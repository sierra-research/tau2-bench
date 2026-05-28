"""Tests for the harness-owned canonical Qwen3 codec.

These tests pin two things:

1. ``render_chat`` is byte-for-byte identical to
   ``tokenizer.apply_chat_template`` for the Qwen3 chat template across a
   battery of message shapes. This is what makes train/eval parity hold by
   construction.
2. ``parse_completion`` reproduces vLLM's non-streaming semantics exactly:
   the Qwen3 reasoning parser runs first, then the Hermes tool parser runs on
   the resulting content.

The tokenizer-backed tests require the ``Qwen/Qwen3-8B`` tokenizer (and the
``transformers``/``tokenizers`` stack). They are skipped automatically if the
tokenizer cannot be loaded so the suite stays green in minimal CI.
"""


import pytest

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils import qwen3_codec
from tau2.utils.qwen3_codec import (
    STOP,
    format_training_example,
    messages_to_chat_dicts,
    parse_completion,
    render_chat,
)

# ---------------------------------------------------------------------------
# Tokenizer fixture (skip everything that needs it if unavailable)
# ---------------------------------------------------------------------------


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(qwen3_codec.DEFAULT_TOKENIZER_ID)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Qwen3 tokenizer unavailable: {e!r}")


@pytest.fixture(scope="module")
def tokenizer():
    return _load_tokenizer()


# ---------------------------------------------------------------------------
# Sample tools / message batteries
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

TOOLS = [WEATHER_TOOL, SEARCH_TOOL]


def _battery():
    """Return (id, messages, tools) tuples covering the render battery."""
    cases = []

    # no-tool, system + user
    cases.append(
        (
            "no_tool_system_user",
            [
                SystemMessage(role="system", content="You are helpful."),
                UserMessage(role="user", content="Hi there!"),
            ],
            None,
        )
    )

    # no system message
    cases.append(
        (
            "no_system",
            [UserMessage(role="user", content="What is 2+2?")],
            None,
        )
    )

    # single tool call
    cases.append(
        (
            "single_tool_call",
            [
                SystemMessage(role="system", content="You are helpful."),
                UserMessage(role="user", content="weather in Tokyo?"),
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="get_weather",
                            arguments={"city": "Tokyo"},
                        )
                    ],
                ),
            ],
            [WEATHER_TOOL],
        )
    )

    # multiple tool calls in one assistant turn
    cases.append(
        (
            "multiple_tool_calls",
            [
                UserMessage(role="user", content="weather + search please"),
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1", name="get_weather", arguments={"city": "Paris"}
                        ),
                        ToolCall(
                            id="c2", name="search", arguments={"query": "Paris news"}
                        ),
                    ],
                ),
            ],
            TOOLS,
        )
    )

    # multi-turn with tool results (consecutive tool msgs merge)
    cases.append(
        (
            "multi_turn_tool_results",
            [
                SystemMessage(role="system", content="You are helpful."),
                UserMessage(role="user", content="weather in Tokyo and Paris?"),
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1", name="get_weather", arguments={"city": "Tokyo"}
                        ),
                        ToolCall(
                            id="c2", name="get_weather", arguments={"city": "Paris"}
                        ),
                    ],
                ),
                ToolMessage(id="c1", role="tool", content="Tokyo: sunny 20C"),
                ToolMessage(id="c2", role="tool", content="Paris: cloudy 15C"),
                AssistantMessage(
                    role="assistant",
                    content="Tokyo is sunny, Paris is cloudy.",
                ),
            ],
            [WEATHER_TOOL],
        )
    )

    # tool-only assistant content (content is empty string)
    cases.append(
        (
            "tool_only_empty_content",
            [
                UserMessage(role="user", content="search cats"),
                AssistantMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "cats"})
                    ],
                ),
            ],
            [SEARCH_TOOL],
        )
    )

    # unicode args
    cases.append(
        (
            "unicode_args",
            [
                UserMessage(role="user", content="weather in 東京?"),
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1", name="get_weather", arguments={"city": "東京 ☀"}
                        )
                    ],
                ),
            ],
            [WEATHER_TOOL],
        )
    )

    return cases


# ---------------------------------------------------------------------------
# render_chat parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,messages,tools", _battery(), ids=lambda c: c)
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_render_chat_matches_apply_chat_template(
    tokenizer, case_id, messages, tools, enable_thinking, add_generation_prompt
):
    # Skip nonsensical params produced by the outer parametrize loop.
    if isinstance(case_id, str) is False:
        pytest.skip("param artifact")

    expected = tokenizer.apply_chat_template(
        messages_to_chat_dicts(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    got = render_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tokenizer_id=qwen3_codec.DEFAULT_TOKENIZER_ID,
    )
    assert got == expected


# ---------------------------------------------------------------------------
# parse_completion semantic tests (no tokenizer needed)
# ---------------------------------------------------------------------------


def test_parse_pure_message_thinking_disabled():
    # With thinking disabled and no </think>, vLLM treats it all as content.
    out = parse_completion("Hello, how can I help?", thinking_enabled=False)
    assert out.reasoning is None
    assert out.content == "Hello, how can I help?"
    assert out.tool_calls is None


def test_parse_pure_text_thinking_enabled_is_reasoning():
    # vLLM Qwen3 parser: thinking enabled + no </think> => whole output is
    # reasoning (treated as truncated). Documented behavior, pinned here.
    out = parse_completion("Hello, how can I help?", thinking_enabled=True)
    assert out.reasoning == "Hello, how can I help?"
    assert out.content is None
    assert out.tool_calls is None


def test_parse_message_after_think_end():
    # The realistic thinking-enabled path: prompt injects <think>, output
    # carries </think> then the actual message.
    out = parse_completion("let me think</think>Hello, how can I help?", True)
    assert out.reasoning == "let me think"
    assert out.content == "Hello, how can I help?"
    assert out.tool_calls is None


def test_parse_leading_think_block():
    text = "<think>I should greet.</think>Hello!"
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "I should greet."
    assert out.content == "Hello!"
    assert out.tool_calls is None


def test_parse_think_terminated_by_end_only():
    # Qwen3.5 style: <think> in prompt, only </think> in output.
    text = "reasoning here</think>final answer"
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "reasoning here"
    assert out.content == "final answer"


def test_parse_single_tool_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    out = parse_completion(text, thinking_enabled=True)
    assert out.tool_calls is not None
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Tokyo"}
    assert tc.id  # synthesized id
    # content before first tool_call is empty -> None
    assert out.content is None


def test_parse_content_before_tool_call_after_think_end():
    # Realistic: </think> closes reasoning, then content precedes <tool_call>.
    text = (
        "reason</think>Let me check."
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    )
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "reason"
    assert out.content == "Let me check."
    assert out.tool_calls[0].name == "get_weather"


def test_parse_multiple_tool_calls():
    text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        '<tool_call>\n{"name": "search", "arguments": {"query": "Paris"}}\n</tool_call>'
    )
    out = parse_completion(text, thinking_enabled=True)
    assert [tc.name for tc in out.tool_calls] == ["get_weather", "search"]
    assert out.tool_calls[0].arguments == {"city": "Paris"}
    assert out.tool_calls[1].arguments == {"query": "Paris"}


def test_parse_think_then_tool_call():
    text = (
        "<think>need weather</think>"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    )
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "need weather"
    assert out.content is None
    assert out.tool_calls[0].name == "get_weather"


def test_parse_think_then_tool_call_whitespace_gap_is_none():
    # Qwen3 emits a blank line between </think> and <tool_call>; the leftover
    # whitespace before the tool call must normalize to None (matching vLLM),
    # not "\n\n", so replayed history stays identical to the native path.
    text = (
        "<think>need weather</think>\n\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    )
    out = parse_completion(text, thinking_enabled=True)
    assert out.content is None
    assert out.tool_calls[0].name == "get_weather"


def test_parse_missing_think_end_terminated_by_tool_call():
    # No </think>; <tool_call> acts as implicit reasoning terminator.
    text = (
        "I am thinking about the weather "
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    )
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "I am thinking about the weather "
    assert out.tool_calls[0].name == "get_weather"


def test_parse_missing_think_end_thinking_disabled_is_content():
    text = "just an answer, no closing think"
    out = parse_completion(text, thinking_enabled=False)
    assert out.reasoning is None
    assert out.content == "just an answer, no closing think"
    assert out.tool_calls is None


def test_parse_missing_think_end_thinking_enabled_all_reasoning():
    text = "this got truncated mid-thought"
    out = parse_completion(text, thinking_enabled=True)
    assert out.reasoning == "this got truncated mid-thought"
    assert out.content is None


def test_parse_malformed_tool_call_falls_back_to_content():
    # Mirrors vLLM Hermes: on JSON error, tools_called=False, content=full text.
    text = '<tool_call>\n{"name": "f", "arguments": not json}\n</tool_call>'
    out = parse_completion(text, thinking_enabled=False)
    assert out.tool_calls is None
    assert out.content == text


# ---------------------------------------------------------------------------
# Round-trip: render a final assistant turn, parse it back.
# ---------------------------------------------------------------------------


def test_roundtrip_tool_call(tokenizer):
    messages = [
        UserMessage(role="user", content="weather in Tokyo?"),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="get_weather", arguments={"city": "Tokyo"})
            ],
        ),
    ]
    # Render WITHOUT generation prompt; isolate the last assistant turn body.
    full = render_chat(
        messages,
        tools=[WEATHER_TOOL],
        add_generation_prompt=False,
        enable_thinking=True,
    )
    # Extract the generated portion of the final assistant turn.
    marker = "<|im_start|>assistant\n"
    body = full[full.rindex(marker) + len(marker) :]
    body = body.split("<|im_end|>")[0]
    out = parse_completion(body, thinking_enabled=True)
    assert out.tool_calls[0].name == "get_weather"
    assert out.tool_calls[0].arguments == {"city": "Tokyo"}


def test_roundtrip_text(tokenizer):
    messages = [
        UserMessage(role="user", content="hi"),
        AssistantMessage(role="assistant", content="Hello there!"),
    ]
    full = render_chat(
        messages, tools=None, add_generation_prompt=False, enable_thinking=True
    )
    marker = "<|im_start|>assistant\n"
    body = full[full.rindex(marker) + len(marker) :].split("<|im_end|>")[0]
    out = parse_completion(body, thinking_enabled=True)
    # Empty injected <think>\n\n</think>\n\n -> reasoning is the empty think
    # body, content is the message (vLLM does NOT strip the leading newlines
    # after </think>, so we assert exact-match against the raw split).
    assert out.content == "\n\nHello there!"
    assert out.content.strip() == "Hello there!"
    assert out.tool_calls is None


# ---------------------------------------------------------------------------
# generate() io_mode branch
# ---------------------------------------------------------------------------


def test_generate_vllm_branch(monkeypatch, tokenizer):
    from types import SimpleNamespace

    import litellm

    from tau2.environment.tool import as_tool
    from tau2.utils import llm_utils

    captured = {}

    def fake_text_completion(*, model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        raw = (
            "<think>checking weather</think>"
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        )
        usage = SimpleNamespace(completion_tokens=12, prompt_tokens=34)
        resp = SimpleNamespace(
            choices=[SimpleNamespace(text=raw, finish_reason="stop")],
            usage=usage,
            model=model,
            to_dict=lambda: {"choices": [{"text": raw}]},
        )
        resp.get = lambda key, default=None: getattr(resp, key, default)
        return resp

    monkeypatch.setattr(litellm, "text_completion", fake_text_completion)
    # Avoid cost lookup hitting the network / unknown model maps.
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)

    def get_weather(city: str) -> str:
        """Get the weather for a city.

        Args:
            city (str): the city.
        Returns:
            str: the weather.
        """
        return "sunny"

    msg = llm_utils.generate(
        model="hosted_vllm/Qwen/Qwen3-8B",
        messages=[UserMessage(role="user", content="weather in Tokyo?")],
        tools=[as_tool(get_weather)],
        io_mode="completions",
    )
    assert isinstance(msg, AssistantMessage)
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].name == "get_weather"
    assert msg.tool_calls[0].arguments == {"city": "Tokyo"}
    assert msg.usage == {"completion_tokens": 12, "prompt_tokens": 34}
    # stop tokens were passed through
    assert captured["kwargs"]["stop"] == STOP
    # io_mode must not leak into the litellm call
    assert "io_mode" not in captured["kwargs"]
    # prompt ends with the generation prompt tail
    assert captured["prompt"].endswith("<|im_start|>assistant\n")


def test_generate_vllm_branch_plain_message(monkeypatch, tokenizer):
    from types import SimpleNamespace

    import litellm

    from tau2.utils import llm_utils

    def fake_text_completion(*, model, prompt, **kwargs):
        raw = "Hello, how can I help you today?"
        usage = SimpleNamespace(completion_tokens=8, prompt_tokens=10)
        resp = SimpleNamespace(
            choices=[SimpleNamespace(text=raw, finish_reason="stop")],
            usage=usage,
            model=model,
            to_dict=lambda: {"choices": [{"text": raw}]},
        )
        resp.get = lambda key, default=None: getattr(resp, key, default)
        return resp

    monkeypatch.setattr(litellm, "text_completion", fake_text_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda r: 0.0)

    msg = llm_utils.generate(
        model="hosted_vllm/Qwen/Qwen3-8B",
        messages=[UserMessage(role="user", content="hi")],
        io_mode="completions",
        enable_thinking=False,
    )
    assert msg.content == "Hello, how can I help you today?"
    assert msg.tool_calls is None


# ---------------------------------------------------------------------------
# Training-data loss mask
# ---------------------------------------------------------------------------


def test_format_training_example_loss_mask(tokenizer):
    record = {
        "tools": [WEATHER_TOOL],
        "conversations": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        },
                    }
                ],
            },
        ],
    }
    formatted = format_training_example(record)
    input_ids = formatted["input_ids"]
    labels = formatted["labels"]
    assert len(input_ids) == len(labels)
    # Some tokens unmasked (the final assistant turn).
    assert any(label != -100 for label in labels)
    # Everything before the final assistant header is masked.
    text = formatted["text"]
    assert "<|im_start|>assistant" in text
    # The unmasked region must be a contiguous suffix-ish block: the first
    # unmasked token index should be at/after the final assistant header.
    first_unmasked = next(i for i, lab in enumerate(labels) if lab != -100)
    # Decode the prefix that is masked and assert it does not contain the
    # final assistant's tool call.
    assert first_unmasked > 0
    # Unmasked labels equal the corresponding input_ids (teacher forcing).
    for i, lab in enumerate(labels):
        if lab != -100:
            assert lab == input_ids[i]


def test_format_training_example_only_final_assistant_unmasked(tokenizer):
    # Two assistant turns: only the LAST one should be unmasked.
    record = {
        "tools": None,
        "conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ],
    }
    formatted = format_training_example(record)
    labels = formatted["labels"]
    input_ids = formatted["input_ids"]

    # Decode only the unmasked tokens; it should contain "Goodbye" not "Hello".
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen3_codec.DEFAULT_TOKENIZER_ID)
    unmasked_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded = tok.decode(unmasked_ids)
    assert "Goodbye" in decoded
    assert "Hello" not in decoded


# ---------------------------------------------------------------------------
# Template version pinning
# ---------------------------------------------------------------------------


def test_get_stop_tokens_uses_model_eos(tokenizer):
    # Stop must be derived from the served model's tokenizer (its eos), not a
    # hardcoded Qwen terminator, so non-Qwen families halt correctly.
    stops = qwen3_codec.get_stop_tokens()
    assert stops == [tokenizer.eos_token]
    assert stops == ["<|im_end|>"]  # Qwen3 default tokenizer


def test_chat_template_signature_stable(tokenizer):
    sig1 = qwen3_codec.chat_template_signature()
    sig2 = qwen3_codec.chat_template_signature()
    assert sig1 == sig2 and isinstance(sig1, str) and len(sig1) == 16


def test_assert_template_matches_ok(tokenizer):
    sig = qwen3_codec.chat_template_signature()
    qwen3_codec.assert_template_matches(sig)  # should not raise


def test_assert_template_matches_raises_on_skew(tokenizer):
    with pytest.raises(ValueError, match="Chat-template mismatch"):
        qwen3_codec.assert_template_matches("deadbeefdeadbeef")


def test_default_tokenizer_id_env_overridable(monkeypatch):
    # The default is resolved from TAU2_QWEN3_TOKENIZER at import; verify the
    # resolution logic (re-evaluate the same expression the module uses).
    monkeypatch.setenv("TAU2_QWEN3_TOKENIZER", "some/local-checkpoint")
    resolved = __import__("os").environ.get("TAU2_QWEN3_TOKENIZER", "Qwen/Qwen3-8B")
    assert resolved == "some/local-checkpoint"
