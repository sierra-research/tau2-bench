# Copyright Sierra
"""Unit tests for `generate()`'s provider-specific reasoning translation.

These mock the `completion` symbol imported into `tau2.utils.llm_utils` so NO
network call is made; they only assert on the kwargs `generate()` forwards to
litellm.

The behavior under test: newer Anthropic models (the adaptive-thinking
generation, e.g. claude-opus-4-8) reject the legacy
`thinking={"type": "enabled", ...}` shape that the pinned litellm derives from
`reasoning_effort`. `generate()` must translate `reasoning_effort` into the
adaptive shape (`thinking={"type": "adaptive"}` + `output_config={"effort": ...}`)
for those models, while leaving gpt-5 (native `reasoning_effort`) and other
models untouched.
"""

from types import SimpleNamespace

import pytest

import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import Message, SystemMessage, UserMessage


@pytest.fixture
def messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Hello."),
    ]


def _fake_response():
    """A minimal litellm ModelResponse stand-in `generate()` can consume."""
    message = SimpleNamespace(role="assistant", content="hi", tool_calls=None)
    choice = SimpleNamespace(finish_reason="stop", message=message)

    class _Resp(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

        def to_dict(self):
            return {}

    return _Resp(model="test-model", choices=[choice], usage=None)


@pytest.fixture
def capture_completion(monkeypatch):
    """Patch the `completion` symbol used by llm_utils; capture its kwargs."""
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    # No-op the cost lookup so we don't hit litellm's pricing tables.
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    return captured


# --- New Claude (adaptive thinking) ----------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-4-8", "anthropic/claude-fable-5"])
def test_new_claude_uses_adaptive_thinking(capture_completion, messages, model):
    # The full model-id sweep lives on the predicate test below; here two
    # representatives prove the completion-kwargs translation.
    llm_utils.generate(model, messages, reasoning_effort="high")
    # Legacy shape must NOT be sent.
    assert capture_completion.get("thinking") == {"type": "adaptive"}
    assert capture_completion.get("thinking", {}).get("type") != "enabled"
    # Effort is forwarded via Anthropic's output_config, not reasoning_effort.
    assert capture_completion.get("output_config") == {"effort": "high"}
    assert "reasoning_effort" not in capture_completion


def test_explicit_thinking_still_drops_reasoning_effort(capture_completion, messages):
    """Bugbot #305: when a caller passes BOTH `thinking` and `reasoning_effort`
    for an adaptive-thinking Claude model, `reasoning_effort` must still be
    removed — otherwise litellm re-derives the legacy `thinking.type.enabled`
    shape from it and the request 400s despite the explicit `thinking` payload.
    The caller's explicit `thinking` is preserved as-is."""
    llm_utils.generate(
        "claude-opus-4-8",
        messages,
        reasoning_effort="high",
        thinking={"type": "adaptive"},
    )
    assert "reasoning_effort" not in capture_completion
    assert capture_completion.get("thinking") == {"type": "adaptive"}


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_recognized_effort_levels_become_output_config(
    capture_completion, messages, effort
):
    llm_utils.generate("claude-opus-4-8", messages, reasoning_effort=effort)
    assert capture_completion.get("output_config") == {"effort": effort}
    assert capture_completion.get("thinking") == {"type": "adaptive"}


@pytest.mark.parametrize("effort", ["minimal", "none", "weird"])
def test_unrecognized_effort_sends_bare_adaptive(capture_completion, messages, effort):
    # Anthropic's output_config.effort only accepts low/medium/high and 400s on
    # anything else, so we must fall back to bare adaptive thinking.
    llm_utils.generate("claude-opus-4-8", messages, reasoning_effort=effort)
    assert capture_completion.get("thinking") == {"type": "adaptive"}
    assert "output_config" not in capture_completion
    assert "reasoning_effort" not in capture_completion


def test_explicit_thinking_is_respected(capture_completion, messages):
    # A caller that passes an explicit `thinking` wins; reasoning_effort is left
    # as-is (litellm will reconcile), and we do not override their choice.
    llm_utils.generate(
        "claude-opus-4-8",
        messages,
        reasoning_effort="high",
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    assert capture_completion.get("thinking") == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


def test_new_claude_without_effort_is_untouched(capture_completion, messages):
    llm_utils.generate("claude-opus-4-8", messages)
    assert "thinking" not in capture_completion
    assert "output_config" not in capture_completion


# --- Models that must be unaffected ----------------------------------------


def test_gpt5_reasoning_effort_is_preserved(capture_completion, messages):
    llm_utils.generate("gpt-5", messages, reasoning_effort="high")
    # gpt-5 takes reasoning_effort natively; no Anthropic translation.
    assert capture_completion.get("reasoning_effort") == "high"
    assert "thinking" not in capture_completion
    assert "output_config" not in capture_completion


def test_gpt5_gets_default_effort_when_unset(capture_completion, messages):
    from tau2.config import DEFAULT_GPT5_REASONING_EFFORT

    llm_utils.generate("gpt-5-mini", messages)
    assert capture_completion.get("reasoning_effort") == DEFAULT_GPT5_REASONING_EFFORT
    assert "thinking" not in capture_completion


_FAKE_TOOL = SimpleNamespace(
    openai_schema={
        "type": "function",
        "function": {"name": "get_x", "description": "get x", "parameters": {}},
    }
)


def test_gpt5_with_tools_keeps_reasoning_via_responses_bridge(
    capture_completion, messages
):
    """OpenAI rejects reasoning_effort alongside function tools on
    /v1/chat/completions, so gpt-5 tool calls are routed through the Responses
    API bridge (openai/responses/...), which DOES support reasoning + tools.
    reasoning_effort must be preserved, not dropped."""
    llm_utils.generate("gpt-5", messages, tools=[_FAKE_TOOL], reasoning_effort="high")
    assert capture_completion.get("reasoning_effort") == "high"
    assert capture_completion.get("model") == "openai/responses/gpt-5"
    assert capture_completion.get("tools")  # tools still forwarded


def test_gpt5_default_effort_and_bridge_without_tools(capture_completion, messages):
    """No tools -> reasoning_effort default still applied, and gpt-5 still routes
    through the Responses bridge (OpenAI serves gpt-5 there)."""
    from tau2.config import DEFAULT_GPT5_REASONING_EFFORT

    llm_utils.generate("gpt-5.5", messages)
    assert capture_completion.get("reasoning_effort") == DEFAULT_GPT5_REASONING_EFFORT
    assert capture_completion.get("model") == "openai/responses/gpt-5.5"


def test_gpt5_responses_bridge_keeps_tool_calls_from_later_choice(
    monkeypatch, messages
):
    """The Responses bridge maps output items to separate chat choices.

    A response can therefore put assistant text in the first choice and its
    function call in a later choice. Tau2 turns cannot contain both, so the
    actionable function-call choice must take precedence.
    """
    text_message = SimpleNamespace(
        role="assistant", content="I'll look that up.", tool_calls=None
    )
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(name="get_x", arguments='{"x": 7}'),
    )
    tool_message = SimpleNamespace(
        role="assistant", content=None, tool_calls=[tool_call]
    )
    second_tool_call = SimpleNamespace(
        id="call_456",
        function=SimpleNamespace(name="get_x", arguments='{"x": 8}'),
    )
    second_tool_message = SimpleNamespace(
        role="assistant", content=None, tool_calls=[second_tool_call]
    )
    response = _fake_response()
    response.choices = [
        SimpleNamespace(finish_reason="stop", message=text_message),
        SimpleNamespace(finish_reason="tool_calls", message=tool_message),
        SimpleNamespace(finish_reason="tool_calls", message=second_tool_message),
    ]

    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: response)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)

    result = llm_utils.generate("gpt-5.5", messages)

    assert result.content is None
    assert result.tool_calls is not None
    assert [call.model_dump() for call in result.tool_calls] == [
        {
            "id": "call_123",
            "name": "get_x",
            "arguments": {"x": 7},
            "requestor": "assistant",
        },
        {
            "id": "call_456",
            "name": "get_x",
            "arguments": {"x": 8},
            "requestor": "assistant",
        },
    ]


def test_openai_prefixed_gpt5_bridged_once(capture_completion, messages):
    """An openai/-prefixed id is bridged to openai/responses/... (not doubled)."""
    llm_utils.generate("openai/gpt-5.5", messages)
    assert capture_completion.get("model") == "openai/responses/gpt-5.5"


def test_non_openai_gpt5_is_not_bridged(capture_completion, messages):
    """Non-OpenAI providers keep their id (only OpenAI serves the bridge)."""
    llm_utils.generate("azure/gpt-5.5", messages)
    assert capture_completion.get("model") == "azure/gpt-5.5"


def test_older_claude_reasoning_effort_is_preserved(capture_completion, messages):
    # Older Claude (e.g. 3.7 sonnet) does NOT use adaptive thinking; litellm's
    # legacy mapping must keep handling it, so we must pass reasoning_effort
    # through untouched.
    llm_utils.generate("claude-3-7-sonnet-20250219", messages, reasoning_effort="high")
    assert capture_completion.get("reasoning_effort") == "high"
    assert "thinking" not in capture_completion
    assert "output_config" not in capture_completion


def test_non_claude_non_gpt5_is_preserved(capture_completion, messages):
    llm_utils.generate("gpt-4o-mini", messages, reasoning_effort="medium")
    assert capture_completion.get("reasoning_effort") == "medium"
    assert "thinking" not in capture_completion


# --- Predicate unit coverage -----------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-8", True),
        ("anthropic/claude-opus-4-8", True),
        ("vertex_ai/claude-opus-4-7", True),
        ("claude-fable-5", True),
        ("claude-opus-4-6", True),
        ("claude-3-7-sonnet-20250219", False),
        ("gpt-5", False),
        ("gpt-4o-mini", False),
    ],
)
def test_uses_anthropic_adaptive_thinking_predicate(model, expected):
    assert llm_utils._uses_anthropic_adaptive_thinking(model) is expected


# --- Responses-bridge system-message positioning ----------------------------
#
# litellm's Responses-API bridge hoists every string-content system message
# into `instructions`; a history with no user/assistant message (the user
# simulator's first voice turn: system prompt + silence annotation) then sends
# an empty `input`, which OpenAI rejects (`missing_required_parameter: input`).
# `generate()` must rewrap non-leading system messages as content-part lists so
# the bridge keeps them positional in `input`.


def _bridge_input_items(litellm_messages):
    """Run litellm's actual bridge conversion on captured messages."""
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    handler = LiteLLMResponsesTransformationHandler()
    return handler.convert_chat_completion_messages_to_responses_api(litellm_messages)


def test_gpt5_non_leading_system_messages_stay_positional(capture_completion):
    history = [
        SystemMessage(role="system", content="system prompt"),
        UserMessage(role="user", content="hello"),
        SystemMessage(role="system", content="[12 seconds of silence]"),
    ]
    llm_utils.generate("gpt-5.5", history)
    sent = capture_completion["messages"]
    assert sent[0]["content"] == "system prompt"  # leading block -> instructions
    assert sent[2]["content"] == [
        {"type": "text", "text": "[12 seconds of silence]"}
    ]  # keeps its place in `input`

    input_items, instructions = _bridge_input_items(sent)
    assert instructions == "system prompt"
    assert [item["role"] for item in input_items] == ["user", "system"]


def test_gpt5_all_system_history_sends_nonempty_input(capture_completion):
    """First user-sim voice turn: nobody has spoken, history is all system."""
    history = [
        SystemMessage(role="system", content="system prompt"),
        SystemMessage(role="system", content="[12 seconds of silence]"),
    ]
    llm_utils.generate("gpt-5.5", history)
    sent = capture_completion["messages"]
    assert sent[0]["content"] == "system prompt"
    assert sent[1]["content"] == [{"type": "text", "text": "[12 seconds of silence]"}]

    input_items, instructions = _bridge_input_items(sent)
    assert instructions == "system prompt"
    assert input_items, "bridge must never produce an empty `input`"


def test_gpt5_single_system_message_sends_nonempty_input(capture_completion):
    llm_utils.generate("gpt-5.5", [SystemMessage(role="system", content="prompt")])
    sent = capture_completion["messages"]
    assert sent[0]["content"] == [{"type": "text", "text": "prompt"}]

    input_items, _ = _bridge_input_items(sent)
    assert input_items, "bridge must never produce an empty `input`"


def test_non_bridged_model_system_messages_untouched(capture_completion):
    history = [
        SystemMessage(role="system", content="system prompt"),
        UserMessage(role="user", content="hello"),
        SystemMessage(role="system", content="[12 seconds of silence]"),
    ]
    llm_utils.generate("azure/gpt-5.5", history)
    sent = capture_completion["messages"]
    assert sent[0]["content"] == "system prompt"
    assert sent[2]["content"] == "[12 seconds of silence]"
