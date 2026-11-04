# Copyright Sierra
"""Offline completion-boundary regressions for provider reasoning and message order."""

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
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    return captured


@pytest.mark.parametrize("model", ["claude-opus-4-8", "anthropic/claude-fable-5"])
def test_new_claude_uses_adaptive_thinking(capture_completion, messages, model):
    llm_utils.generate(model, messages, reasoning_effort="high")
    assert capture_completion.get("thinking") == {"type": "adaptive"}
    assert capture_completion.get("thinking", {}).get("type") != "enabled"
    assert capture_completion.get("output_config") == {"effort": "high"}
    assert "reasoning_effort" not in capture_completion


def test_explicit_thinking_still_drops_reasoning_effort(capture_completion, messages):
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
    llm_utils.generate("claude-opus-4-8", messages, reasoning_effort=effort)
    assert capture_completion.get("thinking") == {"type": "adaptive"}
    assert "output_config" not in capture_completion
    assert "reasoning_effort" not in capture_completion


def test_explicit_thinking_is_respected(capture_completion, messages):
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


def test_gpt5_reasoning_effort_is_preserved(capture_completion, messages):
    llm_utils.generate("gpt-5", messages, reasoning_effort="high")
    assert capture_completion.get("reasoning_effort") == "high"
    assert "thinking" not in capture_completion
    assert "output_config" not in capture_completion


def test_gpt5_preserves_provider_default_effort_when_unset(
    capture_completion, messages
):
    llm_utils.generate("gpt-5-mini", messages)
    assert "reasoning_effort" not in capture_completion
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
    llm_utils.generate("gpt-5", messages, tools=[_FAKE_TOOL], reasoning_effort="high")
    assert capture_completion.get("reasoning_effort") == "high"
    assert capture_completion.get("model") == "openai/responses/gpt-5"
    assert capture_completion.get("tools")  # tools still forwarded


def test_gpt5_bridge_without_tools(capture_completion, messages):
    llm_utils.generate("gpt-5.5", messages)
    assert "reasoning_effort" not in capture_completion
    assert capture_completion.get("model") == "openai/responses/gpt-5.5"


def test_openai_prefixed_gpt5_bridged_once(capture_completion, messages):
    llm_utils.generate("openai/gpt-5.5", messages)
    assert capture_completion.get("model") == "openai/responses/gpt-5.5"


def test_non_openai_gpt5_is_not_bridged(capture_completion, messages):
    llm_utils.generate("azure/gpt-5.5", messages)
    assert capture_completion.get("model") == "azure/gpt-5.5"


def test_older_claude_reasoning_effort_is_preserved(capture_completion, messages):
    llm_utils.generate("claude-3-7-sonnet-20250219", messages, reasoning_effort="high")
    assert capture_completion.get("reasoning_effort") == "high"
    assert "thinking" not in capture_completion
    assert "output_config" not in capture_completion


def test_non_claude_non_gpt5_is_preserved(capture_completion, messages):
    llm_utils.generate("gpt-4o-mini", messages, reasoning_effort="medium")
    assert capture_completion.get("reasoning_effort") == "medium"
    assert "thinking" not in capture_completion


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


def _bridge_input_items(litellm_messages):
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


@pytest.mark.parametrize("model", ["gpt-5.5", "openai/gpt-5.4-mini-2026-03-17"])
def test_responses_bridge_keeps_tool_choice(capture_completion, messages, model):
    """The bridge's failed capability lookup must not drop a requested tool_choice."""
    llm_utils.generate(model, messages, tool_choice="required")
    assert capture_completion["model"].startswith("openai/responses/")
    assert capture_completion["tool_choice"] == "required"
    assert capture_completion["allowed_openai_params"] == ["tool_choice"]


def test_non_bridged_models_keep_default_param_filtering(capture_completion, messages):
    llm_utils.generate("gpt-4.1-2025-04-14", messages, tool_choice="required")
    assert "allowed_openai_params" not in capture_completion


def test_litellm_bridge_param_mapping_keeps_tool_choice_quietly(capsys):
    """Pinned-LiteLLM regression: allowed params survive the bridge's lookup."""
    from litellm.utils import get_optional_params

    params = get_optional_params(
        model="responses/gpt-5.5",
        custom_llm_provider="openai",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="required",
        allowed_openai_params=list(llm_utils._RESPONSES_BRIDGE_PARAMS),
    )
    assert params["tool_choice"] == "required"
    assert "Provider List" not in capsys.readouterr().out
