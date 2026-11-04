import logging
from types import SimpleNamespace

import pytest
from loguru import logger

import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import (
    _UNSUPPORTED_PARAM_WARNINGS,
    generate,
    warn_if_param_unsupported,
)


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


@pytest.mark.live_api
def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


@pytest.mark.live_api
def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"


class TestWarnIfParamUnsupported:
    """A requested parameter that the provider will drop must not stay silent.

    Regression coverage for #558: `litellm.drop_params = True` turns LiteLLM's
    UnsupportedParamsError into a no-op, so `--seed` reached the call, was
    dropped for providers without seed support, and nothing said so. Two
    unseeded runs then look like a reproducible pair, which is easy to believe
    because a model at temperature 0 can be deterministic for unrelated reasons.
    """

    def setup_method(self):
        _UNSUPPORTED_PARAM_WARNINGS.clear()

    def test_supported_param_returns_true_and_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert warn_if_param_unsupported("gpt-4o-mini", "seed") is True
        assert "does not support" not in caplog.text

    def test_unsupported_param_returns_false(self):
        assert warn_if_param_unsupported("gemini/gemini-2.0-flash", "seed") is False

    def test_unsupported_param_warns_once_per_model_and_param(self):
        messages: list[str] = []
        handle = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            warn_if_param_unsupported("gemini/gemini-2.0-flash", "seed")
            warn_if_param_unsupported("gemini/gemini-2.0-flash", "seed")
        finally:
            logger.remove(handle)

        assert len(messages) == 1
        assert "NOT seeded" in messages[0]

    def test_unknown_model_returns_none_and_stays_quiet(self):
        """A lookup failure must not manufacture an unseeded-run warning."""
        messages: list[str] = []
        handle = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            result = warn_if_param_unsupported(
                "not-a-real-provider/not-a-model", "seed"
            )
        finally:
            logger.remove(handle)

        assert result is None
        assert messages == []

    def test_non_seed_param_gets_the_generic_message(self):
        """Only `seed` earns the reproducibility wording; others stay generic.

        A name no provider accepts keeps this independent of which parameters a
        given provider happens to support this week.
        """
        messages: list[str] = []
        handle = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            warn_if_param_unsupported("gpt-4o-mini", "not_a_real_param")
        finally:
            logger.remove(handle)

        assert len(messages) == 1
        assert "has no effect" in messages[0]
        assert "NOT seeded" not in messages[0]


def test_gpt_5_5_xhigh_uses_responses_api_and_preserves_system_turns(
    monkeypatch: pytest.MonkeyPatch, tool: Tool
):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant", content="Done", tool_calls=None
                    ),
                )
            ],
            to_dict=lambda: {},
        )

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(
        llm_utils,
        "get_response_usage",
        lambda response: {"prompt_tokens": 1, "completion_tokens": 1},
    )
    monkeypatch.setattr(llm_utils, "_write_llm_log", lambda *args, **kwargs: None)

    response = generate(
        "gpt-5.5",
        [
            SystemMessage(role="system", content="You are a customer."),
            UserMessage(role="user", content="Hello"),
            SystemMessage(role="system", content="The agent is silent."),
        ],
        tools=[tool],
        reasoning_effort="xhigh",
        extra_body={"metadata": {"source": "test"}},
    )

    assert response.content == "Done"
    assert captured["model"] == "openai/responses/gpt-5.5"
    assert captured["extra_body"] == {
        "metadata": {"source": "test"},
        "reasoning": {"effort": "xhigh"},
    }
    assert "reasoning_effort" not in captured
    assert captured["messages"][0]["content"] == "You are a customer."
    assert captured["messages"][2]["content"] == [
        {"type": "text", "text": "The agent is silent."}
    ]
    assert captured["tools"] == [tool.openai_schema]
