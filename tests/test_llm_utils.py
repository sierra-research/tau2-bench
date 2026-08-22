import pytest
from litellm import ModelResponse

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils import llm_utils
from tau2.utils.llm_utils import generate, get_response_cost


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


def response_with_usage_cost(cost: object) -> ModelResponse:
    return ModelResponse(
        model="openrouter/qwen/qwen3.8-max",
        choices=[],
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cost": cost,
        },
    )


def test_get_response_cost_prefers_litellm_cost(monkeypatch: pytest.MonkeyPatch):
    response = response_with_usage_cost(0.25)
    monkeypatch.setattr(llm_utils, "completion_cost", lambda **_: 0.5)

    assert get_response_cost(response) == 0.5


@pytest.mark.parametrize("litellm_cost", [None, 0, 0.0])
def test_get_response_cost_falls_back_when_litellm_cost_is_missing_or_zero(
    monkeypatch: pytest.MonkeyPatch, litellm_cost: object
):
    response = response_with_usage_cost(0.25)
    monkeypatch.setattr(llm_utils, "completion_cost", lambda **_: litellm_cost)

    assert get_response_cost(response) == 0.25


@pytest.mark.parametrize("usage_cost", [0, 0.25])
def test_get_response_cost_falls_back_when_litellm_raises(
    monkeypatch: pytest.MonkeyPatch, usage_cost: object
):
    response = response_with_usage_cost(usage_cost)

    def raise_unknown_model(**_: object) -> float:
        raise ValueError("unknown model")

    monkeypatch.setattr(llm_utils, "completion_cost", raise_unknown_model)

    assert get_response_cost(response) == usage_cost


@pytest.mark.parametrize(
    "usage_cost", [-0.25, float("nan"), "0.25", True, 1 + 2j, None]
)
def test_get_response_cost_rejects_invalid_provider_cost(
    monkeypatch: pytest.MonkeyPatch, usage_cost: object
):
    response = response_with_usage_cost(usage_cost)

    def raise_unknown_model(**_: object) -> float:
        raise ValueError("unknown model")

    monkeypatch.setattr(llm_utils, "completion_cost", raise_unknown_model)

    assert get_response_cost(response) == 0.0


def test_get_response_cost_returns_zero_when_no_cost_is_available(
    monkeypatch: pytest.MonkeyPatch,
):
    response = response_with_usage_cost(None)
    monkeypatch.setattr(llm_utils, "completion_cost", lambda **_: None)

    assert get_response_cost(response) == 0.0


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


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
