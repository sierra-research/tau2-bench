from unittest.mock import patch

import litellm
import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import _is_unsupported_temperature_error, generate


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


def _temperature_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=(
            "Unsupported value: 'temperature' does not support 0.0 with this "
            "model. Only the default (1) value is supported."
        ),
        model="gpt-5.2",
        llm_provider="openai",
    )


def _ok_response(model: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "tool_calls": None,
                },
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_is_unsupported_temperature_error():
    assert _is_unsupported_temperature_error(_temperature_error())
    other = litellm.BadRequestError(
        message="Invalid 'messages': too long.",
        model="gpt-5.2",
        llm_provider="openai",
    )
    assert not _is_unsupported_temperature_error(other)
    assert not _is_unsupported_temperature_error(ValueError("temperature"))


def test_generate_drops_temperature_on_reasoning_model(messages: list[Message]):
    """Reasoning models reject temperature; generate should drop it and retry."""
    with patch("tau2.utils.llm_utils.completion") as mock_completion:
        mock_completion.side_effect = [_temperature_error(), _ok_response("gpt-5.2")]
        response = generate("gpt-5.2", messages, temperature=0.0)

    assert isinstance(response, AssistantMessage)
    assert mock_completion.call_count == 2
    assert "temperature" not in mock_completion.call_args_list[1].kwargs


def test_generate_reraises_unrelated_bad_request(messages: list[Message]):
    other = litellm.BadRequestError(
        message="Invalid 'messages': too long.",
        model="gpt-5.2",
        llm_provider="openai",
    )
    with patch("tau2.utils.llm_utils.completion") as mock_completion:
        mock_completion.side_effect = other
        with pytest.raises(litellm.BadRequestError):
            generate("gpt-5.2", messages, temperature=0.0)
    assert mock_completion.call_count == 1


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
