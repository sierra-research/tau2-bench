import json

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate, to_litellm_messages


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


def test_to_litellm_messages_tool_call_is_openai_compliant():
    """Assistant tool_calls must match the OpenAI chat-completion schema.

    A tool_call object only allows ``id``, ``type`` and ``function`` (which holds
    ``name`` and ``arguments``). The previous conversion also emitted a top-level
    ``name`` key, which is not part of the spec and trips strict validators in
    downstream consumers that parse these messages with the OpenAI SDK.
    """
    message = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
        ],
    )

    litellm_messages = to_litellm_messages([message])

    tool_calls = litellm_messages[0]["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert set(tool_call.keys()) == {"id", "function", "type"}
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"] == {
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris"}),
    }


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
