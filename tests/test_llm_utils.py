from types import SimpleNamespace

import pytest

import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate


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
