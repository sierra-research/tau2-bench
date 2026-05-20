import pytest
from openai.types.responses.response import Response

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils import llm_utils


class DummyResponsesAPI:
    def __init__(self, responses: list[Response]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No mocked responses remaining")
        return self._responses.pop(0)


class DummyClient:
    def __init__(self, responses: list[Response]):
        self.responses = DummyResponsesAPI(responses)


def make_response(output: list[dict], usage: dict | None = None) -> Response:
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage is not None:
        payload["usage"] = usage
    return Response.model_validate(payload)


@pytest.fixture
def model() -> str:
    return "gpt-5.4-mini"


@pytest.fixture
def messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]


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
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]


def test_generate_no_tool_call(
    monkeypatch: pytest.MonkeyPatch, model: str, messages: list[Message]
):
    response = make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "There is no capital city on the moon.",
                        "annotations": [],
                    }
                ],
            }
        ],
        usage={
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 18,
        },
    )
    client = DummyClient([response])
    monkeypatch.setattr(llm_utils, "get_openai_client", lambda num_retries=None: client)

    message = llm_utils.generate(model, messages)

    assert isinstance(message, AssistantMessage)
    assert message.content == "There is no capital city on the moon."
    assert message.tool_calls is None
    assert message.usage == {
        "completion_tokens": 8,
        "prompt_tokens": 10,
        "reasoning_tokens": 0,
        "total_tokens": 18,
        "cached_tokens": 0,
    }
    assert client.responses.calls[0]["store"] is True


def test_generate_passes_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch, model: str, messages: list[Message]
):
    response = make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                    }
                ],
            }
        ],
    )
    client = DummyClient([response])
    monkeypatch.setattr(llm_utils, "get_openai_client", lambda num_retries=None: client)

    llm_utils.generate(model, messages, parallel_tool_calls=False)

    assert client.responses.calls[0]["parallel_tool_calls"] is False


def test_generate_websocket_uses_incremental_input(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    previous = AssistantMessage(
        role="assistant",
        content="Need account details",
        raw_data={"id": "resp_previous", "output": []},
    )
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Start"),
        previous,
        ToolMessage(role="tool", id="call_1", content='{"status":"ok"}'),
    ]
    response = make_response(
        [
            {
                "type": "message",
                "id": "msg_2",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Done",
                        "annotations": [],
                    }
                ],
            }
        ],
    )
    payloads = []

    def fake_websocket(payload):
        payloads.append(payload)
        return response

    monkeypatch.setattr(llm_utils, "_create_response_websocket", fake_websocket)

    message = llm_utils.generate(model, messages, responses_transport="websocket")

    assert message.content == "Done"
    assert payloads[0]["previous_response_id"] == "resp_previous"
    assert payloads[0]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"status":"ok"}',
        }
    ]


def test_generate_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    tool_call_messages: list[Message],
    tool: Tool,
):
    first = make_response(
        [
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "call_id": "call_square",
                "name": "calculate_square",
                "arguments": "{\"x\": 5}",
            }
        ]
    )
    second = make_response(
        [
            {
                "type": "message",
                "id": "msg_2",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "25",
                        "annotations": [],
                    }
                ],
            }
        ]
    )
    client = DummyClient([first, second])
    monkeypatch.setattr(llm_utils, "get_openai_client", lambda num_retries=None: client)

    response = llm_utils.generate(model, tool_call_messages, tools=[tool])

    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    assert response.tool_calls[0].id == "call_square"

    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    final = llm_utils.generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )

    assert isinstance(final, AssistantMessage)
    assert final.tool_calls is None
    assert final.content == "25"

    second_request_input = client.responses.calls[1]["input"]
    assert second_request_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_square",
        "output": "25",
    }


def test_generate_replays_reasoning_items_from_raw_response(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    tool_call_messages: list[Message],
    tool: Tool,
):
    first = make_response(
        [
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "summary": [],
                "content": [],
                "encrypted_content": "ciphertext",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "call_id": "call_square",
                "name": "calculate_square",
                "arguments": "{\"x\": 5}",
            },
        ]
    )
    second = make_response(
        [
            {
                "type": "message",
                "id": "msg_2",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "25",
                        "annotations": [],
                    }
                ],
            }
        ]
    )
    client = DummyClient([first, second])
    monkeypatch.setattr(llm_utils, "get_openai_client", lambda num_retries=None: client)

    response = llm_utils.generate(model, tool_call_messages, tools=[tool])
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id="call_square", content="25"),
    ]
    llm_utils.generate(model, tool_call_messages + follow_up_messages, tools=[tool])

    second_request_input = client.responses.calls[1]["input"]
    assert second_request_input[1]["type"] == "reasoning"
    assert second_request_input[2]["type"] == "function_call"
    assert second_request_input[2]["call_id"] == "call_square"


def test_required_web_search_only_forces_first_search(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    messages: list[Message],
):
    first = make_response(
        [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "moon capital"},
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Searching...",
                        "annotations": [],
                    }
                ],
            },
        ]
    )
    second = make_response(
        [
            {
                "type": "message",
                "id": "msg_2",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "There is no capital city on the moon.",
                        "annotations": [],
                    }
                ],
            }
        ]
    )
    client = DummyClient([first, second])
    monkeypatch.setattr(llm_utils, "get_openai_client", lambda num_retries=None: client)

    response = llm_utils.generate(model, messages, web_search_mode="required")
    assert client.responses.calls[0]["tool_choice"] == {"type": "web_search"}

    llm_utils.generate(
        model,
        messages + [response, UserMessage(role="user", content="Answer now.")],
        web_search_mode="required",
    )
    assert client.responses.calls[1]["tool_choice"] == "auto"
