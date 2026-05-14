
import pytest
import requests

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils import llm_utils
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


def test_generate_responses_api_text(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message]
):
    captured_payloads: list[dict] = []

    def fake_responses(**kwargs):
        captured_payloads.append(kwargs)
        return {
            "id": "resp_text_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The moon has no capital.",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }

    def fail_post(*args, **kwargs):
        raise AssertionError("Responses mode must not use raw requests.post")

    monkeypatch.setattr(llm_utils, "responses", fake_responses)
    monkeypatch.setattr(requests, "post", fail_post)

    response = generate(
        "openai/gpt-oss-20b",
        messages,
        api_mode="responses",
        api_base="http://example.test/v1",
        api_key="EMPTY",
        temperature=0,
        reasoning={"effort": "high"},
    )

    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "The moon has no capital."
    assert response.usage == {"completion_tokens": 7, "prompt_tokens": 11}
    assert captured_payloads[0]["model"] == "openai/openai/gpt-oss-20b"
    assert captured_payloads[0]["api_base"] == "http://example.test/v1"
    assert captured_payloads[0]["api_key"] == "EMPTY"
    assert captured_payloads[0]["instructions"] == "You are a helpful assistant."
    assert captured_payloads[0]["input"] == [
        {"role": "user", "content": "What is the capital of the moon?"}
    ]


def test_generate_responses_api_tool_call_follow_up(
    monkeypatch: pytest.MonkeyPatch,
    tool_call_messages: list[Message],
    tool: Tool,
):
    captured_payloads: list[dict] = []
    fake_responses = [
        {
            "id": "resp_tool_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": "Need the square tool."}
                    ],
                    "encrypted_content": None,
                    "status": None,
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_square_1",
                    "name": "calculate_square",
                    "arguments": "{\"x\": 5}",
                    "status": "completed",
                },
            ],
            "usage": {"input_tokens": 13, "output_tokens": 9},
        },
        {
            "id": "resp_tool_2",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_2",
                    "type": "message",
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
            ],
            "usage": {"input_tokens": 6, "output_tokens": 1},
        },
    ]

    def fake_litellm_responses(**kwargs):
        captured_payloads.append(kwargs)
        return fake_responses[len(captured_payloads) - 1]

    def fail_post(*args, **kwargs):
        raise AssertionError("Responses mode must not use raw requests.post")

    monkeypatch.setattr(llm_utils, "responses", fake_litellm_responses)
    monkeypatch.setattr(requests, "post", fail_post)

    response = generate(
        "openai/gpt-oss-20b",
        tool_call_messages,
        tools=[tool],
        api_mode="responses",
        api_base="http://example.test/v1",
        api_key="EMPTY",
        temperature=0,
        reasoning={"effort": "high"},
    )

    assert isinstance(response, AssistantMessage)
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_square_1"
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    assert captured_payloads[0]["tool_choice"] == "auto"
    assert captured_payloads[0]["tools"][0]["name"] == "calculate_square"

    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    final_response = generate(
        "openai/gpt-oss-20b",
        tool_call_messages + follow_up_messages,
        tools=[tool],
        api_mode="responses",
        api_base="http://example.test/v1",
        api_key="EMPTY",
        temperature=0,
        reasoning={"effort": "high"},
    )

    assert isinstance(final_response, AssistantMessage)
    assert final_response.tool_calls is None
    assert final_response.content == "25"
    assert captured_payloads[1]["previous_response_id"] == "resp_tool_1"
    assert captured_payloads[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_square_1",
            "output": "25",
        }
    ]


def test_generate_responses_api_retries_empty_turn(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message]
):
    captured_payloads: list[dict] = []
    fake_responses = [
        {
            "id": "resp_empty_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "rs_empty_1",
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "I should answer, but this turn is malformed.",
                        }
                    ],
                    "encrypted_content": None,
                    "status": None,
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
        {
            "id": "resp_empty_2",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_ok_2",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Recovered response.",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    ]

    def fake_litellm_responses(**kwargs):
        captured_payloads.append(kwargs)
        return fake_responses[len(captured_payloads) - 1]

    def fail_post(*args, **kwargs):
        raise AssertionError("Responses mode must not use raw requests.post")

    monkeypatch.setattr(llm_utils, "responses", fake_litellm_responses)
    monkeypatch.setattr(requests, "post", fail_post)

    response = generate(
        "openai/gpt-oss-20b",
        messages,
        api_mode="responses",
        api_base="http://example.test/v1",
        api_key="EMPTY",
        temperature=0,
        empty_response_retries=1,
    )

    assert isinstance(response, AssistantMessage)
    assert response.content == "Recovered response."
    assert response.tool_calls is None
    assert len(captured_payloads) == 2


def test_generate_responses_api_is_generic_for_openai_models(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message]
):
    captured_payloads: list[dict] = []

    def fake_litellm_responses(**kwargs):
        captured_payloads.append(kwargs)
        return {
            "id": "resp_gpt41_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_gpt41_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Generic Responses path.",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }

    def fail_completion(*args, **kwargs):
        raise AssertionError("Responses mode must not use chat completions")

    def fail_post(*args, **kwargs):
        raise AssertionError("Responses mode must not use raw requests.post")

    monkeypatch.setattr(llm_utils, "responses", fake_litellm_responses)
    monkeypatch.setattr(llm_utils, "completion", fail_completion)
    monkeypatch.setattr(requests, "post", fail_post)

    response = generate(
        "gpt-4.1-2025-04-14",
        messages,
        api_mode="responses",
        temperature=0,
    )

    assert response.content == "Generic Responses path."
    assert captured_payloads[0]["model"] == "gpt-4.1-2025-04-14"
    assert captured_payloads[0]["tool_choice"] is None


def test_generate_responses_api_preserves_prefixed_model_for_custom_openai_base(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message]
):
    captured_payloads: list[dict] = []

    def fake_litellm_responses(**kwargs):
        captured_payloads.append(kwargs)
        return {
            "id": "resp_custom_base_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_custom_base_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Custom base path.",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }

    monkeypatch.setattr(llm_utils, "responses", fake_litellm_responses)

    response = generate(
        "openai/gpt-oss-20b",
        messages,
        api_mode="responses",
        api_base="http://64.62.141.218:8000/v1",
        api_key="EMPTY",
        temperature=0,
    )

    assert response.content == "Custom base path."
    assert captured_payloads[0]["model"] == "openai/openai/gpt-oss-20b"


def test_generate_responses_api_converts_required_tool_choice_for_gpt_oss(
    monkeypatch: pytest.MonkeyPatch,
    tool_call_messages: list[Message],
    tool: Tool,
):
    captured_payloads: list[dict] = []

    def fake_litellm_responses(**kwargs):
        captured_payloads.append(kwargs)
        return {
            "id": "resp_required_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "fc_required_1",
                    "type": "function_call",
                    "call_id": "call_required_1",
                    "name": "calculate_square",
                    "arguments": "{\"x\": 5}",
                    "status": "completed",
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 5},
        }

    def fail_post(*args, **kwargs):
        raise AssertionError("Responses mode must not use raw requests.post")

    monkeypatch.setattr(llm_utils, "responses", fake_litellm_responses)
    monkeypatch.setattr(requests, "post", fail_post)

    response = generate(
        "openai/gpt-oss-20b",
        tool_call_messages,
        tools=[tool],
        tool_choice="required",
        api_mode="responses",
        api_base="http://example.test/v1",
        api_key="EMPTY",
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "calculate_square"
    assert captured_payloads[0]["tool_choice"] == "auto"
