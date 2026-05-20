import json
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from openai.types.responses.response import Response

from tau2.config import DEFAULT_MAX_RETRIES
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Suppress Pydantic serialization warnings from SDK response models
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Load project-local .env when present so text benchmark runs can use OPENAI_API_KEY.
load_dotenv()

# Shared HTTP client for OpenAI Responses API calls.
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
_http_client = httpx.Client(limits=httpx_limits, trust_env=False)
_openai_client = OpenAI(http_client=_http_client)
_websocket_state = local()

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")


def get_openai_client(num_retries: Optional[int] = None) -> OpenAI:
    if num_retries is None:
        return _openai_client
    return _openai_client.with_options(max_retries=num_retries)


def _get_websocket_connection():
    try:
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect
    except ImportError as exc:
        raise RuntimeError(
            "Responses WebSocket transport requires the websockets package. "
            "Install tau2 with the websocket-capable dependencies before using "
            "responses_transport='websocket'."
        ) from exc

    ws = getattr(_websocket_state, "connection", None)
    if ws is not None:
        try:
            is_closed = getattr(ws, "closed", False)
            if not is_closed:
                return ws
        except (AttributeError, WebSocketException):
            return ws

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Responses WebSocket mode.")
    ws = connect(
        "wss://api.openai.com/v1/responses",
        additional_headers={"Authorization": f"Bearer {api_key}"},
    )
    _websocket_state.connection = ws
    return ws


def _close_websocket_connection() -> None:
    ws = getattr(_websocket_state, "connection", None)
    if ws is None:
        return
    try:
        ws.close()
    finally:
        _websocket_state.connection = None


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def _participant_message_text_item(message: Message) -> dict:
    if isinstance(message, AssistantMessage):
        return {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": message.content or "",
                    "annotations": [],
                }
            ],
        }
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": [{"type": "input_text", "text": message.content or ""}],
        }
    raise TypeError(f"Unsupported participant message type: {type(message)}")


def _tool_call_item(tool_call: ToolCall) -> dict:
    call_id = tool_call.id or f"call_{uuid.uuid4().hex}"
    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex}",
        "status": "completed",
        "call_id": call_id,
        "name": tool_call.name,
        "arguments": json.dumps(tool_call.arguments),
    }


def _prepare_raw_output_for_replay(raw_output: list[dict]) -> list[dict]:
    return deepcopy(raw_output)


def _message_contains_web_search_call(message: Message) -> bool:
    raw_data = getattr(message, "raw_data", None)
    raw_output = (raw_data or {}).get("output")
    if not isinstance(raw_output, list):
        return False
    return any(item.get("type") == "web_search_call" for item in raw_output)


def _history_has_web_search_call(messages: list[Message]) -> bool:
    return any(_message_contains_web_search_call(message) for message in messages)


def _message_to_response_items(message: Message) -> list[dict]:
    if isinstance(message, SystemMessage):
        return []

    if isinstance(message, ToolMessage):
        return [
            {
                "type": "function_call_output",
                "call_id": message.id,
                "output": message.content or "",
            }
        ]

    if isinstance(message, ParticipantMessageBase):
        raw_output = (message.raw_data or {}).get("output")
        if isinstance(raw_output, list):
            return _prepare_raw_output_for_replay(raw_output)

        items = []
        if message.has_text_content():
            items.append(_participant_message_text_item(message))
        if message.tool_calls:
            items.extend(_tool_call_item(tool_call) for tool_call in message.tool_calls)
        return items

    raise TypeError(f"Unsupported message type: {type(message)}")


def to_responses_incremental_request(
    messages: list[Message],
) -> tuple[Optional[str], Optional[str], list[dict]]:
    """
    Convert Tau2 messages into an incremental Responses request.

    The returned input contains only messages after the most recent assistant
    response that has a response id. This is intended for WebSocket
    `previous_response_id` continuation. If no previous response id exists, the
    input falls back to the full stateless replay window.
    """
    instructions = "\n\n".join(
        message.content.strip()
        for message in messages
        if isinstance(message, SystemMessage) and message.content
    )
    previous_response_id = None
    input_items: list[dict] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        raw_data = getattr(message, "raw_data", None) or {}
        response_id = raw_data.get("id")
        if isinstance(message, AssistantMessage) and response_id:
            previous_response_id = response_id
            input_items = []
            continue
        input_items.extend(_message_to_response_items(message))
    return instructions or None, previous_response_id, input_items


def to_responses_request(messages: list[Message]) -> tuple[Optional[str], list[dict]]:
    """
    Convert Tau2 messages into a Responses API request payload.

    System messages are collapsed into the top-level `instructions` field. All
    other messages are converted to response items so we can replay prior tool
    calls, tool outputs, and reasoning items statelessly.
    """
    instructions = "\n\n".join(
        message.content.strip()
        for message in messages
        if isinstance(message, SystemMessage) and message.content
    )
    input_items: list[dict] = []
    for message in messages:
        input_items.extend(_message_to_response_items(message))
    return instructions or None, input_items


def _create_response_websocket(payload: dict[str, Any]) -> Response:
    from websockets.exceptions import WebSocketException

    def send_once() -> Response:
        ws = _get_websocket_connection()
        ws.send(json.dumps({"type": "response.create", **payload}))
        while True:
            event = json.loads(ws.recv())
            event_type = event.get("type")
            if event_type == "response.completed":
                response_payload = event["response"]
                if response_payload.get("prompt_cache_retention") == "in_memory":
                    response_payload["prompt_cache_retention"] = "in-memory"
                return Response.model_validate(response_payload)
            if event_type in {"response.failed", "response.incomplete"}:
                raise RuntimeError(json.dumps(event))
            if event_type == "error":
                raise RuntimeError(json.dumps(event.get("error", event)))

    try:
        return send_once()
    except WebSocketException:
        _close_websocket_connection()
        return send_once()
    except RuntimeError as exc:
        # A long-lived connection can hit the 60 minute limit. Reconnect and
        # retry once; with store=true the service can hydrate the prior response.
        message = str(exc)
        if "websocket_connection_limit_reached" in message:
            _close_websocket_connection()
            return send_once()
        raise


def validate_message(message: Message) -> None:
    """
    Validate the message.
    """

    def has_text_content(message: Message) -> bool:
        """
        Check if the message has text content.
        """
        return message.content is not None and bool(message.content.strip())

    def has_content_or_tool_calls(message: ParticipantMessageBase) -> bool:
        """
        Check if the message has content or tool calls.
        """
        return message.has_content() or message.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: list[Message]) -> None:
    """
    Validate the message history.
    """
    for message in messages:
        validate_message(message)


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _format_payload_for_logging(payload: Any) -> Any:
    """
    Format request payloads for debug logging by splitting long text on newlines.
    """
    if isinstance(payload, dict):
        return {key: _format_payload_for_logging(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_format_payload_for_logging(value) for value in payload]
    if isinstance(payload, str) and "\n" in payload:
        return payload.split("\n")
    return payload


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    Behavior depends on the current log mode:
    - "all": Saves every LLM call
    - "latest": Only keeps the most recent call of each call_name type

    Args:
        request_data: Dictionary containing request information
        response_data: Dictionary containing response information
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message")
    """
    log_dir = llm_log_dir.get()

    if log_dir is None:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    current_log_mode = llm_log_mode.get()

    if current_log_mode == "latest" and call_name:
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                pass

    call_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[Any] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    validate_message_history(messages)

    request_kwargs = dict(kwargs)
    num_retries = request_kwargs.pop("num_retries", DEFAULT_MAX_RETRIES)
    reasoning_effort = request_kwargs.pop("reasoning_effort", None)
    verbosity = request_kwargs.pop("verbosity", None)
    web_search_mode = request_kwargs.pop("web_search_mode", "off")
    web_search_context_size = request_kwargs.pop("web_search_context_size", "medium")
    web_search_filters = request_kwargs.pop("web_search_filters", None)
    web_search_user_location = request_kwargs.pop("web_search_user_location", None)
    responses_transport = request_kwargs.pop("responses_transport", "http")
    store = request_kwargs.pop("store", True)
    max_tokens = request_kwargs.pop("max_tokens", None)
    request_kwargs.pop("seed", None)
    request_kwargs.pop("custom_llm_provider", None)

    if model.startswith("gpt-5"):
        request_kwargs.pop("temperature", None)

    if max_tokens is not None and "max_output_tokens" not in request_kwargs:
        request_kwargs["max_output_tokens"] = max_tokens

    reasoning = dict(request_kwargs.pop("reasoning", {}) or {})
    if reasoning_effort is not None:
        reasoning["effort"] = reasoning_effort
    if reasoning:
        request_kwargs["reasoning"] = reasoning

    text = dict(request_kwargs.pop("text", {}) or {})
    if verbosity is not None:
        text["verbosity"] = verbosity
    if text:
        request_kwargs["text"] = text

    if web_search_mode not in {"off", "auto", "required"}:
        raise ValueError(
            f"Invalid web_search_mode={web_search_mode!r}. Expected one of off, auto, required."
        )

    instructions, input_items = to_responses_request(messages)

    tools_schema = []
    if tools:
        for tool in tools:
            function_schema = deepcopy(tool.openai_schema["function"])
            tools_schema.append(
                {
                    "type": "function",
                    "name": function_schema["name"],
                    "description": function_schema.get("description", ""),
                    "parameters": function_schema["parameters"],
                    # Preserve existing tool semantics. Responses defaults to strict mode,
                    # which would otherwise make optional parameters required.
                    "strict": False,
                }
            )

    if web_search_mode != "off":
        web_search_tool = {
            "type": "web_search",
            "search_context_size": web_search_context_size,
        }
        if web_search_filters is not None:
            web_search_tool["filters"] = web_search_filters
        if web_search_user_location is not None:
            web_search_tool["user_location"] = web_search_user_location
        tools_schema.append(web_search_tool)

    if tools_schema and tool_choice is None:
        tool_choice_payload: Any = "auto"
    else:
        tool_choice_payload = tool_choice

    if (
        web_search_mode == "required"
        and tool_choice is None
        and not _history_has_web_search_call(messages)
    ):
        tool_choice_payload = {"type": "web_search"}

    request_data = {
        "model": model,
        "responses_transport": responses_transport,
        "instructions": _format_payload_for_logging(instructions),
        "input": _format_payload_for_logging(input_items),
        "tools": _format_payload_for_logging(tools_schema or None),
        "tool_choice": _format_payload_for_logging(tool_choice_payload),
        "store": store,
        "kwargs": _format_payload_for_logging(request_kwargs),
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    try:
        create_payload = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "tools": tools_schema or None,
            "tool_choice": tool_choice_payload,
            "store": store,
            **request_kwargs,
        }
        if responses_transport == "http":
            client = get_openai_client(num_retries=num_retries)
            response = client.responses.create(**create_payload)
        elif responses_transport == "websocket":
            (
                incremental_instructions,
                previous_response_id,
                incremental_input_items,
            ) = to_responses_incremental_request(messages)
            create_payload["instructions"] = incremental_instructions
            create_payload["input"] = incremental_input_items
            if previous_response_id is not None:
                create_payload["previous_response_id"] = previous_response_id
            response = _create_response_websocket(create_payload)
        else:
            raise ValueError(
                f"Invalid responses_transport={responses_transport!r}. "
                "Expected one of http, websocket."
            )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time

    if response.status == "incomplete" and response.incomplete_details is not None:
        if response.incomplete_details.reason == "max_output_tokens":
            logger.warning("Output might be incomplete due to token limit!")

    cost = None
    usage = None
    if response.usage is not None:
        usage = {
            "completion_tokens": response.usage.output_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "total_tokens": response.usage.total_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
        }

    content = response.output_text or None
    tool_calls = []
    for output_item in response.output:
        if output_item.type != "function_call":
            continue
        tool_calls.append(
            ToolCall(
                id=output_item.call_id,
                name=output_item.name,
                arguments=json.loads(output_item.arguments),
            )
        )
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
        generation_time_seconds=generation_time_seconds,
    )

    response_data = {
        "timestamp": datetime.now().isoformat(),
        "response_id": response.id,
        "status": response.status,
        "output_types": [item.type for item in response.output],
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage


def extract_json_from_llm_response(response: str) -> str:
    """
    Extract JSON from an LLM response, handling markdown code blocks.
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    return response
