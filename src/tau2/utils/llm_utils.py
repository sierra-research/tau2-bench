import json
import logging
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import litellm
from litellm import completion, completion_cost, responses
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
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

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to type mismatches between streaming and non-streaming response types
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Configure httpx connection limits for LiteLLM
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
litellm.client_session = httpx.Client(limits=httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")

# litellm._turn_on_debug()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

if USE_LANGFUSE:
    litellm.success_callback = ["langfuse"]
else:
    litellm.success_callback = []

litellm.drop_params = True

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()

RESPONSES_API_MODES = {"responses", "response", "responses_api"}


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


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


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


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


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.

    Args:
        messages: List of litellm message dictionaries

    Returns:
        Modified message list with content split into lines for readability
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            # Split content on newlines for better readability
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


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
        # No log directory set, skip logging
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get current logging mode
    current_log_mode = llm_log_mode.get()

    # If mode is "latest" and call_name is provided, remove existing files with the same call_name
    if current_log_mode == "latest" and call_name:
        # Find and remove existing files with this call_name
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                # File might have been removed by another thread, ignore
                pass

    # Create a new file for this LLM call
    call_id = str(uuid.uuid4())[:8]  # Use short UUID for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

    # Include call_name in filename if provided
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    # Create complete JSON structure with both request and response
    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Write to file with indentation
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


def _use_responses_api(kwargs: dict[str, Any]) -> bool:
    api_mode = kwargs.get("api_mode")
    if isinstance(api_mode, str) and api_mode.lower() in RESPONSES_API_MODES:
        return True
    use_responses_api = kwargs.get("use_responses_api")
    return bool(use_responses_api)


def _normalize_responses_model_name(model: str) -> str:
    """
    Normalize older locally documented model aliases before LiteLLM Responses calls.
    """
    if model.startswith("hosted_vllm/"):
        return model[len("hosted_vllm/") :]
    return model


def _uses_official_openai_api_base(api_base: str) -> bool:
    api_base = api_base.rstrip("/")
    return api_base in {"https://api.openai.com", "https://api.openai.com/v1"}


def _get_litellm_responses_model_name(model: str, api_base: str) -> str:
    """
    LiteLLM strips one provider prefix for OpenAI-compatible calls. Some
    self-hosted endpoints expose model IDs that intentionally contain the
    `openai/` prefix, so add the LiteLLM provider prefix without changing the
    model ID sent to that endpoint.
    """
    normalized_model = _normalize_responses_model_name(model)
    if (
        not _uses_official_openai_api_base(api_base)
        and normalized_model.startswith("openai/")
        and not normalized_model.startswith("openai/openai/")
    ):
        return f"openai/{normalized_model}"
    return normalized_model


def _looks_like_gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _responses_to_dict(response: Any) -> dict:
    if isinstance(response, dict):
        return response
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            data = method()
            if isinstance(data, dict):
                return data
    json_method = getattr(response, "json", None)
    if callable(json_method):
        data = json.loads(json_method())
        if isinstance(data, dict):
            return data
    raise TypeError(f"Unsupported LiteLLM Responses object: {type(response)}")


def _get_responses_tools_schema(tools: list[Tool]) -> list[dict]:
    responses_tools = []
    for tool in tools:
        schema = tool.openai_schema
        function = schema.get("function") or {}
        responses_tools.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            }
        )
    return responses_tools


def _get_responses_usage(response_data: dict) -> Optional[dict]:
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "completion_tokens": usage.get("output_tokens", 0),
        "prompt_tokens": usage.get("input_tokens", 0),
    }


def _extract_responses_output_text(response_data: dict) -> Optional[str]:
    output = response_data.get("output") or []
    text_chunks: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for content_item in item.get("content") or []:
            if content_item.get("type") in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text)
    if text_chunks:
        return "\n".join(text_chunks)
    output_text = response_data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    return None


def _extract_responses_tool_calls(response_data: dict) -> Optional[list[ToolCall]]:
    output = response_data.get("output") or []
    tool_calls: list[ToolCall] = []
    for item in output:
        if item.get("type") != "function_call":
            continue
        raw_arguments = item.get("arguments")
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=item.get("call_id") or item.get("id") or "",
                name=item["name"],
                arguments=arguments,
            )
        )
    return tool_calls or None


def _get_system_instructions(messages: list[Message]) -> Optional[str]:
    instructions = [
        message.content.strip()
        for message in messages
        if isinstance(message, SystemMessage)
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    if not instructions:
        return None
    return "\n\n".join(instructions)


def _assistant_message_to_responses_input(message: AssistantMessage) -> list[dict]:
    raw_data = message.raw_data if isinstance(message.raw_data, dict) else None
    if raw_data and raw_data.get("object") == "response":
        output = raw_data.get("output")
        if isinstance(output, list) and output:
            return deepcopy(output)

    input_items: list[dict] = []
    if message.has_text_content():
        input_items.append({"role": "assistant", "content": message.content})
    if message.tool_calls:
        for tool_call in message.tool_calls:
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                }
            )
    return input_items


def _message_to_responses_input(message: Message) -> list[dict]:
    if isinstance(message, SystemMessage):
        return []
    if isinstance(message, UserMessage):
        if message.has_text_content():
            return [{"role": "user", "content": message.content}]
        return []
    if isinstance(message, AssistantMessage):
        return _assistant_message_to_responses_input(message)
    if isinstance(message, ToolMessage):
        return [
            {
                "type": "function_call_output",
                "call_id": message.id,
                "output": message.content or "",
            }
        ]
    if hasattr(message, "tool_messages"):
        input_items: list[dict] = []
        for tool_message in message.tool_messages:
            input_items.extend(_message_to_responses_input(tool_message))
        return input_items
    raise ValueError(f"Unsupported message type for Responses API: {type(message)}")


def _messages_to_responses_input(messages: list[Message]) -> list[dict]:
    input_items: list[dict] = []
    for message in messages:
        input_items.extend(_message_to_responses_input(message))
    return input_items


def _get_previous_response_id(message: AssistantMessage) -> Optional[str]:
    raw_data = message.raw_data if isinstance(message.raw_data, dict) else None
    if raw_data and raw_data.get("object") == "response":
        response_id = raw_data.get("id")
        if isinstance(response_id, str) and response_id.strip():
            return response_id
    return None


def _find_previous_response_anchor(
    messages: list[Message],
) -> Optional[tuple[int, str]]:
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, AssistantMessage):
            continue
        response_id = _get_previous_response_id(message)
        if response_id is not None:
            return idx, response_id
    return None


def _responses_request(
    *,
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]],
    tool_choice: Optional[str],
    call_name: Optional[str],
    **kwargs: Any,
) -> AssistantMessage:
    request_kwargs = kwargs.copy()
    request_kwargs.pop("api_mode", None)
    request_kwargs.pop("use_responses_api", None)
    request_kwargs.pop("litellm_interface", None)

    if "max_tokens" in request_kwargs and "max_output_tokens" not in request_kwargs:
        request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

    api_base = request_kwargs.pop(
        "api_base", os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    )
    api_key = request_kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
    request_timeout = request_kwargs.pop("request_timeout", None)
    if request_timeout is None:
        request_timeout = request_kwargs.pop("timeout", 120)
    empty_response_retries = int(request_kwargs.pop("empty_response_retries", 2) or 0)
    num_retries = int(request_kwargs.pop("num_retries", DEFAULT_MAX_RETRIES) or 0)
    normalized_model = _normalize_responses_model_name(model)
    litellm_model = _get_litellm_responses_model_name(model, api_base)

    if tool_choice == "required" and _looks_like_gpt_oss_model(normalized_model):
        logger.warning(
            "Responses API tool_choice='required' is not supported by Harmony-style "
            "gpt-oss endpoints. Falling back to tool_choice='auto'."
        )
        tool_choice = "auto"

    instructions = _get_system_instructions(messages)
    previous_response_anchor = _find_previous_response_anchor(messages)
    if previous_response_anchor is None:
        anchored_response_id = None
    else:
        _, anchored_response_id = previous_response_anchor

    responses_tools = _get_responses_tools_schema(tools) if tools else None
    if responses_tools is not None and tool_choice is None:
        tool_choice = "auto"
    if responses_tools is not None and "parallel_tool_calls" not in request_kwargs:
        request_kwargs["parallel_tool_calls"] = True
    effective_tool_choice = tool_choice

    def build_payload(
        use_previous_response_id: bool,
    ) -> tuple[dict[str, Any], Optional[str], list[dict]]:
        if use_previous_response_id and previous_response_anchor is not None:
            anchor_idx, previous_response_id = previous_response_anchor
            input_items = _messages_to_responses_input(messages[anchor_idx + 1 :])
        else:
            previous_response_id = None
            input_items = _messages_to_responses_input(messages)

        payload: dict[str, Any] = {
            "model": litellm_model,
            "input": input_items,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if responses_tools is not None:
            payload["tools"] = responses_tools
        if effective_tool_choice is not None:
            payload["tool_choice"] = effective_tool_choice
        payload.update(request_kwargs)
        return payload, previous_response_id, input_items

    payload, previous_response_id, input_items = build_payload(
        use_previous_response_id=anchored_response_id is not None
    )

    request_data = {
        "model": normalized_model,
        "litellm_model": litellm_model,
        "instructions": instructions,
        "input": _format_messages_for_logging(input_items),
        "tools": responses_tools,
        "tool_choice": effective_tool_choice,
        "previous_response_id": previous_response_id,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in request_kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    response_data: Optional[dict] = None
    empty_response_attempt = 0
    for attempt in range(num_retries + 1):
        try:
            response = responses(
                input=payload["input"],
                model=payload["model"],
                instructions=payload.get("instructions"),
                previous_response_id=payload.get("previous_response_id"),
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
                timeout=request_timeout,
                api_base=api_base,
                api_key=api_key,
                **request_kwargs,
            )
            candidate_response_data = _responses_to_dict(response)
            candidate_content = _extract_responses_output_text(candidate_response_data)
            candidate_tool_calls = _extract_responses_tool_calls(candidate_response_data)
            if candidate_content is None and candidate_tool_calls is None:
                if empty_response_attempt < empty_response_retries:
                    empty_response_attempt += 1
                    logger.warning(
                        "Responses API returned neither content nor tool calls. "
                        f"Retrying the same turn ({empty_response_attempt}/"
                        f"{empty_response_retries})."
                    )
                    time.sleep(min(0.5 * (2**(empty_response_attempt - 1)), 2))
                    continue
                raise RuntimeError(
                    "Responses API returned neither content nor tool calls after "
                    f"{empty_response_attempt + 1} attempts."
                )
            response_data = candidate_response_data
            break
        except Exception as exc:
            if previous_response_id is not None and "response_id" in str(exc):
                logger.warning(
                    "Responses API previous_response_id "
                    f"{previous_response_id} was not found. "
                    "Retrying with full reconstructed history."
                )
                payload, previous_response_id, input_items = build_payload(
                    use_previous_response_id=False
                )
                request_data["input"] = _format_messages_for_logging(input_items)
                request_data["previous_response_id"] = None
                continue
            if attempt >= num_retries:
                raise
            time.sleep(min(2**attempt, 5))

    assert response_data is not None, "Responses API request returned no data"

    generation_time_seconds = time.perf_counter() - start_time
    usage = _get_responses_usage(response_data)
    content = _extract_responses_output_text(response_data)
    tool_calls = _extract_responses_tool_calls(response_data)

    # tau2 half-duplex enforces one action type per turn.
    if tool_calls:
        content = None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=0.0,
        usage=usage,
        raw_data=response_data,
        generation_time_seconds=generation_time_seconds,
    )

    response_data_for_log = {
        "timestamp": datetime.now().isoformat(),
        "response_id": response_data.get("id"),
        "status": response_data.get("status"),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
        "output_item_types": [
            item.get("type") for item in response_data.get("output") or []
        ],
    }
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data_for_log, call_name=call_name)

    return message


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
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
    kwargs = kwargs.copy()
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    if _use_responses_api(kwargs):
        return _responses_request(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_name=call_name,
            **kwargs,
        )

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Prepare request data for logging
    formatted_messages = _format_messages_for_logging(litellm_messages)
    request_data = {
        "model": model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time
    cost = get_response_cost(response)
    usage = get_response_usage(response)

    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in raw_tool_calls
    ]
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

    # Log complete LLM call (request + response)
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    # Add timestamp to request data
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
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content between first { and last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    # Return original response as fallback
    return response
