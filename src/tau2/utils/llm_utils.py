import json
import logging
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger
from pydantic import BaseModel, ValidationError

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
# LiteLLM 1.81's Responses bridge looks up GPT-5 capabilities on the stripped
# "responses/<model>" string with no provider; get_llm_provider then prints a
# "Provider List" banner (twice per call) before the lookup fails and is
# swallowed. generate() repairs that lookup's one consequence (tool_choice, see
# _RESPONSES_BRIDGE_PARAMS); this only silences LiteLLM's console banners.
# Exceptions are still raised and logged.
litellm.suppress_debug_info = True

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
    litellm.disable_cache()


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
    model = response.model.removeprefix("openai/").removeprefix("responses/")
    if (
        model in ("gpt-5.5", "gpt-5.5-2026-04-23")
        and response.get("service_tier") in (None, "default")
        and response.usage is not None
    ):
        # The pinned LiteLLM map lacks GPT-5.5 and its 272K pricing boundary.
        # Standard rates: https://developers.openai.com/api/docs/models/gpt-5.5
        usage = response.usage
        cached = (
            usage.prompt_tokens_details.cached_tokens or 0
            if usage.prompt_tokens_details
            else 0
        )
        input_cost = ((usage.prompt_tokens - cached) * 5 + cached * 0.5) / 1e6
        output_cost = usage.completion_tokens * 30 / 1e6
        if usage.prompt_tokens > 272_000:
            input_cost *= 2
            output_cost *= 1.5
        return input_cost + output_cost
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


def _input_audio_format(message: UserMessage) -> str:
    """Return the container format, preferring WAV/MP3 magic over raw encoding."""
    b64 = message.audio_content or ""
    if b64.startswith("UklG"):  # base64 of b"RIFF" — WAV container
        return "wav"
    if b64.startswith("SUQz"):  # base64 of b"ID3" — MP3 container
        return "mp3"
    if message.audio_format is not None:
        return message.audio_format.encoding.value
    return "wav"


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.

    A UserMessage carrying ``audio_content`` becomes multipart content
    (text part + ``input_audio`` part) so multimodal judges can send audio
    through the single ``generate()`` seam.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            if message.audio_content:
                content: list[dict] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": message.audio_content,
                            "format": _input_audio_format(message),
                        },
                    }
                )
                litellm_messages.append({"role": "user", "content": content})
            else:
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
        content = msg_copy.get("content")
        if isinstance(content, str):
            # Split content on newlines for better readability
            content_lines = content.split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        elif isinstance(content, list):
            # Multipart content: never log raw audio payloads (huge base64).
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    audio = part.get("input_audio") or {}
                    data = audio.get("data") or ""
                    parts.append(
                        {
                            **part,
                            "input_audio": {
                                **audio,
                                "data": f"<audio: {len(data)} b64 chars>",
                            },
                        }
                    )
                else:
                    parts.append(part)
            msg_copy["content"] = parts
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


ANTHROPIC_ADAPTIVE_THINKING_MODEL_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
)
_ANTHROPIC_OUTPUT_CONFIG_EFFORTS = frozenset({"low", "medium", "high"})


def _openai_responses_bridge_model(model: str) -> str:
    """Route OpenAI models through LiteLLM Responses; preserve other providers."""
    if "/" not in model:
        return f"openai/responses/{model}"
    provider, rest = model.split("/", 1)
    if provider == "openai":
        if rest.startswith("responses/"):
            return model  # already bridged
        return f"openai/responses/{rest}"
    return model  # non-OpenAI provider — leave as-is


# Params the bridge drops because its failed capability lookup reports them
# unsupported, although the Responses API accepts them.
_RESPONSES_BRIDGE_PARAMS = ("tool_choice",)


def _keep_non_leading_system_messages_positional(
    litellm_messages: list[dict],
) -> list[dict]:
    """Preserve system annotations in Responses input instead of hoisting them."""
    first_non_system = next(
        (i for i, m in enumerate(litellm_messages) if m.get("role") != "system"),
        None,
    )
    rewrapped = []
    for i, message in enumerate(litellm_messages):
        keep_positional = (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and (
                i > first_non_system
                if first_non_system is not None
                else (i > 0 or len(litellm_messages) == 1)
            )
        )
        if keep_positional:
            message = {
                **message,
                "content": [{"type": "text", "text": message["content"]}],
            }
        rewrapped.append(message)
    return rewrapped


def _uses_anthropic_adaptive_thinking(model: str) -> bool:
    """Identify models requiring adaptive rather than budgeted thinking."""
    base_model = model.split("/")[-1]
    return base_model.startswith(ANTHROPIC_ADAPTIVE_THINKING_MODEL_PREFIXES)


def _apply_anthropic_adaptive_thinking(model: str, kwargs: dict[str, Any]) -> None:
    """Translate portable reasoning effort without overriding explicit thinking."""
    if not _uses_anthropic_adaptive_thinking(model):
        return
    if kwargs.get("reasoning_effort") is None:
        return
    effort = kwargs.pop("reasoning_effort")
    if "thinking" in kwargs:
        return
    kwargs["thinking"] = {"type": "adaptive"}
    if isinstance(effort, str) and effort in _ANTHROPIC_OUTPUT_CONFIG_EFFORTS:
        kwargs.setdefault("output_config", {"effort": effort})


class LLMResponseValidationError(ValueError):
    """A model reply remained invalid after response validation retries."""

    def __init__(self, model: str, raw_responses: list[str], cause: Exception):
        self.model = model
        self.raw_responses = raw_responses
        super().__init__(f"Invalid {model} response: {cause}")


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    response_model: type[BaseModel] | None = None,
    response_validation_retries: int = 0,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """Generate a reply, retrying invalid structured responses in this boundary.

    The response model validates an extracted JSON object. On a validation
    failure, the model sees its prior reply and one short correction request.
    All invalid replies are retained on the raised exception for diagnosis.
    """
    if response_validation_retries < 0:
        raise ValueError("response_validation_retries must be nonnegative")
    if response_validation_retries and response_model is None:
        raise ValueError("response_validation_retries requires response_model")

    attempt_messages = list(messages)
    raw_responses: list[str] = []
    total_cost = 0.0
    has_cost = False
    for attempt in range(response_validation_retries + 1):
        reply = _generate_once(
            model,
            attempt_messages,
            tools=tools,
            tool_choice=tool_choice,
            call_name=call_name,
            **kwargs,
        )
        if response_model is None:
            return reply
        if reply.cost is not None:
            total_cost += reply.cost
            has_cost = True
        raw_responses.append(reply.content or "")
        try:
            response_model.model_validate_json(
                extract_json_from_llm_response(reply.content or "")
            )
            if has_cost:
                reply.cost = total_cost
            return reply
        except (ValidationError, ValueError) as exc:
            logger.warning(
                f"Invalid structured response for {call_name or model} "
                f"(attempt {attempt + 1}/{response_validation_retries + 1}): {exc}"
            )
            if attempt == response_validation_retries:
                raise LLMResponseValidationError(
                    response_model.__name__, raw_responses, exc
                ) from exc
            attempt_messages.extend(
                [
                    reply,
                    UserMessage(
                        role="user",
                        content=(
                            "Your previous reply was invalid JSON or did not match "
                            "the requested schema. Return only a corrected JSON "
                            "object. Escape quotation marks and newlines in strings."
                        ),
                    ),
                ]
            )
    raise AssertionError("response validation loop did not return or raise")


def _generate_once(
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
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # The pinned LiteLLM release rejects (or silently drops) xhigh for GPT-5.5.
    # Pass it through the Responses API body so the requested effort reaches
    # OpenAI, including user-simulator turns that use function tools.
    if (
        model.split("/")[-1].startswith("gpt-5.5")
        and kwargs.get("reasoning_effort") == "xhigh"
    ):
        if "/" not in model or model.startswith("openai/"):
            kwargs = dict(kwargs)
            kwargs.pop("reasoning_effort")
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            if "reasoning" in extra_body:
                raise ValueError("Conflicting GPT-5.5 reasoning settings")
            extra_body["reasoning"] = {"effort": "xhigh"}
            kwargs["extra_body"] = extra_body

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    api_model = (
        _openai_responses_bridge_model(model)
        if model.split("/")[-1].startswith("gpt-5")
        else model
    )
    _apply_anthropic_adaptive_thinking(model, kwargs)
    litellm_messages = to_litellm_messages(messages)
    if api_model.startswith("openai/responses/"):
        litellm_messages = _keep_non_leading_system_messages_positional(
            litellm_messages
        )
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"
    if api_model.startswith("openai/responses/") and tool_choice is not None:
        kwargs["allowed_openai_params"] = list(
            dict.fromkeys(
                [
                    *(kwargs.get("allowed_openai_params") or []),
                    *_RESPONSES_BRIDGE_PARAMS,
                ]
            )
        )

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
            model=api_model,
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


def get_cost(messages: list[Message]) -> tuple[float | None, float | None]:
    """
    Get the (agent_cost, user_cost) of the interaction.

    Each side is computed independently: a side is None if any of its
    messages has no cost. This way an uncosted agent message (e.g. an
    audio-native provider without usage reporting) doesn't discard the
    user side's cost, and vice versa.
    """
    agent_cost: float | None = 0.0
    user_cost: float | None = 0.0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AssistantMessage):
            if message.cost is None:
                logger.warning(f"Agent message has no cost: {message.content}")
                agent_cost = None
            elif agent_cost is not None:
                agent_cost += message.cost
        elif isinstance(message, UserMessage):
            if message.cost is None:
                logger.warning(f"User message has no cost: {message.content}")
                user_cost = None
            elif user_cost is not None:
                user_cost += message.cost
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
