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

# Default generation budget for the harness-owned /v1/completions codec path.
# vLLM's completions endpoint otherwise defaults to 16 tokens. Overridable via
# --agent-llm-args '{"max_tokens": N}'.
DEFAULT_COMPLETIONS_MAX_TOKENS = 8192

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


def _truncate_dicts_to_budget(
    messages: list[dict],
    model: str,
    max_input_tokens: int,
    tools: Optional[list] = None,
) -> tuple[list[dict], Optional[dict]]:
    """Trim oldest litellm-dict messages so the request fits ``max_input_tokens``.

    Keeps a leading system message, drops the oldest non-system messages first,
    and strips any orphaned leading ``tool`` messages (whose triggering assistant
    tool_call was dropped) so the request stays valid for the OpenAI API.

    Returns ``(messages, info)`` where ``info`` is ``None`` if nothing was
    dropped, else a dict describing the truncation (for logging/trajectory).
    """

    def count(msgs: list[dict]) -> int:
        try:
            return litellm.token_counter(model=model, messages=msgs, tools=tools)
        except TypeError:
            # Older litellm without the tools kwarg.
            return litellm.token_counter(model=model, messages=msgs)

    tokens_before = count(messages)
    if tokens_before <= max_input_tokens:
        return messages, None

    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    tail = messages[len(head) :]

    while tail and count(head + tail) > max_input_tokens:
        tail.pop(0)
        # Drop orphaned tool results whose assistant tool_call was just removed.
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)

    truncated = head + tail
    info = {
        "messages_dropped": len(messages) - len(truncated),
        "messages_kept": len(truncated),
        "max_input_tokens": max_input_tokens,
        "tokens_before": tokens_before,
        "tokens_after": count(truncated),
    }
    logger.warning(
        f"Context truncated to fit max_input_tokens={max_input_tokens}: dropped "
        f"{info['messages_dropped']} oldest message(s), {tokens_before} -> "
        f"{info['tokens_after']} tokens."
    )
    return truncated, info


def _truncate_messages_to_budget(
    messages: list[Message],
    model: str,
    max_input_tokens: int,
    tools: Optional[list] = None,
) -> tuple[list[Message], Optional[dict]]:
    """Trim a tau2 ``Message`` list to fit ``max_input_tokens``.

    Delegates to ``_truncate_dicts_to_budget`` on the 1:1 litellm-dict
    representation, then slices the original message list to the same shape
    (preserves a leading system message + a contiguous suffix).
    """
    litellm_dicts = to_litellm_messages(messages)
    truncated_dicts, info = _truncate_dicts_to_budget(
        litellm_dicts, model, max_input_tokens, tools
    )
    if info is None:
        return messages, None
    has_system = bool(messages) and isinstance(messages[0], SystemMessage)
    n_head = 1 if has_system else 0
    n_suffix = len(truncated_dicts) - n_head
    suffix = messages[-n_suffix:] if n_suffix > 0 else []
    return messages[:n_head] + suffix, info


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


def _generate_completions(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    call_name: Optional[str] = None,
    *,
    enable_thinking: bool = True,
    tokenizer_id: Optional[str] = None,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate via a vanilla /v1/completions endpoint using the Qwen3 codec.

    The harness owns BOTH prompt formatting (``render_chat``) and output
    parsing (``parse_completion``) so we can serve a plain text-in/text-out
    vLLM with NO --enable-auto-tool-choice / --tool-call-parser /
    --reasoning-parser flags. The returned object is the SAME
    ``AssistantMessage`` the native path produces, so nothing downstream
    changes.

    Args:
        enable_thinking: Qwen3 thinking switch (pulled from llm_args).
        tokenizer_id: Override the tokenizer providing the chat template.
        **kwargs: forwarded to ``litellm.text_completion`` (e.g. temperature,
            max_tokens, api_base, num_retries).
    """
    from tau2.utils import qwen3_codec

    tid = tokenizer_id or qwen3_codec.DEFAULT_TOKENIZER_ID
    tools_schema = [tool.openai_schema for tool in tools] if tools else None

    # Optional input-context cap (matches the native chat path). Use to leave
    # room for generation under a fixed --max-model-len budget, e.g.
    # --agent-llm-args '{"max_input_tokens": 6080, "max_tokens": 2048}' for an
    # 8k-served model.
    context_truncation: Optional[dict] = None
    max_input_tokens = kwargs.pop("max_input_tokens", None)
    if max_input_tokens is not None:
        messages, context_truncation = _truncate_messages_to_budget(
            messages, model, int(max_input_tokens), tools_schema
        )

    prompt = qwen3_codec.render_chat(
        messages,
        tools=tools_schema,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenizer_id=tid,
    )

    # Stop at the served model's own turn terminator (derived from its
    # tokenizer) so any model family halts correctly — Qwen3 <|im_end|>,
    # Llama-3 <|eot_id|>, etc. — not just Qwen. Merge with caller stops.
    model_stop = qwen3_codec.get_stop_tokens(tid)
    user_stop = kwargs.pop("stop", None)
    if user_stop is None:
        stop = list(model_stop)
    elif isinstance(user_stop, str):
        stop = [user_stop, *model_stop]
    else:
        stop = [*user_stop, *model_stop]

    # vLLM's /v1/completions defaults max_tokens to 16 (the legacy OpenAI
    # default), which truncates the model mid-<think> and yields an empty
    # message ("must have content or tool_calls"). The chat endpoint has no such
    # tiny default, so set a generous one for parity. Override via llm_args.
    kwargs.setdefault("max_tokens", DEFAULT_COMPLETIONS_MAX_TOKENS)

    request_data = {
        "model": model,
        "io_mode": "completions",
        "prompt": prompt.split("\n"),
        "tools": tools_schema,
        "stop": stop,
        "max_tokens": kwargs.get("max_tokens"),
        "enable_thinking": enable_thinking,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in kwargs.items()
        },
        "timestamp": datetime.now().isoformat(),
    }

    start_time = time.perf_counter()
    try:
        response = litellm.text_completion(
            model=model,
            prompt=prompt,
            stop=stop,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time

    cost = get_response_cost(response)
    usage = get_response_usage(response)

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        logger.warning("Output might be incomplete due to token limit!")

    raw_text = choice.text or ""
    parsed = qwen3_codec.parse_completion(raw_text, thinking_enabled=enable_thinking)

    message = AssistantMessage(
        role="assistant",
        content=parsed.content,
        tool_calls=parsed.tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
        generation_time_seconds=generation_time_seconds,
        context_truncation=context_truncation,
    )

    response_data = {
        "timestamp": datetime.now().isoformat(),
        "reasoning": parsed.reasoning,
        "content": parsed.content,
        "tool_calls": (
            [tc.model_dump() for tc in parsed.tool_calls]
            if parsed.tool_calls
            else None
        ),
        "raw_text": raw_text,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    _write_llm_log(request_data, response_data, call_name=call_name)

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
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Harness-owned Qwen3 codec branch: talk to a vanilla text-in/text-out
    # vLLM (/v1/completions) where the harness OWNS prompt formatting and
    # output parsing instead of relying on the server-side chat template +
    # tool/reasoning parsers. Enabled either explicitly via io_mode (in
    # llm_args) or implicitly for hosted_vllm/* models.
    io_mode = kwargs.pop("io_mode", None)
    if io_mode == "completions" or (
        io_mode is None and model.startswith("hosted_vllm/")
    ):
        return _generate_completions(
            model=model,
            messages=messages,
            tools=tools,
            call_name=call_name,
            **kwargs,
        )

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Optional hard cap on input context (e.g. via --agent-llm-args
    # '{"max_input_tokens": 8000}'). Truncates oldest messages before sending so
    # a large-context model can be evaluated under a smaller effective window.
    max_input_tokens = kwargs.pop("max_input_tokens", None)
    context_truncation: Optional[dict] = None
    if max_input_tokens is not None:
        litellm_messages, context_truncation = _truncate_dicts_to_budget(
            litellm_messages, model, int(max_input_tokens), tools_schema
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
        context_truncation=context_truncation,
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
