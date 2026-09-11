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
    DEFAULT_GPT5_REASONING_EFFORT,
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


# Models whose missing litellm cost-map entry we've already logged, so the
# "isn't mapped yet" notice fires once per model instead of on every LLM call.
_UNMAPPED_COST_MODELS_SEEN: set[str] = set()


def get_response_cost(response: ModelResponse) -> Optional[float]:
    """
    Get the cost of the response from the litellm completion.

    Returns None when the cost is UNKNOWN — distinct from a known cost of 0.0.
    When the model has no entry in litellm's cost map (common for newly
    released or vendor-prefixed models, e.g. ``us/claude-opus-4-8``),
    ``completion_cost`` raises "This model isn't mapped yet". That is expected
    and harmless — it just means we can't compute a dollar cost — so we degrade
    gracefully: log it once at WARNING and return None, instead of an ERROR on
    every single call (which floods logs) or a fake 0.0 (which masquerades as
    a real, free cost in sums and averages). Genuinely unexpected failures are
    logged at ERROR and also return None (the cost is unknown either way).
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        if "isn't mapped yet" in str(e) or "not mapped yet" in str(e):
            model = response.model
            if model not in _UNMAPPED_COST_MODELS_SEEN:
                _UNMAPPED_COST_MODELS_SEEN.add(model)
                logger.warning(
                    f"No litellm cost-map entry for model '{model}'; cost is "
                    "unknown and reported as None (this notice is logged once "
                    "per model)."
                )
            return None
        logger.error(e)
        return None
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
    """litellm ``input_audio.format`` string for a message's audio payload.

    litellm wants a container/file format ("wav", "mp3", ...), not a raw PCM
    encoding. Audio prepared for LLM consumption in this repo is a WAV
    container (see ``tau2.judges.delivery.audio``), so sniff the base64 magic first
    and only fall back to the message's declared raw encoding.
    """
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


# ---------------------------------------------------------------------------
# Anthropic adaptive-thinking translation
# ---------------------------------------------------------------------------
#
# Newer Anthropic models (the "adaptive thinking" generation, starting with
# Claude Opus 4.6) reject the legacy `thinking={"type": "enabled", ...}` request
# shape. They require `thinking={"type": "adaptive"}`, optionally with
# `output_config={"effort": "low"|"medium"|"high"}`, to control thinking budget.
#
# The pinned litellm (1.81.x) only emits the adaptive shape for models whose id
# contains "opus-4-6" (see AnthropicConfig._map_reasoning_effort); for every
# other Anthropic model it still translates `reasoning_effort` into the legacy
# `thinking.type.enabled`, which these newer models 400 on. Rather than depend
# on a litellm version that recognizes each new model id (and still wouldn't
# cover ids litellm hasn't shipped yet, e.g. internal/preview names), we
# translate `reasoning_effort` into the adaptive shape ourselves for the models
# we know need it, and hand litellm an explicit `thinking=` (which it forwards
# verbatim instead of re-deriving from `reasoning_effort`).
#
# MAINTENANCE: this is a single source of truth. When a new adaptive-thinking
# Claude model ships, add its base-name prefix here. Models NOT listed keep
# litellm's default `reasoning_effort` handling (legacy thinking budgets for
# older Claude, native reasoning_effort for gpt-5, etc.), so adding an entry is
# the only action ever required and there is no risk to unlisted models.
ANTHROPIC_ADAPTIVE_THINKING_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
)

# Effort levels Anthropic's `output_config.effort` accepts. Anything else
# (e.g. "minimal", "none") must NOT be forwarded — Anthropic 400s on it — so we
# fall back to bare adaptive thinking (no effort hint) in that case.
_ANTHROPIC_OUTPUT_CONFIG_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high"})


def _openai_responses_bridge_model(model: str) -> str:
    """Rewrite an OpenAI gpt-5 model id to litellm's Responses-API bridge form.

    litellm routes ``completion()`` through the OpenAI Responses API when the
    model id is ``openai/responses/<name>`` (see
    ``litellm.responses_api_bridge_check``). Reasoning models (gpt-5 series) are
    served there and, unlike /v1/chat/completions, the Responses API accepts
    ``reasoning_effort`` together with function tools and ``response_format``.

    Only OpenAI ids (bare, e.g. ``gpt-5.5``, or ``openai/``-prefixed) are
    bridged; other providers (``azure/``, ``vertex_ai/``, ...) are returned
    unchanged, as is an already-bridged id.
    """
    if "/" not in model:
        return f"openai/responses/{model}"
    provider, rest = model.split("/", 1)
    if provider == "openai":
        if rest.startswith("responses/"):
            return model  # already bridged
        return f"openai/responses/{rest}"
    return model  # non-OpenAI provider — leave as-is


def _keep_non_leading_system_messages_positional(
    litellm_messages: list[dict],
) -> list[dict]:
    """Rewrap system messages so litellm's Responses-API bridge keeps them in place.

    The bridge hoists every STRING-content system message into the request-level
    ``instructions`` field. That destroys the position of mid-conversation system
    messages (silence annotations, role reminders), and when a history contains
    no user/assistant message at all — e.g. the user simulator's first voice turn
    (system prompt + silence annotation, nobody has spoken yet) — it leaves the
    required ``input`` empty and OpenAI rejects the request with
    ``missing_required_parameter: input``.

    The bridge keeps a system message positional when its content is a
    content-part LIST instead of a bare string, so: the leading system block
    (the actual system prompt) is left as strings and hoisted to
    ``instructions``; every system message after the first non-system message is
    rewrapped to ``[{"type": "text", ...}]`` and keeps its place in ``input``.
    If ALL messages are system, everything after the first is rewrapped (a
    single all-system message is rewrapped too) so ``input`` is never empty.
    """
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
    """Whether ``model`` is an Anthropic model that requires the adaptive
    `thinking={"type": "adaptive"}` request shape instead of the legacy
    `thinking={"type": "enabled", ...}` that older litellm emits from
    `reasoning_effort`. Provider-prefix agnostic (matches the base model id)."""
    base_model = model.split("/")[-1]
    return base_model.startswith(ANTHROPIC_ADAPTIVE_THINKING_MODEL_PREFIXES)


def _apply_anthropic_adaptive_thinking(model: str, kwargs: dict[str, Any]) -> None:
    """Translate a portable ``reasoning_effort`` into the Anthropic adaptive
    thinking request shape, in place, for models that need it.

    Mutates ``kwargs``: pops ``reasoning_effort`` and sets
    ``thinking={"type": "adaptive"}`` plus ``output_config={"effort": ...}``
    when the effort is one Anthropic accepts. A caller that already passed an
    explicit ``thinking`` is left untouched (explicit wins). Popping
    ``reasoning_effort`` is deliberate: otherwise litellm would re-derive the
    legacy `thinking.type.enabled` shape from it and these models would 400.
    """
    if not _uses_anthropic_adaptive_thinking(model):
        return
    if kwargs.get("reasoning_effort") is None:
        return
    # Always remove reasoning_effort for these models: left in place, litellm
    # re-derives the legacy `thinking.type.enabled` shape from it, which these
    # models 400 on — even when an explicit `thinking` payload is also present.
    effort = kwargs.pop("reasoning_effort")
    # A caller that already passed an explicit `thinking` wins; we only needed to
    # strip reasoning_effort so litellm can't re-derive the legacy shape.
    if "thinking" in kwargs:
        return
    kwargs["thinking"] = {"type": "adaptive"}
    # Only forward a known-good effort hint; bare adaptive thinking is valid and
    # is the correct fallback for "minimal"/"none"/unrecognized values.
    if isinstance(effort, str) and effort in _ANTHROPIC_OUTPUT_CONFIG_EFFORTS:
        kwargs.setdefault("output_config", {"effort": effort})


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

    # GPT-5 series are reasoning models: they reject `temperature` (litellm drops
    # it via drop_params) and instead take `reasoning_effort`. Default to fast,
    # low-latency behavior matching the prior gpt-4.1 non-reasoning defaults
    # unless the caller explicitly set an effort.
    is_gpt5 = model.split("/")[-1].startswith("gpt-5")
    if is_gpt5 and kwargs.get("reasoning_effort") is None:
        kwargs["reasoning_effort"] = DEFAULT_GPT5_REASONING_EFFORT

    # OpenAI serves the gpt-5 reasoning models through the Responses API. The
    # legacy /v1/chat/completions endpoint rejects `reasoning_effort` together
    # with function tools ("Function tools with reasoning_effort are not
    # supported ... use /v1/responses instead"). litellm bridges completion() to
    # the Responses API when the model id is prefixed `openai/responses/...`,
    # which supports reasoning_effort WITH tools (and with response_format), and
    # still returns a normal ModelResponse with usage/cost intact. So route
    # OpenAI gpt-5 calls through that bridge rather than dropping reasoning. The
    # original `model` is kept for logging; only the API call uses the bridged id.
    api_model = _openai_responses_bridge_model(model) if is_gpt5 else model

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    # Newer Anthropic models reject the legacy thinking shape litellm derives
    # from `reasoning_effort`; translate to adaptive thinking for those models.
    _apply_anthropic_adaptive_thinking(model, kwargs)

    litellm_messages = to_litellm_messages(messages)
    if api_model.startswith("openai/responses/"):
        litellm_messages = _keep_non_leading_system_messages_positional(
            litellm_messages
        )
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

    is_responses_bridge = api_model.startswith("openai/responses/")
    # Chat Completions choices are alternatives, so consume only the first.
    # LiteLLM's Responses bridge instead maps output items to choices, which can
    # split one assistant turn's text and function calls across several choices.
    response_choices = response.choices if is_responses_bridge else response.choices[:1]
    if not response_choices:
        raise ValueError("The response should contain at least one choice")
    try:
        for response_choice in response_choices:
            if response_choice.finish_reason == "length":
                logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert all(choice.message.role == "assistant" for choice in response_choices), (
        "The response should contain only assistant messages"
    )
    if is_responses_bridge:
        raw_tool_calls = [
            tool_call
            for choice in response_choices
            for tool_call in (choice.message.tool_calls or [])
        ]
        if raw_tool_calls:
            # Tau2's half-duplex contract permits text or tool calls, never both.
            content = None
        else:
            content_parts = [
                choice.message.content
                for choice in response_choices
                if choice.message.content is not None
            ]
            content = "".join(content_parts) if content_parts else None
    else:
        response_choice = response_choices[0]
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
