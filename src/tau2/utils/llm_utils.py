import json
import os
import re
from typing import Any

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
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

# Register custom models for cost tracking (Datadog enhancement)
# Allows tracking costs for self-hosted models (e.g., Nebius API)
# Format: {"model_name": {"input_cost_per_token": X, "output_cost_per_token": Y, ...}}
TAU2_LLM_MODELS_ENV = os.getenv("TAU2_LLM_MODELS")
if TAU2_LLM_MODELS_ENV:
    try:
        custom_models = json.loads(TAU2_LLM_MODELS_ENV)
        litellm.register_model(custom_models)
        logger.info(f"Registered {len(custom_models)} custom model(s) for cost tracking")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse TAU2_LLM_MODELS: {e}")
    except Exception as e:
        logger.warning(f"Failed to register custom models: {e}")

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


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """Parse the base model name from a LiteLLM fine-tuned model string.

    Args:
        model: A fine-tuned model identifier (e.g., "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg").

    Returns:
        Base model name extracted from the identifier, or the original string if not a fine-tuned model.
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    return model


def get_response_cost(response: ModelResponse) -> float:
    """Calculate the cost of an LLM response.

    Args:
        response: A LiteLLM ModelResponse object from completion() call.

    Returns:
        Cost in USD as a float. Returns 0.0 if cost cannot be determined.
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


def get_response_usage(response: ModelResponse) -> dict | None:
    """Extract token usage from an LLM response.

    Args:
        response: A LiteLLM ModelResponse object from completion() call.

    Returns:
        Dict with "completion_tokens" and "prompt_tokens" keys, or None if usage
        information is not available.
    """
    usage: Usage | None = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """Convert message dictionaries to Tau2 message objects.

    Args:
        messages: List of message dicts with at least a "role" key.
        ignore_roles: Set of role names to skip (e.g., {"system"}).

    Returns:
        List of typed message objects (UserMessage, AssistantMessage, ToolMessage, SystemMessage).

    Raises:
        ValueError: If an unknown role is encountered.
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
    """Convert Tau2 message objects to LiteLLM-compatible message dicts.

    Tool calls are serialized to JSON strings as required by the OpenAI API format.

    Args:
        messages: List of Tau2 message objects.

    Returns:
        List of message dicts compatible with litellm.completion().
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


def generate(
    model: str,
    messages: list[Message],
    tools: list[Tool] | None = None,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate a response from an LLM using LiteLLM.

    Sends messages to the specified model and returns a structured response with
    parsed tool calls, cost tracking, and token usage. Disables Sonnet thinking
    if not explicitly enabled.

    Args:
        model: Model identifier (e.g., "gpt-4-turbo", "claude-3-sonnet").
        messages: Conversation history as Tau2 message objects.
        tools: Optional list of tools the model can call.
        tool_choice: How to handle tool calling ("auto", "required", or None).
        **kwargs: Additional LiteLLM parameters (temperature, max_tokens, etc.).
            Defaults num_retries to DEFAULT_MAX_RETRIES if not provided.

    Returns:
        AssistantMessage with parsed tool calls, cost, and usage information.

    Raises:
        Exception: If LiteLLM completion fails or output is unparseable.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"

    response = completion(
        model=model,
        messages=litellm_messages,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )

    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]

    if response.finish_reason == "length":
        logger.warning("Output might be incomplete due to token limit!")

    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ] or None

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """Calculate total interaction cost split by agent and user.

    Args:
        messages: List of messages (ToolMessages are skipped).

    Returns:
        Tuple of (agent_cost, user_cost) in USD. Returns None if any message
        lacks cost information.
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
    """Aggregate token usage across conversation messages.

    Args:
        messages: List of messages (ToolMessages are skipped).

    Returns:
        Dict with "completion_tokens" and "prompt_tokens" keys, summed across
        all messages with available usage information.
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
