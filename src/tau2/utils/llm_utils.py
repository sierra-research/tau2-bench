import json
import os
import re
from typing import Any, Optional

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

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


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


def get_thinking_allowed() -> bool:
    return os.getenv("ALLOW_THINKING", "false").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


if not get_thinking_allowed():
    logger.info("Thinking is disabled")


def get_litellm_debug() -> bool:
    """
    Check if litellm debug mode is enabled.
    """
    return os.getenv("LITELLM_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


if get_litellm_debug():
    logger.info("LiteLLM debug mode is enabled")
    litellm._turn_on_debug()


def get_litellm_params_drop() -> bool:
    """
    Check if litellm params drop mode is enabled.
    """
    return os.getenv("LITELLM_PARAMS_DROP", "false").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


if get_litellm_params_drop():
    logger.info("LiteLLM params drop mode is enabled")
    litellm.drop_params = get_litellm_params_drop()


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


def safe_json_load(s):
    try:
        return json.loads(s) if s else None
    except json.JSONDecodeError:
        return s


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
                    "content": safe_json_load(message.content),
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


def handle_reasoning_blocks_for_claude_models(response: Message):
    text_content = response.message.content
    thinking_blocks = response.message.provider_specific_fields.get("thinking_blocks", [])

    kombinierter_content = []
    if thinking_blocks:
        kombinierter_content.extend(thinking_blocks)

    if text_content:
        kombinierter_content.append({"type": "text", "text": text_content})

    content = json.dumps(kombinierter_content) if kombinierter_content else None

    return content


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    custom_api_base = os.getenv("CUSTOM_API_BASE", None)
    custom_api_key = os.getenv("CUSTOM_API_KEY", None)

    sonnet_max_tokens = os.getenv("SONNET_MAX_TOKENS", 64000)
    opus_max_tokens = os.getenv("OPUS_MAX_TOKENS", 32000)
    anthropic_thinking_budget = os.getenv("ANTHROPIC_THINKING_BUDGET", 1024)
    gpt5_reasoning_effort = os.getenv("GPT5_REASONING_EFFORT", "low")
    gpt5_verbosity_level = os.getenv("GPT5_VERBOSITY", "low")

    allow_thinking = get_thinking_allowed()

    # This approach is needed to avoid collision with the judge o4-mini running over the true OpenAI base/key
    if model.startswith("custom_openai/"):
        kwargs["api_key"] = custom_api_key
        kwargs["api_base"] = custom_api_base

        if "glm-4" in model:
            # Extra Body because working with direct thinking is not supported by litellm
            # https://github.com/BerriAI/litellm/issues/11185
            # kwargs["allowed_openai_params"] = ['thinking']
            # kwargs["thinking"] = {"type": "disabled" / "enabled"}
            if not allow_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    if model.startswith("anthropic/claude"):
        if "sonnet-" in model:
            kwargs["max_tokens"] = sonnet_max_tokens
        elif "opus-" in model:
            kwargs["max_tokens"] = opus_max_tokens

        if not allow_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        else:
            kwargs["temperature"] = 1.0
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": anthropic_thinking_budget}

    if model.startswith("openai/gpt-5"):
        kwargs["temperature"] = 1.0
        if not allow_thinking:
            # we can not really disable thinking for GPT-5, but we can set the reasoning effort to low
            kwargs["reasoning_effort"] = "low"
            kwargs["verbosity"] = "low"
        else:
            kwargs["reasoning_effort"] = gpt5_reasoning_effort
            kwargs["verbosity"] = gpt5_verbosity_level


    litellm_messages = to_litellm_messages(messages)

    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )

    # Handle Reasoning Blocks for Claude models, because need to resupply these in following conversations
    if model.startswith("anthropic/claude") and allow_thinking and response.message.provider_specific_fields:
        content = handle_reasoning_blocks_for_claude_models(response)
    else:
        content = response.message.content or None

    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )

    # Prevent error in benchmarks runs, better make the benchmark fail in that step instead of end the benchmark
    # Needed because open source modell sometimes stuck in reasoning, what would be a failure in the benchmark
    # but not an exception that should be raised
    try:
        message.validate()
    except ValueError as e:
        message.content = "noop"

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
