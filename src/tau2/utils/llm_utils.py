import json
import re
from collections import defaultdict
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

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

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


# Global error tracking for tool calls
TOOL_CALL_ERROR_COUNTS = defaultdict(int)
TOOL_CALL_ERROR_DETAILS = defaultdict(list)


def save_tool_call_error_analysis(filepath: str = "error_call_analysis.txt"):
    """
    Save the tool call error analysis to a file.
    """
    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TOOL CALL ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("ERROR TYPE DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        total_errors = sum(TOOL_CALL_ERROR_COUNTS.values())
        if total_errors == 0:
            f.write("No tool call errors recorded.\n")
        else:
            for error_type, count in sorted(
                TOOL_CALL_ERROR_COUNTS.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_errors) * 100
                f.write(f"{error_type}: {count} ({percentage:.2f}%)\n")

        f.write(f"\nTotal errors: {total_errors}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Detailed error information
        f.write("DETAILED ERROR INFORMATION:\n")
        f.write("-" * 80 + "\n\n")
        for error_type, details in sorted(TOOL_CALL_ERROR_DETAILS.items()):
            f.write(f"\n{error_type} ({len(details)} occurrences):\n")
            f.write("-" * 40 + "\n")
            for i, detail in enumerate(details[:10], 1):  # Show first 10 of each type
                f.write(f"{i}. {detail}\n")
            if len(details) > 10:
                f.write(f"... and {len(details) - 10} more occurrences\n")
            f.write("\n")

    logger.info(f"Tool call error analysis saved to {filepath}")


def get_tool_call_error_stats() -> dict:
    """
    Get current tool call error statistics.
    """
    return {
        "counts": dict(TOOL_CALL_ERROR_COUNTS),
        "total": sum(TOOL_CALL_ERROR_COUNTS.values()),
    }


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
        logger.info(e)
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

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
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
    content = response.message.content
    tool_calls = []
    for raw_call in response.message.tool_calls or []:
        raw_args = raw_call.function.arguments
        error_occurred = False
        error_type = None

        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            error_type = "JSON_DECODE_ERROR"
            error_occurred = True
            TOOL_CALL_ERROR_COUNTS[error_type] += 1
            error_detail = f"Tool: {raw_call.function.name}, Error: {str(e)}, raw_args: {raw_args[:200]}"
            TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
            logger.error(
                "Tool call error [%s]: Tool %s failed initial JSON decode. "
                "Error: %s, raw_args=%r",
                error_type,
                raw_call.function.name,
                str(e),
                raw_args[:200],
            )
            parsed_args = {}
        except Exception as e:
            error_type = "UNEXPECTED_PARSE_ERROR"
            error_occurred = True
            TOOL_CALL_ERROR_COUNTS[error_type] += 1
            error_detail = f"Tool: {raw_call.function.name}, Type: {type(e).__name__}, Error: {str(e)}"
            TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
            logger.error(
                "Tool call error [%s]: Tool %s failed with unexpected error. "
                "Error type: %s, Error: %s, raw_args=%r",
                error_type,
                raw_call.function.name,
                type(e).__name__,
                str(e),
                raw_args[:200],
            )
            parsed_args = {}

        if isinstance(parsed_args, str):
            error_type = "DOUBLE_ENCODED_STRING"
            error_occurred = True
            TOOL_CALL_ERROR_COUNTS[error_type] += 1
            error_detail = f"Tool: {raw_call.function.name}, parsed_args: {parsed_args[:200]}"
            TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
            logger.warning(
                "Tool call error [%s]: Tool %s returned string instead of dict after first decode. "
                "Attempting second decode. parsed_args=%r",
                error_type,
                raw_call.function.name,
                parsed_args[:200],
            )
            try:
                parsed_args = json.loads(parsed_args)
                logger.info(
                    "Tool call recovery [DOUBLE_DECODE_SUCCESS]: Tool %s successfully decoded after second attempt.",
                    raw_call.function.name,
                )
            except json.JSONDecodeError as e:
                error_type = "DOUBLE_DECODE_FAILURE"
                TOOL_CALL_ERROR_COUNTS[error_type] += 1
                error_detail = f"Tool: {raw_call.function.name}, Error: {str(e)}, parsed_args: {parsed_args[:200]}"
                TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
                logger.error(
                    "Tool call error [%s]: Tool %s failed second JSON decode. "
                    "Using empty dict. Error: %s, parsed_args=%r",
                    error_type,
                    raw_call.function.name,
                    str(e),
                    parsed_args[:200],
                )
                parsed_args = {}
        elif isinstance(parsed_args, dict):
            pass
        elif parsed_args is None:
            error_type = "NULL_ARGUMENTS"
            error_occurred = True
            TOOL_CALL_ERROR_COUNTS[error_type] += 1
            error_detail = f"Tool: {raw_call.function.name}"
            TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
            logger.warning(
                "Tool call error [%s]: Tool %s returned null/None arguments. Using empty dict.",
                error_type,
                raw_call.function.name,
            )
            parsed_args = {}
        else:
            error_type = "UNEXPECTED_TYPE"
            error_occurred = True
            TOOL_CALL_ERROR_COUNTS[error_type] += 1
            error_detail = f"Tool: {raw_call.function.name}, Type: {type(parsed_args).__name__}, Value: {str(parsed_args)[:200]}"
            TOOL_CALL_ERROR_DETAILS[error_type].append(error_detail)
            logger.error(
                "Tool call error [%s]: Tool %s returned unexpected arguments type. "
                "Type: %s, Using empty dict. parsed_args=%r",
                error_type,
                raw_call.function.name,
                type(parsed_args).__name__,
                parsed_args,
            )
            parsed_args = {}

        tool_calls.append(
            ToolCall(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=parsed_args,
            )
        )

    # Convert empty tool_calls list to None to match expected behavior
    if not tool_calls:
        tool_calls = None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
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


def reset_tool_call_error_tracking():
    """
    Reset the tool call error tracking counters.
    """
    TOOL_CALL_ERROR_COUNTS.clear()
    TOOL_CALL_ERROR_DETAILS.clear()
    logger.info("Tool call error tracking has been reset")
