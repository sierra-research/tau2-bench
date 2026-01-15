"""Message translation utilities between tau2-bench and A2A protocol formats."""

import json
import re
import uuid

from loguru import logger

from tau2.a2a.exceptions import A2AMessageError
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

A2A_AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.""".strip()


def format_tools_as_text(tools: list[Tool]) -> str:
    """
    Convert tau2 Tools to text description for A2A agent consumption.

    Args:
        tools: List of tau2 Tool objects

    Returns:
        Text representation of tools in a format A2A agents can understand
    """
    if not tools:
        logger.trace("No tools to format for A2A message")
        return ""

    logger.trace(
        "Formatting tools as text for A2A agent",
        num_tools=len(tools),
        tool_names=[tool.name for tool in tools],
    )

    lines = ["<available_tools>"]

    for tool in tools:
        schema = tool.openai_schema
        func_schema = schema.get("function", {})
        name = func_schema.get("name", tool.name)
        description = func_schema.get("description", "No description available")
        parameters = func_schema.get("parameters", {})

        param_parts = []
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "any")
            param_parts.append(f"{param_name}: {param_type}")

        signature = f"{name}({', '.join(param_parts)})"
        lines.append(f"- {signature}")
        lines.append(f"  Description: {description}")

        if properties:
            lines.append("  Parameters:")
            for param_name, param_schema in properties.items():
                param_type = param_schema.get("type", "any")
                param_desc = param_schema.get("description", "No description")
                is_required = param_name in required
                required_str = "required" if is_required else "optional"
                lines.append(
                    f"    - {param_name} ({param_type}, {required_str}): {param_desc}"
                )

        lines.append("")

    lines.append("</available_tools>")

    tool_text = "\n".join(lines)

    # Debug: Log full tool description sent to A2A agent
    logger.trace(
        "Tool descriptions formatted for A2A agent",
        tool_text_length=len(tool_text),
        tool_text=tool_text,
    )

    return tool_text


def format_system_context(
    domain_policy: str,
    tools: list[Tool] | None = None,
) -> str:
    """
    Format the system context for A2A agents.

    Args:
        domain_policy: The domain-specific policy text
        tools: Optional list of tools available to the agent

    Returns:
        Formatted system context string with instructions, policy, and tools
    """
    parts = []

    parts.append("<instructions>")
    parts.append(A2A_AGENT_INSTRUCTION)
    parts.append("</instructions>")

    parts.append("")
    parts.append("<policy>")
    parts.append(domain_policy)
    parts.append("</policy>")

    # Tools section
    if tools:
        parts.append("")
        parts.append(format_tools_as_text(tools))

    # Tool call format instruction
    parts.append("")
    parts.append(
        'To use a tool, respond with JSON: {"tool_call": {"name": "tool_name", "arguments": {"param1": "value"}}}'
    )

    context = "\n".join(parts)

    logger.trace(
        "Formatted system context for A2A agent",
        context_length=len(context),
        has_policy=bool(domain_policy),
        num_tools=len(tools) if tools else 0,
    )

    return context


def tau2_to_a2a_message_content(
    message: UserMessage | AssistantMessage | ToolMessage,
    tools: list[Tool] | None = None,
    domain_policy: str | None = None,
    is_first_message: bool = False,
) -> str:
    """
    Convert tau2 message to A2A text content.

    Args:
        message: tau2 message object
        tools: Optional list of tools to include in user messages
        domain_policy: Optional domain policy to include (typically on first message)
        is_first_message: Whether this is the first message in the conversation

    Returns:
        Text content for A2A message
    """
    if isinstance(message, UserMessage):
        # User messages: include content and system context
        content_parts = []

        if is_first_message and domain_policy:
            logger.debug(
                "Including system context in first user message",
                has_policy=bool(domain_policy),
                num_tools=len(tools) if tools else 0,
            )
            system_context = format_system_context(domain_policy, tools)
            content_parts.append(system_context)
            content_parts.append("")  # Separator

        if message.content:
            content_parts.append(message.content)

        # On subsequent messages, just include tools (agent may need reminder)
        if not is_first_message and tools:
            logger.debug(
                "Including tool descriptions in user message",
                num_tools=len(tools),
            )
            tool_text = format_tools_as_text(tools)
            if tool_text:
                content_parts.append("")
                content_parts.append(tool_text)
                content_parts.append(
                    'To use a tool, respond with JSON: {"tool_call": {"name": "tool_name", "arguments": {"param1": "value"}}}'
                )

        return "\n".join(content_parts)

    if isinstance(message, AssistantMessage):
        # Assistant messages: either text content or tool calls
        if message.has_text_content():
            return message.content or ""
        if message.is_tool_call() and message.tool_calls:
            # Convert tool calls to JSON format for A2A
            tool_calls_data = []
            for tool_call in message.tool_calls:
                tool_calls_data.append(
                    {
                        "tool_call": {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    }
                )
            # Return as JSON string
            if len(tool_calls_data) == 1:
                return json.dumps(tool_calls_data[0])
            return json.dumps({"tool_calls": tool_calls_data})
        return ""

    # Tool messages: return the tool output
    prefix = f"Tool result (id={message.id}):"
    if message.error:
        return f"{prefix} ERROR: {message.content or 'Unknown error'}"
    return f"{prefix} {message.content or ''}"


def _extract_json_from_markdown(content: str) -> str:
    """
    Extract JSON content from markdown code blocks if present.

    Handles common formats from LLMs:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - Raw JSON without code blocks

    Args:
        content: Raw content that may contain markdown code blocks

    Returns:
        Extracted JSON string, or original content if no code block found
    """
    # Pattern to match JSON in code blocks (with or without language specifier)
    # Matches: ```json\n{...}\n``` or ```\n{...}\n```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    match = re.search(code_block_pattern, content.strip())
    if match:
        extracted = match.group(1).strip()
        logger.trace(
            "Extracted JSON from markdown code block",
            original_length=len(content),
            extracted_length=len(extracted),
        )
        return extracted
    return content.strip()


def parse_a2a_tool_calls(content: str) -> list[ToolCall] | None:
    """
    Parse tool calls from A2A agent response content.

    Looks for JSON-formatted tool calls in the response:
    - Single: {"tool_call": {"name": "...", "arguments": {...}}}
    - Multiple: {"tool_calls": [{...}, {...}]}

    Also handles JSON wrapped in markdown code blocks:
    - ```json\n{"tool_call": {...}}\n```

    Args:
        content: A2A agent response content

    Returns:
        List of ToolCall objects if found, None otherwise
    """
    if not content or not content.strip():
        return None

    # Extract JSON from markdown code blocks if present
    json_content = _extract_json_from_markdown(content)

    try:
        # Try to parse as JSON
        data = json.loads(json_content)

        # Handle single tool call format
        if "tool_call" in data:
            tool_data = data["tool_call"]
            tool_call = ToolCall(
                id=tool_data.get("id", str(uuid.uuid4())),
                name=tool_data["name"],
                arguments=tool_data["arguments"],
                requestor="assistant",
            )
            return [tool_call]

        # Handle multiple tool calls format
        if "tool_calls" in data:
            tool_calls = []
            for tool_data in data["tool_calls"]:
                if "tool_call" in tool_data:
                    tc = tool_data["tool_call"]
                    tool_call = ToolCall(
                        id=tc.get("id", str(uuid.uuid4())),
                        name=tc["name"],
                        arguments=tc["arguments"],
                        requestor="assistant",
                    )
                    tool_calls.append(tool_call)
            return tool_calls if tool_calls else None

        # Not a tool call, just regular content
        return None

    except json.JSONDecodeError:
        # Not JSON, treat as regular content
        return None
    except (KeyError, TypeError) as e:
        logger.warning(f"Failed to parse tool call from A2A response: {e}")
        msg = f"Invalid tool call format: {e}"
        raise A2AMessageError(msg) from e


def a2a_to_tau2_assistant_message(content: str) -> AssistantMessage:
    """
    Convert A2A agent response content to tau2 AssistantMessage.

    Args:
        content: A2A agent response content

    Returns:
        tau2 AssistantMessage with either text content or tool calls
    """
    tool_calls = parse_a2a_tool_calls(content)

    if tool_calls:
        # Return AssistantMessage with tool calls
        return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)

    # Handle empty responses - agent may have returned no content
    if not content or not content.strip():
        logger.warning(
            "A2A agent returned empty content, using fallback message",
            original_content=repr(content),
        )
        content = "I apologize, but I was unable to generate a response. Could you please rephrase your request?"

    # Return AssistantMessage with text content
    return AssistantMessage(role="assistant", content=content, tool_calls=None)
