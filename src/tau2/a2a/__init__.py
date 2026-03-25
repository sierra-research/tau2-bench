"""A2A Protocol Integration for tau2-bench."""

from a2a.client.errors import (
    A2AClientError,
    A2AClientHTTPError,
    A2AClientJSONError,
    A2AClientTimeoutError,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tau2.a2a.models import A2AAgentState, A2AConfig
from tau2.a2a.translation import (
    a2a_to_tau2_assistant_message,
    extract_response,
    format_tools_as_text,
    parse_a2a_tool_calls,
    tau2_to_a2a_message,
    tau2_to_a2a_message_content,
)

__all__ = [
    # Models
    "A2AConfig",
    "A2AAgentState",
    # SDK types (re-exported for convenience)
    "AgentCard",
    "AgentCapabilities",
    "AgentSkill",
    # SDK errors (re-exported)
    "A2AClientError",
    "A2AClientHTTPError",
    "A2AClientJSONError",
    "A2AClientTimeoutError",
    # Translation
    "format_tools_as_text",
    "tau2_to_a2a_message_content",
    "tau2_to_a2a_message",
    "parse_a2a_tool_calls",
    "a2a_to_tau2_assistant_message",
    "extract_response",
]
