"""A2A Protocol Integration for tau2-bench."""

from tau2.a2a.client import A2AClient
from tau2.a2a.exceptions import (
    A2AAuthError,
    A2ADiscoveryError,
    A2AError,
    A2AMessageError,
    A2ATimeoutError,
)
from tau2.a2a.metrics import AggregatedMetrics, ProtocolMetrics, estimate_tokens
from tau2.a2a.models import A2AAgentState, A2AConfig, AgentCapabilities, AgentCard
from tau2.a2a.translation import (
    a2a_to_tau2_assistant_message,
    format_tools_as_text,
    parse_a2a_tool_calls,
    tau2_to_a2a_message_content,
)

__all__ = [
    # Client
    "A2AClient",
    # Models
    "A2AConfig",
    "A2AAgentState",
    "AgentCard",
    "AgentCapabilities",
    # Exceptions
    "A2AError",
    "A2ATimeoutError",
    "A2AAuthError",
    "A2ADiscoveryError",
    "A2AMessageError",
    # Metrics
    "ProtocolMetrics",
    "AggregatedMetrics",
    "estimate_tokens",
    # Translation
    "format_tools_as_text",
    "tau2_to_a2a_message_content",
    "parse_a2a_tool_calls",
    "a2a_to_tau2_assistant_message",
]
