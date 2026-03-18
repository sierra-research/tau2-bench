"""
Base agent classes.

This module defines agent-specific base classes for both protocols:
- HalfDuplexAgent: For turn-based agents (uses generate_next_message)
- FullDuplexAgent: For streaming agents (uses get_next_chunk)
"""

from abc import ABC
from typing import Generic, Optional, TypeVar

from tau2.agent.base.participant import (
    FullDuplexParticipant,
    HalfDuplexParticipant,
    VoiceParticipantMixin,
)
from tau2.data_model.message import (
    AssistantMessage,
    EnvironmentMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Type variables
AgentState = TypeVar("AgentState")
ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage


class AgentError(Exception):
    """
    Generic exception for agent errors.
    """


def is_valid_agent_history_message(message: Message) -> bool:
    """Check if the message is a valid agent history message."""
    return (
        isinstance(message, AssistantMessage)
        or (isinstance(message, UserMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == "assistant")
    )


# =============================================================================
# HALF-DUPLEX AGENTS
# =============================================================================


class HalfDuplexAgent(
    HalfDuplexParticipant[ValidAgentInputMessage, AssistantMessage, AgentState],
    ABC,
    Generic[AgentState],
):
    """
    Base class for half-duplex (turn-based) agents.

    Agents are conversation participants that:
    - Receive UserMessage/ToolMessage/MultiToolMessage
    - Produce AssistantMessage
    - Maintain AgentState

    Agent developers must implement:
    - generate_next_message: Generate the next message from user/tool message(s)
    - get_init_state: Get the initial state of the agent
    """

    def __init__(self, tools: list[Tool], domain_policy: str):
        super().__init__()
        self.tools = tools
        self.domain_policy = domain_policy

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[AgentState] = None,
    ) -> None:
        """
        Stops the agent.
        Args:
            message: The last message to the agent.
            state: The agent state.
        """
        pass


# =============================================================================
# FULL-DUPLEX AGENTS
# =============================================================================


class FullDuplexAgent(
    FullDuplexParticipant[UserMessage, AssistantMessage, AgentState],
    ABC,
    Generic[AgentState],
):
    """
    Base class for full-duplex (streaming) agents.

    Streaming agents use get_next_chunk() for continuous communication.
    They do NOT implement generate_next_message().

    Agent developers must implement:
    - get_next_chunk: Process incoming chunks and generate outgoing chunks
    - get_init_state: Get the initial state of the agent
    """

    def __init__(self, tools: list[Tool], domain_policy: str):
        super().__init__()
        self.tools = tools
        self.domain_policy = domain_policy

    def stop(
        self,
        participant_chunk: Optional[Message] = None,
        state: Optional[AgentState] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> None:
        """
        Stops the agent.
        Args:
            participant_chunk: The last chunk from the user.
            state: The agent state.
            tool_results: Any pending tool results not yet delivered.
        """
        pass


# =============================================================================
# VOICE AGENTS (can be used with either protocol)
# =============================================================================


class HalfDuplexVoiceAgent(
    VoiceParticipantMixin[ValidAgentInputMessage, AssistantMessage],
    HalfDuplexAgent[AgentState],
    ABC,
    Generic[AgentState],
):
    """
    Base class for half-duplex agents that support voice communication.
    """

    pass


class FullDuplexVoiceAgent(
    VoiceParticipantMixin[UserMessage, AssistantMessage],
    FullDuplexAgent[AgentState],
    ABC,
    Generic[AgentState],
):
    """
    Base class for full-duplex agents that support voice communication.
    """

    pass


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_message_format(
    message: AssistantMessage, solo: bool = False
) -> tuple[bool, str]:
    """Validate the message format for the agent."""
    if solo:
        return validate_message_format_solo(message)
    else:
        return validate_message_format_default(message)


def validate_message_format_default(message: AssistantMessage) -> tuple[bool, str]:
    """Validate the message format for the agent."""
    has_content = message.has_text_content()
    is_tool_call = message.is_tool_call()
    if not has_content and not is_tool_call:
        return (
            False,
            "You sent an empty message. Each message must contain either a text content (message to the user) or tool calls (actions to perform). Message cannot contain both or be empty.",
        )
    if has_content and is_tool_call:
        return (
            False,
            "You sent a message with both text content and tool calls. Each message must contain either a text content (message to the user) or tool calls (actions to perform). Message cannot contain both or be empty.",
        )
    return True, None


def validate_message_format_solo(message: AssistantMessage) -> tuple[bool, str]:
    """Validate the message format for the solo agent."""
    has_content = message.has_text_content()
    is_tool_call = message.is_tool_call()
    if not has_content and not is_tool_call:
        return (
            False,
            "You sent an empty message. Each message must contain tool calls and no other text content.",
        )
    if has_content:
        return (
            False,
            "You sent a message with text content. Each message must contain tool calls and no other text content.",
        )
    return True, None
