"""
Base user simulator classes.

This module defines user-specific base classes for both protocols:
- HalfDuplexUser: For turn-based users (uses generate_next_message)
- FullDuplexUser: For streaming users (uses get_next_chunk)
"""

from abc import ABC
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from tau2.agent.base.participant import (
    FullDuplexParticipant,
    HalfDuplexParticipant,
    VoiceParticipantMixin,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    EnvironmentMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TerminationReason
from tau2.environment.tool import Tool

ValidUserInputMessage = AssistantMessage | ToolMessage | MultiToolMessage


class UserError(Exception):
    """
    Generic exception for user errors.
    """

    pass


def is_valid_user_history_message(message: Message) -> bool:
    """Check if the message is a valid user history message."""
    return (
        isinstance(message, UserMessage)
        or (isinstance(message, AssistantMessage) and not message.is_tool_call())
        or (isinstance(message, ToolMessage) and message.requestor == "user")
    )


STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"

# Maps each user stop token to the termination reason that describes how the run
# ended. ``###OUT-OF-SCOPE###`` means the simulator gave up because the scenario
# gave it nothing to continue with, so the run was aborted rather than completed
# and must not be graded (see rule 5 in ``tau2/evaluator/AGENTS.md``).
#
# Order matters: the first match wins, so a message carrying both an abort token
# and an ordinary stop token is classified as the abort. Getting that wrong is
# the failure this mapping exists to prevent -- grading a run that never ran.
USER_STOP_TOKENS: tuple[tuple[str, "TerminationReason"], ...] = (
    (OUT_OF_SCOPE, TerminationReason.OUT_OF_SCOPE),
    (STOP, TerminationReason.USER_STOP),
    (TRANSFER, TerminationReason.USER_STOP),
)


def classify_user_stop(message: UserMessage) -> Optional["TerminationReason"]:
    """Return the termination reason for a user message, or None if it is not a stop.

    This is the single place that decides *how* a user-side termination is
    recorded; ``is_stop`` implementations decide *whether* the run is over. Both
    read ``USER_STOP_TOKENS``, so a token added in one is never missing from the
    other.
    """
    if message.is_tool_call():
        return None
    # Audio-only messages (chunks) don't have text content
    if message.content is None:
        return None
    for token, reason in USER_STOP_TOKENS:
        if token in message.content:
            return reason
    return None


class UserState(BaseModel):
    """The state of the user simulator."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]

    def flip_roles(self) -> list[APICompatibleMessage]:
        """
        Returns a list of messages with the roles flipped.
        """
        # NOTE: also clean the message to a api-compatible format
        flipped_messages = []
        for message in self.messages:
            if isinstance(message, UserMessage):
                flipped_messages.append(
                    AssistantMessage(
                        role="assistant",
                        tool_calls=message.tool_calls,
                        content=message.content,
                    )
                )
            elif isinstance(message, AssistantMessage):
                if not message.is_tool_call():
                    # Only add non tool call messages
                    flipped_messages.append(
                        UserMessage(
                            role="user",
                            content=message.content,
                        )
                    )
                else:
                    raise ValueError(
                        f"Tool calls are not supported in the flipped messages: {message}"
                    )
            elif isinstance(message, ToolMessage):
                if message.requestor == "user":
                    # Only add tool messages for the user
                    flipped_messages.append(
                        ToolMessage(
                            id=message.id,
                            role=message.role,
                            content=message.content,
                        )
                    )
                else:
                    raise ValueError(
                        f"Tool messages should be sent to the user in this message history: {message}"
                    )
            else:
                raise ValueError(f"Unknown message role: {message.role}")
        return flipped_messages


UserStateType = TypeVar("UserStateType", bound=UserState)


# =============================================================================
# HALF-DUPLEX USERS
# =============================================================================


class HalfDuplexUser(
    HalfDuplexParticipant[ValidUserInputMessage, UserMessage, UserStateType],
    ABC,
    Generic[UserStateType],
):
    """
    Base class for half-duplex (turn-based) user simulators.

    Users are conversation participants that:
    - Receive AssistantMessage/ToolMessage/MultiToolMessage
    - Produce UserMessage
    - Maintain UserState

    User developers must implement:
    - generate_next_message: Generate the next message from assistant/tool message(s)
    - get_init_state: Get the initial state of the user
    """

    def __init__(
        self,
        instructions: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
    ):
        self.instructions = instructions
        self.tools = tools

    def stop(
        self,
        message: Optional[ValidUserInputMessage] = None,
        state: Optional[UserStateType] = None,
    ) -> None:
        """
        Stops the user simulator.
        """
        pass


# =============================================================================
# FULL-DUPLEX USERS
# =============================================================================


class FullDuplexUser(
    FullDuplexParticipant[AssistantMessage, UserMessage, UserStateType],
    ABC,
    Generic[UserStateType],
):
    """
    Base class for full-duplex (streaming) user simulators.

    Streaming users use get_next_chunk() for continuous communication.
    They do NOT implement generate_next_message().

    User developers must implement:
    - get_next_chunk: Process incoming chunks and generate outgoing chunks
    - get_init_state: Get the initial state of the user
    """

    def __init__(
        self,
        instructions: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
    ):
        self.instructions = instructions
        self.tools = tools

    def stop(
        self,
        participant_chunk: Optional[Message] = None,
        state: Optional[UserStateType] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> None:
        """
        Stops the user simulator.

        Args:
            participant_chunk: The last chunk from the agent.
            state: The user state.
            tool_results: Any pending tool results not yet delivered.
        """
        pass


# =============================================================================
# VOICE USERS (can be used with either protocol)
# =============================================================================


class HalfDuplexVoiceUser(
    VoiceParticipantMixin[ValidUserInputMessage, UserMessage],
    HalfDuplexUser[UserStateType],
    ABC,
    Generic[UserStateType],
):
    """
    Base class for half-duplex users that support voice communication.
    """

    pass


class FullDuplexVoiceUser(
    VoiceParticipantMixin[AssistantMessage, UserMessage],
    FullDuplexUser[UserStateType],
    ABC,
    Generic[UserStateType],
):
    """
    Base class for full-duplex users that support voice communication.
    """

    pass
