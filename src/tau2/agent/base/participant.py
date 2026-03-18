"""
Base participant classes defining communication protocols.

This module defines two distinct communication protocols:
- HalfDuplexParticipant: Turn-based communication (generate_next_message)
- FullDuplexParticipant: Streaming communication (get_next_chunk)

Classes should inherit from ONE protocol, not both.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

from loguru import logger

from tau2.data_model.message import EnvironmentMessage, Message

# Generic type variables for conversation participants
InputMessageType = TypeVar("InputMessageType", bound=Message)
OutputMessageType = TypeVar("OutputMessageType", bound=Message)
StateType = TypeVar("StateType")


# =============================================================================
# HALF-DUPLEX PROTOCOL
# =============================================================================


class HalfDuplexParticipant(
    ABC, Generic[InputMessageType, OutputMessageType, StateType]
):
    """
    Protocol for turn-based (half-duplex) communication.

    Participants take turns: receive a message, produce a response.
    One party speaks at a time.

    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces
        StateType: The type of internal state this participant maintains
    """

    @abstractmethod
    def generate_next_message(
        self, message: InputMessageType, state: StateType
    ) -> tuple[OutputMessageType, StateType]:
        """
        Generate the next message from an input message and current state.

        Args:
            message: The input message to respond to.
            state: The current state.

        Returns:
            A tuple of (output_message, updated_state).
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> StateType:
        """
        Get the initial state of the participant.

        Args:
            message_history: The message history.

        Returns:
            The initial state.
        """
        raise NotImplementedError

    def stop(
        self,
        message: Optional[InputMessageType] = None,
        state: Optional[StateType] = None,
    ) -> None:
        """
        Stop the participant.

        Args:
            message: The last message to the participant.
            state: The last state of the participant.
        """
        pass

    @classmethod
    def is_stop(cls, message: OutputMessageType) -> bool:
        """
        Check if the message is a stop message.

        By default the participant does not stop.

        Args:
            message: The output message to check.

        Returns:
            True if the message indicates stopping, False otherwise.
        """
        return False

    def set_seed(self, seed: int):
        """
        Set the seed for the participant. [Optional]

        Args:
            seed: The random seed to set.
        """
        logger.warning(
            f"Setting seed for participant is not implemented for class "
            f"{self.__class__.__name__}"
        )


# =============================================================================
# FULL-DUPLEX PROTOCOL
# =============================================================================


class FullDuplexParticipant(
    ABC, Generic[InputMessageType, OutputMessageType, StateType]
):
    """
    Protocol for streaming (full-duplex) communication.

    Participants exchange chunks continuously and can interrupt each other.
    Both parties can speak simultaneously.

    Note: This is a SEPARATE protocol from HalfDuplexParticipant.
    Classes should inherit from ONE protocol, not both.

    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces
        StateType: The type of internal state this participant maintains
    """

    @abstractmethod
    def get_next_chunk(
        self,
        state: StateType,
        participant_chunk: Optional[InputMessageType] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[Optional[OutputMessageType], StateType]:
        """
        Get the next chunk of the conversation in streaming mode.

        Each participant has two communication channels per tick:
        - Participant channel: speech/audio from the other participant
        - Environment channel: tool results from previously executed tool calls

        Args:
            state: The current state of the conversation.
            participant_chunk: The incoming chunk from the other participant.
                None if this is a continuation call with no new input.
            tool_results: Tool results from the environment (ToolMessage or
                MultiToolMessage). None if no tool results are pending.

        Returns:
            A tuple of (next_chunk, updated_state) where:
            - next_chunk: The next chunk to send (None if waiting/silent).
                May contain tool_calls if the participant wants to invoke tools.
            - updated_state: The updated conversation state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> StateType:
        """
        Get the initial state of the participant.

        Args:
            message_history: The message history.

        Returns:
            The initial state.
        """
        raise NotImplementedError

    def stop(
        self,
        participant_chunk: Optional[InputMessageType] = None,
        state: Optional[StateType] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> None:
        """
        Stop the participant.

        Args:
            participant_chunk: The last chunk from the other participant.
            state: The last state of the participant.
            tool_results: Any pending tool results from the environment
                that were not yet delivered.
        """
        pass

    @classmethod
    def is_stop(cls, message: OutputMessageType) -> bool:
        """
        Check if the message is a stop message.

        By default the participant does not stop.

        Args:
            message: The output message to check.

        Returns:
            True if the message indicates stopping, False otherwise.
        """
        return False

    def set_seed(self, seed: int):
        """
        Set the seed for the participant. [Optional]

        Args:
            seed: The random seed to set.
        """
        logger.warning(
            f"Setting seed for participant is not implemented for class {self.__class__.__name__}"
        )


# =============================================================================
# VOICE MIXIN (can be used with either protocol)
# =============================================================================


class VoiceParticipantMixin(ABC, Generic[InputMessageType, OutputMessageType]):
    """
    Mixin for participants that support voice communication.

    This can be combined with either HalfDuplexParticipant or FullDuplexParticipant.
    """

    @abstractmethod
    def transcribe_voice(self, message: InputMessageType) -> InputMessageType:
        """
        Transcribe voice for the message.

        Args:
            message: The message with audio data.

        Returns:
            The transcribed message with the content replaced by the transcribed text.
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize_voice(self, message: OutputMessageType) -> OutputMessageType:
        """
        Synthesize voice for the message.

        Args:
            message: The message to synthesize.

        Returns:
            The synthesized message with content replaced by the synthesized audio.
        """
        raise NotImplementedError
