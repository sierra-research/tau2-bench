import contextvars
import json
import random
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Literal, Optional, Tuple, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.audio import AudioData, audio_bytes_to_string
from tau2.data_model.message import (
    EnvironmentMessage,
    Message,
    ParticipantMessageBase,
    TurnTakingAction,
)
from tau2.utils.utils import get_now
from tau2.voice.utils.audio_preprocessing import pad_audio_with_zeros
from tau2.voice.utils.probability import poisson_should_trigger

# Generic type variables for streaming mixins
InputMessageType = TypeVar("InputMessageType", bound=Message)
OutputMessageType = TypeVar("OutputMessageType", bound=Message)


class ParticipantTick(BaseModel, Generic[InputMessageType, OutputMessageType]):
    """
    Represents events from a single tick from a participant's perspective.

    In full-duplex mode, both parties can emit chunks simultaneously at each tick.
    This captures three concurrent channels per tick:

    - self_chunk: what I emitted (speech, or a message containing tool_calls)
    - other_chunk: speech/audio received from the other participant
    - env_chunk: tool results received from the environment (ToolMessage or
      MultiToolMessage), delivered in response to a tool call from a previous tick

    Before linearization, ticks with env_chunk are expanded into old-format
    ticks (env_chunk merged into other_chunk) by _expand_env_chunks, so that
    linearization code sees the same structure as the base branch.

    Attributes:
        tick_id: The sequential identifier for this tick.
        timestamp: When this tick occurred.
        self_chunk: The chunk emitted by this participant (if any, may include tool calls).
        other_chunk: The chunk received from the other participant (speech/audio).
        env_chunk: Tool results from the environment (ToolMessage or MultiToolMessage).
    """

    model_config = {"arbitrary_types_allowed": True}

    tick_id: int
    timestamp: str
    self_chunk: Optional[OutputMessageType] = None
    other_chunk: Optional[InputMessageType] = None
    env_chunk: Optional[Message] = None


class LinearizationStrategy(str, Enum):
    """
    Strategies for converting tick-based history to sequential messages.

    LLMs expect sequential message history, but full-duplex ticks contain
    concurrent events. These strategies define how to order concurrent events
    into a sequence suitable for LLM consumption.
    """

    TIMESTAMP_ORDER = "timestamp"  # Sort all messages by timestamp
    SELF_FIRST_PER_TICK = "self_first"  # Each tick: self chunk first, then other chunk
    OTHER_FIRST_PER_TICK = (
        "other_first"  # Each tick: other chunk first, then self chunk
    )
    CONSOLIDATED = "consolidated"  # Merge consecutive same-speaker messages
    CONTAINMENT_AWARE = "containment_aware"  # Containment-aware splice-merge algorithm


class StreamingState(BaseModel, Generic[InputMessageType, OutputMessageType]):
    """
    Generic streaming state for conversation participants.

    This extends the participant's base state with streaming-specific fields
    for managing chunk buffers, turn-taking, and tick-based history.

    In full-duplex mode, the tick-based history (`ticks`) is the primary record
    of the conversation, capturing the concurrent nature of events. The `messages`
    field from the parent state class is kept for backward compatibility but should
    be populated via linearization when needed for LLM calls.

    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces
    """

    model_config = {"arbitrary_types_allowed": True}

    # Tick-based conversation history (captures concurrent events)
    ticks: list[ParticipantTick[InputMessageType, OutputMessageType]] = Field(
        default_factory=list
    )

    # Chunk buffers for current processing
    input_turn_taking_buffer: list[InputMessageType] = Field(default_factory=list)
    output_streaming_queue: list[OutputMessageType] = Field(default_factory=list)

    # Turn-taking timing
    time_since_last_talk: int = 0
    time_since_last_other_talk: int = 0
    tick_count: int = 0  # Incremented on each get_next_chunk call

    # Backchannel timing - tracks ticks since we last backchanneled or agent speech started
    ticks_since_last_backchannel: int = 0
    backchannel_rng: Optional[random.Random] = None
    is_backchanneling: bool = False

    # Consecutive speaking duration tracking (for overlap detection)
    # These track how many consecutive ticks each party has been speaking
    # Reset to 0 when that party stops speaking
    consecutive_self_speaking_ticks: int = 0
    consecutive_other_speaking_ticks: int = 0

    # Tool result processing state
    # waiting_for_tool_results: True after emitting a tool call, cleared when
    # tool results arrive. While True, _get_next_chunk_core emits noise/silence
    # instead of running normal turn-taking.
    waiting_for_tool_results: bool = False
    # delivering_tool_result_speech: True while streaming speech generated from
    # a tool result. Uses the higher yield_threshold_when_interrupting to make
    # the participant harder to interrupt during tool-result delivery.
    delivering_tool_result_speech: bool = False

    @property
    def overlap_initiator(self) -> Optional[Literal["self", "other"]]:
        """
        Derived property: who initiated the current overlap.

        During an overlap (both parties speaking), we can determine who started first
        by comparing consecutive speaking durations - the one with the longer duration
        was speaking first and got interrupted.

        Returns:
            - "self": Self started speaking while other was already speaking (self initiated interruption)
            - "other": Other started speaking while self was already speaking (other initiated interruption)
            - None: No overlap currently happening (one or both parties not speaking)
        """
        # Both must be speaking for there to be an overlap
        if (
            self.consecutive_self_speaking_ticks == 0
            or self.consecutive_other_speaking_ticks == 0
        ):
            return None

        # Compare durations to determine who was speaking first
        if (
            self.consecutive_self_speaking_ticks
            >= self.consecutive_other_speaking_ticks
        ):
            # Self was speaking first, other started speaking over self
            return "other"
        else:
            # Other was speaking first (or started same tick), self started speaking over other
            return "self"

    @property
    def info(self) -> str:
        """
        Get information about the state.
        """
        return f"Input turn-taking buffer: {len(self.input_turn_taking_buffer)}, Output streaming queue: {len(self.output_streaming_queue)}, Time since last talk: {self.time_since_last_talk}, Time since last other talk: {self.time_since_last_other_talk}, Tick count: {self.tick_count}"

    @property
    def is_talking(self) -> bool:
        """
        Check if the participant is currently talking.
        """
        return len(self.output_streaming_queue) > 0

    def input_total_speech_duration(self) -> int:
        """
        Analyze the pending chunks to determine the speech duration.
        Speech duration is the number of chunks that contain speech.

        Returns:
            The total speech duration in chunks.
        """
        return sum(
            1
            for chunk in reversed(self.input_turn_taking_buffer)
            if not isinstance(chunk, EnvironmentMessage) and chunk.contains_speech
        )

    def input_ongoing_speech_duration(self) -> int:
        """
        Check duration of ongoing speech since last silence.

        Returns:
            The last speech duration in chunks.
        """
        ongoing_speech_duration = 0
        for chunk in reversed(self.input_turn_taking_buffer):
            if not isinstance(chunk, EnvironmentMessage) and chunk.contains_speech:
                ongoing_speech_duration += 1
            else:
                break
        logger.debug(f"Ongoing speech duration: {ongoing_speech_duration}")
        return ongoing_speech_duration

    def input_interrupt(self) -> bool:
        """
        Check if the input is an interruption from the other participant.
        """
        return self.is_talking and self.input_ongoing_speech_duration() > 0

    def input_from_environment(self) -> bool:
        """
        Check if any chunk in the buffer is from the environment.

        With the new tool result flow, EnvironmentMessages should never appear
        in the buffer (they go through the tool_results parameter instead).
        This check is kept as a safety net.
        """
        return any(
            isinstance(chunk, EnvironmentMessage)
            for chunk in self.input_turn_taking_buffer
        )

    def record_tick(
        self,
        tick_id: int,
        timestamp: str,
        self_chunk: Optional[OutputMessageType],
        other_chunk: Optional[InputMessageType],
        env_chunk: Optional[Message] = None,
    ) -> None:
        """
        Record a tick in the conversation history.

        Args:
            tick_id: The sequential tick identifier.
            timestamp: When this tick occurred.
            self_chunk: The chunk emitted by this participant (if any, may include tool calls).
            other_chunk: The chunk received from the other participant (speech/audio).
            env_chunk: Tool results from the environment (ToolMessage or MultiToolMessage).
        """
        tick = ParticipantTick(
            tick_id=tick_id,
            timestamp=timestamp,
            self_chunk=self_chunk,
            other_chunk=other_chunk,
            env_chunk=env_chunk,
        )
        self.ticks.append(tick)

    def get_linearized_messages(
        self,
        strategy: LinearizationStrategy = LinearizationStrategy.CONTAINMENT_AWARE,
        include_pending_input: bool = False,
        indicate_current_incomplete: bool = False,
        integration_ticks: int = 1,
        silence_annotation_threshold_ticks: Optional[int] = None,
        tick_duration_seconds: Optional[float] = None,
    ) -> list[Message]:
        """
        Convert tick-based history to sequential messages for LLM context.

        This bridges the gap between full-duplex tick-based storage and the
        sequential message format that LLMs expect.

        Args:
            strategy: How to order concurrent events into a sequence.
            include_pending_input: If True, include the current incoming chunk
                (last in input_turn_taking_buffer) by creating a temporary tick.
                This is useful when generating a response, as the current tick
                hasn't been recorded yet. The temporary tick is passed to the
                linearization algorithm so it can properly handle segment
                detection and overlap analysis.
            indicate_current_incomplete: If True, indicate that the current incoming chunk is incomplete.
            integration_ticks: Number of consecutive silent ticks before an overlap
                region ends. Higher values are more tolerant of brief pauses in speech.
                Default is 1 (end overlap immediately when self stops speaking).
            silence_annotation_threshold_ticks: If set, insert silence annotations when
                both parties are silent for more than this many consecutive ticks.
                The annotation will be a SystemMessage like "[Both parties silent for X seconds]".
            tick_duration_seconds: Duration of each tick in seconds. Required if
                silence_annotation_threshold_ticks is set (to compute real time).
        Returns:
            A flat list of messages suitable for LLM context.
        """
        from tau2.data_model.message import MultiToolMessage, ToolMessage

        # DEBUG: Store input state
        _debug_save_get_linearized_messages_call(
            state=self,
            strategy=strategy,
            include_pending_input=include_pending_input,
            indicate_current_incomplete=indicate_current_incomplete,
            integration_ticks=integration_ticks,
            stage="input",
        )

        # Build list of ticks to process
        ticks_to_process = list(self.ticks)

        # If include_pending_input, create a temporary tick for the current incoming chunk.
        # This ensures the linearization algorithm can properly handle segment detection
        # and overlap analysis for the current chunk (which hasn't been recorded yet).
        if include_pending_input and self.input_turn_taking_buffer:
            current_incoming = self.input_turn_taking_buffer[-1]

            # Prepare the chunk for the temporary tick
            if isinstance(current_incoming, (MultiToolMessage, ToolMessage)):
                # Tool messages are passed through as-is
                temp_other_chunk = current_incoming
            elif current_incoming is not None and (
                current_incoming.has_text_content() or current_incoming.is_tool_call()
            ):
                temp_other_chunk = deepcopy(current_incoming)
                if indicate_current_incomplete:
                    # Handle None content gracefully - only add suffix if there's content
                    content = temp_other_chunk.content or ""
                    temp_other_chunk.content = (
                        f"{content} [CURRENTLY SPEAKING, INCOMPLETE]"
                    )
            else:
                temp_other_chunk = None

            # Create temporary tick with only other_chunk (we haven't produced output yet)
            if temp_other_chunk is not None:
                temp_tick = ParticipantTick(
                    tick_id=len(self.ticks),
                    timestamp=get_now(),
                    self_chunk=None,
                    other_chunk=temp_other_chunk,
                )
                ticks_to_process.append(temp_tick)

        # Expand ticks with env_chunk into old-format ticks before linearization.
        # This ensures linearization sees the same tick structure as the base branch.
        ticks_to_process = _expand_env_chunks(ticks_to_process)

        messages = linearize_ticks(
            ticks_to_process,
            strategy,
            integration_ticks,
            silence_threshold_ticks=silence_annotation_threshold_ticks,
            tick_duration_seconds=tick_duration_seconds,
        )

        # DEBUG: Store output messages
        _debug_save_get_linearized_messages_call(
            state=self,
            strategy=strategy,
            include_pending_input=include_pending_input,
            indicate_current_incomplete=indicate_current_incomplete,
            integration_ticks=integration_ticks,
            stage="output",
            linearize_ticks_output=messages,
            final_messages=messages,
        )

        return messages

    def update_input_turn_taking_buffer(
        self,
        incoming_chunk: InputMessageType,
    ) -> None:
        """
        Update the pending chunks with a participant chunk (speech/audio).

        Incoming chunk is never None - empty chunks are represented with
        contains_speech=False. EnvironmentMessages (tool results) should NOT
        be added to this buffer — they are passed via the tool_results parameter
        of get_next_chunk and recorded as env_chunk in the tick history.
        """
        self.input_turn_taking_buffer.append(incoming_chunk)


# Silence period map: start_tick_idx -> silence annotation message
# Maps the tick where silence starts to the pre-created annotation message
_SilencePeriodMap = dict[int, Message]


def _expand_env_chunks(
    ticks: list[ParticipantTick],
) -> list[ParticipantTick]:
    """Expand ticks with env_chunk into old-format ticks for linearization.

    In the dual-channel model, a tick can have both other_chunk (participant
    speech) and env_chunk (tool results) simultaneously. The linearization
    code expects tool results in other_chunk (as the base branch's while-loop
    produced). This function expands such ticks into sequential old-format
    ticks so linearization sees the same structure:

    Input:  Tick {other_chunk=speech, self_chunk=response, env_chunk=tool_result}
    Output: Tick {other_chunk=tool_result, self_chunk=response}   (tool result tick)
            Tick {other_chunk=speech, self_chunk=None}             (speech tick, if present)

    Ticks without env_chunk pass through unchanged. Tick IDs are reassigned
    sequentially to remain unique.
    """
    expanded: list[ParticipantTick] = []
    next_id = 0

    for tick in ticks:
        if tick.env_chunk is not None:
            # Tool result tick: env_chunk becomes other_chunk, self_chunk stays
            expanded.append(
                ParticipantTick(
                    tick_id=next_id,
                    timestamp=tick.timestamp,
                    self_chunk=tick.self_chunk,
                    other_chunk=tick.env_chunk,
                )
            )
            next_id += 1

            # Speech tick: only if other_chunk has content
            if tick.other_chunk is not None:
                expanded.append(
                    ParticipantTick(
                        tick_id=next_id,
                        timestamp=tick.timestamp,
                        other_chunk=tick.other_chunk,
                        self_chunk=None,
                    )
                )
                next_id += 1
        else:
            expanded.append(
                ParticipantTick(
                    tick_id=next_id,
                    timestamp=tick.timestamp,
                    self_chunk=tick.self_chunk,
                    other_chunk=tick.other_chunk,
                )
            )
            next_id += 1

    return expanded


def linearize_ticks(
    ticks: list[ParticipantTick],
    strategy: LinearizationStrategy = LinearizationStrategy.CONTAINMENT_AWARE,
    integration_ticks: int = 1,
    silence_threshold_ticks: Optional[int] = None,
    tick_duration_seconds: Optional[float] = None,
) -> list[Message]:
    """
    Convert tick-based history to sequential messages for LLM context.

    This is the core function that bridges the gap between full-duplex tick-based
    storage and the sequential message format that LLMs expect.

    Args:
        ticks: The tick-based conversation history.
        strategy: How to order concurrent events into a sequence.
        integration_ticks: Number of consecutive silent ticks before an overlap
            region ends. Higher values are more tolerant of brief pauses in speech.
            Default is 1 (end overlap immediately when self stops speaking).
        silence_threshold_ticks: If set, insert SystemMessage annotations when both
            parties are silent for more than this many ticks. Requires tick_duration_seconds.
        tick_duration_seconds: Duration of each tick in seconds. Required for silence annotations.

    Returns:
        A flat list of messages suitable for LLM context.

    Strategies:
        - TIMESTAMP_ORDER: Sort all messages by timestamp (may interleave oddly),
          then consolidate consecutive same-speaker messages.
        - SELF_FIRST_PER_TICK: Each tick: self chunk first, then other chunk,
          then consolidate consecutive same-speaker messages.
        - OTHER_FIRST_PER_TICK: Each tick: other chunk first, then self chunk,
          then consolidate consecutive same-speaker messages.
        - CONSOLIDATED: Like OTHER_FIRST, but with special overlap handling.
          During overlapping speech (both parties talking simultaneously), all "other"
          content is placed before "self" content to maintain logical ordering.
        - CONTAINMENT_AWARE: Containment-aware splice-merge algorithm.
          If you speak entirely during someone else's turn, you get inserted where
          you stopped. Otherwise, whoever started first goes first.
    """
    # Pre-compute silence periods if enabled
    silence_map: _SilencePeriodMap = {}
    if silence_threshold_ticks is not None and tick_duration_seconds is not None:
        silence_map = _detect_silence_periods(
            ticks, silence_threshold_ticks, tick_duration_seconds
        )

    messages: list[Message] = []

    if strategy == LinearizationStrategy.CONSOLIDATED:
        # Handle overlapping speech: group other before self within overlap regions
        messages = _linearize_with_overlap_handling(
            ticks, integration_ticks, silence_map
        )
    elif strategy == LinearizationStrategy.CONTAINMENT_AWARE:
        # Containment-aware splice-merge algorithm
        # This strategy handles consolidation internally and deliberately keeps
        # some segments separate (e.g., when gap exceeds integration_ticks)
        return _linearize_with_containment_awareness(
            ticks, integration_ticks, silence_map
        )
    else:
        # Simple strategies: TIMESTAMP_ORDER, SELF_FIRST_PER_TICK, OTHER_FIRST_PER_TICK
        messages = _linearize_simple(ticks, strategy, silence_map)

    # All strategies (except CONTAINMENT_AWARE) consolidate consecutive same-speaker messages
    messages = consolidate_messages(messages)

    return messages


def _linearize_simple(
    ticks: list[ParticipantTick],
    strategy: LinearizationStrategy,
    silence_map: _SilencePeriodMap,
) -> list[Message]:
    """
    Linearize ticks using simple tick-by-tick strategies.

    Handles TIMESTAMP_ORDER, SELF_FIRST_PER_TICK, and OTHER_FIRST_PER_TICK strategies.
    Injects silence annotations at appropriate positions.

    Args:
        ticks: The tick-based conversation history.
        strategy: The linearization strategy.
        silence_map: Map from resume_tick to pre-created silence annotation message.

    Returns:
        List of messages (not yet consolidated).
    """
    messages: list[Message] = []

    for tick_idx, tick in enumerate(ticks):
        # Check if we need to inject silence annotation before this tick
        if tick_idx in silence_map:
            messages.append(silence_map[tick_idx])

        # Extract messages from this tick
        tick_messages = _extract_tick_messages_simple(tick, strategy)
        messages.extend(tick_messages)

    if strategy == LinearizationStrategy.TIMESTAMP_ORDER:
        # Sort by timestamp (silence annotations have no timestamp, they stay in place)
        # Actually, we need to be careful here - silence annotations should maintain position
        # For now, we'll sort and silence annotations will end up at the beginning
        # This may need refinement based on use case
        messages = sorted(messages, key=lambda m: m.timestamp if m.timestamp else "")

    return messages


def _detect_silence_periods(
    ticks: list[ParticipantTick],
    silence_threshold_ticks: int,
    tick_duration_seconds: float,
) -> _SilencePeriodMap:
    """
    Detect periods where both parties are silent and create annotation messages.

    Args:
        ticks: The tick-based conversation history.
        silence_threshold_ticks: Minimum consecutive silent ticks to report.
        tick_duration_seconds: Duration of each tick in seconds.

    Returns:
        Map from start_tick_idx to pre-created silence annotation message.
        start_tick_idx is the tick where silence begins.
    """
    if not ticks or silence_threshold_ticks <= 0:
        return {}

    silence_map: _SilencePeriodMap = {}
    silence_start: Optional[int] = None

    for i, tick in enumerate(ticks):
        # Use _has_speech_content (not _has_meaningful_content) to ignore tool calls
        # Tool activity is not audible, so it counts as silence from the user's perspective
        is_silent = not _has_speech_content(
            tick.self_chunk
        ) and not _has_speech_content(tick.other_chunk)

        if is_silent:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                duration = i - silence_start
                if duration >= silence_threshold_ticks:
                    # Silence started at silence_start - key by start_tick
                    silence_map[silence_start] = _create_silence_annotation(
                        duration, tick_duration_seconds
                    )
                silence_start = None

    # Check for trailing silence (silence at the end of conversation)
    if silence_start is not None:
        duration = len(ticks) - silence_start
        if duration >= silence_threshold_ticks:
            silence_map[silence_start] = _create_silence_annotation(
                duration, tick_duration_seconds
            )

    return silence_map


def _create_silence_annotation(
    duration_ticks: int, tick_duration_seconds: float
) -> Message:
    """Create a SystemMessage annotation for a silence period."""
    from tau2.data_model.message import SystemMessage

    duration_seconds = duration_ticks * tick_duration_seconds
    return SystemMessage(
        role="system",
        content=f"[Both parties silent for {duration_seconds:.1f} seconds]",
    )


def _has_tool_constraints(tick: ParticipantTick) -> tuple[bool, bool]:
    """
    Check if a tick has tool-related constraints that prevent overlap handling.

    Tool messages must maintain strict ordering (result follows call), so they
    cannot be reordered during overlap handling.

    Args:
        tick: The tick to check.

    Returns:
        Tuple of (other_is_tool, self_has_tool_call).
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    other_is_tool = isinstance(tick.other_chunk, (ToolMessage, MultiToolMessage))
    self_has_tool_call = (
        tick.self_chunk is not None
        and hasattr(tick.self_chunk, "is_tool_call")
        and tick.self_chunk.is_tool_call()
    )
    return other_is_tool, self_has_tool_call


def _collect_overlap_region(
    ticks: list[ParticipantTick],
    start_idx: int,
    integration_ticks: int,
) -> tuple[list[Message], list[Message], int]:
    """
    Collect all chunks in an overlap region starting at the given index.

    An overlap region continues while self has content and no tool constraints.
    The integration_ticks parameter provides a grace period for brief pauses.

    Args:
        ticks: The tick-based conversation history.
        start_idx: The index to start collecting from.
        integration_ticks: Number of consecutive silent ticks before overlap ends.

    Returns:
        Tuple of (other_chunks, self_chunks, next_index).
    """
    overlap_other_chunks: list[Message] = []
    overlap_self_chunks: list[Message] = []
    consecutive_silent_ticks = 0
    i = start_idx

    while i < len(ticks):
        tick = ticks[i]
        tick_has_self = _has_meaningful_content(tick.self_chunk)
        tick_has_other = _has_meaningful_content(tick.other_chunk)
        other_is_tool, self_has_tool_call = _has_tool_constraints(tick)

        # Tool constraints always end overlap immediately
        if other_is_tool or self_has_tool_call:
            break

        # Track consecutive silent ticks for grace period
        if not tick_has_self:
            consecutive_silent_ticks += 1
        else:
            consecutive_silent_ticks = 0

        # Overlap region ends when self is silent for integration_ticks
        if consecutive_silent_ticks >= integration_ticks:
            # End overlap now. Ticks during the grace period (where
            # consecutive_silent_ticks < integration_ticks) were already
            # added to the overlap. This tick that triggered the threshold
            # will be processed outside the overlap by the outer loop.
            break

        # Add chunks to their respective lists
        if tick_has_other:
            _add_chunk_to_list(tick.other_chunk, overlap_other_chunks)
        if tick_has_self:
            _add_chunk_to_list(tick.self_chunk, overlap_self_chunks)

        i += 1

    return overlap_other_chunks, overlap_self_chunks, i


def _linearize_with_overlap_handling(
    ticks: list[ParticipantTick],
    integration_ticks: int = 1,
    silence_map: Optional[_SilencePeriodMap] = None,
) -> list[Message]:
    """
    Linearize ticks with special handling for overlapping speech.

    When both parties are speaking simultaneously (overlap region), all "other"
    content from the overlap is placed before all "self" content. This ensures
    that content arriving from the other party during an interruption is
    logically ordered before the interrupting speech.

    An overlap region is defined as consecutive ticks where "self" has content.
    Any "other" content arriving during these ticks is considered part of the
    overlap and is placed before self's content.

    The integration_ticks parameter controls how tolerant the overlap detection
    is to brief pauses. With integration_ticks=3, self must be silent for 3
    consecutive ticks before the overlap ends. This helps prevent fragmentation
    from natural speech pauses.

    IMPORTANT: Tool messages (ToolMessage, MultiToolMessage) are NOT treated as
    overlap content because they have strict ordering requirements - they must
    follow the message containing the tool call. Tool messages are always processed
    with standard ordering (self first if it contains a tool call, then tool result).

    Args:
        ticks: The tick-based conversation history.
        integration_ticks: Number of consecutive silent ticks before overlap ends.
            Default is 1 (end immediately when self stops speaking).
            Minimum value is 1 (values < 1 are treated as 1).
        silence_map: Map from resume_tick to pre-created silence annotation message.

    Returns:
        A flat list of messages with overlap regions properly ordered.
    """
    # DEBUG: Store input ticks to file
    _debug_save_linearization_call(ticks, integration_ticks, stage="input")

    # Ensure integration_ticks is at least 1 to prevent infinite loops
    integration_ticks = max(1, integration_ticks)

    if silence_map is None:
        silence_map = {}

    messages: list[Message] = []
    i = 0

    while i < len(ticks):
        # Check if we need to inject silence annotation before this tick
        if i in silence_map:
            messages.append(silence_map[i])

        tick = ticks[i]
        has_self = _has_meaningful_content(tick.self_chunk)
        has_other = _has_meaningful_content(tick.other_chunk)
        other_is_tool, self_has_tool_call = _has_tool_constraints(tick)

        # Determine if this tick can start an overlap region
        # Overlap is NOT allowed when:
        # - other is a tool message (strict ordering required)
        # - self has a tool call (tool result must follow immediately)
        can_start_overlap = (
            has_self and has_other and not other_is_tool and not self_has_tool_call
        )

        if can_start_overlap:
            # Collect all ticks in the overlap region
            overlap_other, overlap_self, i = _collect_overlap_region(
                ticks, i, integration_ticks
            )
            # Output all other chunks first, then all self chunks
            messages.extend(overlap_other)
            messages.extend(overlap_self)
        else:
            # No overlap OR tool-related tick - use standard ordering
            # Always output other (incoming) first, then self (output).
            #
            # This is critical for tool call chains: when self contains a NEW tool call
            # and other contains the result of a PREVIOUS tool call, the previous result
            # must appear before the new tool call. OpenAI requires tool results to
            # immediately follow their corresponding tool calls.
            #
            # Example flow:
            #   Tick N: self=get_users tool call
            #   Tick N+1: other=get_users result, self=create_task tool call
            # Correct order: get_users call → get_users result → create_task call
            if has_other:
                _add_chunk_to_list(tick.other_chunk, messages)
            if has_self:
                _add_chunk_to_list(tick.self_chunk, messages)
            i += 1

    # DEBUG: Store output messages to file
    _debug_save_linearization_call(
        ticks, integration_ticks, stage="output", messages=messages
    )

    return messages


# =============================================================================
# Containment-Aware Linearization
# =============================================================================

# Type definitions for containment-aware segment collection
# A speech segment: (start_tick, end_tick, {tick: message})
_SpeechSegment = tuple[int, int, dict[int, Message]]
# A tool pair: (call_tick, tool_call_message, tool_result_message)
_ToolPair = tuple[int, Message, Message]


def _collect_containment_segments(
    ticks: list[ParticipantTick],
    integration_ticks: int,
) -> tuple[list[_SpeechSegment], list[_SpeechSegment], list[_ToolPair]]:
    """
    Collect speech segments and tool pairs from ticks for containment-aware linearization.

    Forms continuous speech segments for both self and other, using the
    integration_ticks parameter to tolerate brief pauses within a segment.
    Also matches tool calls with their results.

    Args:
        ticks: The tick-based conversation history.
        integration_ticks: Number of consecutive silent ticks before a speech
            segment ends.

    Returns:
        Tuple of (self_segments, other_segments, tool_pairs).
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    self_segments: list[_SpeechSegment] = []
    other_segments: list[_SpeechSegment] = []
    tool_pairs: list[_ToolPair] = []

    # Track tool calls waiting for results
    tool_call_buffer: dict[str, tuple[int, Message]] = {}

    for i, tick in enumerate(ticks):
        has_self = _has_meaningful_content(tick.self_chunk)
        has_other = _has_meaningful_content(tick.other_chunk)

        # Check tool constraints
        other_is_tool = isinstance(tick.other_chunk, (ToolMessage, MultiToolMessage))
        self_has_tool_call = (
            tick.self_chunk is not None
            and hasattr(tick.self_chunk, "is_tool_call")
            and tick.self_chunk.is_tool_call()
        )

        # Collect self speech segments (non-tool messages only)
        if has_self and not self_has_tool_call:
            if self_segments and i - self_segments[-1][1] <= integration_ticks:
                # Extend current segment
                start, end, msgs = self_segments[-1]
                msgs[i] = tick.self_chunk
                self_segments[-1] = (start, i, msgs)
            else:
                # Start new segment
                self_segments.append((i, i, {i: tick.self_chunk}))

        # Collect other speech segments (non-tool messages only)
        if has_other and not other_is_tool:
            if other_segments and i - other_segments[-1][1] <= integration_ticks:
                # Extend current segment
                start, end, msgs = other_segments[-1]
                msgs[i] = tick.other_chunk
                other_segments[-1] = (start, i, msgs)
            else:
                # Start new segment
                other_segments.append((i, i, {i: tick.other_chunk}))

        # Handle tool calls
        if self_has_tool_call:
            tool_msg = tick.self_chunk
            if hasattr(tool_msg, "tool_calls") and tool_msg.tool_calls:
                for tc in tool_msg.tool_calls:
                    tool_call_buffer[tc.id] = (i, tool_msg)
            else:
                tool_call_buffer[f"tick_{i}"] = (i, tool_msg)

        # Handle tool results
        if other_is_tool:
            tool_result = tick.other_chunk

            # If it's a MultiToolMessage, process each individual tool message
            if isinstance(tool_result, MultiToolMessage):
                for individual_result in tool_result.tool_messages:
                    tool_call_id = getattr(
                        individual_result, "tool_call_id", None
                    ) or getattr(individual_result, "id", None)

                    # Try to match by ID
                    matched = False
                    if tool_call_id and tool_call_id in tool_call_buffer:
                        call_tick, tool_call = tool_call_buffer.pop(tool_call_id)
                        tool_pairs.append((call_tick, tool_call, individual_result))
                        matched = True

                    # Fallback: match by position (most recent call before this result)
                    if not matched and tool_call_buffer:
                        logger.warning(
                            f"No match found for tool call {tool_call_id} at tick {i}, falling back to position matching"
                        )
                        candidates = [
                            (tid, tick_pos, msg)
                            for tid, (tick_pos, msg) in tool_call_buffer.items()
                            if tick_pos < i
                        ]
                        if candidates:
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            tool_id, call_tick, tool_call = candidates[0]
                            tool_call_buffer.pop(tool_id)
                            tool_pairs.append((call_tick, tool_call, individual_result))
            else:
                # Single tool message
                tool_call_id = getattr(tool_result, "tool_call_id", None) or getattr(
                    tool_result, "id", None
                )

                # Try to match by ID
                matched = False
                if tool_call_id and tool_call_id in tool_call_buffer:
                    call_tick, tool_call = tool_call_buffer.pop(tool_call_id)
                    tool_pairs.append((call_tick, tool_call, tool_result))
                    matched = True

                # Fallback: match by position (most recent call before this result)
                if not matched and tool_call_buffer:
                    candidates = [
                        (tid, tick_pos, msg)
                        for tid, (tick_pos, msg) in tool_call_buffer.items()
                        if tick_pos < i
                    ]
                    if candidates:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        tool_id, call_tick, tool_call = candidates[0]
                        tool_call_buffer.pop(tool_id)
                        tool_pairs.append((call_tick, tool_call, tool_result))

    # Add any unmatched tool calls (result hasn't arrived yet)
    # Use None as placeholder for the result
    for tool_id, (call_tick, tool_call) in tool_call_buffer.items():
        tool_pairs.append((call_tick, tool_call, None))

    return self_segments, other_segments, tool_pairs


def _is_segment_contained(
    inner_start: int, inner_end: int, outer_start: int, outer_end: int
) -> bool:
    """Check if inner segment is fully contained within outer segment."""
    return inner_start >= outer_start and inner_end <= outer_end


def _segments_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two tick ranges overlap (share at least one tick)."""
    return not (end1 < start2 or end2 < start1)


def _get_segment_messages(msgs: dict[int, Message]) -> list[Message]:
    """Get all messages from a segment in tick order."""
    return [msgs[t] for t in sorted(msgs.keys())]


# Sort priorities for containment-aware linearization (lower = earlier in output)
_PRIORITY_SILENCE = -1  # Silence annotations come first at their tick
_PRIORITY_OTHER = 0  # Other speech comes first
_PRIORITY_SELF = 1  # Self speech comes second
_PRIORITY_TOOL = 2  # Tool pairs come after speech
_PRIORITY_CONTINUATION = 3  # Continuation of split segment comes last


def _apply_containment_rules(
    self_segments: list[_SpeechSegment],
    other_segments: list[_SpeechSegment],
    tool_pairs: list[_ToolPair],
    silence_map: Optional[_SilencePeriodMap] = None,
) -> list[Message]:
    """
    Apply containment-aware linearization rules to segments.

    Rules:
    - Containment: If X is contained in Y, split Y at X's end and insert X there
    - Partial overlap: Order by start time, other first on tie
    - No overlap: Natural chronological order

    Args:
        self_segments: Speech segments from self.
        other_segments: Speech segments from other.
        tool_pairs: Tool call/result pairs.
        silence_map: Map from resume_tick to pre-created silence annotation message.

    Returns:
        Flattened list of messages in linearized order.
    """
    from collections import namedtuple

    OutputSegment = namedtuple("OutputSegment", ["tick", "priority", "messages"])
    output_segments: list = []

    # Add silence annotations as OutputSegments
    if silence_map:
        for resume_tick, annotation in silence_map.items():
            output_segments.append(
                OutputSegment(resume_tick, _PRIORITY_SILENCE, [annotation])
            )

    # Track which segments have been processed
    processed_self: set[int] = set()
    processed_other: set[int] = set()

    # First pass: Find all containment relationships
    # Map: other_idx -> list of (self_idx, self_start, self_end, self_msgs)
    selfs_contained_in_other: dict[int, list[tuple[int, int, int, dict]]] = {}
    # Map: self_idx -> list of (other_idx, other_start, other_end, other_msgs)
    others_contained_in_self: dict[int, list[tuple[int, int, int, dict]]] = {}

    for self_idx, (self_start, self_end, self_msgs) in enumerate(self_segments):
        for other_idx, (o_start, o_end, o_msgs) in enumerate(other_segments):
            if _is_segment_contained(self_start, self_end, o_start, o_end):
                # Self contained in other
                if other_idx not in selfs_contained_in_other:
                    selfs_contained_in_other[other_idx] = []
                selfs_contained_in_other[other_idx].append(
                    (self_idx, self_start, self_end, self_msgs)
                )
            elif _is_segment_contained(o_start, o_end, self_start, self_end):
                # Other contained in self
                if self_idx not in others_contained_in_self:
                    others_contained_in_self[self_idx] = []
                others_contained_in_self[self_idx].append(
                    (other_idx, o_start, o_end, o_msgs)
                )

    # Process other segments that contain self segment(s)
    for other_idx, contained_selfs in selfs_contained_in_other.items():
        o_start, o_end, o_msgs = other_segments[other_idx]
        processed_other.add(other_idx)

        # Sort contained selfs by end tick
        contained_selfs.sort(key=lambda x: x[2])  # sort by self_end

        last_split = o_start - 1

        for self_idx, self_start, self_end, self_msgs in contained_selfs:
            processed_self.add(self_idx)

            # Get other messages from last_split+1 to self_end
            other_before = [
                o_msgs[t] for t in sorted(o_msgs.keys()) if last_split < t <= self_end
            ]

            if other_before:
                output_segments.append(
                    OutputSegment(
                        self_end, _PRIORITY_OTHER, consolidate_messages(other_before)
                    )
                )

            # Output self segment
            output_segments.append(
                OutputSegment(
                    self_end,
                    _PRIORITY_SELF,
                    consolidate_messages(_get_segment_messages(self_msgs)),
                )
            )

            last_split = self_end

        # Output remaining other messages after last contained self
        other_after = [o_msgs[t] for t in sorted(o_msgs.keys()) if t > last_split]
        if other_after:
            output_segments.append(
                OutputSegment(
                    last_split + 1,
                    _PRIORITY_CONTINUATION,
                    consolidate_messages(other_after),
                )
            )

    # Process self segments that contain other segment(s)
    for self_idx, contained_others in others_contained_in_self.items():
        if self_idx in processed_self:
            continue  # Already handled

        self_start, self_end, self_msgs = self_segments[self_idx]
        processed_self.add(self_idx)

        # Sort contained others by end tick
        contained_others.sort(key=lambda x: x[2])  # sort by other_end

        last_split = self_start - 1

        for other_idx, o_start, o_end, o_msgs in contained_others:
            processed_other.add(other_idx)

            # Get self messages from last_split+1 to o_end
            self_before = [
                self_msgs[t]
                for t in sorted(self_msgs.keys())
                if last_split < t <= o_end
            ]

            if self_before:
                output_segments.append(
                    OutputSegment(
                        o_end,
                        _PRIORITY_SELF,
                        consolidate_messages(self_before),
                    )
                )

            # Output other segment
            output_segments.append(
                OutputSegment(
                    o_end,
                    _PRIORITY_CONTINUATION,
                    consolidate_messages(_get_segment_messages(o_msgs)),
                )
            )

            last_split = o_end

        # Output remaining self messages after last contained other
        self_after = [self_msgs[t] for t in sorted(self_msgs.keys()) if t > last_split]
        if self_after:
            output_segments.append(
                OutputSegment(
                    last_split + 1,
                    _PRIORITY_SELF,
                    consolidate_messages(self_after),
                )
            )

    # Process remaining self segments (partial overlap or no overlap)
    for self_idx, (self_start, self_end, self_msgs) in enumerate(self_segments):
        if self_idx in processed_self:
            continue

        # Check for partial overlap (crossing boundaries)
        overlapping_others = [
            (idx, o_start, o_end, o_msgs)
            for idx, (o_start, o_end, o_msgs) in enumerate(other_segments)
            if idx not in processed_other
            and _segments_overlap(self_start, self_end, o_start, o_end)
        ]

        if overlapping_others:
            # Partial overlap - order by start time, other first on tie
            processed_self.add(self_idx)
            for other_idx, o_start, o_end, o_msgs in overlapping_others:
                processed_other.add(other_idx)
                output_segments.append(
                    OutputSegment(
                        o_start,
                        _PRIORITY_OTHER,
                        consolidate_messages(_get_segment_messages(o_msgs)),
                    )
                )

            # Self comes after on tie (higher priority number)
            output_segments.append(
                OutputSegment(
                    self_start,
                    _PRIORITY_SELF,
                    consolidate_messages(_get_segment_messages(self_msgs)),
                )
            )
            continue

        # No overlap - just add self segment
        processed_self.add(self_idx)
        output_segments.append(
            OutputSegment(
                self_start,
                _PRIORITY_SELF,
                consolidate_messages(_get_segment_messages(self_msgs)),
            )
        )

    # Add any unprocessed other segments
    for other_idx, (o_start, o_end, o_msgs) in enumerate(other_segments):
        if other_idx not in processed_other:
            output_segments.append(
                OutputSegment(
                    o_start,
                    _PRIORITY_OTHER,
                    consolidate_messages(_get_segment_messages(o_msgs)),
                )
            )

    # Add tool pairs (they get inserted at their call tick)
    # Group by call_tick and tool_call message to avoid duplicates
    tool_groups: dict[tuple[int, int], tuple[Message, list[Message]]] = {}
    for call_tick, tool_call, tool_result in tool_pairs:
        # Use id of tool_call message to group (same message can have multiple calls)
        key = (call_tick, id(tool_call))
        if key not in tool_groups:
            tool_groups[key] = (tool_call, [])
        if tool_result is not None:
            tool_groups[key][1].append(tool_result)

    # Add grouped tool pairs to output
    for (call_tick, _), (tool_call, tool_results) in tool_groups.items():
        if tool_results:
            output_segments.append(
                OutputSegment(call_tick, _PRIORITY_TOOL, [tool_call] + tool_results)
            )
        else:
            # Unmatched tool call (results haven't arrived yet)
            output_segments.append(
                OutputSegment(call_tick, _PRIORITY_TOOL, [tool_call])
            )

    # Sort by (tick, priority) and flatten
    output_segments.sort(key=lambda seg: (seg.tick, seg.priority))
    result = []
    for seg in output_segments:
        result.extend(seg.messages)

    return result


def _linearize_with_containment_awareness(
    ticks: list[ParticipantTick],
    integration_ticks: int = 1,
    silence_map: Optional[_SilencePeriodMap] = None,
) -> list[Message]:
    """
    Linearize ticks using containment-aware splice-merge algorithm.

    This algorithm converts concurrent tick-based conversation history into a
    sequential message list suitable for LLM context, with special handling for
    overlapping speech based on containment relationships.

    ## Algorithm Summary

    **"If you speak entirely during someone else's turn, you get inserted where
    you stopped. Otherwise, whoever started first goes first."**

    ## Detailed Rules

    1. **Segment Formation**
       Form continuous speech segments for both `self` and `other`, using the
       `integration_ticks` parameter to tolerate brief pauses within a segment.

    2. **Overlap Classification & Handling**

       For any pair of overlapping segments (one from self, one from other):

       **Case A: Containment** - One segment fully inside another
       If segment X is contained in segment Y (X.start ≥ Y.start AND X.end ≤ Y.end):
       - Split Y at the tick where X ends
       - Insert X at that split point
       - Result: Y[start → X.end] → X → Y[X.end → end]

       If multiple segments are contained within Y, each creates a break:
       - Result: Y[start → X₁.end] → X₁ → Y[X₁.end → X₂.end] → X₂ → Y[X₂.end → end]

       **Case B: Partial Overlap** - Segments cross but neither is contained
       - Order by start time
       - Tie-breaker: if same start time, `other` comes first, `self` comes second

       **Case C: No Overlap**
       - Place segments in natural chronological order by start tick

    Args:
        ticks: The tick-based conversation history.
        integration_ticks: Number of consecutive silent ticks before a speech
            segment ends. Higher values merge segments separated by brief pauses.
            Default is 1 (segments end immediately when speaker stops).
            Minimum value is 1 (values < 1 are treated as 1).
        silence_map: Map from resume_tick to pre-created silence annotation message.

    Returns:
        A flat list of messages with overlaps resolved according to containment rules.
    """
    # Ensure integration_ticks is at least 1
    integration_ticks = max(1, integration_ticks)

    # Phase 1: Collect segments
    self_segments, other_segments, tool_pairs = _collect_containment_segments(
        ticks, integration_ticks
    )

    # Phase 2: Apply containment rules and linearize
    return _apply_containment_rules(
        self_segments, other_segments, tool_pairs, silence_map
    )


def _debug_save_linearization_call(
    ticks: list[ParticipantTick],
    integration_ticks: int,
    stage: str,
    messages: Optional[list[Message]] = None,
) -> None:
    """
    Debug helper to save linearization input/output to a file.

    Creates a JSON file with timestamp showing the ticks and messages
    for each linearization call.

    Args:
        ticks: Input ticks to the linearization function
        integration_ticks: Integration ticks parameter
        stage: "input" or "output" to distinguish call stages
        messages: Output messages (only provided for stage="output")
    """
    try:
        # Create debug directory
        debug_dir = Path("tmp/linearization_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = debug_dir / f"linearization_{timestamp}_{stage}.json"

        # Prepare debug data
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "integration_ticks": integration_ticks,
            "num_ticks": len(ticks),
        }

        # Serialize ticks
        ticks_data = []
        for i, tick in enumerate(ticks):
            tick_data = {
                "tick_id": tick.tick_id,
                "timestamp": tick.timestamp,
                "self_chunk": (
                    _serialize_message(tick.self_chunk) if tick.self_chunk else None
                ),
                "other_chunk": (
                    _serialize_message(tick.other_chunk) if tick.other_chunk else None
                ),
                "env_chunk": (
                    _serialize_message(tick.env_chunk) if tick.env_chunk else None
                ),
            }
            ticks_data.append(tick_data)
        debug_data["ticks"] = ticks_data

        # Serialize output messages if provided
        if messages is not None:
            messages_data = []
            for msg in messages:
                messages_data.append(_serialize_message(msg))
            debug_data["output_messages"] = messages_data
            debug_data["num_output_messages"] = len(messages)
            consolidated = consolidate_messages(messages)
            debug_data["consolidated_messages"] = [
                _serialize_message(msg) for msg in consolidated
            ]
            debug_data["num_consolidated_messages"] = len(consolidated)

        # Write to file
        with open(filename, "w") as f:
            json.dump(debug_data, f, indent=2)

        logger.debug(f"Linearization debug saved to: {filename}")

    except Exception as e:
        # Don't fail the main function if debug logging fails
        logger.warning(f"Failed to save linearization debug: {e}")


def _debug_save_get_linearized_messages_call(
    state: "StreamingState",
    strategy: LinearizationStrategy,
    include_pending_input: bool,
    indicate_current_incomplete: bool,
    integration_ticks: int,
    stage: str,
    linearize_ticks_output: Optional[list[Message]] = None,
    final_messages: Optional[list[Message]] = None,
) -> None:
    """
    Debug helper to save get_linearized_messages input/output to a file.

    Creates a JSON file with timestamp showing the state, intermediate output
    from linearize_ticks, and final messages.

    Args:
        state: The StreamingState object
        strategy: Linearization strategy used
        include_pending_input: Whether to include pending input
        indicate_current_incomplete: Whether to indicate incomplete chunks
        integration_ticks: Integration ticks parameter
        stage: "input" or "output" to distinguish call stages
        linearize_ticks_output: Output from linearize_ticks (only for stage="output")
        final_messages: Final output messages (only for stage="output")
    """
    try:
        # Create debug directory
        debug_dir = Path("tmp/get_linearized_messages_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = debug_dir / f"get_linearized_messages_{timestamp}_{stage}.json"

        # Prepare debug data
        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "strategy": strategy,
            "include_pending_input": include_pending_input,
            "indicate_current_incomplete": indicate_current_incomplete,
            "integration_ticks": integration_ticks,
        }

        # Serialize state
        debug_data["state"] = {
            "num_ticks": len(state.ticks),
            "num_input_turn_taking_buffer": len(state.input_turn_taking_buffer),
            "num_output_streaming_queue": len(state.output_streaming_queue),
            "time_since_last_talk": state.time_since_last_talk,
            "time_since_last_other_talk": state.time_since_last_other_talk,
            "is_talking": state.is_talking,
        }

        # Serialize ticks from state
        ticks_data = []
        for tick in state.ticks:
            tick_data = {
                "tick_id": tick.tick_id,
                "timestamp": tick.timestamp,
                "self_chunk": (
                    _serialize_message(tick.self_chunk) if tick.self_chunk else None
                ),
                "other_chunk": (
                    _serialize_message(tick.other_chunk) if tick.other_chunk else None
                ),
                "env_chunk": (
                    _serialize_message(tick.env_chunk) if tick.env_chunk else None
                ),
            }
            ticks_data.append(tick_data)
        debug_data["state"]["ticks"] = ticks_data

        # Serialize input_turn_taking_buffer
        buffer_data = []
        for msg in state.input_turn_taking_buffer:
            buffer_data.append(_serialize_message(msg))
        debug_data["state"]["input_turn_taking_buffer"] = buffer_data

        # Serialize output_streaming_queue
        queue_data = []
        for msg in state.output_streaming_queue:
            queue_data.append(_serialize_message(msg))
        debug_data["state"]["output_streaming_queue"] = queue_data

        # Serialize intermediate and final outputs if provided (stage="output")
        if linearize_ticks_output is not None:
            linearize_ticks_data = []
            for msg in linearize_ticks_output:
                linearize_ticks_data.append(_serialize_message(msg))
            debug_data["linearize_ticks_output"] = linearize_ticks_data
            debug_data["num_linearize_ticks_output"] = len(linearize_ticks_output)

        if final_messages is not None:
            final_messages_data = []
            for msg in final_messages:
                final_messages_data.append(_serialize_message(msg))
            debug_data["final_messages"] = final_messages_data
            debug_data["num_final_messages"] = len(final_messages)

            # Also include consolidated version for easier reading
            consolidated = consolidate_messages(final_messages)
            debug_data["consolidated_final_messages"] = [
                _serialize_message(msg) for msg in consolidated
            ]
            debug_data["num_consolidated_final_messages"] = len(consolidated)

        # Write to file
        with open(filename, "w") as f:
            json.dump(debug_data, f, indent=2)

        logger.debug(f"get_linearized_messages debug saved to: {filename}")

    except Exception as e:
        # Don't fail the main function if debug logging fails
        logger.warning(f"Failed to save get_linearized_messages debug: {e}")


def _serialize_message(msg: Message) -> dict:
    """Helper to serialize a message to a JSON-compatible dict."""
    from tau2.data_model.message import MultiToolMessage

    if msg is None:
        return None

    # Get basic fields
    data = {
        "role": msg.role,
    }

    # Add type info
    data["message_type"] = type(msg).__name__

    # Handle MultiToolMessage specially - it doesn't have content/timestamp
    if isinstance(msg, MultiToolMessage):
        data["tool_messages"] = [_serialize_message(tm) for tm in msg.tool_messages]
    else:
        # Regular messages have content and timestamp
        data["content"] = getattr(msg, "content", None)
        data["timestamp"] = getattr(msg, "timestamp", None)

    # Add optional fields if present
    if hasattr(msg, "chunk_id") and msg.chunk_id is not None:
        data["chunk_id"] = msg.chunk_id
    if hasattr(msg, "is_final_chunk") and msg.is_final_chunk is not None:
        data["is_final_chunk"] = msg.is_final_chunk
    if hasattr(msg, "contains_speech"):
        data["contains_speech"] = msg.contains_speech
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = [{"id": tc.id, "name": tc.name} for tc in msg.tool_calls]
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "is_audio") and msg.is_audio:
        data["is_audio"] = msg.is_audio
    if hasattr(msg, "audio_script_gold") and msg.audio_script_gold:
        data["audio_script_gold"] = msg.audio_script_gold

    return data


def _add_chunk_to_list(chunk: Message, messages: list[Message]) -> None:
    """Add a chunk to a message list, expanding MultiToolMessage if needed."""
    from tau2.data_model.message import MultiToolMessage

    if chunk is None:
        return
    if isinstance(chunk, MultiToolMessage):
        messages.extend(chunk.tool_messages)
    else:
        messages.append(chunk)


def _extract_tick_messages_simple(
    tick: ParticipantTick,
    strategy: LinearizationStrategy,
) -> list[Message]:
    """
    Extract messages from a single tick in the order determined by strategy.

    This is used for non-CONSOLIDATED strategies (TIMESTAMP_ORDER, SELF_FIRST_PER_TICK,
    OTHER_FIRST_PER_TICK). For CONSOLIDATED strategy, use _linearize_with_overlap_handling.

    Note: Tool results come as other_chunk (ToolMessage or MultiToolMessage) in
    subsequent ticks. MultiToolMessage is expanded to individual ToolMessage items
    since the LLM API expects individual tool messages paired with their tool call IDs.

    Args:
        tick: The tick to extract messages from.
        strategy: The linearization strategy (should not be CONSOLIDATED).

    Returns:
        List of messages from this tick in the appropriate order.
    """
    from tau2.data_model.message import MultiToolMessage

    messages: list[Message] = []

    def add_chunk(chunk: Message) -> None:
        """Add a chunk to messages, expanding MultiToolMessage if needed."""
        if chunk is None or not _has_meaningful_content(chunk):
            return
        # Expand MultiToolMessage to individual ToolMessage items for LLM API compatibility
        if isinstance(chunk, MultiToolMessage):
            messages.extend(chunk.tool_messages)
        else:
            messages.append(chunk)

    # Determine order based on strategy
    if strategy == LinearizationStrategy.OTHER_FIRST_PER_TICK:
        # Other's messages first (typical for responding to incoming)
        # This includes both speech from other party AND tool results
        add_chunk(tick.other_chunk)
        add_chunk(tick.self_chunk)
    elif strategy == LinearizationStrategy.SELF_FIRST_PER_TICK:
        # Self's messages first
        add_chunk(tick.self_chunk)
        add_chunk(tick.other_chunk)
    elif strategy == LinearizationStrategy.TIMESTAMP_ORDER:
        # TIMESTAMP_ORDER - collect all, will be sorted later
        add_chunk(tick.other_chunk)
        add_chunk(tick.self_chunk)
    else:
        raise ValueError(f"Unsupported linerazation strategy: {strategy}")

    return messages


def _has_meaningful_content(chunk: Message) -> bool:
    """
    Check if a chunk has meaningful content for linearization.

    This includes both speech content AND tool-related messages, since tool
    messages must be included in LLM context to pair with their corresponding
    tool calls.

    Use this for linearization and tick recording where tool calls matter.
    Use _has_speech_content() for checking actual audible speech (e.g., silence detection).

    Args:
        chunk: The message chunk to check.

    Returns:
        True if the chunk has meaningful content, False otherwise.
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    if chunk is None:
        return False

    # Tool messages are always meaningful - they must be included to pair with tool calls
    if isinstance(chunk, ToolMessage):
        return True
    if isinstance(chunk, MultiToolMessage):
        return True

    # Messages with tool calls are always meaningful (even if contains_speech=False)
    # Tool calls must be included so tool results can be paired with them
    if hasattr(chunk, "is_tool_call") and chunk.is_tool_call():
        return True

    # Check contains_speech flag if available
    # Return True if contains_speech=True (audio content, even without transcript)
    # Return False if contains_speech=False (explicitly marked as no speech)
    if hasattr(chunk, "contains_speech"):
        if chunk.contains_speech is True:
            return True
        if chunk.contains_speech is False:
            return False

    # Check for actual content
    if hasattr(chunk, "content") and chunk.content:
        return True

    return False


def _has_speech_content(chunk: Message) -> bool:
    """
    Check if a chunk has actual speech content (excludes tool calls).

    Unlike _has_meaningful_content(), this function does NOT consider tool calls
    or tool messages as speech. Use this for silence detection where we only care
    about audible speech, not tool activity.

    Args:
        chunk: The message chunk to check.

    Returns:
        True if the chunk has actual speech content, False otherwise.
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    if chunk is None:
        return False

    # Tool messages are NOT speech - they're tool activity
    if isinstance(chunk, ToolMessage):
        return False
    if isinstance(chunk, MultiToolMessage):
        return False

    # Messages with tool calls are NOT speech - they're tool activity
    if hasattr(chunk, "is_tool_call") and chunk.is_tool_call():
        return False

    # Check contains_speech flag if available
    if hasattr(chunk, "contains_speech"):
        if chunk.contains_speech is True:
            return True
        if chunk.contains_speech is False:
            return False

    # Check for actual content
    if hasattr(chunk, "content") and chunk.content:
        return True

    return False


def compute_responsiveness_info(
    ticks: list,
    integration_ticks: int = 1,
) -> dict:
    """
    Compute agent responsiveness metrics from tick history.

    Analyzes the tick history to detect patterns where the user speaks
    multiple times without getting an agent response, which may indicate
    agent responsiveness issues.

    Args:
        ticks: List of Tick objects from the orchestrator.
        integration_ticks: Number of consecutive silent ticks before a speech
            segment ends. Higher values merge segments separated by brief pauses.

    Returns:
        Dict with responsiveness metrics:
        - total_user_turns: Number of distinct user speech segments
        - total_agent_turns: Number of distinct agent speech/activity segments
        - max_unresponded_user_turns: Max consecutive user turns without agent response
        - had_unresponsive_period: True if max_unresponded_user_turns >= 3
    """
    from tau2.data_model.message import Tick

    if not ticks:
        return {
            "total_user_turns": 0,
            "total_agent_turns": 0,
            "max_unresponded_user_turns": 0,
            "had_unresponsive_period": False,
        }

    # Convert orchestrator Ticks to ParticipantTicks (from agent's perspective)
    # agent_chunk is "self", user_chunk is "other"
    participant_ticks = []
    for t in ticks:
        if not isinstance(t, Tick):
            continue
        participant_ticks.append(
            ParticipantTick(
                tick_id=t.tick_id,
                timestamp=t.timestamp,
                self_chunk=t.agent_chunk,  # agent is "self"
                other_chunk=t.user_chunk,  # user is "other"
            )
        )

    if not participant_ticks:
        return {
            "total_user_turns": 0,
            "total_agent_turns": 0,
            "max_unresponded_user_turns": 0,
            "had_unresponsive_period": False,
        }

    # Collect speech segments using existing machinery
    # self_segments = agent segments, other_segments = user segments
    agent_segments, user_segments, tool_pairs = _collect_containment_segments(
        participant_ticks, integration_ticks
    )

    # Count agent activity events (speech segments + tool pairs)
    # Each tool pair counts as agent activity
    total_agent_turns = len(agent_segments) + len(tool_pairs)
    total_user_turns = len(user_segments)

    if total_user_turns == 0:
        return {
            "total_user_turns": 0,
            "total_agent_turns": total_agent_turns,
            "max_unresponded_user_turns": 0,
            "had_unresponsive_period": False,
        }

    # Count consecutive user turns without agent response
    # A user turn is "responded to" if:
    #   1. Agent was speaking DURING the user's turn (temporal overlap), OR
    #   2. Agent spoke AFTER the user finished but BEFORE the next user started
    # A user turn is "unresponded" if neither condition is met
    max_unresponded = 0
    current_unresponded = 0

    # Sort user and agent segments by start tick
    sorted_user_segments = sorted(user_segments, key=lambda s: s[0])
    sorted_agent_segments = sorted(agent_segments, key=lambda s: s[0])

    for i, (user_start, user_end, _) in enumerate(sorted_user_segments):
        # Determine the boundary for "response window" - either next user start or end of ticks
        if i + 1 < len(sorted_user_segments):
            next_user_start = sorted_user_segments[i + 1][0]
        else:
            # Last user segment - use a large value (end of conversation)
            next_user_start = float("inf")

        # Check if any agent segment overlaps with user OR occurs between this user and next
        # Agent "responds" if: agent was active at any point from user_start to next_user_start
        agent_responded = False
        for agent_start, agent_end, _ in sorted_agent_segments:
            # Agent segment overlaps with window [user_start, next_user_start)
            # This means agent was speaking during user's turn OR after user finished (before next user)
            if agent_start < next_user_start and agent_end >= user_start:
                agent_responded = True
                break

        # Also check tool pairs
        if not agent_responded:
            for call_tick, result_tick, _ in tool_pairs:
                # Tool activity in window [user_start, next_user_start)
                if call_tick < next_user_start and call_tick >= user_start:
                    agent_responded = True
                    break
                if (
                    result_tick
                    and result_tick < next_user_start
                    and result_tick >= user_start
                ):
                    agent_responded = True
                    break

        if agent_responded:
            # Agent responded to this user's turn - reset the unresponded count
            current_unresponded = 0
        else:
            # Agent did not respond - increment unresponded count
            current_unresponded += 1
            max_unresponded = max(max_unresponded, current_unresponded)

    return {
        "total_user_turns": total_user_turns,
        "total_agent_turns": total_agent_turns,
        "max_unresponded_user_turns": max_unresponded,
        "had_unresponsive_period": max_unresponded >= 3,
    }


StateType = TypeVar("StateType", bound=StreamingState)


class StreamingMixin(ABC, Generic[InputMessageType, OutputMessageType, StateType]):
    """
    Generic mixin to add streaming capabilities to conversation participants.

    This mixin provides the infrastructure for implementing get_next_chunk()
    which enables full-duplex communication. It includes:
    - Chunk buffering and management
    - Turn-taking logic hooks
    - Input/output chunk handling

    To use this mixin, your concrete class should inherit from both
    BaseStreamingParticipant (or its subclasses) and this mixin:

    Usage:
        from tau2.agent.base import BaseStreamingParticipant

        class MyStreamingParticipant(StreamingMixin, MyBaseClass):
            # This will implement BaseStreamingParticipant's get_next_chunk requirement
            pass

    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces
        StateType: The type of internal streaming state (should extend StreamingState)

    Note: Concrete classes must:
    1. Inherit from BaseStreamingParticipant (directly or indirectly)
    2. Override get_init_state() to return a StateType with streaming fields
    3. Implement the abstract turn-taking methods
    4. Implement the speech_detection method
    """

    def __init__(self, *args, chunk_size: int = 50, **kwargs):
        """
        Initialize the streaming mixin.

        Args:
            chunk_size: Number of sentences/words/characters/bytes per chunk.
                       Can be ignored if custom chunking is implemented.
        """
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    @abstractmethod
    def _next_turn_taking_action(self, state: StateType) -> TurnTakingAction:
        """
        Decide the next action to take in the turn-taking.

        Subclasses should implement this to define custom turn-taking logic.
        Possible actions:
            - Stop talking
            - Keep talking
            - Generate message
            - Wait
        """
        raise NotImplementedError

    @abstractmethod
    def _perform_turn_taking_action(
        self, state: StateType, action: TurnTakingAction
    ) -> Tuple[OutputMessageType, StateType]:
        """
        Perform the next action in the turn-taking.

        Subclasses should implement this to define custom turn-taking action logic.
        Args:
            state: The current state of the turn-taking.
            action: The action to perform.
        Returns:
            A tuple of the next chunk and the updated state.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_chunk_messages(
        self,
        full_message: OutputMessageType,
    ) -> list[OutputMessageType]:
        """
        Create a list of chunk messages from a full message.

        Subclasses should implement this to define custom chunking logic.

        Args:
            full_message: The complete message to chunk

        Returns:
            List of chunk messages
        """
        raise NotImplementedError

    @abstractmethod
    def speech_detection(self, chunk: InputMessageType) -> bool:
        """
        Check if the chunk is a speech chunk.

        Args:
            chunk: The chunk to check.

        Returns:
            True if the chunk is a speech chunk, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_tool_result(
        self, tool_result: EnvironmentMessage, state: StateType
    ) -> Tuple[OutputMessageType, StateType]:
        """
        Process a tool result by calling the LLM and returning the response.

        Called when tool results arrive (tool_results is not None in get_next_chunk).
        Similar to the generate_message flow but:
        - Takes a ToolMessage/MultiToolMessage directly (not merged from buffer)
        - Does NOT clear input_turn_taking_buffer (participant chunks preserved
          for timing)
        - Temporarily sets buffer to [tool_result] so
          get_linearized_messages(include_pending_input=True) picks it up for
          LLM context

        If the LLM returns a text response, subclasses should queue the chunks
        in output_streaming_queue and return a noise/silence chunk for the
        current tick (deferring speech delivery to normal turn-taking).

        If the LLM returns another tool call, return the tool call message
        directly (get_next_chunk will set waiting_for_tool_results=True).

        Args:
            tool_result: The tool result message(s) from the environment.
            state: The current streaming state.

        Returns:
            A tuple of the output chunk and updated state.
        """
        raise NotImplementedError

    @abstractmethod
    def _emit_waiting_chunk(
        self, state: StateType
    ) -> Tuple[OutputMessageType, StateType]:
        """
        Emit a chunk while waiting for tool results.

        Called on each tick while waiting_for_tool_results is True and no
        tool results have arrived yet. Voice participants should emit
        background noise; text participants should return an empty message.

        Args:
            state: The current streaming state.

        Returns:
            A tuple of the waiting chunk and updated state.
        """
        raise NotImplementedError

    def get_next_chunk(
        self,
        state: StateType,
        participant_chunk: InputMessageType,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[OutputMessageType, StateType]:
        """
        Get the next chunk of the conversation.

        This implements BaseStreamingParticipant's abstract method.
        Records the tick (concurrent events) in the tick-based history.

        Each tick, the participant receives two independent channels:
        - participant_chunk: speech/audio from the other participant
        - tool_results: results from previously executed tool calls (if any)

        Operates in three modes based on tool call state:
        1. NORMAL: standard turn-taking (generate, wait, keep_talking, etc.)
        2. WAITING: tool call was sent, waiting for results — emit noise/silence
        3. PROCESS TOOL RESULT: tool results arrived — call LLM with result

        Tool results are NEVER added to input_turn_taking_buffer. They are
        recorded as env_chunk in the tick history for linearization.

        Args:
            state: The current state of the conversation.
            participant_chunk: The incoming chunk from the other participant.
            tool_results: Tool results from the environment (ToolMessage or
                MultiToolMessage). None if no tool results are pending.

        Returns:
            A tuple of the next chunk and the updated state.
        """
        state.tick_count += 1

        # Always buffer participant chunk and update timing counters
        state.update_input_turn_taking_buffer(participant_chunk)
        is_speech_chunk = self.speech_detection(participant_chunk)
        logger.debug(f"Speech chunk detected: {is_speech_chunk}")

        agent_speech_just_started = (
            is_speech_chunk and state.time_since_last_other_talk > 0
        )

        if is_speech_chunk:
            state.time_since_last_other_talk = 0
            state.consecutive_other_speaking_ticks += 1
            state.ticks_since_last_backchannel += 1
        else:
            state.time_since_last_other_talk += 1
            state.consecutive_other_speaking_ticks = 0

        if agent_speech_just_started:
            state.ticks_since_last_backchannel = 1

        # --- Mode selection ---
        if tool_results is not None:
            # PROCESS TOOL RESULT: results arrived, process immediately
            logger.debug("Tool results received, processing tool result")
            state.waiting_for_tool_results = False
            next_chunk, state = self._process_tool_result(tool_results, state)
            if next_chunk is not None and next_chunk.is_tool_call():
                state.waiting_for_tool_results = True

        elif state.waiting_for_tool_results:
            # WAITING: emit noise/silence, don't respond to other participant
            logger.debug("Waiting for tool results, emitting waiting chunk")
            next_chunk, state = self._emit_waiting_chunk(state)

        else:
            # NORMAL: standard turn-taking
            next_action = self._next_turn_taking_action(state)
            next_chunk, state = self._perform_turn_taking_action(state, next_action)
            if next_chunk is not None and next_chunk.is_tool_call():
                state.waiting_for_tool_results = True

        # TODO: time_since_last_talk should also be updated here, instead of being updated separately in all the different subclasses in _perform_turn_taking_action.
        # Update consecutive self speaking ticks based on whether self emitted speech
        self_emitted_speech = _has_meaningful_content(next_chunk)
        if self_emitted_speech:
            state.consecutive_self_speaking_ticks += 1
        else:
            state.consecutive_self_speaking_ticks = 0

        # Record this tick in the tick-based history.
        # Both channels are recorded separately: other_chunk for participant speech,
        # env_chunk for tool results. Before linearization, _expand_env_chunks
        # transforms these into old-format ticks for backward compatibility.
        timestamp = get_now()
        should_record_other = is_speech_chunk or _has_meaningful_content(
            participant_chunk
        )
        state.record_tick(
            tick_id=len(state.ticks),
            timestamp=timestamp,
            self_chunk=next_chunk if _has_meaningful_content(next_chunk) else None,
            other_chunk=participant_chunk if should_record_other else None,
            env_chunk=tool_results,
        )

        return next_chunk, state


### Utils for streaming operations ###
def consolidate_messages(messages: list[OutputMessageType]) -> list[InputMessageType]:
    """
    Consolidate a sequence of messages by merging contiguous groups of the same type.

    Processes a heterogeneous list of messages and merges contiguous sequences
    of the same message type into single messages. This reduces the number of
    messages while preserving semantic boundaries.

    Only ParticipantMessageBase messages that are NOT tool calls will be merged.
    EnvironmentMessages, tool calls, and None values are kept separate and not merged.

    Args:
        messages: A list of messages that may include different types, tool calls,
                    environment messages, or None values.

    Returns:
        A consolidated list where contiguous mergeable messages of the same type
        have been combined into single messages.

    Example:
        [UserMsg1(chunk1), UserMsg1(chunk2), ToolCall, UserMsg2(chunk1), UserMsg2(chunk2)]
        → [UserMsg1(merged), ToolCall, UserMsg2(merged)]
    """
    result = []
    for message in messages:
        result = append_or_merge_chunk(result, message)
    return result


def merge_homogeneous_chunks(chunks: list[Message]) -> Message:
    """
    Merge a homogeneous list of chunks into a single message.

    All chunks in the list must be of the same type. This method validates
    the chunks and delegates to the message class's merge_chunks method.

    Args:
        chunks: A list of message chunks, all of the same type.

    Returns:
        A single merged message combining all the chunks.

    Raises:
        ValueError: If the chunk list is empty, contains mixed types,
                    or contains multiple EnvironmentMessages.
    """
    if len(chunks) == 0:
        raise ValueError("Cannot merge empty chunk list")

    if any(isinstance(chunk, EnvironmentMessage) for chunk in chunks):
        if len(chunks) > 1:
            raise ValueError(
                "If there is an environment message, it should be the only chunk."
            )
        return chunks[0]

    # All the chunks should be of the same type.
    if not all(isinstance(chunk, type(chunks[0])) for chunk in chunks):
        raise ValueError("All chunks should be of the same type.")

    if len(chunks) == 1:
        return chunks[0]

    # Use Message's merge_chunks if available
    message_class = type(chunks[0])
    if hasattr(message_class, "merge_chunks"):
        return message_class.merge_chunks(chunks)
    else:
        raise ValueError(f"Cannot merge chunks of type {message_class}.")


def append_or_merge_chunk(
    messages: list[Message], chunk: InputMessageType
) -> list[Message]:
    """
    Intelligently append or merge a message chunk to a list of messages.

    If the chunk can be merged with the last message in the list,
    it will be merged. Otherwise, it will be appended.

    Only ParticipantMessageBase messages without tool calls of the same type
    can be merged. EnvironmentMessages, tool calls, and None values are always
    appended.

    Args:
        messages: The list of messages to add to.
        chunk: The message chunk to add.

    Returns:
        Updated list of messages with the chunk added or merged.

    Example:
        # If last message is UserMessage(chunk1) and chunk is UserMessage(chunk2):
        # → Last message becomes UserMessage(merged)
        # If last message is ToolCall and chunk is UserMessage:
        # → UserMessage is appended as a new message
    """
    # If messages list is empty, just append the chunk
    if len(messages) == 0:
        messages.append(chunk)
        return messages

    last_message = messages[-1]

    # Determine if chunk can be merged with last message
    can_merge = can_merge_messages(last_message, chunk)

    if can_merge:
        # Merge the chunk with the last message
        merged = merge_homogeneous_chunks([last_message, chunk])
        messages[-1] = merged
    else:
        messages.append(chunk)

    return messages


def can_merge_messages(msg1: Message, msg2: Message) -> bool:
    """
    Check if two messages can be merged together.

    Messages can only be merged if:
    - Both are non-None ParticipantMessageBase
    - Neither is a tool call
    - Both are of the same type

    Args:
        msg1: First message (typically the last message in history).
        msg2: Second message (typically the incoming chunk).

    Returns:
        True if the messages can be merged, False otherwise.
    """
    # EnvironmentMessages cannot be merged
    if isinstance(msg1, EnvironmentMessage) or isinstance(msg2, EnvironmentMessage):
        return False

    # Both must be ParticipantMessageBase
    if not (
        isinstance(msg1, ParticipantMessageBase)
        and isinstance(msg2, ParticipantMessageBase)
    ):
        return False

    # Tool calls cannot be merged
    if msg1.is_tool_call() or msg2.is_tool_call():
        return False

    # Must be the same type
    if type(msg1) is not type(msg2):
        return False

    return True


class AudioChunkingMixin(
    StreamingMixin[InputMessageType, OutputMessageType, StateType]
):
    """
    Generic streaming mixin with audio chunking.
    Type Parameters:
        InputMessageType: The type of messages this participant receives
        OutputMessageType: The type of messages this participant produces (must support chunking)
        StateType: The type of internal streaming state
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize audio chunking mixin.
        """
        super().__init__(*args, **kwargs)

    def _create_chunk_messages(
        self,
        full_message: OutputMessageType,
    ) -> list[OutputMessageType]:
        """Create chunk messages from a full audio message.

        This method splits both the audio data and text content into chunks for
        streaming. The audio is chunked by bytes (using chunk_size samples per chunk),
        while the text content is distributed evenly across all chunks character-by-character.

        Key behaviors:
        - Audio is chunked into fixed-size chunks (with the last chunk padded if needed)
        - Text content (audio_script_gold) is distributed character-by-character across
          chunks using an interval-based approach to ensure even distribution
        - Each chunk message includes:
          * Base64-encoded audio data
          * A portion of the text content (may be empty if fewer chars than chunks)
          * Chunk metadata (chunk_id, is_final_chunk)
          * A marked audio_script_gold with UUID, active chunk ID, and all chunk boundaries
        - Metadata (raw_data, cost, usage) is only included in the first chunk to avoid
          duplication
        - A unique UUID is generated per message. Each chunk's audio_script_gold contains
          the full template with all chunks tagged, plus an 'active' attribute:
          * Chunk 0: `<message uuid="abc" active="0"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>`
          * Chunk 1: `<message uuid="abc" active="1"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>`
          When merged, only received chunks (based on 'active' attrs) retain their tags.

        Args:
            full_message: The complete audio message to chunk. Must have is_audio=True
                         and valid audio content.

        Returns:
            A list of chunk messages with evenly distributed audio and text content.
        """

        chunks = self._chunk_by_bytes(full_message)
        chunk_messages = []
        chunks_chars = self._create_content_chunks(
            full_message.audio_script_gold, len(chunks)
        )

        # Generate a unique message UUID for chunk correlation
        utterance_id = str(uuid.uuid4())

        # Build the full template with all chunks tagged
        all_chunks_marked = [
            f"<chunk id={j}>{chunk_text}</chunk>"
            for j, chunk_text in enumerate(chunks_chars)
        ]
        template = "".join(all_chunks_marked)

        for i, chunk in enumerate(chunks):
            # Each chunk's audio_script_gold contains the full template with all chunks tagged,
            # plus an 'active' attribute indicating which chunk this message represents.
            # This allows merge to:
            # 1. Know the full text structure from any chunk
            # 2. Determine which chunks were received by checking active attributes
            chunk_audio_script_gold = (
                f'<message uuid="{utterance_id}" active="{i}">{template}</message>'
            )

            # Convert AudioData to base64-encoded string for audio_content
            chunk_message = type(full_message)(
                role=full_message.role,
                content=chunks_chars[i],
                is_audio=True,
                audio_content=audio_bytes_to_string(chunk.data),
                audio_format=chunk.format,
                utterance_ids=[utterance_id],
                chunk_id=i,
                is_final_chunk=i == len(chunks) - 1,
                audio_script_gold=chunk_audio_script_gold,
                raw_data=full_message.raw_data if i == 0 else None,
                cost=full_message.cost if i == 0 else 0.0,
                usage=full_message.usage if i == 0 else None,
                speech_effects=full_message.speech_effects,
                source_effects=full_message.source_effects,
                channel_effects=full_message.channel_effects,
            )
            chunk_messages.append(chunk_message)
        return chunk_messages

    def _create_content_chunks(self, content: str, num_chunks: int) -> list[str]:
        """Create content chunks from a string.

        Distributes characters as evenly as possible across all chunks using a
        symmetric interval-based approach. Characters are assigned to chunks such
        that they are interspersed evenly rather than bunched at the start.

        Args:
            content: The string content to split into chunks
            num_chunks: The number of chunks to create

        Returns:
            A list of string chunks with characters distributed evenly
        """
        num_chars = len(content)

        # Initialize all chunks as empty lists to accumulate characters
        chunk_lists: list[list[str]] = [[] for _ in range(num_chunks)]

        # Distribute each character to a chunk using interval-based assignment
        if num_chars > 0:
            chunk_interval = num_chunks / num_chars
            for char_idx in range(num_chars):
                chunk_idx = int(char_idx * chunk_interval)
                chunk_lists[chunk_idx].append(content[char_idx])

        # Convert lists to strings
        chunks = ["".join(chunk_list) for chunk_list in chunk_lists]

        return chunks

    def _chunk_by_bytes(
        self,
        full_message: OutputMessageType,
    ) -> list[AudioData]:
        """Chunk audio bytes into chunks.
        chunk_size represents the number of samples per chunk.

        All chunks will have exactly chunk_size samples. The last chunk is padded
        with zeros if necessary to maintain consistent chunk sizes.

        Args:
            full_message: The full message to chunk.
        Returns:
            A list of AudioData objects, all with chunk_size samples.
        """

        if not full_message.is_audio:
            raise ValueError(f"Message is not audio: {full_message}")
        if not full_message.has_audio_content():
            raise ValueError(f"Message has no audio content: {full_message}")
        bytes_per_sample = full_message.audio_format.bytes_per_sample
        audio_bytes = full_message.get_audio_bytes()
        bytes_per_chunk = self.chunk_size * bytes_per_sample
        chunks = []
        for i in range(0, len(audio_bytes), bytes_per_chunk):
            chunk_audio = AudioData(
                data=audio_bytes[i : i + bytes_per_chunk],
                format=deepcopy(full_message.audio_format),
                audio_path=None,
            )
            # Pad last chunk if necessary to maintain consistent chunk size
            chunk_audio = pad_audio_with_zeros(chunk_audio, self.chunk_size)
            chunks.append(chunk_audio)
        return chunks


def check_threshold(
    value: float,
    threshold: float,
    variance_factor: Optional[float] = None,
) -> bool:
    """
    Check if a value is above a threshold.

    If variance_factor is not provided, it is set to 0 and the system is deterministic.
    Args:
        value: The current value to evaluate
        threshold: The minimum value below which the probability is always 0
        variance_factor: The factor by which the threshold is varied.
        deterministic: Whether to return a deterministic value or a stochastic value.
    Returns:
        True with probability that increases linearly from 0 to 1 as value goes from threshold to the threshold + variance_factor.
        Always returns False if value < threshold.
        Always returns True if value >= threshold + variance_factor.
    """
    if variance_factor is None:
        variance_factor = 0

    if value < threshold:
        return False

    max_value = threshold + variance_factor if variance_factor > 0 else threshold
    if value >= max_value:
        return True

    # Linear interpolation between threshold and max_value
    probability = (value - threshold) / (max_value - threshold)
    return random.random() < probability


BasicActionType = Literal[  #
    "stop_talking",
    "keep_talking",
    "generate_message",
    "wait",
    "backchannel",
]


def should_backchannel(
    ticks_since_last_backchannel: int,
    ongoing_speech_duration: int,
    min_threshold: int,
    max_threshold: int,
    poisson_rate: float,
    tick_duration_seconds: float,
    rng: Optional[random.Random] = None,
) -> tuple[bool, str]:
    """Determine if a backchannel should be triggered."""
    # No backchannel if no ongoing speech
    if ongoing_speech_duration <= 0:
        return False, "No ongoing speech"

    if ticks_since_last_backchannel < min_threshold:
        return (
            False,
            f"Below min threshold ({ticks_since_last_backchannel} < {min_threshold})",
        )

    if ticks_since_last_backchannel >= max_threshold:
        return (
            True,
            f"Forced at max threshold ({ticks_since_last_backchannel} >= {max_threshold})",
        )

    # Between min and max - use Poisson probability
    if rng is None:
        rng = random.Random()
    triggered = poisson_should_trigger(poisson_rate, tick_duration_seconds, rng)
    return triggered, f"Poisson (rate={poisson_rate:.6f}/s, triggered={triggered})"


@dataclass
class ListenerReactionDecision:
    """Result of a single listener reaction callback with metadata from the LLM call."""

    decision: bool = False
    generation_time_seconds: Optional[float] = None
    cost: Optional[float] = None
    usage: Optional[dict] = None


# Type alias for listener reaction callbacks - now returns full metadata
ListenerReactionCallback = Callable[[StreamingState], ListenerReactionDecision]


@dataclass
class ListenerReactionResult:
    """Result of evaluating listener reactions with timing metadata."""

    should_interrupt: bool = False
    should_backchannel: bool = False
    interrupt_check_seconds: Optional[float] = None
    backchannel_check_seconds: Optional[float] = None
    interrupt_cost: Optional[float] = None
    backchannel_cost: Optional[float] = None
    interrupt_usage: Optional[dict] = None
    backchannel_usage: Optional[dict] = None


def _run_callback(
    callback: ListenerReactionCallback, state: StreamingState
) -> ListenerReactionDecision:
    """Run a callback and return the result, handling exceptions."""
    try:
        return callback(state)
    except Exception as e:
        logger.error(f"Error in listener reaction callback: {e}")
        return ListenerReactionDecision(decision=False)


def evaluate_listener_reactions(
    state: StreamingState,
    should_interrupt_callback: Optional[ListenerReactionCallback],
    should_backchannel_callback: Optional[ListenerReactionCallback],
) -> ListenerReactionResult:
    """
    Evaluate interrupt and backchannel decisions in parallel using ThreadPoolExecutor.

    Both callbacks are executed concurrently to minimize latency. Each callback
    returns a ListenerReactionDecision with the decision and metadata from the LLM call.

    Args:
        state: The current streaming state
        should_interrupt_callback: Callback that returns ListenerReactionDecision
        should_backchannel_callback: Callback that returns ListenerReactionDecision

    Returns:
        ListenerReactionResult with decision bools and metadata (timing, cost, usage).
        Priority is handled by the caller (interrupt > backchannel > none).
    """
    result = ListenerReactionResult()

    # Collect callbacks that need to be executed
    callbacks_to_run: list[tuple[str, ListenerReactionCallback]] = []
    if should_interrupt_callback is not None:
        callbacks_to_run.append(("interrupt", should_interrupt_callback))
    if should_backchannel_callback is not None:
        callbacks_to_run.append(("backchannel", should_backchannel_callback))

    if not callbacks_to_run:
        return result

    # Run callbacks in parallel, each with its own context copy to preserve ContextVars
    # (like llm_log_dir). Each thread needs its own context copy since a context
    # can only be entered once at a time.
    decisions: dict[str, ListenerReactionDecision] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_name = {
            executor.submit(
                contextvars.copy_context().run,
                _run_callback,
                callback,
                state,
            ): name
            for name, callback in callbacks_to_run
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                decisions[name] = future.result()
            except Exception as e:
                logger.error(f"Error in {name} callback: {e}")
                decisions[name] = ListenerReactionDecision(decision=False)

    # Extract results with all metadata
    if "interrupt" in decisions:
        d = decisions["interrupt"]
        result.should_interrupt = d.decision
        result.interrupt_check_seconds = d.generation_time_seconds
        result.interrupt_cost = d.cost
        result.interrupt_usage = d.usage
    if "backchannel" in decisions:
        d = decisions["backchannel"]
        result.should_backchannel = d.decision
        result.backchannel_check_seconds = d.generation_time_seconds
        result.backchannel_cost = d.cost
        result.backchannel_usage = d.usage

    logger.debug(
        f"Listener reactions evaluated: interrupt={result.should_interrupt} ({result.interrupt_check_seconds:.3f}s), "
        f"backchannel={result.should_backchannel} ({result.backchannel_check_seconds:.3f}s)"
        if result.interrupt_check_seconds and result.backchannel_check_seconds
        else f"Listener reactions evaluated: interrupt={result.should_interrupt}, backchannel={result.should_backchannel}"
    )

    return result


def basic_turn_taking_policy(
    state: StateType,
    wait_to_respond_threshold_other: int = 2,
    wait_to_respond_threshold_self: int = 4,
    yield_threshold_when_interrupted: Optional[int] = None,
    yield_threshold_when_interrupting: Optional[int] = None,
    backchannel_min_threshold: Optional[int] = None,
    backchannel_max_threshold: Optional[int] = None,
    backchannel_poisson_rate: Optional[float] = None,
    tick_duration_seconds: float = 0.05,
    should_interrupt_callback: Optional[ListenerReactionCallback] = None,
    should_backchannel_callback: Optional[ListenerReactionCallback] = None,
    use_llm_backchannel: bool = True,
    listener_reaction_check_interval: Optional[int] = None,
    variance_factor: Optional[float] = None,
    discard_overlapping_speech: bool = False,
) -> tuple[BasicActionType, str]:
    """
    Decide the next action to take in the turn-taking.

    Possible actions:
        - stop_talking
        - keep_talking
        - generate_message
        - wait
        - backchannel

    Args:
        state: The current state of the turn-taking.
        wait_to_respond_threshold_other: Minimum time to wait since OTHER last spoke before generating a response.
            Both this AND wait_to_respond_threshold_self must be satisfied.
        wait_to_respond_threshold_self: Minimum time to wait since SELF last spoke before generating a response.
            Both this AND wait_to_respond_threshold_other must be satisfied.
        yield_threshold_when_interrupted: How long self keeps speaking when OTHER initiated the interruption
            (i.e., self was talking first, other started speaking over self). If None, cannot be interrupted.
        yield_threshold_when_interrupting: How long self keeps speaking when SELF initiated the interruption
            (i.e., other was talking first, self started speaking over other). Should be defaulted to
            yield_threshold_when_interrupted at initialization if not explicitly set.
        backchannel_min_threshold: Minimum ticks before Poisson backchanneling is allowed.
            Used when use_llm_backchannel=False. If None, Poisson backchannel is disabled.
        backchannel_max_threshold: Maximum ticks - force Poisson backchannel at this point.
            Used when use_llm_backchannel=False.
        backchannel_poisson_rate: Poisson rate (events per second) for probabilistic backchanneling.
            Used when use_llm_backchannel=False.
        tick_duration_seconds: Duration of each tick in seconds. Used for Poisson calculations.
        should_interrupt_callback: Listener reaction callback for interruption decisions.
            Called with (state) and should return True if user should interrupt, False otherwise.
            Only used when self is not talking and other participant is currently speaking.
        should_backchannel_callback: Listener reaction callback for backchannel decisions.
            Called with (state) and should return True if user should backchannel, False otherwise.
            Only used when use_llm_backchannel=True.
        use_llm_backchannel: If True, use should_backchannel_callback for backchannel decisions.
            If False, use Poisson-based backchannel logic with min/max thresholds.
        listener_reaction_check_interval: If set, only check listener reaction callbacks every N ticks.
            When None (default), checks every tick. Useful to reduce callback frequency.
            Both interrupt and backchannel callbacks use the same interval.
        variance_factor: The factor by which the threshold is varied.
            If None, the system is deterministic.
        discard_overlapping_speech: If True, clear the input buffer when self is talking
            and other has stopped speaking (failed interruption attempt). This makes
            total_speech_duration == 0 for turn-taking purposes, so self won't respond
            to the interrupted speech. Note: the speech is still recorded in tick history
            for analysis purposes, only the buffer used for turn-taking decisions is cleared.
            Default is False.

    Returns:
        A tuple of the next action to take in the turn-taking and a message explaining the action.
    """
    can_be_interrupted = yield_threshold_when_interrupted is not None
    can_use_interrupt_callback = should_interrupt_callback is not None
    can_use_backchannel_callback = (
        use_llm_backchannel and should_backchannel_callback is not None
    )
    can_use_poisson_backchannel = (
        not use_llm_backchannel and backchannel_min_threshold is not None
    )

    if state.is_talking:
        # Backchannels should not be interrupted - they're short acknowledgments
        if state.is_backchanneling:
            logger.debug(
                "DECISION: keep_talking - delivering backchannel (not interruptible)"
            )
            return "keep_talking", "Backchannel in progress"

        # Handle interruption from other party
        if state.input_interrupt():
            interruption_length = state.input_ongoing_speech_duration()

            # Choose threshold based on context
            overlap_initiator = state.overlap_initiator
            if state.delivering_tool_result_speech:
                # Tool result speech is assertive — always use the high
                # "interrupting" threshold so the participant holds its ground
                active_threshold = yield_threshold_when_interrupting
                threshold_type = "when_interrupting (delivering_tool_result)"
            else:
                # Normal overlap: choose based on who initiated
                if overlap_initiator == "self":
                    # Self initiated the interruption (self barged in on other)
                    active_threshold = yield_threshold_when_interrupting
                    threshold_type = "when_interrupting"
                else:
                    # Other initiated the interruption (other barged in on self)
                    active_threshold = yield_threshold_when_interrupted
                    threshold_type = "when_interrupted"

            if (
                can_be_interrupted
                and active_threshold is not None
                and check_threshold(
                    value=interruption_length,
                    threshold=active_threshold,
                    variance_factor=variance_factor,
                )
            ):
                logger.debug(
                    f"DECISION: stop_talking - "
                    f"interruption_length={interruption_length} >= threshold={active_threshold} ({threshold_type}), "
                    f"overlap_initiator={overlap_initiator}, "
                    f"self_speaking={state.consecutive_self_speaking_ticks}, "
                    f"other_speaking={state.consecutive_other_speaking_ticks}"
                )
                return (
                    "stop_talking",
                    f"Interruption above threshold ({threshold_type}), stop.",
                )
            else:
                logger.debug(
                    f"DECISION: keep_talking - "
                    f"interruption_length={interruption_length} < threshold={active_threshold} ({threshold_type}), "
                    f"overlap_initiator={overlap_initiator}, "
                    f"self_speaking={state.consecutive_self_speaking_ticks}, "
                    f"other_speaking={state.consecutive_other_speaking_ticks}"
                )
                return (
                    "keep_talking",
                    f"Interruption below threshold ({threshold_type}), ignore",
                )
        else:
            # No active interruption - self continues talking
            # If discard_overlapping_speech is enabled and other has stopped speaking
            # (but there was speech in the buffer), discard it as a failed interruption attempt
            if (
                discard_overlapping_speech
                and state.time_since_last_other_talk > state.time_since_last_talk
                and state.input_total_speech_duration() > 0
            ):
                discarded_chunks = state.input_total_speech_duration()
                state.input_turn_taking_buffer = []
                logger.debug(
                    f"DECISION: keep_talking - "
                    f"discarding {discarded_chunks} overlapping speech chunks (failed interruption), "
                    f"time_since_other={state.time_since_last_other_talk}, "
                    f"time_since_self={state.time_since_last_talk}"
                )
                return "keep_talking", "Ignoring overlapping speech."
            else:
                logger.debug(
                    f"DECISION: keep_talking - "
                    f"no interruption detected, continuing speech"
                )
                return "keep_talking", "No interruption"
    else:
        # Handle input from environment (e.g., tool results)
        if state.input_from_environment():
            logger.debug(
                "DECISION: generate_message - input from environment (tool result)"
            )
            return "generate_message", "Input from environment"

        # Check for listener reactions while other party is speaking (interrupt, backchannel, or keep listening)
        ongoing_speech_duration = state.input_ongoing_speech_duration()
        is_check_interval = (
            listener_reaction_check_interval is None
            or state.tick_count % listener_reaction_check_interval == 0
        )

        if ongoing_speech_duration > 0 and is_check_interval:
            # Other party is currently speaking - check how we should react

            # Determine which callbacks to run
            interrupt_cb = (
                should_interrupt_callback if can_use_interrupt_callback else None
            )
            backchannel_cb = (
                should_backchannel_callback if can_use_backchannel_callback else None
            )

            # Run listener reaction callbacks in parallel (if any)
            if interrupt_cb is not None or backchannel_cb is not None:
                reaction_result = evaluate_listener_reactions(
                    state, interrupt_cb, backchannel_cb
                )

                # Store timing/cost info on state for later use when creating TurnTakingAction
                # This allows the caller to access timing metadata
                state._listener_reaction_timing = {
                    "interrupt_check_seconds": reaction_result.interrupt_check_seconds,
                    "interrupt_check_cost": reaction_result.interrupt_cost,
                    "interrupt_check_usage": reaction_result.interrupt_usage,
                    "backchannel_check_seconds": reaction_result.backchannel_check_seconds,
                    "backchannel_check_cost": reaction_result.backchannel_cost,
                    "backchannel_check_usage": reaction_result.backchannel_usage,
                }

                # Priority: interrupt > backchannel > none
                if reaction_result.should_interrupt:
                    logger.debug(
                        f"DECISION: generate_message (interrupt) - "
                        f"callback decided to interrupt, ongoing_speech={ongoing_speech_duration} chunks"
                    )
                    return "generate_message", "Callback decided to interrupt."
                elif reaction_result.should_backchannel:
                    logger.debug(
                        f"DECISION: backchannel - "
                        f"callback decided to backchannel, ongoing_speech={ongoing_speech_duration} chunks"
                    )
                    return "backchannel", "Callback decided to backchannel."
                else:
                    # Callbacks decided to keep listening
                    logger.debug(
                        f"DECISION: wait (keep listening) - "
                        f"callbacks decided to keep listening, ongoing_speech={ongoing_speech_duration} chunks"
                    )
                    # Fall through to wait/Poisson backchannel logic below

        # Poisson-based backchannel (when use_llm_backchannel=False)
        if can_use_poisson_backchannel and ongoing_speech_duration > 0:
            ticks_since_bc = state.ticks_since_last_backchannel
            trigger, reason = should_backchannel(
                ticks_since_last_backchannel=ticks_since_bc,
                ongoing_speech_duration=ongoing_speech_duration,
                min_threshold=backchannel_min_threshold,
                max_threshold=backchannel_max_threshold,
                poisson_rate=backchannel_poisson_rate,
                tick_duration_seconds=tick_duration_seconds,
                rng=state.backchannel_rng,
            )
            if trigger:
                logger.debug(
                    f"DECISION: backchannel (Poisson) - "
                    f"{reason}, ongoing_speech={ongoing_speech_duration} chunks, "
                    f"ticks_since_bc={ticks_since_bc}"
                )
                return "backchannel", f"Poisson backchannel: {reason}"

        # Wait for a long enough silence to generate a message
        # Both thresholds must be satisfied:
        # - time_since_last_other_talk > wait_to_respond_threshold_other
        # - time_since_last_talk > wait_to_respond_threshold_self
        other_threshold_met = (
            state.time_since_last_other_talk > wait_to_respond_threshold_other
        )
        self_threshold_met = state.time_since_last_talk > wait_to_respond_threshold_self

        if other_threshold_met and self_threshold_met:
            logger.debug(
                f"DECISION: generate_message - "
                f"both silence thresholds met, "
                f"time_since_other={state.time_since_last_other_talk} > {wait_to_respond_threshold_other}, "
                f"time_since_self={state.time_since_last_talk} > {wait_to_respond_threshold_self}"
            )
            return "generate_message", "Both silence thresholds met."

        # Neither generating nor reacting - wait
        ongoing = state.input_ongoing_speech_duration()
        if ongoing > 0:
            logger.debug(
                f"DECISION: wait - "
                f"listening to ongoing speech ({ongoing} chunks), "
                f"time_since_other={state.time_since_last_other_talk}, "
                f"time_since_self={state.time_since_last_talk}"
            )
            return "wait", "Listening to ongoing speech."
        else:
            logger.debug(
                f"DECISION: wait - "
                f"silence (no ongoing speech), "
                f"time_since_other={state.time_since_last_other_talk}/{wait_to_respond_threshold_other}, "
                f"time_since_self={state.time_since_last_talk}/{wait_to_respond_threshold_self}"
            )
            return "wait", "Silence, waiting for thresholds."
