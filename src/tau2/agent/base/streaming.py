import contextvars
import random
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
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
from tau2.data_model.transcript import LinearizationStrategy, ParticipantTick
from tau2.utils.transcript.linearization import (
    collect_containment_segments,
    expand_env_chunks,
    has_meaningful_content,
    linearize_ticks,
)
from tau2.utils.utils import get_now
from tau2.voice.utils.audio_preprocessing import pad_audio_with_zeros
from tau2.voice.utils.probability import poisson_should_trigger

# Generic type variables for streaming mixins
InputMessageType = TypeVar("InputMessageType", bound=Message)
OutputMessageType = TypeVar("OutputMessageType", bound=Message)


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
        Convert tick-based history to text-only messages for LLM context.

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

        # LLM callers consume transcripts and tools, not concatenated recordings.
        # Keep speech flags/timing for segmentation and leave stored ticks intact.
        for index, tick in enumerate(ticks_to_process):
            updates = {}
            for field in ("self_chunk", "other_chunk"):
                chunk = getattr(tick, field)
                if isinstance(chunk, ParticipantMessageBase) and (
                    chunk.audio_content or chunk.audio_script_gold
                ):
                    updates[field] = chunk.model_copy(
                        update={"audio_content": None, "audio_script_gold": None}
                    )
            if updates:
                ticks_to_process[index] = tick.model_copy(update=updates)

        # Expand ticks with env_chunk into old-format ticks before linearization.
        # This ensures linearization sees the same tick structure as the base branch.
        ticks_to_process = expand_env_chunks(ticks_to_process)

        messages = linearize_ticks(
            ticks_to_process,
            strategy,
            integration_ticks,
            silence_threshold_ticks=silence_annotation_threshold_ticks,
            tick_duration_seconds=tick_duration_seconds,
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
    segments = collect_containment_segments(participant_ticks, integration_ticks)
    agent_segments = [
        (segment.start_index, segment.end_index, segment.chunks)
        for segment in segments.self_speech
    ]
    user_segments = [
        (segment.start_index, segment.end_index, segment.chunks)
        for segment in segments.other_speech
    ]
    tool_pairs = [
        (exchange.call_index, exchange.call, exchange.result)
        for exchange in segments.tool_exchanges
    ]

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
        self_emitted_speech = has_meaningful_content(next_chunk)
        if self_emitted_speech:
            state.consecutive_self_speaking_ticks += 1
        else:
            state.consecutive_self_speaking_ticks = 0

        # Record this tick in the tick-based history.
        # Both channels are recorded separately: other_chunk for participant speech,
        # env_chunk for tool results. Before linearization, expand_env_chunks
        # transforms these into old-format ticks for backward compatibility.
        timestamp = get_now()
        should_record_other = is_speech_chunk or has_meaningful_content(
            participant_chunk
        )
        state.record_tick(
            tick_id=len(state.ticks),
            timestamp=timestamp,
            self_chunk=next_chunk if has_meaningful_content(next_chunk) else None,
            other_chunk=participant_chunk if should_record_other else None,
            env_chunk=tool_results,
        )

        return next_chunk, state


### Utils for streaming operations ###


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
