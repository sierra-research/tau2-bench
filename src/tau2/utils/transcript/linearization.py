"""Participant-context linearization, including overlap, silence and tools.

This view preserves containment-aware prompt ordering. It deliberately differs
from raw call transcripts, utterance-indexed audio, and tool replay order.
"""

from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    EnvironmentMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
)
from tau2.data_model.transcript import (
    InputMessageType,
    LinearizationStrategy,
    OutputMessageType,
    ParticipantSegments,
    ParticipantTick,
    SpeechSegment,
    ToolExchange,
)

_SilencePeriodMap = dict[int, Message]
_SpeechSegment = tuple[int, int, dict[int, Message]]
_ToolPair = tuple[int, Message, Message | None]
_PRIORITY_SILENCE = -1  # Silence annotations come first at their tick
_PRIORITY_OTHER = 0  # Other speech comes first
_PRIORITY_SELF = 1  # Self speech comes second
_PRIORITY_TOOL = 2  # Tool pairs come after speech
_PRIORITY_CONTINUATION = 3  # Continuation of split segment comes last


def expand_env_chunks(
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
        # Use _has_speech_content (not has_meaningful_content) to ignore tool calls
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
        tick_has_self = has_meaningful_content(tick.self_chunk)
        tick_has_other = has_meaningful_content(tick.other_chunk)
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
        has_self = has_meaningful_content(tick.self_chunk)
        has_other = has_meaningful_content(tick.other_chunk)
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

    return messages


def collect_containment_segments(
    ticks: list[ParticipantTick],
    integration_ticks: int,
) -> ParticipantSegments:
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
        Typed speech segments and tool exchanges, indexed in the input list.
    """
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    self_segments: list[_SpeechSegment] = []
    other_segments: list[_SpeechSegment] = []
    tool_pairs: list[ToolExchange] = []

    # Track tool calls waiting for results
    tool_call_buffer: dict[str, tuple[int, Message]] = {}

    for i, tick in enumerate(ticks):
        has_self = has_meaningful_content(tick.self_chunk)
        has_other = has_meaningful_content(tick.other_chunk)

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
                        tool_pairs.append(
                            ToolExchange(
                                call_index=call_tick,
                                call=tool_call,
                                result=individual_result,
                            )
                        )
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
                            tool_pairs.append(
                                ToolExchange(
                                    call_index=call_tick,
                                    call=tool_call,
                                    result=individual_result,
                                )
                            )
            else:
                # Single tool message
                tool_call_id = getattr(tool_result, "tool_call_id", None) or getattr(
                    tool_result, "id", None
                )

                # Try to match by ID
                matched = False
                if tool_call_id and tool_call_id in tool_call_buffer:
                    call_tick, tool_call = tool_call_buffer.pop(tool_call_id)
                    tool_pairs.append(
                        ToolExchange(
                            call_index=call_tick, call=tool_call, result=tool_result
                        )
                    )
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
                        tool_pairs.append(
                            ToolExchange(
                                call_index=call_tick, call=tool_call, result=tool_result
                            )
                        )

    # Add any unmatched tool calls (result hasn't arrived yet)
    # Use None as placeholder for the result
    for tool_id, (call_tick, tool_call) in tool_call_buffer.items():
        tool_pairs.append(
            ToolExchange(call_index=call_tick, call=tool_call, result=None)
        )

    return ParticipantSegments(
        self_speech=[
            SpeechSegment(start_index=start, end_index=end, chunks=chunks)
            for start, end, chunks in self_segments
        ],
        other_speech=[
            SpeechSegment(start_index=start, end_index=end, chunks=chunks)
            for start, end, chunks in other_segments
        ],
        tool_exchanges=tool_pairs,
    )


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
    segments = collect_containment_segments(ticks, integration_ticks)

    # Phase 2: Apply containment rules and linearize
    return _apply_containment_rules(
        [(s.start_index, s.end_index, s.chunks) for s in segments.self_speech],
        [(s.start_index, s.end_index, s.chunks) for s in segments.other_speech],
        [(t.call_index, t.call, t.result) for t in segments.tool_exchanges],
        silence_map,
    )


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
        if chunk is None or not has_meaningful_content(chunk):
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


def has_meaningful_content(chunk: Message | None) -> bool:
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

    # Streaming transcripts can arrive before or after their audible frames.
    # Keep their text in LLM history without treating them as speech activity.
    if hasattr(chunk, "content") and chunk.content:
        return True

    # Check contains_speech flag if available
    # Return True if contains_speech=True (audio content, even without transcript)
    # Return False if contains_speech=False (explicitly marked as no speech)
    if hasattr(chunk, "contains_speech"):
        if chunk.contains_speech is True:
            return True
        if chunk.contains_speech is False:
            return False

    return False


def _has_speech_content(chunk: Message) -> bool:
    """
    Check if a chunk has actual speech content (excludes tool calls).

    Unlike has_meaningful_content(), this function does NOT consider tool calls
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
