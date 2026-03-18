"""Utilities for proportional transcript distribution."""

import math
from typing import Optional


def get_proportional_text(
    transcript: str,
    total_duration: float,
    audio_played: float,
    start_char: int = 0,
) -> tuple[str, int]:
    """Get proportional text based on audio played.

    Returns (text, new_char_position) for stateful tracking across calls.
    """
    if total_duration == 0 or not transcript:
        return "", start_char

    ratio = min(1.0, audio_played / total_duration)
    end_char = math.ceil(ratio * len(transcript))
    # Ensure we never go backwards - when transcript grows faster than audio
    # is played, the ratio can decrease, but we should never re-show text
    end_char = max(start_char, end_char)
    return transcript[start_char:end_char], end_char


def _collect_transcripts_from_ticks(
    ticks: list,
) -> dict[str, tuple[int, int, str]]:
    """Extract transcripts from tick events.

    Returns dict of {item_id: (start_ms, end_ms, transcript)}.
    """
    transcripts: dict[str, tuple[int, int, str]] = {}
    pending: dict[str, dict] = {}

    for tick in ticks:
        raw_data = (
            getattr(tick.agent_chunk, "raw_data", None) if tick.agent_chunk else None
        )
        if not raw_data or "events" not in raw_data:
            continue

        for event in raw_data["events"]:
            event_type = event.get("type", "")
            item_id = event.get("item_id")

            if event_type == "input_audio_buffer.speech_started" and item_id:
                pending[item_id] = {"start_ms": event.get("audio_start_ms", 0)}

            elif event_type == "input_audio_buffer.speech_stopped" and item_id:
                if item_id in pending:
                    pending[item_id]["end_ms"] = event.get("audio_end_ms", 0)

            elif (
                event_type == "conversation.item.input_audio_transcription.completed"
                and item_id
                and item_id in pending
            ):
                # Timing comes from speech_started/stopped events
                transcripts[item_id] = (
                    pending[item_id].get("start_ms", 0),
                    pending[item_id].get("end_ms", 0),
                    event.get("transcript", ""),
                )

    return transcripts


def _merge_overlapping_transcripts(
    transcripts: dict[str, tuple[int, int, str]],
    gap_threshold_ms: int = 500,
) -> list[tuple[int, int, str]]:
    """Merge overlapping or adjacent transcripts into combined units.

    Returns list of (start_ms, end_ms, combined_transcript) tuples.
    """
    if not transcripts:
        return []

    sorted_items = sorted(transcripts.values(), key=lambda x: x[0])
    merged = []
    current_start, current_end, current_text = sorted_items[0]

    for start_ms, end_ms, text in sorted_items[1:]:
        if start_ms <= current_end + gap_threshold_ms:
            current_end = max(current_end, end_ms)
            current_text = current_text + " " + text
        else:
            merged.append((current_start, current_end, current_text))
            current_start, current_end, current_text = start_ms, end_ms, text

    merged.append((current_start, current_end, current_text))
    return merged


def _group_ticks_into_segments(ticks: list) -> list[list[tuple]]:
    """Group consecutive gold ticks into utterance segments.

    Returns list of segments, where each segment is a list of (tick, start_ms, end_ms).
    """
    segments: list[list[tuple]] = []
    current_segment: list[tuple] = []
    prev_tick_idx = -2

    for i, tick in enumerate(ticks):
        if not (tick.user_chunk and tick.user_chunk.content):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            prev_tick_idx = -2
            continue

        if i != prev_tick_idx + 1 and current_segment:
            segments.append(current_segment)
            current_segment = []

        raw_data = (
            getattr(tick.agent_chunk, "raw_data", None) if tick.agent_chunk else None
        )
        if raw_data:
            tick_start = raw_data.get("cumulative_user_audio_at_tick_start_ms", 0)
            tick_duration = raw_data.get("audio_sent_duration_ms", 0)
            current_segment.append((tick, tick_start, tick_start + tick_duration))

        prev_tick_idx = i

    if current_segment:
        segments.append(current_segment)

    return segments


def _assign_transcripts_to_segments(
    merged_transcripts: list[tuple[int, int, str]],
    segments: list[list[tuple]],
) -> list[list[int]]:
    """Assign each merged transcript to a segment.

    Returns list of transcript indices for each segment.
    """
    segment_transcripts: list[list[int]] = [[] for _ in segments]
    assigned: set[int] = set()

    for seg_idx, segment in enumerate(segments):
        seg_start = segment[0][1]
        seg_end = segment[-1][2]

        for t_idx, (start_ms, end_ms, _) in enumerate(merged_transcripts):
            if t_idx in assigned:
                continue

            has_overlap = not (end_ms <= seg_start or start_ms >= seg_end)
            ended_before = end_ms <= seg_start

            if has_overlap or ended_before:
                segment_transcripts[seg_idx].append(t_idx)
                assigned.add(t_idx)

    # Unassigned transcripts go to last segment
    for t_idx in range(len(merged_transcripts)):
        if t_idx not in assigned:
            segment_transcripts[-1].append(t_idx)

    return segment_transcripts


def _distribute_transcript_in_tick(
    tick_start_ms: int,
    tick_end_ms: int,
    start_ms: int,
    end_ms: int,
    transcript: str,
    char_position: int,
    is_first_tick: bool,
    seg_start: int,
) -> tuple[Optional[str], int]:
    """Distribute a transcript's text for a single tick.

    Returns (text_for_tick, new_char_position).
    """
    total_duration = end_ms - start_ms

    if not transcript or char_position >= len(transcript):
        return None, char_position

    # Speech ended before segment - dump all on first tick
    if is_first_tick and end_ms <= seg_start:
        text = transcript[char_position:]
        return text if text else None, len(transcript)

    # No overlap with this tick
    if tick_end_ms <= start_ms or tick_start_ms >= end_ms:
        return None, char_position

    # Proportional distribution
    if total_duration <= 0:
        text = transcript[char_position:]
        return text if text else None, len(transcript)

    audio_played = min(tick_end_ms, end_ms) - start_ms
    text, new_pos = get_proportional_text(
        transcript=transcript,
        total_duration=total_duration,
        audio_played=audio_played,
        start_char=char_position,
    )
    return text if text else None, new_pos


def _process_segment(
    segment: list[tuple],
    seg_start: int,
    merged_transcripts: list[tuple[int, int, str]],
    seg_t_indices: list[int],
    char_positions: list[int],
) -> None:
    """Process a single segment, distributing transcripts across its ticks."""
    last_segment_tick = segment[-1][0]

    for tick_idx, (tick, tick_start_ms, tick_end_ms) in enumerate(segment):
        tick_text_parts = []
        is_first_tick = tick_idx == 0

        for t_idx in seg_t_indices:
            start_ms, end_ms, transcript = merged_transcripts[t_idx]

            text, new_pos = _distribute_transcript_in_tick(
                tick_start_ms=tick_start_ms,
                tick_end_ms=tick_end_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                transcript=transcript,
                char_position=char_positions[t_idx],
                is_first_tick=is_first_tick,
                seg_start=seg_start,
            )
            char_positions[t_idx] = new_pos

            if text:
                tick_text_parts.append(text)

        tick.user_transcript = " ".join(tick_text_parts) if tick_text_parts else None

    # Flush remaining text to last tick
    remaining_parts = []
    for t_idx in seg_t_indices:
        transcript = merged_transcripts[t_idx][2]
        if char_positions[t_idx] < len(transcript):
            remaining_parts.append(transcript[char_positions[t_idx] :])
            char_positions[t_idx] = len(transcript)

    if remaining_parts:
        existing = last_segment_tick.user_transcript or ""
        separator = " " if existing else ""
        last_segment_tick.user_transcript = (
            existing + separator + " ".join(remaining_parts)
        )


def compute_proportional_user_transcripts(ticks: list) -> None:
    """Populate tick.user_transcript by distributing transcripts proportionally.

    Guarantees:
    - Every transcript is assigned to exactly one segment
    - Every character appears in exactly one tick within that segment
    - Overlapping/adjacent transcripts are merged to avoid interleaving
    """
    # Step 1: Extract transcripts from events
    transcripts = _collect_transcripts_from_ticks(ticks)
    if not transcripts:
        return

    # Step 2: Merge overlapping/adjacent transcripts
    merged_transcripts = _merge_overlapping_transcripts(transcripts)

    # Step 3: Group ticks into segments
    segments = _group_ticks_into_segments(ticks)
    if not segments:
        return

    # Step 4: Assign transcripts to segments
    segment_transcripts = _assign_transcripts_to_segments(merged_transcripts, segments)

    # Step 5: Distribute transcripts within each segment
    char_positions = [0] * len(merged_transcripts)

    for seg_idx, segment in enumerate(segments):
        seg_start = segment[0][1]
        _process_segment(
            segment=segment,
            seg_start=seg_start,
            merged_transcripts=merged_transcripts,
            seg_t_indices=segment_transcripts[seg_idx],
            char_positions=char_positions,
        )
