"""Completed-call GPT Live groups on the recorded playback clock.

Split at 1.5 seconds of silence, pad each side by 750 ms, and associate saved
caption ticks by overlap. A tick overlapping two clips belongs to the earlier
clip. Provider transcript timestamps are preserved as evidence, not used to align.
"""

import hashlib
import json
from dataclasses import dataclass

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.transcript import (
    LiveTranscriptFragment,
    LiveTranscriptTick,
    UtteranceGroup,
)

LIVE_SEGMENT_VERSION = "live-tick-padding-v3"
LIVE_SILENCE_MS = 1500
LIVE_PADDING_MS = LIVE_SILENCE_MS // 2


def live_metadata(tick: Tick) -> LiveTranscriptTick | None:
    """Read explicit Live evidence, never infer a provider from missing IDs."""
    raw = tick.agent_chunk.raw_data if tick.agent_chunk else None
    value = raw.get("live_transcript") if isinstance(raw, dict) else None
    return LiveTranscriptTick.model_validate(value) if value is not None else None


@dataclass
class _Fragment:
    value: LiveTranscriptFragment
    index: int
    release_ms: float


@dataclass
class _AudioSpan:
    start_ms: float
    end_ms: float
    start_index: int
    end_index: int
    boundary: str = "audio_silence"


def _nearest(position: float, spans: list[_AudioSpan]) -> tuple[int, float]:
    distances = [max(s.start_ms - position, 0, position - s.end_ms) for s in spans]
    distance = min(distances)
    return distances.index(distance), distance


def _session_groups(
    ticks: list[Tick], session: str, records: list[tuple[int, LiveTranscriptTick]]
) -> list[UtteranceGroup]:
    fragments = []
    activity = []
    durations = {ticks[i].tick_duration_seconds for i, _ in records}
    if len(durations) != 1 or None in durations or next(iter(durations)) <= 0:
        raise ValueError("Live grouping requires one recorded positive tick duration")
    duration = next(iter(durations))
    for index, metadata in records:
        tick = ticks[index]
        start = round(tick.tick_id * duration * 1000, 6)
        for a, b in metadata.speech_intervals_ms:
            if not 0 <= a < b <= duration * 1000 + 1e-6:
                raise ValueError("Live speech interval lies outside its recorded tick")
            activity.append(
                _AudioSpan(round(start + a, 6), round(start + b, 6), index, index)
            )
        for fragment in metadata.fragments:
            if fragment.speaker == "agent" and fragment.text:
                fragments.append(_Fragment(fragment, index, start))
    spans: list[_AudioSpan] = []
    for span in sorted(activity, key=lambda s: (s.start_ms, s.end_ms)):
        if not spans or span.start_ms - spans[-1].end_ms >= LIVE_SILENCE_MS:
            span.boundary = "session_start" if not spans else "audio_silence"
            spans.append(span)
        else:
            spans[-1].end_ms = max(spans[-1].end_ms, span.end_ms)
            spans[-1].end_index = max(spans[-1].end_index, span.end_index)
    if not spans and not fragments:
        return []
    # Preserve orphan text as unresolved evidence, never as a scored silence clip.
    orphan = not spans
    if orphan:
        spans = [
            _AudioSpan(
                fragments[0].release_ms,
                fragments[-1].release_ms + duration * 1000,
                fragments[0].index,
                fragments[-1].index,
                "unmatched_text",
            )
        ]
    session_start = ticks[records[0][0]].tick_id * duration * 1000
    session_end = (ticks[records[-1][0]].tick_id + 1) * duration * 1000
    bounds = [
        (
            max(session_start, s.start_ms - LIVE_PADDING_MS),
            min(session_end, s.end_ms + LIVE_PADDING_MS),
        )
        for s in spans
    ]
    members: list[list[_Fragment]] = [[] for _ in spans]
    problems: list[set[str]] = [set() for _ in spans]
    ties: list[set[int]] = [set() for _ in spans]
    outside: list[set[int]] = [set() for _ in spans]
    previous = 0
    for fragment in fragments:
        tick_end = fragment.release_ms + duration * 1000
        matches = [
            ordinal
            for ordinal, (start, end) in enumerate(bounds)
            if fragment.release_ms < end and tick_end > start
        ]
        if matches:
            # A text-bearing tick crossing both clips belongs to the earlier clip.
            selected = matches[0]
            if len(matches) > 1:
                ties[selected].add(ticks[fragment.index].tick_id)
        else:
            # Retain distant text with the nearest group; record the timing mismatch.
            selected, _ = _nearest(fragment.release_ms, spans)
            outside[selected].add(ticks[fragment.index].tick_id)
        if selected < previous:
            raise ValueError("Live caption association would reorder recorded text")
        previous = selected
        members[selected].append(fragment)
    groups = []
    for ordinal, (span, part) in enumerate(zip(spans, members, strict=True)):
        if orphan:
            problems[ordinal].add("Live transcript has no recorded speech")
        if not part:
            problems[ordinal].add("Recorded Live speech has no transcript")
        identity = [
            session,
            LIVE_SEGMENT_VERSION,
            span.start_ms,
            span.end_ms,
            bounds[ordinal],
            [(f.value.text, ticks[f.index].tick_id) for f in part],
        ]
        identifier = (
            "live-segment:"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:24]
        )
        error = "; ".join(sorted(problems[ordinal])) or None
        metadata = {
            "version": LIVE_SEGMENT_VERSION,
            "evidence_origins": sorted({m.origin for _, m in records}),
            "silence_threshold_ms": LIVE_SILENCE_MS,
            "padding_ms": LIVE_PADDING_MS,
            "association_method": "recorded_tick_overlap_earlier",
            "earlier_clip_tie_tick_ids": sorted(ties[ordinal]),
            "outside_window_tick_ids": sorted(outside[ordinal]),
            "segment_id": identifier,
            "boundary": span.boundary,
            "association_status": "unmatched_text"
            if orphan
            else "unmatched_audio"
            if not part
            else "ambiguous"
            if error
            else "stable",
            "audio_alignment_error": error,
            "speech_start_us": round(span.start_ms * 1000),
            "speech_end_us": round(span.end_ms * 1000),
            "playback_start_us": round(bounds[ordinal][0] * 1000),
            "playback_end_us": round(bounds[ordinal][1] * 1000),
            "fragments": [f.value.model_dump() for f in part],
            "caption_tick_ids": [ticks[f.index].tick_id for f in part],
        }
        groups.append(
            UtteranceGroup(
                speaker="agent",
                start_index=span.start_index,
                end_index=span.end_index,
                start_tick=ticks[span.start_index].tick_id,
                end_tick=ticks[span.end_index].tick_id,
                chunks=[
                    AssistantMessage(
                        role="assistant",
                        content="".join(f.value.text for f in part),
                        contains_speech=True,
                        utterance_ids=[identifier],
                        raw_data={"live_segment": metadata},
                    )
                ],
            )
        )
    # Explicit cuts concern playback. Caller overlap alone is not proof of a cut.
    for index, _ in records:
        if not ticks[index].agent_chunk.raw_data.get("was_truncated"):
            continue
        started = [g for g in groups if g.start_index <= index]
        if not started:
            continue
        group = started[-1]
        raw = group.chunks[0].raw_data
        if len(started) > 1 and started[-2].start_index == group.start_index:
            for g in started[-2:]:
                g.chunks[0].raw_data["live_segment"].update(
                    association_status="ambiguous",
                    audio_alignment_error="Cannot attribute Live truncation within one tick",
                )
            continue
        raw["was_truncated"] = True
        raw["live_segment"].setdefault("truncation_tick_ids", []).append(
            ticks[index].tick_id
        )
    return groups


def live_utterance_groups(ticks: list[Tick]) -> list[UtteranceGroup]:
    """Group recorded audio first; retain unresolved text/audio with explicit reasons."""
    sessions: dict[str, list[tuple[int, LiveTranscriptTick]]] = {}
    for index, tick in enumerate(ticks):
        if metadata := live_metadata(tick):
            sessions.setdefault(metadata.session_id, []).append((index, metadata))
    return [
        group
        for session, records in sessions.items()
        for group in _session_groups(ticks, session, records)
    ]


def require_live_alignment(ticks: list[Tick]) -> None:
    """Prevent uncertain units, including uncaptioned audio, from silently scoring."""
    unresolved = [
        g.chunks[0].raw_data["live_segment"] for g in live_utterance_groups(ticks)
    ]
    unresolved = [m for m in unresolved if m["audio_alignment_error"]]
    if unresolved:
        raise ValueError(
            "Unresolved Live audio/text association: "
            + json.dumps(
                [
                    {
                        "segment_id": m["segment_id"],
                        "reason": m["audio_alignment_error"],
                    }
                    for m in unresolved
                ]
            )
        )
