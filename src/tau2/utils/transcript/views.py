"""Explicit transcript views over recorded ticks, with shared utterance grouping.

Audio/delivery groups every non-tool chunk with intersecting utterance IDs;
missing IDs stay separate. Raw call transcripts first select audible content and
also join adjacent unidentified chunks. Neither view rewrites stored ticks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Literal

from tau2.config import DEFAULT_TICK_DURATION_SECONDS
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    Tick,
    UserMessage,
)
from tau2.data_model.transcript import (
    AgentTurn,
    ParticipantTick,
    SpokenTurn,
    UtteranceGroup,
)
from tau2.utils.transcript.audio_text import (
    extract_delivered_text,
    merge_audio_script_gold,
)
from tau2.utils.transcript.linearization import linearize_ticks
from tau2.utils.transcript.live import live_metadata, live_utterance_groups

if TYPE_CHECKING:
    from tau2.data_model.simulation import SimulationRun

_BARGE_IN_GRACE_SECONDS = 2.0


def utterance_groups(
    ticks: list[Tick], speaker: Literal["agent", "caller"], *, raw_speech: bool = False
) -> list[UtteranceGroup]:
    """Group one speaker with delivery/audio or raw-call boundary semantics.

    The raw view filters content before grouping and joins adjacent no-ID speech;
    delivery preserves silent chunks and treats each absent-ID chunk separately.
    ID intersections accumulate across a group, even when ticks are not adjacent.
    """
    groups: list[UtteranceGroup] = []
    current_ids: set[str] = set()
    for index, tick in enumerate(ticks):
        if speaker == "agent" and live_metadata(tick) is not None:
            continue
        chunk = tick.agent_chunk if speaker == "agent" else tick.user_chunk
        if chunk is None:
            continue
        if raw_speech:
            if not chunk.content or not chunk.contains_speech:
                continue
        elif chunk.is_tool_call():
            continue
        ids = set(chunk.utterance_ids or [])
        current = groups[-1] if groups else None
        overlap = bool(current_ids and ids and current_ids & ids)
        contiguous_unidentified = bool(
            raw_speech
            and current is not None
            and not ids
            and not current_ids
            and index == current.end_index + 1
        )
        if current is not None and (overlap or contiguous_unidentified):
            current.chunks.append(chunk)
            current.end_index = index
            current.end_tick = tick.tick_id
            current_ids.update(ids)
        else:
            groups.append(
                UtteranceGroup(
                    speaker=speaker,
                    start_index=index,
                    end_index=index,
                    start_tick=tick.tick_id,
                    end_tick=tick.tick_id,
                    chunks=[chunk],
                )
            )
            current_ids = ids
    if speaker == "agent":
        groups.extend(live_utterance_groups(ticks))
        groups.sort(key=lambda group: group.start_index)
    return groups


def agent_messages(ticks: list[Tick]) -> list[AssistantMessage]:
    """Agent speech messages indexed exactly like audio/delivery utterance spans."""
    return [
        group.chunks[0]
        if len(group.chunks) == 1
        else AssistantMessage.merge_chunks(group.chunks)
        for group in utterance_groups(ticks, "agent")
    ]


def agent_utterance_tick_spans(sim: SimulationRun) -> list[tuple[int, int]]:
    """Original tick-ID spans aligned one-for-one with ``agent_messages``."""
    return [
        (group.start_tick, group.end_tick)
        for group in utterance_groups(sim.ticks or [], "agent")
    ]


def utterance_interrupted(messages: Iterable[ParticipantMessageBase]) -> bool:
    """Whether chunk records show truncated or undelivered speech.

    This combines ``raw_data.was_truncated`` with the delivered/undelivered
    split in markup golds. Neither signal identifies why speech was cut or
    whether the caller caused it. ``messages`` must include every chunk in one
    utterance group: gold detection uses their merged active-chunk union, not a
    single chunk's template.

    Realtime cancellations can leave every emitted chunk active and therefore
    escape this check. Per-turn consumers combine this signal with the caller
    overlap heuristic in ``barged_in_tick_spans``. The chunk signal also covers
    cuts without observed overlap and runs without ticks.
    """
    messages = list(messages)
    for message in messages:
        raw = getattr(message, "raw_data", None)
        if isinstance(raw, dict) and raw.get("was_truncated"):
            return True
    merged_gold = merge_audio_script_gold(
        [getattr(message, "audio_script_gold", None) for message in messages]
    )
    if not merged_gold or "<chunk id=" not in merged_gold:
        return False
    return extract_delivered_text(merged_gold)[1]


def _barge_in_span_flags(
    simulation: SimulationRun,
) -> list[tuple[tuple[int, int], bool]]:
    """Flag agent utterance spans with caller speech near their final speech tick.

    This heuristic checks for caller speech overlapping the final
    ``_BARGE_IN_GRACE_SECONDS`` of each agent utterance, including its last
    speaking tick. The overlap is observed; whether it caused a cancellation or
    cut is inferred, not established by the timeline.

    It complements chunk delivery accounting: provider cancellations may leave
    all emitted chunks active while omitting the unfinished tail. All caller
    speech counts regardless of turn-taking classification, including
    backchannels and collisions where the caller starts before the agent.
    Those cases can also trigger provider VAD cancellation.
    """
    if not simulation.ticks:
        return []
    spans = agent_utterance_tick_spans(simulation)
    if not spans:
        return []
    tick_seconds = next(
        (
            tick.tick_duration_seconds
            for tick in simulation.ticks
            if tick.tick_duration_seconds
        ),
        DEFAULT_TICK_DURATION_SECONDS,
    )
    grace_ticks = max(1, math.ceil(_BARGE_IN_GRACE_SECONDS / tick_seconds))
    agent_speech_ticks = {
        tick.tick_id
        for tick in simulation.ticks
        if tick.agent_chunk is not None
        and not tick.agent_chunk.is_tool_call()
        and tick.agent_chunk.contains_speech
    }
    user_speech_ticks = {
        tick.tick_id
        for tick in simulation.ticks
        if tick.user_chunk is not None and tick.user_chunk.contains_speech
    }
    flags: list[tuple[tuple[int, int], bool]] = []
    for start, end in spans:
        last_speech = max(
            (t for t in agent_speech_ticks if start <= t <= end), default=None
        )
        if last_speech is None:
            flags.append(((start, end), False))
            continue
        tail = range(max(start, last_speech - grace_ticks), last_speech + 1)
        flags.append(((start, end), any(t in user_speech_ticks for t in tail)))
    return flags


def barged_in_tick_spans(simulation: SimulationRun) -> list[tuple[int, int]]:
    """Return inclusive tick-ID spans flagged by the caller-overlap heuristic.

    See ``_barge_in_span_flags`` for the evidence and its limits.
    """
    return [span for span, barged in _barge_in_span_flags(simulation) if barged]


def tick_span_barged_in(
    start_tick: int, end_tick: int, barged_spans: list[tuple[int, int]]
) -> bool:
    """Whether an agent turn overlaps a span flagged by caller speech overlap.

    Mapping is by tick-range overlap, never by ordinal: consumers group agent
    chunks differently, so the n-th turn is not necessarily the n-th span.
    This carries the overlap heuristic to another grouping without adding
    evidence about what caused the utterance to end.
    """
    return any(start <= end_tick and start_tick <= end for start, end in barged_spans)


def extract_spoken_turns(sim: SimulationRun) -> list[SpokenTurn]:
    """Raw interleaved call transcript, ordered by onset (caller first on ties)."""
    if not sim.ticks:
        return [
            SpokenTurn(
                speaker="agent" if message.role == "assistant" else "caller",
                text=message.content.strip(),
            )
            for message in sim.messages or []
            if message.role in {"assistant", "user"} and (message.content or "").strip()
        ]
    barged = barged_in_tick_spans(sim)
    groups = utterance_groups(sim.ticks, "caller", raw_speech=True) + utterance_groups(
        sim.ticks, "agent", raw_speech=True
    )
    return [
        SpokenTurn(
            speaker=g.speaker,
            text=text,
            interrupted=utterance_interrupted(g.chunks)
            or (
                g.speaker == "agent"
                and not (g.chunks[0].raw_data or {}).get("live_segment")
                and tick_span_barged_in(g.start_tick, g.end_tick, barged)
            ),
        )
        for g in sorted(groups, key=lambda g: g.start_index)
        if (text := "".join(chunk.content for chunk in g.chunks).strip())
    ]


def delivered_text(message: ParticipantMessageBase) -> str | None:
    """Actually delivered gold text, with raw content fallback and speech filtering."""
    if message.contains_speech is False:
        return None
    text = message.audio_script_gold or message.content
    if not text:
        return None
    if message.audio_script_gold:
        text, _ = extract_delivered_text(text)
    return text or None


def build_agent_turns(simulation: SimulationRun) -> list[AgentTurn]:
    """Delivered agent utterances with the last caller utterance completed before onset."""
    turns: list[AgentTurn] = []
    if not simulation.ticks:
        preceding_user = None
        for message in simulation.messages or []:
            if message.role == "user" and message.content:
                preceding_user = message.content
            elif message.role == "assistant" and message.content:
                turns.append(
                    AgentTurn(
                        index=len(turns),
                        text=message.content,
                        preceding_user_text=preceding_user,
                    )
                )
        return turns
    barged = barged_in_tick_spans(simulation)
    user_events = []
    for group in utterance_groups(simulation.ticks, "caller"):
        merged = (
            group.chunks[0]
            if len(group.chunks) == 1
            else UserMessage.merge_chunks(group.chunks)
        )
        if text := delivered_text(merged):
            user_events.append((group.start_tick, group.end_tick, text))
    for group in utterance_groups(simulation.ticks, "agent"):
        merged = (
            group.chunks[0]
            if len(group.chunks) == 1
            else AssistantMessage.merge_chunks(group.chunks)
        )
        if not (text := delivered_text(merged)):
            continue
        prior = [text for _, end, text in user_events if end < group.start_tick]
        live = isinstance(merged.raw_data, dict) and bool(
            merged.raw_data.get("live_segment")
        )
        overlapping = [
            text
            for start, end, text in user_events
            if start <= group.end_tick and end >= group.start_tick
        ]
        turns.append(
            AgentTurn(
                index=len(turns),
                start_tick=group.start_tick,
                end_tick=group.end_tick,
                segment_id=(
                    merged.raw_data.get("live_segment", {}).get("segment_id")
                    if isinstance(merged.raw_data, dict)
                    else None
                ),
                text=text,
                preceding_user_text=prior[-1] if prior else None,
                overlapping_user_text="\n".join(overlapping)
                if live and overlapping
                else None,
                interrupted=utterance_interrupted(group.chunks)
                or (
                    not live
                    and tick_span_barged_in(group.start_tick, group.end_tick, barged)
                ),
            )
        )
    return turns


def speech_messages(ticks: list[Tick]) -> list[Message]:
    """Containment-aware evaluation view, excluding tool calls from both speakers."""
    return linearize_ticks(
        [
            ParticipantTick(
                tick_id=t.tick_id,
                timestamp=t.timestamp,
                self_chunk=t.agent_chunk
                if t.agent_chunk is not None and not t.agent_chunk.is_tool_call()
                else None,
                other_chunk=t.user_chunk
                if t.user_chunk is not None and not t.user_chunk.is_tool_call()
                else None,
            )
            for t in ticks
        ]
    )


def _participant_tick_message(
    tick: Tick, speaker: Literal["agent", "caller"], *, include_audio: bool
) -> AssistantMessage | UserMessage:
    """Materialize recorded fields for a full-row or tool-replay message view."""
    agent = speaker == "agent"
    chunk = tick.agent_chunk if agent else tick.user_chunk
    calls = tick.agent_tool_calls if agent else tick.user_tool_calls
    message_class = AssistantMessage if agent else UserMessage
    fields = dict(
        role="assistant" if agent else "user",
        content=chunk.content if chunk else None,
        tool_calls=calls or None,
        timestamp=chunk.timestamp if chunk else tick.timestamp,
        contains_speech=chunk.contains_speech if chunk else False,
    )
    if include_audio:
        fields.update(
            is_audio=chunk.is_audio if chunk else False,
            audio_content=chunk.audio_content if chunk else None,
            audio_format=chunk.audio_format if chunk else None,
            speech_effects=chunk.speech_effects if chunk else None,
            source_effects=chunk.source_effects if chunk else None,
            channel_effects=chunk.channel_effects if chunk else None,
        )
    return message_class(**fields)


def tick_messages(tick: Tick) -> list[Message]:
    """Full stored-row view: agent and its results, then caller and its results.

    This preserves the ordinary saved-result view, including audio/effects and
    standalone tool results. It is deliberately not environment replay order.
    """
    messages: list[Message] = []
    if tick.agent_chunk or tick.agent_tool_calls:
        messages.append(_participant_tick_message(tick, "agent", include_audio=True))
    messages.extend(tick.agent_tool_results)
    if tick.user_chunk or tick.user_tool_calls:
        messages.append(_participant_tick_message(tick, "caller", include_audio=True))
    messages.extend(tick.user_tool_results)
    return messages


def tool_messages(ticks: list[Tick]) -> list[Message]:
    """Environment replay view: caller calls/results before agent calls/results.

    Speech-only chunks, audio and unmatched results are excluded, exactly as in
    the environment evaluator's execution trajectory.
    """
    messages: list[Message] = []
    for tick in ticks:
        if tick.user_tool_calls:
            messages.append(
                _participant_tick_message(tick, "caller", include_audio=False)
            )
            messages.extend(tick.user_tool_results)
        if tick.agent_tool_calls:
            messages.append(
                _participant_tick_message(tick, "agent", include_audio=False)
            )
            messages.extend(tick.agent_tool_results)
    return messages
