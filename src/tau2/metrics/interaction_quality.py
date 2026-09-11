# Copyright Sierra
"""Deterministic raw measurements used by the universal quality rubric.

Also the single home of caller barge-in (interruption) detection: the
tick-overlap detector (``barged_in_tick_spans`` / ``barge_in_utterance_indices``)
is the source of truth wherever a run has ticks, and every judge family
(delivery, nativeness, quality, semantic) consumes it from here.
"""

import math
import re
from enum import Enum
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

from tau2.agent.base.streaming_utils import (
    extract_delivered_text,
    merge_audio_script_gold,
)
from tau2.config import DEFAULT_TICK_DURATION_SECONDS
from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import SimulationRun, resolve_tick_duration_seconds
from tau2.metrics.voice_interaction_metrics import (
    InteractionMetricsConfig,
    VoiceInteractionMetrics,
    compute_interaction_metrics_for_ticks,
)

QUALITY_METRICS_VERSION = "quality-metrics-v5"

AUTH_TOOLS_BY_DOMAIN: dict[str, frozenset[str]] = {
    "telecom": frozenset(
        {"get_customer_by_phone", "get_customer_by_name", "get_customer_by_id"}
    ),
    "airline": frozenset({"get_user_details"}),
    "retail": frozenset({"find_user_id_by_name_zip", "find_user_id_by_email"}),
    "banking_knowledge": frozenset(
        {
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
        }
    ),
}

_EMPTY_RESULT_CONTENTS = frozenset({"", "[]", "null", "None"})
_NO_RECORDS_RESULT = re.compile(r"^no records found", re.IGNORECASE)
_NOT_FOUND_ERROR = re.compile(r"not\s+found", re.IGNORECASE)


class AuthLookupOutcome(str, Enum):
    """Meaning of an identity lookup result."""

    IDENTIFIED = "identified"
    NO_MATCH = "no_match"
    TOOL_FAILURE = "tool_failure"


def classify_auth_lookup(content: Optional[str], error: bool) -> AuthLookupOutcome:
    """Classify the three transports used for a no-match identity lookup."""
    text = (content or "").strip()
    if not error:
        if text in _EMPTY_RESULT_CONTENTS or _NO_RECORDS_RESULT.match(text):
            return AuthLookupOutcome.NO_MATCH
        return AuthLookupOutcome.IDENTIFIED
    if _NOT_FOUND_ERROR.search(text):
        return AuthLookupOutcome.NO_MATCH
    return AuthLookupOutcome.TOOL_FAILURE


class InteractionQualityMetrics(BaseModel):
    """Raw, deterministic evidence for one call."""

    voice: Optional[VoiceInteractionMetrics] = Field(
        default=None,
        description="Canonical τ-voice panel; None for calls without ticks.",
    )
    max_agent_floor_hold_seconds: Optional[float] = Field(
        default=None, description="Longest uninterrupted agent floor hold."
    )
    agent_turn_count: float = Field(
        description="Number of agent spoken turns.", default=0.0
    )
    exact_repeated_agent_turn_count: float = Field(
        description="Adjacent agent turns repeated after normalization.", default=0.0
    )
    agent_tool_call_count: float = Field(
        description="Agent-requested tool calls.", default=0.0
    )
    agent_tool_error_count: float = Field(
        description="Error results returned to agent tool calls.", default=0.0
    )
    auth_tool_call_count: Optional[float] = Field(
        default=None, description="Identity lookup calls in a registered domain."
    )
    auth_arg_mismatch_count: Optional[float] = Field(
        default=None,
        description="Identity calls returning no match; an LLM attributes cause.",
    )


def _segments(flags: list[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, active in enumerate(flags + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            segments.append((start, i - 1))
            start = None
    return segments


def _normalized_turn(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


class SpokenTurn(BaseModel):
    """One delivered utterance in call order.

    ``text`` is the raw delivered transcript: deterministic checkers, lexical
    stats and corpus builders, and preference inputs consume it as-is. LLM
    judge prompts render an interruption marker separately
    (``tau2.judges.base.interruption_marked_text``) — the marker never lives
    on this model.
    """

    speaker: Literal["agent", "caller"] = Field(
        description="Which participant delivered the turn."
    )
    text: str = Field(
        description="Raw delivered text; never carries judge-facing markers."
    )
    interrupted: bool = Field(
        default=False,
        description="Whether the other participant cut this utterance off "
        "mid-delivery (barge-in). For agent turns in tick runs this is the "
        "union of the chunk-level signal (``utterance_interrupted``) and the "
        "tick-overlap detector (``barged_in_tick_spans``).",
    )


def utterance_interrupted(messages: Iterable) -> bool:
    """Whether a merged utterance's CHUNKS record a mid-delivery cut (barge-in).

    Detection is dual (PR #763): streaming runs carry markup golds whose
    delivered/undelivered chunk split records the cut (undelivered template
    chunks == the caller barged in); runs that predate markup golds carry the
    runner's per-chunk ``raw_data.was_truncated`` instead (the same signal the
    voice metrics use). ``messages`` are all chunk messages of ONE utterance
    group — gold detection is only correct over the merged active-chunk union,
    never over a single chunk's template.

    This chunk signal alone MISSES realtime cancellations (see
    ``barged_in_tick_spans``); consumers of per-turn interrupted flags OR it
    with the tick-overlap detector. It is kept in the union rather than
    replaced because it also fires on cuts the tick timeline understates
    (extra tolerance audited harmless on the judge-calibration corpora), and
    it is the only signal on runs without ticks.
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


def agent_utterance_tick_spans(sim: SimulationRun) -> list[tuple[int, int]]:
    """(first, last) tick id per agent utterance, index-aligned with the
    delivery judge's ``utterance_idx``.

    The delivery judge indexes utterances into
    ``FullDuplexCommunicateEvaluator.ticks_to_message_history(sim.ticks)``;
    this mirrors that grouping (agent chunks that are not tool calls, merged
    while consecutive chunks share an ``utterance_id``) but keeps the tick
    ids the merge discards — so stored delivery verdicts and barge-in spans
    can be anchored back onto the transcript/audio timeline. A paired test
    asserts the alignment against the evaluator's own grouping.
    """
    chunks: list[tuple[int, frozenset[str]]] = []
    for tick in sim.ticks or []:
        chunk = tick.agent_chunk
        if chunk is not None and not chunk.is_tool_call():
            chunks.append((tick.tick_id, frozenset(chunk.utterance_ids or [])))
    if not chunks:
        return []

    spans: list[tuple[int, int]] = []
    start, end = chunks[0][0], chunks[0][0]
    current_ids = set(chunks[0][1])
    for tick_id, ids in chunks[1:]:
        if current_ids and ids and not current_ids.isdisjoint(ids):
            end = tick_id
            current_ids.update(ids)
        else:
            spans.append((start, end))
            start = end = tick_id
            current_ids = set(ids)
    spans.append((start, end))
    return spans


# Caller speech only counts as cutting an utterance short when it overlaps
# the final stretch of the agent's speech; earlier overlaps the agent talks
# through (the provider did not cancel) keep the full check.
_BARGE_IN_GRACE_SECONDS = 2.0


def _barge_in_span_flags(
    simulation: SimulationRun,
) -> list[tuple[tuple[int, int], bool]]:
    """Each agent utterance's tick span plus whether the tick timeline shows
    a caller barge-in cutting it short.

    This tick-overlap detector (PR #855) is the source of truth for
    interruptions wherever ticks exist. The chunk accounting in
    ``extract_delivered_text`` only sees an interruption when trailing chunks
    are marked inactive. On a realtime barge-in the provider CANCELS
    generation instead: every chunk it did emit is marked active even though
    the tail never aired, so the reference keeps unaired text (often ending
    mid-word) and a judge told "not interrupted" reads it as missing words.
    The tick timeline still records the cut: caller speech overlapping the
    last ``_BARGE_IN_GRACE_SECONDS`` of the utterance's agent speech. ALL
    overlapping caller speech counts — turn-taking classification is
    deliberately ignored, because provider VAD cancels generation on
    caller-collision starts (the caller began a tick before the agent, so no
    segment *starts* during agent speech) and on backchannels alike; both
    were observed cutting agent audio mid-word in stored runs.
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


def barge_in_utterance_indices(simulation: SimulationRun) -> set[int]:
    """Utterance indices (``ticks_to_message_history`` order) cut short by
    overlapping caller speech, detected from the tick timeline.

    See ``_barge_in_span_flags`` for the detection rule. The delivery judge
    consumes this index form because its utterance ordering IS the span
    ordering; everything that groups agent turns differently maps by tick
    range instead (``barged_in_tick_spans`` + ``tick_span_barged_in``).
    """
    return {
        idx
        for idx, (_span, barged) in enumerate(_barge_in_span_flags(simulation))
        if barged
    }


def barged_in_tick_spans(simulation: SimulationRun) -> list[tuple[int, int]]:
    """(first, last) tick-id spans of agent utterances the tick timeline
    shows cut short by a caller barge-in (see ``_barge_in_span_flags``)."""
    return [span for span, barged in _barge_in_span_flags(simulation) if barged]


def tick_span_barged_in(
    start_tick: int, end_tick: int, barged_spans: list[tuple[int, int]]
) -> bool:
    """Whether an agent turn spanning ``[start_tick, end_tick]`` was cut short.

    Mapping is by tick-range OVERLAP, never by ordinal: consumers group agent
    chunks differently (``extract_spoken_turns`` and the nativeness harness
    merge on different rules than ``agent_utterance_tick_spans``), so the
    n-th turn is not the n-th span. An agent turn is interrupted iff any
    barged span overlaps its tick range.
    """
    return any(start <= end_tick and start_tick <= end for start, end in barged_spans)


def extract_spoken_turns(sim: SimulationRun) -> list[SpokenTurn]:
    """Interleaved caller/agent turns, preserving caller boundaries.

    An agent turn's ``interrupted`` flag is the union of the chunk-level
    signal (``utterance_interrupted``) and the tick-overlap detector, mapped
    onto this function's turn grouping by tick-range overlap.
    """
    if not sim.ticks:
        turns = []
        for message in sim.messages or []:
            role = getattr(message, "role", None)
            text = (getattr(message, "content", None) or "").strip()
            if role in {"assistant", "user"} and text:
                turns.append(
                    SpokenTurn(
                        speaker="agent" if role == "assistant" else "caller",
                        text=text,
                    )
                )
        return turns

    barged_spans = barged_in_tick_spans(sim)
    # [order, speaker, chunks, first_tick_id, last_tick_id]
    groups: list[list] = []
    current: dict[str, tuple[int, set[str], int]] = {}
    for index, tick in enumerate(sim.ticks):
        for chunk, speaker in (
            (tick.user_chunk, "caller"),
            (tick.agent_chunk, "agent"),
        ):
            if chunk is None or not chunk.content or not chunk.contains_speech:
                continue
            ids = set(chunk.utterance_ids or [])
            previous = current.get(speaker)
            if previous is not None:
                group_index, seen_ids, previous_index = previous
                same_utterance = bool(ids and seen_ids and ids & seen_ids)
                contiguous_unidentified_speech = (
                    not ids and not seen_ids and index == previous_index + 1
                )
                if same_utterance or contiguous_unidentified_speech:
                    groups[group_index][2].append(chunk)
                    groups[group_index][4] = tick.tick_id
                    current[speaker] = (group_index, seen_ids | ids, index)
                    continue
            groups.append([index, speaker, [chunk], tick.tick_id, tick.tick_id])
            current[speaker] = (len(groups) - 1, ids, index)
    return [
        SpokenTurn(
            speaker=speaker,
            text=text,
            interrupted=utterance_interrupted(chunks)
            or (
                speaker == "agent"
                and tick_span_barged_in(first_tick, last_tick, barged_spans)
            ),
        )
        for _, speaker, chunks, first_tick, last_tick in sorted(
            groups, key=lambda group: group[0]
        )
        if (text := "".join(chunk.content for chunk in chunks).strip())
    ]


def _agent_floor_holds(
    agent: list[bool], user: list[bool], duration: float
) -> list[float]:
    """Agent spans bounded by caller speech, including internal agent pauses."""
    holds: list[float] = []
    for block_start, block_end in _segments([not speaking for speaking in user]):
        agent_indexes = [
            index for index in range(block_start, block_end + 1) if agent[index]
        ]
        if agent_indexes:
            holds.append((agent_indexes[-1] - agent_indexes[0] + 1) * duration)
    return holds


def extract_interaction_quality_metrics(
    sim: SimulationRun, *, domain: Optional[str] = None
) -> InteractionQualityMetrics:
    """Extract quality evidence without consulting preference labels or an LLM."""
    voice_metrics: Optional[VoiceInteractionMetrics] = None
    max_agent_floor: Optional[float] = None
    if sim.ticks:
        duration = resolve_tick_duration_seconds(sim.ticks, None)
        voice_metrics = compute_interaction_metrics_for_ticks(
            sim.ticks,
            InteractionMetricsConfig(tick_duration_sec=duration),
        )
        agent = [
            bool(t.agent_chunk and t.agent_chunk.contains_speech) for t in sim.ticks
        ]
        user = [bool(t.user_chunk and t.user_chunk.contains_speech) for t in sim.ticks]
        speech = [a or u for a, u in zip(agent, user)]
        if any(speech):
            holds = _agent_floor_holds(agent, user, duration)
            max_agent_floor = max(holds) if holds else None

    spoken_turns = extract_spoken_turns(sim)
    agent_turns = [turn.text for turn in spoken_turns if turn.speaker == "agent"]
    normalized = [(turn.speaker, _normalized_turn(turn.text)) for turn in spoken_turns]
    repeated_turns = sum(
        1
        for previous, current in zip(normalized, normalized[1:])
        if current[0] == previous[0] == "agent"
        and current[1]
        and current[1] == previous[1]
    )

    calls_by_id: dict[str, ToolCall] = {}
    errors = 0
    auth_mismatches = 0
    agent_tool_calls = 0
    auth_tool_calls = 0
    auth_tools = AUTH_TOOLS_BY_DOMAIN.get(domain or "")
    for message in sim.get_messages():
        role = getattr(message, "role", None)
        for call in getattr(message, "tool_calls", None) or []:
            if call.requestor != "assistant":
                continue
            agent_tool_calls += 1
            if call.id:
                calls_by_id[call.id] = call
            if auth_tools is not None and call.name in auth_tools:
                auth_tool_calls += 1
        if role != "tool" or getattr(message, "requestor", None) != "assistant":
            continue
        call = calls_by_id.get(message.id)
        if message.error:
            errors += 1
        if call is not None and auth_tools is not None and call.name in auth_tools:
            if (
                classify_auth_lookup(message.content, message.error)
                is AuthLookupOutcome.NO_MATCH
            ):
                auth_mismatches += 1

    return InteractionQualityMetrics(
        voice=voice_metrics,
        max_agent_floor_hold_seconds=max_agent_floor,
        agent_turn_count=float(len(agent_turns)),
        exact_repeated_agent_turn_count=float(repeated_turns),
        agent_tool_call_count=float(agent_tool_calls),
        agent_tool_error_count=float(errors),
        auth_tool_call_count=(
            float(auth_tool_calls) if auth_tools is not None else None
        ),
        auth_arg_mismatch_count=(
            float(auth_mismatches) if auth_tools is not None else None
        ),
    )
