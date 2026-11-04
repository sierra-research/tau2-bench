"""Benchmark control speech and terminal cutoffs excluded before judging.

The versioned naturalness and fidelity rules remove these utterances:

* The injected opener. The orchestrator seeds every non-solo call with a fixed
  first agent message: the task's ``agent_opener``, else the recorded language
  context's reviewed ``agent_greeting``, else the English default opener. It is
  identified structurally: it is the call's first agent message (tick 0 in
  full-duplex runs, the first message otherwise) and its text is exactly the
  recorded opener. A model that later repeats the opener is still judged.
* The policy-mandated transfer line. Domain policies require the literal
  English sentence below, including in localized policies. Complete
  occurrences are removed anywhere in a turn; when the turn was interrupted, a
  delivered prefix at its end is removed as well. ``###STOP###`` and
  ``###TRANSFER###`` control markers are removed with it. Under the newer
  naturalness rule, turns containing either marker are dropped in full before
  the line is stripped. Speech fidelity uses the same whole-turn marker rule.
  Turns left empty are dropped.
* A final agent utterance with recorded speech in the same tick that the agent
  or caller ended the call. The precise cutoff within that tick is unknown, so
  the overlapping utterance is excluded as a whole.

Which rule a contract applies is recorded on the contract
(``JudgeContract.input_exclusions``) and in every verdict identity, so a rule
change invalidates cached verdicts.
"""

from __future__ import annotations

import re

from tau2.data_model.judges import InputExclusions
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.transcript import AgentTurn
from tau2.utils.transcript.views import build_agent_turns, delivered_text

BENCHMARK_TRANSFER_MESSAGE = (
    "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
)

_FULL_TRANSFER_RE = re.compile(
    r"YOU\s+ARE\s+BEING\s+TRANSFERRED\s+TO\s+A\s+HUMAN\s+AGENT\."
    r"\s+PLEASE\s+HOLD\s+ON\.",
    flags=re.IGNORECASE,
)
_TRANSFER_START_RE = re.compile(r"YOU\s+ARE\s+BEING", flags=re.IGNORECASE)
_STOP_MARKER_RE = re.compile(r"###(?:STOP|TRANSFER)###", flags=re.IGNORECASE)
_CANONICAL_WORDS = " ".join(re.findall(r"[A-Z]+", BENCHMARK_TRANSFER_MESSAGE))
_MINIMUM_FRAGMENT = "YOU ARE BEING T"


def has_control_marker(text: str) -> bool:
    """Whether delivered agent speech contains a stop or transfer marker."""
    return _STOP_MARKER_RE.search(text) is not None


def terminal_speech_tick(simulation: SimulationRun) -> int | None:
    """Final tick only when a participant ends the call over active agent audio."""
    if (
        simulation.termination_reason
        not in (
            TerminationReason.AGENT_STOP,
            TerminationReason.USER_STOP,
        )
        or not simulation.ticks
    ):
        return None
    tick = simulation.ticks[-1]
    agent = tick.agent_chunk
    if agent is None:
        return None
    raw = agent.raw_data or {}
    live = raw.get("live_transcript") if isinstance(raw, dict) else None
    speaking = (
        bool(live.get("speech_intervals_ms"))
        if isinstance(live, dict)
        else bool(agent.contains_speech)
    )
    if speaking:
        return tick.tick_id
    return None


def _words(text: str) -> str:
    """Normalize an English transfer-message fragment for exact comparison."""
    without_markers = _STOP_MARKER_RE.sub(" ", text.upper())
    return " ".join(re.findall(r"[A-Z]+", without_markers))


def is_benchmark_transfer_fragment(text: str) -> bool:
    """Whether text contains the fixed transfer line or a delivered prefix of it."""
    normalized = _words(text)
    start = normalized.find("YOU ARE BEING")
    if start < 0:
        return False
    candidate = normalized[start:]
    return _CANONICAL_WORDS in candidate or (
        len(candidate) >= len(_MINIMUM_FRAGMENT)
        and _CANONICAL_WORDS.startswith(candidate)
    )


# A cut-off tail may hold only the English line's own characters; any other
# script after the prefix is speech the agent produced and must stay judged.
_TRANSFER_TAIL_CHARS = re.compile(r"[A-Za-z\s.,!?#]*")


def without_benchmark_transfer_message(text: str, *, interrupted: bool) -> str:
    """Remove the policy-mandated transfer wording from one agent turn.

    Turns without the line are returned unchanged.
    """
    filtered = _FULL_TRANSFER_RE.sub("", text)
    if interrupted:
        starts = list(_TRANSFER_START_RE.finditer(filtered))
        if starts:
            tail = filtered[starts[-1].start() :]
            if _TRANSFER_TAIL_CHARS.fullmatch(tail) and is_benchmark_transfer_fragment(
                tail
            ):
                filtered = filtered[: starts[-1].start()]
    filtered = _STOP_MARKER_RE.sub("", filtered)
    if filtered == text:
        return text
    return re.sub(r"[ \t]+", " ", filtered).strip()


def _first_agent_message(simulation: SimulationRun) -> AssistantMessage | None:
    """The orchestrator's seeded slot: tick 0's agent chunk, or the first message."""
    if simulation.ticks:
        first = simulation.ticks[0]
        return first.agent_chunk if first.tick_id == 0 else None
    messages = simulation.messages or []
    if messages and isinstance(messages[0], AssistantMessage):
        return messages[0]
    return None


def injected_opener(simulation: SimulationRun, opener: str | None) -> bool:
    """Whether the call starts with the recorded benchmark opener."""
    if not opener or not opener.strip():
        return False
    first = _first_agent_message(simulation)
    return first is not None and (first.content or "").strip() == opener.strip()


def judged_agent_turns(
    simulation: SimulationRun, opener: str | None, rule: InputExclusions
) -> list[AgentTurn]:
    """Delivered agent turns after the contract's benchmark-speech exclusions.

    Turn indices keep ``build_agent_turns`` ordinals, so unit IDs stay stable.
    """
    turns = build_agent_turns(simulation)
    if rule == InputExclusions.NONE:
        return turns
    first = _first_agent_message(simulation)
    # In full-duplex runs ``build_agent_turns`` yields the opener as turn 0 only
    # when it was delivered as speech; audio-native openers are text-only (no
    # speech) and never become a turn, so a later verbatim repeat by the model
    # is not mistaken for it. Without ticks, the first message is always turn 0.
    if (
        turns
        and first is not None
        and injected_opener(simulation, opener)
        and (not simulation.ticks or delivered_text(first) is not None)
        and turns[0].text.strip() == (opener or "").strip()
    ):
        turns = turns[1:]
    if rule in (InputExclusions.OPENER, InputExclusions.OPENER_AND_CONTROL_TURNS):
        if rule == InputExclusions.OPENER_AND_CONTROL_TURNS:
            return [turn for turn in turns if not has_control_marker(turn.text)]
        return turns
    judged = []
    terminal_tick = terminal_speech_tick(simulation)
    for turn in turns:
        if rule == InputExclusions.OPENER_TRANSFER_AND_CONTROL_TURNS and (
            has_control_marker(turn.text)
            or (
                terminal_tick is not None
                and turn.start_tick is not None
                and turn.end_tick is not None
                and turn.start_tick <= terminal_tick <= turn.end_tick
            )
        ):
            continue
        text = without_benchmark_transfer_message(
            turn.text, interrupted=turn.interrupted
        )
        if text:
            judged.append(turn.model_copy(update={"text": text}))
    return judged
