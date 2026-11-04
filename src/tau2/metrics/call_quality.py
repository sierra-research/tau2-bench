"""Call-level Interaction and Experience (shown as Generation quality) rules.

Both are per-call pass/fail verdicts derived offline from a saved call and its
saved judge verdicts; neither feeds the task reward.

**Interaction** (``voice-v2-interaction-call-v1``) sets five binary flags:

- nonresponse: the agent missed a caller turn, answered late (mean latency over
  3 s) or failed to yield to an interruption;
- interruption: more than 3 agent onsets over caller speech;
- selectivity: more than 3 errors summed over backchannels, vocal tics and
  non-directed speech;
- monologue: an agent floor hold over 45 s;
- tool_misuse: the saved ``tool_use`` judge verdict.

A flag is unknown (``None``) when the call has no opportunity or no usable
verdict. A call fails when at least two known flags are 1, passes when even
setting every unknown flag to 1 cannot reach two, and is otherwise unscored.

**Experience** (``voice-v2-experience-call-v2``): an eligible agent utterance
has an issue when the ``naturalness`` or ``speech_fidelity`` judge flagged it.
A call passes when at most 25% of its eligible utterances have an issue. It is
unscored, never failed, unless both verdicts are complete and every eligible
naturalness utterance pairs with a fidelity unit. A verdict is complete when it
has no whole-call error and no unit errors other than input errors: Live
speech recorded without a transcript is a recording gap, so those utterances
are left out, as in the judge diagnostics. v1 left the whole call unscored.

A fidelity clip whose judge reply hit its output limit even after retries is
a judge failure on that utterance alone. The utterance counts as an issue when
naturalness flagged it and is otherwise left out; the rest of the call is
scored. Submission preparation caps how many such clips a package may have.
"""

from enum import StrEnum

from tau2.data_model.judges import StoredVerdict, is_fidelity_output_limit
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.interaction_quality import InteractionMeasurements

INTERACTION_VERSION = "voice-v2-interaction-call-v1"
EXPERIENCE_VERSION = "voice-v2-experience-call-v2"
INTERACTION_COMPONENTS = (
    "nonresponse",
    "interruption",
    "selectivity",
    "monologue",
    "tool_misuse",
)
RESPONSE_LATENCY_SECONDS = 3.0
MONOLOGUE_SECONDS = 45.0
# Flags fire only above these event counts.
INTERRUPTION_EVENTS = 3
SELECTIVITY_ERRORS = 3
# Experience fails when issues exceed 1/EXPERIENCE_ISSUE_DIVISOR (25%) of the
# eligible utterances; integer arithmetic keeps the boundary exact.
EXPERIENCE_ISSUE_DIVISOR = 4
TOOL_USE_MEASURE = "tool_use"
_LIVE_UNIT_PREFIX = "live-segment:"


class CallOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSCORED = "unscored"


Flags = dict[str, int | None]


def _nonresponse(measured: InteractionMeasurements) -> int | None:
    """Responsiveness or yielding failed; unknown when neither had an opportunity."""
    checks = []
    if measured.response_total:
        latency = measured.response_latency_mean
        checks.append(
            measured.responses != measured.response_total
            or (latency is not None and latency > RESPONSE_LATENCY_SECONDS)
        )
    if measured.yield_total:
        checks.append(measured.yields != measured.yield_total)
    return int(any(checks)) if checks else None


def tool_misuse_flag(verdict: StoredVerdict | None) -> int | None:
    """The saved whole-call ``tool_use`` verdict; unknown if absent or partial."""
    if (
        verdict is None
        or verdict.error is not None
        or verdict.verdict is None
        or verdict.verdict.unit_errors
    ):
        return None
    units = verdict.verdict.units
    if len(units) != 1 or units[0].unit_id != "call":
        raise ValueError(
            f"{verdict.identity.input.simulation_id}: tool_use verdict is not "
            "one whole-call unit"
        )
    return int(units[0].violation) if units[0].opportunity else None


def interaction_flags(
    measured: InteractionMeasurements | None, tool_use: StoredVerdict | None
) -> Flags:
    """The five component flags of one call."""
    flags: Flags = dict.fromkeys(INTERACTION_COMPONENTS)
    if measured is not None:
        flags["nonresponse"] = _nonresponse(measured)
        flags["interruption"] = int(measured.agent_interruptions > INTERRUPTION_EVENTS)
        if measured.selectivity_total:
            flags["selectivity"] = int(measured.selectivity_errors > SELECTIVITY_ERRORS)
        if measured.max_agent_floor_hold_seconds is not None:
            flags["monologue"] = int(
                measured.max_agent_floor_hold_seconds > MONOLOGUE_SECONDS
            )
    flags["tool_misuse"] = tool_misuse_flag(tool_use)
    return flags


def interaction_outcome(flags: Flags) -> CallOutcome:
    """Two known failures fail; pass only if the unknowns cannot make two."""
    failed = sum(flags[name] == 1 for name in INTERACTION_COMPONENTS)
    unknown = sum(flags[name] is None for name in INTERACTION_COMPONENTS)
    if failed >= 2:
        return CallOutcome.FAIL
    if failed + unknown < 2:
        return CallOutcome.PASS
    return CallOutcome.UNSCORED


def experience_outcome(utterances: int, issues: int | None) -> CallOutcome:
    """Pass at or below the issue share; ``issues`` is None when unpaired."""
    if issues is None or utterances == 0:
        return CallOutcome.UNSCORED
    if EXPERIENCE_ISSUE_DIVISOR * issues > utterances:
        return CallOutcome.FAIL
    return CallOutcome.PASS


def _complete(verdict: StoredVerdict | None) -> bool:
    return (
        verdict is not None
        and verdict.error is None
        and verdict.verdict is not None
        and all(
            error.kind == "input"
            or is_fidelity_output_limit(verdict.identity.measure, error)
            for error in verdict.verdict.unit_errors
        )
    )


def experience_counts(
    simulation: SimulationRun, verdicts: dict[str, StoredVerdict]
) -> tuple[int, int | None]:
    """Scored utterances and utterances with an issue (None if unpaired).

    Eligible utterances are the naturalness units with an opportunity; units
    with an input error (no Live transcript) are not units and stay out. An
    eligible utterance whose fidelity clip hit the judge's output limit is
    scored only when naturalness flagged it, as an issue.
    """
    naturalness = verdicts.get("naturalness")
    fidelity = verdicts.get("speech_fidelity")
    eligible = {
        unit.unit_id: unit
        for unit in (
            naturalness.verdict.units
            if naturalness is not None and naturalness.verdict is not None
            else []
        )
        if unit.opportunity
    }
    if not (_complete(naturalness) and _complete(fidelity)):
        return len(eligible), None
    clips = {unit.unit_id: unit for unit in fidelity.verdict.units if unit.opportunity}
    limited = {
        error.unit_id
        for error in fidelity.verdict.unit_errors
        if is_fidelity_output_limit(fidelity.identity.measure, error)
    }
    if eligible and all(unit_id.startswith(_LIVE_UNIT_PREFIX) for unit_id in eligible):
        pairs = {unit_id: unit_id for unit_id in eligible}
    else:
        pairs = utterance_alignment(simulation)
    scored = issues = 0
    for turn_id, turn in eligible.items():
        clip = pairs.get(turn_id)
        if clip in clips:
            scored += 1
            issues += bool(turn.violation or clips[clip].violation)
        elif clip in limited:
            scored += bool(turn.violation)
            issues += bool(turn.violation)
        else:
            return len(eligible), None
    return scored, issues


def utterance_alignment(simulation: SimulationRun) -> dict[str, str]:
    """Map naturalness turn indexes to speech-fidelity clip indexes.

    Outside Live runs the two judges number utterances differently: fidelity
    numbers every non-tool agent chunk group (chunks sharing an utterance ID),
    silent ones included; naturalness numbers only the groups with delivered
    speech text. Live runs use the shared segment ID instead.
    """
    groups: list[list[AssistantMessage]] = []
    current: set[str] = set()
    for tick in simulation.ticks or []:
        chunk = tick.agent_chunk
        if chunk is None or chunk.is_tool_call():
            continue
        ids = set(chunk.utterance_ids or [])
        if groups and current and ids and not current.isdisjoint(ids):
            groups[-1].append(chunk)
            current.update(ids)
        else:
            groups.append([chunk])
            current = ids
    pairs: dict[str, str] = {}
    for clip, chunks in enumerate(groups):
        merged = (
            chunks[0] if len(chunks) == 1 else AssistantMessage.merge_chunks(chunks)
        )
        if merged.contains_speech is False or not (
            merged.audio_script_gold or merged.content
        ):
            continue
        pairs[str(len(pairs))] = str(clip)
    return pairs
