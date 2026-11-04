"""Call-level Interaction and Experience rules on synthetic calls and verdicts."""

import pytest

from tau2.data_model.judges import StoredVerdict
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics import call_quality
from tau2.metrics.call_quality import CallOutcome
from tau2.metrics.interaction_quality import (
    InteractionMeasurements,
    measure_interaction,
)
from tau2.scripts.leaderboard.benchmark_quality import (
    CallQuality,
    InteractionFlags,
    cell_quality,
    score_call,
)


def _simulation(ticks=None) -> SimulationRun:
    return SimulationRun(
        id="sim",
        task_id="task",
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason="user_stop",
        ticks=ticks,
    )


def _timeline(user, agent, length, silent=()) -> SimulationRun:
    """Caller and agent speech spans (inclusive tick ranges), 0.2 s per tick.

    ``silent`` agent spans carry chunks without speech (``contains_speech``
    false), as providers emit around tool calls.
    """

    def owner(spans, tick):
        return next((n for n, (a, b) in enumerate(spans) if a <= tick <= b), None)

    ticks = []
    for tick in range(length):
        caller, speaker, quiet = (
            owner(user, tick),
            owner(agent, tick),
            owner(silent, tick),
        )
        agent_chunk = None
        if speaker is not None:
            agent_chunk = AssistantMessage(
                role="assistant", content=f"a{tick} ", utterance_ids=[f"a{speaker}"]
            )
        elif quiet is not None:
            agent_chunk = AssistantMessage(
                role="assistant",
                content="",
                contains_speech=False,
                utterance_ids=[f"q{quiet}"],
            )
        ticks.append(
            Tick(
                tick_id=tick,
                timestamp=str(tick),
                tick_duration_seconds=0.2,
                user_chunk=UserMessage(
                    role="user", content=f"u{tick} ", utterance_ids=[f"u{caller}"]
                )
                if caller is not None
                else None,
                agent_chunk=agent_chunk,
            )
        )
    return _simulation(ticks)


def _verdict(measure, units, unit_errors=(), error=None) -> StoredVerdict:
    return StoredVerdict.model_validate(
        {
            "identity": {
                "input": {
                    "domain": "airline",
                    "run_id": "r" * 64,
                    "simulation_id": "sim",
                    "trial": 0,
                    "language": "en",
                    "agent_gender": "female",
                    "caller_gender": "male",
                    "input_fingerprint": "f" * 64,
                    "has_audio": True,
                },
                "measure": measure,
                "contract_id": f"{measure}-fixture",
                "implementation": "naturalness-v17",
                "input_exclusions": "none",
                "model": "fixture-judge",
                "model_args": {},
                "prompt_version": "v1",
                "rubric_version": "r1",
                "transport_revision": "multimodal-responses-v1",
                "catalog_revision": "fixture-catalog",
            },
            "verdict": None
            if error
            else {
                "units": [
                    {
                        "unit_id": unit_id,
                        "opportunity": opportunity,
                        "violation": violation,
                        "evidence": "fixture",
                    }
                    for unit_id, opportunity, violation in units
                ],
                "unit_errors": [
                    {
                        "unit_id": unit_id,
                        "kind": kind,
                        "error": "fixture",
                        "finish_reason": finish_reason[0] if finish_reason else None,
                    }
                    for unit_id, kind, *finish_reason in unit_errors
                ],
            },
            "error": error,
        }
    )


def _measured(**overrides) -> InteractionMeasurements:
    values = dict(
        response_total=2,
        responses=2,
        response_latency_mean=1.0,
        yield_total=0,
        yields=0,
        backchannel_total=0,
        backchannel_errors=0,
        vocal_tic_total=0,
        vocal_tic_errors=0,
        non_directed_total=0,
        non_directed_errors=0,
        agent_interruptions=0,
        max_agent_floor_hold_seconds=10.0,
    )
    return InteractionMeasurements(**(values | overrides))


def _tool_use(violation=False, opportunity=True):
    return _verdict("tool_use", [("call", opportunity, violation)])


def _flags(**values):
    return dict.fromkeys(call_quality.INTERACTION_COMPONENTS, 0) | values


@pytest.mark.parametrize(
    ("flags", "outcome"),
    [
        (_flags(), CallOutcome.PASS),
        (_flags(nonresponse=1), CallOutcome.PASS),
        (_flags(nonresponse=1, tool_misuse=1), CallOutcome.FAIL),
        # One failure plus an unknown could still reach two.
        (_flags(nonresponse=1, selectivity=None), CallOutcome.UNSCORED),
        # Two failures decide the call whatever the unknowns are.
        (_flags(interruption=1, monologue=1, tool_misuse=None), CallOutcome.FAIL),
        # A single unknown cannot reach two on its own.
        (_flags(tool_misuse=None), CallOutcome.PASS),
        (_flags(tool_misuse=None, selectivity=None), CallOutcome.UNSCORED),
    ],
)
def test_two_of_five_rule(flags, outcome):
    assert call_quality.interaction_outcome(flags) == outcome


def test_event_flags_need_more_than_three_events():
    at_three = call_quality.interaction_flags(
        _measured(
            agent_interruptions=3,
            backchannel_total=5,
            backchannel_errors=1,
            vocal_tic_total=2,
            vocal_tic_errors=1,
            non_directed_total=1,
            non_directed_errors=1,
        ),
        _tool_use(),
    )
    assert at_three["interruption"] == 0 and at_three["selectivity"] == 0
    at_four = call_quality.interaction_flags(
        _measured(
            agent_interruptions=4,
            backchannel_total=5,
            backchannel_errors=2,
            vocal_tic_total=2,
            vocal_tic_errors=1,
            non_directed_total=1,
            non_directed_errors=1,
        ),
        _tool_use(),
    )
    assert at_four["interruption"] == 1 and at_four["selectivity"] == 1
    # Without any selectivity opportunity the flag is unknown, not a pass.
    assert call_quality.interaction_flags(_measured(), _tool_use())["selectivity"] is (
        None
    )


def test_nonresponse_monologue_and_unmeasurable_calls():
    flags = call_quality.interaction_flags
    assert flags(_measured(responses=1), None)["nonresponse"] == 1
    assert flags(_measured(response_latency_mean=3.01), None)["nonresponse"] == 1
    assert flags(_measured(response_latency_mean=3.0), None)["nonresponse"] == 0
    assert flags(_measured(yield_total=2, yields=1), None)["nonresponse"] == 1
    assert flags(_measured(response_total=0, responses=0), None)["nonresponse"] is None
    assert flags(_measured(max_agent_floor_hold_seconds=45.0), None)["monologue"] == 0
    assert flags(_measured(max_agent_floor_hold_seconds=45.2), None)["monologue"] == 1
    assert flags(_measured(max_agent_floor_hold_seconds=None), None)["monologue"] is (
        None
    )
    # A call without ticks only has the judge-backed component.
    assert flags(None, _tool_use(violation=True)) == dict.fromkeys(
        ["nonresponse", "interruption", "selectivity", "monologue"]
    ) | {"tool_misuse": 1}


def test_tool_misuse_reads_the_whole_call_tool_use_verdict():
    flag = call_quality.tool_misuse_flag
    assert flag(_tool_use(violation=True)) == 1
    assert flag(_tool_use()) == 0
    assert flag(_tool_use(opportunity=False)) is None
    assert flag(None) is None
    assert flag(_verdict("tool_use", [], error="ValueError: fixture")) is None
    assert flag(_verdict("tool_use", [("call", True, True)], [("x", "response")])) is (
        None
    )
    with pytest.raises(ValueError, match="one whole-call unit"):
        flag(_verdict("tool_use", [("0", True, False)]))


def _live(*violations):
    """Paired Live naturalness and fidelity units, keyed by segment ID."""
    units = [
        (f"live-segment:{index}", True, flagged)
        for index, flagged in enumerate(violations)
    ]
    return {
        "naturalness": _verdict("naturalness", units),
        "speech_fidelity": _verdict(
            "speech_fidelity", [(unit_id, True, False) for unit_id, _, _ in units]
        ),
    }


def test_experience_passes_at_a_quarter_and_fails_above_it():
    sim = _simulation()
    assert call_quality.experience_counts(sim, _live(True, False, False, False)) == (
        4,
        1,
    )
    assert call_quality.experience_outcome(4, 1) == CallOutcome.PASS
    assert call_quality.experience_counts(
        sim, _live(True, True, False, False, False)
    ) == (5, 2)
    assert call_quality.experience_outcome(5, 2) == CallOutcome.FAIL
    assert call_quality.experience_outcome(0, 0) == CallOutcome.UNSCORED
    # An issue from either judge counts once per utterance.
    verdicts = _live(True, False, False, False)
    verdicts["speech_fidelity"] = _verdict(
        "speech_fidelity",
        [(f"live-segment:{index}", True, index < 2) for index in range(4)],
    )
    assert call_quality.experience_counts(sim, verdicts) == (4, 2)


def test_incomplete_or_unpaired_judgments_leave_experience_unscored():
    sim = _simulation()
    complete = _live(False, False)
    missing = {"naturalness": complete["naturalness"]}
    assert call_quality.experience_counts(sim, missing) == (2, None)
    # A fidelity-only input error leaves a naturalness utterance unpaired.
    blocked = dict(complete)
    blocked["speech_fidelity"] = _verdict(
        "speech_fidelity",
        [("live-segment:0", True, False)],
        [("live-segment:1", "input")],
    )
    assert call_quality.experience_counts(sim, blocked) == (2, None)
    # Any other unit error (a judge timeout) leaves the call unscored.
    timed_out = dict(complete)
    timed_out["speech_fidelity"] = _verdict(
        "speech_fidelity",
        [("live-segment:0", True, False)],
        [("live-segment:1", "response")],
    )
    assert call_quality.experience_counts(sim, timed_out) == (2, None)
    errored = dict(complete)
    errored["naturalness"] = _verdict("naturalness", [], error="ValueError: x")
    assert call_quality.experience_counts(sim, errored) == (0, None)
    unpaired = dict(complete)
    unpaired["speech_fidelity"] = _verdict(
        "speech_fidelity", [("live-segment:0", True, False)]
    )
    assert call_quality.experience_counts(sim, unpaired) == (2, None)
    # No-opportunity naturalness units are not eligible.
    sparse = dict(complete)
    sparse["naturalness"] = _verdict(
        "naturalness",
        [("live-segment:0", True, True), ("live-segment:1", False, False)],
    )
    assert call_quality.experience_counts(sim, sparse) == (1, 1)


def test_live_speech_without_a_transcript_is_left_out():
    """Both judges skip an utterance with no Live transcript; the rest is scored."""
    sim = _simulation()
    units = [("live-segment:0", True, True), ("live-segment:1", True, False)]
    gap = [("live-segment:2", "input")]
    verdicts = {
        "naturalness": _verdict("naturalness", units, gap),
        "speech_fidelity": _verdict(
            "speech_fidelity", [(u, True, False) for u, _, _ in units], gap
        ),
    }
    assert call_quality.experience_counts(sim, verdicts) == (2, 1)
    assert call_quality.experience_outcome(2, 1) == CallOutcome.FAIL
    # With no transcribed utterance left, there is nothing to score.
    only_gaps = {
        measure: _verdict(measure, [], gap)
        for measure in ("naturalness", "speech_fidelity")
    }
    assert call_quality.experience_counts(sim, only_gaps) == (0, 0)
    assert call_quality.experience_outcome(0, 0) == CallOutcome.UNSCORED


def test_fidelity_output_limit_leaves_out_only_its_utterance():
    """A clip the fidelity judge could not finish drops that utterance alone."""
    sim = _simulation()
    limit = [("live-segment:2", "response", "length")]

    def verdicts(flagged):
        units = [
            ("live-segment:0", True, True),
            ("live-segment:1", True, False),
            ("live-segment:2", True, flagged),
        ]
        return {
            "naturalness": _verdict("naturalness", units),
            "speech_fidelity": _verdict(
                "speech_fidelity", [(u, True, False) for u, _, _ in units[:2]], limit
            ),
        }

    # Unflagged by naturalness: left out, the other two utterances are scored.
    assert call_quality.experience_counts(sim, verdicts(False)) == (2, 1)
    # Flagged by naturalness: a known issue, so it still counts.
    assert call_quality.experience_counts(sim, verdicts(True)) == (3, 2)
    # Every utterance left out: nothing to score.
    only_limit = {
        "naturalness": _verdict("naturalness", [("live-segment:2", True, False)]),
        "speech_fidelity": _verdict("speech_fidelity", [], limit),
    }
    assert call_quality.experience_counts(sim, only_limit) == (0, 0)
    assert call_quality.experience_outcome(0, 0) == CallOutcome.UNSCORED
    # The same error from naturalness, or a fidelity timeout, still blocks.
    blocked = verdicts(False)
    blocked["naturalness"] = _verdict(
        "naturalness", [("live-segment:0", True, False)], limit
    )
    assert call_quality.experience_counts(sim, blocked)[1] is None
    timed_out = verdicts(False)
    timed_out["speech_fidelity"] = _verdict(
        "speech_fidelity",
        [("live-segment:0", True, False), ("live-segment:1", True, False)],
        [("live-segment:2", "response")],
    )
    assert call_quality.experience_counts(sim, timed_out) == (3, None)


def test_non_live_units_pair_by_agent_chunk_groups():
    """Fidelity numbers silent chunk groups too; naturalness skips them."""
    sim = _timeline(user=[(0, 2)], agent=[(4, 6), (12, 14)], length=16, silent=[(8, 9)])
    assert call_quality.utterance_alignment(sim) == {"0": "0", "1": "2"}
    verdicts = {
        "naturalness": _verdict("naturalness", [("0", True, False), ("1", True, True)]),
        "speech_fidelity": _verdict(
            "speech_fidelity", [("0", True, False), ("2", True, False)]
        ),
    }
    assert call_quality.experience_counts(sim, verdicts) == (2, 1)
    # Joining by ordinal position instead would leave unit "1" unpaired.
    verdicts["speech_fidelity"] = _verdict(
        "speech_fidelity", [("0", True, False), ("1", True, False)]
    )
    assert call_quality.experience_counts(sim, verdicts) == (2, None)


def test_measurements_come_from_the_tick_timeline():
    assert measure_interaction(_simulation()) is None
    answered = measure_interaction(_timeline([(0, 9)], [(12, 20)], 40))
    assert (answered.response_total, answered.responses) == (1, 1)
    assert answered.agent_interruptions == 0
    assert answered.max_agent_floor_hold_seconds == pytest.approx(1.8)
    # Four agent onsets over one long caller turn.
    talked_over = measure_interaction(
        _timeline([(0, 30)], [(5, 8), (12, 14), (18, 20), (24, 26)], 60)
    )
    assert talked_over.agent_interruptions == 4
    # Agent pauses between caller turns still hold the floor.
    held = measure_interaction(_timeline([(0, 4), (40, 44)], [(6, 10), (20, 34)], 50))
    assert held.max_agent_floor_hold_seconds == pytest.approx(29 * 0.2)


def test_scored_call_and_cell_totals():
    sim = _timeline([(0, 30)], [(5, 8), (12, 14), (18, 20), (24, 26)], 60)
    verdicts = _live(False, False) | {"tool_use": _tool_use(violation=True)}
    call = score_call(sim, verdicts)
    assert call.flags.interruption == 1 and call.flags.tool_misuse == 1
    assert call.interaction == "fail"
    assert (call.experience, call.experience_utterances, call.experience_issues) == (
        "pass",
        2,
        0,
    )
    cell = cell_quality("en", "airline", [call])
    assert cell.interaction.failed == 1 and cell.experience.passed == 1
    assert cell.interaction_components.interruption.model_dump() == {
        "flagged": 1,
        "known": 1,
    }
    assert cell.interaction_components.selectivity.known == 0
    with pytest.raises(ValueError, match="totals do not match"):
        cell.model_validate(
            cell.model_dump() | {"interaction": cell.experience.model_dump()}
        )


def test_call_records_must_follow_the_rules():
    flags = InteractionFlags(
        nonresponse=1, interruption=1, selectivity=None, monologue=0, tool_misuse=0
    )
    record = dict(
        simulation_id="sim",
        interaction="fail",
        flags=flags,
        experience="pass",
        experience_utterances=4,
        experience_issues=1,
    )
    CallQuality(**record)
    with pytest.raises(ValueError, match="Interaction outcome"):
        CallQuality(**(record | {"interaction": "pass"}))
    with pytest.raises(ValueError, match="Interaction outcome"):
        CallQuality(**(record | {"interaction": "excluded"}))
    with pytest.raises(ValueError, match="Experience outcome"):
        CallQuality(**(record | {"experience_issues": 2}))
    with pytest.raises(ValueError, match="Experience outcome"):
        CallQuality(**(record | {"experience_issues": None}))
