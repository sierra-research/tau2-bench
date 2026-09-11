# Copyright Sierra
"""Facts-first interaction suite: dead air, the facts/shadow split, typed N/A
propagation, and in-verb cell aggregation on constructed typed-event fixtures."""

from pathlib import Path
from typing import Optional

import pytest
from pydantic import ValidationError

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.metrics.interaction_facts import (
    Fact,
    FactCellSummary,
    FactNAReason,
    InteractionCellAggregate,
    InteractionFacts,
    InteractionFactsConfig,
    ShadowCellSummary,
    aggregate_cells,
    build_interaction_facts,
    compute_dead_air,
    extract_interaction_facts,
)
from tau2.metrics.run_loading import LoadedCall, RunMeta
from tau2.metrics.turn_taking import TurnTakingNAReason

DUR = 0.5  # tick clock used by every fixture in this file


def _tick(
    tick_id: int,
    *,
    user: bool = False,
    agent: bool = False,
    action: Optional[str] = None,
) -> Tick:
    user_chunk = None
    if user:
        user_chunk = UserMessage(
            role="user",
            content="caller",
            contains_speech=True,
            turn_taking_action=TurnTakingAction(action=action) if action else None,
        )
    return Tick(
        tick_id=tick_id,
        timestamp=str(tick_id),
        tick_duration_seconds=DUR,
        user_chunk=user_chunk,
        agent_chunk=AssistantMessage(
            role="assistant", content="agent", contains_speech=agent
        ),
    )


def _ticks(spec: list[dict]) -> list[Tick]:
    return [_tick(i, **row) for i, row in enumerate(spec)]


# 1s caller turn, 500ms gap, 1s agent answer, 500ms trailing gap.
ANSWERED_TURN = (
    [{"user": True, "action": "keep_talking"}] * 2 + [{}] + [{"agent": True}] * 2 + [{}]
)
# The same caller turn followed only by silence.
UNANSWERED_TURN = [{"user": True, "action": "keep_talking"}] * 2 + [{}] * 3


def _meta(**overrides) -> RunMeta:
    values = dict(
        results_path="/runs/cell-a/results.json",
        experiment_label="cell-a",
        domain="telecom",
        modality="voice",
        provider="openai",
        agent_model="gpt-realtime-2",
        reasoning_effort="xhigh",
        run_git_commit="0" * 40,
        task_set_name=None,
    )
    values.update(overrides)
    return RunMeta(**values)


def _sim(ticks: Optional[list[Tick]], sim_id: str = "s1") -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id="task_1",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1.0),
        ticks=ticks,
        mode="full_duplex" if ticks else "half_duplex",
    )


def _call(
    ticks: Optional[list[Tick]],
    *,
    sim_id: str = "s1",
    language: str = "ko",
    meta: Optional[RunMeta] = None,
) -> LoadedCall:
    return LoadedCall(
        meta=meta or _meta(),
        sim=_sim(ticks, sim_id=sim_id),
        task=None,
        language=language,
    )


# ---------------------------------------------------------------------------
# Dead air
# ---------------------------------------------------------------------------


def test_dead_air_clean_call():
    # Two 0.5s silent gaps in a 3s call; the first IS the measured ordinary
    # response latency, so it stays in the totals but leaves excl_response.
    facts = compute_dead_air(
        _ticks(ANSWERED_TURN),
        long_gap_threshold_s=3.0,
        ordinary_response_gaps_s=[(1.0, 1.5)],
    )
    assert facts.total_s.value == pytest.approx(1.0)
    assert facts.fraction.value == pytest.approx(1.0 / 3.0)
    assert facts.max_gap_s.value == pytest.approx(0.5)
    assert facts.long_gap_count.value == 0.0
    assert facts.excl_response_s.value == pytest.approx(0.5)


def test_dead_air_overlapping_speech_is_not_dead_air():
    # Simultaneous speech ticks are not silence; only the trailing tick is.
    facts = compute_dead_air(
        _ticks(
            [
                {"agent": True},
                {"agent": True, "user": True, "action": "keep_talking"},
                {"user": True},
                {},
            ]
        ),
        long_gap_threshold_s=3.0,
    )
    assert facts.total_s.value == pytest.approx(0.5)
    assert facts.max_gap_s.value == pytest.approx(0.5)


def test_dead_air_mute_call_counts_the_silence():
    # A mute call is mostly dead air; with no ordinary response gaps the
    # exclusive column equals the total.
    ticks = _ticks(UNANSWERED_TURN * 2)
    facts = compute_dead_air(ticks, long_gap_threshold_s=3.0)
    # Each unanswered block ends with 1.5s silence; the second is trailing.
    assert facts.total_s.value == pytest.approx(3.0)
    assert facts.excl_response_s.value == pytest.approx(facts.total_s.value)
    assert facts.max_gap_s.value == pytest.approx(1.5)
    assert facts.long_gap_count.value == 0.0


def test_dead_air_empty_timeline_is_reasoned_na():
    facts = compute_dead_air([], long_gap_threshold_s=3.0)
    for name in (
        "total_s",
        "fraction",
        "max_gap_s",
        "long_gap_count",
        "excl_response_s",
    ):
        fact: Fact = getattr(facts, name)
        assert fact.value is None
        assert fact.na_reason == FactNAReason.NO_TICK_TIMELINE


def test_dead_air_long_gap_threshold_is_parameterized():
    # A single 3.5s gap: above the default 3.0s threshold, below a 4.0s one.
    spec = (
        [{"user": True, "action": "keep_talking"}] + [{}] * 7 + [{"agent": True}] + [{}]
    )
    default = compute_dead_air(_ticks(spec), long_gap_threshold_s=3.0)
    assert default.max_gap_s.value == pytest.approx(3.5)
    assert default.long_gap_count.value == 1.0
    relaxed = compute_dead_air(_ticks(spec), long_gap_threshold_s=4.0)
    assert relaxed.long_gap_count.value == 0.0


def test_fact_is_value_xor_reasoned_na():
    with pytest.raises(ValidationError):
        Fact()
    with pytest.raises(ValidationError):
        Fact(value=1.0, na_reason=FactNAReason.NO_EVENTS)


# ---------------------------------------------------------------------------
# Per-call extraction: facts vs shadow split, N/A propagation
# ---------------------------------------------------------------------------


def test_clean_call_facts_headline_and_shadow_are_structurally_separate():
    record = extract_interaction_facts(_call(_ticks(ANSWERED_TURN * 2)))
    facts = record.facts
    assert facts.response_latency_mean_ms.value == pytest.approx(500.0)
    assert facts.response_latency_p50_ms.value == pytest.approx(500.0)
    assert facts.plain_turn_latency_mean_ms.value == pytest.approx(500.0)
    # No tool calls anywhere: the tool-turn split is a reasoned N/A.
    assert facts.tool_turn_latency_mean_ms.na_reason == FactNAReason.NO_EVENTS
    assert facts.missed_response_rate.value == 0.0
    assert facts.route_ordinary_fraction.value == pytest.approx(1.0)
    assert facts.route_missed_response_fraction.value == 0.0
    # The promoted panel values are present.
    assert facts.response_rate.value == pytest.approx(1.0)
    assert facts.selectivity_backchannel.na_reason == FactNAReason.NO_EVENTS
    # Dead air over 12 ticks: four 0.5s gaps — two measured response gaps,
    # one inter-turn gap, one trailing gap. The response gaps count once in
    # the total and drop out of the exclusive column.
    assert facts.dead_air_total_s.value == pytest.approx(2.0)
    assert facts.dead_air_excl_response_s.value == pytest.approx(1.0)
    # The composite score never appears among the fact columns.
    assert not any("score" in name for name in InteractionFacts.model_fields)
    # It lives, labeled, in the shadow section.
    shadow = record.shadow_scores
    assert shadow.label == "uncalibrated-shadow"
    assert shadow.turn_taking.label == "uncalibrated-shadow"
    assert shadow.turn_taking.score == pytest.approx(1.0)
    assert shadow.turn_taking.adaptation_version == "eva-x-tau-v3"
    assert record.scored_turn_count == 2
    assert len(record.turns) == 2


def test_text_run_is_a_fully_reasoned_na_row():
    record = extract_interaction_facts(
        _call(None, meta=_meta(modality="text", provider="", reasoning_effort=None))
    )
    for name in InteractionFacts.model_fields:
        fact: Fact = getattr(record.facts, name)
        assert fact.value is None, name
        assert fact.na_reason == FactNAReason.NO_TICK_TIMELINE, name
    assert record.event_counts is None
    assert record.turns == []
    assert record.shadow_scores.turn_taking.score is None
    assert (
        record.shadow_scores.turn_taking.not_applicable.reason
        == TurnTakingNAReason.NO_SCOREABLE_TURNS
    )


def test_mute_call_keeps_descriptive_facts_but_shadow_score_is_na():
    # The #743 pattern: the composite is a reasoned agent-mute N/A while the
    # descriptive facts (missed-response rate, dead air) remain reported.
    record = extract_interaction_facts(
        _call(_ticks(ANSWERED_TURN + UNANSWERED_TURN * 4))
    )
    facts = record.facts
    assert facts.missed_response_rate.value == pytest.approx(0.75)
    assert facts.route_missed_response_fraction.value == pytest.approx(0.75)
    assert facts.dead_air_total_s.value is not None
    shadow = record.shadow_scores.turn_taking
    assert shadow.score is None
    assert shadow.gated_score is None
    assert shadow.passed is None
    assert shadow.not_applicable.reason == TurnTakingNAReason.AGENT_MUTE
    # Per-turn evidence is retained for inspection.
    assert record.scored_turn_count == 4


def test_include_turns_off_drops_per_turn_evidence_only():
    config = InteractionFactsConfig(include_turns=False)
    record = extract_interaction_facts(_call(_ticks(ANSWERED_TURN)), config)
    assert record.turns == []
    assert record.facts.response_latency_mean_ms.value is not None
    assert record.scored_turn_count == 1


# ---------------------------------------------------------------------------
# Cell aggregation
# ---------------------------------------------------------------------------


def test_cells_aggregate_facts_na_counts_and_shadow_means_separately():
    clean = extract_interaction_facts(_call(_ticks(ANSWERED_TURN * 2), sim_id="s1"))
    mute = extract_interaction_facts(
        _call(_ticks(ANSWERED_TURN + UNANSWERED_TURN * 4), sim_id="s2")
    )
    other_cell = extract_interaction_facts(
        _call(
            _ticks(ANSWERED_TURN),
            sim_id="s3",
            language="es",
            meta=_meta(provider="gemini", reasoning_effort="high"),
        )
    )
    text = extract_interaction_facts(
        _call(
            None,
            sim_id="s4",
            meta=_meta(modality="text", provider="", reasoning_effort=None),
        )
    )
    cells = aggregate_cells([clean, mute, other_cell, text])
    assert len(cells) == 3
    by_key = {
        (c.language, c.provider, c.reasoning_effort, c.modality): c for c in cells
    }
    ko = by_key[("ko", "openai", "xhigh", "voice")]
    assert ko.n_calls == 2
    # Facts aggregate over valued calls; both calls carry a latency value.
    assert ko.facts["response_latency_mean_ms"].n == 2
    assert ko.facts["missed_response_rate"].mean == pytest.approx((0.0 + 0.75) / 2)
    # Shadow means drop the mute N/A and account for it by reason.
    shadow = ko.shadow_scores
    assert shadow.label == "uncalibrated-shadow"
    assert shadow.n_scored == 1
    assert shadow.turn_taking_score_mean == pytest.approx(1.0)
    assert shadow.n_na == 1
    assert shadow.na_reasons == {TurnTakingNAReason.AGENT_MUTE: 1}

    es = by_key[("es", "gemini", "high", "voice")]
    assert es.n_calls == 1

    text_cell = by_key[("ko", "", None, "text")]
    summary = text_cell.facts["dead_air_total_s"]
    assert summary.n == 0
    assert summary.n_na == 1
    assert summary.na_reasons == {FactNAReason.NO_TICK_TIMELINE: 1}
    assert text_cell.shadow_scores.n_scored == 0


def test_cell_aggregate_rejects_unknown_fact_keys():
    with pytest.raises(ValueError, match="not InteractionFacts fields"):
        InteractionCellAggregate(
            language="en",
            domain="telecom",
            modality="voice",
            provider="openai",
            reasoning_effort=None,
            n_calls=1,
            facts={"not_a_fact": FactCellSummary(mean=1.0, median=1.0, n=1, n_na=0)},
            shadow_scores=ShadowCellSummary(
                turn_taking_score_mean=None,
                turn_taking_gated_score_mean=None,
                turn_taking_pass_rate=None,
                n_scored=0,
                n_na=0,
            ),
        )


# ---------------------------------------------------------------------------
# End-to-end over a real dir-format run (fixture-based; worktrees carry no
# stored run data)
# ---------------------------------------------------------------------------


@pytest.fixture
def run_on_disk(tmp_path: Path) -> Path:
    from fixtures_runs import make_hi_results

    from tau2.config import ReasoningEffort
    from tau2.data_model.simulation import AudioNativeConfig

    sims = [
        _sim(_ticks(ANSWERED_TURN * 2), sim_id="sim_clean"),
        _sim(_ticks(ANSWERED_TURN + UNANSWERED_TURN * 4), sim_id="sim_mute"),
    ]
    return make_hi_results(
        tmp_path,
        sims,
        domain="telecom",
        audio_native_config=AudioNativeConfig(
            provider="openai",
            model="gpt-realtime-2",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
        agent_llm="openai:gpt-realtime-2",
    )


def test_build_interaction_facts_artifact_is_content_deterministic(run_on_disk):
    first = build_interaction_facts([run_on_disk], InteractionFactsConfig())
    second = build_interaction_facts([run_on_disk], InteractionFactsConfig())
    assert first.artifact_id == second.artifact_id
    assert len(first.records) == 2
    assert first.source_runs[0].n_calls == 2
    assert len(first.cells) == 1
    cell = first.cells[0]
    assert cell.provider == "openai"
    assert cell.reasoning_effort == "xhigh"
    assert cell.shadow_scores.n_scored == 1
    assert cell.shadow_scores.na_reasons == {TurnTakingNAReason.AGENT_MUTE: 1}


def test_cli_verb_writes_validating_artifact(run_on_disk, tmp_path):
    import argparse

    from tau2.metrics.cli import run_metrics_interaction_facts
    from tau2.metrics.interaction_facts import InteractionFactsArtifact

    out = tmp_path / "facts.json"
    run_metrics_interaction_facts(
        argparse.Namespace(
            results=[str(run_on_disk)],
            output=out,
            langs=None,
            domain=None,
            max_sims=None,
            long_gap_threshold=None,
            no_turns=True,
        )
    )
    loaded = InteractionFactsArtifact.model_validate_json(out.read_text())
    assert loaded.records and loaded.cells
    assert all(record.turns == [] for record in loaded.records)
    assert loaded.config.include_turns is False
    assert loaded.config.long_gap_threshold_s == pytest.approx(3.0)
