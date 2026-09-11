# Copyright Sierra
"""tau2 judges recheck-gender: deterministic gender re-derivation on stored runs.

Fixture round trip: a dir-format root whose sims carry stored v1
gender_agreement FAILs — one where the only masculine match is the pinned
tick-0 greeting (must flip to NO_OPPORTUNITY) and one with a genuine
model-turn violation (must stay FAIL at the current checker version) — plus
snapshot, aggregate-consistency, out-of-scope (LLM verdict) and idempotency
guarantees. No LLM calls anywhere.
"""

import json

from fixtures_runs import make_hi_results, make_task

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.voice import SpeechEnvironment
from tau2.judges.export import judge_factor_ids
from tau2.judges.nativeness.checkers import GENDER_AGREEMENT_CHECKER_VERSION
from tau2.judges.nativeness.gender_recheck import (
    GenderRecheckSnapshot,
    recheck_gender_root,
)
from tau2.judges.nativeness.harness import aggregate_nativeness_checks

OLD_PINNED_GREETING = "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?"
V1_AGENT_EVIDENCE = "gender-v1: known female agent contradicted by 'मैं … सकता हूँ'"


def _greeting_tick() -> Tick:
    """Tick 0 exactly as the runner stores the injected greeting."""
    return Tick(
        tick_id=0,
        timestamp="0",
        agent_chunk=AssistantMessage(
            role="assistant",
            content=OLD_PINNED_GREETING,
            is_audio=False,
            contains_speech=False,
            chunk_id=0,
            is_final_chunk=True,
        ),
        tick_duration_seconds=0.05,
    )


def _speech_tick(tick_id: int, text: str, uid: str) -> Tick:
    return Tick(
        tick_id=tick_id,
        timestamp=str(tick_id),
        agent_chunk=AssistantMessage(
            role="assistant", content=text, utterance_ids=[uid]
        ),
        tick_duration_seconds=0.05,
    )


def _stored_checks(gender_evidence: str) -> list[NativenessFactorCheck]:
    """Complete stored hi judge-factor checks with a v1 deterministic gender FAIL."""
    checks = []
    for factor_id in sorted(judge_factor_ids("hi")):
        if factor_id == "gender_agreement":
            checks.append(
                NativenessFactorCheck(
                    id=factor_id,
                    category="grammar",
                    severity=3,
                    outcome=JudgeOutcome.FAIL,
                    evidence=gender_evidence,
                    observed_severity=3,
                    violation_count=1,
                    evaluation_level="utterance",
                )
            )
        else:
            checks.append(
                NativenessFactorCheck(
                    id=factor_id,
                    category="register",
                    severity=2,
                    outcome=JudgeOutcome.PASS,
                    evidence="stored evidence",
                )
            )
    return checks


def _voice_sim(sim_id: str, task_id: str, agent_turns: list[str]) -> SimulationRun:
    """A voice sim: pinned tick-0 greeting + spoken model turns, stored v1 FAIL."""
    checks = _stored_checks(V1_AGENT_EVIDENCE)
    aggregates = aggregate_nativeness_checks(checks)
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason.AGENT_STOP,
        agent_provider="openai",
        agent_voice="marin",  # pinned catalog voice -> agent gender female
        speech_environment=SpeechEnvironment(language="hi"),
        ticks=[_greeting_tick()]
        + [_speech_tick(i + 1, text, f"u{i}") for i, text in enumerate(agent_turns)],
        nativeness_info=NativenessInfo(
            factor_checks=checks,
            language="hi",
            script="deva",
            rubric_version="nativeness-rubric-v20",
            judge_model="stored-judge",
            judge_prompt_version="stored-prompt",
            **aggregates.model_dump(),
        ),
    )


def _make_root(tmp_path):
    """One root with one dir-format cell holding both fixture sims."""
    greeting_only = _voice_sim(
        "sim_greeting_only",
        "t1",
        # Benign model speech: no masculine self-reference anywhere.
        ["जी, मैं आपकी बुकिंग देखती हूँ।", "धन्यवाद, आपका काम हो गया।"],
    )
    genuine = _voice_sim(
        "sim_model_match",
        "t2",
        # The model itself speaks masculine — a real violation.
        ["जी, मैं आपकी मदद कर सकता हूँ।"],
    )
    root = tmp_path / "run_root"
    root.mkdir()
    make_hi_results(
        root,
        [greeting_only, genuine],
        name="hi_cell",
        tasks=[make_task("t1"), make_task("t2")],
    )
    return root


def _gender_check(root, sim_id):
    sim_file = root / "hi_cell" / "simulations" / f"{sim_id}.json"
    sim = SimulationRun.model_validate_json(sim_file.read_text())
    assert sim.nativeness_info is not None
    check = next(
        c for c in sim.nativeness_info.factor_checks if c.id == "gender_agreement"
    )
    return sim, check


def test_recheck_flips_greeting_only_fail_and_keeps_genuine_fail(tmp_path):
    root = _make_root(tmp_path)

    report = recheck_gender_root(root)

    assert report.cells == 1
    assert report.sims_scanned == 2
    assert report.candidates == 2
    assert report.fails_before == 2
    assert report.fails_after == 1
    assert report.patched == 2  # both rewritten: one flip, one v1 -> v2 evidence
    assert report.skipped == 0

    flipped_sim, flipped = _gender_check(root, "sim_greeting_only")
    assert flipped.outcome == JudgeOutcome.NO_OPPORTUNITY
    assert flipped.evidence is None
    assert flipped.violation_count == 0
    assert flipped.observed_severity == 0

    kept_sim, kept = _gender_check(root, "sim_model_match")
    assert kept.outcome == JudgeOutcome.FAIL
    assert (kept.evidence or "").startswith(
        f"gender-{GENDER_AGREEMENT_CHECKER_VERSION}:"
    )

    # Aggregates recomputed with the harness's own rule, consistent with the
    # patched checks; untouched judge provenance fields survive.
    for sim in (flipped_sim, kept_sim):
        info = sim.nativeness_info
        expected = aggregate_nativeness_checks(info.factor_checks)
        assert info.score == expected.score
        assert info.num_pass == expected.num_pass
        assert info.num_fail == expected.num_fail
        assert info.num_no_opportunity == expected.num_no_opportunity
        assert info.score_coverage == expected.score_coverage
        assert info.judge_model == "stored-judge"
        assert info.judge_prompt_version == "stored-prompt"
        assert info.rubric_version == "nativeness-rubric-v20"
    assert flipped_sim.nativeness_info.num_fail == 0
    assert kept_sim.nativeness_info.num_fail == 1


def test_recheck_writes_provenance_snapshot_before_patching(tmp_path):
    root = _make_root(tmp_path)

    report = recheck_gender_root(root)

    snapshot_path = root / "gender_recheck_snapshot.json"
    assert report.snapshot_path == str(snapshot_path)
    snapshot = GenderRecheckSnapshot.model_validate_json(snapshot_path.read_text())
    assert snapshot.checker_version_after == GENDER_AGREEMENT_CHECKER_VERSION
    assert snapshot.checker_versions_before == ["v1"]
    assert snapshot.created_at
    assert {entry.sim_id for entry in snapshot.sims} == {
        "sim_greeting_only",
        "sim_model_match",
    }
    by_id = {entry.sim_id: entry for entry in snapshot.sims}
    prior = by_id["sim_greeting_only"]
    assert prior.prior_outcome == JudgeOutcome.FAIL
    assert prior.prior_evidence == V1_AGENT_EVIDENCE
    assert prior.new_outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["sim_model_match"].new_outcome == JudgeOutcome.FAIL


def test_recheck_is_idempotent_and_never_resnapshots(tmp_path):
    root = _make_root(tmp_path)
    recheck_gender_root(root)
    original_snapshot = (root / "gender_recheck_snapshot.json").read_text()
    files_after_first = {
        p: p.read_text() for p in (root / "hi_cell" / "simulations").glob("*.json")
    }

    second = recheck_gender_root(root)

    assert second.patched == 0
    assert second.snapshot_path is None
    assert second.fails_before == second.fails_after == 1
    assert (root / "gender_recheck_snapshot.json").read_text() == original_snapshot
    assert not (root / "gender_recheck_snapshot_2.json").exists()
    for path, content in files_after_first.items():
        assert path.read_text() == content


def test_recheck_leaves_llm_verdicts_and_non_fail_outcomes_alone(tmp_path):
    # An LLM-judged FAIL (free-text evidence) and a PASS are out of scope for
    # the deterministic recheck, even when the sim carries the pinned greeting.
    llm_fail = _voice_sim("sim_llm_fail", "t1", ["जी, ठीक है।"])
    check = next(
        c for c in llm_fail.nativeness_info.factor_checks if c.id == "gender_agreement"
    )
    check.evidence = "Agent used masculine agreement; a native speaker would not."
    passing = _voice_sim("sim_pass", "t2", ["जी, ठीक है।"])
    gender = next(
        c for c in passing.nativeness_info.factor_checks if c.id == "gender_agreement"
    )
    gender.outcome = JudgeOutcome.PASS
    gender.evidence = None
    gender.violation_count = 0
    gender.observed_severity = 0
    root = tmp_path / "run_root"
    root.mkdir()
    make_hi_results(
        root,
        [llm_fail, passing],
        name="hi_cell",
        tasks=[make_task("t1"), make_task("t2")],
    )
    before = {
        p.name: json.loads(p.read_text())["nativeness_info"]["factor_checks"]
        for p in (root / "hi_cell" / "simulations").glob("*.json")
    }

    report = recheck_gender_root(root)

    assert report.candidates == 0
    assert report.patched == 0
    assert report.fails_before == report.fails_after == 1
    assert report.snapshot_path is None
    after = {
        p.name: json.loads(p.read_text())["nativeness_info"]["factor_checks"]
        for p in (root / "hi_cell" / "simulations").glob("*.json")
    }
    assert after == before


def test_cli_verb_runs_end_to_end(tmp_path):
    from argparse import Namespace

    from tau2.judges.cli import run_judges_recheck_gender

    root = _make_root(tmp_path)

    run_judges_recheck_gender(Namespace(roots=[str(root)]))

    _, flipped = _gender_check(root, "sim_greeting_only")
    assert flipped.outcome == JudgeOutcome.NO_OPPORTUNITY
    assert (root / "gender_recheck_snapshot.json").exists()
