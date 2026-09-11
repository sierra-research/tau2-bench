# Copyright Sierra
"""Frozen τ-Multilingual naturalness artifacts and replay machinery."""

import hashlib
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pytest

from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeUnitResult,
)
from tau2.data_model.voice import SpeechEnvironment
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EXPERIENCE_PATH,
    CANONICAL_VALIDATION_PATH,
    CohortContract,
    TrialRunConfig,
    _aggregate_simulation,
    _aggregate_trial,
    _code_provenance,
    _outcome_summary,
    _work_fingerprint,
    prepare_trial0_naturalness,
    prepare_trial_plan,
    run_trial0_naturalness,
)
from tau2.judges.nativeness.validation import (
    VALIDATION_FACTOR_ID,
    runtime_combined_factor,
)
from tau2.runner.run_lock import claim_run_directories
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results

REPO = Path(__file__).resolve().parents[2]
VALIDATION_ARCHIVE = REPO / CANONICAL_VALIDATION_PATH


def _passing_check(language, turn):
    factor = runtime_combined_factor(language)
    return NativenessFactorCheck(
        id=VALIDATION_FACTOR_ID,
        category=factor.category,
        severity=factor.severity,
        outcome=JudgeOutcome.PASS,
        shadow=factor.shadow,
        evaluation_level="utterance",
        observed_severity=0,
        violation_count=0,
        unit_results=[
            NativenessJudgeUnitResult(
                unit_index=turn.index,
                opportunity=True,
                violated=False,
                severity=0,
                reasoning="fixture pass",
                quote="",
            )
        ],
    )


def _write_mini_cohort(repo: Path) -> tuple[CohortContract, list[Path]]:
    contract = CohortContract(
        languages=("en", "hi"),
        domains=("airline",),
        systems=("Fake",),
        system_slugs=("fake",),
        trial=0,
        calls_per_root=1,
    )
    main_runs = repo / "data/simulations/paper_runs/tau-multi/main_runs"
    source_paths: list[Path] = []
    for language in contract.languages:
        sim = hi_sim(
            f"{language}-sim",
            "t1",
            checks=hi_factor_checks(),
            language="hi",
        )
        if language == "hi":
            sim = sim.model_copy(
                update={"speech_environment": SpeechEnvironment(language="hi")}
            )
        wrapper = main_runs / f"fixture_{language}"
        run = make_hi_results(
            wrapper,
            [sim],
            name=f"{language}_airline_fake",
        )
        source_paths.append(run / "results.json")

    sources = [
        {
            "path": path.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trial_0_calls": 1,
        }
        for path in source_paths
    ]
    fingerprint_payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['trial_0_calls']}" for row in sources
    )
    experience = {
        "instrument": "tau-multilingual-utterance-experience",
        "instrument_version": "fixture-v1",
        "cohort": {
            "calls": 2,
            "calls_per_system": 2,
            "languages": ["en", "hi"],
            "domains": ["airline"],
            "systems": ["Fake"],
            "trial": 0,
        },
        "provenance": {
            "cohort_fingerprint_sha256": hashlib.sha256(
                fingerprint_payload.encode()
            ).hexdigest(),
            "results_files": sources,
        },
    }
    manifest = repo / CANONICAL_EXPERIENCE_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(experience, indent=2))
    return contract, source_paths


def _refresh_fixture_experience(repo: Path) -> None:
    path = repo / CANONICAL_EXPERIENCE_PATH
    experience = json.loads(path.read_text())
    sources = experience["provenance"]["results_files"]
    for source in sources:
        source["sha256"] = hashlib.sha256(
            (repo / source["path"]).read_bytes()
        ).hexdigest()
    payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['trial_0_calls']}" for row in sources
    )
    experience["provenance"]["cohort_fingerprint_sha256"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()
    path.write_text(json.dumps(experience, indent=2))


def test_canonical_mode_pins_manifest_and_transcript_bytes(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial

    contract, sources = _write_mini_cohort(tmp_path)
    baseline = prepare_trial_plan(tmp_path, contract=contract)
    shutil.copytree(
        VALIDATION_ARCHIVE.parent,
        (tmp_path / CANONICAL_VALIDATION_PATH).parent,
    )
    experience_path = tmp_path / CANONICAL_EXPERIENCE_PATH
    monkeypatch.setattr(paper_trial, "CANONICAL_COHORT_CONTRACT", contract)
    monkeypatch.setattr(
        paper_trial,
        "CANONICAL_EXPERIENCE_SHA256",
        hashlib.sha256(experience_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        paper_trial, "CANONICAL_EXPERIENCE_INSTRUMENT_VERSION", "fixture-v1"
    )
    monkeypatch.setattr(
        paper_trial,
        "CANONICAL_WORK_FINGERPRINT_SHA256",
        _work_fingerprint(baseline),
    )
    monkeypatch.setattr(paper_trial, "CANONICAL_UTTERANCES", len(baseline.utterances))
    monkeypatch.setattr(
        paper_trial,
        "CANONICAL_ZERO_UTTERANCE_SIMULATIONS",
        sum(not row.turns for row in baseline.simulations),
    )
    canonical_plan = prepare_trial_plan(tmp_path, contract=contract)
    assert canonical_plan.utterances
    assert canonical_plan.validation_evidence is not None
    assert (
        canonical_plan.validation_evidence.path == CANONICAL_VALIDATION_PATH.as_posix()
    )
    assert (
        "utterance_level/naturalness/hi/manifest.json"
        in canonical_plan.validation_evidence.files
    )
    assert "prompts.json" in canonical_plan.validation_evidence.files
    assert not any(
        path.startswith("provenance/")
        for path in canonical_plan.validation_evidence.files
    )

    original_experience = experience_path.read_bytes()
    experience_path.write_bytes(original_experience + b"\n")
    with pytest.raises(ValueError, match="Experience artifact hash drifted"):
        prepare_trial_plan(tmp_path, contract=contract)
    experience_path.write_bytes(original_experience)

    hi_results = sources[1]
    sim_path = hi_results.parent / "simulations/hi-sim.json"
    simulation = json.loads(sim_path.read_text())
    assistant = next(
        message for message in simulation["messages"] if message["role"] == "assistant"
    )
    assistant["content"] += " बदला हुआ"
    sim_path.write_text(json.dumps(simulation, ensure_ascii=False, indent=2) + "\n")
    with pytest.raises(ValueError, match="transcript work fingerprint drifted"):
        prepare_trial_plan(tmp_path, contract=contract)


def test_trial0_replay_uses_only_manifest_non_english_and_resumes(
    tmp_path, monkeypatch
):
    from tau2.judges.nativeness import paper_trial

    contract, sources = _write_mini_cohort(tmp_path)
    source_bytes = {
        path: path.read_bytes()
        for path in (
            tmp_path / "data/simulations/paper_runs/tau-multi/main_runs"
        ).rglob("*.json")
    }
    calls = {"count": 0}

    def judge(language, turn):
        calls["count"] += 1
        assert language == "hi"
        return _passing_check(language, turn)

    monkeypatch.setattr(paper_trial, "_judge_turn", judge)
    out = tmp_path / "validation-output"
    preparation = prepare_trial0_naturalness(tmp_path, contract=contract)
    assert preparation.counts.verified_roots == 2
    assert preparation.counts.selected_roots == 1
    assert preparation.counts.simulations == 1
    assert preparation.counts.utterances == 1
    assert preparation.simulations_by_language_system == {"hi": {"fake": 1}}
    first = run_trial0_naturalness(
        TrialRunConfig(repo_root=tmp_path, output_root=out, max_concurrency=4),
        contract=contract,
    )
    assert first.status == "complete"
    assert first.counts.verified_roots == 2
    assert first.counts.selected_roots == 1
    assert first.counts.simulations == 1
    assert first.counts.utterances == 1
    assert first.aggregate is not None
    assert first.aggregate.overall.pass_calls == 1
    assert first.aggregate.by_language_system["hi"]["fake"].pass_rate == 1
    assert first.identity.selected_sources[0].system_slug == "fake"
    assert calls["count"] == 1
    assert len(list((out / "utterances").rglob("*.json"))) == 1
    assert len(list((out / "simulations").rglob("*.json"))) == 1
    assert all(path.read_bytes() == content for path, content in source_bytes.items())

    def must_not_run(_language, _turn):
        raise AssertionError("matching trial artifact should be reused")

    monkeypatch.setattr(paper_trial, "_judge_turn", must_not_run)
    original_atomic_write = paper_trial._atomic_write
    rewritten_simulations = []

    def track_atomic_write(path, model):
        if "simulations" in path.parts:
            rewritten_simulations.append(path)
        original_atomic_write(path, model)

    monkeypatch.setattr(paper_trial, "_atomic_write", track_atomic_write)
    second = run_trial0_naturalness(
        TrialRunConfig(repo_root=tmp_path, output_root=out, max_concurrency=1),
        contract=contract,
    )
    assert second.invocations[-1].judged_utterances == 0
    assert second.invocations[-1].reused_utterances == 1
    assert len(second.invocations) == 2
    assert rewritten_simulations == []

    # The 90-root manifest equivalent is authoritative: a changed header fails
    # before any source simulation is judged or an output is mixed in place.
    sources[0].write_text(sources[0].read_text() + "\n")
    with pytest.raises(ValueError, match="frozen results hash mismatch"):
        run_trial0_naturalness(
            TrialRunConfig(
                repo_root=tmp_path,
                output_root=tmp_path / "different-output",
                max_concurrency=1,
            ),
            contract=contract,
        )


def test_trial0_preflight_reads_detached_evidence_bundle(tmp_path):
    contract, _sources = _write_mini_cohort(tmp_path)
    canonical_root = tmp_path / "data/simulations/paper_runs/tau-multi"
    evidence_root = tmp_path / "detached-tau-multi"
    evidence_root.mkdir()
    shutil.move(canonical_root / "main_runs", evidence_root / "main_runs")

    plan = prepare_trial_plan(
        tmp_path,
        contract=contract,
        evidence_root=evidence_root,
    )

    assert len(plan.all_sources) == 2
    assert len(plan.selected_sources) == 1
    assert plan.selected_sources[0].language == "hi"


def test_trial0_replay_rejects_concurrent_writer(tmp_path, monkeypatch):
    from tau2.judges.nativeness import paper_trial

    contract, _sources = _write_mini_cohort(tmp_path)
    out = tmp_path / "locked-output"
    monkeypatch.setattr(
        paper_trial,
        "_judge_turn",
        lambda *_args: (_ for _ in ()).throw(AssertionError("judge must not run")),
    )
    with claim_run_directories([out]):
        with pytest.raises(RuntimeError, match="already being written"):
            run_trial0_naturalness(
                TrialRunConfig(repo_root=tmp_path, output_root=out, max_concurrency=1),
                contract=contract,
            )


def test_trial0_source_simulation_paths_cannot_escape(tmp_path):
    contract, sources = _write_mini_cohort(tmp_path)
    results_path = sources[1]
    results = json.loads(results_path.read_text())
    results["simulation_index"][0]["id"] = "../outside"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    _refresh_fixture_experience(tmp_path)
    with pytest.raises(ValueError, match="unsafe simulation id"):
        prepare_trial_plan(tmp_path, contract=contract)


def test_trial0_source_simulation_symlink_cannot_escape(tmp_path):
    contract, sources = _write_mini_cohort(tmp_path)
    sim_path = sources[1].parent / "simulations/hi-sim.json"
    outside = tmp_path / "outside.json"
    sim_path.rename(outside)
    sim_path.symlink_to(outside)
    with pytest.raises(ValueError, match="simulation path escapes"):
        prepare_trial_plan(tmp_path, contract=contract)


def test_fourteen_zero_utterance_calls_roll_up_without_judging(tmp_path):
    contract, _sources = _write_mini_cohort(tmp_path)
    source_plan = prepare_trial_plan(tmp_path, contract=contract).simulations[0]
    plans = [
        source_plan.model_copy(update={"simulation_id": f"zero-{index}", "turns": []})
        for index in range(14)
    ]
    artifacts = [_aggregate_simulation(plan, []) for plan in plans]
    assert len(artifacts) == 14
    assert all(row.created_at is None for row in artifacts)
    assert all(row.check.outcome is JudgeOutcome.NO_OPPORTUNITY for row in artifacts)
    aggregate = _aggregate_trial(artifacts)
    assert aggregate.overall.calls == 14
    assert aggregate.overall.no_opportunity_calls == 14
    assert aggregate.overall.no_opportunity_rate == 1


def test_paper_rates_exclude_no_opportunity_and_error_from_denominator(tmp_path):
    contract, _sources = _write_mini_cohort(tmp_path)
    plan = (
        prepare_trial_plan(tmp_path, contract=contract)
        .simulations[0]
        .model_copy(update={"turns": []})
    )
    base = _aggregate_simulation(plan, [])
    outcomes = [
        JudgeOutcome.PASS,
        JudgeOutcome.PASS,
        JudgeOutcome.FAIL,
        JudgeOutcome.NO_OPPORTUNITY,
        JudgeOutcome.ERROR,
    ]
    artifacts = [
        base.model_copy(
            update={"check": base.check.model_copy(update={"outcome": outcome})}
        )
        for outcome in outcomes
    ]
    summary = _outcome_summary(artifacts)
    assert summary.calls == 5
    assert summary.scored_calls == 3
    assert summary.pass_rate == pytest.approx(2 / 3)
    assert summary.fail_rate == pytest.approx(1 / 3)
    assert summary.coverage_rate == pytest.approx(3 / 5)
    assert summary.no_opportunity_rate == pytest.approx(1 / 5)
    assert summary.error_rate == pytest.approx(1 / 5)


def test_code_provenance_is_independent_of_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provenance = _code_provenance()
    assert Path(provenance.code_root) == REPO
    assert provenance.git_commit != "unknown"
    assert provenance.source_files


def test_tau_multi_naturalness_cli_has_prepare_and_fixed_defaults():
    from tau2.config import DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY
    from tau2.judges.cli import add_judges_args

    parser = ArgumentParser()
    add_judges_args(parser)
    prepare = parser.parse_args(["tau-multi-naturalness", "prepare"])
    assert prepare.func.__name__ == "prepare_tau_multi_naturalness_trial"
    run = parser.parse_args(
        ["tau-multi-naturalness", "run", "--out", "/tmp/paper-output"]
    )
    assert run.max_concurrency == DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY
