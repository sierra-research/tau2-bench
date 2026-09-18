# Copyright Sierra
"""Safe adapter for replaying the frozen generic paper judge suite."""

from __future__ import annotations

import json
import subprocess
import wave
from argparse import ArgumentParser
from pathlib import Path

import pytest

import tau2.judges.corrected_paper_suite as corrected_suite
from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import (
    DeliveryFinding,
    DeliveryInfo,
    DeliveryUtteranceResult,
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessInfo,
    QualityFactorCheck,
    QualityInfo,
    Results,
)
from tau2.data_model.voice import SpeechEnvironment
from tau2.judges.cli import add_judges_args
from tau2.judges.corrected_paper_suite import (
    CANONICAL_FINAL_WINDOW_CSV_SHA256,
    CANONICAL_FINAL_WINDOW_MANIFEST_SHA256,
    CANONICAL_FINAL_WINDOW_ROWS,
    CANONICAL_GENERIC_COHORT,
    COMPOSE_REPORT_FILENAME,
    EXCLUDED_QUALITY_FACTOR_IDS,
    FROZEN_GENERIC_SUITE_COMMIT,
    PAPER_MODAL_FACTOR_IDS,
    PAPER_QUALITY_FACTOR_IDS,
    FrozenWorktreeIdentity,
    GenericJudgeCell,
    GenericJudgeCohortContract,
    build_composed_final_window_exclusions,
    build_final_window_exclusions,
    compose_cross_workspace_judgments,
    merge_frozen_judgments,
    nonjudge_sha256,
    prepare_frozen_judge_workspace,
    probe_frozen_schema_roundtrip,
    suite_execution_plan,
)
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results


def _voice_sim(sim_id: str, task_id: str, *, trial: int):
    sim = hi_sim(
        sim_id,
        task_id,
        checks=hi_factor_checks(),
        language="hi",
        trial=trial,
    )
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage(
                role="assistant",
                content="namaste",
                audio_script_gold="namaste",
                utterance_ids=["u0"],
            ),
        )
    ]
    sim.speech_environment = SpeechEnvironment(language="hi")
    return sim


def _write_stereo(path: Path, *, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0\0\0" * int(8_000 * seconds))


def _source_cell(root: Path, relative: str, *, prefix: str) -> Path:
    trial0 = _voice_sim(f"{prefix}-t0", f"{prefix}-task", trial=0)
    trial1 = _voice_sim(f"{prefix}-t1", f"{prefix}-task", trial=1)
    cell = make_hi_results(
        root / Path(relative).parent,
        [trial0, trial1],
        name=Path(relative).name,
        num_trials=2,
    )
    for sim in (trial0, trial1):
        _write_stereo(
            cell
            / "artifacts"
            / f"task_{sim.task_id}"
            / f"sim_{sim.id}"
            / "audio"
            / "both.wav"
        )
    return cell


def _fixture_contract() -> GenericJudgeCohortContract:
    return GenericJudgeCohortContract(
        trial=0,
        calls_per_cell=1,
        untouched_trial=1,
        untouched_calls=2,
        cells=(
            GenericJudgeCell(
                language="hi",
                system_slug="s1",
                source_relative_path="run_a/cell_a",
            ),
            GenericJudgeCell(
                language="hi",
                system_slug="s2",
                source_relative_path="run_b/cell_b",
            ),
        ),
    )


def _write_fixture(tmp_path: Path):
    source = tmp_path / "source"
    _source_cell(source, "run_a/cell_a", prefix="a")
    _source_cell(source, "run_b/cell_b", prefix="b")
    workspace = tmp_path / "workspace"
    contract = _fixture_contract()
    manifest = prepare_frozen_judge_workspace(
        source_root=source,
        workspace=workspace,
        contract=contract,
    )
    return source, workspace, contract, manifest


def _judged_fields() -> tuple[dict, dict, dict]:
    nativeness = NativenessInfo(
        score=1.0,
        factor_checks=[],
        language="hi",
        script="deva",
        rubric_version="nativeness-rubric-v19",
        judge_model="gpt-5.5",
        judge_args={"reasoning_effort": "medium"},
        judge_prompt_version="v15",
    )
    quality_args = {
        "unnecessary_repetition": {"reasoning_effort": "none"},
        "agent_caused_tool_error": {"reasoning_effort": "high"},
        "auth_arg_mismatch": {"reasoning_effort": "high"},
        "incorrect_tool_parameters": {"reasoning_effort": "high"},
        "unnecessary_tool_call": {"reasoning_effort": "xhigh"},
    }
    checks = [
        QualityFactorCheck(
            id=factor_id,
            category="test",
            severity=1,
            evaluator="llm",
            outcome=JudgeOutcome.PASS,
            judge_model="gpt-5.5",
            judge_args=args,
            judge_prompt_version="quality-judge-v7",
        )
        for factor_id, args in quality_args.items()
    ]
    for factor_id in (
        "responsiveness",
        "yielding",
        "inappropriate_interruption",
        "backchannel_selectivity",
        "vocal_tic_selectivity",
        "non_directed_selectivity",
        "monologue",
    ):
        checks.append(
            QualityFactorCheck(
                id=factor_id,
                category="test",
                severity=1,
                evaluator="deterministic",
                outcome=JudgeOutcome.PASS,
            )
        )
    quality = QualityInfo(
        score=1.0,
        factor_checks=checks,
        rubric_version="quality-rubric-v4",
        metrics_version="quality-metrics-v5",
        num_pass=len(checks),
        score_coverage=1.0,
    )
    delivery = DeliveryInfo(
        score=0.0,
        fidelity_score=0.0,
        intonation_score=1.0,
        num_judged=1,
        num_flagged=1,
        utterance_results=[
            DeliveryUtteranceResult(
                utterance_idx=0,
                expected_text="namaste",
                outcome=JudgeOutcome.FAIL,
                flag_for_review=True,
                severity=2,
                findings=[
                    DeliveryFinding(
                        axis="fidelity",
                        category="clipped_or_garbled",
                        time_range="0.6-1.0",
                        issue="clipped",
                        severity=2,
                    )
                ],
            )
        ],
        language="hi",
        judge_model="gemini/gemini-3.1-pro-preview",
        judge_args={"temperature": 0.0, "max_tokens": 8192, "timeout": 120},
        judge_prompt_version="v5",
        sample_rate=1.0,
        max_segments=100,
    )
    return (
        nativeness.model_dump(mode="json"),
        quality.model_dump(mode="json"),
        delivery.model_dump(mode="json"),
    )


def _plant_frozen_judgments(workspace: Path) -> None:
    nativeness, quality, delivery = _judged_fields()
    for sim_path in (workspace / "legacy").glob("*/*/simulations/*.json"):
        payload = json.loads(sim_path.read_text())
        payload["nativeness_info"] = nativeness
        payload["quality_info"] = quality
        payload["delivery_info"] = delivery
        # Frozen-schema output is never trusted for non-judge fields.
        payload["duration"] = 999.0
        payload["ticks"][0]["tick_id"] = 999
        sim_path.write_text(json.dumps(payload, indent=2))


def _compose_contract() -> GenericJudgeCohortContract:
    return GenericJudgeCohortContract(
        trial=0,
        calls_per_cell=1,
        untouched_trial=1,
        untouched_calls=2,
        cells=(
            GenericJudgeCell(
                language="ko",
                system_slug="s1",
                source_relative_path="run_ko/cell_ko",
            ),
            GenericJudgeCell(
                language="zh",
                system_slug="s2",
                source_relative_path="run_zh/cell_zh",
            ),
        ),
    )


def _source_cell_for_language(
    root: Path, relative: str, *, prefix: str, language: str
) -> Path:
    simulations = []
    for trial in (0, 1):
        sim = hi_sim(
            f"{prefix}-t{trial}",
            f"{prefix}-task",
            checks=hi_factor_checks(),
            language=language,
            trial=trial,
        )
        sim.ticks = [
            Tick(
                tick_id=0,
                timestamp="2026-01-01T00:00:00",
                tick_duration_seconds=1.0,
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="hello",
                    audio_script_gold="hello",
                    utterance_ids=["u0"],
                ),
            )
        ]
        sim.speech_environment = SpeechEnvironment(language=language)
        simulations.append(sim)
    cell = make_hi_results(
        root / Path(relative).parent,
        simulations,
        name=Path(relative).name,
        num_trials=2,
    )
    for sim in simulations:
        _write_stereo(
            cell
            / "artifacts"
            / f"task_{sim.task_id}"
            / f"sim_{sim.id}"
            / "audio"
            / "both.wav"
        )
    return cell


def _selected_quality_payload() -> dict:
    _nativeness, payload, _delivery = _judged_fields()
    payload["factor_checks"] = [
        check
        for check in payload["factor_checks"]
        if check["id"] in PAPER_QUALITY_FACTOR_IDS
    ]
    for check in payload["factor_checks"]:
        if check["id"] in {"agent_caused_tool_error", "auth_arg_mismatch"}:
            check["evaluator"] = "hybrid"
    payload.update(
        score=1.0,
        num_pass=len(payload["factor_checks"]),
        num_fail=0,
        num_no_opportunity=0,
        num_deferred=0,
        num_errors=0,
        score_coverage=1.0,
    )
    return payload


def _modal_payload() -> dict:
    checks = [
        NativenessFactorCheck(
            id=factor_id,
            category="test",
            severity=3 if factor_id in {"register_formality", "modal_particles"} else 2,
            outcome=(
                JudgeOutcome.FAIL
                if factor_id == "modal_particles"
                else JudgeOutcome.PASS
            ),
            evaluation_level=(
                "call" if factor_id == "modal_particles" else "deterministic"
            ),
            observed_severity=2 if factor_id == "modal_particles" else 0,
            violation_count=1 if factor_id == "modal_particles" else 0,
        )
        for factor_id in sorted(PAPER_MODAL_FACTOR_IDS)
    ]
    return NativenessInfo(
        score=0.75,
        factor_checks=checks,
        language="zh",
        script="hans",
        rubric_version="nativeness-rubric-v19",
        num_pass=3,
        num_fail=1,
        score_coverage=1.0,
        judge_model="gpt-5.5",
        judge_args={"reasoning_effort": "medium"},
        judge_prompt_version="v15",
    ).model_dump(mode="json")


def _write_compose_fixture(tmp_path: Path):
    source = tmp_path / "compose-source"
    _source_cell_for_language(source, "run_ko/cell_ko", prefix="ko", language="ko")
    _source_cell_for_language(source, "run_zh/cell_zh", prefix="zh", language="zh")
    contract = _compose_contract()
    quality = tmp_path / "quality"
    delivery = tmp_path / "delivery"
    modal = tmp_path / "modal"
    quality_manifest = prepare_frozen_judge_workspace(
        source_root=source, workspace=quality, contract=contract
    )
    prepare_frozen_judge_workspace(
        source_root=source, workspace=delivery, contract=contract
    )
    prepare_frozen_judge_workspace(
        source_root=source, workspace=modal, contract=contract
    )

    quality_info = _selected_quality_payload()
    _nativeness, _quality, delivery_info = _judged_fields()
    modal_info = _modal_payload()
    for cell in quality_manifest.cells:
        [call] = cell.calls
        relative = Path(cell.source_relative_path) / call.simulation_path
        quality_path = quality / "legacy" / relative
        quality_payload = json.loads(quality_path.read_text())
        quality_payload["quality_info"] = quality_info
        quality_payload["nativeness_info"] = {"trap": "do not import"}
        quality_path.write_text(json.dumps(quality_payload, indent=2))

        delivery_path = delivery / "legacy" / relative
        delivery_payload = json.loads(delivery_path.read_text())
        cell_delivery = dict(delivery_info)
        cell_delivery["language"] = cell.language
        delivery_payload["delivery_info"] = cell_delivery
        delivery_payload["quality_info"] = {"trap": "do not import"}
        delivery_path.write_text(json.dumps(delivery_payload, indent=2))

        modal_path = modal / "legacy" / relative
        modal_sim = json.loads(modal_path.read_text())
        modal_sim["quality_info"] = {"trap": "do not import"}
        modal_sim["nativeness_info"] = (
            modal_info if cell.language == "zh" else {"trap": "do not import"}
        )
        modal_path.write_text(json.dumps(modal_sim, indent=2))

    # Reproduce the real production hazard: this clone retains the quality
    # workspace declaration. The compose operation must use the explicit modal
    # root and never follow this stale value.
    (modal / "manifest.json").write_bytes((quality / "manifest.json").read_bytes())
    return source, quality, delivery, modal, quality_manifest


def test_prepare_materializes_only_trial_zero_and_locks_source(tmp_path):
    source, workspace, _contract, manifest = _write_fixture(tmp_path)

    assert manifest.counts.cells == 2
    assert manifest.counts.calls == 2
    assert manifest.counts.stereo_audio_files == 2
    assert manifest.counts.infrastructure_errors == 0
    assert manifest.counts.duplicates == 0
    assert len(list((workspace / "legacy").glob("*/*/simulations/*.json"))) == 2
    assert len(list(source.glob("*/*/simulations/*.json"))) == 4

    locked = manifest.cells[0].calls[0]
    source_sim = (
        source / manifest.cells[0].source_relative_path / locked.simulation_path
    )
    staged_sim = (
        workspace
        / "legacy"
        / manifest.cells[0].source_relative_path
        / locked.simulation_path
    )
    assert source_sim.stat().st_ino != staged_sim.stat().st_ino
    assert nonjudge_sha256(json.loads(source_sim.read_text())) == locked.nonjudge_sha256

    # Idempotent preflight verifies the existing copy instead of rewriting it.
    assert (
        prepare_frozen_judge_workspace(
            source_root=source,
            workspace=workspace,
            contract=_contract,
        )
        == manifest
    )


def test_prepare_rejects_duplicate_task_trial_pairs(tmp_path):
    source = tmp_path / "source"
    cell = _source_cell(source, "run_a/cell_a", prefix="a")
    payload = json.loads((cell / "results.json").read_text())
    payload["simulation_index"].append(dict(payload["simulation_index"][0]))
    (cell / "results.json").write_text(json.dumps(payload, indent=2))
    contract = GenericJudgeCohortContract(
        trial=0,
        calls_per_cell=1,
        untouched_trial=1,
        untouched_calls=1,
        cells=(
            GenericJudgeCell(
                language="hi",
                system_slug="s1",
                source_relative_path="run_a/cell_a",
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepare_frozen_judge_workspace(
            source_root=source,
            workspace=tmp_path / "workspace",
            contract=contract,
        )


def test_prepare_requires_exact_untouched_trial_inventory(tmp_path):
    source = tmp_path / "source"
    _source_cell(source, "run_a/cell_a", prefix="a")
    contract = GenericJudgeCohortContract(
        trial=0,
        calls_per_cell=1,
        untouched_trial=1,
        untouched_calls=2,
        cells=(
            GenericJudgeCell(
                language="hi",
                system_slug="s1",
                source_relative_path="run_a/cell_a",
            ),
        ),
    )

    with pytest.raises(ValueError, match="untouched trial-1 calls; expected 2"):
        prepare_frozen_judge_workspace(
            source_root=source,
            workspace=tmp_path / "workspace",
            contract=contract,
        )


def test_execution_plan_pins_commit_and_exact_suite_flags(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    plan = suite_execution_plan(
        manifest=manifest,
        frozen_worktree=tmp_path / "frozen",
        python_executable=Path("/venv/bin/python"),
    )

    assert plan.frozen_commit == FROZEN_GENERIC_SUITE_COMMIT
    assert plan.command[:4] == [
        "/venv/bin/python",
        "-m",
        "tau2.cli",
        "judges",
    ]
    assert plan.command[4:6] == ["suite", str(workspace / "legacy/run_a/cell_a")]
    assert "--rejudge" not in plan.command
    assert plan.command[-8:] == [
        "--concurrency",
        "10",
        "--text-processes",
        "8",
        "--audio-processes",
        "5",
        "--trials",
        "0",
    ]


def test_frozen_schema_probe_roundtrips_all_staged_evidence_without_audio(
    tmp_path, monkeypatch
):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        corrected_suite,
        "verify_frozen_worktree",
        lambda path: FrozenWorktreeIdentity(
            path=str(path.resolve()),
            commit=FROZEN_GENERIC_SUITE_COMMIT,
            clean=True,
        ),
    )

    def fake_run(command, **kwargs):
        observed.setdefault("commands", []).append(command)
        root = (
            Path(command[4]).parent
            if command[4].endswith(".json")
            else Path(command[4])
        )
        assert not list(root.rglob("*.wav"))
        if command[6] == "json":
            for path in (root / "simulations").glob("*.json"):
                payload = json.loads(path.read_text())
                payload.pop("duration", None)
                payload.pop(corrected_suite.ROUNDTRIP_PROBE_FIELD)
                path.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(corrected_suite.subprocess, "run", fake_run)

    report = probe_frozen_schema_roundtrip(
        manifest=manifest,
        frozen_worktree=frozen,
        python_executable=Path("/venv/bin/python"),
    )

    assert report.calls == 2
    assert report.cells == 2
    assert report.judge_evidence_mismatches == 0
    assert report.external_api_calls is False
    assert len(observed["commands"]) == 4
    assert observed["commands"][0] == [
        "/venv/bin/python",
        "-m",
        "tau2.cli",
        "convert-results",
        str(report.commands[0][4]),
        "--to",
        "json",
        "--no-backup",
    ]
    assert report.commands == observed["commands"]


def test_frozen_schema_probe_rejects_evidence_lost_by_legacy_serializer(
    tmp_path, monkeypatch
):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    monkeypatch.setattr(
        corrected_suite,
        "verify_frozen_worktree",
        lambda path: FrozenWorktreeIdentity(
            path=str(path.resolve()),
            commit=FROZEN_GENERIC_SUITE_COMMIT,
            clean=True,
        ),
    )

    def fake_run(command, **kwargs):
        root = (
            Path(command[4]).parent
            if command[4].endswith(".json")
            else Path(command[4])
        )
        if command[6] == "json":
            sims = list((root / "simulations").glob("*.json"))
            for path in sims:
                payload = json.loads(path.read_text())
                payload.pop(corrected_suite.ROUNDTRIP_PROBE_FIELD)
                path.write_text(json.dumps(payload))
            if not getattr(fake_run, "mutated", False):
                payload = json.loads(sims[0].read_text())
                payload["ticks"] = []
                sims[0].write_text(json.dumps(payload))
                fake_run.mutated = True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(corrected_suite.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="round-trip judge evidence drifted"):
        probe_frozen_schema_roundtrip(
            manifest=manifest,
            frozen_worktree=frozen,
            python_executable=Path("/venv/bin/python"),
        )


def test_resume_rejects_drift_in_a_pristine_staged_call(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    sim_path = next((workspace / "legacy").glob("*/*/simulations/*.json"))
    payload = json.loads(sim_path.read_text())
    payload["ticks"][0]["tick_id"] = 999
    sim_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="pristine judge evidence drifted"):
        corrected_suite._verify_runnable_legacy(manifest)


def test_resume_accepts_exact_partial_frozen_output_despite_legacy_rewrite(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    nativeness, _quality, _delivery = _judged_fields()
    sim_path = next((workspace / "legacy").glob("*/*/simulations/*.json"))
    payload = json.loads(sim_path.read_text())
    payload["nativeness_info"] = nativeness
    payload["ticks"][0]["tick_id"] = 999
    sim_path.write_text(json.dumps(payload))

    corrected_suite._verify_runnable_legacy(manifest)


def test_roundtrip_probe_compares_a_valid_partial_output_to_its_staged_evidence(
    tmp_path, monkeypatch
):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    nativeness, _quality, _delivery = _judged_fields()
    sim_path = next((workspace / "legacy").glob("*/*/simulations/*.json"))
    payload = json.loads(sim_path.read_text())
    payload["nativeness_info"] = nativeness
    payload["ticks"][0]["tick_id"] = 999
    sim_path.write_text(json.dumps(payload))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    monkeypatch.setattr(
        corrected_suite,
        "verify_frozen_worktree",
        lambda path: FrozenWorktreeIdentity(
            path=str(path.resolve()),
            commit=FROZEN_GENERIC_SUITE_COMMIT,
            clean=True,
        ),
    )

    def fake_run(command, **kwargs):
        root = (
            Path(command[4]).parent
            if command[4].endswith(".json")
            else Path(command[4])
        )
        if command[6] == "json":
            for path in (root / "simulations").glob("*.json"):
                probe_payload = json.loads(path.read_text())
                probe_payload.pop(corrected_suite.ROUNDTRIP_PROBE_FIELD)
                path.write_text(json.dumps(probe_payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(corrected_suite.subprocess, "run", fake_run)

    report = probe_frozen_schema_roundtrip(
        manifest=manifest,
        frozen_worktree=frozen,
        python_executable=Path("/venv/bin/python"),
    )

    assert report.calls == 2
    assert report.judge_evidence_mismatches == 0


def test_resume_rejects_partial_output_with_wrong_frozen_contract(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    nativeness, _quality, _delivery = _judged_fields()
    nativeness["judge_model"] = "wrong-model"
    sim_path = next((workspace / "legacy").glob("*/*/simulations/*.json"))
    payload = json.loads(sim_path.read_text())
    payload["nativeness_info"] = nativeness
    payload["ticks"][0]["tick_id"] = 999
    sim_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="wrong frozen nativeness contract"):
        corrected_suite._verify_runnable_legacy(manifest)


def test_cli_keeps_prepare_run_merge_and_filter_explicit():
    parser = ArgumentParser()
    add_judges_args(parser)

    prepare = parser.parse_args(
        [
            "corrected-paper-suite",
            "prepare",
            "--source-root",
            "/tmp/source",
            "--workspace",
            "/tmp/workspace",
            "--dry-run",
        ]
    )
    assert prepare.func.__name__ == "prepare_corrected_paper_suite"
    assert prepare.dry_run is True

    run = parser.parse_args(
        [
            "corrected-paper-suite",
            "run",
            "--workspace",
            "/tmp/workspace",
            "--frozen-worktree",
            "/tmp/frozen",
            "--dry-run",
        ]
    )
    assert run.func.__name__ == "run_corrected_paper_suite"

    merge = parser.parse_args(
        ["corrected-paper-suite", "merge", "--workspace", "/tmp/workspace"]
    )
    assert merge.func.__name__ == "merge_corrected_paper_suite"

    compose = parser.parse_args(
        [
            "corrected-paper-suite",
            "compose",
            "--quality-workspace",
            "/tmp/quality",
            "--delivery-workspace",
            "/tmp/delivery",
            "--modal-workspace",
            "/tmp/modal",
            "--output-workspace",
            "/tmp/combined",
        ]
    )
    assert compose.func.__name__ == "compose_corrected_paper_suite"

    composed_final_window = parser.parse_args(
        [
            "corrected-paper-suite",
            "composed-final-window",
            "--quality-workspace",
            "/tmp/quality",
            "--delivery-workspace",
            "/tmp/delivery",
            "--modal-workspace",
            "/tmp/modal",
            "--composed-workspace",
            "/tmp/combined",
            "--output",
            "/tmp/composed-exclusions.csv",
        ]
    )
    assert composed_final_window.func.__name__ == "build_composed_final_window_sidecar"

    final_window = parser.parse_args(
        [
            "corrected-paper-suite",
            "final-window",
            "--workspace",
            "/tmp/workspace",
            "--output",
            "/tmp/exclusions.csv",
        ]
    )
    assert final_window.func.__name__ == "build_corrected_final_window_sidecar"


def test_merge_imports_only_three_judge_fields_and_refreshes_indexes(tmp_path):
    source, workspace, _contract, manifest = _write_fixture(tmp_path)
    _plant_frozen_judgments(workspace)

    report = merge_frozen_judgments(manifest=manifest)

    assert report.calls == 2
    assert report.nonjudge_hash_mismatches == 0
    assert report.source_hash_mismatches == 0
    for cell in manifest.cells:
        source_path = source / cell.source_relative_path
        merged_path = workspace / "merged" / cell.source_relative_path
        merged_meta = Results.load_metadata(merged_path)
        assert len(merged_meta.simulation_index or []) == 1
        assert merged_meta.simulation_index[0].quality == 1.0
        assert merged_meta.simulation_index[0].nativeness == 1.0
        assert merged_meta.simulation_index[0].fidelity == 0.0
        [call] = cell.calls
        source_payload = json.loads((source_path / call.simulation_path).read_text())
        merged_payload = json.loads((merged_path / call.simulation_path).read_text())
        assert merged_payload["duration"] == source_payload["duration"]
        assert nonjudge_sha256(merged_payload) == nonjudge_sha256(source_payload)


def test_cross_workspace_compose_imports_only_owned_fields_atomically(tmp_path):
    source, quality, delivery, modal, manifest = _write_compose_fixture(tmp_path)
    source_hashes = {
        path: corrected_suite.sha256_file(path) for path in source.rglob("*.json")
    }
    output = tmp_path / "combined"

    report = compose_cross_workspace_judgments(
        quality_workspace=quality,
        delivery_workspace=delivery,
        modal_workspace=modal,
        output_workspace=output,
    )

    assert report.calls == 2
    assert report.imported_quality_calls == 2
    assert report.imported_delivery_calls == 2
    assert report.delivery_error_utterances == 0
    assert report.imported_modal_nativeness_calls == 1
    assert report.preserved_source_nativeness_calls == 1
    assert report.quality_factor_ids == tuple(sorted(PAPER_QUALITY_FACTOR_IDS))
    assert report.excluded_quality_factor_ids == tuple(
        sorted(EXCLUDED_QUALITY_FACTOR_IDS)
    )
    assert report.modal_factor_ids == tuple(sorted(PAPER_MODAL_FACTOR_IDS))
    assert report.natural_word_choice_storage == "separate_sidecar"
    assert report.inputs[2].role == "modal_particles"
    assert report.inputs[2].declared_workspace_matches is False
    assert (output / COMPOSE_REPORT_FILENAME).is_file()
    assert source_hashes == {
        path: corrected_suite.sha256_file(path) for path in source.rglob("*.json")
    }

    for cell in manifest.cells:
        [call] = cell.calls
        source_path = source / cell.source_relative_path / call.simulation_path
        merged_path = (
            output / "merged" / cell.source_relative_path / call.simulation_path
        )
        source_payload = json.loads(source_path.read_text())
        merged_payload = json.loads(merged_path.read_text())
        assert nonjudge_sha256(merged_payload) == nonjudge_sha256(source_payload)
        assert {
            check["id"] for check in merged_payload["quality_info"]["factor_checks"]
        } == PAPER_QUALITY_FACTOR_IDS
        assert merged_payload["delivery_info"]["language"] == cell.language
        if cell.language == "zh":
            assert {
                check["id"]
                for check in merged_payload["nativeness_info"]["factor_checks"]
            } == PAPER_MODAL_FACTOR_IDS
        else:
            assert (
                merged_payload["nativeness_info"] == source_payload["nativeness_info"]
            )

    # A second invocation is a full verification pass, not a rewrite.
    assert (
        compose_cross_workspace_judgments(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            output_workspace=output,
        )
        == report
    )


def test_cross_workspace_compose_retains_delivery_errors_as_unscored(tmp_path):
    _source, quality, delivery, modal, manifest = _write_compose_fixture(tmp_path)
    [cell] = [row for row in manifest.cells if row.language == "ko"]
    [call] = cell.calls
    sim_path = delivery / "legacy" / cell.source_relative_path / call.simulation_path
    payload = json.loads(sim_path.read_text())
    error_row = dict(payload["delivery_info"]["utterance_results"][0])
    error_row.update(
        utterance_idx=1,
        expected_text="annyeong",
        outcome=JudgeOutcome.ERROR.value,
        summary="response validation failed",
        flag_for_review=False,
        severity=0,
        findings=[],
    )
    payload["delivery_info"]["utterance_results"].append(error_row)
    payload["delivery_info"]["num_judged"] = 2
    payload["delivery_info"]["num_errors"] = 1
    sim_path.write_text(json.dumps(payload, indent=2))

    report = compose_cross_workspace_judgments(
        quality_workspace=quality,
        delivery_workspace=delivery,
        modal_workspace=modal,
        output_workspace=tmp_path / "combined-errors",
    )

    assert report.delivery_error_utterances == 1


def test_modal_validator_accepts_only_complete_unstamped_no_call_slice():
    payload = _modal_payload()
    payload.update(
        judge_model=None,
        judge_args=None,
        judge_prompt_version=None,
        score=None,
        score_coverage=0.0,
        num_pass=0,
        num_fail=0,
        num_no_opportunity=4,
    )
    for check in payload["factor_checks"]:
        check.update(
            outcome=JudgeOutcome.NO_OPPORTUNITY.value,
            evidence=None,
            quote=None,
            observed_severity=0,
            violation_count=0,
            unit_results=[],
        )

    corrected_suite._validate_modal_nativeness(payload, "no-call")

    payload["factor_checks"][-1]["outcome"] = JudgeOutcome.PASS.value
    with pytest.raises(ValueError, match="judge stamps"):
        corrected_suite._validate_modal_nativeness(payload, "mixed")


def test_cross_workspace_compose_rejects_delivery_error_counter_drift(tmp_path):
    _source, quality, delivery, modal, manifest = _write_compose_fixture(tmp_path)
    [cell] = [row for row in manifest.cells if row.language == "ko"]
    [call] = cell.calls
    sim_path = delivery / "legacy" / cell.source_relative_path / call.simulation_path
    payload = json.loads(sim_path.read_text())
    payload["delivery_info"]["utterance_results"][0]["outcome"] = (
        JudgeOutcome.ERROR.value
    )
    sim_path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(ValueError, match="delivery error count drifted"):
        compose_cross_workspace_judgments(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            output_workspace=tmp_path / "invalid-errors",
        )


def test_cross_workspace_compose_rejects_excluded_quality_factor_without_output(
    tmp_path,
):
    _source, quality, delivery, modal, manifest = _write_compose_fixture(tmp_path)
    [cell] = [row for row in manifest.cells if row.language == "ko"]
    [call] = cell.calls
    sim_path = quality / "legacy" / cell.source_relative_path / call.simulation_path
    payload = json.loads(sim_path.read_text())
    payload["quality_info"]["factor_checks"].append(
        QualityFactorCheck(
            id="unnecessary_repetition",
            category="repetition",
            severity=1,
            evaluator="llm",
            outcome=JudgeOutcome.PASS,
            judge_model="gpt-5.5",
            judge_args={"reasoning_effort": "none"},
            judge_prompt_version="quality-judge-v7",
        ).model_dump(mode="json")
    )
    payload["quality_info"]["num_pass"] += 1
    sim_path.write_text(json.dumps(payload, indent=2))
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="wrong selected quality contract"):
        compose_cross_workspace_judgments(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            output_workspace=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_cross_workspace_compose_rejects_different_cohort_lock(tmp_path):
    _source, quality, delivery, modal, _manifest = _write_compose_fixture(tmp_path)
    delivery_manifest_path = delivery / "manifest.json"
    delivery_manifest = json.loads(delivery_manifest_path.read_text())
    delivery_manifest["counts"]["unselected_calls"] += 1
    delivery_manifest_path.write_text(json.dumps(delivery_manifest, indent=2))

    with pytest.raises(ValueError, match="does not share the quality cohort lock"):
        compose_cross_workspace_judgments(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            output_workspace=tmp_path / "combined",
        )


def test_cross_workspace_compose_rejects_unrecognized_stale_modal_workspace(
    tmp_path,
):
    _source, quality, delivery, modal, _manifest = _write_compose_fixture(tmp_path)
    modal_manifest_path = modal / "manifest.json"
    modal_manifest = json.loads(modal_manifest_path.read_text())
    modal_manifest["workspace"] = str(tmp_path / "unrelated")
    modal_manifest_path.write_text(json.dumps(modal_manifest, indent=2))

    with pytest.raises(ValueError, match="unrecognized stale workspace"):
        compose_cross_workspace_judgments(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            output_workspace=tmp_path / "combined",
        )


def test_final_window_sidecar_uses_one_second_rule(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    _plant_frozen_judgments(workspace)
    merge_frozen_judgments(manifest=manifest)

    report = build_final_window_exclusions(
        manifest=manifest,
        output=workspace / "final_window_exclusions.csv",
    )

    assert report.replacement_calls == 2
    assert report.excluded_findings == 2
    assert report.excluded_severity_2plus_findings == 2
    lines = (workspace / "final_window_exclusions.csv").read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",") == [
        "sim_id",
        "language",
        "domain",
        "system",
        "utterance_idx",
        "finding_index",
        "severity",
        "time_range",
        "clip_duration_seconds",
        "span_start_seconds",
        "span_end_seconds",
    ]
    sidecar = json.loads((workspace / "final_window_exclusions.json").read_text())
    assert sidecar["filter_version"] == "v1"
    assert sidecar["window_seconds"] == 1.0


def test_composed_final_window_reverifies_owned_fields(tmp_path):
    _source, quality, delivery, modal, _manifest = _write_compose_fixture(tmp_path)
    composed = tmp_path / "combined"
    compose_cross_workspace_judgments(
        quality_workspace=quality,
        delivery_workspace=delivery,
        modal_workspace=modal,
        output_workspace=composed,
    )
    output = tmp_path / "composed-exclusions.csv"

    report = build_composed_final_window_exclusions(
        quality_workspace=quality,
        delivery_workspace=delivery,
        modal_workspace=modal,
        composed_workspace=composed,
        output=output,
    )

    assert report.replacement_calls == 2
    assert report.excluded_findings == 2
    assert output.is_file()
    assert output.with_suffix(".json").is_file()


def test_composed_final_window_requires_existing_composition(tmp_path):
    _source, quality, delivery, modal, _manifest = _write_compose_fixture(tmp_path)

    with pytest.raises(ValueError, match="must already exist"):
        build_composed_final_window_exclusions(
            quality_workspace=quality,
            delivery_workspace=delivery,
            modal_workspace=modal,
            composed_workspace=tmp_path / "missing-composition",
            output=tmp_path / "must-not-exist.csv",
        )

    assert not (tmp_path / "must-not-exist.csv").exists()


def test_final_window_sidecar_replaces_old_corrected_slice(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    _plant_frozen_judgments(workspace)
    merge_frozen_judgments(manifest=manifest)
    canonical = tmp_path / "canonical.csv"
    canonical.write_text(
        "sim_id,language,domain,system,utterance_idx,finding_index,severity,"
        "time_range,clip_duration_seconds,span_start_seconds,span_end_seconds\n"
        "keep,es,airline,OpenAI xhigh,0,0,2,0.5-1.0,1.0,0.5,1.0\n"
        "drop,ko,retail,OpenAI xhigh,0,0,2,0.5-1.0,1.0,0.5,1.0\n"
    )
    canonical.with_suffix(".json").write_text(
        json.dumps(
            {
                "artifact": canonical.name,
                "filter_version": "v1",
                "window_seconds": 1.0,
                "cohort": "old",
                "excluded_findings": 2,
                "excluded_severity_2plus_findings": 2,
                "by_language": {"es": 1, "ko": 1},
                "by_system": {"OpenAI xhigh": 2},
                "results_files_sha256": "0" * 64,
            }
        )
    )

    output = workspace / "hybrid.csv"
    report = build_final_window_exclusions(
        manifest=manifest,
        output=output,
        canonical_sidecar=canonical,
    )

    text = output.read_text()
    assert "keep,es,airline,OpenAI xhigh" in text
    assert "drop,ko,retail" not in text
    assert report.excluded_findings == 3
    assert report.by_language == {"es": 1, "hi": 2}
    assert report.canonical_sidecar_sha256 is not None


def test_paper_canonical_sidecar_identity_is_hard_pinned():
    assert CANONICAL_FINAL_WINDOW_CSV_SHA256 == (
        "94575b3f52af53c7471cb3fa044c5cf05fef694541a7e798294c274c26ace591"
    )
    assert CANONICAL_FINAL_WINDOW_MANIFEST_SHA256 == (
        "f372056904e20990a0038a038501482484ee6bd10535a7f25c1a01d4cdd7179b"
    )
    assert CANONICAL_FINAL_WINDOW_ROWS == 7_436


def test_paper_cohort_rejects_a_different_valid_looking_canonical_sidecar(tmp_path):
    _source, workspace, _contract, manifest = _write_fixture(tmp_path)
    _plant_frozen_judgments(workspace)
    merge_frozen_judgments(manifest=manifest)
    canonical = tmp_path / "canonical.csv"
    canonical.write_text(
        "sim_id,language,domain,system,utterance_idx,finding_index,severity,"
        "time_range,clip_duration_seconds,span_start_seconds,span_end_seconds\n"
    )
    canonical.with_suffix(".json").write_text(
        json.dumps(
            {
                "filter_version": "v1",
                "window_seconds": 1.0,
                "excluded_findings": 0,
            }
        )
    )
    paper_manifest = manifest.model_copy(update={"cohort": CANONICAL_GENERIC_COHORT})

    with pytest.raises(ValueError, match="canonical final-window sidecar identity"):
        build_final_window_exclusions(
            manifest=paper_manifest,
            output=workspace / "hybrid.csv",
            canonical_sidecar=canonical,
        )
