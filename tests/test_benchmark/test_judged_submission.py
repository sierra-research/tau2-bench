"""Voice V2 submissions carry complete, recomputable judge diagnostics.

A synthetic complete matrix with fake saved verdicts; no model is ever called.
"""

import argparse
import json
import shutil

import pytest

from tau2.data_model.judges import (
    MeasureVerdict,
    PlanAction,
    StoredVerdict,
    UnitVerdict,
)
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.simulation import (
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    VoiceRunConfig,
)
from tau2.multilingual.runtime import load_language
from tau2.runner.helpers import get_info
from tau2.runner.presets import resolve_configs

# Package name for the conftest `submitter_file` metadata.
PACKAGE_NAME = "fixture_test_2026-10-02"


def _write_verdicts(cell, catalog):
    """Fake one saved verdict per applicable call/measure, as `judges suite` would."""
    from tau2.judges.run_evidence import progress_path
    from tau2.judges.suite import _save_verdicts, judge_saved_run

    plan = judge_saved_run(cell, catalog).plan
    rows = [
        StoredVerdict(
            identity=item.identity,
            verdict=MeasureVerdict(
                units=[
                    UnitVerdict(
                        unit_id="call",
                        opportunity=True,
                        violation=index % 3 == 0,
                        evidence="fixture",
                    )
                ]
            ),
        )
        for index, item in enumerate(plan.items)
        if item.action == PlanAction.MISSING
    ]
    _save_verdicts(progress_path(cell), rows)


@pytest.fixture
def judged_matrix(published_frame, tmp_path, monkeypatch):
    """A complete 18-cell Voice V2 matrix with audio and a saved verdict per measure."""
    from tau2.judges import suite
    from tau2.judges.validation import production_catalog
    from tau2.runner import presets

    monkeypatch.setattr(presets, "load_frame", lambda *a, **kw: published_frame)
    monkeypatch.setattr(presets, "verify_frame", lambda *a, **kw: None)
    monkeypatch.setattr(presets, "preset_names", lambda: ["voice"])
    monkeypatch.setattr(
        suite, "create_judge", lambda *_: pytest.fail("Submission called a judge")
    )
    template = VoiceRunConfig(
        audio_native_config=AudioNativeConfig(provider="openai"),
        communicate_scoring=CommunicateScoring(
            judge=SemanticCommunicateSettings(model="fixture", model_args={})
        ),
    )
    root = tmp_path / "voice-run"
    plan = resolve_configs(published_frame, template, root)
    root.mkdir()
    (root / "benchmark-frame.json").write_text(published_frame.model_dump_json())
    catalog = production_catalog([])
    for config in plan.configs:
        loaded = load_language(config.language, config.domain, mode="voice")
        info = get_info(config)
        cell = root / f"{config.language}-{config.domain}"
        sims = []
        for index, task in enumerate(loaded.tasks):
            sim = SimulationRun(
                id=f"sim-{index}",
                task_id=task.id,
                trial=0,
                start_time="a",
                end_time="b",
                duration=1,
                termination_reason=TerminationReason.USER_STOP,
                reward_info=RewardInfo(reward=float(index < 25)),
                language_context=info.language_tasks[task.id],
            )
            audio = cell / "tasks" / f"task_{task.id}" / f"sim_{sim.id}" / "audio"
            audio.mkdir(parents=True)
            (audio / "both.wav").write_bytes(f"fixture audio {sim.id}".encode())
            sims.append(sim)
        Results(info=info, tasks=loaded.tasks, simulations=sims).save(
            cell / "results.json", format="dir"
        )
        _write_verdicts(cell, catalog)
    return root


def _submit(*argv):
    from tau2.scripts.leaderboard.cli import add_submit_args

    parser = argparse.ArgumentParser()
    add_submit_args(parser)
    args = parser.parse_args([str(value) for value in argv])
    args.func(args)


def test_prepare_and_validate_recompute_rewards_and_judges(
    judged_matrix, tmp_path, submitter_file
):
    from tau2.judges.submission import BenchmarkDiagnostics
    from tau2.scripts.leaderboard.benchmark_package import load_package

    _submit(
        "prepare",
        judged_matrix,
        "--output",
        tmp_path / "out",
        "--metadata-file",
        submitter_file,
    )
    package = tmp_path / "out" / PACKAGE_NAME
    metadata = load_package(package).metadata
    assert (metadata.agent_provider, metadata.evidence) == ("openai", "matrix")
    diagnostics = BenchmarkDiagnostics.model_validate_json(
        (package / "benchmark-diagnostics.json").read_text()
    )
    assert diagnostics.benchmark_version == "voice-v2"
    assert len(diagnostics.cells) == 18
    hindi = next(
        cell
        for cell in diagnostics.cells
        if (cell.language, cell.domain) == ("hi", "retail")
    )
    measures = {summary.measure: summary for summary in hindi.summary}
    assert measures["naturalness"].judged_calls == 50
    assert measures["speech_fidelity"].judged_calls == 50
    # The catalog validates Hindi gender only for male agents; fixture is unknown.
    assert measures["gender_agreement"].applicable_calls == 0
    assert all(summary.complete for summary in hindi.summary)
    assert {judge.contract_id for judge in diagnostics.judges} >= {
        row.contract_id for row in hindi.verdicts
    }
    submission = json.loads((package / "benchmark-submission.json").read_text())
    assert submission["summary"]["complete"] is True
    quality = load_package(package).quality
    assert len(quality.cells) == 18
    # Every scored call gets a record; tool misuse comes from its tool_use verdict.
    assert all(len(cell.calls) == 50 for cell in quality.cells)
    assert all(cell.interaction_components.tool_misuse.known for cell in quality.cells)

    _submit("validate", package, "--results", judged_matrix)

    # A self-consistent but changed call record differs from the saved calls.
    edited = json.loads((package / "benchmark-quality.json").read_text())
    edited["cells"][0]["calls"][0]["experience_utterances"] += 1
    edited_package = tmp_path / "edited"
    shutil.copytree(package, edited_package)
    (edited_package / "benchmark-quality.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="Call quality differs"):
        _submit("validate", edited_package, "--results", judged_matrix)

    # A forged summary in the package fails its own consistency check.
    forged = json.loads((package / "benchmark-diagnostics.json").read_text())
    forged["cells"][0]["summary"][0]["fail_units"] += 1
    forged_package = tmp_path / "forged"
    shutil.copytree(package, forged_package)
    (forged_package / "benchmark-diagnostics.json").write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="does not match verdict fingerprints"):
        _submit("validate", forged_package, "--results", judged_matrix)

    # A changed saved verdict no longer matches the package fingerprints.
    verdicts_path = judged_matrix / "es-airline" / "diagnostics" / "verdicts.json"
    rows = json.loads(verdicts_path.read_text())
    unit = rows[0]["verdict"]["units"][0]
    unit["violation"] = not unit["violation"]
    verdicts_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Judge diagnostics differ"):
        _submit("validate", package, "--results", judged_matrix)

    # A package without diagnostics is not a complete Voice V2 submission.
    unit["violation"] = not unit["violation"]
    verdicts_path.write_text(json.dumps(rows))
    (package / "benchmark-diagnostics.json").unlink()
    with pytest.raises(ValueError, match="lacks benchmark-diagnostics.json"):
        _submit("validate", package, "--results", judged_matrix)


def test_validate_rejects_changed_agent_identity(
    judged_matrix, tmp_path, submitter_file
):
    _submit(
        "prepare",
        judged_matrix,
        "--output",
        tmp_path,
        "--metadata-file",
        submitter_file,
    )
    package = tmp_path / PACKAGE_NAME
    path = package / "benchmark-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["agent_model"] = "someone-else"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Recorded agent identity differs"):
        _submit("validate", package, "--results", judged_matrix)


def test_prepare_refuses_missing_rewards_with_counts_and_commands(
    judged_matrix, tmp_path, submitter_file
):
    shutil.rmtree(judged_matrix / "ko-telecom")
    for index in range(10):
        (judged_matrix / "pt-retail" / "simulations" / f"sim-{index}.json").unlink()
    with pytest.raises(SystemExit) as raised:
        _submit(
            "prepare",
            judged_matrix,
            "--output",
            tmp_path / "package",
            "--metadata-file",
            submitter_file,
        )
    message = str(raised.value.code)
    assert "Traceback" not in message
    assert "60/900 calls lack a scored reward (60 missing" in message
    assert "ko-telecom: 50 missing" in message
    assert "pt-retail: 10 missing" in message
    assert "--auto-resume" in message
    assert f"tau2 judges suite {judged_matrix}" in message
    assert not (tmp_path / "package").exists()


def test_prepare_refuses_missing_judges_per_cell_and_measure(
    judged_matrix, tmp_path, submitter_file
):
    verdicts_path = judged_matrix / "zh-airline" / "diagnostics" / "verdicts.json"
    rows = json.loads(verdicts_path.read_text())
    kept = [
        row
        for row in rows
        if not (
            row["identity"]["measure"] == "naturalness"
            and row["identity"]["input"]["simulation_id"] in {"sim-0", "sim-1"}
        )
    ]
    verdicts_path.write_text(json.dumps(kept))
    with pytest.raises(SystemExit) as raised:
        _submit(
            "prepare",
            judged_matrix,
            "--output",
            tmp_path / "package",
            "--metadata-file",
            submitter_file,
        )
    message = str(raised.value.code)
    assert "Rewards:" not in message
    assert "Judges: 1/18 cells" in message
    assert "zh-airline: naturalness 48/50 judged (2 not judged)" in message
    assert f"tau2 judges suite {judged_matrix}" in message
    assert "--auto-resume" not in message
    assert not (tmp_path / "package").exists()


def test_prepare_refuses_partial_judge_errors_cleanly(
    judged_matrix, tmp_path, submitter_file
):
    """A verdict with unit errors but some units is incomplete, never a crash."""
    verdicts_path = judged_matrix / "es-retail" / "diagnostics" / "verdicts.json"
    rows = json.loads(verdicts_path.read_text())
    partial = next(row for row in rows if row["identity"]["measure"] == "naturalness")
    partial["verdict"]["unit_errors"] = [
        {"unit_id": "late-utterance", "kind": "response", "error": "fixture"}
    ]
    verdicts_path.write_text(json.dumps(rows))
    with pytest.raises(SystemExit) as raised:
        _submit(
            "prepare",
            judged_matrix,
            "--output",
            tmp_path / "package",
            "--metadata-file",
            submitter_file,
        )
    message = str(raised.value.code)
    assert "es-retail: naturalness 49/50 judged (1 judge errors)" in message
    assert f"Run `tau2 judges suite {judged_matrix}`" in message
    assert not (tmp_path / "package").exists()
