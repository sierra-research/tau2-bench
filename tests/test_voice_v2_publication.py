"""The frozen 2026-10-02 release export becomes the same 2.0 package as a matrix.

A synthetic 18-cell export on the real published frame; no model is called.
"""

import argparse
import json
import shutil
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.runner.presets import load_frame
from tau2.scripts.leaderboard.benchmark_package import (
    METADATA_FILE_NAME,
    SUBMISSION_FILE_NAME,
    PackageMetadata,
    SubmitterMetadata,
    load_package,
    package_name,
)
from tau2.scripts.leaderboard.benchmark_quality import QUALITY_FILE_NAME
from tau2.scripts.leaderboard.generate_schema import (
    METADATA_SCHEMA_PATH,
    QUALITY_SCHEMA_PATH,
    SCHEMA_PATH,
    generate_metadata_schema,
    generate_quality_schema,
    generate_schema,
)
from tau2.scripts.leaderboard.submission import LeaderboardManifest
from tau2.scripts.leaderboard.voice_v2_publication import (
    _validate_cell_frame,
    _validate_simulation_files,
)

MEASURES = {"en": ("naturalness", "tool_use"), "ko": ("honorific_agreement",)}
SUBMITTER = {
    "model_name": "Offline Fixture 1.0",
    "model_organization": "Test Org",
    "submitting_organization": "Test",
    "submission_date": "2026-10-02",
    "contact_info": {"email": "review@example.org"},
}
NAME = "offline-fixture-1-0_test-org_2026-10-02"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _verdict(language, domain, simulation_id, measure, violation=False):
    units = [
        {"unit_id": "0", "opportunity": True, "violation": violation},
        {"unit_id": "1", "opportunity": False, "violation": False},
    ]
    if measure == "tool_use":
        # The tool-use judge scores the whole call as one unit.
        units = [{"unit_id": "call", "opportunity": True, "violation": violation}]
    return {
        "identity": {
            "input": {
                "domain": domain,
                "run_id": "r" * 64,
                "simulation_id": simulation_id,
                "trial": 0,
                "language": language,
                "agent_gender": "female",
                "caller_gender": "male",
                "input_fingerprint": "f" * 64,
                "has_audio": True,
                "audio_fingerprint": "a" * 64,
            },
            "measure": measure,
            "contract_id": f"{measure}-fixture",
            "implementation": "naturalness-v17",
            "input_exclusions": "injected-opener-transfer-line-and-control-turns-v1",
            "model": "fixture-judge",
            "model_args": {},
            "prompt_version": "v1",
            "rubric_version": "r1",
            "transport_revision": "multimodal-responses-v1",
            "catalog_revision": "fixture-catalog",
        },
        "verdict": {"units": [dict(unit, evidence="fixture") for unit in units]},
    }


@pytest.fixture
def forbid_paid_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Publication must use only saved offline inputs")

    monkeypatch.setattr("tau2.utils.llm_utils.generate", forbidden)
    monkeypatch.setattr("litellm.completion", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def release_export(tmp_path_factory):
    """Historical layout: trajectories/<lang>/<domain>/ plus per-language verdicts."""
    root = tmp_path_factory.mktemp("release")
    source = root / "export"
    judges = root / "verdicts"
    official = load_frame("voice", num_trials=1)
    for cell in official.cells:
        language, domain = cell.language, cell.domain
        directory = source / "trajectories" / language / domain
        index = []
        for position, task_id in enumerate(cell.canonical_task_ids):
            simulation_id = f"{domain}-{task_id}"
            reward = float(position % 2)
            simulation = SimulationRun(
                id=simulation_id,
                task_id=task_id,
                trial=0,
                timestamp="2026-10-02T00:00:00",
                start_time="2026-10-02T00:00:00",
                end_time="2026-10-02T00:00:01",
                duration=1,
                termination_reason="agent_stop",
                reward_info=RewardInfo(reward=reward),
            )
            _write_json(
                directory / "simulations" / f"{simulation_id}.json",
                simulation.model_dump(mode="json"),
            )
            index.append(
                {
                    "id": simulation_id,
                    "task_id": task_id,
                    "trial": 0,
                    "reward": reward,
                    "termination_reason": "agent_stop",
                }
            )
            measures = MEASURES.get(language, ("naturalness",))
            _write_json(
                judges / language / f"{simulation_id}.json",
                [
                    _verdict(language, domain, simulation_id, m, position % 4 == 0)
                    for m in measures
                ],
            )
        _write_json(
            directory / "results.json",
            {
                "info": {
                    "benchmark": {
                        "benchmark_version": official.benchmark_version,
                        "language": language,
                        "domain": domain,
                        "selected_task_ids": cell.canonical_task_ids,
                        "canonical_task_ids": cell.canonical_task_ids,
                        "trials": [0],
                        "frame_revision": official.revision,
                        "binding_revision": cell.binding_revision,
                        "condition_fingerprint": "c" * 64,
                        "conditions": {
                            # Historical provider values the current parser
                            # rejects are read as recorded strings.
                            "audio_native_config": {
                                "provider": "openai_live",
                                "model": "offline-fixture",
                            }
                        },
                    },
                    "language": language,
                    "environment_info": {"domain_name": domain},
                    "agent_info": {"llm": "offline-fixture"},
                    "user_info": {"llm": "offline-caller"},
                    "num_trials": 1,
                    "language_tasks": {
                        task_id: {"canonical_task_id": task_id}
                        for task_id in cell.canonical_task_ids
                    },
                },
                "tasks": [{"id": task_id} for task_id in cell.canonical_task_ids],
                "simulation_index": index,
            },
        )
    metadata = root / "metadata.json"
    _write_json(metadata, SUBMITTER)
    return source, judges, metadata


def _submit(*argv):
    from tau2.scripts.leaderboard.cli import add_submit_args

    parser = argparse.ArgumentParser()
    add_submit_args(parser)
    args = parser.parse_args([str(value) for value in argv])
    args.func(args)


@pytest.fixture
def package(release_export, tmp_path, forbid_paid_calls):
    source, judges, metadata = release_export
    output = tmp_path / "out"
    _submit(
        "prepare",
        source,
        "--historical-voice-v2",
        "--judges-root",
        judges,
        "--metadata-file",
        metadata,
        "--output",
        output,
    )
    return output / NAME


def _validate(package, release_export, judges=None):
    source, default_judges, _ = release_export
    _submit(
        "validate",
        package,
        "--historical-voice-v2",
        "--judges-root",
        judges or default_judges,
        "--results",
        source,
    )


def test_historical_export_is_a_canonical_package(package, release_export):
    loaded = load_package(package)
    assert loaded.metadata.evidence == "historical-voice-v2-2026-10-02"
    assert (loaded.metadata.agent_provider, loaded.metadata.agent_model) == (
        "openai_live",
        "offline-fixture",
    )
    summary = loaded.submission.summary
    assert summary.complete and len(loaded.submission.trials) == 900
    assert summary.score == pytest.approx(0.5)
    assert set(summary.by_language) == {"en", "es", "hi", "ko", "pt", "zh"}
    diagnostics = loaded.diagnostics
    assert diagnostics.catalog_revision == "fixture-catalog"
    assert len(diagnostics.cells) == 18
    en_airline = next(
        cell
        for cell in diagnostics.cells
        if (cell.language, cell.domain) == ("en", "airline")
    )
    tool_use = next(row for row in en_airline.summary if row.measure == "tool_use")
    assert (tool_use.applicable_calls, tool_use.judged_calls) == (50, 50)
    assert tool_use.complete and tool_use.pass_units + tool_use.fail_units == 50
    assert {row.measure for row in en_airline.summary} == {"naturalness", "tool_use"}
    quality = loaded.quality
    assert (quality.benchmark_version, quality.frame_revision) == (
        "voice-v2",
        loaded.submission.frame.revision,
    )
    assert [(c.language, c.domain) for c in quality.cells] == [
        (c.language, c.domain) for c in loaded.submission.frame.cells
    ]
    en_airline_quality = quality.cells[0]
    assert len(en_airline_quality.calls) == 50
    # The fixture calls have no ticks and no fidelity verdicts: only the tool-use
    # component is known, so neither measure can decide a call.
    assert en_airline_quality.interaction_components.tool_misuse.known == 50
    assert en_airline_quality.interaction_components.nonresponse.known == 0
    assert en_airline_quality.interaction.unscored == 50
    assert en_airline_quality.experience.unscored == 50
    _validate(package, release_export)


def test_upload_layout_matches_the_uploaded_release_entry(package, release_export):
    """Review uploads use the S3 layout of the 2026-10-09 Gemini entry."""
    from tau2.scripts.leaderboard.review_submission import _package_uploads

    source, judges, _ = release_export
    (source / "correction.json").write_text("{}")
    try:
        keys = {
            key for _, key in _package_uploads(load_package(package), source, [judges])
        }
    finally:
        (source / "correction.json").unlink()
    assert "trajectories/hi/airline/" in keys
    assert "source_correction.json" in keys
    assert "judges/openai_live/ko/" in keys
    assert not any(key.startswith(("judges/ko", "correction")) for key in keys)


def test_validate_rejects_changed_package_and_evidence(package, release_export):
    submission = json.loads((package / SUBMISSION_FILE_NAME).read_text())
    submission["summary"]["score"] = 0.9
    (package / SUBMISSION_FILE_NAME).write_text(json.dumps(submission))
    with pytest.raises(ValueError, match="summary does not match"):
        _validate(package, release_export)


def test_validate_recomputes_call_quality(package, release_export):
    path = package / QUALITY_FILE_NAME
    original = path.read_text()
    quality = json.loads(original)
    # A self-consistent edit still differs from the recomputed calls.
    quality["cells"][0]["calls"][0]["experience_utterances"] += 1
    path.write_text(json.dumps(quality))
    with pytest.raises(ValueError, match="Call quality differs"):
        _validate(package, release_export)
    quality["cells"][0]["interaction"]["passed"] += 1
    path.write_text(json.dumps(quality))
    with pytest.raises(ValueError, match="totals do not match the calls"):
        load_package(package)
    path.unlink()
    with pytest.raises(ValueError, match=f"lacks {QUALITY_FILE_NAME}"):
        load_package(package)
    path.write_text(original)
    _validate(package, release_export)


def test_validate_rejects_changed_verdict(package, release_export, tmp_path):
    _, judges, _ = release_export
    changed = tmp_path / "verdicts"
    shutil.copytree(judges, changed)
    path = next((changed / "es").glob("*.json"))
    rows = json.loads(path.read_text())
    rows[0]["verdict"]["units"][0]["violation"] = not rows[0]["verdict"]["units"][0][
        "violation"
    ]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Judge diagnostics differ"):
        _validate(package, release_export, changed)


def test_partial_and_missing_verdicts_are_coverage_gaps(
    release_export, tmp_path, forbid_paid_calls
):
    from tau2.scripts.leaderboard.voice_v2_publication import (
        build_historical_package,
    )

    source, judges, metadata = release_export
    gaps = tmp_path / "verdicts"
    shutil.copytree(judges, gaps)
    partial, missing = sorted((gaps / "en").glob("retail-*.json"))[:2]
    rows = json.loads(partial.read_text())
    rows[0]["verdict"]["unit_errors"] = [
        {"unit_id": "2", "kind": "response", "error": "fixture"}
    ]
    partial.write_text(json.dumps(rows))
    rows = json.loads(missing.read_text())
    missing.write_text(json.dumps(rows[1:]))
    built = build_historical_package(
        source,
        [gaps],
        SubmitterMetadata.model_validate_json(metadata.read_text()),
    )
    cell = next(
        cell
        for cell in built.diagnostics.cells
        if (cell.language, cell.domain) == ("en", "retail")
    )
    naturalness = next(row for row in cell.summary if row.measure == "naturalness")
    assert not naturalness.complete
    assert (naturalness.judged_calls, naturalness.not_judged_calls) == (48, 1)
    assert (naturalness.error_calls, naturalness.partial_calls) == (1, 1)
    # Never a failure: rewards and the official score are unchanged.
    assert built.submission.summary.score == pytest.approx(0.5)


def test_prepare_requires_one_verdict_file_per_scored_call(
    release_export, tmp_path, forbid_paid_calls
):
    source, judges, metadata = release_export
    duplicate = tmp_path / "duplicate"
    shutil.copytree(judges / "hi", duplicate / "hi")
    with pytest.raises(ValueError, match="expected one saved verdict file, found 2"):
        _submit(
            "prepare",
            source,
            "--historical-voice-v2",
            "--judges-root",
            judges,
            "--judges-root",
            duplicate,
            "--metadata-file",
            metadata,
            "--output",
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_historical_flags_are_rejected_without_the_adapter(tmp_path):
    with pytest.raises(ValueError, match="requires --historical-voice-v2"):
        _submit(
            "prepare", tmp_path, "--judges-root", tmp_path, "--output", tmp_path / "o"
        )


def test_metadata_contract():
    submitter = SubmitterMetadata.model_validate(SUBMITTER)
    assert package_name(submitter) == NAME
    # One agent's packages share metadata; domain frames get distinct names.
    names = {
        version: package_name(submitter, version)
        for version in ("voice-v2", "text-v1", "voice-v2-banking", "voice-v2-entity")
    }
    assert names == {
        "voice-v2": NAME,
        "text-v1": NAME,
        "voice-v2-banking": f"{NAME}_banking",
        "voice-v2-entity": f"{NAME}_entity",
    }
    with pytest.raises(ValidationError):
        SubmitterMetadata.model_validate({**SUBMITTER, "contact_info": {}})
    with pytest.raises(ValidationError):
        SubmitterMetadata.model_validate({**SUBMITTER, "results": {}})
    with pytest.raises(ValidationError):
        PackageMetadata(
            **SUBMITTER,
            agent_provider="gemini",
            agent_model="m",
            trajectories="https://example.org/not-s3",
        )


def test_package_rejects_incomplete_or_misfiled_parts(package):
    loaded = load_package(package)
    with pytest.raises(ValueError, match="lacks benchmark-diagnostics.json"):
        loaded.model_copy(update={"diagnostics": None}).check()
    text = loaded.model_copy(
        update={"metadata": loaded.metadata.model_copy(update={"agent_provider": None})}
    )
    with pytest.raises(ValueError, match="agent_provider"):
        text.check()
    (package / METADATA_FILE_NAME).unlink()
    with pytest.raises(ValueError, match="lacks benchmark-metadata.json"):
        load_package(package)


def test_frame_rejects_duplicate_task_trial_even_with_50_entries() -> None:
    official = load_frame("voice", num_trials=1)
    cell = next(
        item
        for item in official.cells
        if (item.language, item.domain) == ("en", "airline")
    )
    task_ids = cell.canonical_task_ids
    simulations = [
        {
            "id": f"sim-{index}",
            "task_id": task_id,
            "trial": 0,
            "reward": 1.0,
            "termination_reason": "agent_stop",
        }
        for index, task_id in enumerate([*task_ids[:-1], task_ids[0]])
    ]
    data = {
        "info": {
            "benchmark": {
                "benchmark_version": official.benchmark_version,
                "frame_revision": official.revision,
                "binding_revision": cell.binding_revision,
                "canonical_task_ids": task_ids,
                "selected_task_ids": task_ids,
                "language": "en",
                "domain": "airline",
                "trials": [0],
            },
            "language": "en",
            "environment_info": {"domain_name": "airline"},
            "num_trials": 1,
            "language_tasks": {
                task_id: {"canonical_task_id": task_id} for task_id in task_ids
            },
        },
        "tasks": [{"id": task_id} for task_id in task_ids],
    }
    with pytest.raises(ValueError, match="missing or duplicate task/trial keys"):
        _validate_cell_frame(data, simulations, "en", "airline", official)


def test_index_must_agree_with_saved_simulation(release_export) -> None:
    source, _, _ = release_export
    results = json.loads((source / "trajectories/en/airline/results.json").read_text())
    index = [dict(entry) for entry in results["simulation_index"]]
    index[0]["reward"] = 1.0 - index[0]["reward"]
    with pytest.raises(ValueError, match="simulation/index reward mismatch"):
        _validate_simulation_files(source, index, "en", "airline")


def test_schemas_are_current_and_manifest_has_a_2_0_key():
    assert SCHEMA_PATH.read_text() == generate_schema()
    assert METADATA_SCHEMA_PATH.read_text() == generate_metadata_schema()
    assert QUALITY_SCHEMA_PATH.read_text() == generate_quality_schema()
    manifest = LeaderboardManifest.model_validate_json(
        (SCHEMA_PATH.parent / "manifest.json").read_text()
    )
    for name in manifest.voice_v2_submissions + manifest.voice_v2_domain_submissions:
        load_package(SCHEMA_PATH.parent / name)


def test_historical_infrastructure_errors_are_excluded_and_validate(
    release_export, tmp_path, forbid_paid_calls
):
    """Unscored infrastructure calls leave the denominator, not the package."""
    source, judges, metadata = release_export
    copied, verdicts = tmp_path / "export", tmp_path / "verdicts"
    shutil.copytree(source, copied)
    shutil.copytree(judges, verdicts)
    directory = copied / "trajectories" / "zh" / "telecom"
    results = json.loads((directory / "results.json").read_text())
    # Positions 1 and 3 would have passed; their tasks drop out of the mean.
    for entry in (results["simulation_index"][1], results["simulation_index"][3]):
        entry.update(reward=None, termination_reason="infrastructure_error")
        path = directory / "simulations" / f"{entry['id']}.json"
        simulation = json.loads(path.read_text())
        simulation.update(reward_info=None, termination_reason="infrastructure_error")
        path.write_text(json.dumps(simulation))
        (verdicts / "zh" / f"{entry['id']}.json").unlink()
    (directory / "results.json").write_text(json.dumps(results))
    output = tmp_path / "out"
    _submit(
        "prepare",
        copied,
        "--historical-voice-v2",
        "--judges-root",
        verdicts,
        "--metadata-file",
        metadata,
        "--output",
        output,
    )
    summary = load_package(output / NAME).submission.summary
    assert summary.complete and summary.error_trials == 2
    cell = next(c for c in summary.cells if (c.language, c.domain) == ("zh", "telecom"))
    assert (cell.scored_trials, cell.error_trials) == (48, 2)
    assert cell.score == pytest.approx(23 / 48)
    assert summary.score == pytest.approx((17 * 0.5 + 23 / 48) / 18)
    _validate(output / NAME, (copied, verdicts, metadata))
