# Copyright Sierra
"""Offline canonical-binding tests for the corrected final-window sidecar."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pytest

import experiments.tau_multilingual.experience_without_fluency as experience
import tau2.judges.corrected_paper_final_window_rebind as rebind_module
from experiments.tau_multilingual.experience_without_fluency import (
    FidelityEndExclusionManifest,
    _load_fidelity_end_exclusions,
    _validate_fidelity_manifest_cohort,
)
from tau2.judges.cli import add_judges_args
from tau2.judges.corrected_paper_final_window_rebind import (
    FinalWindowRebindConfig,
    FinalWindowRebindContract,
    rebind_final_window_manifest,
)
from tau2.judges.corrected_paper_suite import (
    FinalWindowExclusionRow,
    canonical_cohort_fingerprint_sha256,
    canonical_results_files_sha256,
)
from tau2.paper.multilingual import (
    ListeningManifest,
    MultilingualAudit,
    ResultCell,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(**updates) -> FinalWindowExclusionRow:
    values = {
        "sim_id": "sim-1",
        "language": "ko",
        "domain": "retail",
        "system": "OpenAI xhigh",
        "utterance_idx": 2,
        "finding_index": 0,
        "severity": 2,
        "time_range": "0.6-1.0",
        "clip_duration_seconds": 1.0,
        "span_start_seconds": 0.6,
        "span_end_seconds": 1.0,
    }
    values.update(updates)
    return FinalWindowExclusionRow.model_validate(values)


def _write_sidecar(root: Path, rows: list[FinalWindowExclusionRow]) -> Path:
    root.mkdir(parents=True)
    path = root / "final-window.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(FinalWindowExclusionRow.model_fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)
    by_language = {
        language: sum(row.language == language for row in rows)
        for language in {row.language for row in rows}
    }
    by_system = {
        system: sum(row.system == system for row in rows)
        for system in {row.system for row in rows}
    }
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "artifact": path.name,
                "filter_version": "v1",
                "window_seconds": 1.0,
                "cohort": "fixture",
                "excluded_findings": len(rows),
                "excluded_severity_2plus_findings": sum(
                    row.severity >= 2 for row in rows
                ),
                "by_language": dict(sorted(by_language.items())),
                "by_system": dict(sorted(by_system.items())),
                "results_files_sha256": "0" * 64,
                "replacement_cohort_fingerprint_sha256": "1" * 64,
                "replacement_calls": 1,
                "canonical_sidecar_sha256": "2" * 64,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def _write_repo(repo: Path) -> Path:
    cell = (
        repo
        / "data/simulations/paper_runs/tau-multi/main_runs/fixture"
        / "ko_retail_openai_xhigh"
    )
    simulations = cell / "simulations"
    simulations.mkdir(parents=True)
    simulation = {
        "id": "sim-1",
        "task_id": "task-1_ko",
        "delivery_info": {
            "utterance_results": [
                {
                    "utterance_idx": 2,
                    "findings": [
                        {
                            "axis": "fidelity",
                            "severity": 2,
                            "time_range": "0.6-1.0",
                        },
                        {
                            "axis": "fidelity",
                            "severity": 2,
                            "time_range": "0.7-1.0",
                        },
                    ],
                }
            ]
        },
        "ticks": [],
        "trial": 0,
    }
    (simulations / "sim-1.json").write_text(json.dumps(simulation, indent=2) + "\n")
    results = cell / "results.json"
    results.write_text(
        json.dumps(
            {"simulation_index": [{"id": "sim-1", "task_id": "task-1_ko", "trial": 0}]},
            indent=2,
        )
        + "\n"
    )
    audit = MultilingualAudit(
        repository_commit="fixture",
        evidence_root="fixture",
        artifact_id="fixture",
        findings=[],
        voice_cells=[
            ResultCell(
                cohort="voice",
                language="ko",
                domain="retail",
                system="openai_xhigh",
                results_path="main_runs/fixture/ko_retail_openai_xhigh/results.json",
                sha256=_sha256(results),
                git_commit="fixture",
                rows=1,
                unique_tasks=1,
                trials=[0],
                trial_success={0: 1.0},
                task_ids_sha256="3" * 64,
                matches_expected_subset=True,
            )
        ],
        text_cells=[],
        task_success=[],
        text_success=[],
        text_voice_gap=[],
        trial_stability=[],
        retail_ablations=[],
        listening=ListeningManifest(calls=[]),
    )
    audit_content = audit.model_dump(
        mode="json",
        exclude={"artifact_id", "repository_commit", "evidence_root"},
    )
    audit = audit.model_copy(
        update={
            "artifact_id": hashlib.sha256(
                json.dumps(
                    audit_content, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()[:16]
        }
    )
    audit_path = repo / "papers/tau-multilingual/reproduction/audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(audit.model_dump_json(indent=2) + "\n")
    return results


def _contract(sidecar: Path) -> FinalWindowRebindContract:
    return FinalWindowRebindContract(
        languages=("ko",),
        domains=("retail",),
        system_slugs=("openai_xhigh",),
        system_labels={"openai_xhigh": "OpenAI xhigh"},
        calls_per_cell=1,
        expected_csv_sha256=_sha256(sidecar),
        expected_generation_results_files_sha256="0" * 64,
        expected_replacement_cohort_fingerprint_sha256="1" * 64,
        expected_replacement_calls=1,
        expected_canonical_sidecar_sha256="2" * 64,
    )


def _config(repo: Path, sidecar: Path, output: Path) -> FinalWindowRebindConfig:
    return FinalWindowRebindConfig(
        repo_root=repo,
        sidecar_from=sidecar,
        output_root=output,
    )


def test_rebind_accepts_current_headers_and_copies_csv_byte_identically(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    output = tmp_path / "rebound"

    manifest = rebind_final_window_manifest(
        _config(repo, sidecar, output), contract=_contract(sidecar)
    )

    assert (output / sidecar.name).read_bytes() == sidecar.read_bytes()
    assert "results_files_sha256" not in json.loads(
        (output / sidecar.with_suffix(".json").name).read_text()
    )
    assert manifest.generation_results_files_sha256 == "0" * 64
    assert manifest.final_window_csv_sha256 == _sha256(sidecar)
    assert manifest.canonical_results_files == 1
    assert manifest.canonical_trial0_calls == 1


def test_final_window_rebind_cli_is_registered():
    parser = ArgumentParser()
    add_judges_args(parser)

    args = parser.parse_args(
        [
            "corrected-paper-suite",
            "final-window-rebind",
            "--sidecar-from",
            "/tmp/generated.csv",
            "--out",
            "/tmp/rebound",
        ]
    )

    assert args.func.__name__ == "rebind_corrected_final_window_sidecar"


def test_rebind_rejects_stale_active_header(tmp_path):
    repo = tmp_path / "repo"
    results = _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    results.write_text(results.read_text() + " ")

    with pytest.raises(ValueError, match="header hash drifted"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_duplicate_exclusion_rows(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row(), _row()])

    with pytest.raises(ValueError, match="duplicate exclusion rows"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_row_for_missing_canonical_call(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row(sim_id="missing")])

    with pytest.raises(ValueError, match="missing canonical call"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_finding_metadata_drift(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row(severity=3)])

    with pytest.raises(ValueError, match="finding metadata drifted"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row(span_start_seconds=0.5), "span metadata drifted"),
        (_row(clip_duration_seconds=3.0), "no longer satisfies its rule"),
    ],
)
def test_rebind_rejects_span_or_rule_drift(tmp_path, row, message):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [row])

    with pytest.raises(ValueError, match=message):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_generated_provenance_drift(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    manifest_path = sidecar.with_suffix(".json")
    payload = json.loads(manifest_path.read_text())
    payload["results_files_sha256"] = "9" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(ValueError, match="generated final-window provenance drifted"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_mixed_legacy_and_new_generation_schema(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    manifest_path = sidecar.with_suffix(".json")
    payload = json.loads(manifest_path.read_text())
    payload["generation_results_files_sha256"] = payload["results_files_sha256"]
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(ValueError, match="generation_results_files_sha256"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_audit_artifact_identity_drift(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    audit_path = repo / "papers/tau-multilingual/reproduction/audit.json"
    audit = json.loads(audit_path.read_text())
    audit["artifact_id"] = "not-the-content-id"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    with pytest.raises(ValueError, match="audit artifact identity"):
        rebind_final_window_manifest(
            _config(repo, sidecar, tmp_path / "out"), contract=_contract(sidecar)
        )


def test_rebind_rejects_existing_output(tmp_path):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    output = tmp_path / "out"
    output.mkdir()

    with pytest.raises(ValueError, match="output already exists"):
        rebind_final_window_manifest(
            _config(repo, sidecar, output), contract=_contract(sidecar)
        )


def test_rebind_cleans_atomic_staging_after_copy_hash_failure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    output = tmp_path / "out"

    def corrupt_copy(_source, destination):
        Path(destination).write_bytes(b"corrupt")

    monkeypatch.setattr(rebind_module.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="does not match the pinned hash"):
        rebind_final_window_manifest(
            _config(repo, sidecar, output), contract=_contract(sidecar)
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".out.*"))


def test_experience_rejects_same_count_valid_finding_swap(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repo(repo)
    sidecar = _write_sidecar(tmp_path / "source", [_row()])
    rebound = tmp_path / "rebound"
    rebind_final_window_manifest(
        _config(repo, sidecar, rebound), contract=_contract(sidecar)
    )
    analysis = repo / "data/analysis"
    analysis.mkdir(parents=True)
    csv_path = analysis / sidecar.name
    manifest_path = csv_path.with_suffix(".json")
    shutil.copyfile(rebound / sidecar.name, csv_path)
    shutil.copyfile(rebound / sidecar.with_suffix(".json").name, manifest_path)
    monkeypatch.setattr(
        experience,
        "FIDELITY_END_EXCLUSIONS_PATH",
        Path("data/analysis") / sidecar.name,
    )
    loaded, _provenance, keys, _manifest = _load_fidelity_end_exclusions(repo)
    assert len(loaded["sim-1"]) == 1
    assert keys == {("sim-1", 2, 0)}
    rows = list(csv.DictReader(csv_path.open(newline="")))
    rows[0].update(
        {
            "finding_index": "1",
            "time_range": "0.7-1.0",
            "span_start_seconds": "0.7",
        }
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(FinalWindowExclusionRow.model_fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="CSV hash drifted"):
        _load_fidelity_end_exclusions(repo)


def test_experience_requires_exact_rebound_canonical_hashes():
    sources = [{"path": "a/results.json", "sha256": "a" * 64, "trial_0_calls": 1}]
    manifest = FidelityEndExclusionManifest(
        artifact="final-window.csv",
        filter_version="v1",
        window_seconds=1.0,
        cohort="fixture",
        excluded_findings=1,
        excluded_severity_2plus_findings=1,
        by_language={"ko": 1},
        by_system={"OpenAI xhigh": 1},
        final_window_csv_sha256="f" * 64,
        generation_results_files_sha256="0" * 64,
        replacement_cohort_fingerprint_sha256="1" * 64,
        replacement_calls=1,
        canonical_sidecar_sha256="2" * 64,
        canonical_results_files_sha256=canonical_results_files_sha256(sources),
        canonical_cohort_fingerprint_sha256=(
            canonical_cohort_fingerprint_sha256(sources)
        ),
        canonical_results_files=1,
        canonical_trial0_calls=1,
    )

    _validate_fidelity_manifest_cohort(manifest, sources)
    with pytest.raises(ValueError, match="canonical results hash drifted"):
        _validate_fidelity_manifest_cohort(
            manifest,
            [{"path": "a/results.json", "sha256": "b" * 64, "trial_0_calls": 1}],
        )
