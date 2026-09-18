# Copyright Sierra
"""Checks for the corrected τ-Multilingual latency artifact."""

import hashlib
import json
import math
from pathlib import Path

from tau2.paper.latency import (
    RESULT_PATH_PREFIX,
    LatencyCounts,
    _latency_counts,
    _SimulationPrefix,
    _snapshot,
    load_multilingual_latency_artifact,
)
from tau2.paper.retail_replacement import load_retail_replacement_manifest

REPO = Path(__file__).resolve().parents[2]
LATENCY = REPO / "papers/tau-multilingual/reproduction/latency.json"
AUDIT = REPO / "papers/tau-multilingual/reproduction/audit.json"
PREVIOUS_EXPERIENCE = (
    REPO / "papers/tau-multilingual/reproduction/validation_runs/"
    "pre-retail-name-role-v1/experience.json"
)
REPLACEMENT_MANIFEST = (
    REPO / "papers/tau-multilingual/reproduction/retail_name_role_replacement.json"
)
LATENCY_SHA256 = "eb56617f3b216566a9147e49699500d30b9ed9b311af3586bda6b85d19c3e01f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latency_counts_are_event_weighted_then_channel_balanced() -> None:
    first = LatencyCounts(
        response_seconds_sum=12.0,
        response_events=4.0,
        yield_seconds_sum=8.0,
        yield_events=2.0,
    )
    second = LatencyCounts(
        response_seconds_sum=10.0,
        response_events=1.0,
        yield_seconds_sum=0.5,
        yield_events=1.0,
    )

    combined = first.plus(second)

    assert combined.response_seconds_sum == 22.0
    assert combined.response_events == 5.0
    assert combined.yield_seconds_sum == 8.5
    assert combined.yield_events == 3.0
    assert combined.latency_seconds == ((22.0 / 5.0) + (8.5 / 3.0)) / 2.0


def test_partial_latency_metric_pair_is_rejected() -> None:
    simulation = _SimulationPrefix.model_validate(
        {
            "id": "partial",
            "task_id": "task",
            "quality_info": {
                "factor_checks": [
                    {
                        "id": "responsiveness",
                        "metrics": {"response_total": 2},
                    }
                ]
            },
        }
    )

    try:
        _latency_counts(simulation)
    except ValueError as error:
        assert "partial response latency metrics" in str(error)
    else:
        raise AssertionError("partial latency metrics must fail loudly")


def test_snapshot_filters_trial_one_and_hashes_trial_zero_simulations(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "cell" / "results.json"
    simulations = results_path.parent / "simulations"
    simulations.mkdir(parents=True)
    rows = []
    for index in range(51):
        trial = 0 if index < 50 else 1
        sim_id = f"sim-{index}"
        task_id = f"task-{index}"
        rows.append({"id": sim_id, "task_id": task_id, "trial": trial})
        payload = {
            "id": sim_id,
            "task_id": task_id,
            "quality_info": {
                "factor_checks": [
                    {
                        "id": "responsiveness",
                        "metrics": {
                            "response_total": 1,
                            "response_latency_mean": 2.0,
                        },
                    },
                    {
                        "id": "yielding",
                        "metrics": {
                            "yield_total": 1,
                            "yield_latency_mean": 4.0,
                        },
                    },
                ]
            },
            "ticks": [],
        }
        (simulations / f"{sim_id}.json").write_text(json.dumps(payload, indent=2))
    results_path.write_text(json.dumps({"simulation_index": rows}, indent=2))

    snapshot = _snapshot(
        results_path,
        logical_path="data/simulations/paper_runs/tau-multi/main_runs/cell/results.json",
        storage_path="data/simulations/paper_runs/tau-multi/main_runs/cell/results.json",
        expected_results_sha256=_sha256(results_path),
    )

    assert snapshot.calls == 50
    assert snapshot.counts.response_events == 50
    assert snapshot.counts.yield_events == 50
    assert snapshot.latency_seconds == 3.0


def test_checked_in_latency_artifact_binds_the_corrected_active_cohort() -> None:
    artifact = load_multilingual_latency_artifact(LATENCY)
    audit = json.loads(AUDIT.read_text())
    previous = json.loads(PREVIOUS_EXPERIENCE.read_text())

    assert _sha256(LATENCY) == LATENCY_SHA256
    assert artifact.provenance.audit_sha256 == _sha256(AUDIT)
    assert artifact.provenance.audit_artifact_id == audit["artifact_id"]
    assert artifact.provenance.previous_experience_sha256 == _sha256(
        PREVIOUS_EXPERIENCE
    )
    assert (
        artifact.provenance.previous_results_fingerprint_sha256
        == (previous["provenance"]["cohort_fingerprint_sha256"])
    )
    assert "/private/tmp" not in LATENCY.read_text()

    audit_by_key = {
        (cell["language"], cell["domain"], cell["system"]): cell
        for cell in audit["voice_cells"]
    }
    previous_by_path = {
        source["path"]: source for source in previous["provenance"]["results_files"]
    }
    for cell in artifact.cells:
        key = (cell.language, cell.domain, cell.system)
        audit_cell = audit_by_key[key]
        logical_path = (RESULT_PATH_PREFIX / audit_cell["results_path"]).as_posix()
        previous_source = previous_by_path[logical_path]

        assert cell.active.logical_path == logical_path
        assert cell.active.storage_path == logical_path
        assert cell.active.results_sha256 == audit_cell["sha256"]
        assert cell.previous.logical_path == logical_path
        assert cell.previous.results_sha256 == previous_source["sha256"]
        assert cell.previous.calls == previous_source["trial_0_calls"] == 50


def test_only_corrected_korean_and_mandarin_retail_latency_can_change() -> None:
    artifact = load_multilingual_latency_artifact(LATENCY)
    replacement = load_retail_replacement_manifest(REPLACEMENT_MANIFEST)
    replacement_by_key = {
        f"{cell.language}/retail/{cell.system}": cell
        for cell in replacement.cells
        if cell.modality == "voice"
    }
    changed_inputs = {cell.key for cell in artifact.cells if cell.input_changed}
    changed_latencies = {cell.key for cell in artifact.cells if cell.latency_changed}

    assert artifact.comparison.cells == 90
    assert artifact.comparison.unchanged_input_cells == 80
    assert artifact.comparison.changed_input_cells == 10
    assert artifact.comparison.changed_latency_cells == 10
    assert changed_inputs == changed_latencies == set(replacement_by_key)

    for cell in artifact.cells:
        if cell.key in replacement_by_key:
            manifest_cell = replacement_by_key[cell.key]
            assert (
                cell.active.logical_path
                == (RESULT_PATH_PREFIX / manifest_cell.canonical_path).as_posix()
            )
            assert (
                cell.previous.storage_path
                == (RESULT_PATH_PREFIX / manifest_cell.archive_path).as_posix()
            )
            assert cell.previous.results_sha256 == (
                manifest_cell.previous.results_sha256
            )
            assert cell.active.results_sha256 == (
                manifest_cell.replacement.results_sha256
            )
            assert cell.previous.simulations_sha256 != cell.active.simulations_sha256
            assert cell.previous.latency_seconds != cell.active.latency_seconds
        else:
            assert cell.previous == cell.active


def test_historical_inputs_reproduce_the_frozen_latency_rollups() -> None:
    artifact = load_multilingual_latency_artifact(LATENCY)
    previous = json.loads(PREVIOUS_EXPERIENCE.read_text())[
        "descriptive_complete_cohort"
    ]
    recomputed = artifact.previous_descriptive_complete_cohort

    for language, systems in recomputed.language_system.items():
        for system, value in systems.items():
            assert math.isclose(
                value.latency_seconds,
                previous["language_system"][language][system]["latency_seconds"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
    for system, value in recomputed.provider.items():
        assert math.isclose(
            value.latency_seconds,
            previous["provider"][system]["latency_seconds"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    assert math.isclose(
        recomputed.overall.latency_seconds,
        previous["overall"]["latency_seconds"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
