# Copyright Sierra
"""Run identity: what it is stable across, and what it must notice.

A path locates a run; the identity says WHICH run. These tests pin both
halves of that contract against real ``Results`` written to disk.
"""

import json
import shutil

import pytest
from fixtures_runs import make_hi_results

from tau2.data_model.message import AssistantMessage
from tau2.data_model.run_identity import compute_run_identity
from tau2.data_model.simulation import (
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)


def _sim(sim_id: str, task_id: str) -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=1.0),
        messages=[AssistantMessage(role="assistant", content="hello")],
    )


def _run(tmp_path, sim_ids=("s1", "s2"), *, name="run", format="dir"):
    sims = [_sim(sid, f"t{i}") for i, sid in enumerate(sim_ids)]
    return make_hi_results(tmp_path, sims, name=name, format=format)


def test_identity_survives_a_move(tmp_path):
    run = _run(tmp_path / "before")
    before = compute_run_identity(run)

    moved = tmp_path / "after" / "somewhere" / "renamed"
    moved.parent.mkdir(parents=True)
    shutil.move(str(run), str(moved))

    assert compute_run_identity(moved).run_id == before.run_id


def test_a_regenerated_run_gets_a_different_identity(tmp_path):
    first = compute_run_identity(_run(tmp_path / "a"))
    # Same save path, re-executed: new run timestamp, new simulation uuids.
    second = compute_run_identity(_run(tmp_path / "b", sim_ids=("s9", "s10")))
    assert first.run_id != second.run_id


def test_a_different_simulation_set_at_the_same_timestamp_differs(tmp_path):
    """Identity is not the timestamp alone — the sim set is part of it."""
    run_a = _run(tmp_path / "a", sim_ids=("s1", "s2"))
    run_b = _run(tmp_path / "b", sim_ids=("s1", "s3"))
    for run in (run_a, run_b):
        meta = json.loads((run / "results.json").read_text())
        meta["timestamp"] = "2026-07-20T00:00:00"
        (run / "results.json").write_text(json.dumps(meta))

    assert compute_run_identity(run_a).run_id != compute_run_identity(run_b).run_id


def test_identity_is_stable_under_re_judging(tmp_path):
    """Post-hoc judging rewrites simulation FILES but keeps their ids — the
    identity names the run, not the annotations layered onto it."""
    run = _run(tmp_path)
    before = compute_run_identity(run)

    results = Results.load(run)
    for sim in results.simulations:
        sim.reward_info.reward = 0.0
    results.save(run, format="dir")

    assert compute_run_identity(run).run_id == before.run_id


def test_a_run_dir_and_its_results_json_identify_the_same_run(tmp_path):
    run = _run(tmp_path)
    assert (
        compute_run_identity(run).run_id
        == compute_run_identity(run / "results.json").run_id
    )


def test_monolithic_json_runs_identify_too(tmp_path):
    path = _run(tmp_path, name="mono", format="json")
    identity = compute_run_identity(path)
    assert identity.num_simulations == 2
    assert identity.run_id


def test_dir_format_identity_reads_no_simulation_files(tmp_path, monkeypatch):
    """Cheap by construction: a 4 GB voice run must be identified from its
    metadata alone, so nothing under simulations/ may be opened."""
    run = _run(tmp_path)
    expected = compute_run_identity(run)
    sim_files = sorted((run / "simulations").glob("*.json"))
    assert sim_files
    for sim_file in sim_files:
        sim_file.write_text("NOT JSON — reading this would raise")

    assert compute_run_identity(run).run_id == expected.run_id


def test_identity_carries_legible_context(tmp_path):
    identity = compute_run_identity(_run(tmp_path))
    assert identity.num_simulations == 2
    assert identity.timestamp


def test_a_path_with_no_results_is_loud(tmp_path):
    empty = tmp_path / "not_a_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        compute_run_identity(empty)


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param([], id="not_an_object"),
        pytest.param({"simulation_index": {"id": "s1"}}, id="index_not_a_list"),
        pytest.param({"simulation_index": [{"task_id": 1}]}, id="index_entry_no_id"),
        pytest.param({"simulations": [{"task_id": 1}]}, id="inline_sim_no_id"),
        pytest.param({"timestamp": 12345, "simulations": []}, id="timestamp_not_a_str"),
    ],
)
def test_malformed_metadata_raises_value_error_not_key_error(tmp_path, meta):
    """Callers sweep whole run trees and catch ValueError per run, so a
    hand-edited results.json must never escape as a KeyError and abort them."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        compute_run_identity(run)
