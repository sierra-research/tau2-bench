# Copyright Sierra
"""Safety tests for the corrected retail name-role provenance stamp."""

from __future__ import annotations

import hashlib
import json

import pytest

from tau2.data_model.simulation import (
    Results,
    SimulationRun,
    SupersededInfo,
    TaskSubsetInfo,
    TerminationReason,
)
from tau2.multilingual.english_prompts import retail_name_roles_prompt_line
from tau2.registry import registry
from tau2.runner.helpers import get_info
from tau2.runner.name_role_provenance import (
    NameRoleProvenanceError,
    stamp_name_role_provenance,
    write_evidence_report,
)
from tau2.runner.pools import PoolArm, PoolSpec, TaskSelection
from tau2.task_subsets.store import load_subset


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def completed_legacy_cell(tmp_path, monkeypatch):
    monkeypatch.setattr("tau2.runner.pools.DATA_DIR", tmp_path)
    spec = PoolSpec(
        name="test_name_roles",
        description="synthetic corrected-name provenance fixture",
        domain="retail",
        modality="text",
        task_sets={"ko": "retail_ko_identity"},
        roots={"ko": "test_name_roles_korean_retail"},
        arms=(
            PoolArm(
                provider="gpt55",
                reasoning_effort="xhigh",
                agent_llm="gpt-5.5",
            ),
        ),
        tasks=TaskSelection(kind="subset", subset="retail_50"),
        num_trials=1,
        retail_name_roles_prompt_version="v1",
    )
    monkeypatch.setattr(
        "tau2.runner.name_role_provenance.CORRECTED_NAME_ROLE_POOLS",
        frozenset({spec.name}),
    )
    monkeypatch.setattr("tau2.runner.name_role_provenance.get_pool", lambda name: spec)

    language, arm = "ko", spec.arms[0]
    subset = load_subset("retail_50")
    tasks = subset.select(registry.get_tasks_loader("retail_ko_identity")())
    config = spec.cell_config(language, arm)
    info = get_info(config)
    info.task_subset = TaskSubsetInfo(
        name=subset.name,
        size=subset.size,
        tasks_scored=subset.size,
        frame_task_set=subset.frame.task_set,
        frame_size=subset.frame.size,
        frame_digest=subset.frame.digest,
        strategy=subset.strategy.value,
        seed=subset.seed,
    )
    simulations = [
        SimulationRun(
            id=f"sim-{index}",
            task_id=task.id,
            trial=0,
            seed=index,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
        )
        for index, task in enumerate(tasks)
    ]
    results = Results(
        info=info,
        info_history=[SupersededInfo(info=info.model_copy(deep=True))],
        tasks=tasks,
        simulations=simulations,
    )
    run_dir = spec.cell_dir(language, arm)
    results.save(run_dir / "results.json", format="dir")

    for task, simulation in zip(tasks, simulations, strict=True):
        line = retail_name_roles_prompt_line(
            task,
            language=language,
            domain="retail",
            task_set_name="retail_ko_identity",
            version="v1",
        )
        assert line
        debug_dir = (
            run_dir
            / "artifacts"
            / f"task_{task.id}"
            / f"sim_{simulation.id}"
            / "llm_debug"
        )
        debug_dir.mkdir(parents=True)
        (debug_dir / "request.json").write_text(
            json.dumps(
                {
                    "call_id": simulation.id,
                    "call_name": "user_simulator_response",
                    "timestamp": "2026-01-01T00:00:00",
                    "request": {"messages": [{"role": "system", "content": [line]}]},
                    "response": {},
                }
            )
        )

    # Reproduce the completed cells from 758: prompt evidence exists, but the
    # treatment field did not yet exist in current or historical headers.
    results_path = run_dir / "results.json"
    raw = json.loads(results_path.read_text())
    for header in [raw["info"], *[item["info"] for item in raw["info_history"]]]:
        header.pop("retail_name_roles_prompt_version")
        header.pop("retail_name_roles_prompt_backfill", None)
    results_path.write_text(json.dumps(raw, indent=2))
    return spec, results_path


def test_stamp_is_dry_by_default_then_exact_and_idempotent(completed_legacy_cell):
    spec, results_path = completed_legacy_cell
    before = results_path.read_bytes()
    sim_hashes = {
        path.name: _sha(path)
        for path in (results_path.parent / "simulations").glob("*.json")
    }

    dry = stamp_name_role_provenance([spec.name])
    assert {cell.action for cell in dry.cells} == {"would_stamp"}
    assert dry.calls_verified == 50
    assert results_path.read_bytes() == before

    written = stamp_name_role_provenance([spec.name], write=True)
    assert {cell.action for cell in written.cells} == {"stamped"}
    raw = json.loads(results_path.read_text())
    headers = [raw["info"], *[item["info"] for item in raw["info_history"]]]
    assert {header["retail_name_roles_prompt_version"] for header in headers} == {"v1"}
    assert all(header["retail_name_roles_prompt_backfill"] for header in headers)
    assert {
        path.name: _sha(path)
        for path in (results_path.parent / "simulations").glob("*.json")
    } == sim_hashes

    once = results_path.read_bytes()
    again = stamp_name_role_provenance([spec.name], write=True)
    assert {cell.action for cell in again.cells} == {"already_stamped"}
    assert results_path.read_bytes() == once


def test_stamp_rejects_serialized_null_treatment(completed_legacy_cell):
    spec, results_path = completed_legacy_cell
    raw = json.loads(results_path.read_text())
    headers = [raw["info"], *[item["info"] for item in raw["info_history"]]]
    for header in headers:
        header["retail_name_roles_prompt_version"] = None
        header["retail_name_roles_prompt_backfill"] = None
    results_path.write_text(json.dumps(raw, indent=2))
    before = results_path.read_bytes()

    with pytest.raises(NameRoleProvenanceError, match="explicit prompt-off treatment"):
        stamp_name_role_provenance([spec.name], write=True)

    assert results_path.read_bytes() == before


def test_stamp_rejects_noncanonical_retail_subset_ids(completed_legacy_cell):
    spec, results_path = completed_legacy_cell
    raw = json.loads(results_path.read_text())
    outside = next(
        task
        for task in registry.get_tasks_loader("retail_ko_identity")()
        if task.id not in {stored["id"] for stored in raw["tasks"]}
    )
    old_id = raw["tasks"][-1]["id"]
    raw["tasks"][-1] = outside.model_dump(mode="json")
    for row in raw["simulation_index"]:
        if row["task_id"] == old_id:
            row["task_id"] = outside.id
    results_path.write_text(json.dumps(raw, indent=2))

    with pytest.raises(NameRoleProvenanceError, match="canonical task ids"):
        stamp_name_role_provenance([spec.name])


def test_stamp_rejects_noncanonical_retail_subset_digest(completed_legacy_cell):
    spec, results_path = completed_legacy_cell
    raw = json.loads(results_path.read_text())
    raw["info"]["task_subset"]["frame_digest"] = "0" * 64
    results_path.write_text(json.dumps(raw, indent=2))

    with pytest.raises(NameRoleProvenanceError, match="task-subset provenance"):
        stamp_name_role_provenance([spec.name])


def test_wrong_prompt_evidence_fails_before_any_write(completed_legacy_cell):
    spec, results_path = completed_legacy_cell
    log = next(results_path.parent.glob("artifacts/task_*/sim_*/llm_debug/*.json"))
    payload = json.loads(log.read_text())
    payload["request"]["messages"][0]["content"] = ["wrong prompt"]
    log.write_text(json.dumps(payload))
    before = results_path.read_bytes()

    with pytest.raises(NameRoleProvenanceError, match="exact v1 name-role line"):
        stamp_name_role_provenance([spec.name], write=True)

    assert results_path.read_bytes() == before


def test_evidence_sidecar_is_typed_exclusive_and_outside_source_root(
    completed_legacy_cell, tmp_path
):
    spec, results_path = completed_legacy_cell
    before = results_path.read_bytes()
    report = stamp_name_role_provenance([spec.name])
    evidence_path = tmp_path / "name-role-evidence.json"

    assert write_evidence_report(report, evidence_path) == evidence_path
    payload = json.loads(evidence_path.read_text())
    assert payload["treatment"] == "v1"
    assert payload["template_sha256"]
    assert payload["cells"][0]["source_results_sha256"] == _sha(results_path)
    assert results_path.read_bytes() == before
    with pytest.raises(NameRoleProvenanceError, match="refusing to overwrite"):
        write_evidence_report(report, evidence_path)
    with pytest.raises(NameRoleProvenanceError, match="outside verified run roots"):
        write_evidence_report(report, results_path.parent / "evidence.json")
