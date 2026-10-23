"""Replacement trajectory write failures must preserve the previous checkpoint."""

import json
import multiprocessing

import pytest

from tau2.data_model.simulation import (
    SIMULATIONS_DIR,
    Info,
    Results,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.runner.checkpoint import create_checkpoint_fns


@pytest.mark.parametrize("failure_stage", ["mkstemp", "serialize", "replace"])
@pytest.mark.parametrize("reuse_id", [False, True])
def test_failed_replacement_preserves_checkpoint(
    tmp_path, monkeypatch, failure_stage, reuse_id
):
    """A failed trajectory write must preserve the previous checkpoint."""
    info = Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )
    task = Task(
        id="t0",
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )
    save_path = tmp_path / "results.json"
    results = Results(info=info, tasks=[task], simulations=[])
    results.save(save_path, format="dir")
    save_fn, replace_fn = create_checkpoint_fns(save_path, multiprocessing.Lock())
    original = SimulationRun(
        id="sim-t0-t0-s42",
        task_id="t0",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        trial=0,
        seed=42,
    )
    save_fn(original)
    replacement = original.model_copy(
        update={"id": original.id if reuse_id else "replacement", "duration": 120.0}
    )
    original_metadata = save_path.read_bytes()
    sims_dir = tmp_path / SIMULATIONS_DIR
    original_path = sims_dir / f"{original.id}.json"
    original_bytes = original_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected checkpoint write failure")

    targets = {
        "mkstemp": "tau2.runner.checkpoint.tempfile.mkstemp",
        "serialize": "tau2.data_model.simulation.SimulationRun.model_dump_json",
        "replace": "tau2.runner.checkpoint.os.replace",
    }
    with monkeypatch.context() as patch:
        patch.setattr(targets[failure_stage], fail)
        with pytest.raises(OSError, match="injected checkpoint write failure"):
            replace_fn((0, "t0", 42), replacement)

    assert original_path.read_bytes() == original_bytes
    assert save_path.read_bytes() == original_metadata
    assert list(sims_dir.iterdir()) == [original_path]
    assert Results.load(save_path).simulations == [original]

    # A later retry must still replace the original, without duplicate files.
    replace_fn((0, "t0", 42), replacement)
    assert Results.load(save_path).simulations == [replacement]
    assert len(list(sims_dir.iterdir())) == 1
    metadata = json.loads(save_path.read_text())
    assert [entry["id"] for entry in metadata["simulation_index"]] == [replacement.id]
