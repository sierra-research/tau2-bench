"""run_tasks with workers=0 must behave exactly as before the TaskSource seam.

run_single_task is monkeypatched so no LLM is called; what's under test is the
batch loop itself: every (task, trial) executes once, results are
checkpointed, and a resume runs nothing.
"""

import json
import uuid

from tau2.data_model.simulation import (
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.run import get_tasks
from tau2.runner import batch as batch_mod
from tau2.runner.batch import run_tasks


def _make_config(**overrides) -> TextRunConfig:
    defaults = dict(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        task_ids=["create_task_1", "update_task_1"],
        llm_agent="gpt-3.5-turbo",
        llm_args_agent={},
        llm_user="gpt-3.5-turbo",
        llm_args_user={},
        num_trials=2,
        max_steps=20,
        max_errors=10,
        save_to=None,
        max_concurrency=2,
        auto_resume=True,
    )
    defaults.update(overrides)
    return TextRunConfig(**defaults)


def _install_fake_run_single_task(monkeypatch, calls: list):
    def fake_run_single_task(config, task, *, seed=None, **kwargs):
        calls.append((task.id, seed))
        return SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=1.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
            seed=seed,
        )

    monkeypatch.setattr(batch_mod, "run_single_task", fake_run_single_task)


def test_all_task_trials_execute_and_checkpoint(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_run_single_task(monkeypatch, calls)

    config = _make_config()
    tasks = get_tasks("mock", task_ids=config.task_ids)
    save_path = tmp_path / "results.json"

    results = run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=tmp_path,
        console_display=False,
    )

    assert len(calls) == 4  # 2 tasks x 2 trials
    assert len(results.simulations) == 4
    with open(save_path) as f:
        saved = json.load(f)
    assert len(saved["simulations"]) == 4
    pairs = {(s["task_id"], s["trial"]) for s in saved["simulations"]}
    assert len(pairs) == 4


def test_resume_runs_nothing_when_complete(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_run_single_task(monkeypatch, calls)

    config = _make_config()
    tasks = get_tasks("mock", task_ids=config.task_ids)
    save_path = tmp_path / "results.json"

    run_tasks(
        config, tasks, save_path=save_path, save_dir=tmp_path, console_display=False
    )
    first_run_calls = len(calls)

    results = run_tasks(
        config, tasks, save_path=save_path, save_dir=tmp_path, console_display=False
    )

    assert len(calls) == first_run_calls  # nothing re-ran
    assert len(results.simulations) == 4
