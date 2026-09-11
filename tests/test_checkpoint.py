"""Tests for checkpoint save/resume logic."""

import json
import multiprocessing
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tau2.config import VOICE_TIMEOUT_SAFETY_FACTOR
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Info,
    Results,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import (
    EvaluationCriteria,
    Task,
    UserScenario,
)
from tau2.environment.environment import EnvironmentInfo
from tau2.runner import checkpoint as checkpoint_module
from tau2.runner.checkpoint import (
    create_checkpoint_replacer,
    create_checkpoint_saver,
    drop_simulations,
    repair_ceiling,
    try_resume,
)


def _make_info() -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )


def _make_voice_info(max_steps_seconds: int) -> Info:
    """A voice run header whose ceiling-derived fields all agree, the way
    `get_info` writes them (max_steps in ticks, timeout scaled off the
    conversation budget)."""
    audio_cfg = AudioNativeConfig(max_steps_seconds=max_steps_seconds)
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=audio_cfg.max_steps_ticks,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
        timeout=max_steps_seconds * VOICE_TIMEOUT_SAFETY_FACTOR,
        audio_native_config=audio_cfg,
    )


def _make_task(task_id: str, instructions: str = "test instruction") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions=instructions),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(
    task_id: str,
    trial: int = 0,
    seed: int = 42,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        messages=[],
        trial=trial,
        seed=seed,
    )


class TestTryResumeInfraErrorRemoval:
    """Test that try_resume properly handles infrastructure error retries."""

    def test_infra_errors_excluded_from_done_runs(self, tmp_path):
        """Infrastructure error sims should not be in done_runs so tasks get retried."""
        tasks = [_make_task("t0"), _make_task("t1"), _make_task("t2")]
        info = _make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                _make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
                _make_sim("t2", termination_reason=TerminationReason.USER_STOP),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        resumed, done_runs, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )

        done_task_ids = {task_id for _, task_id, _ in done_runs}
        assert "t0" in done_task_ids
        assert "t1" not in done_task_ids, (
            "Infrastructure error task should NOT be in done_runs"
        )
        assert "t2" in done_task_ids

    def test_infra_errors_removed_from_resumed_simulations(self, tmp_path):
        """Infrastructure error sims should be removed from the resumed results."""
        tasks = [_make_task("t0"), _make_task("t1")]
        info = _make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                _make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        resumed, _, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )

        assert len(resumed.simulations) == 1
        assert resumed.simulations[0].task_id == "t0"

    def test_infra_errors_removed_from_disk(self, tmp_path):
        """After try_resume, the on-disk file should NOT contain infra error sims.

        This is the core regression test: without the fix, the on-disk file
        retained infra error entries, causing create_checkpoint_saver's
        duplicate check to reject the retried results.
        """
        tasks = [_make_task("t0"), _make_task("t1")]
        info = _make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                _make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["task_id"] == "t0"

    def test_checkpoint_saver_works_after_infra_error_resume(self, tmp_path):
        """End-to-end: after resuming, saving a retried sim should succeed."""
        tasks = [_make_task("t0"), _make_task("t1")]
        info = _make_info()
        seed = 42

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                _make_sim("t0", seed=seed),
                _make_sim(
                    "t1",
                    seed=seed,
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        retried_sim = _make_sim("t1", seed=seed)
        save_fn(retried_sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 2
        task_ids = {s["task_id"] for s in on_disk["simulations"]}
        assert task_ids == {"t0", "t1"}
        t1_sim = next(s for s in on_disk["simulations"] if s["task_id"] == "t1")
        assert t1_sim["termination_reason"] == "user_stop"

    def test_no_infra_errors_skips_resave(self, tmp_path):
        """When there are no infra errors and no new tasks, don't rewrite the file."""
        tasks = [_make_task("t0")]
        info = _make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[_make_sim("t0")],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        original_mtime = save_path.stat().st_mtime

        new_results = Results(info=info, tasks=tasks, simulations=[])

        import time

        time.sleep(0.05)

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        assert save_path.stat().st_mtime == original_mtime


class TestCheckpointSaver:
    """Tests for checkpoint save function."""

    def test_saves_new_simulation(self, tmp_path):
        save_path = tmp_path / "results.json"
        data = {"simulations": [], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        sim = _make_sim("t0")
        save_fn(sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["task_id"] == "t0"

    def test_skips_duplicate(self, tmp_path):
        sim = _make_sim("t0")
        save_path = tmp_path / "results.json"
        data = {"simulations": [sim.model_dump()], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        save_fn(sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1


class TestCheckpointReplacer:
    """Tests for checkpoint replace function."""

    def test_replaces_existing_simulation(self, tmp_path):
        old_sim = _make_sim(
            "t0", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
        )
        save_path = tmp_path / "results.json"
        data = {"simulations": [old_sim.model_dump()], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        replace_fn = create_checkpoint_replacer(save_path, lock)

        new_sim = _make_sim("t0")
        replace_fn((0, "t0", 42), new_sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["termination_reason"] == "user_stop"


class TestDropSimulations:
    """`tau2 drop-sims`: explicit removal of finished sims by termination."""

    def _write_json_checkpoint(self, tmp_path, sims):
        tasks = [_make_task(s.task_id) for s in sims]
        results = Results(info=_make_info(), tasks=tasks, simulations=sims)
        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(results.model_dump_json(indent=2))
        return save_path, tasks

    def test_drops_matching_terminations_json_format(self, tmp_path):
        sims = [
            _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t1", termination_reason=TerminationReason.TIMEOUT),
            _make_sim("t2", termination_reason=TerminationReason.MAX_STEPS),
        ]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(
            save_path,
            terminations=[TerminationReason.TIMEOUT, TerminationReason.MAX_STEPS],
        )

        assert sorted(d[0] for d in dropped) == ["t1", "t2"]
        on_disk = Results.load(save_path)
        assert [s.task_id for s in on_disk.simulations] == ["t0"]

    def test_no_match_is_a_noop(self, tmp_path):
        sims = [_make_sim("t0", termination_reason=TerminationReason.USER_STOP)]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)
        before = save_path.read_text()

        assert drop_simulations(save_path, [TerminationReason.TIMEOUT]) == []
        assert save_path.read_text() == before

    def test_dir_format_deletes_files_and_rebuilds_index(self, tmp_path):
        sims = [
            _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t1", termination_reason=TerminationReason.TIMEOUT),
        ]
        tasks = [_make_task(s.task_id) for s in sims]
        results = Results(info=_make_info(), tasks=tasks, simulations=sims)
        save_path = tmp_path / "results.json"
        results.save(save_path, format="dir")

        dropped = drop_simulations(save_path, [TerminationReason.TIMEOUT])

        assert [d[0] for d in dropped] == ["t1"]
        sims_dir = tmp_path / "simulations"
        remaining_files = {f.stem for f in sims_dir.glob("*.json")}
        assert remaining_files == {"sim-t0-0"}
        # Results.load errors on an index/file mismatch, so a clean load
        # proves the index was rebuilt alongside the file deletion.
        on_disk = Results.load(save_path)
        assert [s.task_id for s in on_disk.simulations] == ["t0"]
        assert {e.id for e in on_disk.simulation_index} == {"sim-t0-0"}

    def test_drops_by_task_id(self, tmp_path):
        sims = [
            _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t1", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t2", termination_reason=TerminationReason.TIMEOUT),
        ]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(save_path, task_ids=["t0", "t2"])

        assert sorted(d[0] for d in dropped) == ["t0", "t2"]
        on_disk = Results.load(save_path)
        assert [s.task_id for s in on_disk.simulations] == ["t1"]

    def test_task_and_termination_filters_intersect(self, tmp_path):
        sims = [
            _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t1", termination_reason=TerminationReason.TIMEOUT),
        ]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(
            save_path,
            terminations=[TerminationReason.TIMEOUT],
            task_ids=["t0"],
        )
        assert dropped == []

    def test_no_filter_raises(self, tmp_path):
        sims = [_make_sim("t0", termination_reason=TerminationReason.USER_STOP)]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)
        with pytest.raises(ValueError, match="terminations, task_ids or trials"):
            drop_simulations(save_path)

    def test_drops_by_trial(self, tmp_path):
        sims = [
            _make_sim("t0", trial=0),
            _make_sim("t0", trial=1),
            _make_sim("t1", trial=1),
        ]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(save_path, trials=[1])

        assert sorted((d[0], d[1]) for d in dropped) == [("t0", 1), ("t1", 1)]
        on_disk = Results.load(save_path)
        assert [(s.task_id, s.trial) for s in on_disk.simulations] == [("t0", 0)]

    def test_a_sim_with_no_trial_is_trial_zero(self, tmp_path):
        # Callers name that cell trial 0 (refill.SimulationCell does), so the
        # filter has to agree: a plan that says "delete trial 0" and a delete
        # that skips it leaves the old simulation beside the new one.
        sims = [_make_sim("t0", trial=None), _make_sim("t0", trial=1)]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(save_path, trials=[0])

        assert [(d[0], d[1]) for d in dropped] == [("t0", None)]
        assert [s.trial for s in Results.load(save_path).simulations] == [1]

    def test_task_and_trial_filters_intersect(self, tmp_path):
        # The filters AND, so a refill that means "trial 1 of t0" must pass
        # both — passing every task id and every trial at once would delete
        # the cross product (see tau2.runner.refill.execute_refill).
        sims = [
            _make_sim("t0", trial=0),
            _make_sim("t0", trial=1),
            _make_sim("t1", trial=0),
            _make_sim("t1", trial=1),
        ]
        save_path, _ = self._write_json_checkpoint(tmp_path, sims)

        dropped = drop_simulations(save_path, task_ids=["t0"], trials=[1])

        assert [(d[0], d[1]) for d in dropped] == [("t0", 1)]
        on_disk = Results.load(save_path)
        assert sorted((s.task_id, s.trial) for s in on_disk.simulations) == [
            ("t0", 0),
            ("t1", 0),
            ("t1", 1),
        ]

    def test_dropped_cells_rerun_on_resume(self, tmp_path):
        """The point of the verb: dropped (trial, task, seed) cells leave
        done_runs, so --auto-resume re-runs exactly them."""
        sims = [
            _make_sim("t0", termination_reason=TerminationReason.USER_STOP),
            _make_sim("t1", termination_reason=TerminationReason.TIMEOUT),
        ]
        save_path, tasks = self._write_json_checkpoint(tmp_path, sims)
        drop_simulations(save_path, [TerminationReason.TIMEOUT])

        new_results = Results(info=_make_info(), tasks=tasks, simulations=[])
        _, done_runs, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )
        assert {task_id for _, task_id, _ in done_runs} == {"t0"}


class TestRedoStaleTasks:
    """Resuming a checkpoint whose task TEXT changed underneath it.

    The telecom name-auth redo: three merged fixes rewrote the name-auth arm's
    scenario text, so every recorded sim of those tasks measures text that no
    longer exists. ``redo_stale_tasks`` drops exactly those sims and refreshes
    the checkpoint's task snapshot so the redo runs the CURRENT text.
    """

    @staticmethod
    def _checkpoint(tmp_path, tasks, sims, fmt="json"):
        results = Results(info=_make_info(), tasks=tasks, simulations=sims)
        save_path = tmp_path / "results.json"
        results.save(save_path, format=fmt)
        return save_path

    def test_modified_task_still_raises_without_the_flag(self, tmp_path):
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0"), _make_sim("t1")]
        )

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        with pytest.raises(ValueError, match="Tasks were modified"):
            try_resume(save_path, new_results, changed, num_trials=1, auto_resume=True)

    def test_stale_task_sims_leave_done_runs(self, tmp_path):
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0"), _make_sim("t1")]
        )

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        resumed, done_runs, out_tasks = try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        assert {task_id for _, task_id, _ in done_runs} == {"t0"}
        assert [s.task_id for s in resumed.simulations] == ["t0"]

    def test_resumed_checkpoint_carries_the_current_text(self, tmp_path):
        """The crux: try_resume hands the runner its own task list, so a stale
        task must be REPLACED — otherwise the redo re-runs the stale text."""
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0"), _make_sim("t1")]
        )

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        resumed, _, out_tasks = try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        by_id = {t.id: t for t in out_tasks}
        assert by_id["t1"].user_scenario.instructions == "rewritten"
        assert {t.id: t.user_scenario.instructions for t in resumed.tasks} == {
            "t0": "test instruction",
            "t1": "rewritten",
        }

    def test_dir_format_deletes_stale_sim_files_and_rebuilds_index(self, tmp_path):
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0"), _make_sim("t1")], fmt="dir"
        )

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        assert {f.stem for f in (tmp_path / "simulations").glob("*.json")} == {
            "sim-t0-0"
        }
        # Results.load errors on an index/file mismatch, so a clean load proves
        # the index was rebuilt alongside the deletion.
        on_disk = Results.load(save_path)
        assert [s.task_id for s in on_disk.simulations] == ["t0"]
        assert {t.id: t.user_scenario.instructions for t in on_disk.tasks} == {
            "t0": "test instruction",
            "t1": "rewritten",
        }

    def test_stale_and_infra_error_drops_compose(self, tmp_path):
        tasks = [_make_task("t0"), _make_task("t1"), _make_task("t2")]
        save_path = self._checkpoint(
            tmp_path,
            tasks,
            [
                _make_sim("t0"),
                _make_sim("t1"),
                _make_sim(
                    "t2", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
                ),
            ],
        )

        changed = [
            _make_task("t0"),
            _make_task("t1", instructions="rewritten"),
            _make_task("t2"),
        ]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        resumed, done_runs, _ = try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        assert {task_id for _, task_id, _ in done_runs} == {"t0"}
        assert [s.task_id for s in resumed.simulations] == ["t0"]

    def test_refreshed_task_with_no_recorded_sims_is_persisted(self, tmp_path):
        """B1/T2: the case that drops nothing and adds nothing.

        `tau2 drop-sims --tasks` (the workaround this flag replaces) leaves
        exactly this state behind: the stale task's simulations are already
        gone, so the resave guard sees no dropped sims and no added tasks. If
        staleness is not its own trigger the refreshed text lives only in
        memory — the run does the right thing, the checkpoint keeps the old
        text, and a crash before the next flush makes the next resume raise
        `Tasks were modified` on a task that was already redone.
        """
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(tmp_path, tasks, [_make_sim("t0")])

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        on_disk = Results.load(save_path)
        assert {t.id: t.user_scenario.instructions for t in on_disk.tasks} == {
            "t0": "test instruction",
            "t1": "rewritten",
        }
        # And the proof that it stuck: a second resume of the CURRENT text no
        # longer sees a modified task, with or without the flag.
        try_resume(
            save_path,
            Results(info=_make_info(), tasks=changed, simulations=[]),
            changed,
            num_trials=1,
            auto_resume=True,
        )

    def test_refreshed_task_with_no_sims_persists_in_dir_format(self, tmp_path):
        """Same case, dir format — the resave goes through save_metadata."""
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(tmp_path, tasks, [_make_sim("t0")], fmt="dir")

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        try_resume(
            save_path,
            new_results,
            changed,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        on_disk = Results.load(save_path)
        assert [s.task_id for s in on_disk.simulations] == ["t0"]
        assert {t.id: t.user_scenario.instructions for t in on_disk.tasks} == {
            "t0": "test instruction",
            "t1": "rewritten",
        }

    def test_drop_counts_are_disjoint_when_a_sim_is_both(self, tmp_path):
        """B2: a sim can be an infrastructure error AND belong to a stale task.

        The two logged counts are what the redo's verification rests on, so
        they must partition the deletion rather than double-count it: booked
        once, under infra (it would have been dropped either way), and the
        stale count reports only what redoing the stale tasks cost on top.
        """
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path,
            tasks,
            [
                _make_sim("t0"),
                # t1 is stale AND infra-errored: both predicates, one deletion.
                _make_sim(
                    "t1", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
                ),
                # a second stale sim of t1 that is NOT an infra error
                _make_sim("t1", trial=1),
            ],
        )

        changed = [_make_task("t0"), _make_task("t1", instructions="rewritten")]
        new_results = Results(info=_make_info(), tasks=changed, simulations=[])

        # Capture the module's own logger rather than adding a loguru sink:
        # another test module disables tau2 logging process-wide at import
        # time, so a sink would come back empty depending on run order.
        lines: list[str] = []
        recorder = SimpleNamespace(info=lines.append, warning=lines.append)
        with patch.object(checkpoint_module, "logger", recorder):
            resumed, _, _ = try_resume(
                save_path,
                new_results,
                changed,
                num_trials=1,
                auto_resume=True,
                redo_stale_tasks=True,
            )

        assert [s.task_id for s in resumed.simulations] == ["t0"]
        text = "\n".join(lines)
        assert "Removed 1 infrastructure error simulation(s)" in text
        # Not 2 - 1 = 1 double-counted, and not 0: exactly the one stale sim
        # that infra had not already accounted for.
        assert "Removed 1 simulation(s) of 1 stale task(s)" in text

    def test_unchanged_tasks_do_not_trigger_a_resave(self, tmp_path):
        tasks = [_make_task("t0")]
        save_path = self._checkpoint(tmp_path, tasks, [_make_sim("t0")])
        before = save_path.read_text()

        new_results = Results(info=_make_info(), tasks=tasks, simulations=[])
        _, done_runs, _ = try_resume(
            save_path,
            new_results,
            tasks,
            num_trials=1,
            auto_resume=True,
            redo_stale_tasks=True,
        )

        assert {task_id for _, task_id, _ in done_runs} == {"t0"}
        assert save_path.read_text() == before


class TestResumeConfigDrift:
    """A resume that proceeds under a changed run config must leave a header
    that states what actually ran.

    The pt airline rerun (2026-08-27): cells whose results.json was created by
    a `tau2 pool run --max-steps-seconds 1200` invocation and resumed to
    completion at 2400 kept `max_steps_seconds=1200` / `max_steps=6000` in the
    header while every resumed call ran under the 2400 s ceiling. The header
    must follow the invocation now producing simulations, with the displaced
    header kept in info_history so neither regime is lost.
    """

    @staticmethod
    def _checkpoint(tmp_path, tasks, sims, info, fmt="json"):
        results = Results(info=info, tasks=tasks, simulations=sims)
        save_path = tmp_path / "results.json"
        results.save(save_path, format=fmt)
        return save_path

    def test_header_rewritten_and_prior_config_kept(self, tmp_path):
        tasks = [_make_task("t0"), _make_task("t1")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0")], _make_voice_info(1200)
        )

        new_results = Results(info=_make_voice_info(2400), tasks=tasks, simulations=[])
        resumed, done_runs, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )

        assert resumed.info.audio_native_config.max_steps_seconds == 2400
        # The resave must happen even though nothing was dropped or added —
        # config drift is its own trigger.
        on_disk = Results.load(save_path)
        assert on_disk.info.audio_native_config.max_steps_seconds == 2400
        assert on_disk.info.max_steps == 12000
        assert on_disk.info.timeout == 2400 * VOICE_TIMEOUT_SAFETY_FACTOR
        # The displaced header is on record, not overwritten.
        assert [
            h.info.audio_native_config.max_steps_seconds for h in on_disk.info_history
        ] == [1200]
        assert on_disk.info_history[0].superseded_at
        # The surviving simulation is untouched and still done.
        assert [s.task_id for s in on_disk.simulations] == ["t0"]
        assert {task_id for _, task_id, _ in done_runs} == {"t0"}

    def test_dir_format_header_rewrite_keeps_index_consistent(self, tmp_path):
        tasks = [_make_task("t0")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0")], _make_voice_info(1200), fmt="dir"
        )

        new_results = Results(info=_make_voice_info(2400), tasks=tasks, simulations=[])
        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        # Results.load errors on an index/file mismatch, so a clean load proves
        # the metadata rewrite left the index and sim files consistent.
        on_disk = Results.load(save_path)
        assert on_disk.info.audio_native_config.max_steps_seconds == 2400
        assert [
            h.info.audio_native_config.max_steps_seconds for h in on_disk.info_history
        ] == [1200]
        assert [s.task_id for s in on_disk.simulations] == ["t0"]

    def test_second_resume_under_the_same_config_appends_nothing(self, tmp_path):
        tasks = [_make_task("t0")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0")], _make_voice_info(1200)
        )

        try_resume(
            save_path,
            Results(info=_make_voice_info(2400), tasks=tasks, simulations=[]),
            tasks,
            num_trials=1,
            auto_resume=True,
        )
        before = save_path.read_text()

        try_resume(
            save_path,
            Results(info=_make_voice_info(2400), tasks=tasks, simulations=[]),
            tasks,
            num_trials=1,
            auto_resume=True,
        )

        # The rewritten header now matches the invocation, so there is no
        # drift, no new history entry, and no resave.
        assert save_path.read_text() == before
        assert len(Results.load(save_path).info_history) == 1

    def test_unchanged_config_leaves_header_and_history_alone(self, tmp_path):
        tasks = [_make_task("t0")]
        save_path = self._checkpoint(
            tmp_path, tasks, [_make_sim("t0")], _make_voice_info(1200)
        )
        before = save_path.read_text()

        resumed, _, _ = try_resume(
            save_path,
            Results(info=_make_voice_info(1200), tasks=tasks, simulations=[]),
            tasks,
            num_trials=1,
            auto_resume=True,
        )

        assert resumed.info_history == []
        assert save_path.read_text() == before


class TestRepairCeiling:
    """`tau2 pool repair-ceiling`: the backfill for cells written before
    try_resume rewrote headers on config drift (including cells already
    archived), applying the same correction after the fact."""

    @staticmethod
    def _checkpoint(tmp_path, info, fmt="dir"):
        tasks = [_make_task("t0")]
        results = Results(info=info, tasks=tasks, simulations=[_make_sim("t0")])
        save_path = tmp_path / "results.json"
        results.save(save_path, format=fmt)
        return save_path

    def test_rewrites_ceiling_and_derived_fields(self, tmp_path):
        save_path = self._checkpoint(tmp_path, _make_voice_info(1200))

        report = repair_ceiling(save_path, 2400)

        assert report.changed
        on_disk = Results.load(save_path)
        assert on_disk.info.audio_native_config.max_steps_seconds == 2400
        assert on_disk.info.max_steps == 12000
        assert on_disk.info.timeout == 2400 * VOICE_TIMEOUT_SAFETY_FACTOR
        assert [
            h.info.audio_native_config.max_steps_seconds for h in on_disk.info_history
        ] == [1200]
        # Everything but the header is byte-for-byte what it was: the clean
        # load above already proves index/sim-file consistency.
        assert [s.task_id for s in on_disk.simulations] == ["t0"]
        # And the header carries what the repair did not claim to know better:
        # the original commit stays.
        assert on_disk.info.git_commit == "abc123"

    def test_noop_and_dry_run_write_nothing(self, tmp_path):
        save_path = self._checkpoint(tmp_path, _make_voice_info(1200))
        before = save_path.read_text()

        noop = repair_ceiling(save_path, 1200)
        assert not noop.changed
        assert save_path.read_text() == before

        dry = repair_ceiling(save_path, 2400, dry_run=True)
        assert dry.changed
        assert dry.new_max_steps == 12000
        assert save_path.read_text() == before

    def test_monolithic_json_keeps_its_simulations(self, tmp_path):
        save_path = self._checkpoint(tmp_path, _make_voice_info(1200), fmt="json")

        repair_ceiling(save_path, 2400)

        on_disk = Results.load(save_path)
        assert on_disk.info.audio_native_config.max_steps_seconds == 2400
        assert [s.task_id for s in on_disk.simulations] == ["t0"]

    def test_text_header_is_refused(self, tmp_path):
        save_path = self._checkpoint(tmp_path, _make_info())

        with pytest.raises(ValueError, match="no audio_native_config"):
            repair_ceiling(save_path, 2400)

    def test_missing_results_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no stored results"):
            repair_ceiling(tmp_path / "nowhere" / "results.json", 2400)
