"""Tests for Results storage format: directory-based format, format detection,
round-trip compatibility, streaming APIs, and checkpoint operations."""

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
from tau2.runner.checkpoint import create_checkpoint_fns, try_resume
from tau2.utils.io_utils import load_results_dict

# ---- Fixtures ----


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


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(
    task_id: str,
    trial: int = 0,
    seed: int = 42,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}-t{trial}-s{seed}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        messages=[],
        trial=trial,
        seed=seed,
    )


@pytest.fixture
def sample_results():
    return Results(
        info=_make_info(),
        tasks=[_make_task("t0"), _make_task("t1")],
        simulations=[
            _make_sim("t0", trial=0, seed=42),
            _make_sim("t1", trial=0, seed=42),
        ],
    )


# ---- Format detection ----


class TestFormatDetection:
    def test_json_file_detected_as_json(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text("{}")
        assert Results._detect_format(p) == "json"

    def test_directory_detected_as_dir(self, tmp_path):
        assert Results._detect_format(tmp_path) == "dir"

    def test_json_with_sibling_sims_dir_detected_as_dir(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text("{}")
        (tmp_path / SIMULATIONS_DIR).mkdir()
        assert Results._detect_format(p) == "dir"

    def test_resolve_paths_from_json(self, tmp_path):
        p = tmp_path / "results.json"
        meta, sims = Results._resolve_paths(p)
        assert meta == p
        assert sims == tmp_path / SIMULATIONS_DIR

    def test_resolve_paths_from_directory(self, tmp_path):
        meta, sims = Results._resolve_paths(tmp_path)
        assert meta == tmp_path / "results.json"
        assert sims == tmp_path / SIMULATIONS_DIR


# ---- Round-trip: JSON format ----


class TestJsonFormat:
    def test_save_and_load(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="json")
        loaded = Results.load(p)
        assert len(loaded.simulations) == 2
        assert loaded.info.git_commit == "abc123"
        assert {s.id for s in loaded.simulations} == {
            s.id for s in sample_results.simulations
        }

    def test_creates_parent_dirs(self, tmp_path, sample_results):
        p = tmp_path / "nested" / "dir" / "results.json"
        sample_results.save(p, format="json")
        assert p.exists()


# ---- Round-trip: directory format ----


class TestDirFormat:
    def test_save_creates_structure(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        assert p.exists()
        sims_dir = tmp_path / SIMULATIONS_DIR
        assert sims_dir.is_dir()
        sim_files = list(sims_dir.glob("*.json"))
        assert len(sim_files) == 2

    def test_metadata_excludes_simulations(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        with open(p) as f:
            meta = json.load(f)
        assert "simulations" not in meta
        assert "info" in meta
        assert "tasks" in meta

    def test_load_from_json_path(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        loaded = Results.load(p)
        assert len(loaded.simulations) == 2
        assert loaded.info.git_commit == "abc123"

    def test_load_from_directory_path(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        loaded = Results.load(tmp_path)
        assert len(loaded.simulations) == 2

    def test_sim_files_named_by_id(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        sims_dir = tmp_path / SIMULATIONS_DIR
        expected = {f"{s.id}.json" for s in sample_results.simulations}
        actual = {f.name for f in sims_dir.glob("*.json")}
        assert actual == expected

    def test_empty_simulations(self, tmp_path):
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        p = tmp_path / "results.json"
        results.save(p, format="dir")
        loaded = Results.load(p)
        assert len(loaded.simulations) == 0
        assert len(loaded.tasks) == 1


# ---- Cross-format round-trip ----


class TestCrossFormat:
    def test_json_to_dir(self, tmp_path, sample_results):
        json_path = tmp_path / "json" / "results.json"
        dir_path = tmp_path / "dir" / "results.json"
        sample_results.save(json_path, format="json")
        loaded = Results.load(json_path)
        loaded.save(dir_path, format="dir")
        reloaded = Results.load(dir_path)
        assert len(reloaded.simulations) == 2
        assert {s.id for s in reloaded.simulations} == {
            s.id for s in sample_results.simulations
        }

    def test_dir_to_json(self, tmp_path, sample_results):
        dir_path = tmp_path / "dir" / "results.json"
        json_path = tmp_path / "json" / "results.json"
        sample_results.save(dir_path, format="dir")
        loaded = Results.load(dir_path)
        loaded.save(json_path, format="json")
        reloaded = Results.load(json_path)
        assert len(reloaded.simulations) == 2


# ---- Streaming APIs ----


class TestStreamingAPIs:
    def test_load_metadata_json(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="json")
        meta = Results.load_metadata(p)
        assert len(meta.simulations) == 0
        assert len(meta.tasks) == 2
        assert meta.info.git_commit == "abc123"

    def test_load_metadata_dir(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        meta = Results.load_metadata(p)
        assert len(meta.simulations) == 0
        assert len(meta.tasks) == 2

    def test_iter_simulations_json(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="json")
        sims = list(Results.iter_simulations(p))
        assert len(sims) == 2
        assert all(isinstance(s, SimulationRun) for s in sims)

    def test_iter_simulations_dir(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        sims = list(Results.iter_simulations(p))
        assert len(sims) == 2
        assert all(isinstance(s, SimulationRun) for s in sims)

    def test_save_metadata_only(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save_metadata(p)
        assert p.exists()
        assert (tmp_path / SIMULATIONS_DIR).is_dir()
        sims = list((tmp_path / SIMULATIONS_DIR).glob("*.json"))
        assert len(sims) == 0
        meta = Results.load_metadata(p)
        assert len(meta.tasks) == 2


# ---- load_results_dict helper ----


class TestLoadResultsDict:
    def test_json_format(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="json")
        data = load_results_dict(p)
        assert isinstance(data, dict)
        assert len(data["simulations"]) == 2
        assert "info" in data

    def test_dir_format(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        data = load_results_dict(p)
        assert isinstance(data, dict)
        assert len(data["simulations"]) == 2
        assert "format_version" not in data

    def test_dir_format_from_directory(self, tmp_path, sample_results):
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        data = load_results_dict(tmp_path)
        assert len(data["simulations"]) == 2


# ---- Checkpoint with dir format ----


class TestCheckpointDirFormat:
    def test_try_resume_creates_dir_format(self, tmp_path):
        save_path = tmp_path / "results.json"
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[],
        )
        resumed, done_runs, tasks = try_resume(
            save_path,
            results,
            results.tasks,
            1,
            auto_resume=True,
            results_format="dir",
        )
        assert save_path.exists()
        assert (tmp_path / SIMULATIONS_DIR).is_dir()
        assert len(done_runs) == 0

    def test_checkpoint_saver_dir_format(self, tmp_path):
        save_path = tmp_path / "results.json"
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        results.save(save_path, format="dir")

        lock = multiprocessing.Lock()
        save_fn, _ = create_checkpoint_fns(save_path, lock)

        sim = _make_sim("t0", trial=0, seed=42)
        save_fn(sim)

        sims_dir = tmp_path / SIMULATIONS_DIR
        sim_files = list(sims_dir.glob("*.json"))
        assert len(sim_files) == 1
        assert sim_files[0].name == f"{sim.id}.json"

        loaded = Results.load(save_path)
        assert len(loaded.simulations) == 1
        assert loaded.simulations[0].task_id == "t0"

    def test_checkpoint_saver_dedup_dir_format(self, tmp_path):
        save_path = tmp_path / "results.json"
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        results.save(save_path, format="dir")

        lock = multiprocessing.Lock()
        save_fn, _ = create_checkpoint_fns(save_path, lock)

        sim = _make_sim("t0", trial=0, seed=42)
        save_fn(sim)
        save_fn(sim)  # duplicate

        sims_dir = tmp_path / SIMULATIONS_DIR
        assert len(list(sims_dir.glob("*.json"))) == 1

    def test_checkpoint_replacer_dir_format(self, tmp_path):
        save_path = tmp_path / "results.json"
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        results.save(save_path, format="dir")

        lock = multiprocessing.Lock()
        save_fn, replace_fn = create_checkpoint_fns(save_path, lock)

        original = _make_sim("t0", trial=0, seed=42)
        save_fn(original)

        replacement = SimulationRun(
            id="sim-replacement",
            task_id="t0",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:02:00",
            duration=120.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
            trial=0,
            seed=42,
        )
        replace_fn((0, "t0", 42), replacement)

        sims_dir = tmp_path / SIMULATIONS_DIR
        sim_files = list(sims_dir.glob("*.json"))
        assert len(sim_files) == 1
        assert sim_files[0].name == "sim-replacement.json"

    def test_try_resume_dir_format_removes_infra_errors(self, tmp_path):
        save_path = tmp_path / "results.json"
        tasks = [_make_task("t0"), _make_task("t1")]
        info = _make_info()

        ok_sim = _make_sim("t0", termination_reason=TerminationReason.USER_STOP)
        err_sim = _make_sim(
            "t1", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
        )

        results = Results(info=info, tasks=tasks, simulations=[ok_sim, err_sim])
        results.save(save_path, format="dir")

        new_results = Results(info=info, tasks=tasks, simulations=[])
        resumed, done_runs, _ = try_resume(
            save_path, new_results, tasks, 1, auto_resume=True
        )

        assert (0, "t0", 42) in done_runs
        assert (0, "t1", 42) not in done_runs
        assert len(resumed.simulations) == 1

        sims_dir = tmp_path / SIMULATIONS_DIR
        sim_files = list(sims_dir.glob("*.json"))
        assert len(sim_files) == 1

    def test_try_resume_dir_format_adds_new_tasks(self, tmp_path):
        save_path = tmp_path / "results.json"
        tasks = [_make_task("t0")]
        info = _make_info()

        results = Results(
            info=info,
            tasks=tasks,
            simulations=[_make_sim("t0")],
        )
        results.save(save_path, format="dir")

        new_tasks = [_make_task("t0"), _make_task("t1")]
        new_results = Results(info=info, tasks=new_tasks, simulations=[])
        _, done_runs, updated_tasks = try_resume(
            save_path, new_results, new_tasks, 1, auto_resume=True
        )

        assert len(updated_tasks) == 2
        reloaded = Results.load_metadata(save_path)
        assert len(reloaded.tasks) == 2


# ---- Simulation index ----


class TestSimulationIndex:
    def test_dir_save_writes_index(self, tmp_path, sample_results):
        """Dir-format save must write simulation_index into results.json."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        with open(p) as f:
            meta = json.load(f)
        assert "simulation_index" in meta
        assert len(meta["simulation_index"]) == 2
        ids = {e["id"] for e in meta["simulation_index"]}
        assert ids == {s.id for s in sample_results.simulations}

    def test_index_entry_fields(self, tmp_path, sample_results):
        """Each index entry should have the expected lightweight fields."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        with open(p) as f:
            meta = json.load(f)
        entry = meta["simulation_index"][0]
        for field in ("id", "task_id", "trial", "termination_reason", "duration"):
            assert field in entry

    def test_json_save_excludes_index(self, tmp_path, sample_results):
        """Monolithic JSON format should not have simulation_index at top level
        (it stays None and is included in the full dump only if set)."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="json")
        with open(p) as f:
            data = json.load(f)
        # simulation_index should be null/None in JSON format
        assert data.get("simulation_index") is None

    def test_load_validates_index_ok(self, tmp_path, sample_results):
        """Load should succeed when index matches simulation files exactly."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        loaded = Results.load(p)
        assert len(loaded.simulations) == 2
        assert loaded.simulation_index is not None
        assert len(loaded.simulation_index) == 2

    def test_load_raises_on_missing_sim_file(self, tmp_path, sample_results):
        """Load should raise ValueError when a file listed in index is missing."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        # Delete one sim file
        sims_dir = tmp_path / SIMULATIONS_DIR
        sim_files = list(sims_dir.glob("*.json"))
        sim_files[0].unlink()
        with pytest.raises(ValueError, match="Missing simulation files"):
            Results.load(p)

    def test_load_raises_on_extra_sim_file(self, tmp_path, sample_results):
        """Load should raise ValueError when a sim file exists but is not in index."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        # Add an extra sim file not in the index
        sims_dir = tmp_path / SIMULATIONS_DIR
        extra = sims_dir / "sim-rogue.json"
        extra.write_text('{"id": "sim-rogue"}')
        with pytest.raises(ValueError, match="Extra simulation files not in index"):
            Results.load(p)

    def test_load_backward_compat_no_index(self, tmp_path, sample_results):
        """Load should work fine when simulation_index is absent (old data)."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        # Strip the index from results.json to simulate old format
        with open(p) as f:
            meta = json.load(f)
        del meta["simulation_index"]
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        loaded = Results.load(p)
        assert len(loaded.simulations) == 2
        assert loaded.simulation_index is None

    def test_save_metadata_preserves_index(self, tmp_path, sample_results):
        """save_metadata should preserve existing on-disk simulation_index
        when the in-memory index is None."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        # Read back the saved index
        with open(p) as f:
            original_meta = json.load(f)
        original_index = original_meta["simulation_index"]
        assert len(original_index) == 2

        # Create a new Results with no index and call save_metadata
        fresh = Results(
            info=_make_info(),
            tasks=[_make_task("t0"), _make_task("t1")],
            simulations=[],
        )
        assert fresh.simulation_index is None
        fresh.save_metadata(p)

        with open(p) as f:
            updated_meta = json.load(f)
        assert updated_meta["simulation_index"] == original_index

    def test_load_metadata_includes_index(self, tmp_path, sample_results):
        """load_metadata should populate simulation_index from dir format."""
        p = tmp_path / "results.json"
        sample_results.save(p, format="dir")
        meta = Results.load_metadata(p)
        assert len(meta.simulations) == 0
        assert meta.simulation_index is not None
        assert len(meta.simulation_index) == 2

    def test_checkpoint_save_updates_index(self, tmp_path):
        """Checkpoint save should append to simulation_index in results.json."""
        save_path = tmp_path / "results.json"
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        results.save(save_path, format="dir")

        lock = multiprocessing.Lock()
        save_fn, _ = create_checkpoint_fns(save_path, lock)

        sim = _make_sim("t0", trial=0, seed=42)
        save_fn(sim)

        with open(save_path) as f:
            meta = json.load(f)
        assert len(meta["simulation_index"]) == 1
        assert meta["simulation_index"][0]["id"] == sim.id

    def test_checkpoint_replace_updates_index(self, tmp_path):
        """Checkpoint replace should update the index entry."""
        save_path = tmp_path / "results.json"
        results = Results(info=_make_info(), tasks=[_make_task("t0")], simulations=[])
        results.save(save_path, format="dir")

        lock = multiprocessing.Lock()
        save_fn, replace_fn = create_checkpoint_fns(save_path, lock)

        original = _make_sim("t0", trial=0, seed=42)
        save_fn(original)

        replacement = SimulationRun(
            id="sim-replacement",
            task_id="t0",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:02:00",
            duration=120.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
            trial=0,
            seed=42,
        )
        replace_fn((0, "t0", 42), replacement)

        with open(save_path) as f:
            meta = json.load(f)
        assert len(meta["simulation_index"]) == 1
        assert meta["simulation_index"][0]["id"] == "sim-replacement"

    def test_try_resume_rebuilds_index_after_infra_removal(self, tmp_path):
        """try_resume should rebuild the index after removing infra-error sims."""
        save_path = tmp_path / "results.json"
        tasks = [_make_task("t0"), _make_task("t1")]
        info = _make_info()

        ok_sim = _make_sim("t0", termination_reason=TerminationReason.USER_STOP)
        err_sim = _make_sim(
            "t1", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
        )
        results = Results(info=info, tasks=tasks, simulations=[ok_sim, err_sim])
        results.save(save_path, format="dir")

        new_results = Results(info=info, tasks=tasks, simulations=[])
        try_resume(save_path, new_results, tasks, 1, auto_resume=True)

        with open(save_path) as f:
            meta = json.load(f)
        assert len(meta["simulation_index"]) == 1
        assert meta["simulation_index"][0]["id"] == ok_sim.id
