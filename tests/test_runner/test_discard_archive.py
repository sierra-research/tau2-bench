"""Tests for the append-safe hallucination discard archive.

Regression coverage for concurrent workers interleaving a rewrite-the-whole-JSON
archive, and a corrupt archive then failing every subsequent discarding unit.
"""

import json
import multiprocessing
import threading
from pathlib import Path

from tau2.data_model.simulation import (
    Info,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.runner.discard_archive import (
    DISCARDED_DIR_NAME,
    archive_discarded_simulation,
    load_discarded_simulations,
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


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(task_id: str, sim_id: str) -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        trial=0,
        seed=42,
    )


def _worker_append(save_dir: str, worker_id: int, n_records: int) -> None:
    """Runs in a child process: append n_records discards."""
    for i in range(n_records):
        path = archive_discarded_simulation(
            save_dir=save_dir,
            info=_make_info(),
            task=_make_task(f"task_w{worker_id}_{i}"),
            simulation=_make_sim(f"task_w{worker_id}_{i}", f"sim_w{worker_id}_{i}"),
        )
        assert path is not None


def _archive_files(save_dir: Path) -> list[Path]:
    return sorted((save_dir / DISCARDED_DIR_NAME).glob("*.jsonl"))


def _assert_every_line_parses(path: Path) -> list[dict]:
    """Every line of a well-formed archive file must be standalone JSON."""
    payloads = []
    with open(path) as fp:
        for line in fp:
            payloads.append(json.loads(line))
    return payloads


class TestConcurrentAppends:
    def test_two_processes_appending_concurrently(self, tmp_path):
        """Two worker processes hammering the archive must not corrupt it."""
        n_records = 20
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_worker_append, args=(str(tmp_path), w, n_records))
            for w in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0

        # Every line of every file is valid standalone JSON, each file starts
        # with exactly one info header, and no record was lost.
        files = _archive_files(tmp_path)
        assert len(files) == 2  # one file per worker pid
        total = 0
        for path in files:
            payloads = _assert_every_line_parses(path)
            assert payloads[0]["record"] == "info"
            body = [p for p in payloads[1:]]
            assert all(p["record"] == "discarded_simulation" for p in body)
            total += len(body)
        assert total == 2 * n_records

        records = load_discarded_simulations(tmp_path)
        assert len(records) == 2 * n_records
        assert len({r.simulation.id for r in records}) == 2 * n_records

    def test_threads_in_one_process_share_one_file(self, tmp_path):
        """Worker threads within a process serialize onto one intact file."""
        n_threads, n_records = 8, 10

        def work(worker_id: int):
            _worker_append(str(tmp_path), worker_id, n_records)

        threads = [threading.Thread(target=work, args=(w,)) for w in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        files = _archive_files(tmp_path)
        assert len(files) == 1
        payloads = _assert_every_line_parses(files[0])
        headers = [p for p in payloads if p["record"] == "info"]
        assert len(headers) == 1
        assert payloads[0]["record"] == "info"
        assert len(payloads) == 1 + n_threads * n_records


class TestCorruptArchiveNeverFailsTheUnit:
    def test_unwritable_target_is_rotated_and_append_succeeds(self, tmp_path):
        """An unusable archive file is rotated to *.corrupt and the discard
        still lands — the simulation unit must never see a failure."""
        # First append establishes the worker's archive path.
        path = archive_discarded_simulation(
            save_dir=tmp_path,
            info=_make_info(),
            task=_make_task("t0"),
            simulation=_make_sim("t0", "s0"),
        )
        assert path is not None

        # Replace it with something unappendable (a directory).
        path.unlink()
        path.mkdir()

        result = archive_discarded_simulation(
            save_dir=tmp_path,
            info=_make_info(),
            task=_make_task("t1"),
            simulation=_make_sim("t1", "s1"),
        )
        assert result == path
        corrupt = path.with_name(path.name + ".corrupt")
        assert corrupt.is_dir()

        payloads = _assert_every_line_parses(path)
        assert payloads[0]["record"] == "info"
        assert payloads[1]["record"] == "discarded_simulation"
        assert payloads[1]["simulation"]["id"] == "s1"

    def test_totally_unwritable_archive_returns_none(self, tmp_path, monkeypatch):
        """If even the rotated retry fails, the unit gets None, not an exception."""
        import tau2.runner.discard_archive as da

        def boom(*args, **kwargs):
            raise OSError("disk on fire")

        monkeypatch.setattr(da, "_append", boom)
        result = archive_discarded_simulation(
            save_dir=tmp_path,
            info=_make_info(),
            task=_make_task("t0"),
            simulation=_make_sim("t0", "s0"),
        )
        assert result is None

    def test_loader_skips_corrupt_lines(self, tmp_path):
        """A partially corrupt archive is read best-effort, never raises."""
        archive_discarded_simulation(
            save_dir=tmp_path,
            info=_make_info(),
            task=_make_task("t0"),
            simulation=_make_sim("t0", "s0"),
        )
        path = _archive_files(tmp_path)[0]
        with open(path, "a") as fp:
            fp.write('{"record": "discarded_simulation", "task": {"truncat\n')
        archive_discarded_simulation(
            save_dir=tmp_path,
            info=_make_info(),
            task=_make_task("t1"),
            simulation=_make_sim("t1", "s1"),
        )

        records = load_discarded_simulations(tmp_path)
        assert {r.simulation.id for r in records} == {"s0", "s1"}
