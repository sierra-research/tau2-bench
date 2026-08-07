"""Tests for the parallel-run controller and worker loop.

Everything runs in-process: the controller's FastAPI app is exercised through
httpx's ASGI transport (no sockets, no subprocesses), and the worker loop's
simulation execution is monkeypatched so no LLM is called.
"""

import json
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from tau2.data_model.simulation import SimulationRun, TerminationReason, TextRunConfig
from tau2.run import get_tasks
from tau2.runner import worker as worker_mod
from tau2.runner.batch import prepare_batch
from tau2.runner.controller import Controller, ControllerRun
from tau2.runner.work import WorkUnit
from tau2.runner.worker import ControllerClient, worker_loop


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
        num_trials=1,
        max_steps=20,
        max_errors=10,
        save_to=None,
        max_concurrency=2,
        auto_resume=True,
    )
    defaults.update(overrides)
    return TextRunConfig(**defaults)


def _make_sim(
    unit: WorkUnit,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=str(uuid.uuid4()),
        task_id=unit.task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        messages=[],
        trial=unit.trial,
        seed=unit.seed,
    )


@pytest.fixture
def controller(tmp_path):
    config = _make_config()
    tasks = get_tasks("mock", task_ids=config.task_ids)
    save_path = tmp_path / "results.json"
    prep = prepare_batch(
        config,
        tasks,
        save_path=save_path,
        save_dir=tmp_path,
        console_display=False,
    )
    run = ControllerRun.from_prep("test_run", prep)
    ctrl = Controller([run], max_attempts=2)
    ctrl.save_path = save_path  # for test assertions only
    return ctrl


def _client(ctrl: Controller) -> httpx.Client:
    return TestClient(ctrl.build_app())


def _load_sims(save_path) -> list[dict]:
    with open(save_path) as f:
        return json.load(f)["simulations"]


class TestControllerHttpContract:
    def test_lease_returns_unit_config_and_task(self, controller):
        with _client(controller) as client:
            resp = client.post("/lease", json={"worker_id": "w1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "unit"
            unit = body["unit"]
            assert unit["run_id"] == "test_run"
            run = body["run"]
            assert run["config_kind"] == "text"
            assert run["config"]["domain"] == "mock"
            assert run["task"]["id"] == unit["task_id"]
            # The CLI's --llm-log-mode only reaches worker processes through
            # the lease payload ("latest" is the ContextVar default).
            assert run["llm_log_mode"] == "latest"

    def test_complete_writes_exactly_one_checkpoint_entry(self, controller):
        with _client(controller) as client:
            body = client.post("/lease", json={"worker_id": "w1"}).json()
            unit = WorkUnit.model_validate(body["unit"])
            sim = _make_sim(unit)
            resp = client.post(
                "/complete",
                json={
                    "worker_id": "w1",
                    "unit_id": unit.unit_id,
                    "result": sim.model_dump(mode="json"),
                },
            )
            assert resp.json()["status"] == "ok"
            sims = _load_sims(controller.save_path)
            assert len(sims) == 1
            assert sims[0]["task_id"] == unit.task_id

            # A duplicate complete must not write a second entry.
            resp = client.post(
                "/complete",
                json={
                    "worker_id": "w1",
                    "unit_id": unit.unit_id,
                    "result": sim.model_dump(mode="json"),
                },
            )
            assert resp.json()["status"] == "stale"
            assert len(_load_sims(controller.save_path)) == 1

    def test_infra_error_result_requeues_then_writes_when_exhausted(self, controller):
        with _client(controller) as client:
            body = client.post("/lease", json={"worker_id": "w1"}).json()
            unit = WorkUnit.model_validate(body["unit"])
            infra_sim = _make_sim(
                unit, termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
            )

            resp = client.post(
                "/complete",
                json={
                    "worker_id": "w1",
                    "unit_id": unit.unit_id,
                    "result": infra_sim.model_dump(mode="json"),
                },
            )
            assert resp.json()["status"] == "requeued"
            assert _load_sims(controller.save_path) == []

            # The requeued attempt comes back around; max_attempts=2 means this
            # infra failure is final and must be checkpointed.
            leases = [client.post("/lease", json={"worker_id": "w2"}).json()]
            if leases[0]["unit"]["task_id"] != unit.task_id:
                leases.append(client.post("/lease", json={"worker_id": "w2"}).json())
            retry_unit = WorkUnit.model_validate(leases[-1]["unit"])
            assert retry_unit.task_id == unit.task_id
            assert retry_unit.attempt == 1

            infra_sim_2 = _make_sim(
                retry_unit, termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
            )
            resp = client.post(
                "/complete",
                json={
                    "worker_id": "w2",
                    "unit_id": retry_unit.unit_id,
                    "result": infra_sim_2.model_dump(mode="json"),
                },
            )
            assert resp.json()["status"] == "ok"
            sims = _load_sims(controller.save_path)
            infra_written = [
                s for s in sims if s["termination_reason"] == "infrastructure_error"
            ]
            assert len(infra_written) == 1

    def test_fail_requeues_then_dead_unit_gets_placeholder_sim(self, controller):
        with _client(controller) as client:
            body = client.post("/lease", json={"worker_id": "w1"}).json()
            unit = WorkUnit.model_validate(body["unit"])

            resp = client.post(
                "/fail",
                json={"worker_id": "w1", "unit_id": unit.unit_id, "error": "boom"},
            )
            assert resp.json()["status"] == "requeued"

            leases = [client.post("/lease", json={"worker_id": "w1"}).json()]
            if leases[0]["unit"]["task_id"] != unit.task_id:
                leases.append(client.post("/lease", json={"worker_id": "w1"}).json())
            retry_unit = WorkUnit.model_validate(leases[-1]["unit"])
            resp = client.post(
                "/fail",
                json={
                    "worker_id": "w1",
                    "unit_id": retry_unit.unit_id,
                    "error": "boom again",
                },
            )
            assert resp.json()["status"] == "dead"

        # Draining flushes placeholder infra sims for dead units.
        controller.flush_dead_units()
        sims = _load_sims(controller.save_path)
        placeholder = [s for s in sims if s["task_id"] == unit.task_id]
        assert len(placeholder) == 1
        assert placeholder[0]["termination_reason"] == "infrastructure_error"

    def test_status_endpoint_reports_counts(self, controller):
        with _client(controller) as client:
            client.post("/lease", json={"worker_id": "w1"})
            counts = client.get("/status").json()["counts"]
            assert counts["leased"] == 1
            assert counts["pending"] == 1

    def test_done_when_all_resolved(self, controller):
        with _client(controller) as client:
            while True:
                body = client.post("/lease", json={"worker_id": "w1"}).json()
                if body["status"] != "unit":
                    break
                unit = WorkUnit.model_validate(body["unit"])
                client.post(
                    "/complete",
                    json={
                        "worker_id": "w1",
                        "unit_id": unit.unit_id,
                        "result": _make_sim(unit).model_dump(mode="json"),
                    },
                )
            assert body["status"] == "done"


class TestWorkerLoop:
    def test_worker_loop_completes_all_units(self, controller, monkeypatch):
        executed: list[str] = []

        def fake_execute(payload: dict) -> SimulationRun:
            unit = WorkUnit.model_validate(payload["unit"])
            executed.append(unit.task_id)
            return _make_sim(unit)

        monkeypatch.setattr(worker_mod, "execute_lease", fake_execute)

        with TestClient(controller.build_app()) as http_client:
            client = ControllerClient(client=http_client)
            exit_code = worker_loop(client, worker_id="w1", slots=2)

        assert exit_code == 0
        assert sorted(executed) == ["create_task_1", "update_task_1"]
        assert len(_load_sims(controller.save_path)) == 2
        assert controller.queue.all_resolved()

    def test_worker_loop_reports_execution_errors_as_fail(
        self, controller, monkeypatch
    ):
        def broken_execute(payload: dict) -> SimulationRun:
            raise RuntimeError("worker exploded")

        monkeypatch.setattr(worker_mod, "execute_lease", broken_execute)

        with TestClient(controller.build_app()) as http_client:
            client = ControllerClient(client=http_client)
            exit_code = worker_loop(client, worker_id="w1", slots=1)

        assert exit_code == 0
        # max_attempts=2, both attempts fail -> both units dead.
        assert controller.queue.all_resolved()
        assert len(controller.queue.dead_units) == 2
        controller.flush_dead_units()
        sims = _load_sims(controller.save_path)
        assert len(sims) == 2
        assert all(s["termination_reason"] == "infrastructure_error" for s in sims)


class _RecordingClient:
    """Stands in for ControllerClient in HeartbeatThread tests."""

    def __init__(self, leases_by_call: list[dict]):
        self.calls: list[list[str]] = []
        self._leases_by_call = leases_by_call

    def heartbeat(self, worker_id: str, unit_ids: list[str]) -> dict:
        self.calls.append(list(unit_ids))
        return {
            "leases": self._leases_by_call[
                min(len(self.calls) - 1, len(self._leases_by_call) - 1)
            ]
        }


class TestHeartbeatThread:
    def test_beat_posts_inflight_ids(self):
        client = _RecordingClient([{"u1": True, "u2": True}])
        hb = worker_mod.HeartbeatThread(
            client, "w1", lambda: ["u1", "u2"], interval=999
        )
        hb.beat()
        hb.beat()
        assert client.calls == [["u1", "u2"], ["u1", "u2"]]
        assert hb.lost == set()

    def test_lost_lease_is_remembered_and_not_beaten_again(self):
        client = _RecordingClient([{"u1": False, "u2": True}, {"u2": True}])
        hb = worker_mod.HeartbeatThread(
            client, "w1", lambda: ["u1", "u2"], interval=999
        )
        hb.beat()
        assert hb.lost == {"u1"}
        hb.beat()
        assert client.calls[1] == ["u2"]

    def test_heartbeat_errors_do_not_kill_the_thread(self):
        class ExplodingClient:
            def heartbeat(self, worker_id, unit_ids):
                raise RuntimeError("controller unreachable")

        hb = worker_mod.HeartbeatThread(
            ExplodingClient(), "w1", lambda: ["u1"], interval=999
        )
        hb.beat()  # must not raise
        assert hb.lost == set()

    def test_thread_beats_periodically_until_stopped(self):
        client = _RecordingClient([{"u1": True}])
        hb = worker_mod.HeartbeatThread(client, "w1", lambda: ["u1"], interval=0.01)
        hb.start()
        deadline = time.monotonic() + 2.0
        while len(client.calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        hb.stop()
        hb.join(timeout=2.0)
        assert len(client.calls) >= 3
        assert not hb.is_alive()


class _RecordingMonitor:
    def __init__(self):
        self.started: list[str] = []
        self.finished: list[str] = []

    def task_started(self, key, trial):
        self.started.append(key)

    def task_finished(self, key):
        self.finished.append(key)

    def task_restarted(self, key):
        pass


class TestMultiRunMonitorKeys:
    def test_monitor_keys_are_run_scoped_so_runs_do_not_collide(self, tmp_path):
        """Two runs share task ids; 4 in-flight sims must show as 4, not 2."""
        monitor = _RecordingMonitor()
        runs = []
        for name in ["run_a", "run_b"]:
            config = _make_config()
            tasks = get_tasks("mock", task_ids=config.task_ids)
            prep = prepare_batch(
                config,
                tasks,
                save_path=tmp_path / name / "results.json",
                save_dir=tmp_path / name,
                console_display=False,
            )
            runs.append(ControllerRun.from_prep(name, prep))
        ctrl = Controller(runs, max_attempts=2, monitor=monitor)

        with TestClient(ctrl.build_app()) as client:
            leased = []
            while True:
                body = client.post("/lease", json={"worker_id": "w1"}).json()
                if body["status"] != "unit":
                    break
                leased.append(WorkUnit.model_validate(body["unit"]))
            assert len(leased) == 4
            assert len(set(monitor.started)) == 4

            for unit in leased:
                client.post(
                    "/complete",
                    json={
                        "worker_id": "w1",
                        "unit_id": unit.unit_id,
                        "result": _make_sim(unit).model_dump(mode="json"),
                    },
                )
        assert sorted(monitor.finished) == sorted(monitor.started)

    def test_releasing_a_lost_unit_resumes_its_heartbeats(self):
        """Unit ids omit the attempt number: a requeued unit re-leased by the
        same worker must be heartbeated again, not stay muted forever."""
        client = _RecordingClient([{"u1": False}, {"u1": True}])
        hb = worker_mod.HeartbeatThread(client, "w1", lambda: ["u1"], interval=999)
        hb.beat()
        assert hb.lost == {"u1"}
        hb.leased("u1")
        hb.beat()
        assert client.calls[-1] == ["u1"]
        assert hb.lost == set()
