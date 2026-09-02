"""Parallel-run controller: schedules work units and writes all checkpoints.

The controller owns the queue and the results files; worker processes
(``tau2 worker`` / ``python -m tau2.runner.worker``) lease units over HTTP,
execute them, and post results back. One controller can drive many runs at
once — that is what makes per-provider caps and the global in-flight cap
actually global. See docs/designs/parallel-runner.md.

Locally, ``tau2 run --workers N`` binds 127.0.0.1 on an ephemeral port and
spawns its own workers. Remote workers are the same binary pointed at a
routable address, authenticated with TAU2_CONTROLLER_TOKEN.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from loguru import logger

from tau2.data_model.simulation import (
    Results,
    RunConfig,
    SimulationRun,
    TerminationReason,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import BatchPrep
from tau2.runner.progress import StatusMonitor
from tau2.runner.work import (
    DEFAULT_LEASE_TTL_SECONDS,
    DEFAULT_UNIT_ATTEMPTS,
    FailOutcome,
    WorkQueue,
    WorkUnit,
)
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.llm_utils import llm_log_mode
from tau2.utils.utils import DATA_DIR, get_now

AUTH_TOKEN_ENV = "TAU2_CONTROLLER_TOKEN"

# How long drive_batches waits for workers to notice "done" and exit on their
# own before terminating them.
WORKER_EXIT_GRACE_SECONDS = 30.0


def make_infra_failure_sim(
    task_id: str, trial: int, seed: int, error: str
) -> SimulationRun:
    """Placeholder result for a unit that died in the queue (worker crashes,
    expired leases) — same shape run_with_retry produces on exhausted retries,
    so resume repairs it the usual way."""
    now = get_now()
    return SimulationRun(
        id=str(uuid.uuid4()),
        task_id=task_id,
        timestamp=now,
        start_time=now,
        end_time=now,
        duration=0.0,
        termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
        messages=[],
        trial=trial,
        seed=seed,
        info={"error": error, "error_type": "WorkerFailure"},
    )


class ControllerRun:
    """One registered run: its config, tasks, results, and checkpoint writer."""

    def __init__(
        self,
        run_id: str,
        config: RunConfig,
        tasks: list[Task],
        simulation_results: Results,
        units: list[WorkUnit],
        save_fn: Optional[Callable],
        save_dir: Optional[Path],
        evaluation_type: EvaluationType,
    ):
        self.run_id = run_id
        self.config = config
        self.tasks_by_id = {task.id: task for task in tasks}
        self.simulation_results = simulation_results
        # Re-key units onto this run_id so a prep prepared under a default id
        # can be registered under a controller-unique one.
        self.units = [
            u.model_copy(update={"run_id": run_id, "unit_id": _rekey(u, run_id)})
            for u in units
        ]
        self.save_fn = save_fn
        self.save_dir = save_dir
        self.evaluation_type = evaluation_type
        # Captured at registration time (main thread): uvicorn handler threads
        # get a fresh ContextVar context, and workers are separate processes,
        # so the CLI's --llm-log-mode has to travel in the lease payload.
        self.llm_log_mode = llm_log_mode.get()

    @classmethod
    def from_prep(cls, run_id: str, prep: BatchPrep) -> "ControllerRun":
        return cls(
            run_id=run_id,
            config=prep.config,
            tasks=prep.tasks,
            simulation_results=prep.simulation_results,
            units=prep.units,
            save_fn=prep.save_fn,
            save_dir=prep.save_dir,
            evaluation_type=prep.evaluation_type,
        )

    def lease_payload(self, unit: WorkUnit) -> dict:
        """Everything a worker needs to execute the unit: the run config, the
        task, and where artifacts go. Workers share the filesystem locally;
        cross-machine artifact upload is a later addition."""
        return {
            "config_kind": "voice"
            if isinstance(self.config, VoiceRunConfig)
            else "text",
            "config": self.config.model_dump(mode="json"),
            "task": self.tasks_by_id[unit.task_id].model_dump(mode="json"),
            "evaluation_type": self.evaluation_type.value,
            "save_dir": str(self.save_dir) if self.save_dir else None,
            "llm_log_mode": self.llm_log_mode,
        }


def _rekey(unit: WorkUnit, run_id: str) -> str:
    from tau2.runner.work import make_unit_id

    return make_unit_id(run_id, unit.task_id, unit.trial)


class Controller:
    """Queue + policy + single checkpoint writer, exposed as a FastAPI app."""

    def __init__(
        self,
        runs: list[ControllerRun],
        *,
        provider_limits: Optional[dict[str, int]] = None,
        global_limit: Optional[int] = None,
        max_attempts: int = DEFAULT_UNIT_ATTEMPTS,
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        monitor: Optional[StatusMonitor] = None,
    ):
        if len({run.run_id for run in runs}) != len(runs):
            raise ValueError("Controller runs must have unique run_ids")
        self.runs = {run.run_id: run for run in runs}
        self.queue = WorkQueue(
            [unit for run in runs for unit in run.units],
            max_attempts=max_attempts,
            lease_ttl_seconds=lease_ttl_seconds,
            provider_limits=provider_limits,
            global_limit=global_limit,
        )
        self.lease_ttl_seconds = lease_ttl_seconds
        self.monitor = monitor
        self._written: set[str] = set()
        self._write_lock = threading.Lock()
        self._units_by_id = {unit.unit_id: unit for run in runs for unit in run.units}
        self._run_by_unit_id = {unit.unit_id: run for run in runs for unit in run.units}

    # ---- Core handlers (transport-independent) ----

    def _monitor_key(self, unit: WorkUnit) -> str:
        # task_id.trial collides across runs (every run has a task 0), which
        # made 8 in-flight sims display as 6 — so multi-run controllers scope
        # the key by run. Single runs keep the familiar short form.
        if len(self.runs) > 1:
            return f"{unit.run_id}/{unit.task_id}.{unit.trial}"
        return f"{unit.task_id}.{unit.trial}"

    def handle_lease(self, worker_id: str) -> dict:
        unit = self.queue.lease(worker_id)
        if unit is None:
            if self.queue.all_resolved():
                return {"status": "done"}
            return {"status": "wait"}
        run = self.runs[unit.run_id]
        if self.monitor:
            self.monitor.task_started(self._monitor_key(unit), unit.trial)
        return {
            "status": "unit",
            "unit": unit.model_dump(mode="json"),
            "run": run.lease_payload(unit),
            "lease_ttl_seconds": self.lease_ttl_seconds,
        }

    def handle_complete(self, worker_id: str, unit_id: str, result: dict) -> dict:
        sim = SimulationRun.model_validate(result)
        run = self._run_for_unit(unit_id)
        if run is None:
            return {"status": "stale"}
        unit = self._unit_from_id(unit_id)

        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
            # Don't checkpoint a result that can't enter an analysis frame —
            # requeue the unit instead. Only the final attempt is written, so
            # the run completes with an honest record of the failure.
            outcome = self.queue.fail(
                unit_id, "infrastructure_error result", worker_id=worker_id
            )
            if outcome == FailOutcome.REQUEUED:
                if self.monitor:
                    self.monitor.task_restarted(self._monitor_key(unit))
                return {"status": "requeued"}
            if outcome == FailOutcome.STALE:
                return {"status": "stale"}
            self._write(run, unit_id, sim)
            return {"status": "ok"}

        if not self.queue.complete(unit_id, worker_id=worker_id):
            return {"status": "stale"}
        self._write(run, unit_id, sim)
        return {"status": "ok"}

    def handle_fail(self, worker_id: str, unit_id: str, error: str) -> dict:
        outcome = self.queue.fail(unit_id, error, worker_id=worker_id)
        if outcome == FailOutcome.REQUEUED and self.monitor:
            unit = self._unit_from_id(unit_id)
            if unit:
                self.monitor.task_restarted(self._monitor_key(unit))
        return {"status": outcome.value}

    def handle_heartbeat(self, worker_id: str, unit_ids: list[str]) -> dict:
        return {
            "leases": {
                unit_id: self.queue.heartbeat(unit_id, worker_id=worker_id)
                for unit_id in unit_ids
            }
        }

    def flush_dead_units(self) -> int:
        """Write placeholder infra sims for units that died in the queue
        (worker crashes, expired leases) and were never checkpointed."""
        flushed = 0
        for unit, error in list(self.queue.dead_units):
            run = self.runs.get(unit.run_id)
            if run is None or unit.unit_id in self._written:
                continue
            sim = make_infra_failure_sim(unit.task_id, unit.trial, unit.seed, error)
            self._write(run, unit.unit_id, sim)
            flushed += 1
        return flushed

    def _write(self, run: ControllerRun, unit_id: str, sim: SimulationRun) -> None:
        with self._write_lock:
            if unit_id in self._written:
                return
            self._written.add(unit_id)
            if run.save_fn:
                run.save_fn(sim)
            run.simulation_results.simulations.append(sim)
        if self.monitor:
            unit = self._unit_from_id(unit_id)
            key = self._monitor_key(unit) if unit else f"{sim.task_id}.{sim.trial}"
            self.monitor.task_finished(key)

    def _run_for_unit(self, unit_id: str) -> Optional[ControllerRun]:
        return self._run_by_unit_id.get(unit_id)

    def _unit_from_id(self, unit_id: str) -> Optional[WorkUnit]:
        return self._units_by_id.get(unit_id)

    # ---- HTTP layer ----

    def build_app(self) -> FastAPI:
        app = FastAPI(title="tau2 controller")
        expected_token = os.environ.get(AUTH_TOKEN_ENV)

        def check_auth(request: Request) -> None:
            if not expected_token:
                return
            header = request.headers.get("authorization", "")
            if header != f"Bearer {expected_token}":
                raise HTTPException(status_code=401, detail="bad or missing token")

        @app.post("/lease")
        async def lease(request: Request):
            check_auth(request)
            body = await request.json()
            return self.handle_lease(body["worker_id"])

        @app.post("/complete")
        async def complete(request: Request):
            check_auth(request)
            body = await request.json()
            return self.handle_complete(
                body["worker_id"], body["unit_id"], body["result"]
            )

        @app.post("/fail")
        async def fail(request: Request):
            check_auth(request)
            body = await request.json()
            return self.handle_fail(
                body["worker_id"], body["unit_id"], body.get("error", "")
            )

        @app.post("/heartbeat")
        async def heartbeat(request: Request):
            check_auth(request)
            body = await request.json()
            return self.handle_heartbeat(body["worker_id"], body["unit_ids"])

        @app.get("/status")
        async def status(request: Request):
            check_auth(request)
            return {
                "counts": self.queue.counts(),
                "done": self.queue.all_resolved(),
            }

        return app


# =============================================================================
# Driving a controller with locally-spawned workers
# =============================================================================


def _free_port(host: str) -> int:
    sock = socket.socket()
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _start_server(app, host: str, port: int):
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Controller HTTP server failed to start")
        time.sleep(0.05)
    return server


def _spawn_worker(
    controller_url: str, slots: int, index: int, log_dir: Path
) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{index}.log"
    log_file = open(log_path, "w")
    cmd = [
        sys.executable,
        "-m",
        "tau2.runner.worker",
        "--controller",
        controller_url,
        "--slots",
        str(slots),
        "--worker-id",
        f"local-{index}",
    ]
    logger.info(f"Spawning worker {index}: {' '.join(cmd)} (log: {log_path})")
    return subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy()
    )


def drive_batches(
    preps: list[BatchPrep],
    *,
    workers: int,
    slots: int,
    provider_limits: Optional[dict[str, int]] = None,
    global_limit: Optional[int] = None,
    monitor: Optional[StatusMonitor] = None,
    worker_log_dir: Optional[Path] = None,
    host: str = "127.0.0.1",
) -> Controller:
    """Drive prepared runs to completion with locally-spawned worker processes.

    Fills each prep's ``simulation_results`` in place (results are also
    checkpointed as they arrive, exactly as in local mode).
    """
    if workers <= 0:
        raise ValueError("drive_batches needs workers >= 1")

    runs = [ControllerRun.from_prep(prep.run_id, prep) for prep in preps]
    controller = Controller(
        runs,
        provider_limits=provider_limits,
        global_limit=global_limit,
        monitor=monitor,
    )

    total_units = sum(len(run.units) for run in runs)
    if total_units == 0:
        return controller

    port = _free_port(host)
    server = _start_server(controller.build_app(), host, port)
    controller_url = f"http://{host}:{port}"

    if worker_log_dir is None:
        first_save_dir = next(
            (run.save_dir for run in runs if run.save_dir is not None), None
        )
        worker_log_dir = (
            first_save_dir / "workers"
            if first_save_dir
            else DATA_DIR / "simulations" / "workers"
        )

    procs = [
        _spawn_worker(controller_url, slots, i, worker_log_dir) for i in range(workers)
    ]
    ConsoleDisplay.console.print(
        Text(
            text=f"Controller on {controller_url}: {total_units} unit(s), "
            f"{workers} worker(s) x {slots} slot(s). Worker logs: {worker_log_dir}",
            style="bold green",
        )
    )

    try:
        while not controller.queue.all_resolved():
            if all(proc.poll() is not None for proc in procs):
                # Every worker exited while work remains: reap what their
                # leases owe, then surface the failure with the logs to read.
                controller.flush_dead_units()
                if not controller.queue.all_resolved():
                    raise RuntimeError(
                        f"All {workers} workers exited with work remaining "
                        f"({controller.queue.counts()}). "
                        f"See logs in {worker_log_dir}"
                    )
                break
            time.sleep(2.0)

        # Workers exit on their own once the queue reports done.
        deadline = time.monotonic() + WORKER_EXIT_GRACE_SECONDS
        while any(proc.poll() is None for proc in procs):
            if time.monotonic() > deadline:
                for proc in procs:
                    if proc.poll() is None:
                        proc.terminate()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        ConsoleDisplay.console.print(
            "\n[bold red]Ctrl+C received — stopping workers...[/bold red]"
        )
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        n = sum(len(run.simulation_results.simulations) for run in runs)
        ConsoleDisplay.console.print(
            f"[bold yellow]{n} simulation(s) already checkpointed. "
            f"Use --auto-resume to continue later.[/bold yellow]"
        )
        os._exit(130)
    finally:
        controller.flush_dead_units()
        server.should_exit = True

    return controller
