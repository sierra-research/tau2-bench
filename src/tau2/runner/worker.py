"""tau2 worker process: leases simulation units from a controller, runs them,
and posts results back. It never touches results.json — the controller is the
single checkpoint writer. Kill a worker and its leases expire back into the
queue; nothing is lost.

Run as ``tau2 worker --controller http://host:port --slots 10`` or
``python -m tau2.runner.worker ...``. See docs/designs/parallel-runner.md.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from tau2.data_model.simulation import SimulationRun, TextRunConfig, VoiceRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.work import WorkUnit

AUTH_TOKEN_ENV = "TAU2_CONTROLLER_TOKEN"

DEFAULT_SLOTS = 10
HEARTBEAT_INTERVAL_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 1.0
POST_RETRIES = 3

# /complete and /fail carry a full SimulationRun (voice sims with verbose tick
# logs run to many MB) and the controller may be mid-checkpoint-write when
# the request lands, so read/write get minutes, not the httpx 5s default.
HTTP_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# /lease and /heartbeat are tiny; failing fast and retrying beats waiting
# minutes on a stuck controller.
CONTROL_TIMEOUT_SECONDS = 30.0


class ControllerClient:
    """HTTP TaskSource client. ``client`` is injectable for in-process tests
    (httpx ASGITransport)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        token: Optional[str] = None,
    ):
        token = token or os.environ.get(AUTH_TOKEN_ENV)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = client or httpx.Client(
            base_url=base_url, headers=headers, timeout=HTTP_TIMEOUT
        )

    def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(POST_RETRIES):
            try:
                kwargs = {} if timeout is None else {"timeout": timeout}
                response = self._client.post(path, json=payload, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                if attempt < POST_RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
        raise last_error

    def lease(self, worker_id: str) -> dict:
        return self._post(
            "/lease", {"worker_id": worker_id}, timeout=CONTROL_TIMEOUT_SECONDS
        )

    def complete(self, worker_id: str, unit_id: str, result: dict) -> dict:
        return self._post(
            "/complete",
            {"worker_id": worker_id, "unit_id": unit_id, "result": result},
        )

    def fail(self, worker_id: str, unit_id: str, error: str) -> dict:
        return self._post(
            "/fail", {"worker_id": worker_id, "unit_id": unit_id, "error": error}
        )

    def heartbeat(self, worker_id: str, unit_ids: list[str]) -> dict:
        return self._post(
            "/heartbeat",
            {"worker_id": worker_id, "unit_ids": unit_ids},
            timeout=CONTROL_TIMEOUT_SECONDS,
        )


class HeartbeatThread(threading.Thread):
    """Keeps leases alive independently of the worker's main loop.

    The main loop blocks on /complete posts (multi-MB voice results) and on
    execute_lease setup, so heartbeating from there starves under load: in the
    first field run a ~3-minute stall expired live leases and the controller
    re-ran sims that were still going. A lease the controller reports lost is
    remembered so its zombie sim stops being heartbeated and its eventual
    result is expected to come back stale.
    """

    def __init__(
        self,
        client: ControllerClient,
        worker_id: str,
        inflight_ids,
        interval: float,
    ):
        super().__init__(daemon=True, name="tau2-worker-heartbeat")
        self._client = client
        self._worker_id = worker_id
        self._inflight_ids = inflight_ids
        self._interval = interval
        self._stop_event = threading.Event()
        self.lost: set[str] = set()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            self.beat()

    def beat(self) -> None:
        unit_ids = [u for u in self._inflight_ids() if u not in self.lost]
        if not unit_ids:
            return
        try:
            leases = self._client.heartbeat(self._worker_id, unit_ids)["leases"]
        except Exception as e:
            logger.warning(f"Heartbeat failed (will retry): {e}")
            return
        for unit_id, alive in leases.items():
            if not alive:
                self.lost.add(unit_id)
                logger.warning(
                    f"Lease lost for {unit_id}; the controller has requeued it "
                    "and this attempt's result will be discarded as stale"
                )

    def leased(self, unit_id: str) -> None:
        """A fresh lease supersedes a lost one. Unit ids omit the attempt
        number, so a requeued unit re-leased by this same worker would
        otherwise stay muted forever and expire mid-sim again."""
        self.lost.discard(unit_id)

    def stop(self) -> None:
        self._stop_event.set()


def execute_lease(payload: dict) -> SimulationRun:
    """Execute one leased unit: rebuild the run context from the payload and
    run the same code the local loop runs. No checkpoint fns, no monitor —
    the result goes back to the controller."""
    from tau2.runner.batch import _BatchContext, make_voice_run_settings, run_unit
    from tau2.runner.helpers import get_info

    unit = WorkUnit.model_validate(payload["unit"])
    run = payload["run"]
    config_cls = VoiceRunConfig if run["config_kind"] == "voice" else TextRunConfig
    config = config_cls.model_validate(run["config"])
    task = Task.model_validate(run["task"])
    save_dir = Path(run["save_dir"]) if run.get("save_dir") else None

    user_voice_settings, user_persona_config = make_voice_run_settings(config)
    info = get_info(
        config,
        user_persona_config=user_persona_config,
        user_voice_settings=user_voice_settings,
    )
    ctx = _BatchContext(
        config=config,
        evaluation_type=EvaluationType(run["evaluation_type"]),
        save_dir=save_dir,
        user_voice_settings=user_voice_settings,
        user_persona_config=user_persona_config,
        info=info,
        console_display=False,
        llm_log_mode_value=run.get("llm_log_mode"),
    )
    return run_unit(ctx, task, unit.trial, unit.seed, unit.progress_str)


def _maybe_preregister_livekit(run_payload: dict, registered: set) -> None:
    """LiveKit plugins must be registered on the worker's main thread before
    sim threads spawn (same constraint as the local loop)."""
    if "livekit" in registered:
        return
    audio_config = (run_payload.get("config") or {}).get("audio_native_config") or {}
    if audio_config.get("provider") == "livekit":
        from tau2.voice.audio_native.livekit import preregister_livekit_plugins

        preregister_livekit_plugins()
        registered.add("livekit")


def worker_loop(
    client: ControllerClient,
    worker_id: str,
    slots: int,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> int:
    """Lease → execute → report, keeping up to ``slots`` sims in flight.
    Returns a process exit code; exits when the controller reports done."""
    executor = ThreadPoolExecutor(max_workers=slots)
    inflight: dict[Future, str] = {}
    inflight_lock = threading.Lock()
    preregistered: set = set()

    def inflight_ids() -> list[str]:
        with inflight_lock:
            return list(inflight.values())

    heartbeats = HeartbeatThread(
        client, worker_id, inflight_ids, interval=heartbeat_interval
    )
    heartbeats.start()

    try:
        while True:
            # Report finished sims.
            for future in [f for f in list(inflight) if f.done()]:
                with inflight_lock:
                    unit_id = inflight.pop(future)
                try:
                    sim = future.result()
                except BaseException as e:
                    logger.error(f"Unit {unit_id} failed in worker: {e}")
                    resp = client.fail(worker_id, unit_id, f"{type(e).__name__}: {e}")
                else:
                    resp = client.complete(
                        worker_id, unit_id, sim.model_dump(mode="json")
                    )
                status = resp.get("status")
                if status == "stale":
                    logger.warning(
                        f"Result for {unit_id} arrived after its lease was lost; "
                        "the controller discarded it as stale"
                    )
                elif status == "requeued":
                    logger.info(
                        f"Controller requeued {unit_id} (infrastructure_error result)"
                    )

            # Fill free slots.
            if len(inflight) < slots:
                body = client.lease(worker_id)
                if body["status"] == "unit":
                    _maybe_preregister_livekit(body["run"], preregistered)
                    heartbeats.leased(body["unit"]["unit_id"])
                    future = executor.submit(execute_lease, body)
                    with inflight_lock:
                        inflight[future] = body["unit"]["unit_id"]
                    continue  # try to fill the next slot immediately
                if body["status"] == "done" and not inflight:
                    logger.info("Controller reports done; worker exiting")
                    return 0

            # Wait for a sim to finish (or poll again for work).
            if inflight:
                wait(list(inflight), timeout=poll_interval, return_when=FIRST_COMPLETED)
            else:
                time.sleep(poll_interval)
    finally:
        heartbeats.stop()
        executor.shutdown(wait=False, cancel_futures=True)


def run_worker_command(
    controller: str, slots: int = DEFAULT_SLOTS, worker_id: Optional[str] = None
) -> int:
    if worker_id is None:
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
    logger.info(f"Worker {worker_id} connecting to {controller} with {slots} slot(s)")
    client = ControllerClient(base_url=controller)
    try:
        return worker_loop(client, worker_id=worker_id, slots=slots)
    except httpx.HTTPError as e:
        logger.error(f"Lost contact with controller {controller}: {e}")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="tau2 worker: executes simulations for a tau2 controller."
    )
    parser.add_argument(
        "--controller",
        required=True,
        help="Controller base URL, e.g. http://127.0.0.1:8321",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help=f"Concurrent simulations this worker holds (default {DEFAULT_SLOTS}).",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Worker identity in controller logs (default: hostname-pid).",
    )
    args = parser.parse_args(argv)
    return run_worker_command(
        controller=args.controller, slots=args.slots, worker_id=args.worker_id
    )


if __name__ == "__main__":
    sys.exit(main())
