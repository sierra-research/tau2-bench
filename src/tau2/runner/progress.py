"""
Progress monitoring and retry logic for batch simulation runs.
"""

import threading
import time
import traceback
import uuid
from typing import Callable, Optional

from loguru import logger

from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.utils import get_now


def run_with_retry(
    run_fn: Callable[[], SimulationRun],
    task: Task,
    trial: int,
    seed: int,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    console_display: bool = True,
    save_fn: Optional[Callable[[SimulationRun], None]] = None,
    on_retry: Optional[Callable[[], None]] = None,
    shutdown_event: Optional[threading.Event] = None,
) -> SimulationRun:
    """Run a simulation function with retry logic.

    Retries on any exception. If all retries are exhausted, returns a failed
    SimulationRun with INFRASTRUCTURE_ERROR instead of raising.

    Args:
        run_fn: A callable that produces a SimulationRun.
        task: The task being run (for error reporting).
        trial: Trial number.
        seed: Random seed for this trial.
        max_retries: Maximum number of retries (on top of initial attempt).
        retry_delay: Delay in seconds between retries.
        console_display: Whether to show console output.
        save_fn: Optional callable to save the simulation after success/failure.
        on_retry: Optional callback invoked before each retry attempt.
        shutdown_event: If set, aborts retries immediately.

    Returns:
        SimulationRun (either successful or a failed placeholder).
    """
    max_attempts = max_retries + 1
    last_exception = None
    last_error_reason = ""
    last_traceback = ""

    for attempt in range(max_attempts):
        if shutdown_event is not None and shutdown_event.is_set():
            last_error_reason = "Shutdown requested (Ctrl+C)"
            last_exception = KeyboardInterrupt(last_error_reason)
            last_traceback = ""
            break

        try:
            if attempt > 0:
                retry_text = Text(
                    text=f"  Retry {attempt}/{max_retries} for task {task.id}: {last_error_reason}",
                    style="yellow",
                )
                ConsoleDisplay.console.print(retry_text)
                if on_retry:
                    on_retry()
                time.sleep(retry_delay)

            simulation = run_fn()
            simulation.trial = trial

            if console_display:
                ConsoleDisplay.display_simulation(simulation, show_details=False)
            if save_fn:
                save_fn(simulation)

            if attempt > 0:
                success_text = Text(
                    text=f"  Task {task.id} succeeded on retry {attempt}",
                    style="green",
                )
                ConsoleDisplay.console.print(success_text)

            return simulation

        except Exception as e:
            last_exception = e
            last_error_reason = str(e)
            last_traceback = traceback.format_exc()
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Task {task.id} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )
            else:
                logger.error(
                    f"Task {task.id} failed after {max_attempts} attempts: {e}"
                )

    # All retries exhausted
    error_text = Text(
        text=f"  Task {task.id} failed permanently after {max_attempts} attempts: {last_error_reason}",
        style="bold red",
    )
    ConsoleDisplay.console.print(error_text)

    now = get_now()
    failed_simulation = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=task.id,
        timestamp=now,
        start_time=now,
        end_time=now,
        duration=0.0,
        termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
        messages=[],
        trial=trial,
        seed=seed,
        info={
            "error": str(last_exception),
            "error_type": type(last_exception).__name__,
            "error_traceback": last_traceback,
            "failed_after_attempts": max_attempts,
        },
    )
    if save_fn:
        save_fn(failed_simulation)
    return failed_simulation


class StatusMonitor:
    """Periodic status monitoring for concurrent batch runs.

    Prints progress every 30 seconds with running task info and average reward.
    """

    def __init__(self, total_count: int, initial_completed: int = 0):
        self.total_count = total_count
        self.completed_count = initial_completed
        self.running_tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._simulation_results: Optional[Results] = None

    def set_results(self, results: Results):
        """Set the results object for reward tracking."""
        self._simulation_results = results

    def task_started(self, task_key: str, trial: int):
        """Record that a task has started."""
        with self._lock:
            self.running_tasks[task_key] = {
                "start_time": time.time(),
                "trial": trial,
                "retries": 0,
            }

    def task_restarted(self, task_key: str):
        """Reset the start time for a task and increment retry count."""
        with self._lock:
            if task_key in self.running_tasks:
                self.running_tasks[task_key]["start_time"] = time.time()
                self.running_tasks[task_key]["retries"] += 1

    def task_finished(self, task_key: str):
        """Record that a task has finished."""
        with self._lock:
            self.running_tasks.pop(task_key, None)
            self.completed_count += 1

    def start(self):
        """Start the background monitoring thread."""
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor(self):
        """Print status every 30 seconds."""
        while not self._stop_event.wait(timeout=30.0):
            with self._lock:
                running_count = len(self.running_tasks)
                if running_count == 0:
                    continue
                now = time.time()
                task_statuses = []
                for task_id, info in self.running_tasks.items():
                    elapsed = now - info["start_time"]
                    retries = info.get("retries", 0)
                    retry_str = f" R{retries}" if retries > 0 else ""
                    task_statuses.append(f"{task_id}({elapsed:.0f}s{retry_str})")

                reward_str = "Avg reward: N/A."
                if self.completed_count > 0 and self._simulation_results:
                    rewards = [
                        sim.reward_info.reward
                        for sim in self._simulation_results.simulations
                        if sim.reward_info is not None
                    ]
                    if rewards:
                        avg_reward = sum(rewards) / len(rewards)
                        reward_str = f"Avg reward: {avg_reward:.2f} (N={len(rewards)})."

                status_text = Text(
                    text=f"Status: {self.completed_count}/{self.total_count} complete. {reward_str} "
                    f"{running_count} running: {', '.join(task_statuses[:10])}"
                    + (f" +{running_count - 10} more" if running_count > 10 else ""),
                    style="cyan",
                )
                ConsoleDisplay.console.print(status_text)
