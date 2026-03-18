"""
Checkpoint save/resume logic for batch simulation runs.
"""

import json
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import show_dict_diff


def try_resume(
    save_path: Path,
    simulation_results: Results,
    tasks: list,
    num_trials: int,
    auto_resume: bool = False,
) -> tuple[Results, set, list]:
    """Try to resume from an existing checkpoint file.

    Args:
        save_path: Path to the results JSON file.
        simulation_results: The new (empty) results to compare against.
        tasks: Current task list.
        num_trials: Number of trials.
        auto_resume: If True, resume without prompting.

    Returns:
        Tuple of (results, done_runs, tasks):
        - results: The resumed or new Results object.
        - done_runs: Set of (trial, task_id, seed) tuples already completed.
        - tasks: Potentially updated task list (if new tasks were merged).

    Raises:
        FileExistsError: If user declines to resume.
        ValueError: If config changed and user declines, or tasks were modified/removed.
    """
    done_runs = set()

    if not save_path.exists():
        # Create new save file
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving simulation batch to {save_path}")
        with open(save_path, "w") as fp:
            fp.write(simulation_results.model_dump_json(indent=2))
        return simulation_results, done_runs, tasks

    # File exists -- try to resume
    if auto_resume:
        response = "y"
    else:
        response = (
            ConsoleDisplay.console.input(
                "[yellow]File [bold]{}[/bold] already exists. Do you want to resume the run? (y/n)[/yellow] ".format(
                    save_path
                )
            )
            .lower()
            .strip()
        )
    if response != "y":
        raise FileExistsError(
            f"File {save_path} already exists. Please delete it or use a different save_to name."
        )

    with open(save_path, "r") as fp:
        prev_simulation_results = Results.model_validate_json(fp.read())

    # Check if the run config has changed (exclude policy which may change between runs)
    exclude_fields = {"environment_info": {"policy"}}
    if get_pydantic_hash(
        prev_simulation_results.info, exclude=exclude_fields
    ) != get_pydantic_hash(simulation_results.info, exclude=exclude_fields):
        diff = show_dict_diff(
            prev_simulation_results.info.model_dump(exclude=exclude_fields),
            simulation_results.info.model_dump(exclude=exclude_fields),
        )
        if auto_resume:
            logger.warning(
                f"Run config has changed, continuing with auto-resume:\n{diff}"
            )
            response = "y"
        else:
            ConsoleDisplay.console.print(
                f"The run config has changed.\n\n{diff}\n\nDo you want to resume the run? (y/n)"
            )
            response = (
                ConsoleDisplay.console.input(
                    "[yellow]File [bold]{}[/bold] already exists. Do you want to resume the run? (y/n)[/yellow] ".format(
                        save_path
                    )
                )
                .lower()
                .strip()
            )
        if response != "y":
            raise ValueError(
                "The run config has changed. Please delete the existing file or use a different save_to name."
            )

    # Check task set compatibility
    prev_tasks_by_id = {t.id: t for t in prev_simulation_results.tasks}
    new_tasks_by_id = {t.id: t for t in simulation_results.tasks}

    modified_tasks = []
    removed_tasks = []
    for task_id, prev_task in prev_tasks_by_id.items():
        if task_id not in new_tasks_by_id:
            removed_tasks.append(task_id)
        elif get_pydantic_hash(prev_task) != get_pydantic_hash(
            new_tasks_by_id[task_id]
        ):
            modified_tasks.append(task_id)

    if removed_tasks:
        raise ValueError(
            f"Tasks were removed from the task set: {removed_tasks}. "
            "Please delete the existing file or use a different save_to name."
        )
    if modified_tasks:
        raise ValueError(
            f"Tasks were modified: {modified_tasks}. "
            "Please delete the existing file or use a different save_to name."
        )

    # Identify new tasks being added
    added_task_ids = set(new_tasks_by_id.keys()) - set(prev_tasks_by_id.keys())
    if added_task_ids:
        logger.info(
            f"Adding {len(added_task_ids)} new tasks to the run: {sorted(added_task_ids)}"
        )

    # Determine completed runs (exclude infrastructure failures for retry)
    done_runs = set(
        [
            (sim.trial, sim.task_id, sim.seed)
            for sim in prev_simulation_results.simulations
            if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
        ]
    )
    # Remove infrastructure failure simulations so they can be replaced
    infra_error_count = len(prev_simulation_results.simulations) - len(done_runs)
    prev_simulation_results.simulations = [
        sim
        for sim in prev_simulation_results.simulations
        if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    ]

    # Merge tasks: keep previous tasks and add any new ones
    if added_task_ids:
        new_tasks_to_add = [
            t for t in simulation_results.tasks if t.id in added_task_ids
        ]
        prev_simulation_results.tasks = (
            list(prev_simulation_results.tasks) + new_tasks_to_add
        )
        tasks = prev_simulation_results.tasks

    # Re-save checkpoint if anything changed (infra errors removed or tasks added)
    # so that the on-disk file stays in sync with the in-memory state.
    # Without this, create_checkpoint_saver's duplicate check would reject
    # retried simulations because the old infra-error entries still exist on disk.
    if added_task_ids or infra_error_count > 0:
        with open(save_path, "w") as fp:
            fp.write(prev_simulation_results.model_dump_json(indent=2))
        if added_task_ids:
            logger.info(f"Updated results file with {len(added_task_ids)} new tasks")
        if infra_error_count > 0:
            logger.info(
                f"Removed {infra_error_count} infrastructure error simulation(s) "
                "from checkpoint for retry"
            )

    console_text = Text(
        text=f"Resuming run from {len(done_runs)} runs. {len(tasks) * num_trials - len(done_runs)} runs remaining.",
        style="bold yellow",
    )
    ConsoleDisplay.console.print(console_text)

    return prev_simulation_results, done_runs, tasks


def create_checkpoint_saver(
    save_path: Optional[Path],
    lock: multiprocessing.Lock,
):
    """Create a thread-safe checkpoint save function.

    Args:
        save_path: Path to the results JSON file. If None, returns a no-op.
        lock: Multiprocessing lock for thread safety.

    Returns:
        A callable that saves a SimulationRun to the checkpoint file atomically.
    """

    def save(simulation: SimulationRun):
        if save_path is None:
            return
        with lock:
            with open(save_path, "r") as fp:
                ckpt = json.load(fp)
            # Check for duplicates (race condition prevention)
            existing_keys = {
                (sim.get("trial"), sim.get("task_id"), sim.get("seed"))
                for sim in ckpt["simulations"]
            }
            sim_key = (simulation.trial, simulation.task_id, simulation.seed)
            if sim_key in existing_keys:
                logger.warning(
                    f"Skipping duplicate save for task {simulation.task_id}, "
                    f"trial {simulation.trial}, seed {simulation.seed}"
                )
                return
            ckpt["simulations"].append(simulation.model_dump())
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json", prefix=".results_", dir=save_path.parent
            )
            try:
                with os.fdopen(fd, "w") as fp:
                    json.dump(ckpt, fp, indent=2)
                os.replace(tmp_path, save_path)  # Atomic on POSIX
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    return save


def create_checkpoint_replacer(
    save_path: Optional[Path],
    lock: multiprocessing.Lock,
):
    """Create a thread-safe checkpoint replace function.

    Replaces an existing simulation entry in the checkpoint file, identified
    by (trial, task_id, seed). Used to swap a hallucinated result with a
    clean retry result.

    Args:
        save_path: Path to the results JSON file. If None, returns a no-op.
        lock: Multiprocessing lock for thread safety.

    Returns:
        A callable that replaces a SimulationRun in the checkpoint file atomically.
    """

    def replace(
        key: tuple[int, str, int],
        simulation: SimulationRun,
    ):
        if save_path is None:
            return
        trial, task_id, seed = key
        with lock:
            with open(save_path, "r") as fp:
                ckpt = json.load(fp)
            ckpt["simulations"] = [
                sim
                for sim in ckpt["simulations"]
                if not (
                    sim.get("trial") == trial
                    and sim.get("task_id") == task_id
                    and sim.get("seed") == seed
                )
            ]
            ckpt["simulations"].append(simulation.model_dump())
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json", prefix=".results_", dir=save_path.parent
            )
            try:
                with os.fdopen(fd, "w") as fp:
                    json.dump(ckpt, fp, indent=2)
                os.replace(tmp_path, save_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    return replace
