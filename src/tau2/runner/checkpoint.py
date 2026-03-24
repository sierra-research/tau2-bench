"""
Checkpoint save/resume logic for batch simulation runs.

Supports two storage formats:
- "json": single monolithic results.json (default for text runs).
- "dir": metadata in results.json + individual simulation files in
  simulations/ (default for voice runs — O(1) append, O(1) replace).

Format is auto-detected from on-disk state. New runs use the format specified
by the ``results_format`` parameter (default "json" for backward compat).
"""

import json
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

from loguru import logger

from tau2.data_model.simulation import (
    SIMULATIONS_DIR,
    Results,
    SimulationIndexEntry,
    SimulationRun,
    TerminationReason,
)
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import show_dict_diff


def try_resume(
    save_path: Path,
    simulation_results: Results,
    tasks: list,
    num_trials: int,
    auto_resume: bool = False,
    results_format: Literal["json", "dir"] = "json",
) -> tuple[Results, set, list]:
    """Try to resume from an existing checkpoint file.

    Args:
        save_path: Path to the results JSON file.
        simulation_results: The new (empty) results to compare against.
        tasks: Current task list.
        num_trials: Number of trials.
        auto_resume: If True, resume without prompting.
        results_format: Storage format for new runs ("json" or "dir").
            Ignored when resuming — the existing format is auto-detected.

    Returns:
        Tuple of (results, done_runs, tasks):
        - results: The resumed or new Results object.
        - done_runs: Set of (trial, task_id, seed) tuples already completed.
        - tasks: Potentially updated task list (if new tasks were merged).

    Raises:
        FileExistsError: If user declines to resume.
        ValueError: If config changed and user declines, or tasks were modified/removed.
    """
    done_runs: set = set()

    # For dir format, existence means either the results.json or simulations/ exist
    sims_dir = save_path.parent / SIMULATIONS_DIR
    has_existing = save_path.exists() or sims_dir.exists()

    if not has_existing:
        # Create new save file / directory
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving simulation batch to {save_path}")
        simulation_results.save(save_path, format=results_format)
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

    # Auto-detect format from on-disk state
    fmt = Results._detect_format(save_path)
    prev_simulation_results = Results.load(save_path)

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
    infra_error_sim_ids = [
        sim.id
        for sim in prev_simulation_results.simulations
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ]
    done_runs = set(
        [
            (sim.trial, sim.task_id, sim.seed)
            for sim in prev_simulation_results.simulations
            if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
        ]
    )
    # Remove infrastructure failure simulations so they can be replaced
    infra_error_count = len(infra_error_sim_ids)
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
    # so that the on-disk state stays in sync with the in-memory state.
    if added_task_ids or infra_error_count > 0:
        if fmt == "dir":
            for sim_id in infra_error_sim_ids:
                sim_file = sims_dir / f"{sim_id}.json"
                if sim_file.exists():
                    sim_file.unlink()
            # Rebuild index after removing infra-error sims
            prev_simulation_results.simulation_index = (
                prev_simulation_results._build_simulation_index()
            )
            prev_simulation_results.save_metadata(save_path)
        else:
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


def create_checkpoint_fns(
    save_path: Optional[Path],
    lock: multiprocessing.Lock,
) -> tuple[Callable, Callable]:
    """Create thread-safe checkpoint save and replace functions.

    Returns a (save_fn, replace_fn) pair. For directory format, the two
    closures share state so the replacer can locate simulation files written
    by the saver without scanning the directory.

    Args:
        save_path: Path to the results JSON file. If None, returns no-ops.
        lock: Multiprocessing lock for thread safety.

    Returns:
        Tuple of (save_fn, replace_fn).
    """
    if save_path is None:
        return (lambda simulation: None, lambda key, simulation: None)

    fmt = Results._detect_format(save_path)

    if fmt == "dir":
        meta_path, sims_dir = Results._resolve_paths(save_path)
        # Shared state: maps (trial, task_id, seed) -> sim.id for this run
        _key_to_sim_id: dict[tuple, str] = {}
        _saved_keys: set[tuple] = set()
        # Simulation index entries keyed by sim id for efficient updates
        _index_by_id: dict[str, dict] = {}

        # Seed index from existing results.json if present
        if meta_path.exists():
            with open(meta_path, "r") as fp:
                existing_meta = json.load(fp)
            for entry in existing_meta.get("simulation_index") or []:
                _index_by_id[entry["id"]] = entry

        def _index_entry(sim: SimulationRun) -> dict:
            return SimulationIndexEntry(
                id=sim.id,
                task_id=sim.task_id,
                trial=sim.trial,
                reward=sim.reward_info.reward if sim.reward_info else None,
                termination_reason=sim.termination_reason,
                agent_cost=sim.agent_cost,
                duration=sim.duration,
            ).model_dump(mode="json")

        def _flush_index():
            """Rewrite results.json with the current simulation_index."""
            with open(meta_path, "r") as fp:
                meta = json.load(fp)
            meta["simulation_index"] = list(_index_by_id.values())
            fd, tmp = tempfile.mkstemp(
                suffix=".json", prefix=".meta_", dir=meta_path.parent
            )
            try:
                with os.fdopen(fd, "w") as fp:
                    json.dump(meta, fp, indent=2)
                os.replace(tmp, meta_path)
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

        def _save_dir(simulation: SimulationRun):
            sim_key = (simulation.trial, simulation.task_id, simulation.seed)
            with lock:
                if sim_key in _saved_keys:
                    logger.warning(
                        f"Skipping duplicate save for task {simulation.task_id}, "
                        f"trial {simulation.trial}, seed {simulation.seed}"
                    )
                    return
                sim_path = sims_dir / f"{simulation.id}.json"
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".json", prefix=".sim_", dir=sims_dir
                )
                try:
                    with os.fdopen(fd, "w") as fp:
                        fp.write(simulation.model_dump_json(indent=2))
                    os.replace(tmp_path, sim_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
                _saved_keys.add(sim_key)
                _key_to_sim_id[sim_key] = simulation.id
                _index_by_id[simulation.id] = _index_entry(simulation)
                _flush_index()

        def _replace_dir(
            key: tuple[int, str, int],
            simulation: SimulationRun,
        ):
            with lock:
                old_sim_id = _key_to_sim_id.get(key)
                if old_sim_id:
                    old_path = sims_dir / f"{old_sim_id}.json"
                    if old_path.exists():
                        old_path.unlink()
                    _index_by_id.pop(old_sim_id, None)

                sim_path = sims_dir / f"{simulation.id}.json"
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".json", prefix=".sim_", dir=sims_dir
                )
                try:
                    with os.fdopen(fd, "w") as fp:
                        fp.write(simulation.model_dump_json(indent=2))
                    os.replace(tmp_path, sim_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
                _key_to_sim_id[key] = simulation.id
                _saved_keys.discard(key)
                _saved_keys.add(key)
                _index_by_id[simulation.id] = _index_entry(simulation)
                _flush_index()

        return _save_dir, _replace_dir

    # Monolithic JSON format
    def _save_json(simulation: SimulationRun):
        with lock:
            with open(save_path, "r") as fp:
                ckpt = json.load(fp)
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

    def _replace_json(
        key: tuple[int, str, int],
        simulation: SimulationRun,
    ):
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

    return _save_json, _replace_json


# Backward-compatible wrappers (used by external code or older call sites)
def create_checkpoint_saver(
    save_path: Optional[Path],
    lock: multiprocessing.Lock,
) -> Callable:
    """Create a thread-safe checkpoint save function.

    Prefer ``create_checkpoint_fns`` for new code.
    """
    save_fn, _ = create_checkpoint_fns(save_path, lock)
    return save_fn


def create_checkpoint_replacer(
    save_path: Optional[Path],
    lock: multiprocessing.Lock,
) -> Callable:
    """Create a thread-safe checkpoint replace function.

    Prefer ``create_checkpoint_fns`` for new code — the paired save/replace
    closures share state needed for efficient directory-format replacements.
    """
    _, replace_fn = create_checkpoint_fns(save_path, lock)
    return replace_fn
