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
from pydantic import BaseModel

from tau2.config import VOICE_TIMEOUT_SAFETY_FACTOR
from tau2.data_model.simulation import (
    SIMULATIONS_DIR,
    Results,
    SimulationIndexEntry,
    SimulationRun,
    SupersededInfo,
    TerminationReason,
)
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import show_dict_diff

#: Filename prefix of the results.json temp file written mid-``os.replace`` by
#: the dir-format index flush. A temp file with this prefix is evidence that a
#: writer is live in the directory, so anything that edits stored run dirs
#: must key off this constant, never a second copy of the string literal.
CHECKPOINT_META_TMP_PREFIX = ".meta_"


def task_drift_hash(task: BaseModel) -> str:
    """The hash task-drift detection compares (resume AND ``tau2 run refill``).

    ``Task.description`` is excluded: it is provenance prose (purpose/notes,
    including the generator-version string a re-freeze bumps on EVERY task)
    that no simulated call ever consumes. A recorded call is stale when the
    task it actually exercised — scenario, seeds, evaluation — changed, not
    when its documentation did: hashing the notes made a version-note-only
    re-freeze mark all 200 tasks of a run stale and redo the lot.
    """
    return get_pydantic_hash(task, exclude={"description": True})


def try_resume(
    save_path: Path,
    simulation_results: Results,
    tasks: list,
    num_trials: int,
    auto_resume: bool = False,
    results_format: Literal["json", "dir"] = "json",
    redo_stale_tasks: bool = False,
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
        redo_stale_tasks: If True, a task whose text changed since the
            checkpoint was written is STALE rather than fatal: its recorded
            simulations are dropped and the checkpoint's task object is
            replaced with the current one, so the redo runs the new text.
            If False (default), a modified task aborts the resume.

    A resume that proceeds under a CHANGED run config rewrites the
    checkpoint's header (``Results.info``) to this invocation's config and
    appends the displaced header to ``Results.info_history`` — the header
    states what is producing simulations NOW, and the regimes the surviving
    simulations were collected under stay on record instead of being silently
    misattributed to whichever invocation wrote results.json first.

    Returns:
        Tuple of (results, done_runs, tasks):
        - results: The resumed or new Results object.
        - done_runs: Set of (trial, task_id, seed) tuples already completed.
        - tasks: Potentially updated task list (if new tasks were merged, or
          stale ones refreshed).

    Raises:
        FileExistsError: If user declines to resume.
        ValueError: If config changed and user declines, tasks were removed, or
            tasks were modified without ``redo_stale_tasks``.
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
    fmt = Results.detect_format(save_path)
    prev_simulation_results = Results.load(save_path)

    # Check if the run config has changed (exclude the policy, which may change
    # between runs, and the reasoning-effort provenance, which records how the
    # stored value was obtained rather than what the run is configured to do —
    # a backfilled dir must still resume cleanly).
    exclude_fields = {
        "environment_info": {"policy"},
        "audio_native_config": {
            "reasoning_effort_source",
            "reasoning_effort_backfill",
        },
    }
    config_drift = get_pydantic_hash(
        prev_simulation_results.info, exclude=exclude_fields
    ) != get_pydantic_hash(simulation_results.info, exclude=exclude_fields)
    if config_drift:
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
        # Proceeding under a changed config: the header must describe the
        # invocation now producing simulations, not whichever invocation wrote
        # results.json first — a pool cell resumed under a widened ceiling
        # otherwise claims every simulation ran under the old one. The
        # displaced header moves to info_history, so the directory records
        # BOTH regimes its surviving simulations were collected under.
        prev_simulation_results.info_history = [
            *prev_simulation_results.info_history,
            SupersededInfo(info=prev_simulation_results.info),
        ]
        prev_simulation_results.info = simulation_results.info

    # Check task set compatibility
    prev_tasks_by_id = {t.id: t for t in prev_simulation_results.tasks}
    new_tasks_by_id = {t.id: t for t in simulation_results.tasks}

    modified_tasks = []
    removed_tasks = []
    for task_id, prev_task in prev_tasks_by_id.items():
        if task_id not in new_tasks_by_id:
            removed_tasks.append(task_id)
        elif task_drift_hash(prev_task) != task_drift_hash(new_tasks_by_id[task_id]):
            modified_tasks.append(task_id)

    if removed_tasks:
        raise ValueError(
            f"Tasks were removed from the task set: {removed_tasks}. "
            "Please delete the existing file or use a different save_to name."
        )
    stale_task_ids: set[str] = set()
    if modified_tasks:
        if not redo_stale_tasks:
            raise ValueError(
                f"Tasks were modified: {modified_tasks}. "
                "Pass --redo-stale-tasks to drop their recorded simulations "
                "and re-run them against the current text, or delete the "
                "existing file / use a different save_to name."
            )
        stale_task_ids = set(modified_tasks)
        logger.info(
            f"{len(stale_task_ids)} task(s) changed since the checkpoint was "
            "written; their simulations will be dropped and re-run against "
            f"the current text: {sorted(stale_task_ids)}"
        )

    # Identify new tasks being added
    added_task_ids = set(new_tasks_by_id.keys()) - set(prev_tasks_by_id.keys())
    if added_task_ids:
        logger.info(
            f"Adding {len(added_task_ids)} new tasks to the run: {sorted(added_task_ids)}"
        )

    # Determine completed runs. Two kinds of recorded sim are dropped so the
    # run redoes them: infrastructure failures (always) and sims of tasks whose
    # text changed (only under redo_stale_tasks — see above).
    def _is_dropped(sim) -> bool:
        return (
            sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
            or sim.task_id in stale_task_ids
        )

    # The two reported counts must be DISJOINT and must sum to what was
    # deleted: a sim can satisfy both predicates at once (an infrastructure
    # error recorded for a task whose text later changed), and counting it
    # under both — or subtracting one total from the other — makes the redo's
    # own verification arithmetic lie. A sim that is both is booked as an
    # infrastructure error, because it would have been dropped with or without
    # ``redo_stale_tasks``; the stale count then answers the question the flag
    # raises, "how many extra simulations did redoing the stale tasks cost".
    infra_error_ids = {
        sim.id
        for sim in prev_simulation_results.simulations
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    }
    dropped_sim_ids = [
        sim.id for sim in prev_simulation_results.simulations if _is_dropped(sim)
    ]
    infra_error_count = len(infra_error_ids)
    stale_sim_count = sum(
        1 for sim_id in dropped_sim_ids if sim_id not in infra_error_ids
    )
    done_runs = set(
        [
            (sim.trial, sim.task_id, sim.seed)
            for sim in prev_simulation_results.simulations
            if not _is_dropped(sim)
        ]
    )
    prev_simulation_results.simulations = [
        sim for sim in prev_simulation_results.simulations if not _is_dropped(sim)
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

    # Refresh stale tasks: the checkpoint must carry the CURRENT text, or the
    # redo would re-run the very text that was found stale.
    if stale_task_ids:
        prev_simulation_results.tasks = [
            new_tasks_by_id[t.id] if t.id in stale_task_ids else t
            for t in prev_simulation_results.tasks
        ]
        tasks = prev_simulation_results.tasks

    # Re-save checkpoint if anything changed (sims removed, tasks added /
    # refreshed, or the header rewritten for a changed config) so that the
    # on-disk state stays in sync with the in-memory one.
    # ``stale_task_ids`` is its own trigger, not a consequence of
    # ``dropped_sim_ids``: a task whose text changed but whose simulations are
    # already gone — the state `tau2 drop-sims --tasks` leaves behind — drops
    # nothing and adds nothing, yet its refreshed text exists only in memory
    # until this write. Skip it and the run does the right thing while the
    # checkpoint keeps the stale text, so a crash before the next flush makes
    # the next resume raise `Tasks were modified` on a task already redone.
    # ``config_drift`` is its own trigger for the same reason: a resume that
    # drops nothing (the pt airline rerun's completed cells) must still land
    # the rewritten header, or the file keeps claiming the old config.
    if added_task_ids or dropped_sim_ids or stale_task_ids or config_drift:
        if fmt == "dir":
            for sim_id in dropped_sim_ids:
                sim_file = sims_dir / f"{sim_id}.json"
                if sim_file.exists():
                    sim_file.unlink()
            # Rebuild index after removing the dropped sims
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
        if stale_sim_count > 0:
            logger.info(
                f"Removed {stale_sim_count} simulation(s) of {len(stale_task_ids)} "
                "stale task(s) from checkpoint; they will be re-run against the "
                "current task text"
            )
        if config_drift:
            logger.info(
                "Rewrote the results header to this invocation's run config; "
                "the previous header is kept in info_history "
                f"({len(prev_simulation_results.info_history)} superseded "
                "config(s) on record)"
            )

    console_text = Text(
        text=f"Resuming run from {len(done_runs)} runs. {len(tasks) * num_trials - len(done_runs)} runs remaining.",
        style="bold yellow",
    )
    ConsoleDisplay.console.print(console_text)

    return prev_simulation_results, done_runs, tasks


def drop_simulations(
    results_path: Path,
    terminations: Optional[list[TerminationReason]] = None,
    task_ids: Optional[list[str]] = None,
    trials: Optional[list[int]] = None,
) -> list[tuple[str, int, str]]:
    """Remove completed sims by termination reason, task id and/or trial.

    A sim is dropped when it matches EVERY provided filter (at least one
    filter is required): ``terminations`` selects by termination_reason,
    ``task_ids`` by the sim's task, ``trials`` by its trial index. The next
    ``--auto-resume`` run then re-runs those (trial, task, seed) cells — the
    same mechanism try_resume applies to infrastructure errors automatically,
    exposed as an explicit verb for deliberate redos (e.g. re-running
    timeout/max_steps truncations after raising the ceilings, or redoing
    tasks whose recorded sims are invalid).

    Rewrites the checkpoint the way try_resume does: dir format deletes the
    per-sim files AND rebuilds ``simulation_index`` (Results.load fails on an
    index/file mismatch), json format rewrites the monolithic file.

    Returns (task_id, trial, termination_reason) per dropped sim.
    """
    if not terminations and not task_ids and not trials:
        raise ValueError("drop_simulations needs terminations, task_ids or trials")
    results_path = Path(results_path)
    fmt = Results.detect_format(results_path)
    results = Results.load(results_path)

    targets = set(terminations or [])
    wanted_tasks = set(task_ids or [])
    wanted_trials = set(trials if trials is not None else [])
    dropped = [
        s
        for s in results.simulations
        if (not targets or s.termination_reason in targets)
        and (not wanted_tasks or s.task_id in wanted_tasks)
        # A sim that records no trial is the single-trial run's trial 0, which
        # is how every caller names it (see refill.SimulationCell). Matching
        # the raw None instead would let a cell be planned for deletion, not
        # deleted, and then re-run beside the copy that survived.
        and (
            not wanted_trials
            or (s.trial if s.trial is not None else 0) in wanted_trials
        )
    ]
    if not dropped:
        return []
    dropped_ids = {s.id for s in dropped}
    results.simulations = [s for s in results.simulations if s.id not in dropped_ids]

    if fmt == "dir":
        _, sims_dir = Results._resolve_paths(results_path)
        for sim_id in dropped_ids:
            sim_file = sims_dir / f"{sim_id}.json"
            if sim_file.exists():
                sim_file.unlink()
        results.simulation_index = results._build_simulation_index()
        results.save_metadata(results_path)
    else:
        with open(results_path, "w") as fp:
            fp.write(results.model_dump_json(indent=2))

    criteria = ", ".join(
        filter(
            None,
            (
                ", ".join(sorted(t.value for t in targets)),
                f"{len(wanted_tasks)} task id(s)" if wanted_tasks else "",
                f"trial(s) {sorted(wanted_trials)}" if wanted_trials else "",
            ),
        )
    )
    logger.info(
        f"Dropped {len(dropped)} simulation(s) from {results_path} "
        f"({criteria}) — the next --auto-resume run will redo them"
    )
    return [(s.task_id, s.trial, s.termination_reason.value) for s in dropped]


class CeilingRepair(BaseModel):
    """What ``repair_ceiling`` found, and (unless dry-run / no-op) wrote."""

    results_path: str
    changed: bool
    old_max_steps_seconds: int
    new_max_steps_seconds: int
    old_max_steps: int
    new_max_steps: int
    old_timeout: Optional[float]
    new_timeout: Optional[float]

    def describe(self) -> str:
        if not self.changed:
            return (
                f"{self.results_path}: header already records "
                f"max_steps_seconds={self.new_max_steps_seconds}; nothing to do"
            )
        return (
            f"{self.results_path}: max_steps_seconds "
            f"{self.old_max_steps_seconds} -> {self.new_max_steps_seconds}, "
            f"max_steps {self.old_max_steps} -> {self.new_max_steps}, "
            f"timeout {self.old_timeout} -> {self.new_timeout} "
            "(prior header kept in info_history)"
        )


def resolve_results_path(save_to: str | Path) -> Path:
    """The stored results named by ``--save-to``: a directory, a results.json
    (dir-format metadata or monolithic), or the ``--save-to`` name the run was
    launched with (resolved under data/simulations/). Unlike refill's
    ``resolve_run_dir`` this keeps a file path a file path, because a
    monolithic results.json collapsed to its parent directory would be
    re-detected as dir format."""
    from tau2.utils import DATA_DIR

    path = Path(save_to)
    for candidate in (path, DATA_DIR / "simulations" / path):
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(
        f"no stored results at {path} (also looked under {DATA_DIR / 'simulations'})"
    )


def repair_ceiling(
    save_to: str | Path, max_steps_seconds: int, *, dry_run: bool = False
) -> CeilingRepair:
    """Rewrite a stored voice run's header to the conversation ceiling its
    finishing invocation actually ran under.

    For directories written before ``try_resume`` learned to rewrite the
    header on a config drift: a pool cell created at one ``--max-steps-seconds``
    and resumed to completion at another kept the FIRST invocation's ceiling in
    ``info.audio_native_config.max_steps_seconds`` (plus the ``max_steps``
    ticks and ``timeout`` derived from it) — including cells already archived
    to Drive. This verb applies the same correction the resume now applies
    itself: the ceiling fields move to the stated value, the derived fields are
    recomputed with the same arithmetic ``tau2 pool run --max-steps-seconds``
    uses (ticks from the recorded tick duration; the wallclock guard rescaled,
    never narrowed), and the displaced header is appended to ``info_history``.

    Only the header is touched: simulations, tasks and the simulation index
    are byte-for-byte what they were.
    """
    results_path = resolve_results_path(save_to)
    fmt = Results.detect_format(results_path)
    # Monolithic json holds the simulations in the same file, so a
    # metadata-only load would drop them on save; dir format must NOT load
    # hundreds of MB of transcript to edit its header.
    results = (
        Results.load_metadata(results_path)
        if fmt == "dir"
        else Results.load(results_path)
    )
    info = results.info
    audio_cfg = info.audio_native_config
    if audio_cfg is None:
        raise ValueError(
            f"{results_path} records no audio_native_config; the conversation "
            "ceiling is voice tick arithmetic and a text run never had one"
        )

    new_audio = audio_cfg.model_copy(update={"max_steps_seconds": max_steps_seconds})
    new_timeout = info.timeout
    if info.timeout is not None:
        new_timeout = max(info.timeout, max_steps_seconds * VOICE_TIMEOUT_SAFETY_FACTOR)
    report = CeilingRepair(
        results_path=str(results_path),
        changed=audio_cfg.max_steps_seconds != max_steps_seconds,
        old_max_steps_seconds=audio_cfg.max_steps_seconds,
        new_max_steps_seconds=max_steps_seconds,
        old_max_steps=info.max_steps,
        new_max_steps=new_audio.max_steps_ticks,
        old_timeout=info.timeout,
        new_timeout=new_timeout,
    )
    if not report.changed or dry_run:
        return report

    results.info_history = [*results.info_history, SupersededInfo(info=info)]
    results.info = info.model_copy(
        update={
            "audio_native_config": new_audio,
            "max_steps": new_audio.max_steps_ticks,
            "timeout": new_timeout,
        }
    )
    if fmt == "dir":
        results.save_metadata(results_path)
    else:
        results.save(results_path, format="json")
    logger.info(report.describe())
    return report


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

    fmt = Results.detect_format(save_path)

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
                quality=sim.quality_info.score if sim.quality_info else None,
                nativeness=sim.nativeness_info.score if sim.nativeness_info else None,
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
                suffix=".json", prefix=CHECKPOINT_META_TMP_PREFIX, dir=meta_path.parent
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
