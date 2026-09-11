# Copyright Sierra
"""Re-score stored results: reward via the evaluator, quality/nativeness/delivery via
the judges pillar.

``tau2 evaluate-trajs`` is a thin CLI over ``rescore_results``. The judge axes
are NOT re-implemented here — they route through ``tau2.judges.export``
(``iter_judged_sims_detailed``), so post-hoc rescoring gets the exact same
machinery as ``tau2 judges rejudge``: the registered-language guard, complete-
verdict reuse with gap-filling (ERROR verdicts are healed, never treated as
done), bounded-memory streaming, and per-sim error policy. Delivery is
re-scored from the stored ``both.wav`` disk audio (full-duplex voice runs).

Rescored results are persisted back in their original storage format — paid
judge verdicts are never wasted on a display-only pass.
"""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.progress import Progress

from tau2.data_model.simulation import (
    DEFAULT_SCORES,
    DeliveryJudgeSettings,
    NativenessJudgeSettings,
    QualityJudgeSettings,
    Results,
    Score,
    SimulationRun,
    parse_scores,
)
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.judges.export import (
    JudgeStreamStats,
    iter_judged_sims_detailed,
)
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.display import ConsoleDisplay
from tau2.utils.io_utils import expand_paths


def is_solo_mode(results: Results) -> bool:
    """Whether the run was recorded in solo mode (solo agent + dummy user)."""
    return (
        results.info.agent_info.implementation == "llm_agent_solo"
        and results.info.user_info.implementation == "dummy_user"
    )


def get_communication_mode(
    results: Results, simulation: Optional[SimulationRun] = None
) -> CommunicationMode:
    """Detect the communication mode of a simulation in the results.

    Full-duplex (voice streaming) trajectories store the conversation in
    simulation.ticks rather than simulation.messages and must be rescored
    with the tick-based evaluators. Per-simulation signals take precedence;
    the run-level info covers full-duplex trajectories that predate the
    SimulationRun.mode field or were saved without ticks.
    """
    if simulation is not None and (
        simulation.mode == CommunicationMode.FULL_DUPLEX.value
        or simulation.ticks is not None
    ):
        return CommunicationMode.FULL_DUPLEX
    info = results.info
    if info.audio_native_config is not None:
        return CommunicationMode.FULL_DUPLEX
    if info.user_info.implementation == "voice_streaming_user_simulator":
        return CommunicationMode.FULL_DUPLEX
    return CommunicationMode.HALF_DUPLEX


def _load_fresh_tasks(
    results: Results,
    task_set_name: Optional[str] = None,
    console: Optional[Console] = None,
) -> Results:
    """Replace the task definitions embedded in the results with the current
    ones from the data directory, matched by task id.

    The embedded tasks record what the grading criteria were when the run was
    produced. Re-grading against updated criteria (e.g. after a task fix ships)
    requires reloading them; simulations whose task id no longer exists keep
    their embedded definition and a warning is emitted.

    Tasks are reloaded from ``task_set_name`` if given, else from the task set
    recorded in the results file, else from the domain's default set. The last
    fallback matters: localized task sets (e.g. ``airline_hi``) share task ids
    with the domain default, so reloading a localized run from the default set
    would silently swap in English task definitions.
    """
    from tau2.registry import registry

    domain = results.info.environment_info.domain_name
    task_set = task_set_name or results.info.task_set_name
    if task_set is None:
        task_set = domain
        if console:
            console.print(
                f"  ⚠️  Results file does not record a task set; reloading "
                f"tasks from the '{domain}' domain default. If this run used "
                f"a localized/non-default task set, pass --fresh-tasks-set.",
                style="yellow",
            )
    fresh = {task.id: task for task in registry.get_tasks_loader(task_set)(None)}
    missing = [task.id for task in results.tasks if task.id not in fresh]
    if missing and console:
        console.print(
            f"  ⚠️  {len(missing)} task(s) not found in current data dir, "
            f"keeping embedded definitions: {missing}",
            style="yellow",
        )
    results.tasks = [fresh.get(task.id, task) for task in results.tasks]
    return results


def _build_eval_env_kwargs(domain: str, task) -> Optional[dict]:
    """Env kwargs needed so re-grading matches live grading for a domain.

    banking_knowledge needs the per-task read_log_allowlist: without it the
    required-read assertions (derived from the golden trajectory) silently
    stop discriminating. Mirrors tau2.runner.build._build_env_kwargs.
    """
    if domain == "banking_knowledge":
        from tau2.runner.build import _derive_read_log_allowlist

        return {"read_log_allowlist": _derive_read_log_allowlist(task)}
    return None


def compute_simulation_rewards(
    results: Results,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    console: Optional[Console] = None,
    fresh_tasks: bool = False,
    fresh_tasks_set: Optional[str] = None,
) -> Results:
    """
    Compute and update rewards for all simulations in the results.

    Args:
        results: The Results object containing simulations to evaluate
        evaluation_type: Type of evaluation to perform
        console: Optional Rich console for output
        fresh_tasks: Re-grade against the current task definitions from the
            data directory instead of the ones embedded in the results file.
        fresh_tasks_set: Registered task set to reload the tasks from,
            overriding the one recorded in the results file (for results
            predating the recorded task set, e.g. localized runs).
    """
    results = deepcopy(results)
    if fresh_tasks:
        results = _load_fresh_tasks(
            results, task_set_name=fresh_tasks_set, console=console
        )
    domain = results.info.environment_info.domain_name
    solo_mode = is_solo_mode(results)
    tasks = {task.id: task for task in results.tasks}

    progress_context = Progress(console=console) if console else None

    try:
        if progress_context:
            progress_context.__enter__()
            task_progress = progress_context.add_task(
                "🔍 Computing rewards...", total=len(results.simulations)
            )

        for simulation in results.simulations:
            task = tasks[simulation.task_id]
            simulation.reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=evaluation_type,
                solo_mode=solo_mode,
                mode=get_communication_mode(results, simulation),
                env_kwargs=_build_eval_env_kwargs(domain, task),
                strict_replay=False,
            )

            if progress_context:
                progress_context.update(task_progress, advance=1)

    finally:
        if progress_context:
            progress_context.__exit__(None, None, None)
    return results


def rescore_results(
    path: Path,
    *,
    scores: set[Score],
    evaluation_type: EvaluationType = EvaluationType.ALL,
    nativeness_judge: Optional[NativenessJudgeSettings] = None,
    delivery_judge: Optional[DeliveryJudgeSettings] = None,
    quality_judge: Optional[QualityJudgeSettings] = None,
    concurrency: int = 10,
    output: Optional[Path] = None,
    fresh_tasks: bool = False,
    fresh_tasks_set: Optional[str] = None,
) -> tuple[Results, JudgeStreamStats]:
    """Re-score one stored results location and persist it (original format).

    - ``reward``     — re-run the evaluator per simulation.
    - ``nativeness`` — the judges gap-filling stream: complete stored verdicts
      are reused; missing/DEFERRED/ERROR verdicts are (re-)judged.
    - ``delivery``   — same stream, judged from the stored disk audio
      (full-duplex voice runs; unavailable sims are counted, never fatal).

    ``fresh_tasks`` re-grades against the current task definitions from the
    data directory instead of the ones embedded in the results file.

    Saves to ``output`` (default: back over ``path``) preserving the input's
    storage format, and returns the rescored ``Results`` plus the judge-stream
    stats.
    """
    path = Path(path)
    stats = JudgeStreamStats()
    judged: dict[str, SimulationRun] = {}
    if Score.NATIVENESS in scores or Score.DELIVERY in scores:
        for item in iter_judged_sims_detailed(
            path,
            concurrency=concurrency,
            settings=nativeness_judge,
            delivery_settings=(
                (delivery_judge or DeliveryJudgeSettings())
                if Score.DELIVERY in scores
                else None
            ),
            judge_nativeness=Score.NATIVENESS in scores,
            stats=stats,
        ):
            if item.was_judged:
                judged[item.sim.id] = item.sim

    fmt = Results.detect_format(path)
    results = Results.load(path)
    results.simulations = [judged.get(s.id, s) for s in results.simulations]

    if Score.QUALITY in scores:
        from tau2.judges.attach import attach_quality

        tasks_by_id = {task.id: task for task in results.tasks}
        domain = results.info.environment_info.domain_name
        for simulation in results.simulations:
            task = tasks_by_id.get(simulation.task_id)
            if task is None:
                logger.warning(
                    f"no task for quality sim {simulation.id} "
                    f"(task_id={simulation.task_id}); skip"
                )
                continue
            attach_quality(
                simulation,
                task,
                domain=domain,
                settings=quality_judge,
            )

    if Score.REWARD in scores:
        results = compute_simulation_rewards(
            results,
            evaluation_type=evaluation_type,
            console=ConsoleDisplay.console,
            fresh_tasks=fresh_tasks,
            fresh_tasks_set=fresh_tasks_set,
        )

    target = Path(output) if output is not None else path
    if fmt == "dir" and target.suffix != ".json":
        target.mkdir(parents=True, exist_ok=True)
    results.save(target, format=fmt)
    return results, stats


def evaluate_trajectories(
    input_paths: list[str],
    output_dir: str | None = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    scores: Optional[set[Score]] = None,
    nativeness_judge: Optional[NativenessJudgeSettings] = None,
    delivery_judge: Optional[DeliveryJudgeSettings] = None,
    quality_judge: Optional[QualityJudgeSettings] = None,
    fresh_tasks: bool = False,
    fresh_tasks_set: Optional[str] = None,
) -> None:
    """Re-score trajectory files/dirs and persist the updated results.

    Each input is rescored via ``rescore_results`` and saved back in place
    (or under ``output_dir`` as ``updated_<name>`` when provided) — judge
    verdicts paid for along the way are always persisted. Metrics are
    displayed per input. A failed input is reported and counted; the batch
    continues and the process exits non-zero at the end.

    ``fresh_tasks`` re-grades against the current task definitions from the
    data directory instead of the ones embedded in each results file.
    """
    scores = set(DEFAULT_SCORES) if scores is None else scores
    files = expand_paths(input_paths, extension=".json")
    console = ConsoleDisplay.console
    if not files:
        console.print("❌ No trajectory files found", style="red")
        sys.exit(1)

    output_path: Optional[Path] = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    console.print(f"\n🔍 Rescoring {len(files)} trajectory file(s)", style="bold blue")

    failed_files: list[str] = []
    for file_path in files:
        console.print(f"\n📁 {file_path}", style="bold")
        try:
            output = (
                output_path / f"updated_{Path(file_path).name}"
                if output_path is not None
                else None
            )
            results, stats = rescore_results(
                Path(file_path),
                scores=scores,
                evaluation_type=evaluation_type,
                nativeness_judge=nativeness_judge,
                delivery_judge=delivery_judge,
                quality_judge=quality_judge,
                output=output,
                fresh_tasks=fresh_tasks,
                fresh_tasks_set=fresh_tasks_set,
            )
            console.print(
                f"  ✅ Rescored {len(results.simulations)} simulation(s) "
                f"(judged {stats.judged}, reused {stats.reused}, "
                f"skipped {stats.skipped})",
                style="green",
            )
            if stats.skipped:
                logger.warning(
                    f"{file_path}: skipped sims (kept as stored): "
                    f"{stats.skipped_sim_ids}"
                )
            if stats.delivery_unavailable:
                logger.warning(
                    f"{file_path}: delivery unavailable for "
                    f"{stats.delivery_unavailable_sim_ids}"
                )
            ConsoleDisplay.display_agent_metrics(compute_metrics(results))
            console.print(
                f"  💾 Saved to: {output if output is not None else file_path}",
                style="blue",
            )
        except Exception as e:
            console.print(f"  ❌ Error processing file: {e}", style="red")
            failed_files.append(file_path)

    console.print()
    console.print("=" * 60, style="dim")
    console.print(f"📊 Summary: {len(files)} file(s) processed", style="bold")
    if not failed_files:
        console.print("🎉 All files processed successfully!", style="bold green")
    else:
        console.print(f"✅ {len(files) - len(failed_files)} file(s) processed")
        console.print(f"❌ {len(failed_files)} file(s) failed", style="red")
        for failed_file in failed_files:
            console.print(f"  • {failed_file}", style="red")
        sys.exit(1)


def main():
    """Entry point mirroring the ``tau2 evaluate-trajs`` verb."""
    import argparse

    # WARNING and up: judge failures (per-factor ERROR outcomes, skipped sims)
    # are logged at WARNING and must stay visible; only chatty INFO/DEBUG from
    # the evaluator internals is suppressed.
    logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])
    parser = argparse.ArgumentParser(
        description="Re-score trajectories (reward / quality / nativeness / delivery)"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save rescored results as updated_<name>. "
        "Default: update each input in place.",
    )
    parser.add_argument(
        "--scores",
        default="reward,quality,nativeness",
        help="Comma-separated scoring axes to recompute: "
        "reward,quality,nativeness,delivery (or 'all'). Delivery re-scores from the "
        "stored disk audio (full-duplex voice runs). Unselected axes are "
        "left untouched on each simulation.",
    )
    parser.add_argument(
        "--nativeness-llm-judge",
        action="store_true",
        help="Run the LLM judge for judge-type nativeness factors. Off by "
        "default: deterministic checkers only, LLM factors recorded DEFERRED.",
    )
    parser.add_argument(
        "--quality-llm-judge",
        action="store_true",
        help="Run task-aware semantic quality factors. Off by default.",
    )
    parser.add_argument(
        "--fresh-tasks",
        action="store_true",
        help="Re-grade against the current task definitions from the data "
        "directory instead of the ones embedded in each results file.",
    )
    parser.add_argument(
        "--fresh-tasks-set",
        default=None,
        help="Registered task set to reload tasks from with --fresh-tasks "
        "(e.g. 'airline_hi'), overriding the one recorded in the results "
        "file. Needed for localized runs whose results predate the recorded "
        "task set.",
    )
    args = parser.parse_args()
    try:
        scores = parse_scores(args.scores, voice=True)
    except ValueError as exc:
        parser.error(str(exc))
    evaluate_trajectories(
        args.paths,
        args.output_dir,
        scores=scores,
        nativeness_judge=NativenessJudgeSettings(llm_judge=args.nativeness_llm_judge),
        quality_judge=QualityJudgeSettings(llm_judge=args.quality_llm_judge),
        fresh_tasks=args.fresh_tasks,
        fresh_tasks_set=args.fresh_tasks_set,
    )


if __name__ == "__main__":
    main()
