# Copyright Sierra
"""``tau2 pool`` and ``tau2 run refill`` CLI registration (package-owned,
wired from ``tau2.cli``).

The argparse surface lives here; the engines are ``tau2.runner.pool_driver``
and ``tau2.runner.refill``. Kept import-light: the engines pull in the data
model and psutil, so they are imported at dispatch, not at registration.
"""

from __future__ import annotations

import argparse

# =============================================================================
# tau2 pool
# =============================================================================


def add_pool_args(parser: argparse.ArgumentParser) -> None:
    """Attach the ``tau2 pool`` subcommands (list / status / run)."""
    from tau2.runner.pools import DEFAULT_POOL_WORKERS

    sub = parser.add_subparsers(dest="pool_command", required=True)

    list_parser = sub.add_parser("list", help="List the registered run pools")
    list_parser.set_defaults(func=run_pool_list)

    status_parser = sub.add_parser(
        "status",
        help="Print the pool's languages x arms matrix of usable simulations",
    )
    status_parser.add_argument("pool", help="Pool name (see `tau2 pool list`)")
    status_parser.set_defaults(func=run_pool_status)

    run_parser = sub.add_parser(
        "run",
        help="Run the selected cells through one shared multiprocess controller",
    )
    run_parser.add_argument("pool", help="Pool name (see `tau2 pool list`)")
    run_parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated cells to drive, as <lang>/<provider>/<effort> "
        "(default: every cell in the pool)",
    )
    run_parser.add_argument(
        "--langs",
        default=None,
        help="Comma-separated languages to drive (default: all in the pool)",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_POOL_WORKERS,
        help="Worker processes shared by every selected cell; each holds the "
        f"pool's per-worker concurrency (default: {DEFAULT_POOL_WORKERS})",
    )
    run_parser.add_argument(
        "--provider-limit",
        default=None,
        help='Per-provider concurrency caps, e.g. "openai=40,gemini=20"',
    )
    run_parser.add_argument(
        "--max-steps-seconds",
        type=int,
        default=None,
        help="Override the pool's per-simulation duration ceiling for THIS "
        "invocation only (default: the pool's declared ceiling). The cell "
        "--timeout is rescaled to VOICE_TIMEOUT_SAFETY_FACTOR x the new ceiling "
        "so the guard can never pre-empt the conversation it is guarding. For "
        "refilling calls that legitimately outran the ceiling. The registered "
        "pool spec is unchanged, but a resumed cell's results.json header IS "
        "rewritten to this invocation's ceiling (the displaced header moves to "
        "info_history), so the cell records what actually ran.",
    )
    run_parser.add_argument(
        "--redo-stale-tasks",
        action="store_true",
        help="Drop and re-run the simulations of any task whose text changed "
        "since the cell was written. Without it such a cell refuses to resume, "
        "which is the safe default: two vintages of the same task inside one "
        "cell is not a comparison. Destructive — it deletes recorded "
        "simulations. Use it to rebase a whole pool onto one text, never to "
        "unstick a single cell, or the pool ends up more mixed than it started.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the typed per-cell run configs without running them",
    )
    run_parser.set_defaults(func=run_pool_run)

    repair_parser = sub.add_parser(
        "repair-ceiling",
        help="Rewrite a stored voice cell's header to the --max-steps-seconds "
        "its finishing invocation actually ran under (the displaced header is "
        "kept in info_history; simulations are untouched)",
    )
    repair_parser.add_argument(
        "--save-to",
        required=True,
        help="The cell to repair: its directory, its results.json, or the "
        "--save-to name it was launched with (resolved under "
        "data/simulations/)",
    )
    repair_parser.add_argument(
        "--max-steps-seconds",
        type=int,
        required=True,
        help="The ceiling the finishing invocation ran under (the value its "
        "`tau2 pool run --max-steps-seconds` was given). max_steps ticks and "
        "the wallclock timeout are re-derived from it with the same "
        "arithmetic the run override uses.",
    )
    repair_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything",
    )
    repair_parser.set_defaults(func=run_pool_repair_ceiling)


def run_pool_list(args: argparse.Namespace) -> None:
    """Dispatch ``tau2 pool list``."""
    from tau2.runner.pools import POOLS, list_pools

    for name in list_pools():
        spec = POOLS[name]
        print(f"{name}")
        print(f"  {spec.description}")
        print(
            f"  {len(spec.languages)} langs ({' '.join(spec.languages)}) x "
            f"{len(spec.arms)} arms ({' '.join(a.key for a in spec.arms)})"
        )
        print(
            f"  {spec.tasks.describe()}, {spec.num_trials} trials, "
            f"{spec.target_per_cell} terminal results/cell = "
            f"{spec.total_target} simulations"
        )


def run_pool_status(args: argparse.Namespace) -> None:
    """Dispatch ``tau2 pool status``."""
    from tau2.runner.pool_driver import format_matrix
    from tau2.runner.pools import census_pool, get_pool

    spec = get_pool(args.pool)
    print(format_matrix(spec, census_pool(spec)))


def run_pool_run(args: argparse.Namespace) -> None:
    """Dispatch ``tau2 pool run``."""
    from tau2.runner.pool_driver import run_pool
    from tau2.runner.pools import get_pool
    from tau2.runner.work import parse_provider_limits

    spec = get_pool(args.pool)
    if args.max_steps_seconds is not None and spec.modality == "text":
        # The ceiling is simulated conversation seconds, a tick-loop concept.
        # A text cell never reads it, so accepting the flag would report a
        # widened budget that no simulation actually ran under.
        raise SystemExit(
            f"pool {spec.name!r} is a text pool; --max-steps-seconds is voice "
            "tick arithmetic and has no effect on a turn loop"
        )
    if args.max_steps_seconds is not None:
        from tau2.config import VOICE_TIMEOUT_SAFETY_FACTOR

        # PoolSpec is frozen on purpose; an override is a copy, never a mutation
        # of the registered spec. The timeout rides along at the same safety
        # factor the pool constants use, so a raised ceiling cannot be cut short
        # by a stale guard.
        spec = spec.model_copy(
            update={
                "max_steps_seconds": args.max_steps_seconds,
                "timeout_seconds": max(
                    spec.timeout_seconds,
                    int(args.max_steps_seconds * VOICE_TIMEOUT_SAFETY_FACTOR),
                ),
            }
        )
    code = run_pool(
        spec,
        only=[c.strip() for c in args.only.split(",")] if args.only else None,
        languages=[c.strip() for c in args.langs.split(",")] if args.langs else None,
        workers=args.workers,
        provider_limits=parse_provider_limits(args.provider_limit),
        dry_run=args.dry_run,
        redo_stale_tasks=args.redo_stale_tasks,
    )
    raise SystemExit(code)


def run_pool_repair_ceiling(args: argparse.Namespace) -> None:
    """Dispatch ``tau2 pool repair-ceiling``."""
    from tau2.runner.checkpoint import repair_ceiling

    try:
        report = repair_ceiling(
            args.save_to, args.max_steps_seconds, dry_run=args.dry_run
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(report.describe())
    if args.dry_run and report.changed:
        print("--dry-run: nothing written.")


# =============================================================================
# tau2 run refill
# =============================================================================


def add_refill_args(parser: argparse.ArgumentParser) -> None:
    """Attach the ``tau2 run refill`` arguments.

    Deliberately a short surface. Everything that describes the RUN — domain,
    task set, caller persona, provider, effort, seed, timeout, scoring axes —
    comes from the results the directory already holds, because re-typing it
    is how five preference-pool calls ended up in the wrong voice. The flags
    here are the two things a result cannot tell you (which cells to refill,
    how fast to go) plus escapes for directories written before the run config
    was recorded in full.
    """
    parser.add_argument(
        "--save-to",
        required=True,
        help="The run to refill: its directory, its results.json, or the "
        "--save-to name it was launched with (resolved under "
        "data/simulations/)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        required=True,
        help="Task id(s) whose simulations to delete and re-run",
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=None,
        help="Trial indices to refill, 0-based (default: every trial of the "
        "named tasks). A run with --num-trials 1 has only trial 0.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Concurrent simulations (default: the runner default). The one "
        "knob no result records, because it changes nothing about the calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rebuilt config and the cells that would be deleted "
        "and run, then stop. Nothing is deleted.",
    )

    legacy = parser.add_argument_group(
        "pre-recording escapes",
        "Only for directories written before results recorded the whole run "
        "config. Each one is REFUSED when the checkpoint does record the "
        "field, so it can never make a refilled call differ from the calls "
        "beside it.",
    )
    legacy.add_argument(
        "--user-persona-id",
        default=None,
        help="The caller the run spoke as (a language code like 'ko', or a "
        "pack persona id). Checked against the personas the directory's "
        "recorded calls actually used, so a wrong one is refused rather than "
        "written.",
    )
    legacy.add_argument(
        "--task-set-name",
        default=None,
        help="The task set the run's tasks came from (e.g. telecom_ko_identity). "
        "Needed for a directory written before results recorded it, whose "
        "tasks are not the domain's own — checked against the task ids the "
        "directory holds.",
    )
    legacy.add_argument(
        "--verbose-logs",
        action="store_true",
        default=None,
        help="Save per-task and LLM call logs, for a run that did.",
    )
    legacy.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-simulation wallclock guard in seconds, for a run that set "
        "one. 0 for a run that had none, as on `tau2 run`.",
    )
    legacy.add_argument(
        "--scores",
        default=None,
        help="Comma-separated scoring axes (reward,quality,nativeness,delivery) the "
        "run computed inline.",
    )
    legacy.add_argument(
        "--nativeness-llm-judge",
        action="store_true",
        help="The run ran the LLM nativeness judge. Omit it for the default: "
        "pack-selected deterministic checkers only, which is what every "
        "collected pool ran.",
    )
    parser.set_defaults(func=run_refill)


def run_refill(args: argparse.Namespace) -> None:
    """Dispatch ``tau2 run refill``."""
    from tau2.runner.refill import (
        RefillError,
        RefillOverrides,
        execute_refill,
        format_plan,
        plan_refill,
    )

    overrides = RefillOverrides(
        max_concurrency=args.max_concurrency,
        verbose_logs=args.verbose_logs,
        timeout=args.timeout,
        # Left unparsed: 'all' covers delivery only for a voice run, and
        # whether this directory is one is not known until it is read.
        scores_spec=args.scores,
        nativeness_llm_judge=True if args.nativeness_llm_judge else None,
        user_persona_id=args.user_persona_id,
        task_set_name=args.task_set_name,
    )
    try:
        plan = plan_refill(
            args.save_to,
            task_ids=args.task_ids,
            trials=args.trials,
            overrides=overrides,
        )
    except RefillError as exc:
        raise SystemExit(str(exc))

    print(format_plan(plan))
    if args.dry_run:
        print("\n--dry-run: nothing deleted, nothing run.")
        return

    report = execute_refill(plan)
    for cell in report.filled:
        print(f"refilled {cell.describe()}")
    if report.missing:
        raise SystemExit(
            f"{len(report.missing)} requested cell(s) are still empty: "
            + "; ".join(cell.describe() for cell in report.missing)
        )
