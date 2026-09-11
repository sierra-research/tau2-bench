# Copyright Sierra
"""Package-owned registration for the ``tau2 tasks`` verbs.

``subset`` builds a subset artifact, ``list``/``show`` read them, ``verify``
checks every checked-in subset still fits its frame. Building is a CLI verb
rather than a script so the draw is reproducible by anyone with the repo.
"""

import argparse

from tau2.config import DEFAULT_TASK_SUBSET_SEED, DEFAULT_TASK_SUBSET_SIZE


def add_tasks_args(parser: argparse.ArgumentParser) -> None:
    """Attach the sub-subcommands to the ``tau2 tasks`` parser."""
    tasks_sub = parser.add_subparsers(
        dest="tasks_command", help="Task subset commands", required=True
    )

    # --- subset ------------------------------------------------------------
    subset_parser = tasks_sub.add_parser(
        "subset",
        help="Draw a fixed task subset for one domain (seeded, stratified)",
    )
    subset_parser.add_argument(
        "--domain",
        required=True,
        help="Domain the subset belongs to (airline, retail, telecom, ...)",
    )
    subset_parser.add_argument(
        "--task-set-name",
        default=None,
        help="Task set to draw the frame from (default: same name as --domain)",
    )
    subset_parser.add_argument(
        "--task-split-name",
        default="base",
        help="Task split for the frame (default %(default)s)",
    )
    subset_parser.add_argument(
        "--name",
        default=None,
        help="Subset name / filename stem (default <domain>_<size>)",
    )
    subset_parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_TASK_SUBSET_SIZE,
        help="Tasks to draw (default %(default)s)",
    )
    subset_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TASK_SUBSET_SEED,
        help="Seed for the within-stratum draw (default %(default)s)",
    )
    subset_parser.add_argument(
        "--stratifier",
        default=None,
        help="Override the domain's stratifier (see task_subsets.sampler)",
    )
    subset_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact. Only for a subset nothing has run on",
    )
    subset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the draw without writing the artifact",
    )
    subset_parser.set_defaults(func=run_tasks_subset)

    # --- prefix ------------------------------------------------------------
    prefix_parser = tasks_sub.add_parser(
        "prefix",
        help="Freeze the first N ids of an existing subset as a derived "
        "subset (pairs exactly with the parent's runs on the same stems)",
    )
    prefix_parser.add_argument(
        "--of", required=True, help="Parent subset name (e.g. retail_50)"
    )
    prefix_parser.add_argument(
        "--size", type=int, required=True, help="How many leading ids to keep"
    )
    prefix_parser.add_argument(
        "--name",
        default=None,
        help="Derived subset name (default <domain>_<size>)",
    )
    prefix_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact. Only for a subset nothing has run on",
    )
    prefix_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derivation without writing the artifact",
    )
    prefix_parser.set_defaults(func=run_tasks_prefix)

    # --- list --------------------------------------------------------------
    list_parser = tasks_sub.add_parser("list", help="List checked-in task subsets")
    list_parser.set_defaults(func=run_tasks_list)

    # --- show --------------------------------------------------------------
    show_parser = tasks_sub.add_parser(
        "show", help="Show one subset: design, strata, ids"
    )
    show_parser.add_argument("name", help="Subset name")
    show_parser.add_argument(
        "--ids", action="store_true", help="Also print every task id"
    )
    show_parser.set_defaults(func=run_tasks_show)

    # --- verify ------------------------------------------------------------
    verify_parser = tasks_sub.add_parser(
        "verify",
        help="Check every subset still matches its frame and resolves against "
        "the localized task sets that use it",
    )
    verify_parser.add_argument(
        "--name", default=None, help="Verify one subset (default: all)"
    )
    verify_parser.set_defaults(func=run_tasks_verify)


# ---------------------------------------------------------------------------
# Verb implementations
# ---------------------------------------------------------------------------


def _print_summary(subset) -> None:
    print(f"{subset.name}: {subset.size} tasks from {subset.domain}")
    if subset.derived_from:
        print(f"  derived   first {subset.size} of subset {subset.derived_from}")
    print(
        f"  frame     {subset.frame.task_set} ({subset.frame.size} tasks) "
        f"digest {subset.frame.digest[:12]}"
    )
    print(
        f"  design    {subset.strategy.value} / {subset.stratifier} / seed {subset.seed}"
    )
    print(f"  created   {subset.created} @ {subset.git_commit}")
    print("  strata:")
    for stratum in subset.strata:
        print(
            f"    {stratum.key:<48} {stratum.drawn:>3}/{stratum.frame_size:<4} "
            f"pi={stratum.inclusion_probability:.2f}"
        )


def run_tasks_subset(args) -> None:
    from tau2.runner.helpers import get_tasks
    from tau2.task_subsets.sampler import build_subset
    from tau2.task_subsets.store import save_subset

    task_set = args.task_set_name or args.domain
    name = args.name or f"{args.domain}_{args.size}"
    tasks = get_tasks(task_set_name=task_set, task_split_name=args.task_split_name)
    subset = build_subset(
        name=name,
        domain=args.domain,
        task_set=task_set,
        task_split=args.task_split_name,
        tasks=tasks,
        size=args.size,
        seed=args.seed,
        stratifier=args.stratifier,
    )
    _print_summary(subset)
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    path = save_subset(subset, overwrite=args.overwrite)
    print(f"\nWrote {path}")


def run_tasks_prefix(args) -> None:
    from tau2.runner.helpers import get_tasks
    from tau2.task_subsets.sampler import build_prefix_subset
    from tau2.task_subsets.store import load_subset, save_subset

    parent = load_subset(args.of)
    name = args.name or f"{parent.domain}_{args.size}"
    tasks = get_tasks(
        task_set_name=parent.frame.task_set,
        task_split_name=parent.frame.task_split,
    )
    subset = build_prefix_subset(name=name, parent=parent, tasks=tasks, size=args.size)
    _print_summary(subset)
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    path = save_subset(subset, overwrite=args.overwrite)
    print(f"\nWrote {path}")


def run_tasks_list(args) -> None:
    from tau2.task_subsets.store import list_subsets, load_subset

    names = list_subsets()
    if not names:
        print("No task subsets. Build one with `tau2 tasks subset --domain <domain>`.")
        return
    for name in names:
        subset = load_subset(name)
        print(
            f"{name:<20} {subset.domain:<18} {subset.size:>3} of "
            f"{subset.frame.size:<4} {subset.strategy.value}"
        )


def run_tasks_show(args) -> None:
    from tau2.task_subsets.store import load_subset

    subset = load_subset(args.name)
    _print_summary(subset)
    if args.ids:
        print("  task_ids:")
        for task_id in subset.task_ids:
            print(f"    {task_id}")


def run_tasks_verify(args) -> None:
    from tau2.multilingual.task_sets import discover_localized_task_files
    from tau2.runner.helpers import get_tasks
    from tau2.task_subsets.store import check_frame, list_subsets, load_subset

    names = [args.name] if args.name else list_subsets()
    if not names:
        print("No task subsets to verify.")
        return
    problems = 0
    for name in names:
        subset = load_subset(name)
        frame_tasks = get_tasks(
            task_set_name=subset.frame.task_set,
            task_split_name=subset.frame.task_split,
        )
        drift = check_frame(subset, [t.id for t in frame_tasks])
        if drift is not None:
            problems += 1
            print(f"DRIFT {name}: {drift}")
        else:
            print(f"OK    {name}: frame {subset.frame.task_set} unchanged")
        # The LOCALIZED task sets are renames of this frame, so every one of
        # them must resolve the subset's ids — otherwise a language's run is
        # silently short. Sibling sets of the same domain (telecom_full,
        # telecom_small) are different frames, not renames, and are skipped.
        for task_set in sorted(discover_localized_task_files()):
            if not task_set.startswith(f"{subset.domain}_"):
                continue
            try:
                subset.select(get_tasks(task_set_name=task_set))
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                problems += 1
                print(f"FAIL  {name} on {task_set}: {exc}")
    print(f"\n{len(names)} subset(s) checked, {problems} problem(s).")
    if problems:
        raise SystemExit(1)
