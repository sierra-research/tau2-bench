"""Package-owned frozen-subset commands; registration is owned by the root CLI."""

import argparse
import re
from typing import get_args

from pydantic import TypeAdapter

from tau2.task_subsets.sampler import build_subset
from tau2.task_subsets.schema import Domain, DrawRequest, TaskSubset, Verification
from tau2.task_subsets.store import (
    list_subsets,
    load_frame,
    load_subset,
    save_subset,
    verify_installed_bindings,
    verify_subset,
)


def add_tasks_args(
    parser: argparse.ArgumentParser, *, default_size: int, default_seed: int
) -> None:
    """Register verbs using centrally supplied draw defaults at integration."""
    sub = parser.add_subparsers(dest="tasks_command", required=True)
    listing = sub.add_parser("list", help="List frozen canonical subsets")
    listing.set_defaults(func=run_tasks_list)
    show = sub.add_parser("show", help="Show one subset's design and membership")
    show.add_argument("name")
    show.add_argument("--ids", action="store_true", help="Include all canonical IDs")
    show.set_defaults(func=run_tasks_show)
    verify = sub.add_parser(
        "verify", help="Verify frames, draws and installed bindings"
    )
    verify.add_argument("--name", help="One artifact; omitted verifies every artifact")
    verify.set_defaults(func=run_tasks_verify)
    draw = sub.add_parser("subset", help="Publish a seeded draw under a new name")
    draw.add_argument("--domain", choices=get_args(Domain), required=True)
    draw.add_argument(
        "--name", required=True, help="New artifact name; never overwritten"
    )
    draw.add_argument("--size", type=int, default=default_size)
    draw.add_argument("--seed", type=int, default=default_seed)
    draw.add_argument(
        "--dry-run", action="store_true", help="Print the draw without writing"
    )
    draw.set_defaults(func=run_tasks_subset)


def run_tasks_list(args: argparse.Namespace) -> None:
    """Print typed artifact content; corrupt declared data fails on inspection."""
    subsets = [load_subset(name) for name in list_subsets()]
    print(TypeAdapter(list[TaskSubset]).dump_json(subsets, indent=2).decode())


def run_tasks_show(args: argparse.Namespace) -> None:
    """Show the contract, optionally suppressing its potentially long ID list."""
    subset = load_subset(args.name)
    print(subset.model_dump_json(indent=2, exclude=set() if args.ids else {"task_ids"}))


def run_tasks_verify(args: argparse.Namespace) -> None:
    """Verify all selected artifacts before emitting a successful report."""
    names = [args.name] if args.name else list_subsets()
    if not names:
        raise ValueError("no task subset artifacts found")
    reports = []
    for name in names:
        subset = load_subset(name)
        report = verify_subset(subset)
        report.languages = verify_installed_bindings(subset)
        reports.append(report)
    print(TypeAdapter(list[Verification]).dump_json(reports, indent=2).decode())


def run_tasks_subset(args: argparse.Namespace) -> None:
    """Draw in memory first; immutable publication refuses existing changed data."""
    from tau2.utils.utils import get_commit_hash

    request = DrawRequest(
        name=args.name, domain=args.domain, size=args.size, seed=args.seed
    )
    commit = get_commit_hash()
    subset = build_subset(
        request,
        load_frame(request.domain),
        git_commit=commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None,
    )
    if args.dry_run:
        print(subset.model_dump_json(indent=2))
    else:
        print(save_subset(subset))
