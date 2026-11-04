# Copyright Sierra
"""Package-owned registration for deterministic elicitation task generation."""

import argparse
from pathlib import Path

from tau2.config import DEFAULT_INTAKE_TASK_SEED


def add_intake_tasks_args(parser: argparse.ArgumentParser) -> None:
    """Register the enumerate, freeze, and read-only verify commands."""
    commands = parser.add_subparsers(dest="intake_tasks_command", required=True)
    enumerate_parser = commands.add_parser(
        "enumerate", help="Validate the banks and report the released task frame"
    )
    enumerate_parser.set_defaults(func=_enumerate)
    freeze = commands.add_parser(
        "freeze", help="Generate verified tasks, splits, and provenance"
    )
    freeze.add_argument("--seed", type=int, default=DEFAULT_INTAKE_TASK_SEED)
    freeze.add_argument(
        "--out", type=Path, help="Output directory (default: canonical domain data)"
    )
    freeze.set_defaults(func=_freeze)
    verify = commands.add_parser(
        "verify",
        help="Check frozen artifacts against their recorded draw without writing",
    )
    verify.add_argument(
        "--data-dir",
        type=Path,
        help="Artifact directory (default: canonical domain data)",
    )
    verify.set_defaults(func=_verify)


def _enumerate(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.generator import enumerate_frame

    frame = enumerate_frame()
    Console().print(
        f"{frame.banks} families × {frame.tiers} tiers × {frame.draw_per_cell} draws "
        f"= {frame.canonical_size} tasks ({frame.frame_size} eligible values)"
    )


def _freeze(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.generator import freeze_tasks

    outcome = freeze_tasks(seed=args.seed, out_dir=args.out)
    Console().print(
        f"{outcome.task_count} tasks; seed {outcome.seed}; "
        f"{len(outcome.written)} files changed in {outcome.out_dir}"
    )


def _verify(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.generator import verify_tasks

    outcome = verify_tasks(data_dir=args.data_dir)
    Console().print(
        f"Verified {outcome.task_count} tasks; seed {outcome.seed}; {outcome.data_dir}"
    )
