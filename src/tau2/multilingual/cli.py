# Copyright Sierra
"""``tau2 run-preset`` CLI registration (package-owned, wired from tau2.cli).

The argparse surface and mode dispatch for the multilingual run-preset verb
live here; the execution engine (preset arms, the langs x providers matrix,
subprocess plumbing) is ``tau2.multilingual.run_preset_driver``. Deliberately
imports nothing heavy: preset names are validated at dispatch (building the
preset table loads every language pack).
"""

from __future__ import annotations

import argparse
import shlex
import sys

from tau2.config import (
    DEFAULT_MULTILINGUAL_DOMAIN,
    DEFAULT_MULTILINGUAL_PROVIDERS,
    DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
)
from tau2.multilingual import run_preset_driver


def add_run_preset_args(parser: argparse.ArgumentParser) -> None:
    """Attach the ``tau2 run-preset`` arguments (registered from tau2.cli)."""
    parser.add_argument(
        "preset",
        nargs="?",
        default=None,
        help="Run preset name (e.g. multilingual_v1_hindi; see --list). Omit "
        "with --all-languages for matrix mode.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available run presets (generated from the registered "
        "language packs) and exit",
    )
    # Preset-mode-only flags default to None so passing them alongside
    # --all-languages is detectable and rejected (the real defaults are
    # resolved at dispatch).
    parser.add_argument(
        "--stage",
        choices=["smoke", "full"],
        default=None,
        help="smoke = 1 task per arm; full = all tasks (default: smoke; "
        "preset mode only)",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Run only this arm (default: all arms, sequentially; preset mode only)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Trials per task (default: 1)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
        help="Concurrent simulations per tau2 run (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands (with env overrides) without running them",
    )
    parser.add_argument(
        "--extra-run-args",
        default=None,
        help="Extra arguments forwarded verbatim to every `tau2 run` call "
        '(one shell-quoted string, e.g. --extra-run-args "--seed 7")',
    )
    # Matrix mode. These flags also default to None so passing one alongside a
    # preset name is detectable and rejected.
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Matrix mode: run every registered language pack (except en) "
        "across --providers",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help="Matrix mode: comma-separated audio-native providers, one "
        "concurrent lane each (default: "
        f"{','.join(DEFAULT_MULTILINGUAL_PROVIDERS)})",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=None,
        help="Matrix mode: max provider lanes running concurrently "
        "(default: all providers at once)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Matrix mode: only the first N tasks per cell (default: all)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Matrix mode: domain whose localized task sets to run "
        f"(default: {DEFAULT_MULTILINGUAL_DOMAIN})",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Matrix mode: base results dir under data/simulations/ "
        "(default: <domain>_matrix_<YYYY-MM-DD>)",
    )
    parser.set_defaults(func=run_run_preset)


def _reject_wrong_mode_flags(mode: str, flags: list[tuple[str, object]]) -> None:
    """Exit with an actionable message when a flag from the OTHER mode was passed."""
    wrong = [name for name, value in flags if value is not None]
    if wrong:
        sys.exit(
            f"{', '.join(wrong)} only appl{'ies' if len(wrong) == 1 else 'y'} "
            f"in {mode}. Remove the flag(s), or switch mode "
            "(a preset name = preset mode; --all-languages = matrix mode)."
        )


def _list_presets() -> None:
    # Deliberately lazy: building the preset table loads every language pack.
    from tau2.multilingual.run_presets import get_run_presets

    presets = get_run_presets()
    if not presets:
        print(
            "No run presets available (no registered pack has a usable experiment block)."
        )
        return
    width = max(len(name) for name in presets)
    for name in sorted(presets):
        print(f"{name:<{width}}  {presets[name].description}")


def run_run_preset(args) -> None:
    if args.list:
        _list_presets()
        return
    if args.all_languages and args.preset:
        sys.exit("Pass either a preset name or --all-languages, not both.")
    if args.all_languages:
        _reject_wrong_mode_flags(
            "preset mode (a preset name)",
            [("--stage", args.stage), ("--arm", args.arm)],
        )
        providers_csv = args.providers or ",".join(DEFAULT_MULTILINGUAL_PROVIDERS)
        rc = run_preset_driver.run_matrix(
            domain=args.domain or DEFAULT_MULTILINGUAL_DOMAIN,
            providers=[p for p in providers_csv.split(",") if p],
            lanes=args.lanes,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            max_concurrency=args.max_concurrency,
            base=args.base,
            dry_run=args.dry_run,
            extra_args=shlex.split(args.extra_run_args or ""),
        )
    elif args.preset:
        _reject_wrong_mode_flags(
            "matrix mode (--all-languages)",
            [
                ("--providers", args.providers),
                ("--lanes", args.lanes),
                ("--num-tasks", args.num_tasks),
                ("--domain", args.domain),
                ("--base", args.base),
            ],
        )
        rc = run_preset_driver.run_preset(
            args.preset,
            stage=args.stage or "smoke",
            arm=args.arm,
            num_trials=args.num_trials,
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
            extra_args=shlex.split(args.extra_run_args or ""),
        )
    else:
        sys.exit("Pass a preset name (see --list), or --all-languages for matrix mode.")
    if rc:
        sys.exit(rc)
