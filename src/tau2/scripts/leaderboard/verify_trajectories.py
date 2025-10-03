import argparse
import os
import sys
from enum import Enum

from loguru import logger
from rich.console import Console

from tau2.scripts.leaderboard.verify_trajectories_public import (
    check_format,
    check_num_trials,
    check_tasks,
)
from tau2.utils.io_utils import expand_paths


class VerificationMode(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


def verify_single_file(
    file_path: str, verification_mode: VerificationMode, console: Console
) -> bool:
    """Verify a single trajectory file."""
    console.print(f"\n📁 {file_path}", style="bold")
    console.print(f"Verification mode: {verification_mode.value}", style="bold")

    # Run checks
    results, format_ok, format_error = check_format(file_path)

    # Display format check
    if format_ok:
        console.print("  ✅ Format validation", style="green")
    else:
        console.print(f"  ❌ Format validation: {format_error}", style="red")
        return False

    # Only continue if format is valid
    tasks_ok, tasks_error = check_tasks(results)
    trials_ok, trials_error = check_num_trials(results)

    # Display quick checks first
    quick_checks = [
        ("Tasks validation", tasks_ok, tasks_error),
        ("Trial count validation", trials_ok, trials_error),
    ]

    all_passed = True
    for check_name, passed, error in quick_checks:
        if passed:
            console.print(f"  ✅ {check_name}", style="green")
        else:
            console.print(f"  ❌ {check_name}: {error}", style="red")
            all_passed = False

    if verification_mode == VerificationMode.PRIVATE:
        raise ValueError(
            "Private checks are not available in the public version of the leaderboard."
        )

    return all_passed


def verify_trajectories(paths: list[str], mode: VerificationMode):
    """Verify trajectories with given paths."""
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    console = Console()

    # Expand paths to list of files
    files = expand_paths(paths, extension=".json")

    if not files:
        console.print("❌ No trajectory files found", style="red")
        sys.exit(1)

    console.print(f"\n🔍 Verifying {len(files)} trajectory file(s)", style="bold blue")

    # Verify each file
    all_files_passed = True
    failed_files = []

    for file_path in files:
        if not os.path.exists(file_path):
            console.print(f"\n📁 {file_path}", style="bold")
            console.print(f"  ❌ File does not exist", style="red")
            all_files_passed = False
            failed_files.append(file_path)
            continue

        file_passed = verify_single_file(file_path, mode, console)
        if not file_passed:
            all_files_passed = False
            failed_files.append(file_path)

    # Summary
    console.print()
    console.print("=" * 60, style="dim")
    console.print(f"📊 Summary: {len(files)} file(s) processed", style="bold")

    if all_files_passed:
        console.print("🎉 All files passed all checks!", style="bold green")
    else:
        passed_count = len(files) - len(failed_files)
        console.print(f"✅ {passed_count} file(s) passed", style="green")
        console.print(f"❌ {len(failed_files)} file(s) failed", style="red")
        console.print()
        console.print("Failed files:", style="bold red")
        for failed_file in failed_files:
            console.print(f"  • {failed_file}", style="red")
        sys.exit(1)
