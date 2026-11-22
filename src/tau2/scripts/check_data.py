#!/usr/bin/env python3
"""
Script to check if the tau2 data directory is properly configured.
"""

import json
import sys
from pathlib import Path

from tau2.utils.utils import DATA_DIR


def find_reward_basis_issues(tasks: list[dict]) -> tuple[list[str], list[str]]:
    """Find reward criteria that cannot evaluate what their basis declares."""
    errors = []
    warnings = []
    for task in tasks:
        task_id = task.get("id", "<unknown>")
        criteria = task.get("evaluation_criteria") or {}
        basis = set(
            criteria["reward_basis"]
            if "reward_basis" in criteria
            else ("DB", "COMMUNICATE")
        )
        communicate_info = criteria.get("communicate_info")
        nl_assertions = criteria.get("nl_assertions")

        if "COMMUNICATE" in basis and not communicate_info:
            errors.append(
                f"task {task_id}: COMMUNICATE is in reward_basis but "
                "communicate_info is empty"
            )
        if "NL_ASSERTION" in basis and not nl_assertions:
            errors.append(
                f"task {task_id}: NL_ASSERTION is in reward_basis but "
                "nl_assertions is empty"
            )
        if communicate_info and "COMMUNICATE" not in basis:
            warnings.append(
                f"task {task_id}: communicate_info is populated but "
                "COMMUNICATE is not in reward_basis"
            )
    return errors, warnings


def check_task_files(data_dir: Path) -> tuple[list[str], list[str]]:
    """Validate reward criteria in every domain task file."""
    errors = []
    warnings = []
    domains_dir = data_dir / "tau2" / "domains"
    for task_file in sorted(domains_dir.glob("*/tasks.json")):
        if task_file.parent.name == "mock":
            # Mock tasks intentionally exercise omitted/default criteria fields.
            continue
        tasks = json.loads(task_file.read_text(encoding="utf-8"))
        domain_errors, domain_warnings = find_reward_basis_issues(tasks)
        errors.extend(f"{task_file}: {message}" for message in domain_errors)
        warnings.extend(f"{task_file}: {message}" for message in domain_warnings)
    return errors, warnings


def main():
    """Main function to check data directory and task criteria."""
    print("tau2 Data Directory Checker")
    print("=" * 40)
    print(f"Data directory: {DATA_DIR}")

    if DATA_DIR.exists():
        print("Data directory exists")
        errors, warnings = check_task_files(DATA_DIR)
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            print(f"Found {len(errors)} invalid reward criteria.")
            sys.exit(1)
        print("You can now run tau2 commands.")
    else:
        print("Data directory does not exist!")
        print("\nTo fix this, you can:")
        print("1. Set the TAU2_DATA_DIR environment variable:")
        print("   export TAU2_DATA_DIR=/path/to/your/data")
        print("2. Or ensure the data directory exists in the expected location")
        sys.exit(1)


if __name__ == "__main__":
    main()
