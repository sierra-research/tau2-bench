#!/usr/bin/env python3
"""
Script to recombine rewards in results.json using updated reward_basis from tasks.json.

This script does NOT re-run evaluators - it just recombines existing component rewards
using the new reward_basis defined in the current task definitions.

Usage:
    python src/tau2/scripts/recombine_rewards.py <experiment_dir>

Example:
    python src/tau2/scripts/recombine_rewards.py data/tmp/gdrive/experiment_2025_01_22_v3_regular_openai_low_vad
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def compute_component_reward(reward_info: dict, component: str) -> float:
    """Compute reward for a single component from stored data."""
    if component == "DB":
        db_check = reward_info.get("db_check")
        if db_check:
            return db_check.get("db_reward", 1.0)
        return 1.0  # No DB check defined

    elif component == "ENV_ASSERTION":
        env_assertions = reward_info.get("env_assertions")
        if not env_assertions:
            return 1.0  # No assertions defined
        reward = 1.0
        for assertion in env_assertions:
            reward *= assertion.get("reward", 1.0)
        return reward

    elif component == "ACTION":
        action_checks = reward_info.get("action_checks")
        if not action_checks:
            return 1.0  # No actions defined
        reward = 1.0
        for check in action_checks:
            reward *= check.get("action_reward", 1.0)
        return reward

    elif component == "COMMUNICATE":
        communicate_checks = reward_info.get("communicate_checks")
        if not communicate_checks:
            return 1.0  # No communicate checks defined
        reward = 1.0
        for check in communicate_checks:
            reward *= 1.0 if check.get("met", False) else 0.0
        return reward

    elif component == "NL_ASSERTION":
        nl_assertions = reward_info.get("nl_assertions")
        if not nl_assertions:
            return 1.0  # No NL assertions defined
        reward = 1.0
        for assertion in nl_assertions:
            reward *= 1.0 if assertion.get("met", False) else 0.0
        return reward

    else:
        print(f"  Warning: Unknown component: {component}")
        return 1.0


def recombine_reward(reward_info: dict, new_basis: list[str]) -> tuple[float, dict]:
    """
    Recombine reward using a new reward_basis.

    Returns:
        Tuple of (new_reward, new_breakdown)
    """
    reward = 1.0
    breakdown = {}

    for component in new_basis:
        component_reward = compute_component_reward(reward_info, component)
        breakdown[component] = component_reward
        reward *= component_reward

    return reward, breakdown


def recombine_results(experiment_dir: str, domain: str | None = None) -> None:
    """
    Recombine rewards in results.json files using updated task definitions.

    Args:
        experiment_dir: Path to experiment directory containing subdirectories with results.json
        domain: Optional domain to filter (e.g., 'retail'). If None, processes all.
    """
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        print(f"Error: Directory not found: {experiment_dir}")
        sys.exit(1)

    # Find all results.json files (follow symlinks, use find for speed on network drives)
    print(f"Searching for results.json files in {experiment_dir}...")
    try:
        result = subprocess.run(
            ["find", "-L", str(experiment_path), "-name", "results.json", "-type", "f"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for find
        )
        results_files = [Path(p) for p in result.stdout.strip().split("\n") if p]
    except subprocess.TimeoutExpired:
        print("Error: Search timed out. Try a more specific directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error searching for files: {e}")
        sys.exit(1)

    if not results_files:
        print(f"No results.json files found in {experiment_dir}")
        sys.exit(1)

    print(f"Found {len(results_files)} results.json file(s)")

    for results_file in results_files:
        print(f"\nProcessing: {results_file}")

        # Load results
        with open(results_file, "r") as f:
            results = json.load(f)

        # Get domain from results
        domain_name = (
            results.get("info", {}).get("environment_info", {}).get("domain_name")
        )
        if not domain_name:
            print(f"  Warning: Could not determine domain, skipping")
            continue

        if domain and domain_name != domain:
            print(f"  Skipping domain {domain_name}")
            continue

        # Load current task definitions
        tasks_file = Path(f"data/tau2/domains/{domain_name}/tasks.json")
        if not tasks_file.exists():
            print(f"  Warning: Tasks file not found: {tasks_file}, skipping")
            continue

        with open(tasks_file, "r") as f:
            tasks_data = json.load(f)

        # Build task lookup by ID
        tasks_by_id = {task["id"]: task for task in tasks_data}

        # Update tasks in results with new reward_basis
        updated_task_count = 0
        for task in results.get("tasks", []):
            task_id = task.get("id")
            if task_id in tasks_by_id:
                new_ec = tasks_by_id[task_id].get("evaluation_criteria", {})
                new_basis = new_ec.get("reward_basis")
                if new_basis:
                    if task.get("evaluation_criteria") is None:
                        task["evaluation_criteria"] = {}
                    task["evaluation_criteria"]["reward_basis"] = new_basis
                    updated_task_count += 1

        print(f"  Updated reward_basis for {updated_task_count} tasks")

        # Recombine rewards for each simulation
        simulations = results.get("simulations", [])
        updated_sim_count = 0
        skipped_not_evaluated = 0
        for sim in simulations:
            task_id = sim.get("task_id")
            reward_info = sim.get("reward_info")

            if not reward_info:
                continue

            # Skip simulations where evaluation was not done (e.g., max steps, errors)
            # These should keep reward=0.0, not be recombined
            # Check 1: Original reward_basis was None (evaluation never ran)
            original_basis = reward_info.get("reward_basis")
            # Check 2: Termination reason indicates premature end (more reliable)
            termination_reason = sim.get("termination_reason")
            valid_terminations = {"agent_stop", "user_stop"}

            if original_basis is None or termination_reason not in valid_terminations:
                # Fix corrupted rewards: set back to 0.0 if this was a premature termination
                if termination_reason and termination_reason not in valid_terminations:
                    if reward_info.get("reward") != 0.0:
                        reward_info["reward"] = 0.0
                        reward_info["reward_basis"] = None
                        reward_info["reward_breakdown"] = None
                        reward_info["info"] = {
                            "note": f"Simulation terminated prematurely. Termination reason: {termination_reason}"
                        }
                        updated_sim_count += 1
                skipped_not_evaluated += 1
                continue

            # Get new reward_basis from updated task
            task = tasks_by_id.get(task_id)
            if not task:
                continue

            new_basis = task.get("evaluation_criteria", {}).get("reward_basis")
            if not new_basis:
                continue

            # Recombine reward
            old_reward = reward_info.get("reward")
            new_reward, new_breakdown = recombine_reward(reward_info, new_basis)

            # Update reward_info
            reward_info["reward"] = new_reward
            reward_info["reward_basis"] = new_basis
            reward_info["reward_breakdown"] = new_breakdown

            if old_reward != new_reward:
                updated_sim_count += 1

        print(
            f"  Recombined rewards for {len(simulations)} simulations ({updated_sim_count} changed, {skipped_not_evaluated} skipped - not evaluated)"
        )

        # Save updated results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Recombine rewards in results.json using updated task reward_basis"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory containing results.json files",
    )
    parser.add_argument(
        "--domain",
        help="Optional: only process this domain (e.g., 'retail')",
        default=None,
    )

    args = parser.parse_args()
    recombine_results(args.experiment_dir, args.domain)


if __name__ == "__main__":
    main()
