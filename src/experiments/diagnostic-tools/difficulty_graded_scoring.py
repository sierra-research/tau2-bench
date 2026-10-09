#!/usr/bin/env python3
"""
Difficulty-Graded Scoring for τ²-bench

Analyzes tasks by number of action checks (faults) and computes
pass rates per difficulty tier, revealing which difficulty levels
different agents struggle with.

Usage:
    python difficulty_graded_scoring.py --tasks tasks.json --results results.json
    python difficulty_graded_scoring.py --tasks tasks.json --results results.json --domain airline
    python difficulty_graded_scoring.py --tasks tasks.json --results results.json --output json

License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Difficulty Tiers ─────────────────────────────────────────────────

TIERS = {
    "easy": {"min": 1, "max": 3, "label": "Easy (1-3 checks)"},
    "medium": {"min": 4, "max": 6, "label": "Medium (4-6 checks)"},
    "hard": {"min": 7, "max": 999, "label": "Hard (7+ checks)"},
}


def count_action_checks(task: dict) -> int:
    """Count the number of action checks in a task."""
    checks = task.get("action_checks", [])
    if isinstance(checks, list):
        return len(checks)
    return 0


def get_tier(n_checks: int) -> str:
    """Determine difficulty tier based on number of checks."""
    for tier_name, tier_info in TIERS.items():
        if tier_info["min"] <= n_checks <= tier_info["max"]:
            return tier_name
    return "hard"


def analyze(tasks: list[dict], results: list[dict]) -> dict[str, Any]:
    """Analyze tasks and results by difficulty tier."""
    # Build task -> n_checks mapping
    task_checks = {}
    for task in tasks:
        task_id = task.get("id", task.get("task_id", ""))
        n = count_action_checks(task)
        task_checks[task_id] = n

    # Build results lookup
    result_map = {}
    for r in results:
        task_id = r.get("task_id", r.get("id", ""))
        result_map[task_id] = r.get("reward", 0) >= 1.0

    # Categorize by tier
    tier_data: dict[str, dict] = {}
    for tier_name in TIERS:
        tier_data[tier_name] = {
            "label": TIERS[tier_name]["label"],
            "tasks": [],
            "passed": [],
            "failed": [],
        }

    for task_id, n_checks in task_checks.items():
        tier = get_tier(n_checks)
        tier_data[tier]["tasks"].append(task_id)
        if task_id in result_map:
            if result_map[task_id]:
                tier_data[tier]["passed"].append(task_id)
            else:
                tier_data[tier]["failed"].append(task_id)

    # Compute stats
    tiers_summary = []
    for tier_name in ["easy", "medium", "hard"]:
        td = tier_data[tier_name]
        total = len(td["tasks"])
        evaluated = len(td["passed"]) + len(td["failed"])
        passed = len(td["passed"])
        pass_rate = round(passed / max(evaluated, 1) * 100, 1)
        tiers_summary.append(
            {
                "tier": tier_name,
                "label": td["label"],
                "total_tasks": total,
                "evaluated": evaluated,
                "passed": passed,
                "failed": len(td["failed"]),
                "pass_rate": pass_rate,
                "failed_task_ids": td["failed"][:10],
            }
        )

    overall_evaluated = sum(t["evaluated"] for t in tiers_summary)
    overall_passed = sum(t["passed"] for t in tiers_summary)

    return {
        "tiers": tiers_summary,
        "overall_pass_rate": round(overall_passed / max(overall_evaluated, 1) * 100, 1),
        "overall_evaluated": overall_evaluated,
        "overall_passed": overall_passed,
    }


def format_text_report(analysis: dict[str, Any], domain: str = "") -> str:
    """Format analysis as human-readable text report."""
    header = f"τ²-bench Difficulty-Graded Scoring"
    if domain:
        header += f" — {domain}"

    lines = [
        "=" * 65,
        f"  {header}",
        "=" * 65,
        "",
        f"  {'Tier':<25} {'Tasks':>6} {'Eval':>6} {'Pass':>6} {'Fail':>6} {'Rate':>7}",
        "  " + "-" * 60,
    ]

    for t in analysis["tiers"]:
        lines.append(
            f"  {t['label']:<25} {t['total_tasks']:>6} {t['evaluated']:>6} "
            f"{t['passed']:>6} {t['failed']:>6} {t['pass_rate']:>6.1f}%"
        )

    lines.extend(
        [
            "  " + "-" * 60,
            f"  {'Overall':<25} {'':>6} {analysis['overall_evaluated']:>6} "
            f"{analysis['overall_passed']:>6} {'':>6} {analysis['overall_pass_rate']:>6.1f}%",
            "",
        ]
    )

    # Show failed task IDs per tier
    for t in analysis["tiers"]:
        if t["failed_task_ids"]:
            lines.append(
                f"  {t['tier']} failures: {', '.join(t['failed_task_ids'][:5])}"
            )

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Difficulty-graded scoring for τ²-bench"
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--domain", default="", help="Domain label for report")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    results_path = Path(args.results)

    if not tasks_path.exists():
        print(f"Error: Tasks file not found: {tasks_path}", file=sys.stderr)
        sys.exit(1)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])
    if isinstance(results, dict):
        results = results.get("results", results.get("tasks", []))

    analysis = analyze(tasks, results)

    if args.output == "json":
        print(json.dumps(analysis, indent=2))
    else:
        print(format_text_report(analysis, args.domain))


if __name__ == "__main__":
    main()
