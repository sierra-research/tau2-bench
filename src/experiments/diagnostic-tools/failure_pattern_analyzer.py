#!/usr/bin/env python3
"""
Failure Pattern Analyzer for τ²-bench

Analyzes agent evaluation results and groups failures by root cause pattern,
providing actionable recommendations for agent improvement.

Usage:
    python failure_pattern_analyzer.py results.json
    python failure_pattern_analyzer.py results.json --output json
    python failure_pattern_analyzer.py results.json --top 5

License: MIT
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── Failure Pattern Definitions ──────────────────────────────────────

PATTERNS = {
    "identity_not_verified": {
        "description": "Agent acted before identifying the customer",
        "keywords": ["identify", "user_id", "customer_id", "verification", "name", "who"],
        "action_keywords": ["get_user_details", "find_user_id", "get_customer_details"],
        "recommendation": "Ensure the agent always calls an identity tool before any action.",
    },
    "missing_confirmation": {
        "description": "Write action executed without user confirmation",
        "keywords": ["confirm", "approval", "consent", "permission", "agree"],
        "action_keywords": ["cancel", "book", "update", "modify", "change"],
        "recommendation": "Add confirmation gate before all write operations.",
    },
    "wrong_tool": {
        "description": "Agent called a non-existent or incorrect tool",
        "keywords": ["tool", "function", "not found", "invalid", "hallucinate"],
        "action_keywords": [],
        "recommendation": "Validate tool names against available tools list before calling.",
    },
    "data_not_retrieved": {
        "description": "Write action before retrieving relevant data",
        "keywords": ["retrieve", "lookup", "fetch", "details", "information"],
        "action_keywords": ["get_", "search_", "check_", "find_"],
        "recommendation": "Always retrieve current state before attempting modifications.",
    },
    "policy_violation": {
        "description": "Action violated domain policy rules",
        "keywords": ["policy", "rule", "violation", "not allowed", "prohibited", "cannot"],
        "action_keywords": [],
        "recommendation": "Strengthen policy awareness in system prompt or add verification.",
    },
    "communication_error": {
        "description": "Agent did not communicate required information to user",
        "keywords": ["inform", "communicate", "tell", "notify", "message", "response"],
        "action_keywords": [],
        "recommendation": "Add explicit communication checks after each action.",
    },
    "multi_fault_missed": {
        "description": "Agent fixed one issue but missed additional faults",
        "keywords": ["additional", "also", "another", "remaining", "missed", "incomplete"],
        "action_keywords": [],
        "recommendation": "Add checklist-style verification after each fix.",
    },
}


def classify_failure(task_result: dict) -> str:
    """Classify a failed task into a failure pattern category."""
    action_checks = task_result.get("action_checks", [])
    failed_checks = [
        c for c in action_checks if not c.get("passed", True)
    ]

    if not failed_checks:
        # No explicit check failures — might be reward-based failure
        return "unknown"

    # Analyze failed check descriptions
    all_text = " ".join(
        c.get("description", "") + " " + c.get("message", "")
        for c in failed_checks
    ).lower()

    # Score each pattern
    scores = {}
    for pattern_name, pattern_info in PATTERNS.items():
        score = 0
        for kw in pattern_info["keywords"]:
            if kw in all_text:
                score += 1
        for kw in pattern_info["action_keywords"]:
            if kw in all_text:
                score += 2  # Action keywords are more specific
        scores[pattern_name] = score

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "unknown"


def analyze_results(results: list[dict]) -> dict[str, Any]:
    """Analyze a list of task results and group by failure pattern."""
    total = len(results)
    passed = sum(1 for r in results if r.get("reward", 0) >= 1.0)
    failed = total - passed

    failed_tasks = [r for r in results if r.get("reward", 0) < 1.0]

    pattern_groups: dict[str, list[str]] = defaultdict(list)
    for task in failed_tasks:
        pattern = classify_failure(task)
        task_id = task.get("task_id", task.get("id", "unknown"))
        pattern_groups[pattern].append(task_id)

    # Build summary
    patterns_summary = []
    for pattern_name in sorted(pattern_groups.keys(), key=lambda k: -len(pattern_groups[k])):
        task_ids = pattern_groups[pattern_name]
        info = PATTERNS.get(pattern_name, {
            "description": "Unclassifiable failure",
            "recommendation": "Manual investigation required.",
        })
        patterns_summary.append({
            "pattern": pattern_name,
            "description": info.get("description", ""),
            "count": len(task_ids),
            "percentage": round(len(task_ids) / max(failed, 1) * 100, 1),
            "task_ids": task_ids[:10],  # Show up to 10 examples
            "recommendation": info.get("recommendation", ""),
        })

    return {
        "total_tasks": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1) * 100, 1),
        "patterns": patterns_summary,
    }


def format_text_report(analysis: dict[str, Any], top_n: int = 0) -> str:
    """Format analysis as human-readable text report."""
    lines = [
        "=" * 60,
        "  τ²-bench Failure Pattern Analysis",
        "=" * 60,
        "",
        f"  Total tasks:  {analysis['total_tasks']}",
        f"  Passed:       {analysis['passed']}",
        f"  Failed:       {analysis['failed']}",
        f"  Pass rate:    {analysis['pass_rate']}%",
        "",
        "-" * 60,
        "  Failure Patterns (sorted by frequency)",
        "-" * 60,
    ]

    patterns = analysis["patterns"]
    if top_n > 0:
        patterns = patterns[:top_n]

    for p in patterns:
        lines.append("")
        lines.append(f"  [{p['pattern']}] — {p['count']} failures ({p['percentage']}%)")
        lines.append(f"    {p['description']}")
        lines.append(f"    Example task IDs: {', '.join(p['task_ids'][:5])}")
        lines.append(f"    → {p['recommendation']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze τ²-bench results for failure patterns"
    )
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Show only top N patterns (0 = all)"
    )
    args = parser.parse_args()

    path = Path(args.results_file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        results = json.load(f)

    if isinstance(results, dict):
        results = results.get("results", results.get("tasks", []))

    analysis = analyze_results(results)

    if args.output == "json":
        print(json.dumps(analysis, indent=2))
    else:
        print(format_text_report(analysis, args.top))


if __name__ == "__main__":
    main()
