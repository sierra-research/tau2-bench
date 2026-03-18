#!/usr/bin/env python3
"""
Extract experiment results from results.json files.

Usage:
    python src/tau2/scripts/get_experiment_results.py <experiment_dir>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_results_summary(results_file: Path, first_n: int | None = None) -> dict:
    """Extract summary statistics from a results.json file."""
    with open(results_file) as f:
        data = json.load(f)

    sims = data.get("simulations", [])

    # Filter to first N tasks if specified
    if first_n:
        sims = [
            s
            for s in sims
            if s.get("task_id") is not None and int(s.get("task_id")) < first_n
        ]

    n = len(sims)
    if n == 0:
        return None

    # Termination reasons
    term_reasons = {}
    for s in sims:
        tr = s.get("termination_reason", "unknown")
        term_reasons[tr] = term_reasons.get(tr, 0) + 1

    # Failed terminations (max_steps, too_many_errors, infrastructure_error)
    fail_terms = {"max_steps", "too_many_errors", "infrastructure_error"}
    fail_count = sum(term_reasons.get(t, 0) for t in fail_terms)

    # Rewards (count of reward=1.0)
    rewards = [
        s.get("reward_info", {}).get("reward", 0) if s.get("reward_info") else 0
        for s in sims
    ]
    success_count = sum(1 for r in rewards if r == 1.0)

    # DB match
    db_ok = db_fail = 0
    for s in sims:
        db = (s.get("reward_info") or {}).get("db_check")
        if db:
            if db.get("db_match"):
                db_ok += 1
            else:
                db_fail += 1
    db_total = db_ok + db_fail

    # Auth (LLM judge)
    auth_ok = auth_fail = 0
    for s in sims:
        auth = s.get("auth_classification")
        if auth:
            status = auth.get("status")
            if status == "succeeded":
                auth_ok += 1
            elif status == "failed":
                auth_fail += 1
    auth_total = auth_ok + auth_fail

    return {
        "file": str(results_file),
        "n": n,
        "fail_term": fail_count,
        "auth_ok": auth_ok,
        "auth_total": auth_total,
        "db_ok": db_ok,
        "db_total": db_total,
        "reward_ok": success_count,
    }


def fmt(num: int, total: int) -> str:
    """Format as 'num/total (pct%)'."""
    if total == 0:
        return "-"
    pct = num / total * 100
    return f"{num}/{total} ({pct:.1f}%)"


def main():
    parser = argparse.ArgumentParser(description="Extract experiment results")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--first-n", type=int, help="Only use first N tasks")
    args = parser.parse_args()

    exp_path = Path(args.experiment_dir)
    if not exp_path.exists():
        sys.exit(f"Error: {args.experiment_dir} not found")

    # Find results.json files
    result = subprocess.run(
        ["find", "-L", str(exp_path), "-name", "results.json", "-type", "f"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    files = [Path(p) for p in result.stdout.strip().split("\n") if p]

    if not files:
        sys.exit(f"No results.json found in {args.experiment_dir}")

    summaries = [get_results_summary(f, args.first_n) for f in sorted(files)]
    summaries = [s for s in summaries if s]  # Filter out None

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for s in summaries:
            n = s["n"]
            fail_str = fmt(s["fail_term"], n)
            auth_str = fmt(s["auth_ok"], s["auth_total"])
            db_str = fmt(s["db_ok"], s["db_total"])
            reward_str = fmt(s["reward_ok"], n)

            print(f"{s['file']}")
            print(
                f"  n={n}  fail_term={fail_str}  auth={auth_str}  db={db_str}  reward={reward_str}"
            )


if __name__ == "__main__":
    main()
