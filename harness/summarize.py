#!/usr/bin/env python3
"""Summarize v2 per-trial logs into pass^k, task-cluster CIs and a CSV.

Usage:
    python summarize.py --runs harness/runs --out harness/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

B = 300_000
SEED = 20261119


def comb(n: int, k: int) -> int:
    if n < k or k < 0:
        return 0
    return math.comb(n, k)


def pass_k(successes: int, trials: int, k: int) -> float:
    return comb(successes, k) / comb(trials, k) if trials >= k else 0.0


def mean_pass_k(counts: list[int], trials: list[int], k: int) -> float:
    return statistics.mean(pass_k(c, n, k) for c, n in zip(counts, trials))


def bootstrap_ci(
    counts: list[int], trials: list[int], k: int, rng: random.Random
) -> tuple[float, float]:
    n_tasks = len(counts)
    values = []
    for _ in range(B):
        idx = [rng.randrange(n_tasks) for _ in range(n_tasks)]
        values.append(
            statistics.mean(pass_k(counts[i], trials[i], k) for i in idx)
        )
    values.sort()
    return values[int(0.025 * B)], values[int(0.975 * B)]


def load_reward(path: Path) -> tuple[int, dict]:
    data = json.loads(path.read_text())
    reward_info = data.get("reward_info") or {}
    reward = reward_info.get("reward")
    return (1 if reward and reward >= 1.0 else 0), data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("summary"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    per_model_task: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for sim_path in sorted(args.runs.glob("*/*/trial_*/simulation.json")):
        model = sim_path.parts[-4]
        task = sim_path.parts[-3].replace("task_", "")
        reward, data = load_reward(sim_path)
        per_model_task[model][task].append(reward)
        rows.append(
            {
                "model": model,
                "task": task,
                "trial": sim_path.parts[-2].replace("trial_", ""),
                "reward": reward,
                "path": str(sim_path),
                "termination_reason": (data.get("termination_reason") or ""),
            }
        )

    csv_path = args.out / "trials.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    rng = random.Random(SEED)

    for model, task_map in sorted(per_model_task.items()):
        tasks = sorted(task_map, key=lambda x: int(x) if x.isdigit() else x)
        counts = [sum(task_map[t]) for t in tasks]
        trials = [len(task_map[t]) for t in tasks]
        expected_trials = max(trials) if trials else 0
        incomplete = {
            t: len(task_map[t])
            for t in tasks
            if len(task_map[t]) < expected_trials
        }
        if incomplete:
            print(
                f"  WARNING: incomplete trials for {incomplete} "
                f"(expected {expected_trials}); pass^k denominators differ"
            )
        model_summary = {
            "tasks": tasks,
            "successes": counts,
            "trials": trials,
            "incomplete_tasks": incomplete,
            "pass": {},
        }
        for k in args.k:
            value = mean_pass_k(counts, trials, k)
            lo, hi = bootstrap_ci(counts, trials, k, rng)
            model_summary["pass"][f"pass^{k}"] = {
                "value": round(100 * value, 2),
                "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
            }
        summary[model] = model_summary
        print(f"\n{model}")
        print(f"  successes: {sum(counts)}/{sum(trials)}")
        for key, value in model_summary["pass"].items():
            print(f"  {key}: {value['value']}%  CI {value['ci95']}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nWrote {csv_path} and {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
