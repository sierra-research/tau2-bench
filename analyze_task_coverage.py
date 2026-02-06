#!/usr/bin/env python3
"""Analyze task coverage across all models."""

import json
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    results_dir = Path("results/openrouter_eval")
    results = load_json(results_dir / "final_consolidated_all_models.json")

    # Build lookup of tasks by model and config
    tasks_by_model = defaultdict(lambda: defaultdict(set))

    for r in results:
        model = r["model"]
        config_key = (r["domain"], r["strategy"], r["sophistication"])
        for task in r.get("task_results", []):
            tasks_by_model[model][config_key].add(task["task_id"])

    # Print task coverage for grok (reference model with 72 tasks)
    print("=== Task Coverage by Strategy (using grok-4.1-fast as reference) ===")
    grok_tasks = tasks_by_model["grok-4.1-fast"]

    strategy_tasks = defaultdict(set)
    for (domain, strategy, soph), task_ids in grok_tasks.items():
        for task_id in task_ids:
            strategy_tasks[strategy].add(task_id)

    for strategy, task_ids in sorted(strategy_tasks.items()):
        print(f"\n{strategy}: {sorted(task_ids)}")

    # Count per model
    print("\n\n=== Task Counts by Model ===")
    for model in sorted(tasks_by_model.keys()):
        total = sum(len(tasks) for tasks in tasks_by_model[model].values())
        print(f"{model}: {total} tasks")

        # Show breakdown by strategy
        by_strategy = defaultdict(int)
        for (domain, strategy, soph), task_ids in tasks_by_model[model].items():
            by_strategy[strategy] += len(task_ids)

        for strategy, count in sorted(by_strategy.items()):
            print(f"    {strategy}: {count}")

if __name__ == "__main__":
    main()
