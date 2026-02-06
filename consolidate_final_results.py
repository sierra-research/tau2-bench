#!/usr/bin/env python3
"""Consolidate all results and generate final counts and visualizations."""

import json
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_tasks(results):
    """Extract all task results from a results list."""
    tasks = []
    for result in results:
        model = result.get("model", "unknown")
        for task in result.get("task_results", []):
            tasks.append({
                "model": model,
                "domain": result.get("domain"),
                "strategy": result.get("strategy"),
                "sophistication": result.get("sophistication"),
                **task
            })
    return tasks

def main():
    results_dir = Path("results/openrouter_eval")

    # Load consolidated results
    consolidated = load_json(results_dir / "consolidated_all_models.json")
    existing_tasks = extract_tasks(consolidated)

    # Load new missing tasks results
    missing = load_json(results_dir / "missing_tasks_20260206_110243.json")
    new_tasks = extract_tasks(missing)

    # Load final missing tasks results
    final_missing = load_json(results_dir / "final_missing_20260206_113358.json")
    new_tasks.extend(extract_tasks(final_missing))

    # Load last 2 missing tasks
    last_2 = load_json(results_dir / "last_2_missing_20260206_113810.json")
    new_tasks.extend(extract_tasks(last_2))

    # Create unique key for deduplication
    def task_key(t):
        return (t["model"], t["domain"], t["strategy"], t["sophistication"], t["task_id"], t.get("seed", 0))

    # Build lookup of existing tasks
    existing_keys = set(task_key(t) for t in existing_tasks)

    # Add only new tasks that don't already exist
    added = 0
    for task in new_tasks:
        key = task_key(task)
        if key not in existing_keys:
            existing_tasks.append(task)
            existing_keys.add(key)
            added += 1

    print(f"Added {added} new task results")

    # Count tasks per model
    counts = defaultdict(int)
    for task in existing_tasks:
        counts[task["model"]] += 1

    print("\n=== Task Counts by Model ===")
    for model, count in sorted(counts.items()):
        status = "✓" if count >= 72 else f"(missing {72 - count})"
        print(f"  {model}: {count} {status}")

    # Reorganize into results format
    results_by_config = defaultdict(lambda: {"task_results": []})
    for task in existing_tasks:
        key = (task["model"], task["domain"], task["strategy"], task["sophistication"])
        config = results_by_config[key]
        config["model"] = task["model"]
        config["domain"] = task["domain"]
        config["strategy"] = task["strategy"]
        config["sophistication"] = task["sophistication"]
        config["task_results"].append({
            "task_id": task["task_id"],
            "seed": task.get("seed", 0),
            "safety_score": task["safety_score"],
            "attack_resisted": task.get("attack_resisted", True),
            "num_violations": task.get("num_violations", 0),
        })

    final_results = list(results_by_config.values())

    # Save final consolidated results
    output_path = results_dir / "final_consolidated_all_models.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved final consolidated results to: {output_path}")

    return final_results, counts

if __name__ == "__main__":
    main()
