#!/usr/bin/env python3
"""Find missing social engineering tasks for GPT-OSS-120B."""

import json
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Correct expected tasks
EXPECTED_TASKS = {
    "social_engineering": ["adv_se_001", "adv_se_002", "adv_combo_001"],  # 3 tasks
    "prompt_injection": ["adv_pi_001", "adv_pi_002"],  # 2 tasks
    "policy_exploitation": ["adv_pe_001"],  # 1 task
    "identity_manipulation": ["adv_id_001"],  # 1 task
    "information_extraction": ["adv_ie_001"],  # 1 task
}

DOMAINS = ["airline", "retail", "telecom"]
SOPHISTICATIONS = [0.0, 0.5, 1.0]

def main():
    results_dir = Path("results/openrouter_eval")
    results = load_json(results_dir / "final_consolidated_all_models.json")

    # Build lookup of what exists for gpt-oss-120b
    existing = set()
    for r in results:
        if r["model"] == "gpt-oss-120b":
            for task in r.get("task_results", []):
                key = (r["domain"], r["strategy"], r["sophistication"], task["task_id"])
                existing.add(key)

    # Find missing
    missing = []
    for domain in DOMAINS:
        for strategy, tasks in EXPECTED_TASKS.items():
            for soph in SOPHISTICATIONS:
                for task_id in tasks:
                    key = (domain, strategy, soph, task_id)
                    if key not in existing:
                        missing.append(key)

    print(f"GPT-OSS-120B is missing {len(missing)} tasks:")
    for domain, strategy, soph, task_id in sorted(missing):
        print(f"  {domain} | {strategy} | soph={soph} | {task_id}")

if __name__ == "__main__":
    main()
