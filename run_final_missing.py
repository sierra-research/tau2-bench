#!/usr/bin/env python3
"""Re-run final 4 missing tasks for GPT-OSS-120B."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Final 4 missing tasks
MISSING_TASKS = [
    {"domain": "airline", "strategy": "social_engineering", "sophistication": 0.5, "task_id": "adv_se_001"},
    {"domain": "telecom", "strategy": "social_engineering", "sophistication": 0.0, "task_id": "adv_se_002"},
    {"domain": "telecom", "strategy": "social_engineering", "sophistication": 0.5, "task_id": "adv_se_002"},
    {"domain": "telecom", "strategy": "social_engineering", "sophistication": 1.0, "task_id": "adv_se_002"},
]

def run_single_task():
    from run_openrouter_eval import run_single_evaluation

    results = []
    model_name = "gpt-oss-120b"
    model_id = "openrouter/openai/gpt-oss-120b"

    logger.info(f"Re-running {len(MISSING_TASKS)} missing tasks for {model_name}")

    for task_info in MISSING_TASKS:
        logger.info(f"Running: {task_info['domain']} | {task_info['strategy']} | soph={task_info['sophistication']} | {task_info['task_id']}")

        result = run_single_evaluation(
            model_name=model_name,
            model_id=model_id,
            domain=task_info["domain"],
            strategy=task_info["strategy"],
            sophistication=task_info["sophistication"],
            num_seeds=1,
            verbose=True,
        )
        results.append(result)
        time.sleep(2)

    # Save results
    if results:
        output_dir = Path("results/openrouter_eval")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"final_missing_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nSaved results to: {output_file}")

        new_tasks = sum(len(r.get("task_results", [])) for r in results)
        logger.info(f"New task results: {new_tasks}")

    return results

if __name__ == "__main__":
    run_single_task()
