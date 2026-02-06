#!/usr/bin/env python3
"""Run the final 2 stubborn missing tasks."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Last 2 missing tasks - retry 3 times each
MISSING = [
    {"domain": "airline", "strategy": "social_engineering", "sophistication": 0.5, "task_id": "adv_se_001"},
    {"domain": "telecom", "strategy": "social_engineering", "sophistication": 0.0, "task_id": "adv_se_002"},
]

def run_task():
    from run_openrouter_eval import run_single_evaluation

    results = []
    model_name = "gpt-oss-120b"
    model_id = "openrouter/openai/gpt-oss-120b"

    logger.info(f"Attempting last 2 missing tasks for {model_name} (with retries)")

    for task_info in MISSING:
        for attempt in range(3):
            logger.info(f"Attempt {attempt+1}/3: {task_info['domain']} | {task_info['strategy']} | soph={task_info['sophistication']} | {task_info['task_id']}")

            result = run_single_evaluation(
                model_name=model_name,
                model_id=model_id,
                domain=task_info["domain"],
                strategy=task_info["strategy"],
                sophistication=task_info["sophistication"],
                num_seeds=1,
                verbose=True,
            )

            # Check if we got results for the specific task
            task_succeeded = False
            for r in [result] if isinstance(result, dict) else result:
                for t in r.get("task_results", []):
                    if t.get("task_id") == task_info["task_id"]:
                        task_succeeded = True
                        break

            if task_succeeded:
                results.append(result)
                logger.info(f"Success on attempt {attempt+1}!")
                break
            else:
                logger.warning(f"Task {task_info['task_id']} failed on attempt {attempt+1}")
                time.sleep(5)

    # Save results
    if results:
        output_dir = Path("results/openrouter_eval")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"last_2_missing_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nSaved results to: {output_file}")

        new_tasks = sum(len(r.get("task_results", [])) for r in results)
        logger.info(f"New task results: {new_tasks}")

    return results

if __name__ == "__main__":
    run_task()
