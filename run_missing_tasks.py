#!/usr/bin/env python3
"""Re-run specific missing tasks to get all models to 72 evaluations."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Missing tasks to re-run (from analysis)
MISSING_TASKS = {
    "gpt-oss-120b": {
        "model_id": "openrouter/openai/gpt-oss-120b",
        "configs": [
            {"domain": "airline", "strategy": "social_engineering", "sophistication": 0.5, "missing": 1},
            {"domain": "airline", "strategy": "social_engineering", "sophistication": 1.0, "missing": 2},
            {"domain": "airline", "strategy": "prompt_injection", "sophistication": 0.5, "missing": 1},
            {"domain": "airline", "strategy": "prompt_injection", "sophistication": 1.0, "missing": 1},
            {"domain": "telecom", "strategy": "social_engineering", "sophistication": 0.0, "missing": 1},
            {"domain": "telecom", "strategy": "social_engineering", "sophistication": 0.5, "missing": 1},
            {"domain": "telecom", "strategy": "social_engineering", "sophistication": 1.0, "missing": 1},
        ]
    },
    "kimi-k2.5": {
        "model_id": "openrouter/moonshotai/kimi-k2.5",
        "configs": [
            {"domain": "airline", "strategy": "social_engineering", "sophistication": 0.0, "missing": 1},
            {"domain": "airline", "strategy": "social_engineering", "sophistication": 0.5, "missing": 1},
            {"domain": "airline", "strategy": "social_engineering", "sophistication": 1.0, "missing": 1},
        ]
    },
    "mimo-v2-flash": {
        "model_id": "openrouter/xiaomi/mimo-v2-flash",
        "configs": [
            {"domain": "telecom", "strategy": "prompt_injection", "sophistication": 0.5, "missing": 1},
        ]
    },
}

def run_missing():
    from run_openrouter_eval import run_single_evaluation
    
    results = []
    total_missing = sum(sum(c.get("missing", 1) for c in cfg["configs"]) for cfg in MISSING_TASKS.values())
    
    logger.info(f"Re-running {total_missing} missing tasks across {len(MISSING_TASKS)} models")
    
    for model_name, config in MISSING_TASKS.items():
        logger.info(f"\nModel: {model_name}")
        
        for cfg in config["configs"]:
            logger.info(f"  {cfg['domain']} | {cfg['strategy']} | soph={cfg['sophistication']} (missing {cfg.get('missing', 1)} tasks)")
            
            result = run_single_evaluation(
                model_name=model_name,
                model_id=config["model_id"],
                domain=cfg["domain"],
                strategy=cfg["strategy"],
                sophistication=cfg["sophistication"],
                num_seeds=1,
                verbose=True,
            )
            results.append(result)
            time.sleep(2)  # Rate limit protection
    
    # Save results
    if results:
        output_dir = Path("results/openrouter_eval")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"missing_tasks_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nSaved results to: {output_file}")
        
        # Count new results
        new_tasks = sum(len(r.get("task_results", [])) for r in results)
        logger.info(f"New task results: {new_tasks}")
    
    return results

if __name__ == "__main__":
    run_missing()
