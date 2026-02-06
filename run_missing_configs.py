#!/usr/bin/env python3
"""Re-run missing configurations to equalize evaluation counts."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Missing configs to re-run
MISSING_CONFIGS = {
    "gpt-oss-120b": {
        "model_id": "openrouter/openai/gpt-oss-120b",
        "configs": [
            {"domain": "airline", "strategy": "policy_exploitation", "sophistication": 0.5},
            {"domain": "airline", "strategy": "information_extraction", "sophistication": 0.5},
        ]
    },
    "kimi-k2.5": {
        "model_id": "openrouter/moonshotai/kimi-k2.5",
        "configs": [
            {"domain": "airline", "strategy": "policy_exploitation", "sophistication": 0.0},
        ]
    },
    "grok-4.1-fast": {
        "model_id": "openrouter/x-ai/grok-4.1-fast",
        "configs": []  # Complete
    },
    "deepseek-v3.2": {
        "model_id": "openrouter/deepseek/deepseek-v3.2",
        "configs": []  # Complete
    },
    "mimo-v2-flash": {
        "model_id": "openrouter/xiaomi/mimo-v2-flash",
        "configs": []  # Nearly complete (71/72)
    },
}

def run_missing():
    from run_openrouter_eval import run_single_evaluation
    
    results = []
    
    for model_name, config in MISSING_CONFIGS.items():
        if not config["configs"]:
            continue
            
        logger.info(f"Re-running missing configs for {model_name}")
        
        for cfg in config["configs"]:
            logger.info(f"  {cfg['domain']} | {cfg['strategy']} | soph={cfg['sophistication']}")
            
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
    
    # Save results
    if results:
        output_dir = Path("results/openrouter_eval")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"missing_configs_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved missing config results to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_missing()
