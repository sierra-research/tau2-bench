from __future__ import annotations

import argparse
import json
import os

from agent import create_agent

from tau2.data_model.simulation import PostEvaluationMode, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.batch import run_domain_evaluated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="mock")
    parser.add_argument("--split", default="base")
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    model = os.getenv("TAU2_AGENT_MODEL", "gpt-4.1-2025-04-14")
    llm_args = {"temperature": float(os.getenv("TAU2_AGENT_TEMPERATURE", "0"))}
    api_mode = os.getenv("TAU2_AGENT_API_MODE")
    if api_mode:
        llm_args["api_mode"] = api_mode
    api_base = os.getenv("TAU2_AGENT_API_BASE")
    if api_base:
        llm_args["api_base"] = api_base
        llm_args["api_key"] = os.getenv("TAU2_AGENT_API_KEY") or "EMPTY"

    config = TextRunConfig(
        domain=args.domain,
        task_split_name=args.split,
        num_tasks=args.num_tasks,
        num_trials=1,
        max_concurrency=1,
        llm_agent=model,
        llm_args_agent=llm_args,
        post_evaluation_mode=PostEvaluationMode.EVALUATION_ONLY,
    )
    results = run_domain_evaluated(
        config,
        evaluation_type=EvaluationType.ALL,
        score_policy="evaluation_mean_v1",
        agent_factory_override=create_agent,
    )
    scores = [sim.evaluation_outcome.overall_score for sim in results.simulations]
    print(json.dumps({"scores": scores}, indent=2))


if __name__ == "__main__":
    main()

