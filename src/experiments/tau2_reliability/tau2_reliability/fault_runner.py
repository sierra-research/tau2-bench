"""Fault robustness evaluation.

Wraps tau2's environment to inject faults into tool responses,
then re-runs tasks to measure R_fault = Acc(faulted) / Acc(baseline).

No changes to tau2 source code — uses Python API (build_orchestrator + run_simulation)
and wraps the environment object before execution.
"""

from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger
from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import is_successful
from tau2.runner.build import build_orchestrator
from tau2.runner.simulation import run_simulation


class FaultType(Enum):
    TIMEOUT = "timeout"
    ERROR_500 = "error_500"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED = "malformed"


# Fault type distribution
FAULT_DISTRIBUTION = {
    FaultType.TIMEOUT: 0.30,
    FaultType.ERROR_500: 0.35,
    FaultType.EMPTY_RESPONSE: 0.15,
    FaultType.MALFORMED: 0.20,
}

FAULT_RESPONSES = {
    FaultType.TIMEOUT: "Error: Request timed out after 30 seconds. Please try again.",
    FaultType.ERROR_500: "Error: Internal server error (HTTP 500). The service is temporarily unavailable.",
    FaultType.EMPTY_RESPONSE: "",
    FaultType.MALFORMED: '{"error": true, "data": null, "message": "unexpected response format"}',
}


def _select_fault_type(rng: random.Random) -> FaultType:
    """Select a fault type based on the distribution using the seeded RNG."""
    r = rng.random()
    cumulative = 0.0
    for fault_type, prob in FAULT_DISTRIBUTION.items():
        cumulative += prob
        if r <= cumulative:
            return fault_type
    return FaultType.ERROR_500


def _wrap_environment_with_faults(env, fault_rate: float, seed: int):
    """Monkey-patch environment.get_response to inject faults.

    Wraps the environment to inject faults — no orchestrator changes.
    The environment's get_response randomly returns error responses.
    """
    original_get_response = env.get_response
    rng = random.Random(seed)
    fault_log = []

    def faulted_get_response(tool_call):
        roll = rng.random()
        logger.debug(f"Fault check: tool={tool_call.name}, roll={roll:.3f}, threshold={fault_rate}")
        if roll < fault_rate:
            fault_type = _select_fault_type(rng)
            fault_log.append({
                "tool": tool_call.name,
                "fault_type": fault_type.value,
                "turn": len(fault_log),
            })
            logger.info(f"  FAULT INJECTED: {fault_type.value} on {tool_call.name}")
            return ToolMessage(
                role="tool",
                id=tool_call.id,
                content=FAULT_RESPONSES[fault_type],
                error=True,
            )
        return original_get_response(tool_call)

    env.get_response = faulted_get_response
    return fault_log


def run_fault_evaluation(
    baseline_path: Path | str,
    fault_rate: float = 0.2,
    num_trials: int = 3,
    max_tasks: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
    seed: int = 42,
) -> dict:
    """Run fault injection evaluation on tasks from baseline results.

    Args:
        baseline_path: Path to baseline results.json (from tau2 run).
        fault_rate: Probability of injecting a fault per tool call (0-1).
        num_trials: Number of fault-injected trials per task.
        max_tasks: Limit number of tasks to evaluate.
        output_dir: Where to save fault evaluation results.
        seed: Random seed for reproducibility.

    Returns:
        Dict with baseline_accuracy, faulted_accuracy, r_fault, and per-task details.
    """
    baseline_path = Path(baseline_path)
    baseline = Results.load(baseline_path)

    # Extract config from baseline
    info = baseline.info
    config = TextRunConfig(
        domain=info.environment_info.domain_name,
        agent=info.agent_info.implementation,
        llm_agent=info.agent_info.llm or "gpt-4o",
        llm_args_agent=info.agent_info.llm_args or {"temperature": 0.0},
        user=info.user_info.implementation,
        llm_user=info.user_info.llm or "gpt-4o",
        llm_args_user=info.user_info.llm_args or {"temperature": 0.0},
        max_steps=info.max_steps,
        max_errors=info.max_errors,
    )

    # Get unique tasks from baseline
    tasks = baseline.tasks or []
    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(f"Fault evaluation: {len(tasks)} tasks × {num_trials} trials, fault_rate={fault_rate}")

    # Compute baseline accuracy
    baseline_outcomes = {}
    for sim in baseline.simulations:
        if sim.reward_info:
            baseline_outcomes.setdefault(sim.task_id, []).append(
                is_successful(sim.reward_info.reward)
            )

    baseline_accuracy = 0.0
    baseline_count = 0
    for outcomes in baseline_outcomes.values():
        baseline_accuracy += sum(outcomes)
        baseline_count += len(outcomes)
    baseline_accuracy = baseline_accuracy / baseline_count if baseline_count > 0 else 0.0

    # Run faulted trials
    faulted_results = []
    faulted_outcomes = []
    all_fault_logs = []

    for task_idx, task in enumerate(tasks):
        logger.info(f"Task {task.id} ({task_idx + 1}/{len(tasks)})")
        for trial in range(num_trials):
            trial_seed = seed + task_idx * 100 + trial
            try:
                # Build fresh orchestrator
                orchestrator = build_orchestrator(config, task, seed=trial_seed)

                # Wrap environment with fault injection
                fault_log = _wrap_environment_with_faults(
                    orchestrator.environment, fault_rate, trial_seed
                )

                # Run simulation with full evaluation (NL assertions use Azure via env var)
                sim = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)

                outcome = is_successful(sim.reward_info.reward) if sim.reward_info else False
                faulted_outcomes.append(outcome)
                faulted_results.append({
                    "task_id": task.id,
                    "trial": trial,
                    "reward": sim.reward_info.reward if sim.reward_info else 0.0,
                    "outcome": "pass" if outcome else "fail",
                    "faults_injected": len(fault_log),
                    "fault_log": fault_log,
                    "duration": sim.duration,
                    "cost": sim.agent_cost or 0.0,
                    "termination": sim.termination_reason.value if sim.termination_reason else "unknown",
                })
                all_fault_logs.extend(fault_log)

                logger.info(
                    f"  Trial {trial}: {'PASS' if outcome else 'FAIL'}, "
                    f"{len(fault_log)} faults injected"
                )

            except Exception as e:
                logger.error(f"  Trial {trial} failed: {e}")
                faulted_outcomes.append(False)
                faulted_results.append({
                    "task_id": task.id,
                    "trial": trial,
                    "outcome": "error",
                    "error": str(e),
                })

    # Compute metrics
    faulted_accuracy = sum(faulted_outcomes) / len(faulted_outcomes) if faulted_outcomes else 0.0
    r_fault = min(faulted_accuracy / baseline_accuracy, 1.0) if baseline_accuracy > 0 else 1.0

    total_faults = len(all_fault_logs)
    fault_type_counts = {}
    for fl in all_fault_logs:
        ft = fl["fault_type"]
        fault_type_counts[ft] = fault_type_counts.get(ft, 0) + 1

    result = {
        "baseline_accuracy": baseline_accuracy,
        "faulted_accuracy": faulted_accuracy,
        "r_fault": r_fault,
        "fault_rate": fault_rate,
        "num_tasks": len(tasks),
        "num_trials": num_trials,
        "total_faults_injected": total_faults,
        "fault_type_distribution": fault_type_counts,
        "faulted_conversations": faulted_results,
    }

    logger.info("\nFault Robustness Results:")
    logger.info(f"  Baseline accuracy: {baseline_accuracy:.1%}")
    logger.info(f"  Faulted accuracy:  {faulted_accuracy:.1%}")
    logger.info(f"  R_fault:           {r_fault:.3f}")
    logger.info(f"  Total faults:      {total_faults}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "fault_results.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info(f"  Saved: {out_path}")

    return result
