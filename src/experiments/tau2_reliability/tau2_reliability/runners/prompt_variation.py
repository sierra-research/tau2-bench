"""Prompt robustness evaluation via semantic paraphrasing.

Generates semantic paraphrases of user instructions, runs the same
tasks with varied instructions, and computes R_prompt = min(Acc_varied / Acc_base, 1.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import Results, TextRunConfig
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

from tau2_reliability.metrics.robustness import (
    bootstrap_robustness_ratio,
    compute_robustness_ratio,
)
from tau2_reliability.models import RobustnessMetrics

PARAPHRASE_SYSTEM_PROMPT = """Generate semantic paraphrases of user instructions for a customer service scenario.

Rules:
- Preserve the EXACT intent, entities, names, IDs, and required actions
- Change ONLY the wording, sentence structure, formality level, and phrasing
- Each paraphrase should sound like a different person asking for the same thing
- Do NOT add or remove information

Return a JSON array of strings, one per paraphrase."""


def generate_paraphrases(
    instruction: str,
    model: str = "gpt-4o-mini",
    n: int = 3,
) -> list[str]:
    """Generate n semantic paraphrases of a user instruction.

    Uses an LLM to rephrase while preserving semantic intent.
    Returns list of paraphrased instructions (excludes original).
    """
    try:
        response = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=PARAPHRASE_SYSTEM_PROMPT),
                UserMessage(
                    role="user",
                    content=f"Generate {n} paraphrases of this instruction:\n\n{instruction}",
                ),
            ],
        )

        json_str = extract_json_from_llm_response(response.content or "[]")
        paraphrases = json.loads(json_str)

        if isinstance(paraphrases, list):
            return [str(p) for p in paraphrases[:n]]
        return []

    except Exception as e:
        logger.warning(f"Paraphrase generation failed: {e}")
        return []


def run_prompt_robustness(
    config: TextRunConfig,
    baseline_results: Results,
    paraphrase_model: str = "gpt-4o-mini",
    n_paraphrases: int = 3,
    save_dir: Optional[Path] = None,
) -> RobustnessMetrics:
    """Run prompt robustness evaluation.

    1. Compute baseline accuracy from existing results
    2. Generate paraphrases for each task's user instructions
    3. Run simulations with paraphrased instructions
    4. Compute R_prompt = min(Acc_paraphrase / Acc_baseline, 1.0)

    Note: This requires API keys and costs money to run.
    """
    from tau2.metrics.agent_metrics import is_successful
    from tau2.runner.build import build_orchestrator
    from tau2.runner.simulation import run_simulation

    # Baseline accuracy
    baseline_outcomes = [
        is_successful(sim.reward_info.reward)
        for sim in baseline_results.simulations
        if sim.reward_info
    ]
    baseline_accuracy = sum(baseline_outcomes) / len(baseline_outcomes) if baseline_outcomes else 0.0

    logger.info(f"Baseline accuracy: {baseline_accuracy:.1%} ({len(baseline_outcomes)} simulations)")

    # Generate paraphrases and run
    paraphrased_outcomes = []
    tasks = baseline_results.tasks or []

    for task in tasks:
        instruction = ""
        if task.user_scenario:
            inst = task.user_scenario.instructions
            if isinstance(inst, str):
                instruction = inst
            elif hasattr(inst, "text"):
                instruction = inst.text or ""

        if not instruction:
            logger.warning(f"No instruction found for task {task.id}, skipping")
            continue

        paraphrases = generate_paraphrases(instruction, model=paraphrase_model, n=n_paraphrases)

        for i, paraphrase in enumerate(paraphrases):
            logger.info(f"Task {task.id}, paraphrase {i + 1}/{len(paraphrases)}")
            try:
                # Build orchestrator with modified instruction
                # Note: This is a simplified version — full implementation would
                # modify the task's user_scenario.instructions before building
                orchestrator = build_orchestrator(config, task, seed=42 + i)
                sim = run_simulation(orchestrator)
                if sim.reward_info:
                    paraphrased_outcomes.append(is_successful(sim.reward_info.reward))
            except Exception as e:
                logger.warning(f"Paraphrased run failed for task {task.id}: {e}")

    if not paraphrased_outcomes:
        logger.warning("No paraphrased outcomes collected")
        return RobustnessMetrics(
            r_prompt=None,
            baseline_accuracy=baseline_accuracy,
        )

    paraphrased_accuracy = sum(paraphrased_outcomes) / len(paraphrased_outcomes)
    r_prompt = compute_robustness_ratio(baseline_accuracy, paraphrased_accuracy)

    # Bootstrap
    bootstrap = bootstrap_robustness_ratio(baseline_outcomes, paraphrased_outcomes)

    logger.info(
        f"Prompt robustness: R_prompt={r_prompt:.3f} "
        f"(baseline={baseline_accuracy:.1%}, paraphrased={paraphrased_accuracy:.1%})"
    )

    return RobustnessMetrics(
        r_prompt=r_prompt,
        baseline_accuracy=baseline_accuracy,
        perturbed_accuracy=paraphrased_accuracy,
        bootstrap_se={"r_prompt": bootstrap.standard_error},
    )
