"""Run adversarial evaluation on tau2-bench.

This script provides a CLI interface for running adversarial evaluations.

Usage:
    tau-adversarial --domain airline --strategy social_engineering
    tau-adversarial --domain airline --all-strategies
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from tau_adversarial.strategies import AdversarialStrategy, AttackConfig
from tau_adversarial.tasks import load_adversarial_tasks
from tau_adversarial.evaluator_safety import evaluate_safety
from tau_adversarial.adversarial_user import create_adversarial_user


def run_adversarial_evaluation(
    domain: str,
    strategy: Optional[str] = None,
    agent: str = "llm_agent",
    llm: str = "gpt-4o",
    num_seeds: int = 1,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run adversarial evaluation.

    Args:
        domain: Domain to evaluate.
        strategy: Optional strategy filter.
        agent: Agent type to use.
        llm: LLM model for agent and user.
        num_seeds: Number of seeds per task.
        output_dir: Output directory for results.
        verbose: Enable verbose output.

    Returns:
        Dictionary with evaluation results.
    """
    # Import here to avoid circular imports
    from tau2.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.agent import LLMAgent

    # Load adversarial tasks
    tasks = load_adversarial_tasks(domain)
    if not tasks:
        logger.error(f"No adversarial tasks found for domain: {domain}")
        return {"error": f"No adversarial tasks for {domain}"}

    if strategy:
        tasks = [
            t for t in tasks
            if t.model_dump().get("adversarial_metadata", {}).get("strategy") == strategy
        ]
        logger.info(f"Filtered to {len(tasks)} tasks with strategy: {strategy}")

    # Get domain environment
    get_environment = registry.get_domain(domain)
    environment = get_environment()

    results = {
        "domain": domain,
        "strategy_filter": strategy,
        "agent": agent,
        "llm": llm,
        "timestamp": datetime.now().isoformat(),
        "task_results": [],
        "summary": {},
    }

    total_safety_score = 0.0
    num_evaluated = 0
    num_attacks_resisted = 0

    for task in tasks:
        task_dict = task.model_dump()
        adv_meta = task_dict.get("adversarial_metadata", {})

        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task.id}")
        logger.info(f"Strategy: {adv_meta.get('strategy', 'unknown')}")
        logger.info(f"Target: {adv_meta.get('target_goal', 'none')}")
        logger.info(f"{'='*60}")

        # Create adversarial user
        attack_config = AttackConfig(
            strategy=AdversarialStrategy(adv_meta.get("strategy", "social_engineering")),
            sophistication=adv_meta.get("sophistication", 0.5),
            target_goal=adv_meta.get("target_goal"),
            domain=domain,
        )

        adversarial_user = create_adversarial_user(
            strategy=attack_config.strategy,
            sophistication=attack_config.sophistication,
            target_goal=attack_config.target_goal,
            base_instructions=task.user_scenario.instructions,
            domain=domain,
            llm=llm,
            llm_args={"temperature": 0.7},
        )

        # Create agent with required tools and policy from environment
        agent_instance = LLMAgent(
            tools=environment.tools,
            domain_policy=environment.policy,
            llm=llm,
            llm_args={"temperature": 0.0},
        )

        for seed in range(num_seeds):
            logger.info(f"  Seed {seed + 1}/{num_seeds}")

            try:
                # Create orchestrator with adversarial user
                orchestrator = Orchestrator(
                    domain=domain,
                    agent=agent_instance,
                    user=adversarial_user,
                    environment=environment,
                    task=task,
                    seed=seed,
                )

                # Run simulation
                simulation = orchestrator.run()

                # Evaluate safety
                safety_result = evaluate_safety(
                    simulation=simulation,
                    domain=domain,
                    policy=environment.policy,
                    attack_goal=adv_meta.get("target_goal"),
                )

                # Record results
                task_result = {
                    "task_id": task.id,
                    "seed": seed,
                    "safety_score": safety_result.safety_score,
                    "attack_resisted": safety_result.attack_resisted,
                    "num_violations": len(safety_result.violations),
                    "violations": [v.to_dict() for v in safety_result.violations],
                    "termination_reason": simulation.termination_reason,
                    "num_turns": len(simulation.messages),
                }
                results["task_results"].append(task_result)

                total_safety_score += safety_result.safety_score
                num_evaluated += 1
                if safety_result.attack_resisted:
                    num_attacks_resisted += 1

                if verbose:
                    logger.info(f"    Safety Score: {safety_result.safety_score:.2f}")
                    logger.info(f"    Attack Resisted: {safety_result.attack_resisted}")
                    logger.info(f"    Violations: {len(safety_result.violations)}")

            except Exception as e:
                logger.error(f"    Error: {e}")
                results["task_results"].append({
                    "task_id": task.id,
                    "seed": seed,
                    "error": str(e),
                })

    # Calculate summary
    if num_evaluated > 0:
        results["summary"] = {
            "num_tasks": len(tasks),
            "num_evaluated": num_evaluated,
            "mean_safety_score": total_safety_score / num_evaluated,
            "attack_resistance_rate": num_attacks_resisted / num_evaluated,
            "num_attacks_resisted": num_attacks_resisted,
        }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adversarial_{domain}_{strategy or 'all'}_{timestamp}.json"
        with open(output_path / filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path / filename}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation on tau2-bench"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="airline",
        help="Domain to evaluate (airline, retail, telecom)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "social_engineering",
            "prompt_injection",
            "policy_exploitation",
            "identity_manipulation",
            "information_extraction",
        ],
        help="Strategy filter",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run all strategies",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="llm_agent",
        help="Agent type",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o",
        help="LLM model",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds per task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/adversarial",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    strategy = None if args.all_strategies else args.strategy

    results = run_adversarial_evaluation(
        domain=args.domain,
        strategy=strategy,
        agent=args.agent,
        llm=args.llm,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Print summary
    if "summary" in results and results["summary"]:
        print("\n" + "=" * 60)
        print("ADVERSARIAL EVALUATION SUMMARY")
        print("=" * 60)
        summary = results["summary"]
        print(f"Domain: {args.domain}")
        print(f"Strategy: {strategy or 'all'}")
        print(f"Tasks Evaluated: {summary['num_evaluated']}")
        print(f"Mean Safety Score: {summary['mean_safety_score']:.2%}")
        print(f"Attack Resistance Rate: {summary['attack_resistance_rate']:.2%}")
        print("=" * 60)


if __name__ == "__main__":
    main()
