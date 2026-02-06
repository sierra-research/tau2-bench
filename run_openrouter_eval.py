#!/usr/bin/env python3
"""
τ²-Adv Bench: Open-Source Model Evaluation Script

This script runs a comprehensive adversarial evaluation across:
- 3 open-source models (via OpenRouter)
- 3 domains (airline, retail, telecom)
- 5 attack strategies
- 3 sophistication levels (0.0, 0.5, 1.0)

Usage:
    # Set your OpenRouter API key
    export OPENROUTER_API_KEY=your_key_here

    # Run full evaluation
    python run_openrouter_eval.py

    # Run quick test (1 task per config)
    python run_openrouter_eval.py --quick

    # Run specific model only
    python run_openrouter_eval.py --model llama-3.1-70b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Open-source models available on OpenRouter
# See: https://openrouter.ai/models
OPEN_SOURCE_MODELS = {
    # Xiaomi MiMo V2 Flash - needs to be re-run (had errors)
    "mimo-v2-flash": "openrouter/xiaomi/mimo-v2-flash",

    # Moonshot Kimi K2.5 - Moonshot AI's latest model
    "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",

    # DeepSeek V3.2 - Latest DeepSeek model
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
}

# Domains to evaluate
DOMAINS = ["airline", "retail", "telecom"]

# Attack strategies
ATTACK_STRATEGIES = [
    "social_engineering",
    "prompt_injection",
    "policy_exploitation",
    "identity_manipulation",
    "information_extraction",
]

# Sophistication levels
SOPHISTICATION_LEVELS = [0.0, 0.5, 1.0]

# Number of seeds per configuration
DEFAULT_NUM_SEEDS = 1


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def check_openrouter_key():
    """Check if OpenRouter API key is set."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.error("OPENROUTER_API_KEY environment variable not set!")
        logger.info("Set it with: export OPENROUTER_API_KEY=your_key_here")
        sys.exit(1)
    return key


def parse_adversarial_metadata(task) -> dict:
    """Parse adversarial metadata from task description.

    The metadata is embedded in the description purpose field like:
    [ADVERSARIAL: strategy=social_engineering, sophistication=0.5, target=Get refund]
    """
    task_dict = task.model_dump() if hasattr(task, "model_dump") else {}

    # Check for explicit adversarial_metadata field first
    if "adversarial_metadata" in task_dict and task_dict["adversarial_metadata"]:
        return task_dict["adversarial_metadata"]

    # Parse from description
    description = task_dict.get("description", {})
    purpose = description.get("purpose", "") if isinstance(description, dict) else str(description)

    metadata = {}
    if "[ADVERSARIAL:" in purpose:
        # Extract metadata string
        start = purpose.find("[ADVERSARIAL:") + len("[ADVERSARIAL:")
        end = purpose.find("]", start)
        if end > start:
            meta_str = purpose[start:end].strip()
            # Parse key=value pairs
            for pair in meta_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Convert sophistication to float
                    if key == "sophistication":
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    metadata[key] = value

    return metadata


def run_single_evaluation(
    model_name: str,
    model_id: str,
    domain: str,
    strategy: str,
    sophistication: float,
    num_seeds: int = 1,
    verbose: bool = False,
) -> dict:
    """Run evaluation for a single configuration."""
    from tau2.adversarial.strategies import AdversarialStrategy, AttackConfig
    from tau2.adversarial.tasks import load_adversarial_tasks
    from tau2.evaluator.evaluator_safety import evaluate_safety
    from tau2.user.adversarial_user import create_adversarial_user
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.agent.llm_agent import LLMAgent

    result = {
        "model": model_name,
        "model_id": model_id,
        "domain": domain,
        "strategy": strategy,
        "sophistication": sophistication,
        "task_results": [],
        "errors": [],
    }

    try:
        # Load tasks for domain
        tasks = load_adversarial_tasks(domain)

        # Filter by strategy using parsed metadata
        filtered_tasks = []
        for t in tasks:
            meta = parse_adversarial_metadata(t)
            if meta.get("strategy") == strategy:
                filtered_tasks.append((t, meta))
        tasks = filtered_tasks

        if not tasks:
            logger.warning(f"No tasks found for {domain}/{strategy}")
            return result

        # Get domain environment
        get_environment = registry.get_env_constructor(domain)
        environment = get_environment()

        for task, adv_meta in tasks:

            # Get tools and policy from environment
            tools = environment.get_tools()
            policy = environment.get_policy()

            # Create adversarial user with specified sophistication
            adversarial_user = create_adversarial_user(
                strategy=AdversarialStrategy(strategy),
                sophistication=sophistication,
                target_goal=adv_meta.get("target_goal"),
                base_instructions=task.user_scenario.instructions,
                domain=domain,
                llm=model_id,
                llm_args={"temperature": 0.7},
            )

            # Create agent with tools and policy
            agent_instance = LLMAgent(
                tools=tools,
                domain_policy=policy,
                llm=model_id,
                llm_args={"temperature": 0.0},
            )

            for seed in range(num_seeds):
                try:
                    # Run simulation
                    orchestrator = Orchestrator(
                        domain=domain,
                        agent=agent_instance,
                        user=adversarial_user,
                        environment=environment,
                        task=task,
                        seed=seed,
                    )

                    simulation = orchestrator.run()

                    # Evaluate safety
                    safety_result = evaluate_safety(
                        simulation=simulation,
                        domain=domain,
                        policy=environment.policy,
                        attack_goal=adv_meta.get("target_goal"),
                    )

                    task_result = {
                        "task_id": task.id,
                        "seed": seed,
                        "safety_score": safety_result.safety_score,
                        "attack_resisted": safety_result.attack_resisted,
                        "num_violations": len(safety_result.violations),
                        "termination_reason": str(simulation.termination_reason),
                        "num_turns": len(simulation.messages),
                    }
                    result["task_results"].append(task_result)

                    if verbose:
                        status = "✓ RESISTED" if safety_result.attack_resisted else "✗ FAILED"
                        logger.info(f"    {task.id} (seed {seed}): {status} | Safety: {safety_result.safety_score:.2f}")

                except Exception as e:
                    logger.error(f"    Error on {task.id}: {e}")
                    result["errors"].append({"task_id": task.id, "seed": seed, "error": str(e)})

                # Add delay between tasks to avoid rate limiting
                time.sleep(1)

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        result["errors"].append({"error": str(e)})

    return result


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from results."""
    metrics = {
        "by_model": {},
        "by_domain": {},
        "by_strategy": {},
        "by_sophistication": {},
        "by_model_domain": {},
        "by_model_strategy": {},
        "by_strategy_sophistication": {},
        "overall": {},
    }

    # Flatten all task results
    all_results = []
    for r in results:
        for tr in r.get("task_results", []):
            all_results.append({
                **tr,
                "model": r["model"],
                "domain": r["domain"],
                "strategy": r["strategy"],
                "sophistication": r["sophistication"],
            })

    if not all_results:
        return metrics

    # Helper to compute stats
    def compute_stats(filtered_results):
        if not filtered_results:
            return {"count": 0}

        safety_scores = [r["safety_score"] for r in filtered_results]
        attack_resisted = [r["attack_resisted"] for r in filtered_results]

        return {
            "count": len(filtered_results),
            "mean_safety_score": sum(safety_scores) / len(safety_scores),
            "defense_rate": sum(attack_resisted) / len(attack_resisted),
            "attack_success_rate": 1 - sum(attack_resisted) / len(attack_resisted),
        }

    # By model
    for model in OPEN_SOURCE_MODELS.keys():
        filtered = [r for r in all_results if r["model"] == model]
        metrics["by_model"][model] = compute_stats(filtered)

    # By domain
    for domain in DOMAINS:
        filtered = [r for r in all_results if r["domain"] == domain]
        metrics["by_domain"][domain] = compute_stats(filtered)

    # By strategy
    for strategy in ATTACK_STRATEGIES:
        filtered = [r for r in all_results if r["strategy"] == strategy]
        metrics["by_strategy"][strategy] = compute_stats(filtered)

    # By sophistication
    for soph in SOPHISTICATION_LEVELS:
        filtered = [r for r in all_results if r["sophistication"] == soph]
        metrics["by_sophistication"][str(soph)] = compute_stats(filtered)

    # By model x domain
    for model in OPEN_SOURCE_MODELS.keys():
        metrics["by_model_domain"][model] = {}
        for domain in DOMAINS:
            filtered = [r for r in all_results if r["model"] == model and r["domain"] == domain]
            metrics["by_model_domain"][model][domain] = compute_stats(filtered)

    # By model x strategy
    for model in OPEN_SOURCE_MODELS.keys():
        metrics["by_model_strategy"][model] = {}
        for strategy in ATTACK_STRATEGIES:
            filtered = [r for r in all_results if r["model"] == model and r["strategy"] == strategy]
            metrics["by_model_strategy"][model][strategy] = compute_stats(filtered)

    # By strategy x sophistication
    for strategy in ATTACK_STRATEGIES:
        metrics["by_strategy_sophistication"][strategy] = {}
        for soph in SOPHISTICATION_LEVELS:
            filtered = [r for r in all_results if r["strategy"] == strategy and r["sophistication"] == soph]
            metrics["by_strategy_sophistication"][strategy][str(soph)] = compute_stats(filtered)

    # Overall
    metrics["overall"] = compute_stats(all_results)

    return metrics


def print_summary(metrics: dict):
    """Print a formatted summary of results."""
    print("\n" + "=" * 80)
    print(" τ²-ADV BENCH: OPEN-SOURCE MODEL EVALUATION RESULTS")
    print("=" * 80)

    # Overall
    overall = metrics.get("overall", {})
    if overall.get("count", 0) > 0:
        print(f"\n📊 OVERALL ({overall['count']} evaluations)")
        print(f"   Mean Safety Score:    {overall['mean_safety_score']:.2%}")
        print(f"   Defense Rate:         {overall['defense_rate']:.2%}")
        print(f"   Attack Success Rate:  {overall['attack_success_rate']:.2%}")

    # By Model
    print("\n" + "-" * 80)
    print("📱 BY MODEL")
    print("-" * 80)
    print(f"{'Model':<20} {'Safety Score':>15} {'Defense Rate':>15} {'ASR':>15}")
    print("-" * 80)
    for model, stats in metrics.get("by_model", {}).items():
        if stats.get("count", 0) > 0:
            print(f"{model:<20} {stats['mean_safety_score']:>14.2%} {stats['defense_rate']:>14.2%} {stats['attack_success_rate']:>14.2%}")

    # By Domain
    print("\n" + "-" * 80)
    print("🏢 BY DOMAIN")
    print("-" * 80)
    print(f"{'Domain':<20} {'Safety Score':>15} {'Defense Rate':>15} {'ASR':>15}")
    print("-" * 80)
    for domain, stats in metrics.get("by_domain", {}).items():
        if stats.get("count", 0) > 0:
            print(f"{domain:<20} {stats['mean_safety_score']:>14.2%} {stats['defense_rate']:>14.2%} {stats['attack_success_rate']:>14.2%}")

    # By Strategy
    print("\n" + "-" * 80)
    print("⚔️  BY ATTACK STRATEGY")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Safety Score':>15} {'Defense Rate':>15} {'ASR':>15}")
    print("-" * 80)
    for strategy, stats in metrics.get("by_strategy", {}).items():
        if stats.get("count", 0) > 0:
            print(f"{strategy:<25} {stats['mean_safety_score']:>14.2%} {stats['defense_rate']:>14.2%} {stats['attack_success_rate']:>14.2%}")

    # By Sophistication
    print("\n" + "-" * 80)
    print("📈 BY SOPHISTICATION LEVEL")
    print("-" * 80)
    print(f"{'Level':<20} {'Safety Score':>15} {'Defense Rate':>15} {'ASR':>15}")
    print("-" * 80)
    for soph, stats in metrics.get("by_sophistication", {}).items():
        if stats.get("count", 0) > 0:
            level_name = {"0.0": "Low", "0.5": "Medium", "1.0": "High"}.get(soph, soph)
            print(f"{level_name} ({soph}){'':<10} {stats['mean_safety_score']:>14.2%} {stats['defense_rate']:>14.2%} {stats['attack_success_rate']:>14.2%}")

    # Model x Strategy Matrix
    print("\n" + "-" * 80)
    print("📊 MODEL × STRATEGY ATTACK SUCCESS RATES")
    print("-" * 80)
    header = f"{'Model':<20}"
    for strategy in ATTACK_STRATEGIES:
        short = strategy[:10]
        header += f"{short:>12}"
    print(header)
    print("-" * 80)

    for model in OPEN_SOURCE_MODELS.keys():
        row = f"{model:<20}"
        model_stats = metrics.get("by_model_strategy", {}).get(model, {})
        for strategy in ATTACK_STRATEGIES:
            stats = model_stats.get(strategy, {})
            if stats.get("count", 0) > 0:
                row += f"{stats['attack_success_rate']:>11.1%}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    print("\n" + "=" * 80)


def run_full_evaluation(
    models: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    strategies: Optional[list[str]] = None,
    sophistication_levels: Optional[list[float]] = None,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    output_dir: str = "results/openrouter_eval",
    verbose: bool = False,
    quick: bool = False,
) -> dict:
    """Run full evaluation across all configurations."""

    models = models or list(OPEN_SOURCE_MODELS.keys())
    domains = domains or DOMAINS
    strategies = strategies or ATTACK_STRATEGIES
    sophistication_levels = sophistication_levels or SOPHISTICATION_LEVELS

    # Calculate total configurations
    total_configs = len(models) * len(domains) * len(strategies) * len(sophistication_levels)

    logger.info(f"Starting τ²-Adv Bench evaluation")
    logger.info(f"  Models: {models}")
    logger.info(f"  Domains: {domains}")
    logger.info(f"  Strategies: {strategies}")
    logger.info(f"  Sophistication levels: {sophistication_levels}")
    logger.info(f"  Total configurations: {total_configs}")
    logger.info(f"  Seeds per config: {num_seeds}")

    all_results = []
    config_count = 0
    start_time = time.time()

    for model_name in models:
        model_id = OPEN_SOURCE_MODELS[model_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*60}")

        for domain in domains:
            for strategy in strategies:
                for sophistication in sophistication_levels:
                    config_count += 1
                    logger.info(f"\n[{config_count}/{total_configs}] {model_name} | {domain} | {strategy} | soph={sophistication}")

                    result = run_single_evaluation(
                        model_name=model_name,
                        model_id=model_id,
                        domain=domain,
                        strategy=strategy,
                        sophistication=sophistication,
                        num_seeds=num_seeds,
                        verbose=verbose,
                    )
                    all_results.append(result)

                    # Add delay between configurations to avoid rate limiting
                    time.sleep(1)

                    # Quick mode: break after first successful config
                    if quick and result.get("task_results"):
                        logger.info("Quick mode: stopping after first successful evaluation")
                        break
                if quick:
                    break
            if quick:
                break
        if quick:
            break

    elapsed = time.time() - start_time
    logger.info(f"\nEvaluation completed in {elapsed:.1f}s")

    # Compute metrics
    metrics = compute_metrics(all_results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    results_file = output_path / f"raw_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Raw results saved to: {results_file}")

    # Save metrics
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to: {metrics_file}")

    # Print summary
    print_summary(metrics)

    return {
        "results": all_results,
        "metrics": metrics,
        "elapsed_time": elapsed,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="τ²-Adv Bench: Open-Source Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (all models, domains, strategies, sophistication levels)
  python run_openrouter_eval.py

  # Quick test
  python run_openrouter_eval.py --quick

  # Specific model
  python run_openrouter_eval.py --model llama-3.1-70b

  # Specific domain
  python run_openrouter_eval.py --domain airline

  # Specific strategy
  python run_openrouter_eval.py --strategy social_engineering
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(OPEN_SOURCE_MODELS.keys()),
        help="Specific model to evaluate (default: all)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=DOMAINS,
        help="Specific domain to evaluate (default: all)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=ATTACK_STRATEGIES,
        help="Specific strategy to evaluate (default: all)",
    )
    parser.add_argument(
        "--sophistication",
        type=float,
        choices=SOPHISTICATION_LEVELS,
        help="Specific sophistication level (default: all)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of seeds per configuration (default: {DEFAULT_NUM_SEEDS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/openrouter_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (run only one configuration)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable open-source models:")
        print("-" * 60)
        for name, model_id in OPEN_SOURCE_MODELS.items():
            print(f"  {name:<20} -> {model_id}")
        print()
        return

    # Check API key
    check_openrouter_key()

    # Build config
    models = [args.model] if args.model else None
    domains = [args.domain] if args.domain else None
    strategies = [args.strategy] if args.strategy else None
    sophistication_levels = [args.sophistication] if args.sophistication else None

    # Run evaluation
    run_full_evaluation(
        models=models,
        domains=domains,
        strategies=strategies,
        sophistication_levels=sophistication_levels,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        verbose=args.verbose,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
