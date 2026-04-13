"""
Phase 6: Experiment Runner.

CLI entry point for tau2-TRACE. Supports:
  1. Post-hoc analysis of existing simulation Results JSON files.
  2. Live simulation with optional adversarial user-simulator wrapper.

Usage:
    # Analyse existing results (deterministic, zero-cost)
    python -m experiments.tau2_trace.run_experiment analyze \
        --results-file data/tau2/simulations/my_run.json \
        --output src/experiments/tau2_trace/results/augmented.csv

    # Analyse with LLM judge for interaction quality
    python -m experiments.tau2_trace.run_experiment analyze \
        --results-file data/tau2/simulations/my_run.json \
        --llm-judge --llm-judge-model gpt-4.1

    # Run a live simulation with adversarial user perturbations
    python -m experiments.tau2_trace.run_experiment run \
        --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 \
        --adversarial --perturbation-rate 0.2 --num-tasks 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from experiments.tau2_trace.domain_router import evaluate_results_trace
from tau2.data_model.simulation import Results


def analyze_results(
    results_path: Path,
    output_path: Path,
    domain_override: str | None = None,
    recovery_window: int = 3,
    use_llm_judge: bool = False,
    llm_judge_model: str | None = None,
) -> pd.DataFrame:
    """
    Post-hoc analysis mode: load an existing Results file, run tau2-TRACE
    analysis, and produce an augmented DataFrame.

    Args:
        results_path: Path to a tau2-bench simulation Results JSON file.
        output_path: Path to write the augmented CSV.
        domain_override: Override the domain from the Results file.
        recovery_window: Error recovery lookahead window size.
        use_llm_judge: Enable LLM-based interaction quality evaluation.
        llm_judge_model: LLM model for judge calls.

    Returns:
        The augmented pandas DataFrame.
    """
    logger.info(f"Loading results from {results_path}")
    results = Results.load(results_path)

    domain = domain_override or results.info.environment_info.domain_name
    logger.info(f"Domain: {domain}")
    logger.info(f"Simulations: {len(results.simulations)}")
    logger.info(f"Tasks: {len(results.tasks)}")

    # Build task_id -> expected_actions mapping from task metadata
    task_expected_actions: dict[str, int] = {}
    for task in results.tasks:
        if task.evaluation_criteria is not None:
            info = task.evaluation_criteria.info()
            num_actions = info.get("num_agent_actions", 0) + info.get(
                "num_user_actions", 0
            )
            task_expected_actions[task.id] = num_actions

    # Run tau2-TRACE analysis
    logger.info("Running tau2-TRACE analysis...")
    scorecards = evaluate_results_trace(
        simulations=results.simulations,
        domain=domain,
        task_expected_actions=task_expected_actions,
        recovery_window=recovery_window,
        use_llm_judge=use_llm_judge,
        llm_judge_model=llm_judge_model,
    )

    # Build the core DataFrame from Results.
    # to_df() may fail if tasks lack evaluation_criteria, so fall back to
    # building a minimal DataFrame from simulation fields.
    try:
        df_core = results.to_df()
    except (KeyError, AttributeError):
        logger.warning(
            "Results.to_df() failed (likely missing evaluation_criteria). "
            "Building minimal DataFrame from simulation data."
        )
        rows = []
        for sim in results.simulations:
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "seed": sim.seed,
                    "reward": sim.reward_info.reward if sim.reward_info else None,
                    "agent_cost": sim.agent_cost,
                    "user_cost": sim.user_cost,
                    "termination_reason": sim.termination_reason.value,
                    "duration": sim.duration,
                    "num_messages": len(sim.messages),
                    "info_domain": domain,
                }
            )
        df_core = pd.DataFrame(rows)

    # Build the trace DataFrame from scorecards
    trace_rows = [sc.to_dict() for sc in scorecards]
    df_trace = pd.DataFrame(trace_rows)

    # Merge on (task_id, trial)
    merge_cols = ["task_id"]
    if "trial" in df_core.columns and "trial" in df_trace.columns:
        merge_cols.append("trial")

    df_augmented = pd.merge(df_core, df_trace, on=merge_cols, how="left")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(output_path, index=False)
    logger.info(f"Augmented results saved to {output_path}")

    # Print summary
    _print_summary(df_augmented, domain)

    return df_augmented


def run_live(
    domain: str,
    agent_llm: str,
    user_llm: str,
    output_path: Path,
    num_tasks: int = 1,
    num_trials: int = 1,
    task_split: Optional[str] = None,
    max_steps: int = 100,
    adversarial: bool = False,
    perturbation_rate: float = 0.20,
    adversarial_seed: int = 42,
    recovery_window: int = 3,
    use_llm_judge: bool = False,
    llm_judge_model: Optional[str] = None,
    seed: Optional[int] = 300,
) -> pd.DataFrame:
    """
    Live simulation mode: run tau2-bench simulations with optional
    adversarial user-simulator wrapper, then apply tau2-TRACE analysis.

    When --adversarial is set, the UserSimulator is wrapped via the Proxy
    Pattern so the Orchestrator sees the same interface but the user
    occasionally injects interruptions or self-corrections.
    """
    from experiments.tau2_trace.adversarial_wrapper import AdversarialSimulatorWrapper
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.run import get_info, get_tasks
    from tau2.user.user_simulator import UserSimulator

    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split,
        num_tasks=num_tasks,
    )
    logger.info(f"Loaded {len(tasks)} tasks for domain '{domain}'")

    simulations: list[SimulationRun] = []
    task_expected_actions: dict[str, int] = {}

    for task in tasks:
        if task.evaluation_criteria is not None:
            info = task.evaluation_criteria.info()
            num_actions = info.get("num_agent_actions", 0) + info.get(
                "num_user_actions", 0
            )
            task_expected_actions[task.id] = num_actions

        for trial in range(num_trials):
            trial_seed = (seed or 300) + trial
            logger.info(
                f"Running task={task.id} trial={trial} adversarial={adversarial}"
            )

            env_constructor = registry.get_env_constructor(domain)
            environment = env_constructor()

            agent_obj = LLMAgent(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                llm=agent_llm,
            )

            try:
                user_tools = environment.get_user_tools()
            except Exception:
                user_tools = None

            user_obj = UserSimulator(
                tools=user_tools,
                instructions=str(task.user_scenario),
                llm=user_llm,
            )

            if adversarial:
                user_obj = AdversarialSimulatorWrapper(
                    base_simulator=user_obj,
                    perturbation_rate=perturbation_rate,
                    seed=adversarial_seed + trial,
                )

            orchestrator = Orchestrator(
                domain=domain,
                agent=agent_obj,
                user=user_obj,
                environment=environment,
                task=task,
                max_steps=max_steps,
                seed=trial_seed,
            )
            simulation = orchestrator.run()

            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
            )
            simulation.reward_info = reward_info

            if adversarial and isinstance(user_obj, AdversarialSimulatorWrapper):
                n_perturbed = len(user_obj.perturbation_log)
                logger.info(f"  Adversarial perturbations injected: {n_perturbed}")

            simulations.append(simulation)

    # Build Results object
    info = get_info(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=agent_llm,
        llm_user=user_llm,
        num_trials=num_trials,
        max_steps=max_steps,
        seed=seed,
    )
    results = Results(
        info=info,
        tasks=tasks,
        simulations=simulations,
    )

    # Save raw results
    raw_path = output_path.with_suffix(".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(raw_path)
    logger.info(f"Raw results saved to {raw_path}")

    # Run tau2-TRACE analysis
    logger.info("Running tau2-TRACE analysis...")
    scorecards = evaluate_results_trace(
        simulations=simulations,
        domain=domain,
        task_expected_actions=task_expected_actions,
        recovery_window=recovery_window,
        use_llm_judge=use_llm_judge,
        llm_judge_model=llm_judge_model,
    )

    try:
        df_core = results.to_df()
    except (KeyError, AttributeError):
        rows = []
        for sim in simulations:
            rows.append(
                {
                    "simulation_id": sim.id,
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "seed": sim.seed,
                    "reward": sim.reward_info.reward if sim.reward_info else None,
                    "agent_cost": sim.agent_cost,
                    "num_messages": len(sim.messages),
                    "info_domain": domain,
                }
            )
        df_core = pd.DataFrame(rows)

    trace_rows = [sc.to_dict() for sc in scorecards]
    df_trace = pd.DataFrame(trace_rows)

    merge_cols = ["task_id"]
    if "trial" in df_core.columns and "trial" in df_trace.columns:
        merge_cols.append("trial")

    df_augmented = pd.merge(df_core, df_trace, on=merge_cols, how="left")

    csv_path = output_path.with_suffix(".csv")
    df_augmented.to_csv(csv_path, index=False)
    logger.info(f"Augmented results saved to {csv_path}")

    _print_summary(df_augmented, domain)

    return df_augmented


def _print_summary(df: pd.DataFrame, domain: str) -> None:
    """Print a concise summary of tau2-TRACE metrics."""
    print("\n" + "=" * 70)
    print(f"  tau2-TRACE Analysis Summary  |  Domain: {domain}")
    print("=" * 70)

    trace_cols = [c for c in df.columns if c.startswith("trace_")]
    if not trace_cols:
        print("  No trace metrics computed.")
        return

    if "reward" in df.columns:
        avg_reward = df["reward"].mean()
        pass_rate = (df["reward"] >= 1.0 - 1e-6).mean() * 100
        print(f"\n  Core Metrics:")
        print(f"    Average Reward:        {avg_reward:.4f}")
        print(f"    Pass Rate:             {pass_rate:.1f}%")

    print(f"\n  Trajectory Efficiency:")
    for col in [
        "trace_total_turns",
        "trace_total_tool_calls",
        "trace_redundant_tool_calls",
        "trace_loop_count",
        "trace_error_count",
        "trace_error_recovery_rate",
        "trace_error_burst_count",
        "trace_error_bursts_recovered",
        "trace_orphan_tool_messages",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            print(f"    {label:30s} {df[col].mean():>8.2f} (avg)")

    print(f"\n  Policy Adherence:")
    for col in [
        "trace_policy_adherence",
        "trace_read_after_write_score",
        "trace_policy_breach_count",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            print(f"    {label:30s} {df[col].mean():>8.2f} (avg)")

    print(f"\n  Interaction Quality:")
    for col in [
        "trace_action_density",
        "trace_token_to_action_ratio",
        "trace_turns_vs_expected",
        "trace_repeated_info_requests",
        "trace_guidance_precision",
    ]:
        if col in df.columns:
            label = col.replace("trace_", "").replace("_", " ").title()
            val = df[col].mean()
            if val != 0.0 or domain == "telecom" or "guidance" not in col:
                print(f"    {label:30s} {val:>8.2f} (avg)")

    print("\n" + "=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tau2-trace",
        description="tau2-TRACE: Trajectory-Aware Comprehensive Evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- analyze subcommand ----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Post-hoc analysis of existing simulation results",
    )
    analyze_parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to a tau2-bench Results JSON file",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/experiments/tau2_trace/results/tau2_trace_augmented.csv"),
        help="Output path for augmented CSV",
    )
    analyze_parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override domain (auto-detected from results if not specified)",
    )
    analyze_parser.add_argument(
        "--recovery-window",
        type=int,
        default=3,
        help="Error recovery lookahead window size (default: 3)",
    )
    analyze_parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-based interaction quality evaluation (incurs API cost)",
    )
    analyze_parser.add_argument(
        "--llm-judge-model",
        type=str,
        default=None,
        help="LLM model for judge calls (required with --llm-judge)",
    )

    # ---- run subcommand ----
    run_parser = subparsers.add_parser(
        "run",
        help="Run live simulations with optional adversarial wrapper + trace analysis",
    )
    run_parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to run (telecom, retail, airline)",
    )
    run_parser.add_argument(
        "--agent-llm",
        type=str,
        required=True,
        help="LLM model for the agent (e.g. gpt-4.1)",
    )
    run_parser.add_argument(
        "--user-llm",
        type=str,
        required=True,
        help="LLM model for the user simulator (e.g. gpt-4.1)",
    )
    run_parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of tasks to run (default: 1)",
    )
    run_parser.add_argument(
        "--task-split",
        type=str,
        default=None,
        help="Task split to use (e.g. 'base' for full benchmark set)",
    )
    run_parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum orchestrator steps per simulation (default: 100)",
    )
    run_parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="Wrap user simulator with adversarial perturbation proxy",
    )
    run_parser.add_argument(
        "--perturbation-rate",
        type=float,
        default=0.20,
        help="Probability of perturbation per user turn (default: 0.20)",
    )
    run_parser.add_argument(
        "--adversarial-seed",
        type=int,
        default=42,
        help="Base seed for adversarial RNG (default: 42)",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/experiments/tau2_trace/results/live_run"),
        help="Output path prefix (writes .json and .csv)",
    )
    run_parser.add_argument(
        "--recovery-window",
        type=int,
        default=3,
        help="Error recovery lookahead window size (default: 3)",
    )
    run_parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-based interaction quality evaluation",
    )
    run_parser.add_argument(
        "--llm-judge-model",
        type=str,
        default=None,
        help="LLM model for judge calls (required with --llm-judge)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=300,
        help="Base seed for simulations (default: 300)",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        if args.llm_judge and not args.llm_judge_model:
            analyze_parser.error(
                "--llm-judge-model is required when --llm-judge is set"
            )

        analyze_results(
            results_path=args.results_file,
            output_path=args.output,
            domain_override=args.domain,
            recovery_window=args.recovery_window,
            use_llm_judge=args.llm_judge,
            llm_judge_model=args.llm_judge_model,
        )
    elif args.command == "run":
        if args.llm_judge and not args.llm_judge_model:
            run_parser.error("--llm-judge-model is required when --llm-judge is set")

        run_live(
            domain=args.domain,
            agent_llm=args.agent_llm,
            user_llm=args.user_llm,
            output_path=args.output,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            task_split=args.task_split,
            max_steps=args.max_steps,
            adversarial=args.adversarial,
            perturbation_rate=args.perturbation_rate,
            adversarial_seed=args.adversarial_seed,
            recovery_window=args.recovery_window,
            use_llm_judge=args.llm_judge,
            llm_judge_model=args.llm_judge_model,
            seed=args.seed,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
