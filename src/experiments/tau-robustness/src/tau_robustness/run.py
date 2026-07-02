"""
Standalone runner for AVER robustness evaluation.

Entry points:
    python -m tau_robustness.run -d retail --num-tasks 5 --num-trials 3
    tau2 run -d retail --mode robustness  (via thin CLI hook)
"""

import argparse
import json
import multiprocessing
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RunConfig,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.gym.gym_agent import GymAgent
from tau2.metrics.agent_metrics import compute_metrics
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, get_global_user_sim_guidelines
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.environment.environment import ToolResponseMismatchError
from tau2.utils.utils import DATA_DIR, get_commit_hash, get_now

from tau_robustness.injection_config import InjectionConfig
from tau_robustness.injector import ErrorInjector
from tau_robustness.metrics import RobustnessMetrics, compute_robustness_metrics
from tau_robustness.robustness_orchestrator import RobustOrchestrator


def run_robustness_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    error_injector: ErrorInjector,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    enforce_communication_protocol: bool = False,
) -> SimulationRun:
    """Run a single task with error injection via RobustOrchestrator.

    Mirrors tau2.run.run_task but swaps Orchestrator for RobustOrchestrator
    and attaches robustness metrics to the simulation result.
    """
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    logger.info(
        f"STARTING ROBUSTNESS SIMULATION: Domain: {domain}, Task: {task.id}"
    )

    # --- Setup (mirrors tau2.run.run_task) ---
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    AgentConstructor = registry.get_agent_constructor(agent)

    solo_mode = False
    if issubclass(AgentConstructor, LLMAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
        )
    elif issubclass(AgentConstructor, LLMGTAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, LLMSoloAgent):
        solo_mode = True
        environment = environment_constructor(solo_mode=True)
        user_tools = environment.get_user_tools() if environment.user_tools else []
        agent_instance = AgentConstructor(
            tools=environment.get_tools() + user_tools,
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, GymAgent):
        agent_instance = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
        )
    else:
        raise ValueError(f"Unknown agent type: {AgentConstructor}")

    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    if issubclass(UserConstructor, DummyUser):
        assert isinstance(agent_instance, LLMSoloAgent)

    user_instance = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )

    # --- Robustness-specific: configure injector for this task ---
    error_injector.set_task(task)

    orchestrator = RobustOrchestrator(
        domain=domain,
        agent=agent_instance,
        user=user_instance,
        environment=environment,
        task=task,
        error_injector=error_injector,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
        validate_communication=enforce_communication_protocol,
    )
    simulation = orchestrator.run()

    # Evaluate with standard τ²-bench evaluator.
    # The EnvironmentEvaluator replays the trajectory and compares tool responses
    # against the fresh environment. When an injection modified a response, the
    # replay will see a mismatch and raise ValueError. In that case, fall back to
    # evaluating without the ENV component (DB check) — we still get COMMUNICATE
    # and ACTION scores, and Recovery = 0.0 is the correct score since the agent
    # acted on corrupted data.
    try:
        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=evaluation_type,
            solo_mode=solo_mode,
        )
    except ToolResponseMismatchError as e:
        logger.warning(
            f"Evaluation replay mismatch (expected with injection): {str(e)[:200]}"
        )
        # Fall back: evaluate without ENV replay.
        # The injected tool response differs from a clean replay, so ENV/DB
        # evaluation is meaningless. We still get ACTION + COMMUNICATE scores.
        from tau2.evaluator.evaluator_action import ActionEvaluator
        from tau2.evaluator.evaluator_communicate import CommunicateEvaluator

        action_info = ActionEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages,
        )
        comm_info = CommunicateEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages,
        )
        reward_breakdown = {}
        reward = 1.0
        if task.evaluation_criteria and task.evaluation_criteria.reward_basis:
            from tau2.data_model.tasks import RewardType
            basis = set(task.evaluation_criteria.reward_basis)
            if basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
                reward_breakdown[RewardType.DB] = 0.0
                reward *= 0.0
            if basis & {RewardType.ACTION}:
                if action_info.reward_breakdown:
                    reward_breakdown.update(action_info.reward_breakdown)
                reward *= action_info.reward
            if basis & {RewardType.COMMUNICATE}:
                if comm_info.reward_breakdown:
                    reward_breakdown.update(comm_info.reward_breakdown)
                reward *= comm_info.reward

        from tau2.data_model.simulation import RewardInfo
        reward_info = RewardInfo(
            reward=reward,
            action_checks=action_info.action_checks,
            communicate_checks=comm_info.communicate_checks,
            reward_basis=(
                task.evaluation_criteria.reward_basis
                if task.evaluation_criteria else None
            ),
            reward_breakdown=reward_breakdown,
            info={"note": "ENV evaluation skipped due to injection replay mismatch"},
        )
    simulation.reward_info = reward_info

    # Compute robustness metrics and attach to reward_info
    task_reward = reward_info.reward if reward_info else 0.0
    robustness_metrics = orchestrator.get_robustness_metrics(task_reward)

    # Store robustness data in reward_info.info dict
    if reward_info and reward_info.info is None:
        reward_info.info = {}
    if reward_info:
        reward_info.info["robustness"] = robustness_metrics.model_dump()
        reward_info.info["injection_summary"] = (
            error_injector.get_injection_summary()
        )

    logger.info(
        f"FINISHED ROBUSTNESS SIMULATION: Domain: {domain}, Task: {task.id}, "
        f"Reward: {task_reward:.3f}, "
        f"Injections: {error_injector.num_injections}, "
        f"Detection: {robustness_metrics.detection_score}, "
        f"Recovery: {robustness_metrics.recovery_score}"
    )

    return simulation


def run_robustness(
    domain: str,
    agent: str = "llm_agent",
    user: str = "user_simulator",
    llm_agent: Optional[str] = None,
    llm_user: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_args_user: Optional[dict] = None,
    num_tasks: Optional[int] = None,
    num_trials: int = 3,
    max_steps: int = 100,
    max_errors: int = 10,
    max_concurrency: int = 1,
    injection_rate: float = 1.0,
    injection_seed: int = 42,
    max_injections: int = 1,
    persistent: bool = False,
    negative_control_fraction: float = 0.2,
    save_to: Optional[str] = None,
    seed: int = 300,
    log_level: str = "INFO",
    task_split: str = "base",
) -> Results:
    """Run robustness evaluation across tasks.

    Loads tasks from the registry, creates per-task ErrorInjectors,
    runs simulations through RobustOrchestrator, and aggregates results.
    """
    # Set log level
    logger.remove()
    logger.add(lambda msg: print(msg), level=log_level)

    # Load injection config
    injection_config = InjectionConfig.from_domain(domain)
    ConsoleDisplay.console.print(
        f"\n[bold yellow]AVER Robustness Mode[/bold yellow] | "
        f"domain={domain} | injection_rate={injection_rate} | "
        f"persistent={persistent} | "
        f"{len(injection_config.injections)} injection definitions loaded\n"
    )

    # Load tasks (using tau2's registry API)
    from tau2.run import get_tasks
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split,
        num_tasks=num_tasks,
    )

    if not tasks:
        raise ValueError(f"No tasks found for domain '{domain}'")

    ConsoleDisplay.console.print(
        f"Loaded {len(tasks)} tasks, {num_trials} trials each\n"
    )

    # Determine negative control tasks
    random.seed(seed)
    num_negative = max(1, int(len(tasks) * negative_control_fraction))
    negative_control_ids = set(
        t.id for t in random.sample(tasks, min(num_negative, len(tasks)))
    )

    # Setup save path
    if save_to is None:
        mode_tag = "persistent" if persistent else "transient"
        save_to = f"{get_now()}_{domain}_robustness_{mode_tag}"
    save_path = DATA_DIR / "simulations" / f"{save_to}.json"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare seeds
    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    lock = multiprocessing.Lock()

    # Build info object
    environment_info = registry.get_env_constructor(domain)().get_info(
        include_tool_info=False
    )
    info = Info(
        git_commit=get_commit_hash(),
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=UserInfo(
            implementation=user,
            llm=llm_user,
            llm_args=llm_args_user,
            global_simulation_guidelines=get_global_user_sim_guidelines(),
        ),
        agent_info=AgentInfo(
            implementation=agent,
            llm=llm_agent,
            llm_args=llm_args_agent,
        ),
        environment_info=environment_info,
        seed=seed,
    )

    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )

    # Save initial checkpoint
    with open(save_path, "w") as fp:
        fp.write(simulation_results.model_dump_json(indent=2))

    def _save(simulation: SimulationRun):
        with lock:
            with open(save_path, "r") as fp:
                ckpt = json.load(fp)
            ckpt["simulations"].append(simulation.model_dump())
            with open(save_path, "w") as fp:
                json.dump(ckpt, fp, indent=2)

    def _run(
        task: Task, trial: int, seed: int, progress_str: str
    ) -> SimulationRun:
        ConsoleDisplay.console.print(
            Text(
                text=f"{progress_str}. Task {task.id}, trial {trial + 1}",
                style="bold green",
            )
        )

        # Create per-task injector
        is_negative = task.id in negative_control_ids
        task_seed = injection_seed + hash(task.id) % (2**31)
        error_injector = ErrorInjector(
            injection_config=injection_config,
            injection_rate=0.0 if is_negative else injection_rate,
            seed=task_seed,
            max_injections_per_run=max_injections,
            persistent=persistent,
        )

        try:
            simulation = run_robustness_task(
                domain=domain,
                task=task,
                agent=agent,
                user=user,
                error_injector=error_injector,
                llm_agent=llm_agent,
                llm_args_agent=llm_args_agent,
                llm_user=llm_user,
                llm_args_user=llm_args_user,
                max_steps=max_steps,
                max_errors=max_errors,
                seed=seed,
            )
            simulation.trial = trial

            # Mark negative controls in metadata
            if is_negative and simulation.reward_info:
                if simulation.reward_info.info is None:
                    simulation.reward_info.info = {}
                simulation.reward_info.info["is_negative_control"] = True

            ConsoleDisplay.display_simulation(simulation, show_details=False)
            _save(simulation)
        except Exception as e:
            logger.error(f"Error running task {task.id}, trial {trial}: {e}")
            return None
        return simulation

    # Build run args
    args = []
    for trial in range(num_trials):
        for i, task in enumerate(tasks):
            progress_str = (
                f"{i + 1}/{len(tasks)} (trial {trial + 1}/{num_trials})"
            )
            args.append((task, trial, seeds[trial], progress_str))

    # Execute
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        res = list(executor.map(_run, *zip(*args)))
        simulation_results.simulations.extend([r for r in res if r is not None])

    # Display results
    metrics = compute_metrics(simulation_results)
    ConsoleDisplay.display_agent_metrics(metrics)
    _display_robustness_summary(simulation_results, negative_control_ids)

    # Save final
    with open(save_path, "w") as fp:
        fp.write(simulation_results.model_dump_json(indent=2))

    ConsoleDisplay.console.print(
        f"\nResults saved to [bold blue]{save_path}[/bold blue]\n"
    )

    return simulation_results


def _display_robustness_summary(
    results: Results, negative_control_ids: set[str]
) -> None:
    """Display AVER robustness summary after simulation."""
    total_injections = 0
    detection_scores = []
    recovery_scores = []
    false_positives = 0
    total_negatives = 0

    for sim in results.simulations:
        if not sim.reward_info or not sim.reward_info.info:
            continue
        robustness = sim.reward_info.info.get("robustness", {})
        is_negative = sim.reward_info.info.get("is_negative_control", False)

        if is_negative:
            total_negatives += 1
            # Check for false positives on negative controls
            if robustness.get("detection_score") and robustness["detection_score"] > 0:
                false_positives += 1
        else:
            if robustness.get("num_injections", 0) > 0:
                total_injections += robustness["num_injections"]
                if robustness.get("detection_score") is not None:
                    detection_scores.append(robustness["detection_score"])
                if robustness.get("recovery_score") is not None:
                    recovery_scores.append(robustness["recovery_score"])

    if total_injections == 0:
        ConsoleDisplay.console.print(
            "\n[yellow]No errors were injected. "
            "Try increasing --injection-rate.[/yellow]"
        )
        return

    avg_det = sum(detection_scores) / len(detection_scores) if detection_scores else 0
    avg_rec = sum(recovery_scores) / len(recovery_scores) if recovery_scores else 0
    fpr = false_positives / total_negatives if total_negatives > 0 else None

    # Collect diagnosis scores
    diagnosis_scores = []
    for sim in results.simulations:
        if not sim.reward_info or not sim.reward_info.info:
            continue
        robustness = sim.reward_info.info.get("robustness", {})
        is_neg = sim.reward_info.info.get("is_negative_control", False)
        if not is_neg and robustness.get("diagnosis_score") is not None:
            diagnosis_scores.append(robustness["diagnosis_score"])
    avg_diag = sum(diagnosis_scores) / len(diagnosis_scores) if diagnosis_scores else 0

    # AVER = Detection × 0.4 + Diagnosis × 0.2 + Recovery × 0.4
    aver = (avg_det * 0.4 + avg_diag * 0.2 + avg_rec * 0.4) * 100

    summary = (
        f"\n[bold yellow]AVER Robustness Results[/bold yellow]\n"
        f"   Total injections: {total_injections}\n"
        f"   Avg Detection:    {avg_det:.1%}\n"
        f"   Avg Diagnosis:    {avg_diag:.1%}\n"
        f"   Avg Recovery:     {avg_rec:.1%}\n"
        f"   AVER Score:       {aver:.1f} / 100\n"
    )
    if fpr is not None:
        summary += f"   False Positive Rate: {fpr:.0%} ({false_positives}/{total_negatives} negative controls)\n"

    ConsoleDisplay.console.print(summary)



def run_robustness_from_args(args: argparse.Namespace) -> Results:
    """Entry point from thin CLI hook — converts argparse namespace to kwargs."""
    return run_robustness(
        domain=args.domain,
        agent=getattr(args, "agent", "llm_agent"),
        user=getattr(args, "user", "user_simulator"),
        llm_agent=getattr(args, "agent_llm", None),
        llm_user=getattr(args, "user_llm", None),
        llm_args_agent=_build_llm_args(args, "agent"),
        llm_args_user=_build_llm_args(args, "user"),
        num_tasks=getattr(args, "num_tasks", None),
        num_trials=getattr(args, "num_trials", 3),
        max_steps=getattr(args, "max_steps", 100),
        max_errors=getattr(args, "max_errors", 10),
        max_concurrency=getattr(args, "max_concurrency", 1),
        injection_rate=getattr(args, "injection_rate", 1.0),
        injection_seed=getattr(args, "injection_seed", 42),
        max_injections=getattr(args, "max_injections", 1),
        persistent=getattr(args, "persistent", False),
        save_to=getattr(args, "save_to", None),
        seed=getattr(args, "seed", 300),
        log_level=getattr(args, "log_level", "INFO"),
        task_split=getattr(args, "task_split", "base"),
    )


def _build_llm_args(args: argparse.Namespace, role: str) -> dict:
    """Build LLM args dict from argparse namespace."""
    llm_args = {}
    temp = getattr(args, f"{role}_temperature", None)
    if temp is not None:
        llm_args["temperature"] = temp
    seed = getattr(args, f"{role}_seed", None)
    if seed is not None:
        llm_args["seed"] = seed
    return llm_args


def main():
    """Standalone CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AVER: Robustness evaluation for tau2-bench"
    )
    parser.add_argument(
        "-d", "--domain", type=str, required=True,
        help="Domain to evaluate (retail, airline, telecom)",
    )
    parser.add_argument(
        "--agent", type=str, default="llm_agent",
        help="Agent implementation name. Default: llm_agent",
    )
    parser.add_argument(
        "--user", type=str, default="user_simulator",
        help="User simulator implementation. Default: user_simulator",
    )
    parser.add_argument(
        "--agent-llm", type=str, default=None,
        help="LLM for agent (e.g. gpt-4.1). Defaults to config default.",
    )
    parser.add_argument(
        "--user-llm", type=str, default=None,
        help="LLM for user simulator. Defaults to config default.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--num-trials", type=int, default=3,
        help="Trials per task. Default: 3",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Max parallel simulations. Default: 1",
    )
    parser.add_argument(
        "--injection-rate", type=float, default=1.0,
        help="Fraction of eligible tool calls to inject. Default: 1.0",
    )
    parser.add_argument(
        "--injection-seed", type=int, default=42,
        help="Seed for injection randomness. Default: 42",
    )
    parser.add_argument(
        "--max-injections", type=int, default=1,
        help="Max injections per simulation. Default: 1",
    )
    parser.add_argument(
        "--persistent", action="store_true",
        help="Enable persistent injection mode (DB corruption, retry fails).",
    )
    parser.add_argument(
        "--save-to", type=str, default=None,
        help="Save filename (without .json). Default: auto-generated.",
    )
    parser.add_argument(
        "--seed", type=int, default=300,
        help="Random seed. Default: 300",
    )
    parser.add_argument(
        "--task-split", type=str, default="base",
        help="Task split to use. Default: base",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level. Default: INFO",
    )

    args = parser.parse_args()
    run_robustness(
        domain=args.domain,
        agent=args.agent,
        user=args.user,
        llm_agent=args.agent_llm,
        llm_user=args.user_llm,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        injection_rate=args.injection_rate,
        injection_seed=args.injection_seed,
        max_injections=args.max_injections,
        persistent=args.persistent,
        save_to=args.save_to,
        seed=args.seed,
        task_split=args.task_split,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
