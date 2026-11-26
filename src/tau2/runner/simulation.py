"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional, Union

from loguru import logger

from tau2.data_model.simulation import (
    DBCheck,
    EnvAssertionCheck,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator


def _uses_live_crm_evaluation(domain: str, task) -> bool:
    criteria = task.evaluation_criteria
    if domain != "crm" or criteria is None:
        return False
    basis = set(criteria.reward_basis or [])
    return bool(criteria.env_assertions) and basis == {RewardType.ENV_ASSERTION}


def _evaluate_live_crm(orchestrator, task) -> RewardInfo:
    if orchestrator.termination_reason not in {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }:
        return RewardInfo(
            reward=0.0,
            info={
                "note": (
                    "Simulation terminated prematurely. Termination reason: "
                    f"{orchestrator.termination_reason.value}"
                )
            },
        )

    env = orchestrator.environment
    checks = []
    reward = 1.0
    for assertion in task.evaluation_criteria.env_assertions or []:
        try:
            met = env.run_env_assertion(assertion, raise_assertion_error=False)
        except Exception as exc:
            logger.warning(f"Live CRM assertion {assertion.func_name} failed: {exc}")
            met = False
        check = EnvAssertionCheck(
            env_assertion=assertion,
            met=met,
            reward=1.0 if met else 0.0,
        )
        checks.append(check)
        reward *= check.reward

    return RewardInfo(
        reward=reward,
        db_check=DBCheck(db_match=(reward == 1.0), db_reward=reward),
        env_assertions=checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown={RewardType.ENV_ASSERTION: reward},
        info={"note": "Evaluated live CRM environment; replay skipped"},
    )


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
) -> SimulationRun:
    """Run a simulation and evaluate the result.

    Takes a fully constructed orchestrator (with agent, user, environment, and task
    already wired in), runs the simulation, evaluates it, and returns the result
    with reward_info attached.

    This is the lowest-level entry point. It has no dependency on the registry
    or RunConfig -- everything is already encapsulated in the orchestrator.

    Args:
        orchestrator: A fully constructed Orchestrator (half-duplex) or
            FullDuplexOrchestrator (full-duplex/voice). Must have agent, user,
            environment, and task set.
        evaluation_type: The type of evaluation to perform. Defaults to ALL.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor (e.g., retrieval_variant for banking_knowledge).

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="gpt-4.1", instructions=task.user_scenario,
                             tools=env.get_user_tools())
        orchestrator = Orchestrator(
            domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=100,
        )
        result = run_simulation(orchestrator)
        print(result.reward_info.reward)
    """
    # Run the orchestrator
    simulation = orchestrator.run()

    # Save the actual policy used for this simulation
    simulation.policy = orchestrator.environment.get_policy()

    # Extract context from the orchestrator -- no external params needed
    domain = orchestrator.environment.get_domain_name()
    task = orchestrator.task
    is_full_duplex = isinstance(orchestrator, FullDuplexOrchestrator)
    mode = (
        CommunicationMode.FULL_DUPLEX
        if is_full_duplex
        else CommunicationMode.HALF_DUPLEX
    )
    solo_mode = getattr(orchestrator, "solo_mode", False)

    # Evaluate
    if _uses_live_crm_evaluation(domain, task):
        reward_info = _evaluate_live_crm(orchestrator, task)
    else:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=solo_mode,
            domain=domain,
            mode=mode,
            env_kwargs=env_kwargs,
        )
    simulation.reward_info = reward_info

    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, "
        f"reward={reward_info.reward}"
    )

    return simulation
