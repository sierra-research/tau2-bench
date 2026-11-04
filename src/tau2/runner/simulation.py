"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional, Union

from loguru import logger

from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
    communicate_scoring: Optional[CommunicateScoring] = None,
    canonical_tasks: bool = False,
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
        communicate_scoring: Scorer for orchestrators built without a call
            context; defaults to the pinned semantic judge. An orchestrator's
            call context always supplies its own scorer.
        canonical_tasks: Whether the run's plan bound the canonical registry
            tasks (voice-v1); the call's language reload must repeat it.

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
    # Direct callers of this layer need the same fresh scenario for replay that
    # batch execution passes explicitly. Resolve it before any provider work.
    call_context = orchestrator.call_context
    customer_language = None
    if call_context is not None:
        from tau2.multilingual.runtime import (
            language_environment_kwargs,
            task_digest,
        )
        from tau2.runner.presets import reload_recorded_call

        context = call_context.language
        loaded = reload_recorded_call(
            context,
            orchestrator.environment.get_domain_name(),
            canonical_tasks=canonical_tasks,
        )
        if task_digest(orchestrator.task) != context.task_sha256:
            raise ValueError(
                "Evaluation task differs from the constructed language call"
            )
        expected_kwargs = language_environment_kwargs(loaded)
        if expected_kwargs:
            if env_kwargs is None:
                env_kwargs = expected_kwargs
            else:
                if any(
                    key not in env_kwargs or env_kwargs[key] != value
                    for key, value in expected_kwargs.items()
                ):
                    raise ValueError(
                        "Explicit evaluation environment differs from the constructed language call"
                    )
                # Retrieval and intake add domain-specific constructor arguments
                # alongside the pinned regional database and policy inputs.
                env_kwargs = {**env_kwargs, **expected_kwargs}
        if env_kwargs is not None and (
            "task" in env_kwargs and env_kwargs["task"] != orchestrator.task
        ):
            raise ValueError(
                "Evaluation task differs from the constructed language call"
            )
        if communicate_scoring is not None and (
            communicate_scoring != call_context.communicate_scoring
        ):
            raise ValueError("communicate scoring differs from the call context")
        communicate_scoring = call_context.communicate_scoring
        customer_language = context.provider_language
    communicate_scoring = communicate_scoring or CommunicateScoring()

    # Run the orchestrator
    simulation = orchestrator.run()
    if call_context is not None:
        from tau2.voice.voice_gender import VOICE_GENDER_CATALOG_REVISION

        simulation.language_context = call_context.language
        simulation.agent_voice = call_context.agent_voice
        simulation.agent_voice_gender = call_context.agent_voice_gender
        if call_context.agent_voice or call_context.agent_voice_gender:
            simulation.agent_voice_catalog = VOICE_GENDER_CATALOG_REVISION

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
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
        domain=domain,
        mode=mode,
        env_kwargs=env_kwargs,
        communicate_scoring=communicate_scoring,
        customer_language=customer_language,
    )
    simulation.reward_info = reward_info

    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, "
        f"reward={reward_info.reward}"
    )

    return simulation
