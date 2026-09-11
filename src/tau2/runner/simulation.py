"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional, Union

from loguru import logger

from tau2.data_model.simulation import (
    DEFAULT_SCORES,
    DeliveryJudgeSettings,
    NativenessJudgeSettings,
    QualityJudgeSettings,
    Score,
    SimulationRun,
)
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.judges.attach import attach_delivery, attach_nativeness, attach_quality
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    scores: Optional[set[Score]] = None,
    nativeness_judge: Optional[NativenessJudgeSettings] = None,
    quality_judge: Optional[QualityJudgeSettings] = None,
    delivery_judge: Optional[DeliveryJudgeSettings] = None,
    env_kwargs: Optional[dict] = None,
    llm_communicate_judge: Optional[bool] = None,
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
        scores: Which scoring axes to compute (see ``Score``). Defaults to
            ``DEFAULT_SCORES`` (reward + quality + nativeness).
        nativeness_judge: LLM nativeness-judge settings; None uses defaults.
        quality_judge: Universal quality-rubric settings; None uses defaults.
        delivery_judge: Audio delivery judge settings (used only when 'delivery'
            is in scores; voice-only); None uses defaults.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor (e.g., retrieval_variant for banking_knowledge).
        llm_communicate_judge: Force (True) or disable (False) the LLM judge
            for communicate_info checks. None (default) auto-activates the
            judge only for non-English runs.

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="gpt-5.4-mini", instructions=task.user_scenario,
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

    scores = set(DEFAULT_SCORES) if scores is None else scores

    # Evaluate (pass@1 reward) — gated by the selected scores.
    if Score.REWARD in scores:
        simulation.reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=solo_mode,
            domain=domain,
            mode=mode,
            env_kwargs=env_kwargs,
            llm_communicate_judge=llm_communicate_judge,
        )

    # Nativeness — a decoupled axis, never folded into reward (no-op for English).
    # The agent's provider/voice (for the judge's gender factors) is recorded on
    # the simulation by the voice orchestrator, so no need to thread it here.
    if Score.NATIVENESS in scores:
        attach_nativeness(
            simulation,
            task,
            settings=nativeness_judge,
            domain=domain,
        )

    # Universal process/interaction quality — separate from task reward.
    if Score.QUALITY in scores:
        attach_quality(
            simulation,
            task,
            domain=domain,
            settings=quality_judge,
        )

    # Delivery — audio judge (fidelity + intonation), a perceptual-quality axis.
    # Opt-in and voice-only (no-op when the run has no ticks). Must run here while
    # audio_content is in memory (it is excluded from serialization).
    if Score.DELIVERY in scores:
        attach_delivery(simulation, settings=delivery_judge)

    reward = simulation.reward_info.reward if simulation.reward_info else None
    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, reward={reward}"
    )

    return simulation
