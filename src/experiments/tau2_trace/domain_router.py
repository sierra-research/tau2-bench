"""
Phase 4: Domain-Aware Metric Router.

Dispatches to the correct evaluators based on domain and aggregates
results into a CompositeScorecard.
"""

from __future__ import annotations

from typing import Optional

from experiments.tau2_trace.interaction_quality import evaluate_interaction_quality
from experiments.tau2_trace.models import CompositeScorecard
from experiments.tau2_trace.tool_order_evaluator import evaluate_tool_ordering
from experiments.tau2_trace.trajectory_analyzer import analyze_trajectory
from tau2.data_model.simulation import SimulationRun


def evaluate_simulation_trace(
    simulation: SimulationRun,
    domain: str,
    expected_actions: int = 0,
    recovery_window: int = 3,
    use_llm_judge: bool = False,
    llm_judge_model: Optional[str] = None,
) -> CompositeScorecard:
    """
    Run the full tau2-TRACE analysis pipeline on a single SimulationRun.

    Args:
        simulation: A completed simulation run with full message trajectory.
        domain: The domain name (telecom, retail, airline).
        expected_actions: Expected number of actions from task metadata
            (task_num_actions from Results.to_df()). Used for turns_vs_expected.
        recovery_window: How many subsequent tool calls to inspect when
            looking for error recovery (default 3).
        use_llm_judge: Enable LLM-based evaluation for interaction quality
            metrics (higher fidelity, incurs API cost).
        llm_judge_model: LLM model identifier for judge calls.

    Returns:
        A CompositeScorecard containing all computed metrics.
    """
    traj_metrics, records = analyze_trajectory(
        simulation, domain, recovery_window=recovery_window
    )

    ordering_metrics = evaluate_tool_ordering(
        records=records,
        domain=domain,
        task_id=simulation.task_id,
        trial=simulation.trial,
    )

    interaction_metrics = evaluate_interaction_quality(
        messages=simulation.messages,
        records=records,
        domain=domain,
        task_id=simulation.task_id,
        trial=simulation.trial,
        total_turns=traj_metrics.total_turns,
        agent_tool_call_count=traj_metrics.agent_tool_call_count,
        expected_actions=expected_actions,
        use_llm_judge=use_llm_judge,
        llm_judge_model=llm_judge_model,
    )

    return CompositeScorecard(
        task_id=simulation.task_id,
        trial=simulation.trial,
        domain=domain,
        trajectory=traj_metrics,
        ordering=ordering_metrics,
        interaction=interaction_metrics,
    )


def evaluate_results_trace(
    simulations: list[SimulationRun],
    domain: str,
    task_expected_actions: Optional[dict[str, int]] = None,
    recovery_window: int = 3,
    use_llm_judge: bool = False,
    llm_judge_model: Optional[str] = None,
) -> list[CompositeScorecard]:
    """
    Batch-evaluate a list of simulation runs.

    Args:
        simulations: List of SimulationRun objects.
        domain: Domain name.
        task_expected_actions: Optional mapping of task_id -> expected action count.
        recovery_window: Error recovery lookahead window size.
        use_llm_judge: Enable LLM-based interaction quality evaluation.
        llm_judge_model: LLM model identifier for judge calls.

    Returns:
        List of CompositeScorecard, one per simulation.
    """
    if task_expected_actions is None:
        task_expected_actions = {}

    scorecards = []
    for sim in simulations:
        expected = task_expected_actions.get(sim.task_id, 0)
        scorecard = evaluate_simulation_trace(
            sim,
            domain,
            expected,
            recovery_window=recovery_window,
            use_llm_judge=use_llm_judge,
            llm_judge_model=llm_judge_model,
        )
        scorecards.append(scorecard)

    return scorecards
