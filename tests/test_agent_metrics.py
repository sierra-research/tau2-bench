"""Tests for non-evaluable termination handling in agent metrics."""

from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.metrics.agent_metrics import compute_metrics


def _make_info() -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(
    task_id: str,
    termination_reason: TerminationReason,
    reward: float | None = None,
) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        reward_info=RewardInfo(reward=reward) if reward is not None else None,
        messages=[],
        trial=0,
        seed=42,
    )


def test_context_window_sims_excluded_from_metrics_like_infra():
    """CONTEXT_WINDOW_EXCEEDED simulations never produced an evaluable rollout, so
    they are excluded from metric denominators, the same as INFRASTRUCTURE_ERROR."""
    results = Results(
        info=_make_info(),
        tasks=[_make_task("t0"), _make_task("t1")],
        simulations=[
            _make_sim("t0", TerminationReason.INFRASTRUCTURE_ERROR),
            _make_sim("t1", TerminationReason.CONTEXT_WINDOW_EXCEEDED),
        ],
    )

    metrics = compute_metrics(results)

    # Both sims are non-evaluable, so nothing is left to score.
    assert metrics.total_simulations == 0
    # infra_error_count still tracks infrastructure errors specifically.
    assert metrics.infra_error_count == 1


def test_metrics_score_only_evaluable_sims_in_mixed_run():
    """With a mix of evaluable and non-evaluable sims, only the evaluable ones
    reach the denominator. Both infrastructure and context-window failures are
    dropped, so they cannot drag down or inflate the reported metrics."""
    results = Results(
        info=_make_info(),
        tasks=[_make_task("t0"), _make_task("t1"), _make_task("t2")],
        simulations=[
            _make_sim("t0", TerminationReason.AGENT_STOP, reward=1.0),
            _make_sim("t1", TerminationReason.INFRASTRUCTURE_ERROR),
            _make_sim("t2", TerminationReason.CONTEXT_WINDOW_EXCEEDED),
        ],
    )

    metrics = compute_metrics(results)

    # Only the AGENT_STOP sim is evaluable.
    assert metrics.total_simulations == 1
    assert metrics.avg_reward == 1.0
    assert metrics.infra_error_count == 1
