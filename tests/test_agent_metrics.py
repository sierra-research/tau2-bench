"""Regression tests for infrastructure-error handling in agent metrics.

Covers issue #493: tasks for which every trial ended with an
INFRASTRUCTURE_ERROR used to vanish from the task-level denominator (and under
pandas 3.x the infra-error filter itself silently never matched), so pass^k
could be reported as 100% from a single successful simulation.
"""

import pytest

from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.metrics.agent_metrics import (
    compute_metrics,
    get_metrics_df,
    get_tasks_pass_hat_k,
)

NOW = "2026-08-31T00:00:00"


def _make_sim(sim_id, task_id, trial, termination, reward):
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        start_time=NOW,
        end_time=NOW,
        duration=1.0,
        termination_reason=termination,
        trial=trial,
        reward_info=RewardInfo(reward=reward) if reward is not None else None,
    )


def _make_results(sims, num_trials=2, task_ids=("task_001", "task_002")):
    info = Info(
        git_commit="test",
        num_trials=num_trials,
        max_steps=10,
        max_errors=5,
        user_info=UserInfo(implementation="dummy_user"),
        agent_info=AgentInfo(implementation="llm_agent"),
        environment_info=EnvironmentInfo(domain_name="airline", policy=""),
    )
    tasks = [
        Task(
            id=task_id,
            user_scenario=UserScenario(instructions=f"do {task_id}"),
            evaluation_criteria=EvaluationCriteria(),
        )
        for task_id in task_ids
    ]
    return Results(info=info, tasks=tasks, simulations=sims)


def _issue_493_results():
    """The minimal example from issue #493: 2 tasks x 2 trials (4 slots).

    task_001: trial 0 succeeds, trial 1 infra error
    task_002: both trials infra error
    """
    sims = [
        _make_sim("s1", "task_001", 0, TerminationReason.AGENT_STOP, 1.0),
        _make_sim("s2", "task_001", 1, TerminationReason.INFRASTRUCTURE_ERROR, None),
        _make_sim("s3", "task_002", 0, TerminationReason.INFRASTRUCTURE_ERROR, None),
        _make_sim("s4", "task_002", 1, TerminationReason.INFRASTRUCTURE_ERROR, None),
    ]
    return _make_results(sims, num_trials=2)


def test_infra_errors_are_filtered_from_metrics_df():
    """The infra-error filter must actually match under pandas 2.x/3.x."""
    results = _issue_493_results()
    df, _ = get_metrics_df(results)
    assert len(df) == 1
    assert (
        df.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR.value
    ).all()


def test_all_infra_task_stays_in_pass_hat_k_denominator():
    """A task whose trials all hit infra errors must contribute pass^k = 0."""
    results = _issue_493_results()
    df_pk = get_tasks_pass_hat_k(results)
    assert set(df_pk.index) == {"task_001", "task_002"}
    assert df_pk.loc["task_001", "pass^1"] == pytest.approx(1.0)
    assert df_pk.loc["task_002", "pass^1"] == pytest.approx(0.0)


def test_pass_hat_1_not_inflated_to_100_percent():
    """4 slots with a single success must not report pass^1 = 1.0."""
    results = _issue_493_results()
    metrics = compute_metrics(results)
    assert metrics.infra_error_count == 3
    assert metrics.total_simulations == 1
    assert metrics.total_tasks == 2
    assert metrics.pass_hat_ks[1] == pytest.approx(0.5)


def test_all_infra_simulations_yield_empty_pass_hat_k():
    """If every simulation is an infra error, no pass^k is reported."""
    sims = [
        _make_sim("s1", "task_001", 0, TerminationReason.INFRASTRUCTURE_ERROR, None),
        _make_sim("s2", "task_001", 1, TerminationReason.INFRASTRUCTURE_ERROR, None),
    ]
    results = _make_results(sims, num_trials=2, task_ids=("task_001",))
    metrics = compute_metrics(results)
    assert metrics.infra_error_count == 2
    assert metrics.pass_hat_ks == {}


def test_metrics_unchanged_without_infra_errors():
    """Normal runs (no infra errors) keep their previous pass^k values."""
    sims = [
        _make_sim("s1", "task_001", 0, TerminationReason.AGENT_STOP, 1.0),
        _make_sim("s2", "task_001", 1, TerminationReason.AGENT_STOP, 0.0),
        _make_sim("s3", "task_002", 0, TerminationReason.AGENT_STOP, 0.0),
        _make_sim("s4", "task_002", 1, TerminationReason.AGENT_STOP, 0.0),
    ]
    results = _make_results(sims, num_trials=2)
    metrics = compute_metrics(results)
    assert metrics.infra_error_count == 0
    assert metrics.total_simulations == 4
    assert metrics.total_tasks == 2
    # task_001: 1 success of 2 trials -> pass^1 = C(1,1)/C(2,1) = 0.5
    # task_002: 0 successes -> 0.0; mean = 0.25
    assert metrics.pass_hat_ks[1] == pytest.approx(0.25)
