from datetime import date

import pytest
from rich.console import Console

from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.simulation import (
    Results as TrajectoryResults,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.scripts.leaderboard.prepare_submission import (
    validate_submission,
    validate_submission_metrics,
)
from tau2.scripts.leaderboard.submission import (
    ContactInfo,
    DomainResults,
    Methodology,
    Results,
    Submission,
    SubmissionData,
)


def _make_trajectory_results() -> TrajectoryResults:
    tasks = [
        Task(
            id="task_1",
            user_scenario=UserScenario(instructions="Complete task 1"),
            evaluation_criteria=EvaluationCriteria(),
        ),
        Task(
            id="task_2",
            user_scenario=UserScenario(instructions="Complete task 2"),
            evaluation_criteria=EvaluationCriteria(),
        ),
    ]
    simulations = [
        SimulationRun(
            id="sim_task_1",
            task_id="task_1",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60.0,
            termination_reason=TerminationReason.USER_STOP,
            agent_cost=1.0,
            reward_info=RewardInfo(reward=1.0),
            messages=[],
            trial=0,
            seed=1,
        ),
        SimulationRun(
            id="sim_task_2",
            task_id="task_2",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60.0,
            termination_reason=TerminationReason.USER_STOP,
            agent_cost=3.0,
            reward_info=RewardInfo(reward=0.0),
            messages=[],
            trial=0,
            seed=2,
        ),
    ]
    return TrajectoryResults(
        info=Info(
            git_commit="abc123",
            num_trials=1,
            max_steps=100,
            max_errors=10,
            user_info=UserInfo(implementation="user_simulator", llm="user-model"),
            agent_info=AgentInfo(implementation="llm_agent", llm="agent-model"),
            environment_info=EnvironmentInfo(domain_name="retail", policy="policy"),
        ),
        tasks=tasks,
        simulations=simulations,
    )


def _make_submission(
    *,
    retail_results: DomainResults | None = None,
    model_name: str = "agent-model",
    user_simulator: str = "user-model",
) -> Submission:
    return Submission(
        model_name=model_name,
        model_organization="Org",
        submitting_organization="Org",
        submission_date=date(2026, 1, 1),
        contact_info=ContactInfo(email="test@example.com"),
        results=Results(retail=retail_results),
        methodology=Methodology(user_simulator=user_simulator),
    )


def test_validate_submission_metrics_accepts_matching_metrics():
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(retail_results=DomainResults(pass_1=50.0, cost=2.0))

    assert (
        validate_submission_metrics(
            submission, [trajectory_results], Console(record=True)
        )
        is True
    )


def test_validate_submission_metrics_rejects_pass_metric_mismatch():
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(retail_results=DomainResults(pass_1=100.0, cost=2.0))

    assert (
        validate_submission_metrics(
            submission, [trajectory_results], Console(record=True)
        )
        is False
    )


def test_validate_submission_metrics_rejects_cost_mismatch():
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(retail_results=DomainResults(pass_1=50.0, cost=99.0))

    assert (
        validate_submission_metrics(
            submission, [trajectory_results], Console(record=True)
        )
        is False
    )


def test_validate_submission_metrics_rejects_missing_domain_results():
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(retail_results=None)

    assert (
        validate_submission_metrics(
            submission, [trajectory_results], Console(record=True)
        )
        is False
    )


def test_validate_submission_metrics_keeps_metadata_mismatches_as_warnings():
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(
        retail_results=DomainResults(pass_1=50.0, cost=2.0),
        model_name="display-name",
        user_simulator="display-user",
    )

    assert (
        validate_submission_metrics(
            submission, [trajectory_results], Console(record=True)
        )
        is True
    )


def test_validate_submission_exits_nonzero_on_metric_mismatch(monkeypatch, tmp_path):
    trajectory_results = _make_trajectory_results()
    submission = _make_submission(retail_results=DomainResults(pass_1=100.0, cost=2.0))
    submission_data = SubmissionData(
        submission_dir=str(tmp_path),
        submission_file=str(tmp_path / "submission.json"),
        trajectory_files=[str(tmp_path / "trajectories" / "retail.json")],
        submission=submission,
        results=[trajectory_results],
    )

    monkeypatch.setattr(
        "tau2.scripts.leaderboard.prepare_submission.check_and_load_submission_data",
        lambda submission_dir: (True, "", submission_data),
    )
    monkeypatch.setattr(
        "tau2.scripts.leaderboard.prepare_submission.verify_trajectories",
        lambda paths, mode: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_submission(str(tmp_path))

    assert exc_info.value.code == 1
