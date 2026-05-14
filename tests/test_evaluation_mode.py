from pathlib import Path

import pytest

from tau2.data_model.evaluation import (
    ActionEvaluation,
    CheckResult,
    DBEvaluation,
    EvaluatedSimulation,
    EvaluationOutcome,
    EvaluationReport,
)
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import (
    PostEvaluationMode,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator import (
    EvaluationType,
    compute_evaluation_outcome,
)
from tau2.runner.batch import (
    run_domain,
    run_single_task,
    run_single_task_evaluated,
    run_tasks,
    run_tasks_evaluated,
)
from tau2.runner.simulation import run_simulation_evaluated


def _make_base_simulation(task_id: str = "create_task_1") -> SimulationRun:
    return SimulationRun(
        id="sim-1",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:05",
        duration=5.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="create_task",
                        arguments={"user_id": "user_1", "title": "Important Meeting"},
                    )
                ],
            )
        ],
        trial=0,
        seed=123,
    )


def _make_evaluated_simulation() -> EvaluatedSimulation:
    simulation = _make_base_simulation()
    report = EvaluationReport(
        domain="mock",
        task_id=simulation.task_id,
        simulation_id=simulation.id,
        termination_reason=simulation.termination_reason,
        mode="half_duplex",
        evaluation_type=EvaluationType.ACTION.value,
        reward_basis=[RewardType.DB],
        action_checks=[
            ActionEvaluation(
                action={
                    "action_id": "create_1",
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                    "requestor": "assistant",
                },
                action_match=True,
            )
        ],
    )
    outcome = EvaluationOutcome(
        score_policy="evaluation_mean_v1",
        overall_score=1.0,
        component_scores=[
            CheckResult(
                name="action",
                score=1.0,
                passed_count=1,
                total_count=1,
                passed=True,
            )
        ],
    )
    return EvaluatedSimulation(
        simulation=simulation,
        evaluation_report=report,
        evaluation_outcome=outcome,
    )


def test_compute_evaluation_outcome_mean_ignores_reward_basis():
    report = EvaluationReport(
        domain="mock",
        task_id="task-1",
        simulation_id="sim-1",
        termination_reason=TerminationReason.AGENT_STOP,
        mode="half_duplex",
        evaluation_type=EvaluationType.ALL.value,
        reward_basis=[RewardType.DB],
        db_check=DBEvaluation(db_match=False),
        action_checks=[
            ActionEvaluation(
                action={
                    "action_id": "a1",
                    "name": "create_task",
                    "arguments": {},
                    "requestor": "assistant",
                },
                action_match=True,
            ),
            ActionEvaluation(
                action={
                    "action_id": "a2",
                    "name": "create_task",
                    "arguments": {},
                    "requestor": "assistant",
                },
                action_match=False,
            ),
        ],
    )

    mean_outcome = compute_evaluation_outcome(
        report,
        score_policy="evaluation_mean_v1",
    )
    compatible_outcome = compute_evaluation_outcome(
        report,
        score_policy="tau2_reward_compatible",
    )

    assert mean_outcome.overall_score == pytest.approx(0.25)
    assert compatible_outcome.overall_score == 0.0


def test_run_simulation_evaluated_returns_report_without_reward_info(base_task):
    class FakeEnvironment:
        def get_policy(self):
            return "mock policy"

        def get_domain_name(self):
            return "mock"

    class FakeOrchestrator:
        def __init__(self, task):
            self.task = task
            self.environment = FakeEnvironment()
            self.solo_mode = False

        def run(self):
            return _make_base_simulation(task_id=self.task.id)

    evaluated = run_simulation_evaluated(
        FakeOrchestrator(base_task),
        evaluation_type=EvaluationType.ACTION,
        score_policy="evaluation_mean_v1",
    )

    assert evaluated.simulation.reward_info is None
    assert evaluated.evaluation_report.task_id == base_task.id
    assert evaluated.evaluation_outcome.overall_score == 1.0


def test_benchmark_entrypoints_reject_evaluation_only_mode(base_task):
    config = TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        llm_agent="gpt-4.1",
        llm_args_agent={},
        llm_user="gpt-4.1",
        llm_args_user={},
        post_evaluation_mode=PostEvaluationMode.EVALUATION_ONLY,
        num_tasks=1,
    )

    with pytest.raises(ValueError, match="post_evaluation_mode='benchmark'"):
        run_single_task(config, base_task)

    with pytest.raises(ValueError, match="post_evaluation_mode='benchmark'"):
        run_tasks(config, [base_task], save_path=None)

    with pytest.raises(ValueError, match="post_evaluation_mode='benchmark'"):
        run_domain(config)


def test_run_single_task_evaluated_threads_agent_factory_override(
    monkeypatch: pytest.MonkeyPatch,
    base_task,
):
    config = TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        llm_agent="gpt-4.1",
        llm_args_agent={},
        llm_user="gpt-4.1",
        llm_args_user={},
        post_evaluation_mode=PostEvaluationMode.EVALUATION_ONLY,
    )

    captured = {}
    evaluated_simulation = _make_evaluated_simulation()

    class FakeEnvironment:
        def get_policy(self):
            return "mock policy"

    class FakeOrchestrator:
        def __init__(self):
            self.environment = FakeEnvironment()

    def fake_build_orchestrator(*args, **kwargs):
        captured["agent_factory_override"] = kwargs.get("agent_factory_override")
        return FakeOrchestrator()

    def fake_run_simulation_evaluated(*args, **kwargs):
        return evaluated_simulation

    monkeypatch.setattr("tau2.runner.batch.build_orchestrator", fake_build_orchestrator)
    monkeypatch.setattr(
        "tau2.runner.batch.run_simulation_evaluated",
        fake_run_simulation_evaluated,
    )

    def override(**kwargs):
        return None

    result = run_single_task_evaluated(
        config,
        base_task,
        agent_factory_override=override,
    )

    assert result is evaluated_simulation
    assert captured["agent_factory_override"] is override


def test_run_tasks_evaluated_returns_evaluated_results(
    monkeypatch: pytest.MonkeyPatch,
    base_task,
    tmp_path: Path,
):
    config = TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        llm_agent="gpt-4.1",
        llm_args_agent={},
        llm_user="gpt-4.1",
        llm_args_user={},
        post_evaluation_mode=PostEvaluationMode.EVALUATION_ONLY,
        max_concurrency=1,
        num_trials=1,
        auto_resume=True,
    )

    evaluated_simulation = _make_evaluated_simulation()
    evaluated_simulation.simulation.task_id = base_task.id
    evaluated_simulation.evaluation_report.task_id = base_task.id

    monkeypatch.setattr(
        "tau2.runner.batch.run_single_task_evaluated",
        lambda *args, **kwargs: evaluated_simulation,
    )

    results = run_tasks_evaluated(
        config,
        [base_task],
        save_path=tmp_path / "results.json",
        console_display=False,
    )

    assert len(results.simulations) == 1
    assert results.simulations[0].evaluation_outcome.overall_score == 1.0
