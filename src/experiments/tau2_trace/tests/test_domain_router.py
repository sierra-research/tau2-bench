"""Tests for the domain-aware metric router."""

from experiments.tau2_trace.domain_router import (
    evaluate_results_trace,
    evaluate_simulation_trace,
)
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason


def _make_sim(task_id="task_1", trial=0) -> SimulationRun:
    messages = [
        AssistantMessage(role="assistant", content="How can I help you today?"),
        UserMessage(role="user", content="I need help with my account"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="t1", name="get_customer_by_phone", arguments={"phone": "555"}
                )
            ],
        ),
        ToolMessage(id="t1", role="tool", content='{"id": "C1", "name": "Alice"}'),
        AssistantMessage(role="assistant", content="I found your account, Alice."),
        UserMessage(role="user", content="###STOP###"),
    ]
    return SimulationRun(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        trial=trial,
        agent_cost=0.05,
    )


class TestEvaluateSimulationTrace:
    def test_telecom_produces_all_metrics(self):
        sim = _make_sim()
        scorecard = evaluate_simulation_trace(sim, domain="telecom")

        assert scorecard.task_id == "task_1"
        assert scorecard.domain == "telecom"
        assert scorecard.trajectory is not None
        assert scorecard.ordering is not None
        assert scorecard.interaction is not None

    def test_retail_produces_all_metrics(self):
        sim = _make_sim()
        scorecard = evaluate_simulation_trace(sim, domain="retail")

        assert scorecard.trajectory is not None
        assert scorecard.ordering is not None
        assert scorecard.interaction is not None
        assert scorecard.ordering.matched_workflow is None

    def test_scorecard_to_dict(self):
        sim = _make_sim()
        scorecard = evaluate_simulation_trace(sim, domain="telecom")
        d = scorecard.to_dict()

        assert "task_id" in d
        assert "trace_total_turns" in d
        assert "trace_policy_adherence" in d
        assert "trace_action_density" in d
        assert "trace_error_burst_count" in d
        assert "trace_orphan_tool_messages" in d

    def test_recovery_window_threaded(self):
        """recovery_window parameter reaches trajectory analyzer."""
        sim = _make_sim()
        sc = evaluate_simulation_trace(sim, domain="telecom", recovery_window=5)
        assert sc.trajectory is not None


class TestEvaluateResultsTrace:
    def test_batch_evaluation(self):
        sims = [_make_sim(task_id="t1", trial=0), _make_sim(task_id="t2", trial=0)]
        scorecards = evaluate_results_trace(sims, domain="telecom")

        assert len(scorecards) == 2
        assert scorecards[0].task_id == "t1"
        assert scorecards[1].task_id == "t2"

    def test_with_expected_actions(self):
        sims = [_make_sim(task_id="t1", trial=0)]
        scorecards = evaluate_results_trace(
            sims, domain="telecom", task_expected_actions={"t1": 3}
        )

        assert scorecards[0].interaction.turns_vs_expected_ratio > 0.0
