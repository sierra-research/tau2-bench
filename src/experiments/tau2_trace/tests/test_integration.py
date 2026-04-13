"""
End-to-end integration test for tau2-TRACE.

Constructs a realistic Results object with multiple simulations across
domains and verifies the full pipeline: analyse → merge → augmented DataFrame.
"""

import tempfile
from pathlib import Path

import pandas as pd

from experiments.tau2_trace.domain_router import evaluate_results_trace
from experiments.tau2_trace.run_experiment import analyze_results
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Task, UserScenario
from tau2.environment.environment import EnvironmentInfo


def _build_telecom_simulation(task_id: str, trial: int) -> SimulationRun:
    """Build a realistic telecom troubleshooting simulation."""
    messages = [
        SystemMessage(role="system", content="You are a telecom support agent."),
        AssistantMessage(role="assistant", content="Hello! How can I help you today?"),
        UserMessage(role="user", content="I have no service on my phone."),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="get_customer_by_phone",
                    arguments={"phone": "555-1234"},
                )
            ],
        ),
        ToolMessage(
            id="tc1",
            role="tool",
            content='{"customer_id": "C100", "name": "John Doe", "phone": "555-1234"}',
        ),
        AssistantMessage(
            role="assistant",
            content="I found your account, John. Could you please check your status bar and tell me if you see 'No Service' or 'Airplane Mode' displayed?",
        ),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="check_status_bar",
                    arguments={},
                    requestor="user",
                )
            ],
        ),
        ToolMessage(id="tc2", role="tool", content="No Service", requestor="user"),
        UserMessage(role="user", content="It says No Service."),
        AssistantMessage(
            role="assistant",
            content="Please toggle your airplane mode on and then off again. This will reset your network connection.",
        ),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="tc3",
                    name="toggle_airplane_mode",
                    arguments={},
                    requestor="user",
                )
            ],
        ),
        ToolMessage(
            id="tc3", role="tool", content="Airplane mode: ON", requestor="user"
        ),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="tc4",
                    name="toggle_airplane_mode",
                    arguments={},
                    requestor="user",
                )
            ],
        ),
        ToolMessage(
            id="tc4", role="tool", content="Airplane mode: OFF", requestor="user"
        ),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="tc5",
                    name="check_status_bar",
                    arguments={},
                    requestor="user",
                )
            ],
        ),
        ToolMessage(
            id="tc5", role="tool", content="Signal: Full bars", requestor="user"
        ),
        UserMessage(role="user", content="It's working now! I have full signal."),
        AssistantMessage(
            role="assistant",
            content="I'm glad your service is restored. Is there anything else I can help with?",
        ),
        UserMessage(role="user", content="No, that's all. Thanks! ###STOP###"),
    ]

    return SimulationRun(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        start_time="2026-03-07T10:00:00",
        end_time="2026-03-07T10:02:30",
        duration=150.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        trial=trial,
        agent_cost=0.03,
        user_cost=0.02,
        reward_info=RewardInfo(reward=1.0),
    )


def _build_retail_simulation(task_id: str, trial: int) -> SimulationRun:
    """Build a retail order cancellation simulation."""
    messages = [
        SystemMessage(role="system", content="You are a retail support agent."),
        AssistantMessage(role="assistant", content="Hi! How can I help you?"),
        UserMessage(role="user", content="I want to cancel my order."),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="find_user_id_by_email",
                    arguments={"email": "john@example.com"},
                )
            ],
        ),
        ToolMessage(id="tc1", role="tool", content='{"user_id": "U100"}'),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="tc2",
                    name="get_order_details",
                    arguments={"order_id": "O200"},
                )
            ],
        ),
        ToolMessage(
            id="tc2",
            role="tool",
            content='{"order_id": "O200", "status": "pending", "items": ["Widget A"]}',
        ),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="tc3",
                    name="cancel_pending_order",
                    arguments={"order_id": "O200", "reason": "customer_request"},
                )
            ],
        ),
        ToolMessage(id="tc3", role="tool", content='{"status": "cancelled"}'),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="tc4",
                    name="get_order_details",
                    arguments={"order_id": "O200"},
                )
            ],
        ),
        ToolMessage(
            id="tc4",
            role="tool",
            content='{"order_id": "O200", "status": "cancelled"}',
        ),
        AssistantMessage(
            role="assistant",
            content="Your order O200 has been cancelled. Is there anything else?",
        ),
        UserMessage(role="user", content="No thanks. ###STOP###"),
    ]

    return SimulationRun(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        start_time="2026-03-07T10:00:00",
        end_time="2026-03-07T10:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        trial=trial,
        agent_cost=0.02,
        reward_info=RewardInfo(reward=1.0),
    )


def _build_task(task_id: str, num_agent_actions: int = 2, num_user_actions: int = 0):
    """Build a minimal Task object with evaluation criteria."""
    return Task(
        id=task_id,
        description=None,
        user_scenario=UserScenario(
            persona=None,
            instructions="Test scenario",
        ),
    )


def _build_results(domain: str) -> Results:
    """Build a full Results object for a domain."""
    if domain == "telecom":
        sims = [
            _build_telecom_simulation("telecom_task_1", trial=0),
            _build_telecom_simulation("telecom_task_1", trial=1),
            _build_telecom_simulation("telecom_task_2", trial=0),
        ]
        tasks = [
            _build_task("telecom_task_1", num_agent_actions=1, num_user_actions=4),
            _build_task("telecom_task_2", num_agent_actions=1, num_user_actions=4),
        ]
    else:
        sims = [
            _build_retail_simulation("retail_task_1", trial=0),
            _build_retail_simulation("retail_task_1", trial=1),
        ]
        tasks = [
            _build_task("retail_task_1", num_agent_actions=3),
        ]

    return Results(
        info=Info(
            git_commit="test123",
            num_trials=2,
            max_steps=100,
            max_errors=10,
            user_info=UserInfo(implementation="user_simulator", llm="gpt-4.1"),
            agent_info=AgentInfo(implementation="llm_agent", llm="gpt-4.1"),
            environment_info=EnvironmentInfo(domain_name=domain, policy="Test policy"),
        ),
        tasks=tasks,
        simulations=sims,
    )


class TestEndToEndPipeline:
    """Full pipeline: Results → tau2-TRACE → augmented DataFrame."""

    def test_telecom_pipeline(self):
        results = _build_results("telecom")
        scorecards = evaluate_results_trace(results.simulations, domain="telecom")

        assert len(scorecards) == 3
        for sc in scorecards:
            assert sc.domain == "telecom"
            assert sc.trajectory is not None
            assert sc.ordering is not None
            assert sc.interaction is not None

            d = sc.to_dict()
            assert d["trace_total_tool_calls"] > 0
            assert d["trace_action_density"] > 0
            assert d["trace_matched_workflow"] is not None

        df_trace = pd.DataFrame([sc.to_dict() for sc in scorecards])
        assert len(df_trace) == 3
        assert "trace_total_tool_calls" in df_trace.columns
        assert "trace_policy_adherence" in df_trace.columns
        assert "trace_action_density" in df_trace.columns
        assert "trace_error_burst_count" in df_trace.columns
        assert "trace_orphan_tool_messages" in df_trace.columns

        trace_cols = [c for c in df_trace.columns if c.startswith("trace_")]
        for col in trace_cols:
            if col != "trace_matched_workflow":
                assert df_trace[col].notna().all(), f"{col} has NaN values"

    def test_retail_pipeline(self):
        results = _build_results("retail")
        scorecards = evaluate_results_trace(results.simulations, domain="retail")

        assert len(scorecards) == 2
        for sc in scorecards:
            assert sc.domain == "retail"
            assert sc.ordering.matched_workflow is None
            assert sc.ordering.read_after_write_score == 1.0

        df_trace = pd.DataFrame([sc.to_dict() for sc in scorecards])
        assert len(df_trace) == 2
        assert df_trace["trace_read_after_write_score"].mean() == 1.0

    def test_full_analyze_results_cli(self):
        """Test the CLI analyze_results function end-to-end with file I/O."""
        results = _build_results("telecom")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "test_results.json"
            results.save(results_path)

            output_path = Path(tmpdir) / "augmented.csv"
            df = analyze_results(
                results_path=results_path,
                output_path=output_path,
            )

            assert output_path.exists()

            assert len(df) == 3
            assert "reward" in df.columns
            assert "trace_total_tool_calls" in df.columns
            assert "trace_policy_adherence" in df.columns
            assert "trace_error_burst_count" in df.columns

            df_reload = pd.read_csv(output_path)
            assert len(df_reload) == 3
            assert "trace_action_density" in df_reload.columns

    def test_telecom_specific_metrics(self):
        """Verify telecom-specific metrics are computed correctly."""
        results = _build_results("telecom")
        scorecards = evaluate_results_trace(results.simulations, domain="telecom")

        sc = scorecards[0]

        assert sc.trajectory.agent_tool_call_count == 1
        assert sc.trajectory.user_tool_call_count == 4
        assert sc.trajectory.total_tool_call_count == 5
        assert sc.trajectory.redundant_tool_calls >= 0
        assert sc.trajectory.loop_count == 0
        assert sc.trajectory.error_count == 0
        assert sc.trajectory.orphan_tool_messages == 0

        assert sc.ordering.matched_workflow == "path1_no_service"
        assert sc.ordering.policy_adherence_score > 0.0

        assert sc.interaction.guidance_precision > 0.0
        assert sc.interaction.action_density > 0.0

    def test_retail_read_after_write(self):
        """Verify retail read-after-write detection."""
        results = _build_results("retail")
        scorecards = evaluate_results_trace(results.simulations, domain="retail")

        sc = scorecards[0]
        assert sc.ordering.write_calls_total == 1
        assert sc.ordering.write_calls_verified == 1
        assert sc.ordering.read_after_write_score == 1.0
