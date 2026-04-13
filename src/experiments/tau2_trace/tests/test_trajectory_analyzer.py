"""Tests for the trajectory analyzer."""

from experiments.tau2_trace.models import ToolCallRecord
from experiments.tau2_trace.trajectory_analyzer import (
    _compute_error_recovery,
    _count_redundant_calls,
    _detect_loops,
    _is_signature_error,
    analyze_trajectory,
    extract_tool_calls,
)
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason


def _make_sim(messages, task_id="test_1", trial=0, agent_cost=0.5) -> SimulationRun:
    return SimulationRun(
        id="sim-1",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
        trial=trial,
        agent_cost=agent_cost,
    )


class TestExtractToolCalls:
    def test_extracts_agent_tool_calls(self):
        messages = [
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="tc1", name="get_user", arguments={"id": "123"})
                ],
            ),
            ToolMessage(id="tc1", role="tool", content='{"name": "Alice"}'),
        ]
        records, orphans = extract_tool_calls(messages)
        assert len(records) == 1
        assert records[0].name == "get_user"
        assert records[0].requestor == "assistant"
        assert records[0].result_content == '{"name": "Alice"}'
        assert records[0].result_error is False
        assert orphans == 0

    def test_extracts_user_tool_calls(self):
        messages = [
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
            ToolMessage(id="tc2", role="tool", content="No Service"),
        ]
        records, orphans = extract_tool_calls(messages)
        assert len(records) == 1
        assert records[0].requestor == "user"
        assert orphans == 0

    def test_marks_errors(self):
        messages = [
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="tc3", name="bad_call", arguments={"x": 1})],
            ),
            ToolMessage(id="tc3", role="tool", content="Error", error=True),
        ]
        records, orphans = extract_tool_calls(messages)
        assert records[0].result_error is True
        assert orphans == 0

    def test_orphan_tool_message_counted(self):
        """Orphan ToolMessages (no matching pending call) are counted."""
        messages = [
            ToolMessage(id="no_match", role="tool", content="stray result"),
        ]
        records, orphans = extract_tool_calls(messages)
        assert len(records) == 0
        assert orphans == 1


class TestRedundantCalls:
    def test_no_redundancy(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", arguments={"x": 1}, requestor="assistant"
            ),
            ToolCallRecord(
                turn_index=1, name="tool_b", arguments={"y": 2}, requestor="assistant"
            ),
        ]
        assert _count_redundant_calls(records) == 0

    def test_detects_redundancy(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", arguments={"x": 1}, requestor="assistant"
            ),
            ToolCallRecord(
                turn_index=1, name="tool_a", arguments={"x": 1}, requestor="assistant"
            ),
            ToolCallRecord(
                turn_index=2, name="tool_a", arguments={"x": 1}, requestor="assistant"
            ),
        ]
        assert _count_redundant_calls(records) == 2

    def test_different_args_not_redundant(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", arguments={"x": 1}, requestor="assistant"
            ),
            ToolCallRecord(
                turn_index=1, name="tool_a", arguments={"x": 2}, requestor="assistant"
            ),
        ]
        assert _count_redundant_calls(records) == 0

    def test_nested_dict_args_hashed_correctly(self):
        """get_dict_hash handles nested dicts and key ordering."""
        records = [
            ToolCallRecord(
                turn_index=0,
                name="tool_a",
                arguments={"a": 1, "b": {"c": 2}},
                requestor="assistant",
            ),
            ToolCallRecord(
                turn_index=1,
                name="tool_a",
                arguments={"b": {"c": 2}, "a": 1},
                requestor="assistant",
            ),
        ]
        assert _count_redundant_calls(records) == 1


class TestLoopDetection:
    def test_no_loops(self):
        records = [
            ToolCallRecord(turn_index=i, name=f"tool_{i}", requestor="assistant")
            for i in range(6)
        ]
        assert _detect_loops(records) == 0

    def test_detects_loop(self):
        names = ["a", "b", "c", "a", "b", "c"]
        records = [
            ToolCallRecord(turn_index=i, name=n, requestor="assistant")
            for i, n in enumerate(names)
        ]
        assert _detect_loops(records, window=3) == 1


class TestSignatureErrorDetection:
    def test_unknown_tool_error(self):
        rec = ToolCallRecord(
            turn_index=0,
            name="bad_name",
            requestor="assistant",
            result_error=True,
            result_content="Error: unknown tool 'bad_name'",
        )
        assert _is_signature_error(rec) is True

    def test_value_error_not_signature(self):
        rec = ToolCallRecord(
            turn_index=0,
            name="get_user",
            requestor="assistant",
            result_error=True,
            result_content="Error: user_id 'X' not found in database",
        )
        assert _is_signature_error(rec) is False

    def test_no_error_not_signature(self):
        rec = ToolCallRecord(
            turn_index=0,
            name="get_user",
            requestor="assistant",
            result_error=False,
            result_content="OK",
        )
        assert _is_signature_error(rec) is False


class TestErrorRecovery:
    def test_full_recovery(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=1, name="tool_a", requestor="assistant", result_error=False
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 1
        assert recovered == 1
        assert rate == 1.0
        assert len(pairs) == 1
        assert pairs[0].failed.name == "tool_a"
        assert pairs[0].recovered.result_error is False

    def test_no_recovery(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=1, name="tool_b", requestor="assistant", result_error=False
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 1
        assert recovered == 0
        assert rate == 0.0
        assert len(pairs) == 0

    def test_no_errors(self):
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=False
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 0
        assert rate == 1.0
        assert len(pairs) == 0
        assert len(bursts) == 0

    def test_configurable_recovery_window(self):
        """Recovery must be within the configured window."""
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=1, name="tool_b", requestor="assistant", result_error=False
            ),
            ToolCallRecord(
                turn_index=2, name="tool_c", requestor="assistant", result_error=False
            ),
            ToolCallRecord(
                turn_index=3, name="tool_a", requestor="assistant", result_error=False
            ),
        ]
        # window=1: only looks at tool_b → no match
        _, recovered_1, _, _, _ = _compute_error_recovery(records, recovery_window=1)
        assert recovered_1 == 0

        # window=3: looks at tool_b, tool_c, tool_a → matches tool_a
        _, recovered_3, _, _, _ = _compute_error_recovery(records, recovery_window=3)
        assert recovered_3 == 1

    def test_error_burst_grouping(self):
        """Consecutive failures on the same tool form a single burst."""
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=1, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=2, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=3, name="tool_a", requestor="assistant", result_error=False
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 3
        assert recovered == 3
        assert rate == 1.0
        assert len(bursts) == 1
        assert bursts[0].count == 3
        assert bursts[0].recovered is True
        assert len(pairs) == 1

    def test_multiple_independent_bursts(self):
        """Different tools produce separate bursts."""
        records = [
            ToolCallRecord(
                turn_index=0, name="tool_a", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=1, name="tool_a", requestor="assistant", result_error=False
            ),
            ToolCallRecord(
                turn_index=2, name="tool_b", requestor="assistant", result_error=True
            ),
            ToolCallRecord(
                turn_index=3, name="tool_b", requestor="assistant", result_error=False
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 2
        assert recovered == 2
        assert len(bursts) == 2
        assert all(b.recovered for b in bursts)
        assert len(pairs) == 2

    def test_signature_error_recovers_from_different_name(self):
        """When error indicates wrong function name, recovery is allowed
        from any subsequent successful call."""
        records = [
            ToolCallRecord(
                turn_index=0,
                name="get_usr",
                requestor="assistant",
                result_error=True,
                result_content="Error: unknown tool 'get_usr'",
            ),
            ToolCallRecord(
                turn_index=1,
                name="get_user",
                requestor="assistant",
                result_error=False,
                result_content='{"id": "U1"}',
            ),
        ]
        errors, recovered, rate, pairs, bursts = _compute_error_recovery(records)
        assert errors == 1
        assert recovered == 1
        assert rate == 1.0
        assert pairs[0].failed.name == "get_usr"
        assert pairs[0].recovered.name == "get_user"


class TestAnalyzeTrajectory:
    def test_basic_analysis(self):
        messages = [
            AssistantMessage(role="assistant", content="How can I help?"),
            UserMessage(role="user", content="Fix my phone"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="get_customer_by_phone",
                        arguments={"phone": "555"},
                    )
                ],
            ),
            ToolMessage(id="t1", role="tool", content='{"id": "C1"}'),
            AssistantMessage(role="assistant", content="I found your account."),
        ]
        sim = _make_sim(messages)
        metrics, records = analyze_trajectory(sim, "telecom")

        assert metrics.task_id == "test_1"
        assert metrics.domain == "telecom"
        assert metrics.agent_message_count == 3
        assert metrics.user_message_count == 1
        assert metrics.agent_tool_call_count == 1
        assert metrics.total_tool_call_count == 1
        assert metrics.redundant_tool_calls == 0
        assert metrics.agent_cost == 0.5
        assert metrics.orphan_tool_messages == 0
        assert metrics.error_burst_count == 0

    def test_recovery_window_parameter(self):
        """The recovery_window parameter is threaded through."""
        messages = [
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="t1", name="tool_a", arguments={})],
            ),
            ToolMessage(id="t1", role="tool", content="Error", error=True),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="t2", name="tool_b", arguments={})],
            ),
            ToolMessage(id="t2", role="tool", content="OK"),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="t3", name="tool_c", arguments={})],
            ),
            ToolMessage(id="t3", role="tool", content="OK"),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="t4", name="tool_a", arguments={})],
            ),
            ToolMessage(id="t4", role="tool", content="OK"),
        ]
        sim = _make_sim(messages)

        # window=1: tool_b doesn't match tool_a
        m1, _ = analyze_trajectory(sim, "telecom", recovery_window=1)
        assert m1.errors_recovered == 0

        # window=3: tool_a at index 3 matches
        m3, _ = analyze_trajectory(sim, "telecom", recovery_window=3)
        assert m3.errors_recovered == 1
