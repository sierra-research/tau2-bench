"""Tests for SOP-as-DAG policy adherence verification."""


from tau2_reliability.analysis.policy_adherence import (
    AIRLINE_POLICY,
    RETAIL_POLICY,
    TELECOM_POLICY,
    check_trace_adherence,
    compute_policy_adherence,
    get_policy_graph,
)
from tau2_reliability.models import PolicyViolationType, TaskTrialData


class TestPolicyGraph:
    def test_airline_graph_has_nodes(self):
        assert "authenticate" in AIRLINE_POLICY.nodes
        assert "gather_info" in AIRLINE_POLICY.nodes
        assert "execute" in AIRLINE_POLICY.nodes

    def test_tool_to_node_mapping(self):
        assert AIRLINE_POLICY.tool_to_node("get_user_details") == "authenticate"
        assert AIRLINE_POLICY.tool_to_node("cancel_reservation") == "execute"
        assert AIRLINE_POLICY.tool_to_node("get_reservation_details") == "gather_info"
        assert AIRLINE_POLICY.tool_to_node("nonexistent_tool") is None

    def test_topological_order(self):
        order = AIRLINE_POLICY.topological_order()
        assert order.index("authenticate") < order.index("gather_info")
        assert order.index("gather_info") < order.index("execute")

    def test_get_predecessors(self):
        preds = AIRLINE_POLICY.get_predecessors("execute")
        assert "gather_info" in preds

    def test_retail_graph(self):
        assert len(RETAIL_POLICY.nodes) == 4
        assert RETAIL_POLICY.tool_to_node("find_user_id_by_name_zip") == "authenticate"
        assert RETAIL_POLICY.tool_to_node("find_user_id_by_email") == "authenticate"
        assert RETAIL_POLICY.tool_to_node("get_user_details") == "gather_info"
        assert RETAIL_POLICY.tool_to_node("get_order_details") == "gather_info"
        assert RETAIL_POLICY.tool_to_node("cancel_pending_order") == "execute"

    def test_telecom_graph(self):
        assert len(TELECOM_POLICY.nodes) == 4
        assert TELECOM_POLICY.tool_to_node("get_customer_by_name") == "authenticate"
        assert TELECOM_POLICY.tool_to_node("get_customer_by_phone") == "authenticate"

    def test_get_policy_graph(self):
        assert get_policy_graph("airline") is AIRLINE_POLICY
        assert get_policy_graph("retail") is RETAIL_POLICY
        assert get_policy_graph("telecom") is TELECOM_POLICY
        assert get_policy_graph("unknown_domain") is None
        assert get_policy_graph("telecom-workflow") is TELECOM_POLICY


class TestCheckTraceAdherence:
    def test_perfect_adherence(self):
        # authenticate -> gather_info -> execute (correct order)
        actions = ["get_user_details", "get_reservation_details", "cancel_reservation"]
        result = check_trace_adherence(actions, AIRLINE_POLICY)
        assert result.adherence_score > 0.8
        assert len(result.violations) == 0

    def test_skipped_authentication(self):
        # Jump straight to execute without authenticating
        actions = ["cancel_reservation"]
        result = check_trace_adherence(actions, AIRLINE_POLICY)
        skipped = [v for v in result.violations if v.violation_type == PolicyViolationType.SKIPPED_NODE]
        assert len(skipped) >= 1
        assert any("authenticate" in (v.expected_node or "") for v in skipped)

    def test_wrong_order(self):
        # execute before gather_info
        actions = ["cancel_reservation", "get_user_details", "get_reservation_details"]
        result = check_trace_adherence(actions, AIRLINE_POLICY)
        # Should have violations for wrong ordering
        assert result.adherence_score < 1.0

    def test_empty_trace(self):
        result = check_trace_adherence([], AIRLINE_POLICY)
        assert result.adherence_score < 0.5
        # All required nodes are skipped
        skipped = [v for v in result.violations if v.violation_type == PolicyViolationType.SKIPPED_NODE]
        assert len(skipped) >= 2

    def test_read_only_trace(self):
        # Only reads, no execution — misses execute phase
        actions = ["get_user_details", "get_reservation_details", "list_all_airports"]
        result = check_trace_adherence(actions, AIRLINE_POLICY)
        skipped = [v for v in result.violations if v.violation_type == PolicyViolationType.SKIPPED_NODE]
        assert any("execute" in (v.expected_node or "") for v in skipped)

    def test_unknown_tools_ignored(self):
        actions = ["unknown_tool", "get_user_details", "get_reservation_details", "cancel_reservation"]
        result = check_trace_adherence(actions, AIRLINE_POLICY)
        assert result.adherence_score > 0.8

    def test_retail_adherence(self):
        actions = ["find_user_id_by_name_zip", "get_user_details", "get_order_details", "cancel_pending_order"]
        result = check_trace_adherence(actions, RETAIL_POLICY)
        assert result.adherence_score > 0.8
        assert len(result.violations) == 0

    def test_adherence_score_bounded(self):
        for actions in [[], ["cancel_reservation"], ["get_user_details", "cancel_reservation"]]:
            result = check_trace_adherence(actions, AIRLINE_POLICY)
            assert 0.0 <= result.adherence_score <= 1.0


class TestComputePolicyAdherence:
    def test_multiple_tasks(self):
        td1 = TaskTrialData(
            task_id="t1", outcomes=[True],
            action_sequences=[["get_user_details", "get_reservation_details", "cancel_reservation"]],
            costs=[0.1], durations=[30], num_actions=[3],
        )
        td2 = TaskTrialData(
            task_id="t2", outcomes=[False],
            action_sequences=[["cancel_reservation"]],  # Skips authentication
            costs=[0.1], durations=[30], num_actions=[1],
        )
        results = compute_policy_adherence([td1, td2], "airline")
        assert len(results) == 2
        assert results[0].task_id == "t1"
        assert results[0].adherence_score > results[1].adherence_score

    def test_unknown_domain(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True],
            action_sequences=[["some_action"]],
            costs=[0.1], durations=[30], num_actions=[1],
        )
        results = compute_policy_adherence([td], "nonexistent")
        assert results == []

    def test_averages_across_trials(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True, False],
            action_sequences=[
                ["get_user_details", "get_reservation_details", "cancel_reservation"],  # Good
                ["cancel_reservation"],  # Bad — skips identification
            ],
            costs=[0.1, 0.1], durations=[30, 30], num_actions=[3, 1],
        )
        results = compute_policy_adherence([td], "airline")
        assert len(results) == 1
        # Averaged score should be between perfect and poor
        assert 0.3 < results[0].adherence_score < 0.95
