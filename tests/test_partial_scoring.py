"""Unit tests for partial credit scoring in Action.compare_with_tool_call and ActionEvaluator."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Action
from tau2.evaluator.evaluator_action import ActionEvaluator


def make_action(name: str, arguments: dict, allow_partial_match: bool = False) -> Action:
    return Action(
        action_id="test_action",
        name=name,
        arguments=arguments,
        allow_partial_match=allow_partial_match,
    )


def make_tool_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="tc_1", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# compare_with_tool_call — exact mode (partial=False)
# ---------------------------------------------------------------------------


def test_exact_match_wrong_name():
    action = make_action("search_flights", {"origin": "NYC", "destination": "LAX"})
    tool_call = make_tool_call("get_weather", {"origin": "NYC", "destination": "LAX"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=False)
    assert matched is False
    assert reward == 0.0


def test_exact_match_correct_name_and_args():
    action = make_action("search_flights", {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"})
    tool_call = make_tool_call("search_flights", {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=False)
    assert matched is True
    assert reward == 1.0


def test_exact_match_correct_name_wrong_args():
    action = make_action("search_flights", {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"})
    tool_call = make_tool_call("search_flights", {"origin": "NYC", "destination": "LAX"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=False)
    assert matched is False
    assert reward == 0.0


# ---------------------------------------------------------------------------
# compare_with_tool_call — partial mode (partial=True)
# ---------------------------------------------------------------------------


def test_partial_wrong_name_gives_zero():
    action = make_action("search_flights", {"origin": "NYC"})
    tool_call = make_tool_call("get_weather", {"location": "NYC"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=True)
    assert matched is False
    assert reward == 0.0


def test_partial_exact_args_gives_full_reward():
    action = make_action(
        "search_flights",
        {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
    )
    tool_call = make_tool_call(
        "search_flights",
        {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
    )
    matched, reward = action.compare_with_tool_call(tool_call, partial=True)
    assert matched is True
    assert reward == 1.0


def test_partial_missing_arg_gives_half_reward():
    action = make_action(
        "search_flights",
        {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
    )
    # Missing 'date'
    tool_call = make_tool_call("search_flights", {"origin": "NYC", "destination": "LAX"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=True)
    assert matched is True
    assert reward == 0.5


def test_partial_wrong_arg_value_gives_half_reward():
    action = make_action("search_flights", {"origin": "NYC", "destination": "LAX"})
    tool_call = make_tool_call("search_flights", {"origin": "SFO", "destination": "LAX"})
    matched, reward = action.compare_with_tool_call(tool_call, partial=True)
    assert matched is True
    assert reward == 0.5


def test_partial_no_args_gives_full_reward():
    action = make_action("transfer_to_human_agents", {})
    tool_call = make_tool_call("transfer_to_human_agents", {})
    matched, reward = action.compare_with_tool_call(tool_call, partial=True)
    assert matched is True
    assert reward == 1.0


# ---------------------------------------------------------------------------
# ActionEvaluator.calculate_reward — partial scores flow into headline reward
# ---------------------------------------------------------------------------


def _make_trajectory_with_tool_call(name: str, arguments: dict) -> list:
    tool_call = ToolCall(id="tc_1", name=name, arguments=arguments)
    return [AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])]


def test_evaluate_actions_full_credit():
    """Perfect tool call with allow_partial_match → action_reward 1.0"""
    action = Action(
        action_id="a1",
        name="search_flights",
        arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
        allow_partial_match=True,
    )
    trajectory = _make_trajectory_with_tool_call(
        "search_flights", {"origin": "NYC", "destination": "LAX", "date": "2024-03-15"}
    )
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert len(checks) == 1
    assert checks[0].action_match is True
    assert checks[0].action_reward == 1.0


def test_evaluate_actions_partial_credit_flows_to_reward():
    """Right function but missing arg with allow_partial_match → action_reward 0.5 (not 0.0)"""
    action = Action(
        action_id="a1",
        name="search_flights",
        arguments={"origin": "NYC", "destination": "LAX", "date": "2024-03-15"},
        allow_partial_match=True,
    )
    trajectory = _make_trajectory_with_tool_call(
        "search_flights", {"origin": "NYC", "destination": "LAX"}  # missing date
    )
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert len(checks) == 1
    assert checks[0].action_match is True
    assert checks[0].action_reward == 0.5


def test_evaluate_actions_zero_for_wrong_function():
    """Wrong function name with allow_partial_match → action_reward 0.0"""
    action = Action(
        action_id="a1",
        name="search_flights",
        arguments={"origin": "NYC"},
        allow_partial_match=True,
    )
    trajectory = _make_trajectory_with_tool_call("get_weather", {"location": "NYC"})
    checks = ActionEvaluator.evaluate_actions(trajectory, [action])
    assert len(checks) == 1
    assert checks[0].action_match is False
    assert checks[0].action_reward == 0.0


def test_evaluate_actions_averages_multiple_actions():
    """Two actions: one perfect (1.0), one partial (0.5) → per-action rewards [1.0, 0.5]"""
    actions = [
        Action(
            action_id="a1",
            name="search_flights",
            arguments={"origin": "NYC", "destination": "LAX"},
            allow_partial_match=True,
        ),
        Action(
            action_id="a2",
            name="book_flight",
            arguments={"flight_id": "F123", "seat": "12A"},
            allow_partial_match=True,
        ),
    ]
    tc1 = ToolCall(id="tc_1", name="search_flights", arguments={"origin": "NYC", "destination": "LAX"})
    tc2 = ToolCall(id="tc_2", name="book_flight", arguments={"flight_id": "F123"})  # missing seat
    trajectory = [AssistantMessage(role="assistant", content=None, tool_calls=[tc1, tc2])]

    checks = ActionEvaluator.evaluate_actions(trajectory, actions)
    assert len(checks) == 2
    assert checks[0].action_reward == 1.0
    assert checks[1].action_reward == 0.5
    # Average flows to headline reward
    avg = sum(c.action_reward for c in checks) / len(checks)
    assert avg == pytest.approx(0.75)
