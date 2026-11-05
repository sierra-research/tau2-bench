"""ActionEvaluator matching: each gold action needs its own tool call."""

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import Action
from tau2.evaluator.evaluator_action import ActionEvaluator


def _gold(action_id: str, name: str, requestor: str = "assistant", **kwargs) -> Action:
    kwargs.setdefault("arguments", {})
    return Action(action_id=action_id, name=name, requestor=requestor, **kwargs)


def _agent_call(call_id: str, name: str, arguments: dict) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def _user_call(call_id: str, name: str) -> UserMessage:
    return UserMessage(
        role="user",
        tool_calls=[ToolCall(id=call_id, name=name, arguments={}, requestor="user")],
    )


def _matches(trajectory, gold) -> list[bool]:
    return [c.action_match for c in ActionEvaluator.evaluate_actions(trajectory, gold)]


REPEATED_GOLD = [
    *[
        _gold(f"req_{i}", "request_human_agent_transfer", "user", compare_args=[])
        for i in range(3)
    ],
    _gold(
        "transfer",
        "transfer_to_human_agents",
        arguments={"summary": ""},
        compare_args=[],
    ),
]
TRANSFER = _agent_call("t", "transfer_to_human_agents", {"summary": "x"})


def test_one_call_cannot_satisfy_a_repeated_gold_action():
    trajectory = [_user_call("u0", "request_human_agent_transfer"), TRANSFER]
    assert _matches(trajectory, REPEATED_GOLD) == [True, False, False, True]


def test_repeated_gold_action_matched_by_as_many_calls():
    trajectory = [
        *[_user_call(f"u{i}", "request_human_agent_transfer") for i in range(3)],
        TRANSFER,
    ]
    assert all(_matches(trajectory, REPEATED_GOLD))


def test_call_is_assigned_where_it_is_needed():
    # The specific call also satisfies the unconstrained gold action listed
    # first; it must still be credited to the specific one.
    gold = [
        _gold("any", "lookup", arguments={"id": "a"}, compare_args=[]),
        _gold("specific", "lookup", arguments={"id": "b"}, compare_args=["id"]),
    ]
    trajectory = [
        _agent_call("c0", "lookup", {"id": "b"}),
        _agent_call("c1", "lookup", {"id": "z"}),
    ]
    assert _matches(trajectory, gold) == [True, True]
