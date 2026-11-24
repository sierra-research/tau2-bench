"""The airline DB check ignores the order of passengers and payments (issue #736)."""

import copy

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.evaluator.db_equivalence import db_check_hash
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


@pytest.fixture(scope="module")
def tasks():
    return {task.id: task for task in get_tasks("base")}


def reference_actions(task) -> list[dict]:
    return [
        {
            "id": action.action_id,
            "name": action.name,
            "requestor": action.requestor,
            "arguments": copy.deepcopy(action.arguments),
        }
        for action in task.evaluation_criteria.actions
    ]


def replay(actions: list[dict]):
    environment = get_environment()
    for action in actions:
        environment.make_tool_call(
            tool_name=action["name"],
            requestor=action["requestor"],
            **action["arguments"],
        )
    return environment


def edit_action(actions: list[dict], action_id: str, edit) -> list[dict]:
    actions = copy.deepcopy(actions)
    matches = [action for action in actions if action["id"] == action_id]
    assert len(matches) == 1
    edit(matches[0]["arguments"])
    return actions


def trajectory(actions: list[dict]) -> list:
    """Messages that make the agent call these tools, with the real tool outputs."""
    environment = get_environment()
    messages = []
    for index, action in enumerate(actions):
        call = ToolCall(
            id=f"call_{index}",
            name=action["name"],
            arguments=action["arguments"],
            requestor=action["requestor"],
        )
        messages.append(AssistantMessage(role="assistant", tool_calls=[call]))
        messages.append(environment.get_response(call))
    assert all(not m.error for m in messages if isinstance(m, ToolMessage)), (
        "a replayed tool call failed"
    )
    return messages


def test_db_check_passes_when_passengers_are_reordered(tasks):
    task = tasks["8"]
    actions = edit_action(
        reference_actions(task), "8_3", lambda args: args["passengers"].reverse()
    )

    reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory(actions),
    )

    assert reward.db_check.db_match


@pytest.mark.parametrize("task_id,action_id", [("14", "14_1"), ("20", "20_0")])
def test_payment_order_does_not_change_the_db_check_hash(tasks, task_id, action_id):
    actions = reference_actions(tasks[task_id])
    reordered = edit_action(
        actions, action_id, lambda args: args["payment_methods"].reverse()
    )

    assert db_check_hash(replay(reordered)) == db_check_hash(replay(actions))


def test_changed_passenger_still_fails_the_db_check(tasks):
    task = tasks["8"]

    def rename(args):
        args["passengers"][0]["first_name"] += "x"

    actions = edit_action(reference_actions(task), "8_3", rename)
    reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory(actions),
    )

    assert not reward.db_check.db_match


def test_flight_order_still_changes_the_db_check_hash():
    environment = get_environment()
    reservation = next(
        r
        for r in environment.tools.db.reservations.values()
        if len(r.flights) > 1 and len(r.passengers) > 1
    )
    original = db_check_hash(environment)

    reservation.passengers.reverse()
    assert db_check_hash(environment) == original

    reservation.flights.reverse()
    assert db_check_hash(environment) != original


def test_retail_db_check_hash_is_unchanged_and_order_sensitive():
    environment = get_retail_environment()
    order = next(
        o for o in environment.tools.db.orders.values() if len(o.payment_history) > 1
    )

    assert db_check_hash(environment) == environment.get_db_hash()
    original = db_check_hash(environment)
    order.payment_history.reverse()
    assert db_check_hash(environment) != original
