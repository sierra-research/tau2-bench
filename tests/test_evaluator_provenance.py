"""Regression tests for evaluator state provenance."""

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import (
    Action,
    EvaluationCriteria,
    RewardType,
    Task,
    UserScenario,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


class _MockEnvironment:
    """Small deterministic environment for evaluator-only tests."""

    def __init__(self, **_kwargs):
        self.state = 0

    def set_state(
        self,
        initialization_data,
        initialization_actions,
        message_history,
        strict=True,
    ):
        del initialization_data, initialization_actions, strict
        for message in message_history:
            if isinstance(message, AssistantMessage):
                self.state += sum(
                    tool_call.name == "write"
                    for tool_call in (message.tool_calls or [])
                )

    def make_tool_call(self, tool_name, requestor="assistant", **_kwargs):
        del requestor
        if tool_name == "write":
            self.state += 1

    def _has_tool(self, tool_name):
        return tool_name == "write"

    def _is_mutating_tool(self, tool_name):
        return tool_name == "write"

    def get_db_hash(self):
        return str(self.state)

    def get_user_db_hash(self):
        return "user-state"

    def run_env_assertion(self, _assertion, raise_assertion_error=True):
        del raise_assertion_error
        return True


def _task() -> Task:
    return Task(
        id="mock/provenance",
        user_scenario=UserScenario(instructions="write one item"),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="reference-write",
                    name="write",
                    arguments={},
                )
            ],
            reward_basis=[RewardType.DB],
        ),
    )


def _trajectory():
    call = ToolCall(id="call-write", name="write", arguments={})
    return [
        AssistantMessage(role="assistant", tool_calls=[call]),
        ToolMessage(
            id=call.id,
            role="tool",
            requestor="assistant",
            content='{"ok": true}',
        ),
    ]


def test_replay_result_exposes_state_provenance():
    result = EnvironmentEvaluator.calculate_reward(
        environment_constructor=_MockEnvironment,
        task=_task(),
        full_trajectory=_trajectory(),
    )

    assert result.reward == 1.0
    assert result.evaluation_mode == "replay"
    assert result.state_source == "replayed"
    assert [(item.source, item.tool_call.name) for item in result.replayed_actions] == [
        ("trajectory", "write"),
        ("reference", "write"),
    ]
    assert any("does not certify live environment state" in w for w in result.warnings)


def test_live_result_does_not_inherit_replay_success():
    result = EnvironmentEvaluator.evaluate_live(
        environment=_MockEnvironment(),
        environment_constructor=_MockEnvironment,
        task=_task(),
    )

    assert result.reward == 0.0
    assert result.db_check.db_match is False
    assert result.evaluation_mode == "live"
    assert result.state_source == "live"
    assert [(item.source, item.tool_call.name) for item in result.replayed_actions] == [
        ("reference", "write")
    ]


def test_lenient_replay_emits_warning():
    result = EnvironmentEvaluator.calculate_reward(
        environment_constructor=_MockEnvironment,
        task=_task(),
        full_trajectory=_trajectory(),
        strict_replay=False,
    )

    assert any("strict_replay=False" in warning for warning in result.warnings)
