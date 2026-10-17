"""Deterministic tool faults (Task.faults) and the tau2.plugins entry point.

A fault applies to the assistant's nth call to a tool, in the live run only:
set_state replays a trajectory without it, and the orchestrator installs the
task's faults after it has set the initial state.
"""

import sys
from typing import Callable

import pytest

import tau2.registry  # noqa: F401
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Task, ToolFault
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import DummyUser

# tau2 re-exports its registry instance as `tau2.registry`, which shadows the
# module of the same name; the plugin loader lives in the module.
tau2_registry = sys.modules["tau2.registry"]

CREATE = ToolCall(
    id="c1",
    name="create_task",
    arguments={"user_id": "user_1", "title": "Important Meeting"},
)


def _fault(kind: str, nth: int = 1) -> ToolFault:
    return ToolFault(tool_name=CREATE.name, kind=kind, nth=nth)


def _committed_hash(get_environment: Callable[..., Environment]) -> str:
    env = get_environment()
    env.get_response(CREATE)
    return env.get_db_hash()


def test_timeout_before_commit_leaves_the_database_unchanged(
    get_environment: Callable[..., Environment],
):
    env = get_environment()
    before = env.get_db_hash()
    env.set_faults([_fault("timeout_before_commit")])

    result = env.get_response(CREATE)

    assert result.error is True
    assert result.content == "Error: timeout"
    assert env.get_db_hash() == before


def test_timeout_after_commit_keeps_the_change(
    get_environment: Callable[..., Environment],
):
    env = get_environment()
    env.set_faults([_fault("timeout_after_commit")])

    result = env.get_response(CREATE)

    assert result.error is True
    assert result.content == "Error: timeout"
    assert env.get_db_hash() == _committed_hash(get_environment)


def test_only_the_scheduled_call_fails(get_environment: Callable[..., Environment]):
    env = get_environment()
    env.set_faults([_fault("timeout_before_commit", nth=2)])

    first = env.get_response(CREATE)
    second = env.get_response(CREATE.model_copy(update={"id": "c2"}))

    assert first.error is False
    assert second.error is True
    assert env.fault_call_counts == {CREATE.name: 2}


def test_set_faults_resets_the_call_counts(get_environment: Callable[..., Environment]):
    env = get_environment()
    env.set_faults([_fault("timeout_before_commit", nth=2)])
    env.get_response(CREATE)
    env.set_faults([_fault("timeout_before_commit", nth=2)])

    assert env.fault_call_counts == {}
    assert env.get_response(CREATE.model_copy(update={"id": "c2"})).error is False


def test_replay_never_applies_a_fault(get_environment: Callable[..., Environment]):
    recorded = get_environment().get_response(CREATE)
    env = get_environment()
    env.set_faults([_fault("timeout_before_commit")])

    env.set_state(
        initialization_data=None,
        initialization_actions=None,
        message_history=[
            AssistantMessage(role="assistant", content=None, tool_calls=[CREATE]),
            recorded,
        ],
    )

    assert env.get_db_hash() == _committed_hash(get_environment)
    assert env.fault_call_counts == {}


class _ScriptedSoloAgent(LLMSoloAgent):
    """Calls create_task once, then the stop tool; no model."""

    def generate_next_message(self, message, state):
        if message is not None:
            state.messages.append(message)
        if not any(isinstance(m, ToolMessage) for m in state.messages):
            reply = AssistantMessage(
                role="assistant", content=None, tool_calls=[CREATE]
            )
        else:
            reply = self._check_if_stop_toolcall(
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(id="stop", name=self.STOP_FUNCTION_NAME, arguments={})
                    ],
                )
            )
        state.messages.append(reply)
        return reply, state


@pytest.mark.parametrize(
    "kind, reward",
    [("timeout_after_commit", 1.0), ("timeout_before_commit", 0.0)],
)
def test_the_orchestrator_runs_the_task_faults_and_grades_the_live_database(
    get_environment: Callable[..., Environment], base_task: Task, kind: str, reward
):
    task = base_task.model_copy(update={"faults": [_fault(kind)]})
    environment = get_environment(solo_mode=True)
    orchestrator = Orchestrator(
        domain="mock",
        environment=environment,
        user=DummyUser(),
        agent=_ScriptedSoloAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            task=task,
            llm="unused",
        ),
        task=task,
        max_steps=10,
        solo_mode=True,
    )
    simulation = orchestrator.run()
    results = [m for m in simulation.messages if isinstance(m, ToolMessage)]

    assert results[0].content == "Error: timeout"
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=True,
        domain="mock",
    )
    assert reward_info.reward == reward
    assert reward_info.info["replay_divergence"] is True


class _EntryPoint:
    def __init__(self, name: str, plugin: Callable):
        self.name = name
        self._plugin = plugin

    def load(self) -> Callable:
        return self._plugin


def test_registry_plugins_register_task_sets(monkeypatch: pytest.MonkeyPatch):
    registry = tau2_registry.Registry()
    mock_tasks = tau2_registry.registry.get_tasks_loader("mock")

    def plugin(reg: tau2_registry.Registry) -> None:
        reg.register_tasks(mock_tasks, "mock_plugin_tasks")

    def broken(reg: tau2_registry.Registry) -> None:
        raise RuntimeError("cannot register")

    monkeypatch.setattr(
        tau2_registry,
        "entry_points",
        lambda group: [_EntryPoint("ok", plugin), _EntryPoint("broken", broken)]
        if group == tau2_registry.PLUGIN_GROUP
        else [],
    )

    assert tau2_registry.load_registry_plugins(registry) == ["ok"]
    assert "mock_plugin_tasks" in registry.get_task_sets()
