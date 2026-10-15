"""The DB check grades the database the agent left, not a replay of it.

A tool that commits and then reports an error, or reports an error without
committing, makes the replayed trajectory disagree with the live run. These
tests pin both directions, the unchanged path, and the behaviour for stored
trajectories that carry no live hash.
"""

from typing import Callable, Optional

import pytest

from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    ToolMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import DummyUser

CREATE = ToolCall(
    id="c1",
    name="create_task",
    arguments={"user_id": "user_1", "title": "Important Meeting"},
)
TIMEOUT = ToolMessage(id="c1", role="tool", content="Error: timeout", error=True)


def _trajectory(result: ToolMessage) -> list[Message]:
    return [
        AssistantMessage(role="assistant", content=None, tool_calls=[CREATE]),
        result,
        AssistantMessage(role="assistant", content="Done."),
    ]


def _live_hashes(
    get_environment: Callable[..., Environment], commit: bool
) -> tuple[Optional[str], Optional[str]]:
    """Hashes of a live environment where the create_task call did or did not commit."""
    live = get_environment()
    if commit:
        live.get_response(CREATE)
    return live.get_db_hash(), live.get_user_db_hash()


def test_timeout_after_commit_is_graded_on_the_live_database(
    get_environment: Callable[..., Environment], base_task: Task
):
    db_hash, user_db_hash = _live_hashes(get_environment, commit=True)

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=base_task,
        full_trajectory=_trajectory(TIMEOUT),
        live_db_hash=db_hash,
        live_user_db_hash=user_db_hash,
    )

    assert reward_info.reward == 1.0
    assert reward_info.db_check.db_match
    assert reward_info.info["replay_divergence"] is True
    assert reward_info.info["replay_error"] is not None


def test_timeout_before_commit_is_not_credited(
    get_environment: Callable[..., Environment], base_task: Task
):
    db_hash, user_db_hash = _live_hashes(get_environment, commit=False)

    for strict_replay in (True, False):
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=base_task,
            full_trajectory=_trajectory(TIMEOUT),
            strict_replay=strict_replay,
            live_db_hash=db_hash,
            live_user_db_hash=user_db_hash,
        )

        assert reward_info.reward == 0.0
        assert not reward_info.db_check.db_match
        assert reward_info.info["replay_divergence"] is True


def test_unfaulted_trajectory_reports_no_divergence(
    get_environment: Callable[..., Environment], base_task: Task
):
    live = get_environment()
    result = live.get_response(CREATE)

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=base_task,
        full_trajectory=_trajectory(result),
        live_db_hash=live.get_db_hash(),
        live_user_db_hash=live.get_user_db_hash(),
    )

    assert reward_info.reward == 1.0
    assert reward_info.info == {"replay_divergence": False, "replay_error": None}


def test_without_a_live_hash_strict_replay_still_raises(
    get_environment: Callable[..., Environment], base_task: Task
):
    """Stored trajectories without live_db_hash are graded exactly as before."""
    with pytest.raises(ValueError):
        EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=base_task,
            full_trajectory=_trajectory(TIMEOUT),
        )

    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=base_task,
        full_trajectory=_trajectory(TIMEOUT),
        strict_replay=False,
    )
    assert reward_info.info is None


def test_full_duplex_grades_the_live_database(
    get_environment: Callable[..., Environment], base_task: Task
):
    db_hash, user_db_hash = _live_hashes(get_environment, commit=False)
    ticks = [
        Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:00",
            agent_tool_calls=[CREATE],
            agent_tool_results=[TIMEOUT],
        )
    ]

    reward_info = FullDuplexEnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=base_task,
        full_trajectory=ticks,
        live_db_hash=db_hash,
        live_user_db_hash=user_db_hash,
    )

    assert reward_info.reward == 0.0
    assert reward_info.info["replay_divergence"] is True


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


def _run(get_environment, base_task, fault: Optional[str]):
    environment = get_environment(solo_mode=True)
    real_get_response = environment.get_response

    def timeout(call: ToolCall) -> ToolMessage:
        return ToolMessage(
            id=call.id,
            role="tool",
            content="Error: timeout",
            requestor=call.requestor,
            error=True,
        )

    def get_response(call: ToolCall) -> ToolMessage:
        if fault == "before_commit" and call.name == CREATE.name:
            return timeout(call)
        result = real_get_response(call)
        if fault == "after_commit" and call.name == CREATE.name:
            return timeout(call)
        return result

    environment.get_response = get_response
    orchestrator = Orchestrator(
        domain="mock",
        environment=environment,
        user=DummyUser(),
        agent=_ScriptedSoloAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            task=base_task,
            llm="unused",
        ),
        task=base_task,
        max_steps=10,
        solo_mode=True,
    )
    simulation = orchestrator.run()
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=base_task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=True,
        domain="mock",
    )
    return orchestrator, simulation, reward_info


@pytest.mark.parametrize(
    "fault, reward, divergence",
    [(None, 1.0, False), ("after_commit", 1.0, True), ("before_commit", 0.0, True)],
)
def test_orchestrator_records_the_live_hash_and_the_evaluator_uses_it(
    get_environment, base_task, fault, reward, divergence
):
    orchestrator, simulation, reward_info = _run(get_environment, base_task, fault)

    assert simulation.live_db_hash == orchestrator.environment.get_db_hash()
    assert simulation.live_user_db_hash == orchestrator.environment.get_user_db_hash()
    assert reward_info.reward == reward
    assert reward_info.info["replay_divergence"] is divergence
