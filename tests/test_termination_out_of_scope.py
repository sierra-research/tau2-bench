"""Regression tests for out-of-scope user terminations (issue #517).

The user simulator emits ``###OUT-OF-SCOPE###`` when the scenario does not give
it enough information to continue. Such a run is an aborted simulation, not a
completed one, so it must not be recorded with the same termination reason as a
normal user-side end -- otherwise the "premature termination = 0 reward" rule in
``src/tau2/evaluator/AGENTS.md`` never fires for it and the abort can collect
partial credit from whichever assertions happen to already hold.

These tests pin the classification at every layer that can produce it:

* ``classify_user_stop`` maps each stop token to its termination reason;
* the half-duplex orchestrator records it both when resuming from a message
  history and when the stop is produced inside the step loop;
* the evaluator refuses to grade such a run;
* ``###STOP###`` and ``###TRANSFER###`` keep recording ``USER_STOP`` so the
  fix does not move any existing reward.
"""

import inspect
from typing import Callable

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import InitialState, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.scripts import view_simulations
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    classify_user_stop,
)


@pytest.fixture
def user_simulator() -> UserSimulator:
    return UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def agent(get_environment: Callable[[], Environment]):
    from tau2.agent.llm_agent import LLMAgent

    environment = get_environment()
    return LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


# --------------------------------------------------------------------------
# classify_user_stop
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [
        (STOP, TerminationReason.USER_STOP),
        (TRANSFER, TerminationReason.USER_STOP),
        (OUT_OF_SCOPE, TerminationReason.OUT_OF_SCOPE),
    ],
)
def test_classify_user_stop_maps_each_token(
    token: str, expected: TerminationReason
) -> None:
    """Each stop token maps to the reason that describes how the run ended."""
    message = UserMessage(role="user", content=f"Thanks. {token}")
    assert classify_user_stop(message) == expected


def test_classify_user_stop_returns_none_for_an_ordinary_message() -> None:
    """A message with no stop token is not a termination at all."""
    message = UserMessage(role="user", content="I would like to book a flight.")
    assert classify_user_stop(message) is None


def test_classify_user_stop_ignores_tool_calls_and_audio_only_chunks() -> None:
    """Mirrors the two guards in ``UserSimulator.is_stop``."""
    tool_call = UserMessage(
        role="user",
        content=f"{OUT_OF_SCOPE}",
        tool_calls=[ToolCall(id="1", name="search", arguments={})],
    )
    assert classify_user_stop(tool_call) is None

    audio_only = UserMessage(role="user", content=None)
    assert classify_user_stop(audio_only) is None


def test_out_of_scope_wins_when_several_tokens_are_present() -> None:
    """An abort stays an abort even if the simulator also emitted ``###STOP###``.

    Grading an aborted run is the failure mode this issue is about, so when the
    tokens disagree the classification must fall on the side that withholds the
    reward.
    """
    message = UserMessage(role="user", content=f"{STOP} {OUT_OF_SCOPE}")
    assert classify_user_stop(message) == TerminationReason.OUT_OF_SCOPE


def test_classify_user_stop_agrees_with_is_stop() -> None:
    """The two must never disagree about *whether* a message ends the run.

    ``is_stop`` decides that the simulation is over and ``classify_user_stop``
    decides how it is recorded. If a future token is added to one and not the
    other, a run would either stop with no reason or carry a reason without
    stopping.
    """
    for token in (STOP, TRANSFER, OUT_OF_SCOPE):
        message = UserMessage(role="user", content=f"prefix {token} suffix")
        assert UserSimulator.is_stop(message) is True
        assert classify_user_stop(message) is not None

    ordinary = UserMessage(role="user", content="no token here")
    assert UserSimulator.is_stop(ordinary) is False
    assert classify_user_stop(ordinary) is None


# --------------------------------------------------------------------------
# Orchestrator: the two places a user-side stop is recorded
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [
        (STOP, TerminationReason.USER_STOP),
        (TRANSFER, TerminationReason.USER_STOP),
        (OUT_OF_SCOPE, TerminationReason.OUT_OF_SCOPE),
    ],
)
def test_orchestrator_step_records_the_matched_token(
    token: str,
    expected: TerminationReason,
    domain_name: str,
    user_simulator: UserSimulator,
    agent,
    get_environment: Callable[[], Environment],
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step loop is the path a real run takes, and it was unconditionally
    recording ``USER_STOP``."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    stop_message = UserMessage(role="user", content=f"I have to go. {token}")

    def fake_generate_next_message(message, state):
        return stop_message, state

    monkeypatch.setattr(
        orchestrator.user, "generate_next_message", fake_generate_next_message
    )

    orchestrator.from_role = Role.AGENT
    orchestrator.to_role = Role.USER
    orchestrator.message = AssistantMessage(
        role="assistant", content="How can I help?", cost=0.0
    )
    orchestrator.done = False
    orchestrator.termination_reason = None

    orchestrator.step()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == expected


@pytest.mark.parametrize(
    "token, expected",
    [
        (STOP, TerminationReason.USER_STOP),
        (TRANSFER, TerminationReason.USER_STOP),
        (OUT_OF_SCOPE, TerminationReason.OUT_OF_SCOPE),
    ],
)
def test_orchestrator_initialize_records_the_matched_token(
    token: str,
    expected: TerminationReason,
    domain_name: str,
    user_simulator: UserSimulator,
    agent,
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    """Resuming from a history whose last message is a stop takes the other
    branch that records a user-side termination."""
    task = base_task.model_copy(deep=True)
    task.initial_state = InitialState(
        message_history=[
            AssistantMessage(role="assistant", content="How can I help?", cost=0.0),
            UserMessage(role="user", content=f"I have to go. {token}"),
        ]
    )

    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task,
    )
    orchestrator.initialize()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == expected


# --------------------------------------------------------------------------
# Evaluator: the rule this bug was bypassing
# --------------------------------------------------------------------------


def _simulation_with(
    reason: TerminationReason, base_task: Task, domain_name: str
) -> "object":
    from tau2.data_model.simulation import SimulationRun

    return SimulationRun(
        id="sim-517",
        task_id=base_task.id,
        start_time="2026-09-21T00:00:00",
        end_time="2026-09-21T00:00:10",
        duration=10.0,
        termination_reason=reason,
        messages=[
            AssistantMessage(role="assistant", content="How can I help?", cost=0.0),
            UserMessage(role="user", content="I have to go."),
        ],
    )


def test_evaluator_gives_zero_reward_to_an_out_of_scope_abort(
    base_task: Task, domain_name: str
) -> None:
    """This is the rule in ``evaluator/AGENTS.md`` that the old classification
    silently bypassed."""
    simulation = _simulation_with(
        TerminationReason.OUT_OF_SCOPE, base_task, domain_name
    )

    reward_info = evaluate_simulation(
        simulation=simulation,
        task=base_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain_name,
    )

    assert reward_info.reward == 0.0
    assert TerminationReason.OUT_OF_SCOPE.value in reward_info.info["note"]


def test_evaluator_still_grades_a_normal_user_stop(
    base_task: Task, domain_name: str
) -> None:
    """The fix must not turn ordinary user-side endings into premature ones.

    A ``USER_STOP`` run must still reach the evaluators, which is what keeps
    existing rewards unchanged.
    """
    simulation = _simulation_with(TerminationReason.USER_STOP, base_task, domain_name)

    reward_info = evaluate_simulation(
        simulation=simulation,
        task=base_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain_name,
    )

    note = (reward_info.info or {}).get("note", "")
    assert "terminated prematurely" not in note


# --------------------------------------------------------------------------
# Downstream reporting: the abort must stay visible
# --------------------------------------------------------------------------


def test_out_of_scope_is_counted_and_not_silently_dropped() -> None:
    """Splitting the reason out of ``USER_STOP`` must not make it disappear.

    ``compute_metrics`` counts termination reasons with an if/elif chain, so a
    new member that no branch mentions would be recorded nowhere -- trading the
    bug in the issue (an abort counted as a normal end) for a different silent
    gap (an abort counted as nothing at all).
    """
    from tau2.metrics.agent_metrics import AgentMetrics

    metrics = AgentMetrics(
        avg_reward=0.0,
        pass_hat_ks={1: 0.0},
        avg_agent_cost=0.0,
        termination_out_of_scope=2,
    )
    assert metrics.termination_out_of_scope == 2

    source = inspect.getsource(compute_metrics)
    assert "TerminationReason.OUT_OF_SCOPE" in source
    assert "termination_out_of_scope += 1" in source


def test_viewer_has_a_label_for_every_termination_reason() -> None:
    """``view_simulations`` falls back to ``?`` for unmapped reasons, so a new
    member silently renders as unknown unless it is added to both maps."""
    source = inspect.getsource(view_simulations)
    for reason in TerminationReason:
        if reason in {
            TerminationReason.TIMEOUT,
            TerminationReason.INFRASTRUCTURE_ERROR,
            TerminationReason.CONTEXT_WINDOW_EXCEEDED,
            TerminationReason.UNEXPECTED_ERROR,
        }:
            # Pre-existing gaps in the viewer maps; not introduced here.
            continue
        assert f"TerminationReason.{reason.name}" in source, reason.name
