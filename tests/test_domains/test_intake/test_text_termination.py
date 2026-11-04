"""Domain termination tools execute before finalization and do not affect defaults."""

import pytest

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.domains.intake.environment import get_environment, get_tasks
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator


def _end_call(*extra_calls):
    return AssistantMessage(
        role="assistant",
        cost=0,
        tool_calls=[
            ToolCall(id="stop", name="end_call", arguments={}, requestor="assistant"),
            *extra_calls,
        ],
    )


def test_default_text_agent_does_not_interpret_termination_tools():
    agent = LLMAgent(tools=[], domain_policy="", llm="test")
    assert not agent.is_stop(_end_call())
    assert not agent.is_stop(AssistantMessage(role="assistant", content="###STOP###"))


@pytest.mark.parametrize("bundled", [False, True])
def test_configured_end_call_finishes_without_pending_environment_step(
    monkeypatch, bundled
):
    task = get_tasks(strategy="scaffolded")[0]
    environment = get_environment(strategy="scaffolded", channel="text", task=task)
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="test",
        additional_stop_tool_names=frozenset({"end_call"}),
    )
    extra = (
        [
            ToolCall(
                id="other",
                name="get_today",
                arguments={},
                requestor="assistant",
            )
        ]
        if bundled
        else []
    )
    monkeypatch.setattr(
        agent, "_generate_next_message", lambda message, state: _end_call(*extra)
    )
    user = UserSimulator(llm="test", instructions="Answer the call.")
    monkeypatch.setattr(
        user,
        "generate_next_message",
        lambda message, state: (
            UserMessage(role="user", content="Hello?", cost=0),
            state,
        ),
    )
    result = Orchestrator(
        domain="intake",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=10,
    ).run()
    assert result.messages[0].content == task.agent_opener
    assert result.termination_reason == TerminationReason.AGENT_STOP
    results = [message for message in result.messages if message.role == "tool"]
    if bundled:
        assert len(results) == 2
        assert all(message.error for message in results)
        assert all(message.execution_status == "rejected" for message in results)
    else:
        assert len(results) == 1
        assert results[0].id == "stop"
