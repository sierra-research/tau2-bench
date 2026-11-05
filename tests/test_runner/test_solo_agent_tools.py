"""Solo agent factories must not add private tools to the caller's tool list."""

import pytest

from tau2.agent.llm_agent import create_llm_agent, create_llm_solo_agent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage


@pytest.mark.parametrize("empty_tools", [False, True])
def test_solo_agent_keeps_caller_tools_unchanged(
    get_environment, base_task, empty_tools
):
    environment = get_environment()
    tools = [] if empty_tools else environment.get_tools()
    original_tools = list(tools)

    agent = create_llm_solo_agent(
        tools, environment.get_policy(), task=base_task, llm="fixture"
    )

    assert tools == original_tools
    assert agent.tools[:-1] == original_tools
    assert agent.tools[-1].name == agent.STOP_FUNCTION_NAME
    assert agent.tools[-1]() == agent.STOP_TOKEN


def test_solo_agents_do_not_accumulate_stop_tools(get_environment, base_task):
    environment = get_environment()
    tools = environment.get_tools()
    first = create_llm_solo_agent(
        tools, environment.get_policy(), task=base_task, llm="fixture"
    )
    first_tools = list(first.tools)
    second = create_llm_solo_agent(
        tools, environment.get_policy(), task=base_task, llm="fixture"
    )

    assert first.tools == first_tools
    for agent in (first, second):
        assert [tool.name for tool in agent.tools].count(agent.STOP_FUNCTION_NAME) == 1
    assert first.tools[-1] is not second.tools[-1]


def test_solo_agent_does_not_expose_stop_tool_to_normal_agent(
    get_environment, base_task, monkeypatch
):
    environment = get_environment()
    tools = environment.get_tools()
    normal_agent = create_llm_agent(tools, environment.get_policy(), llm="fixture")
    solo_agent = create_llm_solo_agent(
        tools, environment.get_policy(), task=base_task, llm="fixture"
    )

    def generate(**kwargs):
        if kwargs["call_name"] == "agent_solo_response":
            return AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="done", name="done", arguments={})],
            )
        assert solo_agent.STOP_FUNCTION_NAME not in [
            tool.name for tool in kwargs["tools"]
        ]
        return AssistantMessage(role="assistant", content="How can I help?")

    monkeypatch.setattr("tau2.agent.llm_agent.generate", generate)
    normal_agent.generate_next_message(
        UserMessage(role="user", content="Hello"), normal_agent.get_init_state()
    )
    stop_message, _ = solo_agent.generate_next_message(
        None, solo_agent.get_init_state()
    )
    assert solo_agent.is_stop(stop_message)
    assert stop_message.tool_calls is None
