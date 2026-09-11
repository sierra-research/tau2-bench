import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.persona import PersonaConfig, Verbosity
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.build import build_text_orchestrator, build_user
from tau2.user.user_simulator import DummyUser, UserSimulator


@pytest.fixture
def user_instructions() -> str:
    return (
        "You are Mia Li. You want to fly from New York to Seattle on May 20 (one way)."
    )


@pytest.fixture
def bad_user_instructions() -> str:
    return "You are Mia Li. You want to fly from Chicago to San Francisco on May 19 (round trip)."


@pytest.fixture
def first_agent_message() -> AssistantMessage:
    return AssistantMessage(
        content="Hello, how can I help you today?", role="assistant"
    )


@pytest.fixture
def user_simulator(user_instructions: str) -> UserSimulator:
    return UserSimulator(llm="gpt-4o-mini", instructions=user_instructions)


def test_user_simulator(
    user_simulator: UserSimulator, first_agent_message: AssistantMessage
):
    user_state = user_simulator.get_init_state()
    assert user_state is not None
    user_msg, user_state = user_simulator.generate_next_message(
        first_agent_message, user_state
    )
    # Check the response is a user message
    assert isinstance(user_msg, UserMessage)
    # Check the state is updated
    assert user_state is not None
    # Check the messages are of the correct type
    assert isinstance(user_state.messages[0], AssistantMessage)
    assert user_state.messages[0].content == first_agent_message.content
    assert isinstance(user_state.messages[1], UserMessage)


def test_user_simulator_set_state(
    user_simulator: UserSimulator,
):
    user_simulator.get_init_state(
        message_history=[
            UserMessage(content="Hello, can you help me find a flight?", role="user"),
            AssistantMessage(
                content="Hello, I can help you find a flight.", role="assistant"
            ),
        ]
    )


def test_dummy_user_no_args():
    """DummyUser must be instantiable without arguments (solo-mode gym path)."""
    dummy = DummyUser()
    state = dummy.get_init_state()
    assert state.messages == []


def test_build_dummy_user(get_environment, base_task):
    user = build_user(
        "dummy_user",
        get_environment(solo_mode=True),
        base_task,
        llm="unused-user-model",
        llm_args={"temperature": 0.7},
        persona_config=PersonaConfig(verbosity=Verbosity.MINIMAL),
        solo_mode=True,
    )
    assert isinstance(user, DummyUser)
    assert user.llm == "dummy"
    assert user.llm_args == {}
    assert user.get_init_state().messages == []


def test_build_dummy_user_requires_solo_mode(get_environment, base_task):
    with pytest.raises(AssertionError, match="only be used with solo agent"):
        build_user("dummy_user", get_environment(), base_task)


def test_build_regular_user_preserves_configuration(
    get_environment, task_with_user_tools
):
    environment = get_environment()
    persona = PersonaConfig(verbosity=Verbosity.MINIMAL)
    user = build_user(
        "user_simulator",
        environment,
        task_with_user_tools,
        llm="test-user-model",
        llm_args={"temperature": 0.7},
        persona_config=persona,
    )
    assert user.llm == "test-user-model"
    assert user.llm_args == {"temperature": 0.7}
    assert user.instructions == str(task_with_user_tools.user_scenario)
    assert user.persona_config == persona
    assert [tool.name for tool in user.tools] == [
        tool.name
        for tool in environment.get_user_tools(include=task_with_user_tools.user_tools)
    ]


def test_solo_runner_initializes_without_user_generation(base_task, monkeypatch):
    first_message = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="create-task",
                name="create_task",
                arguments={"user_id": "user_1", "title": "Important Meeting"},
                requestor="assistant",
            )
        ],
    )
    monkeypatch.setattr("tau2.agent.llm_agent.generate", lambda **kwargs: first_message)
    config = TextRunConfig(
        domain="mock",
        agent="llm_agent_solo",
        user="dummy_user",
        llm_agent="test-agent-model",
        llm_user="unused-user-model",
    )
    orchestrator = build_text_orchestrator(config, base_task)
    orchestrator.initialize()
    assert orchestrator.solo_mode
    assert isinstance(orchestrator.user, DummyUser)
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.user_state.messages == []
    assert orchestrator.message.tool_calls[0].name == "create_task"
