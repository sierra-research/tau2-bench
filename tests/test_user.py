import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
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


@pytest.mark.parametrize("empty_content", [None, "", "   \n"])
def test_empty_completion_is_retried_with_a_reminder(
    monkeypatch,
    user_simulator: UserSimulator,
    first_agent_message: AssistantMessage,
    empty_content,
):
    """An empty completion (no content, no tool calls) is retried once with an
    explicit reminder in the system prompt, instead of crashing the episode (#470)."""
    prompts = []
    responses = [
        AssistantMessage(role="assistant", content=empty_content),
        AssistantMessage(role="assistant", content="Where is my deposit?"),
    ]

    def fake_generate(model, messages, tools=None, call_name=None, **kwargs):
        prompts.append(messages)
        return responses[len(prompts) - 1]

    monkeypatch.setattr("tau2.user.user_simulator.generate", fake_generate)
    state = user_simulator.get_init_state()
    user_msg, state = user_simulator.generate_next_message(first_agent_message, state)

    assert user_msg.content == "Where is my deposit?"
    assert len(prompts) == 2
    # The retry carries the reminder in the system prompt; the first call does not.
    assert "your previous reply was empty" in prompts[1][0].content
    assert "your previous reply was empty" not in prompts[0][0].content


def test_empty_completion_twice_raises_a_clear_error(
    monkeypatch,
    user_simulator: UserSimulator,
    first_agent_message: AssistantMessage,
):
    """Two empty completions in a row raise a descriptive error rather than letting
    message validation fail deeper in the orchestrator."""

    def fake_generate(model, messages, tools=None, call_name=None, **kwargs):
        return AssistantMessage(role="assistant", content=None)

    monkeypatch.setattr("tau2.user.user_simulator.generate", fake_generate)
    state = user_simulator.get_init_state()
    with pytest.raises(ValueError, match="empty completion twice"):
        user_simulator.generate_next_message(first_agent_message, state)
