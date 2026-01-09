"""Tests for UserGymEnv - playing as the user against an automated agent."""

import pytest

from tau2.data_model.message import UserMessage
from tau2.gym.gym_agent import UserGymEnv


def test_user_gym_env_init():
    """Test that UserGymEnv can be initialized."""
    env = UserGymEnv(
        domain="mock",
        task_id="create_task_1",
        agent_llm="mock_llm",
    )
    assert env.domain == "mock"
    assert env.task_id == "create_task_1"
    assert env.agent_llm == "mock_llm"


def test_user_gym_env_reset():
    """Test that UserGymEnv can be reset."""
    env = UserGymEnv(
        domain="mock",
        task_id="create_task_1",
        agent_llm="mock_llm",
    )

    # Reset should return observation and info
    observation, info = env.reset()

    # Check that we got an observation (agent's greeting)
    assert isinstance(observation, str)

    # Check that info contains expected keys
    assert "task" in info
    assert "agent_tools" in info
    assert "user_tools" in info
    assert "policy" in info


def test_user_gym_env_step():
    """Test that UserGymEnv can process a step."""
    env = UserGymEnv(
        domain="mock",
        task_id="create_task_1",
        agent_llm="mock_llm",
    )

    observation, info = env.reset()

    # Take a step as the user
    action = "I need help with something"
    result = env.step(action)

    # Should return observation, reward, terminated, truncated, info
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result

    assert isinstance(observation, str)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gym_user_properties():
    """Test GymUser properties."""
    from tau2.gym.gym_agent import GymUser

    user = GymUser(tools=None, instructions="Test instructions")

    # Check initial state
    assert user.observation == []
    assert not user.is_user_turn  # Initially not waiting for action

    # Check is_stop method
    msg = UserMessage(role="user", content="Hello")
    assert not GymUser.is_stop(msg)


def test_user_gym_env_no_solo_mode():
    """Test that UserGymEnv doesn't support solo mode."""
    env = UserGymEnv(
        domain="mock",
        task_id="create_task_1",
    )

    # Should always be False
    assert not env._get_orchestrator().solo_mode


def test_gym_user_stop_tokens():
    """Test that GymUser correctly detects stop tokens."""
    from tau2.data_model.message import UserMessage
    from tau2.gym.gym_agent import GymUser

    # Test STOP token
    msg_stop = UserMessage(role="user", content="Thanks, that's perfect! ###STOP###")
    assert GymUser.is_stop(msg_stop)

    # Test TRANSFER token
    msg_transfer = UserMessage(
        role="user", content="I need to speak with a human ###TRANSFER###"
    )
    assert GymUser.is_stop(msg_transfer)

    # Test OUT_OF_SCOPE token
    msg_oos = UserMessage(
        role="user", content="This is not what I need ###OUT-OF-SCOPE###"
    )
    assert GymUser.is_stop(msg_oos)

    # Test normal message (no stop)
    msg_normal = UserMessage(role="user", content="Hello, I need help")
    assert not GymUser.is_stop(msg_normal)

    # Test tool call (should not be stop)
    msg_tool = UserMessage(role="user", tool_calls=[])
    assert not GymUser.is_stop(msg_tool)

    # Test None content (should not be stop)
    msg_none = UserMessage(role="user", content=None)
    assert not GymUser.is_stop(msg_none)


def test_user_gym_env_tool_call_parsing():
    """Test that UserGymEnv correctly parses user tool calls with requestor='user'."""
    from tau2.utils.tools import parse_action_string

    # Test functional tool call is parsed as UserMessage with correct requestor
    action = "check_status_bar()"
    parsed = parse_action_string(action, requestor="user")

    assert isinstance(parsed, UserMessage)
    assert parsed.role == "user"
    assert parsed.tool_calls is not None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "check_status_bar"
    assert parsed.tool_calls[0].requestor == "user"

    # Test with arguments
    action_with_args = "set_user_location(abroad=True)"
    parsed_with_args = parse_action_string(action_with_args, requestor="user")

    assert isinstance(parsed_with_args, UserMessage)
    assert parsed_with_args.tool_calls is not None
    assert len(parsed_with_args.tool_calls) == 1
    assert parsed_with_args.tool_calls[0].name == "set_user_location"
    assert parsed_with_args.tool_calls[0].arguments == {"abroad": True}
    assert parsed_with_args.tool_calls[0].requestor == "user"
