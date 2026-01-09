import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Message, SendMessageResponse, SendMessageSuccessResponse
from agentify_tau_bench.green_agent.agent import (
    RESPOND_ACTION_NAME,
    TauGreenAgentExecutor,
    ask_agent_to_solve,
    load_agent_card_toml,
    start_green_agent,
    tools_to_str,
)


class TestLoadAgentCardToml:
    """Test loading agent card from TOML file."""

    def test_load_agent_card_toml(self):
        """Test loading the tau_green_agent TOML configuration."""
        result = load_agent_card_toml("tau_green_agent")

        assert isinstance(result, dict)
        assert "name" in result
        assert "description" in result


class TestToolsToStr:
    """Test converting tools to string format."""

    def test_tools_to_str_empty(self):
        """Test converting empty tools list to string format."""
        result = tools_to_str([])
        assert result == "[]"

    def test_tools_to_str_tools(self):
        """Test converting tools to string format."""
        mock_tool1 = MagicMock()
        mock_tool1.openai_schema = {"name": "tool1", "description": "First tool"}

        mock_tool2 = MagicMock()
        mock_tool2.openai_schema = {"name": "tool2", "description": "Second tool"}

        result = tools_to_str([mock_tool1, mock_tool2])
        parsed = json.loads(result)

        assert len(parsed) == 2
        assert parsed[0]["name"] == "tool1"
        assert parsed[1]["name"] == "tool2"


class TestAskAgentToSolve:
    """Test the ask_agent_to_solve function."""

    @pytest.mark.asyncio
    async def test_ask_agent_to_solve_basic(self):
        """Test basic agent interaction flow."""
        # Mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = (
            "User message",
            {"policy": "Be helpful", "tools": []},
        )
        mock_env.step.return_value = (
            "Done",
            1.0,
            True,
            False,
            {"simulation_run": None, "reward_info": None},
        )

        # Mock white agent response
        mock_message = Message(
            role="agent", parts=[], message_id="msg1", context_id="ctx1"
        )
        mock_success = SendMessageSuccessResponse(result=mock_message)
        mock_response = SendMessageResponse(root=mock_success)

        with (
            patch("agentify_tau_bench.green_agent.agent.a2a_send_message") as mock_send,
            patch(
                "agentify_tau_bench.green_agent.agent.get_text_parts"
            ) as mock_get_text,
        ):

            mock_send.return_value = mock_response
            mock_get_text.return_value = [
                f'<json>{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": "Hello"}})}</json>'
            ]

            await ask_agent_to_solve("http://white-agent", mock_env)

            # Verify basic interactions
            mock_env.reset.assert_called_once()
            mock_env.step.assert_called_once()
            mock_send.assert_called_once()


class TestTauGreenAgentExecutor:
    """Test the TauGreenAgentExecutor class."""

    def test_executor_initialization(self):
        """Test executor can be initialized."""
        executor = TauGreenAgentExecutor()
        assert executor is not None

    @pytest.mark.asyncio
    async def test_execute_basic_flow(self):
        """Test that execute runs the basic flow."""
        executor = TauGreenAgentExecutor()

        mock_context = MagicMock()
        env_config = {"domain": "mock", "task_ids": ["task1"]}
        user_input = f"""
<white_agent_url>http://localhost:9000</white_agent_url>
<env_config>{json.dumps(env_config)}</env_config>
"""
        mock_context.get_user_input.return_value = user_input
        mock_event_queue = AsyncMock()

        with (
            patch("agentify_tau_bench.green_agent.agent.gym.make") as mock_gym_make,
            patch(
                "agentify_tau_bench.green_agent.agent.get_task_ids"
            ) as mock_get_task_ids,
            patch(
                "agentify_tau_bench.green_agent.agent.ask_agent_to_solve"
            ) as mock_ask,
        ):

            mock_env = MagicMock()
            mock_gym_make.return_value = mock_env
            mock_get_task_ids.return_value = ["task1"]

            # Mock successful simulation
            mock_simulation = MagicMock()
            mock_simulation.reward_info.reward = 1
            mock_ask.return_value = mock_simulation

            await executor.execute(mock_context, mock_event_queue)

            # Verify components were called
            mock_gym_make.assert_called_once()
            mock_ask.assert_called_once()
            mock_event_queue.enqueue_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_not_implemented(self):
        """Test that cancel raises NotImplementedError."""
        executor = TauGreenAgentExecutor()
        with pytest.raises(NotImplementedError):
            await executor.cancel(MagicMock(), AsyncMock())


class TestStartGreenAgent:
    """Test the start_green_agent function."""

    def test_start_green_agent_setup(self):
        """Test that start_green_agent sets up and runs the server."""
        with (
            patch(
                "agentify_tau_bench.green_agent.agent.load_agent_card_toml"
            ) as mock_load,
            patch("agentify_tau_bench.green_agent.agent.AgentCard"),
            patch("agentify_tau_bench.green_agent.agent.DefaultRequestHandler"),
            patch("agentify_tau_bench.green_agent.agent.A2AStarletteApplication"),
            patch("agentify_tau_bench.green_agent.agent.uvicorn.run") as mock_run,
        ):

            mock_load.return_value = {"name": "test_agent", "description": "Test"}

            start_green_agent(agent_name="tau_green_agent", host="localhost", port=9001)

            # Verify TOML was loaded
            mock_load.assert_called_once_with("tau_green_agent")

            # Verify uvicorn.run was called
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 9001
