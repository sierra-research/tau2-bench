from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentify_tau_bench.white_agent.agent import (
    GeneralWhiteAgentExecutor,
    prepare_white_agent_card,
    start_white_agent,
)


class TestPrepareWhiteAgentCard:
    """Test the prepare_white_agent_card function."""

    def test_prepare_white_agent_card(self):
        """Test creating a white agent card."""
        url = "http://localhost:9000"
        card = prepare_white_agent_card(url)

        assert card.name == "file_agent"
        assert card.url == url
        assert card.version == "1.0.0"
        assert len(card.skills) == 1
        assert card.skills[0].name == "Task Fulfillment"

    def test_prepare_white_agent_card_different_url(self):
        """Test creating a card with different URL."""
        url = "http://example.com:8080"
        card = prepare_white_agent_card(url)

        assert card.url == url


class TestGeneralWhiteAgentExecutor:
    """Test the GeneralWhiteAgentExecutor class."""

    def test_executor_initialization(self):
        """Test executor initializes with empty message dict."""
        executor = GeneralWhiteAgentExecutor()

        assert executor.ctx_id_to_messages == {}

    @pytest.mark.asyncio
    async def test_execute_first_message(self):
        """Test executing with first message in new context."""
        executor = GeneralWhiteAgentExecutor()

        mock_context = MagicMock()
        mock_context.get_user_input.return_value = "Hello"
        mock_context.context_id = "ctx123"

        mock_event_queue = AsyncMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.model_dump.return_value = {"content": "Hi there!"}
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch(
            "agentify_tau_bench.white_agent.agent.completion"
        ) as mock_completion:
            mock_completion.return_value = mock_response

            await executor.execute(mock_context, mock_event_queue)

            # Verify message history was created
            assert "ctx123" in executor.ctx_id_to_messages
            messages = executor.ctx_id_to_messages["ctx123"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Hi there!"

            # Verify LLM was called
            mock_completion.assert_called_once()

            # Verify response was sent
            mock_event_queue.enqueue_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_multiple_messages(self):
        """Test executing multiple messages in same context."""
        executor = GeneralWhiteAgentExecutor()

        mock_context = MagicMock()
        mock_context.context_id = "ctx123"

        mock_event_queue = AsyncMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.model_dump.return_value = {"content": "Response"}
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch(
            "agentify_tau_bench.white_agent.agent.completion"
        ) as mock_completion:
            mock_completion.return_value = mock_response

            # First message
            mock_context.get_user_input.return_value = "First message"
            await executor.execute(mock_context, mock_event_queue)

            # Second message
            mock_context.get_user_input.return_value = "Second message"
            await executor.execute(mock_context, mock_event_queue)

            # Verify history accumulates
            messages = executor.ctx_id_to_messages["ctx123"]
            assert len(messages) == 4  # 2 user + 2 assistant
            assert messages[0]["content"] == "First message"
            assert messages[2]["content"] == "Second message"

    @pytest.mark.asyncio
    async def test_cancel_not_implemented(self):
        """Test that cancel raises NotImplementedError."""
        executor = GeneralWhiteAgentExecutor()

        with pytest.raises(NotImplementedError):
            await executor.cancel(MagicMock(), AsyncMock())


class TestStartWhiteAgent:
    """Test the start_white_agent function."""

    def test_start_white_agent_setup(self):
        """Test that start_white_agent sets up and runs the server."""
        with (
            patch("agentify_tau_bench.white_agent.agent.DefaultRequestHandler"),
            patch("agentify_tau_bench.white_agent.agent.A2AStarletteApplication"),
            patch("agentify_tau_bench.white_agent.agent.uvicorn.run") as mock_run,
        ):

            start_white_agent(
                agent_name="general_white_agent", host="localhost", port=9002
            )

            # Verify uvicorn.run was called
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 9002

    def test_start_white_agent_custom_port(self):
        """Test starting white agent on custom port."""
        with (
            patch("agentify_tau_bench.white_agent.agent.DefaultRequestHandler"),
            patch("agentify_tau_bench.white_agent.agent.A2AStarletteApplication"),
            patch("agentify_tau_bench.white_agent.agent.uvicorn.run") as mock_run,
        ):

            start_white_agent(host="0.0.0.0", port=8888)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8888
