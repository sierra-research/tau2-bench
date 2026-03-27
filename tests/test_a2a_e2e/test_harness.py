"""Tests for the E2E test harness itself."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tests.test_a2a_e2e.harness import HarnessAgentExecutor, build_test_server


class TestHarnessAgentExecutor:
    """Verify the harness executor calls openai and enqueues a response."""

    def test_execute_enqueues_agent_message(self):
        """Test execute calls gpt-4o and enqueues an agent Message."""
        executor = HarnessAgentExecutor()

        context = MagicMock()
        context.get_user_input.return_value = "Hello"
        context.context_id = "ctx-1"
        context.task_id = "task-1"

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()
        event_queue.close = AsyncMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello! How can I help?"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            executor._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            asyncio.run(executor.execute(context, event_queue))

        event_queue.enqueue_event.assert_called_once()
        enqueued_msg = event_queue.enqueue_event.call_args[0][0]
        assert enqueued_msg.role.value == "agent"
        event_queue.close.assert_called_once()

    def test_cancel_closes_queue(self):
        """Test cancel closes the event queue."""
        executor = HarnessAgentExecutor()
        context = MagicMock()
        event_queue = MagicMock()
        event_queue.close = AsyncMock()

        asyncio.run(executor.cancel(context, event_queue))
        event_queue.close.assert_called_once()


class TestBuildTestServer:
    """Verify build_test_server returns a working FastAPI app."""

    def test_returns_fastapi_app(self):
        """Test the returned app has a router attribute."""
        app = build_test_server()
        assert hasattr(app, "router")

    def test_agent_card_route_exists(self):
        """Test the agent card well-known route is registered."""
        app = build_test_server()
        route_paths = [route.path for route in app.routes]
        assert "/.well-known/agent-card.json" in route_paths


class TestConftest:
    """Verify conftest fixtures are importable and well-formed."""

    def test_conftest_defines_expected_fixtures(self):
        """Test conftest defines expected fixture functions."""
        import tests.test_a2a_e2e.conftest as conftest

        assert hasattr(conftest, "a2a_e2e_endpoint")
        assert hasattr(conftest, "a2a_e2e_agent")
        assert hasattr(conftest, "pytest_addoption")
