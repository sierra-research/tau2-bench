"""Tests for the E2E test harness itself."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_a2a_e2e.harness import TestAgentExecutor, build_test_server


class TestTestAgentExecutor:
    """Verify the harness executor calls openai and enqueues a response."""

    def test_execute_enqueues_agent_message(self):
        """Execute should call gpt-4o and enqueue an agent Message."""
        executor = TestAgentExecutor()

        # Mock the RequestContext
        context = MagicMock()
        context.get_user_input.return_value = "Hello"
        context.context_id = "ctx-1"
        context.task_id = "task-1"

        # Mock the EventQueue
        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()
        event_queue.close = AsyncMock()

        # Mock openai response
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

        # Verify an event was enqueued
        event_queue.enqueue_event.assert_called_once()
        enqueued_msg = event_queue.enqueue_event.call_args[0][0]
        assert enqueued_msg.role.value == "agent"

        # Verify queue was closed
        event_queue.close.assert_called_once()

    def test_cancel_closes_queue(self):
        """Cancel should close the event queue."""
        executor = TestAgentExecutor()
        context = MagicMock()
        event_queue = MagicMock()
        event_queue.close = AsyncMock()

        asyncio.run(executor.cancel(context, event_queue))
        event_queue.close.assert_called_once()


class TestBuildTestServer:
    """Verify build_test_server returns a working FastAPI app."""

    def test_returns_fastapi_app(self):
        app = build_test_server()
        # FastAPI apps have a router attribute
        assert hasattr(app, "router")

    def test_agent_card_route_exists(self):
        app = build_test_server()
        route_paths = [route.path for route in app.routes]
        assert "/.well-known/agent-card.json" in route_paths


class TestConftest:
    """Verify conftest fixtures are importable and well-formed."""

    def test_conftest_defines_expected_fixtures(self):
        """conftest should define the expected fixture functions."""
        import tests.test_a2a_e2e.conftest as conftest

        assert hasattr(conftest, "a2a_e2e_endpoint")
        assert hasattr(conftest, "a2a_e2e_agent")
        assert hasattr(conftest, "pytest_addoption")
