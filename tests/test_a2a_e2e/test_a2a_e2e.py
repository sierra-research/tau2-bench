"""A2A end-to-end test scenarios.

These tests validate the full A2A agent pipeline against a real server.
Run with: pytest -m full_a2a_integration
"""

import asyncio

import httpx
import pytest

from a2a.client import A2ACardResolver
from a2a.types import AgentCard

from tau2.a2a.models import A2AAgentState
from tau2.agent.a2a_agent import A2AAgent
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

pytestmark = pytest.mark.full_a2a_integration


# ---------------------------------------------------------------------------
# Smoke Layer
# ---------------------------------------------------------------------------


class TestSmoke:
    """Basic connectivity and protocol compliance."""

    def test_agent_card_discovery(self, a2a_e2e_endpoint: str):
        """Fetch the agent card and validate its structure."""

        async def _fetch_card() -> AgentCard:
            async with httpx.AsyncClient(
                base_url=a2a_e2e_endpoint
            ) as client:
                resolver = A2ACardResolver(
                    httpx_client=client,
                    base_url=a2a_e2e_endpoint,
                )
                return await resolver.get_agent_card()

        card = asyncio.run(_fetch_card())

        assert isinstance(card, AgentCard)
        assert card.name
        assert len(card.skills) > 0
        assert card.version

    def test_single_message_round_trip(
        self, a2a_e2e_agent: A2AAgent
    ):
        """Send one message and verify we get a non-empty text response."""
        state = a2a_e2e_agent.get_init_state()
        user_msg = UserMessage(role="user", content="Hello")

        response, new_state = a2a_e2e_agent.generate_next_message(
            user_msg, state
        )

        assert isinstance(response, AssistantMessage)
        assert response.has_text_content()
        assert response.content  # non-empty
        assert not response.is_tool_call()


# ---------------------------------------------------------------------------
# Protocol Layer
# ---------------------------------------------------------------------------


class TestProtocol:
    """A2A protocol mechanics: context tracking and tool calls."""

    def test_context_persistence(self, a2a_e2e_agent: A2AAgent):
        """Context ID should be None initially, set after first call, stable after second."""
        state = a2a_e2e_agent.get_init_state()
        assert state.context_id is None

        msg1 = UserMessage(role="user", content="Hello")
        _, state = a2a_e2e_agent.generate_next_message(msg1, state)
        assert state.context_id is not None
        first_context_id = state.context_id

        msg2 = UserMessage(role="user", content="How are you?")
        _, state = a2a_e2e_agent.generate_next_message(msg2, state)
        assert state.context_id == first_context_id

    def test_tool_call_round_trip(self, a2a_e2e_agent: A2AAgent):
        """Send a message that triggers a tool call, then send the result back."""
        state = a2a_e2e_agent.get_init_state()

        user_msg = UserMessage(
            role="user",
            content="Create a task for user_1 titled 'Test Task'",
        )
        response, state = a2a_e2e_agent.generate_next_message(
            user_msg, state
        )

        assert isinstance(response, AssistantMessage)

        # The LLM should respond with a tool call for create_task
        assert response.is_tool_call(), (
            f"Expected a tool call but got text: {response.content!r}"
        )
        assert response.tool_calls
        tool_call = response.tool_calls[0]
        assert tool_call.name == "create_task"

        # Send back a successful tool result
        tool_result = ToolMessage(
            id=tool_call.id,
            role="tool",
            content=(
                '{"task_id": "task_1", "title": "Test Task",'
                ' "status": "pending"}'
            ),
            error=False,
            requestor="assistant",
        )
        response2, state = a2a_e2e_agent.generate_next_message(
            tool_result, state
        )

        assert isinstance(response2, AssistantMessage)
        assert response2.has_text_content(), (
            f"Expected text response after tool result but got "
            f"tool call: {response2.tool_calls}"
        )


# ---------------------------------------------------------------------------
# Functional Layer
# ---------------------------------------------------------------------------

MAX_TURNS = 10


class TestFunctional:
    """Full task flow using a real mock domain task definition."""

    def test_mock_domain_task_flow(
        self, a2a_e2e_agent: A2AAgent
    ):
        """Run a full conversation loop for the create_task_1 mock domain task."""
        from tau2.domains.mock.environment import (
            get_environment,
            get_tasks,
        )

        tasks = get_tasks()
        task = next(t for t in tasks if t.id == "create_task_1")
        assert task.ticket is not None, (
            "create_task_1 must have a ticket field"
        )

        # Build a fresh environment to execute tool calls against
        env = get_environment()

        state = a2a_e2e_agent.get_init_state()

        # Initial user message from the task's ticket
        next_msg: UserMessage | ToolMessage = UserMessage(
            role="user", content=task.ticket
        )
        tool_was_called = False
        tool_name_called: str | None = None
        last_tool_call = None

        for turn in range(MAX_TURNS):
            response, state = a2a_e2e_agent.generate_next_message(
                next_msg, state
            )
            assert isinstance(response, AssistantMessage)

            if response.has_text_content() and not response.is_tool_call():
                # Agent gave a final text response -- conversation done
                break

            if response.is_tool_call() and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_was_called = True
                tool_name_called = tool_call.name
                last_tool_call = tool_call

                # Execute the tool call against the real mock environment
                tool_result = env.get_response(tool_call)
                next_msg = tool_result
        else:
            pytest.fail(
                f"Conversation did not complete within {MAX_TURNS} turns"
            )

        # The task expects create_task to be called with user_id="user_1"
        assert tool_was_called, "Expected at least one tool call"
        assert tool_name_called == "create_task", (
            f"Expected create_task but got {tool_name_called}"
        )
        assert last_tool_call is not None
        assert last_tool_call.arguments.get("user_id") == "user_1", (
            f"Expected user_id='user_1' but got "
            f"{last_tool_call.arguments}"
        )
