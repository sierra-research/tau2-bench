"""A2A end-to-end test scenarios.

Run with: pytest -m full_a2a_integration
"""

import asyncio

import httpx
import pytest
from a2a.client import A2ACardResolver
from a2a.types import AgentCard

from tau2.agent.a2a_agent import A2AAgent
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

pytestmark = pytest.mark.full_a2a_integration


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestSmoke:
    """Basic connectivity and protocol compliance."""

    def test_agent_card_discovery(self, a2a_e2e_endpoint: str):
        """Test agent card can be fetched and has required fields."""

        async def _fetch_card() -> AgentCard:
            async with httpx.AsyncClient(base_url=a2a_e2e_endpoint) as client:
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

    def test_single_message_round_trip(self, a2a_e2e_agent: A2AAgent):
        """Test single message produces a non-empty text response."""
        state = a2a_e2e_agent.get_init_state()
        user_msg = UserMessage(role="user", content="Hello")

        response, new_state = a2a_e2e_agent.generate_next_message(user_msg, state)

        assert isinstance(response, AssistantMessage)
        assert response.has_text_content()
        assert response.content
        assert not response.is_tool_call()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    """A2A protocol mechanics: context tracking and tool calls."""

    def test_context_persistence(self, a2a_e2e_agent: A2AAgent):
        """Test context_id is set after first call and stable across turns."""
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
        """Test sending a tool result back after receiving a tool call."""
        state = a2a_e2e_agent.get_init_state()

        user_msg = UserMessage(
            role="user",
            content="Create a task for user_1 titled 'Test Task'",
        )
        response, state = a2a_e2e_agent.generate_next_message(user_msg, state)

        assert isinstance(response, AssistantMessage)
        assert response.is_tool_call(), (
            f"Expected a tool call but got text: {response.content!r}"
        )
        assert response.tool_calls
        tool_call = response.tool_calls[0]

        tool_result = ToolMessage(
            id=tool_call.id,
            role="tool",
            content='{"status": "ok", "result": "success"}',
            error=False,
            requestor="assistant",
        )
        response2, state = a2a_e2e_agent.generate_next_message(tool_result, state)

        assert isinstance(response2, AssistantMessage)


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------

MAX_TURNS = 10


class TestFunctional:
    """Full task flow using a real mock domain task definition."""

    def test_mock_domain_task_flow(self, a2a_e2e_agent: A2AAgent):
        """Test full conversation loop with tool execution against mock environment."""
        from tau2.domains.mock.environment import (
            get_environment,
            get_tasks,
        )

        tasks = get_tasks()
        task = next(t for t in tasks if t.id == "create_task_1")
        assert task.ticket is not None, "create_task_1 must have a ticket field"

        env = get_environment()
        state = a2a_e2e_agent.get_init_state()

        next_msg: UserMessage | ToolMessage = UserMessage(
            role="user", content=task.ticket
        )
        tool_names_called: list[str] = []

        for turn in range(MAX_TURNS):
            response, state = a2a_e2e_agent.generate_next_message(next_msg, state)
            assert isinstance(response, AssistantMessage)

            if response.has_text_content() and not response.is_tool_call():
                break

            if response.is_tool_call() and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_names_called.append(tool_call.name)

                tool_result = env.get_response(tool_call)
                next_msg = tool_result
        else:
            pytest.fail(f"Conversation did not complete within {MAX_TURNS} turns")

        assert tool_names_called, "Expected at least one tool call in the flow"
        assert state.context_id is not None, "Context should be established"
        assert state.request_count >= 2, (
            f"Expected at least 2 round trips, got {state.request_count}"
        )
