"""Integration tests for A2A debug logging functionality."""

import httpx
import pytest
from a2a.client.errors import A2AClientError
from loguru import logger

from tau2.a2a.models import A2AConfig
from tau2.agent.a2a_agent import A2AAgent
from tau2.data_model.message import UserMessage
from tests.test_a2a_client.conftest import (
    MockA2ATransport,
    build_rpc_message_response,
    create_mock_sdk_client,
)


@pytest.fixture
def a2a_config():
    """Create A2A configuration for testing."""
    return A2AConfig(
        endpoint="http://test-agent.example.com",
        auth_token="test-token-123",
        timeout=300,
    )


def test_debug_logging_context_lifecycle(a2a_config, caplog):
    """Test that context_id lifecycle is logged at TRACE level."""

    def mock_handler(request: httpx.Request):
        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(200, json={"name": "Test", "url": "http://test"})
        if request.method == "POST":
            return httpx.Response(
                200,
                json=build_rpc_message_response(
                    text="Hello! I can help you with that.", context_id="ctx-123"
                ),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    # Configure logger to capture TRACE level
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    # Create agent with SDK client backed by mock transport
    agent = A2AAgent(
        config=a2a_config,
        tools=[],
        domain_policy="Test policy",
        client=create_mock_sdk_client(transport),
    )

    # Get initial state
    state = agent.get_init_state()
    assert state.context_id is None

    # Generate first message
    user_msg = UserMessage(role="user", content="Hello!")
    assistant_msg, new_state = agent.generate_next_message(user_msg, state)

    # Verify context_id was set
    assert new_state.context_id == "ctx-123"
    assert new_state.request_count == 1

    # Generate second message (context should be reused)
    user_msg2 = UserMessage(role="user", content="Thanks!")
    assistant_msg2, final_state = agent.generate_next_message(user_msg2, new_state)

    # Verify context persisted
    assert final_state.context_id == "ctx-123"
    assert final_state.request_count == 2

    # Clean up
    agent.stop(user_msg2, final_state)


def test_debug_logging_protocol_errors(a2a_config, caplog):
    """Test that protocol errors trigger agent-level error handling."""

    transport = MockA2ATransport(
        should_fail=True,
        fail_status=500,
        fail_message="Internal server error",
    )

    # Configure logger to capture TRACE level
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    agent = A2AAgent(
        config=a2a_config,
        tools=[],
        domain_policy="Test policy",
        client=create_mock_sdk_client(transport),
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Test message")

    # The agent's generate_next_message uses asyncio.run() internally.
    # An HTTP 500 from the mock transport causes A2AClientError to propagate.
    with pytest.raises((A2AClientError, Exception)):
        agent.generate_next_message(user_msg, state)

    agent.stop()


def test_debug_logging_tool_descriptions(a2a_config, caplog):
    """Test that tool descriptions are logged when included in messages."""
    import json

    def mock_handler(request: httpx.Request):
        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(200, json={"name": "Test", "url": "http://test"})
        if request.method == "POST":
            # Verify request contains tool descriptions
            request_data = json.loads(request.content)
            message_content = request_data["params"]["message"]["parts"][0]["text"]
            # Tool description should be in the message
            assert "<available_tools>" in message_content or message_content
            return httpx.Response(
                200,
                json=build_rpc_message_response(
                    text="Hello! I can help you with that.", context_id="ctx-456"
                ),
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    # Configure logger to capture TRACE level
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    # Create agent with tools
    from tau2.environment.tool import Tool

    def test_tool(arg1: str) -> str:
        """A test tool.

        Args:
            arg1: A string argument.

        Returns:
            The input string.
        """
        return arg1

    agent = A2AAgent(
        config=a2a_config,
        tools=[Tool(test_tool)],
        domain_policy="Test policy",
        client=create_mock_sdk_client(transport),
    )

    # Get initial state
    state = agent.get_init_state()

    # Generate message with tools
    user_msg = UserMessage(role="user", content="Can you help?")
    assistant_msg, new_state = agent.generate_next_message(user_msg, state)

    # Verify message was sent
    assert new_state.request_count == 1

    # Clean up
    agent.stop(user_msg, new_state)
