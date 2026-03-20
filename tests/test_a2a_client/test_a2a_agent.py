"""Integration tests for A2AAgent execution."""

import pytest

from tau2.data_model.message import ToolMessage, UserMessage
from tau2.environment.tool import Tool

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


@pytest.fixture
def sample_domain_tools():
    """Create sample domain tools for testing."""

    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Search for available flights."""
        return {
            "flights": [
                {"id": "AA123", "departure": "10:00", "price": 350},
                {"id": "UA456", "departure": "14:00", "price": 400},
            ]
        }

    def book_flight(flight_id: str, passenger_info: dict) -> dict:
        """Book a specific flight."""
        return {
            "booking_id": "BK123456",
            "confirmation": f"Booked flight {flight_id}",
        }

    return [Tool(search_flights), Tool(book_flight)]


def test_a2a_agent_initialization(sample_domain_tools):
    """Test A2AAgent can be initialized with proper configuration."""
    from tau2.a2a.models import A2AAgentState, A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    # Create A2A config
    config = A2AConfig(
        endpoint="http://test-agent.example.com",
        timeout=300,
    )

    # Create A2A agent
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Test airline domain policy",
    )

    # Verify initialization
    assert agent.config.endpoint == "http://test-agent.example.com"
    assert agent.tools == sample_domain_tools
    assert agent.domain_policy == "Test airline domain policy"

    # Get initial state
    state = agent.get_init_state()
    assert isinstance(state, A2AAgentState)
    assert state.context_id is None
    assert state.request_count == 0


def test_a2a_agent_generate_message(mock_a2a_client, sample_domain_tools):
    """Test A2AAgent can generate messages via A2A protocol."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tau2.data_model.message import AssistantMessage

    # Create A2A config
    config = A2AConfig(endpoint="http://test-agent.example.com")

    # Create A2A agent with mock client
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=mock_a2a_client,
    )

    # Get initial state
    state = agent.get_init_state()

    # Create user message
    user_msg = UserMessage(
        role="user",
        content="I need to book a flight from SFO to JFK on December 15th.",
    )

    # Generate response
    assistant_msg, new_state = agent.generate_next_message(user_msg, state)

    # Verify response
    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.role == "assistant"

    # Should be tool call or text response
    assert assistant_msg.has_text_content() or assistant_msg.is_tool_call()

    # Verify state updated
    assert new_state.request_count == 1
    assert new_state.context_id is not None  # Mock returns context_id


def test_a2a_agent_tool_call_flow(mock_a2a_client, sample_domain_tools):
    """Test complete tool call flow through A2A agent."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=mock_a2a_client,
    )

    state = agent.get_init_state()

    # Step 1: User requests flight search
    user_msg = UserMessage(
        role="user",
        content="Search for flights from SFO to JFK on December 15th.",
    )

    assistant_msg, state = agent.generate_next_message(user_msg, state)

    # Agent should request tool call (mock returns search_flights tool call)
    if assistant_msg.is_tool_call():
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) > 0
        assert assistant_msg.tool_calls[0].name == "search_flights"

        # Step 2: Execute tool and send result back
        tool_result = ToolMessage(
            id=assistant_msg.tool_calls[0].id,
            role="tool",
            content='{"flights": [{"id": "AA123", "price": 350}]}',
            error=False,
            requestor="assistant",
        )

        # Agent processes tool result
        assistant_msg_2, state = agent.generate_next_message(tool_result, state)

        # Verify context persisted
        assert state.request_count == 2
        assert state.context_id is not None


def test_a2a_agent_context_persistence(mock_a2a_client, sample_domain_tools):
    """Test that context_id is persisted across multiple turns."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=mock_a2a_client,
    )

    state = agent.get_init_state()
    assert state.context_id is None

    # First turn
    user_msg_1 = UserMessage(role="user", content="Hello")
    _, state = agent.generate_next_message(user_msg_1, state)

    # Context should be set after first response
    first_context_id = state.context_id
    assert first_context_id is not None

    # Second turn
    user_msg_2 = UserMessage(role="user", content="Thank you")
    _, state = agent.generate_next_message(user_msg_2, state)

    # Context should persist
    assert state.context_id == first_context_id
    assert state.request_count == 2


def test_a2a_agent_error_handling(failing_a2a_agent, sample_domain_tools):
    """Test A2AAgent handles errors gracefully."""
    import httpx

    from tau2.a2a.exceptions import A2AError
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    # Create client with failing transport
    failing_client = httpx.AsyncClient(
        transport=failing_a2a_agent,
        base_url="http://test-agent.example.com",
    )

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=failing_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")

    # Should raise A2AError on failure
    with pytest.raises(A2AError):
        agent.generate_next_message(user_msg, state)


def test_a2a_agent_timeout_handling(timeout_a2a_agent, sample_domain_tools):
    """Test A2AAgent handles timeouts properly."""
    import httpx

    from tau2.a2a.exceptions import A2AError
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    # Create client with timeout transport
    timeout_client = httpx.AsyncClient(
        transport=timeout_a2a_agent,
        base_url="http://test-agent.example.com",
    )

    config = A2AConfig(endpoint="http://test-agent.example.com", timeout=1)
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=timeout_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")

    # Should raise timeout error
    with pytest.raises(A2AError):
        agent.generate_next_message(user_msg, state)


def test_a2a_agent_auth_header(sample_domain_tools):
    """Test that A2AAgent sends authentication token properly."""
    import httpx

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    auth_header_received = None

    def auth_capture_handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_header_received
        auth_header_received = request.headers.get("authorization")

        # Return minimal valid response
        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(
                status_code=200,
                json={
                    "name": "Test Agent",
                    "url": str(request.url.copy_with(path="")),
                    "capabilities": {},
                },
            )

        # JSON-RPC message response
        return httpx.Response(
            status_code=200,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "agent",
                        "parts": [{"text": "Hello"}],
                        "contextId": "ctx-123",
                    }
                },
            },
        )

    mock_transport = httpx.MockTransport(auth_capture_handler)
    auth_client = httpx.AsyncClient(
        transport=mock_transport,
        base_url="http://test-agent.example.com",
    )

    # Create agent with auth token
    config = A2AConfig(
        endpoint="http://test-agent.example.com",
        auth_token="secret-token-12345",
    )

    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=auth_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")

    # Generate message
    agent.generate_next_message(user_msg, state)

    # Verify auth header was sent
    assert auth_header_received == "Bearer secret-token-12345"


def test_a2a_agent_stop_method(mock_a2a_client, sample_domain_tools):
    """Test A2AAgent stop method."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=mock_a2a_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Goodbye")

    # Stop should not raise error
    agent.stop(message=user_msg, state=state)


def test_a2a_agent_with_message_history(mock_a2a_client, sample_domain_tools):
    """Test A2AAgent can be initialized with message history."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tau2.data_model.message import AssistantMessage

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        http_client=mock_a2a_client,
    )

    # Create message history
    message_history = [
        UserMessage(role="user", content="Hello"),
        AssistantMessage(role="assistant", content="Hi, how can I help?"),
    ]

    # Initialize with history
    state = agent.get_init_state(message_history=message_history)

    # Verify history preserved
    assert len(state.conversation_history) == 2
    assert state.conversation_history[0].content == "Hello"
