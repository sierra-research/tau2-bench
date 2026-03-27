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
        client=mock_a2a_client,
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
        client=mock_a2a_client,
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
        client=mock_a2a_client,
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
    from a2a.client.errors import A2AClientError

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    sdk_client = create_mock_sdk_client(failing_a2a_agent)

    config = A2AConfig(endpoint="http://test-agent.example.com")
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        client=sdk_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")

    # Should raise A2AClientError on failure
    with pytest.raises(A2AClientError):
        agent.generate_next_message(user_msg, state)


def test_a2a_agent_timeout_handling(timeout_a2a_agent, sample_domain_tools):
    """Test A2AAgent handles timeouts properly."""
    from a2a.client.errors import A2AClientError

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    sdk_client = create_mock_sdk_client(timeout_a2a_agent)

    config = A2AConfig(endpoint="http://test-agent.example.com", timeout=1)
    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        client=sdk_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Hello")

    # Should raise A2AClientError on timeout/error
    with pytest.raises(A2AClientError):
        agent.generate_next_message(user_msg, state)


def test_a2a_agent_auth_header(sample_domain_tools):
    """Test that A2AAgent sends authentication token properly."""
    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.client.client_factory import minimal_agent_card

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import build_rpc_message_response

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

        # JSON-RPC message response using Format 2
        return httpx.Response(
            status_code=200,
            json=build_rpc_message_response(text="Hello", context_id="ctx-123"),
        )

    mock_transport = httpx.MockTransport(auth_capture_handler)
    httpx_client = httpx.AsyncClient(
        transport=mock_transport,
        base_url="http://test-agent.example.com",
        headers={"Authorization": "Bearer secret-token-12345"},
    )
    card = minimal_agent_card("http://test-agent.example.com")
    factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=False))
    sdk_client = factory.create(card)

    # Create agent with auth token
    config = A2AConfig(
        endpoint="http://test-agent.example.com",
        auth_token="secret-token-12345",
    )

    agent = A2AAgent(
        config=config,
        tools=sample_domain_tools,
        domain_policy="Airline customer service",
        client=sdk_client,
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
        client=mock_a2a_client,
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
        client=mock_a2a_client,
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


# ---------------------------------------------------------------------------
# Request payload verification — what the agent actually sends to the endpoint
# ---------------------------------------------------------------------------


def _make_capture_transport():
    """Build a transport that records every POST body and returns a valid response."""
    import json as _json

    import httpx as _httpx

    from tests.test_a2a_client.conftest import build_rpc_message_response

    captured_bodies = []

    def handler(request: _httpx.Request) -> _httpx.Response:
        if request.method == "POST":
            captured_bodies.append(_json.loads(request.content))

        return _httpx.Response(
            status_code=200,
            json=build_rpc_message_response(text="ok", context_id="ctx-1"),
        )

    return captured_bodies, _httpx.MockTransport(handler)


def test_first_user_message_includes_tools_and_policy(sample_domain_tools):
    """First user message must include tool descriptions and domain policy.

    This catches the bug where tools were not sent to the A2A endpoint,
    causing tool-call failures that were hard to diagnose.
    """
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    captured, transport = _make_capture_transport()
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Help with flights. Always verify booking details.",
        client=sdk_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Book me a flight")
    agent.generate_next_message(user_msg, state)

    # Extract the text content sent in the JSON-RPC message
    assert len(captured) == 1
    sent_text = captured[0]["params"]["message"]["parts"][0]["text"]

    # Tools MUST be present
    assert "<available_tools>" in sent_text
    assert "search_flights" in sent_text
    assert "book_flight" in sent_text

    # Policy MUST be present on first message
    assert "Help with flights" in sent_text
    assert "<policy>" in sent_text

    # User content MUST be present
    assert "Book me a flight" in sent_text


def test_subsequent_user_message_includes_tools_but_not_policy(sample_domain_tools):
    """Second user message still includes tools (as a reminder) but NOT the policy."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    captured, transport = _make_capture_transport()
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Secret policy text",
        client=sdk_client,
    )

    state = agent.get_init_state()

    # First message — includes policy
    msg1 = UserMessage(role="user", content="Hello")
    _, state = agent.generate_next_message(msg1, state)

    # Second message — tools yes, policy no
    msg2 = UserMessage(role="user", content="Search for flights")
    agent.generate_next_message(msg2, state)

    assert len(captured) == 2
    second_text = captured[1]["params"]["message"]["parts"][0]["text"]

    assert "<available_tools>" in second_text
    assert "search_flights" in second_text
    assert "Secret policy text" not in second_text
    assert "<policy>" not in second_text


def test_tool_result_message_does_not_include_tools(sample_domain_tools):
    """Successful tool result messages must NOT include tool descriptions."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    captured, transport = _make_capture_transport()
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
        client=sdk_client,
    )

    state = agent.get_init_state()

    # First turn: user message
    msg1 = UserMessage(role="user", content="Search flights")
    _, state = agent.generate_next_message(msg1, state)

    # Second turn: successful tool result
    tool_msg = ToolMessage(
        id="call_1",
        role="tool",
        content='{"flights": [{"id": "AA123"}]}',
        error=False,
        requestor="assistant",
    )
    agent.generate_next_message(tool_msg, state)

    assert len(captured) == 2
    tool_result_text = captured[1]["params"]["message"]["parts"][0]["text"]

    # Tool output must be present
    assert "AA123" in tool_result_text

    # Tool descriptions must NOT be in the tool result message
    assert "<available_tools>" not in tool_result_text


# ---------------------------------------------------------------------------
# Tool call validation
# ---------------------------------------------------------------------------


def _make_sequence_transport(responses: list[str]):
    """Build a transport that returns responses in order and captures requests.

    Args:
        responses: List of text responses to return in sequence.
            If more requests are made than responses, the last response is reused.

    Returns:
        Tuple of (captured_bodies, call_count, MockTransport).
    """
    import json as _json

    import httpx as _httpx

    from tests.test_a2a_client.conftest import build_rpc_message_response

    captured_bodies: list[dict] = []
    call_count = [0]

    def handler(request: _httpx.Request) -> _httpx.Response:
        if request.method == "POST":
            captured_bodies.append(_json.loads(request.content))
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return _httpx.Response(
            status_code=200,
            json=build_rpc_message_response(
                text=responses[idx],
                context_id="ctx-1",
            ),
        )

    return captured_bodies, call_count, _httpx.MockTransport(handler)


def test_a2a_agent_tool_error_includes_tools(sample_domain_tools):
    """Tool error messages must include available tool descriptions."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    captured, transport = _make_capture_transport()
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
        client=sdk_client,
    )

    # First turn: user message (to establish state)
    state = agent.get_init_state()
    msg1 = UserMessage(role="user", content="Hello")
    _, state = agent.generate_next_message(msg1, state)

    # Second turn: tool error (simulating environment rejecting a hallucinated tool)
    tool_error = ToolMessage(
        id="call_bad",
        role="tool",
        content="Error: Tool 'find_flights' not found.",
        error=True,
        requestor="assistant",
    )
    agent.generate_next_message(tool_error, state)

    assert len(captured) == 2
    error_text = captured[1]["params"]["message"]["parts"][0]["text"]

    # Must contain the original error
    assert "find_flights" in error_text

    # Must contain tool descriptions
    assert "<available_tools>" in error_text
    assert "search_flights" in error_text
    assert "book_flight" in error_text


def test_a2a_agent_successful_tool_result_does_not_include_tools(sample_domain_tools):
    """Successful tool result messages must NOT include tool descriptions."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    captured, transport = _make_capture_transport()
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
        client=sdk_client,
    )

    state = agent.get_init_state()
    msg1 = UserMessage(role="user", content="Hello")
    _, state = agent.generate_next_message(msg1, state)

    # Successful tool result
    tool_ok = ToolMessage(
        id="call_1",
        role="tool",
        content='{"flights": [{"id": "AA123"}]}',
        error=False,
        requestor="assistant",
    )
    agent.generate_next_message(tool_ok, state)

    assert len(captured) == 2
    result_text = captured[1]["params"]["message"]["parts"][0]["text"]

    # Must contain the tool output
    assert "AA123" in result_text

    # Must NOT contain tool descriptions
    assert "<available_tools>" not in result_text


def test_a2a_agent_invalid_tool_passes_through_to_orchestrator(sample_domain_tools):
    """Invalid tool calls pass through to the orchestrator unchanged."""
    import json

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    # Remote agent returns a hallucinated tool call
    hallucinated_response = json.dumps(
        {"tool_call": {"name": "find_flights", "arguments": {"query": "SFO"}}}
    )
    captured, call_count, transport = _make_sequence_transport([hallucinated_response])
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
        client=sdk_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Find flights")
    assistant_msg, _ = agent.generate_next_message(user_msg, state)

    assert assistant_msg.is_tool_call()
    assert assistant_msg.tool_calls[0].name == "find_flights"
    assert call_count[0] == 1


def test_a2a_agent_valid_tool_passes_through(sample_domain_tools):
    """Valid tool calls pass through unchanged."""
    import json

    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent
    from tests.test_a2a_client.conftest import create_mock_sdk_client

    valid_response = json.dumps(
        {
            "tool_call": {
                "name": "search_flights",
                "arguments": {
                    "origin": "SFO",
                    "destination": "JFK",
                    "date": "2025-12-15",
                },
            }
        }
    )
    captured, call_count, transport = _make_sequence_transport([valid_response])
    sdk_client = create_mock_sdk_client(transport)

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
        client=sdk_client,
    )

    state = agent.get_init_state()
    user_msg = UserMessage(role="user", content="Search flights")
    assistant_msg, _ = agent.generate_next_message(user_msg, state)

    assert assistant_msg.is_tool_call()
    assert assistant_msg.tool_calls[0].name == "search_flights"
    assert call_count[0] == 1


def test_get_valid_tool_names(sample_domain_tools):
    """Test _valid_tool_names contains the correct set of tool names."""
    from tau2.a2a.models import A2AConfig
    from tau2.agent.a2a_agent import A2AAgent

    agent = A2AAgent(
        config=A2AConfig(endpoint="http://test-agent.example.com"),
        tools=sample_domain_tools,
        domain_policy="Policy",
    )

    assert agent._valid_tool_names == {"search_flights", "book_flight"}
