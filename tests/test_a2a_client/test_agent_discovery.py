"""Integration tests for A2A agent discovery."""

import pytest

from tau2.a2a.exceptions import A2ADiscoveryError
from tau2.a2a.models import A2AConfig, AgentCard

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock

# This test will fail until we implement the client


def test_discover_agent_success(mock_a2a_client):
    """Test successful agent discovery via agent card."""
    import asyncio

    from tau2.a2a.client import A2AClient

    # Create client config
    config = A2AConfig(endpoint="http://test-agent.example.com")

    # Create client with mock transport
    client = A2AClient(config, http_client=mock_a2a_client)

    # Discover agent
    agent_card = asyncio.run(client.discover_agent())

    # Verify agent card
    assert isinstance(agent_card, AgentCard)
    assert agent_card.name == "Test Airline Agent"
    assert agent_card.description == "Mock airline customer service agent for testing"
    assert agent_card.version == "1.0.0"
    assert agent_card.capabilities.streaming is False
    assert agent_card.capabilities.push_notifications is False


def test_discover_agent_failure(failing_a2a_agent):
    """Test agent discovery failure handling."""
    import asyncio

    from tau2.a2a.client import A2AClient

    # Create client with failing transport
    client_with_failure = pytest.importorskip("httpx").AsyncClient(
        transport=failing_a2a_agent,
        base_url="http://test-agent.example.com",
    )

    config = A2AConfig(endpoint="http://test-agent.example.com")
    client = A2AClient(config, http_client=client_with_failure)

    # Should raise discovery error
    with pytest.raises(A2ADiscoveryError) as exc_info:
        asyncio.run(client.discover_agent())

    assert "discovery failed" in str(exc_info.value).lower()


def test_discover_agent_caching(mock_a2a_client):
    """Test that agent card is cached after first discovery."""
    import asyncio

    from tau2.a2a.client import A2AClient

    config = A2AConfig(endpoint="http://test-agent.example.com")
    client = A2AClient(config, http_client=mock_a2a_client)

    # First discovery
    agent_card_1 = asyncio.run(client.discover_agent())

    # Second discovery should return cached result
    agent_card_2 = asyncio.run(client.discover_agent())

    # Should be the same instance (cached)
    assert agent_card_1 is agent_card_2

    # Verify only one request was made to the mock
    assert mock_a2a_client._transport.request_count == 1


def test_discover_agent_validates_response():
    """Test that invalid agent card responses are rejected."""
    import asyncio

    import httpx

    from tau2.a2a.client import A2AClient

    # Create mock that returns invalid agent card (missing required fields)
    def invalid_agent_card_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(
                status_code=200,
                json={"invalid": "response"},  # Missing required 'name' and 'url'
            )
        return httpx.Response(status_code=404)

    mock_transport = httpx.MockTransport(invalid_agent_card_handler)
    client_with_invalid = httpx.AsyncClient(
        transport=mock_transport,
        base_url="http://test-agent.example.com",
    )

    config = A2AConfig(endpoint="http://test-agent.example.com")
    client = A2AClient(config, http_client=client_with_invalid)

    # Should raise error due to validation failure
    with pytest.raises((A2ADiscoveryError, ValueError)):
        asyncio.run(client.discover_agent())


def test_discover_agent_with_auth():
    """Test agent discovery with authentication token."""
    import asyncio

    import httpx

    from tau2.a2a.client import A2AClient

    auth_token_used = None

    def auth_checking_handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_token_used
        # Capture auth header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            auth_token_used = auth_header[7:]  # Remove "Bearer " prefix

        if request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(
                status_code=200,
                json={
                    "name": "Authenticated Agent",
                    "url": str(request.url.copy_with(path="")),
                    "capabilities": {"streaming": False},
                },
            )
        return httpx.Response(status_code=404)

    mock_transport = httpx.MockTransport(auth_checking_handler)
    client_with_auth = httpx.AsyncClient(
        transport=mock_transport,
        base_url="http://test-agent.example.com",
    )

    # Create client with auth token
    config = A2AConfig(
        endpoint="http://test-agent.example.com",
        auth_token="test-token-12345",
    )
    client = A2AClient(config, http_client=client_with_auth)

    # Discover agent
    asyncio.run(client.discover_agent())

    # Verify auth token was sent
    assert auth_token_used == "test-token-12345"
