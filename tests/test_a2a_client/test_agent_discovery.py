"""Tests for A2A agent card discovery via SDK A2ACardResolver."""

import json

import httpx
import pytest
from a2a.client import A2ACardResolver
from a2a.client.errors import A2AClientHTTPError, A2AClientJSONError
from a2a.types import AgentCard

from tests.test_a2a_client.conftest import MockA2ATransport, build_agent_card_json


@pytest.mark.a2a_mock
class TestAgentCardDiscovery:
    """Tests for agent card resolution via SDK A2ACardResolver."""

    @pytest.mark.asyncio
    async def test_discover_agent_card(self):
        transport = MockA2ATransport()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test-agent.example.com"
        ) as client:
            resolver = A2ACardResolver(
                httpx_client=client, base_url="http://test-agent.example.com"
            )
            card = await resolver.get_agent_card()
        assert isinstance(card, AgentCard)
        assert card.name == "Test A2A Agent"

    @pytest.mark.asyncio
    async def test_discovery_404(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=404, json={"error": "Not found"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test-agent.example.com"
        ) as client:
            resolver = A2ACardResolver(
                httpx_client=client, base_url="http://test-agent.example.com"
            )
            with pytest.raises(A2AClientHTTPError):
                await resolver.get_agent_card()

    @pytest.mark.asyncio
    async def test_discovery_invalid_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b"not json",
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test-agent.example.com"
        ) as client:
            resolver = A2ACardResolver(
                httpx_client=client, base_url="http://test-agent.example.com"
            )
            with pytest.raises((A2AClientJSONError, A2AClientHTTPError)):
                await resolver.get_agent_card()

    @pytest.mark.asyncio
    async def test_discovery_auth_header(self):
        received_headers = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received_headers.update(dict(request.headers))
            card = build_agent_card_json()
            return httpx.Response(status_code=200, json=card)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test-agent.example.com",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            resolver = A2ACardResolver(
                httpx_client=client, base_url="http://test-agent.example.com"
            )
            await resolver.get_agent_card()
        assert "authorization" in received_headers
        assert received_headers["authorization"] == "Bearer test-token"
