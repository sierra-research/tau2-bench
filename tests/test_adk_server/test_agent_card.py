"""
Integration test for agent card serving (T023).

Tests that ADK agent properly serves /.well-known/agent-card.json endpoint.

NOTE: As of google-adk version used, the /.well-known/agent-card.json endpoint
is not automatically created. These tests are skipped pending ADK support.
"""

import pytest
from google.adk.cli.fast_api import get_fast_api_app
from httpx import ASGITransport, AsyncClient

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="ADK does not currently expose /.well-known/agent-card.json endpoint. "
    "This feature may need to be implemented separately or awaits ADK update."
)
async def test_agent_card_endpoint_exists():
    """Test that agent card endpoint is accessible"""
    app = get_fast_api_app(agents_dir="./tau2_agent", web=False, a2a=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/.well-known/agent-card.json")

        assert response.status_code == 200, "Agent card endpoint should return 200"
        assert response.headers["content-type"] == "application/json", (
            "Agent card should be JSON"
        )


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="ADK does not currently expose /.well-known/agent-card.json endpoint. "
    "This feature may need to be implemented separately or awaits ADK update."
)
async def test_agent_card_structure():
    """Test that agent card has correct structure and metadata"""
    app = get_fast_api_app(agents_dir="./tau2_agent", web=False, a2a=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        card = response.json()

        # Required fields per A2A spec
        assert "name" in card, "Agent card must have name"
        assert "url" in card, "Agent card must have url"
        assert card["name"] == "tau2_agent", "Agent name should be tau2_agent"

        # Optional but expected fields
        assert "description" in card, "Agent card should have description"
        assert "capabilities" in card, "Agent card should have capabilities"


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="ADK does not currently expose /.well-known/agent-card.json endpoint. "
    "This feature may need to be implemented separately or awaits ADK update."
)
async def test_agent_card_tools_listed():
    """Test that agent card lists tau2 evaluation tools"""
    app = get_fast_api_app(agents_dir="./tau2_agent", web=False, a2a=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        card = response.json()

        # Check that tools/skills are listed
        card_str = str(card).lower()
        assert "run_tau2_evaluation" in card_str or "runtau2evaluation" in card_str, (
            "Agent card should mention RunTau2Evaluation tool"
        )
        assert "list_domains" in card_str or "listdomains" in card_str, (
            "Agent card should mention ListDomains tool"
        )
