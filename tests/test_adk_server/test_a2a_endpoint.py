"""
Integration test for A2A message/send endpoint (T025).

Tests that ADK agent properly handles A2A protocol messages via in-process ASGI.
These tests use the real ADK FastAPI app with the tau2_agent registered.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from google.adk.cli.fast_api import get_fast_api_app
from httpx import ASGITransport, AsyncClient

# Mark all tests in this module as mock-based (no real LLM calls)
pytestmark = pytest.mark.a2a_mock

# Project root for agent discovery
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Note: Use trailing slash for GET, no trailing slash for POST (FastAPI routing)
A2A_AGENT_CARD = "/a2a/tau2_agent/.well-known/agent-card.json"
A2A_ENDPOINT = "/a2a/tau2_agent"  # No trailing slash for POST


@pytest.fixture
def adk_app():
    """Create ADK FastAPI app with tau2_agent registered."""
    return get_fast_api_app(agents_dir=str(PROJECT_ROOT), web=False, a2a=True)


@pytest.fixture
def async_client(adk_app):
    """Create AsyncClient with follow_redirects enabled."""
    return AsyncClient(
        transport=ASGITransport(app=adk_app),
        base_url="http://test",
        follow_redirects=True,
    )


@pytest.mark.asyncio
async def test_agent_card_accessible(async_client):
    """Test that agent card is accessible at well-known endpoint."""
    async with async_client as client:
        response = await client.get(A2A_AGENT_CARD)

        assert response.status_code == 200, (
            f"Agent card should be accessible, got {response.status_code}"
        )

        agent_card = response.json()
        assert "name" in agent_card, "Agent card should have name"
        assert agent_card["name"] == "tau2_agent", "Agent name should be tau2_agent"


@pytest.mark.asyncio
async def test_a2a_message_send_endpoint_exists(async_client):
    """Test that A2A message/send endpoint is accessible and returns valid response."""
    async with async_client as client:
        # A2A message/send uses JSON-RPC 2.0 format
        a2a_message = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-001",
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                }
            },
            "id": "req-001",
        }

        response = await client.post(A2A_ENDPOINT, json=a2a_message)

        # Should return 200 for valid JSON-RPC request
        assert response.status_code == 200, (
            f"A2A endpoint should return 200, got {response.status_code}: {response.text}"
        )

        result = response.json()
        assert "jsonrpc" in result, "Response should be JSON-RPC 2.0"
        assert result["jsonrpc"] == "2.0", "Response should be JSON-RPC 2.0"


@pytest.mark.asyncio
async def test_a2a_message_send_list_domains(async_client):
    """Test A2A message requesting domain list returns valid response."""
    async with async_client as client:
        a2a_message = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-002",
                    "role": "user",
                    "parts": [{"text": "What domains can you evaluate?"}],
                }
            },
            "id": "req-002",
        }

        response = await client.post(A2A_ENDPOINT, json=a2a_message)

        assert response.status_code == 200, (
            f"A2A endpoint should return 200, got {response.status_code}"
        )

        result = response.json()
        # Should have result (success) or error (valid JSON-RPC)
        assert "result" in result or "error" in result, (
            "JSON-RPC response should have result or error"
        )


@pytest.mark.asyncio
async def test_a2a_message_send_evaluation_request(async_client):
    """Test A2A message requesting evaluation with mocked tau2-bench."""
    # Mock tau2-bench results to avoid actual LLM calls
    mock_reward_info = Mock()
    mock_reward_info.reward = 1.0
    mock_reward_info.model_dump = Mock(return_value={"reward": 1.0})

    mock_simulation = Mock()
    mock_simulation.task_id = "task-1"
    mock_simulation.success = True
    mock_simulation.reward_info = mock_reward_info
    mock_simulation.duration = 1.0
    mock_simulation.termination_reason = Mock(value="user_stop")
    mock_simulation.messages = []

    mock_task = Mock()
    mock_task.id = "task-1"
    mock_task.description = Mock()
    mock_task.description.purpose = "Test task"

    mock_results = Mock()
    mock_results.timestamp = "2025-11-24T10:00:00Z"
    mock_results.simulations = [mock_simulation]
    mock_results.tasks = [mock_task]

    mock_metrics = Mock()
    mock_metrics.avg_reward = 1.0
    mock_metrics.pass_hat_ks = {1: 1.0}
    mock_metrics.avg_agent_cost = 0.001

    mock_registry = Mock()
    mock_registry.get_domains.return_value = ["airline", "retail", "telecom", "mock"]

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=[mock_task]),
        patch("tau2.registry.registry", mock_registry),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        async with async_client as client:
            a2a_message = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-003",
                        "role": "user",
                        "parts": [
                            {
                                "text": "Run an evaluation on the airline domain for agent at https://agent.example.com"
                            }
                        ],
                    }
                },
                "id": "req-003",
            }

            response = await client.post(A2A_ENDPOINT, json=a2a_message)

            assert response.status_code == 200, (
                f"A2A endpoint should return 200, got {response.status_code}"
            )

            result = response.json()
            assert "result" in result or "error" in result, (
                "JSON-RPC response should have result or error"
            )


@pytest.mark.asyncio
async def test_a2a_context_id_persistence(async_client):
    """Test that context_id is returned and can be reused for multi-turn conversation."""
    async with async_client as client:
        # First message without context_id
        first_message = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-004",
                    "role": "user",
                    "parts": [{"text": "What can you do?"}],
                }
            },
            "id": "req-004",
        }

        response1 = await client.post(A2A_ENDPOINT, json=first_message)

        assert response1.status_code == 200, (
            f"First message should succeed, got {response1.status_code}"
        )

        result1 = response1.json()
        assert "result" in result1, "Response should have result"

        # Extract context_id from response (may be in different locations)
        context_id = None
        if "result" in result1:
            res = result1["result"]
            # Check various possible locations for context_id
            context_id = (
                res.get("contextId")
                or res.get("context_id")
                or (res.get("message", {}) or {}).get("contextId")
            )

        if context_id:
            # Second message with context_id
            second_message = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-005",
                        "role": "user",
                        "parts": [{"text": "What about the airline domain?"}],
                        "contextId": context_id,
                    }
                },
                "id": "req-005",
            }

            response2 = await client.post(A2A_ENDPOINT, json=second_message)
            assert response2.status_code == 200, (
                f"Second message with context_id should succeed, got {response2.status_code}"
            )


@pytest.mark.asyncio
async def test_a2a_invalid_method_returns_error(async_client):
    """Test that invalid JSON-RPC method returns proper error."""
    async with async_client as client:
        invalid_message = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "req-error-001",
        }

        response = await client.post(A2A_ENDPOINT, json=invalid_message)

        # Should return 200 with JSON-RPC error, or 4xx HTTP error
        # NOTE: 404 is NOT valid - it indicates endpoint routing failure, not method error
        assert response.status_code in [200, 400, 405], (
            f"Expected 200 (with JSON-RPC error), 400, or 405. "
            f"Got {response.status_code}. 404 indicates routing failure."
        )

        if response.status_code == 200:
            result = response.json()
            # JSON-RPC error response should have error field
            assert "error" in result, "Should return JSON-RPC error for invalid method"
            # Verify error code is method not found (-32601) per JSON-RPC spec
            error_code = result["error"].get("code")
            assert error_code == -32601, (
                f"Expected method not found error code -32601, got {error_code}"
            )
