"""
End-to-end tests for A2A protocol compliance with ADK agents.

These tests verify the A2A protocol communication works correctly
with any ADK agent server, including tau2-bench evaluation integration.

All tests are marked with @pytest.mark.a2a_e2e and are NOT run by default.
Run explicitly with: pytest -m a2a_e2e
"""

import httpx
import pytest

# Mark all tests in this module as E2E tests
pytestmark = pytest.mark.a2a_e2e


@pytest.mark.asyncio
async def test_e2e_jsonrpc_message_send(adk_server):
    """
    Test JSON-RPC message/send method over real HTTP.

    Verifies that the A2A protocol JSON-RPC structure works correctly.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Send A2A message with proper JSON-RPC structure
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-msg-001",
                    "role": "user",
                    "parts": [{"text": "Hello, what can you help me with?"}],
                }
            },
            "id": "req-001",
        }

        response = await client.post(f"{adk_server}/", json=jsonrpc_request)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        result = response.json()

        # Verify JSON-RPC 2.0 structure
        assert "jsonrpc" in result, "Response missing jsonrpc field"
        assert result["jsonrpc"] == "2.0", "Response not JSON-RPC 2.0"
        assert "id" in result, "Response missing id field"
        assert result["id"] == "req-001", "Response id should match request"

        # Should have result (not error)
        assert "result" in result, f"Response missing result: {result}"


@pytest.mark.asyncio
async def test_e2e_agent_card_discovery(adk_server):
    """
    Test agent card discovery via well-known endpoint.
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(f"{adk_server}/.well-known/agent-card.json")

        assert response.status_code == 200, (
            f"Agent card not found: {response.status_code}"
        )
        agent_card = response.json()

        # Verify required agent card fields per A2A spec
        assert "name" in agent_card, "Agent card missing 'name'"
        assert len(agent_card["name"]) > 0, "Agent name is empty"

        # Optional but common fields
        if "capabilities" in agent_card:
            assert isinstance(agent_card["capabilities"], dict)

        if "skills" in agent_card:
            assert isinstance(agent_card["skills"], list)


@pytest.mark.asyncio
async def test_e2e_context_persistence(adk_server):
    """
    Test that context_id enables multi-turn conversation.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # First message - no context
        request_1 = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-001",
                    "role": "user",
                    "parts": [{"text": "Remember this: the magic number is 42."}],
                }
            },
            "id": "req-001",
        }

        response_1 = await client.post(f"{adk_server}/", json=request_1)
        assert response_1.status_code == 200
        result_1 = response_1.json()

        # Extract context_id from response
        context_id = None
        if "result" in result_1:
            result = result_1["result"]
            context_id = result.get("contextId") or result.get("context_id")

        # Second message - with context
        request_2 = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-002",
                    "role": "user",
                    "parts": [{"text": "What magic number did I mention?"}],
                    "contextId": context_id,
                }
            },
            "id": "req-002",
        }

        response_2 = await client.post(f"{adk_server}/", json=request_2)
        assert response_2.status_code == 200


@pytest.mark.asyncio
async def test_e2e_error_handling_invalid_method(adk_server):
    """
    Test error handling for invalid JSON-RPC method.
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # Send request with invalid method
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "params": {},
            "id": "req-error-001",
        }

        response = await client.post(f"{adk_server}/", json=jsonrpc_request)

        # Should return 200 with JSON-RPC error, or 4xx HTTP error
        assert response.status_code in [200, 400, 404, 405], (
            f"Unexpected status: {response.status_code}"
        )

        if response.status_code == 200:
            result = response.json()
            # JSON-RPC error response should have error field
            assert "error" in result, "Should return JSON-RPC error"


@pytest.mark.asyncio
async def test_e2e_error_handling_malformed_request(adk_server):
    """
    Test error handling for malformed JSON-RPC request.
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # Send malformed request (missing required fields)
        jsonrpc_request = {
            "jsonrpc": "2.0",
            # Missing method
            "params": {},
            "id": "req-malformed-001",
        }

        response = await client.post(f"{adk_server}/", json=jsonrpc_request)

        # Should return error (either HTTP or JSON-RPC)
        assert response.status_code in [200, 400, 422], (
            f"Unexpected status: {response.status_code}"
        )


@pytest.mark.asyncio
async def test_e2e_protocol_version_in_card(adk_server):
    """
    Test that agent card includes protocol version.
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(f"{adk_server}/.well-known/agent-card.json")
        assert response.status_code == 200

        agent_card = response.json()

        # Protocol version is recommended in A2A spec
        if "protocolVersion" in agent_card:
            version = agent_card["protocolVersion"]
            # Should be a version string like "0.3.0"
            assert isinstance(version, str), "Protocol version should be string"
            assert len(version) > 0, "Protocol version should not be empty"


@pytest.mark.asyncio
async def test_e2e_response_formats(adk_server):
    """
    Test that response follows one of valid A2A response formats.
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test-format-001",
                    "role": "user",
                    "parts": [{"text": "Please respond with a simple greeting."}],
                }
            },
            "id": "req-format-001",
        }

        response = await client.post(f"{adk_server}/", json=jsonrpc_request)
        assert response.status_code == 200

        result = response.json()
        assert "result" in result, "Response missing result"

        res = result["result"]

        # Check for valid A2A response format:
        # 1. Task with artifacts: res.artifacts[].parts[]
        # 2. Direct Message: res.parts[]
        # 3. Task status: res.status.message.parts[]
        # 4. Message wrapper: res.message.parts[]
        # 5. History: res.history[].parts[]

        has_valid_format = (
            "artifacts" in res
            or "parts" in res
            or ("status" in res and "message" in res.get("status", {}))
            or "message" in res
            or "history" in res
        )

        assert has_valid_format, (
            f"Response doesn't match any valid A2A format. Keys: {list(res.keys())}"
        )


@pytest.mark.asyncio
async def test_e2e_concurrent_requests(adk_server):
    """
    Test that server handles concurrent requests correctly.
    """
    import asyncio

    async def send_request(client, msg_id):
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"concurrent-{msg_id}",
                    "role": "user",
                    "parts": [{"text": f"Concurrent request {msg_id}"}],
                }
            },
            "id": f"req-concurrent-{msg_id}",
        }
        return await client.post(f"{adk_server}/", json=jsonrpc_request)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        # Send 3 concurrent requests
        tasks = [send_request(client, i) for i in range(3)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                pytest.fail(f"Request {i} failed: {resp}")
            assert resp.status_code == 200, f"Request {i} returned {resp.status_code}"


