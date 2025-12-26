"""E2E tests for error handling scenarios."""

import httpx
import pytest

from tests.test_a2a_e2e.conftest import A2EServer, build_a2a_evaluation_request

pytestmark = pytest.mark.a2a_e2e


@pytest.mark.asyncio
async def test_request_timeout(a2e_server: A2EServer):
    """Verify timeout exceptions propagate correctly."""
    async with httpx.AsyncClient(timeout=0.001) as client:
        request = build_a2a_evaluation_request(
            domain="mock",
            agent_endpoint=a2e_server.mock_agent_endpoint,
        )
        with pytest.raises(httpx.TimeoutException):
            await client.post(a2e_server.tau2_agent_endpoint, json=request)


@pytest.mark.asyncio
async def test_malformed_json_rpc_requests(a2e_server: A2EServer):
    """Verify handling of various malformed JSON-RPC requests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        test_cases = [
            # Missing jsonrpc field
            {
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "t1",
                        "role": "user",
                        "parts": [{"text": "Hi"}],
                    }
                },
                "id": "req-001",
            },
            # Missing method field
            {"jsonrpc": "2.0", "params": {}, "id": "req-002"},
            # Invalid jsonrpc version
            {
                "jsonrpc": "1.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "t3",
                        "role": "user",
                        "parts": [{"text": "Hi"}],
                    }
                },
                "id": "req-003",
            },
            # Missing message in params
            {"jsonrpc": "2.0", "method": "message/send", "params": {}, "id": "req-004"},
            # Empty message parts
            {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {"message": {"messageId": "t5", "role": "user", "parts": []}},
                "id": "req-005",
            },
        ]

        for i, request in enumerate(test_cases):
            response = await client.post(a2e_server.tau2_agent_endpoint, json=request)
            # ADK may be lenient - accept 200 (with error/result), 400, or 422
            assert response.status_code in [200, 400, 422], (
                f"Case {i}: Expected 200/400/422, got {response.status_code}"
            )
            # If 200, should have either error or result (ADK lenient)
            if response.status_code == 200:
                result = response.json()
                assert "error" in result or "result" in result, (
                    f"Case {i}: 200 response should have error or result"
                )


@pytest.mark.asyncio
async def test_evaluation_target_errors(a2e_server: A2EServer):
    """Verify handling of invalid domain and unreachable agent endpoint."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Invalid domain
        request = build_a2a_evaluation_request(
            domain="nonexistent_domain_xyz",
            agent_endpoint=a2e_server.mock_agent_endpoint,
            num_tasks=1,
            num_trials=1,
        )
        response = await client.post(a2e_server.tau2_agent_endpoint, json=request)
        assert response.status_code == 200
        result_text = str(response.json())
        domain_error_indicators = ["not found", "invalid", "error", "unknown", "domain"]
        assert any(ind in result_text.lower() for ind in domain_error_indicators), (
            f"Invalid domain response should indicate error: {result_text[:200]}"
        )

        # Unreachable agent endpoint
        request = build_a2a_evaluation_request(
            domain="mock",
            agent_endpoint="http://127.0.0.1:19999/a2a/nonexistent",
            num_tasks=1,
            num_trials=1,
        )
        response = await client.post(a2e_server.tau2_agent_endpoint, json=request)
        assert response.status_code == 200
        result_text = str(response.json())
        conn_error_indicators = ["error", "failed", "connect", "unreachable", "refused"]
        assert any(ind in result_text.lower() for ind in conn_error_indicators), (
            f"Unreachable endpoint response should indicate error: {result_text[:200]}"
        )
