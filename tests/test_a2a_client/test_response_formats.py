"""Tests for A2A client response format parsing and protocol error handling.

The A2A client handles 5 different response formats from various A2A server
implementations. These tests verify each format is correctly parsed.
"""

import httpx
import pytest

from tau2.a2a.client import A2AClient
from tau2.a2a.exceptions import A2AError, A2AMessageError, A2ATimeoutError
from tau2.a2a.models import A2AConfig


def _make_jsonrpc_transport(result: dict) -> httpx.MockTransport:
    """Build a MockTransport that returns a JSON-RPC 2.0 success response."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {"jsonrpc": "2.0", "id": "1", "result": result}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _make_client(transport: httpx.MockTransport) -> A2AClient:
    """Build an A2AClient wired to the given transport."""
    config = A2AConfig(endpoint="http://test-agent.example.com")
    http = httpx.AsyncClient(transport=transport, base_url=config.endpoint)
    return A2AClient(config=config, http_client=http)


# ---------------------------------------------------------------------------
# Response format diversity
# ---------------------------------------------------------------------------


class TestResponseFormatDiversity:
    """Verify the client extracts text from all 5 A2A response formats.

    The formats correspond to different A2A server implementations:
    1. Google ADK — artifacts array
    2. Direct Message — parts at result level
    3. TaskStatusUpdateEvent — status.message.parts
    4. Legacy wrapper — result.message.parts
    5. History-based — last agent message in history array
    """

    @pytest.mark.asyncio
    async def test_format1_google_adk_artifacts(self):
        """Client extracts text from result.artifacts[].parts[].text."""
        transport = _make_jsonrpc_transport(
            {
                "artifacts": [{"parts": [{"text": "artifact response"}]}],
                "contextId": "ctx-adk",
            }
        )
        client = _make_client(transport)

        content, ctx = await client.send_message("hi")

        assert content == "artifact response"
        assert ctx == "ctx-adk"

    @pytest.mark.asyncio
    async def test_format1_multiple_artifacts_concatenated(self):
        """Multiple artifacts are joined with newline."""
        transport = _make_jsonrpc_transport(
            {
                "artifacts": [
                    {"parts": [{"text": "first"}]},
                    {"parts": [{"text": "second"}]},
                ],
            }
        )
        client = _make_client(transport)

        content, _ = await client.send_message("hi")

        assert content == "first\nsecond"

    @pytest.mark.asyncio
    async def test_format2_direct_message_parts(self):
        """Client extracts text from result.parts[].text."""
        transport = _make_jsonrpc_transport(
            {
                "parts": [{"text": "direct message"}],
                "contextId": "ctx-direct",
            }
        )
        client = _make_client(transport)

        content, ctx = await client.send_message("hi")

        assert content == "direct message"
        assert ctx == "ctx-direct"

    @pytest.mark.asyncio
    async def test_format3_task_status_update_event(self):
        """Client extracts text from result.status.message.parts[].text."""
        transport = _make_jsonrpc_transport(
            {
                "status": {
                    "message": {"parts": [{"text": "status update"}]},
                },
            }
        )
        client = _make_client(transport)

        content, _ = await client.send_message("hi")

        assert content == "status update"

    @pytest.mark.asyncio
    async def test_format4_legacy_wrapper(self):
        """Client extracts text from result.message.parts[].text (legacy)."""
        transport = _make_jsonrpc_transport(
            {
                "message": {
                    "messageId": "msg-1",
                    "role": "agent",
                    "parts": [{"text": "legacy response"}],
                    "contextId": "ctx-legacy",
                },
            }
        )
        client = _make_client(transport)

        content, ctx = await client.send_message("hi")

        assert content == "legacy response"
        assert ctx == "ctx-legacy"

    @pytest.mark.asyncio
    async def test_format5_history_based(self):
        """Client extracts text from the last agent message in history."""
        transport = _make_jsonrpc_transport(
            {
                "history": [
                    {"role": "user", "parts": [{"text": "hello"}]},
                    {"role": "agent", "parts": [{"text": "from history"}]},
                ],
            }
        )
        client = _make_client(transport)

        content, _ = await client.send_message("hi")

        assert content == "from history"

    @pytest.mark.asyncio
    async def test_format5_history_skips_trailing_user_messages(self):
        """History extraction picks the last agent message, not a trailing user."""
        transport = _make_jsonrpc_transport(
            {
                "history": [
                    {"role": "user", "parts": [{"text": "hello"}]},
                    {"role": "agent", "parts": [{"text": "agent reply"}]},
                    {"role": "user", "parts": [{"text": "thanks"}]},
                ],
            }
        )
        client = _make_client(transport)

        content, _ = await client.send_message("hi")

        assert content == "agent reply"

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_string(self):
        """When no format matches, client returns empty string without error."""
        transport = _make_jsonrpc_transport({})
        client = _make_client(transport)

        content, _ = await client.send_message("hi")

        assert content == ""

    @pytest.mark.asyncio
    async def test_context_id_from_top_level_result(self):
        """context_id extracted from result.contextId (Google ADK format)."""
        transport = _make_jsonrpc_transport(
            {
                "parts": [{"text": "ok"}],
                "contextId": "ctx-top",
            }
        )
        client = _make_client(transport)

        _, ctx = await client.send_message("hi")

        assert ctx == "ctx-top"

    @pytest.mark.asyncio
    async def test_context_id_from_nested_message(self):
        """context_id extracted from result.message.contextId (standard A2A)."""
        transport = _make_jsonrpc_transport(
            {
                "message": {
                    "parts": [{"text": "ok"}],
                    "contextId": "ctx-nested",
                },
            }
        )
        client = _make_client(transport)

        _, ctx = await client.send_message("hi")

        assert ctx == "ctx-nested"


# ---------------------------------------------------------------------------
# JSON-RPC error-in-body
# ---------------------------------------------------------------------------


class TestJsonRpcErrorInBody:
    """Verify that JSON-RPC errors returned inside a 200 response are detected."""

    @pytest.mark.asyncio
    async def test_jsonrpc_error_raises_message_error(self):
        """HTTP 200 with JSON-RPC 'error' key must raise A2AMessageError."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32600, "message": "Invalid request"},
            }
            return httpx.Response(200, json=body)

        client = _make_client(httpx.MockTransport(handler))

        with pytest.raises(A2AMessageError, match="Agent returned error"):
            await client.send_message("hi")

    @pytest.mark.asyncio
    async def test_jsonrpc_error_missing_message_uses_unknown(self):
        """JSON-RPC error without 'message' field shows 'Unknown error'."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32600},
            }
            return httpx.Response(200, json=body)

        client = _make_client(httpx.MockTransport(handler))

        with pytest.raises(A2AMessageError, match="Unknown error"):
            await client.send_message("hi")


# ---------------------------------------------------------------------------
# Real timeout / HTTP error paths
# ---------------------------------------------------------------------------


class TestTimeoutAndHttpErrors:
    """Verify httpx exceptions are translated to A2A exceptions."""

    @pytest.mark.asyncio
    async def test_httpx_read_timeout_raises_a2a_timeout_error(self):
        """httpx.ReadTimeout propagates as A2ATimeoutError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out")

        client = _make_client(httpx.MockTransport(handler))

        with pytest.raises(A2ATimeoutError):
            await client.send_message("hi")

    @pytest.mark.asyncio
    async def test_httpx_connect_timeout_raises_a2a_timeout_error(self):
        """httpx.ConnectTimeout also maps to A2ATimeoutError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("connect timed out")

        client = _make_client(httpx.MockTransport(handler))

        with pytest.raises(A2ATimeoutError):
            await client.send_message("hi")

    @pytest.mark.asyncio
    async def test_httpx_connect_error_raises_a2a_error(self):
        """httpx.ConnectError maps to base A2AError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _make_client(httpx.MockTransport(handler))

        with pytest.raises(A2AError, match="Failed to send message"):
            await client.send_message("hi")
