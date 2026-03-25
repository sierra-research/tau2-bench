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


# ---------------------------------------------------------------------------
# Tests for extract_response() — SDK typed extraction
# ---------------------------------------------------------------------------

import uuid

from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from tau2.a2a.translation import extract_response


def _make_text_part(text: str) -> Part:
    """Helper to create a Part containing a TextPart."""
    return Part(root=TextPart(text=text))


def _make_message(text: str, context_id: str | None = None) -> Message:
    """Helper to create a simple agent text message."""
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.agent,
        parts=[_make_text_part(text)],
        context_id=context_id,
    )


def _make_task(
    *,
    artifacts: list[Artifact] | None = None,
    status_message: Message | None = None,
    history: list[Message] | None = None,
    context_id: str = "ctx-123",
) -> Task:
    """Helper to create a Task with specified fields."""
    status = TaskStatus(state=TaskState.completed)
    if status_message is not None:
        status = TaskStatus(state=TaskState.completed, message=status_message)
    return Task(
        id=str(uuid.uuid4()),
        context_id=context_id,
        status=status,
        artifacts=artifacts,
        history=history,
    )


class TestMessageExtraction:
    """Tests for extract_response() on Message objects."""

    def test_simple_text(self):
        msg = _make_message("Hello world", context_id="ctx-1")
        text, ctx = extract_response(msg)
        assert text == "Hello world"
        assert ctx == "ctx-1"

    def test_multi_part_message(self):
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[_make_text_part("Part one"), _make_text_part("Part two")],
            context_id="ctx-2",
        )
        text, ctx = extract_response(msg)
        assert "Part one" in text
        assert "Part two" in text
        assert ctx == "ctx-2"

    def test_none_context_id(self):
        msg = _make_message("No context")
        text, ctx = extract_response(msg)
        assert text == "No context"
        assert ctx is None

    def test_empty_parts(self):
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[],
        )
        text, ctx = extract_response(msg)
        assert text == ""


class TestTaskExtraction:
    """Tests for extract_response() on Task objects."""

    def test_task_with_artifacts(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("Artifact text")],
        )
        task = _make_task(artifacts=[artifact])
        text, ctx = extract_response(task)
        assert text == "Artifact text"
        assert ctx == "ctx-123"

    def test_task_with_multiple_artifacts(self):
        a1 = Artifact(artifact_id="a1", parts=[_make_text_part("First")])
        a2 = Artifact(artifact_id="a2", parts=[_make_text_part("Second")])
        task = _make_task(artifacts=[a1, a2])
        text, ctx = extract_response(task)
        assert "First" in text
        assert "Second" in text

    def test_task_with_status_message(self):
        status_msg = _make_message("Status update")
        task = _make_task(status_message=status_msg)
        text, ctx = extract_response(task)
        assert text == "Status update"

    def test_task_with_history(self):
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("User said")],
        )
        agent_msg = _make_message("Agent replied")
        task = _make_task(history=[user_msg, agent_msg])
        text, ctx = extract_response(task)
        assert text == "Agent replied"

    def test_task_history_picks_last_agent_message(self):
        agent1 = _make_message("First reply")
        user = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("Follow up")],
        )
        agent2 = _make_message("Second reply")
        task = _make_task(history=[agent1, user, agent2])
        text, _ = extract_response(task)
        assert text == "Second reply"

    def test_task_history_skips_trailing_user_messages(self):
        agent_msg = _make_message("Agent reply")
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("User follow up")],
        )
        task = _make_task(history=[agent_msg, user_msg])
        text, _ = extract_response(task)
        assert text == "Agent reply"

    def test_empty_task(self):
        task = _make_task()
        text, ctx = extract_response(task)
        assert text == ""
        assert ctx == "ctx-123"


class TestTaskFieldPriority:
    """Tests for extract_response() field priority: artifacts > status > history."""

    def test_artifacts_win_over_status(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("From artifact")],
        )
        status_msg = _make_message("From status")
        task = _make_task(artifacts=[artifact], status_message=status_msg)
        text, _ = extract_response(task)
        assert text == "From artifact"

    def test_status_wins_over_history(self):
        status_msg = _make_message("From status")
        agent_msg = _make_message("From history")
        task = _make_task(status_message=status_msg, history=[agent_msg])
        text, _ = extract_response(task)
        assert text == "From status"

    def test_artifacts_win_over_all(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("From artifact")],
        )
        status_msg = _make_message("From status")
        agent_msg = _make_message("From history")
        task = _make_task(
            artifacts=[artifact],
            status_message=status_msg,
            history=[agent_msg],
        )
        text, _ = extract_response(task)
        assert text == "From artifact"
