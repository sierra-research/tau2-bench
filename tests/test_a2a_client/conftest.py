"""Test fixtures for A2A client integration tests."""

import json
import uuid
from typing import Any

import httpx
import pytest


class MockA2ATransport(httpx.MockTransport):
    """
    Mock HTTP transport for A2A agent testing.

    Simulates an A2A-compliant agent endpoint for testing without network calls.
    """

    def __init__(
        self,
        agent_name: str = "Test A2A Agent",
        agent_description: str = "A mock A2A agent for testing",
        context_id: str | None = None,
        should_fail: bool = False,
        fail_status: int = 500,
        fail_message: str = "Mock failure",
    ):
        """
        Initialize mock A2A transport.

        Args:
            agent_name: Name for the mock agent card
            agent_description: Description for the mock agent card
            context_id: Context ID to return (generated if None)
            should_fail: Whether requests should fail
            fail_status: HTTP status code for failures
            fail_message: Error message for failures
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.context_id = context_id or f"ctx-{uuid.uuid4().hex[:12]}"
        self.should_fail = should_fail
        self.fail_status = fail_status
        self.fail_message = fail_message
        self.request_count = 0

        super().__init__(self._handle_request)

    def _handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle mock HTTP requests."""
        self.request_count += 1

        # Check if should fail
        if self.should_fail:
            return httpx.Response(
                status_code=self.fail_status,
                json={"error": self.fail_message},
            )

        # Agent card discovery
        if request.url.path == "/.well-known/agent-card.json":
            return self._handle_agent_card(request)

        # A2A message/send endpoint (JSON-RPC)
        if request.method == "POST":
            return self._handle_message_send(request)

        # Unknown endpoint
        return httpx.Response(
            status_code=404,
            json={"error": "Not found"},
        )

    def _handle_agent_card(self, request: httpx.Request) -> httpx.Response:
        """Handle agent card discovery request."""
        agent_card = {
            "name": self.agent_name,
            "description": self.agent_description,
            "url": str(request.url.copy_with(path="")),
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "push_notifications": False,
            },
            "security_schemes": None,
            "security": None,
            "skills": [
                {
                    "id": "customer_service",
                    "name": "Customer Service",
                    "description": "Handle customer service inquiries",
                    "tags": ["support", "airline"],
                }
            ],
        }

        return httpx.Response(
            status_code=200,
            json=agent_card,
            headers={"content-type": "application/json"},
        )

    def _handle_message_send(self, request: httpx.Request) -> httpx.Response:
        """Handle A2A message/send request (JSON-RPC 2.0)."""
        try:
            # Parse JSON-RPC request
            rpc_request = json.loads(request.content)

            # Validate JSON-RPC structure
            if rpc_request.get("jsonrpc") != "2.0":
                return httpx.Response(
                    status_code=400,
                    json={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing jsonrpc version",
                        },
                        "id": rpc_request.get("id"),
                    },
                )

            if rpc_request.get("method") != "message/send":
                return httpx.Response(
                    status_code=400,
                    json={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {rpc_request.get('method')}",
                        },
                        "id": rpc_request.get("id"),
                    },
                )

            # Extract message from params
            message = rpc_request.get("params", {}).get("message", {})
            message_content = self._extract_message_content(message)

            # Generate mock response
            response_text = self._generate_response(message_content)

            # Build JSON-RPC response
            rpc_response = {
                "jsonrpc": "2.0",
                "id": rpc_request.get("id"),
                "result": {
                    "message": {
                        "messageId": f"msg-{uuid.uuid4()}",
                        "role": "agent",
                        "parts": [{"text": response_text}],
                        "contextId": self.context_id,
                    }
                },
            }

            return httpx.Response(
                status_code=200,
                json=rpc_response,
                headers={"content-type": "application/json"},
            )

        except (json.JSONDecodeError, KeyError) as e:
            return httpx.Response(
                status_code=400,
                json={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                    "id": None,
                },
            )

    def _extract_message_content(self, message: dict[str, Any]) -> str:
        """Extract text content from A2A message parts."""
        parts = message.get("parts", [])
        text_parts = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

        return "\n".join(text_parts)

    def _generate_response(self, message_content: str) -> str:
        """
        Generate mock agent response based on input.

        Provides simple pattern matching for common test scenarios.
        """
        content_lower = message_content.lower()

        # Tool call request - search flights
        if "search_flights" in content_lower or "flight from" in content_lower:
            return json.dumps(
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

        # Tool call request - book flight
        if "book_flight" in content_lower or "book the flight" in content_lower:
            return json.dumps(
                {
                    "tool_call": {
                        "name": "book_flight",
                        "arguments": {
                            "flight_id": "AA123",
                            "passenger_info": {
                                "name": "John Doe",
                                "email": "john@example.com",
                            },
                        },
                    }
                }
            )

        # Tool result acknowledgment
        if "tool result" in content_lower or "tool output" in content_lower:
            return "Thank you for the information. I'll proceed with helping you."

        # Default response
        return "I understand. How can I help you today?"


@pytest.fixture
def mock_a2a_agent():
    """Fixture providing a mock A2A agent transport."""
    return MockA2ATransport(
        agent_name="Test Airline Agent",
        agent_description="Mock airline customer service agent for testing",
    )


@pytest.fixture
def mock_a2a_client(mock_a2a_agent):
    """Fixture providing httpx AsyncClient with mock A2A agent."""
    client = httpx.AsyncClient(
        transport=mock_a2a_agent,
        base_url="http://test-agent.example.com",
    )
    return client


@pytest.fixture
def failing_a2a_agent():
    """Fixture providing a failing mock A2A agent."""
    return MockA2ATransport(
        should_fail=True,
        fail_status=500,
        fail_message="Internal server error",
    )


@pytest.fixture
def unauthorized_a2a_agent():
    """Fixture providing an unauthorized mock A2A agent."""
    return MockA2ATransport(
        should_fail=True,
        fail_status=401,
        fail_message="Unauthorized",
    )


@pytest.fixture
def timeout_a2a_agent():
    """Fixture providing a timeout mock A2A agent."""
    return MockA2ATransport(
        should_fail=True,
        fail_status=408,
        fail_message="Request timeout",
    )
