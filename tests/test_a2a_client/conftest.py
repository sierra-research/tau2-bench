"""Test fixtures for A2A client integration tests."""

import json
import uuid
from typing import Any

import httpx
import pytest
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.client.client_factory import minimal_agent_card
from a2a.utils.message import new_agent_text_message


def create_mock_sdk_client(
    transport: httpx.MockTransport,
    endpoint: str = "http://test-agent.example.com",
) -> Client:
    """Create an SDK Client backed by a mock httpx transport.

    Args:
        transport: Mock httpx transport that handles HTTP requests
        endpoint: Agent endpoint URL

    Returns:
        SDK Client using the mock transport
    """
    httpx_client = httpx.AsyncClient(
        transport=transport,
        base_url=endpoint,
    )
    card = minimal_agent_card(endpoint)
    factory = ClientFactory(
        ClientConfig(httpx_client=httpx_client, streaming=False),
    )
    return factory.create(card)


def build_agent_card_json(
    name: str = "Test A2A Agent",
    description: str = "A mock A2A agent for testing",
    url: str = "http://test-agent.example.com",
) -> dict[str, Any]:
    """Build a valid SDK AgentCard-compatible JSON dict.

    Includes all required fields for a2a.types.AgentCard validation.
    """
    return {
        "name": name,
        "description": description,
        "url": url,
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "customer_service",
                "name": "Customer Service",
                "description": "Handle customer service inquiries",
                "tags": ["support", "airline"],
            }
        ],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }


def build_rpc_message_response(
    text: str,
    request_id: str | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC response containing an SDK-valid Message.

    Uses Format 2 (direct Message response) which is the standard
    SDK server response format.
    """
    msg = new_agent_text_message(text, context_id=context_id)
    result_dict = msg.model_dump(by_alias=True, mode="json", exclude_none=True)
    return {
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "result": result_dict,
    }


class MockA2ATransport(httpx.MockTransport):
    """
    Mock HTTP transport for A2A agent testing.

    Simulates an A2A-compliant agent endpoint, returning SDK-valid
    Message responses (Format 2).
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

        return httpx.Response(status_code=404, json={"error": "Not found"})

    def _handle_agent_card(self, request: httpx.Request) -> httpx.Response:
        """Handle agent card discovery request."""
        card_json = build_agent_card_json(
            name=self.agent_name,
            description=self.agent_description,
            url=str(request.url.copy_with(path="")),
        )
        return httpx.Response(
            status_code=200,
            json=card_json,
            headers={"content-type": "application/json"},
        )

    def _handle_message_send(self, request: httpx.Request) -> httpx.Response:
        """Handle A2A message/send request (JSON-RPC 2.0)."""
        try:
            rpc_request = json.loads(request.content)

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

            message = rpc_request.get("params", {}).get("message", {})
            message_content = self._extract_message_content(message)
            response_text = self._generate_response(message_content)

            rpc_response = build_rpc_message_response(
                text=response_text,
                request_id=rpc_request.get("id"),
                context_id=self.context_id,
            )

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
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
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
        """Generate mock agent response based on input."""
        content_lower = message_content.lower()

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

        if "tool result" in content_lower or "tool output" in content_lower:
            return "Thank you for the information. I'll proceed with helping you."

        return "I understand. How can I help you today?"


@pytest.fixture
def mock_a2a_agent():
    """Fixture providing a mock A2A agent transport."""
    return MockA2ATransport(
        agent_name="Test Airline Agent",
        agent_description="Mock airline customer service agent for testing",
    )


@pytest.fixture
def mock_a2a_client(mock_a2a_agent) -> Client:
    """Fixture providing an SDK Client with mock A2A agent."""
    return create_mock_sdk_client(mock_a2a_agent)


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
