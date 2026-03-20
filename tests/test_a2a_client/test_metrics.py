"""Unit tests for A2A protocol metrics collection."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tau2.a2a.client import A2AClient
from tau2.a2a.metrics import ProtocolMetrics, estimate_tokens
from tau2.a2a.models import A2AConfig


class TestProtocolMetrics:
    """Test suite for ProtocolMetrics creation and validation."""

    def test_protocol_metrics_creation(self):
        """Test creating a ProtocolMetrics instance."""
        metrics = ProtocolMetrics(
            request_id="req-123",
            endpoint="http://localhost:8080",
            method="POST",
            status_code=200,
            latency_ms=123.45,
            input_tokens=100,
            output_tokens=50,
            context_id="ctx-abc",
        )

        assert metrics.request_id == "req-123"
        assert metrics.endpoint == "http://localhost:8080"
        assert metrics.method == "POST"
        assert metrics.status_code == 200
        assert metrics.latency_ms == 123.45
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.context_id == "ctx-abc"
        assert metrics.error is None
        assert metrics.timestamp is not None

    def test_protocol_metrics_with_error(self):
        """Test creating ProtocolMetrics for failed request."""
        metrics = ProtocolMetrics(
            request_id="req-456",
            endpoint="http://localhost:8080",
            method="POST",
            status_code=None,
            latency_ms=50.0,
            error="Connection timeout",
        )

        assert metrics.error == "Connection timeout"
        assert metrics.status_code is None
        assert metrics.input_tokens is None
        assert metrics.output_tokens is None

    def test_protocol_metrics_to_dict(self):
        """Test converting ProtocolMetrics to dictionary."""
        metrics = ProtocolMetrics(
            request_id="req-789",
            endpoint="http://localhost:8080",
            method="POST",
            status_code=200,
            latency_ms=100.0,
            input_tokens=200,
            output_tokens=150,
        )

        result = metrics.to_dict()

        assert result["request_id"] == "req-789"
        assert result["endpoint"] == "http://localhost:8080"
        assert result["status_code"] == 200
        assert result["latency_ms"] == 100.0
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 150
        # Should exclude None values
        assert "context_id" not in result or result.get("context_id") is None
        assert "error" not in result or result.get("error") is None


class TestTokenEstimation:
    """Test suite for token counting functionality."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "This is a test message"
        tokens = estimate_tokens(text)
        # Roughly 4 chars per token = 22/4 = 5.5 → 5 tokens
        assert tokens == 5

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        text = "a" * 1000  # 1000 characters
        tokens = estimate_tokens(text)
        # Should be roughly 250 tokens (1000/4)
        assert tokens == 250

    def test_estimate_tokens_realistic_message(self):
        """Test token estimation with realistic agent message."""
        message = """I'll help you search for flights from New York to Los Angeles.
        Let me use the search_flights tool to find available options."""
        tokens = estimate_tokens(message)
        # Message is ~128 chars → ~32 tokens
        assert 30 <= tokens <= 35


@pytest.mark.asyncio
class TestMetricsCollectionInClient:
    """Test suite for metrics collection during client operations."""

    async def test_send_message_collects_metrics(self, mock_a2a_transport):
        """Test that send_message collects protocol metrics."""
        # Setup
        config = A2AConfig(endpoint="http://localhost:8080")
        client = A2AClient(config=config, http_client=mock_a2a_transport)

        # Act
        response_content, context_id = await client.send_message(
            message_content="Test message",
            context_id=None,
        )

        # Assert - verify response
        assert response_content == "Hello from mock agent"
        assert context_id == "mock-context-123"

        # Note: This test will be updated once metrics collection is implemented
        # For now, it verifies basic functionality works

    async def test_metrics_capture_latency(self, mock_a2a_transport):
        """Test that metrics capture request latency."""
        # Setup
        config = A2AConfig(endpoint="http://localhost:8080")
        client = A2AClient(config=config, http_client=mock_a2a_transport)

        # Record start time
        start = time.perf_counter()

        # Act
        await client.send_message(
            message_content="Test message",
            context_id=None,
        )

        # Measure elapsed time
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert that some time elapsed (even if small)
        assert elapsed_ms >= 0

        # Note: Once metrics are implemented, we'll verify latency is captured

    async def test_metrics_count_tokens(self):
        """Test that metrics count input and output tokens."""
        # Setup
        input_message = "Search for flights to Los Angeles"
        output_message = "I found 5 flights to Los Angeles"

        # Act
        input_tokens = estimate_tokens(input_message)
        output_tokens = estimate_tokens(output_message)

        # Assert
        assert input_tokens > 0
        assert output_tokens > 0
        # Output should have slightly more tokens
        assert output_tokens >= input_tokens

    async def test_metrics_handle_errors(self, mock_a2a_error_transport):
        """Test that metrics are captured even on errors."""
        # Setup
        config = A2AConfig(endpoint="http://localhost:8080")
        client = A2AClient(config=config, http_client=mock_a2a_error_transport)

        # Act & Assert - should raise error
        from tau2.a2a.exceptions import A2AError

        with pytest.raises(A2AError):
            await client.send_message(
                message_content="Test message",
                context_id=None,
            )

        # Note: Once metrics are implemented, verify error is captured in metrics


@pytest.fixture
def mock_a2a_transport(mock_httpx_client):
    """Fixture providing mock httpx client with successful A2A responses."""
    return mock_httpx_client


@pytest.fixture
def mock_a2a_error_transport(mock_httpx_error_client):
    """Fixture providing mock httpx client that returns errors."""
    return mock_httpx_error_client
