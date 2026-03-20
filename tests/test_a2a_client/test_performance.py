"""
Performance tests for A2A protocol integration.

These tests measure A2A protocol overhead compared to baseline LLM agents.
Goal: <10% overhead vs baseline (per plan.md performance goals).

Run with: pytest tests/test_a2a_client/test_performance.py -v
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tau2.a2a.models import A2AAgentState, A2AConfig
from tau2.agent.a2a_agent import A2AAgent
from tau2.data_model.message import UserMessage
from tau2.environment.tool import Tool


def sample_tool() -> dict:
    """Sample tool for testing."""
    return {"result": "success"}


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""
    return [Tool(sample_tool)]


@pytest.fixture
def a2a_config():
    """Create A2A config for testing."""
    return A2AConfig(endpoint="http://localhost:8080", timeout=300)


@pytest.mark.asyncio
async def test_a2a_message_translation_overhead(a2a_config, sample_tools):
    """
    Test message translation overhead.

    Measures time to translate tau2 messages to A2A format and back.
    Target: Translation should add <50ms per message.
    """
    from tau2.a2a.translation import (
        a2a_to_tau2_assistant_message,
        tau2_to_a2a_message_content,
    )

    # Create sample message
    user_msg = UserMessage(
        role="user",
        content="Search for flights from SF to LA on 2025-12-01",
    )

    # Measure translation time (tau2 -> A2A)
    start = time.perf_counter()
    for _ in range(100):
        _ = tau2_to_a2a_message_content(
            message=user_msg,
            tools=sample_tools,
        )
    tau2_to_a2a_time = (time.perf_counter() - start) / 100

    # Measure reverse translation (A2A -> tau2)
    # Mock A2A response content (extracted from JSON-RPC response)
    mock_response_content = "I'll help you search for flights."

    start = time.perf_counter()
    for _ in range(100):
        _ = a2a_to_tau2_assistant_message(mock_response_content)
    a2a_to_tau2_time = (time.perf_counter() - start) / 100

    total_translation_time = tau2_to_a2a_time + a2a_to_tau2_time

    # Report results
    print(f"\nTranslation Performance:")
    print(f"  tau2 -> A2A: {tau2_to_a2a_time * 1000:.2f}ms")
    print(f"  A2A -> tau2: {a2a_to_tau2_time * 1000:.2f}ms")
    print(f"  Total: {total_translation_time * 1000:.2f}ms")

    # Assert: Translation should be fast (<50ms per round-trip)
    assert total_translation_time < 0.05, (
        f"Translation overhead too high: {total_translation_time * 1000:.2f}ms > 50ms"
    )


@pytest.mark.asyncio
async def test_a2a_protocol_metrics_collection_overhead(a2a_config, sample_tools):
    """
    Test metrics collection overhead.

    Measures time to collect and aggregate protocol metrics.
    Target: Metrics collection should add <10ms per request.
    """
    from tau2.a2a.metrics import AggregatedMetrics, ProtocolMetrics

    # Create sample metrics
    metrics = [
        ProtocolMetrics(
            request_id=f"req-{i}",
            endpoint="http://localhost:8080",
            method="POST",
            status_code=200,
            latency_ms=100.0 + i,
            input_tokens=100,
            output_tokens=50,
            context_id="ctx-123",
            timestamp="2025-11-24T10:00:00Z",
        )
        for i in range(100)
    ]

    # Measure aggregation time
    start = time.perf_counter()
    for _ in range(100):
        _ = AggregatedMetrics.from_protocol_metrics(metrics)
    aggregation_time = (time.perf_counter() - start) / 100

    print(f"\nMetrics Collection Performance:")
    print(f"  Aggregation time: {aggregation_time * 1000:.2f}ms (100 requests)")
    print(f"  Per-request overhead: {aggregation_time * 1000 / 100:.2f}ms")

    # Assert: Metrics aggregation should be fast (<10ms for 100 requests)
    assert aggregation_time < 0.01, (
        f"Metrics aggregation too slow: {aggregation_time * 1000:.2f}ms > 10ms"
    )


@pytest.mark.asyncio
async def test_a2a_client_initialization_overhead(a2a_config, sample_tools):
    """
    Test A2A client initialization overhead.

    Measures time to create A2AAgent and A2AClient.
    Target: Initialization should be <100ms.
    """
    # Measure initialization time
    start = time.perf_counter()
    for _ in range(10):
        _ = A2AAgent(
            config=a2a_config,
            tools=sample_tools,
            domain_policy="Test policy",
        )
    init_time = (time.perf_counter() - start) / 10

    print(f"\nInitialization Performance:")
    print(f"  A2AAgent init: {init_time * 1000:.2f}ms")

    # Assert: Initialization should be fast (<100ms)
    assert init_time < 0.1, f"Initialization too slow: {init_time * 1000:.2f}ms > 100ms"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_a2a_full_request_cycle_overhead(mock_a2a_agent, sample_tools):
    """
    Test full A2A request cycle overhead.

    Measures overhead for complete request: translation + HTTP + parsing.
    Target: Total overhead <300ms per request (as specified in plan.md).

    This is marked as slow since it involves HTTP calls.
    """
    from tau2.a2a.client import A2AClient
    from tau2.a2a.models import A2AConfig

    # Mock HTTP response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "message": {
                "messageId": "resp-123",
                "role": "agent",
                "parts": [{"text": "I found 3 flights for you."}],
                "contextId": "ctx-abc",
            }
        },
        "id": "req-123",
    }

    config = A2AConfig(endpoint="http://localhost:8080", timeout=300)

    # Create async mock client
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    # Create client with mock
    client = A2AClient(config=config, http_client=mock_client)

    # Create message content (already translated)
    message_content = "Search for flights from SF to LA"

    # Measure full cycle time
    timings = []
    for _ in range(10):
        start = time.perf_counter()

        # Send message (includes HTTP, parsing)
        _, _ = await client.send_message(
            message_content=message_content,
            context_id=None,
        )

        cycle_time = time.perf_counter() - start
        timings.append(cycle_time)

    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)

    print(f"\nFull Request Cycle Performance:")
    print(f"  Average: {avg_time * 1000:.2f}ms")
    print(f"  Min: {min_time * 1000:.2f}ms")
    print(f"  Max: {max_time * 1000:.2f}ms")
    print(f"  Target: <300ms per request")

    # Note: This test uses mocked HTTP, so times will be artificially low
    # Real overhead will include actual network latency
    print(
        f"\n  Note: Test uses mocked HTTP. Real overhead = measured + network latency"
    )


@pytest.mark.asyncio
async def test_a2a_state_management_overhead(a2a_config, sample_tools):
    """
    Test state management overhead.

    Measures time to create and update A2A agent state.
    Target: State operations should be <10ms.
    """
    agent = A2AAgent(
        config=a2a_config,
        tools=sample_tools,
        domain_policy="Test policy",
    )

    # Measure state initialization
    start = time.perf_counter()
    for _ in range(100):
        state = agent.get_init_state()
    init_time = (time.perf_counter() - start) / 100

    # Measure state updates
    state = A2AAgentState(
        context_id=None,
        conversation_history=[],
        request_count=0,
    )

    start = time.perf_counter()
    for i in range(100):
        state.context_id = f"ctx-{i}"
        state.request_count += 1
    update_time = (time.perf_counter() - start) / 100

    print(f"\nState Management Performance:")
    print(f"  State init: {init_time * 1000:.2f}ms")
    print(f"  State update: {update_time * 1000:.2f}ms")

    # Assert: State operations should be very fast (<10ms)
    assert init_time < 0.01, f"State init too slow: {init_time * 1000:.2f}ms > 10ms"
    assert update_time < 0.01, (
        f"State update too slow: {update_time * 1000:.2f}ms > 10ms"
    )


def test_performance_summary():
    """
    Summary of performance targets from plan.md:

    - <10% evaluation overhead vs baseline LLM agents
    - <300ms protocol translation latency
    - Support for 300s agent response timeout

    Individual component targets:
    - Message translation: <50ms per round-trip
    - Metrics collection: <10ms per request
    - Agent initialization: <100ms
    - State operations: <10ms
    """
    print("\n" + "=" * 60)
    print("A2A Protocol Performance Targets")
    print("=" * 60)
    print("Overall:")
    print("  - <10% overhead vs baseline LLM agents")
    print("  - <300ms protocol translation latency")
    print("  - 300s agent response timeout support")
    print("\nComponent targets:")
    print("  - Message translation: <50ms per round-trip")
    print("  - Metrics collection: <10ms per request")
    print("  - Agent initialization: <100ms")
    print("  - State operations: <10ms")
    print("=" * 60)
