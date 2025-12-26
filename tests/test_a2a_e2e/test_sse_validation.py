"""E2E tests for SSE streaming format validation."""

import pytest

pytestmark = pytest.mark.a2a_e2e

VALID_EVENT_TYPES = {"message", "error", "done", None}


def test_sse_event_format(evaluation_events: list[dict]):
    """Verify SSE events are well-formed with valid types and JSON-RPC structure."""
    assert len(evaluation_events) > 0, "Should receive SSE events"

    # No parse failures (_raw indicates failure)
    malformed = [e for e in evaluation_events if "_raw" in e]
    assert len(malformed) == 0, f"Malformed events: {malformed}"

    # All event types are valid
    for event in evaluation_events:
        event_type = event.get("_event_type")
        assert event_type in VALID_EVENT_TYPES, f"Invalid type: {event_type}"

    # At least one event has JSON-RPC structure
    jsonrpc_events = [
        e for e in evaluation_events if "jsonrpc" in e or "result" in e or "error" in e
    ]
    assert len(jsonrpc_events) > 0, "Should have JSON-RPC structured events"

    # Validate JSON-RPC version where present
    for event in jsonrpc_events:
        if "jsonrpc" in event:
            assert event["jsonrpc"] == "2.0", f"Expected 2.0: {event.get('jsonrpc')}"


def test_sse_stream_completion(evaluation_events: list[dict]):
    """Verify stream completes with result/error, consistent IDs, valid message."""
    assert len(evaluation_events) > 0, "Should receive SSE events"

    # Stream should complete with result, error, or done
    has_result = any("result" in e for e in evaluation_events)
    has_error = any("error" in e for e in evaluation_events)
    has_done = any(e.get("_event_type") == "done" for e in evaluation_events)
    assert has_result or has_error or has_done, "No completion indicator"

    # All request IDs should be consistent (max 1 unique ID)
    request_ids = {e["id"] for e in evaluation_events if "id" in e}
    assert len(request_ids) <= 1, f"Inconsistent IDs: {request_ids}"

    # Result events should have valid message content
    for event in evaluation_events:
        if "result" not in event:
            continue
        result = event["result"]
        assert isinstance(result, dict), f"Result not dict: {type(result)}"
        has_content = any(
            k in result
            for k in [
                "message",
                "parts",
                "artifact",
                "artifacts",
                "status",
                "history",
                "kind",
            ]
        )
        assert has_content, f"Result lacks message content: {list(result.keys())}"
