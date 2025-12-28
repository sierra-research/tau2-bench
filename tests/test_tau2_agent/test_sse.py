"""Tests for SSE parsing utilities.

These tests validate the SSEParser class handles all edge cases correctly:
- Standard SSE format with event: and data: fields
- Multi-line data fields
- Chunked input (events split across chunks)
- Multiple events in single chunk
- Comment lines (: prefix)
- Buffer state preservation between chunks
- End of stream handling
"""

import json

import pytest

from tau2_agent.utils import SSEEvent, SSEParser, parse_sse_events


class TestSSEEvent:
    """Tests for SSEEvent dataclass."""

    def test_event_with_all_fields(self):
        """SSEEvent stores event type and data."""
        event = SSEEvent(event="message", data='{"foo": "bar"}')
        assert event.event == "message"
        assert event.data == '{"foo": "bar"}'

    def test_event_data_only(self):
        """SSEEvent can have data without event type."""
        event = SSEEvent(data='{"status": "ok"}')
        assert event.event is None
        assert event.data == '{"status": "ok"}'

    def test_json_method(self):
        """SSEEvent.json() parses data as JSON."""
        event = SSEEvent(event="message", data='{"count": 42}')
        assert event.json() == {"count": 42}

    def test_json_method_invalid(self):
        """SSEEvent.json() returns None for invalid JSON."""
        event = SSEEvent(data="not json")
        assert event.json() is None

    def test_json_method_empty(self):
        """SSEEvent.json() returns None for empty data."""
        event = SSEEvent(data="")
        assert event.json() is None


class TestSSEParserBasic:
    """Basic SSE parsing tests."""

    def test_single_event_with_type(self):
        """Parse a single event with event type and data."""
        parser = SSEParser()
        events = parser.feed('event: message\ndata: {"status": "ok"}\n\n')

        assert len(events) == 1
        assert events[0].event == "message"
        assert events[0].data == '{"status": "ok"}'

    def test_single_event_data_only(self):
        """Parse a single event with only data field."""
        parser = SSEParser()
        events = parser.feed('data: {"status": "ok"}\n\n')

        assert len(events) == 1
        assert events[0].event is None
        assert events[0].data == '{"status": "ok"}'

    def test_multiple_events_single_chunk(self):
        """Parse multiple events from a single chunk."""
        parser = SSEParser()
        chunk = (
            'event: start\ndata: {"id": 1}\n\n'
            'event: progress\ndata: {"id": 2}\n\n'
            'event: done\ndata: {"id": 3}\n\n'
        )
        events = parser.feed(chunk)

        assert len(events) == 3
        assert events[0].event == "start"
        assert events[1].event == "progress"
        assert events[2].event == "done"

    def test_empty_chunk(self):
        """Empty chunk returns no events."""
        parser = SSEParser()
        events = parser.feed("")
        assert events == []

    def test_whitespace_only_chunk(self):
        """Whitespace-only chunk returns no events."""
        parser = SSEParser()
        events = parser.feed("   \n   ")
        assert events == []


class TestSSEParserMultiLineData:
    """Tests for multi-line data handling."""

    def test_multiline_data_concatenation(self):
        """Multiple data: lines are concatenated with newlines."""
        parser = SSEParser()
        events = parser.feed('data: line1\ndata: line2\ndata: line3\n\n')

        assert len(events) == 1
        assert events[0].data == "line1\nline2\nline3"

    def test_multiline_json_data(self):
        """Multi-line JSON data is handled correctly."""
        parser = SSEParser()
        # JSON split across multiple data: lines
        events = parser.feed('data: {"key":\ndata: "value"}\n\n')

        assert len(events) == 1
        # Data lines are joined with newlines
        assert events[0].data == '{"key":\n"value"}'

    def test_event_type_before_multiline_data(self):
        """Event type is preserved with multi-line data."""
        parser = SSEParser()
        events = parser.feed('event: update\ndata: part1\ndata: part2\n\n')

        assert len(events) == 1
        assert events[0].event == "update"
        assert events[0].data == "part1\npart2"


class TestSSEParserChunkedInput:
    """Tests for handling events split across chunks."""

    def test_event_split_at_delimiter(self):
        """Event split at \\n\\n delimiter is reassembled."""
        parser = SSEParser()

        # First chunk ends before delimiter
        events1 = parser.feed('event: message\ndata: {"id": 1}')
        assert events1 == []

        # Second chunk contains delimiter
        events2 = parser.feed('\n\n')
        assert len(events2) == 1
        assert events2[0].event == "message"

    def test_event_split_mid_line(self):
        """Event split mid-line is reassembled correctly."""
        parser = SSEParser()

        events1 = parser.feed('event: mess')
        assert events1 == []

        events2 = parser.feed('age\ndata: {"status": "ok"}\n\n')
        assert len(events2) == 1
        assert events2[0].event == "message"
        assert events2[0].data == '{"status": "ok"}'

    def test_multiple_chunks_multiple_events(self):
        """Multiple events across multiple chunks."""
        parser = SSEParser()

        events1 = parser.feed('event: start\ndata: 1\n\nevent: ')
        assert len(events1) == 1
        assert events1[0].event == "start"

        events2 = parser.feed('progress\ndata: 2\n\nevent: done\ndata: 3')
        assert len(events2) == 1
        assert events2[0].event == "progress"

        events3 = parser.flush()
        assert len(events3) == 1
        assert events3[0].event == "done"

    def test_tiny_chunks(self):
        """Handle character-by-character streaming."""
        parser = SSEParser()
        full_event = 'data: ok\n\n'

        all_events = []
        for char in full_event:
            all_events.extend(parser.feed(char))

        assert len(all_events) == 1
        assert all_events[0].data == "ok"


class TestSSEParserComments:
    """Tests for SSE comment handling."""

    def test_comment_line_ignored(self):
        """Lines starting with : are ignored (SSE comments)."""
        parser = SSEParser()
        events = parser.feed(': this is a comment\ndata: real data\n\n')

        assert len(events) == 1
        assert events[0].data == "real data"

    def test_multiple_comments(self):
        """Multiple comment lines are all ignored."""
        parser = SSEParser()
        events = parser.feed(
            ': comment 1\n'
            ': comment 2\n'
            'data: actual data\n'
            ': another comment\n'
            '\n'
        )

        assert len(events) == 1
        assert events[0].data == "actual data"

    def test_comment_only_event(self):
        """Event with only comments produces no output."""
        parser = SSEParser()
        events = parser.feed(': just a comment\n\n')

        assert events == []


class TestSSEParserFlush:
    """Tests for end-of-stream handling."""

    def test_flush_incomplete_event(self):
        """flush() returns buffered incomplete event."""
        parser = SSEParser()
        parser.feed('event: final\ndata: {"last": true}')

        events = parser.flush()
        assert len(events) == 1
        assert events[0].event == "final"
        assert events[0].data == '{"last": true}'

    def test_flush_empty_buffer(self):
        """flush() with empty buffer returns empty list."""
        parser = SSEParser()
        events = parser.flush()
        assert events == []

    def test_flush_whitespace_buffer(self):
        """flush() with whitespace-only buffer returns empty list."""
        parser = SSEParser()
        parser.feed('   \n   ')
        events = parser.flush()
        assert events == []

    def test_double_flush(self):
        """Second flush() returns empty (buffer cleared)."""
        parser = SSEParser()
        parser.feed('data: test')

        events1 = parser.flush()
        assert len(events1) == 1

        events2 = parser.flush()
        assert events2 == []


class TestSSEParserEdgeCases:
    """Edge cases and spec compliance tests."""

    def test_space_after_colon_stripped(self):
        """Single space after colon is stripped per SSE spec."""
        parser = SSEParser()
        events = parser.feed('data: value\n\n')
        assert events[0].data == "value"

        parser2 = SSEParser()
        events2 = parser2.feed('data:value\n\n')
        assert events2[0].data == "value"

    def test_multiple_spaces_preserved(self):
        """Only first space after colon is stripped."""
        parser = SSEParser()
        events = parser.feed('data:  two spaces\n\n')
        # First space stripped, second preserved
        assert events[0].data == " two spaces"

    def test_empty_data_line(self):
        """Empty data: line adds empty string."""
        parser = SSEParser()
        events = parser.feed('data:\n\n')
        assert events[0].data == ""

    def test_unknown_field_ignored(self):
        """Unknown fields are ignored per SSE spec."""
        parser = SSEParser()
        events = parser.feed('unknown: value\ndata: real\n\n')

        assert len(events) == 1
        assert events[0].data == "real"

    def test_id_field_ignored(self):
        """id field is parsed but not exposed (not needed for our use case)."""
        parser = SSEParser()
        events = parser.feed('id: 123\ndata: test\n\n')

        assert len(events) == 1
        assert events[0].data == "test"
        # We don't expose id since we don't need reconnection support

    def test_retry_field_ignored(self):
        """retry field is ignored (not needed for our use case)."""
        parser = SSEParser()
        events = parser.feed('retry: 5000\ndata: test\n\n')

        assert len(events) == 1
        assert events[0].data == "test"

    def test_crlf_handling(self):
        """CRLF line endings are handled correctly."""
        parser = SSEParser()
        events = parser.feed('event: test\r\ndata: value\r\n\r\n')

        assert len(events) == 1
        assert events[0].event == "test"
        assert events[0].data == "value"

    def test_mixed_line_endings(self):
        """Mixed line endings (LF, CR, CRLF) are handled."""
        parser = SSEParser()
        events = parser.feed('data: line1\ndata: line2\r\ndata: line3\r\n\n')

        assert len(events) == 1
        assert events[0].data == "line1\nline2\nline3"


class TestSSEParserRealWorldScenarios:
    """Tests based on actual A2A/ADK event patterns."""

    def test_adk_progress_event(self):
        """Parse ADK-style progress event."""
        parser = SSEParser()
        event_data = {
            "jsonrpc": "2.0",
            "method": "message",
            "params": {
                "message": {"parts": [{"text": "Working..."}]},
                "metadata": {"tau2.state": "working", "tau2.progress": 50},
            },
        }
        events = parser.feed(f'event: message\ndata: {json.dumps(event_data)}\n\n')

        assert len(events) == 1
        assert events[0].event == "message"
        parsed = events[0].json()
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["params"]["metadata"]["tau2.progress"] == 50

    def test_adk_error_event(self):
        """Parse ADK-style error event."""
        parser = SSEParser()
        event_data = {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": "Evaluation failed"},
        }
        events = parser.feed(f'event: error\ndata: {json.dumps(event_data)}\n\n')

        assert len(events) == 1
        assert events[0].event == "error"
        parsed = events[0].json()
        assert parsed["error"]["code"] == -32000

    def test_streaming_sequence(self):
        """Simulate realistic streaming sequence."""
        parser = SSEParser()
        all_events = []

        # Submitted
        all_events.extend(
            parser.feed('event: message\ndata: {"state": "submitted"}\n\n')
        )

        # Working (chunked)
        all_events.extend(parser.feed('event: message\ndata: {"state": "wor'))
        all_events.extend(parser.feed('king", "progress": 25}\n\n'))

        # More progress
        all_events.extend(parser.feed('event: message\ndata: {"state": "working", '))
        all_events.extend(parser.feed('"progress": 75}\n\n'))

        # Completed (at end of stream, no trailing delimiter)
        parser.feed('event: message\ndata: {"state": "completed"}')
        all_events.extend(parser.flush())

        assert len(all_events) == 4
        states = [e.json()["state"] for e in all_events]
        assert states == ["submitted", "working", "working", "completed"]


class TestParseSSEEventsHelper:
    """Tests for the parse_sse_events convenience function."""

    def test_basic_usage(self):
        """parse_sse_events handles simple case."""
        raw = 'event: test\ndata: {"ok": true}\n\n'
        events = parse_sse_events(raw)

        assert len(events) == 1
        assert events[0].event == "test"

    def test_multiple_events(self):
        """parse_sse_events handles multiple events."""
        raw = 'data: one\n\ndata: two\n\ndata: three\n\n'
        events = parse_sse_events(raw)

        assert len(events) == 3
        assert [e.data for e in events] == ["one", "two", "three"]

    def test_incomplete_event_included(self):
        """parse_sse_events includes incomplete final event."""
        raw = 'data: complete\n\ndata: incomplete'
        events = parse_sse_events(raw)

        assert len(events) == 2
        assert events[1].data == "incomplete"
