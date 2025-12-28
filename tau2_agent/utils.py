"""Shared utilities for tau2_agent."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

# Fields to exclude from message serialization for tracing (too large for Datadog)
# Full data is preserved in EvaluationStore JSON files for debugging
_LARGE_MESSAGE_FIELDS = {"raw_data", "reasoning_content", "provider_specific_fields"}


def compact_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Create a compact version of a message for tracing output.

    Removes large fields like raw_data and reasoning_content that can
    exceed Datadog's 1MB span limit. Full data is preserved in
    EvaluationStore JSON files.

    Preserves: role, content, tool_calls, turn_idx, timestamp, cost, usage

    Args:
        msg: Full message dict from model_dump().

    Returns:
        Compact message with large fields removed.
    """
    result = {}
    for key, value in msg.items():
        if key in _LARGE_MESSAGE_FIELDS:
            continue
        if isinstance(value, dict):
            # Recursively remove large fields from nested dicts
            cleaned = {k: v for k, v in value.items() if k not in _LARGE_MESSAGE_FIELDS}
            if cleaned:
                result[key] = cleaned
        else:
            result[key] = value
    return result


def sanitize_float(value: float | None) -> float | None:
    """Convert NaN/Inf to None for JSON serialization compatibility.

    JSON does not support NaN or Infinity values. This function converts
    them to None to ensure valid JSON output.

    Args:
        value: A float value that may be NaN or Inf.

    Returns:
        The original value if valid, or None if NaN/Inf.
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sanitize_dict_floats(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize all float values in a dictionary.

    Args:
        data: Dictionary that may contain NaN/Inf float values.

    Returns:
        Dictionary with NaN/Inf values converted to None.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, float):
            result[key] = sanitize_float(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict_floats(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict_floats(item) if isinstance(item, dict)
                else sanitize_float(item) if isinstance(item, float)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


# =============================================================================
# SSE (Server-Sent Events) Parsing
# =============================================================================

# Regex to normalize line endings (CRLF, CR, LF) to LF
_LINE_ENDING_RE = re.compile(r"\r\n|\r")


@dataclass
class SSEEvent:
    """Represents a parsed Server-Sent Event.

    Attributes:
        event: The event type (from "event:" field), or None if not specified.
        data: The event data (from "data:" field(s)), joined with newlines.
    """

    data: str = ""
    event: str | None = None

    def json(self) -> dict[str, Any] | None:
        """Parse the data field as JSON.

        Returns:
            Parsed JSON dict, or None if data is empty or invalid JSON.
        """
        if not self.data:
            return None
        try:
            return json.loads(self.data)
        except json.JSONDecodeError:
            return None


@dataclass
class SSEParser:
    """Stateful SSE parser for handling chunked streaming input.

    This parser correctly handles:
    - Events split across multiple chunks
    - Multi-line data fields (multiple "data:" lines)
    - SSE comments (lines starting with ":")
    - Different line endings (LF, CR, CRLF)
    - Proper space stripping after field colons

    Usage:
        parser = SSEParser()

        # Feed chunks as they arrive
        for chunk in stream:
            for event in parser.feed(chunk):
                process(event)

        # Don't forget to flush at end of stream
        for event in parser.flush():
            process(event)
    """

    _buffer: str = field(default="", repr=False)

    def feed(self, chunk: str) -> list[SSEEvent]:
        """Feed a chunk of SSE data and return any complete events.

        Args:
            chunk: Raw SSE text chunk from the stream.

        Returns:
            List of complete SSEEvent objects parsed from the buffer.
            May be empty if no complete events are available yet.
        """
        if not chunk:
            return []

        # Normalize line endings to LF
        chunk = _LINE_ENDING_RE.sub("\n", chunk)
        self._buffer += chunk

        events: list[SSEEvent] = []

        # SSE events are delimited by blank lines (double newline)
        while "\n\n" in self._buffer:
            event_text, self._buffer = self._buffer.split("\n\n", 1)
            event = self._parse_event(event_text)
            if event is not None:
                events.append(event)

        return events

    def flush(self) -> list[SSEEvent]:
        """Flush any remaining buffered content as a final event.

        Call this at the end of the stream to handle events that
        don't have a trailing delimiter.

        Returns:
            List containing the final event, or empty if buffer is empty.
        """
        if not self._buffer.strip():
            self._buffer = ""
            return []

        event = self._parse_event(self._buffer)
        self._buffer = ""

        return [event] if event is not None else []

    def _parse_event(self, event_text: str) -> SSEEvent | None:
        """Parse a single SSE event block into an SSEEvent.

        Args:
            event_text: Raw text of a single SSE event (without delimiters).

        Returns:
            SSEEvent if the block contains data, None otherwise.
        """
        lines = event_text.strip().split("\n")
        event_type: str | None = None
        data_lines: list[str] = []

        for line in lines:
            if not line:
                continue

            # Comment lines (start with :) are ignored
            if line.startswith(":"):
                continue

            # Parse field:value
            if ":" in line:
                field_name, _, value = line.partition(":")
                # Per SSE spec: if value starts with space, remove first space only
                if value.startswith(" "):
                    value = value[1:]
            else:
                # Line without colon: field name is entire line, value is empty
                field_name = line
                value = ""

            # Handle known fields
            if field_name == "event":
                event_type = value
            elif field_name == "data":
                data_lines.append(value)
            # id and retry fields are ignored (not needed for our use case)

        # No data means no event
        if not data_lines:
            return None

        # Join multiple data lines with newlines per SSE spec
        return SSEEvent(
            event=event_type,
            data="\n".join(data_lines),
        )


def parse_sse_events(raw: str) -> list[SSEEvent]:
    """Parse a complete SSE text into a list of events.

    Convenience function for parsing complete SSE text (not streaming).
    Equivalent to feeding the entire text and flushing.

    Args:
        raw: Complete SSE text to parse.

    Returns:
        List of all SSEEvent objects in the text.
    """
    parser = SSEParser()
    events = parser.feed(raw)
    events.extend(parser.flush())
    return events
