"""Unit tests for GeminiLiveProvider._parse_response.

Tests verify that events are parsed from the correct paths only,
after removing duplicate parsing paths per Gemini team guidance:
- Audio: only from response.data (not inline_data)
- Output transcript: only from server_content.output_transcription (not response.text)
- Function calls: only from response.tool_call (not model_turn.parts)

Uses SimpleNamespace mocks -- no real API calls needed.
"""

import asyncio
from types import SimpleNamespace

from tau2.voice.audio_native.gemini.events import (
    GeminiAudioDeltaEvent,
    GeminiFunctionCallDoneEvent,
    GeminiGoAwayEvent,
    GeminiInputTranscriptionEvent,
    GeminiInterruptionEvent,
    GeminiSessionResumptionEvent,
    GeminiTextDeltaEvent,
    GeminiTurnCompleteEvent,
    GeminiUnknownEvent,
)
from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider

# =============================================================================
# Helpers
# =============================================================================


def _make_provider() -> GeminiLiveProvider:
    """Create a GeminiLiveProvider without connecting (for unit-testing _parse_response)."""
    # Bypass __init__ auth detection by constructing minimally
    provider = object.__new__(GeminiLiveProvider)
    # Set the minimal attributes _parse_response relies on
    provider._current_item_id = "test_item_1"
    provider._go_away_received = False
    provider._reconnect_at_turn_boundary = False
    provider._reconnect_deadline = None
    provider._resumption_handle = None
    return provider


def _empty_response(**overrides) -> SimpleNamespace:
    """Create a minimal mock response with no data by default."""
    defaults = {
        "data": None,
        "inline_data": None,
        "text": None,
        "tool_call": None,
        "go_away": None,
        "session_resumption_update": None,
        "server_content": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_server_content(**overrides) -> SimpleNamespace:
    """Create a mock server_content object."""
    defaults = {
        "interrupted": None,
        "turn_complete": None,
        "output_transcription": None,
        "input_transcription": None,
        "model_turn": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# =============================================================================
# Correct parsing paths
# =============================================================================


class TestCorrectParsingPaths:
    """Each event type is parsed from its correct source."""

    def test_audio_from_data(self):
        """response.data produces GeminiAudioDeltaEvent."""
        provider = _make_provider()
        response = _empty_response(data=b"\x00\x01\x02\x03")

        events = provider._parse_response(response)

        audio_events = [e for e in events if isinstance(e, GeminiAudioDeltaEvent)]
        assert len(audio_events) == 1
        assert audio_events[0].data == b"\x00\x01\x02\x03"
        assert audio_events[0].item_id == "test_item_1"

    def test_transcript_from_output_transcription(self):
        """server_content.output_transcription produces GeminiTextDeltaEvent."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="Hello from Gemini")
        sc = _make_server_content(output_transcription=transcription)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        text_events = [e for e in events if isinstance(e, GeminiTextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello from Gemini"

    def test_function_call_from_tool_call(self):
        """response.tool_call produces GeminiFunctionCallDoneEvent."""
        provider = _make_provider()
        func_call = SimpleNamespace(
            id="fc_123", name="get_weather", args={"location": "NYC"}
        )
        tool_call = SimpleNamespace(function_calls=[func_call])
        response = _empty_response(tool_call=tool_call)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 1
        assert fc_events[0].call_id == "fc_123"
        assert fc_events[0].name == "get_weather"
        assert fc_events[0].arguments == {"location": "NYC"}

    def test_multiple_function_calls(self):
        """Multiple function calls in tool_call are all parsed."""
        provider = _make_provider()
        fc1 = SimpleNamespace(id="fc_1", name="func_a", args={"x": 1})
        fc2 = SimpleNamespace(id="fc_2", name="func_b", args={"y": 2})
        tool_call = SimpleNamespace(function_calls=[fc1, fc2])
        response = _empty_response(tool_call=tool_call)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 2
        assert fc_events[0].name == "func_a"
        assert fc_events[1].name == "func_b"

    def test_interruption_event(self):
        """server_content.interrupted=True produces GeminiInterruptionEvent."""
        provider = _make_provider()
        sc = _make_server_content(interrupted=True)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiInterruptionEvent)]) == 1

    def test_turn_complete_event(self):
        """server_content.turn_complete=True produces GeminiTurnCompleteEvent."""
        provider = _make_provider()
        sc = _make_server_content(turn_complete=True)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiTurnCompleteEvent)]) == 1

    def test_input_transcription(self):
        """server_content.input_transcription produces GeminiInputTranscriptionEvent."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="User said hello")
        sc = _make_server_content(input_transcription=transcription)
        response = _empty_response(server_content=sc)

        events = provider._parse_response(response)

        input_events = [
            e for e in events if isinstance(e, GeminiInputTranscriptionEvent)
        ]
        assert len(input_events) == 1
        assert input_events[0].transcript == "User said hello"

    def test_go_away_event(self):
        """response.go_away produces GeminiGoAwayEvent and sets reconnection state."""
        provider = _make_provider()
        go_away = SimpleNamespace(time_left=SimpleNamespace(seconds=30))
        response = _empty_response(go_away=go_away)

        async def _parse():
            return provider._parse_response(response)

        events = asyncio.run(_parse())

        go_events = [e for e in events if isinstance(e, GeminiGoAwayEvent)]
        assert len(go_events) == 1
        assert go_events[0].time_left_seconds == 30.0
        assert provider._go_away_received is True
        assert provider._reconnect_at_turn_boundary is True
        assert provider._reconnect_deadline is not None

    def test_session_resumption_event(self):
        """response.session_resumption_update produces GeminiSessionResumptionEvent."""
        provider = _make_provider()
        update = SimpleNamespace(new_handle="handle_abc123", resumable=True)
        response = _empty_response(session_resumption_update=update)

        events = provider._parse_response(response)

        sr_events = [e for e in events if isinstance(e, GeminiSessionResumptionEvent)]
        assert len(sr_events) == 1
        assert sr_events[0].new_handle == "handle_abc123"
        assert sr_events[0].resumable is True
        assert provider._resumption_handle == "handle_abc123"

    def test_empty_response_produces_unknown(self):
        """Response with no recognized fields produces GeminiUnknownEvent."""
        provider = _make_provider()
        response = _empty_response()

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiUnknownEvent)]) == 1


# =============================================================================
# No double-emit (regression tests for removed duplicate paths)
# =============================================================================


class TestNoDoubleEmit:
    """Removed duplicate paths must not re-emit events."""

    def test_audio_not_double_emitted(self):
        """When both data and inline_data are present, only one audio event."""
        provider = _make_provider()
        response = _empty_response(
            data=b"\x00\x01\x02\x03",
            inline_data=SimpleNamespace(data=b"\x04\x05\x06\x07"),
        )

        events = provider._parse_response(response)

        audio_events = [e for e in events if isinstance(e, GeminiAudioDeltaEvent)]
        assert len(audio_events) == 1
        assert audio_events[0].data == b"\x00\x01\x02\x03"

    def test_function_call_not_double_emitted(self):
        """When tool_call and model_turn.parts both have FCs, only tool_call emits."""
        provider = _make_provider()
        fc_tool = SimpleNamespace(id="fc_1", name="get_order", args={})
        tool_call = SimpleNamespace(function_calls=[fc_tool])
        fc_part = SimpleNamespace(id="fc_2", name="get_order", args={})
        part = SimpleNamespace(function_call=fc_part)
        sc = _make_server_content(model_turn=SimpleNamespace(parts=[part]))
        response = _empty_response(tool_call=tool_call, server_content=sc)

        events = provider._parse_response(response)

        fc_events = [e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]
        assert len(fc_events) == 1
        assert fc_events[0].call_id == "fc_1"

    def test_transcript_not_double_emitted(self):
        """When both response.text and output_transcription exist, only transcription emits."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="Correct transcript")
        sc = _make_server_content(output_transcription=transcription)
        response = _empty_response(text="Wrong text path", server_content=sc)

        events = provider._parse_response(response)

        text_events = [e for e in events if isinstance(e, GeminiTextDeltaEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Correct transcript"


# =============================================================================
# Combined response
# =============================================================================


class TestCombinedResponse:
    """Test a response with multiple fields set simultaneously."""

    def test_audio_and_transcription_and_tool_call(self):
        """Response with audio, output transcription, and tool call."""
        provider = _make_provider()
        transcription = SimpleNamespace(text="I found the order")
        sc = _make_server_content(output_transcription=transcription)
        fc = SimpleNamespace(id="fc_1", name="lookup_order", args={"id": "123"})
        tool_call = SimpleNamespace(function_calls=[fc])
        response = _empty_response(
            data=b"\x00" * 100, tool_call=tool_call, server_content=sc
        )

        events = provider._parse_response(response)

        assert len([e for e in events if isinstance(e, GeminiAudioDeltaEvent)]) == 1
        assert len([e for e in events if isinstance(e, GeminiTextDeltaEvent)]) == 1
        assert (
            len([e for e in events if isinstance(e, GeminiFunctionCallDoneEvent)]) == 1
        )
