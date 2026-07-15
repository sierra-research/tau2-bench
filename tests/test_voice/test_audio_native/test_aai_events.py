"""Tests for AAI event parsing and models."""

from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent,
    AAIAudioChunkEvent,
    AAIAudioDoneEvent,
    AAICancelledEvent,
    AAIConfigEvent,
    AAICustomEvent,
    AAIErrorEvent,
    AAIIdleTimeoutEvent,
    AAIReplyDoneEvent,
    AAIResetEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIToolCallDoneEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    AAIUserTranscriptEvent,
    parse_aai_event,
)


class TestAAIEventModels:
    """Test individual event model creation and field mapping."""

    def test_agent_transcript_event_parses_text(self) -> None:
        """Test that agent_transcript event parses text field."""
        data = {
            "type": "agent_transcript",
            "text": "Hello, how can I help?",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIAgentTranscriptEvent)
        assert event.text == "Hello, how can I help?"
        assert event.type == "agent_transcript"

    def test_tool_call_maps_camel_case_to_snake_case(self) -> None:
        """Test that toolCallId and toolName are mapped to tool_call_id and tool_name."""
        data = {
            "type": "tool_call",
            "toolCallId": "call_123",
            "toolName": "get_weather",
            "args": {"location": "NYC"},
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIToolCallEvent)
        assert event.tool_call_id == "call_123"
        assert event.tool_name == "get_weather"
        assert event.args == {"location": "NYC"}

    def test_user_transcript_maps_turn_order(self) -> None:
        """Test that turnOrder alias maps to turn_order field."""
        data = {
            "type": "user_transcript",
            "text": "What's the weather?",
            "turnOrder": 2,
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIUserTranscriptEvent)
        assert event.text == "What's the weather?"
        assert event.turn_order == 2

    def test_error_parses_code_and_message(self) -> None:
        """Test that error event parses code and message."""
        data = {
            "type": "error",
            "code": "INVALID_REQUEST",
            "message": "Missing required field",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIErrorEvent)
        assert event.code == "INVALID_REQUEST"
        assert event.message == "Missing required field"

    def test_unknown_type_returns_unknown_event(self) -> None:
        """Test that unknown event type returns AAIUnknownEvent."""
        data = {
            "type": "some_future_event",
            "foo": "bar",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIUnknownEvent)
        assert event.type == "some_future_event"
        assert event.raw == data

    def test_config_event(self) -> None:
        """Test config event parsing."""
        data = {
            "type": "config",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIConfigEvent)
        assert event.type == "config"

    def test_speech_started_event(self) -> None:
        """Test speech_started event."""
        data = {"type": "speech_started"}
        event = parse_aai_event(data)
        assert isinstance(event, AAISpeechStartedEvent)
        assert event.type == "speech_started"

    def test_speech_stopped_event(self) -> None:
        """Test speech_stopped event."""
        data = {"type": "speech_stopped"}
        event = parse_aai_event(data)
        assert isinstance(event, AAISpeechStoppedEvent)
        assert event.type == "speech_stopped"

    def test_tool_call_done_event(self) -> None:
        """Test tool_call_done event with result field."""
        data = {
            "type": "tool_call_done",
            "toolCallId": "call_123",
            "result": '{"temp": 72}',
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIToolCallDoneEvent)
        assert event.tool_call_id == "call_123"
        assert event.result == '{"temp": 72}'

    def test_reply_done_event(self) -> None:
        """Test reply_done event."""
        data = {"type": "reply_done"}
        event = parse_aai_event(data)
        assert isinstance(event, AAIReplyDoneEvent)
        assert event.type == "reply_done"

    def test_audio_done_event(self) -> None:
        """Test audio_done event."""
        data = {"type": "audio_done"}
        event = parse_aai_event(data)
        assert isinstance(event, AAIAudioDoneEvent)
        assert event.type == "audio_done"

    def test_cancelled_event(self) -> None:
        """Test cancelled event."""
        data = {"type": "cancelled"}
        event = parse_aai_event(data)
        assert isinstance(event, AAICancelledEvent)
        assert event.type == "cancelled"

    def test_reset_event(self) -> None:
        """Test reset event."""
        data = {"type": "reset"}
        event = parse_aai_event(data)
        assert isinstance(event, AAIResetEvent)
        assert event.type == "reset"

    def test_idle_timeout_event(self) -> None:
        """Test idle_timeout event."""
        data = {"type": "idle_timeout"}
        event = parse_aai_event(data)
        assert isinstance(event, AAIIdleTimeoutEvent)
        assert event.type == "idle_timeout"

    def test_custom_event(self) -> None:
        """Test custom_event parsing."""
        data = {
            "type": "custom_event",
            "event": "my_custom",
            "data": {"key": "value"},
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAICustomEvent)
        assert event.event == "my_custom"
        assert event.data == {"key": "value"}

    def test_tool_call_args_defaults_to_empty_dict(self) -> None:
        """Test that tool_call args defaults to empty dict."""
        data = {
            "type": "tool_call",
            "toolCallId": "call_456",
            "toolName": "list_items",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIToolCallEvent)
        assert event.args == {}

    def test_tool_call_done_result_defaults_to_empty_string(self) -> None:
        """Test that tool_call_done result defaults to empty string."""
        data = {
            "type": "tool_call_done",
            "toolCallId": "call_456",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIToolCallDoneEvent)
        assert event.result == ""

    def test_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored due to ConfigDict."""
        data = {
            "type": "agent_transcript",
            "text": "Response",
            "extra_field": "should_be_ignored",
            "another_extra": 123,
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIAgentTranscriptEvent)
        assert event.text == "Response"
        assert not hasattr(event, "extra_field")

    def test_audio_chunk_event_construction(self) -> None:
        """Test AAIAudioChunkEvent can be constructed directly (not from parse_aai_event)."""
        # AAIAudioChunkEvent is not produced by parse_aai_event, but constructed directly
        event = AAIAudioChunkEvent(pcm16=b"\x00\x01\x02\x03")
        assert event.pcm16 == b"\x00\x01\x02\x03"
        assert event.type == "audio_chunk"

    def test_user_transcript_turn_order_optional(self) -> None:
        """Test that turn_order is optional in user_transcript."""
        data = {
            "type": "user_transcript",
            "text": "Hello",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAIUserTranscriptEvent)
        assert event.turn_order is None

    def test_error_code_and_message_optional(self) -> None:
        """Test that error code and message are optional."""
        data = {"type": "error"}
        event = parse_aai_event(data)
        assert isinstance(event, AAIErrorEvent)
        assert event.code is None
        assert event.message is None

    def test_custom_event_data_optional(self) -> None:
        """Test that custom_event data is optional."""
        data = {
            "type": "custom_event",
            "event": "my_event",
        }
        event = parse_aai_event(data)
        assert isinstance(event, AAICustomEvent)
        assert event.data is None

    def test_parse_failure_returns_unknown_event(self) -> None:
        """Test that parse failures return AAIUnknownEvent with warning."""
        # Force a validation error by passing invalid type for a required field
        data = {
            "type": "tool_call",
            # Missing required fields, should fail validation
        }
        event = parse_aai_event(data)
        # Should gracefully return UnknownEvent
        assert isinstance(event, AAIUnknownEvent)
