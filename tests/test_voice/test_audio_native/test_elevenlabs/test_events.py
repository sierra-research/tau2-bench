import base64

from tau2.voice.audio_native.elevenlabs.events import (
    ElevenLabsAgentResponseEvent,
    ElevenLabsAudioEvent,
    ElevenLabsClientToolCallEvent,
    ElevenLabsConversationMetadataEvent,
    ElevenLabsUnknownEvent,
    parse_elevenlabs_event,
)


def test_parse_conversation_metadata():
    event = parse_elevenlabs_event(
        {
            "type": "conversation_initiation_metadata",
            "conversation_initiation_metadata_event": {
                "conversation_id": "conv_123",
                "agent_output_audio_format": "pcm_44100",
                "user_input_audio_format": "pcm_16000",
            },
        }
    )

    assert isinstance(event, ElevenLabsConversationMetadataEvent)
    assert event.conversation_id == "conv_123"
    assert event.user_input_audio_format == "pcm_16000"
    assert event.agent_output_audio_format == "pcm_44100"


def test_parse_audio_and_agent_response_events():
    audio = parse_elevenlabs_event(
        {
            "type": "audio",
            "audio_event": {
                "audio_base_64": base64.b64encode(b"audio-bytes").decode("ascii"),
                "event_id": 42,
                "alignment": {"chars": ["H", "i"]},
                "is_final": True,
            },
        }
    )
    response = parse_elevenlabs_event(
        {
            "type": "agent_response",
            "agent_response_event": {
                "agent_response": "Hi",
                "event_id": 42,
                "response_id": "response_42",
            },
        }
    )

    assert isinstance(audio, ElevenLabsAudioEvent)
    assert audio.audio_data == b"audio-bytes"
    assert audio.alignment == {"chars": ["H", "i"]}
    assert audio.is_final is True
    assert isinstance(response, ElevenLabsAgentResponseEvent)
    assert response.transcript == "Hi"
    assert response.event_id == audio.event_id


def test_parse_client_tool_call():
    event = parse_elevenlabs_event(
        {
            "type": "client_tool_call",
            "client_tool_call": {
                "tool_name": "get_order_status",
                "tool_call_id": "call_123",
                "parameters": {"order_id": "A-1"},
                "event_id": 7,
                "expects_response": True,
            },
        }
    )

    assert isinstance(event, ElevenLabsClientToolCallEvent)
    assert event.tool_name == "get_order_status"
    assert event.tool_call_id == "call_123"
    assert event.parameters == {"order_id": "A-1"}
    assert event.expects_response is True


def test_malformed_event_is_preserved_as_unknown():
    event = parse_elevenlabs_event(
        {"type": "audio", "audio_event": {"audio_base_64": ""}}
    )

    assert isinstance(event, ElevenLabsUnknownEvent)
    assert event.type == "audio"
    assert event.raw is not None
