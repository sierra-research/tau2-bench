"""Tests for the AAI discrete-time adapter (PCM16 <-> mu-law conversion).

Unlike AssemblyAI's native G.711 mu-law transport, the aai voice-agent host
speaks PCM16 over its WebSocket (16kHz in, 24kHz out). The adapter's external
interface stays tau2 telephony mu-law/8k, so conversion happens at this
boundary: send path (mu-law 8k -> PCM16 16k) and receive path
(PCM16 24k -> mu-law 8k).
"""

import pytest

from tau2.voice.audio_native.aai.discrete_time_adapter import DiscreteTimeAAIAdapter
from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent,
    AAIAudioChunkEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAIToolCallEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


def _adapter() -> DiscreteTimeAAIAdapter:
    return DiscreteTimeAAIAdapter(tick_duration_ms=200, send_audio_instant=True)


def _result() -> TickResult:
    return TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0.0,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )


def test_reasoning_effort_rejected() -> None:
    with pytest.raises(ValueError):
        DiscreteTimeAAIAdapter(tick_duration_ms=200, reasoning_effort="high")


def test_audio_chunk_converted_from_pcm16_24k_to_ulaw_8k() -> None:
    a = _adapter()
    r = _result()
    # ~20ms of PCM16 audio at 24kHz (480 samples, 2 bytes/sample).
    pcm16_24k = (b"\x10\x00") * 480
    a._process_event(r, AAIAudioChunkEvent(pcm16=pcm16_24k))

    assert r.agent_audio_bytes > 0
    assert len(r.agent_audio_chunks) == 1
    ulaw_bytes, item_id = r.agent_audio_chunks[0]
    # mu-law is 1 byte/sample; 24k->8k resample should yield ~1/3 the samples.
    assert len(ulaw_bytes) == pytest.approx(160, abs=5)
    assert item_id == a._current_item_id


def test_agent_transcript_overwrites_not_appends() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="first text"))
    a._process_event(r, AAIAgentTranscriptEvent(text="second text"))

    turn_id = a._current_item_id
    assert turn_id is not None
    assert a._utterance_transcripts[turn_id].transcript_received == "second text"


def test_tool_call_recorded_with_id_name_arguments() -> None:
    a = _adapter()
    r = _result()
    a._process_event(
        r,
        AAIToolCallEvent(tool_call_id="c-1", tool_name="lookup", args={"x": 1}),
    )

    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].id == "c-1"
    assert r.tool_calls[0].name == "lookup"
    assert r.tool_calls[0].arguments == {"x": 1}


def test_speech_started_sets_truncation_and_vad_event() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="hi there"))
    current_item_id = a._current_item_id

    a._process_event(r, AAISpeechStartedEvent())

    assert r.was_truncated is True
    assert "speech_started" in r.vad_events
    assert r.skip_item_id == current_item_id


def test_speech_started_discards_buffered_agent_audio() -> None:
    a = _adapter()
    r = _result()
    a._buffered_agent_audio = [(b"\xff" * 10, "turn-0")]

    a._process_event(r, AAISpeechStartedEvent())

    assert a._buffered_agent_audio == []
    assert r.truncated_audio_bytes == 10


def test_speech_stopped_adds_vad_event() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAISpeechStoppedEvent())

    assert "speech_stopped" in r.vad_events


def test_barge_in_skip_discards_subsequent_audio_chunk() -> None:
    a = _adapter()
    r = _result()
    a._process_event(r, AAIAgentTranscriptEvent(text="hi"))
    current_item_id = a._current_item_id
    r.skip_item_id = current_item_id

    pcm16_24k = (b"\x10\x00") * 480
    a._process_event(r, AAIAudioChunkEvent(pcm16=pcm16_24k))

    assert r.agent_audio_chunks == []
    assert r.truncated_audio_bytes > 0
