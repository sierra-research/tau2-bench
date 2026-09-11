import asyncio
import audioop
import base64
import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from aiortc.rtcpeerconnection import CODECS
from aiortc.sdp import SessionDescription

from tau2.agent.base.streaming import _has_meaningful_content
from tau2.config import (
    DEFAULT_OPENAI_LIVE_MODEL,
    resolve_audio_native_reasoning_effort,
)
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    FunctionCallArgumentsDoneEvent,
)
from tau2.voice.audio_native.openai.live_adapter import DiscreteTimeOpenAILiveAdapter
from tau2.voice.audio_native.openai.live_config import LiveConfig
from tau2.voice.audio_native.openai.live_provider import (
    LiveInputTranscriptDelta,
    LiveTranscriptDelta,
    OpenAILiveProvider,
    _LiveInputAudioTrack,
)
from tau2.voice.audio_native.tick_result import TickResult


@pytest.fixture
def provider():
    result = OpenAILiveProvider(
        model="test-live",
        config=LiveConfig(backend_model="test-backend"),
        reasoning_effort="low",
        api_key="test-only",
    )
    result._channel = Mock(readyState="open")
    return result


def test_default_templates_slot_system_prompt_and_keep_tools_backend_only(provider):
    tool = Mock(
        openai_schema={
            "function": {
                "name": "lookup",
                "description": "Read a record",
                "parameters": {"type": "object"},
            }
        }
    )
    session = provider._build_session("ORIGINAL DOMAIN POLICY", [tool])
    assert session["instructions"] == (
        "ORIGINAL DOMAIN POLICY\n\n"
        "You are the spoken interface for a live support conversation. You "
        "control how and when to speak, listen, stop, interrupt, and delegate. "
        "Listen to the user's complete turn. For every substantive request, "
        "clarification, answer, or reported action, delegate to the backend and "
        "remain silent until the backend result arrives. The backend owns task "
        "reasoning, policy decisions, tool execution, and the substance of your "
        "reply. Naturally rephrase only its concise customer-facing answer, "
        "question, or requested customer action. Do not independently diagnose, "
        "invent tool results, repeat the user, or narrate delegation. Stop "
        "speaking immediately when the user speaks."
    )
    backend = session["delegation"]["responses"]
    assert backend["instructions"] == (
        "ORIGINAL DOMAIN POLICY\n\n"
        "Normalize ten spoken phone-number digits as XXX-XXX-XXXX before "
        "calling a phone lookup."
    )
    assert backend["model"] == "test-backend"
    assert backend["reasoning"] == {"effort": "low"}
    assert backend["tools"][0]["name"] == "lookup"
    assert "tools" not in session
    assert session["audio"]["output"]["voice"] == "marin"


def test_explicit_prompts_preserve_existing_override_behavior():
    config = LiveConfig(
        backend_model="test-backend",
        frontend_prompt="CUSTOM FRONTEND",
        backend_prompt="CUSTOM BACKEND",
    )
    provider = OpenAILiveProvider(
        model="test-live",
        config=config,
        api_key="test-only",
    )
    session = provider._build_session("ORIGINAL DOMAIN POLICY", [])
    assert session["instructions"] == "CUSTOM FRONTEND"
    assert session["delegation"]["responses"]["instructions"] == (
        "CUSTOM BACKEND\n\nORIGINAL DOMAIN POLICY"
    )

    config.append_system_prompt = False
    session = provider._build_session("ORIGINAL DOMAIN POLICY", [])
    assert session["delegation"]["responses"]["instructions"] == "CUSTOM BACKEND"


def test_config_rejects_missing_or_misrouted_backend():
    with pytest.raises(ValueError, match="live_config is required"):
        AudioNativeConfig(provider="openai_live", model="test-live")
    with pytest.raises(ValueError, match="live_config is required"):
        AudioNativeConfig(provider="openai", live_config={"backend_model": "test"})


def test_create_adapter_uses_default_live_frontend_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    adapter, model = create_adapter(
        "openai_live",
        tick_duration_ms=200,
        reasoning_effort=resolve_audio_native_reasoning_effort("openai_live", None),
        live_config=LiveConfig(backend_model="test-backend"),
    )

    assert model == DEFAULT_OPENAI_LIVE_MODEL == "gpt-live-1-diamond-alpha"
    assert adapter.model == DEFAULT_OPENAI_LIVE_MODEL


def test_offer_advertises_audio_recovery_without_changing_other_providers(provider):
    original_codecs = deepcopy(CODECS)

    async def check():
        await provider.connect()
        try:
            offer = SessionDescription.parse((await provider._create_offer()).sdp)
            audio = next(media for media in offer.media if media.kind == "audio")
            assert all(
                any(feedback.type == "nack" for feedback in codec.rtcpFeedback)
                for codec in audio.rtp.codecs
            )
            assert CODECS == original_codecs
        finally:
            await provider.disconnect()

    asyncio.run(check())


def test_audio_arrives_without_any_turn_or_transcript_marker(provider):
    pcm = b"\x00\x00" * 960 + b"\x01\x00" * 960
    events = provider._normalize_event(
        {"type": "live.media.audio", "audio": base64.b64encode(pcm).decode()}
    )
    assert len(events) == 1
    assert isinstance(events[0], AudioDeltaEvent)
    assert base64.b64decode(events[0].delta) == pcm


def test_input_transcript_delta_preserves_structured_speech_timing(provider):
    events = provider._normalize_event(
        {
            "type": "session.input_transcript.delta",
            "event_id": "event-1",
            "delta": " hello",
            "start_ms": 1200,
            "end_ms": 1400,
        }
    )

    assert events == [
        LiveInputTranscriptDelta(
            type="session.input_transcript.delta",
            event_id="event-1",
            delta=" hello",
            start_ms=1200,
            end_ms=1400,
        )
    ]


def test_input_transcript_delta_surfaces_normalized_speech_event(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    adapter = DiscreteTimeOpenAILiveAdapter(
        tick_duration_ms=200,
        model="test-live",
        config=LiveConfig(backend_model="test-backend"),
    )
    event = LiveInputTranscriptDelta(
        type="session.input_transcript.delta",
        delta=" hello",
        start_ms=1200,
        end_ms=1400,
    )
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
    )

    asyncio.run(adapter._process_event(result, event))

    assert result.events == [event]
    assert result.vad_events == ["speech_started"]
    assert result.was_truncated is False


def response_event(provider, event):
    return provider._normalize_event(
        {"type": "response.event", "delegation_id": "d1", "event": event}
    )


@pytest.mark.parametrize("results_before_completion", [False, True])
def test_tool_batch_continues_once_after_all_outputs(
    provider, results_before_completion
):
    response_event(provider, {"type": "response.created", "response": {"id": "r1"}})
    for call_id in ("c1", "c2"):
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "arguments": "{}",
            },
        }
        assert isinstance(
            response_event(provider, event)[0], FunctionCallArgumentsDoneEvent
        )
        assert not any(
            isinstance(e, FunctionCallArgumentsDoneEvent)
            for e in response_event(provider, event)
        )
    complete = {"type": "response.completed", "response": {"id": "r1", "output": []}}
    if not results_before_completion:
        response_event(provider, complete)
    asyncio.run(provider.send_tool_result("c1", "first", request_response=False))
    sent = [json.loads(c.args[0]) for c in provider._channel.send.call_args_list]
    assert [e["type"] for e in sent] == ["response.item.create"]
    asyncio.run(provider.send_tool_result("c2", "second"))
    if results_before_completion:
        response_event(provider, complete)
    assert not any(
        isinstance(e, FunctionCallArgumentsDoneEvent)
        for e in response_event(provider, event)
    )
    response_event(provider, complete)
    sent = [json.loads(c.args[0]) for c in provider._channel.send.call_args_list]
    assert [e["type"] for e in sent] == [
        "response.item.create",
        "response.item.create",
        "response.create",
    ]
    assert [e["item"]["call_id"] for e in sent[:2]] == ["c1", "c2"]


def test_quiet_pcm_is_not_removed_at_chunk_boundaries():
    async def check():
        track = _LiveInputAudioTrack()
        pcm = np.arange(1920, dtype=np.int16).tobytes()
        # Use the track's rate to isolate queue integrity from resampling.
        track.append(pcm[:1700], sample_rate=48_000)
        track.append(pcm[1700:], sample_rate=48_000)
        frames = [await track.recv(), await track.recv()]
        assert b"".join(f.to_ndarray().tobytes() for f in frames) == pcm
        assert [f.pts for f in frames] == [0, 960]

    asyncio.run(check())


def test_buffered_audio_is_preserved_in_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    adapter = DiscreteTimeOpenAILiveAdapter(
        tick_duration_ms=200,
        model="test-live",
        config=LiveConfig(backend_model="test-backend"),
    )
    provider = adapter.provider
    provider._peer = Mock(connectionState="connected")
    provider.send_audio = AsyncMock()
    provider.disconnect = AsyncMock()
    provider._record = Mock()
    pcm = np.concatenate(
        [np.zeros(4800, dtype=np.int16), np.arange(4800, dtype=np.int16)]
    ).tobytes()
    expected = StreamingTelephonyConverter(
        input_sample_rate=24000, output_sample_rate=24000
    ).convert_output(pcm)

    async def receive_burst():
        for offset in range(0, len(pcm), 960):
            provider._events.put_nowait(
                {
                    "type": "live.media.audio",
                    "audio": base64.b64encode(pcm[offset : offset + 960]).decode(),
                }
            )

    adapter._bg_loop.start()
    adapter._connected = True
    try:
        adapter._bg_loop.run_coroutine(receive_burst())
        played = b"".join(
            adapter.run_tick(b"\xff" * 1600).agent_audio_data for _ in range(3)
        )
        assert played == expected
    finally:
        adapter.disconnect()
    audit = next(
        call.args[1]
        for call in provider._record.call_args_list
        if call.args[0] == "audio_delivery"
    )
    assert audit["audio_preserved_in_order"] is True
    assert audit["received_bytes"] == audit["played_bytes"] == len(expected)
    assert audit["buffered_bytes"] == 0


@pytest.mark.parametrize("amplitude,speech", [(0, False), (1000, True)])
def test_stock_activity_does_not_confuse_continuous_silence_with_speech(
    monkeypatch, amplitude, speech
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    adapter = DiscreteTimeOpenAILiveAdapter(
        tick_duration_ms=200,
        model="test-live",
        config=LiveConfig(backend_model="test-backend"),
    )
    audio = audioop.lin2ulaw(np.full(1600, amplitude, dtype=np.int16).tobytes(), 2)
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
        agent_audio_chunks=[(audio, "speech")],
    )
    monkeypatch.setattr(DiscreteTimeOpenAIAdapter, "run_tick", lambda *args: result)
    assert adapter.run_tick(b"\xff" * 1600).contains_speech is speech
    assert result.agent_audio_data == audio
    if not speech:
        # Native text can arrive on a quiet PCM boundary. The stock caller
        # must retain it, then resume treating transport silence as silence.
        asyncio.run(
            adapter._process_event(
                result,
                LiveTranscriptDelta(
                    type="response.output_audio_transcript.delta",
                    delta="Your order shipped.",
                    audio_position_bytes=0,
                    start_ms=0,
                    end_ms=200,
                ),
            )
        )
        late = adapter.run_tick(b"\xff" * 1600)
        message = AssistantMessage(
            role="assistant",
            content=late.proportional_transcript,
            contains_speech=late.contains_speech,
        )
        assert message.content == "Your order shipped."
        assert late.contains_speech is False
        assert _has_meaningful_content(message)
        assert late.agent_audio_data == audio
        assert adapter.run_tick(b"\xff" * 1600).contains_speech is False


def test_missing_close_ack_does_not_invalidate_completed_task(provider, monkeypatch):
    provider._started.set()
    provider._peer = Mock(close=AsyncMock(), getStats=AsyncMock(return_value={}))
    peer = provider._peer

    async def timed_out(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timed_out)
    asyncio.run(provider.disconnect())
    peer.close.assert_awaited_once()
    assert json.loads(provider._channel.send.call_args.args[0]) == {
        "type": "session.close"
    }
    with pytest.raises(ConnectionError, match="during conversation"):
        provider._closing = False
        provider._normalize_event({"type": "session.closed"})
