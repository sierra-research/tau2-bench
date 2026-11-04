"""Offline tests for the exact configuration passed to LiveKit SDK components."""

import asyncio
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from tau2.data_model.cascaded import (
    CascadedConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
)
from tau2.voice.audio_native.livekit.provider import (
    CascadedEventType,
    CascadedVoiceProvider,
    ProviderState,
)


@pytest.mark.parametrize("close_fails", [False, True])
def test_disconnect_closes_tts_before_http_session(monkeypatch, close_fails):
    from unittest.mock import AsyncMock

    provider = CascadedVoiceProvider(CascadedConfig(tts=ElevenLabsTTSConfig()))
    provider._state = ProviderState.LISTENING
    order = []

    async def close_tts():
        order.append("tts")
        assert provider._http_session is not None
        if close_fails:
            raise RuntimeError("offline close failure")

    async def close_http():
        order.append("http")

    provider._tts_client = SimpleNamespace(aclose=close_tts)
    provider._http_session = SimpleNamespace(close=close_http)
    monkeypatch.setattr(provider, "_stop_stt_stream", AsyncMock())
    if close_fails:
        with pytest.raises(RuntimeError, match="offline close failure"):
            asyncio.run(provider.disconnect())
    else:
        asyncio.run(provider.disconnect())
    assert order == ["tts", "http"]
    assert provider._tts_client is None
    assert provider._http_session is None
    assert provider._state == ProviderState.DISCONNECTED


@pytest.mark.parametrize("language", ["en", "es", "pt", "hi", "zh", "ko"])
def test_elevenlabs_stream_request_serializes_selected_language(monkeypatch, language):
    from urllib.parse import parse_qs, urlsplit

    from livekit.plugins.elevenlabs.tts import _multi_stream_url

    monkeypatch.setenv("ELEVENLABS_API_KEY", "offline-fixture-key")
    provider = CascadedVoiceProvider(
        CascadedConfig(tts=ElevenLabsTTSConfig(language=language))
    )
    asyncio.run(provider._init_tts())
    query = parse_qs(urlsplit(_multi_stream_url(provider._tts_client._opts)).query)
    assert query["language_code"] == [language]
    assert query["output_format"] == ["pcm_24000"]


@pytest.mark.parametrize("tts_type", [ElevenLabsTTSConfig, DeepgramTTSConfig])
@pytest.mark.parametrize("outcome", ["success", "barge_in", "error", "cancel"])
def test_tts_uses_language_aware_stream_and_closes_on_every_exit(tts_type, outcome):
    from unittest.mock import Mock

    provider = CascadedVoiceProvider(CascadedConfig(tts=tts_type()))

    class OfflineStream:
        def __init__(self):
            self.closed = False
            self.text = None
            self.ended = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        def push_text(self, text):
            self.text = text

        def end_input(self):
            self.ended = True

        async def __aiter__(self):
            if outcome == "error":
                raise RuntimeError("offline synthesis failure")
            if outcome == "cancel":
                raise asyncio.CancelledError
            if outcome == "barge_in":
                provider._cancel_event.set()
            yield SimpleNamespace(frame=SimpleNamespace(data=memoryview(b"pcm")))

    stream = OfflineStream()
    client = Mock()
    client.stream.return_value = stream
    client.synthesize.return_value = stream
    provider._tts_client = client

    async def collect():
        return [event async for event in provider._process_tts("hello")]

    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(collect())
    else:
        events = asyncio.run(collect())
        audio = [
            event.audio for event in events if event.type == CascadedEventType.TTS_AUDIO
        ]
        assert audio == ([b"pcm"] if outcome == "success" else [])
        errors = [event for event in events if event.type == CascadedEventType.ERROR]
        assert bool(errors) == (outcome == "error")
    assert stream.closed
    assert provider._state == ProviderState.LISTENING
    if tts_type is ElevenLabsTTSConfig:
        client.stream.assert_called_once_with()
        client.synthesize.assert_not_called()
        assert stream.text == "hello"
        assert stream.ended
    else:
        client.synthesize.assert_called_once_with("hello")
        client.stream.assert_not_called()


@pytest.mark.parametrize("language", ["en", "es", "pt", "hi", "zh", "ko"])
def test_deepgram_receives_selected_language(monkeypatch, language):
    from livekit.plugins import deepgram

    constructor = create_autospec(deepgram.STT)
    monkeypatch.setattr(deepgram, "STT", constructor)
    provider = CascadedVoiceProvider(
        CascadedConfig(stt=DeepgramSTTConfig(model="nova-3", language=language))
    )
    asyncio.run(provider._init_stt())
    assert constructor.call_args.kwargs["language"] == language
    assert constructor.call_args.kwargs["model"] == "nova-3"
    assert provider._stt_client is constructor.return_value


@pytest.mark.parametrize(
    ("repository_key", "plugin_key", "expected_key"),
    [
        ("repository-fixture", "plugin-fixture", "repository-fixture"),
        (None, "plugin-fixture", "plugin-fixture"),
        ("", "plugin-fixture", "plugin-fixture"),
    ],
)
@pytest.mark.parametrize("language", ["ko", None])
def test_elevenlabs_receives_recorded_voice_language_settings_and_credentials(
    monkeypatch, repository_key, plugin_key, expected_key, language
):
    from livekit.plugins import elevenlabs

    # Autospec checks the installed SDK signature without constructing a client.
    constructor = create_autospec(elevenlabs.TTS)
    monkeypatch.setattr(elevenlabs, "TTS", constructor)
    for name, value in (
        ("ELEVENLABS_API_KEY", repository_key),
        ("ELEVEN_API_KEY", plugin_key),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    config = ElevenLabsTTSConfig(
        voice_id="reviewed-voice-id",
        model="eleven_flash_v2_5",
        language=language,
        stability=0.31,
        similarity_boost=0.82,
    )
    provider = CascadedVoiceProvider(CascadedConfig(tts=config))
    asyncio.run(provider._init_tts())
    kwargs = constructor.call_args.kwargs
    assert kwargs["voice_id"] == "reviewed-voice-id"
    assert kwargs["model"] == "eleven_flash_v2_5"
    assert kwargs.get("language") == language
    assert ("language" in kwargs) == (language is not None)
    assert kwargs["api_key"] == expected_key
    assert kwargs["voice_settings"].stability == 0.31
    assert kwargs["voice_settings"].similarity_boost == 0.82
    assert provider._tts_client is constructor.return_value


def test_elevenlabs_missing_credentials_fails_before_client(monkeypatch):
    from livekit.plugins import elevenlabs

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)
    constructor = create_autospec(elevenlabs.TTS)
    monkeypatch.setattr(elevenlabs, "TTS", constructor)
    provider = CascadedVoiceProvider(CascadedConfig(tts=ElevenLabsTTSConfig()))
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        asyncio.run(provider._init_tts())
    constructor.assert_not_called()


@pytest.mark.parametrize("sample_rate", [16000, 22050, 24000, 44100])
def test_real_elevenlabs_constructor_accepts_resolved_config_without_connecting(
    monkeypatch,
    sample_rate,
):
    import socket

    from livekit.plugins import elevenlabs

    def reject_connection(*args, **kwargs):
        pytest.fail("SDK construction must not connect to the network")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "offline-fixture-key")
    config = ElevenLabsTTSConfig(
        voice_id="offline-reviewed-voice-id",
        model="eleven_flash_v2_5",
        language="zh",
        stability=0.31,
        similarity_boost=0.82,
        sample_rate=sample_rate,
    )
    provider = CascadedVoiceProvider(CascadedConfig(tts=config))
    asyncio.run(provider._init_tts())
    client = provider._tts_client
    assert isinstance(client, elevenlabs.TTS)
    assert client._opts.voice_id == config.voice_id
    assert client._opts.model == config.model
    assert client._opts.language == "zh"
    assert client._opts.voice_settings.stability == config.stability
    assert client._opts.voice_settings.similarity_boost == config.similarity_boost
    assert client._opts.encoding == f"pcm_{sample_rate}"
    assert client.sample_rate == sample_rate
    assert client._session is None


@pytest.mark.parametrize("sample_rate", [16000, 24000])
def test_deepgram_receives_recorded_synthesis_sample_rate(monkeypatch, sample_rate):
    from livekit.plugins import deepgram

    constructor = create_autospec(deepgram.TTS)
    monkeypatch.setattr(deepgram, "TTS", constructor)
    provider = CascadedVoiceProvider(
        CascadedConfig(tts=DeepgramTTSConfig(sample_rate=sample_rate))
    )
    asyncio.run(provider._init_tts())
    assert constructor.call_args.kwargs["sample_rate"] == sample_rate


@pytest.mark.parametrize(
    ("tts_type", "sample_rate"),
    [
        (DeepgramTTSConfig, 16000),
        (DeepgramTTSConfig, 24000),
        (ElevenLabsTTSConfig, 16000),
        (ElevenLabsTTSConfig, 22050),
        (ElevenLabsTTSConfig, 24000),
        (ElevenLabsTTSConfig, 44100),
    ],
)
def test_adapter_preserves_one_second_of_synthesized_audio(tts_type, sample_rate):
    from tau2.voice.audio_native.livekit.discrete_time_adapter import (
        LiveKitCascadedAdapter,
    )

    adapter = LiveKitCascadedAdapter(
        tick_duration_ms=200,
        cascaded_config=CascadedConfig(tts=tts_type(sample_rate=sample_rate)),
    )
    # One second of mono PCM16, delivered in chunks as the TTS SDK does.
    pcm = bytes(sample_rate * 2)
    converter = adapter._audio_converter
    converted = b"".join(
        converter.convert_output(pcm[offset : offset + 1024])
        for offset in range(0, len(pcm), 1024)
    )
    assert converter.output_sample_rate == sample_rate
    assert len(converted) == 8000  # One second of 8 kHz, one-byte mu-law samples.
    assert adapter._provider is None
