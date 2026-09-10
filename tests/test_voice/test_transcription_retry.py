from unittest.mock import Mock

import httpx
import pytest
from tenacity import wait_none

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig, TranscriptionResult
from tau2.voice.transcription import transcribe


@pytest.fixture
def pcm_audio() -> AudioData:
    return AudioData(
        data=b"\x00\x00",
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=24000,
            channels=1,
        ),
    )


def test_transcribe_audio_retries_transient_provider_failure(
    monkeypatch: pytest.MonkeyPatch, pcm_audio: AudioData
) -> None:
    provider = Mock(
        side_effect=[
            ConnectionError("transient"),
            TranscriptionResult(transcript="recovered"),
        ]
    )
    monkeypatch.setattr(
        transcribe,
        "_convert_audio_to_pcm16_mono_24000",
        lambda _audio: (pcm_audio, None),
    )
    monkeypatch.setattr(transcribe, "transcribe_deepgram", provider)
    monkeypatch.setattr(
        getattr(transcribe.transcribe_audio, "retry"), "wait", wait_none()
    )

    result = transcribe.transcribe_audio(pcm_audio, TranscriptionConfig(model="nova-2"))

    assert provider.call_count == 2
    assert result == TranscriptionResult(transcript="recovered")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("disconnected"),
        httpx.ReadTimeout("timed out"),
    ],
)
def test_transcribe_deepgram_propagates_transient_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    pcm_audio: AudioData,
    error: Exception,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(transcribe.requests, "post", Mock(side_effect=error))

    with pytest.raises(type(error), match=str(error)):
        transcribe.transcribe_deepgram(pcm_audio, TranscriptionConfig(model="nova-2"))
