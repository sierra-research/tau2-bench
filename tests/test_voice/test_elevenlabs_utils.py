from unittest.mock import MagicMock

from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.voice.utils import elevenlabs_utils


def test_tts_elevenlabs_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_BASE_URL", "http://tts.example:8008")
    client = MagicMock()
    client.text_to_speech.convert.return_value = iter([b"\x00\x01"])
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(elevenlabs_utils, "ElevenLabs", constructor)

    audio = elevenlabs_utils.tts_elevenlabs(
        "hello",
        ElevenLabsTTSConfig(voice_id="test-voice", insert_audio_tags=False),
    )

    constructor.assert_called_once_with(
        api_key="test-key",
        base_url="http://tts.example:8008",
    )
    assert audio.data == b"\x00\x01"
