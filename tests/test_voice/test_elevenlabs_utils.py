"""Tests for direct ElevenLabs TTS credential handling."""

from types import SimpleNamespace

import pytest

from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.voice.utils import elevenlabs_utils


def _install_fake_client(monkeypatch, api_keys: list[str]) -> None:
    class FakeElevenLabs:
        def __init__(self, api_key: str):
            api_keys.append(api_key)
            self.text_to_speech = SimpleNamespace(
                convert=lambda **_kwargs: iter([b"audio"])
            )

    monkeypatch.setattr(elevenlabs_utils, "ElevenLabs", FakeElevenLabs)


@pytest.mark.parametrize(
    ("config_key", "tts_key", "fallback_key", "expected"),
    [
        ("config-key", "tts-key", "fallback-key", "config-key"),
        (None, "tts-key", "fallback-key", "tts-key"),
        (None, None, "fallback-key", "fallback-key"),
    ],
)
def test_tts_api_key_precedence(
    monkeypatch,
    config_key: str | None,
    tts_key: str | None,
    fallback_key: str,
    expected: str,
):
    api_keys: list[str] = []
    _install_fake_client(monkeypatch, api_keys)
    if tts_key is None:
        monkeypatch.delenv("ELEVENLABS_TTS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ELEVENLABS_TTS_API_KEY", tts_key)
    monkeypatch.setenv("ELEVENLABS_API_KEY", fallback_key)

    result = elevenlabs_utils.tts_elevenlabs(
        "hello",
        ElevenLabsTTSConfig(api_key=config_key, voice_id="voice_test"),
    )

    assert result.data == b"audio"
    assert api_keys == [expected]
