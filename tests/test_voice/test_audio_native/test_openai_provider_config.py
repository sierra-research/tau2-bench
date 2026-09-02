import pytest

from tau2.config import DEFAULT_OPENAI_REALTIME_BASE_URL
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider


def test_openai_model_uses_openai_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    provider = OpenAIRealtimeProvider(model="gpt-realtime")

    assert provider.api_key == "openai-key"
    assert provider.base_url == DEFAULT_OPENAI_REALTIME_BASE_URL


def test_pine_model_uses_pine_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("PINE_API_KEY", "pine-key")
    monkeypatch.setenv("PINE_REALTIME_BASE_URL", "wss://pine.example/realtime")

    provider = OpenAIRealtimeProvider(model="pine-voice-preview")

    assert provider.api_key == "pine-key"
    assert provider.base_url == "wss://pine.example/realtime"


def test_pine_model_requires_pine_base_url(monkeypatch):
    monkeypatch.setenv("PINE_API_KEY", "pine-key")
    monkeypatch.delenv("PINE_REALTIME_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="PINE_REALTIME_BASE_URL"):
        OpenAIRealtimeProvider(model="pine-voice-preview")
