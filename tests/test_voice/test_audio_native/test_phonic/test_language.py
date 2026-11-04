"""The call's language reaches Phonic as the session's default language."""

from tau2.voice.audio_native.adapter import create_adapter


def test_factory_passes_call_language_to_provider(monkeypatch):
    monkeypatch.setenv("PHONIC_API_KEY", "test")
    adapter, _ = create_adapter(provider="phonic", tick_duration_ms=200, language="ko")
    assert adapter.provider.language == "ko"


def test_language_defaults_to_unset(monkeypatch):
    monkeypatch.setenv("PHONIC_API_KEY", "test")
    adapter, _ = create_adapter(provider="phonic", tick_duration_ms=200)
    assert adapter.provider.language is None
