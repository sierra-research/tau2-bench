"""Resia Fullplex records its backend model and retries a call with dropped caller audio."""

import asyncio

import pytest

from tau2.data_model.live_config import LiveConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.voice.audio_native.resia_fullplex.provider import ResiaFullplexProvider


def test_run_config_records_the_backend_model():
    config = AudioNativeConfig(
        provider="resia_fullplex", live_config=LiveConfig(backend_model="gpt-6-astra")
    )
    assert config.live_config.backend_model == "gpt-6-astra"


def test_live_config_is_refused_for_other_providers():
    with pytest.raises(ValueError):
        AudioNativeConfig(provider="openai", live_config=LiveConfig(backend_model="x"))


def test_dropped_caller_audio_ends_the_call(monkeypatch):
    monkeypatch.setenv("RESIA_API_KEY", "test-key")
    provider = ResiaFullplexProvider(model="resia-fullplex-1", backend_model="m")
    provider._on_message(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "input_audio_rate_limit_exceeded",
                "message": "Input audio was dropped.",
            },
        }
    )
    assert provider.lost is not None
    assert not provider.is_connected


def test_dropped_caller_audio_fails_the_tick(monkeypatch):
    monkeypatch.setenv("RESIA_API_KEY", "test-key")
    provider = ResiaFullplexProvider(model="resia-fullplex-1", backend_model="m")
    provider._on_message(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "input_audio_rate_limit_exceeded",
                "message": "Input audio was dropped.",
            },
        }
    )
    with pytest.raises(ConnectionError):
        asyncio.run(provider.receive_events_for_duration(0.01))


def test_adapter_uses_the_recorded_backend_model(monkeypatch):
    monkeypatch.setenv("RESIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RESIA_FULLPLEX_BACKEND_MODEL", "env-model")
    from tau2.voice.audio_native.adapter import create_adapter

    adapter, _ = create_adapter(
        provider="resia_fullplex",
        tick_duration_ms=200,
        model="resia-fullplex-1",
        live_config=LiveConfig(backend_model="recorded-model"),
    )
    assert adapter.provider.config.backend_model == "recorded-model"
