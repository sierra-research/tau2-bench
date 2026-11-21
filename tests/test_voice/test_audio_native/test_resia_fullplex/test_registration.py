"""The Resia Fullplex integration is reachable through the public configuration and CLI."""

import argparse

from tau2.cli import add_run_args
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_RESIA_FULLPLEX_MODEL,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.resia_fullplex import DiscreteTimeResiaFullplexAdapter


def test_configuration_accepts_resia_fullplex():
    config = AudioNativeConfig(
        provider="resia_fullplex", model=DEFAULT_RESIA_FULLPLEX_MODEL
    )
    assert config.provider == "resia_fullplex"
    assert AUDIO_NATIVE_PROVIDER_TYPES[config.provider] == "audio_native"


def test_factory_selects_resia_fullplex_without_connecting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("RESIA_API_KEY", "test")
    monkeypatch.setenv("RESIA_FULLPLEX_BACKEND_MODEL", "test-model")
    adapter, model = create_adapter(provider="resia_fullplex", tick_duration_ms=200)
    assert isinstance(adapter, DiscreteTimeResiaFullplexAdapter)
    assert model == DEFAULT_AUDIO_NATIVE_MODELS["resia_fullplex"]
    assert model == DEFAULT_RESIA_FULLPLEX_MODEL
    assert not adapter.is_connected


def test_cli_accepts_resia_fullplex():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(
        [
            "--domain",
            "retail",
            "--audio-native",
            "--audio-native-provider",
            "resia_fullplex",
        ]
    )
    assert args.audio_native_provider == "resia_fullplex"
