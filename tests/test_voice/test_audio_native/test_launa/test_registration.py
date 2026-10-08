"""The Launa integration is reachable through the public configuration and CLI."""

import argparse

from tau2.cli import add_run_args
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_LAUNA_MODEL,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.launa import DiscreteTimeLaunaAdapter


def test_configuration_accepts_launa():
    config = AudioNativeConfig(provider="launa", model=DEFAULT_LAUNA_MODEL)
    assert config.provider == "launa"
    assert AUDIO_NATIVE_PROVIDER_TYPES[config.provider] == "audio_native"


def test_factory_selects_launa_without_connecting():
    adapter, model = create_adapter(provider="launa", tick_duration_ms=200)
    assert isinstance(adapter, DiscreteTimeLaunaAdapter)
    assert adapter.model == DEFAULT_AUDIO_NATIVE_MODELS["launa"]
    assert model == DEFAULT_LAUNA_MODEL
    assert not adapter.is_connected


def test_cli_accepts_launa():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(
        ["--domain", "retail", "--audio-native", "--audio-native-provider", "launa"]
    )
    assert args.audio_native_provider == "launa"
