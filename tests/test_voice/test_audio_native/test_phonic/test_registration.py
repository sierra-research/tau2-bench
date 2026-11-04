"""The Phonic integration is reachable through the public configuration and CLI."""

import argparse

from tau2.cli import add_run_args
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_PHONIC_MODEL,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.phonic import DiscreteTimePhonicAdapter


def test_configuration_accepts_phonic():
    config = AudioNativeConfig(provider="phonic", model=DEFAULT_PHONIC_MODEL)
    assert config.provider == "phonic"
    assert AUDIO_NATIVE_PROVIDER_TYPES[config.provider] == "audio_native"


def test_factory_selects_phonic_without_connecting():
    adapter, model = create_adapter(provider="phonic", tick_duration_ms=200)
    assert isinstance(adapter, DiscreteTimePhonicAdapter)
    assert model == DEFAULT_AUDIO_NATIVE_MODELS["phonic"] == DEFAULT_PHONIC_MODEL
    assert not adapter.is_connected


def test_cli_accepts_phonic():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(
        ["--domain", "retail", "--audio-native", "--audio-native-provider", "phonic"]
    )
    assert args.audio_native_provider == "phonic"
