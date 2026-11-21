"""Public registration tests for the Apriel audio-native provider.

These tests make sure the user-facing entry points recognize Apriel: Pydantic
run configuration, the adapter factory, and the `tau2 run` CLI argument parser.
They do not connect to an Apriel endpoint.
"""

import argparse
import os
from unittest.mock import patch

from tau2.cli import add_run_args
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_APRIEL_MODEL,
    DEFAULT_AUDIO_NATIVE_MODELS,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.apriel import DiscreteTimeAprielAdapter

APRIEL_ENV = {
    "APRIEL_REALTIME_BASE_URL": "wss://apriel.example/v1/realtime",
    "APRIEL_REALTIME_API_KEY": "test-key",
    "APRIEL_MODEL_NAME": "Apriel-Voice-27B-Preview",
}


def test_configuration_accepts_apriel():
    config = AudioNativeConfig(provider="apriel", model=DEFAULT_APRIEL_MODEL)
    assert config.provider == "apriel"
    assert AUDIO_NATIVE_PROVIDER_TYPES[config.provider] == "audio_native"


def test_factory_selects_apriel_without_connecting():
    with patch.dict(os.environ, APRIEL_ENV, clear=False):
        adapter, model = create_adapter(provider="apriel", tick_duration_ms=200)

    assert isinstance(adapter, DiscreteTimeAprielAdapter)
    assert adapter.model == DEFAULT_AUDIO_NATIVE_MODELS["apriel"]
    assert model == DEFAULT_APRIEL_MODEL
    assert not adapter.is_connected


def test_cli_accepts_apriel():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(
        ["--domain", "retail", "--audio-native", "--audio-native-provider", "apriel"]
    )
    assert args.audio_native_provider == "apriel"
