"""The smallest integration is reachable through the public configuration and CLI."""

import argparse

import pytest

from tau2.agent.discrete_time_audio_native_agent import (
    AUDIO_NATIVE_VOICE_INSTRUCTION,
    CASCADED_MODEL_INSTRUCTION,
    DiscreteTimeAudioNativeAgent,
)
from tau2.cli import add_run_args
from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
    DEFAULT_SMALLEST_MODEL,
)
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.smallest import (
    DiscreteTimeSmallestAdapter,
    SmallestProvider,
)


def test_configuration_accepts_smallest():
    config = AudioNativeConfig(provider="smallest", model=DEFAULT_SMALLEST_MODEL)
    assert config.provider == "smallest"
    assert AUDIO_NATIVE_PROVIDER_TYPES["smallest"] == "audio_native"
    assert DEFAULT_AUDIO_NATIVE_REASONING_EFFORT["smallest"] is None


def test_factory_selects_smallest_without_connecting(monkeypatch):
    monkeypatch.delenv("SMALLEST_WS_URL", raising=False)
    adapter, model = create_adapter(provider="smallest", tick_duration_ms=200)
    assert isinstance(adapter, DiscreteTimeSmallestAdapter)
    assert model == DEFAULT_AUDIO_NATIVE_MODELS["smallest"] == DEFAULT_SMALLEST_MODEL
    assert adapter.model == DEFAULT_SMALLEST_MODEL
    assert not adapter.is_connected
    # The provider is built lazily and never raises; connect() checks the URL.
    assert isinstance(adapter.provider, SmallestProvider)
    assert adapter.provider.session_id is None


def test_cli_accepts_smallest():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(
        ["--domain", "retail", "--audio-native", "--audio-native-provider", "smallest"]
    )
    assert args.audio_native_provider == "smallest"


def test_server_side_settings_are_rejected_on_the_client():
    with pytest.raises(ValueError, match="reasoning"):
        create_adapter(
            provider="smallest", tick_duration_ms=200, reasoning_effort="high"
        )
    # create_adapter rejects a voice override for every provider but openai, so no
    # client-side voice setting can reach Hydra (the server chooses the voice).
    with pytest.raises(ValueError, match="voice"):
        create_adapter(provider="smallest", tick_duration_ms=200, voice="sophie")
    pcm16 = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000)
    with pytest.raises(ValueError, match="telephony"):
        create_adapter(provider="smallest", tick_duration_ms=200, audio_format=pcm16)
    adapter, _ = create_adapter(provider="smallest", tick_duration_ms=200)
    with pytest.raises(ValueError, match="modality"):
        adapter.connect("prompt", [], modality="audio_in_text_out")


def test_agent_uses_server_turn_taking_and_the_audio_native_prompt():
    agent = DiscreteTimeAudioNativeAgent(
        tools=[], domain_policy="POLICY", tick_duration_ms=200, provider="smallest"
    )
    assert agent.vad_config is None
    assert AUDIO_NATIVE_VOICE_INSTRUCTION in agent.system_prompt
    assert "Transcribed Speech" not in agent.system_prompt
    assert CASCADED_MODEL_INSTRUCTION not in agent.system_prompt  # pre-2.0 wording
    assert agent.system_prompt.endswith("POLICY")
