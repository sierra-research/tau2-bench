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
from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.runner.batch import get_run_info
from tau2.voice.audio_native.adapter import (
    create_adapter,
    provider_run_settings,
    recorded_run_settings,
)
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


def test_the_effective_reveal_cap_is_recorded_in_the_saved_run_info(monkeypatch):
    # The cap comes from the environment, not the run config, and changes what the
    # caller sees, so a saved run must say which value produced it (maintainer
    # review of the leaderboard submission, 2026-11-21).
    monkeypatch.delenv("SMALLEST_TRANSCRIPT_REVEAL_CPS", raising=False)
    config = VoiceRunConfig(
        domain="retail",
        audio_native_config=AudioNativeConfig(
            provider="smallest", model=DEFAULT_SMALLEST_MODEL
        ),
    )
    assert get_run_info(config, tasks=[]).agent_info.llm_args == {
        "transcript_reveal_cps": 15.0
    }
    monkeypatch.setenv("SMALLEST_TRANSCRIPT_REVEAL_CPS", "0")
    assert get_run_info(config, tasks=[]).agent_info.llm_args == {
        "transcript_reveal_cps": 0.0
    }
    assert provider_run_settings("openai") is None  # providers without such settings


def test_the_contract_check_accepts_a_complete_record_or_none():
    # The saved record is the truth (the run's environment is gone), so the
    # benchmark contract checks its shape: every declared key, each a number.
    record = {"transcript_reveal_cps": 15.0}
    assert recorded_run_settings("smallest", record) == record
    assert recorded_run_settings("smallest", None) is None  # runs before recording
    for bad in (
        {},
        {"reveal": 15.0},
        {"transcript_reveal_cps": "15"},
        {"transcript_reveal_cps": True},
    ):
        assert recorded_run_settings("smallest", bad) != bad
    assert recorded_run_settings("openai", None) is None
    assert (
        recorded_run_settings("openai", {"x": 1}) is None
    )  # differs: no settings declared
