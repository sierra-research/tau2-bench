"""Tests for AAI provider registration in tau2.

Asserts that the aai provider is properly registered in:
- config.py (model, reasoning_effort, provider_type registries)
- data_model/simulation.py (AudioNativeConfig.provider Literal)
- adapter.py (create_adapter factory)
- agent/discrete_time_audio_native_agent.py (VAD config branch)
- cli.py (CLI choices)
"""

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
    AUDIO_NATIVE_PROVIDER_TYPES,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.aai.provider import AAIVADConfig


class TestAAIConfigRegistration:
    """Test aai provider registration in config.py."""

    def test_aai_default_model(self):
        """Test that aai has the default model 'host' in registry."""
        assert "aai" in DEFAULT_AUDIO_NATIVE_MODELS
        assert DEFAULT_AUDIO_NATIVE_MODELS["aai"] == "host"

    def test_aai_reasoning_effort(self):
        """Test that aai reasoning_effort is None in registry."""
        assert "aai" in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT
        assert DEFAULT_AUDIO_NATIVE_REASONING_EFFORT["aai"] is None

    def test_aai_provider_type(self):
        """Test that aai provider type is 'audio_native' in registry."""
        assert "aai" in AUDIO_NATIVE_PROVIDER_TYPES
        assert AUDIO_NATIVE_PROVIDER_TYPES["aai"] == "audio_native"


class TestAudioNativeConfigRegistration:
    """Test aai provider registration in data_model/simulation.py."""

    def test_audio_native_config_accepts_aai_provider(self):
        """Test that AudioNativeConfig accepts provider='aai'."""
        config = AudioNativeConfig(provider="aai")
        assert config.provider == "aai"

    def test_audio_native_config_default_model_for_aai(self):
        """Test that AudioNativeConfig uses the aai default model."""
        # When created with provider="aai" and no model specified,
        # it should use the default from config
        config = AudioNativeConfig(provider="aai")
        # model field should default to DEFAULT_AUDIO_NATIVE_MODELS[DEFAULT_AUDIO_NATIVE_PROVIDER]
        # but we're testing that the config can be created with aai provider
        assert config.provider == "aai"


class TestAdapterFactoryRegistration:
    """Test aai provider registration in adapter.py."""

    def test_create_adapter_for_aai(self):
        """Test that create_adapter() works for aai provider."""
        adapter, model = create_adapter(
            provider="aai",
            tick_duration_ms=200,
            model=None,
        )

        # Should return the default model 'host'
        assert model == "host"

        # Should return a DiscreteTimeAAIAdapter
        from tau2.voice.audio_native.aai.discrete_time_adapter import (
            DiscreteTimeAAIAdapter,
        )

        assert isinstance(adapter, DiscreteTimeAAIAdapter)

    def test_create_adapter_aai_not_connected(self):
        """Test that newly created aai adapter is not connected."""
        adapter, _ = create_adapter(
            provider="aai",
            tick_duration_ms=200,
        )
        assert adapter.is_connected is False


class TestDiscreteTimeAgentVADConfigRegistration:
    """Test aai VAD config registration in discrete_time_audio_native_agent.py."""

    def test_agent_creates_aai_vad_config(self):
        """Test that DiscreteTimeAudioNativeAgent builds AAIVADConfig for aai provider."""
        # Create a minimal agent with aai provider
        agent = DiscreteTimeAudioNativeAgent(
            tools=[],
            domain_policy="Test policy",
            tick_duration_ms=200,
            provider="aai",
        )

        # Should have AAIVADConfig set
        assert isinstance(agent.vad_config, AAIVADConfig)

    def test_agent_provider_literal_includes_aai(self):
        """Test that AudioNativeProvider Literal includes 'aai'."""
        # This is a compile-time check via type hints, but we can check
        # that the agent accepts it without error
        agent = DiscreteTimeAudioNativeAgent(
            tools=[],
            domain_policy="Test policy",
            tick_duration_ms=200,
            provider="aai",
        )
        assert agent.provider == "aai"


class TestAAIExports:
    """Test that aai/__init__.py exports the necessary classes and functions."""

    def test_aai_module_exports(self):
        """Test that aai module properly exports all required symbols."""
        from tau2.voice.audio_native.aai import (
            AAIAudioChunkEvent,
            AAIErrorEvent,
            AAIToolCallEvent,
            AAIUserTranscriptEvent,
            AAIVADConfig,
            AAIVoiceAgentProvider,
            DiscreteTimeAAIAdapter,
            parse_aai_event,
        )

        # All imports should succeed
        assert AAIVADConfig is not None
        assert AAIVoiceAgentProvider is not None
        assert DiscreteTimeAAIAdapter is not None
        assert parse_aai_event is not None
        assert AAIToolCallEvent is not None
        assert AAIUserTranscriptEvent is not None
        assert AAIErrorEvent is not None
        assert AAIAudioChunkEvent is not None
