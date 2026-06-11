# Copyright Sierra
"""Tests for run-language plumbing (PR3).

Covers: the agent-side language clause in the audio-native agent's system
prompt (on iff a pack persona is active AND the pack defines a clause),
TranscriptionConfig language defaulting from the active SpeechEnvironment,
provider STT language plumbing (OpenAI realtime, LiveKit cascaded), and —
critically — that English behavior (language=None) is unchanged.
"""

import base64

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.agent.base.voice import VoiceMixin
from tau2.agent.discrete_time_audio_native_agent import (
    DiscreteTimeAudioNativeAgent,
    create_discrete_time_audio_native_agent,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
from tau2.data_model.message import UserMessage
from tau2.data_model.persona import Verbosity
from tau2.data_model.voice import (
    SpeechEnvironment,
    TranscriptionConfig,
    TranscriptionResult,
    VoiceSettings,
)
from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.livekit.config import (
    CASCADED_CONFIGS,
    CascadedConfig,
)

AGENT_CLAUSE = (
    "Respond in Testlang using its native script; mirror the user's "
    "code-switching register; keep tool calls and structured outputs unchanged."
)


@pytest.fixture(autouse=True)
def clean_registries():
    """Snapshot/restore the pack registry and voice persona registry."""
    saved_packs = dict(ml_registry._LANGUAGE_PACKS)
    saved_personas = dict(voice_personas.ALL_PERSONAS)
    saved_names = list(voice_personas.ALL_PERSONA_NAMES)
    saved_loaded = ml_loader._loaded
    ml_loader._loaded = True  # skip discovery; tests register packs directly
    yield
    ml_registry._LANGUAGE_PACKS.clear()
    ml_registry._LANGUAGE_PACKS.update(saved_packs)
    voice_personas.ALL_PERSONAS.clear()
    voice_personas.ALL_PERSONAS.update(saved_personas)
    voice_personas.ALL_PERSONA_NAMES[:] = saved_names
    ml_loader._loaded = saved_loaded


def make_test_pack(**overrides) -> LanguagePack:
    persona = MultilingualPersonaConfig(
        persona_id="tara_testlang_v1",
        display_name="Tara",
        short_description="Test-language persona",
        language="xx",
        locale="XX-TS",
        script="test",
        voice_id="fake_voice_id_123",
        tts_voice_prompt="A test speaker in her 30s.",
        verbosity=Verbosity.MINIMAL,
    )
    fields = dict(
        language="xx",
        display_name="Testlang",
        personas={persona.persona_id: persona},
        agent_language_clause=AGENT_CLAUSE,
    )
    fields.update(overrides)
    return LanguagePack(**fields)


def make_agent(language=None, **kwargs) -> DiscreteTimeAudioNativeAgent:
    return DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="TEST DOMAIN POLICY",
        language=language,
        **kwargs,
    )


class TestAgentLanguageClause:
    def test_clause_appended_when_pack_persona_active(self):
        ml_registry.register_language_pack(make_test_pack())
        agent = make_agent(language="xx")
        assert agent.system_prompt.endswith(AGENT_CLAUSE)
        assert "TEST DOMAIN POLICY" in agent.system_prompt

    def test_clause_off_when_language_is_none(self):
        ml_registry.register_language_pack(make_test_pack())
        agent = make_agent(language=None)
        assert AGENT_CLAUSE not in agent.system_prompt

    def test_clause_off_when_no_pack_exists(self):
        ml_registry.register_language_pack(make_test_pack())
        agent = make_agent(language="zz")
        assert AGENT_CLAUSE not in agent.system_prompt

    def test_clause_off_when_pack_has_no_clause(self):
        ml_registry.register_language_pack(make_test_pack(agent_language_clause=None))
        agent = make_agent(language="xx")
        assert AGENT_CLAUSE not in agent.system_prompt

    def test_english_prompt_unchanged(self):
        """language=None must produce a byte-identical system prompt."""
        ml_registry.register_language_pack(make_test_pack())
        baseline = DiscreteTimeAudioNativeAgent(
            tools=[], domain_policy="TEST DOMAIN POLICY"
        )
        agent = make_agent(language=None)
        assert agent.system_prompt == baseline.system_prompt
        # The prompt ends exactly at the policy (no appended text)
        assert agent.system_prompt.endswith("TEST DOMAIN POLICY")

    def test_factory_threads_language(self):
        ml_registry.register_language_pack(make_test_pack())
        agent = create_discrete_time_audio_native_agent(
            tools=[],
            domain_policy="TEST DOMAIN POLICY",
            language="xx",
        )
        assert agent.language == "xx"
        assert agent.system_prompt.endswith(AGENT_CLAUSE)

    def test_factory_defaults_to_english(self):
        ml_registry.register_language_pack(make_test_pack())
        agent = create_discrete_time_audio_native_agent(
            tools=[],
            domain_policy="TEST DOMAIN POLICY",
        )
        assert agent.language is None
        assert AGENT_CLAUSE not in agent.system_prompt


class _VoiceHolder(VoiceMixin):
    """Minimal VoiceMixin host for exercising transcribe_voice."""


def make_audio_message() -> UserMessage:
    silence = b"\x7f" * 160  # 20ms of mu-law silence
    return UserMessage(
        role="user",
        is_audio=True,
        audio_content=base64.b64encode(silence).decode("utf-8"),
        audio_format=TELEPHONY_AUDIO_FORMAT,
    )


class TestTranscriptionLanguageDefaulting:
    def _capture_transcribe(self, monkeypatch):
        captured = {}

        def fake_transcribe_audio(audio_data, config):
            captured["config"] = config
            return TranscriptionResult(transcript="hello")

        monkeypatch.setattr(
            "tau2.agent.base.voice.transcribe_audio", fake_transcribe_audio
        )
        return captured

    def test_language_defaults_from_speech_environment(self, monkeypatch):
        captured = self._capture_transcribe(monkeypatch)
        transcription_config = TranscriptionConfig()
        holder = _VoiceHolder(
            voice_settings=VoiceSettings(
                transcription_config=transcription_config,
                speech_environment=SpeechEnvironment(language="xx"),
            )
        )
        holder.transcribe_voice(make_audio_message())
        assert captured["config"].language == "xx"
        # The run-level config is not mutated
        assert transcription_config.language is None

    def test_explicit_language_not_overridden(self, monkeypatch):
        captured = self._capture_transcribe(monkeypatch)
        holder = _VoiceHolder(
            voice_settings=VoiceSettings(
                transcription_config=TranscriptionConfig(language="fr"),
                speech_environment=SpeechEnvironment(language="xx"),
            )
        )
        holder.transcribe_voice(make_audio_message())
        assert captured["config"].language == "fr"

    def test_english_path_unchanged(self, monkeypatch):
        captured = self._capture_transcribe(monkeypatch)
        transcription_config = TranscriptionConfig()
        holder = _VoiceHolder(
            voice_settings=VoiceSettings(
                transcription_config=transcription_config,
                speech_environment=SpeechEnvironment(),  # language=None
            )
        )
        holder.transcribe_voice(make_audio_message())
        assert captured["config"].language is None
        assert captured["config"] is transcription_config


class TestProviderLanguagePlumbing:
    def test_openai_provider_language_defaults_to_en(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from tau2.voice.audio_native.openai.discrete_time_adapter import (
            DiscreteTimeOpenAIAdapter,
        )

        adapter = DiscreteTimeOpenAIAdapter(tick_duration_ms=1000)
        assert adapter.provider.language == "en"

    def test_openai_provider_receives_run_language(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        adapter, _ = create_adapter(
            provider="openai", tick_duration_ms=1000, language="xx"
        )
        assert adapter.language == "xx"
        assert adapter.provider.language == "xx"

    def test_livekit_stt_language_overridden_without_mutating_presets(self):
        preset = CASCADED_CONFIGS["default"]
        adapter, _ = create_adapter(
            provider="livekit",
            tick_duration_ms=1000,
            cascaded_config=preset,
            language="xx",
        )
        assert adapter.cascaded_config.stt.language == "xx"
        # Shared preset untouched
        assert preset.stt.language == "en-US"

    def test_livekit_stt_language_default_unchanged_for_english(self):
        adapter, _ = create_adapter(
            provider="livekit",
            tick_duration_ms=1000,
            cascaded_config=CascadedConfig(),
            language=None,
        )
        assert adapter.cascaded_config.stt.language == "en-US"
