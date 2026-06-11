# Copyright Sierra
"""Tests for persona/language-driven backchannel and side-talk phrases (PR2).

Covers: persona side-talk phrases and pack-level rates flowing into
SpeechEffectsConfig, backchannel phrase/decision-prompt/Poisson-rate
resolution in the streaming user simulator, and — critically — that English
defaults are byte-identical when no language-pack persona is active.
"""

from typing import Optional

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.data_model.persona import PersonaConfig, Verbosity
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig
from tau2.user.user_simulator_streaming import (
    BACKCHANNEL_DECISION_PROMPT,
    VoiceStreamingUserSimulator,
)
from tau2.user_simulation_voice_presets import REGULAR_CONFIG, sample_voice_config
from tau2.voice_config import BACKCHANNEL_PHRASES, NON_DIRECTED_PHRASES, VOCAL_TICS

XX_BACKCHANNEL_PHRASES = ["mhm-xx", "ji-xx"]
XX_SIDE_TALK_PHRASES = ["एक मिनट रुको", "अभी फोन पर हूँ"]
XX_DECISION_PROMPT = "Localized backchannel prompt.\n\n{conversation_history}"


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


def make_test_pack(
    backchannel_phrases: Optional[list[str]] = None,
    side_talk_phrases: Optional[list[str]] = None,
    **pack_overrides,
) -> LanguagePack:
    """Build a one-persona pack with the given persona phrase lists (or none)."""
    persona = MultilingualPersonaConfig(
        persona_id="tara_testlang_v1",
        display_name="Tara",
        short_description="Test-language persona",
        language="xx",
        backchannel_phrases=backchannel_phrases,
        side_talk_phrases=side_talk_phrases,
        voice_id="fake_voice_id_123",
        verbosity=Verbosity.MINIMAL,
    )
    fields = dict(
        language="xx",
        display_name="Testlang",
        personas={persona.persona_id: persona},
    )
    fields.update(pack_overrides)
    return LanguagePack(**fields)


def sample_for_persona(persona_name: Optional[str] = None):
    return sample_voice_config(
        seed=7,
        synthesis_config=SynthesisConfig(),
        complexity="regular",
        persona_name=persona_name,
    )


def make_simulator(persona_config: Optional[PersonaConfig]):
    return VoiceStreamingUserSimulator(
        tools=None,
        instructions="You are a test user.",
        llm="gpt-4o-mini",
        voice_settings=VoiceSettings(
            transcription_config=None,
            synthesis_config=SynthesisConfig(),
        ),
        backchannel_min_threshold=3,
        backchannel_poisson_rate=0.1,
        persona_config=persona_config,
    )


class TestSideTalkLocalization:
    def test_persona_phrases_reach_speech_effects(self):
        ml_registry.register_language_pack(
            make_test_pack(side_talk_phrases=XX_SIDE_TALK_PHRASES)
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert [i.text for i in speech.non_directed_phrases] == XX_SIDE_TALK_PHRASES
        assert all(i.type == "non_directed_phrase" for i in speech.non_directed_phrases)

    def test_pack_rate_applies(self):
        ml_registry.register_language_pack(
            make_test_pack(
                side_talk_phrases=XX_SIDE_TALK_PHRASES,
                default_out_of_turn_events_per_minute=2.0,
            )
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert speech.speech_insert_events_per_minute == 2.0

    def test_rate_falls_back_to_preset_value(self):
        ml_registry.register_language_pack(
            make_test_pack(side_talk_phrases=XX_SIDE_TALK_PHRASES)
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert (
            speech.speech_insert_events_per_minute
            == REGULAR_CONFIG["speech_insert_events_per_minute"]
        )

    def test_pack_rate_applies_without_persona_phrases(self):
        ml_registry.register_language_pack(
            make_test_pack(default_out_of_turn_events_per_minute=2.5)
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        # No persona phrases: English phrases, but the language-level rate applies
        assert [i.text for i in speech.non_directed_phrases] == NON_DIRECTED_PHRASES
        assert speech.speech_insert_events_per_minute == 2.5

    def test_pack_persona_without_overrides_keeps_english_values(self):
        ml_registry.register_language_pack(make_test_pack())
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert [i.text for i in speech.non_directed_phrases] == NON_DIRECTED_PHRASES
        assert (
            speech.speech_insert_events_per_minute
            == REGULAR_CONFIG["speech_insert_events_per_minute"]
        )


class TestEnglishRegression:
    def test_english_sampling_unchanged(self):
        """No persona override: every speech-effects value matches today's English."""
        ml_registry.register_language_pack(
            make_test_pack(
                side_talk_phrases=XX_SIDE_TALK_PHRASES,
                default_out_of_turn_events_per_minute=8.8,
            )
        )
        speech = sample_for_persona().speech_effects_config
        assert [i.text for i in speech.non_directed_phrases] == NON_DIRECTED_PHRASES
        assert [i.text for i in speech.vocal_tics] == VOCAL_TICS
        assert (
            speech.speech_insert_events_per_minute
            == REGULAR_CONFIG["speech_insert_events_per_minute"]
        )

    def test_english_simulator_unchanged(self):
        """Plain PersonaConfig: phrases, prompt, and rate are the English defaults."""
        simulator = make_simulator(PersonaConfig())
        assert simulator.backchannel_phrases is BACKCHANNEL_PHRASES
        assert simulator.backchannel_decision_prompt is BACKCHANNEL_DECISION_PROMPT
        assert simulator.backchannel_poisson_rate == 0.1

    def test_pack_persona_without_overrides_keeps_english_backchannels(self):
        pack = make_test_pack()
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_phrases is BACKCHANNEL_PHRASES
        assert simulator.backchannel_decision_prompt is BACKCHANNEL_DECISION_PROMPT
        assert simulator.backchannel_poisson_rate == 0.1


class TestBackchannelLocalization:
    def test_persona_phrases_used(self):
        pack = make_test_pack(backchannel_phrases=XX_BACKCHANNEL_PHRASES)
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_phrases == XX_BACKCHANNEL_PHRASES

    def test_pack_decision_prompt_used(self):
        pack = make_test_pack(
            backchannel_phrases=XX_BACKCHANNEL_PHRASES,
            backchannel_decision_prompt=XX_DECISION_PROMPT,
        )
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_decision_prompt == XX_DECISION_PROMPT

    def test_decision_prompt_falls_back_to_english(self):
        pack = make_test_pack(backchannel_phrases=XX_BACKCHANNEL_PHRASES)
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_decision_prompt is BACKCHANNEL_DECISION_PROMPT

    def test_pack_poisson_rate_used(self):
        pack = make_test_pack(
            backchannel_phrases=XX_BACKCHANNEL_PHRASES,
            default_backchannel_poisson_rate=0.3,
        )
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_poisson_rate == 0.3

    def test_poisson_rate_falls_back_to_constructor_value(self):
        pack = make_test_pack(backchannel_phrases=XX_BACKCHANNEL_PHRASES)
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_poisson_rate == 0.1
