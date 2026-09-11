# Copyright Sierra
"""Tests for persona/language-driven backchannel and out-of-turn speech phrases.

Covers: persona out-of-turn speech phrases and pack-level rates flowing into
SpeechEffectsConfig, backchannel phrase/decision-prompt resolution in the
streaming user simulator, and — critically — that English defaults are
byte-identical when no language-pack persona is active.
"""

from typing import Optional

import pytest

import tau2.multilingual.registry as ml_registry
from tau2.data_model.persona import PersonaConfig, Verbosity
from tau2.data_model.voice import SynthesisConfig
from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig
from tau2.user.user_simulator_streaming import BACKCHANNEL_DECISION_PROMPT
from tau2.user_simulation_voice_presets import REGULAR_CONFIG, sample_voice_config
from tau2.voice_config import BACKCHANNEL_PHRASES, NON_DIRECTED_PHRASES, VOCAL_TICS
from test_multilingual.conftest import make_voice_sim

XX_BACKCHANNEL_PHRASES = ["mhm-xx", "ji-xx"]
XX_NON_DIRECTED_PHRASES = ["एक मिनट रुको", "अभी फोन पर हूँ"]


# Snapshot/restore the pack + voice-persona registries around every test
# (shared fixture in test_multilingual/conftest.py).
pytestmark = pytest.mark.usefixtures("clean_registries")


def make_test_pack(
    backchannel_phrases: Optional[list[str]] = None,
    non_directed_phrases: Optional[list[str]] = None,
    **pack_overrides,
) -> LanguagePack:
    """Build a one-persona pack with the given persona phrase lists (or none)."""
    persona = MultilingualPersonaConfig(
        persona_id="tara_testlang_v1",
        display_name="Tara",
        short_description="Test-language persona",
        language="xx",
        locale="XX-TS",
        backchannel_phrases=backchannel_phrases,
        non_directed_phrases=non_directed_phrases,
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
    return make_voice_sim(persona_config, llm="gpt-4o-mini")


class TestNonDirectedPhraseLocalization:
    def test_persona_phrases_reach_speech_effects(self):
        ml_registry.register_language_pack(
            make_test_pack(non_directed_phrases=XX_NON_DIRECTED_PHRASES)
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert [i.text for i in speech.non_directed_phrases] == XX_NON_DIRECTED_PHRASES
        assert all(i.type == "non_directed_phrase" for i in speech.non_directed_phrases)

    def test_pack_rate_applies(self):
        ml_registry.register_language_pack(
            make_test_pack(
                non_directed_phrases=XX_NON_DIRECTED_PHRASES,
                default_out_of_turn_events_per_minute=2.0,
            )
        )
        speech = sample_for_persona("tara_testlang_v1").speech_effects_config
        assert speech.speech_insert_events_per_minute == 2.0

    def test_rate_falls_back_to_preset_value(self):
        ml_registry.register_language_pack(
            make_test_pack(non_directed_phrases=XX_NON_DIRECTED_PHRASES)
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
                non_directed_phrases=XX_NON_DIRECTED_PHRASES,
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

    def test_pack_persona_without_overrides_keeps_english_backchannels(self):
        pack = make_test_pack()
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_phrases is BACKCHANNEL_PHRASES
        assert simulator.backchannel_decision_prompt is BACKCHANNEL_DECISION_PROMPT


class TestBackchannelLocalization:
    def test_persona_phrases_used(self):
        pack = make_test_pack(backchannel_phrases=XX_BACKCHANNEL_PHRASES)
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_phrases == XX_BACKCHANNEL_PHRASES

    def test_decision_prompt_falls_back_to_english(self):
        pack = make_test_pack(backchannel_phrases=XX_BACKCHANNEL_PHRASES)
        ml_registry.register_language_pack(pack)
        simulator = make_simulator(pack.personas["tara_testlang_v1"])
        assert simulator.backchannel_decision_prompt is BACKCHANNEL_DECISION_PROMPT


class TestBackchannelLevelResolution:
    """The density knob resolves env-override > pack.backchannel_level >
    English default."""

    def test_pack_level_renders_knob_prompt(self):
        pack = make_test_pack(
            backchannel_phrases=XX_BACKCHANNEL_PHRASES,
            backchannel_level="high",
        )
        ml_registry.register_language_pack(pack)
        sim = make_simulator(pack.personas["tara_testlang_v1"])
        # rendered (not English default), HIGH, in this pack's display_name
        assert sim.backchannel_decision_prompt is not BACKCHANNEL_DECISION_PROMPT
        assert "Testlang-speaking listener" in sim.backchannel_decision_prompt
        assert "per 1-2 substantive" in sim.backchannel_decision_prompt  # HIGH density
        assert "{conversation_history}" in sim.backchannel_decision_prompt

    def test_env_override_forces_level(self, monkeypatch):
        monkeypatch.setenv("TAU2_BACKCHANNEL_LEVEL", "low")
        pack = make_test_pack(
            backchannel_phrases=XX_BACKCHANNEL_PHRASES,
            backchannel_level="high",
        )
        ml_registry.register_language_pack(pack)
        sim = make_simulator(pack.personas["tara_testlang_v1"])
        # env wins: LOW (per 3-4), not the pack's HIGH (per 1-2)
        assert "per 3-4 substantive" in sim.backchannel_decision_prompt

    def test_env_override_applies_to_english(self, monkeypatch):
        monkeypatch.setenv("TAU2_BACKCHANNEL_LEVEL", "high")
        sim = make_simulator(None)  # no persona => English
        assert sim.backchannel_decision_prompt is not BACKCHANNEL_DECISION_PROMPT
        assert "English-speaking listener" in sim.backchannel_decision_prompt
