# Copyright Sierra
"""Tests for the Hindi language pack (tau2.multilingual.languages.hi).

Unlike test_language_pack.py (which registers a synthetic 'xx' pack directly),
these tests exercise the REAL Hindi pack as discovered by the loader. The pack
is present in every test session, so we rely on loader discovery rather than
snapshot/restore — and we additionally assert that the pack's mere existence
does not perturb English default sampling.
"""

import re

import pytest

from tau2.data_model.voice import SynthesisConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.multilingual.registry import (
    get_language_pack,
    get_multilingual_persona,
)
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.user.user_simulator import get_global_user_sim_guidelines_voice
from tau2.user_simulation_voice_presets import sample_voice_config

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

RISHIKA_ID = "rishika_hindi_v1"
FATIMA_ID = "fatima_hindi_v1"


class TestLoaderDiscovery:
    def test_pack_discovered(self):
        pack = get_language_pack("hi")
        assert pack is not None
        assert pack.language == "hi"
        assert pack.display_name == "Hindi"
        assert set(pack.personas) == {RISHIKA_ID, FATIMA_ID}

    @pytest.mark.parametrize("persona_id", [RISHIKA_ID, FATIMA_ID])
    def test_personas_discoverable(self, persona_id):
        found = get_multilingual_persona(persona_id)
        assert found is not None
        pack, persona = found
        assert pack.language == "hi"
        assert persona.persona_id == persona_id
        assert persona.language == "hi"
        assert persona.script == "deva"

    def test_voice_ids_registered(self):
        # Real ElevenLabs ids, resolved through the voice-persona registry.
        assert get_elevenlabs_voice_id(RISHIKA_ID) == "nIMjJ45Ta5OdxlwTZvzz"
        assert get_elevenlabs_voice_id(FATIMA_ID) == "COQqplohXYLk7z9YeFmf"


class TestPackValidity:
    def test_guidelines_file_exists(self):
        pack = get_language_pack("hi")
        assert pack.guidelines_voice_path is not None
        assert pack.guidelines_voice_path.exists()

    def test_decision_prompt_has_placeholder(self):
        pack = get_language_pack("hi")
        assert pack.backchannel_decision_prompt is not None
        assert "{conversation_history}" in pack.backchannel_decision_prompt
        # Output contract preserved from the English continuer prompt.
        assert 'ONLY "YES" or "NO"' in pack.backchannel_decision_prompt

    def test_conversation_norm_rates(self):
        pack = get_language_pack("hi")
        # Higher backchannel density than English (0.1/s) and modestly higher
        # out-of-turn rate than English (0.7/min).
        assert pack.default_backchannel_poisson_rate > 0.1
        assert pack.default_out_of_turn_events_per_minute > 0.7

    def test_agent_language_clause(self):
        pack = get_language_pack("hi")
        assert pack.agent_language_clause is not None
        assert "Devanagari" in pack.agent_language_clause
        assert "tool calls" in pack.agent_language_clause

    @pytest.mark.parametrize("persona_id", [RISHIKA_ID, FATIMA_ID])
    def test_phrase_lists_contain_devanagari(self, persona_id):
        persona = get_multilingual_persona(persona_id)[1]
        assert persona.backchannel_phrases
        assert persona.non_directed_phrases
        assert any(DEVANAGARI.search(p) for p in persona.backchannel_phrases)
        assert all(DEVANAGARI.search(p) for p in persona.non_directed_phrases)

    @pytest.mark.parametrize("persona_id", [RISHIKA_ID, FATIMA_ID])
    def test_guidelines_text_includes_clauses_and_voice_prompt(self, persona_id):
        persona = get_multilingual_persona(persona_id)[1]
        text = persona.to_guidelines_text()
        assert text is not None
        assert "PERSONA AND LANGUAGE" in text
        assert persona.tts_voice_prompt.strip() in text
        for clause in persona.pragmatics_clauses:
            assert clause.strip() in text
        # Pragmatics clauses carry inline Devanagari examples.
        assert DEVANAGARI.search(text)

    def test_decision_prompt_has_devanagari_examples(self):
        pack = get_language_pack("hi")
        assert DEVANAGARI.search(pack.backchannel_decision_prompt)


class TestGlobalGuidelines:
    def test_localized_guidelines_returned(self):
        text = get_global_user_sim_guidelines_voice(language="hi")
        english = get_global_user_sim_guidelines_voice()
        assert text != english
        assert "<PERSONA_GUIDELINES>" in text
        assert DEVANAGARI.search(text)
        # Control tokens preserved verbatim in ASCII.
        assert "###STOP###" in text
        assert "###TRANSFER###" in text
        assert "###OUT-OF-SCOPE###" in text
        # Indian spell-letter convention, not NATO.
        assert "Bombay" in text


class TestSampling:
    def test_rishika_sampling_flows_to_speech_environment(self):
        sampled = sample_voice_config(
            seed=11,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name=RISHIKA_ID,
        )
        assert sampled.persona_name == RISHIKA_ID
        assert isinstance(sampled.persona_config, MultilingualPersonaConfig)
        assert sampled.persona_config.language == "hi"

        env = sampled.to_speech_environment(seed=11)
        assert env.language == "hi"
        assert env.locale == "IN-MH"
        assert env.persona_id == RISHIKA_ID
        assert env.voice_id == "nIMjJ45Ta5OdxlwTZvzz"

    def test_rishika_non_directed_phrases_are_hers(self):
        sampled = sample_voice_config(
            seed=3,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name=RISHIKA_ID,
        )
        nd = sampled.persona_config.non_directed_phrases
        assert nd == [
            "भैया अगले सिग्नल पे left लेना",
            "एक सेकंड, मैं call पे हूँ",
            "हाँ बस five minutes में आती हूँ",
        ]

    def test_fatima_sampling(self):
        sampled = sample_voice_config(
            seed=5,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name=FATIMA_ID,
        )
        env = sampled.to_speech_environment(seed=5)
        assert env.language == "hi"
        assert env.locale == "IN-TG"
        assert env.voice_id == "COQqplohXYLk7z9YeFmf"


class TestEnglishRegression:
    def test_default_sampling_unaffected_by_hindi_pack(self):
        # The Hindi pack is registered in this session; default sampling must
        # still produce a plain English voice persona with no language metadata.
        sampled = sample_voice_config(
            seed=7,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
        )
        assert sampled.persona_name not in (RISHIKA_ID, FATIMA_ID)
        assert not isinstance(sampled.persona_config, MultilingualPersonaConfig)
        assert sampled.environment in ("indoor", "outdoor")
        env = sampled.to_speech_environment(seed=7)
        assert env.language is None
        assert env.locale is None
        assert env.persona_id is None

    def test_english_guidelines_unchanged(self):
        # No language => English guidelines; unknown language => English fallback.
        english = get_global_user_sim_guidelines_voice()
        assert get_global_user_sim_guidelines_voice(language="zz") == english
        assert "<PERSONA_GUIDELINES>" in english
