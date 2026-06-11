# Copyright Sierra
"""Tests for the Language Pack registry (tau2.multilingual).

Covers: schema validation, pack/persona registration, VoicePersona
integration, persona guidelines text, voice-config sampling with a persona
override, SpeechEnvironment metadata, guidelines fallback, and — critically —
that English default behavior is unchanged when no pack persona is active.
"""

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.data_model.persona import Verbosity
from tau2.data_model.voice import SynthesisConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.multilingual.schema import (
    AcousticPreset,
    LanguagePack,
    MultilingualPersonaConfig,
)
from tau2.user.user_simulator import (
    get_global_user_sim_guidelines_voice,
)
from tau2.user_simulation_voice_presets import sample_voice_config


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
        pragmatics_clauses=[
            "You greet with 'testgreeting'.",
            "You assume the agent speaks your language.",
        ],
        backchannel_phrases=["mhm-xx", "ji-xx"],
        non_directed_phrases=["one moment (to family)"],
        voice_id="fake_voice_id_123",
        tts_voice_prompt="A test speaker in her 30s.",
        acoustic_preset_id="xx_market",
        verbosity=Verbosity.MINIMAL,
    )
    fields = dict(
        language="xx",
        display_name="Testlang",
        personas={persona.persona_id: persona},
        acoustic_presets={
            "xx_market": AcousticPreset(
                id="xx_market",
                display_name="Test market",
                background_noise_files=["xx/market.wav"],
                burst_noise_files=["xx/bell.wav"],
            )
        },
    )
    fields.update(overrides)
    return LanguagePack(**fields)


class TestSchema:
    def test_persona_language_must_match_pack(self):
        pack = make_test_pack()
        persona = pack.personas["tara_testlang_v1"].model_copy(
            update={"language": "yy"}
        )
        with pytest.raises(ValueError, match="language"):
            make_test_pack(personas={persona.persona_id: persona})

    def test_unknown_acoustic_preset_rejected(self):
        pack = make_test_pack()
        persona = pack.personas["tara_testlang_v1"].model_copy(
            update={"acoustic_preset_id": "missing"}
        )
        with pytest.raises(ValueError, match="acoustic"):
            make_test_pack(personas={persona.persona_id: persona})

    def test_persona_key_must_match_persona_id(self):
        pack = make_test_pack()
        persona = pack.personas["tara_testlang_v1"]
        with pytest.raises(ValueError, match="does not match"):
            make_test_pack(personas={"wrong_key": persona})

    def test_persona_phrase_lists_and_preset_lookup(self):
        pack = make_test_pack()
        persona = pack.personas["tara_testlang_v1"]
        assert persona.backchannel_phrases == ["mhm-xx", "ji-xx"]
        assert persona.non_directed_phrases == ["one moment (to family)"]
        assert pack.get_acoustic_preset(persona).id == "xx_market"


class TestRegistry:
    def test_register_and_lookup(self):
        pack = make_test_pack()
        ml_registry.register_language_pack(pack)
        assert ml_registry.get_language_pack("xx") is pack
        assert "xx" in ml_registry.list_language_packs()
        found = ml_registry.get_multilingual_persona("tara_testlang_v1")
        assert found is not None
        assert found[0] is pack
        assert found[1].persona_id == "tara_testlang_v1"

    def test_duplicate_language_rejected(self):
        ml_registry.register_language_pack(make_test_pack())
        with pytest.raises(ValueError, match="already registered"):
            ml_registry.register_language_pack(make_test_pack())

    def test_unknown_lookups_return_none(self):
        assert ml_registry.get_language_pack("zz") is None
        assert ml_registry.get_multilingual_persona("nobody") is None

    def test_voice_persona_registered_but_not_in_sampling_pools(self):
        ml_registry.register_language_pack(make_test_pack())
        assert get_elevenlabs_voice_id("tara_testlang_v1") == "fake_voice_id_123"
        vp = voice_personas.ALL_PERSONAS["tara_testlang_v1"]
        assert vp.language == "xx"
        assert vp.prompt == "A test speaker in her 30s."
        assert "tara_testlang_v1" not in voice_personas.CONTROL_PERSONA_NAMES
        assert "tara_testlang_v1" not in voice_personas.REGULAR_PERSONA_NAMES

    def test_duplicate_persona_id_rejected(self):
        ml_registry.register_language_pack(make_test_pack())
        persona = make_test_pack().personas["tara_testlang_v1"]
        other = make_test_pack(
            language="yy",
            personas={
                persona.persona_id: persona.model_copy(update={"language": "yy"})
            },
        )
        with pytest.raises(ValueError, match="already registered"):
            ml_registry.register_language_pack(other)


class TestPersonaGuidelines:
    def test_guidelines_text_is_author_content_verbatim(self):
        persona = make_test_pack().personas["tara_testlang_v1"]
        text = persona.to_guidelines_text()
        assert "PERSONA AND LANGUAGE" in text
        assert "A test speaker in her 30s." in text
        assert "testgreeting" in text
        assert "You assume the agent speaks your language." in text
        # Inherited PersonaConfig behavior still present (minimal verbosity)
        assert "MINIMAL VERBOSITY" in text

    def test_guidelines_text_empty_persona_content(self):
        persona = (
            make_test_pack()
            .personas["tara_testlang_v1"]
            .model_copy(
                update={
                    "tts_voice_prompt": "",
                    "pragmatics_clauses": [],
                    "verbosity": Verbosity.STANDARD,
                }
            )
        )
        assert persona.to_guidelines_text() is None


class TestSampling:
    def test_persona_override_flows_to_speech_environment(self):
        ml_registry.register_language_pack(make_test_pack())
        sampled = sample_voice_config(
            seed=7,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name="tara_testlang_v1",
        )
        assert sampled.persona_name == "tara_testlang_v1"
        assert isinstance(sampled.persona_config, MultilingualPersonaConfig)
        assert sampled.persona_config.language == "xx"
        # Acoustic preset replaces indoor/outdoor environment selection
        assert sampled.environment == "xx_market"

        env = sampled.to_speech_environment(seed=7)
        assert env.language == "xx"
        assert env.locale == "XX-TS"
        assert env.persona_id == "tara_testlang_v1"
        assert env.voice_id == "fake_voice_id_123"

    def test_english_default_unchanged(self):
        ml_registry.register_language_pack(make_test_pack())
        sampled = sample_voice_config(
            seed=7,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
        )
        assert sampled.persona_name != "tara_testlang_v1"
        assert not isinstance(sampled.persona_config, MultilingualPersonaConfig)
        assert sampled.environment in ("indoor", "outdoor")
        env = sampled.to_speech_environment(seed=7)
        assert env.language is None
        assert env.locale is None
        assert env.persona_id is None


class TestGuidelines:
    def test_language_without_pack_falls_back_to_english(self):
        english = get_global_user_sim_guidelines_voice()
        assert get_global_user_sim_guidelines_voice(language="zz") == english

    def test_pack_guidelines_used_when_present(self, tmp_path):
        guidelines = tmp_path / "simulation_guidelines_voice_xx.md"
        guidelines.write_text("LOCALIZED GUIDELINES <PERSONA_GUIDELINES>")
        ml_registry.register_language_pack(
            make_test_pack(guidelines_voice_path=guidelines)
        )
        text = get_global_user_sim_guidelines_voice(language="xx")
        assert text.startswith("LOCALIZED GUIDELINES")

    def test_pack_without_guidelines_falls_back(self):
        ml_registry.register_language_pack(make_test_pack())
        english = get_global_user_sim_guidelines_voice()
        assert get_global_user_sim_guidelines_voice(language="xx") == english
