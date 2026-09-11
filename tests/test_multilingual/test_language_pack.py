# Copyright Sierra
"""Tests for the Language Pack registry (tau2.multilingual).

Covers: schema validation, pack/persona registration, VoicePersona
integration, persona guidelines text, voice-config sampling with a persona
override, SpeechEnvironment metadata, guidelines fallback, and — critically —
that English default behavior is unchanged when no pack persona is active.
"""

import pytest
from pydantic import ValidationError

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.registry as ml_registry
from tau2.data_model.persona import Verbosity
from tau2.data_model.voice import SynthesisConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.multilingual.schema import (
    AcousticPreset,
    LanguagePack,
    MultilingualPersonaConfig,
    english_prompt_string,
)
from tau2.user.user_simulator import (
    get_global_user_sim_guidelines_voice,
)
from tau2.user_simulation_voice_presets import sample_voice_config

# Snapshot/restore the pack + voice-persona registries around every test
# (shared fixture in test_multilingual/conftest.py).
pytestmark = pytest.mark.usefixtures("clean_registries")


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


class TestSchemaRejectsUnknownKeys:
    """Pack authoring typos must fail LOUDLY at load, not silently no-op.

    Historic repro: ``backchannel_levl: high`` loaded fine and silently kept
    the English default. Every model in the schema module carries
    ``extra='forbid'`` so an unknown key anywhere in a pack.yaml raises."""

    def test_typoed_pack_key_rejected(self):
        data = make_test_pack().model_dump(mode="json")
        data["backchannel_levl"] = "high"
        with pytest.raises(ValidationError, match="backchannel_levl"):
            LanguagePack.model_validate(data)

    def test_typoed_persona_key_rejected(self):
        data = make_test_pack().model_dump(mode="json")
        data["personas"]["tara_testlang_v1"]["pragmatic_clauses"] = ["typo"]
        with pytest.raises(ValidationError, match="pragmatic_clauses"):
            LanguagePack.model_validate(data)

    def test_every_schema_model_forbids_unknown_keys(self):
        """Coverage guard: no model in the schema module may tolerate extras."""
        import inspect

        from pydantic import BaseModel

        from tau2.multilingual import schema as schema_mod

        tolerant = [
            name
            for name, obj in vars(schema_mod).items()
            if inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == schema_mod.__name__
            and obj.model_config.get("extra") != "forbid"
        ]
        assert not tolerant, f"schema models tolerate unknown keys: {tolerant}"


class TestEnglishPromptScaffold:
    def test_scaffold_field_is_language_parameterized(self):
        assert (
            english_prompt_string("persona_language_header", "Testlang")
            == "## PERSONA AND LANGUAGE"
        )
        assert "Testlang" in english_prompt_string("localization_header", "Testlang")

    def test_unknown_field_names_the_legal_ones(self):
        """``end_reminder`` has no English scaffold entry (its source is the
        versioned reminder template), so the lookup names the legal fields
        instead of raising a bare KeyError."""
        with pytest.raises(ValueError, match="not a prompt-scaffold field"):
            english_prompt_string("end_reminder", "Testlang")


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
        assert "testgreeting" in text
        assert "You assume the agent speaks your language." in text
        # Inherited PersonaConfig behavior still present (minimal verbosity)
        assert "MINIMAL VERBOSITY" in text

    @pytest.mark.parametrize("mode", ["voice", "text"])
    def test_tts_voice_prompt_never_reaches_llm_guidelines(self, mode):
        # The voice prompt is TTS voice-DESIGN material (affect: warm,
        # composed, pacing) — splicing it into the behavioral prompt would
        # prescribe an attitude that can contradict the task persona, which
        # owns attitude. It must not appear in either mode.
        persona = make_test_pack().personas["tara_testlang_v1"]
        assert persona.tts_voice_prompt  # the fixture carries one
        text = persona.to_guidelines_text(mode=mode)
        assert persona.tts_voice_prompt.strip() not in text

    def test_guidelines_text_empty_persona_content(self):
        # A persona with no clauses renders nothing — the tts_voice_prompt
        # alone contributes no LLM guidelines (voice-design only).
        persona = (
            make_test_pack()
            .personas["tara_testlang_v1"]
            .model_copy(
                update={
                    "pragmatics_clauses": [],
                    "verbosity": Verbosity.STANDARD,
                }
            )
        )
        assert persona.tts_voice_prompt
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
    def test_a_packs_localized_guidelines_are_never_read(self, tmp_path):
        """The localized guidelines file belongs to the retired native
        prompt-language arm: registering a pack that declares one must not
        change what any run reads."""
        english = get_global_user_sim_guidelines_voice()
        guidelines = tmp_path / "simulation_guidelines_voice_xx.md"
        guidelines.write_text("LOCALIZED GUIDELINES <PERSONA_GUIDELINES>")
        ml_registry.register_language_pack(
            make_test_pack(guidelines_voice_path=guidelines)
        )
        assert get_global_user_sim_guidelines_voice() == english
        assert "LOCALIZED GUIDELINES" not in english
