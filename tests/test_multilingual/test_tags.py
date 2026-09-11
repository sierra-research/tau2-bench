# Copyright Sierra
"""Tests for the persona tag vocabulary and its plumbing.

Three layers under test:
- schema level: tags PROVIDED on a persona must be in-vocabulary (unknown
  dimension or value -> validation error), but completeness is NOT
  schema-enforced (packs without tags keep loading mid-migration);
- data level: the hi pack is fully backfilled — every persona
  carries every REQUIRED_TAG_DIMENSIONS dimension;
- runtime level: SpeechEnvironment carries the sampled persona's tags so
  persisted runs can be joined to tags without loading packs.
"""

import pytest
from pydantic import ValidationError

from tau2.data_model.voice import SpeechEnvironment, SynthesisConfig
from tau2.multilingual.loader import load_language_packs
from tau2.multilingual.registry import get_language_pack, list_language_packs
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.multilingual.tags import (
    REQUIRED_TAG_DIMENSIONS,
    TAG_VOCABULARY,
    validate_tags,
)
from tau2.user_simulation_voice_presets import sample_voice_config


def make_persona(**overrides) -> MultilingualPersonaConfig:
    fields = dict(
        persona_id="tess_xx_v1",
        display_name="Tess",
        short_description="Test persona",
        language="xx",
        script="latn",
    )
    fields.update(overrides)
    return MultilingualPersonaConfig(**fields)


class TestVocabulary:
    def test_required_dimensions_are_in_vocabulary(self):
        assert set(REQUIRED_TAG_DIMENSIONS) <= set(TAG_VOCABULARY)

    def test_validate_tags_reports_all_problems(self):
        problems = validate_tags({"nope": "x", "formality": "extreme"})
        assert len(problems) == 2
        assert any("unknown tag dimension 'nope'" in p for p in problems)
        assert any("invalid value 'extreme'" in p for p in problems)

    def test_register_covers_urban_gig_service_work(self):
        """The added urban gig/service/manual values are in-vocabulary so a
        Bucharest ride-share/delivery driver no longer force-fits
        'small_business' (see PR #305)."""
        for value in ("gig_worker", "service_worker", "manual_trade"):
            assert value in TAG_VOCABULARY["register"]
            assert validate_tags({"register": value}) == []

    def test_register_legacy_values_unchanged(self):
        """Appending must never rename/remove existing values (persisted runs
        join on them)."""
        assert {
            "urban_professional",
            "homemaker",
            "small_business",
            "student",
            "retiree",
            "rural",
        } <= set(TAG_VOCABULARY["register"])

    def test_register_is_optional(self):
        assert "register" not in REQUIRED_TAG_DIMENSIONS

    def test_environment_is_not_an_editable_tag(self):
        """Environment is implied by the persona's acoustic_preset_id (the bed),
        not chosen via a tag — so it is neither a vocabulary dimension nor a
        required one, and supplying it is rejected as an unknown dimension."""
        assert "environment" not in TAG_VOCABULARY
        assert "environment" not in REQUIRED_TAG_DIMENSIONS
        problems = validate_tags({"environment": "traffic"})
        assert any("unknown tag dimension 'environment'" in p for p in problems)


class TestSchemaValidation:
    def test_valid_tags_accepted(self):
        persona = make_persona(tags={"code_switch": "high", "formality": "low"})
        assert persona.tags == {"code_switch": "high", "formality": "low"}

    def test_no_tags_is_valid(self):
        """Completeness is factory-level only; untagged personas still load."""
        assert make_persona().tags == {}

    def test_unknown_dimension_rejected(self):
        with pytest.raises(ValidationError, match="unknown tag dimension"):
            make_persona(tags={"mood": "high"})

    def test_unknown_value_rejected(self):
        with pytest.raises(ValidationError, match="invalid value"):
            make_persona(tags={"age_band": "teens"})


class TestPackBackfill:
    """Every persona in every registered pack is fully tagged."""

    def all_personas(self):
        load_language_packs()
        for language in list_language_packs():
            pack = get_language_pack(language)
            yield from pack.personas.values()

    def test_hi_pack_registered(self):
        load_language_packs()
        assert {"hi"} <= set(list_language_packs())

    def test_all_personas_carry_required_dimensions(self):
        personas = list(self.all_personas())
        assert len(personas) >= 2  # 2 hi
        for persona in personas:
            missing = set(REQUIRED_TAG_DIMENSIONS) - set(persona.tags)
            assert not missing, (
                f"{persona.persona_id}: missing required tag dimensions "
                f"{sorted(missing)}"
            )
            assert not validate_tags(persona.tags)

    def test_backfill_matches_persona_descriptions(self):
        """Spot-check the inferred tags against the persona prose."""
        load_language_packs()
        rishika = get_language_pack("hi").get_persona("rishika_hindi_v1")
        assert rishika.tags["code_switch"] == "high"
        assert rishika.tags["english_tolerance"] == "high"
        assert rishika.tags["closing_style"] == "brisk"
        imran = get_language_pack("hi").get_persona("imran_hindi_v1")
        assert imran.tags["formality"] == "high"
        assert imran.tags["english_tolerance"] == "low"
        assert imran.tags["closing_style"] == "extended"
        # Shipped packs carry no acoustic wiring — every language runs on the
        # shared benchmark environments (locale beds are a factory opt-in).
        assert imran.acoustic_preset_id is None


class TestSpeechEnvironmentTags:
    def sample_env(self, persona_name, task_id=None):
        config = sample_voice_config(
            seed=7,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name=persona_name,
            task_id=task_id,
        )
        return config, config.to_speech_environment(7)

    def test_environment_carries_pack_persona_tags(self):
        config, env = self.sample_env("rishika_hindi_v1")
        pack = get_language_pack("hi")
        assert env.persona_tags == pack.get_persona("rishika_hindi_v1").tags
        assert env.persona_tags  # non-empty after the backfill

    def test_language_code_sampling_carries_assigned_persona_tags(self):
        config, env = self.sample_env("hi", task_id="3_hi")
        pack = get_language_pack("hi")
        assert env.persona_tags == pack.get_persona(config.persona_name).tags

    def test_english_personas_have_empty_tags(self):
        _, env = self.sample_env("priya_patil")
        assert env.persona_tags == {}

    def test_default_is_empty_for_persisted_runs(self):
        """Old persisted SpeechEnvironments (no persona_tags key) load fine."""
        env = SpeechEnvironment.model_validate({"voice_seed": 1})
        assert env.persona_tags == {}
