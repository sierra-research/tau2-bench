"""Every released voice persona ships the prompts to recreate its voice."""

import pytest

from tau2.multilingual.runtime import load_pack, voice_id_env_key
from tau2.voice.scripts.setup_voices import (
    design_targets,
    load_voice_design,
    released_voice_languages,
    resolve_languages,
)

VOICE_LANGUAGES = released_voice_languages()


def test_voice_v2_languages_are_the_released_cells():
    assert sorted(VOICE_LANGUAGES) == ["en", "es", "hi", "ko", "pt", "zh"]


@pytest.mark.parametrize("language", VOICE_LANGUAGES)
def test_every_voice_v2_persona_has_a_design_prompt(language):
    pack = load_pack(language)
    design = load_voice_design(language)
    assert set(design.personas) == set(pack.personas)
    for persona_id, persona in design.personas.items():
        assert pack.personas[persona_id].voice is not None
        assert persona.gender == pack.personas[persona_id].gender
        assert persona.prompt.strip()
        assert len(persona.audition_text) >= 100


def test_language_selection_defaults_to_english_and_expands_all():
    assert resolve_languages(None) == ["en"]
    assert resolve_languages(["all"]) == VOICE_LANGUAGES
    assert resolve_languages(["es", "all", "es"])[0] == "es"
    assert sorted(resolve_languages(["es", "all"])) == sorted(VOICE_LANGUAGES)


def test_voice_names_and_env_keys_are_unique_across_languages():
    targets = design_targets(VOICE_LANGUAGES)
    assert len({t.voice_name for t in targets}) == len(targets)
    assert [t.env_key for t in targets] == [
        voice_id_env_key(t.language, t.persona_id) for t in targets
    ]
    assert all(t.voice_name.startswith(f"tau2_{t.language}_") for t in targets)


def test_unknown_language_has_no_prompts():
    with pytest.raises(ValueError, match="no published language pack"):
        load_voice_design("xx")
