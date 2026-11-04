"""Runtime contracts of shipped packs: voice overrides, cross-lingual, backchannel."""

import pytest
from pydantic import ValidationError

from tau2.backchannel import (
    MAX_BACKCHANNEL_PHRASES,
    BackchannelLevel,
    BackchannelSettings,
    default_backchannel_settings,
)
from tau2.data_model.language import cross_lingual
from tau2.multilingual.runtime import (
    list_languages,
    load_language,
    load_pack,
    resolve_persona,
    resolve_task,
)
from tau2.runner.build import backchannel_settings


@pytest.mark.parametrize("language", ["en", "es", "pt"])
def test_voice_id_env_override_replaces_only_the_named_persona(language, monkeypatch):
    pack = load_pack(language)
    target, *others = sorted(pack.personas)
    key = f"TAU2_VOICE_ID_{language}_{target}".replace("-", "_").upper()
    monkeypatch.setenv(key, "override-voice")
    resolved = resolve_persona(pack, target, mode="voice")
    assert resolved.voice.voice_id == "override-voice"
    for persona_id in others:
        assert (
            resolve_persona(pack, persona_id, mode="voice").voice.voice_id
            == pack.personas[persona_id].voice.voice_id
        )
    # Text calls carry no voice at all, so the override cannot leak into them.
    assert resolve_persona(pack, target, mode="text").voice is None
    monkeypatch.delenv(key)
    assert (
        resolve_persona(pack, target, mode="voice").voice.voice_id
        == pack.personas[target].voice.voice_id
    )


@pytest.mark.parametrize(
    "language,expected",
    [("en", False), ("en-US", False), ("es", True), ("pt-BR", True), ("zh", True)],
)
def test_cross_lingual_is_relative_to_the_scenario_language(language, expected):
    assert cross_lingual(language) is expected


@pytest.mark.parametrize("language", list_languages())
def test_shipped_personas_declare_directives_exactly_when_cross_lingual(language):
    pack = load_pack(language)
    assert all(
        (persona.language_directive is not None) == cross_lingual(language)
        for persona in pack.personas.values()
    )


def test_resolved_voice_call_carries_its_persona_backchannel_settings():
    loaded = load_language("es", "retail", mode="voice")
    context = resolve_task(loaded, loaded.tasks[0], seed=0)
    settings = backchannel_settings(context)
    assert settings.level == loaded.pack.voice.backchannel_level
    assert settings.phrases == context.persona.voice.backchannel_phrases
    assert settings.language_name == context.persona.display_language
    assert context.backchannel_level == settings.level
    assert context.backchannel_prompt_version is not None


def test_text_calls_record_no_backchannel_settings():
    loaded = load_language("es", "retail", mode="text")
    context = resolve_task(loaded, loaded.tasks[0], seed=0)
    assert context.backchannel_level is None
    assert context.backchannel_prompt_version is None


@pytest.mark.parametrize("language", list_languages())
def test_domain_owned_callers_use_the_canonical_persona_continuers(language):
    pack = load_pack(language)
    settings = default_backchannel_settings(language)
    canonical = pack.personas[sorted(pack.personas)[0]]
    assert settings.phrases == canonical.voice.backchannel_phrases
    assert settings.level == pack.voice.backchannel_level
    explicit = default_backchannel_settings(language, level=BackchannelLevel.HIGH)
    assert explicit.level == BackchannelLevel.HIGH


def test_backchannel_settings_bound_the_continuer_inventory():
    with pytest.raises(ValidationError):
        BackchannelSettings(level="medium", language_name="English", phrases=[])
    with pytest.raises(ValidationError):
        BackchannelSettings(
            level="medium",
            language_name="English",
            phrases=["Mm"] * (MAX_BACKCHANNEL_PHRASES + 1),
        )
