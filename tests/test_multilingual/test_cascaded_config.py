"""Offline language-aware cascaded configuration and explicit-setting validation."""

import pytest

from tau2.data_model import cascaded
from tau2.data_model.cascaded import CascadedConfigRequest


@pytest.mark.parametrize("language", ["en", "es", "pt", "hi", "zh", "ko"])
@pytest.mark.parametrize("preset", ["default", "openai-thinking"])
def test_language_defaults_resolve_real_voice_and_keep_llm_and_endpointing(
    language, preset
):
    before = cascaded.CASCADED_CONFIGS[preset].model_dump()
    resolved = cascaded.resolve_cascaded_config(
        config=None, preset_name=preset, language=language
    )
    assert resolved.stt.model == "nova-3"
    assert resolved.stt.language == {"pt": "pt-BR", "zh": "zh-CN"}.get(
        language, language
    )
    assert resolved.tts.model == "eleven_flash_v2_5"
    assert resolved.tts.voice_id == "9BWtsMINqrJLrRacOk9x"
    assert resolved.tts.language == language
    assert resolved.llm.model_dump() == before["llm"]
    assert resolved.stt.endpointing_ms == before["stt"]["endpointing_ms"]
    resolved.stt.endpointing_ms = 123
    resolved.llm.model = "changed-copy"
    assert cascaded.CASCADED_CONFIGS[preset].model_dump() == before


def test_absent_language_preserves_ordinary_configuration_and_copies():
    for supplied in (None, cascaded.CascadedConfig(preamble=True, log_prompts=True)):
        before = (supplied or cascaded.CASCADED_CONFIGS["default"]).model_dump()
        resolved = cascaded.resolve_cascaded_config(
            config=supplied, preset_name=None, language=None
        )
        assert resolved.model_dump() == before
        resolved.stt.model = "mutated-copy"
        assert (supplied or cascaded.CASCADED_CONFIGS["default"]).model_dump() == before


@pytest.mark.parametrize(
    "defect",
    [
        "stt-language",
        "stt-model",
        "tts-english",
        "tts-model",
        "tts-language",
        "unsupported-hint",
    ],
)
def test_explicit_incompatible_components_fail_without_replacement(defect):
    config = cascaded.CascadedConfig(
        stt=cascaded.DeepgramSTTConfig(language="es"),
        tts=cascaded.ElevenLabsTTSConfig(model="eleven_flash_v2_5", language="es"),
    )
    if defect == "stt-language":
        config.stt.language = "en-US"
    elif defect == "stt-model":
        config.stt.model = "unverified-stt-model"
    elif defect == "tts-english":
        config.tts = cascaded.DeepgramTTSConfig()
    elif defect == "tts-model":
        config.tts.model = "unverified-tts-model"
    elif defect == "tts-language":
        config.tts.language = "en"
    else:
        config.tts.model = "eleven_multilingual_v2"
    before = config.model_dump()
    with pytest.raises(ValueError, match="(?:STT|TTS)"):
        cascaded.resolve_cascaded_config(config=config, preset_name=None, language="es")
    assert config.model_dump() == before


def test_explicit_models_and_options_are_preserved_with_verified_language_alias():
    config = cascaded.CascadedConfig(
        stt=cascaded.DeepgramSTTConfig(
            language="pt-BR", endpointing_ms=700, smart_format=True
        ),
        tts=cascaded.ElevenLabsTTSConfig(
            model="eleven_multilingual_v2", language=None, stability=0.3
        ),
        llm=cascaded.AnthropicLLMConfig(model="reviewed-model", temperature=0.2),
    )
    resolved = cascaded.resolve_cascaded_config(
        config=config, preset_name=None, language="pt"
    )
    assert resolved.model_dump() == config.model_dump()
    assert resolved is not config and resolved.stt is not config.stt
    assert (
        resolved.tts.language is None
    )  # Model infers language; no unsupported API hint.


@pytest.mark.parametrize("language", ["unknown", "multi", "zh-HK"])
def test_unreviewed_pack_locales_fail_locally(language):
    with pytest.raises(ValueError, match="Unsupported cascaded language"):
        cascaded.resolve_cascaded_config(
            config=None, preset_name=None, language=language
        )


def test_unverified_stt_alias_is_not_accepted_by_prefix():
    config = cascaded.CascadedConfig(
        stt=cascaded.DeepgramSTTConfig(language="zh-HK"),
        tts=cascaded.ElevenLabsTTSConfig(model="eleven_flash_v2_5", language="zh"),
    )
    with pytest.raises(ValueError, match="STT"):
        cascaded.resolve_cascaded_config(config=config, preset_name=None, language="zh")


def test_preamble_requires_explicit_matching_language(monkeypatch):
    preset = cascaded.CASCADED_CONFIGS["default"].model_copy(deep=True)
    preset.preamble = True
    monkeypatch.setitem(cascaded.CASCADED_CONFIGS, "preamble-fixture", preset)
    with pytest.raises(ValueError, match="preamble"):
        cascaded.resolve_cascaded_config(
            config=None, preset_name="preamble-fixture", language="es"
        )
    preset.preamble_text = "Reviewed locale preamble fixture"
    preset.preamble_language = "es"
    resolved = cascaded.resolve_cascaded_config(
        config=None, preset_name="preamble-fixture", language="es"
    )
    assert resolved.preamble and resolved.preamble_text == preset.preamble_text
    assert resolved.preamble_language == "es"


def test_default_elevenlabs_voice_is_id_and_unknown_preset_is_explicit():
    assert cascaded.ElevenLabsTTSConfig().voice_id == "9BWtsMINqrJLrRacOk9x"
    with pytest.raises(ValueError, match="Unknown cascaded preset"):
        cascaded.resolve_cascaded_config(
            config=None, preset_name="unknown", language=None
        )


def test_explicit_english_deepgram_config_remains_supported():
    config = cascaded.CascadedConfig(
        stt=cascaded.DeepgramSTTConfig(language="en-US", endpointing_ms=500),
        tts=cascaded.DeepgramTTSConfig(model="aura-luna-en"),
    )
    resolved = cascaded.resolve_cascaded_config(
        config=config, preset_name=None, language="en"
    )
    assert resolved.model_dump() == config.model_dump()


def test_run_config_records_resolved_components_without_mutating_shared_input():
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    shared = AudioNativeConfig(
        provider="livekit",
        cascaded_request=CascadedConfigRequest(preset="openai-thinking"),
    )
    before = shared.model_dump()
    runs = [
        VoiceRunConfig(domain="retail", language=language, audio_native_config=shared)
        for language in ("en", "es", "pt", "hi", "zh", "ko")
    ]
    for run in runs:
        effective = run.audio_native_config.cascaded_config
        assert effective.stt.language == {"pt": "pt-BR", "zh": "zh-CN"}.get(
            run.language, run.language
        )
        assert effective.tts.language == run.language
        assert (
            effective.llm.model
            == cascaded.CASCADED_CONFIGS["openai-thinking"].llm.model
        )
        restored = VoiceRunConfig.model_validate_json(run.model_dump_json())
        assert restored.audio_native_config.cascaded_config == effective
    assert shared.model_dump() == before


@pytest.mark.parametrize("language", ["es"])
def test_conflicting_effective_settings_are_not_replaced(language):
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    custom = cascaded.CascadedConfig(
        stt=cascaded.DeepgramSTTConfig(language="es"),
        llm=cascaded.OpenAILLMConfig(model="gpt-5.2", reasoning_effort="high"),
        tts=cascaded.ElevenLabsTTSConfig(voice_id="custom-voice", language="es"),
    )
    audio = AudioNativeConfig(provider="livekit", cascaded_config=custom)
    before = audio.model_dump()
    with pytest.raises(ValueError, match="Recorded LiveKit settings differ"):
        VoiceRunConfig(domain="mock", language=language, audio_native_config=audio)
    assert audio.model_dump() == before

    requested = VoiceRunConfig(
        domain="mock",
        language=language,
        audio_native_config=AudioNativeConfig(
            provider="livekit", cascaded_request=CascadedConfigRequest(config=custom)
        ),
    )
    assert requested.audio_native_config.cascaded_config == custom
    assert VoiceRunConfig.model_validate_json(requested.model_dump_json()) == requested


@pytest.mark.parametrize("preset", [None, "openai-thinking"])
def test_prepared_template_can_resolve_language_defaults(preset):
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    template = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(
            provider="livekit",
            cascaded_request=CascadedConfigRequest(preset=preset) if preset else None,
        ),
    )
    before = template.model_dump()
    # A template's own request retargets; its resolved English values do not.
    with pytest.raises(ValueError, match="provider language"):
        VoiceRunConfig.model_validate({**template.model_dump(), "language": "es"})
    values = template.model_dump(exclude_unset=True)
    values["language"] = "es"
    resolved = VoiceRunConfig.model_validate(values)
    assert resolved.audio_native_config.cascaded_config.stt.language == "es"
    assert (
        resolved.audio_native_config.cascaded_config.llm
        == template.audio_native_config.cascaded_config.llm
    )
    assert template.model_dump() == before


def test_legacy_preset_conflict_is_rejected_without_mutating_input():
    from tau2.data_model.simulation import AudioNativeConfig

    values = {
        "provider": "livekit",
        "cascaded_config_name": "openai-thinking",
        "cascaded_request": {"preset": "default"},
    }
    with pytest.raises(ValueError, match="Legacy cascaded preset conflicts"):
        AudioNativeConfig.model_validate(values)
    assert values["cascaded_config_name"] == "openai-thinking"
    values["cascaded_request"]["preset"] = "openai-thinking"
    audio = AudioNativeConfig.model_validate(values)
    assert audio.cascaded_request.preset == "openai-thinking"


@pytest.mark.parametrize(
    "language,stt,tts",
    [
        ("fr", "fr", "fr"),
        ("fr-FR", "fr", "fr"),
        ("fr-CA", "fr-CA", "fr"),
        ("en-US", "en-US", "en"),
        ("ja", "ja", "ja"),
    ],
)
def test_custom_language_and_locale_supported_without_benchmark_membership(
    language, stt, tts
):
    resolved = CascadedConfigRequest().resolve(language)
    assert resolved.stt.language == stt
    assert resolved.tts.language == tts


def test_model_specific_language_capability():
    config = CascadedConfigRequest().resolve("vi")
    config.tts.model = "eleven_multilingual_v2"
    config.tts.language = None
    with pytest.raises(ValueError, match="does not support"):
        CascadedConfigRequest(config=config).resolve("vi")
