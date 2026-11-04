"""Language settings must reach providers and survive result serialization."""

import pytest

from tau2.data_model.cascaded import CascadedConfigRequest
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.voice.voice_gender import selected_agent_voice


@pytest.mark.parametrize("language", ["en", "es", "pt", "hi", "zh", "ko"])
@pytest.mark.parametrize("preset", [None, "default", "openai-thinking"])
def test_language_cascade_is_resolved_and_recorded(language, preset):
    audio = AudioNativeConfig(
        provider="livekit", cascaded_request=CascadedConfigRequest(preset=preset)
    )
    run = VoiceRunConfig(domain="retail", language=language, audio_native_config=audio)
    resolved = run.audio_native_config
    assert resolved.language == language
    assert resolved.cascaded_config.stt.language.split("-")[0] == language
    assert resolved.cascaded_config.tts.language == language
    assert resolved.cascaded_config.tts.model == "eleven_flash_v2_5"
    assert selected_agent_voice(run) == ("9BWtsMINqrJLrRacOk9x", "female")
    assert audio.cascaded_config is None
    recorded = VoiceRunConfig.model_validate_json(run.model_dump_json())
    assert recorded.audio_native_config == resolved


def test_shared_audio_config_does_not_leak_language_across_runs():
    audio = AudioNativeConfig(provider="livekit")
    before = audio.model_dump()
    english = VoiceRunConfig(domain="retail", language="en", audio_native_config=audio)
    korean = VoiceRunConfig(domain="retail", language="ko", audio_native_config=audio)
    assert english.audio_native_config.cascaded_config.stt.language == "en"
    assert korean.audio_native_config.cascaded_config.stt.language == "ko"
    assert audio.model_dump() == before


def test_provider_language_cannot_disagree_with_run():
    with pytest.raises(ValueError, match="language"):
        VoiceRunConfig(
            domain="retail",
            language="ko",
            audio_native_config=AudioNativeConfig(provider="xai", language="es"),
        )


def test_default_run_uses_the_english_language_cascade():
    run = VoiceRunConfig(
        domain="retail", audio_native_config=AudioNativeConfig(provider="livekit")
    )
    explicit = VoiceRunConfig(
        domain="retail",
        language="en",
        audio_native_config=AudioNativeConfig(provider="livekit"),
    )
    cascade = run.audio_native_config.cascaded_config
    assert cascade == explicit.audio_native_config.cascaded_config
    assert cascade.stt.language == "en"
    assert cascade.tts.language == "en"


def test_missing_optional_plugin_does_not_block_installed_plugins(monkeypatch):
    import importlib

    from tau2.voice.audio_native.livekit import preregister_livekit_plugins

    attempts = []

    def import_plugin(name):
        attempts.append(name)
        if name == "livekit.plugins.anthropic":
            raise ImportError("optional dependency missing")

    monkeypatch.setattr(importlib, "import_module", import_plugin)
    preregister_livekit_plugins()
    assert attempts == [
        "livekit.plugins.anthropic",
        "livekit.plugins.deepgram",
        "livekit.plugins.elevenlabs",
        "livekit.plugins.openai",
    ]


def test_livekit_gender_guidance_uses_selected_voice():
    from types import SimpleNamespace

    from tau2.multilingual.runtime import agent_language_guidance

    run = VoiceRunConfig(
        domain="retail",
        language="hi",
        audio_native_config=AudioNativeConfig(provider="livekit"),
    )
    context = SimpleNamespace(
        language="hi",
        cross_lingual=True,
        scenario_mode="canonical",
        agent_language_clause="Speak Hindi.",
        domain_agent_guidance=None,
        persona=SimpleNamespace(agent_background_clause=""),
    )

    def guidance_for(run):
        return agent_language_guidance(
            context, agent_voice_gender=selected_agent_voice(run)[1]
        )

    guidance = guidance_for(run)
    assert guidance.startswith("Speak Hindi.\n")
    assert "Use the canonical database values" in guidance
    assert guidance.endswith("\nYour gender is female.")
    run.audio_native_config.cascaded_config.tts.voice_id = "unknown-custom-voice"
    assert selected_agent_voice(run) == ("unknown-custom-voice", None)
    assert guidance_for(run) == guidance.removesuffix("\nYour gender is female.")


def test_explicit_supported_deepgram_voice_has_matching_gender():
    from tau2.data_model.cascaded import CascadedConfig, DeepgramTTSConfig

    run = VoiceRunConfig(
        domain="retail",
        language="en",
        audio_native_config=AudioNativeConfig(
            provider="livekit",
            cascaded_request=CascadedConfigRequest(
                config=CascadedConfig(
                    tts=DeepgramTTSConfig(model="aura-luna-en"),
                )
            ),
        ),
    )
    assert selected_agent_voice(run) == ("aura-luna-en", "female")
