"""Offline tests for Cartesia customer synthesis and run wiring."""

from types import SimpleNamespace

import httpx
import pytest

from tau2.data_model.audio import AudioEncoding
from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.data_model.voice import (
    CartesiaTTSConfig,
    ElevenLabsTTSConfig,
    SynthesisConfig,
    VoiceSettings,
)
from tau2.runner.batch import make_voice_run_settings
from tau2.runner.build import build_voice_user
from tau2.user_simulation_voice_presets import sample_voice_config
from tau2.utils.retry import _is_retryable_tts_error
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils.cartesia_utils import tts_cartesia


def test_config_roundtrip_and_provider_isolation(monkeypatch):
    monkeypatch.setenv("TAU2_CARTESIA_VOICE_ID_MATT_DELANEY", "cartesia-matt")
    config = SynthesisConfig(provider="cartesia")
    restored = SynthesisConfig.model_validate_json(config.model_dump_json())
    assert isinstance(restored.provider_config, CartesiaTTSConfig)
    assert restored.resolve_voice_id("matt_delaney") == "cartesia-matt"
    with pytest.raises(ValueError, match="TAU2_CARTESIA_VOICE_ID_LISA_BRENNER"):
        restored.resolve_voice_id("lisa_brenner")
    restored.provider_config.persona_voice_ids["matt_delaney"] = "saved-matt"
    assert restored.resolve_voice_id("matt_delaney") == "saved-matt"
    assert isinstance(SynthesisConfig().provider_config, ElevenLabsTTSConfig)
    with pytest.raises(ValueError, match="Wrong provider_config"):
        SynthesisConfig(provider="cartesia", provider_config=ElevenLabsTTSConfig())


def test_http_audio_contract(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test-only")
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs)
        return httpx.Response(
            200, content=b"\x00\x00\x01\x00", request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("tau2.voice.utils.cartesia_utils.httpx.post", post)
    audio = synthesize_voice(
        "Hi [pause] there", "cartesia", CartesiaTTSConfig(voice_id="hao")
    )
    assert audio.format.encoding == AudioEncoding.PCM_S16LE
    assert audio.format.sample_rate == 16000
    assert audio.data == b"\x00\x00\x01\x00"
    assert seen["json"]["voice"] == "hao"
    assert seen["json"]["transcript"] == 'Hi <break time="500ms"/> there'
    assert seen["json"]["output_format"]["container"] == "raw"
    with pytest.raises(ValueError, match="vocal-tic"):
        tts_cartesia("[cough]", CartesiaTTSConfig(voice_id="hao"))


@pytest.mark.parametrize("body", [b"", b"\x00"])
def test_invalid_audio_rejected(monkeypatch, body):
    monkeypatch.setenv("CARTESIA_API_KEY", "test-only")
    monkeypatch.setattr(
        "tau2.voice.utils.cartesia_utils.httpx.post",
        lambda *a, **k: httpx.Response(
            200, content=body, request=httpx.Request("POST", a[0])
        ),
    )
    with pytest.raises(ValueError, match="PCM16"):
        tts_cartesia("hello", CartesiaTTSConfig(voice_id="hao"))


@pytest.mark.parametrize(
    "status,retry", [(401, False), (422, False), (429, True), (503, True)]
)
def test_http_retry_policy(status, retry):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.cartesia.ai/tts/bytes")
    )
    with pytest.raises(httpx.HTTPStatusError) as error:
        response.raise_for_status()
    assert _is_retryable_tts_error(error.value) is retry


def test_batch_and_task_settings_preserve_cartesia(monkeypatch):
    synthesis = SynthesisConfig(
        provider="cartesia", provider_config=CartesiaTTSConfig(voice_id="hao")
    )
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(),
        user_voice_settings=VoiceSettings(
            transcription_config=None, synthesis_config=synthesis
        ),
    )
    restored = VoiceRunConfig.model_validate_json(config.model_dump_json())
    settings, _ = make_voice_run_settings(restored)
    assert settings.synthesis_config.provider == "cartesia"
    assert settings is not restored.user_voice_settings
    sampled = sample_voice_config(
        seed=300, synthesis_config=synthesis, complexity="regular"
    )
    assert sampled.speech_effects_config.enable_vocal_tics
    monkeypatch.setattr(
        "tau2.runner.build.get_or_load_task_voice_config", lambda **kwargs: sampled
    )
    monkeypatch.setattr(
        "tau2.user.user_simulator_streaming.VoiceStreamingUserSimulator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    user = build_voice_user(
        SimpleNamespace(get_user_tools=lambda **kwargs: []),
        SimpleNamespace(id="0", user_tools=None, user_scenario="Return my order"),
        AudioNativeConfig(),
        domain="mock",
        voice_settings=settings,
    )
    actual = user.voice_settings
    assert actual.speech_environment.voice_id == "hao"
    assert not actual.synthesis_config.speech_effects_config.enable_vocal_tics
    assert not actual.speech_environment.speech_effects_config.enable_vocal_tics
    assert actual.synthesis_config.speech_effects_config.enable_non_directed_phrases
    assert (
        actual.synthesis_config.source_effects_config == sampled.source_effects_config
    )
    assert (
        actual.synthesis_config.channel_effects_config == sampled.channel_effects_config
    )
    assert sampled.speech_effects_config.enable_vocal_tics  # original preset unmodified
    persona = actual.speech_environment.persona_name
    assert actual.synthesis_config.provider_config.persona_voice_ids[persona] == "hao"


def test_out_of_turn_uses_cartesia_voice(monkeypatch):
    from tau2.voice.synthesis.audio_effects.speech_generator import (
        create_streaming_audio_generators,
    )

    seen = {}
    monkeypatch.setattr(
        "tau2.voice.synthesis.audio_effects.speech_generator.create_background_noise_generator",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "tau2.voice.synthesis.audio_effects.speech_generator.OutOfTurnSpeechGenerator.generate_all",
        lambda self, items: seen.update(
            provider=self.provider, voice=self.provider_config.voice_id
        ),
    )
    config = SynthesisConfig(
        provider="cartesia",
        provider_config=CartesiaTTSConfig(
            persona_voice_ids={"matt_delaney": "cartesia-matt"}
        ),
    )
    config.speech_effects_config.enable_vocal_tics = False
    config.speech_effects_config.enable_non_directed_phrases = True
    create_streaming_audio_generators(config, "matt_delaney", 16000)
    assert seen == {"provider": "cartesia", "voice": "cartesia-matt"}
