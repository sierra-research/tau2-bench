"""Offline provider payload tests for the resolved run language."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.cascaded import CascadedConfigRequest
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.gemini.discrete_time_adapter import (
    DiscreteTimeGeminiAdapter,
)
from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider, GeminiVADConfig
from tau2.voice.audio_native.xai.provider import XAIAudioFormat, XAIRealtimeProvider


@pytest.mark.parametrize(
    "language,hint",
    [
        ("en", None),
        ("es", "es-ES"),
        ("pt", "pt-BR"),
        ("hi", "hi"),
        ("zh", "zh"),
        ("ko", "ko"),
        ("fr", "fr"),
        ("fr-FR", "fr"),
        (None, None),
    ],
)
def test_xai_resolved_language_reaches_audio_payload(language, hint, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "offline-fixture")
    agent = DiscreteTimeAudioNativeAgent(
        tools=[], domain_policy="Fixture", provider="xai", language=language
    )
    provider = agent.adapter.provider
    audio = provider._build_audio_config()
    assert not provider.is_connected
    assert audio["input"].get("transcription") == (
        {"language_hint": hint} if hint else None
    )
    assert audio["input"]["format"] == {"type": "audio/pcmu"}


@pytest.mark.parametrize("audio_format", list(XAIAudioFormat))
def test_xai_hint_preserves_each_audio_format(audio_format):
    provider = XAIRealtimeProvider(
        api_key="offline-fixture", audio_format=audio_format, language_hint="pt-BR"
    )
    config = provider._build_audio_config()
    assert config["input"]["transcription"] == {"language_hint": "pt-BR"}
    assert config["output"]["format"]["type"] == audio_format.value


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-live-2.5-flash", "es-ES"),
        ("models/gemini-live-2.5-flash", "es-ES"),
        ("gemini-2.5-flash-native-audio-preview-12-2025", None),
        ("gemini-3.1-flash-live-preview", None),
        ("future-unknown-model", None),
    ],
)
def test_gemini_language_payload_and_reconnect_use_supported_model(model, expected):
    provider = GeminiLiveProvider(api_key="offline-fixture", model=model)
    configs = []

    def connect(**kwargs):
        configs.append(kwargs["config"])
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace()
        return context

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
    )
    adapter = DiscreteTimeGeminiAdapter(
        tick_duration_ms=100, provider=provider, language="es"
    )
    asyncio.run(
        adapter._async_connect(
            "Reviewed Spanish prompt", [], GeminiVADConfig(), "audio"
        )
    )
    assert configs[-1].input_audio_transcription.language_codes == ["es-ES"]
    assert configs[-1].speech_config.language_code == expected
    assert configs[-1].system_instruction == "Reviewed Spanish prompt"
    assert provider._connect_config["language"] == "es"
    provider._resumption_handle = "fixture-resume"
    assert asyncio.run(provider._reconnect_with_resumption())
    assert configs[-1].input_audio_transcription.language_codes == ["es-ES"]
    assert configs[-1].speech_config.language_code == expected
    assert configs[-1].session_resumption.handle == "fixture-resume"
    provider._session = None


def test_factory_forwards_gemini_language_and_preserves_openai_hint(monkeypatch):
    adapter, _ = create_adapter("gemini", tick_duration_ms=100, language="ko")
    assert adapter.language == "ko"
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    adapter, _ = create_adapter(
        "openai",
        tick_duration_ms=100,
        language="es",
        transcription_language="es",
        voice="marin",
    )
    assert adapter.provider.transcription_language == "es"
    assert adapter.provider.voice == "marin"


@pytest.mark.parametrize(
    "language,expected",
    [
        ("en", None),
        ("es", "es-ES"),
        ("pt", "pt-BR"),
        ("hi", "hi-IN"),
        ("zh", "cmn-CN"),
        ("ko", "ko-KR"),
        ("fr", "fr-FR"),
        ("fr-FR", "fr-FR"),
        (None, None),
    ],
)
def test_gemini_language_codes_match_published_varieties(language, expected):
    from tau2.voice.audio_native.gemini.provider import gemini_speech_language_code

    assert gemini_speech_language_code("gemini-live-2.5-flash", language) == expected


@pytest.mark.parametrize("provider", ["gemini", "xai"])
def test_agent_factory_preserves_recorded_language(provider):
    from tau2.agent.discrete_time_audio_native_agent import (
        create_discrete_time_audio_native_agent,
    )
    from tau2.data_model.simulation import AudioNativeConfig

    config = AudioNativeConfig(provider=provider, language="pt")
    agent = create_discrete_time_audio_native_agent(
        [], "Fixture", audio_native_config=config
    )
    assert agent.language == config.language == "pt"
    assert agent.adapter.language == config.language


def test_direct_livekit_agent_factory_resolves_selected_preset():
    from tau2.agent.discrete_time_audio_native_agent import (
        create_discrete_time_audio_native_agent,
    )
    from tau2.data_model.cascaded import CASCADED_CONFIGS
    from tau2.data_model.simulation import AudioNativeConfig

    config = AudioNativeConfig(
        provider="livekit",
        cascaded_request=CascadedConfigRequest(preset="openai-thinking"),
    )
    assert config.cascaded_config is None
    agent = create_discrete_time_audio_native_agent(
        [], "Fixture", audio_native_config=config
    )
    assert agent.cascaded_config == CASCADED_CONFIGS["openai-thinking"]
    assert agent.cascaded_config is not CASCADED_CONFIGS["openai-thinking"]


@pytest.mark.parametrize("provider", ["gemini", "xai"])
def test_run_language_reaches_adapter_through_real_runner(
    tmp_path, monkeypatch, provider
):
    import json
    from pathlib import Path

    from tau2.data_model.language import LanguagePack
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.data_model.tasks import Task
    from tau2.multilingual import runtime
    from tau2.runner.build import build_voice_orchestrator

    fixture = json.loads(
        (
            Path(__file__).parents[2]
            / "test_multilingual/fixtures/reviewed_prompt_content.json"
        ).read_text()
    )["es"]
    task = Task(id="0_es", user_scenario={"instructions": "Offline scenario."})
    root = tmp_path / "tau2/multilingual/es"
    root.mkdir(parents=True)
    (root / "tasks.json").write_text(json.dumps([task.model_dump(mode="json")]))
    (root / "metadata.json").write_text(
        json.dumps({task.id: {"canonical_task_id": "0", "caller_gender": "male"}})
    )
    content = fixture["prompt_content"]
    pack = LanguagePack(
        schema_version=3,
        language="es",
        revision="offline",
        display_name="Spanish",
        agent_greeting="Hola.",
        agent_language_clause="Speak Spanish.",
        voice={
            "spellout_section": fixture["spellout_section"],
            "speech_insert_events_per_minute": 1,
            "backchannel_level": "medium",
        },
        personas={
            "speaker": {
                "gender": "male",
                "locale": "Spain",
                "variety": "Spanish",
                "gender_guidance": content["gender_guidance"],
                "language_directive": {"origin": "Spain"},
                "voice": {
                    "voice_id": "fixture",
                    "backchannel_phrases": ["sí"],
                    "voice_examples": content["voice_examples"],
                },
            }
        },
        domains={
            "retail": {
                "revision": "offline",
                "tasks_file": "tasks.json",
                "task_metadata_file": "metadata.json",
            }
        },
    )
    (root / "pack.yaml").write_text(pack.model_dump_json())
    monkeypatch.setattr(runtime, "language_root", lambda: root.parent)
    captured = []

    def fake_adapter(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(), "offline-model"

    monkeypatch.setattr(
        "tau2.agent.discrete_time_audio_native_agent.create_adapter", fake_adapter
    )
    config = VoiceRunConfig(
        domain="retail",
        language="es",
        audio_native_config=AudioNativeConfig(provider=provider),
    )
    built = build_voice_orchestrator(config, task)
    assert config.audio_native_config.language == "es"
    assert built.call_context.language.language == "es"
    assert "Speak Spanish." in built.agent.system_prompt
    assert built.agent._adapter is None
    built.agent.adapter
    assert captured[0]["language"] == "es"


@pytest.mark.parametrize("resolver", ["xai", "gemini"])
def test_unknown_language_hints_fail_explicitly(resolver):
    from tau2.voice.audio_native.languages import (
        gemini_language_code,
        xai_language_hint,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        (xai_language_hint if resolver == "xai" else gemini_language_code)("unknown")


@pytest.mark.parametrize(
    "language,codes,expected",
    [
        ("fr", None, ["fr-FR"]),
        (None, None, ["en-US"]),
        ("fr", [], None),
        ("fr", ["fr-CA"], ["fr-CA"]),
    ],
)
def test_gemini_transcription_defaults_and_explicit_overrides(
    language, codes, expected
):
    provider = GeminiLiveProvider(
        api_key="offline-fixture", transcription_language_codes=codes
    )
    configs = []

    def connect(**kwargs):
        configs.append(kwargs["config"])
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace()
        return context

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
    )
    asyncio.run(provider.connect(system_prompt="Fixture", tools=[], language=language))
    assert configs[-1].input_audio_transcription.language_codes == expected
