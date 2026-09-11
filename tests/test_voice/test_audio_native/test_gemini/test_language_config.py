"""Gemini Live language plumbing: SpeechConfig.language_code.

Covers the run-language path end to end without a network connection:
- the ISO 639-1 → Gemini Live BCP-47 map (regional pins per the language
  packs: es→es-ES, pt→pt-BR, zh→cmn-CN; unmapped codes omit the field);
- the adapter stores the language and forwards it through _async_connect;
- provider.connect() builds SpeechConfig with language_code when the language
  maps, omits the field otherwise, and stores the language in _connect_config
  so session-resumption reconnects preserve it.

Uses a fake genai client capturing LiveConnectConfig -- no real API calls.
"""

import asyncio
from types import SimpleNamespace

import pytest

from tau2.voice.audio_native.gemini.discrete_time_adapter import (
    DiscreteTimeGeminiAdapter,
)
from tau2.voice.audio_native.gemini.provider import (
    GEMINI_LIVE_LANGUAGE_CODES,
    GeminiLiveProvider,
)

# =============================================================================
# Helpers
# =============================================================================


class _FakeLiveConnection:
    """Async context manager standing in for client.aio.live.connect(...)."""

    def __init__(self, recorder: dict):
        self._recorder = recorder

    async def __aenter__(self):
        return SimpleNamespace()  # the "session"

    async def __aexit__(self, *exc_info):
        return False


def _make_provider(monkeypatch, recorder: dict) -> GeminiLiveProvider:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_KEY", raising=False)
    provider = GeminiLiveProvider(max_resumptions=0)

    def fake_connect(model, config):
        recorder["model"] = model
        recorder["config"] = config
        return _FakeLiveConnection(recorder)

    provider._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=fake_connect))
    )
    return provider


def _connect(provider: GeminiLiveProvider, **kwargs) -> None:
    asyncio.run(provider.connect(system_prompt="test prompt", tools=[], **kwargs))


# =============================================================================
# The language-code map
# =============================================================================


class TestLanguageCodeMap:
    def test_regional_pins_match_the_language_packs(self):
        assert GEMINI_LIVE_LANGUAGE_CODES["es"] == "es-ES"  # Spain pin
        assert GEMINI_LIVE_LANGUAGE_CODES["pt"] == "pt-BR"
        assert GEMINI_LIVE_LANGUAGE_CODES["zh"] == "cmn-CN"
        assert GEMINI_LIVE_LANGUAGE_CODES["hi"] == "hi-IN"
        assert GEMINI_LIVE_LANGUAGE_CODES["ko"] == "ko-KR"

    def test_map_covers_exactly_the_non_english_pack_languages(self):
        # A language outside the map (including en itself) omits the field
        # and relies on auto-detection.
        assert set(GEMINI_LIVE_LANGUAGE_CODES) == {"es", "hi", "ko", "pt", "zh"}

    def test_values_are_bcp47_shaped(self):
        for iso, code in GEMINI_LIVE_LANGUAGE_CODES.items():
            assert len(iso) == 2
            assert "-" in code, code


# =============================================================================
# Provider: SpeechConfig.language_code
# =============================================================================


class TestProviderSpeechLanguage:
    def test_mapped_language_pins_language_code(self, monkeypatch):
        recorder: dict = {}
        provider = _make_provider(monkeypatch, recorder)
        _connect(provider, language="es")
        speech_config = recorder["config"].speech_config
        assert speech_config.language_code == "es-ES"
        # The voice pin is untouched.
        assert (
            speech_config.voice_config.prebuilt_voice_config.voice_name
            == provider.DEFAULT_VOICE
        )

    @pytest.mark.parametrize("language", [None, "sw", "qq"])
    def test_unmapped_or_absent_language_omits_the_field(self, monkeypatch, language):
        # None/unmapped → omit: Live auto-detects, and native-audio models
        # auto-detect regardless of this field.
        recorder: dict = {}
        provider = _make_provider(monkeypatch, recorder)
        _connect(provider, language=language)
        assert recorder["config"].speech_config.language_code is None

    def test_language_survives_in_connect_config_for_resumption(self, monkeypatch):
        # _connect_config is replayed by _reconnect_with_resumption; language
        # must ride it so a GoAway reconnect keeps the pinned code.
        recorder: dict = {}
        provider = _make_provider(monkeypatch, recorder)
        _connect(provider, language="ko")
        assert provider._connect_config["language"] == "ko"

    def test_text_modality_has_no_speech_config(self, monkeypatch):
        recorder: dict = {}
        provider = _make_provider(monkeypatch, recorder)
        _connect(provider, language="es", modality="text")
        assert recorder["config"].speech_config is None


# =============================================================================
# Adapter: language pass-through
# =============================================================================


class TestAdapterLanguagePassThrough:
    def test_adapter_forwards_language_to_provider_connect(self):
        captured: dict = {}

        class _RecordingProvider:
            async def connect(self, **kwargs):
                captured.update(kwargs)

        adapter = DiscreteTimeGeminiAdapter(
            tick_duration_ms=1000,
            language="pt",
            provider=_RecordingProvider(),
        )
        assert adapter.language == "pt"
        asyncio.run(
            adapter._async_connect(
                system_prompt="p", tools=[], vad_config=None, modality="audio"
            )
        )
        assert captured["language"] == "pt"

    def test_adapter_defaults_to_no_language(self):
        captured: dict = {}

        class _RecordingProvider:
            async def connect(self, **kwargs):
                captured.update(kwargs)

        adapter = DiscreteTimeGeminiAdapter(
            tick_duration_ms=1000, provider=_RecordingProvider()
        )
        assert adapter.language is None
        asyncio.run(
            adapter._async_connect(
                system_prompt="p", tools=[], vad_config=None, modality="audio"
            )
        )
        assert captured["language"] is None
