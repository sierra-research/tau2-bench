"""xAI Realtime language plumbing: audio.input.transcription.language_hint.

Covers the run-language path without a network connection:
- the language-pack-code → BCP-47 hint map (regional pins per the language
  packs' variety pins; xai rejects bare "es"/"pt");
- the provider includes input.transcription.language_hint in the session
  audio config when a hint is set and omits the block otherwise;
- the adapter maps its run language through XAI_LANGUAGE_HINTS when it
  creates the provider lazily.

OpenAI and Gemini sessions already carry a language prior (transcription
.language / SpeechConfig.language_code); without this hint, multilingual xai
runs confound ASR language detection with agent capability.
"""

import pytest

from tau2.voice.audio_native.xai.discrete_time_adapter import DiscreteTimeXAIAdapter
from tau2.voice.audio_native.xai.provider import (
    XAI_LANGUAGE_HINTS,
    XAIAudioFormat,
    XAIRealtimeProvider,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")


# =============================================================================
# The language-hint map
# =============================================================================


class TestLanguageHintMap:
    def test_es_pt_carry_regional_pins(self):
        # xai rejects bare "es"/"pt": a regional variant is required. The
        # pins follow the language packs (es → Spain, pt → Brazil).
        assert XAI_LANGUAGE_HINTS["es"] == "es-ES"
        assert XAI_LANGUAGE_HINTS["pt"] == "pt-BR"

    def test_paper_languages_are_covered(self):
        for lang in ("es", "pt", "hi", "ko", "zh"):
            assert lang in XAI_LANGUAGE_HINTS

    def test_english_is_deliberately_absent(self):
        # en runs omit the hint and auto-detect, matching the other providers.
        assert "en" not in XAI_LANGUAGE_HINTS

    def test_keys_are_pack_codes(self):
        for iso in XAI_LANGUAGE_HINTS:
            assert len(iso) == 2


# =============================================================================
# Provider: audio.input.transcription.language_hint
# =============================================================================


class TestProviderAudioConfig:
    def test_hint_lands_under_input_transcription(self):
        provider = XAIRealtimeProvider(language_hint="es-ES")
        config = provider._build_audio_config()
        assert config["input"]["transcription"] == {"language_hint": "es-ES"}
        # Output side is never touched: xai has no output-language field.
        assert "transcription" not in config["output"]

    def test_no_hint_omits_the_block(self):
        provider = XAIRealtimeProvider()
        config = provider._build_audio_config()
        assert "transcription" not in config["input"]

    @pytest.mark.parametrize(
        "audio_format", [XAIAudioFormat.PCMU, XAIAudioFormat.PCMA, XAIAudioFormat.PCM]
    )
    def test_hint_rides_along_every_audio_format(self, audio_format):
        provider = XAIRealtimeProvider(language_hint="pt-BR", audio_format=audio_format)
        config = provider._build_audio_config()
        assert config["input"]["transcription"] == {"language_hint": "pt-BR"}


# =============================================================================
# Adapter: run language → provider hint
# =============================================================================


class TestAdapterLanguagePlumbing:
    def test_explicit_model_reaches_websocket_provider(self):
        """The paper pool's model pin must reach the URL-building provider."""
        adapter = DiscreteTimeXAIAdapter(
            tick_duration_ms=1000,
            model="grok-voice-think-fast-1.0",
        )
        assert adapter.provider.model == "grok-voice-think-fast-1.0"

    def test_mapped_language_reaches_the_provider(self):
        adapter = DiscreteTimeXAIAdapter(tick_duration_ms=1000, language="es")
        assert adapter.provider.language_hint == "es-ES"

    @pytest.mark.parametrize("language", [None, "en", "qq"])
    def test_unmapped_or_absent_language_leaves_no_hint(self, language):
        adapter = DiscreteTimeXAIAdapter(tick_duration_ms=1000, language=language)
        assert adapter.provider.language_hint is None
