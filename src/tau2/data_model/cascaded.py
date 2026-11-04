"""Configuration types for LiveKit cascaded voice agent.

Defines typed configurations for each component of the cascaded pipeline:
- STT: Speech-to-Text (Deepgram)
- LLM: Language Model (OpenAI, Anthropic)
- TTS: Text-to-Speech (Deepgram, ElevenLabs)

Also provides preset configurations for common use cases.
"""

from typing import Annotated, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import (
    DEFAULT_CASCADED_MULTILINGUAL_STT_MODEL,
    DEFAULT_CASCADED_MULTILINGUAL_TTS_MODEL,
    DEFAULT_CASCADED_MULTILINGUAL_VOICE_ID,
    DEFAULT_CASCADED_TTS_SAMPLE_RATE,
)
from tau2.voice.audio_native.languages import (
    CASCADED_LANGUAGE_CODES,
    ELEVENLABS_V2_5_ONLY_LANGUAGES,
    cascaded_language,
)

# =============================================================================
# STT Configurations
# =============================================================================


class DeepgramSTTConfig(BaseModel):
    """Configuration for Deepgram STT with integrated VAD.

    Deepgram's streaming API includes built-in Voice Activity Detection,
    making it ideal for real-time voice applications.

    Attributes:
        provider: Provider identifier (always "deepgram").
        model: Deepgram model to use. "nova-3" is the latest and most accurate.
        language: Language code (e.g., "en-US", "es", "fr").
        interim_results: Whether to return interim (partial) transcripts.
        vad_events: Whether to emit VAD events (speech start/end).
        endpointing_ms: Silence duration (ms) before considering speech ended.
            Lower values = faster response, higher values = fewer false endpoints.
        utterance_end_ms: Fallback turn-end detection (ms). If a FINAL_TRANSCRIPT
            arrives but no END_OF_SPEECH follows (common with background noise),
            trigger the LLM after this duration of no new transcript activity.
        smart_format: Apply formatting (numbers, dates, etc.).
        punctuate: Add punctuation to transcripts.
    """

    provider: Literal["deepgram"] = "deepgram"
    model: str = "nova-3"
    language: str = "en-US"
    interim_results: bool = True
    vad_events: bool = True
    endpointing_ms: int = 350
    utterance_end_ms: int = 2000
    smart_format: bool = False
    punctuate: bool = True


# Type alias for STT configs (extensible for future providers)
STTConfig = DeepgramSTTConfig


# =============================================================================
# LLM Configurations
# =============================================================================


class OpenAILLMConfig(BaseModel):
    """Configuration for OpenAI LLM.

    Provides full control over OpenAI model parameters, including:
    - Thinking models (o1, o3) via reasoning_effort
    - Standard models (gpt-4.1, gpt-4.1-mini) via temperature

    Attributes:
        provider: Provider identifier (always "openai").
        model: Model name (e.g., "gpt-4.1", "o3-mini", "gpt-4.1-mini").
        temperature: Sampling temperature (0.0-2.0). Not used for thinking models.
        top_p: Nucleus sampling parameter.
        reasoning_effort: For thinking models (o1, o3): "minimal", "low", "medium", "high".
            Controls how much "thinking" the model does before responding.
        max_completion_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout in seconds.
        parallel_tool_calls: Whether to allow parallel tool calls.
    """

    provider: Literal["openai"] = "openai"
    model: str = "gpt-4.1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None
    max_completion_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None


class AnthropicLLMConfig(BaseModel):
    """Configuration for Anthropic LLM.

    Supports Claude models including extended thinking capabilities.

    Attributes:
        provider: Provider identifier (always "anthropic").
        model: Model name (e.g., "claude-sonnet-4-20250514").
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        thinking_budget_tokens: For extended thinking, max thinking tokens.
            Set to enable Claude's internal reasoning before responding.
    """

    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: Optional[float] = None
    thinking_budget_tokens: Optional[int] = None


# Type alias for LLM configs
LLMConfig = Union[OpenAILLMConfig, AnthropicLLMConfig]


# =============================================================================
# TTS Configurations
# =============================================================================


class DeepgramTTSConfig(BaseModel):
    """Configuration for Deepgram TTS.

    Deepgram's Aura voices provide fast, high-quality speech synthesis.

    Attributes:
        provider: Provider identifier (always "deepgram").
        model: Voice model (e.g., "aura-asteria-en", "aura-luna-en").
        sample_rate: Output sample rate in Hz.
    """

    provider: Literal["deepgram"] = "deepgram"
    model: str = "aura-asteria-en"
    sample_rate: int = 24000


class ElevenLabsTTSConfig(BaseModel):
    """Configuration for ElevenLabs TTS.

    ElevenLabs provides highly expressive, human-like voices.

    Attributes:
        provider: Provider identifier (always "elevenlabs").
        voice_id: Actual provider voice ID (names such as "aria" are not IDs).
        model: Model to use (e.g., "eleven_turbo_v2_5", "eleven_multilingual_v2").
        stability: Voice stability (0.0-1.0). Lower = more expressive.
        similarity_boost: How closely to match the original voice (0.0-1.0).
    """

    provider: Literal["elevenlabs"] = "elevenlabs"
    voice_id: str = DEFAULT_CASCADED_MULTILINGUAL_VOICE_ID
    model: str = "eleven_turbo_v2_5"
    language: Annotated[
        Optional[str],
        Field(
            description="Explicit synthesis language hint, or None for model inference."
        ),
    ] = None
    sample_rate: Annotated[
        Literal[16000, 22050, 24000, 44100],
        Field(description="PCM synthesis rate shared by the SDK and audio converter."),
    ] = DEFAULT_CASCADED_TTS_SAMPLE_RATE
    stability: float = 0.5
    similarity_boost: float = 0.75


# Type alias for TTS configs
TTSConfig = Union[DeepgramTTSConfig, ElevenLabsTTSConfig]


# =============================================================================
# Master Cascaded Configuration
# =============================================================================


class CascadedConfig(BaseModel):
    """Configuration for the complete cascaded STT → LLM → TTS pipeline.

    Combines configurations for all three components of the voice pipeline.
    Each component can be configured independently for easy experimentation.

    Attributes:
        stt: Speech-to-Text configuration.
        llm: Language Model configuration.
        tts: Text-to-Speech configuration.
        log_prompts: If True, log the full prompt sent to LLM for debugging.
    """

    stt: STTConfig = Field(default_factory=DeepgramSTTConfig)
    llm: LLMConfig = Field(default_factory=OpenAILLMConfig)
    tts: TTSConfig = Field(default_factory=DeepgramTTSConfig)
    preamble: bool = False
    preamble_text: str = "One moment please."
    preamble_language: Annotated[
        str,
        Field(
            description="Reviewed language of preamble_text; checked when preamble is enabled."
        ),
    ] = "en"
    log_prompts: bool = False


class CascadedConfigRequest(BaseModel):
    """Caller-owned preset selection or complete explicit component override.

    Resolution copies this request and never stores generated language defaults
    back into it. The effective CascadedConfig is recorded separately on the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: str | None = Field(
        default=None, description="Named pipeline preset; None selects the default."
    )
    config: CascadedConfig | None = Field(
        default=None,
        description="Complete explicit pipeline override, validated for the call language.",
    )

    def resolve(self, language: str | None) -> CascadedConfig:
        """Resolve a fresh effective pipeline without mutating the request."""
        return resolve_cascaded_config(
            config=self.config, preset_name=self.preset, language=language
        )


# =============================================================================
# Preset Configurations
# =============================================================================

CASCADED_CONFIGS: Dict[str, CascadedConfig] = {
    # Default: Balanced speed and quality
    "default": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
    # OpenAI thinking: Uses OpenAI's thinking models with high reasoning effort
    "openai-thinking": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-5.2", reasoning_effort="high"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
        # preamble=True,
    ),
}


# The multilingual-v2 model does not accept a language override; record None.
# https://elevenlabs.io/docs/overview/models
# https://elevenlabs.io/docs/api-reference/text-to-speech/convert
_ELEVENLABS_MULTILINGUAL_MODELS = frozenset(
    {
        "eleven_flash_v2_5",
        "eleven_turbo_v2_5",
        "eleven_multilingual_v2",
    }
)
# Only the existing documented English models are enabled in this resolver.
# https://developers.deepgram.com/docs/tts-models
_DEEPGRAM_ENGLISH_TTS_MODELS = frozenset({"aura-asteria-en", "aura-luna-en"})


def resolve_cascaded_config(
    *,
    config: Optional[CascadedConfig],
    preset_name: Optional[str],
    language: Optional[str],
) -> CascadedConfig:
    """Copy and resolve the effective pipeline before any provider is created.

    A full explicit config owns every component and is only validated. Without
    one, a language run retains preset LLM/endpointing options and selects the
    same multilingual STT/TTS models and voice for every supported language.
    Ordinary runs retain their selected settings. No preset or input is mutated.
    """
    if config is not None:
        resolved = CascadedConfig.model_validate(config.model_dump())
    else:
        name = preset_name or "default"
        if name not in CASCADED_CONFIGS:
            raise ValueError(
                f"Unknown cascaded preset {name!r}; choose from {sorted(CASCADED_CONFIGS)}"
            )
        resolved = CASCADED_CONFIGS[name].model_copy(deep=True)
    if language is None:
        return resolved
    base_language, stt_language = cascaded_language(language)
    if config is None:
        resolved.stt = resolved.stt.model_copy(
            update={
                "model": DEFAULT_CASCADED_MULTILINGUAL_STT_MODEL,
                "language": stt_language,
            }
        )
        tts = (
            resolved.tts
            if isinstance(resolved.tts, ElevenLabsTTSConfig)
            else ElevenLabsTTSConfig()
        )
        resolved.tts = tts.model_copy(
            update={
                "model": DEFAULT_CASCADED_MULTILINGUAL_TTS_MODEL,
                "voice_id": DEFAULT_CASCADED_MULTILINGUAL_VOICE_ID,
                "language": base_language,
            }
        )
    if (
        resolved.stt.model != "nova-3"
        or resolved.stt.language not in CASCADED_LANGUAGE_CODES[base_language]
    ):
        raise ValueError(
            f"STT configuration does not match language {language!r}; configure "
            f"Deepgram nova-3 with language={language!r} (not 'multi')."
        )
    if isinstance(resolved.tts, DeepgramTTSConfig):
        if (
            base_language != "en"
            or resolved.tts.model not in _DEEPGRAM_ENGLISH_TTS_MODELS
        ):
            raise ValueError(
                "TTS Deepgram configuration is only verified for English Aura asteria/luna; "
                "use ElevenLabs eleven_flash_v2_5 with a matching language."
            )
    else:
        tts = resolved.tts
        if tts.model not in _ELEVENLABS_MULTILINGUAL_MODELS:
            raise ValueError(
                f"Unsupported TTS model {tts.model!r}; verified models: "
                + ", ".join(sorted(_ELEVENLABS_MULTILINGUAL_MODELS))
            )
        if tts.language is not None and tts.language != base_language:
            raise ValueError(
                f"TTS language must match {language!r}, or be None for model inference"
            )
        if (
            tts.model == "eleven_multilingual_v2"
            and base_language in ELEVENLABS_V2_5_ONLY_LANGUAGES
        ):
            raise ValueError(
                f"TTS eleven_multilingual_v2 does not support {base_language!r}"
            )
        if tts.model == "eleven_multilingual_v2" and tts.language is not None:
            raise ValueError(
                "TTS eleven_multilingual_v2 does not support a language hint; use "
                "language=None for inference, or eleven_flash_v2_5 for explicit language."
            )
    if resolved.preamble and (
        resolved.preamble_language != language or not resolved.preamble_text.strip()
    ):
        raise ValueError(
            f"Enabled preamble requires reviewed nonempty text and preamble_language={language!r}; "
            "localize it explicitly or disable preamble."
        )
    return resolved
