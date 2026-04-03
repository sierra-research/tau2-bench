"""Configuration types for LiveKit cascaded voice agent.

Defines typed configurations for each component of the cascaded pipeline:
- STT: Speech-to-Text (Deepgram)
- LLM: Language Model (OpenAI, Anthropic)
- TTS: Text-to-Speech (Deepgram, ElevenLabs)

Also provides preset configurations for common use cases.
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

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
        reasoning_effort: For thinking models (o1, o3): "low", "medium", "high".
            Controls how much "thinking" the model does before responding.
        max_completion_tokens: Maximum tokens in the response.
        timeout_seconds: Request timeout in seconds.
        parallel_tool_calls: Whether to allow parallel tool calls.
    """

    provider: Literal["openai"] = "openai"
    model: str = "gpt-4.1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
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
        voice_id: Voice ID or name (e.g., "aria", "roger").
        model: Model to use (e.g., "eleven_turbo_v2_5", "eleven_multilingual_v2").
        stability: Voice stability (0.0-1.0). Lower = more expressive.
        similarity_boost: How closely to match the original voice (0.0-1.0).
    """

    provider: Literal["elevenlabs"] = "elevenlabs"
    voice_id: str = "aria"
    model: str = "eleven_turbo_v2_5"
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
    log_prompts: bool = False


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
    # OpenAI thinking: Uses OpenAI's thinking models for complex reasoning
    "openai-thinking": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-5.2", reasoning_effort="medium"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
    # OpenAI thinking (high): Maximum reasoning effort
    "openai-thinking-high": CascadedConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-5.2", reasoning_effort="high"),
        tts=DeepgramTTSConfig(model="aura-asteria-en"),
    ),
}
