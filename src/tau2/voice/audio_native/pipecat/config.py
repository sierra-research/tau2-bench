"""Configuration types for Pipecat-based cascaded voice agent.

Defines typed configurations for each component of the Pipecat pipeline:
- STT: Speech-to-Text (Deepgram, OpenAI Whisper)
- LLM: Language Model (OpenAI, Anthropic)
- TTS: Text-to-Speech (Cartesia, Deepgram, ElevenLabs, OpenAI)

Also provides preset configurations for common use cases via
``PIPECAT_CONFIGS``.

The Pipecat integration uses a real Pipecat ``Pipeline`` driven by a custom
in-memory transport (``QueueTransport``), so any Pipecat service that fits
into the standard STT → LLM → TTS pipeline can be plugged in here.
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

# =============================================================================
# STT Configurations
# =============================================================================


class DeepgramSTTConfig(BaseModel):
    """Configuration for Deepgram STT (with integrated VAD).

    Attributes:
        provider: Provider identifier (always "deepgram").
        model: Deepgram model to use (e.g. "nova-3").
        language: Language code (e.g. "en-US").
        interim_results: Whether to emit interim transcripts.
        endpointing_ms: Silence (ms) before considering speech ended.
        smart_format: Apply smart formatting (numbers, dates, etc.).
        punctuate: Add punctuation to transcripts.
    """

    provider: Literal["deepgram"] = "deepgram"
    model: str = "nova-3"
    language: str = "en-US"
    interim_results: bool = True
    endpointing_ms: int = 350
    smart_format: bool = False
    punctuate: bool = True


class OpenAISTTConfig(BaseModel):
    """Configuration for OpenAI Whisper / gpt-4o-transcribe STT.

    Attributes:
        provider: Provider identifier (always "openai").
        model: OpenAI STT model to use (e.g. "whisper-1",
            "gpt-4o-transcribe", "gpt-4o-mini-transcribe").
        language: Optional language hint.
    """

    provider: Literal["openai"] = "openai"
    model: str = "gpt-4o-transcribe"
    language: Optional[str] = "en"


STTConfig = Union[DeepgramSTTConfig, OpenAISTTConfig]


# =============================================================================
# LLM Configurations
# =============================================================================


class OpenAILLMConfig(BaseModel):
    """Configuration for OpenAI LLM via Pipecat's ``OpenAILLMService``.

    Attributes:
        provider: Provider identifier (always "openai" since this drives
            ``OpenAILLMService``).
        model: Model identifier (e.g. ``gpt-4.1``).
        api_key_env: Env var holding the API key. Defaults to
            ``OPENAI_API_KEY``.
        temperature, top_p, max_completion_tokens, reasoning_effort:
            Standard OpenAI sampling/inference knobs.
    """

    provider: Literal["openai"] = "openai"
    model: str = "gpt-4.1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None
    max_completion_tokens: Optional[int] = None


class AnthropicLLMConfig(BaseModel):
    """Configuration for Anthropic LLM via Pipecat's ``AnthropicLLMService``."""

    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: Optional[float] = None


LLMConfig = Union[OpenAILLMConfig, AnthropicLLMConfig]


# =============================================================================
# TTS Configurations
# =============================================================================


class CartesiaTTSConfig(BaseModel):
    """Configuration for Cartesia TTS (Pipecat's recommended low-latency TTS)."""

    provider: Literal["cartesia"] = "cartesia"
    voice_id: str = "71a7ad14-091c-4e8e-a314-022ece01c121"  # default Cartesia voice
    model: str = "sonic-2"
    sample_rate: int = 24000


class DeepgramTTSConfig(BaseModel):
    """Configuration for Deepgram Aura TTS."""

    provider: Literal["deepgram"] = "deepgram"
    voice: str = "aura-2-asteria-en"
    sample_rate: int = 24000


class ElevenLabsTTSConfig(BaseModel):
    """Configuration for ElevenLabs TTS."""

    provider: Literal["elevenlabs"] = "elevenlabs"
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    model: str = "eleven_turbo_v2_5"
    sample_rate: int = 24000


class OpenAITTSConfig(BaseModel):
    """Configuration for OpenAI TTS."""

    provider: Literal["openai"] = "openai"
    voice: str = "alloy"
    model: str = "gpt-4o-mini-tts"
    sample_rate: int = 24000


TTSConfig = Union[
    CartesiaTTSConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    OpenAITTSConfig,
]


# =============================================================================
# Master Pipecat Configuration
# =============================================================================


class PipecatConfig(BaseModel):
    """Configuration for the complete Pipecat STT → LLM → TTS pipeline.

    Combines configurations for all three components of the voice pipeline.
    Each component can be configured independently for easy experimentation.

    Attributes:
        stt: Speech-to-Text configuration.
        llm: Language Model configuration.
        tts: Text-to-Speech configuration.
        enable_vad: Whether to use Pipecat's Silero VAD analyzer for the
            input transport. When True, ``UserStartedSpeaking`` /
            ``UserStoppedSpeaking`` system frames are emitted, enabling
            Pipecat's built-in interruption handling.
        allow_interruptions: Allow user speech to interrupt agent speech.
        log_prompts: If True, log the full LLM context for debugging.
        preamble: If True, emit a short preamble utterance before the LLM
            starts thinking (useful with high-latency reasoning models).
        preamble_text: Preamble utterance text.
    """

    stt: STTConfig = Field(default_factory=DeepgramSTTConfig)
    llm: LLMConfig = Field(default_factory=OpenAILLMConfig)
    tts: TTSConfig = Field(default_factory=CartesiaTTSConfig)
    enable_vad: bool = True
    allow_interruptions: bool = True
    log_prompts: bool = False
    preamble: bool = False
    preamble_text: str = "One moment please."


# =============================================================================
# Preset Configurations
# =============================================================================

PIPECAT_CONFIGS: Dict[str, PipecatConfig] = {
    # Default: Balanced speed and quality (Deepgram + OpenAI + Cartesia)
    "default": PipecatConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=CartesiaTTSConfig(),
    ),
    # OpenAI thinking: Uses OpenAI's thinking models with high reasoning
    "openai-thinking": PipecatConfig(
        stt=DeepgramSTTConfig(model="nova-3"),
        llm=OpenAILLMConfig(model="gpt-5.2", reasoning_effort="high"),
        tts=CartesiaTTSConfig(),
    ),
    # All-OpenAI cascade for environments with only OPENAI_API_KEY available
    "openai-only": PipecatConfig(
        stt=OpenAISTTConfig(model="gpt-4o-transcribe"),
        llm=OpenAILLMConfig(model="gpt-4.1"),
        tts=OpenAITTSConfig(voice="alloy"),
    ),
}
