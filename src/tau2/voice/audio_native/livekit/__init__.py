"""LiveKit-based cascaded voice pipeline (STT → LLM → TTS).

This module provides a cascaded voice provider that orchestrates:
- STT: Deepgram streaming transcription with integrated VAD
- LLM: OpenAI/Anthropic with full parameter control (reasoning_effort, etc.)
- TTS: Deepgram/ElevenLabs streaming synthesis

Architecture:
- provider.py: Core pipeline logic (CascadedVoiceProvider)
    - Turn-taking decisions
    - Interruption handling
    - Context management
    - Streaming orchestration

- discrete_time_adapter.py: Thin wrapper (LiveKitCascadedAdapter)
    - Tick-based timing interface
    - Audio buffering for tick alignment
    - Event mapping to TickResult

- config.py: Configuration models
    - STT configs (DeepgramSTTConfig)
    - LLM configs (OpenAILLMConfig, AnthropicLLMConfig)
    - TTS configs (DeepgramTTSConfig, ElevenLabsTTSConfig)
    - Named presets (CASCADED_CONFIGS)

Usage:
    from tau2.voice.audio_native.livekit import (
        CASCADED_CONFIGS,
        LiveKitCascadedAdapter,
    )

    adapter = LiveKitCascadedAdapter(
        tick_duration_ms=1000,
        cascaded_config=CASCADED_CONFIGS["openai-thinking"],
    )
"""

from tau2.voice.audio_native.livekit.config import (
    CASCADED_CONFIGS,
    AnthropicLLMConfig,
    CascadedConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    LLMConfig,
    OpenAILLMConfig,
    STTConfig,
    TTSConfig,
)
from tau2.voice.audio_native.livekit.discrete_time_adapter import (
    LiveKitCascadedAdapter,
    LiveKitVADConfig,
)
from tau2.voice.audio_native.livekit.provider import (
    CascadedEvent,
    CascadedEventType,
    CascadedVoiceProvider,
    ConversationContext,
    ProviderState,
    TurnTakingConfig,
)


def preregister_livekit_plugins() -> None:
    """Pre-register LiveKit plugins on the main thread.

    LiveKit requires plugins to be registered on the main thread. This function
    imports the plugins before ThreadPoolExecutor workers are spawned, ensuring
    they are available in worker threads without thread registration errors.

    Must be called from the main thread before any workers are created.
    """
    from importlib import import_module

    from loguru import logger

    for plugin in ("anthropic", "cartesia", "deepgram", "elevenlabs", "openai"):
        try:
            import_module(f"livekit.plugins.{plugin}")
        except ImportError:
            logger.debug(f"Optional LiveKit plugin {plugin} is not installed")


__all__ = [
    # Main adapter for discrete-time simulation
    "LiveKitCascadedAdapter",
    "LiveKitVADConfig",
    # Core provider
    "CascadedVoiceProvider",
    "CascadedEvent",
    "CascadedEventType",
    "ProviderState",
    "ConversationContext",
    "TurnTakingConfig",
    # Utilities
    "preregister_livekit_plugins",
    # Configuration
    "CascadedConfig",
    "STTConfig",
    "LLMConfig",
    "TTSConfig",
    "DeepgramSTTConfig",
    "OpenAILLMConfig",
    "AnthropicLLMConfig",
    "DeepgramTTSConfig",
    "ElevenLabsTTSConfig",
    "CASCADED_CONFIGS",
]
