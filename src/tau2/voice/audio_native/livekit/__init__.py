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

Configuration models and named presets live in ``tau2.data_model.cascaded`` so
run configs never import the optional voice runtime.

Usage:
    from tau2.data_model.cascaded import CASCADED_CONFIGS
    from tau2.voice.audio_native.livekit.discrete_time_adapter import (
        LiveKitCascadedAdapter,
    )

    adapter = LiveKitCascadedAdapter(
        tick_duration_ms=1000,
        cascaded_config=CASCADED_CONFIGS["openai-thinking"],
    )
"""


def preregister_livekit_plugins() -> None:
    """Pre-register LiveKit plugins on the main thread.

    LiveKit requires plugins to be registered on the main thread. This function
    imports the plugins before ThreadPoolExecutor workers are spawned, ensuring
    they are available in worker threads without thread registration errors.

    Must be called from the main thread before any workers are created.
    """
    from importlib import import_module

    from loguru import logger

    # A missing optional component must not prevent the installed components
    # from registering before workers start.
    for plugin in ("anthropic", "deepgram", "elevenlabs", "openai"):
        try:
            import_module(f"livekit.plugins.{plugin}")
            logger.debug(f"Pre-registered LiveKit {plugin} plugin on main thread")
        except ImportError as e:
            logger.warning(f"Could not pre-register LiveKit {plugin} plugin: {e}")


__all__ = ["preregister_livekit_plugins"]
