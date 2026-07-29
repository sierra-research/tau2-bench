"""Pipecat-based cascaded voice pipeline (STT → LLM → TTS).

This module provides a cascaded voice provider built on top of the
`Pipecat <https://docs.pipecat.ai>`_ framework. Pipecat is a pipeline
architecture that orchestrates STT, LLM, and TTS services with built-in
support for VAD, interruption, context management, and streaming.

Architecture:
- ``provider.py``: Builds and runs a Pipecat ``Pipeline`` and exposes
  a small async API (``connect``, ``process_audio``, ``send_tool_result``,
  ``disconnect``) consumed by the discrete-time adapter.
- ``queue_transport.py``: An in-memory ``BaseTransport`` that bridges
  the tick-based simulation framework to Pipecat's frame-based pipeline.
- ``discrete_time_adapter.py``: ``DiscreteTimePipecatAdapter`` — the
  thin wrapper that exposes a tick interface to the rest of tau2-bench.
- ``config.py``: Configuration models for STT/LLM/TTS components and a
  registry of named presets (``PIPECAT_CONFIGS``).

Usage::

    from tau2.voice.audio_native.pipecat import (
        PIPECAT_CONFIGS,
        DiscreteTimePipecatAdapter,
    )

    adapter = DiscreteTimePipecatAdapter(
        tick_duration_ms=200,
        pipecat_config=PIPECAT_CONFIGS["default"],
    )
    adapter.connect(system_prompt, tools, vad_config=None, modality="audio")
"""

from tau2.voice.audio_native.pipecat.config import (
    PIPECAT_CONFIGS,
    AnthropicLLMConfig,
    CartesiaTTSConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    LLMConfig,
    OpenAILLMConfig,
    OpenAISTTConfig,
    OpenAITTSConfig,
    PipecatConfig,
    STTConfig,
    TTSConfig,
)
from tau2.voice.audio_native.pipecat.discrete_time_adapter import (
    DiscreteTimePipecatAdapter,
    PipecatVADConfig,
)
from tau2.voice.audio_native.pipecat.events import PipecatEvent, PipecatEventType
from tau2.voice.audio_native.pipecat.provider import (
    PipecatVoiceProvider,
    ProviderState,
)
from tau2.voice.audio_native.pipecat.queue_transport import (
    QueueInputTransport,
    QueueOutputTransport,
    QueueTransport,
)


def preregister_pipecat_plugins() -> None:
    """Pre-register Pipecat plugins on the main thread.

    Pipecat does not require thread-affinity for its plugins the way
    LiveKit does, so this is currently a no-op. It is provided as a
    parallel to ``preregister_livekit_plugins`` in case Pipecat ever
    requires similar handling in the future.
    """
    return None


__all__ = [
    # Adapter
    "DiscreteTimePipecatAdapter",
    "PipecatVADConfig",
    # Provider
    "PipecatVoiceProvider",
    "PipecatEvent",
    "PipecatEventType",
    "ProviderState",
    # Transport
    "QueueTransport",
    "QueueInputTransport",
    "QueueOutputTransport",
    # Utilities
    "preregister_pipecat_plugins",
    # Configuration
    "PipecatConfig",
    "STTConfig",
    "LLMConfig",
    "TTSConfig",
    "DeepgramSTTConfig",
    "OpenAISTTConfig",
    "OpenAILLMConfig",
    "AnthropicLLMConfig",
    "CartesiaTTSConfig",
    "DeepgramTTSConfig",
    "ElevenLabsTTSConfig",
    "OpenAITTSConfig",
    "PIPECAT_CONFIGS",
]
