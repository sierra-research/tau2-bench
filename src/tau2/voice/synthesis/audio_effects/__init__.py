# Copyright Sierra
"""Audio effects processing for voice synthesis."""

from .effects import (
    apply_burst_noise,
    apply_constant_muffling,
    apply_dynamic_muffling,
    apply_frame_drops,
)
from .noise_generator import (
    BackgroundNoiseGenerator,
    apply_background_noise,
    create_background_noise_generator,
)
from .processor import (
    BatchAudioEffectsMixin,
    PendingEffectState,
    StreamingAudioEffectsMixin,
    StreamingChunkResult,
)
from .scheduler import (
    EffectScheduler,
    EffectSchedulerState,
    ScheduledEffect,
    generate_turn_effects,
)
from .speech_generator import (
    OutOfTurnSpeechGenerator,
    create_streaming_audio_generators,
)

__all__ = [
    # Effects
    "apply_burst_noise",
    "apply_constant_muffling",
    "apply_dynamic_muffling",
    "apply_frame_drops",
    # Generators
    "BackgroundNoiseGenerator",
    "OutOfTurnSpeechGenerator",
    "apply_background_noise",
    "create_background_noise_generator",
    "create_streaming_audio_generators",
    # Processor
    "BatchAudioEffectsMixin",
    "PendingEffectState",
    "StreamingAudioEffectsMixin",
    "StreamingChunkResult",
    # Scheduler
    "EffectScheduler",
    "EffectSchedulerState",
    "ScheduledEffect",
    "generate_turn_effects",
]
