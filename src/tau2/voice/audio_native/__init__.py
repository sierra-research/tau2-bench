"""
Native voice providers for end-to-end voice/text processing.

Contains adapter and provider implementations for native voice APIs like OpenAI Realtime.
"""

from tau2.voice.audio_native.adapter import DiscreteTimeAdapter, create_adapter
from tau2.voice.audio_native.openai import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    DiscreteTimeAudioNativeAdapter,
    ErrorEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    InputAudioTranscriptionCompletedEvent,
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
    OpenAIVADMode,
    OutputItemAddedEvent,
    PendingUtterance,
    RealtimeEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TextDeltaEvent,
    TickRunner,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

__all__ = [
    # Abstract adapters and factory (discrete-time pattern only)
    "DiscreteTimeAdapter",
    "create_adapter",
    # OpenAI adapters
    "DiscreteTimeAudioNativeAdapter",
    # Config (OpenAI-specific)
    "OpenAIVADConfig",
    "OpenAIVADMode",
    # Events (OpenAI)
    "BaseRealtimeEvent",
    "TextDeltaEvent",
    "AudioDeltaEvent",
    "AudioDoneEvent",
    "AudioTranscriptDeltaEvent",
    "AudioTranscriptDoneEvent",
    "FunctionCallArgumentsDeltaEvent",
    "FunctionCallArgumentsDoneEvent",
    "InputAudioTranscriptionCompletedEvent",
    "OutputItemAddedEvent",
    "ResponseDoneEvent",
    "SpeechStartedEvent",
    "SpeechStoppedEvent",
    "ErrorEvent",
    "TimeoutEvent",
    "UnknownEvent",
    "RealtimeEvent",
    "parse_realtime_event",
    # Provider (OpenAI)
    "OpenAIRealtimeProvider",
    # Tick-based simulation (OpenAI)
    "TickResult",
    "TickRunner",
    "UtteranceTranscript",
    "PendingUtterance",
]
