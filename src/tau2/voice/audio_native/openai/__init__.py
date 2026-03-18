"""
OpenAI Realtime API implementation for audio native adapters.
"""

from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeAudioNativeAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    ErrorEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    InputAudioTranscriptionCompletedEvent,
    OutputItemAddedEvent,
    RealtimeEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TextDeltaEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
    OpenAIVADMode,
)
from tau2.voice.audio_native.openai.tick_runner import (
    PendingUtterance,
    TickRunner,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

__all__ = [
    # Adapters
    "DiscreteTimeAudioNativeAdapter",
    # Config (in provider.py)
    "OpenAIVADConfig",
    "OpenAIVADMode",
    # Events
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
    # Provider
    "OpenAIRealtimeProvider",
    # Tick-based simulation
    "TickResult",
    "TickRunner",
    "UtteranceTranscript",
    "PendingUtterance",
]
