"""xAI Grok Voice Agent API integration for audio-native voice processing.

Key advantages:
- Native G.711 μ-law support (no audio conversion for telephony!)
- OpenAI-compatible WebSocket protocol
- Built-in server VAD
- 5 voice options: Ara, Rex, Sal, Eve, Leo

Reference: https://docs.x.ai/docs/guides/voice/agent
"""

from tau2.voice.audio_native.xai.discrete_time_adapter import DiscreteTimeXAIAdapter
from tau2.voice.audio_native.xai.events import (
    BaseXAIEvent,
    XAIAudioDeltaEvent,
    XAIAudioDoneEvent,
    XAIAudioTranscriptDeltaEvent,
    XAIAudioTranscriptDoneEvent,
    XAIConversationCreatedEvent,
    XAIConversationItemAddedEvent,
    XAIErrorEvent,
    XAIFunctionCallArgumentsDeltaEvent,
    XAIFunctionCallArgumentsDoneEvent,
    XAIInputAudioBufferClearedEvent,
    XAIInputAudioBufferCommittedEvent,
    XAIInputTranscriptionCompletedEvent,
    XAIResponseContentPartAddedEvent,
    XAIResponseContentPartDoneEvent,
    XAIResponseCreatedEvent,
    XAIResponseDoneEvent,
    XAIResponseOutputItemAddedEvent,
    XAIResponseOutputItemDoneEvent,
    XAISessionUpdatedEvent,
    XAISpeechStartedEvent,
    XAISpeechStoppedEvent,
    XAITimeoutEvent,
    XAIUnknownEvent,
    parse_xai_event,
)
from tau2.voice.audio_native.xai.provider import (
    XAIAudioFormat,
    XAIRealtimeProvider,
    XAIVADConfig,
    XAIVADMode,
)

__all__ = [
    # Events
    "BaseXAIEvent",
    "XAIAudioDeltaEvent",
    "XAIAudioDoneEvent",
    "XAIAudioTranscriptDeltaEvent",
    "XAIAudioTranscriptDoneEvent",
    "XAIConversationCreatedEvent",
    "XAIConversationItemAddedEvent",
    "XAIErrorEvent",
    "XAIFunctionCallArgumentsDeltaEvent",
    "XAIFunctionCallArgumentsDoneEvent",
    "XAIInputAudioBufferClearedEvent",
    "XAIInputAudioBufferCommittedEvent",
    "XAIInputTranscriptionCompletedEvent",
    "XAIResponseContentPartAddedEvent",
    "XAIResponseContentPartDoneEvent",
    "XAIResponseCreatedEvent",
    "XAIResponseDoneEvent",
    "XAIResponseOutputItemAddedEvent",
    "XAIResponseOutputItemDoneEvent",
    "XAISessionUpdatedEvent",
    "XAISpeechStartedEvent",
    "XAISpeechStoppedEvent",
    "XAITimeoutEvent",
    "XAIUnknownEvent",
    "parse_xai_event",
    # Provider
    "XAIAudioFormat",
    "XAIRealtimeProvider",
    "XAIVADConfig",
    "XAIVADMode",
    # Adapter
    "DiscreteTimeXAIAdapter",
]
