"""Boson realtime voice chat provider for audio-native simulations.

Environment Variables:
    BOSON_API_KEY: Boson API key.
    BOSON_REALTIME_URL: Optional WebSocket endpoint override.

The default endpoint is the staging realtime voice chat URL configured in
tau2.config.
"""

from tau2.voice.audio_native.boson.discrete_time_adapter import (
    DiscreteTimeBosonAdapter,
)
from tau2.voice.audio_native.boson.events import (
    BaseBosonEvent,
    BosonAudioDeltaEvent,
    BosonAudioDoneEvent,
    BosonAudioTranscriptDeltaEvent,
    BosonAudioTranscriptDoneEvent,
    BosonAudioTranscriptLengthEvent,
    BosonErrorEvent,
    BosonFunctionCallArgumentsDoneEvent,
    BosonInputAudioTranscriptionCompletedEvent,
    BosonResponseDoneEvent,
    BosonSpeechStartedEvent,
    BosonSpeechStoppedEvent,
    BosonTimeoutEvent,
    BosonUnknownEvent,
    parse_boson_event,
)
from tau2.voice.audio_native.boson.provider import (
    BosonRealtimeProvider,
    BosonVADConfig,
    BosonVADMode,
    audio_format_to_boson,
)

__all__ = [
    "BosonRealtimeProvider",
    "BosonVADConfig",
    "BosonVADMode",
    "DiscreteTimeBosonAdapter",
    "audio_format_to_boson",
    "BaseBosonEvent",
    "BosonAudioDeltaEvent",
    "BosonAudioDoneEvent",
    "BosonAudioTranscriptDeltaEvent",
    "BosonAudioTranscriptDoneEvent",
    "BosonAudioTranscriptLengthEvent",
    "BosonErrorEvent",
    "BosonFunctionCallArgumentsDoneEvent",
    "BosonInputAudioTranscriptionCompletedEvent",
    "BosonResponseDoneEvent",
    "BosonSpeechStartedEvent",
    "BosonSpeechStoppedEvent",
    "BosonTimeoutEvent",
    "BosonUnknownEvent",
    "parse_boson_event",
]
