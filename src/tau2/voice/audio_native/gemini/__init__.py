"""
Gemini Live API implementation for audio native adapters.
"""

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.voice.audio_native.gemini.audio_utils import (
    GEMINI_INPUT_FORMAT,
    GEMINI_OUTPUT_FORMAT,
    TELEPHONY_FORMAT,
    StreamingGeminiConverter,
    calculate_gemini_bytes_per_tick,
    calculate_telephony_bytes_per_tick,
    gemini_output_to_telephony,
    telephony_to_gemini_input,
)
from tau2.voice.audio_native.gemini.discrete_time_adapter import (
    DiscreteTimeGeminiAdapter,
)
from tau2.voice.audio_native.gemini.events import (
    BaseGeminiEvent,
    GeminiAudioDeltaEvent,
    GeminiAudioDoneEvent,
    GeminiErrorEvent,
    GeminiEvent,
    GeminiFunctionCallDoneEvent,
    GeminiInputTranscriptionEvent,
    GeminiInterruptionEvent,
    GeminiTextDeltaEvent,
    GeminiTimeoutEvent,
    GeminiTurnCompleteEvent,
    GeminiUnknownEvent,
)
from tau2.voice.audio_native.gemini.provider import (
    GEMINI_INPUT_BYTES_PER_SECOND,
    GEMINI_INPUT_SAMPLE_RATE,
    GEMINI_OUTPUT_BYTES_PER_SECOND,
    GEMINI_OUTPUT_SAMPLE_RATE,
    GeminiLiveProvider,
    GeminiVADConfig,
    GeminiVADMode,
)

__all__ = [
    # Events
    "BaseGeminiEvent",
    "GeminiEvent",
    "GeminiAudioDeltaEvent",
    "GeminiAudioDoneEvent",
    "GeminiTextDeltaEvent",
    "GeminiInputTranscriptionEvent",
    "GeminiInterruptionEvent",
    "GeminiTurnCompleteEvent",
    "GeminiFunctionCallDoneEvent",
    "GeminiErrorEvent",
    "GeminiTimeoutEvent",
    "GeminiUnknownEvent",
    # Provider
    "GeminiLiveProvider",
    "GeminiVADConfig",
    "GeminiVADMode",
    # Audio format constants
    "GEMINI_INPUT_SAMPLE_RATE",
    "GEMINI_OUTPUT_SAMPLE_RATE",
    "GEMINI_INPUT_BYTES_PER_SECOND",
    "GEMINI_OUTPUT_BYTES_PER_SECOND",
    # Audio conversion utilities
    "GEMINI_INPUT_FORMAT",
    "GEMINI_OUTPUT_FORMAT",
    "TELEPHONY_FORMAT",
    "TELEPHONY_ULAW_SILENCE",
    "StreamingGeminiConverter",
    "telephony_to_gemini_input",
    "gemini_output_to_telephony",
    "calculate_gemini_bytes_per_tick",
    "calculate_telephony_bytes_per_tick",
    # Adapter
    "DiscreteTimeGeminiAdapter",
]
