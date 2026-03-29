"""Pydantic models for Gemini Live API events.

Gemini Live API uses a different event structure than OpenAI Realtime.
Key differences:
- Audio comes as raw bytes in response.data
- Text comes in response.text
- Interruptions are signaled via server_content.interrupted = True
- Tool calls come via response.tool_call
- Input transcription via separate events

Reference: https://ai.google.dev/api/live
"""

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseGeminiEvent(BaseModel):
    """Base class for all Gemini Live API events."""

    type: str
    event_id: Optional[str] = None


# =============================================================================
# Audio Events
# =============================================================================


class GeminiAudioDeltaEvent(BaseGeminiEvent):
    """Audio chunk received from Gemini.

    In Gemini Live, audio comes as raw bytes in response.data.
    Format: 24kHz PCM16 mono.
    """

    type: Literal["audio.delta"] = "audio.delta"
    data: bytes = Field(
        default=b"",
        description="Raw audio bytes (24kHz PCM16)",
        exclude=True,  # Exclude from serialization due to large size and non-UTF-8
    )
    item_id: Optional[str] = None


class GeminiAudioDoneEvent(BaseGeminiEvent):
    """Audio stream completed for current response."""

    type: Literal["audio.done"] = "audio.done"
    item_id: Optional[str] = None


# =============================================================================
# Text/Transcript Events
# =============================================================================


class GeminiTextDeltaEvent(BaseGeminiEvent):
    """Text chunk received from Gemini (model's response transcript)."""

    type: Literal["text.delta"] = "text.delta"
    text: str = ""
    item_id: Optional[str] = None


class GeminiInputTranscriptionEvent(BaseGeminiEvent):
    """Transcription of user's audio input.

    Gemini Live provides input_audio_transcription events for user speech.
    Note: May degrade in long sessions - periodic flush recommended.
    """

    type: Literal["input_audio_transcription"] = "input_audio_transcription"
    transcript: str = ""
    item_id: Optional[str] = None


# =============================================================================
# Turn/Interruption Events
# =============================================================================


class GeminiInterruptionEvent(BaseGeminiEvent):
    """User interrupted the model's response.

    In Gemini Live, this comes via server_content with interrupted=True.
    Client should immediately clear audio buffer and stop playback.
    """

    type: Literal["interruption"] = "interruption"
    # No audio_start_ms like OpenAI - Gemini doesn't provide this


class GeminiTurnCompleteEvent(BaseGeminiEvent):
    """Model finished its turn (done speaking)."""

    type: Literal["turn.complete"] = "turn.complete"


# =============================================================================
# Tool/Function Call Events
# =============================================================================


class GeminiFunctionCallDoneEvent(BaseGeminiEvent):
    """Function call arguments are complete."""

    type: Literal["function_call.done"] = "function_call.done"
    call_id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Session/Connection Events
# =============================================================================


class GeminiErrorEvent(BaseGeminiEvent):
    """Error from the Gemini API."""

    type: Literal["error"] = "error"
    code: Optional[str] = None
    message: Optional[str] = None


class GeminiTimeoutEvent(BaseGeminiEvent):
    """Timeout waiting for events (used internally)."""

    type: Literal["timeout"] = "timeout"


class GeminiUnknownEvent(BaseGeminiEvent):
    """Unknown/unrecognized event type."""

    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


class GeminiGoAwayEvent(BaseGeminiEvent):
    """Server is about to close the connection.

    The server sends this message to indicate the current connection will
    soon be terminated. The time_left field indicates remaining time before
    disconnection, allowing the client to prepare for reconnection.
    """

    type: Literal["go_away"] = "go_away"
    time_left_seconds: Optional[float] = None


class GeminiSessionResumptionEvent(BaseGeminiEvent):
    """Session resumption update from the server.

    The server periodically sends these updates with a handle that can be
    used to resume the session after a connection drop. The handle should
    be stored and passed to the next connection.
    """

    type: Literal["session_resumption"] = "session_resumption"
    new_handle: Optional[str] = None
    resumable: bool = False


# =============================================================================
# Type Aliases
# =============================================================================

GeminiEvent = Union[
    GeminiAudioDeltaEvent,
    GeminiAudioDoneEvent,
    GeminiTextDeltaEvent,
    GeminiInputTranscriptionEvent,
    GeminiInterruptionEvent,
    GeminiTurnCompleteEvent,
    GeminiFunctionCallDoneEvent,
    GeminiErrorEvent,
    GeminiTimeoutEvent,
    GeminiUnknownEvent,
    GeminiGoAwayEvent,
    GeminiSessionResumptionEvent,
]
