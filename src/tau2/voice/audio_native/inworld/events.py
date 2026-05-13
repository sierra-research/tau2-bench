"""Pydantic models for Inworld Realtime API events.

Inworld's Realtime API uses a WebSocket protocol that's OpenAI-Realtime compatible.
Confirmed event names (via live probe against api.inworld.ai):
- Audio output: response.output_audio.delta / response.output_audio.done
- Audio transcript: response.output_audio_transcript.delta / .done
- Input transcription: conversation.item.input_audio_transcription.completed
- Function calls: response.function_call_arguments.delta / .done
- VAD: input_audio_buffer.speech_started / .speech_stopped
- Turn completion: response.done

Reference: https://docs.inworld.ai/api-reference/realtimeAPI/realtime/realtime-websocket
"""

from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class BaseInworldEvent(BaseModel):
    """Base class for all Inworld Realtime API events."""

    type: str
    event_id: Optional[str] = None


# =============================================================================
# Session Events
# =============================================================================


class InworldSessionCreatedEvent(BaseInworldEvent):
    """First message at connection — session created."""

    type: Literal["session.created"] = "session.created"
    session: Optional[Dict[str, Any]] = None


class InworldSessionUpdatedEvent(BaseInworldEvent):
    """Session configuration has been updated."""

    type: Literal["session.updated"] = "session.updated"
    session: Optional[Dict[str, Any]] = None


# =============================================================================
# Input Audio Buffer Events (VAD)
# =============================================================================


class InworldSpeechStartedEvent(BaseInworldEvent):
    """Server VAD detected start of speech (used for barge-in)."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    item_id: Optional[str] = None
    audio_start_ms: Optional[int] = None


class InworldSpeechStoppedEvent(BaseInworldEvent):
    """Server VAD detected end of speech."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    item_id: Optional[str] = None
    audio_end_ms: Optional[int] = None


class InworldInputAudioBufferCommittedEvent(BaseInworldEvent):
    """Input audio buffer has been committed."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    previous_item_id: Optional[str] = None
    item_id: Optional[str] = None


class InworldInputAudioBufferClearedEvent(BaseInworldEvent):
    """Input audio buffer has been cleared."""

    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


# =============================================================================
# Conversation Item Events
# =============================================================================


class InworldConversationItemAddedEvent(BaseInworldEvent):
    """A new item has been added to conversation history."""

    type: Literal["conversation.item.added"] = "conversation.item.added"
    previous_item_id: Optional[str] = None
    item: Optional[Dict[str, Any]] = None


class InworldConversationItemDoneEvent(BaseInworldEvent):
    """A conversation item has completed."""

    type: Literal["conversation.item.done"] = "conversation.item.done"
    previous_item_id: Optional[str] = None
    item: Optional[Dict[str, Any]] = None


class InworldInputTranscriptionCompletedEvent(BaseInworldEvent):
    """Transcription of user's audio input is complete."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: Optional[str] = None
    transcript: str = ""


# =============================================================================
# Response Events
# =============================================================================


class InworldResponseCreatedEvent(BaseInworldEvent):
    """A new assistant response turn is in progress."""

    type: Literal["response.created"] = "response.created"
    response: Optional[Dict[str, Any]] = None


class InworldResponseOutputItemAddedEvent(BaseInworldEvent):
    """A new assistant response item is added to message history."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[Dict[str, Any]] = None


class InworldResponseOutputItemDoneEvent(BaseInworldEvent):
    """Response output item is complete."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[Dict[str, Any]] = None


class InworldResponseContentPartAddedEvent(BaseInworldEvent):
    """Content part added to response."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: Optional[str] = None
    response_id: Optional[str] = None
    content_index: Optional[int] = None
    output_index: Optional[int] = None
    part: Optional[Dict[str, Any]] = None


class InworldResponseContentPartDoneEvent(BaseInworldEvent):
    """Content part is complete."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: Optional[str] = None
    response_id: Optional[str] = None
    content_index: Optional[int] = None
    output_index: Optional[int] = None
    part: Optional[Dict[str, Any]] = None


class InworldResponseDoneEvent(BaseInworldEvent):
    """Assistant's response is completed (turn complete)."""

    type: Literal["response.done"] = "response.done"
    response: Optional[Dict[str, Any]] = None


# =============================================================================
# Audio Output Events
# =============================================================================


class InworldAudioDeltaEvent(BaseInworldEvent):
    """Audio chunk received from the model.

    Audio is base64-encoded PCM16 at 24 kHz.
    """

    type: Literal["response.output_audio.delta"] = "response.output_audio.delta"
    delta: str = Field(
        default="",
        description="Base64-encoded audio delta (PCM16 @ 24 kHz)",
        exclude=True,  # Exclude from serialization due to large size
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None


class InworldAudioDoneEvent(BaseInworldEvent):
    """Audio stream completed for current response."""

    type: Literal["response.output_audio.done"] = "response.output_audio.done"
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None


# =============================================================================
# Audio Transcript Events (Model's speech transcription)
# =============================================================================


class InworldAudioTranscriptDeltaEvent(BaseInworldEvent):
    """Transcript delta of the assistant's audio response."""

    type: Literal["response.output_audio_transcript.delta"] = (
        "response.output_audio_transcript.delta"
    )
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None


class InworldAudioTranscriptDoneEvent(BaseInworldEvent):
    """Audio transcript of assistant response is complete."""

    type: Literal["response.output_audio_transcript.done"] = (
        "response.output_audio_transcript.done"
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    transcript: str = ""


# Inworld also emits response.output_text.* events alongside audio. We model
# them so they parse cleanly, but the adapter ignores them (the audio
# transcript is the source of truth for what the model said).
class InworldOutputTextDeltaEvent(BaseInworldEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class InworldOutputTextDoneEvent(BaseInworldEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    text: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


# =============================================================================
# Function Call Events
# =============================================================================


class InworldFunctionCallArgumentsDeltaEvent(BaseInworldEvent):
    """Function call arguments delta (streaming)."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    delta: str = ""
    call_id: Optional[str] = None
    name: Optional[str] = None
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class InworldFunctionCallArgumentsDoneEvent(BaseInworldEvent):
    """Function call arguments are complete."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = "{}"
    response_id: Optional[str] = None
    item_id: Optional[str] = None


# =============================================================================
# Error and Utility Events
# =============================================================================


class InworldErrorEvent(BaseInworldEvent):
    """Error from the Inworld API."""

    type: Literal["error"] = "error"
    error: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    message: Optional[str] = None


class InworldTimeoutEvent(BaseInworldEvent):
    """Timeout waiting for events (used internally)."""

    type: Literal["timeout"] = "timeout"


class InworldUnknownEvent(BaseInworldEvent):
    """Unknown/unrecognized event type."""

    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


# =============================================================================
# Type Aliases
# =============================================================================

InworldEvent = Union[
    InworldSessionCreatedEvent,
    InworldSessionUpdatedEvent,
    InworldSpeechStartedEvent,
    InworldSpeechStoppedEvent,
    InworldInputAudioBufferCommittedEvent,
    InworldInputAudioBufferClearedEvent,
    InworldConversationItemAddedEvent,
    InworldConversationItemDoneEvent,
    InworldInputTranscriptionCompletedEvent,
    InworldResponseCreatedEvent,
    InworldResponseOutputItemAddedEvent,
    InworldResponseOutputItemDoneEvent,
    InworldResponseContentPartAddedEvent,
    InworldResponseContentPartDoneEvent,
    InworldResponseDoneEvent,
    InworldAudioDeltaEvent,
    InworldAudioDoneEvent,
    InworldAudioTranscriptDeltaEvent,
    InworldAudioTranscriptDoneEvent,
    InworldOutputTextDeltaEvent,
    InworldOutputTextDoneEvent,
    InworldFunctionCallArgumentsDeltaEvent,
    InworldFunctionCallArgumentsDoneEvent,
    InworldErrorEvent,
    InworldTimeoutEvent,
    InworldUnknownEvent,
]


# =============================================================================
# Event Parsing
# =============================================================================

_EVENT_TYPE_MAP: Dict[str, type[BaseInworldEvent]] = {
    "session.created": InworldSessionCreatedEvent,
    "session.updated": InworldSessionUpdatedEvent,
    "input_audio_buffer.speech_started": InworldSpeechStartedEvent,
    "input_audio_buffer.speech_stopped": InworldSpeechStoppedEvent,
    "input_audio_buffer.committed": InworldInputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": InworldInputAudioBufferClearedEvent,
    "conversation.item.added": InworldConversationItemAddedEvent,
    "conversation.item.done": InworldConversationItemDoneEvent,
    "conversation.item.input_audio_transcription.completed": InworldInputTranscriptionCompletedEvent,
    "response.created": InworldResponseCreatedEvent,
    "response.output_item.added": InworldResponseOutputItemAddedEvent,
    "response.output_item.done": InworldResponseOutputItemDoneEvent,
    "response.content_part.added": InworldResponseContentPartAddedEvent,
    "response.content_part.done": InworldResponseContentPartDoneEvent,
    "response.done": InworldResponseDoneEvent,
    "response.output_audio.delta": InworldAudioDeltaEvent,
    "response.output_audio.done": InworldAudioDoneEvent,
    "response.output_audio_transcript.delta": InworldAudioTranscriptDeltaEvent,
    "response.output_audio_transcript.done": InworldAudioTranscriptDoneEvent,
    "response.output_text.delta": InworldOutputTextDeltaEvent,
    "response.output_text.done": InworldOutputTextDoneEvent,
    "response.function_call_arguments.delta": InworldFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": InworldFunctionCallArgumentsDoneEvent,
    "error": InworldErrorEvent,
}


def parse_inworld_event(data: Dict[str, Any]) -> InworldEvent:
    """Parse a raw Inworld WebSocket message into a typed event."""
    event_type = data.get("type", "unknown")

    log_data = data.copy()
    if "delta" in log_data and event_type == "response.output_audio.delta":
        delta = log_data.get("delta", "")
        log_data["delta"] = f"<{len(delta)} base64 chars>"
    logger.debug(f"Inworld event: {event_type} - {log_data}")

    event_class = _EVENT_TYPE_MAP.get(event_type)
    if event_class:
        try:
            return event_class(**data)
        except Exception as e:
            logger.warning(f"Failed to parse Inworld event {event_type}: {e}")
            return InworldUnknownEvent(type=event_type, raw=data)
    logger.debug(f"Unknown Inworld event type: {event_type}")
    return InworldUnknownEvent(type=event_type, raw=data)
