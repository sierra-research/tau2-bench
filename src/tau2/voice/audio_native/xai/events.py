"""Pydantic models for xAI Grok Voice Agent API events.

xAI's Realtime API uses a WebSocket protocol very similar to OpenAI's Realtime API.
Key event types:
- Audio output: response.output_audio.delta/done
- Audio transcript: response.output_audio_transcript.delta/done
- Input transcription: conversation.item.input_audio_transcription.completed
- Function calls: response.function_call_arguments.done
- VAD events: input_audio_buffer.speech_started/stopped
- Turn completion: response.done

Reference: https://docs.x.ai/docs/guides/voice/agent
"""

from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class BaseXAIEvent(BaseModel):
    """Base class for all xAI Realtime API events."""

    type: str
    event_id: Optional[str] = None


# =============================================================================
# Session Events
# =============================================================================


class XAIConversationCreatedEvent(BaseXAIEvent):
    """First message at connection - conversation session created."""

    type: Literal["conversation.created"] = "conversation.created"
    conversation: Optional[Dict[str, Any]] = None


class XAISessionUpdatedEvent(BaseXAIEvent):
    """Session configuration has been updated."""

    type: Literal["session.updated"] = "session.updated"
    session: Optional[Dict[str, Any]] = None


# =============================================================================
# Input Audio Buffer Events (VAD)
# =============================================================================


class XAISpeechStartedEvent(BaseXAIEvent):
    """Server VAD detected start of speech."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    item_id: Optional[str] = None


class XAISpeechStoppedEvent(BaseXAIEvent):
    """Server VAD detected end of speech."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    item_id: Optional[str] = None


class XAIInputAudioBufferCommittedEvent(BaseXAIEvent):
    """Input audio buffer has been committed."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    previous_item_id: Optional[str] = None
    item_id: Optional[str] = None


class XAIInputAudioBufferClearedEvent(BaseXAIEvent):
    """Input audio buffer has been cleared."""

    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


# =============================================================================
# Conversation Item Events
# =============================================================================


class XAIConversationItemAddedEvent(BaseXAIEvent):
    """A new item has been added to conversation history."""

    type: Literal["conversation.item.added"] = "conversation.item.added"
    previous_item_id: Optional[str] = None
    item: Optional[Dict[str, Any]] = None


class XAIInputTranscriptionCompletedEvent(BaseXAIEvent):
    """Transcription of user's audio input is complete."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: Optional[str] = None
    transcript: str = ""


# =============================================================================
# Response Events
# =============================================================================


class XAIResponseCreatedEvent(BaseXAIEvent):
    """A new assistant response turn is in progress."""

    type: Literal["response.created"] = "response.created"
    response: Optional[Dict[str, Any]] = None


class XAIResponseOutputItemAddedEvent(BaseXAIEvent):
    """A new assistant response item is added to message history."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[Dict[str, Any]] = None


class XAIResponseOutputItemDoneEvent(BaseXAIEvent):
    """Response output item is complete."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[Dict[str, Any]] = None


class XAIResponseContentPartAddedEvent(BaseXAIEvent):
    """Content part added to response."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: Optional[str] = None
    response_id: Optional[str] = None
    content_index: Optional[int] = None
    output_index: Optional[int] = None
    part: Optional[Dict[str, Any]] = None


class XAIResponseContentPartDoneEvent(BaseXAIEvent):
    """Content part is complete."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: Optional[str] = None
    response_id: Optional[str] = None
    content_index: Optional[int] = None
    output_index: Optional[int] = None


class XAIResponseDoneEvent(BaseXAIEvent):
    """Assistant's response is completed (turn complete)."""

    type: Literal["response.done"] = "response.done"
    response: Optional[Dict[str, Any]] = None


# =============================================================================
# Audio Output Events
# =============================================================================


class XAIAudioDeltaEvent(BaseXAIEvent):
    """Audio chunk received from the model.

    Audio is base64-encoded. Format depends on session configuration
    (G.711 μ-law at 8kHz for telephony, or PCM at configured rate).
    """

    type: Literal["response.output_audio.delta"] = "response.output_audio.delta"
    delta: str = Field(
        default="",
        description="Base64-encoded audio delta",
        exclude=True,  # Exclude from serialization due to large size
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None


class XAIAudioDoneEvent(BaseXAIEvent):
    """Audio stream completed for current response."""

    type: Literal["response.output_audio.done"] = "response.output_audio.done"
    response_id: Optional[str] = None
    item_id: Optional[str] = None


# =============================================================================
# Audio Transcript Events (Model's speech transcription)
# =============================================================================


class XAIAudioTranscriptDeltaEvent(BaseXAIEvent):
    """Transcript delta of the assistant's audio response."""

    type: Literal["response.output_audio_transcript.delta"] = (
        "response.output_audio_transcript.delta"
    )
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class XAIAudioTranscriptDoneEvent(BaseXAIEvent):
    """Audio transcript of assistant response is complete."""

    type: Literal["response.output_audio_transcript.done"] = (
        "response.output_audio_transcript.done"
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    transcript: str = ""


# =============================================================================
# Function Call Events
# =============================================================================


class XAIFunctionCallArgumentsDeltaEvent(BaseXAIEvent):
    """Function call arguments delta (streaming)."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    delta: str = ""
    call_id: Optional[str] = None
    name: Optional[str] = None
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class XAIFunctionCallArgumentsDoneEvent(BaseXAIEvent):
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


class XAIErrorEvent(BaseXAIEvent):
    """Error from the xAI API."""

    type: Literal["error"] = "error"
    error: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    message: Optional[str] = None


class XAITimeoutEvent(BaseXAIEvent):
    """Timeout waiting for events (used internally)."""

    type: Literal["timeout"] = "timeout"


class XAIUnknownEvent(BaseXAIEvent):
    """Unknown/unrecognized event type."""

    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


# =============================================================================
# Type Aliases
# =============================================================================

XAIEvent = Union[
    # Session events
    XAIConversationCreatedEvent,
    XAISessionUpdatedEvent,
    # VAD events
    XAISpeechStartedEvent,
    XAISpeechStoppedEvent,
    XAIInputAudioBufferCommittedEvent,
    XAIInputAudioBufferClearedEvent,
    # Conversation events
    XAIConversationItemAddedEvent,
    XAIInputTranscriptionCompletedEvent,
    # Response events
    XAIResponseCreatedEvent,
    XAIResponseOutputItemAddedEvent,
    XAIResponseOutputItemDoneEvent,
    XAIResponseContentPartAddedEvent,
    XAIResponseContentPartDoneEvent,
    XAIResponseDoneEvent,
    # Audio events
    XAIAudioDeltaEvent,
    XAIAudioDoneEvent,
    XAIAudioTranscriptDeltaEvent,
    XAIAudioTranscriptDoneEvent,
    # Function call events
    XAIFunctionCallArgumentsDeltaEvent,
    XAIFunctionCallArgumentsDoneEvent,
    # Utility events
    XAIErrorEvent,
    XAITimeoutEvent,
    XAIUnknownEvent,
]


# =============================================================================
# Event Parsing
# =============================================================================

# Map event type strings to Pydantic model classes
_EVENT_TYPE_MAP: Dict[str, type[BaseXAIEvent]] = {
    # Session events
    "conversation.created": XAIConversationCreatedEvent,
    "session.updated": XAISessionUpdatedEvent,
    # VAD events
    "input_audio_buffer.speech_started": XAISpeechStartedEvent,
    "input_audio_buffer.speech_stopped": XAISpeechStoppedEvent,
    "input_audio_buffer.committed": XAIInputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": XAIInputAudioBufferClearedEvent,
    # Conversation events
    "conversation.item.added": XAIConversationItemAddedEvent,
    "conversation.item.input_audio_transcription.completed": XAIInputTranscriptionCompletedEvent,
    # Response events
    "response.created": XAIResponseCreatedEvent,
    "response.output_item.added": XAIResponseOutputItemAddedEvent,
    "response.output_item.done": XAIResponseOutputItemDoneEvent,
    "response.content_part.added": XAIResponseContentPartAddedEvent,
    "response.content_part.done": XAIResponseContentPartDoneEvent,
    "response.done": XAIResponseDoneEvent,
    # Audio events
    "response.output_audio.delta": XAIAudioDeltaEvent,
    "response.output_audio.done": XAIAudioDoneEvent,
    "response.output_audio_transcript.delta": XAIAudioTranscriptDeltaEvent,
    "response.output_audio_transcript.done": XAIAudioTranscriptDoneEvent,
    # Function call events
    "response.function_call_arguments.delta": XAIFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": XAIFunctionCallArgumentsDoneEvent,
    # Error
    "error": XAIErrorEvent,
}


def parse_xai_event(data: Dict[str, Any]) -> XAIEvent:
    """Parse a raw xAI WebSocket message into a typed event.

    Args:
        data: Raw JSON data from WebSocket message.

    Returns:
        Typed XAIEvent instance.
    """
    event_type = data.get("type", "unknown")

    # Log the event (with audio data size instead of full content)
    log_data = data.copy()
    if "delta" in log_data and event_type == "response.output_audio.delta":
        delta = log_data.get("delta", "")
        log_data["delta"] = f"<{len(delta)} base64 chars>"
    logger.debug(f"xAI event: {event_type} - {log_data}")

    # Look up the event class
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class:
        try:
            return event_class(**data)
        except Exception as e:
            logger.warning(f"Failed to parse xAI event {event_type}: {e}")
            return XAIUnknownEvent(type=event_type, raw=data)
    else:
        logger.debug(f"Unknown xAI event type: {event_type}")
        return XAIUnknownEvent(type=event_type, raw=data)
