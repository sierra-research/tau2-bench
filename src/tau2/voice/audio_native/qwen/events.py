"""Pydantic models for Qwen Omni Flash Realtime API events.

Qwen uses an OpenAI-compatible protocol, so we reuse the OpenAI event models.
This file provides Qwen-specific aliases and any Qwen-specific event types.

Reference: https://www.alibabacloud.com/help/en/model-studio/realtime
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseQwenEvent(BaseModel):
    """Base class for all Qwen Realtime API events."""

    type: str
    event_id: Optional[str] = None


# =============================================================================
# Response Streaming Events
# =============================================================================


class QwenTextDeltaEvent(BaseQwenEvent):
    """Incremental text content from the model."""

    type: Literal["response.text.delta"]
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class QwenTextDoneEvent(BaseQwenEvent):
    """Text content complete."""

    type: Literal["response.text.done"]
    text: str = ""
    item_id: Optional[str] = None


class QwenAudioDeltaEvent(BaseQwenEvent):
    """Incremental audio data (base64-encoded PCM)."""

    type: Literal["response.audio.delta"]
    delta: str = Field(
        default="", description="Base64-encoded audio delta", exclude=True
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class QwenAudioDoneEvent(BaseQwenEvent):
    """Audio for item complete."""

    type: Literal["response.audio.done"]
    item_id: Optional[str] = None


class QwenAudioTranscriptDeltaEvent(BaseQwenEvent):
    """Incremental transcript of audio output."""

    type: Literal["response.audio_transcript.delta"]
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class QwenAudioTranscriptDoneEvent(BaseQwenEvent):
    """Transcript of audio output complete."""

    type: Literal["response.audio_transcript.done"]
    transcript: str = ""
    item_id: Optional[str] = None


# =============================================================================
# Function Call Events
# =============================================================================


class QwenFunctionCallArgumentsDeltaEvent(BaseQwenEvent):
    """Incremental function call arguments."""

    type: Literal["response.function_call_arguments.delta"]
    delta: str = ""
    call_id: Optional[str] = None
    name: Optional[str] = None


class QwenFunctionCallArgumentsDoneEvent(BaseQwenEvent):
    """Function call complete with full arguments."""

    type: Literal["response.function_call_arguments.done"]
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = "{}"


class QwenOutputItemAddedEvent(BaseQwenEvent):
    """New output item (message or function_call)."""

    type: Literal["response.output_item.added"]
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    call_id: Optional[str] = None


class QwenOutputItemDoneEvent(BaseQwenEvent):
    """Output item complete."""

    type: Literal["response.output_item.done"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class QwenResponseDoneEvent(BaseQwenEvent):
    """Response complete, includes usage stats."""

    type: Literal["response.done"]
    response_id: Optional[str] = None
    status: Optional[str] = None
    usage: Optional[dict] = None


class QwenResponseCancelledEvent(BaseQwenEvent):
    """Response cancelled due to interruption."""

    type: Literal["response.cancelled"]
    response_id: Optional[str] = None


# =============================================================================
# Speech Detection Events (VAD)
# =============================================================================


class QwenSpeechStartedEvent(BaseQwenEvent):
    """User started speaking (VAD detected speech start)."""

    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: Optional[int] = None
    item_id: Optional[str] = None


class QwenSpeechStoppedEvent(BaseQwenEvent):
    """User stopped speaking (VAD detected speech end)."""

    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: Optional[int] = None
    item_id: Optional[str] = None


class QwenInputAudioBufferCommittedEvent(BaseQwenEvent):
    """Audio buffer committed."""

    type: Literal["input_audio_buffer.committed"]
    item_id: Optional[str] = None


class QwenInputAudioBufferClearedEvent(BaseQwenEvent):
    """Audio buffer cleared."""

    type: Literal["input_audio_buffer.cleared"]


# =============================================================================
# Conversation Events
# =============================================================================


class QwenConversationItemTruncatedEvent(BaseQwenEvent):
    """Conversation item truncated."""

    type: Literal["conversation.item.truncated"]
    item_id: Optional[str] = None
    audio_end_ms: Optional[int] = None


class QwenConversationItemCreatedEvent(BaseQwenEvent):
    """Conversation item created."""

    type: Literal["conversation.item.created"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class QwenInputAudioTranscriptionCompletedEvent(BaseQwenEvent):
    """Transcription of user's audio input ready."""

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: Optional[str] = None
    content_index: Optional[int] = None
    transcript: str = ""


# =============================================================================
# Session Events
# =============================================================================


class QwenSessionCreatedEvent(BaseQwenEvent):
    """Session created successfully."""

    type: Literal["session.created"]
    session: Optional[dict] = None


class QwenSessionUpdatedEvent(BaseQwenEvent):
    """Session updated."""

    type: Literal["session.updated"]


# =============================================================================
# Utility Events
# =============================================================================


class QwenResponseCreatedEvent(BaseQwenEvent):
    """Response started."""

    type: Literal["response.created"]


class QwenContentPartAddedEvent(BaseQwenEvent):
    """Content part added to response."""

    type: Literal["response.content_part.added"]


class QwenContentPartDoneEvent(BaseQwenEvent):
    """Content part complete."""

    type: Literal["response.content_part.done"]


class QwenRateLimitsUpdatedEvent(BaseQwenEvent):
    """Rate limits updated."""

    type: Literal["rate_limits.updated"]


class QwenErrorEvent(BaseQwenEvent):
    """Error from the API."""

    type: Literal["error"]
    code: Optional[str] = None
    message: Optional[str] = None


class QwenTimeoutEvent(BaseQwenEvent):
    """Timeout waiting for events (internal use)."""

    type: Literal["timeout"]


class QwenUnknownEvent(BaseQwenEvent):
    """Unknown/unrecognized event type."""

    type: str
    raw: Optional[dict] = None


# =============================================================================
# Type Aliases
# =============================================================================

_KnownQwenEvents = Union[
    # Response streaming
    QwenTextDeltaEvent,
    QwenTextDoneEvent,
    QwenAudioDeltaEvent,
    QwenAudioDoneEvent,
    QwenAudioTranscriptDeltaEvent,
    QwenAudioTranscriptDoneEvent,
    # Function calls
    QwenFunctionCallArgumentsDeltaEvent,
    QwenFunctionCallArgumentsDoneEvent,
    QwenOutputItemAddedEvent,
    QwenOutputItemDoneEvent,
    QwenResponseDoneEvent,
    QwenResponseCancelledEvent,
    # Speech detection
    QwenSpeechStartedEvent,
    QwenSpeechStoppedEvent,
    QwenInputAudioBufferCommittedEvent,
    QwenInputAudioBufferClearedEvent,
    # Conversation
    QwenConversationItemTruncatedEvent,
    QwenConversationItemCreatedEvent,
    QwenInputAudioTranscriptionCompletedEvent,
    # Session
    QwenSessionCreatedEvent,
    QwenSessionUpdatedEvent,
    # Utility
    QwenResponseCreatedEvent,
    QwenContentPartAddedEvent,
    QwenContentPartDoneEvent,
    QwenRateLimitsUpdatedEvent,
    QwenErrorEvent,
    QwenTimeoutEvent,
]

QwenRealtimeEvent = Annotated[_KnownQwenEvents, Field(discriminator="type")]

# =============================================================================
# Event Parsing
# =============================================================================

_EVENT_TYPE_MAP: dict[str, type[BaseQwenEvent]] = {
    # Response streaming
    "response.text.delta": QwenTextDeltaEvent,
    "response.text.done": QwenTextDoneEvent,
    "response.audio.delta": QwenAudioDeltaEvent,
    "response.audio.done": QwenAudioDoneEvent,
    "response.audio_transcript.delta": QwenAudioTranscriptDeltaEvent,
    "response.audio_transcript.done": QwenAudioTranscriptDoneEvent,
    # Function calls
    "response.function_call_arguments.delta": QwenFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": QwenFunctionCallArgumentsDoneEvent,
    "response.output_item.added": QwenOutputItemAddedEvent,
    "response.output_item.done": QwenOutputItemDoneEvent,
    "response.done": QwenResponseDoneEvent,
    "response.cancelled": QwenResponseCancelledEvent,
    "response.created": QwenResponseCreatedEvent,
    "response.content_part.added": QwenContentPartAddedEvent,
    "response.content_part.done": QwenContentPartDoneEvent,
    # Speech detection
    "input_audio_buffer.speech_started": QwenSpeechStartedEvent,
    "input_audio_buffer.speech_stopped": QwenSpeechStoppedEvent,
    "input_audio_buffer.committed": QwenInputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": QwenInputAudioBufferClearedEvent,
    # Conversation
    "conversation.item.truncated": QwenConversationItemTruncatedEvent,
    "conversation.item.created": QwenConversationItemCreatedEvent,
    "conversation.item.input_audio_transcription.completed": QwenInputAudioTranscriptionCompletedEvent,
    # Session
    "session.created": QwenSessionCreatedEvent,
    "session.updated": QwenSessionUpdatedEvent,
    # Utility
    "rate_limits.updated": QwenRateLimitsUpdatedEvent,
    "error": QwenErrorEvent,
    "timeout": QwenTimeoutEvent,
}


def parse_qwen_event(raw_data: dict) -> BaseQwenEvent:
    """Parse raw event data into a typed Qwen event model.

    Args:
        raw_data: Raw event dictionary from WebSocket.

    Returns:
        Typed QwenEvent instance.
    """
    event_type = raw_data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return QwenUnknownEvent(type=event_type, raw=raw_data)

    parsed_data = _extract_event_fields(event_type, raw_data)
    return event_class.model_validate(parsed_data)


def _extract_event_fields(event_type: str, raw_data: dict) -> dict:
    """Extract relevant fields from raw event data based on event type."""
    result = {
        "type": event_type,
        "event_id": raw_data.get("event_id"),
    }

    if event_type == "response.output_item.added":
        item = raw_data.get("item", {})
        result.update(
            {
                "item_id": item.get("id"),
                "item_type": item.get("type"),
                "role": item.get("role"),
                "name": item.get("name"),
                "call_id": item.get("call_id"),
            }
        )

    elif event_type == "response.output_item.done":
        item = raw_data.get("item", {})
        result.update(
            {
                "item_id": item.get("id"),
                "item": item,
            }
        )

    elif event_type == "response.done":
        response = raw_data.get("response", {})
        result.update(
            {
                "response_id": response.get("id"),
                "status": response.get("status"),
                "usage": response.get("usage"),
            }
        )

    elif event_type == "error":
        error = raw_data.get("error", {})
        result.update(
            {
                "code": error.get("code"),
                "message": error.get("message"),
            }
        )

    elif event_type == "conversation.item.created":
        item = raw_data.get("item", {})
        result.update(
            {
                "item_id": item.get("id"),
                "item": item,
            }
        )

    elif event_type == "session.created":
        result["session"] = raw_data.get("session")

    else:
        for key in [
            "delta",
            "text",
            "transcript",
            "response_id",
            "item_id",
            "call_id",
            "name",
            "arguments",
            "audio_start_ms",
            "audio_end_ms",
            "content_index",
        ]:
            if key in raw_data:
                result[key] = raw_data[key]

    return result
