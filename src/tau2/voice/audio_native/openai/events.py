"""Pydantic models for OpenAI Realtime API events."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseRealtimeEvent(BaseModel):
    """Base class for all Realtime API events."""

    type: str
    event_id: Optional[str] = None


class TextDeltaEvent(BaseRealtimeEvent):
    type: Literal["response.text.delta"]
    delta: str
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class TextDoneEvent(BaseRealtimeEvent):
    type: Literal["response.text.done"]
    text: str = ""
    item_id: Optional[str] = None


class AudioDeltaEvent(BaseRealtimeEvent):
    type: Literal["response.audio.delta"]
    delta: str = Field(
        default="", description="Base64-encoded audio delta", exclude=True
    )  # Exclude from serialization due to large size
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class AudioDoneEvent(BaseRealtimeEvent):
    type: Literal["response.audio.done"]
    item_id: Optional[str] = None


class AudioTranscriptDeltaEvent(BaseRealtimeEvent):
    type: Literal["response.audio_transcript.delta"]
    delta: str
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class AudioTranscriptDoneEvent(BaseRealtimeEvent):
    type: Literal["response.audio_transcript.done"]
    transcript: str = ""
    item_id: Optional[str] = None


class FunctionCallArgumentsDeltaEvent(BaseRealtimeEvent):
    type: Literal["response.function_call_arguments.delta"]
    delta: str
    call_id: Optional[str] = None
    name: Optional[str] = None


class FunctionCallArgumentsDoneEvent(BaseRealtimeEvent):
    type: Literal["response.function_call_arguments.done"]
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = "{}"


class OutputItemAddedEvent(BaseRealtimeEvent):
    type: Literal["response.output_item.added"]
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    call_id: Optional[str] = None


class OutputItemDoneEvent(BaseRealtimeEvent):
    type: Literal["response.output_item.done"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class ResponseDoneEvent(BaseRealtimeEvent):
    type: Literal["response.done"]
    response_id: Optional[str] = None
    status: Optional[str] = None
    usage: Optional[dict] = None


class ResponseCancelledEvent(BaseRealtimeEvent):
    """Emitted when the server cancels an in-progress response due to user interruption.

    This event is sent when VAD detects user speech while the assistant is responding.
    The server automatically cancels the response and emits this event.
    """

    type: Literal["response.cancelled"]
    response_id: Optional[str] = None


class SpeechStartedEvent(BaseRealtimeEvent):
    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: Optional[int] = None
    item_id: Optional[str] = None


class SpeechStoppedEvent(BaseRealtimeEvent):
    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: Optional[int] = None
    item_id: Optional[str] = None


class InputAudioBufferCommittedEvent(BaseRealtimeEvent):
    type: Literal["input_audio_buffer.committed"]
    item_id: Optional[str] = None


class InputAudioBufferClearedEvent(BaseRealtimeEvent):
    type: Literal["input_audio_buffer.cleared"]


class ConversationItemTruncatedEvent(BaseRealtimeEvent):
    type: Literal["conversation.item.truncated"]
    item_id: Optional[str] = None
    audio_end_ms: Optional[int] = None


class ConversationItemCreatedEvent(BaseRealtimeEvent):
    type: Literal["conversation.item.created"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class InputAudioTranscriptionCompletedEvent(BaseRealtimeEvent):
    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: Optional[str] = None
    content_index: Optional[int] = None
    transcript: str = ""


class ErrorEvent(BaseRealtimeEvent):
    type: Literal["error"]
    code: Optional[str] = None
    message: Optional[str] = None


class TimeoutEvent(BaseRealtimeEvent):
    type: Literal["timeout"]


# Informational events (not critical to process, but good to recognize)
class SessionCreatedEvent(BaseRealtimeEvent):
    type: Literal["session.created"]


class SessionUpdatedEvent(BaseRealtimeEvent):
    type: Literal["session.updated"]


class RateLimitsUpdatedEvent(BaseRealtimeEvent):
    type: Literal["rate_limits.updated"]


class ResponseCreatedEvent(BaseRealtimeEvent):
    type: Literal["response.created"]


class ContentPartAddedEvent(BaseRealtimeEvent):
    type: Literal["response.content_part.added"]


class ContentPartDoneEvent(BaseRealtimeEvent):
    type: Literal["response.content_part.done"]


class UnknownEvent(BaseRealtimeEvent):
    type: str
    raw: Optional[dict] = None


_KnownEvents = Union[
    TextDeltaEvent,
    TextDoneEvent,
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ResponseDoneEvent,
    ResponseCancelledEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferClearedEvent,
    ConversationItemTruncatedEvent,
    ConversationItemCreatedEvent,
    InputAudioTranscriptionCompletedEvent,
    ErrorEvent,
    TimeoutEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    RateLimitsUpdatedEvent,
    ResponseCreatedEvent,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
]

RealtimeEvent = Annotated[_KnownEvents, Field(discriminator="type")]

_EVENT_TYPE_MAP: dict[str, type[BaseRealtimeEvent]] = {
    "response.text.delta": TextDeltaEvent,
    "response.text.done": TextDoneEvent,
    "response.audio.delta": AudioDeltaEvent,
    "response.audio.done": AudioDoneEvent,
    "response.audio_transcript.delta": AudioTranscriptDeltaEvent,
    "response.audio_transcript.done": AudioTranscriptDoneEvent,
    "response.function_call_arguments.delta": FunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": FunctionCallArgumentsDoneEvent,
    "response.output_item.added": OutputItemAddedEvent,
    "response.output_item.done": OutputItemDoneEvent,
    "response.done": ResponseDoneEvent,
    "response.cancelled": ResponseCancelledEvent,
    "response.created": ResponseCreatedEvent,
    "response.content_part.added": ContentPartAddedEvent,
    "response.content_part.done": ContentPartDoneEvent,
    "input_audio_buffer.speech_started": SpeechStartedEvent,
    "input_audio_buffer.speech_stopped": SpeechStoppedEvent,
    "input_audio_buffer.committed": InputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": InputAudioBufferClearedEvent,
    "conversation.item.truncated": ConversationItemTruncatedEvent,
    "conversation.item.created": ConversationItemCreatedEvent,
    "conversation.item.input_audio_transcription.completed": InputAudioTranscriptionCompletedEvent,
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "rate_limits.updated": RateLimitsUpdatedEvent,
    "error": ErrorEvent,
    "timeout": TimeoutEvent,
}


def parse_realtime_event(raw_data: dict) -> BaseRealtimeEvent:
    """Parse raw event data into a typed Pydantic event model."""
    event_type = raw_data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return UnknownEvent(type=event_type, raw=raw_data)

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
        ]:
            if key in raw_data:
                result[key] = raw_data[key]

    return result
