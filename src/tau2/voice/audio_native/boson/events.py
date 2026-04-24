"""Pydantic models for Boson realtime voice chat events."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class BaseBosonEvent(BaseModel):
    """Base class for Boson realtime events."""

    type: str
    event_id: Optional[str] = None


class BosonTextDeltaEvent(BaseBosonEvent):
    """Incremental text content from the model."""

    type: Literal["response.text.delta"]
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class BosonTextDoneEvent(BaseBosonEvent):
    """Text content complete."""

    type: Literal["response.text.done"]
    text: str = ""
    item_id: Optional[str] = None


class BosonAudioDeltaEvent(BaseBosonEvent):
    """Incremental audio data."""

    type: Literal["response.audio.delta"]
    delta: str = Field(
        default="", description="Base64-encoded audio delta", exclude=True
    )
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class BosonAudioDoneEvent(BaseBosonEvent):
    """Audio for an item is complete."""

    type: Literal["response.audio.done"]
    item_id: Optional[str] = None


class BosonAudioTranscriptDeltaEvent(BaseBosonEvent):
    """Incremental transcript of audio output."""

    type: Literal["response.audio_transcript.delta"]
    delta: str = ""
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class BosonAudioTranscriptLengthEvent(BaseBosonEvent):
    """Transcript delta with audio length metadata."""

    type: Literal["response.audio_transcript.length"]
    delta: str = ""
    length_ms: Optional[int] = None
    response_id: Optional[str] = None
    item_id: Optional[str] = None


class BosonAudioTranscriptDoneEvent(BaseBosonEvent):
    """Transcript of audio output is complete."""

    type: Literal["response.audio_transcript.done"]
    transcript: str = ""
    item_id: Optional[str] = None


class BosonFunctionCallArgumentsDeltaEvent(BaseBosonEvent):
    """Incremental function call arguments."""

    type: Literal["response.function_call_arguments.delta"]
    delta: str = ""
    call_id: Optional[str] = None
    name: Optional[str] = None


class BosonFunctionCallArgumentsDoneEvent(BaseBosonEvent):
    """Function call arguments are complete."""

    type: Literal["response.function_call_arguments.done"]
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = "{}"


class BosonOutputItemAddedEvent(BaseBosonEvent):
    """New output item was added to the response."""

    type: Literal["response.output_item.added"]
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    call_id: Optional[str] = None


class BosonOutputItemDoneEvent(BaseBosonEvent):
    """Output item is complete."""

    type: Literal["response.output_item.done"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class BosonResponseCreatedEvent(BaseBosonEvent):
    """Response generation started."""

    type: Literal["response.created"]
    response_id: Optional[str] = None
    status: Optional[str] = None


class BosonResponseDoneEvent(BaseBosonEvent):
    """Response generation completed, cancelled, or failed."""

    type: Literal["response.done"]
    response_id: Optional[str] = None
    status: Optional[str] = None
    usage: Optional[dict] = None


class BosonSpeechStartedEvent(BaseBosonEvent):
    """Server-side VAD detected speech start."""

    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: Optional[int] = None
    item_id: Optional[str] = None


class BosonSpeechStoppedEvent(BaseBosonEvent):
    """Server-side VAD detected speech end."""

    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: Optional[int] = None
    item_id: Optional[str] = None


class BosonInputAudioBufferCommittedEvent(BaseBosonEvent):
    """The input audio buffer was committed."""

    type: Literal["input_audio_buffer.committed"]
    item_id: Optional[str] = None


class BosonInputAudioBufferClearedEvent(BaseBosonEvent):
    """The input audio buffer was cleared."""

    type: Literal["input_audio_buffer.cleared"]


class BosonConversationItemCreatedEvent(BaseBosonEvent):
    """A conversation item was created."""

    type: Literal["conversation.item.created"]
    item_id: Optional[str] = None
    item: Optional[dict] = None


class BosonConversationItemTruncatedEvent(BaseBosonEvent):
    """An assistant conversation item's audio was truncated."""

    type: Literal["conversation.item.truncated"]
    item_id: Optional[str] = None
    audio_end_ms: Optional[int] = None


class BosonInputAudioTranscriptionCompletedEvent(BaseBosonEvent):
    """User audio transcription completed."""

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: Optional[str] = None
    content_index: Optional[int] = None
    transcript: str = ""


class BosonSessionCreatedEvent(BaseBosonEvent):
    """Session created successfully."""

    type: Literal["session.created"]
    session: Optional[dict] = None


class BosonSessionUpdatedEvent(BaseBosonEvent):
    """Session updated successfully."""

    type: Literal["session.updated"]
    session: Optional[dict] = None


class BosonConversationCreatedEvent(BaseBosonEvent):
    """Conversation created for the session."""

    type: Literal["conversation.created"]
    conversation: Optional[dict] = None


class BosonRateLimitsUpdatedEvent(BaseBosonEvent):
    """Rate limits were updated."""

    type: Literal["rate_limits.updated"]
    limits: Optional[list[dict]] = None


class BosonShouldEndCallEvent(BaseBosonEvent):
    """Server suggested ending the call."""

    type: Literal["should_end_call"]
    response_id: Optional[str] = None


class BosonErrorEvent(BaseBosonEvent):
    """Boson error event."""

    type: Literal["error"]
    code: Optional[str] = None
    message: Optional[str] = None


class BosonTimeoutEvent(BaseBosonEvent):
    """Synthetic timeout event emitted by the local adapter."""

    type: Literal["timeout"]


class BosonUnknownEvent(BaseBosonEvent):
    """Fallback for events that are not yet modeled."""

    type: str
    raw: Optional[dict] = None


_KnownEvents = Union[
    BosonTextDeltaEvent,
    BosonTextDoneEvent,
    BosonAudioDeltaEvent,
    BosonAudioDoneEvent,
    BosonAudioTranscriptDeltaEvent,
    BosonAudioTranscriptLengthEvent,
    BosonAudioTranscriptDoneEvent,
    BosonFunctionCallArgumentsDeltaEvent,
    BosonFunctionCallArgumentsDoneEvent,
    BosonOutputItemAddedEvent,
    BosonOutputItemDoneEvent,
    BosonResponseCreatedEvent,
    BosonResponseDoneEvent,
    BosonSpeechStartedEvent,
    BosonSpeechStoppedEvent,
    BosonInputAudioBufferCommittedEvent,
    BosonInputAudioBufferClearedEvent,
    BosonConversationItemCreatedEvent,
    BosonConversationItemTruncatedEvent,
    BosonInputAudioTranscriptionCompletedEvent,
    BosonSessionCreatedEvent,
    BosonSessionUpdatedEvent,
    BosonConversationCreatedEvent,
    BosonRateLimitsUpdatedEvent,
    BosonShouldEndCallEvent,
    BosonErrorEvent,
    BosonTimeoutEvent,
]

BosonEvent = Annotated[_KnownEvents, Field(discriminator="type")]

_EVENT_TYPE_MAP: dict[str, type[BaseBosonEvent]] = {
    "response.text.delta": BosonTextDeltaEvent,
    "response.text.done": BosonTextDoneEvent,
    "response.audio.delta": BosonAudioDeltaEvent,
    "response.audio.done": BosonAudioDoneEvent,
    "response.audio_transcript.delta": BosonAudioTranscriptDeltaEvent,
    "response.audio_transcript.length": BosonAudioTranscriptLengthEvent,
    "response.audio_transcript.done": BosonAudioTranscriptDoneEvent,
    "response.function_call_arguments.delta": BosonFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": BosonFunctionCallArgumentsDoneEvent,
    "response.output_item.added": BosonOutputItemAddedEvent,
    "response.output_item.done": BosonOutputItemDoneEvent,
    "response.created": BosonResponseCreatedEvent,
    "response.done": BosonResponseDoneEvent,
    "input_audio_buffer.speech_started": BosonSpeechStartedEvent,
    "input_audio_buffer.speech_stopped": BosonSpeechStoppedEvent,
    "input_audio_buffer.committed": BosonInputAudioBufferCommittedEvent,
    "input_audio_buffer.cleared": BosonInputAudioBufferClearedEvent,
    "conversation.item.created": BosonConversationItemCreatedEvent,
    "conversation.item.truncated": BosonConversationItemTruncatedEvent,
    "conversation.item.input_audio_transcription.completed": (
        BosonInputAudioTranscriptionCompletedEvent
    ),
    "session.created": BosonSessionCreatedEvent,
    "session.updated": BosonSessionUpdatedEvent,
    "conversation.created": BosonConversationCreatedEvent,
    "rate_limits.updated": BosonRateLimitsUpdatedEvent,
    "should_end_call": BosonShouldEndCallEvent,
    "error": BosonErrorEvent,
    "timeout": BosonTimeoutEvent,
}


def parse_boson_event(raw_data: dict) -> BaseBosonEvent:
    """Parse raw event data into a typed Boson event."""
    event_type = raw_data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return BosonUnknownEvent(type=event_type, raw=raw_data)

    parsed_data = _extract_event_fields(event_type, raw_data)
    return event_class.model_validate(parsed_data)


def _extract_event_fields(event_type: str, raw_data: dict) -> dict:
    """Extract relevant fields from a raw Boson event."""
    result = {
        "type": event_type,
        "event_id": raw_data.get("event_id"),
    }

    if event_type in ("session.created", "session.updated"):
        result["session"] = raw_data.get("session")

    elif event_type == "conversation.created":
        result["conversation"] = raw_data.get("conversation")

    elif event_type == "response.output_item.added":
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
        result.update({"item_id": item.get("id"), "item": item})

    elif event_type in ("response.created", "response.done"):
        response = raw_data.get("response", {})
        result.update(
            {
                "response_id": response.get("id"),
                "status": response.get("status"),
                "usage": response.get("usage"),
            }
        )

    elif event_type == "conversation.item.created":
        item = raw_data.get("item", {})
        result.update({"item_id": item.get("id"), "item": item})

    elif event_type == "error":
        error = raw_data.get("error", {})
        result.update({"code": error.get("code"), "message": error.get("message")})

    elif event_type == "rate_limits.updated":
        result["limits"] = raw_data.get("limits")

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
            "length_ms",
        ]:
            if key in raw_data:
                result[key] = raw_data[key]

    return result
