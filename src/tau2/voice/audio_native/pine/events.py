"""Pydantic models for Pine Realtime protocol events.

Pine's wire format mirrors OpenAI Realtime's. We re-export OpenAI's event
classes under Pine-prefixed aliases so the adapter is fully typed without
duplicating definitions, but keep our own parser dispatch so unsupported
events do not crash if a future Pine server emits them.
"""

from typing import Optional

from pydantic import BaseModel

from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    ConversationItemCreatedEvent,
    ConversationItemTruncatedEvent,
    ErrorEvent,
    FunctionCallArgumentsDoneEvent,
    InputAudioTranscriptionCompletedEvent,
    RateLimitsUpdatedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TimeoutEvent,
    UnknownEvent,
)


# Aliases - Pine uses the same event types as OpenAI Realtime.
BasePineEvent = BaseRealtimeEvent
PineTimeoutEvent = TimeoutEvent
PineUnknownEvent = UnknownEvent


class PineErrorDetails(BaseModel):
    """Structured Pine error payload."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    fatal: bool = False


# ---------------------------------------------------------------------------
# Event-type -> class map.
# ---------------------------------------------------------------------------
_EVENT_TYPE_MAP: dict[str, type[BaseRealtimeEvent]] = {
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "response.created": ResponseCreatedEvent,
    "response.output_audio.delta": AudioDeltaEvent,
    "response.output_audio.done": AudioDoneEvent,
    "response.output_audio_transcript.delta": AudioTranscriptDeltaEvent,
    "response.output_audio_transcript.done": AudioTranscriptDoneEvent,
    "response.function_call_arguments.done": FunctionCallArgumentsDoneEvent,
    "response.done": ResponseDoneEvent,
    "input_audio_buffer.speech_started": SpeechStartedEvent,
    "input_audio_buffer.speech_stopped": SpeechStoppedEvent,
    "conversation.item.created": ConversationItemCreatedEvent,
    "conversation.item.truncated": ConversationItemTruncatedEvent,
    "conversation.item.input_audio_transcription.completed": InputAudioTranscriptionCompletedEvent,
    "rate_limits.updated": RateLimitsUpdatedEvent,
    "error": ErrorEvent,
    "timeout": TimeoutEvent,
}


def parse_pine_event(raw_data: dict) -> BasePineEvent:
    """Parse raw event data into a typed event model.

    Mirrors OpenAI Realtime's event-extraction logic since Pine's wire format
    is identical for the supported event subset. Unknown event types fall back
    to UnknownEvent rather than crashing.
    """
    event_type = raw_data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return UnknownEvent(type=event_type, raw=raw_data)

    parsed = _extract_fields(event_type, raw_data)
    return event_class.model_validate(parsed)


def _extract_fields(event_type: str, raw: dict) -> dict:
    """Pull the fields each event class expects from the raw JSON payload."""
    out: dict = {"type": event_type, "event_id": raw.get("event_id")}

    if event_type == "response.done":
        out["response_id"] = raw.get("response_id") or raw.get("response", {}).get("id")
        out["status"] = raw.get("status") or raw.get("response", {}).get("status")
        out["usage"] = raw.get("usage") or raw.get("response", {}).get("usage")
    elif event_type == "error":
        err = raw.get("error", {})
        out["code"] = err.get("code")
        out["message"] = err.get("message")
    elif event_type == "conversation.item.created":
        item = raw.get("item", {})
        out["item_id"] = item.get("id")
        out["item"] = item
    elif event_type == "session.created":
        sess = raw.get("session", {})
        out["event_id"] = out["event_id"] or sess.get("id")
    else:
        for key in (
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
        ):
            if key in raw:
                out[key] = raw[key]
    return out
