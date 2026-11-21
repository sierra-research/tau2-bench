"""Apriel's extra ownership fields without changing Tau's shared event schema."""

import hashlib
import json
from typing import Optional

from pydantic import ConfigDict, Field

from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
)
from tau2.voice.audio_native.openai.events import (
    parse_realtime_event as parse_openai_event,
)

APRIEL_RESPONSE_EVENTS = frozenset(
    {
        "response.created",
        "response.done",
        "response.cancelled",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.output_audio_transcript.delta",
        "response.output_audio_transcript.done",
    }
)
_DIAGNOSTIC_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "history_transcription_unavailable",
        "response_preparation_failed",
        "input_audio_transcription_failed",
        "generation_failed",
        "generation_no_final_answer",
        "tts_generation_failed",
        "tts_audio_empty",
        "tool_dispatch_incomplete",
        "server_error",
        "context_length_exceeded",
        "session_expired",
        "invalid_api_key",
        "invalid_request_error",
        "response_cancel_not_active",
        "item_not_found",
        "unsupported_response_overrides",
        "conversation_already_has_active_response",
        "invalid_value",
        "unknown_event_type",
        "immutable_session_field",
        "input_audio_buffer_active",
    }
)


def _reference(value) -> Optional[str]:
    if isinstance(value, str) and value:
        return hashlib.sha256(
            value.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:16]
    return None


def diagnostic_summary(event, *, response_id=None, status=None) -> str:
    """Bounded metadata only; arbitrary server text remains in the event."""
    error = getattr(event, "error", {})
    kind = event.type
    known_kinds = APRIEL_RESPONSE_EVENTS | {
        "error",
        "conversation.item.input_audio_transcription.failed",
        "conversation.item.input_audio_transcription.completed",
    }
    code = getattr(event, "code", None)
    error_type = error.get("type")
    status = status or getattr(event, "status", None)
    data = {
        "type": kind if kind in known_kinds else "unknown",
        "code": code if code in _DIAGNOSTIC_CODES else "unknown",
        "code_ref": _reference(code) if code not in _DIAGNOSTIC_CODES else None,
        "error_type": error_type
        if isinstance(error_type, str)
        and error_type in {"server_error", "invalid_request_error", "rate_limit_error"}
        else None,
        "event_ref": _reference(event.event_id),
        "request_ref": _reference(error.get("event_id")),
        "response_ref": _reference(response_id or getattr(event, "response_id", None)),
        "item_ref": _reference(getattr(event, "item_id", None)),
        "status": status
        if isinstance(status, str)
        and status in {"completed", "failed", "incomplete", "cancelled", "in_progress"}
        else None,
        "terminal": error.get("terminal")
        if isinstance(error.get("terminal"), bool)
        else None,
    }
    return json.dumps(
        {key: value for key, value in data.items() if value is not None}, sort_keys=True
    )


class AprielRealtimeEvent(BaseRealtimeEvent):
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    content_index: int = 0
    delta: str = Field(default="", exclude=True)
    transcript: str = ""
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""
    status: Optional[str] = None
    audio_start_ms: Optional[int] = None
    audio_end_ms: Optional[int] = None
    output: list[dict] = Field(default_factory=list)
    usage: Optional[dict] = None


class AprielErrorEvent(BaseRealtimeEvent):
    """Keep the server's error classification and correlation fields intact."""

    model_config = ConfigDict(extra="allow")
    error: dict = Field(default_factory=dict)
    code: Optional[str] = None
    message: Optional[str] = None


class AprielCapacityError(RuntimeError):
    def __init__(self, event: AprielErrorEvent):
        self.event = event
        super().__init__(
            "Apriel rate_limit_exceeded: server request capacity unavailable"
        )


def parse_realtime_event(raw: dict) -> BaseRealtimeEvent:
    """Keep the standard parser for unrelated events and own Apriel lifecycle data."""
    if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
        raise ValueError("Realtime frame must be an object with a string type")
    kind = raw["type"]
    if kind in ("error", "conversation.item.input_audio_transcription.failed"):
        error = raw.get("error") or {}
        return AprielErrorEvent.model_validate(
            {**raw, "code": error.get("code"), "message": error.get("message")}
        )
    if not (
        kind in APRIEL_RESPONSE_EVENTS
        or kind
        in {
            "output_audio_buffer.cleared",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        }
    ):
        return parse_openai_event(raw)
    data = dict(raw)
    response = raw.get("response") or {}
    item = raw.get("item") or {}
    data["response_id"] = raw.get("response_id") or response.get("id")
    data["item_id"] = raw.get("item_id") or item.get("id")
    for key in ("status", "output", "usage"):
        if key in response:
            data[key] = response[key]
    return AprielRealtimeEvent.model_validate(data)
