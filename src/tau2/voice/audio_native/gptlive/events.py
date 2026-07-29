"""Pydantic models for GPT-Live API (alpha) events.

Based exclusively on the confidential alpha integration guide
("GPT-Live API Alpha"). Key differences from the OpenAI Realtime API:

- Output audio arrives as ``output_audio.delta`` with a server-assigned
  half-open time range (start_ms/end_ms) and NO item_id. There is no
  ``output_audio.done`` event; a gap between ranges is omitted silence.
- Transcripts arrive as complete timed fragments
  (``input_transcript.added`` / ``output_transcript.added``), not deltas.
- ``turn.*`` events are a heuristic projection over transcript fragments.
- Tool calls arrive via Responses delegation: ``delegation.created``
  followed by pass-through ``response.*`` events, of which
  ``response.function_call_arguments.done`` identifies actionable calls.
- There are no VAD events (full-duplex architecture).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BaseGPTLiveEvent(BaseModel):
    """Base class for all GPT-Live API events."""

    type: str


class SessionStartedEvent(BaseGPTLiveEvent):
    """Startup configuration was accepted; the session is ready."""

    type: Literal["session.started"]
    session_id: Optional[str] = None


class SessionUpdatedEvent(BaseGPTLiveEvent):
    type: Literal["session.updated"]


class OutputAudioDeltaEvent(BaseGPTLiveEvent):
    """Base64 24kHz PCM16LE mono audio with a server-timeline range."""

    type: Literal["output_audio.delta"]
    audio: str = Field(
        default="", description="Base64-encoded audio", exclude=True
    )  # Exclude from serialization due to large size
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class InputTranscriptAddedEvent(BaseGPTLiveEvent):
    """One complete timed fragment of the user's speech transcript."""

    type: Literal["input_transcript.added"]
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    item_id: Optional[str] = None
    text: str = ""


class OutputTranscriptAddedEvent(BaseGPTLiveEvent):
    """One complete timed fragment of the agent's speech transcript."""

    type: Literal["output_transcript.added"]
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    item_id: Optional[str] = None
    text: str = ""


class TurnCreatedEvent(BaseGPTLiveEvent):
    """A projected transcript turn began (heuristic grouping, not state)."""

    type: Literal["turn.created"]
    turn_id: Optional[str] = None
    role: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    transcript: str = ""


class TurnDeltaEvent(BaseGPTLiveEvent):
    type: Literal["turn.delta"]
    turn_id: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    delta: str = ""


class TurnDoneEvent(BaseGPTLiveEvent):
    type: Literal["turn.done"]
    turn_id: Optional[str] = None
    role: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    transcript: str = ""


class DelegationCreatedEvent(BaseGPTLiveEvent):
    """The model created a unit of delegated work.

    For Responses delegation, response_id binds this item to the
    response.* lifecycle that follows.
    """

    type: Literal["delegation.created"]
    offset_ms: Optional[int] = None
    item_id: Optional[str] = None
    target: Optional[str] = None  # "client" or "responses"
    response_id: Optional[str] = None
    text: str = ""  # concatenated input_text content


class SessionContextAppendedEvent(BaseGPTLiveEvent):
    type: Literal["session.context.appended"]
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class DelegationContextAppendedEvent(BaseGPTLiveEvent):
    type: Literal["delegation.context.appended"]
    delegation_item_id: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class DelegationFunctionCallOutputCreatedEvent(BaseGPTLiveEvent):
    """A delegated function result was accepted by the server."""

    type: Literal["delegation.function_call_output.created"]
    item_id: Optional[str] = None
    call_id: Optional[str] = None


class ResponseCreatedEvent(BaseGPTLiveEvent):
    """A Responses delegation started (pass-through response.* event)."""

    type: Literal["response.created"]
    response_id: Optional[str] = None
    status: Optional[str] = None


class ResponseOutputItemAddedEvent(BaseGPTLiveEvent):
    """A Responses delegation output item started (pass-through event).

    For function calls (item_type == "function_call"), this is where
    call_id and name are announced. NOTE: contrary to the alpha guide,
    the observed response.function_call_arguments.done events do NOT carry
    call_id/name — adapters must join on item_id using this event.
    """

    type: Literal["response.output_item.added"]
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    output_index: Optional[int] = None
    call_id: Optional[str] = None
    name: Optional[str] = None


class ResponseFunctionCallArgumentsDoneEvent(BaseGPTLiveEvent):
    """A client-actionable function call from a Responses delegation.

    call_id/name are documented in the alpha guide but observed to be
    absent in practice; join on item_id via ResponseOutputItemAddedEvent.
    """

    type: Literal["response.function_call_arguments.done"]
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = "{}"


class ResponseCompletedEvent(BaseGPTLiveEvent):
    """Terminal lifecycle event for a Responses delegation."""

    type: Literal["response.completed"]
    response_id: Optional[str] = None
    status: Optional[str] = None


class SessionUsageUpdatedEvent(BaseGPTLiveEvent):
    type: Literal["session.usage.updated"]
    usage: Optional[dict] = None


class SessionClosedEvent(BaseGPTLiveEvent):
    type: Literal["session.closed"]
    reason: Optional[str] = None
    usage: Optional[dict] = None


class ErrorEvent(BaseGPTLiveEvent):
    type: Literal["error"]
    error_type: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    event_id: Optional[str] = None


class TimeoutEvent(BaseGPTLiveEvent):
    """Synthetic event: no API events arrived during the collection window."""

    type: Literal["timeout"]


class UnknownEvent(BaseGPTLiveEvent):
    """Unrecognized event type (the alpha adds new response.* events)."""

    type: str
    raw: Optional[dict] = None


def _content_text(item: dict) -> str:
    """Concatenate input_text parts from an item's content list."""
    parts = item.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "input_text")


def parse_gptlive_event(raw_data: dict) -> BaseGPTLiveEvent:
    """Parse raw event data into a typed Pydantic event model.

    Unknown event types (including unhandled response.* lifecycle events,
    which the alpha docs say to tolerate) become UnknownEvent.
    """
    event_type = raw_data.get("type", "unknown")

    if event_type == "session.started":
        session = raw_data.get("session") or {}
        return SessionStartedEvent(type=event_type, session_id=session.get("id"))

    if event_type == "session.updated":
        return SessionUpdatedEvent(type=event_type)

    if event_type == "output_audio.delta":
        return OutputAudioDeltaEvent(
            type=event_type,
            audio=raw_data.get("audio", ""),
            start_ms=raw_data.get("start_ms"),
            end_ms=raw_data.get("end_ms"),
        )

    if event_type in ("input_transcript.added", "output_transcript.added"):
        item = raw_data.get("item") or {}
        cls = (
            InputTranscriptAddedEvent
            if event_type == "input_transcript.added"
            else OutputTranscriptAddedEvent
        )
        return cls(
            type=event_type,
            start_ms=raw_data.get("start_ms"),
            end_ms=raw_data.get("end_ms"),
            item_id=item.get("id"),
            text=item.get("text", ""),
        )

    if event_type in ("turn.created", "turn.done"):
        turn = raw_data.get("turn") or {}
        cls = TurnCreatedEvent if event_type == "turn.created" else TurnDoneEvent
        return cls(
            type=event_type,
            turn_id=turn.get("id"),
            role=turn.get("role"),
            start_ms=turn.get("start_ms"),
            end_ms=turn.get("end_ms"),
            transcript=turn.get("transcript", ""),
        )

    if event_type == "turn.delta":
        return TurnDeltaEvent(
            type=event_type,
            turn_id=raw_data.get("turn_id"),
            start_ms=raw_data.get("start_ms"),
            end_ms=raw_data.get("end_ms"),
            delta=raw_data.get("delta", ""),
        )

    if event_type == "delegation.created":
        item = raw_data.get("item") or {}
        return DelegationCreatedEvent(
            type=event_type,
            offset_ms=raw_data.get("offset_ms"),
            item_id=item.get("id"),
            target=item.get("target"),
            response_id=item.get("response_id"),
            text=_content_text(item),
        )

    if event_type == "session.context.appended":
        return SessionContextAppendedEvent(
            type=event_type,
            start_ms=raw_data.get("start_ms"),
            end_ms=raw_data.get("end_ms"),
        )

    if event_type == "delegation.context.appended":
        return DelegationContextAppendedEvent(
            type=event_type,
            delegation_item_id=raw_data.get("delegation_item_id"),
            start_ms=raw_data.get("start_ms"),
            end_ms=raw_data.get("end_ms"),
        )

    if event_type == "delegation.function_call_output.created":
        item = raw_data.get("item") or {}
        return DelegationFunctionCallOutputCreatedEvent(
            type=event_type,
            item_id=item.get("id"),
            call_id=item.get("call_id"),
        )

    if event_type == "response.created":
        response = raw_data.get("response") or {}
        return ResponseCreatedEvent(
            type=event_type,
            response_id=response.get("id"),
            status=response.get("status"),
        )

    if event_type == "response.output_item.added":
        item = raw_data.get("item") or {}
        return ResponseOutputItemAddedEvent(
            type=event_type,
            item_id=item.get("id"),
            item_type=item.get("type"),
            output_index=raw_data.get("output_index"),
            call_id=item.get("call_id"),
            name=item.get("name"),
        )

    if event_type == "response.function_call_arguments.done":
        return ResponseFunctionCallArgumentsDoneEvent(
            type=event_type,
            response_id=raw_data.get("response_id"),
            item_id=raw_data.get("item_id"),
            output_index=raw_data.get("output_index"),
            call_id=raw_data.get("call_id"),
            name=raw_data.get("name"),
            arguments=raw_data.get("arguments", "{}"),
        )

    if event_type == "response.completed":
        response = raw_data.get("response") or {}
        return ResponseCompletedEvent(
            type=event_type,
            response_id=response.get("id"),
            status=response.get("status"),
        )

    if event_type == "session.usage.updated":
        return SessionUsageUpdatedEvent(type=event_type, usage=raw_data.get("usage"))

    if event_type == "session.closed":
        return SessionClosedEvent(
            type=event_type,
            reason=raw_data.get("reason"),
            usage=raw_data.get("usage"),
        )

    if event_type == "error":
        error = raw_data.get("error") or {}
        return ErrorEvent(
            type=event_type,
            error_type=error.get("type"),
            code=error.get("code"),
            message=error.get("message"),
            param=error.get("param"),
            event_id=error.get("event_id"),
        )

    if event_type == "timeout":
        return TimeoutEvent(type=event_type)

    return UnknownEvent(type=event_type, raw=raw_data)
