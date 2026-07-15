"""Pydantic models for AAI voice agent events.

AAI server sends camelCase JSON messages that are parsed into typed event models.
All events inherit from BaseAAIEvent with configurable extra field handling.
"""

from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class BaseAAIEvent(BaseModel):
    """Base class for all AAI events."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str


# =============================================================================
# Configuration Events
# =============================================================================


class AAIConfigEvent(BaseAAIEvent):
    """Server configuration acknowledgment."""

    type: Literal["config"] = "config"


# =============================================================================
# Speech Detection Events
# =============================================================================


class AAISpeechStartedEvent(BaseAAIEvent):
    """User speech detection started."""

    type: Literal["speech_started"] = "speech_started"


class AAISpeechStoppedEvent(BaseAAIEvent):
    """User speech detection stopped."""

    type: Literal["speech_stopped"] = "speech_stopped"


# =============================================================================
# Transcript Events
# =============================================================================


class AAIUserTranscriptEvent(BaseAAIEvent):
    """User speech transcript from the voice agent."""

    type: Literal["user_transcript"] = "user_transcript"
    text: str
    turn_order: Optional[int] = Field(default=None, alias="turnOrder")


class AAIAgentTranscriptEvent(BaseAAIEvent):
    """Agent response transcript."""

    type: Literal["agent_transcript"] = "agent_transcript"
    text: str


# =============================================================================
# Tool Events
# =============================================================================


class AAIToolCallEvent(BaseAAIEvent):
    """Agent is requesting to call a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    args: dict = Field(default_factory=dict)


class AAIToolCallDoneEvent(BaseAAIEvent):
    """Tool execution has completed."""

    type: Literal["tool_call_done"] = "tool_call_done"
    tool_call_id: str = Field(alias="toolCallId")
    result: str = ""


# =============================================================================
# Response Events
# =============================================================================


class AAIReplyDoneEvent(BaseAAIEvent):
    """Agent response is complete."""

    type: Literal["reply_done"] = "reply_done"


class AAIAudioDoneEvent(BaseAAIEvent):
    """Audio output is complete."""

    type: Literal["audio_done"] = "audio_done"


# =============================================================================
# Session Control Events
# =============================================================================


class AAICancelledEvent(BaseAAIEvent):
    """Session was cancelled."""

    type: Literal["cancelled"] = "cancelled"


class AAIResetEvent(BaseAAIEvent):
    """Session was reset."""

    type: Literal["reset"] = "reset"


class AAIIdleTimeoutEvent(BaseAAIEvent):
    """Session idle timeout occurred."""

    type: Literal["idle_timeout"] = "idle_timeout"


# =============================================================================
# Error and Custom Events
# =============================================================================


class AAIErrorEvent(BaseAAIEvent):
    """Error from the AAI server."""

    type: Literal["error"] = "error"
    code: Optional[str] = None
    message: Optional[str] = None


class AAICustomEvent(BaseAAIEvent):
    """Custom application-defined event."""

    type: Literal["custom_event"] = "custom_event"
    event: str
    data: Optional[Any] = None


# =============================================================================
# Internal/Helper Events
# =============================================================================


class AAITimeoutEvent(BaseAAIEvent):
    """Timeout waiting for events (used internally, not from server)."""

    type: Literal["timeout"] = "timeout"


class AAIUnknownEvent(BaseAAIEvent):
    """Unknown or unrecognized event type."""

    type: str
    raw: Optional[dict] = None


class AAIAudioChunkEvent(BaseAAIEvent):
    """Audio frame chunk (constructed directly by provider, not parsed).

    pcm16 field is excluded from serialization due to binary size.
    """

    type: Literal["audio_chunk"] = "audio_chunk"
    pcm16: bytes = Field(default=b"", exclude=True)


# =============================================================================
# Event Type Mapping
# =============================================================================

_EVENT_TYPE_MAP: dict[str, type[BaseAAIEvent]] = {
    "config": AAIConfigEvent,
    "speech_started": AAISpeechStartedEvent,
    "speech_stopped": AAISpeechStoppedEvent,
    "user_transcript": AAIUserTranscriptEvent,
    "agent_transcript": AAIAgentTranscriptEvent,
    "tool_call": AAIToolCallEvent,
    "tool_call_done": AAIToolCallDoneEvent,
    "reply_done": AAIReplyDoneEvent,
    "audio_done": AAIAudioDoneEvent,
    "cancelled": AAICancelledEvent,
    "reset": AAIResetEvent,
    "idle_timeout": AAIIdleTimeoutEvent,
    "error": AAIErrorEvent,
    "custom_event": AAICustomEvent,
    "timeout": AAITimeoutEvent,
}


# =============================================================================
# Event Parsing
# =============================================================================


def parse_aai_event(data: dict) -> BaseAAIEvent:
    """Parse raw AAI event data into a typed event model.

    Args:
        data: Raw event dictionary from AAI server.

    Returns:
        Typed event instance. Unknown types return AAIUnknownEvent.
        Parse failures return AAIUnknownEvent with warning logged.
    """
    event_type = data.get("type", "unknown")
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class is None:
        return AAIUnknownEvent(type=event_type, raw=data)

    # Log event, redacting large binary/base64 fields
    log_data = _prepare_log_data(data)
    logger.debug(f"AAI event: {event_type} - {log_data}")

    try:
        return event_class.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse AAI event {event_type}: {e}")
        return AAIUnknownEvent(type=event_type, raw=data)


def _prepare_log_data(data: dict) -> dict:
    """Prepare event data for logging, eliding large fields like base64.

    Args:
        data: Raw event data.

    Returns:
        Copy of data with large fields redacted for logging.
    """
    log_data = data.copy()
    # Redact any large base64 or binary fields if present
    for key in list(log_data.keys()):
        if key in ("audio", "data", "content") and isinstance(log_data[key], str):
            if len(log_data[key]) > 100:
                log_data[key] = f"<{len(log_data[key])} chars>"
    return log_data
