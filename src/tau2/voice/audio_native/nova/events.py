"""Pydantic models for Amazon Nova Sonic API events.

Nova Sonic uses a different event protocol than OpenAI/xAI.
Events are wrapped in a specific structure with event types like:
- sessionStart, sessionEnd
- promptStart, contentStart, textInput, audioInput, toolResult, contentEnd
- contentStart (for output), textOutput, audioOutput, toolUse, contentEnd
- completionStart, completionEnd

Reference: AWS Bedrock Nova Sonic documentation
"""

import json
from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, model_validator


class BaseNovaEvent(BaseModel):
    """Base class for all Nova Sonic API events."""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    event_type: str


# =============================================================================
# Session Events
# =============================================================================


class NovaSessionStartEvent(BaseNovaEvent):
    """Session has started successfully."""

    event_type: Literal["sessionStart"] = "sessionStart"


class NovaSessionEndEvent(BaseNovaEvent):
    """Session has ended."""

    event_type: Literal["sessionEnd"] = "sessionEnd"


# =============================================================================
# Content Start/End Events
# =============================================================================


class NovaContentStartEvent(BaseNovaEvent):
    """Content block is starting (input or output).

    Nova Sonic uses generationStage to indicate whether content is speculative
    (may be revised) or final (committed). We only process FINAL content.
    """

    event_type: Literal["contentStart"] = "contentStart"
    role: Optional[str] = None  # "USER" or "ASSISTANT"
    content_id: Optional[str] = Field(default=None, alias="contentId")
    type: Optional[str] = (
        None  # "AUDIO", "TEXT", "TOOL_USE", "TOOL_RESULT" (maps from API 'type' field)
    )
    tool_use_id: Optional[str] = Field(default=None, alias="toolUseId")
    tool_name: Optional[str] = Field(default=None, alias="toolName")
    # generationStage: "SPECULATIVE" or "FINAL" - parsed from additionalModelFields
    generation_stage: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def parse_generation_stage_from_additional_fields(
        cls, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse generationStage from additionalModelFields JSON string."""
        if isinstance(data, dict):
            additional_fields = data.get("additionalModelFields")
            if additional_fields and isinstance(additional_fields, str):
                try:
                    parsed = json.loads(additional_fields)
                    if "generationStage" in parsed:
                        data["generation_stage"] = parsed["generationStage"]
                except json.JSONDecodeError:
                    pass
        return data


class NovaContentEndEvent(BaseNovaEvent):
    """Content block has ended."""

    event_type: Literal["contentEnd"] = "contentEnd"
    content_id: Optional[str] = Field(default=None, alias="contentId")
    stop_reason: Optional[str] = Field(default=None, alias="stopReason")


# =============================================================================
# Audio Events
# =============================================================================


class NovaAudioOutputEvent(BaseNovaEvent):
    """Audio chunk from the model.

    Audio is base64-encoded LPCM 16kHz mono.
    """

    event_type: Literal["audioOutput"] = "audioOutput"
    content: str = Field(
        default="",
        description="Base64-encoded audio data (LPCM 16kHz)",
    )
    content_id: Optional[str] = Field(default=None, alias="contentId")


class NovaAudioInputEvent(BaseNovaEvent):
    """Audio input acknowledgment from server."""

    event_type: Literal["audioInput"] = "audioInput"
    content_id: Optional[str] = Field(default=None, alias="contentId")


# =============================================================================
# Text Events
# =============================================================================


class NovaTextOutputEvent(BaseNovaEvent):
    """Text output (transcript) from the model."""

    event_type: Literal["textOutput"] = "textOutput"
    content: str = ""  # The API uses 'content', not 'text'
    content_id: Optional[str] = Field(default=None, alias="contentId")
    role: Optional[str] = None  # "USER" (ASR) or "ASSISTANT" (response)


class NovaTextInputEvent(BaseNovaEvent):
    """Text input acknowledgment."""

    event_type: Literal["textInput"] = "textInput"
    content_id: Optional[str] = Field(default=None, alias="contentId")


# =============================================================================
# Tool Events
# =============================================================================


class NovaToolUseEvent(BaseNovaEvent):
    """Model is requesting to use a tool."""

    event_type: Literal["toolUse"] = "toolUse"
    tool_use_id: str = Field(default="", alias="toolUseId")
    tool_name: str = Field(default="", alias="toolName")
    content: str = ""  # JSON string of tool arguments
    content_id: Optional[str] = Field(default=None, alias="contentId")


class NovaToolResultEvent(BaseNovaEvent):
    """Tool result acknowledgment."""

    event_type: Literal["toolResult"] = "toolResult"
    tool_use_id: Optional[str] = Field(default=None, alias="toolUseId")
    content_id: Optional[str] = Field(default=None, alias="contentId")


# =============================================================================
# Turn/Completion Events
# =============================================================================


class NovaCompletionStartEvent(BaseNovaEvent):
    """Model is starting a completion/response."""

    event_type: Literal["completionStart"] = "completionStart"


class NovaCompletionEndEvent(BaseNovaEvent):
    """Model has finished the completion/response."""

    event_type: Literal["completionEnd"] = "completionEnd"
    stop_reason: Optional[str] = Field(
        default=None, alias="stopReason"
    )  # "END_TURN", "TOOL_USE", "INTERRUPTED"


# =============================================================================
# VAD/Barge-in Events
# =============================================================================


class NovaSpeechStartedEvent(BaseNovaEvent):
    """Server detected start of user speech (barge-in)."""

    event_type: Literal["speechStarted"] = "speechStarted"


class NovaSpeechEndedEvent(BaseNovaEvent):
    """Server detected end of user speech."""

    event_type: Literal["speechEnded"] = "speechEnded"


class NovaBargeInEvent(BaseNovaEvent):
    """User interrupted the model (barge-in occurred)."""

    event_type: Literal["bargeIn"] = "bargeIn"


# =============================================================================
# Error and Utility Events
# =============================================================================


class NovaErrorEvent(BaseNovaEvent):
    """Error from the Nova API."""

    event_type: Literal["error"] = "error"
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class NovaTimeoutEvent(BaseNovaEvent):
    """Timeout waiting for events (used internally)."""

    event_type: Literal["timeout"] = "timeout"


class NovaUnknownEvent(BaseNovaEvent):
    """Unknown/unrecognized event type."""

    event_type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


# =============================================================================
# Usage and Metadata Events
# =============================================================================


class NovaUsageEvent(BaseNovaEvent):
    """Token usage tracking event (sent frequently during processing)."""

    event_type: Literal["usageEvent"] = "usageEvent"
    completion_id: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


class NovaMetadataEvent(BaseNovaEvent):
    """Metadata about the response (token usage, etc.)."""

    event_type: Literal["metadata"] = "metadata"
    usage: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None


# =============================================================================
# Type Aliases
# =============================================================================

NovaSonicEvent = Union[
    # Session events
    NovaSessionStartEvent,
    NovaSessionEndEvent,
    # Content events
    NovaContentStartEvent,
    NovaContentEndEvent,
    # Audio events
    NovaAudioOutputEvent,
    NovaAudioInputEvent,
    # Text events
    NovaTextOutputEvent,
    NovaTextInputEvent,
    # Tool events
    NovaToolUseEvent,
    NovaToolResultEvent,
    # Completion events
    NovaCompletionStartEvent,
    NovaCompletionEndEvent,
    # VAD events
    NovaSpeechStartedEvent,
    NovaSpeechEndedEvent,
    NovaBargeInEvent,
    # Utility events
    NovaErrorEvent,
    NovaTimeoutEvent,
    NovaUnknownEvent,
    NovaUsageEvent,
    NovaMetadataEvent,
]


# =============================================================================
# Event Parsing
# =============================================================================

# Map event type strings to Pydantic model classes
_EVENT_TYPE_MAP: Dict[str, type[BaseNovaEvent]] = {
    # Session events
    "sessionStart": NovaSessionStartEvent,
    "sessionEnd": NovaSessionEndEvent,
    # Content events
    "contentStart": NovaContentStartEvent,
    "contentEnd": NovaContentEndEvent,
    # Audio events
    "audioOutput": NovaAudioOutputEvent,
    "audioInput": NovaAudioInputEvent,
    # Text events
    "textOutput": NovaTextOutputEvent,
    "textInput": NovaTextInputEvent,
    # Tool events
    "toolUse": NovaToolUseEvent,
    "toolResult": NovaToolResultEvent,
    # Completion events
    "completionStart": NovaCompletionStartEvent,
    "completionEnd": NovaCompletionEndEvent,
    # VAD events
    "speechStarted": NovaSpeechStartedEvent,
    "speechEnded": NovaSpeechEndedEvent,
    "bargeIn": NovaBargeInEvent,
    # Utility
    "error": NovaErrorEvent,
    "usageEvent": NovaUsageEvent,
    "metadata": NovaMetadataEvent,
}


def parse_nova_event(data: Dict[str, Any]) -> NovaSonicEvent:
    """Parse a raw Nova Sonic event into a typed event.

    Nova Sonic events have a nested structure. This function extracts
    the event type and data from the wrapper.

    Args:
        data: Raw event data from the stream.

    Returns:
        Typed NovaSonicEvent instance.
    """
    # Nova events come in a nested structure
    # Extract the actual event type and data
    event_type = None
    event_data = {}

    # Nova Sonic events use the structure: {"event": {"eventType": {...}}}
    if "event" in data:
        event_wrapper = data["event"]
        for key in event_wrapper:
            if key in _EVENT_TYPE_MAP:
                event_type = key
                event_data = event_wrapper[key] or {}
                break

    if event_type is None:
        logger.debug(f"Unknown Nova event structure: {data}")
        return NovaUnknownEvent(event_type="unknown", raw=data)

    # Log the event
    log_data = event_data.copy() if isinstance(event_data, dict) else {}
    if event_type == "audioOutput" and "content" in log_data:
        content = log_data.get("content", "")
        log_data["content"] = f"<{len(content)} base64 chars>"
    logger.debug(f"Nova event: {event_type} - {log_data}")

    # Look up and instantiate the event class
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class:
        try:
            return event_class(event_type=event_type, **event_data)
        except Exception as e:
            logger.warning(f"Failed to parse Nova event {event_type}: {e}")
            return NovaUnknownEvent(event_type=event_type, raw=data)
    else:
        logger.debug(f"Unknown Nova event type: {event_type}")
        return NovaUnknownEvent(event_type=event_type, raw=data)
