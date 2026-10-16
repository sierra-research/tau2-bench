"""Typed events for the ElevenLabs Agents WebSocket API.

Reference:
https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket
"""

import base64
from typing import Any, Dict, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class BaseElevenLabsEvent(BaseModel):
    """Base class for normalized ElevenLabs WebSocket events."""

    type: str
    event_id: Optional[int] = None


class ElevenLabsConversationMetadataEvent(BaseElevenLabsEvent):
    """Conversation ID and negotiated input/output audio formats."""

    type: Literal["conversation_initiation_metadata"] = (
        "conversation_initiation_metadata"
    )
    conversation_id: str
    agent_output_audio_format: str
    user_input_audio_format: str


class ElevenLabsAudioEvent(BaseElevenLabsEvent):
    """One decoded agent audio chunk."""

    type: Literal["audio"] = "audio"
    event_id: int
    audio_data: bytes = Field(default=b"", exclude=True)
    alignment: Optional[Dict[str, Any]] = None
    is_final: bool = False


class ElevenLabsAgentResponseEvent(BaseElevenLabsEvent):
    """Complete transcript for an agent response."""

    type: Literal["agent_response"] = "agent_response"
    event_id: int
    transcript: str = ""
    response_id: Optional[str] = None


class ElevenLabsAgentResponseCorrectionEvent(BaseElevenLabsEvent):
    """Corrected transcript after an interrupted response."""

    type: Literal["agent_response_correction"] = "agent_response_correction"
    event_id: int
    original_transcript: str = ""
    corrected_transcript: str = ""
    response_id: Optional[str] = None


class ElevenLabsAgentResponseCompleteEvent(BaseElevenLabsEvent):
    """Marks the end of an agent response."""

    type: Literal["agent_response_complete"] = "agent_response_complete"
    event_id: int


class ElevenLabsUserTranscriptEvent(BaseElevenLabsEvent):
    """Finalized transcription of a user utterance."""

    type: Literal["user_transcript"] = "user_transcript"
    event_id: int
    transcript: str = ""


class ElevenLabsInterruptionEvent(BaseElevenLabsEvent):
    """Indicates that user speech interrupted an agent response."""

    type: Literal["interruption"] = "interruption"
    event_id: int
    reason: Optional[str] = None


class ElevenLabsClientToolCallEvent(BaseElevenLabsEvent):
    """A hosted agent request to execute a client-side tool."""

    type: Literal["client_tool_call"] = "client_tool_call"
    event_id: int
    tool_name: str
    tool_call_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expects_response: bool = True


class ElevenLabsPingEvent(BaseElevenLabsEvent):
    """Keepalive ping that must receive a matching pong."""

    type: Literal["ping"] = "ping"
    event_id: int
    ping_ms: int = 0


class ElevenLabsClientErrorEvent(BaseElevenLabsEvent):
    """Error reported by the ElevenLabs conversation service."""

    type: Literal["client_error"] = "client_error"
    code: Optional[int] = None
    error_name: Optional[str] = None
    message: Optional[str] = None


class ElevenLabsTimeoutEvent(BaseElevenLabsEvent):
    """Synthetic timeout used by tick-based event collection."""

    type: Literal["timeout"] = "timeout"


class ElevenLabsUnknownEvent(BaseElevenLabsEvent):
    """Unrecognized or malformed server event."""

    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


ElevenLabsEvent = Union[
    ElevenLabsConversationMetadataEvent,
    ElevenLabsAudioEvent,
    ElevenLabsAgentResponseEvent,
    ElevenLabsAgentResponseCorrectionEvent,
    ElevenLabsAgentResponseCompleteEvent,
    ElevenLabsUserTranscriptEvent,
    ElevenLabsInterruptionEvent,
    ElevenLabsClientToolCallEvent,
    ElevenLabsPingEvent,
    ElevenLabsClientErrorEvent,
    ElevenLabsTimeoutEvent,
    ElevenLabsUnknownEvent,
]


def parse_elevenlabs_event(data: Dict[str, Any]) -> ElevenLabsEvent:
    """Normalize one raw ElevenLabs WebSocket message."""
    event_type = data.get("type", "unknown")

    try:
        if event_type == "conversation_initiation_metadata":
            payload = data["conversation_initiation_metadata_event"]
            return ElevenLabsConversationMetadataEvent(
                conversation_id=payload["conversation_id"],
                agent_output_audio_format=payload["agent_output_audio_format"],
                user_input_audio_format=payload["user_input_audio_format"],
            )

        if event_type == "audio":
            payload = data["audio_event"]
            return ElevenLabsAudioEvent(
                event_id=payload["event_id"],
                audio_data=base64.b64decode(payload.get("audio_base_64", "")),
                alignment=payload.get("alignment"),
                is_final=payload.get("is_final", False),
            )

        if event_type == "agent_response":
            payload = data["agent_response_event"]
            return ElevenLabsAgentResponseEvent(
                event_id=payload["event_id"],
                transcript=payload.get("agent_response", ""),
                response_id=payload.get("response_id"),
            )

        if event_type == "agent_response_correction":
            payload = data["agent_response_correction_event"]
            return ElevenLabsAgentResponseCorrectionEvent(
                event_id=payload["event_id"],
                original_transcript=payload.get("original_agent_response", ""),
                corrected_transcript=payload.get("corrected_agent_response", ""),
                response_id=payload.get("response_id"),
            )

        if event_type == "agent_response_complete":
            payload = data["agent_response_complete_event"]
            return ElevenLabsAgentResponseCompleteEvent(event_id=payload["event_id"])

        if event_type == "user_transcript":
            payload = data["user_transcription_event"]
            return ElevenLabsUserTranscriptEvent(
                event_id=payload["event_id"],
                transcript=payload.get("user_transcript", ""),
            )

        if event_type == "interruption":
            payload = data["interruption_event"]
            return ElevenLabsInterruptionEvent(
                event_id=payload["event_id"],
                reason=payload.get("reason"),
            )

        if event_type == "client_tool_call":
            payload = data["client_tool_call"]
            return ElevenLabsClientToolCallEvent(
                event_id=payload["event_id"],
                tool_name=payload["tool_name"],
                tool_call_id=payload["tool_call_id"],
                parameters=payload.get("parameters", {}),
                expects_response=payload.get("expects_response", True),
            )

        if event_type == "ping":
            payload = data["ping_event"]
            return ElevenLabsPingEvent(
                event_id=payload["event_id"],
                ping_ms=payload.get("ping_ms") or 0,
            )

        if event_type == "client_error":
            payload = data.get("error_event", {})
            return ElevenLabsClientErrorEvent(
                code=payload.get("code"),
                error_name=payload.get("error_name"),
                message=payload.get("message"),
            )
    except (KeyError, TypeError, ValueError) as error:
        logger.warning(f"Failed to parse ElevenLabs event {event_type}: {error}")
        return ElevenLabsUnknownEvent(type=event_type, raw=data)

    logger.debug(f"Unknown ElevenLabs event type: {event_type}")
    return ElevenLabsUnknownEvent(type=event_type, raw=data)
