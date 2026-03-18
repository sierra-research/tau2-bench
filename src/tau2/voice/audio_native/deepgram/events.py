"""Pydantic models for Deepgram Voice Agent API events.

Deepgram Voice Agent uses WebSocket events for real-time communication.
Events include:
- Welcome: Initial connection acknowledgment
- SettingsApplied: Configuration confirmed
- UserStartedSpeaking: VAD detected user speech start
- ConversationText: Transcription (user) or response text (agent)
- AgentThinking: Agent is processing
- AgentStartedSpeaking: Agent audio output starting
- AgentAudioDone: Agent finished speaking
- FunctionCallRequest: Agent wants to call a function
- FunctionCalling: Function call in progress
- Error: Error from the API

Reference: Deepgram Voice Agent API documentation
https://developers.deepgram.com/docs/voice-agent
"""

from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class BaseDeepgramEvent(BaseModel):
    """Base class for all Deepgram Voice Agent API events."""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    type: str


# =============================================================================
# Connection Events
# =============================================================================


class DeepgramWelcomeEvent(BaseDeepgramEvent):
    """Initial connection acknowledgment from Deepgram."""

    type: Literal["Welcome"] = "Welcome"
    request_id: Optional[str] = Field(default=None, alias="request_id")


class DeepgramSettingsAppliedEvent(BaseDeepgramEvent):
    """Settings have been applied to the session."""

    type: Literal["SettingsApplied"] = "SettingsApplied"


# =============================================================================
# Speech Detection Events (VAD)
# =============================================================================


class DeepgramUserStartedSpeakingEvent(BaseDeepgramEvent):
    """VAD detected start of user speech."""

    type: Literal["UserStartedSpeaking"] = "UserStartedSpeaking"


class DeepgramUserStoppedSpeakingEvent(BaseDeepgramEvent):
    """VAD detected end of user speech."""

    type: Literal["UserStoppedSpeaking"] = "UserStoppedSpeaking"


# =============================================================================
# Transcription and Text Events
# =============================================================================


class DeepgramConversationTextEvent(BaseDeepgramEvent):
    """Text content in the conversation (transcription or response).

    Attributes:
        role: "user" for transcription, "assistant" for agent response
        content: The text content
        is_final: Whether this is a final transcription (for user) or complete response
    """

    type: Literal["ConversationText"] = "ConversationText"
    role: str = ""  # "user" or "assistant"
    content: str = ""
    is_final: bool = Field(default=False, alias="is_final")


# =============================================================================
# Agent State Events
# =============================================================================


class DeepgramAgentThinkingEvent(BaseDeepgramEvent):
    """Agent is processing/thinking."""

    type: Literal["AgentThinking"] = "AgentThinking"


class DeepgramAgentStartedSpeakingEvent(BaseDeepgramEvent):
    """Agent has started generating audio output."""

    type: Literal["AgentStartedSpeaking"] = "AgentStartedSpeaking"


class DeepgramAgentAudioDoneEvent(BaseDeepgramEvent):
    """Agent has finished generating audio for this turn."""

    type: Literal["AgentAudioDone"] = "AgentAudioDone"


# =============================================================================
# Audio Events
# =============================================================================


class DeepgramAudioEvent(BaseDeepgramEvent):
    """Audio chunk from the agent (TTS output).

    Audio is base64-encoded in the configured format.
    """

    type: Literal["Audio"] = "Audio"
    audio: str = ""  # Base64-encoded audio data


# =============================================================================
# Function Calling Events
# =============================================================================


class DeepgramFunctionCallBeginEvent(BaseDeepgramEvent):
    """Agent is beginning a function call."""

    type: Literal["FunctionCallBegin"] = "FunctionCallBegin"
    function_name: str = Field(default="", alias="function_name")
    call_id: str = Field(default="", alias="call_id")


class DeepgramFunctionCallingEvent(BaseDeepgramEvent):
    """Incremental function call arguments (streaming)."""

    type: Literal["FunctionCalling"] = "FunctionCalling"
    call_id: str = Field(default="", alias="call_id")
    arguments_delta: str = Field(default="", alias="arguments_delta")


class DeepgramFunctionCallInfo(BaseModel):
    """Info about a single function call in a FunctionCallRequest."""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    id: str = ""
    name: str = ""
    arguments: str = ""
    client_side: bool = True


class DeepgramFunctionCallRequestEvent(BaseDeepgramEvent):
    """Function call request from the agent.

    Deepgram sends a 'functions' array with one or more function calls.
    Each function has an id, name, arguments JSON string, and client_side flag.

    Attributes:
        functions: List of function calls to execute
    """

    type: Literal["FunctionCallRequest"] = "FunctionCallRequest"
    functions: List[DeepgramFunctionCallInfo] = Field(default_factory=list)

    @property
    def function_name(self) -> str:
        """Get the first function name for convenience."""
        return self.functions[0].name if self.functions else ""

    @property
    def call_id(self) -> str:
        """Get the first function call ID for convenience."""
        return self.functions[0].id if self.functions else ""

    @property
    def arguments(self) -> str:
        """Get the first function arguments for convenience."""
        return self.functions[0].arguments if self.functions else ""


# =============================================================================
# Error and Utility Events
# =============================================================================


class DeepgramErrorEvent(BaseDeepgramEvent):
    """Error from the Deepgram API."""

    type: Literal["Error"] = "Error"
    error_code: Optional[str] = Field(default=None, alias="code")
    error_message: Optional[str] = Field(default=None, alias="message")
    description: Optional[str] = None


class DeepgramTimeoutEvent(BaseDeepgramEvent):
    """Timeout waiting for events (used internally for tick-based processing)."""

    type: Literal["timeout"] = "timeout"


class DeepgramUnknownEvent(BaseDeepgramEvent):
    """Unknown/unrecognized event type."""

    type: str = "unknown"
    raw: Optional[Dict[str, Any]] = None


# =============================================================================
# Type Aliases
# =============================================================================

DeepgramEvent = Union[
    # Connection events
    DeepgramWelcomeEvent,
    DeepgramSettingsAppliedEvent,
    # Speech detection events
    DeepgramUserStartedSpeakingEvent,
    DeepgramUserStoppedSpeakingEvent,
    # Text events
    DeepgramConversationTextEvent,
    # Agent state events
    DeepgramAgentThinkingEvent,
    DeepgramAgentStartedSpeakingEvent,
    DeepgramAgentAudioDoneEvent,
    # Audio events
    DeepgramAudioEvent,
    # Function calling events
    DeepgramFunctionCallBeginEvent,
    DeepgramFunctionCallingEvent,
    DeepgramFunctionCallRequestEvent,
    # Utility events
    DeepgramErrorEvent,
    DeepgramTimeoutEvent,
    DeepgramUnknownEvent,
]


# =============================================================================
# Event Parsing
# =============================================================================

# Map event type strings to Pydantic model classes
_EVENT_TYPE_MAP: Dict[str, type[BaseDeepgramEvent]] = {
    # Connection events
    "Welcome": DeepgramWelcomeEvent,
    "SettingsApplied": DeepgramSettingsAppliedEvent,
    # Speech detection events
    "UserStartedSpeaking": DeepgramUserStartedSpeakingEvent,
    "UserStoppedSpeaking": DeepgramUserStoppedSpeakingEvent,
    # Text events
    "ConversationText": DeepgramConversationTextEvent,
    # Agent state events
    "AgentThinking": DeepgramAgentThinkingEvent,
    "AgentStartedSpeaking": DeepgramAgentStartedSpeakingEvent,
    "AgentAudioDone": DeepgramAgentAudioDoneEvent,
    # Audio events
    "Audio": DeepgramAudioEvent,
    # Function calling events
    "FunctionCallBegin": DeepgramFunctionCallBeginEvent,
    "FunctionCalling": DeepgramFunctionCallingEvent,
    "FunctionCallRequest": DeepgramFunctionCallRequestEvent,
    # Error
    "Error": DeepgramErrorEvent,
}


def parse_deepgram_event(data: Dict[str, Any]) -> DeepgramEvent:
    """Parse a raw Deepgram event into a typed event.

    Args:
        data: Raw event data from the WebSocket.

    Returns:
        Typed DeepgramEvent instance.
    """
    event_type = data.get("type")

    if event_type is None:
        logger.debug(f"Deepgram event missing 'type': {data}")
        return DeepgramUnknownEvent(type="unknown", raw=data)

    # Log the event (exclude audio content for readability)
    log_data = data.copy()
    if event_type == "Audio" and "audio" in log_data:
        audio_len = len(log_data.get("audio", ""))
        log_data["audio"] = f"<{audio_len} base64 chars>"
    logger.debug(f"Deepgram event: {event_type} - {log_data}")

    # Look up and instantiate the event class
    event_class = _EVENT_TYPE_MAP.get(event_type)

    if event_class:
        try:
            return event_class(**data)
        except Exception as e:
            logger.warning(f"Failed to parse Deepgram event {event_type}: {e}")
            return DeepgramUnknownEvent(type=event_type, raw=data)
    else:
        logger.debug(f"Unknown Deepgram event type: {event_type}")
        return DeepgramUnknownEvent(type=event_type, raw=data)
