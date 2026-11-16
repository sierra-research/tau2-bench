"""Hydra wire events, translated into the typed events the smallest adapter uses.

Hydra (Smallest AI's speech-to-speech server) speaks an OpenAI-Realtime-style
JSON protocol on ``/v1/hydra``. Every audio and caption delta carries the
``response_id`` of the reply it belongs to, so the adapter keys utterances by
it. Only the fields the adapter needs are kept; the rest of the protocol
(conversation items, turn-detection telemetry) is passed through as
``SmallestOther`` for logging.
"""

import base64
import binascii
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

SESSION_CREATED = "session.created"
SESSION_CONFIGURED = "session.configured"
SESSION_UPDATED = "session.updated"
RESPONSE_CREATED = "response.created"
RESPONSE_DONE = "response.done"
AUDIO_DELTA = "response.output_audio.delta"
AUDIO_DONE = "response.output_audio.done"
# The caption of each clause, sent just before the clause's audio. Text mode
# uses output_text; both are treated as the reply's transcript.
TRANSCRIPT_DELTAS = (
    "response.output_audio_transcript.delta",
    "response.output_text.delta",
)
TOOL_CALL = "response.function_call_arguments.done"
TOOL_CALL_DELTA = "response.function_call_arguments.delta"
SPEECH_STARTED = "input_audio_buffer.speech_started"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
# Sent the moment Hydra commits a barge-in, ahead of any stale audio.
BARGE_IN = "response.barge_in"
ERROR = "error"


class SmallestEvent(BaseModel):
    """One Hydra event. ``type`` is the wire type (tick activity reads it)."""

    model_config = ConfigDict(frozen=True)

    type: str
    response_id: Optional[str] = None


class SmallestSession(SmallestEvent):
    """session.created / session.configured / session.updated."""

    session_id: Optional[str] = None
    output_sample_rate: Optional[int] = None


class SmallestResponseCreated(SmallestEvent):
    sample_rate: Optional[int] = None


class SmallestResponseDone(SmallestEvent):
    """End of a reply: completed, cancelled, incomplete or failed."""

    status: Optional[str] = None


class SmallestAudioDelta(SmallestEvent):
    """Agent speech: PCM16 mono at ``sample_rate``."""

    audio: bytes = b""
    sample_rate: int


class SmallestAudioDone(SmallestEvent):
    pass


class SmallestTranscriptDelta(SmallestEvent):
    text: str = ""


class SmallestToolCall(SmallestEvent):
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Any = None


class SmallestSpeech(SmallestEvent):
    """Caller speech started or stopped, or a barge-in Hydra committed."""


class SmallestError(SmallestEvent):
    code: Optional[str] = None
    message: Optional[str] = None


class SmallestOther(SmallestEvent):
    raw: Optional[dict] = None


def _rate(value: Any) -> Optional[int]:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


def _decode_audio(delta: Any) -> bytes:
    try:
        return base64.b64decode(delta or "")
    except (binascii.Error, TypeError, ValueError):
        logger.warning("Smallest: dropping an audio delta that is not valid base64")
        return b""


def parse_hydra_event(data: dict, sample_rate: int) -> Optional[SmallestEvent]:
    """Translate one decoded Hydra JSON message.

    ``sample_rate`` is the output rate Hydra announced; audio deltas carry it.
    Returns None for streaming tool-argument chunks (the call is read whole
    from ``response.function_call_arguments.done``).
    """
    kind = data.get("type") or "unknown"
    response = data.get("response")
    response = response if isinstance(response, dict) else {}

    if kind in (SESSION_CREATED, SESSION_CONFIGURED, SESSION_UPDATED):
        session = data.get("session")
        session = session if isinstance(session, dict) else {}
        return SmallestSession(
            type=kind,
            session_id=data.get("session_id"),
            output_sample_rate=_rate(session.get("output_audio_sample_rate")),
        )
    if kind == RESPONSE_CREATED:
        return SmallestResponseCreated(
            type=kind,
            response_id=response.get("id"),
            sample_rate=_rate(response.get("sample_rate")),
        )
    if kind == RESPONSE_DONE:
        return SmallestResponseDone(
            type=kind,
            response_id=response.get("id"),
            status=response.get("status"),
        )
    if kind == AUDIO_DELTA:
        return SmallestAudioDelta(
            type=kind,
            response_id=data.get("response_id"),
            audio=_decode_audio(data.get("delta")),
            sample_rate=sample_rate,
        )
    if kind == AUDIO_DONE:
        return SmallestAudioDone(type=kind, response_id=data.get("response_id"))
    if kind in TRANSCRIPT_DELTAS:
        return SmallestTranscriptDelta(
            type=kind,
            response_id=data.get("response_id"),
            text=data.get("delta") or "",
        )
    if kind == TOOL_CALL:
        return SmallestToolCall(
            type=kind,
            response_id=data.get("response_id"),
            call_id=data.get("call_id"),
            name=data.get("name"),
            arguments=data.get("arguments"),
        )
    if kind == TOOL_CALL_DELTA:
        return None
    if kind in (SPEECH_STARTED, SPEECH_STOPPED, BARGE_IN):
        return SmallestSpeech(type=kind)
    if kind == ERROR:
        error = data.get("error")
        error = error if isinstance(error, dict) else {}
        return SmallestError(
            type=kind, code=error.get("code"), message=error.get("message")
        )
    return SmallestOther(type=kind, raw=data)
