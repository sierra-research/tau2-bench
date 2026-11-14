"""Pipecat RTVI session with a hosted Avartha agent.

``AvarthaPipecatProvider`` holds everything that does not depend on how the
call is carried: the boot verdict, the tool exchange, output acknowledgement
and routing of RTVI messages into events. The call itself is a WebRTC
connection, in ``webrtc.py``, kept apart because aiortc ships in the optional
live extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from loguru import logger

from tau2.utils.errors import NonRetryableError
from tau2.voice.audio_native.avartha_pipecat.audio_timing import (
    AudioClockBinding,
    WordEndTime,
    parse_audio_timeline,
    parse_word_end,
)
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig

# Wire identifier of the tool-exchange protocol the agent must acknowledge.
HARNESS_TOOLS_PROTOCOL = "avartha-harness-tools/1"
# Pipecat RTVI message label and the protocol version tau2 announces.
RTVI_MESSAGE_LABEL = "rtvi-ai"
RTVI_PROTOCOL_VERSION = "2.0.0"


@dataclass
class AgentAudioEvent:
    """PCM16 audio emitted by Avartha."""

    # Read by TickResult.has_provider_activity; without it every tick looks idle.
    type: ClassVar[str] = "agent_audio"

    audio: bytes
    sample_rate: int
    # Where this audio sits on the RTP clock (48 kHz, unwrapped), when the
    # carrier keeps its decoded frames' timestamps (``audio_timing.py``):
    # its first sample, and the earliest position from which media was decoded
    # without a gap up to it. Between the previous audio and this one, that
    # media was silence the receiver dropped; before it, media was lost.
    rtp_start: Optional[int] = None
    rtp_decoded_from: Optional[int] = None


@dataclass
class ToolCallEvent:
    """Domain tool invocation requested by Avartha."""

    type: ClassVar[str] = "tool_call"

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TranscriptEvent:
    """One sentence of agent text, sent as soon as the LLM writes it.

    RTVI ``bot-transcription``. It arrives ahead of the audio that speaks it,
    often by the whole utterance, so it must not be shown on arrival.
    """

    type: ClassVar[str] = "transcript"

    text: str


@dataclass
class AgentLLMTextEvent:
    """RTVI ``bot-llm-text``: one LLM token, the input bot-transcription
    sentences are built from."""

    type: ClassVar[str] = "agent_llm_text"

    text: str


@dataclass
class AgentWordEvent:
    """One word of agent speech, sent by the TTS clock as it is played.

    RTVI ``bot-tts-text``. Alignment splits and drops some words, so this is a
    progress signal for ``TranscriptEvent`` text, not the text itself.
    """

    type: ClassVar[str] = "agent_word"

    text: str
    # Where the word's sound ends, when Avartha times it (``audio_time``).
    end: Optional[WordEndTime] = None


@dataclass
class AgentAudioTimelineEvent:
    """Avartha ``audio_timeline``: binds word-end times to the audio's RTP clock."""

    type: ClassVar[str] = "agent_audio_timeline"

    binding: AudioClockBinding


@dataclass
class AgentLLMStartedEvent:
    """RTVI ``bot-llm-started``: the agent began generating a new response.

    After an interruption, text from before this belongs to the response
    that was cut off.
    """

    type: ClassVar[str] = "agent_llm_started"


@dataclass
class AgentStartedSpeakingEvent:
    """RTVI ``bot-started-speaking``: audio of a new utterance follows.

    After an interruption, audio from before this is the cut-off response
    still in flight.
    """

    type: ClassVar[str] = "agent_started_speaking"


@dataclass
class AgentPlaybackEvent:
    """What one agent utterance actually played, reported when it ends.

    Avartha ``assistant_playback`` server-message; ``cut`` when the
    caller barged in and ``text`` holds only the played part.
    """

    type: ClassVar[str] = "agent_playback"

    text: str
    cut: bool


@dataclass
class SpeechEvent:
    """Normalized speech or playback-interruption boundary from RTVI."""

    type: ClassVar[str] = "speech"

    kind: str


@dataclass
class ProviderErrorEvent:
    """Unexpected reader failure surfaced to the tick loop."""

    type: ClassVar[str] = "provider_error"

    message: str


class AvarthaBootError(RuntimeError):
    """The Avartha session failed to become ready."""


class AvarthaProtocolError(NonRetryableError):
    """The Avartha agent does not support this harness tool protocol."""


class AvarthaPipecatProvider:
    """Exchange transcripts, tool calls and boot state with Avartha.

    Subclasses carry the call: ``connect``, ``send_audio``, ``disconnect`` and
    ``_send_message``.
    """

    def __init__(self, config: AvarthaPipecatConfig):
        self.config = config

        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._reader: Optional[asyncio.Task[None]] = None
        self._registered = asyncio.Event()
        self._registration_error: Optional[RuntimeError] = None
        self._session_error: Optional[RuntimeError] = None
        self._boot_settled = asyncio.Event()
        self._message_id = 0
        self._delivered_outputs: set[str] = set()

        self.session_id: Optional[str] = None
        self.boot_ready: Optional[bool] = None
        self.boot_error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Whether the provider currently owns a call."""

        raise NotImplementedError

    async def connect(self, system_prompt: str = "") -> None:
        """Start a call, announce RTVI readiness and wait for a healthy boot."""

        raise NotImplementedError

    async def register_tools(
        self,
        schemas: list[dict[str, Any]],
    ) -> None:
        """Register a self-contained tools-only MCP surface."""

        if self._session_error:
            raise self._session_error
        self._registered.clear()
        self._registration_error = None
        await self._client_message(
            "register_tools",
            {"protocol": HARNESS_TOOLS_PROTOCOL, "tools": schemas},
        )
        try:
            await asyncio.wait_for(
                self._registered.wait(),
                timeout=self.config.tool_registration_timeout_seconds,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "Avartha did not acknowledge register_tools within "
                f"{self.config.tool_registration_timeout_seconds:g}s"
            ) from exc
        if self._session_error:
            raise self._session_error
        if self._registration_error:
            raise self._registration_error
        logger.info(f"Avartha registered {len(schemas)} tool(s)")

    async def send_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Send one mono PCM16 chunk of caller audio."""

        raise NotImplementedError

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Return a tau2 tool result to Avartha."""

        del request_response
        await self._client_message(
            "tool_result",
            {
                "protocol": HARNESS_TOOLS_PROTOCOL,
                "id": call_id,
                "result": {
                    "content": [{"type": "text", "text": result}],
                    "isError": is_error,
                },
            },
        )

    async def receive_events_for_duration(self, duration_seconds: float) -> list[Any]:
        """Drain provider events until the allotted tick time expires."""

        events: list[Any] = []
        end = asyncio.get_running_loop().time() + duration_seconds
        while True:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            events.append(event)
        return events

    def drain_pending_events(self) -> list[Any]:
        """Take events already queued, in order, without extending the tick."""

        events: list[Any] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return events

    async def disconnect(self) -> None:
        """Hang up and release the call."""

        raise NotImplementedError

    def _reset_session_state(self) -> None:
        """Clear state owned by one call."""

        self._reader = None
        self._queue = asyncio.Queue()
        self._registered = asyncio.Event()
        self._registration_error = None
        self._session_error = None
        self._boot_settled = asyncio.Event()
        self._message_id = 0
        self._delivered_outputs = set()
        self.session_id = None
        self.boot_ready = None
        self.boot_error = None

    def reset_for_retry(self) -> None:
        """Clear per-session state before opening a replacement session."""

        self._reset_session_state()

    async def _await_boot(self) -> None:
        try:
            await asyncio.wait_for(
                self._boot_settled.wait(),
                timeout=self.config.boot_timeout_seconds,
            )
        except TimeoutError as exc:
            raise AvarthaBootError(
                "Avartha session_boot never settled within "
                f"{self.config.boot_timeout_seconds:g}s"
            ) from exc
        if self._session_error:
            raise self._session_error
        if not self.boot_ready:
            raise AvarthaBootError(
                self.boot_error or "Avartha session boot reported not ready"
            )
        logger.info("Avartha session boot is ready")

    def _note_boot(self, data: dict[str, Any]) -> None:
        if data.get("ready"):
            self.boot_ready = True
            self._boot_settled.set()
        elif data.get("error"):
            self.boot_ready = False
            self.boot_error = str(data["error"])
            self._boot_settled.set()

    async def _send_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    async def _client_message(self, message_type: str, data: dict[str, Any]) -> None:
        self._message_id += 1
        await self._send_message(
            {
                "label": RTVI_MESSAGE_LABEL,
                "type": "client-message",
                "id": f"tau2-{self._message_id}",
                "data": {"t": message_type, "d": data},
            }
        )

    async def _accept_output(self, message: Any) -> bool:
        """Acknowledge a reliable caller output and route each one only once.

        Avartha resends every server-message carrying an
        ``output_event_id`` once a second until the client answers with
        ``output_ack``, as its browser client does. Unacknowledged, each played
        caption came back about a hundred times a call, and every copy reached
        the adapter as newly aired speech: the simulated caller read old lines
        mid-utterance and talked over the agent. An id's payload never changes,
        so a repeat is dropped; it is acknowledged again in case the first ack
        was lost.
        """

        if not isinstance(message, dict) or message.get("type") != "server-message":
            return True
        data = message.get("data")
        event_id = data.get("output_event_id") if isinstance(data, dict) else None
        if not isinstance(event_id, str) or not event_id:
            return True
        # The browser acknowledges a display artifact only once it is on
        # screen, and the agent is told the caller saw it. The simulated
        # caller never sees one.
        if data.get("type") != "anchor_output":
            try:
                await self._client_message("output_ack", {"event_id": event_id})
            except Exception as exc:  # it is resent, and the repeat dropped
                logger.debug(f"Avartha output_ack {event_id} failed: {exc!r}")
        if event_id in self._delivered_outputs:
            return False
        self._delivered_outputs.add(event_id)
        return True

    def _fail_session(self, message: str) -> None:
        """Preserve the first failure and wake every pending handshake."""

        if self._session_error is not None:
            return
        self._session_error = RuntimeError(message)
        self._boot_settled.set()
        self._registered.set()
        self._queue.put_nowait(ProviderErrorEvent(message))

    def _route_message(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("label") != RTVI_MESSAGE_LABEL:
            return

        message_type = message.get("type")
        data = message.get("data")
        if message_type == "error" and isinstance(data, dict):
            detail = str(data.get("error") or "unknown Avartha error")
            if data.get("fatal"):
                self._fail_session(f"Avartha fatal error: {detail}")
            else:
                logger.warning("Avartha reported a nonfatal error: {}", detail)
        elif message_type == "server-message" and isinstance(data, dict):
            inner_type = data.get("type")
            if inner_type in {"tools_registered", "tools_registration_error"}:
                protocol = data.get("protocol")
                if protocol != HARNESS_TOOLS_PROTOCOL:
                    received = (
                        repr(protocol)
                        if protocol is not None
                        else "missing (legacy tool format)"
                    )
                    self._registration_error = AvarthaProtocolError(
                        "Incompatible Avartha tool protocol: "
                        f"expected {HARNESS_TOOLS_PROTOCOL!r}, received {received} "
                        f"in {inner_type}. Point --avartha-base-url at an agent "
                        f"that supports {HARNESS_TOOLS_PROTOCOL}."
                    )
                    self._registered.set()
                    return
            if inner_type == "session_boot":
                self._note_boot(data)
            elif inner_type == "tools_registered":
                self._registered.set()
            elif inner_type == "tools_registration_error":
                error = data.get("error")
                detail = (
                    str(error.get("message"))
                    if isinstance(error, dict)
                    else str(error or "unknown registration error")
                )
                self._registration_error = RuntimeError(
                    f"Avartha rejected register_tools: {detail}"
                )
                self._registered.set()
            elif inner_type == "audio_timeline":
                binding = parse_audio_timeline(data)
                if binding is None:
                    logger.warning("Avartha sent an unusable audio_timeline: {}", data)
                else:
                    self._queue.put_nowait(AgentAudioTimelineEvent(binding))
            elif inner_type == "assistant_playback":
                self._queue.put_nowait(
                    AgentPlaybackEvent(
                        text=str(data.get("text") or ""),
                        cut=data.get("delivery_status") == "cut",
                    )
                )
            elif inner_type == "tool_call":
                arguments = data.get("arguments")
                self._queue.put_nowait(
                    ToolCallEvent(
                        id=str(data.get("id", "")),
                        name=str(data.get("name", "")),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                )
        elif message_type == "bot-transcription" and isinstance(data, dict):
            self._queue.put_nowait(TranscriptEvent(str(data.get("text", ""))))
        elif message_type == "bot-llm-text" and isinstance(data, dict):
            self._queue.put_nowait(AgentLLMTextEvent(str(data.get("text", ""))))
        elif message_type == "bot-tts-text" and isinstance(data, dict):
            self._queue.put_nowait(
                AgentWordEvent(str(data.get("text", "")), parse_word_end(data))
            )
        elif message_type == "bot-llm-started":
            self._queue.put_nowait(AgentLLMStartedEvent())
        elif message_type == "bot-started-speaking":
            self._queue.put_nowait(AgentStartedSpeakingEvent())
        elif message_type in {
            "bot-interrupted",
            "user-started-speaking",
            "user-stopped-speaking",
        }:
            kind = {
                "bot-interrupted": "interrupted",
                "user-started-speaking": "speech_started",
                "user-stopped-speaking": "speech_stopped",
            }[message_type]
            self._queue.put_nowait(SpeechEvent(kind))
