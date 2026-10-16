"""Pipecat RTVI WebSocket client for the Ava Zero voice agent."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import websockets
from loguru import logger

from tau2.config import DEFAULT_AVA_ZERO_OUTPUT_SAMPLE_RATE
from tau2.utils.errors import NonRetryableError
from tau2.voice.audio_native.ava_zero_pipecat.config import AvaZeroPipecatConfig

# Wire identifier of the tool-exchange protocol the server must acknowledge.
HARNESS_TOOLS_PROTOCOL = "avartha-harness-tools/1"
_AUTH_QUERY_PARAMETERS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "key",
        "sig",
        "signature",
        "token",
    }
)


@dataclass
class AgentAudioEvent:
    """PCM16 audio emitted by Ava Zero."""

    # Read by TickResult.has_provider_activity; without it every tick looks idle.
    type: ClassVar[str] = "agent_audio"

    audio: bytes
    sample_rate: int


@dataclass
class ToolCallEvent:
    """Domain tool invocation requested by Ava Zero."""

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

    Ava Zero ``assistant_playback`` server-message; ``cut`` when the
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


class AvaZeroBootError(RuntimeError):
    """The Ava Zero session failed to become ready."""


class AvaZeroProtocolError(NonRetryableError):
    """The Ava Zero server does not support this harness tool protocol."""


def build_websocket_url(
    config: AvaZeroPipecatConfig,
    base_url: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Build the direct Pipecat WebSocket URL for one call."""

    raw_base = base_url or os.getenv("TAU2_AVA_ZERO_BASE_URL") or config.base_url
    parts = urlsplit(raw_base)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    if scheme not in {"ws", "wss"}:
        raise ValueError(
            f"Ava Zero base_url must use http(s) or ws(s), got {raw_base!r}"
        )

    path = parts.path.rstrip("/")
    if not path.endswith("/ws-client"):
        path = f"{path}/ws-client" if path else "/ws-client"

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["agent"] = config.agent
    if session_id:
        query["session_id"] = session_id
    if config.websocket_token_env:
        token = os.getenv(config.websocket_token_env)
        if not token:
            raise ValueError(
                "websocket_token_env names an environment variable that is not set: "
                f"{config.websocket_token_env}"
            )
        query["token"] = token

    return urlunsplit((scheme, parts.netloc, path, urlencode(query), ""))


def _redact_websocket_url(url: str) -> str:
    """Remove authentication values before a WebSocket URL reaches logs."""

    parts = urlsplit(url)
    query = [
        (key, "REDACTED" if key.lower() in _AUTH_QUERY_PARAMETERS else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


class AvaZeroPipecatProvider:
    """Exchange audio, transcripts, and tool calls with Ava Zero."""

    def __init__(
        self,
        config: Optional[AvaZeroPipecatConfig] = None,
        *,
        base_url: Optional[str] = None,
    ):
        self.config = config or AvaZeroPipecatConfig()
        self._base_url = base_url
        self.url = build_websocket_url(self.config, base_url)

        self._ws: Any = None
        self._serializer: Any = None
        self._input_audio_frame: Any = None
        self._input_transport_frame: Any = None
        self._output_audio_frame: Any = None
        self._output_transport_frame: Any = None
        self._message_label: Optional[str] = None
        self._protocol_version: Optional[str] = None

        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._reader: Optional[asyncio.Task[None]] = None
        self._registered = asyncio.Event()
        self._registration_error: Optional[RuntimeError] = None
        self._session_error: Optional[RuntimeError] = None
        self._cost_received = asyncio.Event()
        self._boot_settled = asyncio.Event()
        self._message_id = 0
        self._delivered_outputs: set[str] = set()

        self.session_id: Optional[str] = None
        self.cost_report: Optional[dict[str, Any]] = None
        self.boot_ready: Optional[bool] = None
        self.boot_error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Whether the provider currently owns an open WebSocket."""

        return self._ws is not None

    def _load_protocol(self) -> None:
        if self._serializer is not None:
            return
        try:
            from pipecat.frames.frames import (
                InputAudioRawFrame,
                InputTransportMessageFrame,
                OutputAudioRawFrame,
                OutputTransportMessageFrame,
            )
            from pipecat.processors.frameworks.rtvi import models as rtvi_models
            from pipecat.serializers.protobuf import ProtobufFrameSerializer
        except ImportError as exc:
            raise RuntimeError(
                "The ava_zero_pipecat provider requires Pipecat. "
                "Install tau2 with the voice extra: uv sync --extra voice"
            ) from exc

        self._serializer = ProtobufFrameSerializer()
        self._input_audio_frame = InputAudioRawFrame
        self._input_transport_frame = InputTransportMessageFrame
        self._output_audio_frame = OutputAudioRawFrame
        self._output_transport_frame = OutputTransportMessageFrame
        self._message_label = rtvi_models.MESSAGE_LABEL
        self._protocol_version = rtvi_models.PROTOCOL_VERSION

    def _http_base_url(self) -> str:
        """Return the product HTTP origin corresponding to the RTVI endpoint."""

        raw_base = (
            self._base_url
            or os.getenv("TAU2_AVA_ZERO_BASE_URL")
            or self.config.base_url
        )
        parts = urlsplit(raw_base)
        scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
        if scheme not in {"http", "https"}:
            raise ValueError(
                f"Ava Zero base_url must use http(s) or ws(s), got {raw_base!r}"
            )
        path = parts.path.rstrip("/")
        if path.endswith("/ws-client"):
            path = path[: -len("/ws-client")]
        return urlunsplit((scheme, parts.netloc, path, "", "")).rstrip("/")

    async def _create_session(self, system_prompt: str) -> str:
        """Install the Tau policy through the product configuration boundary."""

        payload = {
            "title": "Tau2 evaluation",
            "agent": {
                "base_agent_id": self.config.agent,
                "prompt": system_prompt,
                "tools": [],
                "capabilities": [],
                "chat_artifacts_enabled": False,
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{self._http_base_url()}/api/sessions",
                json=payload,
            )
            response.raise_for_status()
            session_id = response.json().get("id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("Ava Zero session API returned no session id")
        return session_id

    async def connect(self, system_prompt: str = "") -> None:
        """Connect, announce RTVI readiness, and wait for a healthy boot."""

        if self.is_connected:
            raise RuntimeError("Ava Zero WebSocket is already connected")
        self._reset_session_state()
        self._load_protocol()
        public_session_id = await self._create_session(system_prompt)
        self.url = build_websocket_url(
            self.config,
            self._base_url,
            session_id=public_session_id,
        )
        connect = websockets.connect(self.url, max_size=None)
        self._ws = await asyncio.wait_for(
            connect,
            timeout=self.config.connect_timeout_seconds,
        )
        self._reader = asyncio.create_task(
            self._read_loop(),
            name="tau2-ava-zero-reader",
        )
        await self._send_message(
            {
                "label": self._message_label,
                "type": "client-ready",
                "id": "tau2-0",
                "data": {
                    "version": self._protocol_version,
                    "about": {"library": "tau2"},
                },
            }
        )
        logger.info(
            f"Ava Zero connected and client-ready ({_redact_websocket_url(self.url)})"
        )
        await self._await_boot()

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
                "Ava Zero did not acknowledge register_tools within "
                f"{self.config.tool_registration_timeout_seconds:g}s"
            ) from exc
        if self._session_error:
            raise self._session_error
        if self._registration_error:
            raise self._registration_error
        logger.info(f"Ava Zero registered {len(schemas)} tool(s)")

    async def send_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Send one mono PCM16 audio frame."""

        if self._session_error:
            raise self._session_error
        if not audio_data or self._ws is None:
            return
        frame = self._output_audio_frame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=1,
        )
        payload = await self._serializer.serialize(frame)
        if payload:
            await self._ws.send(payload)

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Return a tau2 tool result to Ava Zero."""

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

    async def request_cost_report(self) -> None:
        """Best-effort retrieval of Ava Zero's per-session cost report."""

        if self._ws is None or self.cost_report is not None or self._session_error:
            return
        try:
            await self._client_message("get_cost_report", {})
            await asyncio.wait_for(
                self._cost_received.wait(),
                timeout=self.config.cost_report_timeout_seconds,
            )
        except TimeoutError:
            logger.warning("Ava Zero returned no cost_report before disconnect")
        except Exception as exc:
            logger.warning(f"Ava Zero cost_report request failed: {exc!r}")

    async def disconnect(self, request_cost_report: bool = True) -> None:
        """Request final metadata and close the socket and reader."""

        if self._ws is None:
            return
        if request_cost_report:
            await self.request_cost_report()
        websocket, self._ws = self._ws, None
        try:
            await websocket.close()
        except Exception as exc:
            logger.debug(f"Ava Zero WebSocket close failed: {exc!r}")
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None

    def _reset_session_state(self) -> None:
        """Clear state owned by one WebSocket session."""

        self._ws = None
        self._reader = None
        self._queue = asyncio.Queue()
        self._registered = asyncio.Event()
        self._registration_error = None
        self._session_error = None
        self._cost_received = asyncio.Event()
        self._boot_settled = asyncio.Event()
        self._message_id = 0
        self._delivered_outputs = set()
        self.session_id = None
        self.cost_report = None
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
            raise AvaZeroBootError(
                "Ava Zero session_boot never settled within "
                f"{self.config.boot_timeout_seconds:g}s"
            ) from exc
        if self._session_error:
            raise self._session_error
        if not self.boot_ready:
            raise AvaZeroBootError(
                self.boot_error or "Ava Zero session boot reported not ready"
            )
        logger.info("Ava Zero session boot is ready")

    def _note_boot(self, data: dict[str, Any]) -> None:
        if data.get("ready"):
            self.boot_ready = True
            self._boot_settled.set()
        elif data.get("error"):
            self.boot_ready = False
            self.boot_error = str(data["error"])
            self._boot_settled.set()

    async def _send_message(self, message: dict[str, Any]) -> None:
        if self._session_error:
            raise self._session_error
        if self._ws is None:
            raise RuntimeError("Ava Zero WebSocket is not connected")
        frame = self._output_transport_frame(message=message)
        payload = await self._serializer.serialize(frame)
        await self._ws.send(payload)

    async def _client_message(self, message_type: str, data: dict[str, Any]) -> None:
        self._message_id += 1
        await self._send_message(
            {
                "label": self._message_label,
                "type": "client-message",
                "id": f"tau2-{self._message_id}",
                "data": {"t": message_type, "d": data},
            }
        )

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, str):
                    continue
                frame = await self._serializer.deserialize(raw)
                if isinstance(frame, self._input_audio_frame):
                    await self._queue.put(
                        AgentAudioEvent(
                            audio=bytes(frame.audio),
                            sample_rate=(
                                frame.sample_rate or DEFAULT_AVA_ZERO_OUTPUT_SAMPLE_RATE
                            ),
                        )
                    )
                elif isinstance(frame, self._input_transport_frame):
                    if await self._accept_output(frame.message):
                        self._route_message(frame.message)
            if self._ws is not None:
                self._fail_session("Ava Zero WebSocket closed before disconnect")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._ws is not None:
                logger.warning(f"Ava Zero provider reader failed: {exc!r}")
                self._fail_session(f"Ava Zero WebSocket failed: {exc}")

    async def _accept_output(self, message: Any) -> bool:
        """Acknowledge a reliable caller output and route each one only once.

        Ava Zero resends every server-message carrying an
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
                logger.debug(f"Ava Zero output_ack {event_id} failed: {exc!r}")
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
        self._cost_received.set()
        self._queue.put_nowait(ProviderErrorEvent(message))

    def _route_message(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("label") != self._message_label:
            return

        message_type = message.get("type")
        data = message.get("data")
        if message_type == "error" and isinstance(data, dict):
            detail = str(data.get("error") or "unknown Ava Zero error")
            if data.get("fatal"):
                self._fail_session(f"Ava Zero fatal error: {detail}")
            else:
                logger.warning("Ava Zero reported a nonfatal error: {}", detail)
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
                    self._registration_error = AvaZeroProtocolError(
                        "Incompatible Ava Zero tool protocol: "
                        f"expected {HARNESS_TOOLS_PROTOCOL!r}, received {received} "
                        f"in {inner_type}. Point --ava-zero-base-url at a server "
                        f"that supports {HARNESS_TOOLS_PROTOCOL}."
                    )
                    self._registered.set()
                    return
            # "conductor_session" is the server's wire name for its session id.
            if inner_type == "conductor_session":
                self.session_id = data.get("sid")
                # The per-attempt task log is the join key for every boot,
                # including failed boots that have no final trajectory.
                logger.info("Ava Zero session sid={}", self.session_id)
            elif inner_type == "cost_report":
                report = data.get("data")
                self.cost_report = report if isinstance(report, dict) else {}
                self._cost_received.set()
            elif inner_type == "session_boot":
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
                    f"Ava Zero rejected register_tools: {detail}"
                )
                self._registered.set()
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
            self._queue.put_nowait(AgentWordEvent(str(data.get("text", ""))))
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
