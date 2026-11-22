"""Resia Fullplex 1 over its WebSocket Live API.

Resia Fullplex 1 serves the OpenAI GPT-Live session protocol (session.start,
delegated Responses backend events in response.event, timed transcripts) over
one WebSocket instead of WebRTC. This provider keeps every part of
OpenAILiveProvider except the transport and the session instructions:

- Both the voice model and the delegated backend get the benchmark's system
  prompt unchanged. No frontend prompt template is added.
- Caller audio is 8 kHz mu-law on the wire, the benchmark's telephony format.
  It leaves in 20 ms frames, paced by the simulator's feeds: audio fed before
  the latest feed is due at once, and the latest feed's audio is due at real
  time from its arrival. The client never adds silence. A four-second burst
  allowance refills at 1.1 times real time. The adapter waits for each tick's
  paced socket sends. It collects received events during this wait. Plain
  ws drains the asyncio TCP buffer; wss drains only the TLS layer. The TCP
  buffer below TLS and the kernel can retain bytes. Only server counters
  can confirm API receipt.
- Agent audio arrives as session.output_audio.delta events, paced by the
  server at real time with a small lead. It is converted to the 24 kHz PCM
  that the Live adapter reads.
"""

import asyncio
import audioop
import base64
import json
import os
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
    DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_RESIA_FULLPLEX_URL,
    RESIA_FULLPLEX_API_KEY_ENV,
    RESIA_FULLPLEX_BACKEND_MODEL_ENV,
    RESIA_FULLPLEX_SAMPLE_RATE,
    RESIA_FULLPLEX_URL_ENV,
)
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.live_config import LiveConfig
from tau2.data_model.usage import UsageRecord
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    ErrorEvent,
    TimeoutEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.openai.live_provider import OpenAILiveProvider

_FINAL_RESPONSES = {"response.completed", "response.incomplete", "response.failed"}
_FRAME_BYTES = (
    RESIA_FULLPLEX_SAMPLE_RATE * DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS // 1000
)
# The server starts its engine with no deadline of its own; the adapter's connect
# call allows DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT in total, so wait almost all of it.
_START_TIMEOUT_S = DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT - 2.0
# The server sends session.closed when its open backend responses end, or 10 s
# after session.close; the adapter's disconnect call allows
# DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT in total.
_CLOSE_TIMEOUT_S = DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT - 0.5
# The Live API's input audio budget: at most 1.2 times real time after a burst of
# at most 5 s; an append past it is dropped with input_audio_rate_limit_exceeded.
# The client keeps under it with a margin for network jitter, which can bring
# frames to the server closer together than they left.
_BUDGET_BURST_S = 4.0
_BUDGET_RATE = 1.1

# One server message: an audio delta or a forwarded backend event, far below this.
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class _ErrorCode(StrEnum):
    CONNECTION_LOST = "connection_lost"
    INPUT_AUDIO_RATE_LIMIT_EXCEEDED = "input_audio_rate_limit_exceeded"


class _NoRedirectConnect(websockets.connect):
    """Never forward the bearer credential to a redirect destination."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


def resolve_endpoint(url: str | None = None) -> str:
    """The WebSocket URL: wss, or ws only on a loopback host (local tests)."""
    url = url or os.environ.get(RESIA_FULLPLEX_URL_ENV) or DEFAULT_RESIA_FULLPLEX_URL
    parts = urlsplit(url)
    if parts.scheme == "wss" and parts.hostname:
        return url
    if parts.scheme == "ws" and parts.hostname:
        try:
            loopback = ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname == "localhost"
        if loopback:
            return url
    raise ValueError(f"{RESIA_FULLPLEX_URL_ENV} must be a wss:// URL")


def resolve_backend_model(model: str | None = None) -> str:
    """The delegated backend model. There is no default: the run states it."""
    model = model or os.environ.get(RESIA_FULLPLEX_BACKEND_MODEL_ENV)
    if not model:
        raise ValueError(
            f"{RESIA_FULLPLEX_BACKEND_MODEL_ENV} is required (the backend model)"
        )
    return model


def resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get(RESIA_FULLPLEX_API_KEY_ENV)
    if not key:
        raise ValueError(f"{RESIA_FULLPLEX_API_KEY_ENV} is required")
    if len(key) > 4096 or not all(33 <= ord(c) <= 126 for c in key):
        raise ValueError("API key must be an ASCII token without whitespace")
    return key


class ResiaFullplexProvider(OpenAILiveProvider):
    """OpenAILiveProvider with a Resia Fullplex WebSocket as its connection."""

    def __init__(
        self,
        *,
        model: str,
        backend_model: str | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        url: str | None = None,
        trace_path: Path | None = None,
    ) -> None:
        super().__init__(
            model=model,
            config=LiveConfig(backend_model=resolve_backend_model(backend_model)),
            reasoning_effort=reasoning_effort,
            api_key=resolve_api_key(api_key),
            trace_path=trace_path,
        )
        self.url = resolve_endpoint(url)
        self._ws = None
        self._tasks: list[asyncio.Task] = []
        self._outbox: asyncio.Queue[str] = asyncio.Queue()
        self._down = self._up = None  # audioop.ratecv states across chunks
        # Feed-paced sender state.
        self._send_buf = bytearray()
        self._fed = asyncio.Event()
        self._sent = asyncio.Event()
        self._fed_bytes = 0  # all bytes fed so far
        self._fed_mark = 0  # bytes fed before the latest feed
        self._t_fed = 0.0  # loop time of the latest feed
        self.sent_bytes = 0
        self.max_pending_bytes = 0
        self.transmit_s = 0.0
        self.max_transmit_s = 0.0
        # Backend usage of each final response, drained by the adapter.
        self.helper_usage: list[UsageRecord] = []
        self._counted_responses: set[str] = set()
        self.session_seconds: float | None = None
        self.direct_calls = 0
        self.rate_limit_errors = 0  # dropped appends the server reported
        self._credit_s = _BUDGET_BURST_S  # the audio budget, in seconds of audio
        self._credit_at: float | None = None
        self.lost: str | None = None  # why the socket closed during the call

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._closed.is_set() and not self._closing

    async def connect(self) -> None:
        if self._ws is not None or self._closing:
            raise RuntimeError("Create a new provider for each session")

    def _build_session(self, system_prompt: str, tools: list[Tool]) -> dict:
        responses = {
            "model": self.config.backend_model,
            "instructions": system_prompt,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        if self.reasoning_effort is not None:
            responses["reasoning"] = {"effort": self.reasoning_effort}
        return {
            "model": self.model,
            "instructions": system_prompt,
            "audio": {
                "format": {"type": "audio/pcmu", "rate": RESIA_FULLPLEX_SAMPLE_RATE}
            },
            "delegation": {
                "type": "responses",
                "responses": responses,
                "mode": "always",
            },
        }

    async def configure_session(
        self,
        system_prompt: str,
        tools: list[Tool],
        vad_config: object = None,
        modality: str = "audio",
        audio_format: AudioFormat | None = None,
    ) -> None:
        if modality != "audio":
            raise ValueError("Resia Fullplex requires audio modality")
        self._audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
        )
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.trace_path.open("a", encoding="utf-8", buffering=1)
        session = self._build_session(system_prompt, tools)
        self._record("configuration", session)
        try:
            try:
                self._ws = await _NoRedirectConnect(
                    self.url,
                    additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    max_size=_MAX_MESSAGE_BYTES,
                    # CPython TLS pauses even an empty buffer when high is zero.
                    write_limit=(1, 0) if urlsplit(self.url).scheme == "wss" else 0,
                )
            except websockets.InvalidStatus as e:
                status = e.response.status_code
                raise RuntimeError(
                    f"Resia Fullplex refused the session: HTTP {status}"
                ) from None
            self._tasks = [
                asyncio.create_task(self._read()),
                asyncio.create_task(self._write()),
            ]
            self._send_event({"type": "session.start", "session": session})
            waiter = asyncio.create_task(self._started.wait())
            closer = asyncio.create_task(self._closed.wait())
            await asyncio.wait(
                {waiter, closer},
                timeout=_START_TIMEOUT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
            waiter.cancel()
            closer.cancel()
            if not self._started.is_set():
                raise RuntimeError(
                    f"Resia Fullplex session did not start: {self._start_error()}"
                )
            self._tasks.append(asyncio.create_task(self._feed_clock()))
        except BaseException:
            await self.disconnect()
            raise

    def _start_error(self) -> str:
        while not self._events.empty():
            event = self._events.get_nowait()
            if event.get("type") == "error":
                return json.dumps(event.get("error"))
        return "no session.started"

    # ------------------------------------------------------------------ audio in

    async def send_audio(self, audio_data: bytes) -> None:
        """Finish one tick's paced socket sends before tau advances."""
        if not self.is_connected:
            raise ConnectionError(
                f"Resia Fullplex session is not connected: {self.lost or 'closed'}"
            )
        pcm, self._down = audioop.ratecv(
            audio_data,
            2,
            1,
            DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
            RESIA_FULLPLEX_SAMPLE_RATE,
            self._down,
        )
        started = self._now()
        self.feed(audioop.lin2ulaw(pcm, 2))
        target = self._fed_bytes
        while self.sent_bytes < target:
            self._sent.clear()
            if not self.is_connected:
                raise ConnectionError(
                    f"Resia Fullplex transmission failed: {self.lost or 'closed'}"
                )
            await self._sent.wait()
        elapsed = self._now() - started
        self.transmit_s += elapsed
        self.max_transmit_s = max(self.max_transmit_s, elapsed)
        self._record(
            "input_transmitted",
            {
                "fed_bytes": target,
                "transmit_s": elapsed,
            },
        )

    def feed(self, data: bytes) -> None:
        """Audio fed before this call is due at once; this audio is due at real
        time from now."""
        self._fed_mark, self._t_fed = self._fed_bytes, self._now()
        self._fed_bytes += len(data)
        self._send_buf += data
        self.max_pending_bytes = max(
            self.max_pending_bytes, self._fed_bytes - self.sent_bytes
        )
        self._fed.set()

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _due_bytes(self, now: float) -> float:
        """The input bytes that may have started to leave by `now` (mu-law: one
        byte per sample)."""
        return self._fed_mark + (now - self._t_fed) * RESIA_FULLPLEX_SAMPLE_RATE

    async def _feed_clock(self) -> None:
        """Send 20 ms frames within the client budget: a burst of at most 4 s,
        then 1.1 times real time. The last piece is not padded. The due rule
        matches the reference client; send_audio also holds each
        tick until its socket sends complete. TLS can retain a lower TCP buffer."""
        try:
            while True:
                if not self._send_buf:
                    self._fed.clear()
                    await self._fed.wait()
                    continue
                wait = (
                    self.sent_bytes - self._due_bytes(self._now())
                ) / RESIA_FULLPLEX_SAMPLE_RATE
                if wait > 0:
                    self._fed.clear()
                    try:  # a feed can make the audio due earlier
                        await asyncio.wait_for(
                            self._fed.wait(),
                            min(
                                wait,
                                DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS / 1000,
                            ),
                        )
                    except TimeoutError:
                        pass
                    continue
                n = min(_FRAME_BYTES, len(self._send_buf))
                budget_wait = self._spend_budget(n / RESIA_FULLPLEX_SAMPLE_RATE)
                if budget_wait > 0:
                    await asyncio.sleep(budget_wait)
                    continue
                frame = bytes(self._send_buf[:n])
                await self._ws.send(
                    json.dumps(
                        {
                            "type": "session.input_audio.append",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }
                    )
                )
                del self._send_buf[:n]
                self.sent_bytes += n
                self._sent.set()
        except Exception as exc:
            self._connection_lost(f"audio transmission failed: {type(exc).__name__}")

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> list[BaseRealtimeEvent]:
        """Collect through the tick window and its pending caller transmission."""
        events = []
        end = self._now() + duration_seconds
        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)
            if self._now() >= end and self.sent_bytes >= self._fed_bytes:
                break
        if self.rate_limit_errors:
            raise ConnectionError(f"Resia Fullplex dropped caller audio: {self.lost}")
        return events

    def _spend_budget(self, seconds: float) -> float:
        """Take `seconds` of audio from the budget and return 0, or return how
        long to wait until the budget holds them."""
        now = self._now()
        if self._credit_at is not None:
            self._credit_s = min(
                _BUDGET_BURST_S,
                self._credit_s + (now - self._credit_at) * _BUDGET_RATE,
            )
        self._credit_at = now
        if self._credit_s < seconds:
            return (seconds - self._credit_s) / _BUDGET_RATE
        self._credit_s -= seconds
        return 0.0

    # ------------------------------------------------------------------ socket

    def _send_event(self, event: dict) -> None:
        if self._ws is None or self._closed.is_set():
            raise ConnectionError("Resia Fullplex session is not connected")
        self._outbox.put_nowait(json.dumps(event))
        self._record("sent", event)

    async def _write(self) -> None:
        try:
            while True:
                text = await self._outbox.get()
                await self._ws.send(text)
        except Exception as exc:
            self._connection_lost(f"control transmission failed: {type(exc).__name__}")

    def _connection_lost(self, reason: str) -> None:
        if not self._closing and not self._closed.is_set():
            self.lost = reason
            self._events.put_nowait(
                {
                    "type": "error",
                    "error": {
                        "code": _ErrorCode.CONNECTION_LOST.value,
                        "message": reason,
                    },
                }
            )
        self._closed.set()
        self._sent.set()

    async def _read(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                self._on_message(json.loads(message))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connection_lost("connection_lost: socket closed")

    def _on_message(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "error":
            error = parse_realtime_event(event)
            if (
                isinstance(error, ErrorEvent)
                and error.code == _ErrorCode.INPUT_AUDIO_RATE_LIMIT_EXCEEDED
            ):
                self.rate_limit_errors += 1
                self._record("received", event)
                logger.warning(
                    "Resia Fullplex dropped caller audio: {}", self.session_id
                )
                self._connection_lost(
                    "input_audio_rate_limit_exceeded: the server dropped caller audio"
                )
                return
        if kind == "session.output_audio.delta":
            ulaw = base64.b64decode(event["delta"])
            pcm, self._up = audioop.ratecv(
                audioop.ulaw2lin(ulaw, 2),
                2,
                1,
                RESIA_FULLPLEX_SAMPLE_RATE,
                DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
                self._up,
            )
            self._received_audio_bytes += len(pcm)
            self._events.put_nowait(
                {
                    "type": "live.media.audio",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
            return
        if kind == "session.started":
            self.session_id = (event.get("session") or {}).get("id")
        elif kind in ("session.usage.updated", "session.closed"):
            seconds = (event.get("usage") or {}).get("seconds")
            if seconds is not None:
                self.session_seconds = float(seconds)
        elif kind == "response.event":
            inner = event.get("event") or {}
            if inner.get("type") in _FINAL_RESPONSES:
                self._count_helper(inner.get("response") or {})
            if event.get("delegation_id") is None:
                self._outside_delegation(event)
                return
        self._accept_event(event)

    def _count_helper(self, response: dict) -> None:
        """One llm usage record per final backend response, parsed by tau's
        OpenAI usage reader, with the model that the response reports
        ("unreported" when it names none). A response without usage adds no
        record. The server can forward a final event twice; an ID counts once."""
        response_id = response.get("id")
        usage = response.get("usage")
        if response_id in self._counted_responses or not isinstance(usage, dict):
            return
        if response_id:
            self._counted_responses.add(response_id)
        record = UsageRecord.from_openai_realtime_usage(
            usage,
            provider="openai",
            model=response.get("model") or "unreported",
            scope_id=response_id,
        )
        self.helper_usage.append(record.model_copy(update={"component": "llm"}))

    def _outside_delegation(self, event: dict) -> None:
        """A function call outside a delegation is a server-side call tool (for
        example hang up), not a benchmark tool. It gets an error output, so the
        benchmark tools stay the only tools that act."""
        self._record("received", event)
        item = (event.get("event") or {}).get("item") or {}
        if item.get("type") != "function_call":
            return
        self.direct_calls += 1
        logger.warning(
            "Resia Fullplex call outside a delegation refused: {}", item.get("name")
        )
        self._send_event(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": item.get("call_id"),
                    "output": json.dumps({"error": "This tool is not available."}),
                },
            }
        )

    async def truncate_item(self, item_id, content_index, audio_end_ms) -> None:
        """The Live protocol has no truncate event; the server tracks what played."""

    async def disconnect(self) -> None:
        self._closing = True
        try:
            if (
                self._ws is not None
                and self._started.is_set()
                and not self._closed.is_set()
            ):
                self._outbox.put_nowait(json.dumps({"type": "session.close"}))
                self._record("sent", {"type": "session.close"})
                try:
                    await asyncio.wait_for(self._closed.wait(), _CLOSE_TIMEOUT_S)
                except TimeoutError:
                    logger.warning(
                        "Resia Fullplex close not confirmed: {}", self.session_id
                    )
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []
            if self._ws is not None:
                await self._ws.close()
        finally:
            self._record(
                "media",
                {
                    "sent_bytes": self.sent_bytes,
                    "fed_bytes": self._fed_bytes,
                    "pending_bytes": self._fed_bytes - self.sent_bytes,
                    "max_pending_bytes": self.max_pending_bytes,
                    "transmit_s": self.transmit_s,
                    "max_transmit_s": self.max_transmit_s,
                    "received_audio_bytes": self._received_audio_bytes,
                    "direct_calls_refused": self.direct_calls,
                    "rate_limit_errors": self.rate_limit_errors,
                },
            )
            if self._trace is not None:
                self._trace.close()
                self._trace = None
