"""WebSocket client for Smallest AI's Hydra speech-to-speech server (smallest).

Hydra serves one full-duplex call per WebSocket on ``/v1/hydra``. Client to
server: ``session.configure`` (the benchmark's system prompt as
``instructions``, its tool schemas, an optional voice), then
``input_audio_buffer.append`` (base64 PCM16 mono 16 kHz), and for each tool
result a ``conversation.item.create`` (``function_call_output``) followed by
``response.create``. Server to client: OpenAI-Realtime-style events (see
``events.py``). Hydra runs the conversation itself and executes no tools: every tool call comes back to τ-bench.

``connect`` waits for ``session.configured``, which announces the rate of the
agent's audio; an error or a missing rate fails the connection instead of
letting the call run unconfigured or play audio at the wrong speed.
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import websockets
from loguru import logger
from websockets.protocol import State

from tau2.config import DEFAULT_SMALLEST_HANDSHAKE_TIMEOUT_SECONDS
from tau2.environment.tool import Tool
from tau2.voice.audio_native.smallest import events as ev

WS_URL_ENV = "SMALLEST_WS_URL"
API_KEY_ENV = "SMALLEST_API_KEY"
VOICE_ENV = "SMALLEST_VOICE"
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class _NoRedirectConnect(websockets.connect):
    """Never forward a credential to a redirect destination."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


def redact_url(url: str) -> str:
    """The URL without credentials, query or fragment, for logs."""
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"


def validate_ws_url(url: str, has_api_key: bool) -> str:
    """Check the Hydra URL; plaintext credentials only to a local server."""
    if not url:
        raise ValueError(
            f"{WS_URL_ENV} is not set: point it at the Hydra server, "
            "e.g. ws://localhost:<port>/v1/hydra"
        )
    parts = urlsplit(url)
    try:
        parts.port  # raises on a malformed port; the message must not echo the URL
    except ValueError:
        raise ValueError(f"{WS_URL_ENV} has an invalid port") from None
    if parts.scheme not in ("ws", "wss") or not parts.hostname:
        raise ValueError(f"{WS_URL_ENV} must be a ws:// or wss:// URL with a host")
    if parts.username or parts.password:
        raise ValueError(f"Do not put credentials in {WS_URL_ENV}; use {API_KEY_ENV}")
    sends_key = has_api_key or "api_key" in parse_qs(parts.query)
    if sends_key and parts.scheme == "ws" and parts.hostname not in _LOOPBACK:
        raise ValueError("Use wss:// to send an API key to a non-local Hydra server")
    return url


def format_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
    """τ-bench tools as Hydra's ``{"type":"function", name, ...}`` entries."""
    out: List[Dict[str, Any]] = []
    for tool in tools:
        schema = tool.openai_schema
        fn = schema.get("function", schema)
        out.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object"}),
            }
        )
    return out


def _close_reason(e: websockets.ConnectionClosed) -> str:
    frame = e.rcvd or e.sent
    return f"code={frame.code}, reason={frame.reason!r}" if frame else "no close frame"


class SmallestProvider:
    """One Hydra session: ``connect`` -> send/receive -> ``disconnect``."""

    def __init__(
        self,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        handshake_timeout_s: float = DEFAULT_SMALLEST_HANDSHAKE_TIMEOUT_SECONDS,
    ):
        """Settings default to ``SMALLEST_WS_URL`` / ``SMALLEST_API_KEY`` / ``SMALLEST_VOICE``.

        The API key is sent as ``Authorization: Bearer``; without one no
        credential is sent (a local server). Without a voice Hydra uses its
        configured default.
        """
        if ws_url is None:
            ws_url = os.environ.get(WS_URL_ENV, "")
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV, "")
        if voice is None:
            voice = os.environ.get(VOICE_ENV, "")
        self.ws_url = ws_url.strip()
        self.api_key = api_key.strip() or None
        self.voice = voice.strip() or None
        self.handshake_timeout_s = handshake_timeout_s
        self.ws: Optional[Any] = None
        # Hydra's session id (session.created), saved on the simulation run.
        self.session_id: Optional[str] = None
        # Rate of the agent's PCM16 audio, from session.configured.
        self.output_sample_rate: Optional[int] = None
        self._pending: List[ev.SmallestEvent] = []

    @property
    def is_connected(self) -> bool:
        return self.ws is not None and self.ws.state == State.OPEN

    async def connect(self, system_prompt: str, tools: List[Tool]) -> None:
        """Open the socket, send ``session.configure``, await ``session.configured``."""
        if self.api_key and not all(33 <= ord(c) <= 126 for c in self.api_key):
            raise ValueError(f"{API_KEY_ENV} must be an ASCII token without spaces")
        url = validate_ws_url(self.ws_url, bool(self.api_key))
        if self.ws is not None:
            await self.disconnect()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        logger.info(f"Smallest: connecting to {redact_url(url)}")
        self.ws = await _NoRedirectConnect(
            url, additional_headers=headers, max_size=None
        )
        self.session_id, self.output_sample_rate, self._pending = None, None, []
        session: Dict[str, Any] = {
            "instructions": system_prompt,
            "tools": format_tools(tools),
        }
        if self.voice:
            session["voice"] = self.voice
        try:
            await self._send({"type": "session.configure", "session": session})
            await asyncio.wait_for(self._await_configured(), self.handshake_timeout_s)
        except asyncio.TimeoutError as e:
            await self.disconnect()
            raise RuntimeError(
                f"Hydra did not confirm session.configure within "
                f"{self.handshake_timeout_s:g}s"
            ) from e
        except Exception:
            await self.disconnect()
            raise
        logger.info(
            f"Smallest: session {self.session_id} configured "
            f"(tools={len(tools)}, voice={self.voice}, "
            f"output={self.output_sample_rate} Hz)"
        )

    async def _await_configured(self) -> None:
        while True:
            event = self._parse(await self._recv())
            if isinstance(event, ev.SmallestSession):
                if event.type != ev.SESSION_CONFIGURED:
                    continue
                if not self.output_sample_rate:
                    raise RuntimeError(
                        "Hydra's session.configured announced no "
                        "output_audio_sample_rate"
                    )
                return
            if isinstance(event, ev.SmallestError):
                raise RuntimeError(
                    f"Hydra rejected the session: {event.code}: {event.message}"
                )
            if event is not None:
                self._pending.append(event)

    async def disconnect(self) -> None:
        ws, self.ws = self.ws, None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception as e:
            logger.debug(f"Smallest: error closing the socket: {e}")
        logger.info("Smallest: disconnected")

    async def send_audio(self, audio_pcm16: bytes) -> None:
        """Send PCM16 mono 16 kHz caller audio."""
        if audio_pcm16:
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_pcm16).decode("ascii"),
                }
            )

    async def send_tool_result(
        self, call_id: str, output: str, request_response: bool = True
    ) -> None:
        """Return a tool's output; ``response.create`` lets the reply continue."""
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )
        if request_response:
            await self._send({"type": "response.create", "response": None})

    async def receive_events_for_duration(
        self, duration_s: float
    ) -> List[ev.SmallestEvent]:
        """Every event that arrives within ``duration_s`` seconds."""
        if self.ws is not None and self.ws.state != State.OPEN:
            raise RuntimeError(
                f"Hydra connection closed (code={self.ws.close_code}, "
                f"reason={self.ws.close_reason!r})"
            )
        events, self._pending = self._pending, []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_s
        while (remaining := deadline - loop.time()) > 0:
            try:
                raw = await asyncio.wait_for(self._recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            event = self._parse(raw)
            if event is not None:
                events.append(event)
        return events

    async def _send(self, message: Dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("Not connected to Hydra. Call connect() first.")
        try:
            await self.ws.send(json.dumps(message))
        except websockets.ConnectionClosed as e:
            raise RuntimeError(f"Hydra connection closed ({_close_reason(e)})") from e

    async def _recv(self) -> Any:
        if self.ws is None:
            raise RuntimeError("Not connected to Hydra. Call connect() first.")
        try:
            return await self.ws.recv()
        except websockets.ConnectionClosed as e:
            raise RuntimeError(f"Hydra connection closed ({_close_reason(e)})") from e

    def _parse(self, raw: Any) -> Optional[ev.SmallestEvent]:
        if isinstance(raw, (bytes, bytearray)):
            logger.warning(f"Smallest: ignoring a binary frame ({len(raw)} bytes)")
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning(f"Smallest: ignoring a non-JSON frame: {raw[:80]!r}")
            return None
        if not isinstance(data, dict):
            return None
        event = ev.parse_hydra_event(data, self.output_sample_rate or 0)
        if isinstance(event, ev.SmallestSession):
            self.session_id = event.session_id or self.session_id
            if event.output_sample_rate:
                self.output_sample_rate = event.output_sample_rate
        elif isinstance(event, ev.SmallestResponseCreated) and event.sample_rate:
            self.output_sample_rate = event.sample_rate
        return event
