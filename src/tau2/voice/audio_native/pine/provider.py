"""Pine Realtime WebSocket provider for tau2-bench.

Pine exposes an OpenAI-Realtime-API-compatible endpoint: the wire protocol
(``session.update``, ``input_audio_buffer.*``, ``response.*``, function calls,
and barge-in via ``conversation.item.truncate``) matches the OpenAI Realtime
protocol for the subset this adapter uses. This provider is therefore a thin
sibling of ``tau2.voice.audio_native.openai.provider`` with the base URL and
bearer credential pointed at the Pine gateway.

Authentication is a single durable bearer token presented on the WebSocket
upgrade:

    PINE_API_KEY     Pine API key (bearer token on the WS upgrade)
    PINE_BASE_URL    WebSocket endpoint
                     (default: wss://api-preview.pinevoice.ai/v1/realtime)

See https://tau-bench.pinevoice.ai/ for how to obtain an API key.
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_PINE_BASE_URL,
    DEFAULT_PINE_MODEL,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioEncoding, AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.pine.events import (
    BasePineEvent,
    PineTimeoutEvent,
    PineUnknownEvent,
    parse_pine_event,
)

load_dotenv()


DEFAULT_PINE_VOICE = "alloy"
DEFAULT_PINE_VAD_THRESHOLD = 0.5


def _audio_format_to_pine(audio_format: AudioFormat) -> str:
    """Map a tau2 AudioFormat to the Pine wire-format string.

    Pine accepts:
        g711_ulaw: 8 kHz mu-law (telephony default)
        pcm16:     signed-LE PCM (24 kHz)
    """
    enc = audio_format.encoding
    rate = audio_format.sample_rate
    if enc == AudioEncoding.ULAW and rate == 8000:
        return "g711_ulaw"
    if enc == AudioEncoding.PCM_S16LE:
        if rate != 24000:
            logger.warning(
                f"Pine pcm16 is specified at 24 kHz; got {rate} Hz - "
                f"the server will receive audio at this rate. Consider "
                f"g711_ulaw 8 kHz for telephony."
            )
        return "pcm16"
    raise ValueError(
        f"Audio format {audio_format} not supported by Pine "
        f"(expected ULAW 8kHz or PCM_S16LE 24kHz)"
    )


class PineVADMode(str, Enum):
    """Server-side VAD turn detection."""

    SERVER_VAD = "server_vad"


class PineVADConfig(BaseModel):
    """Configuration for Pine server VAD."""

    mode: PineVADMode = PineVADMode.SERVER_VAD
    threshold: float = DEFAULT_PINE_VAD_THRESHOLD
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class PineProvider:
    """Pine Realtime WebSocket provider.

    Manages a single WebSocket session against the Pine gateway. One provider
    instance maps to one connection, which maps to one conversation.

    Example:
        ```python
        provider = PineProvider()
        await provider.connect()
        await provider.configure_session(system_prompt, tools, PineVADConfig())
        async for event in provider.receive_events():
            ...
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_PINE_BASE_URL
    DEFAULT_MODEL = DEFAULT_PINE_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        """Initialize the Pine provider.

        Args:
            api_key: Pine API key. Defaults to env PINE_API_KEY.
            base_url: WebSocket endpoint. Defaults to env PINE_BASE_URL or the
                public gateway.
            model: Model identifier (sent in session.update).
            voice: TTS voice name (server's choice if None).

        Raises:
            ValueError: If no API key is provided.
        """
        self.api_key = api_key or os.environ.get("PINE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pine API key not provided. Set the PINE_API_KEY env var. "
                "See https://tau-bench.pinevoice.ai/ for how to obtain one."
            )

        self.base_url = base_url or os.environ.get("PINE_BASE_URL", self.BASE_URL)
        self.model = model or self.DEFAULT_MODEL
        self.voice = voice or DEFAULT_PINE_VOICE
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[PineVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        return self._audio_format

    @websocket_retry
    async def connect(self) -> None:
        """Open the WebSocket and read session.created.

        The durable PINE_API_KEY is presented as the bearer token on the
        WebSocket upgrade.
        """
        if self.is_connected:
            return

        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.info(f"Pine: connecting to {url}")
        # ping_interval/ping_timeout are raised from the websockets library
        # defaults (20/20) to 30/60 so long sessions are not closed with 1011
        # keepalive errors when the agent's monologue or local audio playback
        # keeps the main thread busy past the default 20s pong deadline.
        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=60,
        )

        raw = await self.ws.recv()
        data = json.loads(raw)
        if data.get("type") != "session.created":
            raise RuntimeError(
                f"Expected session.created, got {data.get('type')!r}: {data}"
            )

        sess = data.get("session", {})
        self.session_id = sess.get("id")
        proto_v = sess.get("protocol_version", "?")
        server_v = sess.get("server_version", "?")
        logger.info(
            f"Pine: session_id={self.session_id} protocol={proto_v} server={server_v}"
        )

    async def disconnect(self) -> None:
        if self.ws:
            logger.info("Pine: disconnecting")
            await self.ws.close()
            self.ws = None
            logger.info("Pine: disconnected")

    def _build_turn_detection_config(self, vad_config: PineVADConfig) -> Dict:
        """Build the turn_detection block for session.update."""
        return {
            "type": "server_vad",
            "threshold": vad_config.threshold,
            "prefix_padding_ms": vad_config.prefix_padding_ms,
            "silence_duration_ms": vad_config.silence_duration_ms,
        }

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Convert tau2 Tool objects to Pine tool definitions (OpenAI format)."""
        formatted: List[Dict] = []
        for tool in tools:
            schema = tool.openai_schema
            formatted.append(
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "parameters": schema["function"]["parameters"],
                }
            )
        return formatted

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: PineVADConfig,
        modality: str = "audio",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """Send session.update and block until session.updated arrives.

        Args:
            system_prompt: The agent's instructions.
            tools: Tool definitions.
            vad_config: Server VAD config.
            modality: Must be "audio".
            audio_format: Wire audio format (telephony mu-law by default).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        if modality != "audio":
            raise ValueError(
                f"Pine only supports modality='audio'; got {modality!r}"
            )

        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT
        self._audio_format = audio_format
        wire_fmt = _audio_format_to_pine(audio_format)

        session = {
            "type": "realtime",
            "instructions": system_prompt,
            "output_modalities": ["audio"],
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": wire_fmt,
                    "transcription": {"model": "whisper-1", "language": "en"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": self._build_turn_detection_config(vad_config),
                },
                "output": {
                    "format": wire_fmt,
                    "voice": self.voice,
                },
            },
        }
        payload = {"type": "session.update", "session": session}
        await self.ws.send(json.dumps(payload))

        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            t = data.get("type", "")
            if t == "session.updated":
                self._current_vad_config = vad_config
                logger.info(
                    f"Pine: session configured (format={wire_fmt}, tools={len(tools)})"
                )
                return
            if t == "error":
                err = data.get("error", {})
                msg = err.get("message", "unknown")
                code = err.get("code", "?")
                raise RuntimeError(f"Pine session.update rejected: {code}: {msg}")
            # Tolerate informational events between update and updated.
            logger.debug(f"Pine: ignoring pre-updated event {t}")

    async def send_audio(self, audio_data: bytes) -> None:
        """Append a chunk of user audio to the input buffer."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Pine")
        b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": b64})
        )

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Return a tool result and (by default) trigger response.create."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Pine")
        item = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self.ws.send(json.dumps(item))
        if request_response:
            await self.ws.send(json.dumps({"type": "response.create"}))

    async def truncate_item(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> None:
        """Tell the server how much audio was actually played for an item."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Pine")
        await self.ws.send(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )
        )
        logger.debug(
            f"Pine: truncate item={item_id} content_index={content_index} "
            f"audio_end_ms={audio_end_ms}"
        )

    async def receive_events(self) -> AsyncGenerator[BasePineEvent, None]:
        """Yield parsed events from the WebSocket. Emits TimeoutEvent every 10ms."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Pine")

        while self.is_connected:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw)
                yield parse_pine_event(data)
            except asyncio.TimeoutError:
                yield PineTimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Pine: WebSocket closed code={e.code} reason={e.reason!r}"
                )
                raise RuntimeError(
                    f"Pine WebSocket closed (code={e.code} reason={e.reason!r})"
                ) from e
            except Exception as e:
                logger.error(f"Pine: error receiving event: {type(e).__name__}: {e}")
                yield PineUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BasePineEvent]:
        """Collect non-timeout events for `duration_seconds`."""
        events: List[BasePineEvent] = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, PineTimeoutEvent):
                events.append(event)
            if asyncio.get_event_loop().time() >= end_time:
                break
        return events
