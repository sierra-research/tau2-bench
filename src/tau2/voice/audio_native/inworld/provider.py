"""Inworld Realtime API provider for real-time voice processing.

Uses WebSocket for bidirectional audio streaming with the Inworld Realtime API.
The wire protocol is OpenAI-Realtime-compatible with three notable differences:

1. **Auth:** ``Authorization: Basic <api_key>`` (the key is already base64-encoded
   from the Inworld Portal; do NOT base64-encode it again).
2. **URL:** ``wss://api.inworld.ai/api/v1/realtime/session`` with required query
   params ``key=voice-<timestamp_ms>`` and ``protocol=realtime``.
3. **Audio:** PCM16 at 24 kHz only. ``audio/pcmu`` requests are silently
   coerced to PCM. Conversion from telephony μ-law is done by the adapter.

Session config places audio settings under nested input/output objects, with
``semantic_vad`` turn detection (``eagerness``) and TTS engine + voice
selected under ``audio.output``.

Reference: https://docs.inworld.ai/api-reference/realtimeAPI/realtime/realtime-websocket
"""

import asyncio
import base64
import json
import os
import time
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_INWORLD_EAGERNESS,
    DEFAULT_INWORLD_MODEL,
    DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE,
    DEFAULT_INWORLD_REALTIME_URL,
    DEFAULT_INWORLD_TTS_MODEL,
    DEFAULT_INWORLD_VOICE,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.inworld.events import (
    BaseInworldEvent,
    InworldTimeoutEvent,
    InworldUnknownEvent,
    parse_inworld_event,
)

load_dotenv()

# Inworld output format (PCM16 mono @ 24 kHz) — 48000 bytes/sec
INWORLD_OUTPUT_BYTES_PER_SECOND = DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE * 2


class InworldEagerness(str, Enum):
    """Semantic VAD eagerness levels for Inworld Realtime API."""

    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InworldVADConfig(BaseModel):
    """Configuration for Inworld's semantic Voice Activity Detection.

    Attributes:
        eagerness: How aggressively the server detects end-of-turn. Higher
            eagerness → faster turn-taking but more interruptions.
        create_response: Auto-create a response when VAD detects end-of-turn.
        interrupt_response: Allow user audio to interrupt agent response.
    """

    eagerness: InworldEagerness = InworldEagerness.HIGH
    create_response: bool = True
    interrupt_response: bool = True


class InworldRealtimeProvider:
    """Inworld Realtime API provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to Inworld's
    Realtime API, enabling real-time bidirectional voice agent interaction.

    Attributes:
        api_key: The Inworld API key (already base64-encoded from the portal).
        model: LLM backbone identifier (e.g., ``openai/gpt-4.1-mini``).
        voice: TTS voice name (e.g., ``Clive``).
        tts_model: TTS engine (e.g., ``inworld-tts-1.5-mini``).
        ws: The active WebSocket connection, or None if disconnected.
    """

    BASE_URL = DEFAULT_INWORLD_REALTIME_URL
    DEFAULT_MODEL = DEFAULT_INWORLD_MODEL
    DEFAULT_VOICE = DEFAULT_INWORLD_VOICE
    DEFAULT_TTS_MODEL = DEFAULT_INWORLD_TTS_MODEL
    DEFAULT_EAGERNESS = DEFAULT_INWORLD_EAGERNESS

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        tts_model: Optional[str] = None,
        sample_rate: int = DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE,
    ):
        """Initialize the Inworld Realtime provider.

        Args:
            api_key: Inworld API key. If not provided, reads ``INWORLD_API_KEY``
                from the environment.
            model: LLM backbone (e.g., ``openai/gpt-4.1-mini``). Reads
                ``INWORLD_MODEL`` env var if not provided.
            voice: TTS voice name. Reads ``INWORLD_VOICE`` env var if not
                provided.
            tts_model: TTS engine name. Reads ``INWORLD_TTS_MODEL`` env var if
                not provided.
            sample_rate: PCM sample rate (24000 — Inworld's only supported
                rate at present).

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("INWORLD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Inworld API key not provided. Set INWORLD_API_KEY env var."
            )

        self.model = model or os.environ.get("INWORLD_MODEL") or self.DEFAULT_MODEL
        self.voice = voice or os.environ.get("INWORLD_VOICE") or self.DEFAULT_VOICE
        self.tts_model = (
            tts_model or os.environ.get("INWORLD_TTS_MODEL") or self.DEFAULT_TTS_MODEL
        )
        self.sample_rate = sample_rate
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[InworldVADConfig] = None
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @websocket_retry
    async def connect(self) -> None:
        """Establish a WebSocket connection to the Inworld Realtime API.

        Raises:
            RuntimeError: If the initial handshake fails.
        """
        if self.is_connected:
            return

        headers = {"Authorization": f"Basic {self.api_key}"}
        url = f"{self.BASE_URL}?key=voice-{int(time.time() * 1000)}&protocol=realtime"

        logger.info(f"Inworld Realtime API: Connecting to {self.BASE_URL}")
        self.ws = await websockets.connect(url, additional_headers=headers)

        # Wait for session.created event
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        session = data.get("session", {})
        self.session_id = session.get("id") or data.get("event_id")
        logger.info(f"Inworld Realtime API: Connected (session_id={self.session_id})")

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            logger.info("Inworld Realtime API: Disconnecting")
            await self.ws.close()
            self.ws = None
            logger.info("Inworld Realtime API: Disconnected")

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the Inworld API (OpenAI-compatible schema)."""
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            formatted_tools.append(
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "parameters": schema["function"]["parameters"],
                }
            )
        return formatted_tools

    def _build_turn_detection_config(self, vad_config: InworldVADConfig) -> Dict:
        """Build the semantic VAD turn-detection config."""
        return {
            "type": "semantic_vad",
            "eagerness": vad_config.eagerness.value
            if hasattr(vad_config.eagerness, "value")
            else vad_config.eagerness,
            "create_response": vad_config.create_response,
            "interrupt_response": vad_config.interrupt_response,
        }

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: InworldVADConfig,
        modality: str = "audio",
    ) -> None:
        """Configure the realtime session with instructions, tools, and audio.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.
            modality: "audio" or "text" (Inworld only supports audio in practice).

        Raises:
            RuntimeError: If not connected or if session configuration fails.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality == "audio":
            output_modalities = ["audio"]
        elif modality == "text":
            output_modalities = ["text"]
        else:
            output_modalities = ["audio"]

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": system_prompt,
                "output_modalities": output_modalities,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.sample_rate,
                        },
                        "turn_detection": self._build_turn_detection_config(vad_config),
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.sample_rate,
                        },
                        "model": self.tts_model,
                        "voice": self.voice,
                    },
                },
                "tools": self._format_tools_for_api(tools),
            },
        }

        logger.debug(f"Inworld session config: {json.dumps(session_config, indent=2)}")
        await self.ws.send(json.dumps(session_config))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")
            if event_type == "session.updated":
                self._current_vad_config = vad_config
                logger.info("Inworld Realtime API: Session configured")
                return
            if event_type == "error":
                error = data.get("error", {})
                error_msg = error.get("message", str(error))
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def send_audio(self, audio_data: bytes) -> None:
        """Append PCM16 audio bytes to the input audio buffer (base64-encoded)."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )

    async def cancel_response(self) -> None:
        """Cancel the in-flight assistant response (used for barge-in)."""
        if not self.is_connected:
            return
        await self.ws.send(json.dumps({"type": "response.cancel"}))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send a tool result and optionally request a continuation."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self.ws.send(json.dumps(item_create))

        if request_response:
            await self.ws.send(json.dumps({"type": "response.create"}))

    async def receive_events(self) -> AsyncGenerator[BaseInworldEvent, None]:
        """Receive and yield events from the WebSocket connection."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                yield parse_inworld_event(data)
            except asyncio.TimeoutError:
                yield InworldTimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Inworld Realtime API: WebSocket closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except Exception as e:
                logger.error(f"Inworld Realtime API: Error receiving event: {e}")
                yield InworldUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseInworldEvent]:
        """Receive events for the specified duration (tick-based polling)."""
        events: List[BaseInworldEvent] = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, InworldTimeoutEvent):
                events.append(event)
            if asyncio.get_event_loop().time() >= end_time:
                break
        return events
