"""Boson realtime voice chat provider."""

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
    DEFAULT_BOSON_MODEL,
    DEFAULT_BOSON_REALTIME_URL,
    DEFAULT_BOSON_VOICE,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioEncoding, AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.boson.events import (
    BaseBosonEvent,
    BosonTimeoutEvent,
    BosonUnknownEvent,
    parse_boson_event,
)

load_dotenv()


class BosonVADMode(str, Enum):
    """Voice Activity Detection modes for Boson realtime voice chat."""

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"
    MANUAL = "manual"


class BosonVADConfig(BaseModel):
    """Configuration for Boson's turn detection."""

    mode: BosonVADMode = BosonVADMode.SERVER_VAD
    threshold: float = 0.55
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration: float = 0.125
    idle_timeout_ms: Optional[int] = None
    create_response: bool = True
    interrupt_response: bool = True
    enable_speaker_id: bool = False
    eagerness: str = "auto"
    timeout_sec: float = 0.4


def audio_format_to_boson(audio_format: AudioFormat) -> Dict:
    """Convert tau2 audio format metadata to Boson's realtime format object."""
    if audio_format.channels != 1:
        raise ValueError(f"Boson realtime requires mono audio, got {audio_format}")

    if audio_format.encoding == AudioEncoding.ULAW:
        if audio_format.sample_rate != 8000:
            raise ValueError(f"Boson PCMU requires 8kHz audio, got {audio_format}")
        return {"type": "audio/pcmu"}

    if audio_format.encoding == AudioEncoding.ALAW:
        if audio_format.sample_rate != 8000:
            raise ValueError(f"Boson PCMA requires 8kHz audio, got {audio_format}")
        return {"type": "audio/pcma"}

    if audio_format.encoding == AudioEncoding.PCM_S16LE:
        if audio_format.sample_rate not in (8000, 16000, 24000, 48000):
            raise ValueError(
                "Boson PCM16 sample rate must be one of "
                f"8000, 16000, 24000, or 48000 Hz, got {audio_format}"
            )
        return {"type": "audio/pcm", "rate": audio_format.sample_rate}

    raise ValueError(f"Unsupported Boson audio format: {audio_format}")


class BosonRealtimeProvider:
    """WebSocket client for Boson realtime voice chat."""

    BASE_URL = DEFAULT_BOSON_REALTIME_URL
    DEFAULT_MODEL = DEFAULT_BOSON_MODEL
    DEFAULT_VOICE = DEFAULT_BOSON_VOICE

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the Boson realtime provider.

        Args:
            api_key: Boson API key. If omitted, reads BOSON_API_KEY.
            model: Model identifier to put in session.update.
            voice: Voice identifier to put in session.audio.output.voice.
            base_url: WebSocket endpoint. Defaults to DEFAULT_BOSON_REALTIME_URL.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        self.api_key = api_key or os.environ.get("BOSON_API_KEY")
        if not self.api_key:
            raise ValueError("Boson API key not provided. Set BOSON_API_KEY env var.")

        self.model = model or self.DEFAULT_MODEL
        self.voice = voice or self.DEFAULT_VOICE
        self.base_url = base_url or os.environ.get("BOSON_REALTIME_URL") or self.BASE_URL
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[BosonVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Return True when the WebSocket connection is open."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        """Get the session audio format."""
        return self._audio_format

    @websocket_retry
    async def connect(self) -> None:
        """Open the Boson realtime WebSocket and wait for session.created."""
        if self.is_connected:
            return

        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info(f"Boson realtime: connecting to {self.base_url}")
        self.ws = await websockets.connect(
            self.base_url,
            additional_headers=headers,
            subprotocols=["realtime"],
        )

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type")
            if event_type == "session.created":
                session = data.get("session", {})
                self.session_id = session.get("id")
                logger.info(
                    f"Boson realtime: session created (session_id={self.session_id})"
                )
                return
            if event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    "Boson connection failed: "
                    f"{error.get('message') or json.dumps(data)}"
                )
            logger.debug(f"Boson realtime: received pre-session event {event_type}")

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            logger.info("Boson realtime: disconnecting WebSocket connection")
            await self.ws.close()
            self.ws = None
            logger.info("Boson realtime: WebSocket connection closed")

    def _build_turn_detection_config(
        self, vad_config: BosonVADConfig
    ) -> Optional[Dict]:
        """Build Boson's turn detection configuration."""
        if vad_config.mode == BosonVADMode.MANUAL:
            return None

        config: Dict = {
            "type": vad_config.mode.value,
            "threshold": vad_config.threshold,
            "prefix_padding_ms": vad_config.prefix_padding_ms,
            "silence_duration_ms": vad_config.silence_duration_ms,
            "min_speech_duration": vad_config.min_speech_duration,
            "idle_timeout_ms": vad_config.idle_timeout_ms,
            "create_response": vad_config.create_response,
            "interrupt_response": vad_config.interrupt_response,
            "enable_speaker_id": vad_config.enable_speaker_id,
        }

        if vad_config.mode == BosonVADMode.SEMANTIC_VAD:
            config.update(
                {
                    "eagerness": vad_config.eagerness,
                    "timeout_sec": vad_config.timeout_sec,
                }
            )

        return config

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tau2 tools for the Boson realtime API."""
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

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: BosonVADConfig,
        modality: str = "audio",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """Configure the Boson realtime session."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality == "text":
            modalities = ["text"]
        elif modality == "audio":
            modalities = ["audio"]
        elif modality == "audio_in_text_out":
            modalities = ["text"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT
        self._audio_format = audio_format
        audio_fmt = audio_format_to_boson(audio_format)

        session = {
            "type": "realtime",
            "model": self.model,
            "instructions": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": "inf",
            "output_modalities": modalities,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
            "truncation": None,
        }

        if modality in ("audio", "audio_in_text_out"):
            session["audio"] = {
                "input": {
                    "format": audio_fmt,
                    "noise_reduction": None,
                    "transcription": {
                        "model": "",
                        "language": "en",
                        "prompt": "",
                        "temperature": None,
                    },
                    "turn_detection": self._build_turn_detection_config(vad_config),
                },
            }

        if modality == "audio":
            session.setdefault("audio", {})["output"] = {
                "format": audio_fmt,
                "voice": self.voice,
                "model": None,
                "speed": 1.0,
                "temperature": None,
            }

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.updated":
                self._current_vad_config = vad_config
                logger.info("Boson realtime: session configured")
                return
            if event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    "Boson session configuration failed: "
                    f"{error.get('message') or json.dumps(data)}"
                )
            logger.debug(f"Boson realtime: received config-time event {event_type}")

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to Boson's input audio buffer."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send a tool result and optionally request an assistant continuation."""
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
            await self.ws.send(json.dumps({"type": "response.create", "response": None}))

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Tell Boson how much assistant audio was played before interruption."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }
        await self.ws.send(json.dumps(truncate_event))
        logger.debug(
            f"Boson truncate sent: item_id={item_id}, content_index={content_index}, "
            f"audio_end_ms={audio_end_ms}"
        )

    async def receive_events(self) -> AsyncGenerator[BaseBosonEvent, None]:
        """Yield parsed Boson events, with synthetic timeout events."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                yield parse_boson_event(data)
            except asyncio.TimeoutError:
                yield BosonTimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Boson realtime: WebSocket closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except websockets.ConnectionClosedError as e:
                logger.error(
                    f"Boson realtime: WebSocket connection error "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except Exception as e:
                logger.error(
                    f"Boson realtime: error receiving event: {type(e).__name__}: {e}"
                )
                yield BosonUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseBosonEvent]:
        """Collect events for a fixed wall-clock duration."""
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, BosonTimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
