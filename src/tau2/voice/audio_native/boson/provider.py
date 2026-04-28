"""Boson realtime voice chat provider."""

import asyncio
import base64
import json
import os
import uuid
from copy import deepcopy
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_BOSON_ASR_LANGUAGE,
    DEFAULT_BOSON_ASR_MODEL,
    DEFAULT_BOSON_MODEL,
    DEFAULT_BOSON_REALTIME_URL,
    DEFAULT_BOSON_TTS_MODEL,
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
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration: Optional[float] = None
    idle_timeout_ms: Optional[int] = None
    create_response: bool = False
    interrupt_response: bool = True
    enable_speaker_id: Optional[bool] = None
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
    DEFAULT_TTS_MODEL = DEFAULT_BOSON_TTS_MODEL
    DEFAULT_ASR_MODEL = DEFAULT_BOSON_ASR_MODEL
    DEFAULT_ASR_LANGUAGE = DEFAULT_BOSON_ASR_LANGUAGE

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        tts_model: Optional[str] = None,
        asr_model: Optional[str] = None,
        asr_language: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the Boson realtime provider.

        Args:
            api_key: Boson API key. If omitted, reads BOSON_API_KEY.
            model: Model identifier to put in session.update.
            voice: Voice identifier to put in session.audio.output.voice.
            tts_model: Boson audio generation model.
            asr_model: Boson audio understanding/transcription model.
            asr_language: Language name for input audio transcription.
            base_url: WebSocket endpoint. Defaults to DEFAULT_BOSON_REALTIME_URL.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        self.api_key = api_key or os.environ.get("BOSON_API_KEY")
        if not self.api_key:
            raise ValueError("Boson API key not provided. Set BOSON_API_KEY env var.")

        self.model = model or self.DEFAULT_MODEL
        self.voice = voice or self.DEFAULT_VOICE
        self.tts_model = tts_model or self.DEFAULT_TTS_MODEL
        self.asr_model = asr_model or self.DEFAULT_ASR_MODEL
        self.asr_language = asr_language or self.DEFAULT_ASR_LANGUAGE
        self.base_url = (
            base_url or os.environ.get("BOSON_REALTIME_URL") or self.BASE_URL
        )
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[BosonVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    def _event(self, event_type: str, **payload) -> str:
        """Serialize a Boson realtime client event with a unique event id."""
        return json.dumps(
            {"event_id": str(uuid.uuid4()), "type": event_type, **payload}
        )

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
        """Open the Boson realtime WebSocket.

        The integration guide says the server sends ``session.created`` immediately
        after connection. The staging endpoint may instead wait until the first
        ``session.update`` and then return a configured ``session.created``. To
        support both behaviors, connect opens the socket and only opportunistically
        consumes an immediate event if one is already available.
        """
        if self.is_connected:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        logger.info(f"Boson realtime: connecting to {self.base_url}")
        self.ws = await websockets.connect(
            self.base_url,
            additional_headers=headers,
            subprotocols=["realtime"],
            ping_interval=20,
            open_timeout=15,
            max_size=16 * 1024 * 1024,
        )

        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=0.25)
        except asyncio.TimeoutError:
            logger.debug(
                "Boson realtime: WebSocket opened; waiting to configure session"
            )
            return

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
                f"Boson connection failed: {error.get('message') or json.dumps(data)}"
            )
        logger.debug(f"Boson realtime: received pre-session event {event_type}")

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            logger.info("Boson realtime: disconnecting WebSocket connection")
            await self.ws.close()
            await self.ws.wait_closed()
            await asyncio.sleep(0)
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
            "create_response": vad_config.create_response,
            "interrupt_response": vad_config.interrupt_response,
        }
        if vad_config.min_speech_duration is not None:
            config["min_speech_duration"] = vad_config.min_speech_duration
        if vad_config.idle_timeout_ms is not None:
            config["idle_timeout_ms"] = vad_config.idle_timeout_ms
        if vad_config.enable_speaker_id is not None:
            config["enable_speaker_id"] = vad_config.enable_speaker_id

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
                    "parameters": self._format_parameters_for_api(
                        schema["function"]["parameters"]
                    ),
                }
            )
        return formatted_tools

    @classmethod
    def _format_parameters_for_api(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt OpenAI-style JSON schema to Boson's current tool limits.

        Boson realtime currently rejects function parameters whose runtime type
        is a list. For array parameters, ask the model to pass a JSON-array
        string; the adapter converts those strings back to lists before tau2
        executes the local tool.
        """
        formatted = deepcopy(parameters)
        for property_schema in formatted.get("properties", {}).values():
            cls._stringify_array_schema(property_schema)
        return formatted

    @classmethod
    def _stringify_array_schema(cls, schema: Dict[str, Any]) -> None:
        schema_type = schema.get("type")
        is_array = schema_type == "array" or (
            isinstance(schema_type, list) and "array" in schema_type
        )
        if is_array:
            description = schema.get("description", "")
            suffix = (
                "Pass this value as a JSON-array string, for example "
                '["item_1", "item_2"].'
            )
            schema.clear()
            schema.update(
                {
                    "type": "string",
                    "description": f"{description} {suffix}".strip(),
                }
            )
            return

        for nested_schema in schema.get("properties", {}).values():
            cls._stringify_array_schema(nested_schema)
        for option_key in ("anyOf", "oneOf", "allOf"):
            for option in schema.get(option_key, []):
                if isinstance(option, dict):
                    cls._stringify_array_schema(option)

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

        if modality not in ("text", "audio", "audio_in_text_out"):
            raise ValueError(f"Unknown modality: {modality}")

        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT
        self._audio_format = audio_format
        audio_fmt = audio_format_to_boson(audio_format)

        session = {
            "type": "realtime",
            "model": self.model,
            "instructions": system_prompt,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
        }

        if modality in ("audio", "audio_in_text_out"):
            session["audio"] = {
                "input": {
                    "format": audio_fmt,
                    "transcription": {
                        "model": self.asr_model,
                        "language": self.asr_language,
                    },
                    "turn_detection": self._build_turn_detection_config(vad_config),
                },
            }

        if modality == "audio":
            session.setdefault("audio", {})["output"] = {
                "format": audio_fmt,
                "voice": self.voice,
                "model": self.tts_model,
            }

        await self.ws.send(self._event("session.update", session=session))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type in ("session.updated", "session.created"):
                if not self._session_matches_config(data.get("session", {}), session):
                    logger.debug(
                        f"Boson realtime: received unconfigured {event_type}; "
                        "continuing to wait for configured session"
                    )
                    continue

                self.session_id = data.get("session", {}).get("id", self.session_id)
                self._current_vad_config = vad_config
                logger.info(
                    "Boson realtime: session configured "
                    f"(event={event_type}, session_id={self.session_id})"
                )
                return
            if event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    "Boson session configuration failed: "
                    f"{error.get('message') or json.dumps(data)}"
                )
            logger.debug(f"Boson realtime: received config-time event {event_type}")

    def _session_matches_config(self, received: dict, expected: dict) -> bool:
        """Return True when a session event reflects the requested config."""
        if not received:
            return False

        expected_audio = expected.get("audio") or {}
        received_audio = received.get("audio") or {}

        return (
            received.get("model") == expected.get("model")
            and received.get("instructions") == expected.get("instructions")
            and received.get("tool_choice") == expected.get("tool_choice")
            and received_audio.get("input", {}).get("format")
            == expected_audio.get("input", {}).get("format")
            and received_audio.get("output", {}).get("format")
            == expected_audio.get("output", {}).get("format")
        )

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to Boson's input audio buffer."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(self._event("input_audio_buffer.append", audio=audio_b64))

    async def commit_audio(self) -> None:
        """Commit Boson's currently buffered streaming input audio."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(self._event("input_audio_buffer.commit"))

    async def create_response(self) -> None:
        """Request an assistant response for the latest committed user input."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(self._event("response.create", response=None))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send a tool result and optionally request an assistant continuation."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(
            self._event(
                "conversation.item.create",
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            )
        )

        if request_response:
            await self.create_response()

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Tell Boson how much assistant audio was played before interruption."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(
            self._event(
                "conversation.item.truncate",
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        )
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
