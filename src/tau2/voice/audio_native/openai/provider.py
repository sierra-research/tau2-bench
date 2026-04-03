"""
OpenAI Realtime API provider for end-to-end voice/text processing.
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
    DEFAULT_OPENAI_NOISE_REDUCTION,
    DEFAULT_OPENAI_REALTIME_BASE_URL,
    DEFAULT_OPENAI_REALTIME_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_VAD_THRESHOLD,
    DEFAULT_OPENAI_VOICE,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.utils.openai_utils import audio_format_to_openai_string

load_dotenv()


class OpenAIVADMode(str, Enum):
    """VAD modes for the Realtime API (server, semantic, or manual turn commits)."""

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"
    MANUAL = "manual"


## TODO: We should have enum to specify output modality (text, audio, text_and_audio).
## TODO: Not sure where speech_in_speech_out and speech_in_text_out should go.


class OpenAIVADConfig(BaseModel):
    """Realtime API turn detection settings; fields apply per ``mode`` (see OpenAI docs)."""

    mode: OpenAIVADMode = OpenAIVADMode.SERVER_VAD
    threshold: float = DEFAULT_OPENAI_VAD_THRESHOLD
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    eagerness: str = "medium"  # For semantic_vad mode


class OpenAIRealtimeProvider:
    """WebSocket client for OpenAI Realtime (VAD, tools, audio/text modalities)."""

    BASE_URL = DEFAULT_OPENAI_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_OPENAI_REALTIME_MODEL

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Create a provider; ``api_key`` defaults to ``OPENAI_API_KEY``."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")

        self.model = model or self.DEFAULT_MODEL
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[OpenAIVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket exists and is open."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        """Session input/output audio format."""
        return self._audio_format

    @websocket_retry
    async def connect(self) -> None:
        """Connect and wait for ``session.created``; no-op if already connected."""
        if self.is_connected:
            return

        url = f"{self.BASE_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.ws = await websockets.connect(url, additional_headers=headers)

        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        # Store and log session ID for debugging with OpenAI
        session_data = data.get("session", {})
        self.session_id = session_data.get("id")
        logger.info(
            f"OpenAI Realtime API: session created (session_id={self.session_id})"
        )

    async def disconnect(self) -> None:
        """Close the WebSocket if open."""
        if self.ws:
            logger.info("OpenAI Realtime API: disconnecting WebSocket connection")
            await self.ws.close()
            self.ws = None
            logger.info("OpenAI Realtime API: WebSocket connection closed")

    def _build_turn_detection_config(
        self, vad_config: OpenAIVADConfig
    ) -> Optional[Dict]:
        """Map ``vad_config`` to Realtime ``turn_detection`` (``None`` for manual)."""
        if vad_config.mode == OpenAIVADMode.MANUAL:
            return None
        elif vad_config.mode == OpenAIVADMode.SERVER_VAD:
            return {
                "type": "server_vad",
                "threshold": vad_config.threshold,
                "prefix_padding_ms": vad_config.prefix_padding_ms,
                "silence_duration_ms": vad_config.silence_duration_ms,
            }
        elif vad_config.mode == OpenAIVADMode.SEMANTIC_VAD:
            return {
                "type": "semantic_vad",
                "eagerness": vad_config.eagerness,
            }
        else:
            raise ValueError(f"Unknown VAD mode: {vad_config.mode}")

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Convert domain ``Tool`` list to Realtime function tools."""
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
        vad_config: OpenAIVADConfig,
        modality: str = "text",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """Send ``session.update`` and block until ``session.updated`` or API error.

        ``modality``: ``text`` | ``audio`` | ``audio_in_text_out``. ``audio_format``
        defaults to telephony (8kHz μ-law); must be a format the Realtime API accepts.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality == "text":
            modalities = ["text"]
        elif modality == "audio":
            modalities = ["text", "audio"]
        elif modality == "audio_in_text_out":
            modalities = ["text"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Default to telephony format if not specified
        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT

        # Store audio format for reference
        self._audio_format = audio_format

        session_config = {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "modalities": modalities,
                "tools": self._format_tools_for_api(tools),
                "tool_choice": "auto",
                "turn_detection": self._build_turn_detection_config(vad_config),
            },
        }

        if modality in ("audio", "audio_in_text_out"):
            # Get OpenAI format string from AudioFormat
            openai_format = audio_format_to_openai_string(audio_format)
            input_config = {
                "input_audio_format": openai_format,
                "input_audio_transcription": {
                    "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
                    "language": "en",
                },
                "input_audio_noise_reduction": {
                    "type": DEFAULT_OPENAI_NOISE_REDUCTION,
                },
            }
            session_config["session"].update(input_config)

        if modality == "audio":
            # Get OpenAI format string from AudioFormat
            openai_format = audio_format_to_openai_string(audio_format)
            session_config["session"].update(
                {
                    "voice": DEFAULT_OPENAI_VOICE,
                    "output_audio_format": openai_format,
                }
            )

        await self.ws.send(json.dumps(session_config))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.updated":
                self._current_vad_config = vad_config
                break
            elif event_type == "error":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def send_audio(self, audio_data: bytes) -> None:
        """Append base64-encoded PCM/codec bytes to ``input_audio_buffer``."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Submit ``function_call_output``; optionally send ``response.create``."""
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

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Barge-in: tell the server how much assistant audio was played (``conversation.item.truncate``).

        Aligns server history/transcript with heard audio; server may emit
        ``conversation.item.truncated``.
        """
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
            f"Truncate sent: item_id={item_id}, content_index={content_index}, "
            f"audio_end_ms={audio_end_ms}"
        )

    async def receive_events(self) -> AsyncGenerator[BaseRealtimeEvent, None]:
        """Yield parsed Realtime events until disconnect.

        Uses a short recv timeout: no message in time yields ``TimeoutEvent``.
        Parse errors yield ``UnknownEvent``. Unexpected close raises ``RuntimeError``.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                event = parse_realtime_event(data)
                yield event

            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"OpenAI Realtime API: WebSocket connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except websockets.ConnectionClosedError as e:
                logger.error(
                    f"OpenAI Realtime API: WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except Exception as e:
                logger.error(
                    f"OpenAI Realtime API: Error receiving event: {type(e).__name__}: {e}"
                )
                yield UnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseRealtimeEvent]:
        """Collect non-``TimeoutEvent`` events for ``duration_seconds``."""
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
