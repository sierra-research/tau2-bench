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
from tau2.voice.utils.openai_utils import audio_format_to_openai

load_dotenv()


class OpenAIVADMode(str, Enum):
    """Voice Activity Detection modes supported by OpenAI's Realtime API.

    Attributes:
        SERVER_VAD: Server-side VAD using audio level thresholds and silence detection.
        SEMANTIC_VAD: Semantic-aware VAD that understands speech patterns and pauses.
        MANUAL: Manual turn detection where the client explicitly commits audio turns.
    """

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"
    MANUAL = "manual"


## TODO: We should have enum to specify output modality (text, audio, text_and_audio).
## TODO: Not sure where speech_in_speech_out and speech_in_text_out should go.


class OpenAIVADConfig(BaseModel):
    """Configuration for OpenAI's Voice Activity Detection.

    Configures how the API detects when the user has finished speaking.
    Different parameters apply depending on the selected mode.

    Attributes:
        mode: The VAD mode to use. Defaults to SERVER_VAD.
        threshold: Audio level threshold for SERVER_VAD (0.0-1.0).
            Higher values require louder speech to trigger. Default: 0.5.
        prefix_padding_ms: Milliseconds of audio to include before detected
            speech start (SERVER_VAD only). Default: 300.
        silence_duration_ms: Milliseconds of silence required to end a turn
            (SERVER_VAD only). Default: 500.
        eagerness: How eagerly to end turns for SEMANTIC_VAD mode.
            One of "low", "medium", "high". Default: "medium".
    """

    mode: OpenAIVADMode = OpenAIVADMode.SERVER_VAD
    threshold: float = DEFAULT_OPENAI_VAD_THRESHOLD
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    eagerness: str = "medium"  # For semantic_vad mode


class OpenAIRealtimeProvider:
    """OpenAI Realtime API provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to OpenAI's Realtime API,
    enabling real-time bidirectional communication for voice and text processing.
    It supports configurable Voice Activity Detection (VAD), tool/function calling,
    and both audio and text modalities.

    Attributes:
        BASE_URL: The WebSocket endpoint for OpenAI's Realtime API.
        DEFAULT_MODEL: The default model to use for realtime sessions.
        api_key: The OpenAI API key for authentication.
        model: The model identifier to use for the session.
        ws: The active WebSocket connection, or None if disconnected.

    Example:
        ```python
        provider = OpenAIRealtimeProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],
            vad_config=OpenAIVADConfig(),
            modality="text"
        )
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_OPENAI_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_OPENAI_REALTIME_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize the OpenAI Realtime provider.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY
                environment variable.
            model: Model identifier to use. Defaults to DEFAULT_MODEL.
            reasoning_effort: Reasoning effort for thinking models ("minimal",
                "low", "medium", "high"). If None, not sent to the API.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")

        self.model = model or self.DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[OpenAIVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active.

        Returns:
            True if connected and the WebSocket is in OPEN state, False otherwise.
        """
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        """Get the configured audio format.

        Returns:
            The AudioFormat configured for this session.
        """
        return self._audio_format

    @websocket_retry
    async def connect(self) -> None:
        """Establish a WebSocket connection to the OpenAI Realtime API.

        Opens a new WebSocket connection and waits for the session.created event
        to confirm successful connection. If already connected, this is a no-op.

        Raises:
            RuntimeError: If the initial handshake fails or receives unexpected response.
        """
        if self.is_connected:
            return

        url = f"{self.BASE_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

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
        """Close the WebSocket connection.

        Gracefully closes the WebSocket connection if one exists.
        Safe to call even if not connected.
        """
        if self.ws:
            logger.info("OpenAI Realtime API: disconnecting WebSocket connection")
            await self.ws.close()
            self.ws = None
            logger.info("OpenAI Realtime API: WebSocket connection closed")

    def _build_turn_detection_config(
        self, vad_config: OpenAIVADConfig
    ) -> Optional[Dict]:
        """Build the turn detection configuration for the API.

        Converts the internal VAD configuration to the format expected by
        OpenAI's Realtime API.

        Args:
            vad_config: The VAD configuration to convert.

        Returns:
            A dictionary with turn detection settings, or None for manual mode.

        Raises:
            ValueError: If the VAD mode is unknown.
        """
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
        """Format tools for the OpenAI Realtime API.

        Converts internal Tool objects to the format expected by the API,
        extracting the function name, description, and parameters from
        each tool's OpenAI schema.

        Args:
            tools: List of Tool objects to format.

        Returns:
            List of dictionaries in OpenAI's tool format.
        """
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
        """Configure the realtime session with instructions, tools, and settings.

        Sets up the session with the provided system prompt, available tools,
        VAD configuration, and modality settings. Waits for confirmation from
        the API before returning.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.
            modality: The input/output modality. One of:
                - "text": Text-only input and output.
                - "audio": Audio input and audio output (with text transcription).
                - "audio_in_text_out": Audio input with text-only output.
            audio_format: Audio format for input/output. Defaults to telephony
                (8kHz μ-law). Must be compatible with OpenAI Realtime API:
                g711_ulaw (8kHz), g711_alaw (8kHz), or pcm16 (24kHz).

        Raises:
            RuntimeError: If not connected or if session configuration fails.
            ValueError: If an unknown modality is specified or audio format unsupported.
        """
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

        # Default to telephony format if not specified
        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT

        # Store audio format for reference
        self._audio_format = audio_format

        audio_fmt = audio_format_to_openai(audio_format)

        session = {
            "type": "realtime",
            "instructions": system_prompt,
            "output_modalities": modalities,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
        }

        if self.reasoning_effort is not None:
            session["reasoning"] = {"effort": self.reasoning_effort}

        if modality in ("audio", "audio_in_text_out"):
            session["audio"] = {
                "input": {
                    "format": audio_fmt,
                    "transcription": {
                        "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
                        "language": "en",
                    },
                    "noise_reduction": {"type": DEFAULT_OPENAI_NOISE_REDUCTION},
                    "turn_detection": self._build_turn_detection_config(vad_config),
                },
            }

        if modality == "audio":
            session.setdefault("audio", {})["output"] = {
                "format": audio_fmt,
                "voice": DEFAULT_OPENAI_VOICE,
            }

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

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
        """Append audio data to the input audio buffer.

        Sends raw audio bytes to the API's input buffer. The audio is
        base64-encoded before transmission. With server VAD, the API commits
        turns automatically from buffered audio.

        Args:
            audio_data: Raw audio bytes in the configured input format (g711_ulaw).

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send a tool result and optionally request a continuation response."""
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
        """Tell the server how much audio was played before interruption."""
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
        """Async generator yielding parsed events from the WebSocket.

        Yields TimeoutEvent when no message arrives within 10ms.
        Raises RuntimeError if the connection closes unexpectedly.
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
        """Receive events for a specified duration.

        Collects all non-timeout events within the time window.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
