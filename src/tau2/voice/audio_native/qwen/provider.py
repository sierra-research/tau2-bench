"""
Qwen Omni Flash Realtime API provider for real-time voice processing.

Uses WebSocket for bidirectional audio streaming with Alibaba Cloud's DashScope API.
The protocol is OpenAI-compatible with minor differences.

Key features:
- Input: PCM16 16kHz audio
- Output: PCM24 24kHz audio
- Server-side VAD for barge-in
- Bidirectional audio conversation

Reference: https://www.alibabacloud.com/help/en/model-studio/realtime

================================================================================
⚠️  CRITICAL LIMITATION: TOOL/FUNCTION CALLING DOES NOT WORK  ⚠️
================================================================================

Despite accepting tools configuration, the Qwen Realtime WebSocket API
(qwen3-omni-flash-realtime) does NOT actually invoke functions:

1. The model accepts tool definitions in session.update
2. The model may SAY "let me check that for you"
3. BUT it never emits function_call events - it generates audio instead

This is a limitation of the REALTIME API specifically. The HTTP API
(qwen3-omni-flash) DOES support tool calling correctly.

Tested: January 2026
Status: Audio streaming works, tool calling broken
Workaround: None available for realtime API

If tool calling is required, use OpenAI, Gemini, or xAI providers instead.
================================================================================
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
    DEFAULT_QWEN_INPUT_SAMPLE_RATE,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_OUTPUT_SAMPLE_RATE,
    DEFAULT_QWEN_REALTIME_URL,
    DEFAULT_QWEN_VOICE,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.qwen.events import (
    BaseQwenEvent,
    QwenTimeoutEvent,
    QwenUnknownEvent,
    parse_qwen_event,
)

load_dotenv()

# Audio format constants (from config)
QWEN_INPUT_SAMPLE_RATE = DEFAULT_QWEN_INPUT_SAMPLE_RATE
QWEN_INPUT_BYTES_PER_SECOND = QWEN_INPUT_SAMPLE_RATE * 2  # 16-bit = 2 bytes

QWEN_OUTPUT_SAMPLE_RATE = DEFAULT_QWEN_OUTPUT_SAMPLE_RATE
QWEN_OUTPUT_BYTES_PER_SECOND = QWEN_OUTPUT_SAMPLE_RATE * 2  # 16-bit samples


class QwenVADMode(str, Enum):
    """Voice Activity Detection modes for Qwen Realtime API."""

    SERVER_VAD = "server_vad"  # Server handles VAD automatically
    MANUAL = "manual"  # Client controls turns explicitly


class QwenVADConfig(BaseModel):
    """Configuration for Qwen's Voice Activity Detection.

    Attributes:
        mode: VAD mode. Defaults to SERVER_VAD.
        threshold: Speech detection sensitivity (0.0-1.0). Default: 0.5.
        prefix_padding_ms: Audio to include before speech start. Default: 300.
        silence_duration_ms: Silence before turn end. Default: 800.
    """

    mode: QwenVADMode = QwenVADMode.SERVER_VAD
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 800


class QwenRealtimeProvider:
    """Qwen Omni Flash Realtime API provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to Alibaba Cloud's
    DashScope Realtime API, enabling real-time bidirectional voice processing.

    The protocol is OpenAI-compatible, using the same event types:
    - session.created / session.updated
    - conversation.item.create
    - response.create / response.done
    - response.audio.delta / response.audio_transcript.delta
    - input_audio_buffer.append / speech_started / speech_stopped

    Attributes:
        BASE_URL: WebSocket endpoint for DashScope Realtime API.
        DEFAULT_MODEL: Default Qwen realtime model.
        api_key: DashScope API key for authentication.
        model: Model identifier.
        voice: Voice to use (e.g., Cherry).
        ws: Active WebSocket connection.

    Example:
        ```python
        provider = QwenRealtimeProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],  # NOTE: Tools are accepted but never invoked!
            vad_config=QwenVADConfig(),
        )
        await provider.send_text("Hello!")
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_QWEN_REALTIME_URL
    DEFAULT_MODEL = DEFAULT_QWEN_MODEL
    DEFAULT_VOICE = DEFAULT_QWEN_VOICE

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        """Initialize the Qwen Realtime provider.

        Args:
            api_key: DashScope API key. If not provided, reads from
                DASHSCOPE_API_KEY environment variable.
            model: Model identifier. Defaults to qwen3-omni-flash-realtime.
            voice: Voice to use. Defaults to Cherry.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key not provided. Set DASHSCOPE_API_KEY env var."
            )

        self.model = model or self.DEFAULT_MODEL
        self.voice = voice or self.DEFAULT_VOICE
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[QwenVADConfig] = None
        self._session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @websocket_retry
    async def connect(self) -> None:
        """Establish a WebSocket connection to the Qwen Realtime API.

        Opens a new WebSocket connection and waits for the session.created
        event to confirm successful connection.

        Raises:
            RuntimeError: If the initial handshake fails.
        """
        if self.is_connected:
            return

        url = f"{self.BASE_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.info(f"Qwen Realtime API: Connecting to {url}")
        self.ws = await websockets.connect(url, additional_headers=headers)

        # Wait for session.created
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        self._session_id = data.get("session", {}).get("id")
        logger.info(f"Qwen Realtime API: Connected, session={self._session_id}")

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            logger.info("Qwen Realtime API: Disconnecting")
            await self.ws.close()
            self.ws = None
            self._session_id = None
            logger.info("Qwen Realtime API: Disconnected")

    def _build_turn_detection_config(self, vad_config: QwenVADConfig) -> Optional[Dict]:
        """Build the turn detection configuration for the API."""
        if vad_config.mode == QwenVADMode.MANUAL:
            return None
        else:
            return {
                "type": "server_vad",
                "threshold": vad_config.threshold,
                "prefix_padding_ms": vad_config.prefix_padding_ms,
                "silence_duration_ms": vad_config.silence_duration_ms,
            }

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the Qwen API (OpenAI-compatible format)."""
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
        vad_config: QwenVADConfig,
        modality: str = "audio",
    ) -> None:
        """Configure the session with instructions, tools, and settings.

        Args:
            system_prompt: System instructions for the assistant.
            tools: Must be empty - Qwen Realtime API does not support tools.
            vad_config: Voice Activity Detection configuration.
            modality: "audio" for audio in/out, "text" for text only.

        Raises:
            NotImplementedError: If tools are provided (Qwen Realtime doesn't support them).
            RuntimeError: If not connected or configuration fails.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        # Raise error if tools are provided - Qwen Realtime API does NOT support function calling
        if tools:
            raise NotImplementedError(
                f"QWEN REALTIME API LIMITATION: {len(tools)} tools configured but "
                "tool/function calling does NOT work with qwen3-omni-flash-realtime. "
                "The model accepts tool configurations but NEVER invokes them - it generates "
                "audio responses instead."
            )

        self._current_vad_config = vad_config

        # Build session update
        if modality == "text":
            modalities = ["text"]
        else:
            modalities = ["text", "audio"]

        session_config = {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "modalities": modalities,
                "voice": self.voice,
                "tools": self._format_tools_for_api(tools),
                "tool_choice": "auto",
                "turn_detection": self._build_turn_detection_config(vad_config),
            },
        }

        logger.debug(f"Qwen session config: {json.dumps(session_config, indent=2)}")
        await self.ws.send(json.dumps(session_config))

        # Wait for session.updated confirmation
        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.updated":
                logger.info("Qwen Realtime API: Session configured")
                break
            elif event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    f"Session configuration failed: {error.get('message', data)}"
                )

    async def send_text(self, text: str, commit: bool = True) -> None:
        """Send a text message from the user.

        Args:
            text: Text content of the user's message.
            commit: If True, immediately request a response.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self.ws.send(json.dumps(item_create))

        if commit:
            await self.ws.send(json.dumps({"type": "response.create"}))

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to the input audio buffer.

        Audio should be PCM16 at 16kHz.

        Args:
            audio_data: Raw audio bytes.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def commit_audio(self) -> None:
        """Commit the audio buffer and request a response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def clear_audio_buffer(self) -> None:
        """Clear the input audio buffer."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send the result of a tool/function call back to the API.

        Args:
            call_id: The unique identifier of the function call.
            result: The string result of the function execution.
            request_response: If True, request the assistant to continue.
        """
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

    async def cancel_response(self) -> None:
        """Cancel an in-progress model response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "response.cancel"}))
        logger.debug("Response cancel sent")

    async def receive_events(self) -> AsyncGenerator[BaseQwenEvent, None]:
        """Receive and yield events from the WebSocket connection.

        Yields:
            BaseQwenEvent: Parsed event objects.

        Raises:
            RuntimeError: If connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                event = parse_qwen_event(data)
                yield event

            except asyncio.TimeoutError:
                yield QwenTimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Qwen Realtime API: Connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except websockets.ConnectionClosedError as e:
                logger.error(
                    f"Qwen Realtime API: Connection error "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except Exception as e:
                logger.error(f"Qwen Realtime API: Error receiving event: {e}")
                yield QwenUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseQwenEvent]:
        """Receive events for a specified duration.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, QwenTimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
