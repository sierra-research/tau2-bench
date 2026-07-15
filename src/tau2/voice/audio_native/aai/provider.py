"""
AAI voice agent provider for local WebSocket-based voice processing.

Communicates with a local AAI voice agent host via WebSocket, sending PCM16 audio
and receiving transcripts, audio responses, and tool calls.

The protocol uses:
- Binary frames for PCM16 audio chunks (16kHz in, 24kHz out)
- JSON frames for control messages and events
- Server-side VAD (voice activity detection) and turn-taking
"""

import asyncio
import json
import os
from typing import AsyncGenerator, Dict, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_AAI_CONFIG_FRAME_TIMEOUT,
    DEFAULT_AAI_INPUT_SAMPLE_RATE,
    DEFAULT_AAI_MODEL,
    DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_AAI_WS_URL,
)
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.aai.events import (
    AAIAudioChunkEvent,
    AAIConfigEvent,
    AAIErrorEvent,
    AAITimeoutEvent,
    BaseAAIEvent,
    parse_aai_event,
)

load_dotenv()

# Audio format constants (from config)
AAI_INPUT_SAMPLE_RATE = DEFAULT_AAI_INPUT_SAMPLE_RATE
AAI_INPUT_BYTES_PER_SECOND = AAI_INPUT_SAMPLE_RATE * 2  # 16-bit = 2 bytes

AAI_OUTPUT_SAMPLE_RATE = DEFAULT_AAI_OUTPUT_SAMPLE_RATE
AAI_OUTPUT_BYTES_PER_SECOND = AAI_OUTPUT_SAMPLE_RATE * 2  # 16-bit samples


class AAIVADConfig(BaseModel):
    """Configuration for AAI's Voice Activity Detection.

    AAI handles VAD server-side; this config is empty for interface parity
    with other providers.
    """

    pass


class AAIVoiceAgentProvider:
    """AAI voice agent provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to a local AAI
    voice agent host, enabling real-time bidirectional voice processing.

    The connection uses:
    - Binary frames (bytes) for PCM16 audio chunks
    - Text frames (JSON) for control messages and events
    - Server-side VAD and turn-taking

    Attributes:
        ws_url: WebSocket endpoint URL (from AAI_WS_URL env var or DEFAULT_AAI_WS_URL).
        input_sample_rate: Sample rate for audio sent to AAI (16000 Hz).
        tts_sample_rate: Sample rate for audio received from AAI (24000 Hz).
        system_prompt: System instructions for the voice agent.
        tools: Tools available to the voice agent.
        ws: Active WebSocket connection.

    Example:
        ```python
        provider = AAIVoiceAgentProvider(
            system_prompt="You are a helpful assistant.",
            tools=[],
        )

        await provider.connect()

        # Send audio
        await provider.send_audio(pcm16_audio_bytes)

        # Receive events for a tick duration
        events = await provider.receive_events_for_duration(0.2)
        for event in events:
            if isinstance(event, AAIAudioChunkEvent):
                play_audio(event.pcm16)
            elif isinstance(event, AAIToolCallEvent):
                result = await execute_tool(event.tool_name, event.args)
                await provider.send_tool_result(event.tool_call_id, result)

        await provider.disconnect()
        ```
    """

    DEFAULT_MODEL = DEFAULT_AAI_MODEL

    def __init__(
        self,
        ws_url: Optional[str] = None,
        input_sample_rate: int = DEFAULT_AAI_INPUT_SAMPLE_RATE,
        tts_sample_rate: int = DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
        system_prompt: str = "",
        tools: tuple = (),
    ):
        """Initialize the AAI voice agent provider.

        Args:
            ws_url: WebSocket URL for AAI host. If not provided, reads from
                AAI_WS_URL environment variable, falling back to DEFAULT_AAI_WS_URL.
            input_sample_rate: Sample rate for audio sent to AAI (default: 16000).
            tts_sample_rate: Sample rate for audio received from AAI (default: 24000).
            system_prompt: System instructions for the agent.
            tools: Tools available to the agent.
        """
        self.ws_url = ws_url or os.environ.get("AAI_WS_URL") or DEFAULT_AAI_WS_URL
        self.input_sample_rate = input_sample_rate
        self.tts_sample_rate = tts_sample_rate
        self.system_prompt = system_prompt
        self.tools = tools
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._buffered_events: List[BaseAAIEvent] = []

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @staticmethod
    def _with_host_flag(url: str) -> str:
        """Append ?host=1 to the WebSocket URL to activate host mode.

        Args:
            url: The WebSocket URL.

        Returns:
            The URL with ?host=1 appended (or merged with existing query params).
        """
        parsed = urlparse(url)
        # Parse existing query parameters
        query_params = {}
        if parsed.query:
            for param in parsed.query.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    query_params[key] = value
                else:
                    query_params[param] = ""
        # Add host=1
        query_params["host"] = "1"
        # Rebuild URL with new query string
        new_query = urlencode(query_params)
        new_parsed = parsed._replace(query=new_query)
        return urlunparse(new_parsed)

    def _format_tools_for_api(self, tools: tuple) -> List[Dict]:
        """Format tools for the AAI API.

        Converts Tool objects to flat function schema (not nested under "function").

        Args:
            tools: Tuple of Tool objects.

        Returns:
            List of flat function declarations.
        """
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            # Extract the nested function schema and flatten it
            func_schema = schema["function"]
            formatted_tools.append(
                {
                    "type": "function",
                    "name": func_schema["name"],
                    "description": func_schema["description"],
                    "parameters": func_schema["parameters"],
                }
            )
        return formatted_tools

    def _build_config_message(self, system_prompt: str, tools: tuple) -> Dict:
        """Build the configuration message for the AAI host.

        Args:
            system_prompt: System instructions for the agent.
            tools: Tools available to the agent.

        Returns:
            Configuration message dict with proper AAI host format.
        """
        return {
            "type": "config",
            "audioFormat": "pcm16",
            "sampleRate": self.input_sample_rate,
            "ttsSampleRate": self.tts_sample_rate,
            "host": {
                "systemPrompt": system_prompt,
                "tools": self._format_tools_for_api(tools),
            },
        }

    @websocket_retry
    async def connect(self) -> None:
        """Connect to the AAI voice agent host.

        Opens a WebSocket connection, appends the ?host=1 query parameter,
        sends the configuration message, and waits for acknowledgment.

        Raises:
            RuntimeError: If connection fails or handshake is incomplete.
        """
        if self.is_connected:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        # Build the URL with host=1 flag
        url = self._with_host_flag(self.ws_url)

        try:
            logger.info(f"AAI Provider: Connecting to {url}")
            self.ws = await websockets.connect(url)

            # Build and send config message
            config_msg = self._build_config_message(self.system_prompt, self.tools)
            await self.ws.send(json.dumps(config_msg))

            # Wait for the config handshake frame, bounded by a timeout so we
            # never hang forever if the host accepts the socket but never
            # initializes the agent. Each recv is individually bounded; any
            # non-config frame received in the meantime is buffered (not
            # dropped) so it can be surfaced by the next
            # receive_events_for_duration() call.
            while True:
                try:
                    frame = await asyncio.wait_for(
                        self.ws.recv(), timeout=DEFAULT_AAI_CONFIG_FRAME_TIMEOUT
                    )
                except asyncio.TimeoutError as e:
                    raise RuntimeError(
                        f"aai host did not send a config frame within "
                        f"{DEFAULT_AAI_CONFIG_FRAME_TIMEOUT}s; the agent did not "
                        "initialize"
                    ) from e

                if isinstance(frame, bytes):
                    # Binary frame (audio); buffer as audio chunk event
                    self._buffered_events.append(AAIAudioChunkEvent(pcm16=frame))
                    continue

                try:
                    data = json.loads(frame)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON frame: {frame}")
                    continue

                event = parse_aai_event(data)
                if isinstance(event, AAIConfigEvent):
                    logger.info("AAI Provider: Config acknowledged")
                    break
                if isinstance(event, AAIErrorEvent):
                    raise RuntimeError(
                        f"aai host returned an error during handshake: {event.message}"
                    )
                # Buffer any other event for later retrieval
                self._buffered_events.append(event)

            logger.info("AAI Provider: Connected and configured")

        except Exception as e:
            logger.error(f"Failed to connect to AAI host: {e}")
            raise RuntimeError(f"Failed to connect to AAI host: {e}") from e

    async def disconnect(self) -> None:
        """Close the WebSocket connection.

        Gracefully closes the connection if one exists.
        Safe to call even if not connected.
        """
        if self.ws:
            logger.info("AAI Provider: Disconnecting")
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            self.ws = None
            logger.info("AAI Provider: Disconnected")

    async def send_audio(self, pcm16: bytes) -> None:
        """Send audio data to the AAI host.

        Audio should be PCM16 at 16kHz.

        Args:
            pcm16: Raw PCM16 audio bytes.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AAI host")

        await self.ws.send(pcm16)

    async def send_tool_result(self, tool_call_id: str, result: str) -> None:
        """Send the result of a tool call back to the AAI host.

        Args:
            tool_call_id: The unique identifier of the tool call.
            result: The string result of the tool execution.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AAI host")

        message = {
            "type": "tool_result",
            "toolCallId": tool_call_id,
            "result": result,
        }
        await self.ws.send(json.dumps(message))

    async def receive_events(self) -> AsyncGenerator[BaseAAIEvent, None]:
        """Receive and yield events from the AAI WebSocket connection.

        Handles both binary frames (audio chunks) and text frames (JSON events).
        Uses per-frame timeouts to allow interleaved reception of multiple events.

        Yields:
            BaseAAIEvent: Parsed event objects or audio chunks.

        Raises:
            RuntimeError: If connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AAI host")

        while self.is_connected:
            try:
                frame = await asyncio.wait_for(self.ws.recv(), timeout=0.01)

                if isinstance(frame, bytes):
                    # Binary frame: audio chunk
                    yield AAIAudioChunkEvent(pcm16=frame)
                else:
                    # Text frame: JSON event
                    data = json.loads(frame)
                    event = parse_aai_event(data)
                    yield event

            except asyncio.TimeoutError:
                yield AAITimeoutEvent(type="timeout")
            except (websockets.ConnectionClosed, websockets.ConnectionClosedError) as e:
                logger.error(
                    f"AAI Provider: Connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason'}')"
                ) from e
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON frame: {e}")
            except Exception as e:
                logger.error(f"AAI Provider: Error receiving event: {e}")

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseAAIEvent]:
        """Receive events for a specified duration.

        Prepends and clears buffered events from the handshake before
        collecting new events.

        Args:
            duration_seconds: How long to collect events (in seconds).

        Returns:
            List of events received during the duration.
        """
        events = self._buffered_events.copy()
        self._buffered_events.clear()

        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, AAITimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
