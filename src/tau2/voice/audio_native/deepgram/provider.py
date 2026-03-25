"""
Deepgram Voice Agent API provider for cascaded voice processing.

NOTE: This is a CASCADED provider (STT → LLM → TTS), architecturally different
from native audio providers like OpenAI Realtime, Gemini Live, or Nova Sonic.

Uses WebSocket connection to Deepgram's Voice Agent API endpoint.
The API handles STT (Nova-3), LLM orchestration, and TTS (Aura-2) in one stream.

Key features:
- Input: Configurable audio format (default: linear16 @ 16kHz)
- Output: Configurable audio format (default: linear16 @ 16kHz)
- BYO LLM: Supports OpenAI, Anthropic, or custom LLM providers
- BYO TTS: Supports ElevenLabs, OpenAI, or custom TTS providers
- Built-in VAD, barge-in, turn-taking

Reference: Deepgram Voice Agent API documentation
https://developers.deepgram.com/docs/voice-agent
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_DEEPGRAM_INPUT_ENCODING,
    DEFAULT_DEEPGRAM_INPUT_SAMPLE_RATE,
    DEFAULT_DEEPGRAM_LLM_MODEL,
    DEFAULT_DEEPGRAM_LLM_PROVIDER,
    DEFAULT_DEEPGRAM_OUTPUT_ENCODING,
    DEFAULT_DEEPGRAM_OUTPUT_SAMPLE_RATE,
    DEFAULT_DEEPGRAM_STT_MODEL,
    DEFAULT_DEEPGRAM_TTS_MODEL,
    DEFAULT_DEEPGRAM_VOICE_AGENT_URL,
)
from tau2.environment.tool import Tool
from tau2.voice.audio_native.deepgram.events import (
    BaseDeepgramEvent,
    DeepgramTimeoutEvent,
    DeepgramUnknownEvent,
    parse_deepgram_event,
)

load_dotenv()

# Audio format constants (from config, with derived values)
DEEPGRAM_VOICE_AGENT_URL = DEFAULT_DEEPGRAM_VOICE_AGENT_URL
DEEPGRAM_INPUT_SAMPLE_RATE = DEFAULT_DEEPGRAM_INPUT_SAMPLE_RATE
DEEPGRAM_INPUT_ENCODING = DEFAULT_DEEPGRAM_INPUT_ENCODING
DEEPGRAM_OUTPUT_SAMPLE_RATE = DEFAULT_DEEPGRAM_OUTPUT_SAMPLE_RATE
DEEPGRAM_OUTPUT_ENCODING = DEFAULT_DEEPGRAM_OUTPUT_ENCODING

# Bytes per second for default format (16kHz, 16-bit mono)
DEEPGRAM_INPUT_BYTES_PER_SECOND = DEEPGRAM_INPUT_SAMPLE_RATE * 2
DEEPGRAM_OUTPUT_BYTES_PER_SECOND = DEEPGRAM_OUTPUT_SAMPLE_RATE * 2


class DeepgramVADMode(str, Enum):
    """Voice Activity Detection modes for Deepgram Voice Agent.

    Deepgram handles VAD automatically with configurable sensitivity.
    """

    SERVER_VAD = "server_vad"  # Server handles VAD (default)


class DeepgramVADConfig(BaseModel):
    """Configuration for Deepgram's Voice Activity Detection.

    Deepgram Voice Agent handles VAD automatically. This config
    allows tuning end-of-turn detection behavior.

    Attributes:
        mode: VAD mode (currently only SERVER_VAD is supported)
        endpointing_ms: Milliseconds of silence to detect end of speech
    """

    mode: DeepgramVADMode = DeepgramVADMode.SERVER_VAD
    endpointing_ms: int = 500  # Silence duration to trigger end of turn


class DeepgramVoiceAgentProvider:
    """Deepgram Voice Agent API provider with WebSocket-based communication.

    This provider manages a WebSocket connection to Deepgram's Voice Agent API,
    enabling real-time cascaded voice processing (STT → LLM → TTS).

    Unlike native audio providers (OpenAI Realtime, Gemini Live, Nova Sonic),
    this is a cascaded system where audio is transcribed to text, processed by
    an LLM, and synthesized back to audio.

    Attributes:
        api_key: Deepgram API key for authentication
        stt_model: Speech-to-text model (default: nova-3)
        llm_provider: LLM provider (openai, anthropic, deepgram)
        llm_model: LLM model identifier
        tts_model: Text-to-speech model (default: aura-2)
        tts_voice: TTS voice identifier
        ws: The active WebSocket connection, or None if disconnected

    Example:
        ```python
        provider = DeepgramVoiceAgentProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],
            vad_config=DeepgramVADConfig(),
        )
        await provider.send_audio(audio_bytes)
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    VOICE_AGENT_URL = DEEPGRAM_VOICE_AGENT_URL

    def __init__(
        self,
        api_key: Optional[str] = None,
        stt_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        input_sample_rate: int = DEEPGRAM_INPUT_SAMPLE_RATE,
        output_sample_rate: int = DEEPGRAM_OUTPUT_SAMPLE_RATE,
    ):
        """Initialize the Deepgram Voice Agent provider.

        Args:
            api_key: Deepgram API key. If not provided, reads from
                DEEPGRAM_API_KEY environment variable.
            stt_model: STT model identifier. Defaults to nova-3.
            llm_provider: LLM provider (openai, anthropic, deepgram).
            llm_model: LLM model identifier.
            tts_model: TTS model identifier including voice. For Deepgram,
                the voice is part of the model name (e.g., "aura-2-thalia-en").
            input_sample_rate: Input audio sample rate in Hz.
            output_sample_rate: Output audio sample rate in Hz.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not provided. Set DEEPGRAM_API_KEY env var."
            )

        self.stt_model = stt_model or DEFAULT_DEEPGRAM_STT_MODEL
        self.llm_provider = llm_provider or DEFAULT_DEEPGRAM_LLM_PROVIDER
        self.llm_model = llm_model or DEFAULT_DEEPGRAM_LLM_MODEL
        self.tts_model = tts_model or DEFAULT_DEEPGRAM_TTS_MODEL
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[DeepgramVADConfig] = None
        self._system_prompt: Optional[str] = None
        self._tools: List[Tool] = []
        self._context_messages: List[Dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active.

        Returns:
            True if connected and WebSocket is in OPEN state.
        """
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    async def connect(self) -> None:
        """Establish a WebSocket connection to Deepgram Voice Agent API.

        Opens a new WebSocket connection with the API key in the header.

        Raises:
            RuntimeError: If the connection fails.
        """
        if self.is_connected:
            return

        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
            }

            logger.info("Deepgram Voice Agent: Connecting...")
            self.ws = await websockets.connect(
                self.VOICE_AGENT_URL,
                additional_headers=headers,
            )
            logger.info("Deepgram Voice Agent: Connected successfully")

        except Exception as e:
            logger.error(f"Deepgram Voice Agent: Connection failed: {e}")
            raise RuntimeError(f"Failed to connect to Deepgram: {e}")

    async def disconnect(self) -> None:
        """Close the WebSocket connection.

        Gracefully closes the connection if one exists.
        """
        if self.ws:
            logger.info("Deepgram Voice Agent: Disconnecting...")
            await self.ws.close()
            self.ws = None
            logger.info("Deepgram Voice Agent: Disconnected")

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the Deepgram Voice Agent API.

        Deepgram expects a simpler format than OpenAI's wrapper format.
        Just name, description, and parameters at the top level.

        Args:
            tools: List of Tool objects to format.

        Returns:
            List of formatted tool dictionaries.
        """
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            # Deepgram uses a simpler format without the "type"/"function" wrapper
            formatted_tools.append(
                {
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
        vad_config: DeepgramVADConfig,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
        context_messages: Optional[List[Dict[str, Any]]] = None,
        greeting: Optional[str] = None,
    ) -> None:
        """Configure the voice agent session.

        Sends a Settings message to configure STT, LLM, and TTS providers.

        Args:
            system_prompt: System instructions for the LLM.
            tools: List of tools available for the agent.
            vad_config: Voice Activity Detection configuration.
            llm_provider: Override LLM provider (openai, anthropic, deepgram).
            llm_model: Override LLM model.
            tts_provider: Override TTS provider (deepgram, elevenlabs, openai).
            tts_model: Override TTS model. For Deepgram TTS, this includes the
                voice (e.g., "aura-2-thalia-en").
            context_messages: Prior conversation messages for context.
            greeting: Optional greeting message for the agent to speak on connect.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")

        self._current_vad_config = vad_config
        self._system_prompt = system_prompt
        self._tools = tools
        self._context_messages = context_messages or []

        # Build the Settings message
        # Reference: https://developers.deepgram.com/docs/voice-agent-settings
        settings = {
            "type": "Settings",
            "audio": {
                "input": {
                    "encoding": DEEPGRAM_INPUT_ENCODING,
                    "sample_rate": self.input_sample_rate,
                },
                "output": {
                    "encoding": DEEPGRAM_OUTPUT_ENCODING,
                    "sample_rate": self.output_sample_rate,
                    "container": "none",  # Raw audio, no container
                },
            },
            "agent": {
                "language": "en",
                "listen": {
                    "provider": {
                        "type": "deepgram",
                        "model": self.stt_model,
                    },
                },
                "think": {
                    "provider": {
                        "type": llm_provider or self.llm_provider,
                        "model": llm_model or self.llm_model,
                    },
                    "prompt": system_prompt,
                },
                "speak": {
                    "provider": {
                        "type": tts_provider or "deepgram",
                        # For Deepgram TTS, voice is part of the model name
                        "model": tts_model or self.tts_model,
                    },
                },
            },
        }

        # Add greeting if specified (agent speaks first)
        if greeting:
            settings["agent"]["greeting"] = greeting

        # Add tools/functions if provided
        if tools:
            settings["agent"]["think"]["functions"] = self._format_tools_for_api(tools)

        # Add context messages if provided
        if context_messages:
            settings["agent"]["context"] = {
                "messages": context_messages,
            }

        # Add VAD/endpointing configuration
        settings["agent"]["listen"]["provider"]["endpointing"] = (
            vad_config.endpointing_ms
        )

        # Send the settings
        await self.ws.send(json.dumps(settings))
        logger.info("Deepgram Voice Agent: Settings sent")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to the voice agent.

        Audio should be in the configured input format (default: linear16 @ 16kHz).

        Args:
            audio_data: Raw audio bytes.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        # Deepgram expects binary audio frames directly
        await self.ws.send(audio_data)

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        function_name: str = "",
        request_response: bool = True,
    ) -> None:
        """Send the result of a function call back to the agent.

        Args:
            call_id: The function call ID from FunctionCallRequest.
            result: The result as a JSON string.
            function_name: Name of the function (optional, for logging).
            request_response: Ignored (Deepgram continues automatically).

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        # Deepgram FunctionCallResponse format
        # Reference: https://developers.deepgram.com/docs/voice-agent-function-call-response
        function_result = {
            "type": "FunctionCallResponse",
            "id": call_id,
            "name": function_name,
            "content": result,  # Note: "content" not "output"
        }

        await self.ws.send(json.dumps(function_result))
        logger.debug(f"Deepgram: Sent function result for {call_id}")

    async def receive_events(self) -> AsyncGenerator[BaseDeepgramEvent, None]:
        """Receive and yield events from the WebSocket connection.

        An async generator that listens for events and yields them as
        typed event objects. Handles timeouts gracefully.

        Yields:
            BaseDeepgramEvent: Parsed event objects.

        Raises:
            RuntimeError: If not connected or connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.1)

                # Check if it's binary (audio) or text (JSON event)
                if isinstance(raw_message, bytes):
                    # Binary audio data - wrap in an Audio event
                    audio_b64 = base64.b64encode(raw_message).decode("utf-8")
                    event = parse_deepgram_event({"type": "Audio", "audio": audio_b64})
                else:
                    # JSON event
                    data = json.loads(raw_message)
                    event = parse_deepgram_event(data)

                yield event

            except asyncio.TimeoutError:
                yield DeepgramTimeoutEvent(type="timeout")

            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Deepgram: WebSocket closed (code={e.code}, reason='{e.reason}')"
                )
                raise RuntimeError(f"WebSocket closed: {e}")

            except Exception as e:
                logger.error(f"Deepgram: Error receiving event: {e}")
                yield DeepgramUnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseDeepgramEvent]:
        """Receive events for a specified duration.

        Useful for tick-based processing. Uses dynamic timeouts to avoid
        waiting longer than the specified duration.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        while self.is_connected:
            # Calculate remaining time - use short timeout to avoid overshooting
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            # Use remaining time as timeout, but cap at a small value for responsiveness
            timeout = min(remaining, 0.01)  # 10ms max, or remaining time

            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=timeout)

                # Check if it's binary (audio) or text (JSON event)
                if isinstance(raw_message, bytes):
                    # Binary audio data - wrap in an Audio event
                    audio_b64 = base64.b64encode(raw_message).decode("utf-8")
                    event = parse_deepgram_event({"type": "Audio", "audio": audio_b64})
                else:
                    # JSON event
                    data = json.loads(raw_message)
                    event = parse_deepgram_event(data)

                events.append(event)

            except asyncio.TimeoutError:
                # No message within timeout - check if we should continue
                continue

            except websockets.ConnectionClosed as e:
                logger.error(
                    f"Deepgram: WebSocket closed (code={e.code}, reason='{e.reason}')"
                )
                raise RuntimeError(f"WebSocket closed: {e}")

            except Exception as e:
                logger.error(f"Deepgram: Error receiving event: {e}")

        return events
