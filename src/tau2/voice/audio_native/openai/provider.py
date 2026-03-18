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
        await provider.send_text("Hello!")
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_OPENAI_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_OPENAI_REALTIME_MODEL

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the OpenAI Realtime provider.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY
                environment variable.
            model: Model identifier to use. Defaults to DEFAULT_MODEL.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
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

    async def set_vad_mode(self, vad_config: OpenAIVADConfig) -> None:
        """Update the Voice Activity Detection mode for the session.

        Sends a session update to change VAD settings. This is a fire-and-forget
        operation that does not wait for confirmation from the API.

        Args:
            vad_config: The new VAD configuration to apply.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": self._build_turn_detection_config(vad_config),
            },
        }

        await self.ws.send(json.dumps(session_update))
        self._current_vad_config = vad_config
        logger.debug(f"VAD mode update sent: {vad_config.mode}")

    async def send_text(self, text: str, commit: bool = True) -> None:
        """Send a text message from the user to the conversation.

        Creates a user message in the conversation and optionally triggers
        a response from the assistant.

        Args:
            text: The text content of the user's message.
            commit: If True, immediately request a response from the assistant.
                If False, the message is added but no response is triggered.

        Raises:
            RuntimeError: If not connected to the API.
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

    async def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the conversation history.

        Injects a message as if it came from the assistant, useful for
        seeding conversation context or simulating prior exchanges.
        Does not trigger a new response.

        Args:
            text: The text content of the assistant's message.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
        await self.ws.send(json.dumps(item_create))

    async def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation without triggering a response.

        Convenience method that calls send_text with commit=False.
        Useful for building up conversation context.

        Args:
            text: The text content of the user's message.

        Raises:
            RuntimeError: If not connected to the API.
        """
        await self.send_text(text, commit=False)

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to the input audio buffer.

        Sends raw audio bytes to the API's input buffer. The audio is
        base64-encoded before transmission. Audio accumulates in the buffer
        until commit_audio() is called.

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

    async def commit_audio(self) -> None:
        """Commit the audio buffer and request a response.

        Finalizes the accumulated audio in the input buffer and triggers
        the assistant to process it and generate a response.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def clear_audio_buffer(self) -> None:
        """Clear the input audio buffer without processing.

        Discards any accumulated audio data in the input buffer.
        Useful for canceling or resetting audio input.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send the result of a tool/function call back to the API.

        After the assistant requests a function call, this method is used to
        provide the result of that function execution.

        Args:
            call_id: The unique identifier of the function call being responded to.
                This must match the call_id from the original function call event.
            result: The string result of the function execution.
            request_response: If True, immediately request the assistant to
                continue generating a response. If False, just submit the result.

        Raises:
            RuntimeError: If not connected to the API.
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
        """Cancel an in-progress model response.

        Use this in push-to-talk scenarios (when VAD is disabled) to manually
        cancel the model's response when the user wants to interrupt. In VAD
        mode, the server automatically cancels responses when user speech is
        detected, so this is typically not needed.

        After canceling, you should also send a truncate_item() to inform the
        server how much audio was actually played.

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        await self.ws.send(json.dumps({"type": "response.cancel"}))
        logger.debug("Response cancel sent")

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Truncate an assistant response item to remove unplayed audio.

        When the user interrupts the assistant (barge-in), this method should be
        called to inform the server how much of the response was actually played.
        The server will truncate the audio at the specified point and remove the
        corresponding portion of the transcript from the conversation history.

        This ensures the model's memory of the conversation matches what the user
        actually heard, enabling natural follow-ups like "what was that last thing?".

        Args:
            item_id: The item ID of the assistant's response being truncated.
                This is the item that was interrupted.
            content_index: Index of the content part being truncated (usually 0
                for single-part responses).
            audio_end_ms: Milliseconds of audio that was actually played before
                the interruption. Audio after this point will be removed from
                the conversation.

        Raises:
            RuntimeError: If not connected to the API.

        Note:
            The server will respond with a conversation.item.truncated event
            to confirm the truncation.
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
        """Receive and yield events from the WebSocket connection.

        An async generator that continuously listens for events from the API
        and yields them as typed event objects. Handles connection timeouts
        gracefully by yielding TimeoutEvent, allowing the caller to perform
        other operations.

        Yields:
            BaseRealtimeEvent: Parsed event objects, which may be:
                - Typed events (e.g., ResponseTextDeltaEvent, FunctionCallEvent)
                - TimeoutEvent: When no message received within 0.1 seconds
                - UnknownEvent: For unrecognized or error events

        Raises:
            RuntimeError: If not connected to the API when called, or if the
                WebSocket connection closes unexpectedly during operation.
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
