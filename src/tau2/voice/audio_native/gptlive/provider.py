"""GPT-Live API (alpha) provider for full-duplex voice processing.

CONFIDENTIAL: gpt-live is a limited-access alpha ("internal testing only").
The API surface differs from the OpenAI Realtime API:

- Endpoint is /v1/live and requires the OpenAI-Alpha header.
- The client sends session.update as the FIRST event; the server replies
  with session.started (there is no server-initiated session.created).
- Audio is raw headerless 24kHz mono PCM16LE, base64-encoded, both ways.
- Tools are configured via Responses delegation
  (session.delegation.responses.tools); function-call results are returned
  with delegation.function_call_output.create. Clients must NOT send
  response.create.
- Graceful close: send session.close, drain until session.closed.
"""

import asyncio
import base64
import json
import os
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger

from tau2.config import (
    DEFAULT_GPTLIVE_BASE_URL,
    DEFAULT_GPTLIVE_DELEGATION_MODEL,
    DEFAULT_GPTLIVE_MODEL,
    DEFAULT_GPTLIVE_VOICE,
    GPTLIVE_ALPHA_HEADER_NAME,
    GPTLIVE_ALPHA_HEADER_VALUE,
)
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.gptlive.events import (
    BaseGPTLiveEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_gptlive_event,
)

load_dotenv()

# How long to wait for session.closed after sending session.close.
# Server-side graceful shutdown has a 10s maximum; we bound much lower
# because the adapter's total disconnect timeout is 5s (drain + close
# handshake must both fit).
_CLOSE_DRAIN_TIMEOUT_SECONDS = 2.0
_CLOSE_HANDSHAKE_TIMEOUT_SECONDS = 2.0


class GPTLiveProvider:
    """GPT-Live API client with WebSocket-based communication.

    Manages a persistent WebSocket connection to the /v1/live endpoint.
    The model is full-duplex: it continuously processes input while
    generating output, so there is no VAD configuration and no explicit
    turn-taking protocol.

    Tool calling uses Responses delegation: the live model delegates work
    to a configured Responses API backend model, which emits
    response.function_call_arguments.done events for client-actionable
    calls. Results go back via delegation.function_call_output.create.
    """

    BASE_URL = DEFAULT_GPTLIVE_BASE_URL
    DEFAULT_MODEL = DEFAULT_GPTLIVE_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: str = DEFAULT_GPTLIVE_VOICE,
        delegation_model: str = DEFAULT_GPTLIVE_DELEGATION_MODEL,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize the GPT-Live provider.

        Args:
            api_key: OpenAI project API key. Falls back to OPENAI_API_KEY.
            model: Live model slug. Defaults to DEFAULT_GPTLIVE_MODEL.
            voice: Output voice (immutable after session start).
            delegation_model: Responses API model that executes delegated
                work (tool selection/calling). Note: calls delegated to the
                backend text model ARE billed, unlike live-model traffic.
            reasoning_effort: Optional reasoning effort for the delegation
                model ("minimal", "low", "medium", "high"). Not sent if None.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")

        self.model = model or self.DEFAULT_MODEL
        self.voice = voice
        self.delegation_model = delegation_model
        self.reasoning_effort = reasoning_effort
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self._event_counter = 0

    @property
    def is_connected(self) -> bool:
        """True if the WebSocket is in OPEN state."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    def _next_event_id(self, prefix: str) -> str:
        """Generate a client event_id (used by the server for error correlation)."""
        self._event_counter += 1
        return f"event_{prefix}_{self._event_counter}"

    @websocket_retry
    async def connect(self) -> None:
        """Open the WebSocket connection.

        The model is selected via the URL query string. Unlike the Realtime
        API, the server does not send anything until the client supplies
        startup configuration — call configure_session() next.
        """
        if self.is_connected:
            return

        url = f"{self.BASE_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            GPTLIVE_ALPHA_HEADER_NAME: GPTLIVE_ALPHA_HEADER_VALUE,
        }

        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            close_timeout=_CLOSE_HANDSHAKE_TIMEOUT_SECONDS,
        )
        logger.info(f"GPT-Live API: WebSocket connected (model={self.model})")

    async def disconnect(self) -> None:
        """Gracefully close the session.

        Sends session.close and drains events until session.closed (bounded),
        then closes the transport. Safe to call when not connected.
        """
        if self.ws is None:
            return

        if self.is_connected:
            try:
                await self.ws.send(
                    json.dumps(
                        {
                            "type": "session.close",
                            "event_id": self._next_event_id("close"),
                        }
                    )
                )
                await asyncio.wait_for(
                    self._drain_until_closed(), timeout=_CLOSE_DRAIN_TIMEOUT_SECONDS
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                logger.warning(
                    "GPT-Live API: did not observe session.closed before timeout; "
                    "closing transport"
                )
            except Exception as e:
                logger.warning(f"GPT-Live API: error during graceful close: {e}")

        await self.ws.close()
        self.ws = None
        logger.info("GPT-Live API: WebSocket connection closed")

    async def _drain_until_closed(self) -> None:
        """Read events until session.closed arrives."""
        while True:
            raw_message = await self.ws.recv()
            data = json.loads(raw_message)
            if data.get("type") == "session.closed":
                logger.info(
                    f"GPT-Live API: session closed "
                    f"(reason={data.get('reason')}, usage={data.get('usage')})"
                )
                return

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for delegation.responses.tools.

        The format matches ordinary Responses API function tool entries
        (same shape as the Realtime API's flattened function tools).
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
        vad_config: object = None,
        modality: str = "audio",
    ) -> None:
        """Send startup configuration and wait for session.started.

        The initial session.update must be the first client event. The model
        must NOT be repeated in the session object (it comes from the URL).
        Instructions and voice are immutable after initialization.

        Args:
            system_prompt: System instructions. Passed both to the live model
                and to the Responses delegation backend (which selects and
                executes tools, so it needs the policy too).
            tools: Tools exposed via Responses delegation.
            vad_config: Ignored — gpt-live is full-duplex and has no VAD.
            modality: Must be "audio" (gpt-live has no text-only mode).

        Raises:
            RuntimeError: If not connected or if configuration fails.
            ValueError: If a non-audio modality is requested.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality != "audio":
            raise ValueError(
                f"GPT-Live only supports 'audio' modality, got '{modality}'"
            )

        if vad_config is not None:
            logger.warning(
                "GPT-Live is full-duplex and has no VAD configuration; "
                "ignoring vad_config"
            )

        responses_config: Dict = {
            "model": self.delegation_model,
            "instructions": system_prompt,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
        }
        if self.reasoning_effort is not None:
            responses_config["reasoning"] = {"effort": self.reasoning_effort}

        session = {
            "instructions": system_prompt,
            "audio": {"output": {"voice": self.voice}},
            "delegation": {
                "type": "responses",
                "responses": responses_config,
            },
        }

        await self.ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": self._next_event_id("start"),
                    "session": session,
                }
            )
        )

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.started":
                session_data = data.get("session", {})
                self.session_id = session_data.get("id")
                logger.info(
                    f"GPT-Live API: session started (session_id={self.session_id})"
                )
                return
            elif event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    f"Session configuration failed: "
                    f"{error.get('code')}: {error.get('message', 'Unknown error')}"
                )

    async def send_audio(self, audio_data: bytes) -> None:
        """Append raw 24kHz mono PCM16LE audio to the session.

        The server does not acknowledge raw audio events. The decoded payload
        must be non-empty and contain an even number of bytes (2-byte samples).

        Args:
            audio_data: Raw PCM16LE bytes at 24kHz mono (no container/header).

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        if len(audio_data) == 0:
            return
        if len(audio_data) % 2 != 0:
            # PCM16 samples are 2 bytes; the API rejects odd payloads.
            audio_data = audio_data[:-1]
            if len(audio_data) == 0:
                return

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Return the result for one client-actionable function call.

        Unlike the Realtime API, the client must NOT send response.create —
        the Responses delegation lifecycle continues on its own after the
        result is accepted (delegation.function_call_output.created).

        Args:
            call_id: The outstanding call_id from
                response.function_call_arguments.done.
            result: Function output (JSON string or plain text).
            request_response: Ignored (no response.create in the Live API).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        message = {
            "type": "delegation.function_call_output.create",
            "event_id": self._next_event_id("function_output"),
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self.ws.send(json.dumps(message))

    async def receive_events(self) -> AsyncGenerator[BaseGPTLiveEvent, None]:
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
                yield parse_gptlive_event(data)

            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    f"GPT-Live API: WebSocket connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    f"WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except Exception as e:
                logger.error(
                    f"GPT-Live API: Error receiving event: {type(e).__name__}: {e}"
                )
                yield UnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseGPTLiveEvent]:
        """Collect all non-timeout events within the time window."""
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
