"""Thin WebSocket transport for the Launa v1 hosted voice service."""

import asyncio
import json
import os
from typing import AsyncGenerator, List, Optional

import websockets
from loguru import logger
from pydantic import JsonValue, TypeAdapter

from tau2.config import DEFAULT_LAUNA_MODEL, LAUNA_SAMPLE_RATE
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseCancelledEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)

from .config import validate_endpoint

_MESSAGE = TypeAdapter(dict[str, JsonValue])


class _NoRedirectConnect(websockets.connect):
    """Never forward the bearer credential to a redirect destination."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


LAUNA_AUDIO_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=LAUNA_SAMPLE_RATE,
)


class LaunaRealtimeWSProvider(OpenAIRealtimeProvider):
    """Client for a Launa service exposed through an OAI Realtime-style socket.

    Unlike OpenAI's hosted endpoint, the Launa server expects the client
    to send ``session.update`` before it emits ``session.created``. It also owns
    turn-taking and response generation, and fixes both audio directions to
    mono 24 kHz PCM16.
    """

    DEFAULT_MODEL = DEFAULT_LAUNA_MODEL

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        allow_insecure_local: bool = False,
    ) -> None:
        selected_key = api_key or os.environ.get("LAUNA_API_KEY")
        if not selected_key:
            raise ValueError("LAUNA_API_KEY is required")
        if len(selected_key) > 4096 or not all(
            33 <= ord(c) <= 126 for c in selected_key
        ):
            raise ValueError("API key must be an ASCII token without whitespace")
        selected_endpoint = endpoint or os.environ.get("LAUNA_WS_URL", "")
        self.endpoint = validate_endpoint(selected_endpoint, allow_insecure_local)
        super().__init__(
            api_key=selected_key,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self._send_authorization = True
        self._audio_format = LAUNA_AUDIO_FORMAT

    @websocket_retry
    async def connect(self) -> None:
        """Open the socket; session negotiation happens in configure_session."""
        if self.is_connected:
            return

        headers = None
        if self._send_authorization:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        self.ws = await _NoRedirectConnect(
            self.endpoint,
            additional_headers=headers,
            max_size=None,
        )

    async def disconnect(self) -> None:
        """Close the Launa WebSocket connection."""
        if self.ws is not None:
            logger.info("Launa realtime WS: disconnecting")
            await self.ws.close()
            self.ws = None

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: OpenAIVADConfig,
        modality: str = "audio",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """Send the client-first session configuration and await creation."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")
        if modality != "audio":
            raise ValueError("launa supports only modality='audio'")

        # ``audio_format`` is the simulation-side format. the hosted voice service accepts
        # continuous PCM16/24 kHz and performs server-side VAD/endpointing.
        del audio_format
        formatted_tools = []
        for tool in self._format_tools_for_api(tools):
            formatted_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            )
        session_update = {
            "type": "session.update",
            "instructions": system_prompt,
            "tools": formatted_tools,
            "audio": {
                "input": {
                    "type": "audio/pcm",
                    "sample_rate_hz": LAUNA_SAMPLE_RATE,
                }
            },
        }

        assert self.ws is not None
        await self.ws.send(json.dumps(session_update))

        while True:
            data = _MESSAGE.validate_json(await self.ws.recv())
            event_type = data.get("type", "")
            if event_type == "latency.ping":
                await self._send_latency_pong(data)
            elif event_type in {"session.created", "session.updated"}:
                session_data = data.get("session", {})
                self.session_id = session_data.get("id")
                self._current_vad_config = vad_config
                logger.info(
                    f"Launa realtime WS: session created (session_id={self.session_id})"
                )
                return
            elif event_type == "error":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def _send_latency_pong(self, data: dict[str, JsonValue]) -> None:
        """Acknowledge the harness latency probe."""
        assert self.ws is not None
        await self.ws.send(json.dumps({"type": "latency.pong", "id": data.get("id")}))

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Return a tool result; the Launa service resumes automatically."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        del request_response
        assert self.ws is not None
        await self.ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }
            )
        )

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Skip unsupported client truncation; the harness owns barge-in."""
        logger.debug(
            "Launa realtime WS owns response truncation "
            f"(item_id={item_id}, content_index={content_index}, "
            f"audio_end_ms={audio_end_ms})"
        )

    async def receive_events(self) -> AsyncGenerator[BaseRealtimeEvent, None]:
        """Yield OAI realtime events while handling harness control messages."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        assert self.ws is not None
        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = _MESSAGE.validate_json(raw_message)
                event_type = data.get("type")
                if event_type == "latency.ping":
                    await self._send_latency_pong(data)
                    continue
                if event_type == "agent-turn.tool-call.fired":
                    yield FunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        call_id=data.get("tool_call_id"),
                        name=data.get("name"),
                        arguments=data.get("arguments") or "{}",
                    )
                    continue
                if event_type == "agent-turn.interrupted":
                    yield ResponseCancelledEvent(type="response.cancelled")
                    continue
                yield parse_realtime_event(data)
            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as exc:
                raise RuntimeError(
                    "Launa realtime WebSocket closed unexpectedly "
                    f"(code={exc.code}, reason='{exc.reason or 'no reason provided'}')"
                ) from exc
            except Exception as exc:
                logger.error(
                    f"Launa realtime WS receive error: {type(exc).__name__}: {exc}"
                )
                yield UnknownEvent(type="error", raw={"error": str(exc)})
