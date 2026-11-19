"""Apriel endpoint configuration for the OpenAI-compatible realtime client.

This implementation is intentionally aligned with tau's OpenAI realtime provider:
- tick-by-tick audio streaming via input_audio_buffer.append
- OpenAI-compatible realtime events and tool calling

The only differences are configuration (APRIEL_* env vars) and optional session
parameters.
"""

import asyncio
import base64
import json
import os
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_APRIEL_MODEL,
    DEFAULT_APRIEL_WEBSOCKET_MAX_SIZE_BYTES,
    DEFAULT_OPENAI_NOISE_REDUCTION,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)


def _audio_format_to_gateway(audio_format: AudioFormat) -> str:
    enc = str(audio_format.encoding).lower()
    if enc.endswith("ulaw"):
        return "g711_ulaw"
    if enc.endswith("alaw"):
        return "g711_alaw"
    if "pcm" in enc:
        return "pcm16"
    raise ValueError(f"Unsupported audio encoding: {audio_format.encoding}")


class AprielRealtimeProvider:
    """OpenAI-wire-compatible realtime provider configured via APRIEL_* env vars."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = True,
        pre_tool_speech: bool = True,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        trace_enabled: bool = True,
        websocket_max_size: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.model = model or DEFAULT_APRIEL_MODEL
        if not self.model:
            raise ValueError(
                "Apriel model not provided. Set APRIEL_MODEL_NAME in .env."
            )

        resolved_url = base_url or os.environ.get("APRIEL_REALTIME_BASE_URL")
        if not resolved_url:
            raise ValueError(
                "Apriel realtime URL not provided. Set APRIEL_REALTIME_BASE_URL."
            )
        if not resolved_url.startswith(("ws://", "wss://")):
            raise ValueError(
                "APRIEL_REALTIME_BASE_URL must start with ws:// or wss://."
            )
        self.base_url = resolved_url

        resolved_key = api_key or os.environ.get("APRIEL_API_KEY")
        if not resolved_key:
            raise ValueError("Apriel API key not provided. Set APRIEL_API_KEY.")
        self.api_key = resolved_key

        self.reasoning_effort = reasoning_effort
        self._session_parameters: dict = {
            "enable_thinking": enable_thinking,
            "pre_tool_speech": pre_tool_speech,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if trace_enabled:
            self._session_parameters["apriel"] = {
                "extensions": True,
                "vad_level": True,
                "trace_vllm_payloads": True,
            }

        self._websocket_max_size = (
            DEFAULT_APRIEL_WEBSOCKET_MAX_SIZE_BYTES
            if websocket_max_size is None
            else websocket_max_size
        )

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        return self._audio_format

    async def connect(self) -> None:
        if self.is_connected:
            return

        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        self.ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=self._websocket_max_size,
        )

        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        session_data = data.get("session", {})
        self.session_id = session_data.get("id")
        logger.info(f"Apriel realtime: session created (session_id={self.session_id})")

    async def disconnect(self) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.close()
        finally:
            self.ws = None

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
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
        vad_config,
        modality: str = "text",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality == "text":
            modalities = ["text"]
        elif modality == "audio":
            modalities = ["audio", "text"]
        elif modality == "audio_in_text_out":
            modalities = ["text"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT
        self._audio_format = audio_format
        audio_fmt = _audio_format_to_gateway(audio_format)

        session: dict = {
            "type": "realtime",
            "instructions": system_prompt,
            "output_modalities": modalities,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
            **self._session_parameters,
        }

        if self.reasoning_effort is not None:
            session["reasoning"] = {"effort": self.reasoning_effort}

        if modality in ("audio", "audio_in_text_out"):
            # Match the source behavior: the gateway owns endpointing/VAD.
            # Do not send turn_detection.
            session["audio"] = {
                "input": {
                    "format": audio_fmt,
                    "noise_reduction": {"type": DEFAULT_OPENAI_NOISE_REDUCTION},
                },
            }

        if modality == "audio":
            audio_output: dict = {"format": audio_fmt}
            session.setdefault("audio", {})["output"] = audio_output

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")
            if event_type == "session.updated":
                break
            if event_type == "error":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def send_audio(self, audio_data: bytes) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        await self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
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
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }
        await self.ws.send(json.dumps(truncate_event))

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
                yield parse_realtime_event(data)

            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                logger.error(
                    "Apriel realtime: WebSocket connection closed "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                )
                raise RuntimeError(
                    "WebSocket connection closed unexpectedly "
                    f"(code={e.code}, reason='{e.reason or 'no reason provided'}')"
                ) from e
            except Exception as e:
                logger.error(
                    f"Apriel realtime: Error receiving event: {type(e).__name__}: {e}"
                )
                yield UnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseRealtimeEvent]:
        """Receive events for a specified duration.

        Collects all non-timeout events within the time window.
        """
        events: List[BaseRealtimeEvent] = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events
