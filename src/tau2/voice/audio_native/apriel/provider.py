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
from pydantic import ValidationError

from tau2.config import (
    DEFAULT_APRIEL_MODEL,
    DEFAULT_APRIEL_WEBSOCKET_MAX_SIZE_BYTES,
    DEFAULT_OPENAI_NOISE_REDUCTION,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.apriel.connection import (
    normalize_realtime_url,
    resolve_credentials,
    websocket_request,
)
from tau2.voice.audio_native.apriel.events import parse_realtime_event
from tau2.voice.audio_native.openai.events import BaseRealtimeEvent, TimeoutEvent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false (got {value!r}).")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number (got {value!r}).") from exc


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {value!r}).") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1 (got {value!r}).")
    return parsed


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            line = handle.readline()
    except OSError:
        return None
    value = line.strip()
    return value or None


def _env_file_secret(name: str) -> Optional[str]:
    path = os.environ.get(name)
    if not path:
        return None
    return _read_first_line(path)


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

    DEFAULT_MODEL = DEFAULT_APRIEL_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        pre_tool_speech: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace_enabled: bool = True,
        websocket_max_size: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        cloud_auth_token: Optional[str] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        if not self.model:
            raise ValueError(
                "Apriel model not provided. Set APRIEL_MODEL_NAME in .env."
            )

        resolved_url = base_url or os.environ.get("APRIEL_REALTIME_BASE_URL")
        if not resolved_url:
            raise ValueError(
                "Apriel realtime URL not provided. Set APRIEL_REALTIME_BASE_URL."
            )
        self.base_url = normalize_realtime_url(resolved_url)
        self.api_key, self.cloud_auth_token = resolve_credentials(
            api_key, cloud_auth_token
        )

        resolved_thinking = (
            _env_bool("APRIEL_ENABLE_THINKING", True)
            if enable_thinking is None
            else enable_thinking
        )
        resolved_pre_tool_speech = (
            _env_bool("APRIEL_PRE_TOOL_SPEECH", True)
            if pre_tool_speech is None
            else pre_tool_speech
        )
        resolved_temperature = (
            _env_float("APRIEL_TEMPERATURE", 1.0)
            if temperature is None
            else temperature
        )
        resolved_max_tokens = (
            _env_int("APRIEL_MAX_TOKENS", 8192) if max_tokens is None else max_tokens
        )

        self.reasoning_effort = reasoning_effort

        self._session_parameters: dict = {
            "enable_thinking": resolved_thinking,
            "pre_tool_speech": resolved_pre_tool_speech,
            "temperature": resolved_temperature,
            "max_output_tokens": resolved_max_tokens,
        }

        resolved_trace = trace_enabled or _env_bool("APRIEL_TRACE", False)
        if resolved_trace:
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
        self.effective_session: dict = {}
        self._configured = False

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

        url, options = websocket_request(
            self.base_url, self.model, self.api_key, self.cloud_auth_token
        )
        self.ws = await websockets.connect(
            url, max_size=self._websocket_max_size, **options
        )

        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        session_data = data.get("session", {})
        self.session_id = session_data.get("id")
        logger.info(f"Apriel realtime: session created (session_id={self.session_id})")

    async def disconnect(self) -> None:
        self._configured = False
        self.effective_session = {}
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

        self._configured = False
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

        session["apriel"] = {
            **session.get("apriel", {}),
            "playback_clock": "media",
            "turn_clock": "media",
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
            voice = os.environ.get("APRIEL_VOICE")
            if voice:
                audio_output["voice"] = voice
            session.setdefault("audio", {})["output"] = audio_output

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")
            if event_type == "session.updated":
                self.effective_session = data.get("session", {})
                canonical = self.effective_session.get("apriel")
                legacy = self.effective_session.get("aurion")
                if canonical is not None and legacy is not None and canonical != legacy:
                    raise RuntimeError(
                        "Apriel session clock/configuration echoes conflict"
                    )
                clocks = canonical if canonical is not None else legacy
                if not isinstance(clocks, dict):
                    raise RuntimeError(
                        "Apriel session clock acknowledgement is missing"
                    )
                if any(
                    clocks.get(key) != "media"
                    for key in ("playback_clock", "turn_clock")
                ):
                    raise RuntimeError(
                        "Apriel requires verified media playback_clock and turn_clock"
                    )
                self._configured = True
                logger.info(f"Apriel effective session clocks: {clocks}")
                break
            if event_type == "error":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Session configuration failed: {error_msg}")

    async def send_audio(self, audio_data: bytes) -> None:
        if not self._configured:
            raise RuntimeError(
                "Apriel media clocks have not been configured and verified"
            )
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
            await self.request_response()

    async def request_response(self) -> None:
        """Continue using the full server conversation, including all tool results."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
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

        while True:
            if not self.is_connected:
                raise RuntimeError("Apriel connection closed while receiving events")
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(raw_message)
                yield parse_realtime_event(data)

            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except websockets.ConnectionClosed as e:
                code = (
                    e.code if isinstance(e.code, int) and 0 <= e.code <= 4999 else None
                )
                message = (
                    f"Apriel WebSocket connection closed unexpectedly (code={code})"
                )
                logger.error(message)
                raise RuntimeError(message) from None
            except Exception as e:
                category = (
                    "invalid_json"
                    if isinstance(e, json.JSONDecodeError)
                    else "invalid_text_encoding"
                    if isinstance(e, UnicodeDecodeError)
                    else "invalid_event_schema"
                    if isinstance(e, ValidationError)
                    else "invalid_event_shape"
                    if isinstance(e, (ValueError, TypeError, AttributeError))
                    else "receive_failure"
                )
                message = f"Apriel error receiving or decoding event ({category})"
                logger.error(message)
                raise RuntimeError(message) from None

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
