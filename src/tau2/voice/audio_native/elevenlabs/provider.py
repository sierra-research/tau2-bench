"""WebSocket client for the ElevenLabs Agents Platform.

An ElevenLabs agent owns its LLM, voice, ASR, turn-taking, and client-tool
configuration. tau2 supplies the benchmark system prompt at conversation start,
streams audio, and executes client-tool calls through the environment.

References:
https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets
https://elevenlabs.io/docs/eleven-agents/customization/events/client-events
"""

import asyncio
import base64
import json
import os
from typing import AsyncGenerator, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
import websockets
from dotenv import load_dotenv
from loguru import logger

from tau2.config import (
    DEFAULT_ELEVENLABS_API_BASE_URL,
    DEFAULT_ELEVENLABS_CONVERSATION_BASE_URL,
    ELEVENLABS_AGENT_ID_PREFIXES,
)
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.elevenlabs.events import (
    BaseElevenLabsEvent,
    ElevenLabsClientErrorEvent,
    ElevenLabsConversationMetadataEvent,
    ElevenLabsPingEvent,
    ElevenLabsTimeoutEvent,
    ElevenLabsUnknownEvent,
    parse_elevenlabs_event,
)

load_dotenv()

_SUPPORTED_ENCODINGS = {"pcm", "ulaw"}
_SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


def parse_elevenlabs_audio_format(value: str) -> Tuple[str, int]:
    """Parse an ElevenLabs audio format such as ``pcm_16000``."""
    try:
        encoding, sample_rate_text = value.rsplit("_", 1)
        sample_rate = int(sample_rate_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported ElevenLabs audio format: {value!r}") from error

    if (
        encoding not in _SUPPORTED_ENCODINGS
        or sample_rate not in _SUPPORTED_SAMPLE_RATES
        or (encoding == "ulaw" and sample_rate != 8000)
    ):
        raise ValueError(f"Unsupported ElevenLabs audio format: {value!r}")
    return encoding, sample_rate


class ElevenLabsAgentsProvider:
    """Manage one realtime conversation with a hosted ElevenLabs agent."""

    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        api_base_url: str = DEFAULT_ELEVENLABS_API_BASE_URL,
        conversation_base_url: str = DEFAULT_ELEVENLABS_CONVERSATION_BASE_URL,
    ):
        """Initialize the provider.

        ``agent_id`` is the ``agent_...`` identifier from ElevenLabs. When an
        API key is available, a signed URL is requested so private agents work.
        Without one, the public-agent WebSocket endpoint is used.
        """
        self.agent_id = agent_id.strip()
        if not self.agent_id.startswith(ELEVENLABS_AGENT_ID_PREFIXES):
            raise ValueError(
                "ElevenLabs requires an agent ID beginning with 'agent_' or "
                "'seng_'. Pass it with --audio-native-model <agent_id>."
            )

        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.api_base_url = api_base_url.rstrip("/")
        self.conversation_base_url = conversation_base_url
        self.ws = None
        self.session_id: Optional[str] = None
        self.user_input_audio_format: Optional[str] = None
        self.agent_output_audio_format: Optional[str] = None
        self._prefetched_events: List[BaseElevenLabsEvent] = []
        self._pong_tasks: set[asyncio.Task] = set()

    @property
    def is_connected(self) -> bool:
        """Whether the conversation WebSocket is open."""
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    async def _get_websocket_url(self) -> str:
        if not self.api_key:
            query = urlencode({"agent_id": self.agent_id})
            return f"{self.conversation_base_url}?{query}"

        url = f"{self.api_base_url}/v1/convai/conversation/get-signed-url"
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                params={"agent_id": self.agent_id},
                headers={"xi-api-key": self.api_key},
            )
            response.raise_for_status()
            payload = response.json()

        signed_url = payload.get("signed_url")
        if not signed_url:
            raise RuntimeError("ElevenLabs signed-URL response omitted signed_url")
        return signed_url

    @staticmethod
    def _build_initiation_payload(system_prompt: str) -> dict:
        """Build the per-conversation prompt and first-message overrides."""
        return {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "agent": {
                    "first_message": "",
                    "prompt": {"prompt": system_prompt},
                }
            },
        }

    @websocket_retry
    async def connect(self, system_prompt: str) -> None:
        """Connect and wait for negotiated audio-format metadata."""
        if self.is_connected:
            return

        websocket_url = await self._get_websocket_url()
        logger.info(f"ElevenLabs Agents: connecting agent_id={self.agent_id}")
        try:
            self.ws = await websockets.connect(websocket_url)
            await self.ws.send(
                json.dumps(self._build_initiation_payload(system_prompt))
            )

            while True:
                raw_message = await self.ws.recv()
                event = parse_elevenlabs_event(json.loads(raw_message))

                if isinstance(event, ElevenLabsPingEvent):
                    self._schedule_pong(event)
                    continue
                if isinstance(event, ElevenLabsClientErrorEvent):
                    message = event.message or event.error_name or "unknown error"
                    raise RuntimeError(
                        "ElevenLabs conversation initialization failed: "
                        f"{message}. Enable System prompt and First message "
                        "overrides for this agent."
                    )
                if isinstance(event, ElevenLabsConversationMetadataEvent):
                    parse_elevenlabs_audio_format(event.user_input_audio_format)
                    parse_elevenlabs_audio_format(event.agent_output_audio_format)
                    self.session_id = event.conversation_id
                    self.user_input_audio_format = event.user_input_audio_format
                    self.agent_output_audio_format = event.agent_output_audio_format
                    logger.info(
                        "ElevenLabs Agents: connected "
                        f"(conversation_id={self.session_id}, "
                        f"input={self.user_input_audio_format}, "
                        f"output={self.agent_output_audio_format})"
                    )
                    return

                self._prefetched_events.append(event)
        except Exception:
            if self.ws is not None:
                await self.ws.close()
                self.ws = None
            raise

    async def disconnect(self) -> None:
        """Close the conversation WebSocket and pending pong tasks."""
        for task in self._pong_tasks:
            task.cancel()
        if self._pong_tasks:
            await asyncio.gather(*self._pong_tasks, return_exceptions=True)
        self._pong_tasks.clear()

        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        logger.info("ElevenLabs Agents: disconnected")

    async def _send_json(self, payload: dict) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected to ElevenLabs Agents API")
        await self.ws.send(json.dumps(payload))

    async def send_audio(self, audio_data: bytes) -> None:
        """Send one base64-encoded user audio chunk."""
        await self._send_json(
            {"user_audio_chunk": base64.b64encode(audio_data).decode("ascii")}
        )

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """Return a client-tool result to the hosted agent."""
        await self._send_json(
            {
                "type": "client_tool_result",
                "tool_call_id": call_id,
                "result": result,
                "is_error": is_error,
            }
        )

    def _schedule_pong(self, event: ElevenLabsPingEvent) -> None:
        task = asyncio.create_task(self._send_pong(event.event_id, event.ping_ms))
        self._pong_tasks.add(task)
        task.add_done_callback(self._pong_tasks.discard)

    async def _send_pong(self, event_id: int, ping_ms: int) -> None:
        if ping_ms > 0:
            await asyncio.sleep(ping_ms / 1000)
        if self.is_connected:
            await self._send_json({"type": "pong", "event_id": event_id})

    async def receive_events(self) -> AsyncGenerator[BaseElevenLabsEvent, None]:
        """Yield normalized conversation events as they arrive."""
        if not self.is_connected:
            raise RuntimeError("Not connected to ElevenLabs Agents API")

        while self._prefetched_events:
            yield self._prefetched_events.pop(0)

        while self.is_connected:
            try:
                raw_message = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                event = parse_elevenlabs_event(json.loads(raw_message))
                if isinstance(event, ElevenLabsPingEvent):
                    self._schedule_pong(event)
                yield event
            except asyncio.TimeoutError:
                yield ElevenLabsTimeoutEvent()
            except websockets.ConnectionClosed as error:
                raise RuntimeError(
                    "ElevenLabs conversation WebSocket closed unexpectedly "
                    f"(code={error.code}, reason={error.reason!r})"
                ) from error
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning(f"Invalid ElevenLabs WebSocket event: {error}")
                yield ElevenLabsUnknownEvent(raw={"error": str(error)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseElevenLabsEvent]:
        """Collect non-timeout events for one simulation tick."""
        events: List[BaseElevenLabsEvent] = []
        end_time = asyncio.get_running_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, ElevenLabsTimeoutEvent):
                events.append(event)
            if asyncio.get_running_loop().time() >= end_time:
                break
        return events
