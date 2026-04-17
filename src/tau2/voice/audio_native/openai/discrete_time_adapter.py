"""Discrete-time adapter for OpenAI Realtime API.

Provides a tick-based interface for the OpenAI Realtime API using the
shared DiscreteTimeAdapter template method for tick lifecycle management.

The only OpenAI-specific behavior is in _process_event: on user
interruption (SpeechStartedEvent), the adapter calls truncate_item()
on the OpenAI API to inform the server how much audio was played.

Usage:
    adapter = DiscreteTimeOpenAIAdapter(
        tick_duration_ms=200,
        send_audio_instant=False,
    )
    adapter.connect(system_prompt, tools, vad_config)

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)

    adapter.disconnect()
"""

import asyncio
import base64
import json
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_OPENAI_VAD_THRESHOLD,
)
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
    OpenAIVADMode,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimeOpenAIAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time simulation with OpenAI Realtime API.

    Uses the shared DiscreteTimeAdapter template for tick lifecycle.
    OpenAI-specific behavior: truncate_item on interruption.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[OpenAIRealtimeProvider] = None,
        audio_format: Optional[AudioFormat] = None,
    ):
        super().__init__(
            tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )

        self._chunk_size = int(
            self.audio_format.bytes_per_second * self._voip_interval_ms / 1000
        )

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self._provider = provider
        self._owns_provider = provider is None

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> OpenAIRealtimeProvider:
        if self._provider is None:
            self._provider = OpenAIRealtimeProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        if vad_config is None:
            vad_config = OpenAIVADConfig(
                mode=OpenAIVADMode.SERVER_VAD,
                threshold=DEFAULT_OPENAI_VAD_THRESHOLD,
            )

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeOpenAIAdapter connected to OpenAI Realtime API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to OpenAI Realtime API: {e}") from e

    async def _async_connect(self, system_prompt, tools, vad_config, modality) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
            audio_format=self.audio_format,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        logger.info("DiscreteTimeOpenAIAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to OpenAI Realtime API. Call connect() first."
            )

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    async def _flush_pending_tool_results(self) -> None:
        for (
            call_id,
            result_str,
            request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str, request_response)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """OpenAI-specific: send audio, receive events, process events."""

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                user_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            await self._process_event(result, event)

    async def _process_event(self, result: TickResult, event: Any) -> None:
        """Process an OpenAI Realtime event."""
        result.events.append(event)

        if isinstance(event, AudioDeltaEvent):
            item_id = getattr(event, "item_id", None)

            if result.skip_item_id is not None:
                if item_id == result.skip_item_id:
                    result.truncated_audio_bytes += len(base64.b64decode(event.delta))
                    return
                else:
                    result.skip_item_id = None

            decoded = base64.b64decode(event.delta)
            result.agent_audio_chunks.append((decoded, item_id))

            if item_id:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_audio(len(decoded))

        elif isinstance(event, AudioTranscriptDeltaEvent):
            item_id = getattr(event, "item_id", None)
            if item_id and event.delta:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.delta)

        elif isinstance(event, SpeechStartedEvent):
            logger.debug(f"Speech started detected at {event.audio_start_ms}ms")
            result.vad_events.append("speech_started")

            has_agent_audio = result.agent_audio_chunks or self._buffered_agent_audio
            if has_agent_audio:
                last_item_id = None
                if result.agent_audio_chunks:
                    last_item_id = result.agent_audio_chunks[-1][1]
                elif self._buffered_agent_audio:
                    last_item_id = self._buffered_agent_audio[-1][1]

                if self._buffered_agent_audio:
                    buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                    result.truncated_audio_bytes += buffered_bytes
                    self._buffered_agent_audio.clear()

                audio_start_ms = (
                    event.audio_start_ms if event.audio_start_ms is not None else 0
                )
                result.truncate_agent_audio(
                    item_id=last_item_id,
                    audio_start_ms=audio_start_ms,
                    cumulative_user_audio_at_tick_start_ms=result.cumulative_user_audio_at_tick_start_ms,
                    bytes_per_tick=result.bytes_per_tick,
                )

                # OpenAI-specific: tell server to truncate and cancel
                if last_item_id is not None:
                    played = result.get_played_agent_audio()
                    audio_end_ms = int(
                        len(played) / self.audio_format.bytes_per_second * 1000
                    )
                    await self.provider.truncate_item(
                        item_id=last_item_id,
                        content_index=0,
                        audio_end_ms=audio_end_ms,
                    )

        elif isinstance(event, SpeechStoppedEvent):
            logger.debug(f"Speech stopped detected at {event.audio_end_ms}ms")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            if event.call_id and event.name:
                try:
                    arguments = json.loads(event.arguments) if event.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=event.call_id,
                    name=event.name,
                    arguments=arguments,
                )
                result.tool_calls.append(tool_call)
                logger.debug(f"Tool call detected: {event.name}({event.call_id})")

        elif isinstance(event, AudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, AudioTranscriptDoneEvent):
            logger.debug(f"Transcript done for item {event.item_id}")

        elif isinstance(event, ResponseDoneEvent):
            logger.debug("Response done")

        else:
            logger.debug(f"Event {event.type} received")
