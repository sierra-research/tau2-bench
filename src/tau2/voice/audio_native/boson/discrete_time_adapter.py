"""Discrete-time adapter for Boson realtime voice chat."""

import asyncio
import base64
import json
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.boson.events import (
    BosonAudioDeltaEvent,
    BosonAudioDoneEvent,
    BosonAudioTranscriptDeltaEvent,
    BosonAudioTranscriptDoneEvent,
    BosonAudioTranscriptLengthEvent,
    BosonErrorEvent,
    BosonFunctionCallArgumentsDoneEvent,
    BosonInputAudioTranscriptionCompletedEvent,
    BosonResponseDoneEvent,
    BosonSpeechStartedEvent,
    BosonSpeechStoppedEvent,
    BosonTimeoutEvent,
)
from tau2.voice.audio_native.boson.provider import (
    BosonRealtimeProvider,
    BosonVADConfig,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimeBosonAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time simulation with Boson realtime voice chat.

    Boson supports 8kHz PCMU directly, so the default tau2 telephony format can
    be sent and received without provider-specific audio conversion.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[BosonRealtimeProvider] = None,
        audio_format: Optional[AudioFormat] = None,
        voice: str = "default",
    ):
        """Initialize the discrete-time Boson adapter."""
        if reasoning_effort is not None:
            raise ValueError(
                "Boson provider does not support reasoning_effort "
                f"(got '{reasoning_effort}')"
            )
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
        self.voice = voice
        self._provider = provider
        self._owns_provider = provider is None

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> BosonRealtimeProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = BosonRealtimeProvider(model=self.model, voice=self.voice)
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Return True if the adapter is connected."""
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to Boson and configure the session."""
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        if vad_config is None:
            vad_config = BosonVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeBosonAdapter connected to Boson realtime "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Boson realtime: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Boson realtime: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: BosonVADConfig,
        modality: str,
    ) -> None:
        """Async connection and session configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
            audio_format=self.audio_format,
        )

    def disconnect(self) -> None:
        """Disconnect and reset adapter state."""
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
        logger.info("DiscreteTimeBosonAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """Async disconnection."""
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """Run one simulation tick."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Boson realtime. Call connect() first.")

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
        """Send queued tool results to Boson."""
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
        """Send user audio, receive Boson events, and process them."""

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
        """Process a Boson event and mutate the tick result."""
        result.events.append(event)

        if isinstance(event, BosonAudioDeltaEvent):
            item_id = event.item_id or self._current_item_id

            if result.skip_item_id is not None:
                if item_id == result.skip_item_id:
                    if event.delta:
                        result.truncated_audio_bytes += len(
                            base64.b64decode(event.delta)
                        )
                    return
                result.skip_item_id = None

            if event.delta:
                decoded = base64.b64decode(event.delta)
                result.agent_audio_chunks.append((decoded, item_id))

                if item_id:
                    self._current_item_id = item_id
                    if item_id not in self._utterance_transcripts:
                        self._utterance_transcripts[item_id] = UtteranceTranscript(
                            item_id=item_id
                        )
                    self._utterance_transcripts[item_id].add_audio(len(decoded))

        elif isinstance(
            event, (BosonAudioTranscriptDeltaEvent, BosonAudioTranscriptLengthEvent)
        ):
            item_id = event.item_id or self._current_item_id
            if item_id and event.delta:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.delta)

        elif isinstance(event, BosonSpeechStartedEvent):
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

        elif isinstance(event, BosonSpeechStoppedEvent):
            logger.debug(f"Speech stopped detected at {event.audio_end_ms}ms")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, BosonFunctionCallArgumentsDoneEvent):
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

        elif isinstance(event, BosonInputAudioTranscriptionCompletedEvent):
            logger.debug(f"Input transcription: {event.transcript}")

        elif isinstance(event, BosonAudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, BosonAudioTranscriptDoneEvent):
            logger.debug(f"Transcript done for item {event.item_id}")

        elif isinstance(event, BosonResponseDoneEvent):
            logger.debug(f"Response done with status={event.status}")

        elif isinstance(event, BosonErrorEvent):
            logger.error(f"Boson error: {event.message or event.code}")

        elif isinstance(event, BosonTimeoutEvent):
            pass

        else:
            logger.debug(f"Event {event.type} received")
