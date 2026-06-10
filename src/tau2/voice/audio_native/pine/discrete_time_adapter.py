"""Discrete-time tick adapter for Pine.

Structurally identical to `tau2.voice.audio_native.openai.discrete_time_adapter`,
which is the right reference because Pine and OpenAI Realtime share a
wire-compatible event taxonomy. The only Pine-specific behavior is that this
adapter routes through PineProvider (different endpoint, auth, and
session-config fields).
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
from tau2.voice.audio_native.pine.provider import (
    PineProvider,
    PineVADConfig,
    PineVADMode,
    DEFAULT_PINE_VAD_THRESHOLD,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimePineAdapter(DiscreteTimeAdapter):
    """Tick-based adapter for the Pine protocol."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        provider: Optional[PineProvider] = None,
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
            raise ValueError("model and provider cannot both be supplied")

        self.model = model
        self.voice = voice
        self._provider = provider
        self._owns_provider = provider is None

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> PineProvider:
        if self._provider is None:
            self._provider = PineProvider(model=self.model, voice=self.voice)
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
            logger.warning("Pine: already connected; reconnecting")
            self.disconnect()

        # Coerce any incoming vad_config to a PineVADConfig. The orchestrator
        # may pass another provider's VAD config (e.g. NovaVADConfig) when no
        # provider-specific override is set.
        if not isinstance(vad_config, PineVADConfig):
            vad_config = PineVADConfig(
                mode=PineVADMode.SERVER_VAD,
                threshold=getattr(vad_config, "threshold", DEFAULT_PINE_VAD_THRESHOLD)
                if vad_config is not None
                else DEFAULT_PINE_VAD_THRESHOLD,
                prefix_padding_ms=getattr(vad_config, "prefix_padding_ms", 300)
                if vad_config is not None
                else 300,
                silence_duration_ms=getattr(vad_config, "silence_duration_ms", 500)
                if vad_config is not None
                else 500,
            )

        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"Pine adapter connected (tick={self.tick_duration_ms}ms, "
                f"bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"Pine connect failed: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Pine connect failed: {e}") from e

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
                logger.warning(f"Pine disconnect: {e}")
        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        logger.info("Pine adapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Pine adapter not connected")
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
            logger.error(f"Pine run_tick (tick={tick_number}): {e}")
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
        """Send user audio, receive Pine events for the tick window, process them."""

        async def receive_events():
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed)
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
        """Handle a single Pine event (mirrors OpenAI adapter behavior).

        Load-bearing branches:
        - AudioDeltaEvent -> buffer assistant audio (skip if item was truncated)
        - AudioTranscriptDeltaEvent -> accumulate transcript
        - SpeechStartedEvent -> barge-in: clear buffers, call truncate_item
        - SpeechStoppedEvent -> record VAD edge
        - FunctionCallArgumentsDoneEvent -> emit ToolCall

        Done/end events are logged. Unknown events are also logged for forward compat.
        """
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
            logger.debug(
                f"Pine: speech_started @ {getattr(event, 'audio_start_ms', None)}ms"
            )
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

        elif isinstance(event, SpeechStoppedEvent):
            logger.debug(
                f"Pine: speech_stopped @ {getattr(event, 'audio_end_ms', None)}ms"
            )
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
                logger.debug(f"Pine: tool call {event.name}({event.call_id})")

        elif isinstance(event, AudioDoneEvent):
            logger.debug(f"Pine: audio done item={event.item_id}")
        elif isinstance(event, AudioTranscriptDoneEvent):
            logger.debug(f"Pine: transcript done item={event.item_id}")
        elif isinstance(event, ResponseDoneEvent):
            logger.debug("Pine: response done")
        else:
            logger.debug(f"Pine: event {event.type}")
