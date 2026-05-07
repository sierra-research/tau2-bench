"""Discrete-time adapter for the Pipecat-based cascaded voice provider.

This adapter is a thin glue layer between the tick-based simulation
framework and ``PipecatVoiceProvider``. It mirrors the design of
``LiveKitCascadedAdapter``:

- Runs the async provider inside its own background asyncio loop.
- Converts tick-based μ-law 8kHz audio to PCM16 16kHz for the provider.
- Buffers TTS output (24kHz PCM16) into μ-law 8kHz tick-aligned chunks.
- Maps Pipecat events to ``TickResult``.

The cascaded pipeline is fundamentally event-driven and operates on its
own clock (LLM / TTS latency >> 200ms tick), so this adapter has its own
``run_tick`` rather than using the ``DiscreteTimeAdapter`` template
method (matching ``LiveKitCascadedAdapter``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.pipecat.config import PipecatConfig
from tau2.voice.audio_native.pipecat.events import PipecatEvent, PipecatEventType
from tau2.voice.audio_native.pipecat.provider import PipecatVoiceProvider
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class PipecatVADConfig(BaseModel):
    """VAD config for the Pipecat adapter.

    Pipecat's VAD is configured via ``PipecatConfig.enable_vad`` (Silero
    VAD is wired into the input transport). This config exists for
    interface parity with other adapters.
    """

    pass


class DiscreteTimePipecatAdapter(DiscreteTimeAdapter):
    """Tick-based adapter wrapping ``PipecatVoiceProvider``."""

    def __init__(
        self,
        tick_duration_ms: int,
        pipecat_config: Optional[PipecatConfig] = None,
        send_audio_instant: bool = True,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        super().__init__(tick_duration_ms, audio_format=audio_format)
        self.pipecat_config = pipecat_config or PipecatConfig()
        self.send_audio_instant = send_audio_instant

        self._provider: Optional[PipecatVoiceProvider] = None

        # Async event loop on its own thread (so adapter calls remain sync).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Tick state
        self._tick_count = 0
        self._pending_tool_results: List[Tuple[str, str, bool]] = []
        self._buffered_audio_chunks: List[Tuple[bytes, Optional[str]]] = []
        self._cumulative_user_audio_ms: int = 0

        # Utterance transcript accumulators
        self._utterance_counter: int = 0
        self._utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self._current_utterance_id: Optional[str] = None
        self._llm_response_text: str = ""

        # Audio format converter (telephony 8kHz μ-law ↔ PCM16 16kHz/24kHz)
        tts_sample_rate = self.pipecat_config.tts.sample_rate
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=16000,
            output_sample_rate=tts_sample_rate,
        )

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    @property
    def provider(self) -> PipecatVoiceProvider:
        if self._provider is None:
            self._provider = PipecatVoiceProvider(config=self.pipecat_config)
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._loop is not None

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if self._connected:
            logger.warning("DiscreteTimePipecatAdapter already connected; reconnecting")
            self.disconnect()

        self._start_background_loop()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.provider.connect(system_prompt, tools),
                self._loop,
            )
            future.result(timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT)
            self._connected = True
            self._audio_converter.reset()
            self._buffered_audio_chunks = []
            self._tick_count = 0
            self._cumulative_user_audio_ms = 0
            self._utterance_transcripts.clear()
            self._utterance_counter = 0
            self._current_utterance_id = None
            self._llm_response_text = ""
            logger.info(
                f"DiscreteTimePipecatAdapter connected "
                f"(tick={self.tick_duration_ms}ms, "
                f"bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"Failed to connect Pipecat provider: {e}")
            self._stop_background_loop()
            raise RuntimeError(f"Failed to connect Pipecat adapter: {e}") from e

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self._loop is not None and self._provider is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._provider.disconnect(),
                    self._loop,
                )
                future.result(timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT)
            except Exception as e:
                logger.warning(f"Error during Pipecat disconnect: {e}")

        self._stop_background_loop()
        self._connected = False
        self._provider = None
        self._tick_count = 0
        self._buffered_audio_chunks = []
        self._audio_converter.reset()
        logger.info("DiscreteTimePipecatAdapter disconnected")

    def _start_background_loop(self) -> None:
        if self._loop is not None:
            return

        ready = threading.Event()

        def run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        ready.wait()

    def _stop_background_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT)
            self._loop = None
            self._thread = None

    # ------------------------------------------------------------------
    # Tick processing (own implementation, not the template method)
    # ------------------------------------------------------------------

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("DiscreteTimePipecatAdapter not connected")

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        future = asyncio.run_coroutine_threadsafe(
            self._async_run_tick(user_audio, tick_number),
            self._loop,
        )

        tick_timeout = (
            self.tick_duration_ms / 1000 + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER
        )

        try:
            return future.result(timeout=tick_timeout)
        except concurrent.futures.TimeoutError as e:
            # Empty str() repr from concurrent.futures.TimeoutError makes
            # the bare message useless. Surface the timeout explicitly,
            # along with any background-runner exception that may have
            # silently crashed the pipeline.
            future.cancel()
            runner_exc = self._get_runner_task_exception()
            if runner_exc is not None:
                logger.error(
                    f"DiscreteTimePipecatAdapter.run_tick timed out after "
                    f"{tick_timeout:.1f}s on tick {tick_number}; the Pipecat "
                    f"pipeline runner has crashed with: "
                    f"{type(runner_exc).__name__}: {runner_exc}"
                )
            else:
                logger.error(
                    f"DiscreteTimePipecatAdapter.run_tick timed out after "
                    f"{tick_timeout:.1f}s on tick {tick_number} (background "
                    f"asyncio loop appears stuck; runner task is still "
                    f"running with no exception). This usually means a "
                    f"FrameProcessor in the pipeline is blocking the loop "
                    f"or a downstream queue is full."
                )
            raise TimeoutError(
                f"Pipecat tick {tick_number} exceeded {tick_timeout:.1f}s"
            ) from e
        except Exception:
            logger.exception(
                f"DiscreteTimePipecatAdapter.run_tick failed on tick {tick_number}"
            )
            raise

    def _get_runner_task_exception(self) -> Optional[BaseException]:
        """Return the Pipecat runner task's exception if it has crashed.

        ``PipelineRunner.run`` is launched as ``self._runner_task`` inside
        ``PipecatVoiceProvider.connect``. If it raised, the asyncio loop
        keeps spinning but the pipeline is dead — every subsequent tick
        will time out with no useful info. This helper extracts that
        exception so we can include it in the timeout error message.
        """
        provider = self._provider
        if provider is None:
            return None
        runner_task = getattr(provider, "_runner_task", None)
        if runner_task is None or not runner_task.done():
            return None
        try:
            return runner_task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            return None

    async def _async_run_tick(
        self,
        user_audio: bytes,
        tick_number: int,
    ) -> TickResult:
        tick_start = asyncio.get_running_loop().time()
        deadline = tick_start + (self.tick_duration_ms / 1000)

        events: List[PipecatEvent] = []
        tool_calls: List[ToolCall] = []
        agent_audio_chunks: List[Tuple[bytes, Optional[str]]] = []
        vad_events: List[str] = []

        # Track cumulative audio for timing
        user_audio_duration_ms = (
            len(user_audio) / self.audio_format.bytes_per_second * 1000
        )
        cumulative_at_tick_start = self._cumulative_user_audio_ms
        self._cumulative_user_audio_ms += int(user_audio_duration_ms)

        # Carry over any audio buffered from the previous tick
        if self._buffered_audio_chunks:
            agent_audio_chunks.extend(self._buffered_audio_chunks)
            self._buffered_audio_chunks = []

        # Flush pending tool results (one task per result, drained via queue)
        for call_id, result, request_response in self._pending_tool_results:
            asyncio.create_task(
                self._drain_async_gen(
                    self.provider.send_tool_result(
                        call_id, result, request_response=request_response
                    )
                )
            )
        self._pending_tool_results.clear()

        # Convert telephony to PCM16 16kHz and feed into the pipeline.
        stt_audio = self._audio_converter.convert_input(user_audio)
        asyncio.create_task(
            self._drain_async_gen(self.provider.process_audio(stt_audio))
        )

        # Drain events from the provider until the tick deadline.
        await self._drain_provider_events(
            deadline, events, agent_audio_chunks, vad_events, tool_calls
        )

        # Cap audio at bytes_per_tick (carry excess to next tick).
        capped_chunks, buffered_chunks = self._cap_audio_chunks(
            agent_audio_chunks, self.bytes_per_tick
        )
        self._buffered_audio_chunks = buffered_chunks

        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=user_audio_duration_ms,
            user_audio_data=user_audio,
            events=events,
            vad_events=vad_events,
            tool_calls=tool_calls,
            agent_audio_chunks=capped_chunks,
            proportional_transcript=self._get_proportional_transcript(capped_chunks),
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
            tick_sim_duration_ms=self.tick_duration_ms,
            cumulative_user_audio_at_tick_start_ms=cumulative_at_tick_start,
        )

        elapsed = asyncio.get_running_loop().time() - tick_start
        remaining = (self.tick_duration_ms / 1000) - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

        logger.debug(f"Pipecat tick {tick_number}: {result.summary()}")
        return result

    async def _drain_async_gen(self, async_gen) -> None:
        """Push events from an async generator into our event queue."""
        try:
            async for evt in async_gen:
                # Push into the provider's queue so the tick loop drains it.
                if self._provider is not None:
                    self._provider._event_queue.put_nowait(evt)
        except Exception as e:
            logger.error(f"Pipecat pipeline error: {e}")
            if self._provider is not None:
                self._provider._event_queue.put_nowait(
                    PipecatEvent(
                        type=PipecatEventType.ERROR,
                        data={"error": str(e)},
                    )
                )

    async def _drain_provider_events(
        self,
        deadline: float,
        events: List[PipecatEvent],
        agent_audio_chunks: List[Tuple[bytes, Optional[str]]],
        vad_events: List[str],
        tool_calls: List[ToolCall],
    ) -> None:
        if self._provider is None:
            return
        loop = asyncio.get_running_loop()
        queue = self._provider._event_queue

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                evt = await asyncio.wait_for(
                    queue.get(),
                    timeout=max(0.001, remaining),
                )
                events.append(evt)
                self._handle_event(evt, agent_audio_chunks, vad_events, tool_calls)
            except asyncio.TimeoutError:
                break

    # ------------------------------------------------------------------
    # Event -> TickResult mapping
    # ------------------------------------------------------------------

    def _handle_event(
        self,
        event: PipecatEvent,
        agent_audio_chunks: List[Tuple[bytes, Optional[str]]],
        vad_events: List[str],
        tool_calls: List[ToolCall],
    ) -> None:
        et = event.type

        if et == PipecatEventType.LLM_TOKEN:
            tok = event.data.get("token") or ""
            self._llm_response_text += tok
            utt_id = f"utt_{self._utterance_counter}"
            self._current_utterance_id = utt_id
            if utt_id not in self._utterance_transcripts:
                self._utterance_transcripts[utt_id] = UtteranceTranscript(
                    item_id=utt_id
                )
            if tok:
                self._utterance_transcripts[utt_id].add_transcript(tok)

        elif et == PipecatEventType.LLM_COMPLETED:
            # Reset accumulator for the next utterance.
            self._llm_response_text = ""

        elif et == PipecatEventType.TTS_AUDIO:
            audio = event.audio
            if audio:
                telephony_audio = self._audio_converter.convert_output(audio)
                utt_id = self._current_utterance_id or f"utt_{self._utterance_counter}"
                if utt_id not in self._utterance_transcripts:
                    self._utterance_transcripts[utt_id] = UtteranceTranscript(
                        item_id=utt_id
                    )
                self._utterance_transcripts[utt_id].add_audio(len(telephony_audio))
                agent_audio_chunks.append((telephony_audio, utt_id))

        elif et == PipecatEventType.TTS_COMPLETED:
            self._utterance_counter += 1
            self._current_utterance_id = None

        elif et == PipecatEventType.SPEECH_STARTED:
            vad_events.append("speech_started")
            self._check_barge_in(agent_audio_chunks, vad_events)

        elif et == PipecatEventType.TRANSCRIPT_PARTIAL:
            self._check_barge_in(agent_audio_chunks, vad_events)

        elif et == PipecatEventType.TRANSCRIPT_FINAL:
            self._check_barge_in(agent_audio_chunks, vad_events)

        elif et == PipecatEventType.SPEECH_ENDED:
            vad_events.append("speech_stopped")

        elif et == PipecatEventType.INTERRUPTED:
            vad_events.append("interrupted")
            self._clear_agent_audio(agent_audio_chunks)

        elif et == PipecatEventType.TOOL_CALL:
            tc = event.tool_call
            if tc:
                tool_calls.append(tc)

    def _check_barge_in(
        self,
        agent_audio_chunks: List[Tuple[bytes, Optional[str]]],
        vad_events: List[str],
    ) -> None:
        has_agent_audio = (
            len(agent_audio_chunks) > 0 or len(self._buffered_audio_chunks) > 0
        )
        if has_agent_audio:
            vad_events.append("interrupted")
            self._clear_agent_audio(agent_audio_chunks)
            logger.debug("Pipecat barge-in: cleared buffered agent audio")

    def _clear_agent_audio(
        self,
        agent_audio_chunks: List[Tuple[bytes, Optional[str]]],
    ) -> None:
        self._buffered_audio_chunks = []
        agent_audio_chunks.clear()
        self._utterance_transcripts.clear()
        self._current_utterance_id = None

    # ------------------------------------------------------------------
    # Audio capping & proportional transcript
    # ------------------------------------------------------------------

    def _cap_audio_chunks(
        self,
        chunks: List[Tuple[bytes, Optional[str]]],
        max_bytes: int,
    ) -> Tuple[List[Tuple[bytes, Optional[str]]], List[Tuple[bytes, Optional[str]]]]:
        if not chunks:
            return [], []

        total_bytes = sum(len(chunk[0]) for chunk in chunks)
        if total_bytes <= max_bytes:
            return chunks, []

        kept: List[Tuple[bytes, Optional[str]]] = []
        buffered: List[Tuple[bytes, Optional[str]]] = []
        current_bytes = 0
        for data, item_id in chunks:
            if current_bytes + len(data) <= max_bytes:
                kept.append((data, item_id))
                current_bytes += len(data)
            else:
                space_left = max_bytes - current_bytes
                if space_left > 0:
                    kept.append((data[:space_left], item_id))
                    buffered.append((data[space_left:], item_id))
                else:
                    buffered.append((data, item_id))
                current_bytes = max_bytes
        return kept, buffered

    def _get_proportional_transcript(
        self,
        chunks: List[Tuple[bytes, Optional[str]]],
    ) -> str:
        if not chunks:
            return ""
        audio_by_item: dict[str, int] = {}
        for data, item_id in chunks:
            if item_id:
                audio_by_item[item_id] = audio_by_item.get(item_id, 0) + len(data)

        parts: List[str] = []
        for item_id, audio_bytes in audio_by_item.items():
            if item_id in self._utterance_transcripts:
                text = self._utterance_transcripts[item_id].get_transcript_for_audio(
                    audio_bytes
                )
                if text:
                    parts.append(text)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Tool result handling
    # ------------------------------------------------------------------

    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        self._pending_tool_results.append((call_id, result, request_response))
        logger.debug(f"Queued Pipecat tool result for call_id={call_id}")

    # ------------------------------------------------------------------
    # Template method overrides (not used; Pipecat has its own run_tick)
    # ------------------------------------------------------------------

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        raise NotImplementedError("Pipecat adapter uses its own run_tick")

    async def _flush_pending_tool_results(self):
        raise NotImplementedError("Pipecat adapter uses its own run_tick")
