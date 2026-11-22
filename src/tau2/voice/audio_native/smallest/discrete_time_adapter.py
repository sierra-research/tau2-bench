"""Discrete-time adapter for Smallest AI's Hydra speech-to-speech server (smallest).

Each tick sends the caller's 8 kHz u-law audio to Hydra as 16 kHz PCM16 in
20 ms frames (or one frame with ``send_audio_instant``), collects Hydra's
events for the rest of the tick, and converts the agent's PCM16 audio, at the
rate ``session.configured`` announced, back to telephony audio. Tool calls
are returned to τ-bench; their results go back on the next tick. Events are
applied by ``HydraEventProcessing`` (``processing.py``).

Transcript: text is revealed through ``PacedUtteranceTranscript`` (see
``pacing.py``), never faster than the audio that speaks it is played.
"""

import asyncio
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_SMALLEST_INPUT_SAMPLE_RATE,
    DEFAULT_SMALLEST_MODEL,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.smallest import events as ev
from tau2.voice.audio_native.smallest.pacing import log_reveal_mode, resolve_reveal_cps
from tau2.voice.audio_native.smallest.processing import HydraEventProcessing
from tau2.voice.audio_native.smallest.provider import SmallestProvider
from tau2.voice.audio_native.smallest.responses import HydraResponses
from tau2.voice.audio_native.tick_result import TickResult


class DiscreteTimeSmallestAdapter(HydraEventProcessing, DiscreteTimeAdapter):
    """Tick-based adapter for one Hydra session per call."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        provider: Optional[SmallestProvider] = None,
        transcript_reveal_cps: Optional[float] = None,
    ):
        """Initialize the adapter (no connection is made).

        Args:
            tick_duration_ms: Duration of each tick in milliseconds.
            send_audio_instant: Send each tick's audio as one frame instead of
                20 ms frames.
            model: Label recorded with the results.
            reasoning_effort: Not supported by this provider.
            audio_format: Only telephony (8 kHz u-law) is supported.
            provider: Optional provider; created from the environment if None.
            transcript_reveal_cps: Transcript reveal cap in characters per
                second of agent audio played; None reads
                ``SMALLEST_TRANSCRIPT_REVEAL_CPS`` (default 15, 0 = off).
        """
        if reasoning_effort is not None:
            raise ValueError(
                "--reasoning-effort is not supported by the smallest provider"
            )
        if audio_format is not None and audio_format != TELEPHONY_AUDIO_FORMAT:
            raise ValueError(
                f"Smallest supports only telephony audio (8 kHz u-law), got {audio_format}"
            )
        super().__init__(
            tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )
        self.model = model or DEFAULT_SMALLEST_MODEL
        self.transcript_reveal_cps = resolve_reveal_cps(transcript_reveal_cps)
        log_reveal_mode(self.transcript_reveal_cps)
        self._chunk_size = int(
            DEFAULT_SMALLEST_INPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
        )
        self._input_converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_SMALLEST_INPUT_SAMPLE_RATE
        )
        self._output_converter: Optional[StreamingTelephonyConverter] = None
        self._responses = HydraResponses(
            self._utterance_transcripts,
            self.transcript_reveal_cps,
            self.audio_format.bytes_per_second,
        )
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> SmallestProvider:
        if self._provider is None:
            self._provider = SmallestProvider()
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
        """Open the Hydra session; ``vad_config`` is unused (server-side turns)."""
        if modality != "audio":
            raise ValueError(
                f"Smallest supports only modality='audio', got {modality!r}"
            )
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self.provider.connect(system_prompt, tools),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"DiscreteTimeSmallestAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Hydra: {e}") from e
        self._connected = True
        logger.info(
            f"DiscreteTimeSmallestAdapter connected (tick={self.tick_duration_ms}ms, "
            f"bytes_per_tick={self.bytes_per_tick})"
        )

    def disconnect(self) -> None:
        """Close the session; the usage ledger survives for collection."""
        if not self._connected:
            return
        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during Hydra disconnect: {e}")
        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._responses.reset()
        self._input_converter.reset()
        self._output_converter = None
        logger.info("DiscreteTimeSmallestAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self._provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Not connected to Hydra. Call connect() first.")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=self.tick_duration_ms / 1000
            + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
        )

    async def _flush_pending_tool_results(self) -> None:
        for call_id, output, request_response, _is_error in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, output, request_response)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        pcm = self._input_converter.convert_input(user_audio)

        async def receive_events() -> List[ev.SmallestEvent]:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, self.tick_duration_ms / 1000 - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        tasks = [
            asyncio.create_task(
                self._send_audio_chunked(
                    pcm, self.provider.send_audio, self._chunk_size
                )
            ),
            asyncio.create_task(receive_events()),
        ]
        try:
            _, events = await asyncio.gather(*tasks)
        finally:
            # gather does not cancel the sibling of a failed task.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for event in events:
            self._process_event(result, event)
        # Spoken tails of replies with no audio in this tick (see pacing.py).
        keys = {key for _, key in result.agent_audio_chunks if key}
        result.agent_audio_chunks[:0] = self._responses.flush_chunks(keys)
