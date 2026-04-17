"""Discrete-time audio native adapter for Qwen Omni Flash Realtime API.

This adapter provides a tick-based interface for Qwen Realtime API, designed
for discrete-time simulation where audio time is the primary clock.

Key features:
- Tick-based interface via run_tick()
- Audio format conversion (telephony ↔ Qwen's PCM16)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via VAD speech_started events

Audio format notes:
- Telephony: 8kHz μ-law (8000 bytes/sec)
- Qwen input: 16kHz PCM16 (32000 bytes/sec)
- Qwen output: 24kHz PCM16 (48000 bytes/sec)

Usage:
    adapter = DiscreteTimeQwenAdapter(
        tick_duration_ms=1000,
        send_audio_instant=True,
    )
    adapter.connect(system_prompt, tools, vad_config)

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)
        # result.get_played_agent_audio() - capped agent audio (telephony format)
        # result.proportional_transcript - text for this tick
        # result.tool_calls - function calls

    adapter.disconnect()

Reference: https://www.alibabacloud.com/help/en/model-studio/realtime
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
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.qwen.events import (
    QwenAudioDeltaEvent,
    QwenAudioDoneEvent,
    QwenAudioTranscriptDeltaEvent,
    QwenErrorEvent,
    QwenFunctionCallArgumentsDoneEvent,
    QwenInputAudioTranscriptionCompletedEvent,
    QwenResponseDoneEvent,
    QwenSpeechStartedEvent,
    QwenTimeoutEvent,
)
from tau2.voice.audio_native.qwen.provider import (
    QWEN_INPUT_BYTES_PER_SECOND,
    QwenRealtimeProvider,
    QwenVADConfig,
)
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
)

# Telephony format constants


class DiscreteTimeQwenAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Qwen Realtime API.

    Implements DiscreteTimeAdapter for Qwen Omni Flash Realtime API.

    This adapter runs an async event loop in a background thread to communicate
    with the Qwen API, while exposing a synchronous interface for the agent
    and orchestrator.

    Audio conversion is handled automatically:
    - Input: telephony (8kHz μ-law) → Qwen (16kHz PCM16)
    - Output: Qwen (24kHz PCM16) → telephony (8kHz μ-law)

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick in telephony format (8kHz μ-law).
        send_audio_instant: If True, send audio in one call per tick.
            If False, send in 20ms chunks with sleeps (VoIP-style streaming).
        provider: Optional provider instance. Created lazily if not provided.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[QwenRealtimeProvider] = None,
        voice: str = "Cherry",
    ):
        """Initialize the discrete-time Qwen adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            model: Model to use. Defaults to None (provider default).
                If provider is also provided, this is ignored.
            reasoning_effort: Not supported by Qwen. Must be None.
            provider: Optional provider instance. Created lazily if not provided.
            voice: Voice to use. Default: Cherry.
        """
        if reasoning_effort is not None:
            raise ValueError(
                f"Qwen provider does not support reasoning_effort (got '{reasoning_effort}')"
            )
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        self._chunk_size = int(
            QWEN_INPUT_BYTES_PER_SECOND * self._voip_interval_ms / 1000
        )
        self.voice = voice

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model

        # Audio converter for telephony ↔ Qwen format
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=16000,
            output_sample_rate=24000,
        )

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Async event loop management
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> QwenRealtimeProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = QwenRealtimeProvider(model=self.model, voice=self.voice)
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Check if connected to the API."""
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to the Qwen API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to server VAD.
            modality: "audio" for audio in/out, "text" for text only.
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = QwenVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeQwenAdapter connected to Qwen API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeQwenAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Qwen API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: QwenVADConfig,
        modality: str,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
        )

    def disconnect(self) -> None:
        """Disconnect from the API and clean up resources."""
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
        self._audio_converter.reset()
        logger.info("DiscreteTimeQwenAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """Async disconnection."""
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """Run one tick of the simulation.

        Args:
            user_audio: User audio bytes in telephony format (8kHz μ-law).
            tick_number: Optional tick number for logging.

        Returns:
            TickResult with audio in telephony format (8kHz μ-law).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Qwen API. Call connect() first.")

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
        """Send pending tool results to Qwen."""
        for (
            call_id,
            result_str,
            _request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str, _request_response)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Qwen-specific: convert audio, send, receive events, process."""
        # Convert telephony audio to Qwen format
        qwen_audio = self._audio_converter.convert_input(user_audio)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                qwen_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            self._process_event(result, event)

    def _process_event(self, result: TickResult, event: Any) -> None:
        """Process a Qwen event."""
        result.events.append(event)

        if isinstance(event, QwenAudioDeltaEvent):
            item_id = event.item_id or self._current_item_id

            # Skip audio from truncated item
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                if event.delta:
                    qwen_audio = base64.b64decode(event.delta)
                    telephony_audio = self._audio_converter.convert_output(qwen_audio)
                    result.truncated_audio_bytes += len(telephony_audio)
                return

            # Decode base64 and convert from Qwen format (24kHz PCM16) to telephony (8kHz μ-law)
            if event.delta:
                qwen_audio = base64.b64decode(event.delta)
                telephony_audio = self._audio_converter.convert_output(qwen_audio)

                if telephony_audio:
                    result.agent_audio_chunks.append((telephony_audio, item_id))

                # Track for transcript distribution (using telephony bytes)
                if item_id:
                    self._current_item_id = item_id
                    if item_id not in self._utterance_transcripts:
                        self._utterance_transcripts[item_id] = UtteranceTranscript(
                            item_id=item_id
                        )
                    self._utterance_transcripts[item_id].add_audio(len(telephony_audio))

        elif isinstance(event, QwenAudioTranscriptDeltaEvent):
            item_id = event.item_id or self._current_item_id
            if item_id and event.delta:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.delta)

        elif isinstance(event, QwenSpeechStartedEvent):
            logger.debug("Speech started - interruption detected")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Reset audio converter state on interruption
            self._audio_converter.reset()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

        elif isinstance(event, QwenFunctionCallArgumentsDoneEvent):
            try:
                arguments = json.loads(event.arguments) if event.arguments else {}
            except json.JSONDecodeError:
                arguments = {}

            tool_call = ToolCall(
                id=event.call_id or "",
                name=event.name or "",
                arguments=arguments,
            )
            result.tool_calls.append(tool_call)
            logger.debug(f"Tool call detected: {event.name}({event.call_id})")

        elif isinstance(event, QwenInputAudioTranscriptionCompletedEvent):
            logger.debug(f"Input transcription: {event.transcript}")

        elif isinstance(event, QwenResponseDoneEvent):
            logger.debug("Response done (turn complete)")

        elif isinstance(event, QwenAudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, QwenErrorEvent):
            logger.error(f"Qwen error: {event.message}")

        elif isinstance(event, QwenTimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
