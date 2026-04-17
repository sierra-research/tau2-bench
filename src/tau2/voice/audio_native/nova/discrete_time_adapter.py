"""Discrete-time audio native adapter for Amazon Nova Sonic API.

This adapter provides a tick-based interface for Amazon Nova Sonic API, designed
for discrete-time simulation where audio time is the primary clock.

Key features:
- Tick-based interface via run_tick()
- Audio format conversion: telephony (8kHz μ-law) ↔ Nova (16kHz/24kHz LPCM)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via VAD speech detection events

Usage:
    adapter = DiscreteTimeNovaAdapter(
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

Reference: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
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
from tau2.voice.audio_native.nova.events import (
    NovaAudioOutputEvent,
    NovaBargeInEvent,
    NovaCompletionEndEvent,
    NovaContentStartEvent,
    NovaSpeechEndedEvent,
    NovaSpeechStartedEvent,
    NovaTextOutputEvent,
    NovaTimeoutEvent,
    NovaToolUseEvent,
)
from tau2.voice.audio_native.nova.provider import (
    NOVA_BYTES_PER_SECOND,
    NovaSonicProvider,
    NovaVADConfig,
)
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
)

# Telephony at 8kHz μ-law = 8000 bytes per second


class DiscreteTimeNovaAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Amazon Nova Sonic API.

    Implements DiscreteTimeAdapter for Nova Sonic.

    This adapter runs an async event loop in a background thread to communicate
    with the Nova Sonic API, while exposing a synchronous interface for the agent
    and orchestrator.

    Audio format handling:
    - Input: Receives telephony audio (8kHz μ-law), converts to 16kHz PCM16
    - Output: Receives 24kHz PCM16 from Nova, converts to 8kHz μ-law

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick (8kHz μ-law = 8000 bytes/sec).
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
        provider: Optional[NovaSonicProvider] = None,
        voice: str = "tiffany",
    ):
        """Initialize the discrete-time Nova Sonic adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            model: Model to use. Defaults to None (provider default).
                If provider is also provided, this is ignored.
            reasoning_effort: Not supported by Nova. Must be None.
            provider: Optional provider instance. Created lazily if not provided.
            voice: Voice to use. Options: matthew, tiffany, amy. Default: tiffany.
        """
        if reasoning_effort is not None:
            raise ValueError(
                f"Nova provider does not support reasoning_effort (got '{reasoning_effort}')"
            )
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        self._chunk_size = int(NOVA_BYTES_PER_SECOND * self._voip_interval_ms / 1000)
        self.voice = voice

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Audio format converter (preserves state for streaming)
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=16000,
            output_sample_rate=24000,
        )

        # Async event loop management
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # Audio stream state
        self._audio_content_id: Optional[str] = None

        # Nova-specific: uses different content_ids for text vs audio
        # _item_id_map (from base class) maps audio_content_id -> text_content_id
        self._item_id_map = {}
        self._last_assistant_text_content_id: Optional[str] = None

        # Nova uses skip_content_id instead of skip_item_id internally
        self._skip_content_id: Optional[str] = None
        self._current_content_id: Optional[str] = None

        # Background receive task and event queue
        self._receive_task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None
        self._receive_active = False

        # Track FINAL content IDs - only process audio/text from FINAL (not SPECULATIVE)
        self._final_content_ids: set[str] = set()

    @property
    def provider(self) -> NovaSonicProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = NovaSonicProvider(model_id=self.model, voice=self.voice)
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
        """Connect to the Nova Sonic API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to server VAD.
            modality: Ignored (Nova Sonic always uses audio).
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = NovaVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeNovaAdapter connected to Nova Sonic API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeNovaAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Nova Sonic API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: NovaVADConfig,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
        )
        # Start the audio stream for continuous input
        self._audio_content_id = await self.provider.start_audio_stream()

        # Start background receive task BEFORE sending any audio
        logger.debug("Starting background receive task...")
        self._event_queue = asyncio.Queue()
        self._receive_active = True
        self._receive_task = asyncio.create_task(self._background_receive_loop())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Send initial audio to trigger Nova's response
        logger.debug("Sending initial silence to prime audio stream...")
        initial_silence = b"\x00" * 32000  # 1 second of 16kHz PCM16 silence
        await self.provider.send_audio(initial_silence, self._audio_content_id)

        logger.debug("Background receive task started")

    async def _background_receive_loop(self) -> None:
        """Background task that continuously receives events from Nova Sonic.

        Events are placed in _event_queue for consumption by _execute_tick.
        This keeps the bidirectional stream alive and responsive.
        """
        try:
            if not await self.provider._ensure_output_stream():
                logger.error("Failed to initialize output stream in background task")
                return

            logger.debug(
                "Background receive loop: output stream ready, starting event loop"
            )

            while self._receive_active:
                try:
                    event_data = await self.provider._read_next_event()

                    if event_data is None:
                        logger.info("Background receive loop: stream ended")
                        break

                    from tau2.voice.audio_native.nova.events import parse_nova_event

                    event = parse_nova_event(event_data)
                    await self._event_queue.put(event)

                except Exception as e:
                    if self._receive_active:
                        logger.debug(f"Background receive loop error: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug("Background receive loop cancelled")
        except Exception as e:
            logger.error(f"Background receive loop fatal error: {e}")

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
        self._audio_content_id = None
        self.clear_buffers()
        self._audio_converter.reset()
        self._item_id_map.clear()
        self._last_assistant_text_content_id = None
        self._final_content_ids.clear()
        logger.info("DiscreteTimeNovaAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """Async disconnection."""
        # Stop the background receive task first
        self._receive_active = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._receive_task = None

        # End the audio content block if we started one
        if self._audio_content_id:
            try:
                await self.provider.end_audio_content(self._audio_content_id)
            except Exception as e:
                logger.debug(f"Error ending audio content: {e}")
            self._audio_content_id = None

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
            raise RuntimeError("Not connected to Nova Sonic API. Call connect() first.")

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
        """Send pending tool results to Nova."""
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
        """Nova-specific: convert audio, send via stream, receive from queue."""
        # Nova uses skip_content_id; sync with result's skip_item_id
        result.skip_item_id = self._skip_content_id

        # Convert telephony audio to Nova format (8kHz μ-law → 16kHz PCM16)
        nova_audio = self._audio_converter.convert_input(user_audio)

        async def send_nova_audio(chunk: bytes) -> None:
            if not self._audio_content_id:
                return
            await self.provider.send_audio(chunk, self._audio_content_id)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            end_time = asyncio.get_running_loop().time() + remaining
            collected = []
            while asyncio.get_running_loop().time() < end_time:
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.05,
                    )
                    collected.append(event)
                except asyncio.TimeoutError:
                    continue
            return collected

        _, events = await asyncio.gather(
            self._send_audio_chunked(nova_audio, send_nova_audio, self._chunk_size),
            receive_events(),
        )

        for event in events:
            self._process_event(result, event)

        # Sync back skip state from Nova's content_id space
        self._skip_content_id = result.skip_item_id

    def _process_event(self, result: TickResult, event: Any) -> None:
        """Process a Nova Sonic event."""
        result.events.append(event)

        if isinstance(event, NovaAudioOutputEvent):
            content_id = event.content_id or self._current_content_id

            # Skip audio from truncated content
            if result.skip_item_id is not None and content_id == result.skip_item_id:
                audio_bytes = base64.b64decode(event.content) if event.content else b""
                result.truncated_audio_bytes += len(audio_bytes)
                return

            # Skip SPECULATIVE audio - only process FINAL content
            text_content_id = self._item_id_map.get(content_id, content_id)
            if (
                content_id not in self._final_content_ids
                and text_content_id not in self._final_content_ids
            ):
                logger.debug(
                    f"Skipping SPECULATIVE audio: {content_id[:8] if content_id else 'None'}..."
                )
                return

            # Decode base64 audio (24kHz PCM16 from Nova)
            nova_audio = base64.b64decode(event.content) if event.content else b""
            if nova_audio:
                telephony_audio = self._audio_converter.convert_output(nova_audio)
                if telephony_audio:
                    result.agent_audio_chunks.append((telephony_audio, content_id))

            # Track for transcript distribution
            if content_id:
                self._current_content_id = content_id
                if text_content_id not in self._utterance_transcripts:
                    self._utterance_transcripts[text_content_id] = UtteranceTranscript(
                        item_id=text_content_id
                    )
                self._utterance_transcripts[text_content_id].add_audio(
                    len(telephony_audio)
                )

        elif isinstance(event, NovaTextOutputEvent):
            if event.role == "ASSISTANT":
                content_id = event.content_id or self._current_content_id

                # Skip SPECULATIVE text - only process FINAL content
                if content_id not in self._final_content_ids:
                    logger.debug(
                        f"Skipping SPECULATIVE text: {content_id[:8] if content_id else 'None'}..."
                    )
                    return

                if content_id and event.content:
                    if content_id not in self._utterance_transcripts:
                        self._utterance_transcripts[content_id] = UtteranceTranscript(
                            item_id=content_id
                        )
                    self._utterance_transcripts[content_id].add_transcript(
                        event.content
                    )
                    self._last_assistant_text_content_id = content_id
                    logger.debug(f"Agent transcript added (FINAL): {content_id[:8]}...")

        elif isinstance(event, NovaContentStartEvent):
            if event.content_id:
                self._current_content_id = event.content_id

                if event.generation_stage == "FINAL":
                    self._final_content_ids.add(event.content_id)
                    logger.debug(
                        f"FINAL content started: {event.content_id[:8]}... "
                        f"(role={event.role}, type={event.type})"
                    )
                elif event.generation_stage == "SPECULATIVE":
                    logger.debug(
                        f"SPECULATIVE content (ignoring): {event.content_id[:8]}... "
                        f"(role={event.role}, type={event.type})"
                    )

                # Nova sends TEXT before AUDIO with different content_ids
                if event.type == "AUDIO" and event.role == "ASSISTANT":
                    if self._last_assistant_text_content_id:
                        self._item_id_map[event.content_id] = (
                            self._last_assistant_text_content_id
                        )
                        logger.debug(
                            f"Audio→text mapping: {event.content_id[:8]}... -> {self._last_assistant_text_content_id[:8]}..."
                        )

        elif isinstance(event, (NovaSpeechStartedEvent, NovaBargeInEvent)):
            logger.debug("Speech started / barge-in - interruption detected")
            result.vad_events.append("speech_started")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_content_id

            # Clear FINAL content tracking - new response will have new content IDs
            self._final_content_ids.clear()

            # Reset audio converter on interruption
            self._audio_converter.reset()

        elif isinstance(event, NovaSpeechEndedEvent):
            logger.debug("Speech ended")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, NovaToolUseEvent):
            try:
                arguments = json.loads(event.content) if event.content else {}
            except json.JSONDecodeError:
                arguments = {}

            tool_call = ToolCall(
                id=event.tool_use_id or "",
                name=event.tool_name or "",
                arguments=arguments,
            )
            result.tool_calls.append(tool_call)
            logger.debug(f"Tool call detected: {event.tool_name}({event.tool_use_id})")

        elif isinstance(event, NovaCompletionEndEvent):
            logger.debug(f"Completion done (stop_reason={event.stop_reason})")

        elif isinstance(event, NovaTimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
