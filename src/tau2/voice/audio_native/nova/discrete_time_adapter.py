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
from typing import Any, List, Optional, Tuple

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
    DEFAULT_TELEPHONY_RATE,
    TELEPHONY_ULAW_SILENCE,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.nova.audio_utils import (
    StreamingNovaConverter,
)
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
    buffer_excess_audio,
    get_proportional_transcript,
)

# Telephony at 8kHz μ-law = 8000 bytes per second
NOVA_TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 8000


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

    VOIP_PACKET_INTERVAL_MS = DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        provider: Optional[NovaSonicProvider] = None,
        voice: str = "tiffany",
    ):
        """Initialize the discrete-time Nova Sonic adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            model: Model to use. Defaults to None (provider default).
                If provider is also provided, this is ignored.
            provider: Optional provider instance. Created lazily if not provided.
            voice: Voice to use. Options: matthew, tiffany, amy. Default: tiffany.
        """
        super().__init__(tick_duration_ms)

        self.send_audio_instant = send_audio_instant
        self._chunk_size = int(
            NOVA_BYTES_PER_SECOND * self.VOIP_PACKET_INTERVAL_MS / 1000
        )
        self.voice = voice

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Audio format converter (preserves state for streaming)
        self._audio_converter = StreamingNovaConverter()

        # Async event loop management
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # Tick state
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0

        # Audio stream state
        self._audio_content_id: Optional[str] = None

        # Buffered audio and transcripts
        self._buffered_agent_audio: List[Tuple[bytes, Optional[str]]] = []
        self._utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self._current_content_id: Optional[str] = None
        self._skip_content_id: Optional[str] = None

        # Nova sends TEXT before AUDIO with different content_ids
        # Track the mapping: audio_content_id -> text_content_id
        self._audio_to_text_map: dict[str, str] = {}
        self._last_assistant_text_content_id: Optional[str] = None

        # Tool result queue (for sending tool results in next tick)
        self._pending_tool_results: List[Tuple[str, str, bool]] = []

        # Background receive task and event queue
        self._receive_task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None  # Created in event loop
        self._receive_active = False

        # Track FINAL content IDs - only process audio/text from FINAL (not SPECULATIVE)
        # Nova Sonic uses speculative generation; we ignore speculative content
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
        # This matches the standalone test flow which works
        logger.debug("Starting background receive task...")
        self._event_queue = asyncio.Queue()  # Create queue in the event loop
        self._receive_active = True
        self._receive_task = asyncio.create_task(self._background_receive_loop())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Send initial audio to trigger Nova's response
        # The standalone test sends actual speech audio here - we send silence
        # but this may need to be real audio for proper VAD triggering
        logger.debug("Sending initial silence to prime audio stream...")
        initial_silence = b"\x00" * 32000  # 1 second of 16kHz PCM16 silence
        await self.provider.send_audio(initial_silence, self._audio_content_id)

        logger.debug("Background receive task started")

    async def _background_receive_loop(self) -> None:
        """Background task that continuously receives events from Nova Sonic.

        Events are placed in _event_queue for consumption by _async_run_tick.
        This keeps the bidirectional stream alive and responsive.
        """
        try:
            # First, ensure output stream is ready
            if not await self.provider._ensure_output_stream():
                logger.error("Failed to initialize output stream in background task")
                return

            logger.debug(
                "Background receive loop: output stream ready, starting event loop"
            )

            while self._receive_active:
                try:
                    # Read next event (this may block at C level)
                    event_data = await self.provider._read_next_event()

                    if event_data is None:
                        logger.info("Background receive loop: stream ended")
                        break

                    # Parse and queue the event
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
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._audio_converter.reset()
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

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Async tick execution."""
        # Send any pending tool results first
        for call_id, result_str, request_response in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str, request_response)
        self._pending_tool_results.clear()

        # Calculate timing
        tick_start = asyncio.get_running_loop().time()

        # Create tick result
        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=(len(user_audio) / NOVA_TELEPHONY_BYTES_PER_SECOND)
            * 1000,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self._cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=NOVA_TELEPHONY_BYTES_PER_SECOND,
            silence_byte=TELEPHONY_ULAW_SILENCE,
        )

        # Add any buffered agent audio from previous tick
        for chunk_data, content_id in self._buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, content_id))
        self._buffered_agent_audio.clear()

        # Carry over skip state from previous tick
        result.skip_item_id = self._skip_content_id

        # Convert telephony audio to Nova format (8kHz μ-law → 16kHz PCM16)
        nova_audio = self._audio_converter.convert_input(user_audio)

        # Send audio and receive events concurrently
        async def send_audio():
            """Send audio (instant or chunked based on config)."""
            if not nova_audio or not self._audio_content_id:
                return
            if self.send_audio_instant:
                await self.provider.send_audio(nova_audio, self._audio_content_id)
            else:
                offset = 0
                while offset < len(nova_audio):
                    chunk = nova_audio[offset : offset + self._chunk_size]
                    await self.provider.send_audio(chunk, self._audio_content_id)
                    offset += len(chunk)
                    await asyncio.sleep(self.VOIP_PACKET_INTERVAL_MS / 1000)

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

        _, events = await asyncio.gather(send_audio(), receive_events())

        # Process all received events
        for event in events:
            await self._process_event(result, event)

        # Record simulation timing
        result.tick_sim_duration_ms = result.audio_sent_duration_ms

        # Move excess agent audio to buffer for next tick
        self._buffered_agent_audio = buffer_excess_audio(result, self.bytes_per_tick)

        # Calculate proportional transcript (Nova needs audio->text ID mapping)
        result.proportional_transcript = get_proportional_transcript(
            result.agent_audio_chunks,
            self._utterance_transcripts,
            item_id_map=self._audio_to_text_map,
        )

        # Update skip state for next tick
        self._skip_content_id = result.skip_item_id

        # Update cumulative user audio tracking
        self._cumulative_user_audio_ms += int(result.audio_sent_duration_ms)

        logger.info(f"Tick {tick_number} completed:\n{result.summary()}")

        return result

    async def _process_event(self, result: TickResult, event: Any) -> None:
        """Process a Nova Sonic event."""
        result.events.append(event)

        if isinstance(event, NovaAudioOutputEvent):
            content_id = event.content_id or self._current_content_id

            # Skip audio from truncated content
            if result.skip_item_id is not None and content_id == result.skip_item_id:
                # Decode and count discarded bytes
                audio_bytes = base64.b64decode(event.content) if event.content else b""
                result.truncated_audio_bytes += len(audio_bytes)
                return

            # Skip SPECULATIVE audio - only process FINAL content
            # Check both the audio content_id and its mapped text content_id
            text_content_id = self._audio_to_text_map.get(content_id, content_id)
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
                # Convert to telephony format (24kHz PCM16 → 8kHz μ-law)
                telephony_audio = self._audio_converter.convert_output(nova_audio)
                if telephony_audio:
                    result.agent_audio_chunks.append((telephony_audio, content_id))

            # Track for transcript distribution
            # Use the audio->text mapping to add bytes to the correct transcript
            if content_id:
                self._current_content_id = content_id
                if text_content_id not in self._utterance_transcripts:
                    self._utterance_transcripts[text_content_id] = UtteranceTranscript(
                        item_id=text_content_id
                    )
                # Track telephony bytes (what we output) on the TEXT transcript
                self._utterance_transcripts[text_content_id].add_audio(
                    len(telephony_audio)
                )

        elif isinstance(event, NovaTextOutputEvent):
            # Only track ASSISTANT transcripts for proportional display
            # USER textOutput events are ASR transcripts of user speech
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
                    # Track as most recent text content for audio->text mapping
                    self._last_assistant_text_content_id = content_id
                    logger.debug(f"Agent transcript added (FINAL): {content_id[:8]}...")

        elif isinstance(event, NovaContentStartEvent):
            # Track new content block
            if event.content_id:
                self._current_content_id = event.content_id

                # Track FINAL content IDs - we only process FINAL, not SPECULATIVE
                # Nova uses speculative generation; speculative content may be revised
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
                # Track the mapping so audio can find its transcript
                if event.type == "AUDIO" and event.role == "ASSISTANT":
                    # Map this audio content to the most recent text content
                    if self._last_assistant_text_content_id:
                        self._audio_to_text_map[event.content_id] = (
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
            # Parse arguments
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
            # Normal timeout, continue
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")

    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Queue a tool result to be sent in the next tick.

        Args:
            call_id: The tool call ID.
            result: The tool result as a string.
            request_response: Ignored for Nova (always continues automatically).
            is_error: If True, the tool call failed. Currently unused by Nova.
        """
        self._pending_tool_results.append((call_id, result, request_response))
        logger.debug(f"Queued tool result for call_id={call_id}")

    def clear_buffers(self) -> None:
        """Clear all internal audio and transcript buffers."""
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._pending_tool_results.clear()
        self._skip_content_id = None
        self._audio_converter.reset()
        self._audio_to_text_map.clear()
        self._last_assistant_text_content_id = None
        self._final_content_ids.clear()
