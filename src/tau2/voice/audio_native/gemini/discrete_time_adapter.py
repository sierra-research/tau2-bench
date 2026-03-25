"""Discrete-time audio native adapter for Gemini Live API.

This adapter provides a tick-based interface for Gemini Live API, designed
for discrete-time simulation where audio time is the primary clock.

Key features:
- Tick-based interface via run_tick()
- Audio format conversion (telephony ↔ Gemini formats)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via server_content.interrupted

Usage:
    adapter = DiscreteTimeGeminiAdapter(
        tick_duration_ms=1000,
        send_audio_instant=True,
    )
    adapter.connect(system_prompt, tools, vad_config, modality="audio")

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)
        # result.get_played_agent_audio() - capped agent audio (telephony format)
        # result.proportional_transcript - text for this tick
        # result.tool_calls - function calls

    adapter.disconnect()
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
    TELEPHONY_ULAW_SILENCE,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.gemini.audio_utils import (
    GEMINI_INPUT_BYTES_PER_SECOND,
    GEMINI_OUTPUT_BYTES_PER_SECOND,
    TELEPHONY_BYTES_PER_SECOND,
    StreamingGeminiConverter,
    calculate_gemini_bytes_per_tick,
)
from tau2.voice.audio_native.gemini.events import (
    GeminiAudioDeltaEvent,
    GeminiAudioDoneEvent,
    GeminiFunctionCallDoneEvent,
    GeminiGoAwayEvent,
    GeminiInputTranscriptionEvent,
    GeminiInterruptionEvent,
    GeminiSessionResumptionEvent,
    GeminiTextDeltaEvent,
    GeminiTimeoutEvent,
    GeminiTurnCompleteEvent,
)
from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider, GeminiVADConfig
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
    buffer_excess_audio,
    get_proportional_transcript,
)


class DiscreteTimeGeminiAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Gemini Live API.

    Implements DiscreteTimeAdapter for Gemini Live.

    This adapter runs an async event loop in a background thread to communicate
    with the Gemini Live API, while exposing a synchronous interface for
    the agent and orchestrator.

    Audio format handling:
    - Input: Receives telephony audio (8kHz μ-law), converts to 16kHz PCM16
    - Output: Receives 24kHz PCM16 from Gemini, converts to 8kHz μ-law

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick in telephony format (8kHz μ-law).
        send_audio_instant: If True, send audio in one call per tick.
            If False, send in 20ms chunks with sleeps (VoIP-style streaming).
        model: Model to use. Defaults to None. If provider is also provided, this is ignored.
        provider: Optional provider instance. Created lazily if not provided.
            If not provided, auto-detects auth from env vars (GEMINI_API_KEY
            or GOOGLE_APPLICATION_CREDENTIALS).
    """

    VOIP_PACKET_INTERVAL_MS = DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        provider: Optional[GeminiLiveProvider] = None,
        max_resumptions: int = 3,
        resume_only_on_timeout: bool = True,
    ):
        """Initialize the discrete-time Gemini adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            model: Optional model to use. Defaults to None. If provider is also provided, this is ignored.
            provider: Optional provider instance. Created lazily if not provided.
                If not provided, auto-detects auth from env vars (GEMINI_API_KEY
                or GOOGLE_APPLICATION_CREDENTIALS).
            max_resumptions: Maximum number of session resumptions to attempt
                when the WebSocket connection is closed. Set to 0 to disable
                session resumption. Defaults to 3.
            resume_only_on_timeout: If True (default), only attempt resumption
                when the connection closes due to the planned ~10 minute timeout
                (indicated by a GoAway message). If False, attempt resumption
                on any connection close.
        """
        super().__init__(tick_duration_ms)

        self.send_audio_instant = send_audio_instant
        self._chunk_size = int(
            GEMINI_INPUT_BYTES_PER_SECOND * self.VOIP_PACKET_INTERVAL_MS / 1000
        )

        # Gemini output format (24kHz PCM16) - for internal processing
        self._gemini_output_bytes_per_tick = calculate_gemini_bytes_per_tick(
            tick_duration_ms, direction="output"
        )

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model
        self._max_resumptions = max_resumptions
        self._resume_only_on_timeout = resume_only_on_timeout

        # Provider - created lazily if not provided (auto-detects auth from env)
        self._provider = provider
        self._owns_provider = provider is None

        # Audio format converter (preserves state for streaming)
        self._audio_converter = StreamingGeminiConverter()

        # Async event loop management
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # Tick state
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0

        # Buffered audio and transcripts
        self._buffered_agent_audio: List[Tuple[bytes, Optional[str]]] = []
        self._utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self._current_item_id: Optional[str] = None
        self._skip_item_id: Optional[str] = None

        # Tool result queue (for sending tool results in next tick)
        # Each entry: (call_id, name, result_str, request_response, is_error)
        self._pending_tool_results: List[Tuple[str, str, str, bool, bool]] = []

        # Track tool call info for Gemini (which sends null IDs)
        # Maps synthetic call_id -> (original_gemini_id, function_name)
        # We use synthetic IDs internally for tracking, but send original IDs back to Gemini
        self._tool_call_info: Dict[str, Tuple[str, str]] = {}

    @property
    def provider(self) -> GeminiLiveProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = GeminiLiveProvider(
                model=self.model,
                max_resumptions=self._max_resumptions,
                resume_only_on_timeout=self._resume_only_on_timeout,
            )
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
        """Connect to the Gemini Live API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to automatic VAD.
            modality: "audio" or "text".
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = GeminiVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeGeminiAdapter connected to Gemini Live API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(
                f"DiscreteTimeGeminiAdapter failed to connect to Gemini Live API: "
                f"{type(e).__name__}: {e}"
            )
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Gemini Live API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: GeminiVADConfig,
        modality: str,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect(
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
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._tool_call_info.clear()
        self._audio_converter.reset()
        logger.info("DiscreteTimeGeminiAdapter disconnected")

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
            raise RuntimeError(
                "Not connected to Gemini Live API. Call connect() first."
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

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Async tick execution."""
        # Handle pending GoAway reconnection before any session I/O.
        if self.provider.needs_reconnection:
            logger.info(
                f"Tick {tick_number}: performing GoAway reconnection before session I/O"
            )
            success = await self.provider.perform_pending_reconnection()
            if not success:
                raise RuntimeError("GoAway reconnection failed at tick boundary")

        # Send all pending tool results in a single batch call.
        # Gemini may respond after each individual send_tool_response,
        # so batching prevents the model from re-calling tools it hasn't
        # seen results for yet.
        if self._pending_tool_results:
            from google.genai import types

            function_responses = []
            for (
                call_id,
                name,
                result_str,
                request_response,
                is_error,
            ) in self._pending_tool_results:
                payload = {"error": result_str} if is_error else {"output": result_str}
                function_responses.append(
                    types.FunctionResponse(id=call_id, name=name, response=payload)
                )
            await self.provider._session.send_tool_response(
                function_responses=function_responses
            )
            logger.debug(
                f"Sent {len(function_responses)} tool results to Gemini in one batch"
            )
            self._pending_tool_results.clear()

        # Convert user audio from telephony to Gemini format
        gemini_audio = self._audio_converter.convert_input(user_audio)

        # Calculate timing
        tick_start = asyncio.get_running_loop().time()
        _ = tick_start + (self.tick_duration_ms / 1000)

        # Create tick result
        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=(len(user_audio) / TELEPHONY_BYTES_PER_SECOND)
            * 1000,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self._cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=TELEPHONY_BYTES_PER_SECOND,
            silence_byte=TELEPHONY_ULAW_SILENCE,
        )

        # Add any buffered agent audio from previous tick
        for chunk_data, item_id in self._buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, item_id))
        self._buffered_agent_audio.clear()

        # Carry over skip state from previous tick
        result.skip_item_id = self._skip_item_id

        # Receive events for tick duration
        gemini_audio_received: List[Tuple[bytes, Optional[str]]] = []

        async def send_audio():
            """Send audio (instant or chunked based on config)."""
            if len(gemini_audio) == 0:
                return
            if self.send_audio_instant:
                await self.provider.send_audio(gemini_audio)
            else:
                offset = 0
                while offset < len(gemini_audio):
                    chunk = gemini_audio[offset : offset + self._chunk_size]
                    await self.provider.send_audio(chunk)
                    offset += len(chunk)
                    await asyncio.sleep(self.VOIP_PACKET_INTERVAL_MS / 1000)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(send_audio(), receive_events())

        # Process ALL received events - don't break early
        # Text events (transcripts) and audio events must all be processed
        # to ensure transcript text is properly associated with audio.
        # The _buffer_excess_audio step will limit audio output to bytes_per_tick
        # and buffer any excess for the next tick.
        for event in events:
            await self._process_event(result, event, gemini_audio_received)

        # Convert Gemini audio to telephony format and add to result
        for gemini_bytes, item_id in gemini_audio_received:
            telephony_bytes = self._audio_converter.convert_output(gemini_bytes)
            result.agent_audio_chunks.append((telephony_bytes, item_id))

        # Record simulation timing
        result.tick_sim_duration_ms = result.audio_sent_duration_ms

        # Move excess agent audio to buffer for next tick
        self._buffered_agent_audio = buffer_excess_audio(result, self.bytes_per_tick)

        # Calculate proportional transcript
        result.proportional_transcript = get_proportional_transcript(
            result.agent_audio_chunks, self._utterance_transcripts
        )

        # Update skip state for next tick
        self._skip_item_id = result.skip_item_id

        # Update cumulative user audio tracking
        self._cumulative_user_audio_ms += int(result.audio_sent_duration_ms)

        logger.info(f"Tick {tick_number} completed:\n{result.summary()}")

        return result

    async def _process_event(
        self,
        result: TickResult,
        event: Any,
        gemini_audio_received: List[Tuple[bytes, Optional[str]]],
    ) -> None:
        """Process a Gemini event."""
        result.events.append(event)

        if isinstance(event, GeminiAudioDeltaEvent):
            item_id = event.item_id or self._current_item_id

            # Skip audio from truncated item
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                # Estimate discarded bytes after conversion
                estimated_telephony_bytes = int(
                    len(event.data)
                    * TELEPHONY_BYTES_PER_SECOND
                    / GEMINI_OUTPUT_BYTES_PER_SECOND
                )
                result.truncated_audio_bytes += estimated_telephony_bytes
                return

            # Store Gemini audio for conversion after tick
            gemini_audio_received.append((event.data, item_id))

            # Track for transcript distribution
            if item_id:
                self._current_item_id = item_id
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                # Estimate telephony bytes for transcript tracking
                estimated_telephony_bytes = int(
                    len(event.data)
                    * TELEPHONY_BYTES_PER_SECOND
                    / GEMINI_OUTPUT_BYTES_PER_SECOND
                )
                self._utterance_transcripts[item_id].add_audio(
                    estimated_telephony_bytes
                )

        elif isinstance(event, GeminiTextDeltaEvent):
            item_id = event.item_id or self._current_item_id
            if item_id:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.text)

        elif isinstance(event, GeminiInterruptionEvent):
            logger.debug("Interruption detected from Gemini")
            result.vad_events.append("interrupted")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

            # Reset audio converter state after interruption
            self._audio_converter.reset()

        elif isinstance(event, GeminiFunctionCallDoneEvent):
            # Store original Gemini ID (often empty/null)
            original_gemini_id = event.call_id or ""

            # Generate synthetic ID for internal tracking (Gemini often sends null IDs)
            synthetic_id = (
                event.call_id if event.call_id else f"gemini_{uuid.uuid4().hex[:8]}"
            )

            # Store both original ID and name for when we send the result back
            # We need the original ID to send to Gemini, and the name for matching
            self._tool_call_info[synthetic_id] = (original_gemini_id, event.name)

            # Extract tool call with synthetic ID (for internal tracking)
            tool_call = ToolCall(
                id=synthetic_id,
                name=event.name,
                arguments=event.arguments,
            )
            result.tool_calls.append(tool_call)
            logger.debug(f"Tool call detected: {event.name}({synthetic_id})")

        elif isinstance(event, GeminiInputTranscriptionEvent):
            logger.debug(f"Input transcription: {event.transcript}")

        elif isinstance(event, GeminiTurnCompleteEvent):
            logger.debug("Turn complete")

        elif isinstance(event, GeminiAudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, GeminiTimeoutEvent):
            # Normal timeout, continue
            pass

        elif isinstance(event, GeminiGoAwayEvent):
            logger.warning(
                f"GoAway received, server will disconnect in {event.time_left_seconds}s"
            )
            # The provider will handle reconnection automatically

        elif isinstance(event, GeminiSessionResumptionEvent):
            logger.debug(f"Session resumption update: resumable={event.resumable}")
            # The provider stores the handle automatically

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
            call_id: The tool call ID (synthetic ID from our tracking).
            result: The tool result as a string.
            request_response: If True, request a response after sending.
            is_error: If True, the tool call failed and result contains error details.
        """
        # Look up original Gemini ID and function name from our tracking dictionary
        # Pop it since each tool call should only be responded to once
        info = self._tool_call_info.pop(call_id, None)
        if info is None:
            logger.warning(
                f"Unknown tool call ID: {call_id}, using empty ID and 'unknown' name"
            )
            original_id = ""
            name = "unknown"
        else:
            original_id, name = info

        # Queue with original Gemini ID (not our synthetic one) so Gemini can match it
        self._pending_tool_results.append(
            (original_id, name, result, request_response, is_error)
        )
        logger.debug(
            f"Queued tool result for {name}(gemini_id={original_id!r}, is_error={is_error})"
        )

    def clear_buffers(self) -> None:
        """Clear all internal audio and transcript buffers."""
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._pending_tool_results.clear()
        self._tool_call_info.clear()
        self._audio_converter.reset()
        self._skip_item_id = None
