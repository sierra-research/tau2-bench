"""Discrete-time audio native adapter for tick-based full-duplex simulation.

This adapter provides a tick-based interface for audio native APIs, designed
for discrete-time simulation where audio time is the primary clock and
wall-clock time must meet minimum guarantees per tick.

Key features:
- Tick-based interface via run_tick() instead of request-response
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Timing guarantee: wall-clock time >= tick duration
- Proportional transcript: text distributed based on audio played
- Interruption handling: client-side truncation on SpeechStartedEvent

Usage:
    adapter = DiscreteTimeAudioNativeAdapter(
        tick_duration_ms=1000,
        send_audio_instant=True,
        buffer_until_complete=False,
    )
    adapter.connect(system_prompt, tools, vad_config, modality="audio")

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)
        # result.get_played_agent_audio() - capped agent audio
        # result.proportional_transcript - text for this tick
        # result.events - all events received

    adapter.disconnect()

See docs/architecture/discrete_time_audio_native_agent.md for design details.
"""

import asyncio
import threading
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
    DEFAULT_OPENAI_VAD_THRESHOLD,
)
from tau2.data_model.audio import AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
    OpenAIVADMode,
)
from tau2.voice.audio_native.openai.tick_runner import TickResult, TickRunner


class DiscreteTimeAudioNativeAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex audio native simulation.

    Implements DiscreteTimeAdapter for the OpenAI Realtime API.

    This adapter runs an async event loop in a background thread to communicate
    with the OpenAI Realtime API, while exposing a synchronous interface for
    the agent and orchestrator.

    The primary method is run_tick(), which handles one tick of simulation:
    - Sends user audio to the API
    - Collects events for the tick duration
    - Returns TickResult with audio, transcript, and timing info

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick (derived from tick_duration_ms and audio_format).
        audio_format: Audio format for communication with the API.
        send_audio_instant: If True, send audio in one call per tick.
        buffer_until_complete: If True, wait for complete utterances.
        model: Model to use. Defaults to None. If provider is also provided, this is ignored.
        provider: Optional provider instance. Created lazily if not provided.
            If not provided, auto-detects auth from env vars (OPENAI_API_KEY).
        audio_format: Audio format for API communication. Defaults to telephony
            (8kHz μ-law). This determines bytes_per_tick calculation and must
            match the format configured in the provider session.
        fast_forward_mode: If True, exit tick early when we have enough audio.
    """

    DEFAULT_VOIP_PACKET_INTERVAL_MS = DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool,
        buffer_until_complete: bool,
        model: Optional[str] = None,
        provider: Optional[OpenAIRealtimeProvider] = None,
        audio_format: Optional[AudioFormat] = None,
        fast_forward_mode: bool = False,
    ):
        """Initialize the discrete-time adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
                If False, send in 20ms chunks with sleeps (VoIP-style).
            buffer_until_complete: If True, wait until an utterance is complete
                (AudioDoneEvent received) before including its audio/text in results.
                This guarantees accurate timing since we know the full utterance length.
                If False, stream audio/text as received and use proportional
                distribution for text.
            model: Model to use. Defaults to None. If provider is also provided, this is ignored.
            provider: Optional provider instance. Created lazily if not provided.
            audio_format: Audio format for API communication. Defaults to telephony
                (8kHz μ-law). This determines bytes_per_tick calculation and must
                match the format configured in the provider session.
            fast_forward_mode: If True, exit tick early when we have enough audio
                buffered (>= bytes_per_tick), rather than waiting for wall-clock time.
                This speeds up simulation when the API responds quickly.

        Raises:
            ValueError: If tick_duration_ms is <= 0.
        """
        super().__init__(tick_duration_ms, audio_format=audio_format)

        self.send_audio_instant = send_audio_instant
        self.buffer_until_complete = buffer_until_complete
        self.fast_forward_mode = fast_forward_mode

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Async event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Tick runner - created on connect
        self._tick_runner: Optional[TickRunner] = None
        self._tick_count = 0

        # Tool result queue (for sending tool results in next tick)
        self._pending_tool_results: List[tuple[str, str, bool]] = []

    @property
    def provider(self) -> OpenAIRealtimeProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = OpenAIRealtimeProvider(model=self.model)
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Check if connected to the API."""
        return self._connected and self._loop is not None

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to the API and configure the session.

        Starts a background thread with an async event loop for API communication.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to SERVER_VAD.
            modality: "audio" or "audio_in_text_out".
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = OpenAIVADConfig(
                mode=OpenAIVADMode.SERVER_VAD,
                threshold=DEFAULT_OPENAI_VAD_THRESHOLD,
            )

        # Store config for async initialization (including audio_format)
        self._connect_config = {
            "system_prompt": system_prompt,
            "tools": tools,
            "vad_config": vad_config,
            "modality": modality,
            "audio_format": self.audio_format,
            "fast_forward_mode": self.fast_forward_mode,
        }

        # Start background thread with event loop
        self._start_background_loop()

        # Connect and configure in background loop
        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(**self._connect_config),
            self._loop,
        )

        # Wait for connection with timeout
        try:
            future.result(timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT)
            self._connected = True
            logger.info(
                f"DiscreteTimeAudioNativeAdapter connected to OpenAI Realtime API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(
                f"DiscreteTimeAudioNativeAdapter failed to connect to OpenAI Realtime API: "
                f"{type(e).__name__}: {e}"
            )
            self._stop_background_loop()
            raise RuntimeError(f"Failed to connect to OpenAI Realtime API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any,
        modality: str,
        audio_format: AudioFormat,
        fast_forward_mode: bool,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
            audio_format=audio_format,
        )

        # Calculate chunk size from audio format (20ms chunks)
        chunk_size = int(
            audio_format.bytes_per_second * self.DEFAULT_VOIP_PACKET_INTERVAL_MS / 1000
        )

        # Create tick runner
        self._tick_runner = TickRunner(
            provider=self.provider,
            tick_duration_ms=self.tick_duration_ms,
            bytes_per_tick=self.bytes_per_tick,
            send_audio_instant=self.send_audio_instant,
            chunk_size=chunk_size,
            voip_packet_interval_ms=self.DEFAULT_VOIP_PACKET_INTERVAL_MS,
            buffer_until_complete=self.buffer_until_complete,
            audio_format=audio_format,
            fast_forward_mode=fast_forward_mode,
        )

    def disconnect(self) -> None:
        """Disconnect from the API and clean up resources."""
        if not self._connected:
            return

        # Disconnect in background loop
        if self._loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._async_disconnect(),
                self._loop,
            )
            try:
                future.result(timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT)
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        # Stop background loop
        self._stop_background_loop()
        self._connected = False
        self._tick_runner = None
        self._tick_count = 0
        logger.info("DiscreteTimeAudioNativeAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """Async disconnection."""
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def _start_background_loop(self) -> None:
        """Start the background thread with async event loop."""
        if self._loop is not None:
            return

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            import time

            time.sleep(0.01)

    def _stop_background_loop(self) -> None:
        """Stop the background thread and event loop."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT)
            self._loop = None
            self._thread = None

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """Run one tick of the simulation.

        This is the primary method for discrete-time simulation.

        Args:
            user_audio: User audio bytes for this tick (telephony format, 8kHz μ-law).
            tick_number: Optional tick number for logging. Auto-incremented if not provided.

        Returns:
            TickResult containing:
            - user_audio_data: The user audio sent
            - agent_audio_chunks: Agent audio for this tick (capped at bytes_per_tick)
            - events: All events received during the tick
            - proportional_transcript: Text corresponding to played audio
            - tick_sim_duration_ms: Simulated duration of this tick
            - truncation info if interruption occurred

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            # Provide detailed state for debugging
            loop_status = "loop running" if self._loop is not None else "no event loop"
            provider_status = (
                "provider connected"
                if self._provider and self._provider.is_connected
                else "provider not connected"
            )
            raise RuntimeError(
                f"Not connected to OpenAI Realtime API. Call connect() first. "
                f"[internal state: _connected={self._connected}, {loop_status}, {provider_status}]"
            )

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        # Run tick in background loop
        future = asyncio.run_coroutine_threadsafe(
            self._async_run_tick(user_audio, tick_number),
            self._loop,
        )

        # Wait for result
        try:
            result = future.result(
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER
            )
            return result
        except Exception as e:
            # Add context about connection state to help diagnose issues
            connection_status = "connected" if self.is_connected else "disconnected"
            provider_connected = (
                "provider connected"
                if self.provider.is_connected
                else "provider disconnected"
            )
            logger.error(
                f"Error in run_tick (tick={tick_number}): {type(e).__name__}: {e} "
                f"[adapter={connection_status}, {provider_connected}]"
            )
            raise

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Async tick execution."""
        # Send any pending tool results first
        for call_id, result, request_response in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result, request_response)
        self._pending_tool_results.clear()

        # Run the tick
        result = await self._tick_runner.run_tick(
            user_audio=user_audio,
            tick_number=tick_number,
        )

        # Log tick summary at debug level
        logger.info(f"Tick {tick_number} completed:\n{result.summary()}")

        return result

    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Queue a tool result to be sent in the next tick.

        Tool results are queued and sent at the start of the next run_tick() call.
        This ensures proper timing in the discrete-time simulation.

        Args:
            call_id: The tool call ID.
            result: The tool result as a string.
            request_response: If True, request a response after sending.
            is_error: If True, the tool call failed. Currently unused by OpenAI
                (error info is embedded in the result string).
        """
        self._pending_tool_results.append((call_id, result, request_response))
        logger.debug(f"Queued tool result for call_id={call_id}")

    def clear_buffers(self) -> None:
        """Clear all internal audio and transcript buffers.

        Call this when an interruption requires discarding buffered content.
        """
        if self._tick_runner is not None:
            self._tick_runner.buffered_agent_audio = []
            self._tick_runner.utterance_transcripts = {}
            self._tick_runner.pending_utterances = {}
            self._tick_runner.completed_utterances = []
            self._tick_runner.skip_item_id = None
        self._pending_tool_results.clear()
