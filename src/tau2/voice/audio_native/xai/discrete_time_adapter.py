"""Discrete-time audio native adapter for xAI Grok Voice Agent API.

This adapter provides a tick-based interface for xAI Realtime API, designed
for discrete-time simulation where audio time is the primary clock.

Key features:
- Tick-based interface via run_tick()
- Native G.711 μ-law support (NO audio conversion needed!)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via VAD speech_started events

Usage:
    adapter = DiscreteTimeXAIAdapter(
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

Reference: https://docs.x.ai/docs/guides/voice/agent
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
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
    buffer_excess_audio,
    get_proportional_transcript,
)
from tau2.voice.audio_native.xai.events import (
    XAIAudioDeltaEvent,
    XAIAudioDoneEvent,
    XAIAudioTranscriptDeltaEvent,
    XAIFunctionCallArgumentsDoneEvent,
    XAIInputTranscriptionCompletedEvent,
    XAIResponseDoneEvent,
    XAISpeechStartedEvent,
    XAISpeechStoppedEvent,
    XAITimeoutEvent,
)
from tau2.voice.audio_native.xai.provider import (
    XAIAudioFormat,
    XAIRealtimeProvider,
    XAIVADConfig,
)

# xAI with G.711 μ-law at 8kHz = 8000 bytes per second (1 byte per sample)
XAI_TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 8000


def calculate_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate bytes per tick for G.711 μ-law at 8kHz."""
    return int(XAI_TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)


class DiscreteTimeXAIAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with xAI Grok Voice Agent API.

    Implements DiscreteTimeAdapter for xAI Realtime API.

    This adapter runs an async event loop in a background thread to communicate
    with the xAI API, while exposing a synchronous interface for the agent
    and orchestrator.

    Key advantage: xAI natively supports G.711 μ-law at 8kHz, so NO audio
    conversion is needed! Audio passes through directly.

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
        provider: Optional[XAIRealtimeProvider] = None,
        voice: str = "Ara",
    ):
        """Initialize the discrete-time xAI adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            provider: Optional provider instance. Created lazily if not provided.
            voice: Voice to use. One of: Ara, Rex, Sal, Eve, Leo. Default: Ara.
        """
        super().__init__(tick_duration_ms)

        self.send_audio_instant = send_audio_instant
        self._chunk_size = int(
            XAI_TELEPHONY_BYTES_PER_SECOND * self.VOIP_PACKET_INTERVAL_MS / 1000
        )
        self.voice = voice

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

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
        self._pending_tool_results: List[Tuple[str, str, bool]] = []

    @property
    def provider(self) -> XAIRealtimeProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = XAIRealtimeProvider(
                voice=self.voice,
                audio_format=XAIAudioFormat.PCMU,  # G.711 μ-law
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
        """Connect to the xAI API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to server VAD.
            modality: Ignored (xAI always uses audio).
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = XAIVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeXAIAdapter connected to xAI API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeXAIAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to xAI API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: XAIVADConfig,
    ) -> None:
        """Async connection and configuration."""
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
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
        logger.info("DiscreteTimeXAIAdapter disconnected")

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
            raise RuntimeError("Not connected to xAI API. Call connect() first.")

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
            audio_sent_duration_ms=(len(user_audio) / XAI_TELEPHONY_BYTES_PER_SECOND)
            * 1000,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self._cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=XAI_TELEPHONY_BYTES_PER_SECOND,
            silence_byte=TELEPHONY_ULAW_SILENCE,
        )

        # Add any buffered agent audio from previous tick
        for chunk_data, item_id in self._buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, item_id))
        self._buffered_agent_audio.clear()

        # Carry over skip state from previous tick
        result.skip_item_id = self._skip_item_id

        # Send audio and receive events concurrently
        async def send_audio():
            """Send audio (instant or chunked based on config)."""
            if len(user_audio) == 0:
                return
            if self.send_audio_instant:
                await self.provider.send_audio(user_audio)
            else:
                offset = 0
                while offset < len(user_audio):
                    chunk = user_audio[offset : offset + self._chunk_size]
                    await self.provider.send_audio(chunk)
                    offset += len(chunk)
                    await asyncio.sleep(self.VOIP_PACKET_INTERVAL_MS / 1000)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(send_audio(), receive_events())

        # Process all received events
        for event in events:
            await self._process_event(result, event)

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

    async def _process_event(self, result: TickResult, event: Any) -> None:
        """Process an xAI event."""
        result.events.append(event)

        if isinstance(event, XAIAudioDeltaEvent):
            item_id = event.item_id or self._current_item_id

            # Skip audio from truncated item
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                # Decode and count discarded bytes
                audio_bytes = base64.b64decode(event.delta) if event.delta else b""
                result.truncated_audio_bytes += len(audio_bytes)
                return

            # Decode base64 audio (already in G.711 μ-law format!)
            audio_bytes = base64.b64decode(event.delta) if event.delta else b""
            if audio_bytes:
                result.agent_audio_chunks.append((audio_bytes, item_id))

            # Track for transcript distribution
            if item_id:
                self._current_item_id = item_id
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_audio(len(audio_bytes))

        elif isinstance(event, XAIAudioTranscriptDeltaEvent):
            item_id = event.item_id or self._current_item_id
            if item_id and event.delta:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.delta)

        elif isinstance(event, XAISpeechStartedEvent):
            logger.debug("Speech started - interruption detected")
            result.vad_events.append("speech_started")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

        elif isinstance(event, XAIFunctionCallArgumentsDoneEvent):
            # Parse arguments
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

        elif isinstance(event, XAISpeechStoppedEvent):
            logger.debug("Speech stopped")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, XAIInputTranscriptionCompletedEvent):
            logger.debug(f"Input transcription: {event.transcript}")

        elif isinstance(event, XAIResponseDoneEvent):
            logger.debug("Response done (turn complete)")

        elif isinstance(event, XAIAudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, XAITimeoutEvent):
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
            request_response: If True, request a response after sending.
            is_error: If True, the tool call failed. Currently unused by xAI.
        """
        self._pending_tool_results.append((call_id, result, request_response))
        logger.debug(f"Queued tool result for call_id={call_id}")

    def clear_buffers(self) -> None:
        """Clear all internal audio and transcript buffers."""
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._pending_tool_results.clear()
        self._skip_item_id = None
