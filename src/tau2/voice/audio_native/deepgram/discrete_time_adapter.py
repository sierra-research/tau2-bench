"""Discrete-time audio native adapter for Deepgram Voice Agent API.

This adapter provides a tick-based interface for Deepgram Voice Agent API, designed
for discrete-time simulation where audio time is the primary clock.

Note: Deepgram Voice Agent is a CASCADED system (STT → LLM → TTS), unlike native
audio models (OpenAI Realtime, Gemini Live, Nova Sonic) that process audio directly.

Key features:
- Tick-based interface via run_tick()
- Audio format conversion (telephony ↔ Deepgram formats)
- Audio capping: max bytes_per_tick of agent audio per tick
- Audio buffering: excess agent audio carries to next tick
- Proportional transcript: text distributed based on audio played
- Interruption handling via UserStartedSpeaking events

Usage:
    adapter = DiscreteTimeDeepgramAdapter(
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
import base64
import json
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
from tau2.voice.audio_native.deepgram.audio_utils import (
    DEEPGRAM_OUTPUT_BYTES_PER_SECOND,
    TELEPHONY_BYTES_PER_SECOND,
    StreamingDeepgramConverter,
    calculate_deepgram_bytes_per_tick,
)
from tau2.voice.audio_native.deepgram.events import (
    DeepgramAgentAudioDoneEvent,
    DeepgramAgentStartedSpeakingEvent,
    DeepgramAudioEvent,
    DeepgramConversationTextEvent,
    DeepgramErrorEvent,
    DeepgramFunctionCallRequestEvent,
    DeepgramTimeoutEvent,
    DeepgramUserStartedSpeakingEvent,
)
from tau2.voice.audio_native.deepgram.provider import (
    DEEPGRAM_INPUT_BYTES_PER_SECOND,
    DeepgramVADConfig,
    DeepgramVoiceAgentProvider,
)
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
    buffer_excess_audio,
    get_proportional_transcript,
)


class DiscreteTimeDeepgramAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Deepgram Voice Agent API.

    Implements DiscreteTimeAdapter for Deepgram Voice Agent.

    This adapter runs an async event loop in a background thread to communicate
    with the Deepgram Voice Agent API, while exposing a synchronous interface for
    the agent and orchestrator.

    Audio format handling:
    - Input: Receives telephony audio (8kHz μ-law), converts to 16kHz PCM16
    - Output: Receives 16kHz PCM16 from Deepgram, converts to 8kHz μ-law

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        bytes_per_tick: Audio bytes per tick in telephony format (8kHz μ-law).
        send_audio_instant: If True, send audio in one call per tick.
            If False, send in 20ms chunks with sleeps (VoIP-style streaming).
        provider: Optional provider instance. Created lazily if not provided.
    """

    VOIP_PACKET_INTERVAL_MS = DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        provider: Optional[DeepgramVoiceAgentProvider] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
    ):
        """Initialize the discrete-time Deepgram adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
            provider: Optional provider instance. Created lazily if not provided.
            llm_provider: LLM provider (e.g., "open_ai", "anthropic").
            llm_model: LLM model (e.g., "gpt-4o-mini").
            tts_model: TTS model including voice (e.g., "aura-2-thalia-en").
        """
        super().__init__(tick_duration_ms)

        self.send_audio_instant = send_audio_instant
        self._chunk_size = int(
            DEEPGRAM_INPUT_BYTES_PER_SECOND * self.VOIP_PACKET_INTERVAL_MS / 1000
        )
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tts_model = tts_model

        # Deepgram output format (16kHz PCM16) - for internal processing
        self._deepgram_output_bytes_per_tick = calculate_deepgram_bytes_per_tick(
            tick_duration_ms, direction="output"
        )

        # Provider - created lazily if not provided
        self._provider = provider
        self._owns_provider = provider is None

        # Audio format converter (preserves state for streaming)
        self._audio_converter = StreamingDeepgramConverter()

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
        self._pending_tool_results: List[Tuple[str, str, str, bool]] = []

        # Track tool call info for Deepgram
        # Maps call_id -> function_name
        self._tool_call_info: Dict[str, str] = {}

    @property
    def provider(self) -> DeepgramVoiceAgentProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = DeepgramVoiceAgentProvider(
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                tts_model=self.tts_model,
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
        """Connect to the Deepgram Voice Agent API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration. Defaults to automatic VAD.
            modality: "audio" or "text" (Deepgram always uses audio).
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        # Default VAD config
        if vad_config is None:
            vad_config = DeepgramVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeDeepgramAdapter connected to Deepgram Voice Agent API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(
                f"DiscreteTimeDeepgramAdapter failed to connect to Deepgram Voice Agent API: "
                f"{type(e).__name__}: {e}"
            )
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to Deepgram Voice Agent API: {e}"
            ) from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: DeepgramVADConfig,
        modality: str,
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
        self._tool_call_info.clear()
        self._audio_converter.reset()
        logger.info("DiscreteTimeDeepgramAdapter disconnected")

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
                "Not connected to Deepgram Voice Agent API. Call connect() first."
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
        # Send any pending tool results first
        for call_id, name, result_str, request_response in self._pending_tool_results:
            await self.provider.send_tool_result(
                call_id=call_id,
                function_name=name,
                result=result_str,
            )
        self._pending_tool_results.clear()

        # Convert user audio from telephony to Deepgram format
        deepgram_audio = self._audio_converter.convert_input(user_audio)

        # Calculate timing
        tick_start = asyncio.get_running_loop().time()

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
        deepgram_audio_received: List[Tuple[bytes, Optional[str]]] = []

        # Send audio and receive events concurrently
        async def send_audio():
            """Send audio (instant or chunked based on config)."""
            if len(deepgram_audio) == 0:
                return
            if self.send_audio_instant:
                await self.provider.send_audio(deepgram_audio)
            else:
                offset = 0
                while offset < len(deepgram_audio):
                    chunk = deepgram_audio[offset : offset + self._chunk_size]
                    await self.provider.send_audio(chunk)
                    offset += len(chunk)
                    await asyncio.sleep(self.VOIP_PACKET_INTERVAL_MS / 1000)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(send_audio(), receive_events())

        # Process ALL received events
        for event in events:
            await self._process_event(result, event, deepgram_audio_received)

        # Convert Deepgram audio to telephony format and add to result
        for deepgram_bytes, item_id in deepgram_audio_received:
            telephony_bytes = self._audio_converter.convert_output(deepgram_bytes)
            result.agent_audio_chunks.append((telephony_bytes, item_id))

        # Log audio conversion result
        if deepgram_audio_received:
            total_deepgram = sum(len(d) for d, _ in deepgram_audio_received)
            total_telephony = sum(len(d) for d, _ in result.agent_audio_chunks)
            logger.debug(
                f"Audio conversion: {len(deepgram_audio_received)} chunks, "
                f"{total_deepgram} deepgram bytes -> {total_telephony} telephony bytes"
            )

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
        deepgram_audio_received: List[Tuple[bytes, Optional[str]]],
    ) -> None:
        """Process a Deepgram event."""
        result.events.append(event)

        # Handle audio events - binary audio is converted to DeepgramAudioEvent by provider
        if isinstance(event, DeepgramAudioEvent):
            # Decode base64 audio
            audio_data = base64.b64decode(event.audio) if event.audio else b""
            logger.debug(
                f"DeepgramAudioEvent: {len(audio_data)} bytes, "
                f"item_id={self._current_item_id}"
            )
            if audio_data:
                item_id = self._current_item_id or str(uuid.uuid4())[:8]

                # Skip audio from truncated item
                if result.skip_item_id is not None and item_id == result.skip_item_id:
                    estimated_telephony_bytes = int(
                        len(audio_data)
                        * TELEPHONY_BYTES_PER_SECOND
                        / DEEPGRAM_OUTPUT_BYTES_PER_SECOND
                    )
                    result.truncated_audio_bytes += estimated_telephony_bytes
                else:
                    deepgram_audio_received.append((audio_data, item_id))

                    # Track for transcript distribution
                    if item_id not in self._utterance_transcripts:
                        self._utterance_transcripts[item_id] = UtteranceTranscript(
                            item_id=item_id
                        )
                    estimated_telephony_bytes = int(
                        len(audio_data)
                        * TELEPHONY_BYTES_PER_SECOND
                        / DEEPGRAM_OUTPUT_BYTES_PER_SECOND
                    )
                    self._utterance_transcripts[item_id].add_audio(
                        estimated_telephony_bytes
                    )

        elif isinstance(event, DeepgramConversationTextEvent):
            # Track transcripts for proportional distribution
            if event.role == "assistant" and event.content:
                item_id = self._current_item_id or str(uuid.uuid4())[:8]
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.content)
                self._current_item_id = item_id

        elif isinstance(event, DeepgramUserStartedSpeakingEvent):
            logger.debug("User started speaking - interruption detected")
            result.vad_events.append("user_started_speaking")
            # Clear buffered audio
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Mark truncation
            result.was_truncated = True
            result.skip_item_id = self._current_item_id

            # Clear current item_id so next response gets a new one
            # This prevents the skip_item_id from blocking all future audio
            self._current_item_id = None

            # Reset audio converter state after interruption
            self._audio_converter.reset()

        elif isinstance(event, DeepgramAgentStartedSpeakingEvent):
            # New agent utterance
            self._current_item_id = str(uuid.uuid4())[:8]
            logger.debug(f"Agent started speaking (item_id={self._current_item_id})")

        elif isinstance(event, DeepgramAgentAudioDoneEvent):
            logger.debug("Agent audio done")
            # Clear skip state - the previous utterance is complete
            self._skip_item_id = None
            result.skip_item_id = None

        elif isinstance(event, DeepgramFunctionCallRequestEvent):
            # Process function calls
            for func in event.functions:
                # Store function name for when we send the result back
                self._tool_call_info[func.id] = func.name

                # Parse arguments
                try:
                    arguments = json.loads(func.arguments) if func.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=func.id,
                    name=func.name,
                    arguments=arguments,
                )
                result.tool_calls.append(tool_call)
                logger.debug(f"Tool call detected: {func.name}({func.id})")

        elif isinstance(event, DeepgramErrorEvent):
            logger.error(
                f"Deepgram error: {event.error_message} (code={event.error_code})"
            )

        elif isinstance(event, DeepgramTimeoutEvent):
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
            is_error: If True, the tool call failed. Currently unused by Deepgram.
        """
        # Look up function name from our tracking dictionary
        name = self._tool_call_info.pop(call_id, "unknown")

        # Queue for sending in next tick
        self._pending_tool_results.append((call_id, name, result, request_response))
        logger.debug(f"Queued tool result for {name}(call_id={call_id})")

    def clear_buffers(self) -> None:
        """Clear all internal audio and transcript buffers."""
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._pending_tool_results.clear()
        self._tool_call_info.clear()
        self._audio_converter.reset()
        self._skip_item_id = None
