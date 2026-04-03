"""Core cascaded voice provider for LiveKit-based STT → LLM → TTS pipeline.

This module contains the core logic for the cascaded voice pipeline:
- STT: Deepgram streaming with integrated VAD
- LLM: OpenAI/Anthropic with full parameter control
- TTS: Deepgram/ElevenLabs streaming synthesis

The provider handles:
- Turn-taking decisions (when to trigger LLM based on VAD/endpointing)
- Interruption handling (barge-in detection and response cancellation)
- Context management (conversation history accumulation)
- Streaming orchestration (LLM tokens → TTS sentences)

The DiscreteTimeAdapter wraps this provider for tick-based simulation.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from loguru import logger

from tau2.config import DEFAULT_PCM_SAMPLE_RATE
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.livekit.config import (
    AnthropicLLMConfig,
    CascadedConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    ElevenLabsTTSConfig,
    OpenAILLMConfig,
)

# =============================================================================
# Event Types
# =============================================================================


class CascadedEventType(str, Enum):
    """Event types emitted by the cascaded provider."""

    # VAD/Speech events
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"

    # Transcription events
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"

    # LLM events
    LLM_STARTED = "llm_started"
    LLM_TOKEN = "llm_token"
    LLM_COMPLETED = "llm_completed"
    TOOL_CALL = "tool_call"

    # TTS events
    TTS_STARTED = "tts_started"
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETED = "tts_completed"

    # Control events
    INTERRUPTED = "interrupted"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class CascadedEvent:
    """Event from the cascaded pipeline."""

    type: CascadedEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Convenience accessors
    @property
    def transcript(self) -> Optional[str]:
        return self.data.get("transcript")

    @property
    def audio(self) -> Optional[bytes]:
        return self.data.get("audio")

    @property
    def tool_call(self) -> Optional[ToolCall]:
        return self.data.get("tool_call")

    @property
    def text(self) -> Optional[str]:
        return self.data.get("text")


# =============================================================================
# Provider State
# =============================================================================


class ProviderState(str, Enum):
    """State of the cascaded provider."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    LISTENING = "listening"  # STT active, waiting for user
    PROCESSING = "processing"  # LLM generating response
    SPEAKING = "speaking"  # TTS playing audio


@dataclass
class ConversationContext:
    """Manages conversation history for the LLM."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""

    def add_system(self, content: str) -> None:
        """Set the system prompt."""
        self.system_prompt = content
        # Ensure system message is first
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call from the assistant."""
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                ],
            }
        )

    def add_tool_result(self, call_id: str, result: str) -> None:
        """Add a tool result."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            }
        )

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Get messages in OpenAI API format."""
        return self.messages.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }


# =============================================================================
# Turn-Taking Configuration
# =============================================================================


@dataclass
class TurnTakingConfig:
    """Configuration for turn-taking behavior.

    These parameters control when the provider decides the user has
    finished speaking and it's time to generate a response.
    """

    # TODO: Not functional yet - needs timer/polling mechanism.
    # Endpointing: How long to wait after speech ends before triggering LLM
    # This is in addition to Deepgram's endpointing_ms
    additional_silence_ms: int = 0

    # Minimum transcript length to trigger LLM (avoid triggering on "um", "uh")
    min_transcript_chars: int = 1

    # TODO: Not implemented - currently interruption is instant on SPEECH_STARTED
    # Interruption: How quickly to detect barge-in
    interruption_threshold_ms: int = 200

    # Whether to allow interruptions during TTS playback
    allow_interruptions: bool = True


# =============================================================================
# Core Provider
# =============================================================================


class CascadedVoiceProvider:
    """Core provider for cascaded STT → LLM → TTS voice pipeline.

    This class manages the full voice interaction loop:
    1. STT: Streams user audio to Deepgram, receives transcripts with VAD
    2. Turn-taking: Decides when user is done speaking
    3. LLM: Sends transcript to OpenAI/Anthropic, streams response
    4. TTS: Streams response text to Deepgram/ElevenLabs, receives audio

    The provider is event-driven and emits CascadedEvents that can be
    consumed by the discrete-time adapter or other consumers.

    Usage:
        provider = CascadedVoiceProvider(config)
        await provider.connect(system_prompt, tools)

        # Feed audio and receive events
        async for event in provider.process_audio(audio_chunk):
            if event.type == CascadedEventType.TTS_AUDIO:
                play_audio(event.audio)
            elif event.type == CascadedEventType.TOOL_CALL:
                result = execute_tool(event.tool_call)
                await provider.send_tool_result(event.tool_call.id, result)

        await provider.disconnect()
    """

    def __init__(
        self,
        config: Optional[CascadedConfig] = None,
        turn_taking: Optional[TurnTakingConfig] = None,
    ):
        """Initialize the cascaded provider.

        Args:
            config: Cascaded pipeline configuration. Uses defaults if None.
            turn_taking: Turn-taking configuration. Uses defaults if None.
        """
        self.config = config or CascadedConfig()
        self.turn_taking = turn_taking or TurnTakingConfig()

        # State
        self._state = ProviderState.DISCONNECTED
        self._context = ConversationContext()
        self._tools: List[Tool] = []

        # HTTP session for LiveKit plugins (required outside job context)
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Component clients
        self._stt_client: Optional[Any] = None
        self._llm_client: Optional[Any] = None
        self._tts_client: Optional[Any] = None

        # STT streaming state
        self._stt_stream: Optional[Any] = None
        self._stt_task: Optional[asyncio.Task] = None
        self._stt_event_queue: asyncio.Queue = asyncio.Queue()
        self._current_transcript: str = ""
        self._accumulated_transcript: str = ""
        self._is_user_speaking: bool = False
        self._speech_ended_time: Optional[float] = None
        self._last_transcript_time: Optional[float] = None

        # Pending operations
        self._pending_tool_results: List[tuple[str, str]] = []

        # Cancellation
        self._cancel_event: asyncio.Event = asyncio.Event()

    @property
    def state(self) -> ProviderState:
        """Current provider state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the provider is connected."""
        return self._state != ProviderState.DISCONNECTED

    @property
    def context(self) -> ConversationContext:
        """Current conversation context."""
        return self._context

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
    ) -> None:
        """Connect and initialize all components.

        Args:
            system_prompt: System prompt for the LLM.
            tools: List of tools available to the LLM.
        """
        if self.is_connected:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        self._tools = tools
        self._context = ConversationContext()
        self._context.add_system(system_prompt)
        self._cancel_event.clear()

        # Create HTTP session for LiveKit plugins (required outside job context)
        self._http_session = aiohttp.ClientSession()

        # Initialize components
        await self._init_stt()
        await self._init_llm()
        await self._init_tts()

        # Start STT stream
        await self._start_stt_stream()

        self._state = ProviderState.LISTENING
        logger.info(
            f"CascadedVoiceProvider connected "
            f"(STT={self.config.stt.provider}, "
            f"LLM={self.config.llm.provider}/{self.config.llm.model}, "
            f"TTS={self.config.tts.provider})"
        )

    async def disconnect(self) -> None:
        """Disconnect and cleanup all components."""
        if not self.is_connected:
            return

        # Signal cancellation
        self._cancel_event.set()

        # Stop STT stream
        await self._stop_stt_stream()

        # Cleanup clients
        self._stt_client = None
        self._llm_client = None
        self._tts_client = None

        # Close HTTP session
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

        self._state = ProviderState.DISCONNECTED
        self._accumulated_transcript = ""
        self._current_transcript = ""
        self._is_user_speaking = False
        self._last_transcript_time = None
        logger.info("CascadedVoiceProvider disconnected")

    # =========================================================================
    # Component Initialization
    # =========================================================================

    async def _init_stt(self) -> None:
        """Initialize STT client."""
        config = self.config.stt

        if isinstance(config, DeepgramSTTConfig):
            try:
                from livekit.plugins import deepgram

                self._stt_client = deepgram.STT(
                    model=config.model,
                    language=config.language,
                    interim_results=config.interim_results,
                    vad_events=config.vad_events,
                    endpointing_ms=config.endpointing_ms,
                    smart_format=config.smart_format,
                    punctuate=config.punctuate,
                    http_session=self._http_session,
                )
                logger.debug(f"Initialized Deepgram STT: {config.model}")
            except ImportError as e:
                logger.error(f"Failed to import livekit-plugins-deepgram: {e}")
                raise
        else:
            raise ValueError(f"Unknown STT config type: {type(config)}")

    async def _init_llm(self) -> None:
        """Initialize LLM client."""
        config = self.config.llm

        if isinstance(config, OpenAILLMConfig):
            try:
                from livekit.plugins import openai

                kwargs: Dict[str, Any] = {"model": config.model}
                if config.temperature is not None:
                    kwargs["temperature"] = config.temperature
                if config.top_p is not None:
                    kwargs["top_p"] = config.top_p
                if config.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = config.reasoning_effort
                if config.max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = config.max_completion_tokens

                self._llm_client = openai.LLM(**kwargs)
                logger.debug(f"Initialized OpenAI LLM: {config.model}")
            except ImportError as e:
                logger.error(f"Failed to import livekit-plugins-openai: {e}")
                raise

        elif isinstance(config, AnthropicLLMConfig):
            try:
                from livekit.plugins import anthropic

                kwargs = {
                    "model": config.model,
                    "max_tokens": config.max_tokens,
                }
                if config.temperature is not None:
                    kwargs["temperature"] = config.temperature

                self._llm_client = anthropic.LLM(**kwargs)
                logger.debug(f"Initialized Anthropic LLM: {config.model}")
            except ImportError as e:
                logger.error(f"Failed to import livekit-plugins-anthropic: {e}")
                raise
        else:
            raise ValueError(f"Unknown LLM config type: {type(config)}")

    async def _init_tts(self) -> None:
        """Initialize TTS client."""
        config = self.config.tts

        if isinstance(config, DeepgramTTSConfig):
            try:
                from livekit.plugins import deepgram

                self._tts_client = deepgram.TTS(
                    model=config.model,
                    http_session=self._http_session,
                )
                logger.debug(f"Initialized Deepgram TTS: {config.model}")
            except ImportError as e:
                logger.error(f"Failed to import livekit-plugins-deepgram: {e}")
                raise

        elif isinstance(config, ElevenLabsTTSConfig):
            try:
                from livekit.plugins import elevenlabs

                self._tts_client = elevenlabs.TTS(
                    voice=config.voice_id,
                    model=config.model,
                    http_session=self._http_session,
                )
                logger.debug(f"Initialized ElevenLabs TTS: {config.voice_id}")
            except ImportError as e:
                logger.error(f"Failed to import livekit-plugins-elevenlabs: {e}")
                raise
        else:
            raise ValueError(f"Unknown TTS config type: {type(config)}")

    # =========================================================================
    # STT Streaming
    # =========================================================================

    async def _start_stt_stream(self) -> None:
        """Start the persistent STT stream and background receiver task."""
        if self._stt_client is None:
            return

        try:
            # Create the STT stream
            self._stt_stream = self._stt_client.stream()

            # Start background task to receive STT events
            self._stt_task = asyncio.create_task(self._stt_receiver_task())
            logger.debug("Started STT stream")
        except Exception as e:
            logger.error(f"Failed to start STT stream: {e}")
            raise

    async def _stop_stt_stream(self) -> None:
        """Stop the STT stream and receiver task."""
        # Cancel receiver task
        if self._stt_task is not None:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                pass
            self._stt_task = None

        # Close stream
        if self._stt_stream is not None:
            try:
                await self._stt_stream.aclose()
            except Exception as e:
                logger.debug(f"Error closing STT stream: {e}")
            self._stt_stream = None

        # Clear queue
        while not self._stt_event_queue.empty():
            try:
                self._stt_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _stt_receiver_task(self) -> None:
        """Background task that receives events from the STT stream.

        Events are queued for processing in process_audio().
        """
        try:
            from livekit.agents import stt as lk_stt

            logger.debug("STT receiver task started, waiting for events...")
            async for event in self._stt_stream:
                logger.debug(f"STT event received: type={event.type}")
                if self._cancel_event.is_set():
                    break

                # Convert LiveKit STT events to our CascadedEvents
                if event.type == lk_stt.SpeechEventType.START_OF_SPEECH:
                    await self._stt_event_queue.put(
                        CascadedEvent(type=CascadedEventType.SPEECH_STARTED)
                    )
                elif event.type == lk_stt.SpeechEventType.END_OF_SPEECH:
                    await self._stt_event_queue.put(
                        CascadedEvent(type=CascadedEventType.SPEECH_ENDED)
                    )
                elif event.type == lk_stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    await self._stt_event_queue.put(
                        CascadedEvent(
                            type=CascadedEventType.TRANSCRIPT_PARTIAL,
                            data={"transcript": text},
                        )
                    )
                elif event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    await self._stt_event_queue.put(
                        CascadedEvent(
                            type=CascadedEventType.TRANSCRIPT_FINAL,
                            data={"transcript": text},
                        )
                    )

        except asyncio.CancelledError:
            logger.debug("STT receiver task cancelled")
        except Exception as e:
            logger.error(f"STT receiver task error: {e}")
            await self._stt_event_queue.put(
                CascadedEvent(
                    type=CascadedEventType.ERROR,
                    data={"error": str(e), "stage": "stt"},
                )
            )

    async def _push_audio_to_stt(self, audio: bytes) -> None:
        """Push audio bytes to the STT stream.

        Args:
            audio: Audio bytes (16kHz mono PCM expected by Deepgram).
        """
        if self._stt_stream is None:
            logger.warning("STT stream is None, cannot push audio")
            return

        if len(audio) == 0:
            return

        try:
            from livekit import rtc

            # Create an AudioFrame from raw bytes
            # Deepgram expects 16kHz mono PCM
            sample_rate = DEFAULT_PCM_SAMPLE_RATE
            num_channels = 1
            samples_per_channel = len(audio) // 2  # 16-bit samples

            frame = rtc.AudioFrame(
                data=audio,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_channel,
            )

            # Push to stream
            self._stt_stream.push_frame(frame)
            logger.debug(
                f"Pushed {len(audio)} bytes ({samples_per_channel} samples) to STT"
            )

        except Exception as e:
            logger.error(f"Failed to push audio to STT: {e}", exc_info=True)

    async def _drain_stt_events(self, timeout: float = 0.0) -> List[CascadedEvent]:
        """Drain all available STT events from the queue.

        Args:
            timeout: Max time to wait for first event (0 = no wait).

        Returns:
            List of STT events.
        """
        events: List[CascadedEvent] = []

        # Try to get first event with timeout
        if timeout > 0:
            try:
                event = await asyncio.wait_for(
                    self._stt_event_queue.get(),
                    timeout=timeout,
                )
                events.append(event)
            except asyncio.TimeoutError:
                return events

        # Drain remaining events without waiting
        while not self._stt_event_queue.empty():
            try:
                event = self._stt_event_queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break

        return events

    # =========================================================================
    # Audio Processing
    # =========================================================================

    async def process_audio(
        self,
        audio: bytes,
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Process an audio chunk through the pipeline.

        This is the main entry point for audio processing. Feed audio chunks
        and receive events as the pipeline processes them.

        Args:
            audio: Audio bytes to process (16kHz mono PCM).

        Yields:
            CascadedEvent objects as they occur.
        """
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call connect() first.")

        # Push audio to STT stream
        await self._push_audio_to_stt(audio)

        # Give STT a moment to process, then drain events
        await asyncio.sleep(0.01)  # Small delay for processing

        # Drain and yield STT events
        stt_events = await self._drain_stt_events(timeout=0.05)

        for event in stt_events:
            yield event

            # Track speech state
            if event.type == CascadedEventType.SPEECH_STARTED:
                self._is_user_speaking = True
                self._speech_ended_time = None

            elif event.type == CascadedEventType.TRANSCRIPT_PARTIAL:
                self._current_transcript = event.transcript or ""
                self._last_transcript_time = time.time()

            elif event.type == CascadedEventType.TRANSCRIPT_FINAL:
                transcript = event.transcript or ""
                self._accumulated_transcript += " " + transcript
                self._accumulated_transcript = self._accumulated_transcript.strip()
                self._current_transcript = ""
                self._last_transcript_time = time.time()

            elif event.type == CascadedEventType.SPEECH_ENDED:
                self._is_user_speaking = False
                self._speech_ended_time = time.time()

                # Now that speech has ended, check if we should trigger LLM
                if self._should_trigger_llm():
                    async for llm_event in self._trigger_llm():
                        yield llm_event

        # Fallback: if we have accumulated transcript and no new transcript
        # activity for utterance_end_ms, trigger the LLM. This handles the
        # case where END_OF_SPEECH never fires (common with background noise).
        if self._accumulated_transcript and self._last_transcript_time is not None:
            silence_ms = (time.time() - self._last_transcript_time) * 1000
            utterance_end_ms = self.config.stt.utterance_end_ms
            if utterance_end_ms > 0 and silence_ms >= utterance_end_ms:
                if self._state == ProviderState.LISTENING:
                    self._is_user_speaking = False
                    async for llm_event in self._trigger_llm():
                        yield llm_event

    async def _trigger_llm(self) -> AsyncGenerator[CascadedEvent, None]:
        """Trigger LLM with the accumulated transcript."""
        final_transcript = self._accumulated_transcript
        self._accumulated_transcript = ""
        self._last_transcript_time = None

        logger.debug(f"Triggering LLM with transcript: {final_transcript[:50]}...")

        async for llm_event in self._process_llm(final_transcript):
            yield llm_event

    def _should_trigger_llm(self) -> bool:
        """Decide whether to trigger LLM generation.

        Returns:
            True if we should send accumulated transcript to LLM.
        """
        # Check minimum transcript length
        if len(self._accumulated_transcript) < self.turn_taking.min_transcript_chars:
            return False

        # Check if user has stopped speaking (via VAD endpointing)
        if self._is_user_speaking:
            return False

        # TODO: additional_silence_ms is not functional yet.
        # The current logic checks once immediately after TRANSCRIPT_FINAL,
        # but if elapsed_ms < threshold, it returns False and never retries.
        # To implement properly, we'd need a timer/polling mechanism that
        # re-checks after the silence threshold has elapsed.
        # For now, rely on Deepgram's endpointing_ms for turn detection.
        #
        # if self.turn_taking.additional_silence_ms > 0:
        #     if self._speech_ended_time is None:
        #         return False
        #     elapsed_ms = (time.time() - self._speech_ended_time) * 1000
        #     if elapsed_ms < self.turn_taking.additional_silence_ms:
        #         return False

        return True

    async def _process_llm(
        self,
        user_text: str,
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Process user text through LLM and TTS.

        Args:
            user_text: Transcribed user speech.

        Yields:
            LLM and TTS events.
        """
        if self._llm_client is None:
            return

        self._state = ProviderState.PROCESSING

        # Add user message to context
        self._context.add_user(user_text)

        # Log prompt if configured
        if self.config.log_prompts:
            logger.info(f"LLM Prompt:\n{json.dumps(self._context.to_dict(), indent=2)}")

        yield CascadedEvent(type=CascadedEventType.LLM_STARTED)

        try:
            # Build chat context for LiveKit LLM
            chat_ctx = self._build_chat_context()

            # Format tools for the LLM
            tools = self._format_tools()

            # Stream LLM response
            response_text = ""
            tool_calls: List[ToolCall] = []

            async with self._llm_client.chat(
                chat_ctx=chat_ctx,
                tools=tools if tools else None,
            ) as stream:
                async for chunk in stream:
                    if self._cancel_event.is_set():
                        break

                    if chunk.delta:
                        if chunk.delta.content:
                            token = chunk.delta.content
                            response_text += token
                            yield CascadedEvent(
                                type=CascadedEventType.LLM_TOKEN,
                                data={"token": token, "accumulated": response_text},
                            )

                        if chunk.delta.tool_calls:
                            for tc in chunk.delta.tool_calls:
                                tool_calls.append(self._parse_tool_call(tc))

            yield CascadedEvent(
                type=CascadedEventType.LLM_COMPLETED,
                data={"text": response_text, "tool_calls": tool_calls},
            )

            if response_text:
                self._context.add_assistant(response_text)

            if tool_calls:
                for tc in tool_calls:
                    self._context.add_tool_call(tc)
                    yield CascadedEvent(
                        type=CascadedEventType.TOOL_CALL,
                        data={"tool_call": tc},
                    )
            elif response_text:
                async for tts_event in self._process_tts(response_text):
                    yield tts_event

        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield CascadedEvent(
                type=CascadedEventType.ERROR,
                data={"error": str(e), "stage": "llm"},
            )
        finally:
            self._state = ProviderState.LISTENING

    async def _process_tts(
        self,
        text: str,
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Process text through TTS.

        Args:
            text: Text to synthesize.

        Yields:
            TTS audio events.
        """
        if self._tts_client is None:
            return

        self._state = ProviderState.SPEAKING
        yield CascadedEvent(type=CascadedEventType.TTS_STARTED)

        try:
            # Use synthesize for simpler TTS
            async for event in self._tts_client.synthesize(text):
                # Check for cancellation (barge-in)
                if self._cancel_event.is_set():
                    break

                if hasattr(event, "frame") and event.frame:
                    audio_data = event.frame.data
                    if isinstance(audio_data, memoryview):
                        audio_data = bytes(audio_data)
                    yield CascadedEvent(
                        type=CascadedEventType.TTS_AUDIO,
                        data={"audio": audio_data},
                    )

            yield CascadedEvent(type=CascadedEventType.TTS_COMPLETED)

        except Exception as e:
            logger.error(f"TTS error: {e}")
            yield CascadedEvent(
                type=CascadedEventType.ERROR,
                data={"error": str(e), "stage": "tts"},
            )
        finally:
            self._state = ProviderState.LISTENING

    # =========================================================================
    # Tool Handling
    # =========================================================================

    async def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Send a tool result and optionally get the continuation response.

        When multiple tool results are pending (parallel tool calls), only
        the last one should set request_response=True. Results are batched
        into the context first, and the LLM is only re-invoked once.

        Args:
            call_id: The tool call ID.
            result: The tool result.
            request_response: If True, continue LLM generation after this result.

        Yields:
            LLM and TTS events for the continuation (only if request_response).
        """
        self._context.add_tool_result(call_id, result)

        if request_response:
            async for event in self._continue_after_tool():
                yield event

    async def _continue_after_tool(self) -> AsyncGenerator[CascadedEvent, None]:
        """Continue LLM generation after tool result."""
        if self._llm_client is None:
            return

        self._state = ProviderState.PROCESSING
        yield CascadedEvent(type=CascadedEventType.LLM_STARTED)

        try:
            chat_ctx = self._build_chat_context()

            tools = self._format_tools()
            response_text = ""
            tool_calls: List[ToolCall] = []

            async with self._llm_client.chat(
                chat_ctx=chat_ctx,
                tools=tools if tools else None,
            ) as stream:
                async for chunk in stream:
                    if self._cancel_event.is_set():
                        break

                    if chunk.delta:
                        if chunk.delta.content:
                            token = chunk.delta.content
                            response_text += token
                            yield CascadedEvent(
                                type=CascadedEventType.LLM_TOKEN,
                                data={"token": token, "accumulated": response_text},
                            )

                        if chunk.delta.tool_calls:
                            for tc in chunk.delta.tool_calls:
                                tool_calls.append(self._parse_tool_call(tc))

            yield CascadedEvent(
                type=CascadedEventType.LLM_COMPLETED,
                data={"text": response_text, "tool_calls": tool_calls},
            )

            if response_text:
                self._context.add_assistant(response_text)

            if tool_calls:
                for tc in tool_calls:
                    self._context.add_tool_call(tc)
                    yield CascadedEvent(
                        type=CascadedEventType.TOOL_CALL,
                        data={"tool_call": tc},
                    )
            elif response_text:
                async for tts_event in self._process_tts(response_text):
                    yield tts_event

        except Exception as e:
            logger.error(f"LLM continuation error: {e}")
            yield CascadedEvent(
                type=CascadedEventType.ERROR,
                data={"error": str(e), "stage": "llm_continuation"},
            )
        finally:
            self._state = ProviderState.LISTENING

    @staticmethod
    def _parse_tool_call(tc: Any) -> ToolCall:
        """Parse a LiveKit FunctionToolCall into our ToolCall model.

        LiveKit's FunctionToolCall has `arguments` as a JSON string,
        but our ToolCall expects a dict.
        """
        args = tc.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return ToolCall(
            id=tc.call_id or str(uuid.uuid4()),
            name=tc.name,
            arguments=args,
        )

    def _build_chat_context(self) -> Any:
        """Build LiveKit ChatContext from internal conversation context.

        Properly handles:
        - System, user, and assistant messages
        - Tool calls (as FunctionCall items)
        - Tool results (as FunctionCallOutput items)

        Returns:
            LiveKit ChatContext with full conversation history.
        """
        from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput

        chat_ctx = ChatContext()

        for msg in self._context.messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                chat_ctx.add_message(role="system", content=content)

            elif role == "user":
                chat_ctx.add_message(role="user", content=content)

            elif role == "assistant":
                # Add assistant message if there's content
                if content:
                    chat_ctx.add_message(role="assistant", content=content)

                # Add tool calls as FunctionCall items
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    # tc is a dict: {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    func_call = FunctionCall(
                        call_id=tc["id"],
                        name=func.get("name", ""),
                        arguments=args if isinstance(args, str) else json.dumps(args),
                    )
                    chat_ctx.insert(func_call)

            elif role == "tool":
                # Add tool result as FunctionCallOutput
                func_output = FunctionCallOutput(
                    call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", ""),
                    output=content,
                    is_error=False,
                )
                chat_ctx.insert(func_output)

        return chat_ctx

    def _format_tools(self) -> List[Any]:
        """Format tools for the LLM.

        Returns:
            List of LiveKit tool objects.
        """
        if not self._tools:
            return []

        try:
            from livekit.agents.llm.tool_context import (
                RawFunctionTool,
                RawFunctionToolInfo,
                ToolFlag,
            )

            formatted: List[Any] = []
            for tool in self._tools:
                # Get OpenAI schema from tool
                # Tool.openai_schema returns {"type": "function", "function": {...}}
                # But LiveKit's to_fnc_ctx wraps raw_schema with {"type": "function", "function": raw_schema}
                # So we need to pass only the inner "function" part
                try:
                    full_schema = tool.openai_schema
                    # Extract just the function part
                    raw_schema = full_schema.get("function", full_schema)
                except Exception:
                    # Fallback: construct schema from known attributes
                    raw_schema = {
                        "name": tool.name,
                        "description": getattr(tool, "short_desc", "") or "",
                        "parameters": {"type": "object", "properties": {}},
                    }

                # Create tool info
                info = RawFunctionToolInfo(
                    name=tool.name,
                    raw_schema=raw_schema,
                    flags=ToolFlag.NONE,
                )

                # Create a placeholder callable (execution happens externally)
                async def placeholder_fn(**kwargs):
                    return None

                raw_tool = RawFunctionTool(placeholder_fn, info)
                formatted.append(raw_tool)

            return formatted
        except Exception as e:
            logger.warning(f"Failed to format tools: {e}")
            return []

    # =========================================================================
    # Interruption Handling
    # =========================================================================

    async def interrupt(self) -> CascadedEvent:
        """Interrupt current generation/playback.

        Call this when barge-in is detected (user starts speaking
        while agent is speaking).

        Returns:
            Interrupted event with details.
        """
        old_state = self._state
        self._state = ProviderState.LISTENING

        # Signal cancellation to any running LLM/TTS tasks
        self._cancel_event.set()

        # Clear the event for future operations
        await asyncio.sleep(0.01)
        self._cancel_event.clear()

        return CascadedEvent(
            type=CascadedEventType.INTERRUPTED,
            data={"previous_state": old_state.value},
        )

    # =========================================================================
    # Greeting Generation
    # =========================================================================

    async def generate_greeting(
        self,
        text: str = "Hi! How can I help you today?",
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Generate TTS audio for an initial greeting (bypasses LLM).

        Use this to generate the agent's opening message with audio.

        Args:
            text: Greeting text to synthesize.

        Yields:
            TTS audio events.
        """
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call connect() first.")

        # Add greeting to context as assistant message
        self._context.add_assistant(text)

        # Emit LLM completed event (for tracking)
        yield CascadedEvent(
            type=CascadedEventType.LLM_COMPLETED,
            data={"text": text, "tool_calls": []},
        )

        # Generate TTS audio for the greeting
        async for event in self._process_tts(text):
            yield event

    # =========================================================================
    # Direct Text Input (for testing)
    # =========================================================================

    async def process_text(
        self,
        text: str,
    ) -> AsyncGenerator[CascadedEvent, None]:
        """Process text directly through LLM and TTS (bypasses STT).

        Useful for testing without audio.

        Args:
            text: User text input.

        Yields:
            LLM and TTS events.
        """
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call connect() first.")

        # Yield a synthetic transcript event
        yield CascadedEvent(
            type=CascadedEventType.TRANSCRIPT_FINAL,
            data={"transcript": text},
        )

        # Process through LLM and TTS
        async for event in self._process_llm(text):
            yield event
