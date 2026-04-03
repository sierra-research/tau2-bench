"""Abstract base classes and factory for audio native adapters.
DiscreteTimeAdapter: Tick-based pattern for discrete-time simulation.
   - run_tick() as the primary method
   - Audio time is the primary clock
   - Used by DiscreteTimeAudioNativeAgent

create_adapter(): Factory function that validates parameters and constructs
   the appropriate adapter subclass for a given provider.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
    DEFAULT_BUFFER_UNTIL_COMPLETE,
    DEFAULT_FAST_FORWARD_MODE,
    DEFAULT_SEND_AUDIO_INSTANT,
    TELEPHONY_ULAW_SILENCE,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.tick_result import (
    TickResult,
    UtteranceTranscript,
    buffer_excess_audio,
    get_proportional_transcript,
)


class DiscreteTimeAdapter(ABC):
    """Abstract base class for discrete-time audio native adapters.

    This adapter pattern is designed for discrete-time simulation where:
    - Audio time is the primary clock (not wall-clock time)
    - Interaction happens in fixed-duration "ticks"
    - Each tick: send user audio, receive agent audio (capped to tick duration)

    The primary method is run_tick(), which handles one tick of the simulation.

    Attributes:
        tick_duration_ms: Duration of each tick in milliseconds.
        audio_format: Audio format for the external interface (user audio in,
            agent audio out). Defaults to telephony (8kHz μ-law).
        bytes_per_tick: Maximum agent audio bytes per tick, derived from
            tick_duration_ms and audio_format.

    Usage:
        adapter = SomeAdapter(tick_duration_ms=200)
        adapter.connect(system_prompt, tools, vad_config, modality="audio")

        for tick in range(max_ticks):
            result = adapter.run_tick(user_audio_bytes, tick_number=tick)
            # result.get_played_agent_audio() - padded to exactly bytes_per_tick
            # result.agent_audio_data - raw audio (for speech detection)
            # result.proportional_transcript - text for this tick
            # result.events - all API events received

        adapter.disconnect()
    """

    def __init__(
        self,
        tick_duration_ms: int,
        audio_format: Optional[AudioFormat] = None,
        send_audio_instant: bool = DEFAULT_SEND_AUDIO_INSTANT,
    ):
        """Initialize the adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            audio_format: Audio format for the external interface. Defaults to
                telephony (8kHz μ-law). Subclasses may pass a different format
                if their provider uses a non-telephony external format.
            send_audio_instant: If True, send all audio in one call per tick.
                If False, send in VoIP-style 20ms chunks with sleeps.

        Raises:
            ValueError: If tick_duration_ms is <= 0.
        """
        if tick_duration_ms <= 0:
            raise ValueError(f"tick_duration_ms must be > 0, got {tick_duration_ms}")

        self.tick_duration_ms = tick_duration_ms
        self.audio_format = audio_format or TELEPHONY_AUDIO_FORMAT
        self.bytes_per_tick = int(
            self.audio_format.bytes_per_second * tick_duration_ms / 1000
        )
        self.send_audio_instant = send_audio_instant
        self._voip_interval_ms = DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS

        # Shared tick state (managed by _async_run_tick template)
        self._tick_count: int = 0
        self._cumulative_user_audio_ms: int = 0
        self._buffered_agent_audio: List[Tuple[bytes, Optional[str]]] = []
        self._utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self._current_item_id: Optional[str] = None
        self._skip_item_id: Optional[str] = None
        self._pending_tool_results: List[Tuple] = []

        # Optional item-ID mapping (used by Nova where audio and text have
        # different content IDs). Subclasses that need it should populate this
        # dict; get_proportional_transcript will forward it automatically.
        self._item_id_map: Optional[dict[str, str]] = None

    @abstractmethod
    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any,
        modality: str = "audio",
    ) -> None:
        """Connect to the API and configure the session.

        Args:
            system_prompt: System prompt for the agent.
            tools: List of tools the agent can use.
            vad_config: VAD configuration (e.g., OpenAIVADConfig).
            modality: "audio" for full audio, "audio_in_text_out" for audio input only.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the API and clean up resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the adapter is connected to the API."""
        raise NotImplementedError

    @abstractmethod
    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        """Run one tick of the simulation.

        Subclasses implement this to bridge sync/async (e.g., via BackgroundAsyncLoop).
        The async work should delegate to _async_run_tick() which provides the
        shared tick lifecycle (buffering, transcript, etc.).

        Args:
            user_audio: User audio bytes for this tick (in audio_format encoding).
            tick_number: Optional tick number for logging/tracking.

        Returns:
            TickResult with capped audio, proportional transcript, events.
        """
        raise NotImplementedError

    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Queue a tool result to be sent in the next tick."""
        self._pending_tool_results.append((call_id, result, request_response, is_error))
        logger.debug(f"Queued tool result for call_id={call_id}")

    def clear_buffers(self) -> None:
        """Reset all internal tick state."""
        self._buffered_agent_audio.clear()
        self._utterance_transcripts.clear()
        self._pending_tool_results.clear()
        self._skip_item_id = None

    # -----------------------------------------------------------------------
    # Tick lifecycle template
    # -----------------------------------------------------------------------

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        """Template method for the tick lifecycle.

        Handles shared pre/post processing around _execute_tick():
        - Flush pending tool results
        - Create TickResult with timing info
        - Prepend buffered audio from previous tick
        - Call _execute_tick() for provider-specific work
        - Cap audio to bytes_per_tick, buffer excess
        - Compute proportional transcript
        - Update cumulative state

        Subclasses implement _execute_tick() and _flush_pending_tool_results().
        """
        tick_start = asyncio.get_running_loop().time()

        # 1. Flush pending tool results
        await self._flush_pending_tool_results()

        # 2. Create tick result
        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=len(user_audio),
            audio_sent_duration_ms=(
                len(user_audio) / self.audio_format.bytes_per_second
            )
            * 1000,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self._cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
            silence_byte=TELEPHONY_ULAW_SILENCE,
        )

        # 3. Prepend buffered audio from previous tick
        for chunk_data, item_id in self._buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, item_id))
        self._buffered_agent_audio.clear()

        # 4. Carry over skip state
        result.skip_item_id = self._skip_item_id

        # 5. Provider-specific: send audio, receive events, process events
        await self._execute_tick(user_audio, tick_number, result, tick_start)

        # 6. Record simulation timing
        result.tick_sim_duration_ms = result.audio_sent_duration_ms

        # 7. Cap audio to bytes_per_tick, buffer excess for next tick
        self._buffered_agent_audio = buffer_excess_audio(result, self.bytes_per_tick)

        # 8. Compute proportional transcript
        result.proportional_transcript = get_proportional_transcript(
            result.agent_audio_chunks,
            self._utterance_transcripts,
            item_id_map=self._item_id_map,
        )

        # 9. Update skip state for next tick
        self._skip_item_id = result.skip_item_id

        # 10. Update cumulative user audio tracking
        self._cumulative_user_audio_ms += int(result.audio_sent_duration_ms)

        # 11. Enforce minimum tick duration (safety net)
        elapsed = asyncio.get_running_loop().time() - tick_start
        remaining = (self.tick_duration_ms / 1000) - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

        logger.info(f"Tick {tick_number} completed:\n{result.summary()}")
        return result

    @abstractmethod
    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Provider-specific tick execution. Mutate result in place.

        Args:
            user_audio: User audio bytes (in external format).
            tick_number: Current tick number.
            result: TickResult to populate with events, audio, etc.
            tick_start: asyncio loop time when the tick started. Use this
                to compute remaining time for receive_events_for_duration().

        Responsibilities:
        - Convert audio format if needed
        - Send audio to provider API
        - Receive events for the remaining tick duration
        - Process events (append to result.agent_audio_chunks,
          result.tool_calls, result.vad_events, result.events)
        - Track utterance transcripts via self._utterance_transcripts
        - Set result.was_truncated and result.skip_item_id on barge-in
        """
        raise NotImplementedError

    @abstractmethod
    async def _flush_pending_tool_results(self) -> None:
        """Send all pending tool results to the provider.

        Called at the start of each tick. Provider-specific because each
        API has different batching semantics (e.g., Gemini batches all
        results in one call, others send individually).

        After sending, clear self._pending_tool_results.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    async def _send_audio_chunked(
        self,
        audio: bytes,
        send_fn: Callable[[bytes], Awaitable[None]],
        chunk_size: int,
    ) -> None:
        """Send audio either instantly or in VoIP-style chunks.

        Args:
            audio: Audio bytes to send (in provider's format).
            send_fn: Async callable that sends a chunk to the provider.
            chunk_size: Bytes per chunk when using VoIP-style sending.
        """
        if len(audio) == 0:
            return
        if self.send_audio_instant:
            await send_fn(audio)
        else:
            offset = 0
            while offset < len(audio):
                chunk = audio[offset : offset + chunk_size]
                await send_fn(chunk)
                offset += len(chunk)
                await asyncio.sleep(self._voip_interval_ms / 1000)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

# Providers where the model is determined by the endpoint, not a parameter
_PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL = ("xai",)


def create_adapter(
    provider: str,
    tick_duration_ms: int,
    send_audio_instant: bool = DEFAULT_SEND_AUDIO_INSTANT,
    buffer_until_complete: bool = DEFAULT_BUFFER_UNTIL_COMPLETE,
    fast_forward_mode: bool = DEFAULT_FAST_FORWARD_MODE,
    model: Optional[str] = None,
    audio_format: Optional[AudioFormat] = None,
    cascaded_config: Any = None,
) -> Tuple[DiscreteTimeAdapter, str]:
    """Create a discrete-time adapter for the given provider.

    Validates parameter/provider compatibility, resolves the model default,
    constructs the appropriate adapter subclass, and returns both the adapter
    and the resolved model name.

    Args:
        provider: Provider identifier (openai, gemini, xai, nova, qwen,
            livekit).
        tick_duration_ms: Duration of each tick in milliseconds.
        send_audio_instant: If True, send audio in one call per tick.
        buffer_until_complete: If True, wait for complete utterances before
            releasing audio. Only supported by the OpenAI provider.
        fast_forward_mode: If True, exit tick early when enough audio is
            buffered. Only supported by the OpenAI provider.
        model: Model identifier. If None, uses the provider's default.
        audio_format: Audio format for external communication. Defaults to
            telephony (8kHz μ-law).
        cascaded_config: Configuration for cascaded providers (livekit).

    Returns:
        Tuple of (adapter, resolved_model).

    Raises:
        ValueError: If buffer_until_complete or fast_forward_mode is used
            with a non-OpenAI provider, or if the provider is unknown.
    """
    # --- Validate OpenAI-only parameters ---
    if buffer_until_complete and provider != "openai":
        raise ValueError(
            f"buffer_until_complete is only supported by the 'openai' provider, "
            f"got provider='{provider}'."
        )
    if fast_forward_mode:
        if provider != "openai":
            raise ValueError(
                f"fast_forward_mode is only supported by the 'openai' provider, "
                f"got provider='{provider}'."
            )
        logger.warning(
            "Fast-forward mode is enabled. The simulation will run as fast as "
            "possible rather than in real-time. This may affect timing-sensitive "
            "behaviors and produce results that differ from real-time execution."
        )

    # --- Resolve model default ---
    if model is None:
        if provider == "livekit":
            from tau2.voice.audio_native.livekit.config import CascadedConfig

            config = cascaded_config or CascadedConfig()
            model = config.llm.model
        else:
            model = DEFAULT_AUDIO_NATIVE_MODELS[provider]
        logger.debug(
            f"No model provided, using default model for provider {provider}: {model}"
        )
    elif provider in _PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL:
        logger.warning(
            f"model='{model}' was provided but the '{provider}' provider's model "
            f"is determined by its endpoint — the provided model will be ignored."
        )

    # --- Construct adapter ---
    adapter: DiscreteTimeAdapter
    if provider == "openai":
        from tau2.voice.audio_native.openai.discrete_time_adapter import (
            DiscreteTimeAudioNativeAdapter,
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            buffer_until_complete=buffer_until_complete,
            model=model,
            audio_format=audio_format,
            fast_forward_mode=fast_forward_mode,
        )
    elif provider == "gemini":
        from tau2.voice.audio_native.gemini.discrete_time_adapter import (
            DiscreteTimeGeminiAdapter,
        )

        adapter = DiscreteTimeGeminiAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            model=model,
        )
    elif provider == "xai":
        from tau2.voice.audio_native.xai.discrete_time_adapter import (
            DiscreteTimeXAIAdapter,
        )

        adapter = DiscreteTimeXAIAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
        )
    elif provider == "nova":
        from tau2.voice.audio_native.nova.discrete_time_adapter import (
            DiscreteTimeNovaAdapter,
        )

        adapter = DiscreteTimeNovaAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            model=model,
        )
    elif provider == "qwen":
        from tau2.voice.audio_native.qwen.discrete_time_adapter import (
            DiscreteTimeQwenAdapter,
        )

        adapter = DiscreteTimeQwenAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            model=model,
        )
    elif provider == "livekit":
        from tau2.voice.audio_native.livekit.config import CascadedConfig
        from tau2.voice.audio_native.livekit.discrete_time_adapter import (
            LiveKitCascadedAdapter,
        )

        config = cascaded_config or CascadedConfig()
        adapter = LiveKitCascadedAdapter(
            tick_duration_ms=tick_duration_ms,
            cascaded_config=config,
            send_audio_instant=send_audio_instant,
            audio_format=audio_format,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return adapter, model
