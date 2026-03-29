"""Abstract base classes and factory for audio native adapters.
DiscreteTimeAdapter: Tick-based pattern for discrete-time simulation.
   - run_tick() as the primary method
   - Audio time is the primary clock
   - Used by DiscreteTimeAudioNativeAgent

create_adapter(): Factory function that validates parameters and constructs
   the appropriate adapter subclass for a given provider.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_BUFFER_UNTIL_COMPLETE,
    DEFAULT_FAST_FORWARD_MODE,
    DEFAULT_SEND_AUDIO_INSTANT,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool

if TYPE_CHECKING:
    from tau2.voice.audio_native.tick_result import TickResult


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
    ):
        """Initialize the adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            audio_format: Audio format for the external interface. Defaults to
                telephony (8kHz μ-law). Subclasses may pass a different format
                if their provider uses a non-telephony external format.

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
    ) -> "TickResult":
        """Run one tick of the simulation.

        This is the primary method for discrete-time interaction.

        Guarantees imposed by tick_duration_ms:
        - Agent audio output is capped to at most bytes_per_tick bytes per
          tick. Any excess is buffered for the next tick.
        - For streaming providers (bidirectional WebSocket), each tick takes
          at least tick_duration_ms of wall-clock time to maintain real-time
          pacing. Implementations achieve this either via an explicit sleep
          or by collecting events for the full duration. Cascaded providers
          (e.g., STT → LLM → TTS pipelines) may complete ticks faster since
          processing is request/response rather than continuous streaming.

        Args:
            user_audio: User audio bytes for this tick (in audio_format encoding).
            tick_number: Optional tick number for logging/tracking.

        Returns:
            TickResult containing:
            - Raw agent audio (agent_audio_data) for speech detection
            - Padded agent audio (get_played_agent_audio()) for playback
            - Proportional transcript for this tick
            - All API events received during the tick
            - Timing and interruption information
        """
        raise NotImplementedError

    @abstractmethod
    def send_tool_result(
        self,
        call_id: str,
        result: str,
        request_response: bool = True,
        is_error: bool = False,
    ) -> None:
        """Queue a tool result to be sent in the next tick.

        Tool results are typically queued and sent at the start of the next
        run_tick() call to maintain proper timing in discrete-time simulation.

        Args:
            call_id: The tool call ID.
            result: The tool result as a string.
            request_response: If True, request a response after sending.
            is_error: If True, the tool call failed and result contains error details.
        """
        raise NotImplementedError


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
            deepgram, livekit).
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
    elif provider == "deepgram":
        from tau2.voice.audio_native.deepgram.discrete_time_adapter import (
            DiscreteTimeDeepgramAdapter,
        )

        adapter = DiscreteTimeDeepgramAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            llm_model=model,
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
