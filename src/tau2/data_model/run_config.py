"""Shared typed run settings, independent of task selection and saved provenance."""

from copy import deepcopy
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from typing_extensions import Annotated

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_BACKCHANNEL_MAX_THRESHOLD_SECONDS,
    DEFAULT_BACKCHANNEL_MIN_THRESHOLD_SECONDS,
    DEFAULT_BACKCHANNEL_POISSON_RATE,
    DEFAULT_CHANNEL_EFFECTS_MODE,
    DEFAULT_INTEGRATION_DURATION_SECONDS,
    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_EVAL_USER_SIMULATOR,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_STEPS_SECONDS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_PCM_SAMPLE_RATE,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_SEED,
    DEFAULT_SEND_AUDIO_INSTANT,
    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
    DEFAULT_SPEECH_EFFECTS_MODE,
    DEFAULT_TELEPHONY_RATE,
    DEFAULT_TICK_DURATION_SECONDS,
    DEFAULT_USE_LLM_BACKCHANNEL,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
)
from tau2.data_model.cascaded import CascadedConfig, CascadedConfigRequest
from tau2.data_model.elicitation import (
    ElicitationConfig,
)
from tau2.data_model.live_config import LiveConfig
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.voice import SpeechComplexity, VoiceSettings
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig


class AudioNativeConfig(BaseModel):
    """Configuration for audio-native mode using DiscreteTimeAudioNativeAgent.

    This configuration is used when running full-duplex voice simulations
    with audio native APIs (OpenAI Realtime or Gemini Live).
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def migrate_cascaded_preset(cls, value):
        """Read main's saved preset field without inventing effective settings."""
        if not isinstance(value, dict) or "cascaded_config_name" not in value:
            return value
        value = dict(value)
        preset = value.pop("cascaded_config_name")
        if preset is not None:
            request = CascadedConfigRequest(preset=preset)
            supplied = value.get("cascaded_request")
            if (
                supplied is not None
                and CascadedConfigRequest.model_validate(supplied) != request
            ):
                raise ValueError(
                    "Legacy cascaded preset conflicts with cascaded_request"
                )
            value["cascaded_request"] = request
        return value

    # Provider selection
    provider: Literal[
        "openai",
        "openai_live",
        "gemini",
        "xai",
        "nova",
        "qwen",
        "avartha_pipecat",
        "phonic",
        "livekit",
        "launa",
    ] = Field(
        default=DEFAULT_AUDIO_NATIVE_PROVIDER,
        description="Audio native API provider: 'openai' (OpenAI Realtime), 'openai_live' (OpenAI Live), 'gemini' (Gemini Live), 'xai' (xAI Grok Voice Agent), 'nova' (Amazon Nova Sonic), 'qwen' (Alibaba Qwen Omni), 'avartha_pipecat' (hosted Avartha agent over Pipecat SmallWebRTC), 'phonic' (Phonic 1), 'launa' (Salesforce Launa), or 'livekit' (LiveKit cascaded STT→LLM→TTS)",
    )

    transcription_language: Optional[str] = Field(
        default=None,
        description="OpenAI Realtime input transcription language; runs resolve their base language code.",
    )

    language: Optional[str] = Field(
        default=None,
        description="Resolved run call language supplied to provider language settings.",
    )

    voice: Optional[str] = Field(
        default=None,
        description="Explicit OpenAI Realtime voice; Live uses live_config.voice.",
    )

    # Cascaded config (for livekit provider)
    cascaded_request: Optional[CascadedConfigRequest] = Field(
        default=None,
        description="Caller-requested LiveKit preset or complete overrides; never replaced by generated settings.",
    )

    cascaded_config: Optional[CascadedConfig] = Field(
        default=None,
        description="Effective LiveKit STT, LLM and TTS settings, resolved before construction and recorded on results.",
    )

    avartha_config: Optional[AvarthaPipecatConfig] = Field(
        default=None,
        description="Hosted agent connection settings for the avartha_pipecat provider",
    )

    model: str = Field(
        default=DEFAULT_AUDIO_NATIVE_MODELS[DEFAULT_AUDIO_NATIVE_PROVIDER],
        description=(
            "Audio native model to use (a recorded label for Avartha, whose "
            "agent URL selects the agent)"
        ),
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort for thinking models: 'minimal', 'low', 'medium', 'high'. If None, not sent.",
    )
    live_config: Optional[LiveConfig] = Field(
        default=None,
        description="Backend model, voice, and optional prompt overrides for the openai_live provider",
    )
    realtime_generation: Optional[bool] = Field(
        default=None,
        description="Whether user LLM and TTS generation run without blocking audio ticks. "
        "Defaults to enabled for openai_live and avartha_pipecat and disabled for "
        "other providers.",
    )

    @property
    def realtime_generation_enabled(self) -> bool:
        """Resolve the provider-aware default for real-time user generation."""
        if self.realtime_generation is not None:
            return self.realtime_generation
        return self.provider in ("openai_live", "avartha_pipecat")

    @model_validator(mode="after")
    def validate_live_config(self):
        if (self.provider == "openai_live") != (self.live_config is not None):
            raise ValueError(
                "live_config is required for openai_live and only valid for that provider"
            )
        return self

    @model_validator(mode="after")
    def validate_avartha_config(self):
        """Require the hosted agent's settings for the Avartha provider."""
        if self.provider == "avartha_pipecat":
            if self.avartha_config is None:
                raise ValueError(
                    "avartha_pipecat needs avartha_config with the hosted agent's URL"
                )
            default_model = DEFAULT_AUDIO_NATIVE_MODELS[DEFAULT_AUDIO_NATIVE_PROVIDER]
            if self.model == default_model:
                self.model = DEFAULT_AUDIO_NATIVE_MODELS["avartha_pipecat"]
        elif self.avartha_config is not None:
            raise ValueError(
                "avartha_config is only valid when provider='avartha_pipecat'"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_avartha_config(self, handler):
        """Other providers keep their existing serialized contract."""
        value = handler(self)
        if self.avartha_config is None:
            value.pop("avartha_config", None)
        return value

    # Timing configuration
    tick_duration_seconds: float = Field(
        default=DEFAULT_TICK_DURATION_SECONDS,
        description="Duration of each tick in seconds (e.g., 0.2 = 200ms)",
    )
    max_steps_seconds: int = Field(
        default=DEFAULT_MAX_STEPS_SECONDS,
        description="Maximum conversation duration in seconds",
    )

    # Audio configuration
    pcm_sample_rate: int = Field(
        default=DEFAULT_PCM_SAMPLE_RATE,
        description="User simulator PCM synthesis sample rate",
    )
    telephony_rate: int = Field(
        default=DEFAULT_TELEPHONY_RATE,
        description="API/agent telephony sample rate (OpenAI Realtime API)",
    )

    # User simulator turn-taking thresholds (in seconds)
    wait_to_respond_threshold_other_seconds: float = Field(
        default=DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
        description="Min time to wait since OTHER (agent) last spoke before responding",
    )
    wait_to_respond_threshold_self_seconds: float = Field(
        default=DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
        description="Min time to wait since SELF (user) last spoke before responding",
    )
    yield_threshold_when_interrupted_seconds: float = Field(
        default=DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
        description="How long user keeps speaking when agent interrupts user",
    )
    yield_threshold_when_interrupting_seconds: float = Field(
        default=DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
        description="How long user keeps speaking when user interrupts agent",
    )
    interruption_check_interval_seconds: float = Field(
        default=DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
        description="Interval for checking interruptions",
    )
    integration_duration_seconds: float = Field(
        default=DEFAULT_INTEGRATION_DURATION_SECONDS,
        description="Integration duration for linearization",
    )
    silence_annotation_threshold_seconds: float = Field(
        default=DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
        description="Silence threshold for adding annotations to conversation history",
    )
    backchannel_min_threshold_seconds: Optional[float] = Field(
        default=DEFAULT_BACKCHANNEL_MIN_THRESHOLD_SECONDS,
        description="Backchannel min threshold in seconds (None = disabled). Used with Poisson policy.",
    )
    backchannel_max_threshold_seconds: Optional[float] = Field(
        default=DEFAULT_BACKCHANNEL_MAX_THRESHOLD_SECONDS,
        description="Backchannel max threshold in seconds - force backchannel after this duration. Used with Poisson policy.",
    )
    backchannel_poisson_rate: float = Field(
        default=DEFAULT_BACKCHANNEL_POISSON_RATE,
        description="Backchannel Poisson rate (events per second). Used with Poisson policy.",
    )
    use_llm_backchannel: bool = Field(
        default=DEFAULT_USE_LLM_BACKCHANNEL,
        description="If True, use LLM-based backchannel policy. If False, use Poisson-based policy with min/max thresholds.",
    )

    # Agent behavior
    use_xml_prompt: bool = Field(
        default=False,
        description="Use XML tags in system prompt. Defaults to False (plain text) for all providers.",
    )
    send_audio_instant: bool = Field(
        default=DEFAULT_SEND_AUDIO_INSTANT,
        description="If True, send all audio at once per tick. If False (default), stream audio in 20ms chunks at real-time rate.",
    )

    # Derived properties (computed from seconds and tick_duration)
    @property
    def tick_duration_ms(self) -> float:
        """Tick duration in milliseconds."""
        return self.tick_duration_seconds * 1000

    @property
    def user_chunk_size(self) -> int:
        """User audio chunk size in samples."""
        return int(self.pcm_sample_rate * self.tick_duration_seconds)

    @property
    def wait_to_respond_threshold_other_ticks(self) -> int:
        """Wait to respond threshold (other) in ticks."""
        return int(
            self.wait_to_respond_threshold_other_seconds / self.tick_duration_seconds
        )

    @property
    def wait_to_respond_threshold_self_ticks(self) -> int:
        """Wait to respond threshold (self) in ticks."""
        return int(
            self.wait_to_respond_threshold_self_seconds / self.tick_duration_seconds
        )

    @property
    def yield_threshold_when_interrupted_ticks(self) -> int:
        """Yield threshold when interrupted (agent interrupts user) in ticks."""
        return int(
            self.yield_threshold_when_interrupted_seconds / self.tick_duration_seconds
        )

    @property
    def yield_threshold_when_interrupting_ticks(self) -> int:
        """Yield threshold when interrupting (user interrupts agent) in ticks."""
        return int(
            self.yield_threshold_when_interrupting_seconds / self.tick_duration_seconds
        )

    @property
    def interruption_check_interval_ticks(self) -> int:
        """Interruption check interval in ticks."""
        return int(
            self.interruption_check_interval_seconds / self.tick_duration_seconds
        )

    @property
    def integration_ticks(self) -> int:
        """Integration ticks for linearization."""
        return max(
            1, int(self.integration_duration_seconds / self.tick_duration_seconds)
        )

    @property
    def silence_annotation_threshold_ticks(self) -> int:
        """Silence annotation threshold in ticks."""
        return int(
            self.silence_annotation_threshold_seconds / self.tick_duration_seconds
        )

    @property
    def backchannel_min_threshold_ticks(self) -> Optional[int]:
        """Backchannel min threshold in ticks (None if disabled)."""
        if self.backchannel_min_threshold_seconds is None:
            return None
        return int(self.backchannel_min_threshold_seconds / self.tick_duration_seconds)

    @property
    def backchannel_max_threshold_ticks(self) -> Optional[int]:
        """Backchannel max threshold in ticks (None if not set)."""
        if self.backchannel_max_threshold_seconds is None:
            return None
        return int(self.backchannel_max_threshold_seconds / self.tick_duration_seconds)

    @property
    def max_steps_ticks(self) -> int:
        """Maximum steps in ticks."""
        return int(self.max_steps_seconds / self.tick_duration_seconds)


class RunSettings(BaseModel):
    """Ordinary execution settings shared by text, voice and recorded conditions."""

    elicitation: Annotated[
        Optional[ElicitationConfig],
        Field(
            description="Caller stimulus settings for elicitation domains only.",
            default=None,
        ),
    ]

    communicate_scoring: CommunicateScoring = Field(
        default_factory=CommunicateScoring,
        description="Communicate-info scorer: the pinned semantic LLM judge by default; "
        "substring matching only when explicitly selected.",
    )

    llm_user: Annotated[
        str,
        Field(
            description="The model to use for the user simulator",
            default=DEFAULT_LLM_USER,
        ),
    ]

    llm_args_user: Annotated[
        dict,
        Field(
            description="The arguments to pass to the LLM for the user simulator",
            default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_USER),
        ),
    ]

    num_trials: Annotated[
        int,
        Field(
            description="The number of trials to run the simulation on",
            default=DEFAULT_NUM_TRIALS,
        ),
    ]

    max_errors: Annotated[
        int,
        Field(
            description="The maximum number of tool errors allowed in a row in the simulation",
            default=DEFAULT_MAX_ERRORS,
        ),
    ]

    timeout: Annotated[
        Optional[float],
        Field(
            description="Maximum wallclock time in seconds for a single simulation. None means no timeout.",
            default=None,
        ),
    ]

    max_concurrency: Annotated[
        int,
        Field(
            description="The maximum number of concurrent simulations to run",
            default=DEFAULT_MAX_CONCURRENCY,
        ),
    ]

    workers: Annotated[
        int,
        Field(
            description="Number of worker processes to spawn. 0 (default) runs "
            "simulations in this process; N > 0 makes this process a controller "
            "that schedules and checkpoints while N workers execute, each "
            "holding up to max_concurrency simulations in flight.",
            default=0,
        ),
    ]

    provider_limits: Annotated[
        Optional[dict[str, int]],
        Field(
            description="Max concurrently-running simulations per provider, "
            "enforced at lease time in controller mode (workers > 0), "
            'e.g. {"openai": 40, "gemini": 20}.',
            default=None,
        ),
    ]

    seed: Annotated[
        Optional[int],
        Field(
            description="The seed to use for the simulation",
            default=DEFAULT_SEED,
        ),
    ]

    log_level: Annotated[
        Optional[str],
        Field(
            description="The log level to use for the simulation",
            default=DEFAULT_LOG_LEVEL,
        ),
    ]

    verbose_logs: Annotated[
        bool,
        Field(
            description="Enable verbose logging: saves LLM call logs, audio files, per-task logs, and ticks (for audio-native).",
            default=False,
        ),
    ]

    max_retries: Annotated[
        int,
        Field(
            description="Maximum number of retries for failed tasks.",
            default=DEFAULT_RETRY_ATTEMPTS,
        ),
    ]

    retry_delay: Annotated[
        float,
        Field(
            description="Delay in seconds between retries.",
            default=DEFAULT_RETRY_MIN_WAIT,
        ),
    ]

    auto_resume: Annotated[
        bool,
        Field(
            description="Automatically resume from existing save file without prompting.",
            default=False,
        ),
    ]

    auto_review: Annotated[
        bool,
        Field(
            description="Automatically run LLM conversation review after each simulation.",
            default=False,
        ),
    ]

    review_mode: Annotated[
        Literal["full", "user"],
        Field(
            description="Review mode when auto_review is enabled: 'full' (agent+user errors, default) or 'user' (user simulator only).",
            default="full",
        ),
    ]

    review_model: Annotated[
        str,
        Field(
            description="LLM model for conversation review and hallucination checks.",
            default=DEFAULT_LLM_EVAL_USER_SIMULATOR,
        ),
    ]

    hallucination_retries: Annotated[
        int,
        Field(
            description="Maximum number of retries when a user simulator hallucination is detected. "
            "Set to 0 to disable. "
            "Each retry re-runs the simulation with a different seed and feedback.",
            default=3,
        ),
    ]

    is_remote: Annotated[
        bool,
        Field(
            description="Whether to run the simulation remotely",
            default=False,
        ),
    ]

    retrieval_config: Annotated[
        Optional[str],
        Field(
            description="Knowledge retrieval config name (knowledge domain only).",
            default=None,
        ),
    ]

    retrieval_config_kwargs: Annotated[
        Optional[dict],
        Field(
            description="Arguments to pass to the retrieval config constructor (e.g., top_k for RAG configs).",
            default=None,
        ),
    ]


class TextRunSettings(RunSettings):
    """Text (half-duplex) execution settings without benchmark identity or task selection."""

    # ---- Agent ----
    agent: Annotated[
        str,
        Field(
            description="The agent implementation to use (e.g., 'llm_agent', 'llm_agent_gt', 'llm_agent_solo')",
            default="llm_agent",
        ),
    ]
    llm_agent: Annotated[
        str,
        Field(
            description="The model to use for the agent",
            default=DEFAULT_LLM_AGENT,
        ),
    ]
    llm_args_agent: Annotated[
        dict,
        Field(
            description="The arguments to pass to the LLM for the agent",
            default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_AGENT),
        ),
    ]

    # ---- User ----
    user: Annotated[
        str,
        Field(
            description="The user implementation to use (e.g., 'user_simulator', 'dummy_user')",
            default="user_simulator",
        ),
    ]

    # ---- Text-specific ----
    max_steps: Annotated[
        int,
        Field(
            description="The maximum number of conversation turns",
            default=DEFAULT_MAX_STEPS,
        ),
    ]
    enforce_communication_protocol: Annotated[
        bool,
        Field(
            description="Whether to enforce communication protocol rules (e.g., no mixed messages with text and tool calls)",
            default=False,
        ),
    ]
    text_streaming_config: Annotated[
        Optional[dict],
        Field(
            description="Text streaming configuration",
            default=None,
        ),
    ]


class VoiceRunSettings(RunSettings):
    """Voice execution settings without benchmark identity or task selection."""

    audio_native_config: Annotated[
        AudioNativeConfig,
        Field(
            description="Configuration for audio-native mode (provider, model, timing, thresholds, etc.).",
        ),
    ]

    speech_complexity: Annotated[
        SpeechComplexity,
        Field(
            description="Speech environment complexity level: 'control' (clean speech, no effects), 'regular' (realistic with background noise and effects), plus ablation variants",
            default="regular",
        ),
    ]

    channel_effects_mode: Annotated[
        Literal["light", "regular", "heavy"],
        Field(
            default=DEFAULT_CHANNEL_EFFECTS_MODE,
            description="Caller channel condition overlay.",
        ),
    ]
    speech_effects_mode: Annotated[
        Literal["light", "regular", "heavy"],
        Field(
            default=DEFAULT_SPEECH_EFFECTS_MODE,
            description="Caller speech condition overlay.",
        ),
    ]
    pass3_condition: Literal["regular", "channel_heavy", "speech_heavy"] | None = None
    voice_sampling_seed: int | None = Field(
        default=None,
        description="Optional seed for caller voice sampling, independent of the run seed.",
    )

    agent_voice_settings: Annotated[
        Optional[VoiceSettings],
        Field(
            description="Voice synthesis and transcription settings for the agent",
            default=None,
        ),
    ]

    user_voice_settings: Annotated[
        Optional[VoiceSettings],
        Field(
            description="Voice synthesis and transcription settings for the user",
            default=None,
        ),
    ]

    audio_debug: Annotated[
        bool,
        Field(
            description="Enable audio debugging: saves per-tick audio files and analysis report.",
            default=False,
        ),
    ]

    audio_taps: Annotated[
        bool,
        Field(
            description="Enable audio tap recording at each pipeline stage for signal analysis.",
            default=False,
        ),
    ]
