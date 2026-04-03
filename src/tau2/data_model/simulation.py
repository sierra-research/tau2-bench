import json
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Annotated

if TYPE_CHECKING:
    from tau2.voice.audio_native.livekit.config import CascadedConfig

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION,
    DEFAULT_BACKCHANNEL_MAX_THRESHOLD_SECONDS,
    DEFAULT_BACKCHANNEL_MIN_THRESHOLD_SECONDS,
    DEFAULT_BACKCHANNEL_POISSON_RATE,
    DEFAULT_BUFFER_UNTIL_COMPLETE,
    DEFAULT_FAST_FORWARD_MODE,
    DEFAULT_INTEGRATION_DURATION_SECONDS,
    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
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
    DEFAULT_SAVE_TO,
    DEFAULT_SEED,
    DEFAULT_SEND_AUDIO_INSTANT,
    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
    DEFAULT_TELEPHONY_RATE,
    DEFAULT_TEXT_STREAMING_CONFIG,
    DEFAULT_TICK_DURATION_SECONDS,
    DEFAULT_USE_LLM_BACKCHANNEL,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
)
from tau2.data_model.audio_effects import EffectTimeline
from tau2.data_model.message import Message, Tick
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.tasks import Action, EnvAssertion, RewardType, Task
from tau2.data_model.voice import SpeechComplexity, SpeechEnvironment, VoiceSettings
from tau2.environment.environment import EnvironmentInfo
from tau2.environment.toolkit import ToolType
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.utils import get_now

SIMULATIONS_DIR = "simulations"


class AudioNativeConfig(BaseModel):
    """Configuration for audio-native mode using DiscreteTimeAudioNativeAgent.

    This configuration is used when running full-duplex voice simulations
    with audio native APIs (OpenAI Realtime or Gemini Live).
    """

    # Provider selection
    provider: Literal[
        "openai", "gemini", "xai", "nova", "qwen", "livekit"
    ] = Field(
        default=DEFAULT_AUDIO_NATIVE_PROVIDER,
        description="Audio native API provider: 'openai' (OpenAI Realtime), 'gemini' (Gemini Live), 'xai' (xAI Grok Voice Agent), 'nova' (Amazon Nova Sonic), 'qwen' (Alibaba Qwen Omni), or 'livekit' (LiveKit cascaded STT→LLM→TTS)",
    )

    # Cascaded config (for livekit provider)
    cascaded_config_name: Optional[str] = Field(
        default=None,
        description="Name of cascaded config preset for livekit provider (e.g., 'default', 'openai-thinking', 'openai-thinking-high')",
    )

    model: str = Field(
        default=DEFAULT_AUDIO_NATIVE_MODELS[DEFAULT_AUDIO_NATIVE_PROVIDER],
        description="Audio native model to use",
    )

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
    buffer_until_complete: bool = Field(
        default=DEFAULT_BUFFER_UNTIL_COMPLETE,
        description="Buffer audio until complete utterance (OpenAI only)",
    )
    fast_forward_mode: bool = Field(
        default=DEFAULT_FAST_FORWARD_MODE,
        description="Skip wall-clock waiting when enough audio is buffered (OpenAI only)",
    )
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

    @property
    def cascaded_config(self) -> Optional["CascadedConfig"]:
        """Get the CascadedConfig for livekit provider.

        Returns the config from CASCADED_CONFIGS if a name is specified,
        otherwise returns None (will use defaults).
        """
        if self.cascaded_config_name is None:
            return None

        from tau2.voice.audio_native.livekit.config import CASCADED_CONFIGS

        if self.cascaded_config_name not in CASCADED_CONFIGS:
            raise ValueError(
                f"Unknown cascaded config: '{self.cascaded_config_name}'. "
                f"Available: {list(CASCADED_CONFIGS.keys())}"
            )
        return CASCADED_CONFIGS[self.cascaded_config_name]


class BaseRunConfig(BaseModel):
    """Base configuration shared by both text (half-duplex) and voice (full-duplex) modes.

    Do not instantiate directly. Use TextRunConfig or VoiceRunConfig.
    """

    # ---- Domain and task selection ----
    domain: Annotated[
        str,
        Field(
            description="The domain to run the simulation on",
            default="airline",
        ),
    ]
    task_set_name: Annotated[
        Optional[str],
        Field(
            description="The task set to run the simulation on. If not provided, will load default task set for the domain.",
            default=None,
        ),
    ]
    task_split_name: Annotated[
        Optional[str],
        Field(
            description="The task split to run the simulation on. If not provided, will load 'base' split.",
            default="base",
        ),
    ]
    task_ids: Annotated[
        Optional[list[str]],
        Field(
            description="The task IDs to run the simulation on",
            default=None,
        ),
    ]
    num_tasks: Annotated[
        Optional[int],
        Field(
            description="The number of tasks to run the simulation on",
            default=None,
        ),
    ]

    # ---- User simulator ----
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

    # ---- Execution parameters ----
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
    save_to: Annotated[
        Optional[str],
        Field(
            description="The path to json file where to save the simulation results",
            default=DEFAULT_SAVE_TO,
        ),
    ]
    max_concurrency: Annotated[
        int,
        Field(
            description="The maximum number of concurrent simulations to run",
            default=DEFAULT_MAX_CONCURRENCY,
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

    # ---- Retry ----
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

    # ---- Resume and review ----
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
    hallucination_retries: Annotated[
        int,
        Field(
            description="Maximum number of retries when a user simulator hallucination is detected. "
            "Set to 0 to disable. "
            "Each retry re-runs the simulation with a different seed and feedback.",
            default=3,
        ),
    ]

    # ---- Misc ----
    is_remote: Annotated[
        bool,
        Field(
            description="Whether to run the simulation remotely",
            default=False,
        ),
    ]

    # ---- Knowledge retrieval ----
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

    # ---- Abstract-ish properties (subclasses must override) ----

    @property
    def effective_agent(self) -> str:
        """The agent implementation name to use."""
        raise NotImplementedError("Subclasses must implement effective_agent")

    @property
    def effective_user(self) -> str:
        """The user implementation name to use."""
        raise NotImplementedError("Subclasses must implement effective_user")

    @property
    def effective_max_steps(self) -> int:
        """Maximum simulation steps (turns for text, ticks for voice)."""
        raise NotImplementedError("Subclasses must implement effective_max_steps")

    @property
    def effective_agent_model(self) -> str:
        """The agent model identifier."""
        raise NotImplementedError("Subclasses must implement effective_agent_model")

    @property
    def effective_agent_provider(self) -> Optional[str]:
        """The agent provider (e.g., 'openai'). None for text mode."""
        raise NotImplementedError("Subclasses must implement effective_agent_provider")

    @property
    def effective_user_model(self) -> str:
        """The user model identifier. Always llm_user."""
        return self.llm_user

    @property
    def is_voice(self) -> bool:
        """Whether this is a voice (full-duplex) configuration."""
        return isinstance(self, VoiceRunConfig)

    def validate(self) -> None:
        """Validate the run config."""
        pass


class TextRunConfig(BaseRunConfig):
    """Configuration for half-duplex (text) simulations.

    Text mode uses turn-based message exchange between an LLM agent and a
    user simulator, with an Orchestrator managing the conversation.
    """

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

    # ---- Properties ----

    @property
    def effective_agent(self) -> str:
        return self.agent

    @property
    def effective_user(self) -> str:
        return self.user

    @property
    def effective_max_steps(self) -> int:
        return self.max_steps

    @property
    def effective_agent_model(self) -> str:
        return self.llm_agent

    @property
    def effective_agent_provider(self) -> Optional[str]:
        return None


class VoiceRunConfig(BaseRunConfig):
    """Configuration for full-duplex (voice/audio-native) simulations.

    Voice mode uses real-time audio exchange between a discrete-time audio-native
    agent and a voice streaming user simulator, with a FullDuplexOrchestrator
    managing the tick-based simulation.
    """

    # ---- Audio-native config (required) ----
    audio_native_config: Annotated[
        AudioNativeConfig,
        Field(
            description="Configuration for audio-native mode (provider, model, timing, thresholds, etc.).",
        ),
    ]

    # ---- Voice-specific ----
    speech_complexity: Annotated[
        SpeechComplexity,
        Field(
            description="Speech environment complexity level: 'control' (clean speech, no effects), 'regular' (realistic with background noise and effects), plus ablation variants",
            default="regular",
        ),
    ]
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

    # ---- Properties ----

    @property
    def effective_agent(self) -> str:
        return DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION

    @property
    def effective_user(self) -> str:
        return DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION

    @property
    def effective_max_steps(self) -> int:
        return self.audio_native_config.max_steps_ticks

    @property
    def effective_agent_model(self) -> str:
        return self.audio_native_config.model

    @property
    def effective_agent_provider(self) -> Optional[str]:
        return self.audio_native_config.provider


# Type alias for backward compatibility: accepts either text or voice config
RunConfig = Union[TextRunConfig, VoiceRunConfig]


class NLAssertionCheck(BaseModel):
    """
    A natural language assertion.
    """

    nl_assertion: str
    met: bool
    justification: str


class CommunicateCheck(BaseModel):
    """
    A communication check.
    """

    info: str
    met: bool
    justification: str


class DBCheck(BaseModel):
    """
    A database check.
    """

    db_match: bool
    db_reward: float


class ActionCheck(BaseModel):
    """
    An action check.
    """

    action: Action
    action_match: bool
    action_reward: float
    tool_type: Optional[ToolType] = Field(
        description="The type of tool (read/write/think/generic).",
        default=None,
    )


class EnvAssertionCheck(BaseModel):
    """
    An environment assertion check.
    """

    env_assertion: EnvAssertion
    met: bool
    reward: float


# =============================================================================
# Review Data Models (for LLM-based conversation review)
# =============================================================================


class ReviewError(BaseModel):
    """
    Represents an error found during conversation review.
    """

    source: Literal["user", "agent", "unknown"] = Field(
        description="Who made the error: 'user', 'agent', or 'unknown'."
    )
    error_type: Optional[str] = Field(
        description="Type of error: 'content_error' or 'interruption_error'.",
        default=None,
    )
    error_tags: list[str] = Field(
        description="Tags classifying the error. Must have at least one tag.",
        default_factory=list,
    )
    severity: Optional[
        Literal["minor", "critical", "critical_helped", "critical_hindered"]
    ] = Field(
        description="Error severity. For user errors: 'critical_helped' (helped agent inappropriately), 'critical_hindered' (made task harder/impossible), 'minor' (no impact). For agent errors: 'critical' (caused failure or policy violation), 'minor' (suboptimal but no impact).",
        default=None,
    )
    # For full-duplex conversations, use tick_start/tick_end to identify the segment
    # For turn-based conversations, turn_idx is still used
    turn_idx: Optional[int] = Field(
        description="The turn index where the error occurred (turn-based only).",
        default=None,
    )
    tick_start: Optional[int] = Field(
        description="Start tick of the segment where the error occurred (full-duplex only).",
        default=None,
    )
    tick_end: Optional[int] = Field(
        description="End tick of the segment where the error occurred (full-duplex only).",
        default=None,
    )
    reasoning: str = Field(
        description="Explanation of the error or why there is no error."
    )
    correct_behavior: Optional[str] = Field(
        description="What should have been done instead.",
        default=None,
    )


class Review(BaseModel):
    """
    Result of reviewing a conversation for both user and agent errors.
    """

    summary: str = Field(
        description="Brief summary of the conversation review.",
        default="",
    )
    agent_error: bool = Field(
        description="Whether the agent made at least one error.",
        default=False,
    )
    user_error: bool = Field(
        description="Whether the user simulator made at least one error.",
        default=False,
    )
    critical_user_error: bool = Field(
        description="Whether at least one critical user error was found (severity 'critical_helped' or 'critical_hindered').",
        default=False,
    )
    has_errors: bool = Field(
        description="Whether any errors were found in the conversation."
    )
    errors: list[ReviewError] = Field(
        description="List of errors found in the conversation.",
        default_factory=list,
    )
    cost: Optional[float] = Field(
        description="The cost of the review.",
        default=None,
    )


class UserOnlyReviewError(BaseModel):
    """
    Represents an error made by the user simulator during a conversation.
    """

    # For full-duplex conversations, use tick_start/tick_end to identify the segment
    # For turn-based conversations, turn_idx is still used
    turn_idx: Optional[int] = Field(
        description="The turn index where the error occurred (turn-based only).",
        default=None,
    )
    tick_start: Optional[int] = Field(
        description="Start tick of the segment where the error occurred (full-duplex only).",
        default=None,
    )
    tick_end: Optional[int] = Field(
        description="End tick of the segment where the error occurred (full-duplex only).",
        default=None,
    )
    error_type: str = Field(
        description="Type of error: 'content_error' or 'interruption_error'.",
        default="content_error",
    )
    error_tags: list[str] = Field(
        description="Tags classifying the error. Must have at least one tag.",
        default_factory=list,
    )
    severity: Optional[Literal["minor", "critical"]] = Field(
        description="Severity of user error: 'critical' if it influenced the outcome, 'minor' otherwise.",
        default=None,
    )
    reasoning: str = Field(description="Explanation of why this is an error.")
    user_message: Optional[str] = Field(
        description="The problematic user message content.",
        default=None,
    )
    correct_behavior: Optional[str] = Field(
        description="What the user should have said or done instead.",
        default=None,
    )


class UserOnlyReview(BaseModel):
    """
    Result of reviewing a user simulator's behavior in a conversation.
    """

    summary: str = Field(
        description="Brief summary of the conversation review.",
        default="",
    )
    user_error: bool = Field(
        description="Whether the user simulator made at least one error.",
        default=False,
    )
    critical_user_error: bool = Field(
        description="Whether at least one critical user error was found (severity 'critical_helped' or 'critical_hindered').",
        default=False,
    )
    has_errors: bool = Field(description="Whether the user simulator made any errors.")
    errors: list[UserOnlyReviewError] = Field(
        description="List of errors made by the user simulator.",
        default_factory=list,
    )
    cost: Optional[float] = Field(
        description="The cost of the review.",
        default=None,
    )


class HallucinationCheckError(BaseModel):
    """
    Represents a hallucination detected in the user simulator's messages.
    """

    reasoning: str = Field(description="Explanation of why this is a hallucination.")
    user_message: Optional[str] = Field(
        description="The problematic user message content.",
        default=None,
    )
    correct_behavior: Optional[str] = Field(
        description="What the user should have said or done instead.",
        default=None,
    )


class HallucinationCheck(BaseModel):
    """
    Result of checking a conversation for user simulator hallucinations.
    """

    reasoning: str = Field(
        description="Step-by-step reasoning about the conversation before the decision.",
        default="",
    )
    hallucination_found: bool = Field(
        description="Whether any hallucinations were detected.",
        default=False,
    )
    errors: list[HallucinationCheckError] = Field(
        description="List of hallucinations found.",
        default_factory=list,
    )
    summary: str = Field(
        description="Brief summary of the hallucination check.",
        default="",
    )
    cost: Optional[float] = Field(
        description="The cost of the hallucination check.",
        default=None,
    )


class AuthenticationClassification(BaseModel):
    """
    Classification of user authentication outcome in a conversation.
    """

    status: Literal["succeeded", "failed", "not_needed"] = Field(
        description="Authentication status: 'succeeded', 'failed', or 'not_needed'."
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen.",
        default="",
    )
    cost: Optional[float] = Field(
        description="The cost of the classification.",
        default=None,
    )


class ErrorSource(str, Enum):
    """Source of the error in a simulation."""

    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"  # Orchestrator, framework, or infrastructure error


class ErrorType(str, Enum):
    """Type of error in a simulation."""

    TRANSCRIPTION = "transcription"  # ASR/speech-to-text errors
    VAD = "vad"  # Voice activity detection / turn-taking issues
    LOGICAL = "logical"  # Reasoning, tool call, or instruction following errors
    HALLUCINATION = "hallucination"  # Made up information
    UNRESPONSIVE = "unresponsive"  # Agent disappeared / no response / latency
    EARLY_TERMINATION = "early_termination"  # Ended conversation prematurely


class SimulationNote(BaseModel):
    """
    A note about a simulation run.
    Used to record observations, comments, or annotations about specific simulation runs.
    Unlike TaskIssue, this does NOT modify the task definition.
    """

    id: str = Field(description="Unique identifier for the note.")
    note: Annotated[
        str,
        Field(description="The note/observation about the simulation."),
    ]
    author_email: Annotated[
        Optional[str],
        Field(
            description="Email of the person who created this note.",
            default=None,
        ),
    ]
    created_at: Annotated[
        Optional[str],
        Field(
            description="ISO datetime when the note was created.",
            default=None,
        ),
    ]
    # Simulation metadata
    simulation_id: Annotated[
        str,
        Field(description="ID of the simulation this note is about."),
    ]
    task_id: Annotated[
        str,
        Field(description="ID of the task for this simulation."),
    ]
    trial: Annotated[
        int,
        Field(description="Trial number of the simulation."),
    ]
    # Source location
    source_results_file: Annotated[
        Optional[str],
        Field(
            description="Path to the original results file where the simulation was found.",
            default=None,
        ),
    ]
    simulation_file: Annotated[
        Optional[str],
        Field(
            description="Path to the simulation JSON file associated with this note.",
            default=None,
        ),
    ]
    # Qualitative analysis fields
    error_source: Annotated[
        Optional[ErrorSource],
        Field(
            description="Source of the error: agent, user, or system (framework/orchestrator).",
            default=None,
        ),
    ]
    error_type: Annotated[
        Optional[ErrorType],
        Field(
            description="Type of error: transcription, vad, logical, hallucination, unresponsive, or early_termination.",
            default=None,
        ),
    ]

    def __str__(self) -> str:
        lines = []
        lines.append(
            f"📝 [{self.id}] {self.note[:50]}{'...' if len(self.note) > 50 else ''}"
        )
        lines.append(
            f"  Simulation: {self.simulation_id} (Task: {self.task_id}, Trial: {self.trial})"
        )
        if self.source_results_file:
            lines.append(f"  Source: {self.source_results_file}")
        if self.author_email:
            lines.append(f"  Author: {self.author_email}")
        if self.created_at:
            lines.append(f"  Created: {self.created_at}")
        return "\n".join(lines)


class RewardInfo(BaseModel):
    """
    The reward received by the agent.
    """

    reward: Annotated[float, Field(description="The reward received by the agent.")]
    db_check: Annotated[
        Optional[DBCheck], Field(description="The database check.", default=None)
    ]
    env_assertions: Annotated[
        Optional[list[EnvAssertionCheck]],
        Field(description="The environment assertions.", default=None),
    ]
    action_checks: Annotated[
        Optional[list[ActionCheck]],
        Field(description="The action checks.", default=None),
    ]
    nl_assertions: Annotated[
        Optional[list[NLAssertionCheck]],
        Field(description="The natural language assertions.", default=None),
    ]
    communicate_checks: Annotated[
        Optional[list[CommunicateCheck]],
        Field(
            description="Checks that the agent communicated the required information.",
            default=None,
        ),
    ]
    reward_basis: Annotated[
        Optional[list[RewardType]],
        Field(
            description="The basis of the reward. Fields that are used to calculate the reward.",
            default_factory=lambda: [RewardType.DB],
        ),
    ]
    reward_breakdown: Annotated[
        Optional[dict[RewardType, float]],
        Field(
            description="The breakdown of the reward.",
            default=None,
        ),
    ]
    info: Annotated[
        Optional[dict],
        Field(description="Additional information about the reward.", default=None),
    ]

    @property
    def partial_action_reward(self) -> Optional[dict]:
        """
        Get the partial reward breakdown for actions.

        Returns a dict with:
        - total: (correct, count, proportion)
        - read: (correct, count, proportion) or None if no read actions
        - write: (correct, count, proportion) or None if no write actions

        Returns None if there are no action_checks.
        """
        if not self.action_checks:
            return None

        total_correct = sum(1 for ac in self.action_checks if ac.action_match)
        total_count = len(self.action_checks)
        total_proportion = total_correct / total_count if total_count > 0 else 0.0

        # Filter by tool type
        read_checks = [ac for ac in self.action_checks if ac.tool_type == ToolType.READ]
        write_checks = [
            ac for ac in self.action_checks if ac.tool_type == ToolType.WRITE
        ]

        read_correct = sum(1 for ac in read_checks if ac.action_match)
        read_count = len(read_checks)
        read_proportion = read_correct / read_count if read_count > 0 else None

        write_correct = sum(1 for ac in write_checks if ac.action_match)
        write_count = len(write_checks)
        write_proportion = write_correct / write_count if write_count > 0 else None

        return {
            "total": {
                "correct": total_correct,
                "count": total_count,
                "proportion": total_proportion,
            },
            "read": (
                {
                    "correct": read_correct,
                    "count": read_count,
                    "proportion": read_proportion,
                }
                if read_count > 0
                else None
            ),
            "write": (
                {
                    "correct": write_correct,
                    "count": write_count,
                    "proportion": write_proportion,
                }
                if write_count > 0
                else None
            ),
        }


class AgentInfo(BaseModel):
    """
    Agent information.
    """

    implementation: str = Field(description="The type of agent.")
    llm: Optional[str] = Field(description="The LLM used by the agent.", default=None)
    llm_args: Optional[dict] = Field(
        description="The arguments to pass to the LLM for the agent.", default=None
    )
    voice_settings: Optional[VoiceSettings] = Field(
        description="Voice synthesis and transcription settings for the agent",
        default=None,
    )


class UserInfo(BaseModel):
    """
    User information.
    """

    implementation: str = Field(description="The type of user.")
    llm: Optional[str] = Field(description="The LLM used by the user.", default=None)
    llm_args: Optional[dict] = Field(
        description="The arguments to pass to the LLM for the user.", default=None
    )
    global_simulation_guidelines: Optional[str] = Field(
        description="The global simulation guidelines for the user.", default=None
    )
    voice_settings: Optional[VoiceSettings] = Field(
        description="Voice synthesis and transcription settings for the user",
        default=None,
    )
    persona_config: Optional[PersonaConfig] = Field(
        description="Runtime persona configuration for the user simulator",
        default=None,
    )


class Info(BaseModel):
    """Information about the simulator."""

    git_commit: str = Field(description="The git commit hash.")
    num_trials: int = Field(description="The number of trials.")
    max_steps: int = Field(description="The maximum number of steps.")
    max_errors: int = Field(description="The maximum number of errors.")
    user_info: UserInfo = Field(description="User information.")
    agent_info: AgentInfo = Field(description="Agent information.")
    environment_info: EnvironmentInfo = Field(description="Environment information.")
    seed: Optional[int] = Field(
        description="The seed used for the simulation.", default=None
    )
    text_streaming_config: Optional[dict] = Field(
        description="Text streaming configuration",
        default=deepcopy(DEFAULT_TEXT_STREAMING_CONFIG),
    )
    speech_complexity: Optional[SpeechComplexity] = Field(
        description="Speech complexity level for audio-native mode",
        default=None,
    )
    audio_native_config: Optional["AudioNativeConfig"] = Field(
        description="Configuration for audio-native mode",
        default=None,
    )
    retrieval_config: Optional[str] = Field(
        description="Knowledge retrieval config name (knowledge domain only).",
        default=None,
    )
    retrieval_config_kwargs: Optional[dict] = Field(
        description="Arguments passed to the retrieval config constructor.",
        default=None,
    )


class TerminationReason(str, Enum):
    USER_STOP = "user_stop"
    AGENT_STOP = "agent_stop"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    TOO_MANY_ERRORS = "too_many_errors"
    AGENT_ERROR = "agent_error"
    USER_ERROR = "user_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"  # Task failed due to infrastructure (e.g., API disconnect)
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    UNEXPECTED_ERROR = "unexpected_error"


class SimulationRun(BaseModel):
    """
    Simulation run for the given task.
    """

    id: str = Field(description="The unique identifier for the simulation run.")
    task_id: str = Field(description="The unique identifier for the task.")
    timestamp: str = Field(
        description="The timestamp of the simulation.", default_factory=get_now
    )
    start_time: str = Field(description="The start time of the simulation.")
    end_time: str = Field(description="The end time of the simulation.")
    duration: float = Field(description="The duration of the simulation.")
    termination_reason: TerminationReason = Field(
        description="The reason for the termination of the simulation."
    )
    agent_cost: Optional[float] = Field(
        description="The cost of the agent.", default=None
    )
    user_cost: Optional[float] = Field(
        description="The cost of the user.", default=None
    )
    reward_info: Optional[RewardInfo] = Field(
        description="The reward received by the agent.", default=None
    )
    messages: Optional[list[Message]] = Field(
        description="The messages exchanged between the user, agent and environment. "
        "Populated for half-duplex simulations. For full-duplex, use get_messages() "
        "which derives messages from ticks when this field is None.",
        default=None,
    )
    ticks: Optional[list[Tick]] = Field(
        description="The ticks of the simulation. Only available in full-duplex mode.",
        default=None,
    )
    trial: Optional[int] = Field(description="Trial number", default=None)
    seed: Optional[int] = Field(
        description="Seed used for the simulation.", default=None
    )
    mode: str = Field(
        description="The communication mode used for the simulation.",
        default=CommunicationMode.HALF_DUPLEX.value,
    )
    speech_environment: Optional[SpeechEnvironment] = Field(
        description="Speech environment used for this simulation",
        default=None,
    )
    review: Optional[Review] = Field(  # TODO: Add auth_classification to review field
        description="LLM-based review of the conversation (agent + user errors).",
        default=None,
    )
    user_only_review: Optional[UserOnlyReview] = Field(
        description="LLM-based review of user simulator behavior only.",
        default=None,
    )
    info: Optional[dict] = Field(
        description="Additional diagnostics and metrics from the simulation.",
        default=None,
    )
    auth_classification: Optional[AuthenticationClassification] = (
        Field(  # TODO: Add to review field
            description="Classification of user authentication outcome.",
            default=None,
        )
    )
    hallucination_retries_used: int = Field(
        description="Number of retries triggered by user simulator hallucinations.",
        default=0,
    )
    hallucination_check: Optional[HallucinationCheck] = Field(
        description="Result of the hallucination check for this simulation.",
        default=None,
    )
    provider_session_id: Optional[str] = Field(
        description="Provider session ID (e.g., OpenAI session ID, xAI conversation ID) for debugging.",
        default=None,
    )
    policy: Optional[str] = Field(
        description="The policy/system prompt used for this simulation (knowledge domain only).",
        default=None,
    )
    effect_timeline: Optional[EffectTimeline] = Field(
        description="Timeline of audio effect events during the simulation (full-duplex voice only).",
        default=None,
    )

    def get_messages(self) -> list[Message]:
        """Return the flat message list, deriving from ticks if messages is not stored.

        For half-duplex simulations, returns the stored messages directly.
        For full-duplex simulations where messages were not stored (to save space),
        derives them by flattening ticks.
        """
        if self.messages is not None:
            return self.messages
        if self.ticks is not None:
            messages: list[Message] = []
            for tick in self.ticks:
                messages.extend(tick.get_all_messages())
            messages = sorted(messages, key=lambda m: m.timestamp)
            for i, msg in enumerate(messages):
                msg.turn_idx = i
            return messages
        return []


class SimulationIndexEntry(BaseModel):
    """Lightweight summary of a simulation for the dir-format index.

    Stored in results.json alongside metadata so that external consumers
    (e.g. the web leaderboard) can access simulation summaries without
    fetching individual simulation files. Also used for integrity
    validation on load.
    """

    id: str
    task_id: int | str
    trial: int
    reward: float | None = None
    termination_reason: str | None = None
    agent_cost: float | None = None
    duration: float | None = None


class Results(BaseModel):
    """
    Run results.

    Supports two storage formats:
    - "json": single monolithic JSON file with all data (default for text runs).
    - "dir": metadata in results.json + individual simulation files in a
      simulations/ subdirectory (default for voice runs — enables random
      access and O(1) checkpointing for large simulation files).

    Use load()/save() for full round-trip. Use load_metadata() for fast metadata
    access, iter_simulations() for streaming, and df_from_path() for streaming
    DataFrame construction.
    """

    timestamp: Optional[str] = Field(
        description="The timestamp of the simulation.", default_factory=get_now
    )
    info: Info = Field(description="Information.")
    tasks: list[Task] = Field(description="The list of tasks.")
    simulations: list[SimulationRun] = Field(
        description="The list of simulations.", default_factory=list
    )
    simulation_index: list[SimulationIndexEntry] | None = Field(
        default=None,
        description="Lightweight simulation summaries for dir format. "
        "Populated on save (dir format) and used for integrity validation "
        "on load and by the web frontend.",
    )

    # ---- Format detection and path resolution ----

    @staticmethod
    def _detect_format(path: Path) -> Literal["json", "dir"]:
        """Detect storage format from path.

        Returns "dir" if path is a directory or has a sibling simulations/
        subdirectory, otherwise "json" (monolithic format).
        """
        path = Path(path)
        if path.is_dir():
            return "dir"
        sims_dir = path.parent / SIMULATIONS_DIR
        if sims_dir.is_dir():
            return "dir"
        return "json"

    @staticmethod
    def _resolve_paths(path: Path) -> tuple[Path, Path]:
        """Resolve metadata file path and simulations directory from a path.

        Args:
            path: Either a directory or a results.json file path.

        Returns:
            Tuple of (metadata_json_path, simulations_directory_path).
        """
        path = Path(path)
        if path.is_dir():
            return path / "results.json", path / SIMULATIONS_DIR
        return path, path.parent / SIMULATIONS_DIR

    def _build_simulation_index(self) -> list[SimulationIndexEntry]:
        """Build a simulation index from the current simulations list."""
        return [
            SimulationIndexEntry(
                id=sim.id,
                task_id=sim.task_id,
                trial=sim.trial,
                reward=sim.reward_info.reward if sim.reward_info else None,
                termination_reason=sim.termination_reason,
                agent_cost=sim.agent_cost,
                duration=sim.duration,
            )
            for sim in self.simulations
        ]

    # ---- Load / Save ----

    @classmethod
    def load(cls, path: Path) -> "Results":
        """Load Results from disk, auto-detecting format.

        Supports both monolithic JSON and directory-based formats.
        For directory format, loads results.json metadata and all individual
        simulation files from the simulations/ subdirectory.
        """
        path = Path(path)
        fmt = cls._detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                return cls.model_validate_json(f.read())

        meta_path, sims_dir = cls._resolve_paths(path)
        with open(meta_path, "r") as f:
            meta = json.loads(f.read())

        meta.pop("format_version", None)

        # Validate simulation files against index if present
        index = meta.get("simulation_index")
        simulations = []
        if sims_dir.exists():
            for sim_file in sorted(sims_dir.glob("*.json")):
                with open(sim_file, "r") as f:
                    simulations.append(json.loads(f.read()))

        if index is not None:
            indexed_ids = {entry["id"] for entry in index}
            on_disk_ids = (
                {f.stem for f in sims_dir.glob("*.json")}
                if sims_dir.exists()
                else set()
            )
            missing = indexed_ids - on_disk_ids
            extra = on_disk_ids - indexed_ids
            errors = []
            if missing:
                errors.append(f"Missing simulation files: {sorted(missing)}")
            if extra:
                errors.append(f"Extra simulation files not in index: {sorted(extra)}")
            if errors:
                raise ValueError(
                    f"Dir format integrity check failed for {meta_path}: "
                    + "; ".join(errors)
                )

        meta["simulations"] = simulations
        return cls.model_validate(meta)

    def save(self, path: Path, format: Literal["json", "dir"] = "json") -> None:
        """Save the results to disk.

        Args:
            path: File path (for "json") or directory/file path (for "dir").
                  For "dir" format, if path ends in .json, the simulations/
                  subdirectory is created alongside it. If path is a directory,
                  results.json is created inside it.
            format: Storage format. "json" writes a single monolithic JSON file.
                    "dir" writes metadata to results.json and each
                    simulation to a separate file in simulations/.
        """
        path = Path(path)
        if format == "json":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(self.model_dump_json(indent=2))
            return

        meta_path, sims_dir = self._resolve_paths(path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        sims_dir.mkdir(parents=True, exist_ok=True)

        self.simulation_index = self._build_simulation_index()
        meta = self.model_dump(mode="json", exclude={"simulations"})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        for sim in self.simulations:
            sim_path = sims_dir / f"{sim.id}.json"
            with open(sim_path, "w") as f:
                f.write(sim.model_dump_json(indent=2))

    def save_metadata(self, path: Path) -> None:
        """Save only metadata to a dir-format results.json.

        Creates the simulations/ subdirectory if needed but does not write
        or modify any simulation files. Used by the checkpoint system to update
        metadata (e.g. after adding tasks) without rewriting all sim files.

        Preserves the existing simulation_index from the on-disk results.json
        if the in-memory simulation_index is not populated.
        """
        meta_path, sims_dir = self._resolve_paths(path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        sims_dir.mkdir(parents=True, exist_ok=True)

        # Preserve on-disk simulation_index if we don't have one in memory
        if self.simulation_index is None and meta_path.exists():
            with open(meta_path, "r") as f:
                existing = json.loads(f.read())
            existing_index = existing.get("simulation_index")
            if existing_index is not None:
                self.simulation_index = [
                    SimulationIndexEntry.model_validate(e) for e in existing_index
                ]

        meta = self.model_dump(mode="json", exclude={"simulations"})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ---- Streaming / lightweight access ----

    @classmethod
    def load_metadata(cls, path: Path) -> "Results":
        """Load only metadata without simulations.

        Returns a Results instance with simulations=[]. For dir format,
        simulation_index is populated if present. Works with both
        JSON and directory-based formats.
        """
        path = Path(path)
        fmt = cls._detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                data = json.loads(f.read())
        else:
            meta_path, _ = cls._resolve_paths(path)
            with open(meta_path, "r") as f:
                data = json.loads(f.read())

        data.pop("format_version", None)
        data.pop("simulations", None)
        data["simulations"] = []
        return cls.model_validate(data)

    @classmethod
    def iter_simulations(cls, path: Path) -> Iterator[SimulationRun]:
        """Yield simulations one at a time without loading all into memory.

        For directory format, reads each simulation file individually —
        peak memory is bounded by a single simulation.
        For JSON format, parses the full file but yields SimulationRun models
        one at a time to avoid constructing all at once.
        """
        path = Path(path)
        fmt = cls._detect_format(path)

        if fmt == "json":
            with open(path, "r") as f:
                data = json.loads(f.read())
            for sim_data in data.get("simulations", []):
                yield SimulationRun.model_validate(sim_data)
        else:
            _, sims_dir = cls._resolve_paths(path)
            if sims_dir.exists():
                for sim_file in sorted(sims_dir.glob("*.json")):
                    with open(sim_file, "r") as f:
                        yield SimulationRun.model_validate_json(f.read())

    # ---- DataFrame construction helpers ----

    @staticmethod
    def _transfer_only(task: Task) -> bool:
        if task.evaluation_criteria is None:
            return False
        if task.evaluation_criteria.actions is None:
            return False
        actions = task.evaluation_criteria.actions
        if len(actions) != 1:
            return False
        return "transfer" in actions[0].name.lower()

    @staticmethod
    def _task_metrics(task: Task) -> dict:
        eval_metrics = (
            task.evaluation_criteria.info()
            if task.evaluation_criteria is not None
            else {}
        )
        num_actions = (
            eval_metrics["num_agent_actions"] + eval_metrics["num_user_actions"]
        )
        if Results._transfer_only(task):
            num_actions = -1
        return {
            "task_num_agent_actions": eval_metrics["num_agent_actions"],
            "task_num_user_actions": eval_metrics["num_user_actions"],
            "task_num_actions": num_actions,
            "task_num_env_assertions": eval_metrics["num_env_assertions"],
            "task_num_nl_assertions": eval_metrics["num_nl_assertions"],
        }

    @staticmethod
    def _sim_to_row(sim: "SimulationRun", info: "Info") -> dict:
        return {
            "simulation_id": sim.id,
            "task_id": sim.task_id,
            "trial": sim.trial,
            "seed": sim.seed,
            "reward": sim.reward_info.reward if sim.reward_info else None,
            "agent_cost": sim.agent_cost,
            "user_cost": sim.user_cost,
            "termination_reason": sim.termination_reason,
            "duration": sim.duration,
            "num_messages": len(sim.get_messages()),
            "info_git_commit": info.git_commit,
            "info_seed": info.seed,
            "info_num_trials": info.num_trials,
            "info_max_steps": info.max_steps,
            "info_max_errors": info.max_errors,
            "info_domain": info.environment_info.domain_name,
            "info_user_implementation": info.user_info.implementation,
            "info_user_llm": info.user_info.llm,
            "info_user_llm_args": info.user_info.llm_args,
            "info_agent_implementation": info.agent_info.implementation,
            "info_agent_llm": info.agent_info.llm,
            "info_agent_llm_args": info.agent_info.llm_args,
        }

    def to_df(self) -> pd.DataFrame:
        """Convert a Results object to a pandas DataFrame."""
        rows = []
        for sim in self.simulations:
            row = self._sim_to_row(sim, self.info)
            task = next(t for t in self.tasks if t.id == sim.task_id)
            row.update(self._task_metrics(task))
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def df_from_path(cls, path: Path) -> pd.DataFrame:
        """Build a metrics DataFrame by streaming simulations from disk.

        Like to_df() but loads simulations one at a time, keeping peak memory
        bounded by the size of a single simulation. Works with both formats.
        """
        metadata = cls.load_metadata(path)
        tasks_by_id = {t.id: t for t in metadata.tasks}
        rows = []
        for sim in cls.iter_simulations(path):
            row = cls._sim_to_row(sim, metadata.info)
            task = tasks_by_id.get(sim.task_id)
            if task:
                row.update(cls._task_metrics(task))
            rows.append(row)
        return pd.DataFrame(rows)
