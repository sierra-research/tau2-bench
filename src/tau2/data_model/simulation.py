import json
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION,
    DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION,
    DEFAULT_BANKING_RETRIEVAL_CONFIG,
    DEFAULT_ELICITATION_LLM_ARGS_USER,
    DEFAULT_ELICITATION_LLM_USER,
    DEFAULT_ELICITATION_MAX_STEPS_SECONDS,
    DEFAULT_LANGUAGE,
    DEFAULT_LOCALIZE_DB,
    DEFAULT_SAVE_TO,
    DEFAULT_TEXT_STREAMING_CONFIG,
    ELICITATION_DOMAINS,
)
from tau2.data_model.audio_effects import EffectTimeline
from tau2.data_model.benchmark import BenchmarkRunInfo
from tau2.data_model.elicitation import (
    ElicitationConfig,
    ElicitationStrategy,
    SampledComplication,
)
from tau2.data_model.language import (
    LanguageArtifactPins,
    RecordedLanguageContext,
)
from tau2.data_model.message import Message, Tick
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.run_config import (
    AudioNativeConfig,
    RunSettings,
    TextRunSettings,
    VoiceRunSettings,
)
from tau2.data_model.semantic_communication import (
    CommunicateScorer,
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.tasks import Action, EnvAssertion, RewardType, Task
from tau2.data_model.usage import SessionUsage
from tau2.data_model.voice import SpeechComplexity, SpeechEnvironment, VoiceSettings
from tau2.environment.environment import EnvironmentInfo
from tau2.environment.toolkit import ToolType
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.utils import get_now

SIMULATIONS_DIR = "simulations"


class BaseRunConfig(RunSettings):
    """Base configuration shared by both text (half-duplex) and voice (full-duplex) modes.

    Do not instantiate directly. Use TextRunConfig or VoiceRunConfig.
    """

    language: str = Field(
        default=DEFAULT_LANGUAGE,
        description="Published language pack; canonical English unless a language is requested.",
    )
    pack_extension: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Explicit separate-run domain extension.",
    )

    localize_db: Optional[bool] = Field(
        default=None,
        description="Use the language's published regional database, tasks and policy "
        "where it has one (default). False keeps canonical scenarios with the same "
        "call language.",
    )

    language_artifacts: Optional[LanguageArtifactPins] = Field(
        default=None,
        description="Resolved content pins carried to workers; mutations fail before provider use.",
    )

    @model_validator(mode="after")
    def _resolve_elicitation(self) -> "BaseRunConfig":
        """Resolve domain defaults before either CLI or programmatic execution."""
        if self.domain not in ELICITATION_DOMAINS:
            if self.elicitation is not None:
                raise ValueError("elicitation settings require an elicitation domain")
            return self
        if self.elicitation is None:
            self.elicitation = ElicitationConfig()
        if (
            self.elicitation.strategy == ElicitationStrategy.AGENT_DIRECTED
            and not self.is_voice
        ):
            raise ValueError(
                "Agent-directed intake requires a voice run; "
                "select the scaffolded strategy for the text ablation"
            )
        if "llm_user" not in self.model_fields_set:
            self.llm_user = DEFAULT_ELICITATION_LLM_USER
        if (
            "llm_args_user" not in self.model_fields_set
            and self.llm_user == DEFAULT_ELICITATION_LLM_USER
        ):
            self.llm_args_user = deepcopy(DEFAULT_ELICITATION_LLM_ARGS_USER)
        if (
            self.is_voice
            and "max_steps_seconds" not in self.audio_native_config.model_fields_set
        ):
            self.audio_native_config = self.audio_native_config.model_copy(
                update={"max_steps_seconds": DEFAULT_ELICITATION_MAX_STEPS_SECONDS}
            )
        return self

    benchmark: Optional[BenchmarkRunInfo] = Field(
        default=None, description="Explicit versioned benchmark selection, if any."
    )

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

    # ---- Execution parameters ----
    save_to: Annotated[
        Optional[str],
        Field(
            description="The path to json file where to save the simulation results",
            default=DEFAULT_SAVE_TO,
        ),
    ]

    # ---- Abstract-ish properties (subclasses must override) ----

    @model_validator(mode="after")
    def _default_banking_retrieval_config(self) -> "BaseRunConfig":
        """Default retrieval_config to alltools for banking_knowledge."""
        if self.domain == "banking_knowledge" and self.retrieval_config is None:
            object.__setattr__(
                self, "retrieval_config", DEFAULT_BANKING_RETRIEVAL_CONFIG
            )
        return self

    @model_validator(mode="after")
    def _resolve_language_settings(self) -> "BaseRunConfig":
        """Resolve provider language settings from the run's call language."""
        audio = getattr(self, "audio_native_config", None)
        if audio is None:
            return self
        recorded_language = audio.language
        if audio.language is not None and audio.language != self.language:
            raise ValueError("provider language must match the run language")
        audio = audio.model_copy(deep=True)
        # Derived values are not marked as explicitly set, so an unset-only dump
        # of this config retargets cleanly to another language.
        derived = set()
        if audio.language is None:
            audio.language = self.language
            derived.add("language")
        if audio.provider == "openai" and audio.transcription_language is None:
            audio.transcription_language = self.language.split("-")[0]
            derived.add("transcription_language")
        if audio.transcription_language is not None and audio.provider != "openai":
            raise ValueError("transcription_language applies only to OpenAI Realtime")
        if audio.voice is not None and audio.provider != "openai":
            raise ValueError(
                "voice override is supported for openai; openai_live uses live_config.voice"
            )
        if audio.provider == "livekit":
            from tau2.data_model.cascaded import CascadedConfigRequest

            request = audio.cascaded_request or CascadedConfigRequest()
            if (
                audio.cascaded_config is not None
                and audio.cascaded_config != request.resolve(recorded_language)
            ):
                raise ValueError(
                    "Recorded LiveKit settings differ from the requested pipeline; "
                    "supply overrides through CascadedConfigRequest(config=...)"
                )
            if audio.cascaded_config is None:
                derived.add("cascaded_config")
            audio.cascaded_config = request.resolve(self.language)
        elif audio.cascaded_config is not None or audio.cascaded_request is not None:
            raise ValueError("cascaded configuration applies only to LiveKit")
        audio.__pydantic_fields_set__.difference_update(derived)
        self.audio_native_config = audio
        return self

    def template_values(self) -> dict:
        """This config's explicit request, without its language binding.

        Settings equal to what the run language derives are dropped, so a
        (possibly serialized) config can be re-resolved for another language.
        """
        values = self.model_dump(mode="python", exclude_unset=True)
        values.pop("language", None)
        audio = getattr(self, "audio_native_config", None)
        requested = values.get("audio_native_config")
        if audio is not None and requested is not None:
            if audio.language == self.language:
                requested.pop("language", None)
            if (
                audio.provider == "openai"
                and audio.transcription_language == self.language.split("-")[0]
            ):
                requested.pop("transcription_language", None)
            if audio.provider == "livekit":
                from tau2.data_model.cascaded import CascadedConfigRequest

                request = audio.cascaded_request or CascadedConfigRequest()
                if audio.cascaded_config == request.resolve(self.language):
                    requested.pop("cascaded_config", None)
        return values

    @property
    def effective_localize_db(self) -> bool:
        """Regional scenarios are the default wherever a language publishes one."""
        return DEFAULT_LOCALIZE_DB if self.localize_db is None else self.localize_db

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
        """Validate executable benchmark conditions before starting providers."""
        if self.pack_extension is not None and not self.effective_localize_db:
            raise ValueError("pack extensions require a localized language")
        if (
            self.benchmark is not None
            and self.communicate_scoring.scorer != CommunicateScorer.SEMANTIC
        ):
            raise ValueError(
                "Benchmark scoring requires the pinned semantic communicate-info judge"
            )


class TextRunConfig(BaseRunConfig, TextRunSettings):
    """Configuration for half-duplex (text) simulations.

    Text mode uses turn-based message exchange between an LLM agent and a
    user simulator, with an Orchestrator managing the conversation.
    """

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


class VoiceRunConfig(BaseRunConfig, VoiceRunSettings):
    """Configuration for full-duplex (voice/audio-native) simulations.

    Voice mode uses real-time audio exchange between a discrete-time audio-native
    agent and a voice streaming user simulator, with a FullDuplexOrchestrator
    managing the tick-based simulation.
    """

    def validate(self) -> None:
        """Validate voice-provider requirements before any simulation starts."""
        super().validate()
        if self.audio_native_config.provider == "gemini":
            from tau2.voice.audio_native.gemini.provider import (
                require_extended_thinking_sdk,
            )

            require_extended_thinking_sdk(self.audio_native_config.model)

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


class HallucinationReviewFailure(BaseModel):
    """Evidence retained when the post-call reviewer cannot return a verdict."""

    error_type: str = Field(description="Class of the reviewer failure.")
    message: str = Field(description="Reviewer error message.")
    raw_responses: list[str] = Field(
        description="Invalid raw LLM replies, when a response was received.",
        default_factory=list,
    )


class HallucinationCheck(BaseModel):
    """
    Result of checking a conversation for user simulator hallucinations.
    """

    reasoning: str = Field(
        description="Concise evidence summary for the hallucination verdict.",
        default="",
    )
    hallucination_found: Optional[bool] = Field(
        description="Whether hallucinations were detected; null when review failed.",
        default=False,
    )
    review_status: Literal["completed", "failed"] = Field(
        description="Whether the reviewer produced a valid verdict.",
        default="completed",
    )
    review_failure: Optional[HallucinationReviewFailure] = Field(
        description="Failure evidence when the reviewer could not return a verdict.",
        default=None,
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
    review_model: Optional[str] = Field(
        description="Exact model used for this hallucination review.", default=None
    )
    prompt_version: Optional[str] = Field(
        description="Version of the hallucination review prompt.", default=None
    )
    response_validation_retries: Optional[int] = Field(
        description="Maximum structured-response repair attempts for the review.",
        default=None,
    )

    @model_validator(mode="after")
    def _verdict_matches_status(self) -> "HallucinationCheck":
        if self.review_status == "failed":
            if self.hallucination_found is not None or self.review_failure is None:
                raise ValueError(
                    "Failed review requires no verdict and failure evidence"
                )
        elif self.hallucination_found is None or self.review_failure is not None:
            raise ValueError("Completed review requires a verdict without failure")
        return self


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

    domain_tools_version: Optional[str] = Field(
        default=None,
        description="Observed domain lookup semantics; absent for legacy results.",
    )
    voice_instruction_prompt_version: Optional[str] = Field(
        default=None,
        description="Revision of shared voice instructions used by this run; absent for text, domain overrides, or unrecorded older results.",
    )

    language: Optional[str] = Field(
        default=None,
        description="Resolved published language; absent only in results recorded "
        "before every run resolved a language pack.",
    )
    pack_extension: Optional[str] = Field(
        default=None, description="Explicit separate-run extension, if selected."
    )
    localize_db: Optional[bool] = Field(
        default=None,
        description="Recorded regional database request; null when not requested.",
    )
    language_artifacts: Optional[LanguageArtifactPins] = Field(
        default=None,
        description="Exact published scenario and language artifacts used by this run.",
    )
    language_tasks: dict[str, RecordedLanguageContext] = Field(
        default_factory=dict,
        description="Resolved per-task pack, identity, persona and artifact provenance.",
    )

    communicate_scoring: Optional[CommunicateScoring] = Field(
        default=None,
        description="Communicate-info scorer used by this run; null only in results "
        "recorded before it was explicit (substring matching).",
    )

    @model_validator(mode="before")
    @classmethod
    def read_recorded_semantic_communicate(cls, value):
        """Read the scorer that results saved before ``communicate_scoring`` used."""
        if not isinstance(value, dict) or "semantic_communicate" not in value:
            return value
        value = dict(value)
        judge = value.pop("semantic_communicate")
        if value.get("communicate_scoring") is None:
            value["communicate_scoring"] = (
                CommunicateScoring(scorer=CommunicateScorer.SUBSTRING)
                if judge is None
                else CommunicateScoring(
                    judge=SemanticCommunicateSettings.model_validate(judge)
                )
            )
        return value

    @property
    def effective_communicate_scoring(self) -> CommunicateScoring:
        """The recorded scorer; unrecorded results used substring matching."""
        return self.communicate_scoring or CommunicateScoring(
            scorer=CommunicateScorer.SUBSTRING
        )

    benchmark: Optional[BenchmarkRunInfo] = Field(
        default=None, description="Recorded versioned benchmark selection, if any."
    )

    git_commit: str = Field(description="The git commit hash.")
    elicitation: Optional[ElicitationConfig] = Field(
        default=None, description="Resolved elicitation caller settings, when enabled."
    )
    elicitation_scoring_version: Optional[str] = Field(
        default=None,
        description="Version of the elicitation value normalization and reward contract.",
    )
    elicitation_prompt_version: Optional[str] = Field(
        default=None,
        description="Version of the elicitation-specific agent and caller prompts.",
    )
    elicitation_catalog_version: Optional[str] = Field(
        default=None,
        description="Version of the elicitation stimulus catalog, including clean draws.",
    )
    elicitation_spell_detection_version: Optional[str] = Field(
        default=None,
        description="Version of the spell-event detector used by agent-directed voice runs.",
    )
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
    channel_effects_mode: Optional[Literal["light", "regular", "heavy"]] = None
    speech_effects_mode: Optional[Literal["light", "regular", "heavy"]] = None
    pass3_condition: Optional[Literal["regular", "channel_heavy", "speech_heavy"]] = (
        None
    )
    voice_sampling_seed: Optional[int] = Field(
        default=None,
        description="Caller voice sampling seed, separate from the task and trial seed.",
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

    language_context: Optional[RecordedLanguageContext] = Field(
        default=None,
        description="Exact published language/task/persona used for this simulation.",
    )
    agent_voice: Optional[str] = Field(
        default=None, description="Effective selected agent voice, when known."
    )
    agent_voice_gender: Optional[str] = Field(
        default=None,
        description="Gender resolved by the pinned benchmark voice catalog.",
    )
    agent_voice_catalog: Optional[str] = Field(
        default=None, description="Source revision of the voice metadata catalog."
    )
    id: str = Field(description="The unique identifier for the simulation run.")
    complication: Optional[SampledComplication] = Field(
        default=None,
        description="The seeded elicitation stimulus used for this simulation.",
    )
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
    agent_usage: Optional[SessionUsage] = Field(
        description="Aggregated provider usage (and cost breakdown) for the "
        "agent side. Populated for audio-native full-duplex runs.",
        default=None,
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
        description="The actual domain policy used for this simulation, including any task-specific context.",
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
    def load_metadata(cls, path: Path, *, context: dict | None = None) -> "Results":
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
        return cls.model_validate(data, context=context)

    @classmethod
    def iter_simulations(
        cls, path: Path, *, context: dict | None = None
    ) -> Iterator[SimulationRun]:
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
                yield SimulationRun.model_validate(sim_data, context=context)
        else:
            _, sims_dir = cls._resolve_paths(path)
            if sims_dir.exists():
                for sim_file in sorted(sims_dir.glob("*.json")):
                    with open(sim_file, "r") as f:
                        yield SimulationRun.model_validate_json(
                            f.read(), context=context
                        )

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
