import json
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Annotated

if TYPE_CHECKING:
    from tau2.voice.audio_native.livekit.config import CascadedConfig

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_PROVIDER,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
    DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION,
    DEFAULT_COMMUNICATE_JUDGE_MODE,
    DEFAULT_DELIVERY_JUDGE_CONCURRENCY,
    DEFAULT_DELIVERY_MAX_SEGMENTS,
    DEFAULT_DELIVERY_SAMPLE_RATE,
    DEFAULT_INTEGRATION_DURATION_SECONDS,
    DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_DELIVERY_JUDGE,
    DEFAULT_LLM_DELIVERY_JUDGE_ARGS,
    DEFAULT_LLM_EVAL_USER_SIMULATOR,
    DEFAULT_LLM_NATIVENESS_JUDGE,
    DEFAULT_LLM_NATIVENESS_JUDGE_ARGS,
    DEFAULT_LLM_QUALITY_JUDGE_ARGS,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_STEPS_SECONDS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_PCM_SAMPLE_RATE,
    DEFAULT_QUALITY_MONOLOGUE_SECONDS,
    DEFAULT_QUALITY_RESPONSE_LATENCY_SECONDS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_SAVE_TO,
    DEFAULT_SEED,
    DEFAULT_SEND_AUDIO_INSTANT,
    DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS,
    DEFAULT_TELEPHONY_RATE,
    DEFAULT_TEXT_MAX_CONCURRENCY,
    DEFAULT_TEXT_NOISE_SEED,
    DEFAULT_TEXT_STREAMING_CONFIG,
    DEFAULT_TEXT_WORKERS,
    DEFAULT_TICK_DURATION_SECONDS,
    DEFAULT_USE_LLM_BACKCHANNEL,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS,
    DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS,
    SUBSET_AUTO,
    VOICE_TIMEOUT_SAFETY_FACTOR,
    ReasoningEffort,
    resolve_audio_native_reasoning_effort,
)
from tau2.data_model.audio_effects import EffectTimeline
from tau2.data_model.message import Message, Tick
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.tasks import Action, EnvAssertion, RewardType, Task
from tau2.data_model.usage import SessionUsage
from tau2.data_model.voice import SpeechComplexity, SpeechEnvironment, VoiceSettings
from tau2.environment.environment import EnvironmentInfo
from tau2.environment.toolkit import ToolType
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.utils import get_now
from tau2.voice.audio_native.openai.live_config import LiveConfig

SIMULATIONS_DIR = "simulations"

#: Validation context passed by every path that parses a results file back off
#: disk (`Results.load`, `Results.load_metadata`, the backfill's fragment
#: check). It is the ONLY thing that distinguishes a stored `null` — a result
#: written before a field was resolved eagerly — from a Python `None` handed to
#: a live construction, because both reach a `mode="before"` validator as the
#: same dict. Pass it wherever stored JSON is revived; omit it everywhere else.
STORED_PAYLOAD_CONTEXT: dict = {"stored_payload": True}


def is_stored_payload(info: ValidationInfo) -> bool:
    """True when this validation is reviving a payload read off disk."""
    context = info.context
    return bool(isinstance(context, dict) and context.get("stored_payload"))


class ReasoningEffortSource(str, Enum):
    """Where a recorded ``reasoning_effort`` came from."""

    RUN = "run"
    """Resolved when the run's config was built — the run really used it."""

    INFERRED = "inferred"
    """Not recorded by the run. Derived after the fact from
    ``DEFAULT_AUDIO_NATIVE_REASONING_EFFORT`` keyed on the recorded provider,
    either by the now-retired ``tau2 backfill-reasoning-effort`` repair pass
    or when loading a result that predates eager resolution."""


class ReasoningEffortBackfill(BaseModel):
    """Provenance stamped by the now-retired ``tau2 backfill-reasoning-effort``
    verb when it filled a ``reasoning_effort`` that the run itself never
    recorded. Historical runs carry these stamps, so the model stays."""

    backfilled_at: str = Field(description="When the backfill verb wrote it.")
    git_commit: str = Field(description="Git commit of the backfilling code.")
    tool: str = Field(description="The tau2 verb that wrote the value.")
    basis: str = Field(
        description="Why this value: the table it was read from, or the "
        "measurement it rests on. Written verbatim so the claim can be "
        "weighed without finding the source at this commit."
    )


class AudioNativeConfig(BaseModel):
    """Configuration for audio-native mode using DiscreteTimeAudioNativeAgent.

    This configuration is used when running full-duplex voice simulations
    with audio native APIs (OpenAI Realtime or Gemini Live).
    """

    # Provider selection
    provider: Literal[
        "openai", "openai_live", "gemini", "xai", "nova", "qwen", "livekit"
    ] = Field(
        default=DEFAULT_AUDIO_NATIVE_PROVIDER,
        description="Audio native API provider: 'openai' (OpenAI Realtime), 'openai_live' (OpenAI Live), 'gemini' (Gemini Live), 'xai' (xAI Grok Voice Agent), 'nova' (Amazon Nova Sonic), 'qwen' (Alibaba Qwen Omni), or 'livekit' (LiveKit cascaded STT→LLM→TTS)",
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
    reasoning_effort: ReasoningEffort = Field(
        default=DEFAULT_AUDIO_NATIVE_REASONING_EFFORT[DEFAULT_AUDIO_NATIVE_PROVIDER],
        description=(
            "The reasoning effort this run actually ran at — resolved from "
            "`tau2.config.DEFAULT_AUDIO_NATIVE_REASONING_EFFORT` when the CLI "
            "does not pin one, BEFORE the config is recorded. "
            "'provider_default' is not a level: it means no reasoning setting "
            "was sent to the provider at all, so the server-side default of "
            "that date applied — never read it as equivalent to a named level. "
            "Together with `provider`, `model` and the run's git_commit/"
            "timestamp this fully identifies the arm."
        ),
    )
    reasoning_effort_source: ReasoningEffortSource = Field(
        default=ReasoningEffortSource.RUN,
        description=(
            "Whether `reasoning_effort` was observed at run time ('run') or "
            "derived after the fact from the provider default table "
            "('inferred'). Results written before eager resolution stored null "
            "and load as 'inferred'."
        ),
    )
    reasoning_effort_backfill: Optional[ReasoningEffortBackfill] = Field(
        default=None,
        description=(
            "Set only by the now-retired `tau2 backfill-reasoning-effort` "
            "verb: when/by what an inferred `reasoning_effort` was written to "
            "disk. None on values the run recorded itself."
        ),
    )
    live_config: Optional[LiveConfig] = Field(
        default=None,
        description="Backend model, voice, and optional prompt overrides for the openai_live provider",
    )
    realtime_generation: Optional[bool] = Field(
        default=None,
        description="Whether user LLM and TTS generation run without blocking audio ticks. "
        "Defaults to enabled for openai_live and disabled for other providers.",
    )

    @property
    def realtime_generation_enabled(self) -> bool:
        """Resolve the provider-aware default for real-time user generation."""
        if self.realtime_generation is not None:
            return self.realtime_generation
        return self.provider == "openai_live"

    @model_validator(mode="after")
    def validate_live_config(self):
        if (self.provider == "openai_live") != (self.live_config is not None):
            raise ValueError(
                "live_config is required for openai_live and only valid for that provider"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_reasoning_effort(cls, data, info: ValidationInfo):
        """Fill `reasoning_effort` from the provider default table.

        A null carries two incompatible meanings and only the CALLER knows
        which: a stored `null` read back off disk is a pre-resolution result
        (every run written before this field became eager), while a Python
        `reasoning_effort=None` on a live construction just means "the CLI
        pinned nothing" — the natural shape when threading an unset flag
        through. Both arrive here as the same dict, so the discriminator is the
        validation CONTEXT the from-disk loaders set (STORED_PAYLOAD_CONTEXT),
        not the shape of the data.

        Either way the value is resolved, so readers see the effective effort.
        Only the stored null is additionally marked `inferred`, so the
        distinction survives a re-save — and a value a live run genuinely
        resolved is never recorded as an after-the-fact inference, which is the
        confusion this field exists to prevent.
        """
        if not isinstance(data, dict):
            return data
        if data.get("reasoning_effort") is not None:
            return data
        provider = data.get("provider") or DEFAULT_AUDIO_NATIVE_PROVIDER
        data = dict(data)
        legacy_null = "reasoning_effort" in data and is_stored_payload(info)
        data["reasoning_effort"] = resolve_audio_native_reasoning_effort(provider, None)
        if legacy_null and "reasoning_effort_source" not in data:
            data["reasoning_effort_source"] = ReasoningEffortSource.INFERRED
        return data

    @model_validator(mode="after")
    def _check_provider_supports_the_effort(self) -> "AudioNativeConfig":
        """Reject an effort the selected provider will not accept.

        The levels are not a shared vocabulary (`xhigh` is OpenAI-only; xai and
        friends take no level at all), and a provider that rejects one does so
        at websocket connect — 40 minutes into a matrix run. Fail here instead.
        """
        resolve_audio_native_reasoning_effort(self.provider, self.reasoning_effort)
        return self

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
    use_llm_backchannel: bool = Field(
        default=DEFAULT_USE_LLM_BACKCHANNEL,
        description="If True, enable backchanneling via the LLM decision prompt. If False, disable backchanneling.",
    )

    # Agent behavior
    use_xml_prompt: bool = Field(
        default=False,
        description="Use XML tags in system prompt. Defaults to False (plain text) for all providers.",
    )
    disclose_voice_gender: bool = Field(
        default=True,
        description=(
            "If True (default), the agent system prompt states the gender of "
            "the provider voice (resolved from "
            "the reviewed provider-voice catalog, tau2.voice.voice_gender — "
            "the same source judging uses). Providers give the model no "
            "signal tying its voice to grammatical self-reference, so "
            "ungrounded agents default masculine in gendered languages; the "
            "disclosure closes that gap (hi ablation: 0 masc-only/30 "
            "disclosed vs 10/10 undisclosed). --no-disclose-voice-gender "
            "runs the ungrounded ablation arm; required for providers whose "
            "default voice has no catalog gender (livekit)."
        ),
    )

    @model_validator(mode="after")
    def _resolve_disclosure_feasibility(self) -> "AudioNativeConfig":
        """Auto-disable disclosure the voice catalog cannot honor.

        With disclosure on by default, a provider whose default voice has no
        catalog gender (livekit) cannot make a truthful disclosure. The field
        flips to False here — so results.info records what the prompt actually
        contained — with a warning rather than an error, keeping default
        construction working for every provider.
        """
        if self.disclose_voice_gender:
            from loguru import logger

            from tau2.voice.voice_gender import resolve_agent_gender

            if resolve_agent_gender(self.provider, None) is None:
                logger.warning(
                    f"disclose_voice_gender: provider {self.provider!r}'s "
                    "default voice has no gender in the provider-voice "
                    "catalog; disclosure disabled for this run"
                )
                self.disclose_voice_gender = False
        return self

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
    def max_steps_ticks(self) -> int:
        """Maximum steps in ticks."""
        return int(self.max_steps_seconds / self.tick_duration_seconds)

    @property
    def default_wallclock_timeout_seconds(self) -> float:
        """Wallclock guard that cannot pre-empt this run's conversation budget.

        ``max_steps_seconds`` is simulated time and so caps every call
        identically; wallclock is latency-dependent and would not. Scaling the
        guard off the budget keeps the simulated cap the only limit that fires
        on a healthy run, whatever the budget is set to.
        """
        return self.max_steps_seconds * VOICE_TIMEOUT_SAFETY_FACTOR

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


class Score(str, Enum):
    """A scoring axis computed for a simulation.

    The axes are decoupled siblings — quality, nativeness, and delivery are never
    folded into reward / pass@1. Any subset can be computed per run; each axis writes
    only its own ``*_info`` field and leaves the others untouched (useful for
    post-hoc re-scoring via ``tau2 evaluate-trajs``). Delivery listens to the
    agent's audio, so it is voice-only: judged inline at run time, or post hoc
    from the stored ``both.wav`` disk audio (full-duplex runs).
    """

    REWARD = "reward"
    QUALITY = "quality"
    NATIVENESS = "nativeness"
    DELIVERY = "delivery"


DEFAULT_SCORES = frozenset({Score.REWARD, Score.QUALITY, Score.NATIVENESS})
DEFAULT_TEXT_SCORES = frozenset({Score.REWARD})


def parse_scores(value: str, *, voice: bool = False) -> set[Score]:
    """Parse a comma-separated ``--scores`` value into a set of ``Score`` axes.

    Accepts the axis names (``reward,quality,nativeness,delivery``) plus ``all``,
    which expands per mode: text → reward + quality + nativeness; voice → all
    four (delivery is voice-only). Raises ValueError on unknown tokens.
    """
    tokens = [t.strip().lower() for t in value.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--scores must name at least one axis")
    scores: set[Score] = set()
    for token in tokens:
        if token == "all":
            scores |= set(DEFAULT_SCORES)
            if voice:
                scores.add(Score.DELIVERY)
            continue
        try:
            scores.add(Score(token))
        except ValueError:
            valid = ", ".join(s.value for s in Score)
            raise ValueError(
                f"unknown score '{token}' (valid: {valid}, or 'all')"
            ) from None
    return scores


class TextNoiseSettings(BaseModel):
    """Settings for noisy-text entity probes (text runs only).

    The text analog of acoustic degradation: deterministic, seeded corruption
    of the user simulator's FIRST conveyance of each pinned identity entity
    (see ``tau2.multilingual.text_noise``). Presence of this settings object
    on a ``TextRunConfig`` arms the probes; None (the default) leaves the run
    untouched.
    """

    seed: Annotated[
        int,
        Field(
            description="Noise seed. Corruptions derive ONLY from (catalog "
            "version, this seed, task id, entity string), so two runs with "
            "the same seed face byte-identical corrupted stimuli (paired "
            "design).",
            default=DEFAULT_TEXT_NOISE_SEED,
        ),
    ]
    operators: Annotated[
        Optional[list[str]],
        Field(
            description="Optional RESTRICTION of the language's closed "
            "operator set (catalog ids from tau2.multilingual.text_noise). "
            "None uses the language's full mapped set; ids outside the "
            "catalog are rejected.",
            default=None,
        ),
    ]

    @field_validator("operators")
    @classmethod
    def _operators_in_catalog(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        from tau2.multilingual.text_noise import get_noise_operator

        for operator_id in v:
            try:
                get_noise_operator(operator_id)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
        return v


class TextNoiseEntity(BaseModel):
    """One pinned entity's corruption: clean truth recorded alongside."""

    kind: Annotated[
        str,
        Field(description="Entity class: email | user_id | date | phone | name."),
    ]
    clean: Annotated[
        str,
        Field(description="The clean ground-truth string from the scenario."),
    ]
    corrupted: Annotated[
        str,
        Field(description="The corrupted form injected on first conveyance."),
    ]
    operator: Annotated[
        str,
        Field(description="Catalog id of the noise operator that produced it."),
    ]


class TextNoiseInfo(BaseModel):
    """Per-simulation record of the entity noise plan (provenance contract).

    Written by the runner from ``tau2.multilingual.text_noise`` plans; enough
    for downstream capture/repair checks without re-deriving anything: the
    catalog version, the seed, and every clean → corrupted pair.
    """

    catalog_version: Annotated[
        str,
        Field(description="TEXT_NOISE_CATALOG_VERSION the plan was built with."),
    ]
    seed: Annotated[int, Field(description="The run's noise seed.")]
    entities: Annotated[
        list[TextNoiseEntity],
        Field(description="Every corruption planned for this task."),
    ]


class NativenessJudgeSettings(BaseModel):
    """Settings for the LLM nativeness judge (subjective per-language factors).

    Deterministic nativeness checkers always run when the nativeness score is
    computed; these settings govern only the judge-type factors.

    The LLM judge is OPT-IN (2026-08-03). It was on by default when nativeness
    was the thing being measured; since the paper's scores come from reward and
    from human preference annotation, the judge's per-simulation cost buys
    nothing on a production run — and every run pool collected so far
    (preference_v1, airline_v1, retail_v1) turns it off explicitly anyway. A
    judge that has to be disabled at every call site is a default pointing the
    wrong way. Turn it on with ``--nativeness-llm-judge`` when the judge is what
    you are studying, e.g. a calibration round.
    """

    llm_judge: Annotated[
        bool,
        Field(
            description="Run the LLM judge for judge-type nativeness factors. "
            "False (the default) => deterministic checkers only; LLM factors "
            "are recorded DEFERRED (excluded from the score).",
            default=False,
        ),
    ]
    model: Annotated[
        str,
        Field(
            description="Model id for the nativeness judge.",
            default=DEFAULT_LLM_NATIVENESS_JUDGE,
        ),
    ]
    model_args: Annotated[
        dict,
        Field(
            description="Extra args forwarded to generate() for each judge call.",
            default_factory=lambda: dict(DEFAULT_LLM_NATIVENESS_JUDGE_ARGS),
        ),
    ]


class QualityJudgeSettings(BaseModel):
    """Settings for the universal call-quality rubric.

    Deterministic high-precision checks always run when quality is selected.
    The task-aware LLM checks are opt-in until their rubric is calibrated.
    """

    llm_judge: Annotated[
        bool,
        Field(
            description="Run task-aware semantic quality factors. False records "
            "those factors as DEFERRED.",
            default=False,
        ),
    ]
    model: Annotated[
        Optional[str],
        Field(
            description="Optional model override for every LLM quality factor. "
            "None uses each factor's catalog model.",
            default=None,
        ),
    ]
    model_args: Annotated[
        dict,
        Field(
            description="Overrides merged onto each factor's catalog judge args.",
            default_factory=lambda: dict(DEFAULT_LLM_QUALITY_JUDGE_ARGS),
        ),
    ]
    response_latency_seconds: Annotated[
        float,
        Field(
            description="A mean response latency above this fails responsiveness.",
            default=DEFAULT_QUALITY_RESPONSE_LATENCY_SECONDS,
            gt=0,
        ),
    ]
    monologue_seconds: Annotated[
        float,
        Field(
            description="An uninterrupted agent floor hold longer than this fails monologue.",
            default=DEFAULT_QUALITY_MONOLOGUE_SECONDS,
            gt=0,
        ),
    ]


class DeliveryJudgeSettings(BaseModel):
    """Settings for the multimodal audio delivery judge (fidelity + intonation).

    Voice-only: the judge listens to the agent's synthesized audio — inline
    after each simulation, or post-hoc from the stored ``both.wav`` agent
    channel (``tau2 judges rejudge --delivery``; full-duplex runs only).
    """

    model: Annotated[
        str,
        Field(
            description="Model id for the audio delivery judge.",
            default=DEFAULT_LLM_DELIVERY_JUDGE,
        ),
    ]
    model_args: Annotated[
        dict,
        Field(
            description="Extra args forwarded to the judge LLM call "
            "(incl. per-request timeout).",
            default_factory=lambda: dict(DEFAULT_LLM_DELIVERY_JUDGE_ARGS),
        ),
    ]
    sample_rate: Annotated[
        float,
        Field(
            description="Fraction of voice conversations to judge (deterministic "
            "per sim; a sampled conversation gets all its agent utterances "
            "judged, others are skipped entirely). 1.0 judges every conversation.",
            default=DEFAULT_DELIVERY_SAMPLE_RATE,
        ),
    ]
    max_segments: Annotated[
        int,
        Field(
            description="Max agent utterances judged per conversation (cost cap).",
            default=DEFAULT_DELIVERY_MAX_SEGMENTS,
        ),
    ]
    concurrency: Annotated[
        int,
        Field(
            description="Concurrent judge calls per simulation.",
            default=DEFAULT_DELIVERY_JUDGE_CONCURRENCY,
        ),
    ]


class BaseRunConfig(BaseModel):
    """Base configuration shared by both text (half-duplex) and voice (full-duplex) modes.

    Do not instantiate directly. Use TextRunConfig or VoiceRunConfig.
    """

    scores: Annotated[
        set[Score],
        Field(
            description="Scoring axes to compute per simulation. Defaults to "
            "reward + quality + nativeness; delivery (voice-only) is opt-in as it adds "
            "real per-utterance judge cost.",
            default_factory=lambda: set(DEFAULT_SCORES),
        ),
    ]
    nativeness_judge: Annotated[
        NativenessJudgeSettings,
        Field(
            description="LLM nativeness-judge settings (on/off, model, args). "
            "Only consulted when 'nativeness' is in scores.",
            default_factory=NativenessJudgeSettings,
        ),
    ]
    quality_judge: Annotated[
        QualityJudgeSettings,
        Field(
            description="Universal quality-rubric settings. Only consulted when "
            "'quality' is in scores.",
            default_factory=QualityJudgeSettings,
        ),
    ]
    user_persona_id: Annotated[
        Optional[str],
        Field(
            description="Force a specific user persona by id (e.g. a language-pack "
            "persona like 'priya_hindi_v1') or a bare language code (e.g. 'hi', "
            "sampling per task among that pack's personas). None keeps the default "
            "per-task persona sampling. Works in both text and voice modes. See "
            "tau2.multilingual.",
            default=None,
        ),
    ]
    communicate_judge_mode: Annotated[
        Literal["auto", "llm", "exact"],
        Field(
            description="How communicate_info checks are scored: 'auto' uses "
            "the semantic LLM judge for non-English and exact matching for "
            "English; 'llm' or 'exact' pins one evaluator for the whole run.",
            default=DEFAULT_COMMUNICATE_JUDGE_MODE,
        ),
    ]

    results_format: Annotated[
        Optional[Literal["json", "dir"]],
        Field(
            description="Checkpoint storage format override. None keeps the "
            "modality default (voice: 'dir' — tick transcripts are too large "
            "for one file; text: 'json'). Text runs that feed the pool census "
            "or the annotation packet builders pin 'dir' so every downstream "
            "reader sees one format.",
            default=None,
        ),
    ]

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
    task_subset: Annotated[
        Optional[str],
        Field(
            description="Fixed task subset to run (see data/tau2/task_subsets/ "
            "and tau2.task_subsets). 'auto' (the default) uses the domain's "
            "canonical subset unless task_ids/num_tasks selects tasks "
            "explicitly; 'all' runs the whole task set; any other value names "
            "a subset artifact.",
            default=SUBSET_AUTO,
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
    redo_stale_tasks: Annotated[
        bool,
        Field(
            description="On resume, treat a task whose text changed since the "
            "checkpoint was written as stale rather than fatal: drop its "
            "recorded simulations and re-run them against the CURRENT text. "
            "Without this, a modified task aborts the resume.",
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

    @model_validator(mode="after")
    def _default_banking_retrieval_config(self) -> "BaseRunConfig":
        """Default retrieval_config to alltools for banking_knowledge."""
        if self.domain == "banking_knowledge" and self.retrieval_config is None:
            object.__setattr__(self, "retrieval_config", "alltools")
        return self

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

    @property
    def llm_communicate_judge_override(self) -> Optional[bool]:
        """Evaluator override consumed by ``evaluate_simulation``."""
        if self.communicate_judge_mode == "auto":
            return None
        return self.communicate_judge_mode == "llm"

    def validate(self) -> None:
        """Validate the run config."""
        pass


class TextRunConfig(BaseRunConfig):
    """Configuration for half-duplex (text) simulations.

    Text mode uses turn-based message exchange between an LLM agent and a
    user simulator, with an Orchestrator managing the conversation.
    """

    scores: Annotated[
        set[Score],
        Field(
            description="Text runs score task reward only by default. Interaction "
            "quality and nativeness are reserved for FCE review unless explicitly requested.",
            default_factory=lambda: set(DEFAULT_TEXT_SCORES),
        ),
    ]
    max_concurrency: Annotated[
        int,
        Field(
            description="Concurrent text simulations held by each worker process.",
            default=DEFAULT_TEXT_MAX_CONCURRENCY,
        ),
    ]
    workers: Annotated[
        int,
        Field(
            description="Worker processes used by the text-run controller.",
            default=DEFAULT_TEXT_WORKERS,
        ),
    ]

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
    text_input_style: Annotated[
        Optional[str],
        Field(
            description="Text input-style arm: how the user simulator TYPES "
            "the target language (closed catalog in "
            "tau2.multilingual.text_input_catalog — native_script, romanized, "
            "diacritic_free, code_mixed). None ≡ native_script (the default "
            "arm, no directive rendered). A non-default style requires a "
            "language-pack run whose pack declares the style; validated at "
            "build time, recorded in results provenance. Stored as the plain "
            "style value (like user_persona_id) so the data model stays "
            "decoupled from the multilingual package.",
            default=None,
        ),
    ]

    text_noise: Annotated[
        Optional[TextNoiseSettings],
        Field(
            description="Noisy-text entity probes: presence arms deterministic "
            "seeded corruption of first entity conveyance (see "
            "tau2.multilingual.text_noise). None (the default) leaves the run "
            "untouched. Recorded in results provenance.",
            default=None,
        ),
    ]

    # ---- Validation ----

    @field_validator("text_input_style")
    @classmethod
    def _text_input_style_in_catalog(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        from tau2.multilingual.text_input_catalog import TextInputStyle

        return TextInputStyle(v).value

    @model_validator(mode="after")
    def _reject_delivery_score(self) -> "TextRunConfig":
        if Score.DELIVERY in self.scores:
            raise ValueError("delivery is judged inline on voice runs only")
        return self

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
    delivery_judge: Annotated[
        DeliveryJudgeSettings,
        Field(
            description="Audio delivery judge settings (model, args, sampling). "
            "Only consulted when 'delivery' is in scores — voice-only, since "
            "the judge listens to audio that is never serialized.",
            default_factory=DeliveryJudgeSettings,
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
        str,
        Field(
            description="Channel-effects intensity overlay on the complexity preset: 'light' (clean line: no frame drops or muffling), 'regular' (preset values), 'heavy' (elevated frame drops in longer bursts, more muffling)",
            default="regular",
        ),
    ]
    speech_effects_mode: Annotated[
        str,
        Field(
            description="Speech-effects intensity overlay on the complexity preset: 'light' (patient caller: no interruptions, backchannels, or inserts), 'regular' (preset values), 'heavy' (chatty interruptive caller: 2x insert rate)",
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


class JudgeOutcome(str, Enum):
    """Outcome of one judged check (nativeness factor / delivery utterance)."""

    PASS = "pass"
    FAIL = "fail"
    NO_OPPORTUNITY = "no_opportunity"  # check had no chance to fire this run
    DEFERRED = "deferred"  # checker not run (e.g. LLM judge disabled)
    ERROR = "error"  # checker raised (e.g. judge API/parse failure); see evidence


class QualityFactorCheck(BaseModel):
    """Result of one universal quality-rubric factor."""

    id: Annotated[str, Field(description="Stable factor identifier.")]
    category: Annotated[str, Field(description="Rubric category.")]
    severity: Annotated[int, Field(description="Factor weight.", ge=1, le=3)]
    evaluator: Annotated[
        Literal["deterministic", "llm", "hybrid", "audio"],
        Field(description="Mechanism that produced the verdict."),
    ]
    outcome: JudgeOutcome
    evidence: Optional[str] = Field(
        description="Failure explanation or ERROR detail.", default=None
    )
    quote: Optional[str] = Field(
        description="Exact offending transcript span when available.", default=None
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Raw deterministic measurements used by this verdict.",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Thresholds used by this verdict.",
    )
    judge_model: Optional[str] = Field(
        default=None, description="Model used for this factor's LLM call."
    )
    judge_args: Optional[dict] = Field(
        default=None, description="Arguments used for this factor's LLM call."
    )
    judge_prompt_version: Optional[str] = Field(
        default=None, description="Versioned prompt used for this factor."
    )


class QualityInfo(BaseModel):
    """Per-simulation universal call-quality score, separate from reward."""

    score: Optional[float] = Field(
        description="Equal-weight pass fraction over binary factors that fired.",
        default=None,
    )
    factor_checks: list[QualityFactorCheck] = Field(
        default_factory=list, description="One binary verdict per universal factor."
    )
    rubric_version: Annotated[str, Field(description="Versioned factor catalog.")]
    metrics_version: Annotated[
        str, Field(description="Deterministic extractor version.")
    ]
    num_pass: int = Field(default=0, description="Factors answered PASS.")
    num_fail: int = Field(default=0, description="Factors answered FAIL.")
    num_no_opportunity: int = Field(
        default=0, description="Factors with no opportunity in this conversation."
    )
    num_deferred: int = Field(
        default=0, description="Factors configured but not evaluated."
    )
    num_errors: int = Field(default=0, description="Factors that failed to evaluate.")
    score_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of rubric checks contributing PASS or FAIL to score.",
    )


class NativenessJudgeUnitResult(BaseModel):
    """One typed LLM verdict before factor results are aggregated to a call."""

    unit_index: Optional[int] = Field(
        default=None,
        description="Zero-based agent utterance index; None for a call-level unit.",
    )
    opportunity: bool = Field(description="Whether the criterion applied in this unit.")
    violated: bool = Field(description="Whether this unit violated the criterion.")
    severity: int = Field(
        ge=0,
        le=4,
        description="Observed violation severity; zero when there is no violation.",
    )
    reasoning: str = Field(default="", description="Brief judge explanation.")
    quote: str = Field(default="", description="Exact offending agent span, if any.")

    @model_validator(mode="after")
    def _valid_verdict_state(self) -> "NativenessJudgeUnitResult":
        if not self.opportunity and self.violated:
            raise ValueError("a no-opportunity result cannot be a violation")
        if self.violated and self.severity == 0:
            raise ValueError("a violation needs severity 1..4")
        if not self.violated and self.severity != 0:
            raise ValueError("severity must be zero when there is no violation")
        return self


class NativenessFactorCheck(BaseModel):
    """Result of evaluating one nativeness factor against a simulation."""

    id: str = Field(description="Stable factor identifier.")
    category: str = Field(description="Nativeness category.")
    severity: int = Field(description="Legacy impact metadata.", ge=1, le=3)
    outcome: JudgeOutcome = Field(description="Binary verdict or non-scoring state.")
    evidence: Optional[str] = Field(
        description="Deterministic measurement details or judge explanation.",
        default=None,
    )
    quote: Optional[str] = Field(
        description="Exact offending phrase the judge flagged (verbatim from the "
        "transcript), when the FAIL localizes to one span; None otherwise.",
        default=None,
    )
    shadow: bool = Field(
        default=False,
        description="Calibration mode: the check was judged and recorded but "
        "excluded from the aggregate score, coverage, and outcome counts.",
    )
    evaluation_level: Literal["deterministic", "call", "utterance"] = Field(
        default="deterministic",
        description="Unit on which the factor was evaluated.",
    )
    observed_severity: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Aggregated observed violation severity; zero unless FAIL.",
    )
    violation_count: int = Field(
        default=0,
        ge=0,
        description="Number of violated utterance units behind this call "
        "verdict. Each unit is violated or not (one violation per utterance "
        "at most); call-level and deterministic factors record 0 or 1.",
    )
    unit_results: list[NativenessJudgeUnitResult] = Field(
        default_factory=list,
        description="Raw typed LLM unit verdicts used to produce this call verdict.",
    )


class NativenessInfo(BaseModel):
    """Per-simulation nativeness score — a decoupled axis, never folded into reward.

    Populated for supported-language runs, including English for deterministic
    interaction factors. ``score``
    is None when no factor had an opportunity to fire (nothing to measure), which
    is distinct from a score of 0.0 (factors fired and all failed). The score is
    the equal-weight fraction of fired binary factors that passed, in [0, 1].
    """

    score: Optional[float] = Field(
        description="Equal-weight pass fraction over binary factors that fired; "
        "None when no factor had an opportunity.",
        default=None,
    )
    factor_checks: list[NativenessFactorCheck] = Field(
        default_factory=list, description="One binary verdict per enabled factor."
    )
    language: Optional[str] = Field(default=None, description="ISO language code.")
    script: Optional[str] = Field(
        default=None, description="ISO script code, if known."
    )
    rubric_version: Optional[str] = Field(
        default=None,
        description="Version of the binary nativeness rubric and aggregation contract.",
    )
    num_pass: int = Field(default=0, description="Factors answered PASS.")
    num_fail: int = Field(default=0, description="Factors answered FAIL.")
    num_no_opportunity: int = Field(
        default=0, description="Factors with no opportunity in this conversation."
    )
    num_deferred: int = Field(
        default=0, description="Factors configured but not evaluated."
    )
    num_errors: Annotated[
        int,
        Field(
            description="Factor checks that ERRORed (e.g. judge API/parse "
            "failure); excluded from the score but still counted.",
            default=0,
        ),
    ]
    score_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of enabled checks contributing PASS or FAIL to score.",
    )
    # Judge configuration that produced the LLM-judged factor verdicts —
    # recorded for reproducibility (None when the LLM judge did not run).
    judge_model: Annotated[
        Optional[str],
        Field(
            description="Model id that produced the LLM-judge factor verdicts; "
            "None when only deterministic checkers ran.",
            default=None,
        ),
    ]
    judge_args: Annotated[
        Optional[dict],
        Field(
            description="Model args used for the LLM-judge calls; None when "
            "only deterministic checkers ran.",
            default=None,
        ),
    ]
    judge_prompt_version: Annotated[
        Optional[str],
        Field(
            description="Version tag of the nativeness-judge system prompt "
            "that produced the LLM-judge verdicts; None when the judge made "
            "no LLM call.",
            default=None,
        ),
    ]


DeliveryAxis = Literal["fidelity", "intonation"]


class DeliveryFinding(BaseModel):
    """One issue the audio delivery judge flagged in a single agent utterance.

    ``axis`` tags whether it is a fidelity (did it say the right words) or an
    intonation (did it sound natural) problem, so the combined judge's output can
    be split back into per-axis sub-scores.
    """

    axis: DeliveryAxis
    category: str = Field(
        description="Judge finding category, e.g. 'mispronunciation'."
    )
    time_range: Optional[str] = Field(
        description="Approx. time span of the issue in the clip, if given.",
        default=None,
    )
    issue: Optional[str] = Field(
        description="Description of the problem (PII-redacted by the judge).",
        default=None,
    )
    severity: int = Field(description="1 (minor) .. 3 (critical).", ge=1, le=3)
    confidence: Optional[float] = None


class DeliveryFactorCheck(BaseModel):
    """Result of one language-specific delivery factor against one clip / sim.

    Mirrors ``NativenessFactorCheck`` field-for-field (id / category / severity /
    outcome / evidence / quote) so delivery factor verdicts flow into the same
    calibration and annotation machinery. Only produced in pack-rubric mode
    (``DeliveryInfo.rubric_source == "pack"``).
    """

    id: str
    category: str
    severity: int
    outcome: JudgeOutcome
    evidence: Optional[str] = Field(
        description="Judge reasoning / explanation when the outcome is FAIL.",
        default=None,
    )
    quote: Optional[str] = Field(
        description="Exact offending span from the reference transcript, when "
        "the FAIL localizes to one (PII described by class, never copied); "
        "None otherwise.",
        default=None,
    )
    shadow: bool = Field(
        default=False,
        description="Calibration mode: recorded but excluded from factor_score.",
    )


class DeliveryUtteranceResult(BaseModel):
    """Result of the combined audio delivery judge for one agent utterance."""

    utterance_idx: int
    was_interrupted: bool = Field(
        description="True when the utterance was cut short by the caller "
        "(inactive trailing chunks, or tick-level barge-in detection); the "
        "judge is told not to penalize the missing tail.",
        default=False,
    )
    expected_text: Optional[str] = Field(
        description="Excerpt of the intended synthesis text (audio_script_gold).",
        default=None,
    )
    outcome: JudgeOutcome = Field(
        description="PASS (no issue), FAIL (issue found), NO_OPPORTUNITY (no audio), "
        "or ERROR (judge raised; see summary). FAIL <=> findings is non-empty.",
    )
    flag_for_review: bool = Field(
        description="True when any finding is severity >= 2.", default=False
    )
    severity: int = Field(description="Overall 0 (clean) .. 3 (critical).", default=0)
    confidence: Optional[float] = None
    summary: Optional[str] = None
    findings: list[DeliveryFinding] = Field(default_factory=list)
    factor_checks: list[DeliveryFactorCheck] = Field(
        default_factory=list,
        description="Per-factor verdicts for this clip against the language's "
        "pack delivery rubric; empty in fallback/generic mode.",
    )


class DeliveryInfo(BaseModel):
    """Per-simulation audio delivery scores (fidelity + intonation).

    A decoupled perceptual-quality axis, never folded into reward. Only populated
    for voice runs (None otherwise). Sub-scores are the severity-weighted fraction
    of judged utterances that were clean on that axis, in [0, 1]; each is None when
    no utterance was judged (nothing to measure), distinct from 0.0.
    """

    score: Optional[float] = Field(
        description="Overall clean fraction across both axes; None if nothing judged.",
        default=None,
    )
    fidelity_score: Optional[float] = Field(
        description="Severity-weighted clean fraction on the fidelity axis.",
        default=None,
    )
    intonation_score: Optional[float] = Field(
        description="Severity-weighted clean fraction on the intonation axis.",
        default=None,
    )
    num_judged: int = Field(
        description="Agent utterances actually sent to the judge, including "
        "calls that failed (ERROR outcome) — those still incur cost.",
        default=0,
    )
    num_errors: int = Field(
        description="Judge calls that failed (ERROR outcome); excluded from "
        "the scores but included in num_judged.",
        default=0,
    )
    num_flagged: int = Field(
        description="Utterances with a severity-2/3 finding (flag_for_review).",
        default=0,
    )
    utterance_results: list[DeliveryUtteranceResult] = Field(default_factory=list)
    factor_score: Optional[float] = Field(
        description="Equal-weight pass fraction over binary language-specific "
        "delivery factors that fired (pack-rubric mode only); None when no "
        "factor fired or the language has no pack delivery factors.",
        default=None,
    )
    factor_checks: list[DeliveryFactorCheck] = Field(
        default_factory=list,
        description="Sim-level aggregation of the per-utterance factor verdicts "
        "(one check per pack delivery factor: FAIL if any judged utterance "
        "failed it, else PASS if any passed, else ERROR/NO_OPPORTUNITY).",
    )
    rubric_source: Optional[Literal["pack", "fallback"]] = Field(
        description="Where the judge's language-specific criteria came from: "
        "'pack' (the language pack's delivery factors), 'fallback' (the fixed "
        "native-listener prompt for the language), or None (no language / "
        "judge never ran).",
        default=None,
    )
    language: Optional[str] = Field(default=None, description="ISO language code.")
    locale: Optional[str] = Field(
        default=None,
        description="Configured regional locale supplied to the audio judge.",
    )
    # Judge configuration that produced these scores — recorded for
    # reproducibility, since sampled scores are not comparable across different
    # models/rates and audio is not serialized (no re-judging from results.json).
    judge_model: Optional[str] = None
    judge_args: Annotated[
        Optional[dict],
        Field(
            description="Model args used for the delivery judge calls.",
            default=None,
        ),
    ]
    judge_prompt_version: Annotated[
        Optional[str],
        Field(
            description="Version tag of the delivery-judge prompt that "
            "produced these verdicts; None when no LLM call was made.",
            default=None,
        ),
    ]
    sample_rate: Optional[float] = None
    max_segments: Optional[int] = None
    seed: Optional[int] = None


def perceptual_quality_score(
    sim: "SimulationRun",
    *,
    weights: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """Composite "perceptual quality" over the components that are present.

    Rolls up nativeness + delivery (fidelity, intonation) into one weighted mean,
    skipping components that are None (e.g. a run with no eligible nativeness
    opportunity, or a non-voice run with no delivery). Returns None when no
    component is present.

    NOTE: intentionally NOT wired into the default reporting in v1 — an equal-weight
    mean over heterogeneous, sometimes-missing components is not comparable across
    languages/modes yet. Kept (and unit-tested) so the umbrella can be turned on
    later without a redesign; ``weights`` allows tuning per-component weighting.
    """
    components: dict[str, Optional[float]] = {
        "nativeness": sim.nativeness_info.score if sim.nativeness_info else None,
        "fidelity": sim.delivery_info.fidelity_score if sim.delivery_info else None,
        "intonation": sim.delivery_info.intonation_score if sim.delivery_info else None,
    }
    w = weights or {"nativeness": 1.0, "fidelity": 1.0, "intonation": 1.0}
    num = 0.0
    denom = 0.0
    for name, value in components.items():
        if value is None:
            continue
        weight = w.get(name, 1.0)
        num += weight * value
        denom += weight
    return num / denom if denom > 0 else None


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
    check_error: Optional[str] = Field(
        description="Error message if the check itself failed to run (e.g. reviewer "
        "LLM provider error). When set, the conversation was NOT checked and "
        "hallucination_found is not meaningful.",
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


class TaskSubsetInfo(BaseModel):
    """Which fixed subset a run scored on, recorded in its results.

    Enough to tell two runs apart without loading the artifact: a run on
    telecom_50 and a run on telecom_50_v2 are not comparable, and neither is a
    run on telecom_50 before and after a frame change.
    """

    name: str = Field(description="Subset name (data/tau2/task_subsets/<name>.json).")
    size: int = Field(description="Number of tasks the subset pins.")
    tasks_scored: Optional[int] = Field(
        default=None,
        description="Tasks this run actually scored. Below `size` when the "
        "agent's registered task filter dropped some (llm_agent_gt and "
        "llm_agent_solo need ground-truth actions, which not every task has), "
        "so the run covers a sub-part of the subset — recorded rather than "
        "left to be inferred from the simulation count, which counts trials.",
    )
    frame_task_set: str = Field(description="Task set the subset was drawn from.")
    frame_size: int = Field(description="Size of that frame.")
    frame_digest: str = Field(description="sha256 of the frame's ids at draw time.")
    strategy: str = Field(description="census or stratified.")
    seed: int = Field(description="Seed of the within-stratum draw.")


class Info(BaseModel):
    """Information about the simulator."""

    git_commit: str = Field(description="The git commit hash.")
    num_trials: int = Field(description="The number of trials.")
    max_steps: int = Field(description="The maximum number of steps.")
    max_errors: int = Field(description="The maximum number of errors.")
    user_info: UserInfo = Field(description="User information.")
    agent_info: AgentInfo = Field(description="Agent information.")
    environment_info: EnvironmentInfo = Field(description="Environment information.")
    task_set_name: Optional[str] = Field(
        description="Registered task set the tasks were loaded from (e.g. a "
        "localized set like 'airline_hi'). None means the domain's default "
        "task set, or a results file predating this field.",
        default=None,
    )
    user_persona_id: Optional[str] = Field(
        description="The run's --user-persona-id: a language-pack persona id, "
        "a bare language code (per-task assignment among that pack's "
        "personas), or None for the stock English persona sampling. Recorded "
        "so a results directory states which caller it was configured to "
        "speak as — a resume that drops it changes what the calls ARE, and "
        "without this field that change is invisible to the resume config "
        "check. None also means a results file predating this field.",
        default=None,
    )
    text_input_style: Optional[str] = Field(
        description="The run's --text-input-style arm (closed catalog in "
        "tau2.multilingual.text_input_catalog): how the user simulator TYPED "
        "the target language. None means native_script (the default arm) or "
        "a results file predating this field. Recorded because a resume that "
        "drops it changes what the chats ARE.",
        default=None,
    )
    text_noise: Optional[TextNoiseSettings] = Field(
        description="The run's noisy-text probe settings (seed, operator "
        "restriction). None means the run was not noise-armed, or a results "
        "file predating this field. Per-simulation corruption plans live on "
        "each SimulationRun.text_noise.",
        default=None,
    )
    communicate_judge_mode: Optional[Literal["auto", "llm", "exact"]] = Field(
        description="The run's communicate_info evaluator mode. None means the "
        "result predates this field.",
        default=None,
    )
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
    channel_effects_mode: Optional[str] = Field(
        description="Channel-effects intensity overlay applied over the "
        "complexity preset (light/regular/heavy). None means a text run or "
        "a results file predating this field.",
        default=None,
    )
    speech_effects_mode: Optional[str] = Field(
        description="Speech-effects intensity overlay applied over the "
        "complexity preset (light/regular/heavy). None means a text run or "
        "a results file predating this field.",
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
    task_subset: Optional["TaskSubsetInfo"] = Field(
        description="The fixed task subset the run scored on, if any. None "
        "means the run took the whole task set (or a --num-tasks prefix of "
        "it, which is what every result predating the subsets did).",
        default=None,
    )

    # ---- Run knobs recorded so the config can be rebuilt from the result ----
    # Every field below is Optional with a None default meaning "this results
    # file predates the field", not "the run set it to nothing". `tau2 run
    # refill` reconstructs a RunConfig from this Info; a knob that is absent
    # falls back to the repo default and is REPORTED as unrecorded, because a
    # silent fallback is exactly the failure this record exists to prevent.
    task_split_name: Optional[str] = Field(
        description="The run's --task-split-name (the split the tasks were "
        "loaded from, normally 'base').",
        default=None,
    )
    timeout: Optional[float] = Field(
        description="Per-simulation wallclock guard in seconds, as resolved "
        "(a voice run scales it off its conversation budget). Recorded "
        "because it decides whether a slow call ends as a TIMEOUT. 0 is the "
        "`--timeout 0` opt-out — the run had no guard — written explicitly so "
        "it is not confused with None, which means a results file predating "
        "this field.",
        default=None,
    )
    verbose_logs: Optional[bool] = Field(
        description="Whether the run saved per-task logs and LLM call logs "
        "under the run directory's artifacts/ tree.",
        default=None,
    )
    scores: Optional[list[Score]] = Field(
        description="Scoring axes computed inline for each simulation, sorted "
        "by name. An axis absent here was never computed at run time (it may "
        "still have been added post hoc by `tau2 judges`).",
        default=None,
    )
    nativeness_judge: Optional[NativenessJudgeSettings] = Field(
        description="Nativeness LLM-judge settings the run used. Its "
        "`llm_judge` flag is what separates a run whose judge factors are "
        "real verdicts from one where they are all DEFERRED.",
        default=None,
    )
    quality_judge: Optional[QualityJudgeSettings] = Field(
        description="Quality-rubric settings used by the run.", default=None
    )
    delivery_judge: Optional[DeliveryJudgeSettings] = Field(
        description="Audio delivery judge settings (voice runs only; None for "
        "text runs and for results predating this field).",
        default=None,
    )
    hallucination_retries: Optional[int] = Field(
        description="Retries allowed when the user simulator is caught "
        "hallucinating; each one re-runs the simulation with a new seed, so "
        "it changes which simulation the directory ends up holding.",
        default=None,
    )
    enforce_communication_protocol: Optional[bool] = Field(
        description="Text runs only: whether the orchestrator enforced the "
        "communication protocol. None for voice runs.",
        default=None,
    )
    auto_review: Optional[bool] = Field(
        description="Whether each simulation was reviewed by an LLM inline "
        "(populating `review` / `user_only_review` on the simulation).",
        default=None,
    )
    review_mode: Optional[Literal["full", "user"]] = Field(
        description="Which inline review ran when auto_review was on.",
        default=None,
    )
    review_model: Optional[str] = Field(
        description="Model that produced the inline reviews.",
        default=None,
    )


def resolve_tick_duration_seconds(ticks: list[Tick], info: Optional[Info]) -> float:
    """The tick clock's step, from the ticks themselves, then the run config,
    then the repo default."""
    for tick in ticks:
        if tick.tick_duration_seconds:
            return tick.tick_duration_seconds
    audio_cfg = info.audio_native_config if info is not None else None
    configured = audio_cfg.tick_duration_seconds if audio_cfg is not None else None
    return configured or DEFAULT_TICK_DURATION_SECONDS


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
    agent_usage: Optional[SessionUsage] = Field(
        description="Aggregated provider usage (and cost breakdown) for the "
        "agent side. Populated for audio-native full-duplex runs.",
        default=None,
    )
    reward_info: Optional[RewardInfo] = Field(
        description="The reward received by the agent.", default=None
    )
    nativeness_info: Optional[NativenessInfo] = Field(
        description="Deterministic/LLM nativeness and interaction score for a "
        "resolved run language (decoupled from reward; None without a language).",
        default=None,
    )
    quality_info: Optional[QualityInfo] = Field(
        description="Universal binary-rubric quality score, decoupled from reward.",
        default=None,
    )
    delivery_info: Optional[DeliveryInfo] = Field(
        description="Audio delivery scores (fidelity + intonation) from the "
        "multimodal judge; a perceptual-quality axis decoupled from reward. Only "
        "populated for voice runs scored with the 'delivery' axis; None otherwise.",
        default=None,
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
    text_noise: Optional[TextNoiseInfo] = Field(
        description="The entity noise plan this text simulation ran under "
        "(catalog version, seed, every clean → corrupted pair). None for "
        "un-armed runs and voice runs.",
        default=None,
    )
    agent_provider: Optional[str] = Field(
        description="Audio-native provider the agent spoke with (e.g. 'openai'). "
        "None for text/half-duplex runs.",
        default=None,
    )
    agent_voice: Optional[str] = Field(
        description="The provider voice the agent spoke with, when the provider "
        "exposes it. None means the provider default applies (see "
        "tau2.voice.voice_gender). When that reviewed catalog knows the voice's "
        "perceived gender, gender-sensitive nativeness factors receive it.",
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
    quality: float | None = None
    nativeness: float | None = None
    fidelity: float | None = None
    intonation: float | None = None
    termination_reason: str | None = None
    agent_cost: float | None = None
    duration: float | None = None


class SupersededInfo(BaseModel):
    """A run header displaced by a later invocation of the same directory.

    Written when a resume proceeds under a changed run config (or when a
    repair verb rewrites a header field): the surviving simulations were
    collected under THIS header, the header that replaced it describes the
    invocation that finished the directory. Without this record a cell whose
    ceiling was widened mid-collection claims all its simulations ran under
    whichever config wrote results.json first.
    """

    superseded_at: str = Field(
        description="When a later invocation displaced this header.",
        default_factory=get_now,
    )
    info: Info = Field(description="The displaced run header.")


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
    info_history: list[SupersededInfo] = Field(
        description="Run headers of earlier invocations that some of this "
        "directory's simulations were collected under, oldest first. Appended "
        "when a resume proceeds under a changed run config, so `info` always "
        "describes the invocation that most recently produced simulations "
        "while the regimes it displaced stay on record. Empty for a directory "
        "collected under one config (and for results predating this field).",
        default_factory=list,
    )
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
    def detect_format(path: Path) -> Literal["json", "dir"]:
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

    @staticmethod
    def index_entry(sim: SimulationRun) -> SimulationIndexEntry:
        """The simulation-index row for one sim (id + headline scores)."""
        return SimulationIndexEntry(
            id=sim.id,
            task_id=sim.task_id,
            trial=sim.trial,
            reward=sim.reward_info.reward if sim.reward_info else None,
            quality=sim.quality_info.score if sim.quality_info else None,
            nativeness=sim.nativeness_info.score if sim.nativeness_info else None,
            fidelity=sim.delivery_info.fidelity_score if sim.delivery_info else None,
            intonation=(
                sim.delivery_info.intonation_score if sim.delivery_info else None
            ),
            termination_reason=sim.termination_reason,
            agent_cost=sim.agent_cost,
            duration=sim.duration,
        )

    def _build_simulation_index(self) -> list[SimulationIndexEntry]:
        """Build a simulation index from the current simulations list."""
        return [self.index_entry(sim) for sim in self.simulations]

    # ---- Load / Save ----

    @classmethod
    def load(cls, path: Path) -> "Results":
        """Load Results from disk, auto-detecting format.

        Supports both monolithic JSON and directory-based formats.
        For directory format, loads results.json metadata and all individual
        simulation files from the simulations/ subdirectory.
        """
        path = Path(path)
        fmt = cls.detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                return cls.model_validate_json(f.read(), context=STORED_PAYLOAD_CONTEXT)

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
        return cls.model_validate(meta, context=STORED_PAYLOAD_CONTEXT)

    def save(self, path: Path, format: Optional[Literal["json", "dir"]] = None) -> None:
        """Save the results to disk.

        Args:
            path: File path (for "json") or directory/file path (for "dir").
                  For "dir" format, if path ends in .json, the simulations/
                  subdirectory is created alongside it. If path is a directory,
                  results.json is created inside it.
            format: Storage format. "json" writes a single monolithic JSON file.
                    "dir" writes metadata to results.json and each
                    simulation to a separate file in simulations/.
                    None (default) auto-detects from ``path`` — an existing
                    directory or a results.json with a sibling simulations/
                    dir round-trips as "dir"; anything else as "json" — so
                    saving over a loaded results location never silently
                    switches format (e.g. writing a monolithic results.json
                    that a subsequent load would ignore in favor of
                    simulations/*.json).
        """
        path = Path(path)
        if format is None:
            format = self.detect_format(path)
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
        fmt = cls.detect_format(path)
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
        return cls.model_validate(data, context=STORED_PAYLOAD_CONTEXT)

    @classmethod
    def iter_simulations(cls, path: Path) -> Iterator[SimulationRun]:
        """Yield simulations one at a time without loading all into memory.

        For directory format, reads each simulation file individually —
        peak memory is bounded by a single simulation.
        For JSON format, parses the full file but yields SimulationRun models
        one at a time to avoid constructing all at once.
        """
        path = Path(path)
        fmt = cls.detect_format(path)

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
            "quality": sim.quality_info.score if sim.quality_info else None,
            "nativeness": sim.nativeness_info.score if sim.nativeness_info else None,
            "fidelity": sim.delivery_info.fidelity_score if sim.delivery_info else None,
            "intonation": (
                sim.delivery_info.intonation_score if sim.delivery_info else None
            ),
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
