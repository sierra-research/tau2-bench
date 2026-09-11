# Copyright Sierra
"""Typed contracts for the LLM conversation-judge suite.

LLM-judged computation lives under ``tau2 judges``: this package owns the
EVA-ported conversation-progression and conciseness judges. The HEADLINE of
its artifact is per-dimension progression verdicts and per-turn conciseness
failure modes — the composites (the EVA-X conjunction with its thresholds,
and the τ quality composite) are demoted to a permanent ``shadow_scores``
section: still computed and version-stamped for comparability, labeled
``uncalibrated-shadow``, never ranked or headlined (binding decision,
2026-08-18; the progression composite is floor-saturated on the paper
corpus, means 0.00-0.14, which is exactly why the dimensions are the
headline). The deterministic interaction facts live in
``tau2 metrics interaction-facts``.
"""

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.config import (
    DEFAULT_CONVERSATION_JUDGE_CONCURRENCY,
    DEFAULT_EVA_X_JUDGE,
    DEFAULT_EVA_X_JUDGE_ARGS,
)
from tau2.data_model.simulation import QualityInfo
from tau2.metrics.turn_taking import (
    EVA_X_ADAPTATION_VERSION,
    EVA_X_SOURCE_VERSION,
    UNCALIBRATED_SHADOW_LABEL,
    EvaTurnTakingMetric,
)

CONVERSATION_ARTIFACT_VERSION = "conversation-judge-v2"
# v2: interruption-aware — barged-in agent turns carry "[cut off by caller]"
# and both prompts forbid reading truncation artifacts as defects; flagged
# progression dimensions and sub-3 conciseness ratings must carry evidence.
# v3: prompt text unchanged; interruption detection unified on the
# tick-overlap barge-in detector (union with the chunk/gold signal), so more
# truly cut-off turns carry the marker in the transcript.
EVA_X_SEMANTIC_PROMPT_VERSION = "eva-x-semantic-v3"
EVA_X_MULTILINGUAL_POLICY_VERSION = "eva-x-multilingual-v1"
EVA_X_PROGRESSION_THRESHOLD = 0.5
EVA_X_CONCISENESS_THRESHOLD = 0.5


class EvaConcisenessFailureMode(str, Enum):
    """EVA conciseness failure modes for one assistant turn."""

    VERBOSITY_OR_FILLER = "verbosity_or_filler"
    EXCESS_INFORMATION_DENSITY = "excess_information_density"
    OVER_ENUMERATION_OR_LIST_EXHAUSTION = "over_enumeration_or_list_exhaustion"
    CONTEXTUALLY_DISPROPORTIONATE_DETAIL = "contextually_disproportionate_detail"


class EvaConcisenessTurnResult(BaseModel):
    """One assistant turn's EVA conciseness judgment."""

    turn_id: Annotated[
        int, Field(ge=0, description="Zero-based spoken agent-turn identifier.")
    ]
    rating: Annotated[
        int, Field(ge=1, le=3, description="EVA conciseness rating from 1 to 3.")
    ]
    failure_modes: list[EvaConcisenessFailureMode] = Field(
        default_factory=list,
        description="Applicable closed-set conciseness failure modes.",
    )
    evidence: str = Field(
        default="", description="Brief transcript-grounded rationale."
    )

    @model_validator(mode="after")
    def _sub_perfect_rating_requires_evidence(self) -> "EvaConcisenessTurnResult":
        # A conciseness deduction without evidence is unadjudicable: 27-37%
        # of semantic verdicts were evidence-free before this was required.
        if self.rating < 3 and not self.evidence.strip():
            raise ValueError("a conciseness rating below 3 requires non-empty evidence")
        return self


class EvaConcisenessMetric(BaseModel):
    """Call-level mean of normalized per-turn EVA conciseness ratings."""

    score: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Mean normalized turn rating; None when the call has no "
        "spoken agent turns (no evidence is never a fail).",
    )
    turns: list[EvaConcisenessTurnResult] = Field(
        description="Per-agent-turn conciseness judgments; the evidence count."
    )


class EvaProgressionDimensionName(str, Enum):
    """Closed EVA conversation-progression dimensions."""

    UNNECESSARY_TOOL_CALLS = "unnecessary_tool_calls"
    INFORMATION_LOSS = "information_loss"
    REDUNDANT_STATEMENTS = "redundant_statements"
    QUESTION_QUALITY = "question_quality"


class EvaProgressionDimension(BaseModel):
    """One call-level EVA progression dimension."""

    name: EvaProgressionDimensionName = Field(
        description="Closed conversation-progression dimension."
    )
    flagged: bool = Field(description="Whether this progression defect occurred.")
    rating: Annotated[
        int, Field(ge=1, le=3, description="Dimension severity rating from 1 to 3.")
    ]
    evidence: str = Field(
        default="", description="Brief trajectory-grounded rationale."
    )

    @model_validator(mode="after")
    def _flag_and_rating_agree(self) -> "EvaProgressionDimension":
        if not self.flagged and self.rating != 3:
            raise ValueError("an unflagged progression dimension must have rating=3")
        if self.flagged and self.rating == 3:
            raise ValueError(
                "a flagged progression dimension must have rating 1 or 2; "
                "contradictory judge output is a validation error, not a "
                "silent downgrade"
            )
        if self.flagged and not self.evidence.strip():
            raise ValueError(
                "a flagged progression dimension requires non-empty evidence; "
                "an unexplained flag is unadjudicable"
            )
        return self


class EvaProgressionMetric(BaseModel):
    """EVA's rule-derived conversation progression rating and score.

    The composite ``rating``/``score`` is SHADOW material (floor-saturated on
    the corpus); the ``dimensions`` are the headline.
    """

    rating: Annotated[
        int, Field(ge=1, le=3, description="Rule-derived overall EVA rating.")
    ]
    score: Annotated[
        float, Field(ge=0, le=1, description="Normalized overall EVA rating.")
    ]
    dimensions: list[EvaProgressionDimension] = Field(
        description="Every closed conversation-progression dimension."
    )


class EvaXScore(BaseModel):
    """The conjunctive EVA-X call score (SHADOW: uncalibrated, never headlined)."""

    passed: Optional[bool] = Field(
        default=None,
        description="Conjunction of the EVA-X components that produced "
        "evidence; components with a None score are excluded, and the "
        "conjunction is None only when every component is N/A.",
    )
    turn_taking: EvaTurnTakingMetric = Field(
        description="Event-routed deterministic turn-taking component."
    )
    conversation_progression: EvaProgressionMetric = Field(
        description="Call-level semantic progression component."
    )
    conciseness: EvaConcisenessMetric = Field(
        description="Per-turn spoken-load component."
    )
    adaptation_version: str = Field(
        default=EVA_X_ADAPTATION_VERSION,
        description="Version of the complete tau EVA-X adaptation.",
    )
    source_version: str = Field(
        default=EVA_X_SOURCE_VERSION, description="Upstream EVA contract version."
    )
    semantic_prompt_version: str = Field(
        default=EVA_X_SEMANTIC_PROMPT_VERSION,
        description="Version of the semantic judge prompts.",
    )
    multilingual_policy_version: str = Field(
        default=EVA_X_MULTILINGUAL_POLICY_VERSION,
        description="Version of the multilingual interpretation policy.",
    )
    progression_threshold: float = Field(
        default=EVA_X_PROGRESSION_THRESHOLD,
        description="Applied conversation-progression pass threshold.",
    )
    conciseness_threshold: float = Field(
        default=EVA_X_CONCISENESS_THRESHOLD,
        description="Applied conciseness pass threshold.",
    )
    judge_model: str = Field(description="Model used for semantic components.")
    judge_args: dict = Field(description="Effective semantic judge arguments.")
    adaptation_notes: list[str] = Field(
        default_factory=lambda: [
            "EVA v0.2 score curves and conjunction are preserved.",
            "τ tick events route ordinary, agent-interruption, caller-interruption, "
            "dual-interruption, and missed-response turns.",
            "EVA audit-log fallback nudges and pre-tool-speech diagnostics are "
            "unavailable in τ trajectories and are not synthesized.",
            "Semantic judging uses explicit multilingual proportionality rules.",
        ],
        description="Declared differences from stock EVA and unavailable signals.",
    )


class ConversationShadowScores(BaseModel):
    """The demoted composites: structurally separate from the headline."""

    label: Literal["uncalibrated-shadow"] = Field(
        default=UNCALIBRATED_SHADOW_LABEL,
        description="Fixed shadow marker: computed for comparability only.",
    )
    eva_x: Optional[EvaXScore] = Field(
        default=None,
        description="The complete versioned EVA-X adaptation (conjunctive "
        "gate, thresholds, embedded turn-taking); None when the suite "
        "errored or was disabled.",
    )
    ours: Optional[QualityInfo] = Field(
        default=None,
        description="The τ universal quality composite; None when disabled or errored.",
    )


class ConversationCallError(BaseModel):
    """A call-local failure that does not discard other results in the batch."""

    suite: Literal["ours", "eva-x", "input"] = Field(
        description="Suite whose local scoring unit failed; 'input' marks a "
        "call that could not be scored at all (missing task contract, "
        "unresolvable multilingual language)."
    )
    message: str = Field(description="Recorded failure message.")


class ConversationCallResult(BaseModel):
    """One stored simulation: headline judgments + shadow composites."""

    source: str = Field(description="Absolute source results path.")
    sim_id: str = Field(description="Source simulation identifier.")
    task_id: str = Field(description="Source task identifier.")
    domain: str = Field(description="Task domain used for scoring context.")
    language: str = Field(description="Resolved conversation language.")
    input_sha256: str = Field(
        description="Digest of the judge-relevant inputs (trajectory, ticks, "
        "policy, task contract, domain, language); empty for calls that "
        "failed at input time."
    )
    progression_dimensions: Annotated[
        list[EvaProgressionDimension],
        Field(
            default_factory=list,
            description="HEADLINE: per-dimension progression verdicts (empty "
            "when the eva-x suite errored).",
        ),
    ]
    conciseness_turns: Annotated[
        list[EvaConcisenessTurnResult],
        Field(
            default_factory=list,
            description="HEADLINE: per-agent-turn conciseness verdicts with "
            "closed failure modes (empty when errored or no agent turns).",
        ),
    ]
    shadow_scores: ConversationShadowScores = Field(
        default_factory=ConversationShadowScores,
        description="Demoted composite scores; never headlined.",
    )
    errors: list[ConversationCallError] = Field(
        default_factory=list, description="Suite-local failures for this call."
    )


class ProgressionDimensionSummary(BaseModel):
    """One progression dimension over one cell: the headline aggregate.

    Flags are reported by severity — the HEADLINE number is the major
    (rating-1) rate; minors (rating-2) are plentiful (question_quality flags
    most calls, ~87% of them minor) and are reported separately so they
    never inflate the headline.
    """

    name: EvaProgressionDimensionName
    n_calls: Annotated[int, Field(ge=0, description="Calls with a verdict.")]
    major_count: Annotated[
        int, Field(ge=0, description="Calls flagged at rating 1 (major).")
    ]
    major_rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="HEADLINE: major_count / n_calls; None when empty.",
    )
    minor_count: Annotated[
        int, Field(ge=0, description="Calls flagged at rating 2 (minor).")
    ]
    minor_rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None, description="minor_count / n_calls; None when empty."
    )
    mean_rating: Optional[float] = Field(
        default=None, description="Mean 1-3 severity rating; None when empty."
    )


class ConcisenessFailureModeSummary(BaseModel):
    """One conciseness failure mode over one cell's judged turns."""

    mode: EvaConcisenessFailureMode
    turn_count: Annotated[
        int, Field(ge=0, description="Judged turns carrying this mode.")
    ]
    rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="turn_count over all judged turns; None when no turns.",
    )


class ConversationShadowCellSummary(BaseModel):
    """Shadow composite means for one cell — separate from the headline."""

    label: Literal["uncalibrated-shadow"] = Field(
        default=UNCALIBRATED_SHADOW_LABEL,
        description="Fixed shadow marker for the whole block.",
    )
    eva_x_pass_rate: Optional[float] = Field(
        default=None,
        description="Fraction of calls passing the EVA-X conjunction, over "
        "calls with a non-None gate.",
    )
    progression_score_mean: Optional[float] = Field(
        default=None, description="Mean shadow progression composite score."
    )
    conciseness_score_mean: Optional[float] = Field(
        default=None, description="Mean shadow conciseness composite score."
    )
    turn_taking_score_mean: Optional[float] = Field(
        default=None,
        description="Mean embedded shadow turn-taking score over scored calls.",
    )
    ours_score_mean: Optional[float] = Field(
        default=None, description="Mean τ quality composite score."
    )
    n_gated: Annotated[
        int, Field(ge=0, description="Calls with a non-None EVA-X gate.")
    ] = 0


class ConversationCellSummary(BaseModel):
    """Per (language x domain) headline aggregates plus shadow means."""

    language: str
    domain: str
    n_calls: Annotated[int, Field(ge=0, description="Calls in the cell.")]
    n_judged: Annotated[
        int, Field(ge=0, description="Calls with progression verdicts.")
    ]
    n_judged_turns: Annotated[
        int, Field(ge=0, description="Agent turns with conciseness verdicts.")
    ]
    dimensions: list[ProgressionDimensionSummary] = Field(
        default_factory=list,
        description="HEADLINE: per-dimension flag rates and mean ratings.",
    )
    conciseness_failure_modes: list[ConcisenessFailureModeSummary] = Field(
        default_factory=list,
        description="HEADLINE: per-failure-mode turn counts and rates.",
    )
    conciseness_mean_rating: Optional[float] = Field(
        default=None, description="Mean per-turn 1-3 conciseness rating."
    )
    shadow_scores: ConversationShadowCellSummary = Field(
        default_factory=ConversationShadowCellSummary,
        description="Demoted composite means; never headlined.",
    )


class ConversationJudgeConfig(BaseModel):
    """All behavior-affecting settings for a conversation-judge run."""

    eva_x_model: str = Field(
        default=DEFAULT_EVA_X_JUDGE, description="EVA-X semantic judge model."
    )
    eva_x_model_args: dict = Field(
        default_factory=lambda: dict(DEFAULT_EVA_X_JUDGE_ARGS),
        description="EVA-X semantic judge arguments.",
    )
    include_ours: bool = Field(
        default=True,
        description="Whether to compute the τ quality composite (shadow).",
    )
    ours_llm_judge: bool = Field(
        default=True, description="Whether to run tau semantic quality factors."
    )
    ours_model: Optional[str] = Field(
        default=None, description="Optional tau quality-judge model override."
    )
    ours_model_args: dict = Field(
        default_factory=dict, description="Tau quality-judge argument overrides."
    )
    max_concurrency: Annotated[
        int, Field(gt=0, description="Maximum concurrently scored calls.")
    ] = DEFAULT_CONVERSATION_JUDGE_CONCURRENCY
    limit: Optional[
        Annotated[int, Field(gt=0, description="Optional first-N smoke-run limit.")]
    ] = None


class ConversationArtifactProvenance(BaseModel):
    """Reproducibility record for the complete sidecar artifact."""

    created_at: str = Field(description="Artifact creation timestamp.")
    git_commit: str = Field(
        description="Tau source revision that produced the artifact."
    )
    artifact_version: str = Field(
        default=CONVERSATION_ARTIFACT_VERSION,
        description="Serialized conversation-judge artifact contract version.",
    )
    source_paths: list[str] = Field(description="Resolved input results paths.")
    config: ConversationJudgeConfig = Field(
        description="Complete behavior-affecting run configuration."
    )
    eva_repository: str = Field(
        default="https://github.com/ServiceNow/eva",
        description="Upstream EVA source repository.",
    )
    eva_source_version: str = Field(
        default=EVA_X_SOURCE_VERSION, description="Upstream EVA contract version."
    )


class ConversationJudgeArtifact(BaseModel):
    """One production artifact spanning one or more result inputs."""

    provenance: ConversationArtifactProvenance = Field(
        description="Inputs, versions, source revision, and run settings."
    )
    calls: list[ConversationCallResult] = Field(
        description="Deterministically ordered per-call results."
    )
    cells: list[ConversationCellSummary] = Field(
        default_factory=list,
        description="Per (language x domain) headline aggregates + shadow means.",
    )
    num_calls: Annotated[
        int, Field(ge=0, description="Number of emitted call records.")
    ]
    num_errors: Annotated[
        int, Field(ge=0, description="Total recorded suite-local failures.")
    ]

    @model_validator(mode="after")
    def _validate_counts_and_identity(self) -> "ConversationJudgeArtifact":
        identities = [(call.source, call.sim_id) for call in self.calls]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "conversation artifact contains duplicate source/sim pairs"
            )
        if self.num_calls != len(self.calls):
            raise ValueError("num_calls does not match calls")
        if self.num_errors != sum(len(call.errors) for call in self.calls):
            raise ValueError("num_errors does not match call errors")
        return self
