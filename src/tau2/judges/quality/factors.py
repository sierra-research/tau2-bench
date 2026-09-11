# Copyright Sierra
"""Closed, versioned catalog for the universal call-quality rubric."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.config import (
    DEFAULT_LLM_QUALITY_FACTOR_ARGS,
    DEFAULT_LLM_QUALITY_JUDGE,
)

QUALITY_RUBRIC_VERSION = "quality-rubric-v4"

QualityEvaluator = Literal["deterministic", "llm", "hybrid", "audio"]


class QualityFactorConfig(BaseModel):
    """One binary factor in the universal quality rubric."""

    id: Annotated[str, Field(description="Stable factor identifier.")]
    category: Annotated[str, Field(description="Rubric category.")]
    question: Annotated[str, Field(description="Binary violation question.")]
    evaluator: Annotated[
        QualityEvaluator, Field(description="Mechanism that produces the verdict.")
    ]
    evaluation_level: Annotated[
        Literal["call", "utterance"],
        Field(
            description="Unit the human instrument labels for this factor. "
            "'utterance' = the violation lives in clickable agent-utterance "
            "content, so annotation packets require the offending agent "
            "turns to be selected; 'call' = the violation lives in the tool "
            "trace or timing/event stream, where no single agent utterance "
            "carries it."
        ),
    ] = "call"
    severity: Annotated[
        int, Field(description="Stored factor weight; fixed to 1 in V1.", ge=1, le=3)
    ] = 1
    judge_model: Annotated[
        Optional[str],
        Field(description="Default model for an LLM-backed factor."),
    ] = None
    judge_args: Annotated[
        dict,
        Field(description="Default generate() arguments for an LLM-backed factor."),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _llm_config_matches_evaluator(self) -> "QualityFactorConfig":
        uses_llm = self.evaluator in {"llm", "hybrid"}
        if uses_llm and not self.judge_model:
            raise ValueError(f"{self.id}: LLM-backed factor requires judge_model")
        if not uses_llm and (self.judge_model or self.judge_args):
            raise ValueError(f"{self.id}: deterministic factor cannot configure an LLM")
        return self


def _llm_factor(
    *,
    id: str,
    category: str,
    question: str,
    evaluator: Literal["llm", "hybrid"],
    evaluation_level: Literal["call", "utterance"] = "call",
) -> QualityFactorConfig:
    """Build one LLM-backed factor from the closed default configuration."""
    return QualityFactorConfig(
        id=id,
        category=category,
        question=question,
        evaluator=evaluator,
        evaluation_level=evaluation_level,
        judge_model=DEFAULT_LLM_QUALITY_JUDGE,
        judge_args=dict(DEFAULT_LLM_QUALITY_FACTOR_ARGS[id]),
    )


QUALITY_FACTORS: tuple[QualityFactorConfig, ...] = (
    QualityFactorConfig(
        id="responsiveness",
        category="responsiveness",
        question="Did the agent respond reliably and without excessive delay?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="yielding",
        category="turn_taking",
        question="Did the agent yield when the caller interrupted?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="inappropriate_interruption",
        category="turn_taking",
        question="Did the agent interrupt or talk over the caller?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="backchannel_selectivity",
        category="selectivity",
        question="Did the agent correctly continue through caller backchannels?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="vocal_tic_selectivity",
        category="selectivity",
        question="Did the agent correctly ignore caller vocal tics?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="non_directed_selectivity",
        category="selectivity",
        question="Did the agent correctly ignore non-directed caller speech?",
        evaluator="deterministic",
    ),
    QualityFactorConfig(
        id="monologue",
        category="conversational_structure",
        question="Did the agent monologue instead of allowing interaction?",
        evaluator="deterministic",
        evaluation_level="utterance",
    ),
    _llm_factor(
        id="unnecessary_repetition",
        category="repetition",
        question="Did the agent unnecessarily repeat itself?",
        evaluator="llm",
        evaluation_level="utterance",
    ),
    _llm_factor(
        id="agent_caused_tool_error",
        category="workflow",
        question="Did the agent cause an avoidable tool error?",
        evaluator="hybrid",
    ),
    _llm_factor(
        id="auth_arg_mismatch",
        category="workflow",
        question="Did the agent use inconsistent authentication arguments?",
        evaluator="hybrid",
    ),
    _llm_factor(
        id="incorrect_tool_parameters",
        category="workflow",
        question="Did the agent use incorrect or unjustified tool parameters?",
        evaluator="llm",
    ),
    _llm_factor(
        id="unnecessary_tool_call",
        category="workflow",
        question="Did the agent make an unnecessary tool call?",
        evaluator="llm",
    ),
)

QUALITY_FACTOR_BY_ID = {factor.id: factor for factor in QUALITY_FACTORS}
DETERMINISTIC_FACTORS = tuple(
    factor for factor in QUALITY_FACTORS if factor.evaluator == "deterministic"
)
SEMANTIC_FACTORS = tuple(
    factor for factor in QUALITY_FACTORS if factor.evaluator == "llm"
)
HYBRID_FACTORS = tuple(
    factor for factor in QUALITY_FACTORS if factor.evaluator == "hybrid"
)
LLM_FACTORS = SEMANTIC_FACTORS + HYBRID_FACTORS

if len(QUALITY_FACTOR_BY_ID) != len(QUALITY_FACTORS):
    raise ValueError("quality factor ids must be unique")
