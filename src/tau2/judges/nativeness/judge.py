# Copyright Sierra
"""Batched text judge for language-specific agent nativeness criteria."""

import json
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_LLM_NATIVENESS_JUDGE,
    DEFAULT_LLM_NATIVENESS_JUDGE_ARGS,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.judges.base import VerdictReplyBase, judge_structured
from tau2.judges.nativeness.factors import NativenessRubricText

NATIVENESS_JUDGE_SYSTEM_PROMPT = (
    "You evaluate language use in an AI agent's speech. Return JSON only."
)

# v13: interruption-aware — barged-in turns carry "[cut off by caller]" and
# the prompt forbids reading truncation artifacts as language violations.
# v14: every violation with a quoted span must include the natural phrasing
# a native speaker would use instead, so annotators can adjudicate the flag
# directly.
# v15: prompt text unchanged; interruption detection unified on the
# tick-overlap barge-in detector (union with the chunk/gold signal), so more
# truly cut-off turns carry the marker in the judged input.
# v16: natural_word_choice becomes the frozen, per-language combined utterance
# naturalness criterion. It replaces separate word-choice, translationese, and
# verb-morphology decisions; Spanish also absorbs regional consistency and
# written-diacritic correctness.
NATIVENESS_JUDGE_PROMPT_VERSION = "v16"


class NativenessJudgeCriterion(BaseModel):
    """One factor rubric supplied in a batched judge request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str
    question: str
    opportunity: str
    positive_examples: str
    negative_examples: str

    @classmethod
    def from_rubric(
        cls, factor_id: str, rubric: NativenessRubricText
    ) -> "NativenessJudgeCriterion":
        return cls(
            factor_id=factor_id,
            question=rubric.language_question or rubric.question,
            opportunity=rubric.opportunity,
            positive_examples=rubric.language_allowed,
            negative_examples=rubric.language_violation,
        )


class NativenessJudgeInput(BaseModel):
    """Typed inputs for one call-level or utterance-level judge invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    evaluation_level: Literal["call", "utterance"]
    criteria: list[NativenessJudgeCriterion]
    agent_text: str
    customer_context: Optional[str] = None
    agent_context: Optional[str] = None

    @model_validator(mode="after")
    def _criteria_are_nonempty_and_unique(self) -> "NativenessJudgeInput":
        ids = [criterion.factor_id for criterion in self.criteria]
        if not ids:
            raise ValueError("nativeness judge input needs at least one criterion")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate nativeness factor ids: {ids}")
        return self


class NativenessJudgeResult(VerdictReplyBase):
    """One criterion result with explicit opportunity and observed severity."""

    model_config = ConfigDict(extra="forbid")

    factor_id: str
    severity: Annotated[
        int,
        Field(
            ge=0,
            le=4,
            description="Zero without a violation; otherwise 1 (minor) to 4 (egregious).",
        ),
    ]

    @model_validator(mode="after")
    def _valid_state(self) -> "NativenessJudgeResult":
        if not self.opportunity and self.violated:
            raise ValueError("a no-opportunity result cannot be a violation")
        if self.violated and self.severity == 0:
            raise ValueError("a violation needs severity 1..4")
        if not self.violated and self.severity != 0:
            raise ValueError("severity must be zero when there is no violation")
        return self


class NativenessJudgeReply(BaseModel):
    """Structured reply for one batched nativeness judge invocation."""

    model_config = ConfigDict(extra="forbid")

    results: list[NativenessJudgeResult]


def build_user_prompt(request: NativenessJudgeInput) -> str:
    """Render the fixed v12 user prompt from typed request data."""
    criteria = [criterion.model_dump() for criterion in request.criteria]
    context = request.customer_context or "(none)"
    unit_label = (
        "all agent turns in this call"
        if request.evaluation_level == "call"
        else "one agent utterance"
    )
    return f"""Language: {request.language}
Evaluation unit: {unit_label}

For every criterion:
- opportunity is true only when this speech gave the agent a genuine chance to exhibit the criterion.
- violated is true only when the agent actually violated the criterion. The criterion question asks about the desired behavior; answer violated=true when the answer is no.
- severity is 0 when violated is false. When violated is true: 1 = minor, 2 = clear, 3 = major, 4 = egregious. Repetition may increase severity at call level.
- quote is an exact offending span copied from AGENT SPEECH. Use an empty string for no violation or for a diffuse call-level pattern.
- Judge only AGENT SPEECH. Use CUSTOMER CONTEXT only to interpret it.
- A turn ending in "[cut off by caller]" was interrupted mid-delivery: its final word may be truncated mid-word and endings (agreement suffixes, polite verb endings, sentence-final particles) may be missing. Never count a truncation artifact at the cut-off point as a violation of any criterion; judge only what was audibly complete.
- When violated is true and the violation localizes to a quoted span, reasoning must also give the natural way a native speaker would phrase that span, in the target language.

Criteria:
{json.dumps(criteria, ensure_ascii=False, indent=2)}

About the agent being judged:
{request.agent_context or "(none provided)"}

CUSTOMER CONTEXT — DO NOT JUDGE:
{context}

AGENT SPEECH TO JUDGE:
{request.agent_text}

Return JSON only:
{{
  "results": [
    {{
      "factor_id": "<exact factor_id>",
      "opportunity": true,
      "violated": false,
      "severity": 0,
      "reasoning": "<brief explanation>",
      "quote": ""
    }}
  ]
}}
"""


def _no_opportunity_results(
    request: NativenessJudgeInput,
) -> list[NativenessJudgeResult]:
    return [
        NativenessJudgeResult(
            factor_id=criterion.factor_id,
            opportunity=False,
            violated=False,
            severity=0,
            reasoning="No agent speech to judge.",
            quote="",
        )
        for criterion in request.criteria
    ]


def run_nativeness_judge(
    request: NativenessJudgeInput,
    *,
    model: str = DEFAULT_LLM_NATIVENESS_JUDGE,
    model_args: Optional[dict] = None,
) -> list[NativenessJudgeResult]:
    """Judge every requested criterion in one structured LLM call."""
    if not request.agent_text.strip():
        return _no_opportunity_results(request)

    reply = judge_structured(
        model=model,
        messages=[
            SystemMessage(role="system", content=NATIVENESS_JUDGE_SYSTEM_PROMPT),
            UserMessage(role="user", content=build_user_prompt(request)),
        ],
        response_model=NativenessJudgeReply,
        call_name="nativeness_judge_batch",
        model_args=(
            model_args if model_args is not None else DEFAULT_LLM_NATIVENESS_JUDGE_ARGS
        ),
    )

    expected = [criterion.factor_id for criterion in request.criteria]
    returned = [result.factor_id for result in reply.results]
    if len(returned) != len(set(returned)):
        raise ValueError(f"nativeness judge returned duplicate factor ids: {returned}")
    if set(returned) != set(expected):
        raise ValueError(
            "nativeness judge factor mismatch: "
            f"expected={sorted(expected)}, returned={sorted(returned)}"
        )

    by_id = {result.factor_id: result for result in reply.results}
    ordered = [by_id[factor_id] for factor_id in expected]
    for result in ordered:
        quote = result.quote.strip()
        if result.violated and request.evaluation_level == "utterance" and not quote:
            raise ValueError(
                f"utterance violation for {result.factor_id} requires an exact quote"
            )
        if quote and quote not in request.agent_text:
            raise ValueError(
                f"quote for {result.factor_id} is not present in agent speech: {quote!r}"
            )
    return ordered
