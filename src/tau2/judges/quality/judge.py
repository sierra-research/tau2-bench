# Copyright Sierra
"""Per-factor, task-aware LLM judgments for the universal quality rubric."""

import json
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import JudgeOutcome, SimulationRun
from tau2.data_model.tasks import Task, UserInstructions
from tau2.judges.base import (
    VerdictReplyBase,
    interruption_marked_text,
    judge_structured,
)
from tau2.judges.quality.factors import LLM_FACTORS, QualityFactorConfig
from tau2.metrics.interaction_quality import (
    InteractionQualityMetrics,
    extract_spoken_turns,
)

# v5: redundant_successful_tool_call retired — exact duplicates fold into
# unnecessary_tool_call's instruction instead of being diverted.
# v6: interruption-aware — barged-in agent turns carry "[cut off by caller]"
# in the transcript and the prompt forbids reading truncation artifacts as
# violations; a violated verdict must fill reasoning (enforced on the reply
# model too — see VerdictReplyBase._violation_requires_reasoning).
# v7: prompt text unchanged; interruption detection unified on the
# tick-overlap barge-in detector (union with the chunk/gold signal), so more
# truly cut-off turns carry the marker in the transcript.
# v8: calibrated against the recall_20 gold set (judge_returns_2026-08-31).
# ASR carve-out with an availability test for parameter/auth mismatches,
# summary fields are not parameters, interruption-forced repetition and
# corrected retries / post-mutation verification reads are not violations,
# re-asking for already-provided details is repetition.
# v9: v8's carve-outs over-generalized on the same gold set. The ASR
# carve-out is restricted to values the caller spoke aloud (never tool
# results or invented values), the stale-context/verification-read excuse is
# removed, retries count when repeated with unchanged arguments, and the
# repetition carve-out needs an explicit caller request or cut-off marker.
# v10: incorrect_tool_parameters and auth_arg_mismatch revert to the v7 text
# (both were above the 0.80 F1 hands-off line; v8/v9 carve-outs measured as
# regressions). unnecessary_repetition pivots on whether anything failed
# between the two occurrences; unnecessary_tool_call excludes transfers and
# tolerates one redundant re-read or retry. Plain short-sentence style.
# v11: unnecessary_repetition and unnecessary_tool_call also revert to the v7
# text — v10's carve-outs measured as full-corpus regressions (rep 0.72→0.57,
# utc 0.76→0.45 vs recall_20 gold). Every factor instruction is now identical
# to v7; the disagreement residue is factor-boundary misattribution handled by
# bucket-level calibration reporting, not prompt lines. The stamp stays
# monotonic so version-aware verdict reuse never confuses lineages.
QUALITY_JUDGE_PROMPT_VERSION = "quality-judge-v11"

_SYSTEM_PROMPT = (
    "You evaluate process quality in AI customer-service calls. Return JSON only."
)

_FACTOR_INSTRUCTIONS: dict[str, str] = {
    "unnecessary_repetition": """\
FAIL when the agent repeats materially the same information, question, or action
without a conversational need; rephrased repetition counts. Do not fail necessary
confirmations, requested clarifications, concise recaps, or restatements after a
misunderstanding. On failure, identify both occurrences.""",
    "agent_caused_tool_error": """\
FAIL only when an agent-requested tool error was avoidable and caused by the
agent's comprehension, tool choice, workflow, or arguments. Do not fail errors from
caller misinformation, unavailable state, policy constraints, or infrastructure.
On failure, name the errored call and the cause.""",
    "auth_arg_mismatch": """\
FAIL only when an identity lookup returned no match because the agent altered,
confused, invented, or reused unsupported authentication information. Do not fail a
faithful lookup of caller-provided information that happens to match no record. On
failure, name the lookup and the conflicting facts.""",
    "incorrect_tool_parameters": """\
FAIL when a tool argument contradicts caller facts, prior tool results, the task,
or policy, or when the agent invents an unsupported value. A successful call does
not prove its arguments were correct. On failure, name the call, parameter, value,
and contradiction.""",
    "unnecessary_tool_call": """\
Judge each call using only information available when it was made, not hindsight.
FAIL only when a concrete call could be removed while preserving a correct,
policy-compliant path to the same outcome. Discovery calls are necessary when they
resolve information the agent does not yet have. In particular, an agent may need
to inspect multiple candidate orders to find which order contains the relevant
items; do not fail the lookups needed to identify that order. A bad parameter belongs
to incorrect_tool_parameters. On failure, supply unnecessary_call and
correct_path_without_call.""",
}

if set(_FACTOR_INSTRUCTIONS) != {factor.id for factor in LLM_FACTORS}:
    raise ValueError("quality factor prompt coverage does not match LLM catalog")


class QualityFactorReply(VerdictReplyBase):
    """Validated reply for one LLM-backed quality factor."""

    violated: Annotated[
        bool,
        Field(
            description=(
                "Whether the agent violated the process-quality criterion "
                "(only meaningful when opportunity is true)."
            )
        ),
    ] = False


class QualityJudgeCriterion(BaseModel):
    """The one process-quality criterion supplied to a judge call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    question: str
    decision_rule: str


class QualityTaskContract(BaseModel):
    """Caller-visible task context without evaluator-only reference data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    user_instructions: UserInstructions
    ticket: Optional[str] = None


class QualityJudgeInput(BaseModel):
    """Typed evidence for one process-quality judge invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str
    criterion: QualityJudgeCriterion
    task_contract: QualityTaskContract
    domain_policy: str
    transcript: str
    tool_trace: str


class UnnecessaryToolCallReply(QualityFactorReply):
    """Counterfactual proof required to fail unnecessary_tool_call."""

    unnecessary_call: Annotated[
        str,
        Field(description="Concrete tool call that should be removed."),
    ] = ""
    correct_path_without_call: Annotated[
        str,
        Field(description="Viable policy-compliant path that omits that call."),
    ] = ""

    @field_validator("unnecessary_call", "correct_path_without_call", mode="before")
    @classmethod
    def _coerce_structured_call(cls, value: object) -> object:
        # Judges at high reasoning effort sometimes describe the call as a
        # structured object ({"tool_name": ..., "arguments": {...}}) instead
        # of prose; the description is still usable evidence.
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @model_validator(mode="after")
    def _failure_requires_counterfactual(self) -> "UnnecessaryToolCallReply":
        if self.opportunity and self.violated:
            if not self.unnecessary_call.strip():
                raise ValueError("FAIL requires unnecessary_call")
            if not self.correct_path_without_call.strip():
                raise ValueError("FAIL requires correct_path_without_call")
        return self


class QualityFactorJudgeResult(BaseModel):
    """One factor verdict plus the exact LLM provenance that produced it."""

    outcome: Annotated[JudgeOutcome, Field(description="Scoring outcome.")]
    evidence: Annotated[
        Optional[str], Field(description="Failure explanation when present.")
    ] = None
    quote: Annotated[
        Optional[str], Field(description="Exact offending transcript span.")
    ] = None
    model: Annotated[str, Field(description="Model used for this factor.")]
    model_args: Annotated[dict, Field(description="generate() arguments used.")]
    prompt_version: Annotated[str, Field(description="Versioned factor prompt.")]


def _full_transcript(sim: SimulationRun) -> str:
    """LLM-judge-facing transcript: interrupted agent turns carry the marker."""
    return (
        "\n".join(
            f"{turn.speaker}: "
            + interruption_marked_text(
                turn.text, turn.speaker == "agent" and turn.interrupted
            )
            for turn in extract_spoken_turns(sim)
        )
        or "(no spoken transcript)"
    )


def _tool_trace(sim: SimulationRun) -> str:
    lines: list[str] = []
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for message in sim.get_messages():
        for call in getattr(message, "tool_calls", None) or []:
            identity = call.id or f"{call.name}:{len(lines)}"
            if identity in seen_calls or call.requestor != "assistant":
                continue
            seen_calls.add(identity)
            lines.append(
                f"tool_call id={call.id} name={call.name} "
                f"arguments={json.dumps(call.arguments, ensure_ascii=False, default=str)}"
            )
        if getattr(message, "role", None) != "tool":
            continue
        if message.id in seen_results or message.requestor != "assistant":
            continue
        seen_results.add(message.id)
        content = (message.content or "").replace("\n", " ")[:1200]
        lines.append(f"tool_result id={message.id} error={message.error} {content}")
    return "\n".join(lines) or "(no tool calls)"


def build_user_prompt(request: QualityJudgeInput) -> str:
    """Render the concise v5 user prompt from typed evidence."""
    criterion = json.dumps(request.criterion.model_dump(), ensure_ascii=False, indent=2)
    return f"""Evaluate this call for the following criterion.

For this criterion:
- opportunity is true only when the call gave the agent a genuine chance to exhibit the behavior.
- violated is true only when the agent violated the criterion.
- Judge the agent, not the caller or infrastructure. Allow any policy-compliant path.
- When the evidence does not establish a violation, set violated=false.
- An agent turn ending in "[cut off by caller]" was interrupted mid-delivery by the caller. Truncation artifacts at the cut-off (an incomplete final word or sentence, a question cut off before completion) are never violations; judge only what was actually delivered.
- When violated is true, reasoning must give a brief concrete explanation of the violation, grounded in the transcript or tool trace; never leave it empty.
- quote is an exact offending transcript span, or an empty string when none applies.

Criterion:
{criterion}

Domain: {request.domain}

TASK CONTRACT
{request.task_contract.model_dump_json(exclude_none=True)}

DOMAIN POLICY
{request.domain_policy}

COMPLETE TRANSCRIPT
{request.transcript}

TOOL TRACE
{request.tool_trace}

Return one JSON object matching the requested schema.
"""


def build_quality_judge_input(
    sim: SimulationRun,
    task: Task,
    factor: QualityFactorConfig,
    *,
    domain: Optional[str],
) -> QualityJudgeInput:
    """Build typed evidence for one semantic-quality criterion."""
    return QualityJudgeInput(
        domain=domain or "(unknown)",
        criterion=QualityJudgeCriterion(
            criterion_id=factor.id,
            question=factor.question,
            decision_rule=_FACTOR_INSTRUCTIONS[factor.id],
        ),
        task_contract=QualityTaskContract(
            task_id=task.id,
            user_instructions=task.user_scenario.instructions,
            ticket=task.ticket,
        ),
        domain_policy=sim.policy or "(not recorded)",
        transcript=_full_transcript(sim),
        tool_trace=_tool_trace(sim),
    )


def build_quality_factor_prompt(
    sim: SimulationRun,
    task: Task,
    factor: QualityFactorConfig,
    *,
    domain: Optional[str],
) -> str:
    """Render one versioned semantic-quality prompt."""
    return build_user_prompt(
        build_quality_judge_input(sim, task, factor, domain=domain)
    )


def preclassified_quality_outcome(
    factor: QualityFactorConfig, metrics: InteractionQualityMetrics
) -> Optional[JudgeOutcome]:
    """Return a high-precision outcome when no LLM call is necessary."""
    if factor.id == "unnecessary_repetition":
        return JudgeOutcome.NO_OPPORTUNITY if metrics.agent_turn_count < 2 else None
    if factor.id == "agent_caused_tool_error":
        return (
            JudgeOutcome.NO_OPPORTUNITY if metrics.agent_tool_error_count == 0 else None
        )
    if factor.id == "auth_arg_mismatch":
        if not metrics.auth_tool_call_count:
            return JudgeOutcome.NO_OPPORTUNITY
        if not metrics.auth_arg_mismatch_count:
            return JudgeOutcome.PASS
        return None
    if factor.id in {"incorrect_tool_parameters", "unnecessary_tool_call"}:
        return (
            JudgeOutcome.NO_OPPORTUNITY if metrics.agent_tool_call_count == 0 else None
        )
    raise ValueError(f"no LLM preclassification rule for {factor.id}")


def run_quality_factor_judge(
    sim: SimulationRun,
    task: Task,
    factor: QualityFactorConfig,
    *,
    domain: Optional[str],
    model: str,
    model_args: dict,
) -> QualityFactorJudgeResult:
    """Judge one factor in one isolated, validated LLM call."""
    response_model: type[QualityFactorReply] = (
        UnnecessaryToolCallReply
        if factor.id == "unnecessary_tool_call"
        else QualityFactorReply
    )
    reply = judge_structured(
        model=model,
        messages=[
            SystemMessage(role="system", content=_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=build_quality_factor_prompt(sim, task, factor, domain=domain),
            ),
        ],
        response_model=response_model,
        call_name=f"quality_judge_{factor.id}",
        model_args=model_args,
    )
    if not reply.opportunity:
        outcome = JudgeOutcome.NO_OPPORTUNITY
    elif reply.violated:
        outcome = JudgeOutcome.FAIL
    else:
        outcome = JudgeOutcome.PASS
    evidence = reply.reasoning.strip() or None if outcome == JudgeOutcome.FAIL else None
    if isinstance(reply, UnnecessaryToolCallReply) and outcome == JudgeOutcome.FAIL:
        counterfactual = (
            f"Unnecessary call: {reply.unnecessary_call.strip()}. "
            f"Correct path without it: {reply.correct_path_without_call.strip()}"
        )
        evidence = f"{evidence} {counterfactual}" if evidence else counterfactual
    return QualityFactorJudgeResult(
        outcome=outcome,
        evidence=evidence,
        quote=reply.quote.strip() or None if outcome == JudgeOutcome.FAIL else None,
        model=model,
        model_args=dict(model_args),
        prompt_version=QUALITY_JUDGE_PROMPT_VERSION,
    )
