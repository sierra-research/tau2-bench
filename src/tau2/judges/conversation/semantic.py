# Copyright Sierra
"""Faithful EVA conciseness and conversation-progression judgments."""

import json

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.judges.base import interruption_marked_text, judge_structured
from tau2.judges.conversation.models import (
    EVA_X_CONCISENESS_THRESHOLD,
    EVA_X_MULTILINGUAL_POLICY_VERSION,
    EVA_X_PROGRESSION_THRESHOLD,
    EVA_X_SEMANTIC_PROMPT_VERSION,
    EvaConcisenessMetric,
    EvaConcisenessTurnResult,
    EvaProgressionDimension,
    EvaProgressionDimensionName,
    EvaProgressionMetric,
)
from tau2.metrics.interaction_quality import extract_spoken_turns

_CONCISENESS_SYSTEM = """\
You are evaluating one assistant turn in a spoken customer-service conversation
using EVA conciseness. Judge information density and contextual proportionality,
not mere word count. Return only the requested JSON object.

Rating:
- 3: concise and proportionate; no meaningful excess.
- 2: a minor conciseness issue that does not seriously burden the caller.
- 1: substantial excess, density, enumeration, or disproportionate detail.

Select only applicable failure modes from:
verbosity_or_filler, excess_information_density,
over_enumeration_or_list_exhaustion, contextually_disproportionate_detail.

Judge in the language actually spoken; do not translate the turn into English
and impose English brevity norms. Do not penalize essential identifiers or
codes, phonetic confirmations, a reasonable final wrap-up, content truncated by
an interruption, obligatory honorifics or politeness, discourse particles,
grammatical morphology, or context-required repetition and explicitness. Use
the entire conversation to decide what was proportionate at this moment.

A turn ending in "[cut off by caller]" was interrupted mid-delivery by the
caller. Truncation artifacts at the cut-off — an incomplete final word or
sentence, a question cut off before completion — are never conciseness
failures; judge only what was actually delivered.

When the rating is below 3, evidence must give a brief concrete explanation of
the excess, grounded in the turn; never leave it empty.
"""

_PROGRESSION_SYSTEM = """\
You are evaluating call-level conversation progression using EVA. Return one
entry for each of the four named dimensions and only the requested JSON object.

Dimensions:
- unnecessary_tool_calls: tool use that did not advance a valid path.
- information_loss: caller information was lost, forcing avoidable recovery.
- redundant_statements: needless repetition that stalled progress.
- question_quality: questions were unclear, compound, premature, or failed to
  request the information needed for the next step.

For each dimension, set flagged and a rating: 3 when clean, 2 for a minor issue,
1 for a major issue. An unflagged dimension must have rating 3. Assess flow and
progress only. Do not score policy compliance, factual correctness, task success,
or interruption mechanics except where they directly caused information loss or
repetition. Judge questions and confirmations in the language actually spoken;
do not penalize language-appropriate honorifics, particles, explicitness,
confirmation, or turn-organization conventions merely for differing from English.

An agent turn ending in "[cut off by caller]" was interrupted mid-delivery by
the caller. Truncation artifacts at the cut-off — an incomplete final word or
sentence, a question cut off before completion — are never unclear questions,
repetition, or any other flagged defect; judge only what was actually delivered.

Every flagged dimension must carry brief concrete evidence naming the offending
turns or calls; never leave evidence empty on a flagged dimension.
"""


class EvaProgressionJudgeReply(BaseModel):
    """Validated closed-set progression reply."""

    dimensions: list[EvaProgressionDimension] = Field(
        description="Every closed EVA conversation-progression dimension."
    )

    @model_validator(mode="after")
    def _closed_dimension_set(self) -> "EvaProgressionJudgeReply":
        names = [dimension.name for dimension in self.dimensions]
        expected = set(EvaProgressionDimensionName)
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("progression reply must contain every dimension once")
        return self


def extract_agent_turns(sim: SimulationRun) -> list[tuple[int, str]]:
    """Return stable zero-based IDs for all non-empty spoken agent turns.

    The text is LLM-judge-facing: a turn the caller barged into carries the
    interruption marker so conciseness never reads caller-caused truncation
    as an agent defect. Raw text lives on ``extract_spoken_turns``.
    """
    return [
        (turn_id, interruption_marked_text(turn.text, turn.interrupted))
        for turn_id, turn in enumerate(
            turn for turn in extract_spoken_turns(sim) if turn.speaker == "agent"
        )
    ]


def _render_transcript(sim: SimulationRun) -> str:
    lines: list[str] = []
    agent_id = 0
    for turn in extract_spoken_turns(sim):
        if turn.speaker == "agent":
            text = interruption_marked_text(turn.text, turn.interrupted)
            lines.append(f"AGENT TURN {agent_id}: {text}")
            agent_id += 1
        else:
            lines.append(f"CALLER: {turn.text}")
    return "\n".join(lines) or "(no spoken transcript)"


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


def judge_conciseness_turn(
    sim: SimulationRun,
    *,
    turn_id: int,
    turn_text: str,
    language: str,
    model: str,
    model_args: dict,
) -> EvaConcisenessTurnResult:
    """Judge one assistant turn with full conversational context."""
    prompt = f"""\
PROMPT VERSION: {EVA_X_SEMANTIC_PROMPT_VERSION}
MULTILINGUAL POLICY: {EVA_X_MULTILINGUAL_POLICY_VERSION}
LANGUAGE: {language}

COMPLETE TRANSCRIPT
{_render_transcript(sim)}

TARGET AGENT TURN {turn_id}
{turn_text}

Return: turn_id, rating, failure_modes, evidence. Evidence must be brief and
specific. The returned turn_id must be {turn_id}.
"""
    result = judge_structured(
        model=model,
        messages=[
            SystemMessage(role="system", content=_CONCISENESS_SYSTEM),
            UserMessage(role="user", content=prompt),
        ],
        response_model=EvaConcisenessTurnResult,
        call_name="eva_x_conciseness_turn",
        model_args=model_args,
    )
    if result.turn_id != turn_id:
        raise ValueError(
            f"eva_x_conciseness_turn returned turn_id={result.turn_id}, "
            f"expected {turn_id}"
        )
    return result


def judge_progression(
    sim: SimulationRun,
    task: Task,
    *,
    domain: str,
    language: str,
    model: str,
    model_args: dict,
) -> list[EvaProgressionDimension]:
    """Judge the four closed EVA progression dimensions once per call."""
    prompt = f"""\
PROMPT VERSION: {EVA_X_SEMANTIC_PROMPT_VERSION}
MULTILINGUAL POLICY: {EVA_X_MULTILINGUAL_POLICY_VERSION}
LANGUAGE: {language}
DOMAIN: {domain}

TASK CONTRACT (context only; do not score task completion)
{task.model_dump_json(exclude_none=True)}

DOMAIN POLICY (context only)
{sim.policy or "(not recorded)"}

COMPLETE TRANSCRIPT
{_render_transcript(sim)}

TOOL TRACE
{_tool_trace(sim)}

Return an object with a dimensions list. Every item must contain name, flagged,
rating, and brief evidence.
"""
    reply = judge_structured(
        model=model,
        messages=[
            SystemMessage(role="system", content=_PROGRESSION_SYSTEM),
            UserMessage(role="user", content=prompt),
        ],
        response_model=EvaProgressionJudgeReply,
        call_name="eva_x_conversation_progression",
        model_args=model_args,
    )
    return reply.dimensions


def aggregate_conciseness(
    turns: list[EvaConcisenessTurnResult],
) -> EvaConcisenessMetric:
    """Normalize ratings 1/2/3 to 0/.5/1 and average them.

    A call with no spoken agent turns (e.g. a mute-agent failure) has no
    conciseness evidence: the score is None (N/A), never a zero-fail.
    """
    score = (
        sum((turn.rating - 1) / 2.0 for turn in turns) / len(turns) if turns else None
    )
    return EvaConcisenessMetric(score=score, turns=turns)


def aggregate_progression(
    dimensions: list[EvaProgressionDimension],
) -> EvaProgressionMetric:
    """Derive EVA's overall rating from flagged dimension severities."""
    names = [dimension.name for dimension in dimensions]
    expected = set(EvaProgressionDimensionName)
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError("progression aggregation requires every dimension once")
    flagged = [dimension for dimension in dimensions if dimension.flagged]
    if any(dimension.rating == 1 for dimension in flagged) or len(flagged) >= 3:
        rating = 1
    elif flagged:
        rating = 2
    else:
        rating = 3
    return EvaProgressionMetric(
        rating=rating,
        score=(rating - 1) / 2.0,
        dimensions=dimensions,
    )


def semantic_thresholds_pass(
    conciseness: EvaConcisenessMetric,
    progression: EvaProgressionMetric,
) -> bool:
    """Return whether both EVA semantic components clear their thresholds.

    An N/A conciseness score (no agent turns) is excluded rather than failed.
    """
    return progression.score >= EVA_X_PROGRESSION_THRESHOLD and (
        conciseness.score is None or conciseness.score >= EVA_X_CONCISENESS_THRESHOLD
    )
