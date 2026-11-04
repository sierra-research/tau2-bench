"""Criterion execution with explicit calibrated-batch and single-criterion protocols."""

import json
import re
from typing import Annotated

from pydantic import Field, StrictBool, model_validator

from tau2.data_model.judges import (
    ContractModel,
    DiagnosticId,
    ImplementationFamily,
    JudgeContract,
    Measure,
    MeasureVerdict,
    NativenessCriterion,
    UnitVerdict,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.data_model.transcript import AgentTurn
from tau2.judges.base import BaseJudge, require_complete_response, response_error
from tau2.judges.exclusions import judged_agent_turns
from tau2.judges.run_evidence import (
    LIVE_INPUT_CALL_ERROR,
    RunEvidence,
    live_input_errors,
)
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from tau2.utils.transcript.display import marked_text

CALIBRATED_CALL_FACTORS = frozenset(
    {
        "register_formality",
        "regional_consistency",
        "honorific_levels",
        "routines",
        "modal_particles",
        "discourse_markers",
    }
)


# Stable per-implementation LLM call names (see docs/CODE_DESIGN.md).
CALL_NAMES = {
    ImplementationFamily.NATIVENESS: "nativeness_judge_batch",
    ImplementationFamily.NATURALNESS: "naturalness_judge",
    ImplementationFamily.HINDI_GENDER: "hindi_gender_judge_batch",
    ImplementationFamily.AGENT_TEXT: "agent_diagnostic_judge",
}


class BatchedCriterionResult(ContractModel):
    """Frozen v15/v16 response contract, including quoted-span consistency."""

    factor_id: Annotated[str, Field(description="Exact requested factor identifier.")]
    opportunity: Annotated[bool, Field(description="Explicit opportunity.")]
    violated: Annotated[bool, Field(description="Observed violation.")]
    severity: Annotated[int, Field(ge=0, le=4, description="Frozen severity scale.")]
    reasoning: Annotated[str, Field(min_length=1, description="Judge explanation.")]
    quote: Annotated[str, Field(description="Exact offending span, or empty.")]

    @model_validator(mode="after")
    def consistent(self):
        """Invalid states remain errors rather than silently passing."""
        if self.violated and (not self.opportunity or self.severity == 0):
            raise ValueError("Violation requires opportunity and severity")
        if not self.violated and self.severity:
            raise ValueError("Nonviolations require severity zero")
        return self


class BatchedCriterionReply(ContractModel):
    """Typed batched model reply."""

    results: Annotated[
        list[BatchedCriterionResult], Field(description="Exactly requested criteria.")
    ]


class CriteriaBatchCache(ContractModel):
    """Within-invocation peer results, scoped to one exact simulation/contract."""

    replies: Annotated[
        dict[str, BatchedCriterionReply],
        Field(
            default_factory=dict, description="Validated reply by unit level and index."
        ),
    ]


def participant_context(agent_gender: str, caller_gender: str) -> str:
    """Preserve frozen role context using recorded genders only."""
    lines = ["The speaker is an AI customer-service voice agent."]
    if agent_gender in {"male", "female"}:
        lines.append(
            f"Agent gender: {agent_gender}. Use this for the agent's first-person gender agreement."
        )
    if caller_gender in {"male", "female"}:
        lines.append(
            f"Caller gender: {caller_gender}. Use this only when the agent directly addresses or refers to the caller with gender-marked language."
        )
    lines.append("Do not infer any missing gender from a name or stereotype.")
    return " ".join(lines)


def _replace(template: str, values: dict[str, str]) -> str:
    # A single regex pass avoids both interpreting literal JSON braces and
    # substituting placeholder-like strings inside recorded conversation text.
    pattern = re.compile("|".join(re.escape(key) for key in values))
    return pattern.sub(lambda m: values[m.group()], template)


def _hindi_precheck(text: str, task: Task, caller_gender: str) -> bool:
    # Exact ordered v26 patterns. Literal exclusions use only structured task
    # data; neither names nor participant genders are guessed from text.
    instructions = task.user_scenario.instructions
    if isinstance(instructions, str):
        source = str(task.model_dump())
    else:
        source = "\n".join(
            getattr(instructions, field, None) or ""
            for field in (
                "task_instructions",
                "reason_for_call",
                "known_info",
                "unknown_info",
            )
        )
    patterns = [
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        r"\b[a-z]+_[a-z]+_\d+\b",
        r"\b(?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{2,}\b|\b\d[A-Z0-9]{2,}\b",
        r"\d+",
    ]
    literals = {
        value
        for pattern in patterns
        for value in re.findall(pattern, source)
        if any(ch.isdigit() for ch in value) or "@" in value
    }
    for literal in sorted(set(literals), key=len, reverse=True):
        if literal:
            text = re.sub(re.escape(literal), "", text, flags=re.IGNORECASE)
    patterns = [r"मैं\s+[^।.!?\n]{0,48}?सकती\s+ह(?:ूँ|ूं)"]
    if caller_gender == "male":
        patterns.append(r"आप\s+[^।.!?\n]{0,48}?रही\s+हैं")
    patterns.append(
        r"(?:[क-ह][ा-ौँ-ः]*)?(?:ता|ती|रहा|रही|गा|गी|ूँगा|ूँगी)\s*[/|]\s*(?:[क-ह][ा-ौँ-ः]*)?(?:ता|ती|रहा|रही|गा|गी|ूँगा|ूँगी)(?![क-हा-ौ])"
    )
    return any(re.search(pattern, text) for pattern in patterns)


class CriterionDecision(ContractModel):
    """Single-criterion reply, independent of the calibrated batch reply shape."""

    opportunity: Annotated[
        StrictBool, Field(description="A reviewed opportunity exists.")
    ]
    violation: Annotated[
        StrictBool, Field(description="The agent violated the criterion.")
    ]
    evidence: Annotated[
        str, Field(description="Evidence or explicit no-opportunity explanation.")
    ]


class CriterionSelection(ContractModel):
    """Exact criteria and unit semantics selected before preparing a model request."""

    factor_id: Annotated[str, Field(description="Selected reported factor.")]
    criteria: Annotated[
        list[NativenessCriterion],
        Field(description="Complete calibrated peer batch or one generic criterion."),
    ]
    batched: Annotated[
        bool, Field(description="Use the frozen multi-factor response protocol.")
    ]
    is_call: Annotated[
        bool,
        Field(description="Evaluate the whole call rather than each delivered turn."),
    ]


class CriterionUnit(ContractModel):
    """Prepared evidence for one stable statistical unit."""

    unit_id: Annotated[str, Field(description="Call or delivered-turn identity.")]
    agent_speech: Annotated[
        str, Field(description="Protocol-specific speech or structured call evidence.")
    ]
    customer_context: Annotated[
        str, Field(description="Only the customer context supplied by this protocol.")
    ]


def _select_criteria(
    contract: JudgeContract,
    language: str,
    measure: str,
    agent_gender: str,
    caller_gender: str,
) -> CriterionSelection:
    if contract.implementation == ImplementationFamily.AGENT_TEXT:
        criteria = [
            c
            for c in contract.nativeness_criteria.get(language, [])
            if c.factor_id == measure
        ]
        if len(criteria) != 1:
            raise ValueError("Diagnostic requires exactly one matching agent criterion")
        return CriterionSelection(
            factor_id=measure,
            criteria=criteria,
            batched=False,
            is_call=contract.evaluation_level == "call",
        )
    if contract.implementation not in {
        ImplementationFamily.NATIVENESS,
        ImplementationFamily.NATURALNESS,
        ImplementationFamily.HINDI_GENDER,
    }:
        raise ValueError("Contract does not use a criterion execution protocol")
    factor = str(measure.value if isinstance(measure, Measure) else measure)
    is_call = factor in CALIBRATED_CALL_FACTORS
    criteria = [
        c
        for c in contract.nativeness_criteria.get(language, [])
        if (c.factor_id in CALIBRATED_CALL_FACTORS) == is_call
    ]
    if factor not in {c.factor_id for c in criteria}:
        raise ValueError("Measure absent from calibrated factor batch")
    if contract.implementation == ImplementationFamily.HINDI_GENDER and (
        agent_gender != "male" or caller_gender not in {"male", "female"}
    ):
        raise ValueError(
            "Hindi gender requires the validated male-agent and recorded caller-gender frame"
        )
    if (
        language == "pt"
        and measure == Measure.GENDER_AGREEMENT
        and agent_gender != "female"
    ):
        raise ValueError("Portuguese gender requires validated female-agent frame")
    return CriterionSelection(
        factor_id=factor, criteria=criteria, batched=True, is_call=is_call
    )


def _prepare_units(
    simulation: SimulationRun,
    turns: list[AgentTurn],
    selection: CriterionSelection,
    policy: str,
) -> list[CriterionUnit]:
    if not selection.is_call:
        return [
            CriterionUnit(
                unit_id=t.segment_id or str(t.index),
                agent_speech=marked_text(
                    t.text, t.interrupted, marker="[cut off by caller]"
                ),
                customer_context="\n".join(
                    part
                    for part in [
                        t.preceding_user_text,
                        (
                            "Caller speech overlapping this utterance (not proof of a cutoff): "
                            + t.overlapping_user_text
                        )
                        if t.overlapping_user_text
                        else None,
                    ]
                    if part
                ),
            )
            for t in turns
        ]
    if selection.batched:
        customer = []
        for turn in turns:
            text = (turn.preceding_user_text or "").strip()
            if text and (not customer or customer[-1] != text):
                customer.append(text)
        return [
            CriterionUnit(
                unit_id="call",
                agent_speech="\n".join(
                    "agent: "
                    + marked_text(t.text, t.interrupted, marker="[cut off by caller]")
                    for t in turns
                ),
                customer_context="\n".join(f"customer: {t}" for t in customer),
            )
        ]
    tools = []
    for message in simulation.messages or []:
        if message.role == "tool" or getattr(message, "tool_calls", None):
            tools.append(
                message.model_dump(
                    mode="json",
                    include={"role", "content", "tool_calls", "tool_call_id"},
                )
            )
    for tick in simulation.ticks or []:
        tools.extend(
            x.model_dump(mode="json")
            for x in tick.agent_tool_calls + tick.agent_tool_results
        )
    return [
        CriterionUnit(
            unit_id="call",
            agent_speech=json.dumps(
                {
                    "agent_turns": [t.model_dump() for t in turns],
                    "tools": tools,
                    "policy": policy,
                },
                ensure_ascii=False,
            ),
            customer_context="",
        )
    ]


def _render_prompt(
    unit: CriterionUnit,
    selection: CriterionSelection,
    contract: JudgeContract,
    language: str,
    agent_gender: str,
    caller_gender: str,
) -> str:
    if selection.batched:
        context = participant_context(agent_gender, caller_gender)
        customer = unit.customer_context or "(none)"
        values = {
            "{language}": language,
            "{evaluation_unit}": "all agent turns in this call"
            if selection.is_call
            else "one agent utterance",
            "{criteria_json}": json.dumps(
                [c.model_dump() for c in selection.criteria],
                ensure_ascii=False,
                indent=2,
            ),
            "{agent_context}": context,
            "{customer_context}": customer,
            "{agent_speech}": unit.agent_speech,
            "<ROW_AGENT_CONTEXT>": context,
            "<ROW_CUSTOMER_CONTEXT>": customer,
            "<ROW_AGENT_SPEECH_WITH_INTERRUPTION_MARKER>": unit.agent_speech,
        }
    else:
        values = {
            "{language}": language,
            "{agent_context}": f"Agent gender: {agent_gender}; customer gender: {caller_gender}.",
            "{criteria_json}": json.dumps(
                selection.criteria[0].model_dump(), ensure_ascii=False
            ),
            "{agent_speech}": unit.agent_speech,
            "{customer_context}": unit.customer_context,
        }
    return _replace(contract.prompt.user_prompt_template, values)


def _batch_verdict(
    reply: BatchedCriterionReply, unit: CriterionUnit, selection: CriterionSelection
) -> UnitVerdict:
    ids = [r.factor_id for r in reply.results]
    if len(ids) != len(set(ids)) or set(ids) != {
        c.factor_id for c in selection.criteria
    }:
        raise ValueError("Nativeness factor batch mismatch")
    for result in reply.results:
        if result.quote and result.quote.strip() not in unit.agent_speech:
            raise ValueError("Quote is not present in agent speech")
        if not selection.is_call and result.violated and not result.quote.strip():
            raise ValueError("Utterance violation requires exact quote")
    result = next(r for r in reply.results if r.factor_id == selection.factor_id)
    return UnitVerdict(
        unit_id=unit.unit_id,
        opportunity=result.opportunity,
        violation=result.violated,
        evidence=result.model_dump_json(),
    )


class CriteriaJudge(BaseJudge):
    """Execute the calibrated criteria protocol."""

    def __init__(self, contract: JudgeContract):
        super().__init__(contract)
        self._evidence: RunEvidence | None = None
        self._cache = CriteriaBatchCache()

    def resume(
        self, evidence: RunEvidence, measure: DiagnosticId, previous: MeasureVerdict
    ) -> MeasureVerdict:
        """Retry failed units while retaining successful units from matching inputs."""
        return self.evaluate(evidence, measure, previous=previous)

    def evaluate(
        self,
        evidence: RunEvidence,
        measure: DiagnosticId,
        *,
        previous: MeasureVerdict | None = None,
    ) -> MeasureVerdict:
        """Evaluate recorded evidence without changing the calibrated request or parser."""
        contract = self.contract
        simulation, task = evidence.simulation, evidence.task
        language = evidence.language
        agent_gender, caller_gender = evidence.agent_gender, evidence.caller_gender
        policy = evidence.policy
        if self._evidence != evidence:
            self._evidence = evidence.model_copy(deep=True)
            self._cache = CriteriaBatchCache()
        cache = self._cache
        selection = _select_criteria(
            contract, language, measure, agent_gender, caller_gender
        )
        errors = live_input_errors(simulation)
        if errors and selection.is_call:
            raise ValueError(
                f"{LIVE_INPUT_CALL_ERROR}: " + "; ".join(e.error for e in errors)
            )
        blocked = {e.unit_id for e in errors}
        turns = judged_agent_turns(
            simulation, evidence.opener, contract.input_exclusions
        )
        if not turns and errors:
            return MeasureVerdict(units=[], unit_errors=errors)
        if not turns:
            return MeasureVerdict(
                units=[
                    UnitVerdict(
                        unit_id="call",
                        opportunity=False,
                        violation=False,
                        evidence="No agent speech to judge."
                        if selection.batched
                        else "No delivered agent speech.",
                    )
                ]
            )
        outcomes = []
        saved = {u.unit_id: u for u in previous.units} if previous else {}
        for unit in _prepare_units(simulation, turns, selection, policy):
            if unit.unit_id in blocked:
                continue
            if unit.unit_id in saved:
                outcomes.append(saved[unit.unit_id])
                continue
            if (
                contract.implementation == ImplementationFamily.HINDI_GENDER
                and _hindi_precheck(unit.agent_speech, task, caller_gender)
            ):
                outcomes.append(
                    UnitVerdict(
                        unit_id=unit.unit_id,
                        opportunity=True,
                        violation=True,
                        evidence="Frozen Hindi gender precheck: explicit participant contradiction or voiced slash alternatives.",
                    )
                )
                continue
            response = None
            try:
                cache_key = (
                    f"{'call' if selection.is_call else 'utterance'}:{unit.unit_id}"
                )
                reply = cache.replies.get(cache_key) if selection.batched else None
                if reply is None:
                    prompt = _render_prompt(
                        unit, selection, contract, language, agent_gender, caller_gender
                    )
                    response = generate(
                        model=contract.model,
                        messages=[
                            SystemMessage(
                                role="system", content=contract.prompt.system_prompt
                            ),
                            UserMessage(role="user", content=prompt),
                        ],
                        call_name=CALL_NAMES[contract.implementation],
                        **contract.model_args,
                    )
                    require_complete_response(response)
                    content = extract_json_from_llm_response(response.content or "")
                    reply = (
                        BatchedCriterionReply.model_validate_json(content)
                        if selection.batched
                        else CriterionDecision.model_validate_json(content)
                    )
                if isinstance(reply, BatchedCriterionReply):
                    verdict = _batch_verdict(reply, unit, selection)
                    cache.replies[cache_key] = reply
                else:
                    verdict = UnitVerdict(unit_id=unit.unit_id, **reply.model_dump())
            except Exception as exc:
                errors.append(response_error(unit.unit_id, exc, response))
                continue
            outcomes.append(verdict)
        return MeasureVerdict(units=outcomes, unit_errors=errors)
