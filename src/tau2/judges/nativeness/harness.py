# Copyright Sierra
"""Per-simulation nativeness scoring entrypoint.

``evaluate_nativeness`` runs each enabled factor's checker and aggregates the
binary checks that FIRED (PASS or FAIL) with equal weight. NO_OPPORTUNITY,
DEFERRED, and ERROR are excluded and reported separately as score coverage.
"""

from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessInfo,
    NativenessJudgeSettings,
    NativenessJudgeUnitResult,
    SimulationRun,
)
from tau2.data_model.tasks import Task
from tau2.judges.base import binary_pass_fraction, interruption_marked_text
from tau2.judges.nativeness.agent_voice import ParticipantGender
from tau2.judges.nativeness.caller_identity import CallerNameContext
from tau2.judges.nativeness.checkers import (
    BACKCHANNEL_FREQUENCY_SPECS,
    CHECKER_REGISTRY,
    CheckerContext,
)
from tau2.judges.nativeness.factors import (
    DEFAULT_FACTORS,
    TEXT_ONLY_FACTORS_BY_LANGUAGE,
    NativenessFactorConfig,
    NativenessRubricText,
    email_symbols_for,
    enabled_deterministic_factor_ids_for,
    judge_factors_for,
)
from tau2.judges.nativeness.judge import (
    NATIVENESS_JUDGE_PROMPT_VERSION,
    NativenessJudgeCriterion,
    NativenessJudgeInput,
    NativenessJudgeResult,
    run_nativeness_judge,
)
from tau2.metrics.interaction_quality import (
    barged_in_tick_spans,
    tick_span_barged_in,
    utterance_interrupted,
)
from tau2.multilingual.invariants import extract_values, instruction_text

# v26: hybrid-precheck context carries per-turn interruption metadata, so a
# clipped Hindi ``आप बता…`` falls through to the interruption-aware LLM
# instead of short-circuiting as a deterministic honorific failure. The LLM
# prompt itself is unchanged and keeps its independent prompt version.
NATIVENESS_RUBRIC_VERSION = "nativeness-rubric-v26"

# This calibrated criterion was validated as the only criterion in each
# utterance request. Keep that request boundary in production: neighboring
# criteria change the model's decision context even when their prose is
# otherwise unchanged.
ISOLATED_UTTERANCE_FACTOR_IDS = frozenset({"natural_word_choice"})


class AgentTurn(BaseModel):
    """One delivered agent utterance plus its immediately preceding user turn."""

    index: int = Field(ge=0)
    text: str
    preceding_user_text: Optional[str] = None
    interrupted: bool = False

    @property
    def judged_text(self) -> str:
        """The text as shown to the LLM judge: raw text, plus the
        interruption marker when the caller barged in mid-utterance
        (``tau2.judges.base.INTERRUPTED_TURN_MARKER``). Without it the judge
        reads a truncated final word ("transcripció", "बता") as a
        grammar/honorific violation."""
        return interruption_marked_text(self.text, self.interrupted)


class AggregatedNativenessResult(BaseModel):
    """One utterance factor after deterministic call aggregation."""

    outcome: JudgeOutcome
    severity: int = Field(ge=0, le=4)
    violation_count: int = Field(
        default=0,
        ge=0,
        description="Number of violated utterance units (each unit is violated "
        "or not — never more than one violation per utterance). Recorded even "
        "when the call verdict is PASS (dose aggregation can pass with one).",
    )
    evidence: Optional[str] = None
    quote: Optional[str] = None


class JudgeFactorEvaluation(BaseModel):
    """All LLM-factor checks produced from an explicit sequence of agent turns."""

    checks: list[NativenessFactorCheck]
    judge_ran: bool = False


def _delivered_text(message) -> Optional[str]:
    """Plain text actually delivered for one participant message.

    Interruption detection is separate — see
    ``tau2.metrics.interaction_quality.utterance_interrupted``, which is only
    correct over a whole utterance group.
    """
    if getattr(message, "contains_speech", True) is False:
        return None
    text = message.audio_script_gold or message.content
    if not text:
        return None
    if message.audio_script_gold:
        from tau2.agent.base.streaming_utils import extract_delivered_text

        text, _interrupted = extract_delivered_text(text)
    return text or None


def _tick_utterances(ticks, attribute: str):
    """Merge one participant's timestamped chunks into utterance events.

    Events are ``(start_tick_id, end_tick_id, text, interrupted)`` — tick ids,
    not list positions, so an event's range is directly comparable with the
    tick-overlap barge-in spans (``barged_in_tick_spans``). ``interrupted``
    here carries only the chunk-level signal; ``build_agent_turns`` ORs in
    the tick-overlap detector for agent events.
    """
    chunks = [
        (tick.tick_id, chunk)
        for tick in ticks
        if (chunk := getattr(tick, attribute)) is not None and not chunk.is_tool_call()
    ]
    if not chunks:
        return []

    groups = []
    current = [chunks[0]]
    current_ids = set(chunks[0][1].utterance_ids or [])
    for item in chunks[1:]:
        chunk_ids = set(item[1].utterance_ids or [])
        overlaps = bool(
            current_ids and chunk_ids and not current_ids.isdisjoint(chunk_ids)
        )
        if overlaps:
            current.append(item)
            current_ids.update(chunk_ids)
        else:
            groups.append(current)
            current = [item]
            current_ids = chunk_ids
    groups.append(current)

    events = []
    for group in groups:
        messages = [item[1] for item in group]
        merged = (
            messages[0]
            if len(messages) == 1
            else type(messages[0]).merge_chunks(messages)
        )
        text = _delivered_text(merged)
        if text:
            events.append(
                (group[0][0], group[-1][0], text, utterance_interrupted(messages))
            )
    return events


def pinned_agent_greeting(simulation: SimulationRun) -> Optional[str]:
    """The harness-injected agent opening greeting, identified STRUCTURALLY.

    The runner injects a pinned greeting as the agent's first message: the
    full-duplex orchestrator stores it as the tick-0 agent chunk (text-only,
    silence audio, ``contains_speech=False``); the half-duplex builder injects
    it as the initial assistant message. Neither is model output, so checkers
    must never score it. Identification is structural — never keyed on the
    current pack's greeting text — so frozen sims carrying an older pack
    greeting resolve correctly. Returns None when the first agent event does
    not have the injected-greeting shape.
    """
    if simulation.ticks:
        for tick in simulation.ticks:
            chunk = tick.agent_chunk
            if chunk is None or chunk.is_tool_call():
                continue
            if (
                chunk.contains_speech is False
                and not chunk.is_audio
                and chunk.content
                and chunk.content.strip()
            ):
                return chunk.content
            return None
        return None
    for message in simulation.messages or []:
        if getattr(message, "role", None) != "assistant":
            continue
        content = getattr(message, "content", None)
        return content if content and content.strip() else None
    return None


def build_agent_turns(simulation: SimulationRun) -> list[AgentTurn]:
    """Return delivered agent utterances with preceding-user context only.

    ``interrupted`` is the union of the chunk-level signal
    (``utterance_interrupted``) and the tick-overlap barge-in detector.
    The detector's spans group agent chunks like the delivery judge
    (``agent_utterance_tick_spans``), which differs from this function's
    utterance grouping — so spans map onto turns by tick-range overlap,
    never by ordinal.
    """
    turns: list[AgentTurn] = []
    if simulation.ticks:
        barged_spans = barged_in_tick_spans(simulation)
        agent_events = _tick_utterances(simulation.ticks, "agent_chunk")
        user_events = _tick_utterances(simulation.ticks, "user_chunk")
        for agent_start, agent_end, text, interrupted in agent_events:
            prior = [
                user_text
                for _, user_end, user_text, _interrupted in user_events
                if user_end < agent_start
            ]
            turns.append(
                AgentTurn(
                    index=len(turns),
                    text=text,
                    preceding_user_text=prior[-1] if prior else None,
                    interrupted=interrupted
                    or tick_span_barged_in(agent_start, agent_end, barged_spans),
                )
            )
        return turns

    preceding_user: Optional[str] = None
    for message in simulation.messages or []:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "user" and content:
            preceding_user = content
        elif role == "assistant" and content:
            turns.append(
                AgentTurn(
                    index=len(turns),
                    text=content,
                    preceding_user_text=preceding_user,
                )
            )
    return turns


def _user_turns_and_speech_seconds(
    simulation: SimulationRun,
) -> tuple[list[str], Optional[float]]:
    """Delivered user turns plus tick-aligned speech duration when available."""
    if simulation.ticks:
        turns = [
            text
            for _start, _end, text, _interrupted in _tick_utterances(
                simulation.ticks, "user_chunk"
            )
        ]
        speech_ticks = [
            tick
            for tick in simulation.ticks
            if tick.user_chunk is not None
            and not tick.user_chunk.is_tool_call()
            and tick.user_chunk.contains_speech
        ]
        if any(tick.tick_duration_seconds is None for tick in speech_ticks):
            return turns, None
        return turns, sum(tick.tick_duration_seconds or 0.0 for tick in speech_ticks)

    turns = [
        message.content
        for message in simulation.messages or []
        if getattr(message, "role", None) == "user" and message.content
    ]
    return turns, None


def build_agent_corpus(simulation: SimulationRun) -> str:
    """All agent speech in the run, as one readable string (one line per utterance).

    Full-duplex/voice → merge per-tick chunks by utterance (reusing the
    communicate evaluator's chunk-merge) so the gold transcript reads as flowing
    speech instead of mid-word fragments. Half-duplex/text → assistant contents.

    Streaming voice runs store ``audio_script_gold`` as a marked-up chunk
    template; the judge must see only the plain text of the chunks whose audio
    was actually delivered — never the XML markup, and never text the caller
    interrupted before it was spoken (see ``extract_delivered_text``).
    """
    return "\n".join(f"agent: {turn.text}" for turn in build_agent_turns(simulation))


def aggregate_utterance_results(
    results: list[NativenessJudgeResult],
    *,
    strategy: Literal["any", "dose"],
) -> AggregatedNativenessResult:
    """Aggregate local verdicts without asking a second LLM to reinterpret them."""
    opportunities = [result for result in results if result.opportunity]
    if not opportunities:
        return AggregatedNativenessResult(
            outcome=JudgeOutcome.NO_OPPORTUNITY, severity=0
        )

    violations = [result for result in opportunities if result.violated]
    should_fail = (
        bool(violations)
        if strategy == "any"
        else (
            any(result.severity >= 3 for result in violations) or len(violations) >= 2
        )
    )
    if not should_fail:
        return AggregatedNativenessResult(
            outcome=JudgeOutcome.PASS,
            severity=0,
            violation_count=len(violations),
        )

    strongest = max(violations, key=lambda result: result.severity)
    severity = min(
        4,
        strongest.severity + (1 if len(violations) >= 2 else 0),
    )
    explanations = list(
        dict.fromkeys(
            result.reasoning.strip()
            for result in violations
            if result.reasoning.strip()
        )
    )
    return AggregatedNativenessResult(
        outcome=JudgeOutcome.FAIL,
        severity=severity,
        violation_count=len(violations),
        evidence="; ".join(explanations) or None,
        quote=strongest.quote.strip() or None,
    )


def _task_values(task: Task) -> set[str]:
    """Concrete task values (ids/numbers/emails), robust to free-text instructions.

    ``instruction_text`` expects structured instructions; tasks with a bare-string
    instruction would otherwise raise, so fall back to scanning the dumped text.
    """
    try:
        return extract_values(instruction_text(task.model_dump()))
    except (AttributeError, KeyError, TypeError):
        return extract_values(str(task.model_dump()))


def _deterministic_factors_for(
    language: str, *, is_voice: bool
) -> list[NativenessFactorConfig]:
    """Deterministic factors defined for this language, filtered by modality.

    Every factor carries its own modality profile (voice | text | both);
    the run mode selects the applicable subset. Universal deterministic
    factors are also selected explicitly by the language pack. English has no
    nativeness pack and retains both universal diagnostics.
    """
    candidates = [*DEFAULT_FACTORS, *TEXT_ONLY_FACTORS_BY_LANGUAGE.get(language, [])]
    enabled_defaults = enabled_deterministic_factor_ids_for(language)
    return [
        factor
        for factor in candidates
        if factor.modality.applies(is_voice=is_voice)
        and (factor not in DEFAULT_FACTORS or factor.id in enabled_defaults)
        and (
            factor.id != "backchannel_frequency"
            or language in BACKCHANNEL_FREQUENCY_SPECS
        )
    ]


def _criterion(factor: NativenessFactorConfig) -> NativenessJudgeCriterion:
    if not isinstance(factor.params, NativenessRubricText):
        raise ValueError(f"judge factor {factor.id!r} has no typed rubric")
    return NativenessJudgeCriterion.from_rubric(factor.id, factor.params)


def utterance_factor_batches(
    factors: list[NativenessFactorConfig],
) -> list[list[NativenessFactorConfig]]:
    """Return the exact runtime batches for utterance-level LLM factors."""
    shared = [
        factor for factor in factors if factor.id not in ISOLATED_UTTERANCE_FACTOR_IDS
    ]
    batches: list[list[NativenessFactorConfig]] = []
    shared_added = False
    for factor in factors:
        if factor.id in ISOLATED_UTTERANCE_FACTOR_IDS:
            batches.append([factor])
        elif not shared_added:
            batches.append(shared)
            shared_added = True
    return batches


def _call_customer_context(turns: list[AgentTurn]) -> Optional[str]:
    """Customer turns relevant to the delivered agent turns, in call order."""
    context: list[str] = []
    for turn in turns:
        text = turn.preceding_user_text
        if text and (not context or context[-1] != text):
            context.append(text)
    return "\n".join(f"customer: {text}" for text in context) or None


def checker_context_from_turns(turns: list[AgentTurn], language: str) -> CheckerContext:
    """Build the transcript-only checker context available to cold replay."""
    user_turns: list[str] = []
    for turn in turns:
        text = turn.preceding_user_text
        if text and (not user_turns or user_turns[-1] != text):
            user_turns.append(text)
    return CheckerContext(
        agent_text="\n".join(f"agent: {turn.text}" for turn in turns),
        agent_turns=[turn.text for turn in turns],
        agent_turn_interruptions=[turn.interrupted for turn in turns],
        user_turns=user_turns,
        has_email=False,
        language=language.lower(),
    )


def checker_context_for_simulation(
    simulation: SimulationRun,
    task: Task,
    language: str,
    *,
    turns: Optional[list[AgentTurn]] = None,
    agent_gender: Optional[ParticipantGender] = None,
    caller_gender: Optional[ParticipantGender] = None,
    caller_name: Optional[CallerNameContext] = None,
) -> CheckerContext:
    """The full simulation-backed checker context, exactly as scoring builds it.

    This is the one place the deterministic-checker context is assembled from a
    simulation + task; ``evaluate_nativeness`` and post-hoc recheck verbs share
    it so a recheck can never drift from what benchmark scoring saw. ``turns``
    accepts pre-built agent turns to avoid rebuilding them when the caller
    already has them.
    """
    is_voice = bool(simulation.ticks)
    turns = turns if turns is not None else build_agent_turns(simulation)
    user_turns, user_speech_seconds = _user_turns_and_speech_seconds(simulation)
    task_values = _task_values(task)
    return CheckerContext(
        agent_text="\n".join(f"agent: {turn.text}" for turn in turns),
        has_email=any("@" in value for value in task_values),
        language=language.lower(),
        agent_turns=[turn.text for turn in turns],
        agent_turn_interruptions=[turn.interrupted for turn in turns],
        user_turns=user_turns,
        user_speech_seconds=user_speech_seconds,
        email_symbols=email_symbols_for(language),
        is_voice=is_voice,
        allowed_literals=sorted(task_values),
        agent_gender=agent_gender,
        caller_gender=caller_gender,
        caller_name=caller_name,
        pinned_greeting=pinned_agent_greeting(simulation),
    )


class NativenessAggregates(BaseModel):
    """Score and outcome counts derived from a full set of factor checks."""

    score: Optional[float] = Field(
        default=None,
        description="Equal-weight pass fraction over fired binary factors; "
        "None when no factor had an opportunity.",
    )
    num_pass: int = Field(ge=0, description="Scored factors answered PASS.")
    num_fail: int = Field(ge=0, description="Scored factors answered FAIL.")
    num_no_opportunity: int = Field(
        ge=0, description="Scored factors with no opportunity."
    )
    num_deferred: int = Field(ge=0, description="Scored factors left DEFERRED.")
    num_errors: int = Field(ge=0, description="Scored factors that ERRORed.")
    score_coverage: float = Field(
        ge=0.0, le=1.0, description="Fraction of scored checks that fired."
    )


def aggregate_nativeness_checks(
    checks: list[NativenessFactorCheck],
) -> NativenessAggregates:
    """Derive the NativenessInfo aggregates from factor checks.

    The single aggregation rule shared by ``evaluate_nativeness`` and post-hoc
    factor rechecks: shadow checks are recorded but never influence the score,
    coverage, or outcome counts; the score is the equal-weight pass fraction of
    fired (PASS/FAIL) non-shadow checks.
    """
    scored = [check for check in checks if not check.shadow]
    fired = [
        check
        for check in scored
        if check.outcome in (JudgeOutcome.PASS, JudgeOutcome.FAIL)
    ]
    return NativenessAggregates(
        score=binary_pass_fraction(scored),
        num_pass=sum(check.outcome == JudgeOutcome.PASS for check in scored),
        num_fail=sum(check.outcome == JudgeOutcome.FAIL for check in scored),
        num_no_opportunity=sum(
            check.outcome == JudgeOutcome.NO_OPPORTUNITY for check in scored
        ),
        num_deferred=sum(check.outcome == JudgeOutcome.DEFERRED for check in scored),
        num_errors=sum(1 for c in scored if c.outcome == JudgeOutcome.ERROR),
        score_coverage=len(fired) / len(scored) if scored else 0.0,
    )


def _stored_unit_result(
    result: NativenessJudgeResult, unit_index: Optional[int]
) -> NativenessJudgeUnitResult:
    return NativenessJudgeUnitResult(
        unit_index=unit_index,
        opportunity=result.opportunity,
        violated=result.violated,
        severity=result.severity,
        reasoning=result.reasoning,
        quote=result.quote,
    )


def _direct_result(result: NativenessJudgeResult) -> AggregatedNativenessResult:
    if not result.opportunity:
        return AggregatedNativenessResult(
            outcome=JudgeOutcome.NO_OPPORTUNITY, severity=0
        )
    if not result.violated:
        return AggregatedNativenessResult(outcome=JudgeOutcome.PASS, severity=0)
    return AggregatedNativenessResult(
        outcome=JudgeOutcome.FAIL,
        severity=result.severity,
        violation_count=1,
        evidence=result.reasoning.strip() or None,
        quote=result.quote.strip() or None,
    )


def _factor_check(
    factor: NativenessFactorConfig,
    aggregate: AggregatedNativenessResult,
    *,
    units: Optional[list[NativenessJudgeUnitResult]] = None,
) -> NativenessFactorCheck:
    return NativenessFactorCheck(
        id=factor.id,
        category=factor.category,
        severity=factor.severity,
        outcome=aggregate.outcome,
        evidence=aggregate.evidence,
        quote=aggregate.quote,
        shadow=factor.shadow,
        evaluation_level=factor.evaluation_level,
        observed_severity=aggregate.severity,
        violation_count=aggregate.violation_count,
        unit_results=units or [],
    )


def evaluate_judge_factors(
    turns: list[AgentTurn],
    language: str,
    *,
    settings: NativenessJudgeSettings,
    agent_context: Optional[str] = None,
    factors: Optional[list[NativenessFactorConfig]] = None,
) -> JudgeFactorEvaluation:
    """Run the production LLM-factor path over already-extracted agent turns.

    The simulation harness and cold-workbook replay both call this function so
    calibration exercises the same batching, opportunity handling, and
    utterance-to-call aggregation used in benchmark scoring.
    """
    factors = [
        factor
        for factor in (factors if factors is not None else judge_factors_for(language))
        if factor.enabled and factor.type == "judge"
    ]
    checks_by_id: dict[str, NativenessFactorCheck] = {}
    judge_ran = False

    if not settings.llm_judge:
        for factor in factors:
            checks_by_id[factor.id] = _factor_check(
                factor,
                AggregatedNativenessResult(outcome=JudgeOutcome.DEFERRED, severity=0),
            )
    elif not turns:
        for factor in factors:
            checks_by_id[factor.id] = _factor_check(
                factor,
                AggregatedNativenessResult(
                    outcome=JudgeOutcome.NO_OPPORTUNITY, severity=0
                ),
            )
    else:
        # LLM judges see the interruption marker (raw text everywhere else).
        agent_text = "\n".join(f"agent: {turn.judged_text}" for turn in turns)
        call_factors = [
            factor for factor in factors if factor.evaluation_level == "call"
        ]
        if call_factors:
            request = NativenessJudgeInput(
                language=language.lower(),
                evaluation_level="call",
                criteria=[_criterion(factor) for factor in call_factors],
                agent_text=agent_text,
                customer_context=_call_customer_context(turns),
                agent_context=agent_context,
            )
            judge_ran = True
            try:
                results = run_nativeness_judge(
                    request,
                    model=settings.model,
                    model_args=settings.model_args,
                )
            except Exception as exc:
                logger.warning(
                    f"call-level nativeness judge failed ({language}): {exc}"
                )
                for factor in call_factors:
                    checks_by_id[factor.id] = _factor_check(
                        factor,
                        AggregatedNativenessResult(
                            outcome=JudgeOutcome.ERROR,
                            severity=0,
                            evidence=str(exc),
                        ),
                    )
            else:
                for factor, result in zip(call_factors, results, strict=True):
                    checks_by_id[factor.id] = _factor_check(
                        factor,
                        _direct_result(result),
                        units=[_stored_unit_result(result, None)],
                    )

        utterance_factors = [
            factor for factor in factors if factor.evaluation_level == "utterance"
        ]
        for factor_batch in utterance_factor_batches(utterance_factors):
            results_by_factor: dict[str, list[NativenessJudgeResult]] = {
                factor.id: [] for factor in factor_batch
            }
            units_by_factor: dict[str, list[NativenessJudgeUnitResult]] = {
                factor.id: [] for factor in factor_batch
            }
            batch_error: Optional[Exception] = None
            for turn in turns:
                request = NativenessJudgeInput(
                    language=language.lower(),
                    evaluation_level="utterance",
                    criteria=[_criterion(factor) for factor in factor_batch],
                    agent_text=turn.judged_text,
                    customer_context=turn.preceding_user_text,
                    agent_context=agent_context,
                )
                judge_ran = True
                try:
                    results = run_nativeness_judge(
                        request,
                        model=settings.model,
                        model_args=settings.model_args,
                    )
                except Exception as exc:
                    batch_error = exc
                    logger.warning(
                        f"utterance-level nativeness judge failed ({language}, "
                        f"turn {turn.index}, factors "
                        f"{[factor.id for factor in factor_batch]}): {exc}"
                    )
                    break
                for factor, result in zip(factor_batch, results, strict=True):
                    results_by_factor[factor.id].append(result)
                    units_by_factor[factor.id].append(
                        _stored_unit_result(result, turn.index)
                    )

            for factor in factor_batch:
                if batch_error is not None:
                    aggregate = AggregatedNativenessResult(
                        outcome=JudgeOutcome.ERROR,
                        severity=0,
                        evidence=str(batch_error),
                    )
                else:
                    if factor.aggregation not in {"any", "dose"}:
                        raise ValueError(
                            f"utterance factor {factor.id!r} has invalid "
                            f"aggregation {factor.aggregation!r}"
                        )
                    aggregate = aggregate_utterance_results(
                        results_by_factor[factor.id],
                        strategy=factor.aggregation,
                    )
                checks_by_id[factor.id] = _factor_check(
                    factor,
                    aggregate,
                    units=units_by_factor[factor.id],
                )

    return JudgeFactorEvaluation(
        checks=[checks_by_id[factor.id] for factor in factors],
        judge_ran=judge_ran,
    )


def precheck_factor_check(
    factor: NativenessFactorConfig,
    outcome: JudgeOutcome,
    evidence: Optional[str] = None,
) -> NativenessFactorCheck:
    """A factor check exactly as the hybrid-precheck path records it.

    FAIL carries the checker evidence at the factor's severity with one
    violation; NO_OPPORTUNITY carries nothing. Prechecks produce no other
    outcome (``evaluate_pack_judge_factors`` enforces the same contract).
    """
    if outcome is JudgeOutcome.FAIL:
        return _factor_check(
            factor,
            AggregatedNativenessResult(
                outcome=JudgeOutcome.FAIL,
                severity=factor.severity,
                violation_count=1,
                evidence=evidence,
            ),
        )
    if outcome is JudgeOutcome.NO_OPPORTUNITY:
        return _factor_check(
            factor,
            AggregatedNativenessResult(outcome=JudgeOutcome.NO_OPPORTUNITY, severity=0),
        )
    raise ValueError(
        f"precheck outcomes are FAIL or NO_OPPORTUNITY, got {outcome.value!r}"
    )


def evaluate_pack_judge_factors(
    turns: list[AgentTurn],
    language: str,
    *,
    settings: NativenessJudgeSettings,
    agent_context: Optional[str] = None,
    checker_context: Optional[CheckerContext] = None,
    factors: Optional[list[NativenessFactorConfig]] = None,
) -> JudgeFactorEvaluation:
    """Run hybrid prechecks and LLM fallbacks for pack-authored judge factors."""
    factors = [
        factor
        for factor in (factors if factors is not None else judge_factors_for(language))
        if factor.enabled and factor.type == "judge"
    ]
    context = checker_context or checker_context_from_turns(turns, language)
    checks_by_id: dict[str, NativenessFactorCheck] = {}

    for factor in (factor for factor in factors if factor.precheck_type is not None):
        checker = CHECKER_REGISTRY.get(factor.precheck_type)
        if checker is None:
            raise ValueError(
                f"hybrid factor {factor.id!r} references unknown precheck "
                f"{factor.precheck_type!r}"
            )
        outcome, evidence = checker(context)
        if outcome is JudgeOutcome.FAIL:
            checks_by_id[factor.id] = precheck_factor_check(
                factor, JudgeOutcome.FAIL, evidence
            )
        elif outcome is not JudgeOutcome.NO_OPPORTUNITY:
            raise ValueError(
                f"hybrid precheck {factor.precheck_type!r} returned {outcome.value}; "
                "prechecks may return only FAIL or NO_OPPORTUNITY"
            )

    llm_evaluation = evaluate_judge_factors(
        turns,
        language,
        settings=settings,
        agent_context=agent_context,
        factors=[factor for factor in factors if factor.id not in checks_by_id],
    )
    checks_by_id.update({check.id: check for check in llm_evaluation.checks})
    return JudgeFactorEvaluation(
        checks=[checks_by_id[factor.id] for factor in factors],
        judge_ran=llm_evaluation.judge_ran,
    )


def evaluate_nativeness(
    simulation: SimulationRun,
    task: Task,
    language: Optional[str],
    script: Optional[str],
    *,
    settings: NativenessJudgeSettings,
    agent_context: Optional[str] = None,
    agent_gender: Optional[ParticipantGender] = None,
    caller_gender: Optional[ParticipantGender] = None,
    caller_name: Optional[CallerNameContext] = None,
) -> Optional[NativenessInfo]:
    """Score one simulation's nativeness, or None if not applicable.

    Returns None only when no language is resolved. For an applicable run it
    always returns a ``NativenessInfo`` recording every factor's outcome; the
    aggregate ``score`` may still be None if no factor had an opportunity to
    fire. English has deterministic interaction factors but no pack-authored
    LLM factors.

    Pack-selected deterministic factors always run. Judge factors
    (``type="judge"``) run only when ``settings.llm_judge`` is True; otherwise
    they are recorded as DEFERRED (and so excluded from the score).
    ``agent_context`` carries the agent role and any explicitly resolved
    agent/customer grammatical gender. Typed Korean caller-name metadata is
    available only to local prechecks and is never serialized into the LLM
    prompt.
    """
    if not language:
        return None

    is_voice = bool(simulation.ticks)
    turns = build_agent_turns(simulation)
    base_ctx = checker_context_for_simulation(
        simulation,
        task,
        language,
        turns=turns,
        agent_gender=agent_gender,
        caller_gender=caller_gender,
        caller_name=caller_name,
    )

    # Modality filtering: a voice-only factor never scores a text transcript
    # and vice versa. Deterministic factors are pre-filtered by
    # _deterministic_factors_for; judge factors carry their catalog modality.
    factors = [
        factor
        for factor in _deterministic_factors_for(language.lower(), is_voice=is_voice)
        + judge_factors_for(language)
        if factor.enabled and factor.modality.applies(is_voice=is_voice)
    ]
    checks_by_id: dict[str, NativenessFactorCheck] = {}
    judge_ran = False

    for factor in (factor for factor in factors if factor.type != "judge"):
        if (checker := CHECKER_REGISTRY.get(factor.type)) is None:
            aggregate = AggregatedNativenessResult(
                outcome=JudgeOutcome.DEFERRED, severity=0
            )
        else:
            outcome, evidence = checker(base_ctx)
            aggregate = AggregatedNativenessResult(
                outcome=outcome,
                severity=factor.severity if outcome is JudgeOutcome.FAIL else 0,
                violation_count=1 if outcome is JudgeOutcome.FAIL else 0,
                evidence=evidence,
            )
        checks_by_id[factor.id] = _factor_check(factor, aggregate)

    judge_evaluation = evaluate_pack_judge_factors(
        turns,
        language,
        settings=settings,
        agent_context=agent_context,
        checker_context=base_ctx,
        factors=[factor for factor in factors if factor.type == "judge"],
    )
    judge_ran = judge_evaluation.judge_ran
    checks_by_id.update({check.id: check for check in judge_evaluation.checks})

    checks = [checks_by_id[factor.id] for factor in factors]

    # Shadow checks are judged and recorded (for cold-eval calibration) but
    # never influence the score, coverage, or outcome counts (see
    # aggregate_nativeness_checks).
    aggregates = aggregate_nativeness_checks(checks)
    return NativenessInfo(
        factor_checks=checks,
        language=language.lower(),
        script=script,
        rubric_version=NATIVENESS_RUBRIC_VERSION,
        **aggregates.model_dump(),
        # Judge configuration is stamped only when an LLM call was actually
        # made (judge enabled AND a non-empty transcript to judge).
        judge_model=settings.model if judge_ran else None,
        judge_args=dict(settings.model_args) if judge_ran else None,
        judge_prompt_version=NATIVENESS_JUDGE_PROMPT_VERSION if judge_ran else None,
    )
