# Copyright Sierra
"""Per-simulation audio delivery scoring (fidelity + intonation + factors).

``evaluate_delivery`` runs the combined multimodal judge over ALL agent
utterances of a *sampled subset of conversations* (sierra-style per-conversation
sampling; ``sample_rate`` is the fraction of sims judged) and aggregates
per-axis, severity-weighted "clean" fractions. The language's delivery rubric
(pack factors from the closed catalog, or the fixed native-listener fallback)
is built once per sim and rides every judge call; pack-mode factor verdicts are
folded into sim-level ``factor_checks`` + ``factor_score`` (nativeness-shaped).
Voice-only: returns None when the run has no ticks (no audio), and None for
conversations that were not sampled.

Scoring — for each judged (non-ERROR) utterance, per axis, take the worst finding
severity on that axis (0 if none) and map it to a cleanliness in [0, 1]:
``clean = 1 - worst_severity / 3`` (sev0->1.0, sev1->0.67, sev2->0.33, sev3->0.0).
An axis score is the mean cleanliness over judged utterances (None if none judged);
the overall score uses each utterance's worst severity across both axes.
"""

import hashlib
from typing import TYPE_CHECKING, Callable, Optional

from loguru import logger

from tau2.data_model.simulation import (
    DeliveryAxis,
    DeliveryFactorCheck,
    DeliveryInfo,
    DeliveryJudgeSettings,
    DeliveryUtteranceResult,
    JudgeOutcome,
    SimulationRun,
)
from tau2.judges.base import binary_pass_fraction, ordered_map
from tau2.judges.delivery.audio import message_audio_to_wav_b64
from tau2.judges.delivery.factors import DeliveryFactorConfig, build_delivery_rubric
from tau2.judges.delivery.judge import (
    DELIVERY_JUDGE_PROMPT_VERSION,
    run_delivery_judge,
)
from tau2.metrics.interaction_quality import (
    agent_utterance_tick_spans,
    barge_in_utterance_indices,
    utterance_interrupted,
)

if TYPE_CHECKING:
    from tau2.data_model.message import Message


def _sample_value(key: str) -> float:
    """Deterministic hash of ``key`` to [0, 1) — reproducible across runs/resumes."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _even_spread_indices(n: int, k: int) -> list[int]:
    """``k`` list positions spread evenly across ``range(n)`` (deterministic).

    Always includes the first and last position (for k >= 2), with the rest at
    a fixed index stride — so a capped long call is sampled across its whole
    duration instead of keeping only the opening utterances (opening bias).
    No randomness: the same (n, k) always selects the same positions.
    """
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0] if k == 1 else []
    # Strictly increasing positions; step (n-1)/(k-1) > 1 since k < n, so
    # rounding can never collide two positions.
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _cleanliness(worst_severity: int) -> float:
    """Map a worst-finding severity in {0,1,2,3} to cleanliness in [0, 1]."""
    return 1.0 - max(0, min(3, worst_severity)) / 3.0


def _axis_score(
    results: list[DeliveryUtteranceResult], axis: DeliveryAxis
) -> Optional[float]:
    """Mean cleanliness on one axis over successfully-judged utterances."""
    judged = [r for r in results if r.outcome != JudgeOutcome.ERROR]
    if not judged:
        return None
    total = 0.0
    for r in judged:
        worst = max(
            (f.severity for f in r.findings if f.axis == axis),
            default=0,
        )
        total += _cleanliness(worst)
    return total / len(judged)


_OUTCOME_PRECEDENCE = [
    JudgeOutcome.FAIL,
    JudgeOutcome.PASS,
    JudgeOutcome.ERROR,
    JudgeOutcome.NO_OPPORTUNITY,
]


def _aggregate_factor_checks(
    results: list[DeliveryUtteranceResult],
    factors: list[DeliveryFactorConfig],
) -> list[DeliveryFactorCheck]:
    """Sim-level verdict per rubric factor, folded over the judged utterances.

    Mirrors nativeness' one-check-per-factor-per-sim shape so delivery factor
    verdicts flow into the same calibration machinery. Precedence: any FAIL on
    any utterance fails the factor (evidence/quote from the first failing
    utterance); else any PASS passes it; else any ERROR is surfaced (a factor
    with only failed judge calls has no verdict); else NO_OPPORTUNITY.
    """
    checks: list[DeliveryFactorCheck] = []
    for factor in factors:
        per_utterance = [
            c
            for r in results
            if r.outcome != JudgeOutcome.ERROR
            for c in r.factor_checks
            if c.id == factor.id
        ]
        if not per_utterance:
            continue
        outcome = next(
            (
                o
                for o in _OUTCOME_PRECEDENCE
                if any(c.outcome == o for c in per_utterance)
            ),
            JudgeOutcome.NO_OPPORTUNITY,
        )
        exemplar = next(c for c in per_utterance if c.outcome == outcome)
        checks.append(
            DeliveryFactorCheck(
                id=factor.id,
                category=factor.category,
                severity=factor.severity,
                outcome=outcome,
                evidence=exemplar.evidence,
                quote=exemplar.quote,
                shadow=factor.shadow,
            )
        )
    return checks


# Control token the full-duplex agent appends on transfer_to_human — part of
# the stored message content but never synthesized, so it must not reach the
# judge as expected speech.
_STOP_TOKEN = "###STOP###"


def _delivered_reference(msg: "Message") -> tuple[str, bool]:
    """The reference text actually delivered, plus whether it was interrupted.

    Streaming voice runs store ``audio_script_gold`` as a marked-up chunk
    template; the judge must see only the plain text of the *delivered* chunks,
    so a caller barge-in never reads as "missing words" on the fidelity axis.
    See ``extract_delivered_text`` for the format and parsing rules.
    """
    # Lazy import: the agent package is heavy and only needed here.
    from tau2.agent.base.streaming_utils import extract_delivered_text

    reference, markup_interrupted = extract_delivered_text(
        msg.audio_script_gold or msg.content or ""
    )
    interrupted = markup_interrupted or utterance_interrupted([msg])
    return reference.replace(_STOP_TOKEN, "").strip(), interrupted


def _chunk_interrupted_utterance_indices(
    simulation: SimulationRun, spans: list[tuple[int, int]]
) -> set[int]:
    """Utterance indices cut according to their original, unmerged chunks.

    ``ticks_to_message_history`` merges multi-chunk utterances for delivery and
    the generic message merger does not preserve provider ``raw_data``. Read
    this legacy cutoff signal before that lossy boundary;
    ``spans`` is index-aligned with the merged delivery messages.
    """
    agent_chunks = [
        (tick.tick_id, tick.agent_chunk)
        for tick in simulation.ticks or []
        if tick.agent_chunk is not None and not tick.agent_chunk.is_tool_call()
    ]
    interrupted: set[int] = set()
    chunk_index = 0
    for idx, (start, end) in enumerate(spans):
        while chunk_index < len(agent_chunks) and agent_chunks[chunk_index][0] < start:
            chunk_index += 1
        chunks = []
        while chunk_index < len(agent_chunks) and agent_chunks[chunk_index][0] <= end:
            chunks.append(agent_chunks[chunk_index][1])
            chunk_index += 1
        if utterance_interrupted(chunks):
            interrupted.add(idx)
    return interrupted


def evaluate_delivery(
    simulation: SimulationRun,
    language: Optional[str],
    *,
    locale: Optional[str] = None,
    settings: Optional[DeliveryJudgeSettings] = None,
    seed: int = 0,
    wav_source: Optional[Callable[[int, "Message"], Optional[str]]] = None,
) -> Optional[DeliveryInfo]:
    """Score one voice simulation's audio delivery, or None if not applicable.

    Returns None for non-voice runs (no ticks) and for conversations that were
    not sampled. For a sampled voice run it always returns a ``DeliveryInfo``
    (possibly with all-None scores if nothing was judged). Judge errors are
    recorded per-utterance as ERROR (excluded from scoring) and logged, never
    aborting the batch. Judge calls run concurrently (results stay ordered by
    utterance index).

    ``wav_source`` is the audio seam: ``(utterance_idx, message) -> wav_b64``
    (or None for an utterance with no audio). The default reads the in-memory
    ``audio_content`` (inline judging at run time); rejudge-from-disk passes
    ``SimAgentAudio.wav_b64_for``, which slices the stored ``both.wav`` agent
    channel by tick span. Everything downstream — eligibility, sampling,
    judging, scoring — is identical for both sources.
    """
    if not simulation.ticks:
        return None
    if wav_source is None:
        wav_source = lambda _idx, msg: message_audio_to_wav_b64(msg)  # noqa: E731

    settings = settings if settings is not None else DeliveryJudgeSettings()
    sample_rate = settings.sample_rate

    # Per-CONVERSATION sampling: a sampled sim gets ALL its utterances judged
    # (trustworthy per-sim scores); an unsampled one is skipped entirely.
    # Deterministic hash of the sim id → stable sampling across re-runs/resume.
    if sample_rate < 1.0 and _sample_value(f"{simulation.id}:{seed}") >= sample_rate:
        logger.debug(f"delivery: sim {simulation.id} not sampled (rate {sample_rate})")
        return None

    # Lazy import (evaluator_communicate does not import judges → no cycle).
    from tau2.evaluator.evaluator_communicate import FullDuplexCommunicateEvaluator

    agent_messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(
        simulation.ticks
    )

    # Three truncation-signal families, OR'd: chunk accounting (inactive
    # trailing gold chunks or legacy per-chunk ``raw_data.was_truncated``),
    # the shared tick-overlap barge-in detector
    # (tau2.metrics.interaction_quality) — provider-cancelled tails the chunk
    # accounting misses because every emitted chunk stays active — and the
    # recording ending mid-speech: a call that hits its ceiling or hangs up
    # while the agent is talking stops the audio mid-word with no barge-in
    # anywhere (the tick-span slice clamps at the audio tail), which the
    # recall_20 calibration measured as a fidelity false-positive class.
    # Charge that cut to the call, not the synthesis.
    barged_in = barge_in_utterance_indices(simulation)
    spans = agent_utterance_tick_spans(simulation)
    chunk_interrupted = _chunk_interrupted_utterance_indices(simulation, spans)
    last_tick_id = simulation.ticks[-1].tick_id
    cut_at_recording_end = {
        idx for idx, (_start, end) in enumerate(spans) if end >= last_tick_id
    }

    eligible: list[tuple[int, str, str, bool]] = []  # (idx, wav_b64, ref, interrupted)
    for idx, msg in enumerate(agent_messages):
        wav_b64 = wav_source(idx, msg)
        if wav_b64 is None:
            continue
        # Skip non-speech audio: discrete-time padding ticks carry silence with
        # no script gold or content (contains_speech=False). Judging them wastes
        # budget and would skew aggregates with non-speech clips — mirror the
        # nativeness corpus, which only collects utterances with text.
        reference, interrupted = _delivered_reference(msg)
        if not reference.strip():
            continue
        eligible.append(
            (
                idx,
                wav_b64,
                reference,
                interrupted
                or idx in chunk_interrupted
                or idx in barged_in
                or idx in cut_at_recording_end,
            )
        )

    # Cost cap: a deterministic even spread across the whole conversation
    # (index-stride, first and last always included) — never "first N", which
    # would bias long calls toward their opening.
    n_with_audio = len(eligible)
    selected = [
        eligible[p] for p in _even_spread_indices(n_with_audio, settings.max_segments)
    ]
    n_capped = n_with_audio - len(selected)

    # Built ONCE per sim: the language's delivery rubric (pack factors, or the
    # fixed native-listener fallback). Every judged utterance of the sim sees
    # the same rubric; its source is stamped onto DeliveryInfo as provenance.
    rubric = build_delivery_rubric(language, locale)

    def _judge_one(item: tuple[int, str, str, bool]) -> DeliveryUtteranceResult:
        idx, wav_b64, reference, interrupted = item
        try:
            return run_delivery_judge(
                wav_b64,
                reference,
                utterance_idx=idx,
                rubric=rubric,
                was_interrupted=interrupted,
                model=settings.model,
                model_args=settings.model_args,
            )
        except Exception as exc:  # noqa: BLE001 - record, never abort the batch
            logger.warning(
                f"delivery judge failed for utterance {idx} "
                f"(sim {simulation.id}): {exc}"
            )
            return DeliveryUtteranceResult(
                utterance_idx=idx,
                was_interrupted=interrupted,
                expected_text=(reference[:200] or None),
                outcome=JudgeOutcome.ERROR,
                summary=str(exc),
            )

    # ordered_map preserves input order → results stay sorted by utterance idx.
    results = ordered_map(_judge_one, selected, settings.concurrency)

    if n_capped:
        logger.info(
            f"delivery: sim {simulation.id} — {len(results)} judged of "
            f"{n_with_audio} audio utterances (capped {n_capped})"
        )

    # Scoring excludes ERROR outcomes, but num_judged counts every utterance
    # actually sent to the judge.
    scored = [r for r in results if r.outcome != JudgeOutcome.ERROR]
    if scored:
        overall = sum(_cleanliness(r.severity) for r in scored) / len(scored)
    else:
        overall = None

    # Sim-level factor verdicts + score (pack-rubric mode only; empty/None in
    # fallback and language-generic modes).
    factor_checks = _aggregate_factor_checks(results, rubric.enabled_factors)
    scored_factor_checks = [check for check in factor_checks if not check.shadow]

    # Judge configuration is stamped only when at least one utterance was
    # actually sent to the judge (an LLM call was made).
    judge_ran = bool(selected)
    return DeliveryInfo(
        score=overall,
        fidelity_score=_axis_score(results, "fidelity"),
        intonation_score=_axis_score(results, "intonation"),
        num_judged=len(results),
        num_errors=len(results) - len(scored),
        num_flagged=sum(1 for r in scored if r.flag_for_review),
        utterance_results=results,
        factor_score=binary_pass_fraction(scored_factor_checks),
        factor_checks=factor_checks,
        rubric_source=rubric.source if judge_ran else None,
        language=language.lower() if language else None,
        locale=locale,
        judge_model=settings.model if judge_ran else None,
        judge_args=dict(settings.model_args) if judge_ran else None,
        judge_prompt_version=DELIVERY_JUDGE_PROMPT_VERSION if judge_ran else None,
        sample_rate=sample_rate,
        max_segments=settings.max_segments,
        seed=seed,
    )
