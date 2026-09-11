# Copyright Sierra
"""High-precision deterministic checks for the universal quality rubric."""

from typing import Callable, Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import JudgeOutcome, QualityJudgeSettings
from tau2.metrics.interaction_quality import InteractionQualityMetrics


class QualityCheckerResult(BaseModel):
    """Typed output from one deterministic quality checker."""

    outcome: JudgeOutcome = Field(description="Binary factor outcome.")
    evidence: Optional[str] = Field(
        default=None, description="Failure evidence, when the factor fails."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Raw measurements used by the check."
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Thresholds used by the check."
    )


CheckerFn = Callable[
    [InteractionQualityMetrics, QualityJudgeSettings], QualityCheckerResult
]


def check_responsiveness(
    metrics: InteractionQualityMetrics, settings: QualityJudgeSettings
) -> QualityCheckerResult:
    voice = metrics.voice
    if voice is None or voice.counts.response_total == 0:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    response_rate = voice.response_rate
    latency = voice.response_latency_mean
    failed = response_rate != 1.0 or (
        latency is not None and latency > settings.response_latency_seconds
    )
    raw_metrics = {
        "response_rate": response_rate or 0.0,
        "response_total": float(voice.counts.response_total),
    }
    if latency is not None:
        raw_metrics["response_latency_mean"] = latency
    evidence_parts = []
    if response_rate != 1.0:
        evidence_parts.append(f"response rate was {response_rate:.2%}")
    if latency is not None and latency > settings.response_latency_seconds:
        evidence_parts.append(f"mean response latency was {latency:.2f}s")
    return QualityCheckerResult(
        outcome=JudgeOutcome.FAIL if failed else JudgeOutcome.PASS,
        evidence="; ".join(evidence_parts) if failed else None,
        metrics=raw_metrics,
        thresholds={
            "minimum_response_rate": 1.0,
            "response_latency_seconds": settings.response_latency_seconds,
        },
    )


def check_yielding(
    metrics: InteractionQualityMetrics, _settings: QualityJudgeSettings
) -> QualityCheckerResult:
    voice = metrics.voice
    if voice is None or voice.counts.yield_total == 0:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    yield_rate = voice.yield_rate
    failed = yield_rate != 1.0
    raw_metrics = {
        "yield_rate": yield_rate or 0.0,
        "yield_total": float(voice.counts.yield_total),
    }
    if voice.yield_latency_mean is not None:
        raw_metrics["yield_latency_mean"] = voice.yield_latency_mean
    return QualityCheckerResult(
        outcome=JudgeOutcome.FAIL if failed else JudgeOutcome.PASS,
        evidence=(f"yield rate was {yield_rate:.2%}" if failed else None),
        metrics=raw_metrics,
        thresholds={"minimum_yield_rate": 1.0},
    )


def check_monologue(
    metrics: InteractionQualityMetrics, settings: QualityJudgeSettings
) -> QualityCheckerResult:
    value = metrics.max_agent_floor_hold_seconds
    if value is None:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    failed = value > settings.monologue_seconds
    return QualityCheckerResult(
        outcome=JudgeOutcome.FAIL if failed else JudgeOutcome.PASS,
        evidence=(
            f"longest uninterrupted agent floor hold was {value:.2f}s"
            if failed
            else None
        ),
        metrics={"max_agent_floor_hold_seconds": value},
        thresholds={"monologue_seconds": settings.monologue_seconds},
    )


def check_inappropriate_interruption(
    metrics: InteractionQualityMetrics, _settings: QualityJudgeSettings
) -> QualityCheckerResult:
    voice = metrics.voice
    if voice is None:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    count = voice.counts.agent_interrupts_count
    raw_metrics = {"agent_interrupts_count": float(count)}
    if voice.agent_interruption_rate is not None:
        raw_metrics["agent_interruption_rate"] = voice.agent_interruption_rate
    return QualityCheckerResult(
        outcome=JudgeOutcome.FAIL if count > 0 else JudgeOutcome.PASS,
        evidence=(
            f"agent speech began over caller speech {count} time(s)" if count else None
        ),
        metrics=raw_metrics,
        thresholds={"maximum_agent_interruption_rate": 0.0},
    )


def _check_selectivity(
    metrics: InteractionQualityMetrics,
    *,
    metric_field: str,
    total_field: str,
) -> QualityCheckerResult:
    voice = metrics.voice
    if voice is None:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    total = getattr(voice.counts, total_field)
    if total == 0:
        return QualityCheckerResult(outcome=JudgeOutcome.NO_OPPORTUNITY)
    value = getattr(voice, metric_field)
    failed = value != 1.0
    errors = round(total * (1.0 - value))
    return QualityCheckerResult(
        outcome=JudgeOutcome.FAIL if failed else JudgeOutcome.PASS,
        evidence=(f"mishandled {errors} of {total} event(s)" if failed else None),
        metrics={metric_field: value, total_field: float(total)},
        thresholds={f"minimum_{metric_field}": 1.0},
    )


def check_backchannel_selectivity(
    metrics: InteractionQualityMetrics, _settings: QualityJudgeSettings
) -> QualityCheckerResult:
    return _check_selectivity(
        metrics,
        metric_field="selectivity_backchannel",
        total_field="backchannel_total",
    )


def check_vocal_tic_selectivity(
    metrics: InteractionQualityMetrics, _settings: QualityJudgeSettings
) -> QualityCheckerResult:
    return _check_selectivity(
        metrics,
        metric_field="selectivity_vocal_tic",
        total_field="vocal_tic_total",
    )


def check_non_directed_selectivity(
    metrics: InteractionQualityMetrics, _settings: QualityJudgeSettings
) -> QualityCheckerResult:
    return _check_selectivity(
        metrics,
        metric_field="selectivity_non_directed",
        total_field="non_directed_total",
    )


CHECKER_REGISTRY: dict[str, CheckerFn] = {
    "responsiveness": check_responsiveness,
    "yielding": check_yielding,
    "inappropriate_interruption": check_inappropriate_interruption,
    "backchannel_selectivity": check_backchannel_selectivity,
    "vocal_tic_selectivity": check_vocal_tic_selectivity,
    "non_directed_selectivity": check_non_directed_selectivity,
    "monologue": check_monologue,
}
