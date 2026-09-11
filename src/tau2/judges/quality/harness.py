# Copyright Sierra
"""Per-simulation entrypoint for the universal binary quality rubric."""

from typing import Optional

from loguru import logger

from tau2.data_model.simulation import (
    JudgeOutcome,
    QualityFactorCheck,
    QualityInfo,
    QualityJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import Task
from tau2.judges.quality.checkers import CHECKER_REGISTRY
from tau2.judges.quality.factors import QUALITY_FACTORS, QUALITY_RUBRIC_VERSION
from tau2.judges.quality.judge import (
    QUALITY_JUDGE_PROMPT_VERSION,
    preclassified_quality_outcome,
    run_quality_factor_judge,
)
from tau2.metrics.interaction_quality import (
    QUALITY_METRICS_VERSION,
    extract_interaction_quality_metrics,
)


def evaluate_quality(
    simulation: SimulationRun,
    task: Task,
    *,
    domain: Optional[str],
    settings: QualityJudgeSettings,
) -> QualityInfo:
    """Evaluate every catalog factor and return a provenance-bearing score."""
    metrics = extract_interaction_quality_metrics(simulation, domain=domain)
    checks: list[QualityFactorCheck] = []
    for factor in QUALITY_FACTORS:
        if factor.evaluator == "deterministic":
            checker = CHECKER_REGISTRY[factor.id]
            result = checker(metrics, settings)
            checks.append(
                QualityFactorCheck(
                    id=factor.id,
                    category=factor.category,
                    severity=factor.severity,
                    evaluator="deterministic",
                    outcome=result.outcome,
                    evidence=result.evidence,
                    metrics=result.metrics,
                    thresholds=result.thresholds,
                )
            )
            continue

        preclassified = preclassified_quality_outcome(factor, metrics)
        if preclassified is not None:
            checks.append(
                QualityFactorCheck(
                    id=factor.id,
                    category=factor.category,
                    severity=factor.severity,
                    evaluator=factor.evaluator,
                    outcome=preclassified,
                )
            )
            continue
        if not settings.llm_judge:
            checks.append(
                QualityFactorCheck(
                    id=factor.id,
                    category=factor.category,
                    severity=factor.severity,
                    evaluator=factor.evaluator,
                    outcome=JudgeOutcome.DEFERRED,
                )
            )
            continue

        model = settings.model or factor.judge_model
        if model is None:  # guarded by catalog validation; keeps typing honest
            raise ValueError(f"{factor.id} has no configured quality judge model")
        model_args = dict(factor.judge_args)
        model_args.update(settings.model_args)
        try:
            result = run_quality_factor_judge(
                simulation,
                task,
                factor,
                domain=domain,
                model=model,
                model_args=model_args,
            )
            outcome = result.outcome
            evidence = result.evidence
            quote = result.quote
        except Exception as exc:  # noqa: BLE001 - one factor fails locally
            outcome = JudgeOutcome.ERROR
            evidence = str(exc)
            quote = None
            logger.warning(
                f"quality judge {factor.id} failed for sim {simulation.id}: {exc}"
            )
        checks.append(
            QualityFactorCheck(
                id=factor.id,
                category=factor.category,
                severity=factor.severity,
                evaluator=factor.evaluator,
                outcome=outcome,
                evidence=evidence,
                quote=quote,
                judge_model=model,
                judge_args=model_args,
                judge_prompt_version=QUALITY_JUDGE_PROMPT_VERSION,
            )
        )

    fired = [
        check
        for check in checks
        if check.outcome in (JudgeOutcome.PASS, JudgeOutcome.FAIL)
    ]
    score = (
        sum(check.outcome == JudgeOutcome.PASS for check in fired) / len(fired)
        if fired
        else None
    )
    return QualityInfo(
        score=score,
        factor_checks=checks,
        rubric_version=QUALITY_RUBRIC_VERSION,
        metrics_version=QUALITY_METRICS_VERSION,
        num_pass=sum(check.outcome == JudgeOutcome.PASS for check in checks),
        num_fail=sum(check.outcome == JudgeOutcome.FAIL for check in checks),
        num_no_opportunity=sum(
            check.outcome == JudgeOutcome.NO_OPPORTUNITY for check in checks
        ),
        num_deferred=sum(check.outcome == JudgeOutcome.DEFERRED for check in checks),
        num_errors=sum(check.outcome == JudgeOutcome.ERROR for check in checks),
        score_coverage=len(fired) / len(checks) if checks else 0.0,
    )
