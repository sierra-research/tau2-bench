"""
Robustness-specific metrics for the AVER evaluation mode.

Three-dimension scoring model:
    AVER Score = (Detection × 0.4 + Diagnosis × 0.2 + Recovery × 0.4) × 100

Detection: Did the agent notice the corrupted data? (temporal-aware)
Diagnosis: Did the agent understand what went wrong? (depth-based)
Recovery:  Did the agent complete the task correctly? (= τ²-bench reward)
"""

import math
from typing import Any, Optional

from pydantic import BaseModel, Field

# AVER score weights
WEIGHT_DETECTION = 0.4
WEIGHT_DIAGNOSIS = 0.2
WEIGHT_RECOVERY = 0.4


class RobustnessMetrics(BaseModel):
    """Per-simulation robustness metrics."""

    num_injections: int = Field(
        description="Number of errors injected in this simulation."
    )

    # Three AVER dimensions
    detection_score: Optional[float] = Field(
        description="Detection score (0.0-1.0) with temporal multiplier. "
        "None if no injections occurred.",
        default=None,
    )
    diagnosis_score: Optional[float] = Field(
        description="Diagnosis depth score (0.0-1.0). "
        "None if no injections occurred.",
        default=None,
    )
    recovery_score: float = Field(
        description="Recovery score = standard τ²-bench reward under injection (0.0-1.0)."
    )

    # Composite
    aver_score: Optional[float] = Field(
        description="AVER Score (0-100): (Det×0.4 + Diag×0.2 + Rec×0.4) × 100. "
        "None if no injections occurred.",
        default=None,
    )

    # Metadata
    temporal_pattern: Optional[str] = Field(
        description="'proactive', 'reactive', or 'none' — when detection happened.",
        default=None,
    )
    causal_chain_valid: Optional[bool] = Field(
        description="True if detection preceded recovery behavior (no gaming).",
        default=None,
    )
    is_negative_control: bool = Field(
        description="True if this was a negative control (no injection attempted).",
        default=False,
    )

    # Reward detail
    reward_breakdown: Optional[dict[str, float]] = Field(
        description="Per-component reward scores (e.g., DB, COMMUNICATE) under injection.",
        default=None,
    )
    per_injection_scores: Optional[list[dict[str, Any]]] = Field(
        description="Per-injection detection, diagnosis, and recovery scores.",
        default=None,
    )

    def compute_aver(self) -> None:
        """Compute composite AVER score from the three dimensions."""
        if self.detection_score is None or self.diagnosis_score is None:
            self.aver_score = None
            return
        self.aver_score = round(
            (
                self.detection_score * WEIGHT_DETECTION
                + self.diagnosis_score * WEIGHT_DIAGNOSIS
                + self.recovery_score * WEIGHT_RECOVERY
            )
            * 100,
            1,
        )


class AggregateRobustness(BaseModel):
    """Aggregate metrics across tasks — maps to leaderboard columns."""

    num_tasks: int = Field(description="Number of tasks evaluated.")
    num_trials: int = Field(description="Number of trials per task.")
    total_injections: int = Field(description="Total injections across all runs.")

    # Leaderboard headline
    pass_k_robust: dict[int, float] = Field(
        description="Pass^k under robustness injection (headline metric)."
    )

    # AVER Score decomposition
    avg_detection: float = Field(
        description="Average detection score (temporal-adjusted)."
    )
    avg_diagnosis: float = Field(
        description="Average diagnosis depth score."
    )
    avg_recovery: float = Field(
        description="Average recovery score (≈ avg task reward under injection)."
    )
    aver_score: float = Field(
        description="AVER Score (0-100): (Det×0.4 + Diag×0.2 + Rec×0.4) × 100."
    )

    # Calibration
    false_positive_rate: Optional[float] = Field(
        description="Fraction of negative controls with detection > 0.",
        default=None,
    )
    temporal_breakdown: dict[str, float] = Field(
        description="Percentage breakdown: proactive, reactive, none.",
        default_factory=dict,
    )

    # Detailed breakdowns (for report, not leaderboard)
    avg_reward_breakdown: Optional[dict[str, float]] = Field(
        description="Average per-component reward scores (DB, COMMUNICATE, etc.).",
        default=None,
    )


def compute_robustness_metrics(
    per_task_metrics: dict[str, list[RobustnessMetrics]],
    ks: list[int] = [1, 2, 4, 8],
) -> AggregateRobustness:
    """Compute aggregate robustness metrics from per-task, per-trial metrics.

    Args:
        per_task_metrics: Mapping of task_id → list of RobustnessMetrics (one per trial).
        ks: Values of k for Pass^k computation.

    Returns:
        AggregateRobustness with averaged metrics and Pass^k.
    """
    all_detection = []
    all_diagnosis = []
    all_recovery = []
    total_injections = 0

    # Temporal pattern tracking
    temporal_counts = {"proactive": 0, "reactive": 0, "none": 0}

    # Negative control tracking
    num_negative_controls = 0
    num_false_positives = 0

    # For Pass^k: count successes per task
    task_trials: dict[str, list[float]] = {}

    # For per-component reward averages
    component_sums: dict[str, float] = {}
    component_counts: dict[str, int] = {}

    for task_id, trials in per_task_metrics.items():
        task_trials[task_id] = []
        for m in trials:
            # Handle negative controls separately
            if m.is_negative_control:
                num_negative_controls += 1
                if m.detection_score is not None and m.detection_score > 0:
                    num_false_positives += 1
                continue

            total_injections += m.num_injections
            if m.detection_score is not None:
                all_detection.append(m.detection_score)
            if m.diagnosis_score is not None:
                all_diagnosis.append(m.diagnosis_score)
            all_recovery.append(m.recovery_score)

            if m.temporal_pattern and m.temporal_pattern in temporal_counts:
                temporal_counts[m.temporal_pattern] += 1

            task_trials[task_id].append(m.recovery_score)

            if m.reward_breakdown:
                for component, score in m.reward_breakdown.items():
                    component_sums[component] = (
                        component_sums.get(component, 0.0) + score
                    )
                    component_counts[component] = (
                        component_counts.get(component, 0) + 1
                    )

    avg_det = sum(all_detection) / len(all_detection) if all_detection else 0.0
    avg_diag = sum(all_diagnosis) / len(all_diagnosis) if all_diagnosis else 0.0
    avg_rec = sum(all_recovery) / len(all_recovery) if all_recovery else 0.0

    avg_breakdown = (
        {k: component_sums[k] / component_counts[k] for k in component_sums}
        if component_sums
        else None
    )

    # Compute Pass^k under injection
    pass_k = {}
    for k in ks:
        pass_k[k] = _compute_pass_hat_k(task_trials, k)

    # AVER score
    aver = round(
        (avg_det * WEIGHT_DETECTION + avg_diag * WEIGHT_DIAGNOSIS + avg_rec * WEIGHT_RECOVERY)
        * 100,
        1,
    )

    # Temporal breakdown (as percentages)
    total_temporal = sum(temporal_counts.values())
    temporal_pct = (
        {k: round(v / total_temporal, 2) for k, v in temporal_counts.items()}
        if total_temporal > 0
        else {"proactive": 0.0, "reactive": 0.0, "none": 0.0}
    )

    # False positive rate
    fpr = (
        num_false_positives / num_negative_controls
        if num_negative_controls > 0
        else None
    )

    num_trials = (
        max(len(v) for v in per_task_metrics.values()) if per_task_metrics else 0
    )

    return AggregateRobustness(
        num_tasks=len(per_task_metrics),
        num_trials=num_trials,
        total_injections=total_injections,
        pass_k_robust=pass_k,
        avg_detection=avg_det,
        avg_diagnosis=avg_diag,
        avg_recovery=avg_rec,
        aver_score=aver,
        false_positive_rate=fpr,
        temporal_breakdown=temporal_pct,
        avg_reward_breakdown=avg_breakdown,
    )


def _compute_pass_hat_k(
    task_trials: dict[str, list[float]],
    k: int,
) -> float:
    """Compute Pass^k metric: average over tasks of C(s,k)/C(n,k).

    A trial is successful if reward ≈ 1.0 (within 1e-6).
    """
    if not task_trials:
        return 0.0

    scores = []
    for task_id, rewards in task_trials.items():
        n = len(rewards)
        if n < k:
            scores.append(0.0)
            continue
        s = sum(1 for r in rewards if abs(r - 1.0) < 1e-6)
        if s < k:
            scores.append(0.0)
        else:
            scores.append(math.comb(s, k) / math.comb(n, k))

    return sum(scores) / len(scores) if scores else 0.0
