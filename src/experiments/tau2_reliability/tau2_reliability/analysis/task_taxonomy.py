"""Reliability-driven task classification.

Classifies benchmark tasks by their reliability profile, identifying
the 'bimodal' tasks most informative for reliability research.
"""

from __future__ import annotations

from typing import Optional

from tau2_reliability.models import (
    ConsistencyMetrics,
    TaskReliabilityClass,
    TaskTrialData,
    TaxonomySummary,
)


def classify_tasks(
    task_data: list[TaskTrialData],
    consistency: ConsistencyMetrics,
) -> dict[str, TaskReliabilityClass]:
    """Classify each task by its reliability profile.

    Classification rules:
    - STABLE_PASS:  c_out > 0.8 AND pass_rate > 0.8
    - STABLE_FAIL:  c_out > 0.8 AND pass_rate < 0.2
    - BIMODAL:      c_out < 0.3
    - FRAGILE:      c_out > 0.8 AND c_traj_s < 0.5 (succeeds via different paths)
    - Else:         defaults to STABLE_PASS if pass_rate > 0.5, STABLE_FAIL otherwise
    """
    td_map = {td.task_id: td for td in task_data}
    classifications = {}

    for tid, metrics in consistency.per_task.items():
        c_out = metrics.get("c_out", 0.5)
        c_traj_s = metrics.get("c_traj_s", 0.5)

        td = td_map.get(tid)
        pass_rate = td.pass_rate if td else 0.5

        if c_out < 0.3:
            classifications[tid] = TaskReliabilityClass.BIMODAL
        elif c_out > 0.8 and pass_rate > 0.8:
            if c_traj_s < 0.5:
                classifications[tid] = TaskReliabilityClass.FRAGILE
            else:
                classifications[tid] = TaskReliabilityClass.STABLE_PASS
        elif c_out > 0.8 and pass_rate < 0.2:
            classifications[tid] = TaskReliabilityClass.STABLE_FAIL
        elif pass_rate > 0.5:
            # Moderate consistency, leans toward passing
            if c_traj_s < 0.5:
                classifications[tid] = TaskReliabilityClass.FRAGILE
            else:
                classifications[tid] = TaskReliabilityClass.STABLE_PASS
        else:
            classifications[tid] = TaskReliabilityClass.STABLE_FAIL

    return classifications


def compute_taxonomy_summary(
    classifications: dict[str, TaskReliabilityClass],
    task_data: Optional[list[TaskTrialData]] = None,
) -> TaxonomySummary:
    """Aggregate task classifications into a summary."""
    counts: dict[str, int] = {}
    for cls in TaskReliabilityClass:
        counts[cls.value] = 0
    for tid, cls in classifications.items():
        counts[cls.value] = counts.get(cls.value, 0) + 1

    bimodal_tasks = sorted(
        tid for tid, cls in classifications.items()
        if cls == TaskReliabilityClass.BIMODAL
    )

    return TaxonomySummary(
        classifications={tid: cls.value for tid, cls in classifications.items()},
        counts=counts,
        bimodal_tasks=bimodal_tasks,
    )
