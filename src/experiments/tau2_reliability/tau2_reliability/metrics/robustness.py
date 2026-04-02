"""Robustness metrics for agent reliability evaluation.

Robustness measures accuracy degradation under perturbations.
All ratios are capped at 1.0 (can't be "more robust than perfect").
"""

from __future__ import annotations

import numpy as np

from tau2_reliability.models import BootstrapResult


def compute_robustness_ratio(
    baseline_accuracy: float,
    perturbed_accuracy: float,
) -> float:
    """R = min(Acc_perturbed / Acc_baseline, 1.0).

    Returns 1.0 if baseline is zero (degenerate case).
    """
    if baseline_accuracy <= 0:
        return 1.0
    return min(perturbed_accuracy / baseline_accuracy, 1.0)


def bootstrap_robustness_ratio(
    baseline_outcomes: list[bool],
    perturbed_outcomes: list[bool],
    n_resamples: int = 200,
    seed: int = 42,
) -> BootstrapResult:
    """Compute robustness ratio with bootstrap confidence interval.

    Resamples task-level outcomes with replacement.
    """
    rng = np.random.default_rng(seed)
    n = min(len(baseline_outcomes), len(perturbed_outcomes))
    if n == 0:
        return BootstrapResult(
            point_estimate=1.0, standard_error=0.0, ci_lower=1.0, ci_upper=1.0
        )

    base_acc = sum(baseline_outcomes[:n]) / n
    pert_acc = sum(perturbed_outcomes[:n]) / n
    point = compute_robustness_ratio(base_acc, pert_acc)

    ratios = []
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        b_acc = sum(baseline_outcomes[i] for i in indices) / n
        p_acc = sum(perturbed_outcomes[i] for i in indices) / n
        ratios.append(compute_robustness_ratio(b_acc, p_acc))

    se = float(np.std(ratios, ddof=1))
    ci_lower = float(np.percentile(ratios, 2.5))
    ci_upper = float(np.percentile(ratios, 97.5))

    return BootstrapResult(
        point_estimate=point,
        standard_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
