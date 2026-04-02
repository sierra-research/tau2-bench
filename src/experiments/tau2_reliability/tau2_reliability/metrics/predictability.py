"""Predictability metrics for agent reliability evaluation.

All metrics return values in [0, 1] where higher = better calibrated/discriminating.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# P_cal: Calibration (1 - ECE)
# ---------------------------------------------------------------------------


def compute_p_cal(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Calibration: 1 - Expected Calibration Error.

    Bins confidence scores into n_bins equal-width bins and measures
    alignment between average confidence and average accuracy per bin.
    """
    if not confidences or len(confidences) != len(outcomes):
        return float("nan")

    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, out in zip(confidences, outcomes):
        conf = max(0.0, min(1.0, conf))
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, float(out)))

    n = len(confidences)
    ece = 0.0
    bin_details = []
    for bin_items in bins:
        if not bin_items:
            bin_details.append({"count": 0, "avg_conf": 0, "avg_acc": 0})
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(o for _, o in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(avg_acc - avg_conf)
        bin_details.append({
            "count": len(bin_items),
            "avg_conf": avg_conf,
            "avg_acc": avg_acc,
        })

    return 1.0 - ece


# ---------------------------------------------------------------------------
# P_auroc: Discrimination (AUC-ROC via Mann-Whitney U)
# ---------------------------------------------------------------------------


def compute_p_auroc(
    confidences: list[float],
    outcomes: list[bool],
) -> float:
    """Discrimination: AUC-ROC computed via Mann-Whitney U statistic.

    Measures whether the agent assigns higher confidence to tasks it succeeds on.
    Returns NaN if all outcomes are the same class.
    """
    if not confidences or len(confidences) != len(outcomes):
        return float("nan")

    positives = [c for c, o in zip(confidences, outcomes) if o]
    negatives = [c for c, o in zip(confidences, outcomes) if not o]

    if not positives or not negatives:
        return float("nan")

    concordant = 0
    tied = 0
    for p in positives:
        for n in negatives:
            if p > n:
                concordant += 1
            elif p == n:
                tied += 1

    return (concordant + 0.5 * tied) / (len(positives) * len(negatives))


# ---------------------------------------------------------------------------
# P_brier: Brier Score
# ---------------------------------------------------------------------------


def compute_p_brier(
    confidences: list[float],
    outcomes: list[bool],
) -> float:
    """Brier score: 1 - mean((confidence - outcome)^2).

    A proper scoring rule combining calibration and discrimination.
    Used as the aggregate predictability score (r_pred = p_brier).
    """
    if not confidences or len(confidences) != len(outcomes):
        return float("nan")

    brier = sum(
        (c - float(o)) ** 2 for c, o in zip(confidences, outcomes)
    ) / len(confidences)
    return 1.0 - brier


# ---------------------------------------------------------------------------
# Calibration bin details (for visualization)
# ---------------------------------------------------------------------------


def compute_calibration_bins(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute per-bin calibration data for reliability diagrams."""
    if not confidences:
        return []

    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, out in zip(confidences, outcomes):
        conf = max(0.0, min(1.0, conf))
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, float(out)))

    result = []
    for i, bin_items in enumerate(bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        if not bin_items:
            result.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": 0,
                "avg_confidence": (bin_lower + bin_upper) / 2,
                "avg_accuracy": 0.0,
            })
        else:
            result.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": len(bin_items),
                "avg_confidence": sum(c for c, _ in bin_items) / len(bin_items),
                "avg_accuracy": sum(o for _, o in bin_items) / len(bin_items),
            })
    return result
