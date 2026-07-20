#!/usr/bin/env python3
"""Analyze uncertainty for BFCL v4 non-live candidate shifts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_bfcl_shift_inventory import (  # noqa: E402, I001
    DATASET_NAME,
    DEFAULT_INPUT_JSONL,
    DEFAULT_OUTPUT_JSONL as DEFAULT_INVENTORY_JSONL,
    DEFAULT_OUTPUT_SUMMARY_JSON as DEFAULT_INVENTORY_SUMMARY_JSON,
    EXPLORATORY_WARNING,
    LABEL_ORIGIN,
    LABEL_SCOPE,
    MODEL_NAME,
    NO_IID_MIXING_WARNING,
    OUTCOME_TYPE,
    SOURCE_DATASET,
    build_outputs as build_inventory_outputs,
    load_jsonl,
    write_json,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_shift_uncertainty.jsonl"
)
DEFAULT_OUTPUT_SUMMARY_JSON = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_shift_uncertainty_summary.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs/bfcl_shift_uncertainty.md"

PRACTICAL_THRESHOLDS = (0.05, 0.10, 0.15)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 1
Z_95 = 1.959963984540054
PRIMARY_CI_METHOD = "Newcombe-Wilson score interval for difference in proportions"
BOOTSTRAP_CI_METHOD = "deterministic nonparametric bootstrap percentile interval"
MULTIPLE_TESTING_METHOD = "Benjamini-Hochberg false-discovery-rate adjustment"
NON_INDEPENDENCE_WARNING = (
    "BFCL category groups are disjoint within each contrast, but candidate "
    "contrasts reuse categories across rows and should not be treated as "
    "independent discoveries."
)
NO_DEPLOYMENT_CLAIM_WARNING = (
    "Classifications are exploratory statistical labels only; they do not imply "
    "deployment safety or a retraining requirement."
)


def finite_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    p_hat = successes / n
    denominator = 1.0 + (z * z / n)
    center = (p_hat + (z * z / (2.0 * n))) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z / (4.0 * n * n)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def newcombe_wilson_delta_ci(
    source_positive: int,
    source_n: int,
    target_positive: int,
    target_n: int,
) -> tuple[float, float]:
    source_rate = source_positive / source_n
    target_rate = target_positive / target_n
    source_low, source_high = wilson_interval(source_positive, source_n)
    target_low, target_high = wilson_interval(target_positive, target_n)
    delta = target_rate - source_rate
    lower = delta - math.sqrt(
        (target_rate - target_low) ** 2 + (source_high - source_rate) ** 2
    )
    upper = delta + math.sqrt(
        (target_high - target_rate) ** 2 + (source_rate - source_low) ** 2
    )
    return max(-1.0, lower), min(1.0, upper)


def bootstrap_delta_ci(
    source_y: np.ndarray,
    target_y: np.ndarray,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    source_draws = rng.integers(0, source_y.size, size=(replicates, source_y.size))
    target_draws = rng.integers(0, target_y.size, size=(replicates, target_y.size))
    deltas = target_y[target_draws].mean(axis=1) - source_y[source_draws].mean(axis=1)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return float(lower), float(upper)


def expected_counts(table: np.ndarray) -> np.ndarray:
    total = table.sum()
    if total == 0:
        return np.zeros_like(table, dtype=np.float64)
    return np.outer(table.sum(axis=1), table.sum(axis=0)) / total


def proportions_p_value(
    source_positive: int,
    source_n: int,
    target_positive: int,
    target_n: int,
) -> tuple[float, str]:
    table = np.asarray(
        [
            [source_positive, source_n - source_positive],
            [target_positive, target_n - target_positive],
        ],
        dtype=np.float64,
    )
    if np.any(expected_counts(table) < 5.0):
        return (
            float(stats.fisher_exact(table, alternative="two-sided").pvalue),
            "fisher_exact",
        )

    pooled = (source_positive + target_positive) / (source_n + target_n)
    standard_error = math.sqrt(
        pooled * (1.0 - pooled) * ((1.0 / source_n) + (1.0 / target_n))
    )
    if standard_error == 0.0:
        return 1.0, "two_proportion_z_test"
    z_value = ((target_positive / target_n) - (source_positive / source_n)) / (
        standard_error
    )
    return float(2.0 * stats.norm.sf(abs(z_value))), "two_proportion_z_test"


def risk_ratio(source_rate: float, target_rate: float) -> float | None:
    if source_rate == 0.0:
        return None
    return finite_float(target_rate / source_rate)


def odds_ratio_with_correction(
    source_positive: int,
    source_n: int,
    target_positive: int,
    target_n: int,
) -> tuple[float | None, bool]:
    source_negative = source_n - source_positive
    target_negative = target_n - target_positive
    cells = np.asarray(
        [source_positive, source_negative, target_positive, target_negative],
        dtype=np.float64,
    )
    corrected = bool(np.any(cells == 0.0))
    if corrected:
        cells = cells + 0.5
    source_positive_f, source_negative_f, target_positive_f, target_negative_f = cells
    denominator = source_positive_f * target_negative_f
    if denominator == 0.0:
        return None, corrected
    return (
        finite_float((target_positive_f * source_negative_f) / denominator),
        corrected,
    )


def bh_adjusted_p_values(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running_min = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = m - reverse_rank + 1
        value = min(1.0, p_values[index] * m / rank)
        running_min = min(running_min, value)
        adjusted[index] = max(p_values[index], running_min)
    return adjusted


def classify_delta_ci(ci: tuple[float, float], threshold: float) -> str:
    lower, upper = ci
    if upper < -threshold:
        return "candidate_harmful"
    if lower >= -threshold and upper <= threshold:
        return "candidate_harmless"
    if lower > threshold:
        return "candidate_beneficial"
    return "inconclusive"


def classification_by_threshold(ci: tuple[float, float]) -> dict[str, str]:
    return {
        f"{threshold:.2f}": classify_delta_ci(ci, threshold)
        for threshold in PRACTICAL_THRESHOLDS
    }


def classification_stability(classifications: dict[str, str]) -> str:
    if len(set(classifications.values())) == 1:
        return "stable_across_thresholds"
    return "changes_with_threshold"


def source_target_rows(
    shift: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_category = shift["source_group"]
    target_category = shift["target_group"]
    return (
        [row for row in rows if row["category"] == source_category],
        [row for row in rows if row["category"] == target_category],
    )


def analyze_shift(
    shift: dict[str, Any],
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    source_y = np.asarray([row["y"] for row in source_rows], dtype=np.float64)
    target_y = np.asarray([row["y"] for row in target_rows], dtype=np.float64)
    source_n = len(source_rows)
    target_n = len(target_rows)
    source_positive = int(source_y.sum())
    target_positive = int(target_y.sum())
    source_rate = float(source_y.mean())
    target_rate = float(target_y.mean())
    delta_y = target_rate - source_rate
    delta_ci = newcombe_wilson_delta_ci(
        source_positive,
        source_n,
        target_positive,
        target_n,
    )
    bootstrap_ci = bootstrap_delta_ci(source_y, target_y, rng)
    raw_p_value, p_value_method = proportions_p_value(
        source_positive,
        source_n,
        target_positive,
        target_n,
    )
    odds_ratio, odds_ratio_corrected = odds_ratio_with_correction(
        source_positive,
        source_n,
        target_positive,
        target_n,
    )
    source_ids = {row["id"] for row in source_rows}
    target_ids = {row["id"] for row in target_rows}
    classifications = classification_by_threshold(delta_ci)
    warnings = [
        EXPLORATORY_WARNING,
        NO_IID_MIXING_WARNING,
        NON_INDEPENDENCE_WARNING,
        NO_DEPLOYMENT_CLAIM_WARNING,
    ]
    if odds_ratio_corrected:
        warnings.append("zero 2x2 table cell; odds ratio uses 0.5 correction")

    return {
        "shift_id": shift["shift_id"],
        "shift_family": shift["family"],
        "shift_type": shift["shift_type"],
        "is_primary_complexity_shift": shift["is_primary_complexity_shift"],
        "source_rule": {
            "name": shift["source_group"],
            "grouping_rule": shift["grouping_rule"],
            "fields": shift["group_definition_fields"],
        },
        "target_rule": {
            "name": shift["target_group"],
            "grouping_rule": shift["grouping_rule"],
            "fields": shift["group_definition_fields"],
        },
        "source_n": source_n,
        "target_n": target_n,
        "source_positive": source_positive,
        "target_positive": target_positive,
        "source_success_rate": source_rate,
        "target_success_rate": target_rate,
        "delta_y": delta_y,
        "delta_y_ci_method": PRIMARY_CI_METHOD,
        "delta_y_ci_95": [float(delta_ci[0]), float(delta_ci[1])],
        "bootstrap_delta_y_ci_method": BOOTSTRAP_CI_METHOD,
        "bootstrap_delta_y_ci_95": [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "risk_ratio": risk_ratio(source_rate, target_rate),
        "odds_ratio": odds_ratio,
        "raw_p_value": raw_p_value,
        "bh_adjusted_p_value": raw_p_value,
        "p_value_method": p_value_method,
        "source_target_overlap_count": len(source_ids & target_ids),
        "classification_by_threshold": classifications,
        "classification_stability": classification_stability(classifications),
        "warnings": warnings,
        "dataset": "bfcl_v4_non_live",
        "source_dataset": SOURCE_DATASET,
        "model": MODEL_NAME,
        "outcome_type": OUTCOME_TYPE,
        "label_scope": LABEL_SCOPE,
        "label_origin": LABEL_ORIGIN,
        "is_synthetic": False,
    }


def eligible_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        shift
        for shift in inventory
        if shift.get("dataset") == "bfcl_v4_non_live"
        and shift.get("status") == "eligible"
        and shift.get("outcome_type") == OUTCOME_TYPE
    ]


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts = {
        f"{threshold:.2f}": dict(
            sorted(
                Counter(
                    row["classification_by_threshold"][f"{threshold:.2f}"]
                    for row in results
                ).items()
            )
        )
        for threshold in PRACTICAL_THRESHOLDS
    }
    return {
        "dataset": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "model": MODEL_NAME,
        "label_scope": LABEL_SCOPE,
        "label_origin": LABEL_ORIGIN,
        "is_synthetic": False,
        "outcome_type": OUTCOME_TYPE,
        "analyzed_shift_count": len(results),
        "primary_complexity_shift_count": sum(
            row["is_primary_complexity_shift"] for row in results
        ),
        "behavioral_abstention_shift_count": sum(
            row["shift_type"] == "behavioral_abstention" for row in results
        ),
        "classification_counts_by_threshold": classification_counts,
        "stable_classification_shifts": [
            row["shift_id"]
            for row in results
            if row["classification_stability"] == "stable_across_thresholds"
        ],
        "threshold_sensitive_classification_shifts": [
            row["shift_id"]
            for row in results
            if row["classification_stability"] == "changes_with_threshold"
        ],
        "multiple_testing_method": MULTIPLE_TESTING_METHOD,
        "confidence_interval_methods": {
            "primary_delta_y_ci_95": PRIMARY_CI_METHOD,
            "bootstrap_delta_y_ci_95": BOOTSTRAP_CI_METHOD,
        },
        "bootstrap_configuration": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "resampling": "within source and target category groups separately",
        },
        "warnings": [
            EXPLORATORY_WARNING,
            NO_IID_MIXING_WARNING,
            NON_INDEPENDENCE_WARNING,
            NO_DEPLOYMENT_CLAIM_WARNING,
        ],
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def shifts_for_class_at_threshold(
    results: list[dict[str, Any]],
    threshold_key: str,
    class_name: str,
) -> list[str]:
    return [
        row["shift_id"]
        for row in results
        if row["classification_by_threshold"][threshold_key] == class_name
    ]


def classification_summary_by_threshold(results: list[dict[str, Any]]) -> str:
    class_names = [
        "candidate_harmful",
        "candidate_harmless",
        "candidate_beneficial",
        "inconclusive",
    ]
    sections = []
    for threshold in PRACTICAL_THRESHOLDS:
        threshold_key = f"{threshold:.2f}"
        lines = [f"### d={threshold_key}"]
        for class_name in class_names:
            shifts = shifts_for_class_at_threshold(results, threshold_key, class_name)
            if shifts:
                joined = ", ".join(f"`{shift_id}`" for shift_id in shifts)
            else:
                joined = "None."
            lines.append(f"- `{class_name}`: {joined}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def write_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_lines = [
        "| Shift | Type | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | Bootstrap 95% CI | Raw p | BH p | Test | d=0.05 | d=0.10 | d=0.15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in results:
        ci = row["delta_y_ci_95"]
        bootstrap_ci = row["bootstrap_delta_y_ci_95"]
        result_lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    row["shift_type"],
                    str(row["source_n"]),
                    str(row["target_n"]),
                    fmt_float(row["source_success_rate"]),
                    fmt_float(row["target_success_rate"]),
                    fmt_float(row["delta_y"]),
                    f"[{fmt_float(ci[0])}, {fmt_float(ci[1])}]",
                    f"[{fmt_float(bootstrap_ci[0])}, {fmt_float(bootstrap_ci[1])}]",
                    fmt_float(row["raw_p_value"]),
                    fmt_float(row["bh_adjusted_p_value"]),
                    row["p_value_method"],
                    row["classification_by_threshold"]["0.05"],
                    row["classification_by_threshold"]["0.10"],
                    row["classification_by_threshold"]["0.15"],
                ]
            )
            + " |"
        )

    report = f"""# BFCL Shift Uncertainty

## Purpose

Estimate statistical uncertainty for BFCL v4 non-live single-turn category contrasts using the already evaluated 1,240 sample-level labels. This is exploratory analysis only.

## Statistical Formulation

For each source/target category contrast, `Delta_Y = P_target(Y=1) - P_source(Y=1)`. The primary interval is a 95% Newcombe-Wilson difference-in-proportions interval. A deterministic nonparametric bootstrap interval uses {BOOTSTRAP_REPLICATES:,} replicates with seed {BOOTSTRAP_SEED}, resampling within source and target groups separately. Equality-of-proportions p-values use Fisher's exact test when expected counts are small and a two-proportion z-test otherwise, followed by Benjamini-Hochberg adjustment across the analyzed BFCL shifts.

## Practical-Significance Thresholds

Classifications are reported separately for `delta_practical = [0.05, 0.10, 0.15]`.

- `candidate_harmful`: upper 95% CI < `-d`
- `candidate_harmless`: full 95% CI is inside `[-d, d]`
- `candidate_beneficial`: lower 95% CI > `d`
- `inconclusive`: all other cases

## Results Table

{chr(10).join(result_lines)}

## Classification Summary by Threshold

{classification_summary_by_threshold(results)}

## Constraints

- No causal claims are made.
- No deployment-safe or retraining-required claims are made.
- BFCL rows are not mixed with tau2 or API-Bank as IID samples.
- `label_scope={summary["label_scope"]}` and `label_origin={summary["label_origin"]}` are preserved.
- Candidate groups are defined without using `y`.
- The analysis does not use partial BFCL leaderboard overall scores.
- The `simple_python -> irrelevance` contrast is reported as behavioral/abstention, not as a primary complexity shift.

## Limitations

The category contrasts are exploratory and reuse some category groups across multiple candidate shifts. The intervals describe uncertainty in this processed 1,240-row BFCL subset and should not be interpreted as causal evidence about context changes.
"""
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    input_jsonl: Path = DEFAULT_INPUT_JSONL,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    inventory_summary_json: Path = DEFAULT_INVENTORY_SUMMARY_JSON,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    summary_json: Path = DEFAULT_OUTPUT_SUMMARY_JSON,
    report_path: Path = DEFAULT_REPORT,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(input_jsonl)
    if not inventory_jsonl.exists() or not inventory_summary_json.exists():
        if (
            inventory_jsonl == DEFAULT_INVENTORY_JSONL
            and inventory_summary_json == DEFAULT_INVENTORY_SUMMARY_JSON
        ):
            build_inventory_outputs(write_outputs=True)
        else:
            raise FileNotFoundError(
                "Custom BFCL inventory paths must exist before uncertainty analysis: "
                f"{inventory_jsonl}, {inventory_summary_json}"
            )
    inventory = load_jsonl(inventory_jsonl)
    json.loads(inventory_summary_json.read_text(encoding="utf-8"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for shift in eligible_inventory(inventory):
        source_rows, target_rows = source_target_rows(shift, rows)
        results.append(analyze_shift(shift, source_rows, target_rows, rng))

    adjusted = bh_adjusted_p_values([row["raw_p_value"] for row in results])
    for row, adjusted_p_value in zip(results, adjusted, strict=True):
        row["bh_adjusted_p_value"] = adjusted_p_value

    summary = build_summary(results)
    if write_outputs:
        write_jsonl(results, output_jsonl)
        write_json(summary, summary_json)
        write_report(results, summary, report_path)
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--inventory-jsonl", type=Path, default=DEFAULT_INVENTORY_JSONL)
    parser.add_argument(
        "--inventory-summary-json",
        type=Path,
        default=DEFAULT_INVENTORY_SUMMARY_JSON,
    )
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = build_outputs(
        input_jsonl=args.input_jsonl,
        inventory_jsonl=args.inventory_jsonl,
        inventory_summary_json=args.inventory_summary_json,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        report_path=args.report_path,
    )
    print(f"wrote {len(results)} BFCL uncertainty rows")
    print(f"classification counts: {summary['classification_counts_by_threshold']}")
    print(EXPLORATORY_WARNING)


if __name__ == "__main__":
    main()
