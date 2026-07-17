#!/usr/bin/env python3
"""Add uncertainty estimates for eligible tau2 tool-calling shifts."""

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

from build_toolcalling_numerical_representation import (  # noqa: E402, I001
    DEFAULT_API_BANK_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TAU2_INPUT,
    FULL_JSONL_NAME,
    FULL_NPZ_NAME,
    load_jsonl,
    write_json,
    write_jsonl,
)
from build_toolcalling_shift_inventory import (  # noqa: E402
    DEFAULT_INVENTORY_JSONL,
    DEFAULT_SUMMARY_JSON as DEFAULT_INVENTORY_SUMMARY_JSON,
    TAU2_OUTCOME_TYPE,
    load_npz_arrays,
    make_analysis_rows,
    numeric_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUMERICAL_JSONL = DEFAULT_OUTPUT_DIR / FULL_JSONL_NAME
DEFAULT_NUMERICAL_NPZ = DEFAULT_OUTPUT_DIR / FULL_NPZ_NAME
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty.jsonl"
DEFAULT_OUTPUT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty_summary.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "tau2_shift_uncertainty.md"

PRACTICAL_THRESHOLDS = (0.05, 0.10, 0.15)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 1
SMALL_GROUP_WARNING_N = 30
UNSTABLE_INTERVAL_WIDTH = 0.50
Z_95 = 1.959963984540054

EXPLORATORY_WARNING = (
    "This is an exploratory small-sample tau2 analysis; the results do not imply "
    "causal effects and do not independently authorize deployment decisions."
)
MULTIPLE_TESTING_METHOD = "Benjamini-Hochberg false-discovery-rate adjustment"
PRIMARY_CI_METHOD = "Newcombe-Wilson score interval for difference in proportions"
BOOTSTRAP_CI_METHOD = "deterministic nonparametric bootstrap percentile interval"
NON_INDEPENDENCE_WARNING = (
    "Candidate shifts reuse tau2 records across non-independent exploratory definitions."
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
    lower = delta - math.sqrt((target_rate - target_low) ** 2 + (source_high - source_rate) ** 2)
    upper = delta + math.sqrt((target_high - target_rate) ** 2 + (source_rate - source_low) ** 2)
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
        return float(stats.fisher_exact(table, alternative="two-sided").pvalue), "fisher_exact"

    pooled = (source_positive + target_positive) / (source_n + target_n)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * ((1.0 / source_n) + (1.0 / target_n)))
    if standard_error == 0.0:
        return 1.0, "two_proportion_z_test"
    z_value = ((target_positive / target_n) - (source_positive / source_n)) / standard_error
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
    cells = np.asarray([source_positive, source_negative, target_positive, target_negative], dtype=np.float64)
    corrected = bool(np.any(cells == 0.0))
    if corrected:
        cells = cells + 0.5
    source_positive_f, source_negative_f, target_positive_f, target_negative_f = cells
    denominator = source_positive_f * target_negative_f
    if denominator == 0.0:
        return None, corrected
    return finite_float((target_positive_f * source_negative_f) / denominator), corrected


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
    return {f"{threshold:.2f}": classify_delta_ci(ci, threshold) for threshold in PRACTICAL_THRESHOLDS}


def classification_stability(classifications: dict[str, str]) -> str:
    values = set(classifications.values())
    if len(values) == 1:
        return "stable_across_thresholds"
    return "changes_with_threshold"


def candidate_source_target_rows(
    shift: dict[str, Any],
    tau2_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = shift["thresholds"]
    shift_id = shift["shift_id"]
    if shift_id == "tau2_retail_to_airline":
        source = [row for row in tau2_rows if row["domain"] == "retail"]
        target = [row for row in tau2_rows if row["domain"] == "airline"]
    elif shift_id == "tau2_no_write_to_write_required":
        field = "x_numeric_features.expected_write_action_count"
        source = [row for row in tau2_rows if numeric_value(row, field) == thresholds["source_max"]]
        target = [row for row in tau2_rows if (numeric_value(row, field) or 0) >= thresholds["target_min"]]
    elif shift_id == "tau2_zero_or_one_write_to_two_plus_writes":
        field = "x_numeric_features.expected_write_action_count"
        source = [row for row in tau2_rows if (numeric_value(row, field) or 0) <= thresholds["source_max"]]
        target = [row for row in tau2_rows if (numeric_value(row, field) or 0) >= thresholds["target_min"]]
    elif shift_id == "tau2_few_to_many_expected_actions":
        field = "x_numeric_features.expected_action_count"
        source = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) <= thresholds["lower_threshold"]]
        target = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) >= thresholds["upper_threshold"]]
    elif shift_id == "tau2_short_to_long_trajectory":
        field = "s_structural_features.trajectory_length"
        source = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) <= thresholds["lower_threshold"]]
        target = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) >= thresholds["upper_threshold"]]
    elif shift_id == "tau2_few_to_many_tool_calls":
        field = "metadata.num_tool_calls"
        source = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) <= thresholds["lower_threshold"]]
        target = [row for row in tau2_rows if numeric_value(row, field) is not None and numeric_value(row, field) >= thresholds["upper_threshold"]]
    else:
        raise ValueError(f"Unsupported tau2 shift_id for uncertainty analysis: {shift_id}")
    return source, target


def centroid_distance(array: np.ndarray, source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> float:
    source_indices = [row["array_index"] for row in source_rows]
    target_indices = [row["array_index"] for row in target_rows]
    source_centroid = array[source_indices].mean(axis=0)
    target_centroid = array[target_indices].mean(axis=0)
    return float(np.linalg.norm(target_centroid - source_centroid))


def build_warning_list(
    *,
    source_n: int,
    target_n: int,
    source_positive: int,
    target_positive: int,
    delta_ci: tuple[float, float],
    bootstrap_ci: tuple[float, float],
    overlap_count: int,
    odds_ratio_corrected: bool,
) -> list[str]:
    warnings = [NON_INDEPENDENCE_WARNING]
    if source_n < SMALL_GROUP_WARNING_N or target_n < SMALL_GROUP_WARNING_N:
        warnings.append(
            f"small group size warning: source_n={source_n}, target_n={target_n}, "
            f"threshold={SMALL_GROUP_WARNING_N}"
        )
    if overlap_count:
        warnings.append(f"overlapping source/target groups: overlap_count={overlap_count}")
    zero_cells = [
        source_positive,
        source_n - source_positive,
        target_positive,
        target_n - target_positive,
    ]
    if any(cell == 0 for cell in zero_cells):
        warnings.append("zero 2x2 table cell; odds ratio uses Haldane-Anscombe 0.5 correction")
    elif odds_ratio_corrected:
        warnings.append("odds ratio uses Haldane-Anscombe 0.5 correction")
    if (delta_ci[1] - delta_ci[0]) > UNSTABLE_INTERVAL_WIDTH or (
        bootstrap_ci[1] - bootstrap_ci[0]
    ) > UNSTABLE_INTERVAL_WIDTH:
        warnings.append(
            f"unstable interval warning: CI width exceeds {UNSTABLE_INTERVAL_WIDTH:.2f}"
        )
    return warnings


def source_rule(shift: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": shift["source_group"],
        "grouping_rule": shift["grouping_rule"],
        "fields": shift["group_definition_fields"],
        "thresholds": shift["thresholds"],
    }


def target_rule(shift: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": shift["target_group"],
        "grouping_rule": shift["grouping_rule"],
        "fields": shift["group_definition_fields"],
        "thresholds": shift["thresholds"],
    }


def analyze_shift(
    shift: dict[str, Any],
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
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
    delta_ci = newcombe_wilson_delta_ci(source_positive, source_n, target_positive, target_n)
    bootstrap_ci = bootstrap_delta_ci(source_y, target_y, rng)
    raw_p_value, p_value_method = proportions_p_value(
        source_positive,
        source_n,
        target_positive,
        target_n,
    )
    ratio = risk_ratio(source_rate, target_rate)
    odds_ratio, odds_ratio_corrected = odds_ratio_with_correction(
        source_positive,
        source_n,
        target_positive,
        target_n,
    )
    source_ids = {row["sample_id"] for row in source_rows}
    target_ids = {row["sample_id"] for row in target_rows}
    overlap_count = len(source_ids & target_ids)
    classifications = classification_by_threshold(delta_ci)
    warnings = build_warning_list(
        source_n=source_n,
        target_n=target_n,
        source_positive=source_positive,
        target_positive=target_positive,
        delta_ci=delta_ci,
        bootstrap_ci=bootstrap_ci,
        overlap_count=overlap_count,
        odds_ratio_corrected=odds_ratio_corrected,
    )
    return {
        "shift_id": shift["shift_id"],
        "shift_family": shift["family"],
        "source_rule": source_rule(shift),
        "target_rule": target_rule(shift),
        "source_n": source_n,
        "target_n": target_n,
        "source_positive": source_positive,
        "target_positive": target_positive,
        "source_success_rate": source_rate,
        "target_success_rate": target_rate,
        "delta_y": delta_y,
        "delta_y_ci_method": PRIMARY_CI_METHOD,
        "delta_y_ci_95": [float(delta_ci[0]), float(delta_ci[1])],
        "bootstrap_delta_y_ci_95": [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
        "risk_ratio": ratio,
        "odds_ratio": odds_ratio,
        "raw_p_value": raw_p_value,
        "bh_adjusted_p_value": raw_p_value,
        "p_value_method": p_value_method,
        "x_centroid_distance": centroid_distance(arrays["X"], source_rows, target_rows),
        "s_centroid_distance": centroid_distance(arrays["S"], source_rows, target_rows),
        "source_target_overlap_count": overlap_count,
        "classification_by_threshold": classifications,
        "classification_stability": classification_stability(classifications),
        "warnings": warnings,
        "outcome_type": TAU2_OUTCOME_TYPE,
    }


def eligible_tau2_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        shift
        for shift in inventory
        if shift.get("dataset") == "tau2"
        and shift.get("status") == "eligible"
        and shift.get("outcome_type") == TAU2_OUTCOME_TYPE
    ]


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    group_sizes = [size for row in results for size in (row["source_n"], row["target_n"])]
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
    stable = [
        row["shift_id"]
        for row in results
        if row["classification_stability"] == "stable_across_thresholds"
    ]
    changing = [
        row["shift_id"]
        for row in results
        if row["classification_stability"] == "changes_with_threshold"
    ]
    return {
        "eligible_tau2_shift_count": len(results),
        "classification_counts_by_threshold": classification_counts,
        "stable_classification_shifts": stable,
        "threshold_sensitive_classification_shifts": changing,
        "smallest_group_size": min(group_sizes) if group_sizes else None,
        "largest_group_size": max(group_sizes) if group_sizes else None,
        "overlapping_group_shift_count": sum(
            row["source_target_overlap_count"] > 0 for row in results
        ),
        "small_sample_warning_count": sum(
            any("small group size warning" in warning for warning in row["warnings"])
            for row in results
        ),
        "multiple_testing_method": MULTIPLE_TESTING_METHOD,
        "confidence_interval_methods": {
            "primary_delta_y_ci_95": PRIMARY_CI_METHOD,
            "bootstrap_delta_y_ci_95": BOOTSTRAP_CI_METHOD,
        },
        "bootstrap_configuration": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "resampling": "within source and target groups separately",
        },
        "exploratory_analysis_warning": EXPLORATORY_WARNING,
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def shifts_for_class(results: list[dict[str, Any]], class_name: str) -> list[str]:
    return [
        row["shift_id"]
        for row in results
        if class_name in set(row["classification_by_threshold"].values())
    ]


def bullet_list(items: list[str], fallback: str = "None.") -> str:
    if not items:
        return fallback
    return "\n".join(f"- `{item}`" for item in items)


def write_report(results: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_lines = [
        "| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | Raw p | BH p | d=0.05 | d=0.10 | d=0.15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in results:
        ci = row["delta_y_ci_95"]
        result_lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    str(row["source_n"]),
                    str(row["target_n"]),
                    fmt_float(row["source_success_rate"]),
                    fmt_float(row["target_success_rate"]),
                    fmt_float(row["delta_y"]),
                    f"[{fmt_float(ci[0])}, {fmt_float(ci[1])}]",
                    fmt_float(row["raw_p_value"]),
                    fmt_float(row["bh_adjusted_p_value"]),
                    row["classification_by_threshold"]["0.05"],
                    row["classification_by_threshold"]["0.10"],
                    row["classification_by_threshold"]["0.15"],
                ]
            )
            + " |"
        )

    sensitivity_lines = [
        "| Shift | Classification stability | d=0.05 | d=0.10 | d=0.15 |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        sensitivity_lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    row["classification_stability"],
                    row["classification_by_threshold"]["0.05"],
                    row["classification_by_threshold"]["0.10"],
                    row["classification_by_threshold"]["0.15"],
                ]
            )
            + " |"
        )

    warning_lines = []
    for row in results:
        warning_lines.append(f"- `{row['shift_id']}`: " + "; ".join(row["warnings"]))

    report = f"""# Tau2 Shift Uncertainty Analysis

## Purpose

Estimate statistical uncertainty for the eligible tau2 candidate shifts only. API-Bank rows are excluded because their labels include synthetic corrupted negatives and do not support harmful/harmless deployment claims.

## Statistical formulation

For each preserved source/target definition, `Delta_Y = P_target(Y=1) - P_source(Y=1)`. The primary interval is a 95% Newcombe-Wilson difference-in-proportions interval. A deterministic nonparametric bootstrap interval uses {BOOTSTRAP_REPLICATES:,} replicates with seed {BOOTSTRAP_SEED}, resampling within source and target groups separately. Equality-of-proportions p-values use Fisher's exact test when expected counts are small and a two-proportion z-test otherwise, followed by Benjamini-Hochberg adjustment across the eligible tau2 shifts.

## Practical-significance thresholds

Classifications are reported separately for `delta_practical = [0.05, 0.10, 0.15]`. A shift is `candidate_harmful` only when the full primary 95% CI is below `-d`, `candidate_harmless` only when the full CI lies within `[-d, +d]`, and `candidate_beneficial` only when the full CI is above `+d`. All other cases are `inconclusive`.

## Results table

{chr(10).join(result_lines)}

## Classification sensitivity

{chr(10).join(sensitivity_lines)}

Stable across all three thresholds:

{bullet_list(summary['stable_classification_shifts'])}

Changes with threshold:

{bullet_list(summary['threshold_sensitive_classification_shifts'])}

## Candidate harmful shifts

{bullet_list(shifts_for_class(results, 'candidate_harmful'))}

## Candidate harmless shifts

{bullet_list(shifts_for_class(results, 'candidate_harmless'))}

## Candidate beneficial shifts

{bullet_list(shifts_for_class(results, 'candidate_beneficial'))}

## Inconclusive shifts

{bullet_list(shifts_for_class(results, 'inconclusive'))}

## Small-sample and overlap warnings

{chr(10).join(warning_lines)}

## Retraining interpretation

`candidate_harmful` may motivate additional evaluation or adaptation. `candidate_harmless` is evidence consistent with a practically small outcome change and may suggest retraining is not immediately justified, but it is not proof that retraining is unnecessary. `inconclusive` means more data are needed. These results do not independently authorize a deployment decision.

## Limitations

The dataset is small, the analysis is exploratory, candidate shift definitions reuse records and are not independent, and the estimates should not be interpreted causally. No predictive model is trained here, and group membership is not redefined using `y`.

## Next step

Collect additional real tau2-style task outcomes for the most decision-relevant shifts, then rerun this uncertainty analysis before deciding whether adaptation or retraining is warranted.
"""
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    numerical_jsonl: Path = DEFAULT_NUMERICAL_JSONL,
    numerical_npz: Path = DEFAULT_NUMERICAL_NPZ,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    inventory_summary_json: Path = DEFAULT_INVENTORY_SUMMARY_JSON,
    tau2_input: Path = DEFAULT_TAU2_INPUT,
    api_bank_input: Path = DEFAULT_API_BANK_INPUT,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    summary_json: Path = DEFAULT_OUTPUT_SUMMARY_JSON,
    report_path: Path = DEFAULT_REPORT,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numerical_records = load_jsonl(numerical_jsonl)
    arrays = load_npz_arrays(numerical_npz)
    inventory = load_jsonl(inventory_jsonl)
    # The summary is an explicit input artifact; loading it also fails fast if the
    # uncertainty analysis is not aligned with the inventory build that produced it.
    json.loads(inventory_summary_json.read_text(encoding="utf-8"))
    unified_records = load_jsonl(tau2_input) + load_jsonl(api_bank_input)
    rows = make_analysis_rows(numerical_records, unified_records, arrays)
    tau2_rows = [row for row in rows if row["source_dataset"] == "tau2"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    results = []
    for shift in eligible_tau2_inventory(inventory):
        source_rows, target_rows = candidate_source_target_rows(shift, tau2_rows)
        result = analyze_shift(shift, source_rows, target_rows, arrays, rng)
        results.append(result)

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
    parser.add_argument("--numerical-jsonl", type=Path, default=DEFAULT_NUMERICAL_JSONL)
    parser.add_argument("--numerical-npz", type=Path, default=DEFAULT_NUMERICAL_NPZ)
    parser.add_argument("--inventory-jsonl", type=Path, default=DEFAULT_INVENTORY_JSONL)
    parser.add_argument("--inventory-summary-json", type=Path, default=DEFAULT_INVENTORY_SUMMARY_JSON)
    parser.add_argument("--tau2-input", type=Path, default=DEFAULT_TAU2_INPUT)
    parser.add_argument("--api-bank-input", type=Path, default=DEFAULT_API_BANK_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = build_outputs(
        numerical_jsonl=args.numerical_jsonl,
        numerical_npz=args.numerical_npz,
        inventory_jsonl=args.inventory_jsonl,
        inventory_summary_json=args.inventory_summary_json,
        tau2_input=args.tau2_input,
        api_bank_input=args.api_bank_input,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        report_path=args.report_path,
    )
    print(f"wrote {len(results)} tau2 uncertainty rows")
    print(f"classification counts: {summary['classification_counts_by_threshold']}")
    print(EXPLORATORY_WARNING)


if __name__ == "__main__":
    main()
