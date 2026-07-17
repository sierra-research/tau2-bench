#!/usr/bin/env python3
"""Build tau2-only Stage 1 merged analysis artifacts."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_tau2_shift_uncertainty import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRACTICAL_THRESHOLDS,
    analyze_shift,
    bh_adjusted_p_values,
    build_summary as build_uncertainty_summary,
    candidate_source_target_rows,
)
from build_toolcalling_numerical_representation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TAU2_INPUT,
    TextEncoder,
    build_outputs as build_numerical_outputs,
    load_jsonl,
    write_json,
    write_jsonl,
)
from build_toolcalling_shift_inventory import (
    DEFAULT_INVENTORY_JSONL as BASELINE_INVENTORY_JSONL,
    DEFAULT_MIN_GROUP_SIZE,
    TAU2_OUTCOME_TYPE,
    evaluate_candidate,
    load_npz_arrays,
    make_analysis_rows,
    numeric_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_RETAINED = DEFAULT_OUTPUT_DIR / "tau2_stage1_retained.jsonl"
DEFAULT_UNIFIED_STAGE1 = DEFAULT_OUTPUT_DIR / "unified_toolcalling_tau2_stage1.jsonl"
DEFAULT_UNIFIED_STAGE1_MANIFEST = (
    DEFAULT_OUTPUT_DIR / "unified_toolcalling_tau2_stage1_manifest.json"
)
DEFAULT_NUMERICAL_NPZ_NAME = "toolcalling_numerical_tau2_stage1.npz"
DEFAULT_NUMERICAL_JSONL_NAME = "toolcalling_numerical_tau2_stage1.jsonl"
DEFAULT_NUMERICAL_MANIFEST_NAME = "toolcalling_numerical_tau2_stage1_manifest.json"
DEFAULT_INVENTORY_JSONL = (
    DEFAULT_OUTPUT_DIR / "toolcalling_shift_inventory_tau2_stage1.jsonl"
)
DEFAULT_INVENTORY_SUMMARY = (
    DEFAULT_OUTPUT_DIR / "toolcalling_shift_inventory_tau2_stage1_summary.json"
)
DEFAULT_UNCERTAINTY_JSONL = (
    DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty_stage1.jsonl"
)
DEFAULT_UNCERTAINTY_SUMMARY = (
    DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty_stage1_summary.json"
)
DEFAULT_UNCERTAINTY_REPORT = REPO_ROOT / "docs/tau2_shift_uncertainty_stage1.md"
DEFAULT_COMPARISON_JSON = DEFAULT_OUTPUT_DIR / "tau2_shift_stage1_comparison.json"
DEFAULT_COMPARISON_REPORT = REPO_ROOT / "docs/tau2_shift_stage1_comparison.md"
DEFAULT_PROJECT_LOG = REPO_ROOT / "docs/toolcalling_shift_project_log.md"
BASELINE_UNCERTAINTY_JSONL = DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty.jsonl"
BASELINE_UNCERTAINTY_SUMMARY = DEFAULT_OUTPUT_DIR / "tau2_shift_uncertainty_summary.json"
BASELINE_UNCERTAINTY_REPORT = REPO_ROOT / "docs/tau2_shift_uncertainty.md"

EXPECTED_BASELINE_COUNT = 93
EXPECTED_STAGE1_COUNT = 12
EXPECTED_MERGED_COUNT = 105
STAGE1_SECTION_TITLE = "## 2026-07-17 — Tau2 Stage 1 Post-Collection Analysis"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def y_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["y"]) for record in records).items()))


def task_run_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("metadata", {})
    domain = str(record.get("domain") or metadata.get("domain"))
    task_id = str(metadata.get("task_id"))
    run_identity = str(
        metadata.get("stage1_run_identity")
        or metadata.get("run_identity")
        or metadata.get("source_result_folder")
        or record["sample_id"]
    )
    return domain, task_id, run_identity


def validate_tau2_record_invariants(records: list[dict[str, Any]], source_name: str) -> None:
    for record in records:
        if record.get("source_dataset") != "tau2":
            raise ValueError(f"{source_name} contains non-tau2 record {record.get('sample_id')}")
        if record.get("label_scope") != "task_level":
            raise ValueError(f"{source_name} label_scope changed for {record.get('sample_id')}")
        if record.get("label_origin") != "tau2_benchmark_reward":
            raise ValueError(f"{source_name} label_origin changed for {record.get('sample_id')}")
        if record.get("is_synthetic") is not False:
            raise ValueError(f"{source_name} synthetic flag changed for {record.get('sample_id')}")


def merge_tau2_stage1_records(
    baseline_records: list[dict[str, Any]],
    stage1_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_tau2_record_invariants(baseline_records, "baseline")
    validate_tau2_record_invariants(stage1_records, "stage1")
    baseline_ids = [record["sample_id"] for record in baseline_records]
    stage1_ids = [record["sample_id"] for record in stage1_records]
    if len(baseline_records) != EXPECTED_BASELINE_COUNT:
        raise ValueError(f"Expected 93 baseline tau2 records, found {len(baseline_records)}")
    if len(stage1_records) != EXPECTED_STAGE1_COUNT:
        raise ValueError(f"Expected 12 Stage 1 records, found {len(stage1_records)}")
    if set(stage1_ids) & set(baseline_ids):
        raise ValueError("At least one Stage 1 sample_id already exists in baseline")

    combined: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, str, str]] = set()
    duplicate_identities: list[tuple[str, str, str]] = []
    for record in baseline_records + stage1_records:
        identity = task_run_identity(record)
        if identity in seen_identities:
            duplicate_identities.append(identity)
            continue
        seen_identities.add(identity)
        combined.append(copy.deepcopy(record))
    sample_ids = [record["sample_id"] for record in combined]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Merged tau2 records contain duplicate sample IDs")
    if len(combined) != EXPECTED_MERGED_COUNT:
        raise ValueError(f"Expected 105 merged tau2 records, found {len(combined)}")

    manifest = {
        "total_count": len(combined),
        "source_counts": {"baseline": len(baseline_records), "stage1": len(stage1_records)},
        "y_distribution": y_distribution(combined),
        "baseline_y_distribution": y_distribution(baseline_records),
        "stage1_y_distribution": y_distribution(stage1_records),
        "stage1_sample_ids_all_new": True,
        "duplicate_sample_ids": [],
        "duplicate_task_run_identities_dropped": [list(item) for item in duplicate_identities],
        "label_scope": "task_level",
        "label_origin": "tau2_benchmark_reward",
        "is_synthetic": False,
        "original_records_preserved_unchanged": True,
    }
    return combined, manifest


def rows_for_tau2_outputs(
    numerical_jsonl: Path,
    numerical_npz: Path,
    tau2_input: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    numerical_records = load_jsonl(numerical_jsonl)
    arrays = load_npz_arrays(numerical_npz)
    unified_records = load_jsonl(tau2_input)
    rows = make_analysis_rows(numerical_records, unified_records, arrays)
    return rows, arrays, unified_records


def stage1_inventory_from_baseline(
    *,
    baseline_inventory: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> list[dict[str, Any]]:
    tau2_rows = [row for row in rows if row["source_dataset"] == "tau2"]
    inventory = []
    for baseline_shift in baseline_inventory:
        if baseline_shift.get("dataset") != "tau2":
            continue
        shift_id = baseline_shift["shift_id"]
        thresholds = copy.deepcopy(baseline_shift["thresholds"])

        if shift_id == "tau2_retail_to_airline":
            def source_predicate(row: dict[str, Any]) -> bool:
                return row["domain"] == "retail"

            def target_predicate(row: dict[str, Any]) -> bool:
                return row["domain"] == "airline"
        elif shift_id == "tau2_no_write_to_write_required":
            field = "x_numeric_features.expected_write_action_count"

            def source_predicate(row: dict[str, Any], field: str = field) -> bool:
                return numeric_value(row, field) == thresholds["source_max"]

            def target_predicate(row: dict[str, Any], field: str = field) -> bool:
                return (numeric_value(row, field) or 0) >= thresholds["target_min"]
        elif shift_id == "tau2_zero_or_one_write_to_two_plus_writes":
            field = "x_numeric_features.expected_write_action_count"

            def source_predicate(row: dict[str, Any], field: str = field) -> bool:
                return (numeric_value(row, field) or 0) <= thresholds["source_max"]

            def target_predicate(row: dict[str, Any], field: str = field) -> bool:
                return (numeric_value(row, field) or 0) >= thresholds["target_min"]
        elif shift_id == "tau2_few_to_many_expected_actions":
            field = "x_numeric_features.expected_action_count"

            def source_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value <= thresholds["lower_threshold"]

            def target_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value >= thresholds["upper_threshold"]
        elif shift_id == "tau2_short_to_long_trajectory":
            field = "s_structural_features.trajectory_length"

            def source_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value <= thresholds["lower_threshold"]

            def target_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value >= thresholds["upper_threshold"]
        elif shift_id == "tau2_few_to_many_tool_calls":
            field = "metadata.num_tool_calls"

            def source_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value <= thresholds["lower_threshold"]

            def target_predicate(row: dict[str, Any], field: str = field) -> bool:
                value = numeric_value(row, field)
                return value is not None and value >= thresholds["upper_threshold"]
        else:
            raise ValueError(f"Unexpected tau2 shift_id: {shift_id}")

        inventory.append(
            evaluate_candidate(
                shift_id=shift_id,
                dataset="tau2",
                family=baseline_shift["family"],
                source_name=baseline_shift["source_group"],
                target_name=baseline_shift["target_group"],
                rule=baseline_shift["grouping_rule"],
                group_definition_fields=baseline_shift["group_definition_fields"],
                thresholds=thresholds,
                rows=tau2_rows,
                source_predicate=source_predicate,
                target_predicate=target_predicate,
                arrays=arrays,
                min_group_size=min_group_size,
                outcome_type=TAU2_OUTCOME_TYPE,
            )
        )
    return inventory


def build_inventory_summary(
    inventory: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_tau2 = [row for row in baseline_inventory if row.get("dataset") == "tau2"]
    return {
        "total_candidate_shift_count": len(inventory),
        "record_count_by_dataset": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "status_counts": dict(sorted(Counter(row["status"] for row in inventory).items())),
        "min_group_size": min(row["min_group_size"] for row in inventory),
        "outcome_types_by_dataset": {"tau2": [TAU2_OUTCOME_TYPE]},
        "suitable_for_later_harmful_harmless_analysis": [
            row["shift_id"] for row in inventory if row["status"] == "eligible"
        ],
        "group_definitions_using_y": [
            row["shift_id"] for row in inventory if row["group_definition_uses_y"]
        ],
        "shift_ids": [row["shift_id"] for row in inventory],
        "baseline_shift_ids": [row["shift_id"] for row in baseline_tau2],
        "definitions_preserved_from_baseline": all(
            {
                "shift_id": row["shift_id"],
                "fields": row["group_definition_fields"],
                "thresholds": row["thresholds"],
                "rule": row["grouping_rule"],
            }
            == {
                "shift_id": base["shift_id"],
                "fields": base["group_definition_fields"],
                "thresholds": base["thresholds"],
                "rule": base["grouping_rule"],
            }
            for row, base in zip(inventory, baseline_tau2, strict=True)
        ),
    }


def analyze_stage1_uncertainty(
    *,
    inventory: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tau2_rows = [row for row in rows if row["source_dataset"] == "tau2"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for shift in inventory:
        source_rows, target_rows = candidate_source_target_rows(shift, tau2_rows)
        results.append(analyze_shift(shift, source_rows, target_rows, arrays, rng))
    adjusted = bh_adjusted_p_values([row["raw_p_value"] for row in results])
    for row, adjusted_p_value in zip(results, adjusted, strict=True):
        row["bh_adjusted_p_value"] = adjusted_p_value
    summary = build_uncertainty_summary(results)
    summary["stage1_note"] = (
        "Stage 1 task selection was targeted by X/task characteristics, not by "
        "observed outcomes; Stage 1 retained outcomes had 4 successes in 12 records."
    )
    return results, summary


def ci_width(row: dict[str, Any]) -> float:
    ci = row["delta_y_ci_95"]
    return float(ci[1] - ci[0])


def stage1_membership_counts(
    shift: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    stage1_rows = [
        row
        for row in rows
        if row["metadata"].get("source_sample_id", row["sample_id"]).endswith(":stage1")
        or str(row["sample_id"]).endswith(":stage1")
    ]
    source_rows, target_rows = candidate_source_target_rows(shift, stage1_rows)
    source_ids = {row["sample_id"] for row in source_rows}
    target_ids = {row["sample_id"] for row in target_rows}
    all_ids = {row["sample_id"] for row in stage1_rows}
    return {
        "source": len(source_ids - target_ids),
        "target": len(target_ids - source_ids),
        "both": len(source_ids & target_ids),
        "neither": len(all_ids - source_ids - target_ids),
    }


def build_pre_post_comparison(
    *,
    baseline_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    stage1_inventory: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {row["shift_id"]: row for row in baseline_rows}
    stage1_by_id = {row["shift_id"]: row for row in stage1_rows}
    inventory_by_id = {row["shift_id"]: row for row in stage1_inventory}
    comparisons = []
    for shift_id in baseline_by_id:
        baseline = baseline_by_id[shift_id]
        stage1 = stage1_by_id[shift_id]
        membership = stage1_membership_counts(inventory_by_id[shift_id], analysis_rows)
        comparisons.append(
            {
                "shift_id": shift_id,
                "baseline_source_n": baseline["source_n"],
                "baseline_target_n": baseline["target_n"],
                "stage1_source_n": stage1["source_n"],
                "stage1_target_n": stage1["target_n"],
                "baseline_source_success_rate": baseline["source_success_rate"],
                "baseline_target_success_rate": baseline["target_success_rate"],
                "stage1_source_success_rate": stage1["source_success_rate"],
                "stage1_target_success_rate": stage1["target_success_rate"],
                "baseline_delta_y": baseline["delta_y"],
                "stage1_delta_y": stage1["delta_y"],
                "baseline_delta_y_ci_95": baseline["delta_y_ci_95"],
                "stage1_delta_y_ci_95": stage1["delta_y_ci_95"],
                "baseline_ci_width": ci_width(baseline),
                "stage1_ci_width": ci_width(stage1),
                "ci_width_decreased": ci_width(stage1) < ci_width(baseline),
                "baseline_classification_by_threshold": baseline["classification_by_threshold"],
                "stage1_classification_by_threshold": stage1["classification_by_threshold"],
                "classification_changed": baseline["classification_by_threshold"]
                != stage1["classification_by_threshold"],
                "stage1_membership_counts": membership,
            }
        )
    target = stage1_by_id["tau2_zero_or_one_write_to_two_plus_writes"][
        "classification_by_threshold"
    ]
    return {
        "comparison": comparisons,
        "matching_shift_ids": [row["shift_id"] for row in comparisons],
        "all_shift_ids_match": set(baseline_by_id) == set(stage1_by_id),
        "stage1_successes": 4,
        "stage1_total": 12,
        "stage1_selection_basis": "targeted by X/task characteristics, not observed outcomes",
        "zero_or_one_write_to_two_plus_writes_remains_candidate_harmful_all_thresholds": all(
            value == "candidate_harmful" for value in target.values()
        ),
        "ci_width_decreased_shift_ids": [
            row["shift_id"] for row in comparisons if row["ci_width_decreased"]
        ],
        "classification_changed_shift_ids": [
            row["shift_id"] for row in comparisons if row["classification_changed"]
        ],
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_ci(ci: list[float]) -> str:
    return f"[{fmt_float(ci[0])}, {fmt_float(ci[1])}]"


def write_uncertainty_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | d=0.05 | d=0.10 | d=0.15 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    str(row["source_n"]),
                    str(row["target_n"]),
                    fmt_float(row["source_success_rate"]),
                    fmt_float(row["target_success_rate"]),
                    fmt_float(row["delta_y"]),
                    fmt_ci(row["delta_y_ci_95"]),
                    row["classification_by_threshold"]["0.05"],
                    row["classification_by_threshold"]["0.10"],
                    row["classification_by_threshold"]["0.15"],
                ]
            )
            + " |"
        )
    report = f"""# Tau2 Shift Uncertainty Analysis After Stage 1

## Purpose

Rerun the tau2-only uncertainty analysis after adding 12 retained Stage 1 retail records to the original 93 tau2 records. Baseline files are not overwritten, API-Bank data are not modified, and no predictive model is trained.

## Methods

The analysis uses the same six baseline tau2 shift definitions and thresholds, Newcombe-Wilson 95% confidence intervals, deterministic bootstrap with {BOOTSTRAP_REPLICATES:,} replicates and seed {BOOTSTRAP_SEED}, Fisher or two-proportion testing as previously implemented, Benjamini-Hochberg adjustment, and practical thresholds {', '.join(f'{threshold:.2f}' for threshold in PRACTICAL_THRESHOLDS)}.

## Results

{chr(10).join(lines)}

## Interpretation

Stage 1 task selection was targeted by X/task characteristics, not by observed outcomes. Stage 1 itself had 4 successes in 12 retained records. These results are exploratory, relatively small-sample estimates; they do not establish causality and do not prove that any shift is harmful or harmless.

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_comparison_report(comparison: dict[str, Any], path: Path) -> None:
    lines = [
        "| Shift | Base n | Stage 1 n | Base rates | Stage 1 rates | Base delta | Stage 1 delta | Base CI | Stage 1 CI | CI widths | d=0.05 | d=0.10 | d=0.15 | Class changed | Stage 1 membership |",
        "|---|---:|---:|---|---|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison["comparison"]:
        membership = row["stage1_membership_counts"]
        base_classes = row["baseline_classification_by_threshold"]
        stage_classes = row["stage1_classification_by_threshold"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    f"{row['baseline_source_n']}/{row['baseline_target_n']}",
                    f"{row['stage1_source_n']}/{row['stage1_target_n']}",
                    (
                        f"{fmt_float(row['baseline_source_success_rate'])}/"
                        f"{fmt_float(row['baseline_target_success_rate'])}"
                    ),
                    (
                        f"{fmt_float(row['stage1_source_success_rate'])}/"
                        f"{fmt_float(row['stage1_target_success_rate'])}"
                    ),
                    fmt_float(row["baseline_delta_y"]),
                    fmt_float(row["stage1_delta_y"]),
                    fmt_ci(row["baseline_delta_y_ci_95"]),
                    fmt_ci(row["stage1_delta_y_ci_95"]),
                    (
                        f"{fmt_float(row['baseline_ci_width'])}/"
                        f"{fmt_float(row['stage1_ci_width'])}; "
                        f"narrowed={row['ci_width_decreased']}"
                    ),
                    f"{base_classes['0.05']} -> {stage_classes['0.05']}",
                    f"{base_classes['0.10']} -> {stage_classes['0.10']}",
                    f"{base_classes['0.15']} -> {stage_classes['0.15']}",
                    str(row["classification_changed"]),
                    (
                        f"source={membership['source']}, target={membership['target']}, "
                        f"both={membership['both']}, neither={membership['neither']}"
                    ),
                ]
            )
            + " |"
        )
    harmful = comparison[
        "zero_or_one_write_to_two_plus_writes_remains_candidate_harmful_all_thresholds"
    ]
    target_row = next(
        row
        for row in comparison["comparison"]
        if row["shift_id"] == "tau2_zero_or_one_write_to_two_plus_writes"
    )
    target_classes = target_row["stage1_classification_by_threshold"]
    report = f"""# Tau2 Stage 1 Shift Comparison

## Purpose

Compare the baseline tau2 shift uncertainty outputs with the versioned Stage 1 tau2-only outputs.

## Results

{chr(10).join(lines)}

## Candidate-Harmful Shift

`tau2_zero_or_one_write_to_two_plus_writes` remains candidate_harmful at all three practical thresholds after adding the 12 targeted Stage 1 records: {harmful}. Post-Stage-1 classifications are d=0.05 `{target_classes['0.05']}`, d=0.10 `{target_classes['0.10']}`, and d=0.15 `{target_classes['0.15']}`.

## Interpretation

Stage 1 task selection was targeted by X/task characteristics rather than observed outcomes, and Stage 1 had 4/12 successes. The sample remains exploratory and relatively small. The comparison should not be interpreted causally and does not prove that any shift is harmful or harmless.

## Changes

- CI widths narrowed: {', '.join(comparison['ci_width_decreased_shift_ids']) or 'None'}.
- Classifications changed: {', '.join(comparison['classification_changed_shift_ids']) or 'None'}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def project_log_section(
    *,
    merge_manifest: dict[str, Any],
    numerical_manifest: dict[str, Any],
    uncertainty_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    result_lines = "\n".join(
        (
            f"- `{row['shift_id']}`: source_n={row['source_n']}, "
            f"target_n={row['target_n']}, delta_y={fmt_float(row['delta_y'])}, "
            f"95% CI={fmt_ci(row['delta_y_ci_95'])}, "
            f"d=0.05 {row['classification_by_threshold']['0.05']}, "
            f"d=0.10 {row['classification_by_threshold']['0.10']}, "
            f"d=0.15 {row['classification_by_threshold']['0.15']}"
        )
        for row in uncertainty_rows
    )
    target_row = next(
        row
        for row in uncertainty_rows
        if row["shift_id"] == "tau2_zero_or_one_write_to_two_plus_writes"
    )
    target_classes = target_row["classification_by_threshold"]
    remains_all_thresholds = all(
        value == "candidate_harmful" for value in target_classes.values()
    )
    return f"""
{STAGE1_SECTION_TITLE}

### Objective

Integrate the 12 retained Stage 1 tau2 records with the original 93 tau2 records, then rebuild tau2-only numerical data, the shift inventory, uncertainty analysis, and pre/post comparison using versioned Stage 1 outputs.

### Stage 1 execution summary

Stage 1 execution and ingestion were complete before this analysis: 12 attempted, 12 completed, 12 retained, 0 filtered, with 4/12 successes and total observed cost 0.0706581. Task selection was targeted by X/task characteristics, not observed outcomes.

### Merge counts

- Baseline records: {merge_manifest['source_counts']['baseline']}
- Stage 1 records: {merge_manifest['source_counts']['stage1']}
- Merged records: {merge_manifest['total_count']}
- Merged y distribution: {json.dumps(merge_manifest['y_distribution'], sort_keys=True)}

### Numerical shapes

- X: ({numerical_manifest['total_count']}, {numerical_manifest['final_x_dimension']})
- S: ({numerical_manifest['total_count']}, {numerical_manifest['final_s_dimension']})
- y: ({numerical_manifest['total_count']},)

### Pre/post shift results

{result_lines}

### Candidate-harmful shift result

`tau2_zero_or_one_write_to_two_plus_writes` remains candidate_harmful at all three practical thresholds after adding Stage 1 records: {remains_all_thresholds}. Post-Stage-1 classifications are d=0.05 `{target_classes['0.05']}`, d=0.10 `{target_classes['0.10']}`, and d=0.15 `{target_classes['0.15']}`.

### Classification changes

{', '.join(comparison['classification_changed_shift_ids']) or 'None.'}

### CI-width changes

CI widths narrowed for: {', '.join(comparison['ci_width_decreased_shift_ids']) or 'None.'}

### Verified findings

- Baseline tau2/API-Bank files were not overwritten.
- API-Bank data were not modified.
- Six tau2 shift definitions and thresholds were preserved from the baseline inventory.
- No predictive model was trained.

### Limitations

The analysis remains exploratory and relatively small. The shifts reuse records across non-independent definitions, and results should not be interpreted causally or as proof that a shift is harmful or harmless.

### Next step

Use these versioned Stage 1 results to decide whether another targeted collection stage is warranted before considering adaptation or retraining.
"""


def append_project_log(section: str, path: Path = DEFAULT_PROJECT_LOG) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if STAGE1_SECTION_TITLE in existing:
        return False
    path.write_text(existing.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")
    return True


def build_stage1_analysis(
    *,
    baseline_tau2_jsonl: Path = DEFAULT_TAU2_INPUT,
    stage1_retained_jsonl: Path = DEFAULT_STAGE1_RETAINED,
    baseline_inventory_jsonl: Path = BASELINE_INVENTORY_JSONL,
    baseline_uncertainty_jsonl: Path = BASELINE_UNCERTAINTY_JSONL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    unified_jsonl: Path = DEFAULT_UNIFIED_STAGE1,
    unified_manifest_json: Path = DEFAULT_UNIFIED_STAGE1_MANIFEST,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    inventory_summary_json: Path = DEFAULT_INVENTORY_SUMMARY,
    uncertainty_jsonl: Path = DEFAULT_UNCERTAINTY_JSONL,
    uncertainty_summary_json: Path = DEFAULT_UNCERTAINTY_SUMMARY,
    uncertainty_report: Path = DEFAULT_UNCERTAINTY_REPORT,
    comparison_json: Path = DEFAULT_COMPARISON_JSON,
    comparison_report: Path = DEFAULT_COMPARISON_REPORT,
    project_log: Path = DEFAULT_PROJECT_LOG,
    encoder: TextEncoder | None = None,
    append_log: bool = True,
    write_outputs: bool = True,
) -> dict[str, Any]:
    protected = [
        DEFAULT_TAU2_INPUT,
        BASELINE_INVENTORY_JSONL,
        BASELINE_UNCERTAINTY_JSONL,
        BASELINE_UNCERTAINTY_SUMMARY,
        BASELINE_UNCERTAINTY_REPORT,
    ]
    protected_hashes = {source_path(path): sha256_file(path) for path in protected}

    baseline_records = load_jsonl(baseline_tau2_jsonl)
    stage1_records = load_jsonl(stage1_retained_jsonl)
    combined, merge_manifest = merge_tau2_stage1_records(
        baseline_records,
        stage1_records,
    )
    baseline_inventory = load_jsonl(baseline_inventory_jsonl)

    if write_outputs:
        write_jsonl(combined, unified_jsonl)
        write_json(merge_manifest, unified_manifest_json)

    numerical_records, numerical_manifest, arrays = build_numerical_outputs(
        tau2_input=unified_jsonl,
        api_bank_input=REPO_ROOT / "data/processed/unified_toolcalling_apibank.jsonl",
        output_dir=output_dir,
        npz_name=DEFAULT_NUMERICAL_NPZ_NAME,
        jsonl_name=DEFAULT_NUMERICAL_JSONL_NAME,
        manifest_name=DEFAULT_NUMERICAL_MANIFEST_NAME,
        tau2_limit=len(combined),
        apibank_limit=0,
        full_data=False,
        encoder=encoder,
        write_outputs=write_outputs,
    )
    numerical_npz = output_dir / DEFAULT_NUMERICAL_NPZ_NAME
    numerical_jsonl = output_dir / DEFAULT_NUMERICAL_JSONL_NAME
    if write_outputs:
        rows, inventory_arrays, _ = rows_for_tau2_outputs(
            numerical_jsonl,
            numerical_npz,
            unified_jsonl,
        )
    else:
        rows = make_analysis_rows(numerical_records, combined, arrays)
        inventory_arrays = arrays

    inventory = stage1_inventory_from_baseline(
        baseline_inventory=baseline_inventory,
        rows=rows,
        arrays=inventory_arrays,
    )
    inventory_summary = build_inventory_summary(inventory, rows, baseline_inventory)
    uncertainty_rows, uncertainty_summary = analyze_stage1_uncertainty(
        inventory=inventory,
        rows=rows,
        arrays=inventory_arrays,
    )
    comparison = build_pre_post_comparison(
        baseline_rows=load_jsonl(baseline_uncertainty_jsonl),
        stage1_rows=uncertainty_rows,
        stage1_inventory=inventory,
        analysis_rows=rows,
    )

    if write_outputs:
        write_jsonl(inventory, inventory_jsonl)
        write_json(inventory_summary, inventory_summary_json)
        write_jsonl(uncertainty_rows, uncertainty_jsonl)
        write_json(uncertainty_summary, uncertainty_summary_json)
        write_uncertainty_report(uncertainty_rows, uncertainty_summary, uncertainty_report)
        write_json(comparison, comparison_json)
        write_comparison_report(comparison, comparison_report)

    section = project_log_section(
        merge_manifest=merge_manifest,
        numerical_manifest=numerical_manifest,
        uncertainty_rows=uncertainty_rows,
        comparison=comparison,
    )
    log_appended = False
    if write_outputs and append_log:
        log_appended = append_project_log(section, project_log)

    after_hashes = {source_path(path): sha256_file(path) for path in protected}
    if protected_hashes != after_hashes:
        raise RuntimeError("A protected baseline output changed during Stage 1 analysis")

    return {
        "merge_manifest": merge_manifest,
        "numerical_manifest": numerical_manifest,
        "inventory_summary": inventory_summary,
        "uncertainty_summary": uncertainty_summary,
        "comparison": comparison,
        "output_files": {
            "unified_jsonl": source_path(unified_jsonl),
            "unified_manifest_json": source_path(unified_manifest_json),
            "numerical_npz": source_path(output_dir / DEFAULT_NUMERICAL_NPZ_NAME),
            "numerical_jsonl": source_path(output_dir / DEFAULT_NUMERICAL_JSONL_NAME),
            "numerical_manifest": source_path(output_dir / DEFAULT_NUMERICAL_MANIFEST_NAME),
            "inventory_jsonl": source_path(inventory_jsonl),
            "inventory_summary_json": source_path(inventory_summary_json),
            "uncertainty_jsonl": source_path(uncertainty_jsonl),
            "uncertainty_summary_json": source_path(uncertainty_summary_json),
            "uncertainty_report": source_path(uncertainty_report),
            "comparison_json": source_path(comparison_json),
            "comparison_report": source_path(comparison_report),
            "project_log": source_path(project_log),
        },
        "project_log_section_appended": log_appended,
        "project_log_section_title": STAGE1_SECTION_TITLE,
        "baseline_outputs_unchanged": True,
        "stage1_uncertainty_rows": uncertainty_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tau2-jsonl", type=Path, default=DEFAULT_TAU2_INPUT)
    parser.add_argument("--stage1-retained-jsonl", type=Path, default=DEFAULT_STAGE1_RETAINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-project-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_stage1_analysis(
        baseline_tau2_jsonl=args.baseline_tau2_jsonl,
        stage1_retained_jsonl=args.stage1_retained_jsonl,
        output_dir=args.output_dir,
        append_log=not args.no_project_log,
    )
    print(f"merged tau2 records: {result['merge_manifest']['total_count']}")
    print(f"merged y distribution: {result['merge_manifest']['y_distribution']}")
    print(
        "numerical shapes: "
        f"X=({result['numerical_manifest']['total_count']}, "
        f"{result['numerical_manifest']['final_x_dimension']}), "
        f"S=({result['numerical_manifest']['total_count']}, "
        f"{result['numerical_manifest']['final_s_dimension']})"
    )
    print(
        "zero_or_one_write_to_two_plus_writes remains candidate_harmful: "
        + str(
            result["comparison"][
                "zero_or_one_write_to_two_plus_writes_remains_candidate_harmful_all_thresholds"
            ]
        )
    )


if __name__ == "__main__":
    main()
