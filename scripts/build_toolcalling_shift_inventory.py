#!/usr/bin/env python3
"""Build a descriptive inventory of candidate tool-calling context shifts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_toolcalling_numerical_representation import (  # noqa: E402
    API_BANK_SOURCE_DATASET,
    DEFAULT_API_BANK_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TAU2_INPUT,
    FULL_JSONL_NAME,
    FULL_NPZ_NAME,
    load_jsonl,
    write_json,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUMERICAL_JSONL = DEFAULT_OUTPUT_DIR / FULL_JSONL_NAME
DEFAULT_NUMERICAL_NPZ = DEFAULT_OUTPUT_DIR / FULL_NPZ_NAME
DEFAULT_INVENTORY_JSONL = DEFAULT_OUTPUT_DIR / "toolcalling_shift_inventory.jsonl"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "toolcalling_shift_inventory_summary.json"
DEFAULT_REPORT = REPO_ROOT / "docs/toolcalling_shift_inventory.md"
DEFAULT_MIN_GROUP_SIZE = 10

TAU2_OUTCOME_TYPE = "real_benchmark_task_outcome"
API_BANK_OUTCOME_TYPE = "synthetic_api_call_correctness"
API_BANK_WARNING = (
    "API-Bank delta_y values cannot be interpreted as real deployment success-rate "
    "shifts because the negative samples are synthetic corruptions and labels are "
    "balanced by construction."
)
FORBIDDEN_GROUP_FIELDS = {
    "corruption_type",
    "is_synthetic",
    "label_origin",
    "variant",
    "validation_error",
    "validation_status",
    "y",
}


def field_uses_forbidden_metadata(field_path: str, forbidden_field: str) -> bool:
    return forbidden_field in field_path.split(".")


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["y"]) for row in rows).items()))


def mean_y(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([row["y"] for row in rows]))


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def centroid_distance(array: np.ndarray, source_indices: list[int], target_indices: list[int]) -> float | None:
    if not source_indices or not target_indices:
        return None
    source_centroid = array[source_indices].mean(axis=0)
    target_centroid = array[target_indices].mean(axis=0)
    return finite_float(float(np.linalg.norm(target_centroid - source_centroid)))


def quantile_thresholds(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "lower_quantile": 0.25,
        "upper_quantile": 0.75,
        "lower_threshold": float(np.quantile(array, 0.25)),
        "upper_threshold": float(np.quantile(array, 0.75)),
    }


def descriptive_direction(delta_y: float | None) -> str | None:
    if delta_y is None:
        return None
    if delta_y < 0:
        return "candidate negative-outcome shift"
    if delta_y > 0:
        return "candidate positive-outcome shift"
    return "candidate stable-outcome shift"


def api_name_is_authentication(api_name: str | None) -> bool:
    if not api_name:
        return False
    lowered = api_name.lower()
    return "token" in lowered or "auth" in lowered or "password" in lowered


def numeric_value(row: dict[str, Any], path: str) -> float | None:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def make_analysis_rows(
    numerical_records: list[dict[str, Any]],
    unified_records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    unified_by_id = {record["sample_id"]: record for record in unified_records}
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    rows = []
    for numerical_record in numerical_records:
        sample_id = numerical_record["sample_id"]
        unified = unified_by_id[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "array_index": index_by_id[sample_id],
                "source_dataset": numerical_record["source_dataset"],
                "label_scope": numerical_record["label_scope"],
                "domain": numerical_record["metadata"].get("domain"),
                "y": int(numerical_record["y"]),
                "x_structural_features": numerical_record["x_structural_features"],
                "s_structural_features": numerical_record["s_structural_features"],
                "x_numeric_features": unified.get("x_numeric_features", {}),
                "s_numeric_features": unified.get("s_numeric_features", {}),
                "x_raw": unified.get("x_raw", {}),
                "s_raw": unified.get("s_raw", {}),
                "metadata": unified.get("metadata", {}),
            }
        )
    return rows


def evaluate_candidate(
    *,
    shift_id: str,
    dataset: str,
    family: str,
    source_name: str,
    target_name: str,
    rule: str,
    group_definition_fields: list[str],
    thresholds: dict[str, Any],
    rows: list[dict[str, Any]],
    source_predicate: Callable[[dict[str, Any]], bool],
    target_predicate: Callable[[dict[str, Any]], bool],
    arrays: dict[str, np.ndarray],
    min_group_size: int,
    outcome_type: str,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    forbidden_hits = sorted(
        field
        for field in FORBIDDEN_GROUP_FIELDS
        if any(
            field_uses_forbidden_metadata(definition, field)
            for definition in group_definition_fields
        )
    )
    source_rows = [row for row in rows if source_predicate(row)]
    target_rows = [row for row in rows if target_predicate(row)]
    source_count = len(source_rows)
    target_count = len(target_rows)
    status = "eligible"
    failure_reasons = []
    if forbidden_hits:
        status = "failed"
        failure_reasons.append(f"forbidden grouping fields used: {', '.join(forbidden_hits)}")
    if source_count < min_group_size:
        status = "failed"
        failure_reasons.append(f"source group size {source_count} is below minimum {min_group_size}")
    if target_count < min_group_size:
        status = "failed"
        failure_reasons.append(f"target group size {target_count} is below minimum {min_group_size}")

    source_y_mean = mean_y(source_rows)
    target_y_mean = mean_y(target_rows)
    delta_y = None
    if source_y_mean is not None and target_y_mean is not None:
        delta_y = float(target_y_mean - source_y_mean)

    source_indices = [row["array_index"] for row in source_rows]
    target_indices = [row["array_index"] for row in target_rows]
    x_distance = centroid_distance(arrays["X"], source_indices, target_indices)
    s_distance = centroid_distance(arrays["S"], source_indices, target_indices)
    warnings = list(extra_warnings or [])

    return {
        "shift_id": shift_id,
        "dataset": dataset,
        "family": family,
        "outcome_type": outcome_type,
        "status": status,
        "failure_reasons": failure_reasons,
        "source_group": source_name,
        "target_group": target_name,
        "source_sample_count": source_count,
        "target_sample_count": target_count,
        "source_y_mean": source_y_mean,
        "target_y_mean": target_y_mean,
        "raw_delta_y": delta_y,
        "descriptive_direction": descriptive_direction(delta_y),
        "source_label_counts": label_counts(source_rows),
        "target_label_counts": label_counts(target_rows),
        "x_centroid_distance": x_distance,
        "s_centroid_distance": s_distance,
        "grouping_rule": rule,
        "thresholds": thresholds,
        "min_group_size": min_group_size,
        "group_definition_fields": group_definition_fields,
        "group_definition_uses_y": any(field.endswith(".y") or field == "y" for field in group_definition_fields),
        "warnings": warnings,
    }


def quantile_candidate(
    *,
    shift_id: str,
    dataset: str,
    family: str,
    field_path: str,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    min_group_size: int,
    outcome_type: str,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    values = [numeric_value(row, field_path) for row in rows]
    numeric_values = [value for value in values if value is not None]
    thresholds = quantile_thresholds(numeric_values)
    lower = thresholds["lower_threshold"]
    upper = thresholds["upper_threshold"]
    if lower == upper:
        thresholds["degenerate_quantiles"] = True
    return evaluate_candidate(
        shift_id=shift_id,
        dataset=dataset,
        family=family,
        source_name=f"{field_path} <= q25 ({lower:g})",
        target_name=f"{field_path} >= q75 ({upper:g})",
        rule=(
            f"Source is the lower quartile of {field_path}; target is the upper quartile. "
            "Membership is defined before looking at y."
        ),
        group_definition_fields=[field_path],
        thresholds=thresholds,
        rows=rows,
        source_predicate=lambda row: (numeric_value(row, field_path) is not None)
        and numeric_value(row, field_path) <= lower,
        target_predicate=lambda row: (numeric_value(row, field_path) is not None)
        and numeric_value(row, field_path) >= upper,
        arrays=arrays,
        min_group_size=min_group_size,
        outcome_type=outcome_type,
        extra_warnings=extra_warnings,
    )


def build_tau2_candidates(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    min_group_size: int,
) -> list[dict[str, Any]]:
    tau2_rows = [row for row in rows if row["source_dataset"] == "tau2"]
    return [
        evaluate_candidate(
            shift_id="tau2_retail_to_airline",
            dataset="tau2",
            family="domain",
            source_name="retail",
            target_name="airline",
            rule="Source group has domain == retail; target group has domain == airline.",
            group_definition_fields=["domain"],
            thresholds={},
            rows=tau2_rows,
            source_predicate=lambda row: row["domain"] == "retail",
            target_predicate=lambda row: row["domain"] == "airline",
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
        evaluate_candidate(
            shift_id="tau2_no_write_to_write_required",
            dataset="tau2",
            family="expected_write_requirement",
            source_name="no write required",
            target_name="write required",
            rule=(
                "Source has expected_write_action_count == 0; target has "
                "expected_write_action_count >= 1."
            ),
            group_definition_fields=["x_numeric_features.expected_write_action_count"],
            thresholds={"source_max": 0, "target_min": 1},
            rows=tau2_rows,
            source_predicate=lambda row: numeric_value(row, "x_numeric_features.expected_write_action_count") == 0,
            target_predicate=lambda row: (
                numeric_value(row, "x_numeric_features.expected_write_action_count") or 0
            )
            >= 1,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
        evaluate_candidate(
            shift_id="tau2_zero_or_one_write_to_two_plus_writes",
            dataset="tau2",
            family="expected_write_count",
            source_name="zero or one expected write",
            target_name="two or more expected writes",
            rule=(
                "Source has expected_write_action_count <= 1; target has "
                "expected_write_action_count >= 2."
            ),
            group_definition_fields=["x_numeric_features.expected_write_action_count"],
            thresholds={"source_max": 1, "target_min": 2},
            rows=tau2_rows,
            source_predicate=lambda row: (
                numeric_value(row, "x_numeric_features.expected_write_action_count") or 0
            )
            <= 1,
            target_predicate=lambda row: (
                numeric_value(row, "x_numeric_features.expected_write_action_count") or 0
            )
            >= 2,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
        quantile_candidate(
            shift_id="tau2_few_to_many_expected_actions",
            dataset="tau2",
            family="expected_action_count",
            field_path="x_numeric_features.expected_action_count",
            rows=tau2_rows,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
        quantile_candidate(
            shift_id="tau2_short_to_long_trajectory",
            dataset="tau2",
            family="trajectory_length",
            field_path="s_structural_features.trajectory_length",
            rows=tau2_rows,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
        quantile_candidate(
            shift_id="tau2_few_to_many_tool_calls",
            dataset="tau2",
            family="observed_tool_call_count",
            field_path="metadata.num_tool_calls",
            rows=tau2_rows,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=TAU2_OUTCOME_TYPE,
        ),
    ]


def build_api_bank_candidates(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    min_group_size: int,
) -> list[dict[str, Any]]:
    api_rows = [row for row in rows if row["source_dataset"] == API_BANK_SOURCE_DATASET]
    warning = [API_BANK_WARNING]
    candidates = [
        evaluate_candidate(
            shift_id="api_bank_no_auth_to_auth_required",
            dataset=API_BANK_SOURCE_DATASET,
            family="authentication_requirement",
            source_name="no authentication required",
            target_name="authentication required",
            rule=(
                "Source has no authentication signal in the pre-call context and candidate API; "
                "target has requires_authentication == 1 or an authentication-like API name."
            ),
            group_definition_fields=[
                "x_structural_features.requires_authentication",
                "s_raw.api_name",
            ],
            thresholds={"source_value": 0, "target_value": 1},
            rows=api_rows,
            source_predicate=lambda row: numeric_value(row, "x_structural_features.requires_authentication") == 0
            and not api_name_is_authentication(row["s_raw"].get("api_name")),
            target_predicate=lambda row: numeric_value(row, "x_structural_features.requires_authentication") == 1
            or api_name_is_authentication(row["s_raw"].get("api_name")),
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning,
        ),
        quantile_candidate(
            shift_id="api_bank_short_to_long_dialogue_history",
            dataset=API_BANK_SOURCE_DATASET,
            family="dialogue_history_length",
            field_path="x_numeric_features.history_length",
            rows=api_rows,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning,
        ),
        evaluate_candidate(
            shift_id="api_bank_one_tool_to_multiple_tools_available",
            dataset=API_BANK_SOURCE_DATASET,
            family="available_tool_count",
            source_name="one tool available",
            target_name="multiple tools available",
            rule="Source has available_api_count == 1; target has available_api_count >= 2.",
            group_definition_fields=["x_numeric_features.available_api_count"],
            thresholds={"source_value": 1, "target_min": 2},
            rows=api_rows,
            source_predicate=lambda row: numeric_value(row, "x_numeric_features.available_api_count") == 1,
            target_predicate=lambda row: (
                numeric_value(row, "x_numeric_features.available_api_count") or 0
            )
            >= 2,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning,
        ),
        quantile_candidate(
            shift_id="api_bank_few_to_many_arguments",
            dataset=API_BANK_SOURCE_DATASET,
            family="candidate_argument_count",
            field_path="s_structural_features.argument_count",
            rows=api_rows,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning,
        ),
        evaluate_candidate(
            shift_id="api_bank_simple_call_to_multi_step_context",
            dataset=API_BANK_SOURCE_DATASET,
            family="previous_api_call_count",
            source_name="simple API call",
            target_name="multi-step context",
            rule=(
                "Source has previous_api_call_count == 0; target has "
                "previous_api_call_count >= 1 in the pre-call dialogue history."
            ),
            group_definition_fields=["x_numeric_features.previous_api_call_count"],
            thresholds={"source_value": 0, "target_min": 1},
            rows=api_rows,
            source_predicate=lambda row: numeric_value(row, "x_numeric_features.previous_api_call_count") == 0,
            target_predicate=lambda row: (
                numeric_value(row, "x_numeric_features.previous_api_call_count") or 0
            )
            >= 1,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning,
        ),
    ]

    domain_values = sorted({row["domain"] for row in api_rows if row["domain"]})
    candidates.append(
        evaluate_candidate(
            shift_id="api_bank_domain_or_tool_family_comparison",
            dataset=API_BANK_SOURCE_DATASET,
            family="domain_or_tool_family",
            source_name="reliable source domain/tool family",
            target_name="reliable target domain/tool family",
            rule=(
                "Domain/tool-family comparison is recorded as unsupported because the unified "
                "API-Bank records have no reliable domain metadata."
            ),
            group_definition_fields=["domain"],
            thresholds={"observed_non_null_domain_values": domain_values},
            rows=api_rows,
            source_predicate=lambda row: False,
            target_predicate=lambda row: False,
            arrays=arrays,
            min_group_size=min_group_size,
            outcome_type=API_BANK_OUTCOME_TYPE,
            extra_warnings=warning + ["No reliable API-Bank domain metadata is available."],
        )
    )
    return candidates


def build_summary(inventory: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = dict(sorted(Counter(item["status"] for item in inventory).items()))
    dataset_counts = dict(sorted(Counter(row["source_dataset"] for row in rows).items()))
    outcome_types = {
        dataset: sorted(
            {
                item["outcome_type"]
                for item in inventory
                if item["dataset"] == dataset
            }
        )
        for dataset in sorted({item["dataset"] for item in inventory})
    }
    return {
        "total_candidate_shift_count": len(inventory),
        "status_counts": status_counts,
        "candidate_count_by_dataset": dict(sorted(Counter(item["dataset"] for item in inventory).items())),
        "record_count_by_dataset": dataset_counts,
        "min_group_size": min(item["min_group_size"] for item in inventory) if inventory else None,
        "outcome_types_by_dataset": outcome_types,
        "forbidden_group_definition_fields": sorted(FORBIDDEN_GROUP_FIELDS),
        "group_definitions_using_y": [
            item["shift_id"] for item in inventory if item["group_definition_uses_y"]
        ],
        "api_bank_warning": API_BANK_WARNING,
        "suitable_for_later_harmful_harmless_analysis": [
            item["shift_id"]
            for item in inventory
            if item["dataset"] == "tau2" and item["status"] == "eligible"
        ],
        "unsuitable_for_final_harmful_harmless_analysis": [
            item["shift_id"]
            for item in inventory
            if item["dataset"] == API_BANK_SOURCE_DATASET or item["status"] != "eligible"
        ],
    }


def markdown_table(inventory: list[dict[str, Any]], dataset: str) -> str:
    rows = [item for item in inventory if item["dataset"] == dataset]
    lines = [
        "| Shift | Status | Source n | Target n | Delta_y | X distance | S distance | Direction |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        delta = "" if item["raw_delta_y"] is None else f"{item['raw_delta_y']:.4f}"
        x_distance = "" if item["x_centroid_distance"] is None else f"{item['x_centroid_distance']:.4f}"
        s_distance = "" if item["s_centroid_distance"] is None else f"{item['s_centroid_distance']:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    item["shift_id"],
                    item["status"],
                    str(item["source_sample_count"]),
                    str(item["target_sample_count"]),
                    delta,
                    x_distance,
                    s_distance,
                    item["descriptive_direction"] or "",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(inventory: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eligible_tau2 = summary["suitable_for_later_harmful_harmless_analysis"]
    unsuitable = summary["unsuitable_for_final_harmful_harmless_analysis"]
    rules = "\n".join(
        f"- `{item['shift_id']}`: {item['grouping_rule']} Fields: "
        f"{', '.join(item['group_definition_fields'])}. Thresholds: "
        f"{json.dumps(item['thresholds'], sort_keys=True)}."
        for item in inventory
    )
    report = f"""# Tool-Calling Candidate Shift Inventory

## Formulation

This inventory defines deterministic source and target groups from existing metadata and structural features, then computes descriptive statistics only. Group membership is defined without using `y`, `variant`, `corruption_type`, `label_origin`, `is_synthetic`, validation status, or validation error fields.

`Delta_Y = P_target(Y=1) - P_source(Y=1)` is reported descriptively. A negative `delta_y` is not a final harmful-shift label, and a near-zero `delta_y` is not a final harmless-shift label. Final classification requires confidence intervals and a practical significance threshold.

## Grouping Rules

Minimum group size: {summary['min_group_size']}.

{rules}

## Candidate Shifts By Dataset

### tau2

Outcome type: `{TAU2_OUTCOME_TYPE}`.

{markdown_table(inventory, 'tau2')}

### API-Bank

Outcome type: `{API_BANK_OUTCOME_TYPE}`.

{markdown_table(inventory, API_BANK_SOURCE_DATASET)}

## Validity Warnings

- tau2 labels are real benchmark task outcomes.
- API-Bank labels are synthetic API-call correctness labels, not task-level deployment success labels.
- {API_BANK_WARNING}
- Task-level and API-call-level labels are kept separate and are not treated as equivalent.
- The terms candidate negative-outcome shift, candidate stable-outcome shift, and candidate positive-outcome shift are descriptive only.

## Later Analysis Suitability

Potentially suitable for later harmful/harmless analysis after confidence intervals and a practical threshold are defined:

{chr(10).join(f'- `{shift_id}`' for shift_id in eligible_tau2)}

Unsuitable for final harmful/harmless conclusions at this stage because labels are synthetic, domains are unavailable, or groups are too small:

{chr(10).join(f'- `{shift_id}`' for shift_id in unsuitable)}
"""
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    numerical_jsonl: Path = DEFAULT_NUMERICAL_JSONL,
    numerical_npz: Path = DEFAULT_NUMERICAL_NPZ,
    tau2_input: Path = DEFAULT_TAU2_INPUT,
    api_bank_input: Path = DEFAULT_API_BANK_INPUT,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    report_path: Path = DEFAULT_REPORT,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numerical_records = load_jsonl(numerical_jsonl)
    unified_records = load_jsonl(tau2_input) + load_jsonl(api_bank_input)
    arrays = load_npz_arrays(numerical_npz)
    rows = make_analysis_rows(numerical_records, unified_records, arrays)
    inventory = build_tau2_candidates(rows, arrays, min_group_size)
    inventory.extend(build_api_bank_candidates(rows, arrays, min_group_size))
    summary = build_summary(inventory, rows)
    if write_outputs:
        write_jsonl(inventory, inventory_jsonl)
        write_json(summary, summary_json)
        write_report(inventory, summary, report_path)
    return inventory, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical-jsonl", type=Path, default=DEFAULT_NUMERICAL_JSONL)
    parser.add_argument("--numerical-npz", type=Path, default=DEFAULT_NUMERICAL_NPZ)
    parser.add_argument("--tau2-input", type=Path, default=DEFAULT_TAU2_INPUT)
    parser.add_argument("--api-bank-input", type=Path, default=DEFAULT_API_BANK_INPUT)
    parser.add_argument("--inventory-jsonl", type=Path, default=DEFAULT_INVENTORY_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory, summary = build_outputs(
        numerical_jsonl=args.numerical_jsonl,
        numerical_npz=args.numerical_npz,
        tau2_input=args.tau2_input,
        api_bank_input=args.api_bank_input,
        inventory_jsonl=args.inventory_jsonl,
        summary_json=args.summary_json,
        report_path=args.report_path,
        min_group_size=args.min_group_size,
    )
    print(f"wrote {len(inventory)} candidate shift rows")
    print(f"status counts: {summary['status_counts']}")
    print(API_BANK_WARNING)


if __name__ == "__main__":
    main()
