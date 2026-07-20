#!/usr/bin/env python3
"""Build a BFCL v4 non-live single-turn candidate shift inventory."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl"
)
DEFAULT_SOURCE_SUMMARY_JSON = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_1240_summary.json"
)
DEFAULT_OUTPUT_JSONL = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_shift_inventory.jsonl"
)
DEFAULT_OUTPUT_SUMMARY_JSON = (
    REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_shift_inventory_summary.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs/bfcl_shift_inventory.md"
DEFAULT_AUDIT_REPORT = REPO_ROOT / "docs/bfcl_data_source_audit.md"

MODEL_NAME = "gpt-4o-mini-2024-07-18-FC"
DATASET_NAME = "BFCL v4 non-live single-turn subset"
SOURCE_DATASET = "bfcl_v4"
LABEL_SCOPE = "test_case_level"
LABEL_ORIGIN = "bfcl_evaluator"
OUTCOME_TYPE = "bfcl_test_case_correctness"
X_FIELD = "x_raw"
S_FIELD = "s_raw"
EXPECTED_TOTAL = 1_240
EXPECTED_Y_DISTRIBUTION = {"0": 181, "1": 1059}
EXPECTED_CATEGORY_TOTALS = {
    "simple_python": {"total": 400, "y_1": 350, "y_0": 50},
    "multiple": {"total": 200, "y_1": 176, "y_0": 24},
    "parallel": {"total": 200, "y_1": 174, "y_0": 26},
    "parallel_multiple": {"total": 200, "y_1": 160, "y_0": 40},
    "irrelevance": {"total": 240, "y_1": 199, "y_0": 41},
}
FORBIDDEN_GROUP_FIELDS = {
    "is_synthetic",
    "label_origin",
    "label_scope",
    "metadata.evaluation_error",
    "metadata.evaluation_error_type",
    "s_raw",
    "y",
}
PRIMARY_SHIFT_PAIRS = [
    ("simple_python", "multiple"),
    ("simple_python", "parallel"),
    ("simple_python", "parallel_multiple"),
    ("multiple", "parallel_multiple"),
    ("parallel", "parallel_multiple"),
]
BEHAVIORAL_SHIFT_PAIRS = [("simple_python", "irrelevance")]

EXPLORATORY_WARNING = (
    "This BFCL analysis is exploratory, category-level, and non-causal; it uses "
    "the 1,240 sample-level evaluator labels only."
)
NO_IID_MIXING_WARNING = (
    "BFCL rows are analyzed separately and are not pooled with tau2 or API-Bank "
    "as IID samples."
)
NO_Y_GROUPING_WARNING = (
    "Candidate groups are defined from BFCL category metadata only; y is not used "
    "to define source or target membership."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["y"]) for row in rows).items()))


def success_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("success_rate requires at least one row")
    return sum(int(row["y"]) for row in rows) / len(rows)


def category_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats = {}
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        counts = label_counts(category_rows)
        stats[category] = {
            "total": len(category_rows),
            "y_1": counts.get("1", 0),
            "y_0": counts.get("0", 0),
            "success_rate": success_rate(category_rows),
        }
    return stats


def validate_source_rows(
    rows: list[dict[str, Any]],
    source_summary: dict[str, Any],
) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} BFCL rows, found {len(rows)}")
    if source_summary.get("total") != EXPECTED_TOTAL:
        raise ValueError("Source summary total does not match expected BFCL total")
    if source_summary.get("model") != MODEL_NAME:
        raise ValueError("Source summary model does not match expected BFCL model")
    if source_summary.get("label_scope") != LABEL_SCOPE:
        raise ValueError("Source summary label_scope does not match expected value")
    if source_summary.get("label_origin") != LABEL_ORIGIN:
        raise ValueError("Source summary label_origin does not match expected value")
    if source_summary.get("is_synthetic") is not False:
        raise ValueError("Source summary must mark BFCL rows as non-synthetic")

    for row in rows:
        if row.get("id") in {None, ""}:
            raise ValueError("BFCL row is missing id")
        if row.get(X_FIELD) is None:
            raise ValueError(f"Missing {X_FIELD} for row {row.get('id')}")
        if row.get(S_FIELD) is None:
            raise ValueError(f"Missing {S_FIELD} for row {row.get('id')}")
        if row.get("y") not in {0, 1}:
            raise ValueError(f"Unexpected y for row {row.get('id')}")

    id_counts = Counter(row["id"] for row in rows)
    duplicate_ids = sorted(
        row_id for row_id, count in id_counts.items() if count > 1
    )
    if duplicate_ids:
        preview = ", ".join(str(row_id) for row_id in duplicate_ids[:5])
        raise ValueError(
            f"Duplicate BFCL row ids found: {len(duplicate_ids)} duplicate id(s); "
            f"examples: {preview}"
        )

    row_y_distribution = label_counts(rows)
    if row_y_distribution != EXPECTED_Y_DISTRIBUTION:
        raise ValueError(
            f"Expected y distribution {EXPECTED_Y_DISTRIBUTION}, found "
            f"{row_y_distribution}"
        )
    if source_summary.get("y_distribution") != EXPECTED_Y_DISTRIBUTION:
        raise ValueError("Source summary y_distribution does not match expected values")

    stats = category_stats(rows)
    for category, expected in EXPECTED_CATEGORY_TOTALS.items():
        observed = stats.get(category)
        if observed is None:
            raise ValueError(f"Missing expected BFCL category: {category}")
        comparable = {
            "total": observed["total"],
            "y_1": observed["y_1"],
            "y_0": observed["y_0"],
        }
        if comparable != expected:
            raise ValueError(
                f"Category {category} expected {expected}, found {comparable}"
            )

    for row in rows:
        if row.get("source_dataset") != SOURCE_DATASET:
            raise ValueError(f"Unexpected source_dataset for row {row.get('id')}")
        if row.get("model") != MODEL_NAME:
            raise ValueError(f"Unexpected model for row {row.get('id')}")
        if row.get("label_scope") != LABEL_SCOPE:
            raise ValueError(f"Unexpected label_scope for row {row.get('id')}")
        if row.get("label_origin") != LABEL_ORIGIN:
            raise ValueError(f"Unexpected label_origin for row {row.get('id')}")
        if row.get("is_synthetic") is not False:
            raise ValueError(f"Unexpected is_synthetic for row {row.get('id')}")


def shift_id(source_category: str, target_category: str) -> str:
    return f"bfcl_{source_category}_to_{target_category}"


def category_rows(rows: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["category"] == category]


def category_label_counts(rows: list[dict[str, Any]], category: str) -> dict[str, int]:
    return label_counts(category_rows(rows, category))


def build_shift_row(
    rows: list[dict[str, Any]],
    *,
    source_category: str,
    target_category: str,
    family: str,
    shift_type: str,
    is_primary_complexity_shift: bool,
) -> dict[str, Any]:
    source_rows = category_rows(rows, source_category)
    target_rows = category_rows(rows, target_category)
    source_rate = success_rate(source_rows)
    target_rate = success_rate(target_rows)
    source_ids = {row["id"] for row in source_rows}
    target_ids = {row["id"] for row in target_rows}
    group_definition_fields = ["category"]
    forbidden_hits = sorted(FORBIDDEN_GROUP_FIELDS & set(group_definition_fields))
    status = "eligible"
    failure_reasons = []
    if forbidden_hits:
        status = "failed"
        failure_reasons.append(
            "forbidden grouping fields used: " + ", ".join(forbidden_hits)
        )
    return {
        "shift_id": shift_id(source_category, target_category),
        "dataset": "bfcl_v4_non_live",
        "source_dataset": SOURCE_DATASET,
        "model": MODEL_NAME,
        "family": family,
        "shift_type": shift_type,
        "is_primary_complexity_shift": is_primary_complexity_shift,
        "outcome_type": OUTCOME_TYPE,
        "status": status,
        "failure_reasons": failure_reasons,
        "source_group": source_category,
        "target_group": target_category,
        "source_sample_count": len(source_rows),
        "target_sample_count": len(target_rows),
        "source_y_mean": source_rate,
        "target_y_mean": target_rate,
        "raw_delta_y": target_rate - source_rate,
        "source_label_counts": category_label_counts(rows, source_category),
        "target_label_counts": category_label_counts(rows, target_category),
        "grouping_rule": (
            f"Source rows have category == {source_category}; target rows have "
            f"category == {target_category}. Membership is defined before looking "
            "at y."
        ),
        "group_definition_fields": group_definition_fields,
        "group_definition_uses_y": False,
        "source_target_overlap_count": len(source_ids & target_ids),
        "warnings": [
            EXPLORATORY_WARNING,
            NO_IID_MIXING_WARNING,
            NO_Y_GROUPING_WARNING,
        ],
        "label_scope": LABEL_SCOPE,
        "label_origin": LABEL_ORIGIN,
        "is_synthetic": False,
    }


def build_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    for source_category, target_category in PRIMARY_SHIFT_PAIRS:
        inventory.append(
            build_shift_row(
                rows,
                source_category=source_category,
                target_category=target_category,
                family="bfcl_complexity",
                shift_type="primary_complexity",
                is_primary_complexity_shift=True,
            )
        )
    for source_category, target_category in BEHAVIORAL_SHIFT_PAIRS:
        inventory.append(
            build_shift_row(
                rows,
                source_category=source_category,
                target_category=target_category,
                family="bfcl_behavioral_abstention",
                shift_type="behavioral_abstention",
                is_primary_complexity_shift=False,
            )
        )
    return inventory


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def build_summary(
    rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    category_summary = category_stats(rows)
    return {
        "dataset": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "model": MODEL_NAME,
        "total": len(rows),
        "label_scope": LABEL_SCOPE,
        "label_origin": LABEL_ORIGIN,
        "is_synthetic": False,
        "x_representation_field": X_FIELD,
        "s_representation_field": S_FIELD,
        "y_distribution": label_counts(rows),
        "overall_success_rate": finite_float(success_rate(rows)),
        "categories": category_summary,
        "source_summary_path": str(DEFAULT_SOURCE_SUMMARY_JSON.relative_to(REPO_ROOT)),
        "source_summary_total": source_summary.get("total"),
        "shift_count": len(inventory),
        "primary_complexity_shift_count": sum(
            row["is_primary_complexity_shift"] for row in inventory
        ),
        "behavioral_abstention_shift_count": sum(
            row["shift_type"] == "behavioral_abstention" for row in inventory
        ),
        "eligible_shift_count": sum(row["status"] == "eligible" for row in inventory),
        "group_definitions_using_y": [
            row["shift_id"] for row in inventory if row["group_definition_uses_y"]
        ],
        "group_definition_fields": sorted(
            {field for row in inventory for field in row["group_definition_fields"]}
        ),
        "primary_candidate_shift_ids": [
            row["shift_id"] for row in inventory if row["is_primary_complexity_shift"]
        ],
        "behavioral_abstention_shift_ids": [
            row["shift_id"]
            for row in inventory
            if row["shift_type"] == "behavioral_abstention"
        ],
        "warnings": [
            EXPLORATORY_WARNING,
            NO_IID_MIXING_WARNING,
            NO_Y_GROUPING_WARNING,
        ],
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_inventory_report(
    inventory: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Shift | Type | Source n | Target n | Source rate | Target rate | Delta_y |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in inventory:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    row["shift_type"],
                    str(row["source_sample_count"]),
                    str(row["target_sample_count"]),
                    fmt_float(row["source_y_mean"]),
                    fmt_float(row["target_y_mean"]),
                    fmt_float(row["raw_delta_y"]),
                ]
            )
            + " |"
        )

    category_lines = [
        "| Category | n | Y=1 | Y=0 | Success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, stats in summary["categories"].items():
        category_lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(stats["total"]),
                    str(stats["y_1"]),
                    str(stats["y_0"]),
                    fmt_float(stats["success_rate"]),
                ]
            )
            + " |"
        )

    report = f"""# BFCL Shift Inventory

## Purpose

Define reproducible candidate source and target groups for the BFCL v4 non-live single-turn context-shift analysis. Groups are defined only from BFCL category metadata and never from `y`.

## Dataset

- Dataset: {summary["dataset"]}
- Model: `{summary["model"]}`
- Rows: {summary["total"]:,}
- Label scope: `{summary["label_scope"]}`
- Label origin: `{summary["label_origin"]}`
- Synthetic rows: `{summary["is_synthetic"]}`
- Outcome: `{OUTCOME_TYPE}`
- X representation field: `{summary["x_representation_field"]}`
- S representation field: `{summary["s_representation_field"]}`

## Category Summary

{chr(10).join(category_lines)}

## Candidate Shifts

Primary complexity shifts:

{chr(10).join(f"- `{shift_id}`" for shift_id in summary["primary_candidate_shift_ids"])}

Separately labeled behavioral/abstention shift:

{chr(10).join(f"- `{shift_id}`" for shift_id in summary["behavioral_abstention_shift_ids"])}

## Inventory Table

{chr(10).join(lines)}

## Constraints

- This is exploratory analysis only.
- The inventory makes no causal, deployment-safe, or retraining-required claims.
- BFCL rows are not mixed with tau2 or API-Bank as IID samples.
- `label_scope` and `label_origin` are preserved in the outputs.
- Candidate groups are defined from `category`; `y`, evaluator errors, and model responses are not used for membership.
- The analysis relies on the 1,240 sample-level labels, not partial leaderboard overall scores.
"""
    path.write_text(report, encoding="utf-8")


def write_audit_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    category_lines = [
        "| Category | n | Y=1 | Y=0 | Success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, stats in summary["categories"].items():
        category_lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(stats["total"]),
                    str(stats["y_1"]),
                    str(stats["y_0"]),
                    fmt_float(stats["success_rate"]),
                ]
            )
            + " |"
        )
    report = f"""# BFCL Data Source Audit

## Source Artifacts

- Sample labels: `data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl`
- Source summary: `data/processed/bfcl/bfcl_v4_non_live_1240_summary.json`

## Validated Facts

- Dataset: {summary["dataset"]}
- Model: `{summary["model"]}`
- Total rows: {len(rows):,}
- Label scope: `{summary["label_scope"]}`
- Label origin: `{summary["label_origin"]}`
- Synthetic rows: `{summary["is_synthetic"]}`
- Y distribution: `{summary["y_distribution"]}`
- X representation field: `{summary["x_representation_field"]}`
- S representation field: `{summary["s_representation_field"]}`

{chr(10).join(category_lines)}

## Audit Decisions

- Treat each BFCL row as one evaluator-labeled test case.
- Preserve `label_scope={summary["label_scope"]}` and `label_origin={summary["label_origin"]}` in derived outputs.
- Define candidate groups from category metadata only.
- Exclude `y`, `s_raw`, evaluator error fields, `label_scope`, `label_origin`, and `is_synthetic` from group membership rules.
- Do not pool these BFCL rows with tau2 or API-Bank as IID samples.
- Do not use partial BFCL leaderboard overall scores; all estimates come from the 1,240 sample-level labels.

## Limitations

This audit verifies the local processed artifacts and their internal counts. It does not re-run the upstream BFCL evaluator or establish causal relationships between category membership and correctness.
"""
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    input_jsonl: Path = DEFAULT_INPUT_JSONL,
    source_summary_json: Path = DEFAULT_SOURCE_SUMMARY_JSON,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    summary_json: Path = DEFAULT_OUTPUT_SUMMARY_JSON,
    report_path: Path = DEFAULT_REPORT,
    audit_report_path: Path = DEFAULT_AUDIT_REPORT,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(input_jsonl)
    source_summary = json.loads(source_summary_json.read_text(encoding="utf-8"))
    validate_source_rows(rows, source_summary)
    inventory = build_inventory(rows)
    summary = build_summary(rows, inventory, source_summary)
    if write_outputs:
        write_jsonl(inventory, output_jsonl)
        write_json(summary, summary_json)
        write_inventory_report(inventory, summary, report_path)
        write_audit_report(rows, summary, audit_report_path)
    return inventory, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--source-summary-json",
        type=Path,
        default=DEFAULT_SOURCE_SUMMARY_JSON,
    )
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit-report-path", type=Path, default=DEFAULT_AUDIT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory, summary = build_outputs(
        input_jsonl=args.input_jsonl,
        source_summary_json=args.source_summary_json,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        report_path=args.report_path,
        audit_report_path=args.audit_report_path,
    )
    print(f"wrote {len(inventory)} BFCL shift inventory rows")
    print(
        "primary complexity shifts: "
        f"{summary['primary_complexity_shift_count']}; "
        "behavioral/abstention shifts: "
        f"{summary['behavioral_abstention_shift_count']}"
    )
    print(EXPLORATORY_WARNING)


if __name__ == "__main__":
    main()
