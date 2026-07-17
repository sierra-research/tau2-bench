#!/usr/bin/env python3
"""Build unified wrappers for the tau2 and API-Bank tool-calling pilots."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAU2_INPUT = (
    REPO_ROOT
    / "data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl"
)
DEFAULT_API_BANK_INPUT = (
    REPO_ROOT.parent
    / "DAMO-ConvAI/api-bank/data/processed/apibank_api_call_correctness_pilot.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed"

TAU2_OUTPUT_NAME = "unified_toolcalling_tau2.jsonl"
API_BANK_OUTPUT_NAME = "unified_toolcalling_apibank.jsonl"
MANIFEST_OUTPUT_NAME = "unified_toolcalling_manifest.json"

SCHEMA_FIELDS = [
    "sample_id",
    "source_dataset",
    "domain",
    "label_scope",
    "label_origin",
    "is_synthetic",
    "x_raw",
    "s_raw",
    "x_numeric_features",
    "s_numeric_features",
    "y",
    "metadata",
]
TAU2_LABEL_ORIGIN = "tau2_benchmark_reward"
API_BANK_POSITIVE_LABEL_ORIGIN = "reference_api_call"
API_BANK_NEGATIVE_LABEL_ORIGIN = "synthetic_corruption"
TAU2_LABEL_SCOPE = "task_level"
API_BANK_LABEL_SCOPE = "api_call_level"
TAU2_SOURCE_DATASET = "tau2"
API_BANK_SOURCE_DATASET = "api_bank"
TAU2_EXPECTED_COUNT = 93
API_BANK_EXPECTED_COUNT = 1016
API_BANK_METADATA_ONLY_LEAKAGE_FIELDS = {"corruption_type", "variant"}
COMPATIBILITY_WARNINGS = [
    "tau2 uses task-level labels while API-Bank uses API-call-level labels.",
    "tau2 labels are benchmark outcomes; API-Bank negative labels are synthetic corruptions.",
    "The datasets have different X dimensions.",
    "The datasets have different S representations.",
    "These datasets should not yet be treated as IID samples from one task.",
]


def json_default(value: Any) -> Any:
    """Convert NumPy scalar-like values without importing NumPy directly."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=json_default,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def as_python_number(value: Any) -> int | float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def tau2_event_feature_names(length: int) -> list[str]:
    return [f"event_{idx:02d}" for idx in range(length)]


def build_tau2_records(tau2_input: Path = DEFAULT_TAU2_INPUT) -> list[dict[str, Any]]:
    with tau2_input.open("rb") as handle:
        dataset = pickle.load(handle)

    feature_names = list(dataset["feature_names"])
    x_rows = dataset["X"]
    y_values = dataset["y"]
    trajectories = dataset["traj"]["s"]
    metadata_rows = list(dataset["metadata"])

    if not (len(x_rows) == len(y_values) == len(trajectories) == len(metadata_rows)):
        raise ValueError("tau2 input has inconsistent X, y, S, and metadata lengths")

    records = []
    for idx, (x_row, y_value, s_row, metadata) in enumerate(
        zip(x_rows, y_values, trajectories, metadata_rows, strict=True)
    ):
        domain = metadata.get("domain")
        task_id = metadata.get("task_id")
        if domain is None or task_id is None:
            sample_id = f"tau2:row_{idx:04d}"
        else:
            sample_id = f"tau2:{domain}:task_{task_id}"

        event_values = [int(as_python_number(value)) for value in s_row.tolist()]
        event_features = dict(zip(tau2_event_feature_names(len(event_values)), event_values, strict=True))
        x_features = {
            name: as_python_number(value)
            for name, value in zip(feature_names, x_row.tolist(), strict=True)
        }
        label = 1 if int(as_python_number(y_value)) == 1 else 0

        record_metadata = copy.deepcopy(metadata)
        record_metadata.update(
            {
                "sample_id": sample_id,
                "source_dataset": TAU2_SOURCE_DATASET,
                "source_input_path": str(tau2_input.relative_to(REPO_ROOT))
                if tau2_input.is_relative_to(REPO_ROOT)
                else str(tau2_input),
                "trajectory_event_encoding": {
                    "0": "padding",
                    "1": "user_message",
                    "2": "assistant_message",
                    "3": "assistant_read_tool_call",
                    "4": "assistant_write_tool_call",
                    "5": "successful_tool_result",
                    "6": "errored_tool_result",
                    "7": "end_of_trajectory",
                },
            }
        )

        records.append(
            {
                "sample_id": sample_id,
                "source_dataset": TAU2_SOURCE_DATASET,
                "domain": domain,
                "label_scope": TAU2_LABEL_SCOPE,
                "label_origin": TAU2_LABEL_ORIGIN,
                "is_synthetic": False,
                "x_raw": {},
                "s_raw": event_values,
                "x_numeric_features": x_features,
                "s_numeric_features": event_features,
                "y": label,
                "metadata": record_metadata,
            }
        )

    return records


def api_bank_domain(_: dict[str, Any]) -> None:
    return None


def build_api_bank_records(
    api_bank_input: Path = DEFAULT_API_BANK_INPUT,
) -> list[dict[str, Any]]:
    source_records = load_jsonl(api_bank_input)
    records = []
    for idx, source_record in enumerate(source_records):
        source_metadata = copy.deepcopy(source_record.get("metadata") or {})
        old_sample_id = source_metadata.get("sample_id") or f"row_{idx:04d}"
        sample_id = f"api_bank:{old_sample_id}"
        variant = source_metadata.get("variant")
        label = int(source_record["y"])

        if variant == "positive" or label == 1:
            label_origin = API_BANK_POSITIVE_LABEL_ORIGIN
            is_synthetic = False
        else:
            label_origin = API_BANK_NEGATIVE_LABEL_ORIGIN
            is_synthetic = True

        source_metadata.update(
            {
                "sample_id": sample_id,
                "original_sample_id": old_sample_id,
                "source_dataset": API_BANK_SOURCE_DATASET,
                "source_input_path": str(api_bank_input),
            }
        )

        records.append(
            {
                "sample_id": sample_id,
                "source_dataset": API_BANK_SOURCE_DATASET,
                "domain": api_bank_domain(source_record),
                "label_scope": API_BANK_LABEL_SCOPE,
                "label_origin": label_origin,
                "is_synthetic": is_synthetic,
                "x_raw": copy.deepcopy(source_record["x_raw"]),
                "s_raw": copy.deepcopy(source_record["s_raw"]),
                "x_numeric_features": copy.deepcopy(source_record["x_numeric_features"]),
                "s_numeric_features": copy.deepcopy(source_record["s_numeric_features"]),
                "y": 1 if label == 1 else 0,
                "metadata": source_metadata,
            }
        )
    return records


def label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["y"]) for record in records)
    return dict(sorted(counts.items()))


def distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(
        "null" if record.get(field) is None else str(record.get(field))
        for record in records
    )
    return dict(sorted(counts.items()))


def feature_names(records: list[dict[str, Any]], field: str) -> list[str]:
    names: set[str] = set()
    for record in records:
        value = record.get(field)
        if isinstance(value, dict):
            names.update(value)
    return sorted(names)


def missing_field_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(1 for record in records if field not in record or record[field] is None)
        for field in SCHEMA_FIELDS
    }


def duplicate_sample_ids(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(record["sample_id"] for record in records)
    return sorted(sample_id for sample_id, count in counts.items() if count > 1)


def s_numeric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = feature_names(records, "s_numeric_features")
    value_kinds = Counter()
    dimensions = Counter()
    for record in records:
        values = record["s_numeric_features"]
        dimensions[len(values)] += 1
        for value in values.values():
            value_kinds[type(value).__name__] += 1
    return {
        "feature_names": names,
        "dimension": len(names),
        "record_dimensions": dict(sorted(dimensions.items())),
        "value_type_distribution": dict(sorted(value_kinds.items())),
    }


def tau2_missing_metadata_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    expected = [
        "domain",
        "task_id",
        "source_result_folder",
        "termination_reason",
        "reward",
        "db_match",
        "num_messages",
        "num_tool_calls",
        "expected_action_count",
        "expected_read_action_count",
        "expected_write_action_count",
    ]
    return {
        field: sum(
            1
            for record in records
            if field not in record["metadata"] or record["metadata"][field] is None
        )
        for field in expected
    }


def model_facing_leakage_fields(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    leakage = {}
    for source_dataset in sorted({record["source_dataset"] for record in records}):
        source_records = [
            record for record in records if record["source_dataset"] == source_dataset
        ]
        keys: set[str] = set()
        for record in source_records:
            for field in ("x_raw", "s_raw", "x_numeric_features", "s_numeric_features"):
                value = record[field]
                if isinstance(value, dict):
                    keys.update(value)
        leaked = sorted(API_BANK_METADATA_ONLY_LEAKAGE_FIELDS & keys)
        if leaked:
            leakage[source_dataset] = leaked
    return leakage


def build_manifest(
    tau2_records: list[dict[str, Any]],
    api_bank_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = tau2_records + api_bank_records
    records_by_source = {
        TAU2_SOURCE_DATASET: tau2_records,
        API_BANK_SOURCE_DATASET: api_bank_records,
    }
    label_distribution_by_source = {
        source: label_distribution(source_records)
        for source, source_records in records_by_source.items()
    }
    return {
        "record_count_by_source_dataset": {
            source: len(source_records)
            for source, source_records in records_by_source.items()
        },
        "label_distribution_by_source_dataset": label_distribution_by_source,
        "label_scope_distribution": distribution(records, "label_scope"),
        "synthetic_vs_non_synthetic_counts": distribution(records, "is_synthetic"),
        "synthetic_vs_non_synthetic_counts_by_source_dataset": {
            source: distribution(source_records, "is_synthetic")
            for source, source_records in records_by_source.items()
        },
        "domain_distribution": distribution(records, "domain"),
        "domain_distribution_by_source_dataset": {
            source: distribution(source_records, "domain")
            for source, source_records in records_by_source.items()
        },
        "x_numeric_feature_names_by_source_dataset": {
            source: feature_names(source_records, "x_numeric_features")
            for source, source_records in records_by_source.items()
        },
        "s_numeric_feature_names_and_dimensions_by_source_dataset": {
            source: s_numeric_summary(source_records)
            for source, source_records in records_by_source.items()
        },
        "missing_field_counts": {
            "overall": missing_field_counts(records),
            TAU2_SOURCE_DATASET: missing_field_counts(tau2_records),
            API_BANK_SOURCE_DATASET: missing_field_counts(api_bank_records),
        },
        "tau2_metadata_missing_counts": tau2_missing_metadata_counts(tau2_records),
        "duplicate_sample_ids": duplicate_sample_ids(records),
        "model_facing_leakage_fields": model_facing_leakage_fields(records),
        "compatibility_warnings": COMPATIBILITY_WARNINGS,
    }


def validate_schema(records: list[dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        missing = set(SCHEMA_FIELDS) - set(record)
        extra = set(record) - set(SCHEMA_FIELDS)
        if missing or extra:
            raise ValueError(f"Record {idx} schema mismatch: missing={missing}, extra={extra}")
        if record["source_dataset"] not in {TAU2_SOURCE_DATASET, API_BANK_SOURCE_DATASET}:
            raise ValueError(f"Record {idx} has invalid source_dataset")
        if record["label_scope"] not in {TAU2_LABEL_SCOPE, API_BANK_LABEL_SCOPE}:
            raise ValueError(f"Record {idx} has invalid label_scope")
        if not isinstance(record["sample_id"], str) or not record["sample_id"]:
            raise ValueError(f"Record {idx} has invalid sample_id")
        if not isinstance(record["is_synthetic"], bool):
            raise ValueError(f"Record {idx} has invalid is_synthetic")
        if not isinstance(record["x_raw"], dict):
            raise ValueError(f"Record {idx} has invalid x_raw")
        if not isinstance(record["s_raw"], (dict, list)):
            raise ValueError(f"Record {idx} has invalid s_raw")
        if not isinstance(record["x_numeric_features"], dict):
            raise ValueError(f"Record {idx} has invalid x_numeric_features")
        if not isinstance(record["s_numeric_features"], dict):
            raise ValueError(f"Record {idx} has invalid s_numeric_features")
        if record["y"] not in {0, 1}:
            raise ValueError(f"Record {idx} has non-binary y")
        if not isinstance(record["metadata"], dict):
            raise ValueError(f"Record {idx} has invalid metadata")


def build_outputs(
    tau2_input: Path = DEFAULT_TAU2_INPUT,
    api_bank_input: Path = DEFAULT_API_BANK_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tau2_records = build_tau2_records(tau2_input)
    api_bank_records = build_api_bank_records(api_bank_input)
    if len(tau2_records) != TAU2_EXPECTED_COUNT:
        raise ValueError(f"Expected {TAU2_EXPECTED_COUNT} tau2 records, got {len(tau2_records)}")
    if len(api_bank_records) != API_BANK_EXPECTED_COUNT:
        raise ValueError(
            f"Expected {API_BANK_EXPECTED_COUNT} API-Bank records, got {len(api_bank_records)}"
        )

    validate_schema(tau2_records)
    validate_schema(api_bank_records)
    manifest = build_manifest(tau2_records, api_bank_records)

    write_jsonl(tau2_records, output_dir / TAU2_OUTPUT_NAME)
    write_jsonl(api_bank_records, output_dir / API_BANK_OUTPUT_NAME)
    write_json(manifest, output_dir / MANIFEST_OUTPUT_NAME)
    return tau2_records, api_bank_records, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-input", type=Path, default=DEFAULT_TAU2_INPUT)
    parser.add_argument("--api-bank-input", type=Path, default=DEFAULT_API_BANK_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau2_records, api_bank_records, manifest = build_outputs(
        tau2_input=args.tau2_input,
        api_bank_input=args.api_bank_input,
        output_dir=args.output_dir,
    )
    print(f"wrote {len(tau2_records)} tau2 records")
    print(f"wrote {len(api_bank_records)} API-Bank records")
    print(f"manifest warnings: {len(manifest['compatibility_warnings'])}")


if __name__ == "__main__":
    main()
