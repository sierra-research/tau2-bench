#!/usr/bin/env python3
"""Convert BFCL correctness records to Minxing L2T-compatible pickle format."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from l2t_model_bridge import (
    CONVERTER_VERSION,
    DEFAULT_SPLIT_SEED,
    REPO_ROOT,
    l2t_contract_summary,
    read_jsonl,
    save_pickle,
    split_summary,
    validate_l2t_payload,
    write_json,
)

DEFAULT_INPUT = REPO_ROOT / "data/processed/bfcl/bfcl_v4_non_live_1240_xy.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/l2t/bfcl"
DEFAULT_OUTPUT_PKL = DEFAULT_OUTPUT_DIR / "bfcl_v4_non_live_1240_l2t.pkl"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "bfcl_v4_non_live_1240_l2t_manifest.json"

TRAJ_LENGTH = 32
CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]
X_FEATURE_NAMES = [
    *(f"category_{category}" for category in CATEGORIES),
    "question_message_count",
    "question_text_length_log1p",
    "function_count",
    "function_name_chars_log1p",
    "function_description_chars_log1p",
    "parameter_count",
    "required_parameter_count",
    "max_parameters_per_function",
    "max_required_parameters_per_function",
    "has_parallel_category",
    "has_multiple_category",
    "has_irrelevance_category",
]
S_EVENT_VOCAB = {
    0: "padding",
    1: "candidate_function_call",
    2: "arguments_json_parse_success",
    3: "arguments_json_parse_error",
    4: "scalar_argument",
    5: "list_or_object_argument",
    6: "null_argument",
    7: "end",
}


def _log1p_len(value: Any) -> float:
    return float(math.log1p(len(str(value or ""))))


def _flatten_question_messages(question: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not isinstance(question, list):
        return messages
    for item in question:
        if isinstance(item, list):
            messages.extend(message for message in item if isinstance(message, dict))
        elif isinstance(item, dict):
            messages.append(item)
    return messages


def _function_specs(record: dict[str, Any]) -> list[dict[str, Any]]:
    functions = record.get("x_raw", {}).get("function") or []
    return [function for function in functions if isinstance(function, dict)]


def build_x_row(record: dict[str, Any]) -> list[float]:
    category = str(record.get("category") or "")
    messages = _flatten_question_messages(record.get("x_raw", {}).get("question"))
    question_text = "\n".join(str(message.get("content") or "") for message in messages)
    functions = _function_specs(record)

    parameter_counts = []
    required_counts = []
    name_chars = 0
    description_chars = 0
    for function in functions:
        name_chars += len(str(function.get("name") or ""))
        description_chars += len(str(function.get("description") or ""))
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = parameters.get("required") or []
        parameter_counts.append(len(properties) if isinstance(properties, dict) else 0)
        required_counts.append(len(required) if isinstance(required, list) else 0)

    parameter_count = sum(parameter_counts)
    required_parameter_count = sum(required_counts)
    return [
        *(1.0 if category == known else 0.0 for known in CATEGORIES),
        float(len(messages)),
        _log1p_len(question_text),
        float(len(functions)),
        float(math.log1p(name_chars)),
        float(math.log1p(description_chars)),
        float(parameter_count),
        float(required_parameter_count),
        float(max(parameter_counts, default=0)),
        float(max(required_counts, default=0)),
        float("parallel" in category),
        float("multiple" in category),
        float(category == "irrelevance"),
    ]


def _parse_arguments(raw_value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw_value, dict):
        return raw_value, True
    if not isinstance(raw_value, str):
        return {}, False
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}, False
    return parsed if isinstance(parsed, dict) else {}, isinstance(parsed, dict)


def encode_s_sequence(record: dict[str, Any]) -> np.ndarray:
    events: list[int] = []
    model_result = record.get("s_raw", {}).get("model_result") or []
    if isinstance(model_result, dict):
        result_items = [model_result]
    elif isinstance(model_result, list):
        result_items = [item for item in model_result if isinstance(item, dict)]
    else:
        result_items = []

    for call in result_items:
        for _, raw_arguments in sorted(call.items(), key=lambda item: str(item[0])):
            events.append(1)
            arguments, parsed_ok = _parse_arguments(raw_arguments)
            events.append(2 if parsed_ok else 3)
            for _, value in sorted(arguments.items(), key=lambda item: str(item[0])):
                if value is None:
                    events.append(6)
                elif isinstance(value, (dict, list)):
                    events.append(5)
                else:
                    events.append(4)

    events.append(7)
    if len(events) >= TRAJ_LENGTH:
        events = events[:TRAJ_LENGTH]
    else:
        events.extend([0] * (TRAJ_LENGTH - len(events)))
    return np.asarray(events, dtype=np.float32)


def build_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [str(record["id"]) for record in records]
    metadata = [
        {
            "sample_id": str(record["id"]),
            "source_dataset": record.get("source_dataset"),
            "category": record.get("category"),
            "model": record.get("model"),
            "label_scope": record.get("label_scope"),
            "label_origin": record.get("label_origin"),
        }
        for record in records
    ]
    payload = {
        "X": np.asarray([build_x_row(record) for record in records], dtype=np.float32),
        "y": np.asarray([int(record["y"]) for record in records], dtype=np.int64),
        "traj": {
            "s": np.vstack([encode_s_sequence(record) for record in records]).astype(
                np.float32
            )
        },
        "metadata": metadata,
        "sample_ids": sample_ids,
        "feature_names": X_FEATURE_NAMES,
        "s_event_vocabulary": S_EVENT_VOCAB,
    }
    validate_l2t_payload(payload, sample_ids=sample_ids)
    return payload


def build_manifest(
    *,
    input_path: Path,
    output_path: Path,
    payload: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    records_metadata = payload["metadata"]
    source_counts = Counter(str(row["source_dataset"]) for row in records_metadata)
    category_counts = Counter(str(row["category"]) for row in records_metadata)
    label_origins = Counter(str(row["label_origin"]) for row in records_metadata)
    validation = validate_l2t_payload(payload, sample_ids=payload["sample_ids"])
    return {
        "dataset": "bfcl_v4_non_live",
        "source_input": str(input_path),
        "output_pickle": str(output_path),
        "converter_script": "scripts/convert_bfcl_to_l2t_pkl.py",
        "converter_version": CONVERTER_VERSION,
        "sample_count": int(payload["y"].shape[0]),
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "feature_definitions": {
            "X": {
                "mapping": "BFCL prompt/tool schema structural features only; no evaluator result or label fields.",
                "ordered_feature_names": X_FEATURE_NAMES,
            },
            "traj.s": {
                "mapping": "Fixed-length candidate tool-call event sequence from s_raw.model_result.",
                "trajectory_length": TRAJ_LENGTH,
                "event_vocabulary": S_EVENT_VOCAB,
            },
        },
        "array_contract": validation,
        "split": split_summary(payload["y"], seed=split_seed),
        "label_scope": "test_case_level",
        "label_origin": dict(sorted(label_origins.items())),
        "label_semantics": "Y=1 means BFCL evaluator marked the test case correct; Y=0 means incorrect.",
        "model_facing_exclusions": [
            "metadata.evaluation_error_type",
            "metadata.evaluation_error",
            "label_scope",
            "label_origin",
            "is_synthetic",
            "y",
        ],
        "minxing_contract": l2t_contract_summary(),
    }


def convert(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT_PKL,
    manifest_path: Path = DEFAULT_MANIFEST,
    split_seed: int = DEFAULT_SPLIT_SEED,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = read_jsonl(input_path)
    payload = build_payload(records)
    manifest = build_manifest(
        input_path=input_path,
        output_path=output_path,
        payload=payload,
        split_seed=split_seed,
    )
    if write_outputs:
        save_pickle(payload, output_path)
        write_json(manifest, manifest_path)
    return payload, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, manifest = convert(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        split_seed=args.split_seed,
    )
    print(f"output path: {args.output}")
    print(f"manifest path: {args.manifest}")
    print(f"X shape: {payload['X'].shape}")
    print(f"y shape: {payload['y'].shape}")
    print(f"traj['s'] shape: {payload['traj']['s'].shape}")
    print(f"class distribution: {manifest['array_contract']['class_distribution']}")


if __name__ == "__main__":
    main()
