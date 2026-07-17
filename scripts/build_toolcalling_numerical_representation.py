#!/usr/bin/env python3
"""Build numerical representations for tool-calling shift analysis."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAU2_INPUT = REPO_ROOT / "data/processed/unified_toolcalling_tau2.jsonl"
DEFAULT_API_BANK_INPUT = REPO_ROOT / "data/processed/unified_toolcalling_apibank.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed"
DEFAULT_NPZ_NAME = "toolcalling_numerical_pilot.npz"
DEFAULT_JSONL_NAME = "toolcalling_numerical_pilot.jsonl"
DEFAULT_MANIFEST_NAME = "toolcalling_numerical_pilot_manifest.json"
FULL_NPZ_NAME = "toolcalling_numerical_full.npz"
FULL_JSONL_NAME = "toolcalling_numerical_full.jsonl"
FULL_MANIFEST_NAME = "toolcalling_numerical_full_manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TAU2_SOURCE_DATASET = "tau2"
API_BANK_SOURCE_DATASET = "api_bank"
TAU2_LABEL_SCOPE = "task_level"
API_BANK_LABEL_SCOPE = "api_call_level"

TAU2_EVENT_NAMES = {
    0: "padding",
    1: "user_message",
    2: "assistant_text",
    3: "read_tool_call",
    4: "write_tool_call",
    5: "successful_tool_response",
    6: "tool_error",
    7: "end",
}

BASE_X_STRUCTURAL_FEATURES = [
    "context_text_length",
    "history_turn_count",
    "user_turn_count",
    "assistant_turn_count",
    "available_tool_count",
    "target_or_expected_tool_call_count",
    "requires_authentication",
    "requires_write",
    "domain_id",
]
BASE_S_STRUCTURAL_FEATURES = [
    "trajectory_length",
    "tool_call_count",
    "unique_tool_count",
    "read_call_count",
    "write_call_count",
    "tool_error_count",
    "retry_count",
    "argument_count",
    "has_exception",
    "termination_success_signal",
]
X_STRUCTURAL_FEATURE_NAMES = BASE_X_STRUCTURAL_FEATURES + [
    f"{name}__missing" for name in BASE_X_STRUCTURAL_FEATURES
]
S_STRUCTURAL_FEATURE_NAMES = BASE_S_STRUCTURAL_FEATURES + [
    f"{name}__missing" for name in BASE_S_STRUCTURAL_FEATURES
]

LEAKAGE_FIELD_NAMES = {
    "corruption_type",
    "is_synthetic",
    "label_origin",
    "validation_error",
    "validation_status",
    "variant",
    "y",
}
COMPATIBILITY_WARNINGS = [
    "tau2 uses task-level labels while API-Bank uses API-call-level labels.",
    "tau2 labels are benchmark outcomes; API-Bank negative labels are synthetic corruptions.",
    "Synthetic API-Bank negatives are not naturally occurring LLM failures.",
    "tau2 trajectories are coarse structural event sequences when raw messages are unavailable.",
    "The pilot does not treat tau2 and API-Bank as IID samples from one task.",
    "This builder creates representations only and does not train or score a predictive model.",
]
LEAKAGE_AUDIT_DECISIONS = [
    "Excluded y, label_origin, is_synthetic, variant, corruption_type, validation_status, and validation_error from model-facing text and structural features.",
    "API-Bank X text and X structural features are derived from pre-call history and available API names only, so positive/negative members of a pair share X.",
    "API-Bank execution results are excluded from S text to avoid leaking evaluator outcomes.",
    "API-Bank has_exception and termination_success_signal are marked unavailable in structural S features because they could trivially encode synthetic-negative validation status in future builds.",
    "tau2 reward and db_match metadata are excluded from model-facing text and structural features.",
]


class TextEncoder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        ...


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
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
                    separators=(",", ":"),
                    default=json_default,
                )
            )
            handle.write("\n")


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def with_missing_indicators(
    base_values: dict[str, float | int | bool | None],
    base_order: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    missing: dict[str, float] = {}
    for name in base_order:
        value = safe_float(base_values.get(name))
        if value is None:
            values[name] = 0.0
            missing[f"{name}__missing"] = 1.0
        else:
            values[name] = float(value)
            missing[f"{name}__missing"] = 0.0
    return {**values, **missing}


def feature_vector(features: dict[str, float], ordered_names: list[str]) -> list[float]:
    return [float(features[name]) for name in ordered_names]


def chat_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    history = record.get("x_raw", {}).get("chat_history")
    return history if isinstance(history, list) else []


def available_api_names(record: dict[str, Any]) -> list[str]:
    names = record.get("x_raw", {}).get("available_api_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def role_name(role: Any) -> str:
    role_text = str(role or "unknown").strip().lower()
    if role_text == "ai":
        return "assistant"
    return role_text


def count_history_roles(history: list[dict[str, Any]]) -> tuple[int, int]:
    user_count = 0
    assistant_count = 0
    for turn in history:
        role = role_name(turn.get("role"))
        if role == "user":
            user_count += 1
        elif role == "assistant":
            assistant_count += 1
    return user_count, assistant_count


def api_name_is_write(api_name: str | None) -> bool | None:
    if not api_name:
        return None
    lowered = api_name.lower()
    write_prefixes = (
        "add",
        "book",
        "buy",
        "cancel",
        "create",
        "delete",
        "modify",
        "order",
        "pay",
        "post",
        "remove",
        "reserve",
        "schedule",
        "send",
        "set",
        "submit",
        "transfer",
        "update",
    )
    read_prefixes = ("check", "find", "get", "list", "query", "read", "retrieve", "search")
    if lowered.startswith(write_prefixes):
        return True
    if lowered.startswith(read_prefixes):
        return False
    return None


def record_domain_id(record: dict[str, Any]) -> int | None:
    domain = record.get("domain")
    if domain == "retail":
        return 1
    if domain == "airline":
        return 2
    if isinstance(domain, str) and domain:
        return 99
    return None


def tau2_event_sequence(record: dict[str, Any]) -> list[int]:
    if isinstance(record.get("s_raw"), list):
        return [int(value) for value in record["s_raw"]]
    return []


def non_padding_tau2_events(record: dict[str, Any]) -> list[int]:
    events = []
    for event in tau2_event_sequence(record):
        if event == 0:
            break
        events.append(event)
        if event == 7:
            break
    return events


def serialize_x_text(record: dict[str, Any]) -> str:
    source = record["source_dataset"]
    if source == TAU2_SOURCE_DATASET:
        metadata = record.get("metadata", {})
        x_features = record.get("x_numeric_features", {})
        parts = [
            f"Dataset: tau2",
            f"Domain: {record.get('domain') or 'unknown'}",
            f"Task id: {metadata.get('task_id', 'unknown')}",
            "Task description: unavailable in unified pilot record",
            (
                "Expected action requirements: "
                f"total={x_features.get('expected_action_count', 'unknown')}; "
                f"read={x_features.get('expected_read_action_count', 'unknown')}; "
                f"write={x_features.get('expected_write_action_count', 'unknown')}"
            ),
            (
                "Structured task requirements: "
                f"requires_db_mutation={x_features.get('requires_db_mutation', 'unknown')}; "
                f"has_communication_checks={x_features.get('has_communication_checks', 'unknown')}; "
                f"has_env_assertions={x_features.get('has_env_assertions', 'unknown')}; "
                f"has_nl_assertions={x_features.get('has_nl_assertions', 'unknown')}"
            ),
            "Tool context: exact tool names and schemas unavailable in unified pilot record",
        ]
        return "\n".join(parts)

    if source == API_BANK_SOURCE_DATASET:
        history = chat_history(record)
        lines = ["Dataset: API-Bank", "Pre-call dialogue history:"]
        for index, turn in enumerate(history, start=1):
            role = role_name(turn.get("role"))
            if role == "api":
                api_name = turn.get("api_name", "unknown_api")
                argument_names = sorted((turn.get("param_dict") or {}).keys())
                lines.append(
                    f"{index}. api: {api_name}({', '.join(argument_names)})"
                )
            else:
                text = str(turn.get("text", ""))
                lines.append(f"{index}. {role}: {text}")
        api_names = available_api_names(record)
        if api_names:
            lines.append(f"Available APIs: {', '.join(api_names)}")
        else:
            lines.append("Available APIs: unavailable")
        return "\n".join(lines)

    raise ValueError(f"Unsupported source_dataset: {source}")


def serialize_s_text(record: dict[str, Any]) -> str:
    source = record["source_dataset"]
    if source == TAU2_SOURCE_DATASET:
        event_names = [
            TAU2_EVENT_NAMES.get(event, f"unknown_event_{event}")
            for event in non_padding_tau2_events(record)
        ]
        return "\n".join(
            [
                "Trajectory representation: coarse structural event sequence",
                "Warning: tool names, arguments, output content, and full message semantics are unavailable",
                "Events: " + " -> ".join(event_names),
            ]
        )

    if source == API_BANK_SOURCE_DATASET:
        s_raw = record.get("s_raw", {})
        api_name = s_raw.get("api_name", "unknown_api")
        params = s_raw.get("param_dict") or {}
        return "\n".join(
            [
                f"Candidate API: {api_name}",
                f"Candidate arguments: {stable_json(params)}",
                "Execution result characteristics: excluded from model-facing text",
            ]
        )

    raise ValueError(f"Unsupported source_dataset: {source}")


def build_x_structural_features(record: dict[str, Any], x_text: str) -> dict[str, float]:
    source = record["source_dataset"]
    x_features = record.get("x_numeric_features", {})
    if source == TAU2_SOURCE_DATASET:
        base_values = {
            "context_text_length": len(x_text),
            "history_turn_count": None,
            "user_turn_count": None,
            "assistant_turn_count": None,
            "available_tool_count": None,
            "target_or_expected_tool_call_count": x_features.get("expected_action_count"),
            "requires_authentication": None,
            "requires_write": 1
            if safe_float(x_features.get("expected_write_action_count")) not in (None, 0.0)
            else 0,
            "domain_id": record_domain_id(record),
        }
    elif source == API_BANK_SOURCE_DATASET:
        history = chat_history(record)
        user_count, assistant_count = count_history_roles(history)
        base_values = {
            "context_text_length": len(x_text),
            "history_turn_count": len(history),
            "user_turn_count": user_count,
            "assistant_turn_count": assistant_count,
            "available_tool_count": len(available_api_names(record)),
            "target_or_expected_tool_call_count": 1,
            "requires_authentication": 1
            if "token" in x_text.lower() or "authenticate" in x_text.lower()
            else 0,
            "requires_write": None,
            "domain_id": record_domain_id(record),
        }
    else:
        raise ValueError(f"Unsupported source_dataset: {source}")
    return with_missing_indicators(base_values, BASE_X_STRUCTURAL_FEATURES)


def count_tau2_retries(events: list[int]) -> int:
    retries = 0
    for previous, current in zip(events, events[1:], strict=False):
        if previous == 6 and current in {3, 4}:
            retries += 1
    return retries


def build_s_structural_features(record: dict[str, Any]) -> dict[str, float]:
    source = record["source_dataset"]
    if source == TAU2_SOURCE_DATASET:
        events = non_padding_tau2_events(record)
        read_count = events.count(3)
        write_count = events.count(4)
        tool_error_count = events.count(6)
        termination_reason = record.get("metadata", {}).get("termination_reason")
        base_values = {
            "trajectory_length": len(events),
            "tool_call_count": read_count + write_count,
            "unique_tool_count": None,
            "read_call_count": read_count,
            "write_call_count": write_count,
            "tool_error_count": tool_error_count,
            "retry_count": count_tau2_retries(events),
            "argument_count": None,
            "has_exception": 1 if tool_error_count else 0,
            "termination_success_signal": 1 if termination_reason == "user_stop" else 0,
        }
    elif source == API_BANK_SOURCE_DATASET:
        s_raw = record.get("s_raw", {})
        api_name = s_raw.get("api_name")
        is_write = api_name_is_write(api_name)
        params = s_raw.get("param_dict") or {}
        base_values = {
            "trajectory_length": 1,
            "tool_call_count": 1,
            "unique_tool_count": 1,
            "read_call_count": 0 if is_write else 1 if is_write is False else None,
            "write_call_count": 1 if is_write else 0 if is_write is False else None,
            "tool_error_count": None,
            "retry_count": None,
            "argument_count": len(params),
            "has_exception": None,
            "termination_success_signal": None,
        }
    else:
        raise ValueError(f"Unsupported source_dataset: {source}")
    return with_missing_indicators(base_values, BASE_S_STRUCTURAL_FEATURES)


def select_records(
    tau2_records: list[dict[str, Any]],
    api_bank_records: list[dict[str, Any]],
    *,
    tau2_limit: int,
    apibank_limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_tau2 = list(tau2_records[:tau2_limit])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in api_bank_records:
        pair_id = record.get("metadata", {}).get("pair_id") or record["sample_id"]
        grouped[str(pair_id)].append(record)

    rng = random.Random(seed)
    pair_ids = sorted(grouped)
    rng.shuffle(pair_ids)

    selected_api_bank: list[dict[str, Any]] = []
    selected_pair_ids: list[str] = []
    for pair_id in pair_ids:
        if len(selected_api_bank) >= apibank_limit:
            break
        members = sorted(grouped[pair_id], key=lambda item: item["sample_id"])
        remaining = apibank_limit - len(selected_api_bank)
        if remaining >= len(members):
            selected_api_bank.extend(members)
        elif remaining > 0:
            shuffled_members = list(members)
            rng.shuffle(shuffled_members)
            selected_api_bank.extend(sorted(shuffled_members[:remaining], key=lambda item: item["sample_id"]))
        selected_pair_ids.append(pair_id)

    selected_pair_counts = Counter(
        str(record.get("metadata", {}).get("pair_id") or record["sample_id"])
        for record in selected_api_bank
    )
    complete_pair_count = sum(1 for count in selected_pair_counts.values() if count == 2)
    incomplete_pair_count = sum(1 for count in selected_pair_counts.values() if count != 2)
    sampling_summary = {
        "tau2_limit": tau2_limit,
        "apibank_limit": apibank_limit,
        "seed": seed,
        "selected_api_bank_pair_count": len(selected_pair_counts),
        "selected_api_bank_complete_pair_count": complete_pair_count,
        "selected_api_bank_incomplete_pair_count": incomplete_pair_count,
        "selected_api_bank_all_pairs_complete": incomplete_pair_count == 0,
        "selected_api_bank_pair_integrity_note": (
            "API-Bank sampling preserves complete pairs until an odd limit requires one unpaired member."
        ),
        "selected_pair_ids_considered": selected_pair_ids,
    }
    return selected_tau2 + selected_api_bank, sampling_summary


def full_output_names(full_data: bool) -> tuple[str, str, str]:
    if full_data:
        return FULL_NPZ_NAME, FULL_JSONL_NAME, FULL_MANIFEST_NAME
    return DEFAULT_NPZ_NAME, DEFAULT_JSONL_NAME, DEFAULT_MANIFEST_NAME


def load_sentence_transformer_encoder() -> TextEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for real embedding generation. "
            "Install the repository dependencies with the configured optional dependency "
            "that includes sentence-transformers; do not substitute a different model."
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def encode_texts(
    texts: list[str],
    *,
    encoder: TextEncoder | None,
    batch_size: int,
) -> np.ndarray:
    encoder = encoder or load_sentence_transformer_encoder()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {array.shape}")
    return array


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def prepared_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    output_records = []
    x_struct_rows = []
    s_struct_rows = []
    for record in records:
        x_text = serialize_x_text(record)
        s_text = serialize_s_text(record)
        x_structural = build_x_structural_features(record, x_text)
        s_structural = build_s_structural_features(record)
        output_records.append(
            {
                "sample_id": record["sample_id"],
                "source_dataset": record["source_dataset"],
                "label_scope": record["label_scope"],
                "x_text": x_text,
                "s_text": s_text,
                "x_structural_features": x_structural,
                "s_structural_features": s_structural,
                "y": int(record["y"]),
                "metadata": {
                    "domain": record.get("domain"),
                    "is_synthetic": bool(record["is_synthetic"]),
                    "label_origin": record["label_origin"],
                    "source_sample_id": record["sample_id"],
                    "pair_id": record.get("metadata", {}).get("pair_id"),
                    "coarse_tau2_trajectory_serialization": record["source_dataset"]
                    == TAU2_SOURCE_DATASET,
                },
            }
        )
        x_struct_rows.append(feature_vector(x_structural, X_STRUCTURAL_FEATURE_NAMES))
        s_struct_rows.append(feature_vector(s_structural, S_STRUCTURAL_FEATURE_NAMES))
    return (
        output_records,
        np.asarray(x_struct_rows, dtype=np.float32),
        np.asarray(s_struct_rows, dtype=np.float32),
    )


def duplicate_sample_ids(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(record["sample_id"] for record in records)
    return sorted(sample_id for sample_id, count in counts.items() if count > 1)


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def distribution_by_dataset(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    result = {}
    for source in sorted({record["source_dataset"] for record in records}):
        source_records = [record for record in records if record["source_dataset"] == source]
        result[source] = distribution(source_records, key)
    return result


def feature_missing_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter()
    for record in records:
        features = record[field]
        for name, value in features.items():
            if name.endswith("__missing") and value == 1.0:
                counts[name.removesuffix("__missing")] += 1
    return dict(sorted(counts.items()))


def numeric_summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def norm_summary(array: np.ndarray) -> dict[str, float | None]:
    if array.size == 0:
        return {"min": None, "mean": None, "max": None}
    norms = np.linalg.norm(array, axis=1)
    return {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "max": float(np.max(norms)),
    }


def leakage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    field_name_hits: dict[str, list[str]] = {}
    text_fields = {field.lower() for field in LEAKAGE_FIELD_NAMES if field != "y"}
    for record in records:
        combined = "\n".join([record["x_text"], record["s_text"]]).lower()
        hits = sorted(
            field
            for field in text_fields
            if re.search(rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])", combined)
        )
        feature_hits = sorted(
            LEAKAGE_FIELD_NAMES
            & (
                set(record["x_structural_features"])
                | set(record["s_structural_features"])
            )
        )
        all_hits = sorted(set(hits) | set(feature_hits))
        if all_hits:
            field_name_hits[record["sample_id"]] = all_hits
    return {
        "excluded_metadata_only_fields": sorted(LEAKAGE_FIELD_NAMES),
        "model_facing_field_name_hits": field_name_hits,
        "decisions": LEAKAGE_AUDIT_DECISIONS,
    }


def build_manifest(
    jsonl_records: list[dict[str, Any]],
    X: np.ndarray,
    S: np.ndarray,
    x_embeddings: np.ndarray,
    s_embeddings: np.ndarray,
    x_structural: np.ndarray,
    s_structural: np.ndarray,
    sampling_summary: dict[str, Any],
) -> dict[str, Any]:
    source_counts = distribution(jsonl_records, "source_dataset")
    text_lengths = {
        "x_text": numeric_summary([len(record["x_text"]) for record in jsonl_records]),
        "s_text": numeric_summary([len(record["s_text"]) for record in jsonl_records]),
    }
    return {
        "total_count": len(jsonl_records),
        "count_by_dataset": source_counts,
        "label_counts_by_dataset": distribution_by_dataset(jsonl_records, "y"),
        "label_scope_counts": distribution(jsonl_records, "label_scope"),
        "synthetic_non_synthetic_counts": distribution(
            [
                {
                    **record,
                    "is_synthetic": record["metadata"]["is_synthetic"],
                }
                for record in jsonl_records
            ],
            "is_synthetic",
        ),
        "x_embedding_dimension": int(x_embeddings.shape[1]),
        "s_embedding_dimension": int(s_embeddings.shape[1]),
        "x_structural_dimension": int(x_structural.shape[1]),
        "s_structural_dimension": int(s_structural.shape[1]),
        "final_x_dimension": int(X.shape[1]),
        "final_s_dimension": int(S.shape[1]),
        "ordered_x_structural_feature_names": X_STRUCTURAL_FEATURE_NAMES,
        "ordered_s_structural_feature_names": S_STRUCTURAL_FEATURE_NAMES,
        "missing_feature_counts": {
            "x": feature_missing_counts(jsonl_records, "x_structural_features"),
            "s": feature_missing_counts(jsonl_records, "s_structural_features"),
        },
        "text_length_summaries": text_lengths,
        "duplicate_sample_ids": duplicate_sample_ids(jsonl_records),
        "nan_counts": {"X": int(np.isnan(X).sum()), "S": int(np.isnan(S).sum())},
        "infinite_value_counts": {
            "X": int(np.isinf(X).sum()),
            "S": int(np.isinf(S).sum()),
        },
        "embedding_norm_summaries": {
            "X_embedding": norm_summary(x_embeddings),
            "S_embedding": norm_summary(s_embeddings),
        },
        "coarse_tau2_trajectory_serialization_count": sum(
            1
            for record in jsonl_records
            if record["metadata"]["coarse_tau2_trajectory_serialization"]
        ),
        "selected_api_bank_complete_pair_count": sampling_summary[
            "selected_api_bank_complete_pair_count"
        ],
        "selected_api_bank_all_pairs_complete": sampling_summary[
            "selected_api_bank_all_pairs_complete"
        ],
        "sampling_summary": sampling_summary,
        "leakage_audit": leakage_audit(jsonl_records),
        "compatibility_warnings": COMPATIBILITY_WARNINGS,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_package_version": package_version("sentence-transformers"),
    }


def validate_outputs(
    jsonl_records: list[dict[str, Any]],
    X: np.ndarray,
    S: np.ndarray,
    y: np.ndarray,
) -> None:
    if len(jsonl_records) != len(X) or len(jsonl_records) != len(S) or len(jsonl_records) != len(y):
        raise ValueError("Record and array lengths are inconsistent")
    if duplicate_sample_ids(jsonl_records):
        raise ValueError("Duplicate sample IDs found")
    if set(y.tolist()) - {0, 1}:
        raise ValueError("Non-binary y values found")
    if np.isnan(X).any() or np.isnan(S).any():
        raise ValueError("NaN values found in numeric arrays")
    if np.isinf(X).any() or np.isinf(S).any():
        raise ValueError("Infinite values found in numeric arrays")
    leakage_hits = leakage_audit(jsonl_records)["model_facing_field_name_hits"]
    if leakage_hits:
        raise ValueError(f"Leakage field names found in model-facing values: {leakage_hits}")


def build_outputs(
    *,
    tau2_input: Path = DEFAULT_TAU2_INPUT,
    api_bank_input: Path = DEFAULT_API_BANK_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    npz_name: str = DEFAULT_NPZ_NAME,
    jsonl_name: str = DEFAULT_JSONL_NAME,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    tau2_limit: int = 93,
    apibank_limit: int = 107,
    full_data: bool = False,
    seed: int = 1,
    encoder: TextEncoder | None = None,
    batch_size: int = 32,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    tau2_records = load_jsonl(tau2_input)
    api_bank_records = load_jsonl(api_bank_input)
    if full_data:
        npz_name, jsonl_name, manifest_name = full_output_names(full_data)
        tau2_limit = len(tau2_records)
        apibank_limit = len(api_bank_records)
    selected_records, sampling_summary = select_records(
        tau2_records,
        api_bank_records,
        tau2_limit=tau2_limit,
        apibank_limit=apibank_limit,
        seed=seed,
    )
    jsonl_records, x_structural, s_structural = prepared_records(selected_records)

    x_embeddings = encode_texts(
        [record["x_text"] for record in jsonl_records],
        encoder=encoder,
        batch_size=batch_size,
    )
    s_embeddings = encode_texts(
        [record["s_text"] for record in jsonl_records],
        encoder=encoder,
        batch_size=batch_size,
    )
    x_embeddings = normalize_rows(x_embeddings).astype(np.float32)
    s_embeddings = normalize_rows(s_embeddings).astype(np.float32)

    X = np.concatenate([x_embeddings, x_structural], axis=1).astype(np.float32)
    S = np.concatenate([s_embeddings, s_structural], axis=1).astype(np.float32)
    y = np.asarray([record["y"] for record in jsonl_records], dtype=np.int64)
    sample_ids = np.asarray([record["sample_id"] for record in jsonl_records], dtype=object)
    source_dataset = np.asarray(
        [record["source_dataset"] for record in jsonl_records],
        dtype=object,
    )
    label_scope = np.asarray(
        [record["label_scope"] for record in jsonl_records],
        dtype=object,
    )
    is_synthetic = np.asarray(
        [record["metadata"]["is_synthetic"] for record in jsonl_records],
        dtype=bool,
    )

    arrays = {
        "X": X,
        "S": S,
        "y": y,
        "sample_ids": sample_ids,
        "source_dataset": source_dataset,
        "label_scope": label_scope,
        "is_synthetic": is_synthetic,
        "x_embeddings": x_embeddings,
        "s_embeddings": s_embeddings,
        "x_structural": x_structural,
        "s_structural": s_structural,
    }
    validate_outputs(jsonl_records, X, S, y)
    manifest = build_manifest(
        jsonl_records,
        X,
        S,
        x_embeddings,
        s_embeddings,
        x_structural,
        s_structural,
        sampling_summary,
    )

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / npz_name,
            X=X,
            S=S,
            y=y,
            sample_ids=sample_ids,
            source_dataset=source_dataset,
            label_scope=label_scope,
            is_synthetic=is_synthetic,
        )
        write_jsonl(jsonl_records, output_dir / jsonl_name)
        write_json(manifest, output_dir / manifest_name)

    return jsonl_records, manifest, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-input", type=Path, default=DEFAULT_TAU2_INPUT)
    parser.add_argument("--api-bank-input", type=Path, default=DEFAULT_API_BANK_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau2-limit", type=int, default=93)
    parser.add_argument("--apibank-limit", type=int, default=107)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help=(
            "Use all unified tau2 and API-Bank records and write "
            "toolcalling_numerical_full.* outputs."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, manifest, arrays = build_outputs(
        tau2_input=args.tau2_input,
        api_bank_input=args.api_bank_input,
        output_dir=args.output_dir,
        tau2_limit=args.tau2_limit,
        apibank_limit=args.apibank_limit,
        full_data=args.full_data,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"wrote {len(records)} records")
    print(f"X shape: {arrays['X'].shape}")
    print(f"S shape: {arrays['S'].shape}")
    print(
        "API-Bank complete pairs: "
        f"{manifest['selected_api_bank_complete_pair_count']} "
        f"(all complete: {manifest['selected_api_bank_all_pairs_complete']})"
    )


if __name__ == "__main__":
    main()
