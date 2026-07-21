#!/usr/bin/env python3
"""Convert API-Bank numerical artifacts to Minxing L2T-compatible pickle format."""

from __future__ import annotations

import argparse
import json
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

DEFAULT_NUMERICAL_NPZ = REPO_ROOT / "data/processed/toolcalling_numerical_full.npz"
DEFAULT_NUMERICAL_MANIFEST = (
    REPO_ROOT / "data/processed/toolcalling_numerical_full_manifest.json"
)
DEFAULT_UNIFIED_JSONL = REPO_ROOT / "data/processed/unified_toolcalling_apibank.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/l2t/apibank"
DEFAULT_OUTPUT_PKL = DEFAULT_OUTPUT_DIR / "apibank_full_l2t.pkl"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "apibank_full_l2t_manifest.json"
API_BANK_SOURCE = "api_bank"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(record["sample_id"]): record for record in rows}


def build_payload(
    *,
    numerical_npz: Path = DEFAULT_NUMERICAL_NPZ,
    unified_jsonl: Path = DEFAULT_UNIFIED_JSONL,
) -> dict[str, Any]:
    metadata_lookup = metadata_by_sample_id(unified_jsonl)
    with np.load(numerical_npz, allow_pickle=True) as arrays:
        source_dataset = arrays["source_dataset"].astype(str)
        mask = source_dataset == API_BANK_SOURCE
        sample_ids = [str(value) for value in arrays["sample_ids"][mask].tolist()]
        metadata = []
        group_ids = []
        for sample_id in sample_ids:
            record = metadata_lookup[sample_id]
            pair_id = str(record.get("metadata", {}).get("pair_id"))
            group_ids.append(pair_id)
            metadata.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": record.get("source_dataset"),
                    "label_scope": record.get("label_scope"),
                    "label_origin": record.get("label_origin"),
                    "is_synthetic": bool(record.get("is_synthetic")),
                    "pair_id": pair_id,
                    "source_input_path": record.get("metadata", {}).get(
                        "source_input_path"
                    ),
                }
            )
        payload = {
            "X": np.asarray(arrays["X"][mask], dtype=np.float32),
            "y": np.asarray(arrays["y"][mask], dtype=np.int64),
            "traj": {"s": np.asarray(arrays["S"][mask], dtype=np.float32)},
            "metadata": metadata,
            "sample_ids": sample_ids,
            "group_ids": group_ids,
        }
    validate_l2t_payload(payload, sample_ids=sample_ids)
    return payload


def build_manifest(
    *,
    numerical_npz: Path,
    numerical_manifest: Path,
    unified_jsonl: Path,
    output_path: Path,
    payload: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    source_manifest = load_json(numerical_manifest)
    validation = validate_l2t_payload(payload, sample_ids=payload["sample_ids"])
    metadata = payload["metadata"]
    label_origins = Counter(str(row["label_origin"]) for row in metadata)
    synthetic_counts = Counter(str(row["is_synthetic"]) for row in metadata)
    pair_counts = Counter(str(row["pair_id"]) for row in metadata)
    pair_size_counts = Counter(str(count) for count in pair_counts.values())
    return {
        "dataset": "api_bank",
        "source_input": {
            "numerical_npz": str(numerical_npz),
            "numerical_manifest": str(numerical_manifest),
            "unified_jsonl": str(unified_jsonl),
        },
        "output_pickle": str(output_path),
        "converter_script": "scripts/convert_apibank_to_l2t_pkl.py",
        "converter_version": CONVERTER_VERSION,
        "sample_count": int(payload["y"].shape[0]),
        "feature_definitions": {
            "X": {
                "mapping": "Existing toolcalling_numerical_full API-Bank X rows: normalized x_text embedding concatenated with structural pre-call context features.",
                "shape_source": "toolcalling_numerical_full.npz['X'][source_dataset == 'api_bank']",
                "ordered_structural_feature_names": source_manifest[
                    "ordered_x_structural_feature_names"
                ],
                "embedding_model_name": source_manifest.get("embedding_model_name"),
            },
            "traj.s": {
                "mapping": "Existing toolcalling_numerical_full API-Bank S rows used as the fixed sequence consumed by Minxing make_sequence_pairs.",
                "shape_source": "toolcalling_numerical_full.npz['S'][source_dataset == 'api_bank']",
                "ordered_structural_feature_names": source_manifest[
                    "ordered_s_structural_feature_names"
                ],
                "embedding_model_name": source_manifest.get("embedding_model_name"),
                "compatibility_note": "Minxing's loader has no separate fixed S-vector input; each S row is supplied as traj['s'] so the existing loader can form one-step sequence pairs.",
            },
        },
        "array_contract": validation,
        "split": split_summary(payload["y"], seed=split_seed),
        "label_scope": "api_call_level",
        "label_origin": dict(sorted(label_origins.items())),
        "synthetic_counts": dict(sorted(synthetic_counts.items())),
        "label_semantics": "Y=1 means correct API call under the current API-Bank pilot definition; Y=0 means synthetic corrupted API call.",
        "pair_grouping": {
            "group_id_payload_key": "group_ids",
            "metadata_key": "metadata[].pair_id",
            "pair_count": int(len(pair_counts)),
            "pair_size_distribution": dict(sorted(pair_size_counts.items())),
            "supervised_diagnostic_note": "Positive and synthetic-negative rows are paired; supervised diagnostics must split by pair_id to avoid train/validation leakage.",
        },
        "model_facing_exclusions": [
            "label_scope",
            "label_origin",
            "is_synthetic",
            "pair_id",
            "group_ids",
            "corruption_type",
            "validation_status",
            "validation_error",
            "y",
        ],
        "source_numerical_manifest_summary": {
            "final_x_dimension": source_manifest["final_x_dimension"],
            "final_s_dimension": source_manifest["final_s_dimension"],
            "label_counts_by_dataset": source_manifest["label_counts_by_dataset"],
            "leakage_audit": source_manifest["leakage_audit"],
        },
        "minxing_contract": l2t_contract_summary(),
    }


def convert(
    *,
    numerical_npz: Path = DEFAULT_NUMERICAL_NPZ,
    numerical_manifest: Path = DEFAULT_NUMERICAL_MANIFEST,
    unified_jsonl: Path = DEFAULT_UNIFIED_JSONL,
    output_path: Path = DEFAULT_OUTPUT_PKL,
    manifest_path: Path = DEFAULT_MANIFEST,
    split_seed: int = DEFAULT_SPLIT_SEED,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = build_payload(numerical_npz=numerical_npz, unified_jsonl=unified_jsonl)
    manifest = build_manifest(
        numerical_npz=numerical_npz,
        numerical_manifest=numerical_manifest,
        unified_jsonl=unified_jsonl,
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
    parser.add_argument("--numerical-npz", type=Path, default=DEFAULT_NUMERICAL_NPZ)
    parser.add_argument(
        "--numerical-manifest", type=Path, default=DEFAULT_NUMERICAL_MANIFEST
    )
    parser.add_argument("--unified-jsonl", type=Path, default=DEFAULT_UNIFIED_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, manifest = convert(
        numerical_npz=args.numerical_npz,
        numerical_manifest=args.numerical_manifest,
        unified_jsonl=args.unified_jsonl,
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
