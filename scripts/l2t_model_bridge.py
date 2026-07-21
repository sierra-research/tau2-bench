#!/usr/bin/env python3
"""Shared helpers for Minxing L2T-compatible tool-calling datasets."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_SEED = 1
CONVERTER_VERSION = "l2t_model_bridge_20260720"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def class_distribution(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y.astype(np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def split_summary(y: np.ndarray, *, seed: int = DEFAULT_SPLIT_SEED) -> dict[str, Any]:
    n = int(y.shape[0])
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(0.8 * n)
    train_y = y[perm[:n_train]]
    val_y = y[perm[n_train:]]
    return {
        "seed": int(seed),
        "splitter": "numpy.random.RandomState(seed).permutation with first int(0.8*N) samples used for training",
        "train_size": int(train_y.shape[0]),
        "validation_size": int(val_y.shape[0]),
        "train_class_distribution": class_distribution(train_y),
        "validation_class_distribution": class_distribution(val_y),
    }


def validate_l2t_payload(
    payload: dict[str, Any],
    *,
    sample_ids: list[str],
) -> dict[str, Any]:
    missing = [key for key in ("X", "y", "traj") if key not in payload]
    if missing:
        raise KeyError(f"L2T payload missing required keys: {missing}")
    if not isinstance(payload["traj"], dict) or "s" not in payload["traj"]:
        raise KeyError("L2T payload must contain traj['s']")

    x = np.asarray(payload["X"])
    y = np.asarray(payload["y"])
    s = np.asarray(payload["traj"]["s"])
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if s.ndim != 2:
        raise ValueError(f"traj['s'] must be 2D, got {s.shape}")
    if not (x.shape[0] == y.shape[0] == s.shape[0] == len(sample_ids)):
        raise ValueError(
            "X, y, traj['s'], and sample_ids must have the same first dimension. "
            f"Got X={x.shape}, y={y.shape}, s={s.shape}, ids={len(sample_ids)}"
        )
    if set(np.unique(y).astype(int).tolist()) - {0, 1}:
        raise ValueError(f"y must contain only binary 0/1 labels, got {np.unique(y)}")
    duplicate_ids = duplicate_values(sample_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate sample IDs found: {duplicate_ids[:5]}")
    if not np.isfinite(x).all():
        raise ValueError("X contains NaN or infinite values")
    if not np.isfinite(s).all():
        raise ValueError("traj['s'] contains NaN or infinite values")

    return {
        "X_shape": list(x.shape),
        "y_shape": list(y.shape),
        "traj_s_shape": list(s.shape),
        "X_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "traj_s_dtype": str(s.dtype),
        "class_distribution": class_distribution(y),
        "duplicate_sample_ids": [],
        "nan_counts": {"X": int(np.isnan(x).sum()), "traj_s": int(np.isnan(s).sum())},
        "infinite_value_counts": {
            "X": int(np.isinf(x).sum()),
            "traj_s": int(np.isinf(s).sum()),
        },
    }


def save_pickle(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def l2t_contract_summary() -> dict[str, Any]:
    return {
        "required_pickle_keys": ["X", "y", "traj"],
        "required_traj_keys": ["s"],
        "X": "float32 array with shape (N, d_x); consumed as context/features",
        "y": "binary array with shape (N,); consumed as positive/negative label for safety-style classification metrics",
        "traj.s": "float32 array with shape (N, T); converted by Minxing loader into one-step sequence pairs",
        "loader_behavior": "run_baseline.py casts X, y, and traj['s'] to float32 and builds o_hist/o_next via make_sequence_pairs",
    }
