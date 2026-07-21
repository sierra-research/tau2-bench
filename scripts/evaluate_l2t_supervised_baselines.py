#!/usr/bin/env python3
"""Evaluate supervised diagnostics for L2T bridge artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from l2t_model_bridge import (
    DEFAULT_SPLIT_SEED,
    REPO_ROOT,
    class_distribution,
    duplicate_values,
    json_default,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_BFCL_PKL = (
    REPO_ROOT / "data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl"
)
DEFAULT_APIBANK_PKL = REPO_ROOT / "data/processed/l2t/apibank/apibank_full_l2t.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/l2t/diagnostics"
DEFAULT_PERMUTATION_TRIALS = 10
LEAKAGE_FIELD_NAMES = {
    "corruption_type",
    "evaluation_error",
    "evaluation_error_type",
    "is_synthetic",
    "label_origin",
    "label_scope",
    "validation_error",
    "validation_status",
    "variant",
    "y",
}


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def minxing_split_indices(
    n_samples: int, seed: int = DEFAULT_SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.RandomState(seed).permutation(int(n_samples))
    n_train = int(0.8 * int(n_samples))
    return perm[:n_train], perm[n_train:]


def grouped_split_indices(
    group_ids: np.ndarray | list[str],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids).astype(str)
    if groups.ndim != 1:
        raise ValueError(f"group_ids must be 1D, got {groups.shape}")
    if groups.size == 0:
        raise ValueError("group_ids must not be empty")
    if np.any(groups == "") or np.any(groups == "None"):
        raise ValueError("all rows must have non-empty group_ids")

    unique_groups = np.array(sorted(np.unique(groups).tolist()), dtype=object)
    permuted_groups = unique_groups[
        np.random.RandomState(seed).permutation(unique_groups.shape[0])
    ]
    n_train_groups = int(train_fraction * int(unique_groups.shape[0]))
    train_groups = set(str(value) for value in permuted_groups[:n_train_groups])
    train_mask = np.asarray([str(value) in train_groups for value in groups])
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(~train_mask)
    return train_idx.astype(int), val_idx.astype(int)


def load_l2t_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}")
    missing = [key for key in ("X", "y", "traj") if key not in payload]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    if not isinstance(payload["traj"], dict) or "s" not in payload["traj"]:
        raise KeyError(f"{path} missing traj['s']")

    x = np.asarray(payload["X"], dtype=np.float32)
    s = np.asarray(payload["traj"]["s"], dtype=np.float32)
    y = np.asarray(payload["y"], dtype=np.int64).reshape(-1)
    sample_ids = payload.get("sample_ids")
    if sample_ids is None:
        sample_ids = [
            str(row.get("sample_id", index))
            for index, row in enumerate(payload.get("metadata", []))
        ]
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    if not sample_ids:
        sample_ids = [str(index) for index in range(int(y.shape[0]))]

    if x.ndim != 2 or s.ndim != 2 or y.ndim != 1:
        raise ValueError(f"Invalid array ranks in {path}: X={x.shape}, S={s.shape}, y={y.shape}")
    if not (x.shape[0] == s.shape[0] == y.shape[0] == len(sample_ids)):
        raise ValueError(
            f"Inconsistent first dimensions in {path}: X={x.shape}, S={s.shape}, "
            f"y={y.shape}, sample_ids={len(sample_ids)}"
        )
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(f"Non-binary labels in {path}: {np.unique(y).tolist()}")
    if not np.isfinite(x).all() or not np.isfinite(s).all():
        raise ValueError(f"NaN or infinite values in {path}")
    duplicate_ids = duplicate_values(sample_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicate_ids[:5]}")

    return {
        "path": str(path),
        "payload": payload,
        "keys": sorted(payload.keys()),
        "X": x,
        "S": s,
        "y": y,
        "sample_ids": sample_ids,
    }


def feature_views(artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    x = artifact["X"]
    s = artifact["S"]
    return {
        "X-only": x,
        "S-only": s,
        "X+S": np.concatenate([x, s], axis=1).astype(np.float32),
    }


def class_counts_dict(y: np.ndarray) -> dict[str, int]:
    return class_distribution(np.asarray(y, dtype=np.int64))


def extract_metadata_values(artifact: dict[str, Any], key: str) -> np.ndarray:
    values = []
    for row in artifact["payload"].get("metadata", []):
        if not isinstance(row, dict):
            values.append("")
        else:
            values.append(str(row.get(key, "")))
    if values and len(values) != int(artifact["y"].shape[0]):
        raise ValueError(f"metadata key {key!r} length does not match y")
    return np.asarray(values, dtype=object)


def extract_group_ids(artifact: dict[str, Any]) -> np.ndarray:
    payload = artifact["payload"]
    if "group_ids" in payload:
        group_ids = np.asarray(payload["group_ids"]).astype(str)
    else:
        group_ids = extract_metadata_values(artifact, "pair_id").astype(str)
    if group_ids.shape != artifact["y"].shape:
        raise ValueError(
            f"group_ids shape {group_ids.shape} does not match y {artifact['y'].shape}"
        )
    return group_ids


def split_group_crossings(
    group_ids: np.ndarray | list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    groups = np.asarray(group_ids).astype(str)
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    crossed = sorted(train_groups & val_groups)
    return {
        "cross_split_group_count": int(len(crossed)),
        "cross_split_group_ids_first10": crossed[:10],
        "train_group_count": int(len(train_groups)),
        "validation_group_count": int(len(val_groups)),
    }


def split_report(
    *,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    split_policy: str,
    seed: int,
    group_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    report = {
        "split_policy": split_policy,
        "seed": int(seed),
        "train_size": int(train_idx.shape[0]),
        "validation_size": int(val_idx.shape[0]),
        "train_class_counts": class_counts_dict(y[train_idx]),
        "validation_class_counts": class_counts_dict(y[val_idx]),
    }
    if group_ids is not None:
        report.update(split_group_crossings(group_ids, train_idx, val_idx))
    return report


def audit_feature_matrix(matrix: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix)
    labels = np.asarray(y).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {values.shape}")
    if values.shape[0] != labels.shape[0]:
        raise ValueError("matrix and y must have the same first dimension")

    y_float = labels.astype(float)
    inverse_y_float = (1 - labels).astype(float)
    exact_y: list[int] = []
    exact_inverse_y: list[int] = []
    constants: list[int] = []
    low_cardinality: list[dict[str, Any]] = []
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        if np.array_equal(column.astype(float), y_float):
            exact_y.append(int(column_index))
        if np.array_equal(column.astype(float), inverse_y_float):
            exact_inverse_y.append(int(column_index))
        unique_values = np.unique(column)
        if unique_values.shape[0] == 1:
            constants.append(int(column_index))
        elif unique_values.shape[0] <= 3:
            low_cardinality.append(
                {
                    "column_index": int(column_index),
                    "unique_value_count": int(unique_values.shape[0]),
                }
            )

    return {
        "n_columns": int(values.shape[1]),
        "exact_y_column_count": int(len(exact_y)),
        "exact_y_column_indices": exact_y,
        "exact_inverse_y_column_count": int(len(exact_inverse_y)),
        "exact_inverse_y_column_indices": exact_inverse_y,
        "constant_column_count": int(len(constants)),
        "constant_column_indices_first50": constants[:50],
        "low_cardinality_nonconstant_columns_first50": low_cardinality[:50],
    }


def leakage_audit(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = artifact["payload"]
    feature_names = [str(name).lower() for name in payload.get("feature_names", [])]
    event_vocab = [str(value).lower() for value in payload.get("s_event_vocabulary", {}).values()]
    field_hits = sorted(
        field
        for field in LEAKAGE_FIELD_NAMES
        if field in feature_names or field in event_vocab
    )
    x = artifact["X"]
    s = artifact["S"]
    y = artifact["y"]
    return {
        "model_facing_field_name_hits": field_hits,
        "numeric_array_note": "Per-column audit over numeric model-facing arrays; metadata fields are outside X and S.",
        "X": audit_feature_matrix(x, y),
        "S": audit_feature_matrix(s, y),
    }


def artifact_verification(
    name: str,
    artifact: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    y = artifact["y"]
    train_idx, val_idx = minxing_split_indices(len(y), seed=seed)
    result = {
        "dataset": name,
        "path": artifact["path"],
        "keys": artifact["keys"],
        "X_shape": list(artifact["X"].shape),
        "S_shape": list(artifact["S"].shape),
        "y_shape": list(y.shape),
        "X_dtype": str(artifact["X"].dtype),
        "S_dtype": str(artifact["S"].dtype),
        "y_dtype": str(y.dtype),
        "class_counts": class_counts_dict(y),
        "duplicate_id_count": 0,
        "nan_counts": {
            "X": int(np.isnan(artifact["X"]).sum()),
            "S": int(np.isnan(artifact["S"]).sum()),
        },
        "infinite_value_counts": {
            "X": int(np.isinf(artifact["X"]).sum()),
            "S": int(np.isinf(artifact["S"]).sum()),
        },
        "split_seed": int(seed),
        "train_indices_first10": train_idx[:10].astype(int).tolist(),
        "validation_indices_first10": val_idx[:10].astype(int).tolist(),
        "minxing_row_split": split_report(
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            split_policy="minxing_row_random",
            seed=seed,
        ),
        "leakage_audit": leakage_audit(artifact),
    }
    if name == "apibank":
        group_ids = extract_group_ids(artifact)
        grouped_train_idx, grouped_val_idx = grouped_split_indices(
            group_ids, seed=seed
        )
        legacy_crossing = split_group_crossings(group_ids, train_idx, val_idx)
        grouped_report = split_report(
            y=y,
            train_idx=grouped_train_idx,
            val_idx=grouped_val_idx,
            split_policy="pair_grouped",
            seed=seed,
            group_ids=group_ids,
        )
        if grouped_report["cross_split_group_count"] != 0:
            raise AssertionError("API-Bank grouped split leaked pair IDs")
        result["pair_grouping"] = {
            "group_key": "pair_id",
            "group_count": int(len(np.unique(group_ids))),
            "legacy_minxing_row_split_cross_split_pair_count": int(
                legacy_crossing["cross_split_group_count"]
            ),
            "grouped_split": grouped_report,
        }
    if name == "bfcl":
        categories = extract_metadata_values(artifact, "category")
        result["category_counts"] = dict(
            sorted(Counter(str(category) for category in categories).items())
        )
        category_summary = {}
        for category in sorted(np.unique(categories).tolist()):
            mask = categories == category
            category_summary[str(category)] = {
                "sample_count": int(np.sum(mask)),
                "class_counts": class_counts_dict(y[mask]),
            }
        result["category_summary"] = category_summary
    return result


def metric_dict(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, Any]:
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "predicted_class_counts": class_counts_dict(y_pred),
        "class_0_precision": float(precision[0]),
        "class_0_recall": float(recall[0]),
        "class_0_f1": float(f1[0]),
        "class_0_support": int(support[0]),
        "class_1_precision": float(precision[1]),
        "class_1_recall": float(recall[1]),
        "class_1_f1": float(f1[1]),
        "class_1_support": int(support[1]),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        score = np.asarray(y_score, dtype=float)
        result["roc_auc"] = float(roc_auc_score(y_true, score))
        result["pr_auc_y1"] = float(average_precision_score(y_true, score))
        result["pr_auc_y0"] = float(average_precision_score(1 - y_true, 1.0 - score))
    else:
        result["roc_auc"] = None
        result["pr_auc_y1"] = None
        result["pr_auc_y0"] = None
    return result


def predict_scores(model: Any, x_val: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_val)
        classes = list(model.classes_)
        if 1 in classes:
            return np.asarray(proba[:, classes.index(1)], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x_val), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
    return np.asarray(y_pred, dtype=float)


def make_model_specs(seed: int) -> list[tuple[str, Any]]:
    return [
        ("majority_class", DummyClassifier(strategy="most_frequent")),
        ("stratified_random", DummyClassifier(strategy="stratified", random_state=seed)),
        (
            "logistic_regression_standardized",
            Pipeline(
                [
                    ("standardize", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "class_weighted_logistic_regression_standardized",
            Pipeline(
                [
                    ("standardize", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=seed,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=seed,
            ),
        ),
        (
            "small_mlp_standardized",
            Pipeline(
                [
                    ("standardize", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(32,),
                            activation="relu",
                            alpha=1e-4,
                            batch_size=64,
                            early_stopping=False,
                            learning_rate_init=1e-3,
                            max_iter=300,
                            random_state=seed,
                            solver="adam",
                        ),
                    ),
                ]
            ),
        ),
    ]


def fit_evaluate_one(
    *,
    dataset: str,
    subset: str,
    view: str,
    split_policy: str,
    model_name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    label_permuted: bool,
    seed: int,
    permutation_trial: int | None = None,
) -> dict[str, Any]:
    if label_permuted:
        if permutation_trial is None:
            permutation_trial = 0
        rng = np.random.RandomState(seed + 1009 + int(permutation_trial))
        fit_y = np.asarray(y_train, dtype=np.int64).copy()
        rng.shuffle(fit_y)
    else:
        fit_y = np.asarray(y_train, dtype=np.int64)
    model.fit(x_train, fit_y)
    y_pred = np.asarray(model.predict(x_val), dtype=np.int64)
    y_score = predict_scores(model, x_val, y_pred)
    metrics = metric_dict(y_true=y_val, y_pred=y_pred, y_score=y_score)
    return {
        "dataset": dataset,
        "subset": subset,
        "view": view,
        "split_policy": split_policy,
        "model": model_name,
        "label_permuted": bool(label_permuted),
        "permutation_trial": permutation_trial,
        "seed": int(seed),
        "train_size": int(y_train.shape[0]),
        "validation_size": int(y_val.shape[0]),
        "train_class_counts": class_counts_dict(y_train),
        "validation_class_counts": class_counts_dict(y_val),
        **metrics,
    }


def evaluate_dataset(
    *,
    dataset: str,
    artifact: dict[str, Any],
    seed: int,
    subset: str = "all",
    row_mask: np.ndarray | None = None,
    split_policy: str = "minxing_row_random",
    group_ids: np.ndarray | None = None,
    permutation_trials: int = DEFAULT_PERMUTATION_TRIALS,
) -> list[dict[str, Any]]:
    y_all = artifact["y"]
    if row_mask is None:
        row_mask = np.ones(y_all.shape[0], dtype=bool)
    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.shape != y_all.shape:
        raise ValueError(f"row_mask shape {row_mask.shape} does not match y {y_all.shape}")

    y = y_all[row_mask]
    subset_group_ids = None if group_ids is None else np.asarray(group_ids)[row_mask]
    if split_policy == "pair_grouped":
        if subset_group_ids is None:
            raise ValueError("pair_grouped split requires group_ids")
        train_idx, val_idx = grouped_split_indices(subset_group_ids, seed=seed)
        crossing = split_group_crossings(subset_group_ids, train_idx, val_idx)
        if crossing["cross_split_group_count"] != 0:
            raise AssertionError(f"{dataset} {subset} grouped split leaked groups")
    elif split_policy == "minxing_row_random":
        train_idx, val_idx = minxing_split_indices(len(y), seed=seed)
    else:
        raise ValueError(f"Unknown split_policy: {split_policy}")

    y_train = y[train_idx]
    y_val = y[val_idx]
    rows: list[dict[str, Any]] = []
    for view, features_all in feature_views(artifact).items():
        features = features_all[row_mask]
        x_train = features[train_idx]
        x_val = features[val_idx]
        for model_name, model in make_model_specs(seed):
            rows.append(
                fit_evaluate_one(
                    dataset=dataset,
                    subset=subset,
                    view=view,
                    split_policy=split_policy,
                    model_name=model_name,
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    label_permuted=False,
                    seed=seed,
                )
            )
        for trial in range(int(permutation_trials)):
            for model_name, model in make_model_specs(seed):
                rows.append(
                    fit_evaluate_one(
                        dataset=dataset,
                        subset=subset,
                        view=view,
                        split_policy=split_policy,
                        model_name=model_name,
                        model=model,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        label_permuted=True,
                        seed=seed,
                        permutation_trial=trial,
                    )
                )
    return rows


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=True, sort_keys=True, default=json_default)
        else:
            flat[key] = value
    return flat


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_for_csv(row) for row in rows]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def bfcl_subset_masks(artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    categories = extract_metadata_values(artifact, "category").astype(str)
    if categories.size == 0:
        raise ValueError("BFCL artifact is missing metadata[].category")
    return {
        "all_categories": np.ones(categories.shape[0], dtype=bool),
        "non_irrelevance": categories != "irrelevance",
        "irrelevance_only": categories == "irrelevance",
    }


def subset_report(
    *,
    artifact: dict[str, Any],
    row_mask: np.ndarray,
    split_policy: str,
    seed: int,
    group_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    y = artifact["y"][row_mask]
    subset_group_ids = None if group_ids is None else np.asarray(group_ids)[row_mask]
    if split_policy == "pair_grouped":
        if subset_group_ids is None:
            raise ValueError("pair_grouped split requires group_ids")
        train_idx, val_idx = grouped_split_indices(subset_group_ids, seed=seed)
    elif split_policy == "minxing_row_random":
        train_idx, val_idx = minxing_split_indices(y.shape[0], seed=seed)
    else:
        raise ValueError(f"Unknown split_policy: {split_policy}")
    return split_report(
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        split_policy=split_policy,
        seed=seed,
        group_ids=subset_group_ids,
    )


def best_non_permuted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["label_permuted"]:
            continue
        grouped.setdefault(
            (row["dataset"], row["subset"], row["split_policy"], row["view"]),
            [],
        ).append(row)
    return [
        max(items, key=lambda row: (row["balanced_accuracy"], row["macro_f1"]))
        for items in grouped.values()
    ]


def permutation_control_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1")
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row["label_permuted"]:
            continue
        grouped.setdefault(
            (
                row["dataset"],
                row["subset"],
                row["split_policy"],
                row["view"],
                row["model"],
            ),
            [],
        ).append(row)

    summary_rows = []
    for (dataset, subset, split_policy, view, model), items in grouped.items():
        summary: dict[str, Any] = {
            "dataset": dataset,
            "subset": subset,
            "split_policy": split_policy,
            "view": view,
            "model": model,
            "n_trials": int(len(items)),
            "trial_indices": sorted(int(row["permutation_trial"]) for row in items),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in items], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values, ddof=0))
            summary[f"{metric_name}_min"] = float(np.min(values))
            summary[f"{metric_name}_max"] = float(np.max(values))
        summary_rows.append(summary)
    return summary_rows


def run_diagnostics(
    *,
    bfcl_path: Path = DEFAULT_BFCL_PKL,
    apibank_path: Path = DEFAULT_APIBANK_PKL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SPLIT_SEED,
    permutation_trials: int = DEFAULT_PERMUTATION_TRIALS,
) -> dict[str, Any]:
    datasets = {
        "bfcl": load_l2t_artifact(bfcl_path),
        "apibank": load_l2t_artifact(apibank_path),
    }
    verifications = {
        name: artifact_verification(name, artifact, seed=seed)
        for name, artifact in datasets.items()
    }
    rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    bfcl = datasets["bfcl"]
    split_reports["bfcl"] = {}
    for subset, mask in bfcl_subset_masks(bfcl).items():
        split_reports["bfcl"][subset] = subset_report(
            artifact=bfcl,
            row_mask=mask,
            split_policy="minxing_row_random",
            seed=seed,
        )
        rows.extend(
            evaluate_dataset(
                dataset="bfcl",
                artifact=bfcl,
                seed=seed,
                subset=subset,
                row_mask=mask,
                split_policy="minxing_row_random",
                permutation_trials=permutation_trials,
            )
        )

    apibank = datasets["apibank"]
    apibank_group_ids = extract_group_ids(apibank)
    apibank_mask = np.ones(apibank["y"].shape[0], dtype=bool)
    split_reports["apibank"] = {
        "all_pairs": subset_report(
            artifact=apibank,
            row_mask=apibank_mask,
            split_policy="pair_grouped",
            seed=seed,
            group_ids=apibank_group_ids,
        )
    }
    if split_reports["apibank"]["all_pairs"]["cross_split_group_count"] != 0:
        raise AssertionError("API-Bank grouped split leaked pair IDs")
    rows.extend(
        evaluate_dataset(
            dataset="apibank",
            artifact=apibank,
            seed=seed,
            subset="all_pairs",
            row_mask=apibank_mask,
            split_policy="pair_grouped",
            group_ids=apibank_group_ids,
            permutation_trials=permutation_trials,
        )
    )

    permutation_summary = permutation_control_summary(rows)

    summary = {
        "seed": int(seed),
        "permutation_trials": int(permutation_trials),
        "artifacts": verifications,
        "split_reports": split_reports,
        "result_count": len(rows),
        "best_non_permuted_by_dataset_view": best_non_permuted(rows),
        "permutation_control_summary": permutation_summary,
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, output_dir / "l2t_supervised_diagnostic_summary.json")
    write_csv(rows, output_dir / "l2t_supervised_diagnostic_results.csv")
    write_csv(
        summary["best_non_permuted_by_dataset_view"],
        output_dir / "l2t_supervised_diagnostic_best_by_view.csv",
    )
    write_csv(
        permutation_summary,
        output_dir / "l2t_supervised_diagnostic_permutation_summary.csv",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl", type=Path, default=DEFAULT_BFCL_PKL)
    parser.add_argument("--apibank", type=Path, default=DEFAULT_APIBANK_PKL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--permutation-trials", type=int, default=DEFAULT_PERMUTATION_TRIALS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_diagnostics(
        bfcl_path=args.bfcl,
        apibank_path=args.apibank,
        output_dir=args.output_dir,
        seed=args.seed,
        permutation_trials=args.permutation_trials,
    )
    print(f"wrote results to {args.output_dir}")
    for row in summary["best_non_permuted_by_dataset_view"]:
        print(
            f"{row['dataset']} {row['subset']} {row['view']}: {row['model']} "
            f"balanced_accuracy={row['balanced_accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
