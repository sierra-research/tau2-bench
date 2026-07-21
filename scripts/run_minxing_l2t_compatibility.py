#!/usr/bin/env python3
"""Run Minxing compatibility diagnostics on L2T bridge artifacts.

This script intentionally imports Minxing's existing implementation at runtime
instead of copying its model or training code into tau2-bench.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shlex
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from l2t_model_bridge import DEFAULT_SPLIT_SEED, REPO_ROOT, json_default

VALID_MODES = ("proposed_only", "label_bce", "reconstruction_only")
MODE_TO_MINXING_NAME = {
    "proposed_only": "proposed",
    "label_bce": "label_bce",
    "reconstruction_only": "reconstruction_only",
}

DATASETS = {
    "tau2": {
        "path": REPO_ROOT
        / "data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl",
        "batch_train": 16,
        "label_scope": "tau2 retail/airline task success after filtering",
    },
    "bfcl": {
        "path": REPO_ROOT / "data/processed/l2t/bfcl/bfcl_v4_non_live_1240_l2t.pkl",
        "batch_train": 128,
        "label_scope": "BFCL v4 non-live executable/correct call outcome",
    },
    "apibank": {
        "path": REPO_ROOT / "data/processed/l2t/apibank/apibank_full_l2t.pkl",
        "batch_train": 128,
        "label_scope": "API-Bank synthetic success/failure label",
    },
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
    """Match Minxing's `np.random.RandomState(seed).permutation` split exactly."""
    perm = np.random.RandomState(seed).permutation(int(n_samples))
    n_train = int(0.8 * int(n_samples))
    return perm[:n_train], perm[n_train:]


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("at least one prediction is required")

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    n0 = tn + fp
    n1 = fn + tp
    total = int(y_true.size)

    recall0 = (tn / n0) if n0 else None
    recall1 = (tp / n1) if n1 else None
    precision0 = (tn / (tn + fn)) if (tn + fn) else 0.0
    precision1 = (tp / (tp + fp)) if (tp + fp) else 0.0
    f1_0 = (
        2 * precision0 * recall0 / (precision0 + recall0)
        if recall0 is not None and (precision0 + recall0)
        else 0.0
    )
    f1_1 = (
        2 * precision1 * recall1 / (precision1 + recall1)
        if recall1 is not None and (precision1 + recall1)
        else 0.0
    )
    recalls = [value for value in (recall0, recall1) if value is not None]

    values, counts = np.unique(y_true, return_counts=True)
    class_distribution = {
        str(int(value)): int(count) for value, count in zip(values, counts, strict=True)
    }
    majority_index = int(np.argmax(counts))
    predicted_classes = [int(value) for value in np.unique(y_pred).tolist()]

    return {
        "n": total,
        "class_distribution": class_distribution,
        "accuracy": float((tn + tp) / total),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float((f1_0 + f1_1) / 2.0),
        "recall_y0": None if recall0 is None else float(recall0),
        "recall_y1": None if recall1 is None else float(recall1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "majority_baseline": float(int(counts[majority_index]) / total),
        "majority_class": int(values[majority_index]),
        "one_class_collapse": bool(len(predicted_classes) == 1),
        "collapse_status": (
            f"collapsed_to_y{predicted_classes[0]}"
            if len(predicted_classes) == 1
            else "not_collapsed"
        ),
        "predicted_classes": predicted_classes,
    }


def source_fingerprint(root: Path) -> dict[str, str]:
    """Hash Minxing source files that this diagnostic may import."""
    root = root.resolve()
    paths = [root / "share_code/experiment/run_baseline.py"]
    src_root = root / "share_code/src"
    if src_root.exists():
        paths.extend(sorted(src_root.rglob("*.py")))

    fingerprints: dict[str, str] = {}
    for path in paths:
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprints[str(path.relative_to(root))] = digest
    return fingerprints


def changed_fingerprints(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, dict[str, str | None]]:
    changed: dict[str, dict[str, str | None]] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changed[key] = {"before": before.get(key), "after": after.get(key)}
    return changed


def load_minxing_run_baseline(minxing_repo: Path) -> ModuleType:
    minxing_repo = minxing_repo.resolve()
    run_baseline = minxing_repo / "share_code/experiment/run_baseline.py"
    if not run_baseline.exists():
        raise FileNotFoundError(f"Minxing run_baseline.py not found: {run_baseline}")

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "minxing_share_run_baseline", run_baseline
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {run_baseline}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def command_string(argv: list[str] | None = None) -> str:
    argv = sys.argv if argv is None else argv
    return " ".join(shlex.quote(arg) for arg in argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run existing Minxing modes on one L2T-compatible tool-calling dataset "
            "and save complete compatibility metrics."
        )
    )
    parser.add_argument("--minxing-repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--input-data", type=Path, default=None)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-train", type=int, default=None, dest="batch_train")
    parser.add_argument("--batch-val", type=int, default=512, dest="batch_val")
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--latent-batch-size", type=int, default=4096, dest="latent_batch_size")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time-window", type=float, nargs=2, default=None, dest="time_window")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--loss-type", type=str, default="stability_asymmetric_recon", dest="loss_type")
    parser.add_argument("--lambda-mmd", type=float, default=1e-2, dest="lambda_mmd")
    parser.add_argument("--asym-recon-delta", type=float, default=10.0, dest="asym_recon_delta")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay")
    parser.add_argument("--lr-enc", type=float, default=3e-3, dest="lr_enc")
    parser.add_argument("--weight-decay-enc", type=float, default=1e-5, dest="weight_decay_enc")
    parser.add_argument("--d-x", type=int, default=2, dest="d_x")
    parser.add_argument("--d-o", type=int, default=1, dest="d_o")
    parser.add_argument("--d-v", type=int, default=1, dest="d_v")
    parser.add_argument("--d-h", type=int, default=128, dest="d_h")
    parser.add_argument("--hidden-enc", type=int, default=512, dest="hidden_enc")
    parser.add_argument("--hidden-dec", type=int, default=128, dest="hidden_dec")

    args = parser.parse_args(argv)
    dataset_defaults = DATASETS[args.dataset]
    args.input_data = (
        dataset_defaults["path"] if args.input_data is None else args.input_data
    )
    args.batch_train = (
        int(dataset_defaults["batch_train"])
        if args.batch_train is None
        else int(args.batch_train)
    )
    args.mode_resolved = MODE_TO_MINXING_NAME[args.mode]
    if args.time_window is not None:
        args.time_window = tuple(float(value) for value in args.time_window)
    return args


def load_external_data(rb: ModuleType, input_data: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    x_np, y, traj = rb._load_external_data(str(input_data))
    return np.asarray(x_np, dtype=np.float32), np.asarray(y).reshape(-1), traj


def build_sequence_pairs(minxing_repo: Path, s: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    src = minxing_repo.resolve() / "share_code/src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from data.datasets import make_sequence_pairs  # noqa: PLC0415

    return make_sequence_pairs(s=s, stride=stride)


def resolve_time_window(
    requested: tuple[float, float] | None,
    *,
    sequence_dt: float,
    sequence_length: int,
) -> tuple[float, float]:
    seq_start = float(sequence_dt)
    seq_end = float(sequence_dt * sequence_length)
    window = (seq_start, seq_end) if requested is None else requested
    if window[0] > window[1]:
        raise ValueError(f"Invalid time window {window}: start must be <= end")
    if window[0] < seq_start or window[1] > seq_end:
        raise ValueError(
            f"time_window={window} outside valid sequence range [{seq_start}, {seq_end}]"
        )
    return window


def make_loaders(
    *,
    minxing_repo: Path,
    rb: ModuleType,
    device: Any,
    x_np: np.ndarray,
    y: np.ndarray,
    o_hist: np.ndarray,
    o_next: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_train: int,
    batch_val: int,
    num_workers: int,
) -> tuple[Any, Any, Any, Any, Any]:
    from torch.utils.data import DataLoader  # noqa: PLC0415

    src = minxing_repo.resolve() / "share_code/src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from data.datasets import SpringDataset  # noqa: PLC0415

    train_ds = SpringDataset(x_np[train_idx], o_hist[train_idx], o_next[train_idx], y=y[train_idx])
    val_ds = SpringDataset(x_np[val_idx], o_hist[val_idx], o_next[val_idx], y=y[val_idx])
    train_eval_ds = SpringDataset(
        x_np[train_idx], o_hist[train_idx], o_next[train_idx], y=y[train_idx]
    )
    loader_kwargs = rb._make_loader_kwargs(device, num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=batch_val,
        shuffle=False,
        **loader_kwargs,
    )
    return train_ds, val_ds, train_loader, val_loader, train_eval_loader


def _prediction_rows(
    *,
    split_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for row_number, original_index in enumerate(split_indices.astype(int).tolist()):
        rows.append(
            {
                "row": int(row_number),
                "original_index": int(original_index),
                "y_true": int(y_true[row_number]),
                "y_pred": int(y_pred[row_number]),
                "score": float(scores[row_number]),
            }
        )
    return rows


def predict_sequence(
    *,
    minxing_repo: Path,
    model: Any,
    loader: Any,
    split_indices: np.ndarray,
    device: Any,
    stability_cfg: Any,
) -> pd.DataFrame:
    import torch  # noqa: PLC0415

    src = minxing_repo.resolve() / "share_code/src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from train.training_clean import _stability_logits_from_o_hat  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, o_hist, _o_next, y = batch
            x = x.to(device)
            o_hist = o_hist.to(device)
            y_true = y.numpy().reshape(-1).astype(int)
            o_hat, _ = model(x, o_hist)
            logits = _stability_logits_from_o_hat(
                o_hat=o_hat,
                threshold=float(stability_cfg.threshold),
                window=stability_cfg.window,
                dt=float(stability_cfg.dt),
                t0=float(stability_cfg.t0),
                smooth_beta=float(stability_cfg.smooth_beta),
            )
            scores = logits.detach().cpu().numpy()
            y_pred = (scores >= 0.0).astype(int)
            batch_indices = split_indices[offset : offset + y_true.shape[0]]
            rows.extend(
                _prediction_rows(
                    split_indices=batch_indices,
                    y_true=y_true,
                    y_pred=y_pred,
                    scores=scores,
                )
            )
            offset += y_true.shape[0]
    return pd.DataFrame(rows)


def predict_label(
    *,
    model: Any,
    loader: Any,
    split_indices: np.ndarray,
    device: Any,
) -> pd.DataFrame:
    import torch  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, _o_hist, _o_next, y = batch
            x = x.to(device)
            y_true = y.numpy().reshape(-1).astype(int)
            scores = torch.sigmoid(model(x)).detach().cpu().numpy()
            y_pred = (scores >= 0.5).astype(int)
            batch_indices = split_indices[offset : offset + y_true.shape[0]]
            rows.extend(
                _prediction_rows(
                    split_indices=batch_indices,
                    y_true=y_true,
                    y_pred=y_pred,
                    scores=scores,
                )
            )
            offset += y_true.shape[0]
    return pd.DataFrame(rows)


def train_existing_mode(
    *,
    rb: ModuleType,
    cfg: argparse.Namespace,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    sequence_dt: float,
) -> tuple[Any, pd.DataFrame, str, str, Any]:
    if cfg.mode_resolved == "proposed":
        train_cfg = rb._make_sequence_train_config(
            lambda_mmd=cfg.lambda_mmd,
            loss_type=cfg.loss_type,
            tau=cfg.tau,
            time_window=cfg.time_window,
            sequence_dt=sequence_dt,
            asym_recon_delta=cfg.asym_recon_delta,
        )
        model, history = rb.train_sequence_model(
            model_name="proposed",
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=train_cfg,
        )
        objective = (
            f"stability_asymmetric_recon(delta={cfg.asym_recon_delta}) "
            f"+ lambda_mmd={cfg.lambda_mmd} * MMD"
        )
        return model, history, "X and S", objective, train_cfg.stability

    if cfg.mode_resolved == "reconstruction_only":
        train_cfg = rb._make_sequence_train_config(
            lambda_mmd=0.0,
            loss_type="stability_asymmetric_recon",
            tau=cfg.tau,
            time_window=cfg.time_window,
            sequence_dt=sequence_dt,
            asym_recon_delta=1.0,
        )
        model, history = rb.train_sequence_model(
            model_name="reconstruction_only",
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=train_cfg,
        )
        objective = "plain sequence reconstruction MSE (asym_recon_delta=1, lambda_mmd=0)"
        return model, history, "X and S", objective, train_cfg.stability

    if cfg.mode_resolved == "label_bce":
        model, history = rb.train_label_classifier(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        return model, history, "X only", "binary cross-entropy on benchmark success label", None

    raise ValueError(f"Unsupported resolved mode: {cfg.mode_resolved}")


def build_metrics_payload(
    *,
    cfg: argparse.Namespace,
    runtime_sec: float,
    device: Any,
    inputs_used: str,
    objective: str,
    train_pred_df: pd.DataFrame,
    val_pred_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_np: np.ndarray,
    s: np.ndarray,
    o_hist: np.ndarray,
    minxing_changed_files: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    train_metrics = metrics_from_predictions(
        train_pred_df["y_true"].to_numpy(), train_pred_df["y_pred"].to_numpy()
    )
    val_metrics = metrics_from_predictions(
        val_pred_df["y_true"].to_numpy(), val_pred_df["y_pred"].to_numpy()
    )
    return {
        "schema_version": "minxing_l2t_compatibility_v1",
        "dataset": cfg.dataset,
        "dataset_metadata": {
            "input_data": str(cfg.input_data),
            "label_scope": DATASETS[cfg.dataset]["label_scope"],
        },
        "mode": cfg.mode,
        "mode_resolved": cfg.mode_resolved,
        "inputs_used": inputs_used,
        "training_objective": objective,
        "runtime_sec": float(runtime_sec),
        "device": str(device),
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "split": {
            "seed": int(cfg.seed),
            "train_fraction": 0.8,
            "implementation": "np.random.RandomState(seed).permutation(N)",
            "train_indices": train_idx.astype(int).tolist(),
            "val_indices": val_idx.astype(int).tolist(),
        },
        "data": {
            "X_shape": list(x_np.shape),
            "s_shape": list(s.shape),
            "o_hist_shape": list(o_hist.shape),
        },
        "command": command_string(),
        "configuration": serializable_config(cfg),
        "minxing_integrity": {
            "source_changed": bool(minxing_changed_files),
            "changed_files": minxing_changed_files,
        },
    }


def serializable_config(cfg: argparse.Namespace) -> dict[str, Any]:
    data = vars(cfg).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def run(cfg: argparse.Namespace) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    cfg.minxing_repo = cfg.minxing_repo.resolve()
    cfg.input_data = cfg.input_data.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    before = source_fingerprint(cfg.minxing_repo)
    rb = load_minxing_run_baseline(cfg.minxing_repo)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = rb._choose_device()
    x_np, y, traj = load_external_data(rb, cfg.input_data)
    s = np.asarray(traj["s"], dtype=np.float32)
    cfg.d_x = int(x_np.shape[1])
    o_hist, o_next = build_sequence_pairs(cfg.minxing_repo, s, cfg.stride)
    cfg.d_o = int(o_hist.shape[-1])

    sequence_dt = float(cfg.dt * cfg.stride)
    cfg.time_window = resolve_time_window(
        cfg.time_window,
        sequence_dt=sequence_dt,
        sequence_length=int(o_next.shape[1]),
    )
    train_idx, val_idx = minxing_split_indices(int(x_np.shape[0]), seed=cfg.seed)
    train_ds, _val_ds, train_loader, val_loader, train_eval_loader = make_loaders(
        minxing_repo=cfg.minxing_repo,
        rb=rb,
        device=device,
        x_np=x_np,
        y=y,
        o_hist=o_hist,
        o_next=o_next,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_train=cfg.batch_train,
        batch_val=cfg.batch_val,
        num_workers=cfg.num_workers,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Training loader is empty; reduce --batch-train below train size {len(train_ds)}"
        )

    start = time.perf_counter()
    model, history, inputs_used, objective, stability_cfg = train_existing_mode(
        rb=rb,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        sequence_dt=sequence_dt,
    )
    runtime_sec = time.perf_counter() - start

    history.to_csv(
        cfg.output_dir / f"training_history_{cfg.mode_resolved}.csv", index=False
    )
    if cfg.mode_resolved == "label_bce":
        train_pred_df = predict_label(
            model=model,
            loader=train_eval_loader,
            split_indices=train_idx,
            device=device,
        )
        val_pred_df = predict_label(
            model=model,
            loader=val_loader,
            split_indices=val_idx,
            device=device,
        )
    else:
        train_pred_df = predict_sequence(
            minxing_repo=cfg.minxing_repo,
            model=model,
            loader=train_eval_loader,
            split_indices=train_idx,
            device=device,
            stability_cfg=stability_cfg,
        )
        val_pred_df = predict_sequence(
            minxing_repo=cfg.minxing_repo,
            model=model,
            loader=val_loader,
            split_indices=val_idx,
            device=device,
            stability_cfg=stability_cfg,
        )

    train_pred_df.to_csv(cfg.output_dir / "train_predictions.csv", index=False)
    val_pred_df.to_csv(cfg.output_dir / "val_predictions.csv", index=False)

    after = source_fingerprint(cfg.minxing_repo)
    changed = changed_fingerprints(before, after)
    if changed:
        write_json(
            {"error": "Minxing source changed during diagnostic run", "changed": changed},
            cfg.output_dir / "minxing_integrity_error.json",
        )
        raise RuntimeError("Minxing source files changed during diagnostic run")

    metrics = build_metrics_payload(
        cfg=cfg,
        runtime_sec=runtime_sec,
        device=device,
        inputs_used=inputs_used,
        objective=objective,
        train_pred_df=train_pred_df,
        val_pred_df=val_pred_df,
        train_idx=train_idx,
        val_idx=val_idx,
        x_np=x_np,
        s=s,
        o_hist=o_hist,
        minxing_changed_files=changed,
    )
    write_json(metrics, cfg.output_dir / "metrics.json")
    write_json(serializable_config(cfg), cfg.output_dir / "run_config.json")

    print(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "mode": cfg.mode,
                "output_dir": str(cfg.output_dir),
                "runtime_sec": round(runtime_sec, 3),
                "val_metrics": metrics["val_metrics"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()
