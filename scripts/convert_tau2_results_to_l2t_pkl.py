#!/usr/bin/env python3
"""Convert selected tau2 result folders into a toy L2T pickle dataset."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_FOLDERS = [
    REPO_ROOT
    / "data/simulations/20260714_131548_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini",
    REPO_ROOT
    / "data/simulations/20260714_140540_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini",
]
OUTPUT_PATH = (
    REPO_ROOT
    / "data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl"
)

NORMAL_TERMINATION_REASONS = {"user_stop", "agent_stop"}
EXPECTED_RETAINED_COUNTS = {"retail": 46, "airline": 47}

FEATURE_NAMES = [
    "domain_retail",
    "domain_airline",
    "expected_action_count",
    "expected_read_action_count",
    "expected_write_action_count",
    "requires_db_mutation",
    "has_communication_checks",
    "has_nl_assertions",
    "has_env_assertions",
    "reward_basis_has_DB",
    "reward_basis_has_COMMUNICATE",
    "reward_basis_has_NL_ASSERTION",
]

TRAJ_LENGTH = 64
WRITE_NAME_HINTS = (
    "update",
    "modify",
    "cancel",
    "book",
    "return",
    "exchange",
    "send",
    "create",
    "delete",
)


def load_results(folder: Path) -> dict[str, Any]:
    results_path = folder / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    return json.loads(results_path.read_text())


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set, str)):
        return len(value) > 0
    return bool(value)


def normalize_tool_type(tool_type: Any) -> str | None:
    if not tool_type:
        return None
    value = str(tool_type).lower()
    if value in {"read", "write"}:
        return value
    return None


def infer_tool_type_from_name(name: str | None) -> str:
    lowered = (name or "").lower()
    if any(hint in lowered for hint in WRITE_NAME_HINTS):
        return "write"
    return "read"


def action_id(action: dict[str, Any]) -> str | None:
    return action.get("action_id") or action.get("id")


def action_name(action: dict[str, Any] | None) -> str | None:
    if not isinstance(action, dict):
        return None
    return action.get("name") or action.get("tool_name")


def build_action_type_maps(
    expected_actions: list[dict[str, Any]], action_checks: Any
) -> tuple[dict[str, str], dict[str, str]]:
    """Build action-id and tool-name maps from task actions and evaluated checks."""
    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}

    for action in expected_actions:
        tool_type = normalize_tool_type(action.get("tool_type"))
        if tool_type is None:
            continue
        aid = action_id(action)
        name = action_name(action)
        if aid:
            by_id[aid] = tool_type
        if name:
            by_name[name] = tool_type

    for check in action_checks or []:
        if not isinstance(check, dict):
            continue
        checked_action = check.get("action") or {}
        tool_type = normalize_tool_type(check.get("tool_type"))
        if tool_type is None:
            continue
        aid = action_id(checked_action)
        name = action_name(checked_action)
        if aid:
            by_id.setdefault(aid, tool_type)
        if name:
            by_name.setdefault(name, tool_type)

    return by_id, by_name


def infer_action_tool_type(
    action: dict[str, Any],
    type_by_id: dict[str, str],
    type_by_name: dict[str, str],
) -> str:
    tool_type = normalize_tool_type(action.get("tool_type"))
    if tool_type is not None:
        return tool_type

    aid = action_id(action)
    if aid and aid in type_by_id:
        return type_by_id[aid]

    name = action_name(action)
    if name and name in type_by_name:
        return type_by_name[name]

    return infer_tool_type_from_name(name)


def reward_basis_from(task_criteria: dict[str, Any], reward_info: dict[str, Any]) -> set[str]:
    reward_basis = task_criteria.get("reward_basis")
    if reward_basis is None:
        reward_basis = reward_info.get("reward_basis") or []
    return {str(item) for item in reward_basis}


def task_expects_action_checks(task: dict[str, Any]) -> bool:
    criteria = task.get("evaluation_criteria") or {}
    return is_non_empty(criteria.get("actions"))


def get_exclusion_reasons(
    simulation: dict[str, Any], task: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    termination_reason = simulation.get("termination_reason")
    if termination_reason not in NORMAL_TERMINATION_REASONS:
        reasons.append(f"non_normal_stop:{termination_reason}")

    reward_info = simulation.get("reward_info")
    if reward_info is None:
        reasons.append("missing_reward_info")
        return reasons
    if not isinstance(reward_info, dict):
        reasons.append("invalid_reward_info")
        return reasons

    if reward_info.get("db_check") is None:
        reasons.append("null_db_check")
    if task_expects_action_checks(task) and reward_info.get("action_checks") is None:
        reasons.append("null_action_checks")
    return reasons


def extract_features(
    domain: str,
    task: dict[str, Any],
    reward_info: dict[str, Any],
) -> tuple[list[float], dict[str, int]]:
    criteria = task.get("evaluation_criteria") or {}
    expected_actions = criteria.get("actions") or []
    type_by_id, type_by_name = build_action_type_maps(
        expected_actions, reward_info.get("action_checks")
    )

    expected_read_action_count = 0
    expected_write_action_count = 0
    for action in expected_actions:
        tool_type = infer_action_tool_type(action, type_by_id, type_by_name)
        if tool_type == "write":
            expected_write_action_count += 1
        else:
            expected_read_action_count += 1

    basis = reward_basis_from(criteria, reward_info)
    has_communication_checks = is_non_empty(criteria.get("communicate_info")) or is_non_empty(
        criteria.get("communicate_checks")
    )

    counts = {
        "expected_action_count": len(expected_actions),
        "expected_read_action_count": expected_read_action_count,
        "expected_write_action_count": expected_write_action_count,
    }
    features = [
        float(domain == "retail"),
        float(domain == "airline"),
        float(counts["expected_action_count"]),
        float(counts["expected_read_action_count"]),
        float(counts["expected_write_action_count"]),
        float(expected_write_action_count > 0),
        float(has_communication_checks),
        float(is_non_empty(criteria.get("nl_assertions"))),
        float(is_non_empty(criteria.get("env_assertions"))),
        float("DB" in basis),
        float("COMMUNICATE" in basis),
        float("NL_ASSERTION" in basis),
    ]
    return features, counts


def tool_call_name(tool_call: dict[str, Any]) -> str | None:
    name = tool_call.get("name")
    if name:
        return name
    function = tool_call.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return None


def message_sort_key(message: dict[str, Any]) -> tuple[float, str]:
    turn_idx = message.get("turn_idx")
    if turn_idx is None:
        turn_idx = float("inf")
    return float(turn_idx), str(message.get("timestamp") or "")


def encode_trajectory(
    simulation: dict[str, Any],
    expected_actions: list[dict[str, Any]],
) -> np.ndarray:
    reward_info = simulation.get("reward_info") or {}
    _, type_by_name = build_action_type_maps(
        expected_actions, reward_info.get("action_checks")
    )

    events: list[int] = []
    for message in sorted(simulation.get("messages") or [], key=message_sort_key):
        role = message.get("role")
        if role == "user":
            events.append(1)
        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    name = tool_call_name(tool_call)
                    tool_type = type_by_name.get(name or "") or infer_tool_type_from_name(name)
                    events.append(4 if tool_type == "write" else 3)
            elif is_non_empty(message.get("content")):
                events.append(2)
        elif role == "tool":
            events.append(6 if bool(message.get("error")) else 5)

    events.append(7)
    if len(events) >= TRAJ_LENGTH:
        events = events[:TRAJ_LENGTH]
    else:
        events.extend([0] * (TRAJ_LENGTH - len(events)))
    return np.asarray(events, dtype=np.float32)


def count_tool_calls(simulation: dict[str, Any]) -> int:
    count = 0
    for message in simulation.get("messages") or []:
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def validate_dataset(dataset: dict[str, Any]) -> None:
    x = dataset["X"]
    y = dataset["y"]
    s = dataset["traj"]["s"]

    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if s.ndim != 2:
        raise ValueError(f'traj["s"] must be 2D, got shape {s.shape}')
    if not (len(x) == len(y) == len(s)):
        raise ValueError(f"Length mismatch: X={len(x)}, y={len(y)}, traj={len(s)}")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"y must contain only 0/1 labels, got {np.unique(y)}")
    if np.isnan(x).any():
        raise ValueError("X contains NaNs")
    if np.isnan(s).any():
        raise ValueError('traj["s"] contains NaNs')


def convert() -> dict[str, Any]:
    rows: list[list[float]] = []
    labels: list[int] = []
    trajectories: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    retained_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    exclusion_reasons_by_domain: dict[str, Counter[str]] = defaultdict(Counter)

    for folder in INPUT_FOLDERS:
        results = load_results(folder)
        domain = results["info"]["environment_info"]["domain_name"]
        tasks_by_id = {str(task["id"]): task for task in results.get("tasks") or []}

        for simulation in results.get("simulations") or []:
            task_id = str(simulation["task_id"])
            task = tasks_by_id[task_id]
            exclusion_reasons = get_exclusion_reasons(simulation, task)
            if exclusion_reasons:
                excluded_counts[domain] += 1
                exclusion_reasons_by_domain[domain].update(exclusion_reasons)
                continue

            reward_info = simulation["reward_info"]
            features, counts = extract_features(domain, task, reward_info)
            expected_actions = (task.get("evaluation_criteria") or {}).get("actions") or []

            reward = float(reward_info.get("reward") or 0.0)
            label = 1 if reward == 1.0 else 0
            rows.append(features)
            labels.append(label)
            trajectories.append(encode_trajectory(simulation, expected_actions))
            retained_counts[domain] += 1
            positive_counts[domain] += label
            metadata.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "source_result_folder": str(folder.relative_to(REPO_ROOT)),
                    "termination_reason": simulation.get("termination_reason"),
                    "reward": reward,
                    "db_match": (reward_info.get("db_check") or {}).get("db_match"),
                    "num_messages": len(simulation.get("messages") or []),
                    "num_tool_calls": count_tool_calls(simulation),
                    **counts,
                }
            )

    if dict(retained_counts) != EXPECTED_RETAINED_COUNTS:
        raise ValueError(
            "Unexpected retained counts: "
            f"{dict(retained_counts)} != {EXPECTED_RETAINED_COUNTS}"
        )

    dataset = {
        "X": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "traj": {"s": np.vstack(trajectories).astype(np.float32)},
        "metadata": metadata,
        "feature_names": FEATURE_NAMES,
        "stats": {
            "retained_counts_by_domain": dict(retained_counts),
            "excluded_counts_by_domain": dict(excluded_counts),
            "positive_label_count_by_domain": dict(positive_counts),
            "exclusion_reasons_by_domain": {
                domain: dict(reasons)
                for domain, reasons in exclusion_reasons_by_domain.items()
            },
        },
    }
    validate_dataset(dataset)
    return dataset


def main() -> None:
    dataset = convert()
    stats = dataset["stats"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"output path: {OUTPUT_PATH}")
    print(f"X shape: {dataset['X'].shape}")
    print(f"y shape: {dataset['y'].shape}")
    print(f"traj['s'] shape: {dataset['traj']['s'].shape}")
    print(f"total N: {len(dataset['y'])}")
    print(f"retained counts by domain: {stats['retained_counts_by_domain']}")
    print(f"excluded counts by domain: {stats['excluded_counts_by_domain']}")
    print(
        "positive label count by domain: "
        f"{stats['positive_label_count_by_domain']}"
    )
    success_rates = {
        domain: (
            stats["positive_label_count_by_domain"].get(domain, 0)
            / retained_count
            if retained_count
            else 0.0
        )
        for domain, retained_count in stats["retained_counts_by_domain"].items()
    }
    print(f"success rate by domain: {success_rates}")
    print(f"feature names: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
