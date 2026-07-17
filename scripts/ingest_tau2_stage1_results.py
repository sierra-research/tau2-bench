#!/usr/bin/env python3
"""Ingest retained tau2 Stage 1 raw results."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_tau2_stage1_manifest import (  # noqa: E402
    DEFAULT_OUTPUT_JSON as DEFAULT_MANIFEST,
)
from build_tau2_stage1_manifest import (
    DEFAULT_RAW_DIR,
)
from build_toolcalling_numerical_representation import (  # noqa: E402
    write_jsonl,
)
from build_toolcalling_numerical_representation import (
    write_json as write_json_file,
)
from build_unified_toolcalling_dataset import (  # noqa: E402
    SCHEMA_FIELDS,
    TAU2_LABEL_ORIGIN,
    TAU2_LABEL_SCOPE,
    TAU2_SOURCE_DATASET,
    tau2_event_feature_names,
    validate_schema,
)
from convert_tau2_results_to_l2t_pkl import (  # noqa: E402
    FEATURE_NAMES,
    count_tool_calls,
    encode_trajectory,
    extract_features,
    get_exclusion_reasons,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETAINED_JSONL = REPO_ROOT / "data/processed/tau2_stage1_retained.jsonl"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "data/processed/tau2_stage1_ingestion_summary.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: dict[str, Any], path: Path) -> None:
    write_json_file(value, path)


def manifest_by_task_id(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    return {str(task["task_id"]): task for task in manifest["tasks"]}


def raw_result_paths(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("task_*.json"))


def x_features_from_row(values: list[float]) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, values, strict=True))


def s_features_from_events(events: list[int]) -> dict[str, int]:
    return dict(zip(tau2_event_feature_names(len(events)), events, strict=True))


def source_path(path: Path) -> str:
    return (
        str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else str(path)
    )


def record_from_simulation(
    *,
    result_path: Path,
    domain: str,
    task: dict[str, Any],
    simulation: dict[str, Any],
    manifest_task: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(simulation["task_id"])
    reward_info = simulation["reward_info"]
    feature_values, counts = extract_features(domain, task, reward_info)
    expected_actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    events = [
        int(value) for value in encode_trajectory(simulation, expected_actions).tolist()
    ]
    reward = float(reward_info.get("reward") or 0.0)
    label = 1 if reward == 1.0 else 0
    sample_id = f"tau2:{domain}:task_{task_id}:stage1"
    metadata = {
        "domain": domain,
        "task_id": task_id,
        "sample_id": sample_id,
        "source_result_folder": source_path(result_path.parent),
        "source_input_path": source_path(result_path),
        "termination_reason": simulation.get("termination_reason"),
        "reward": reward,
        "db_match": (reward_info.get("db_check") or {}).get("db_match"),
        "num_messages": len(simulation.get("messages") or []),
        "num_tool_calls": count_tool_calls(simulation),
        "stage1_selection_group": manifest_task["selection_group"],
        "stage1_run_identity": f"{domain}:task_{task_id}:stage1",
        **counts,
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
    return {
        "sample_id": sample_id,
        "source_dataset": TAU2_SOURCE_DATASET,
        "domain": domain,
        "label_scope": TAU2_LABEL_SCOPE,
        "label_origin": TAU2_LABEL_ORIGIN,
        "is_synthetic": False,
        "x_raw": {},
        "s_raw": events,
        "x_numeric_features": x_features_from_row(feature_values),
        "s_numeric_features": s_features_from_events(events),
        "y": label,
        "metadata": metadata,
    }


def ingest_raw_results(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_tasks = manifest_by_task_id(manifest_path)
    retained: list[dict[str, Any]] = []
    attempted_count = 0
    completed_count = 0
    filtered_count = 0
    filter_reasons: Counter[str] = Counter()
    selection_group_counts: Counter[str] = Counter()

    for result_path in raw_result_paths(raw_dir):
        data = load_json(result_path)
        domain = data["info"]["environment_info"]["domain_name"]
        tasks_by_id = {str(task["id"]): task for task in data.get("tasks") or []}
        for simulation in data.get("simulations") or []:
            attempted_count += 1
            completed_count += 1
            task_id = str(simulation["task_id"])
            if task_id not in manifest_tasks:
                filtered_count += 1
                filter_reasons["task_not_in_stage1_manifest"] += 1
                continue
            task = tasks_by_id[task_id]
            reasons = get_exclusion_reasons(simulation, task)
            if reasons:
                filtered_count += 1
                filter_reasons.update(reasons)
                continue
            record = record_from_simulation(
                result_path=result_path,
                domain=domain,
                task=task,
                simulation=simulation,
                manifest_task=manifest_tasks[task_id],
            )
            retained.append(record)
            selection_group_counts[manifest_tasks[task_id]["selection_group"]] += 1

    validate_schema(retained)
    y_distribution = dict(
        sorted(Counter(str(record["y"]) for record in retained).items())
    )
    summary = {
        "attempted_count": attempted_count,
        "completed_count": completed_count,
        "retained_count": len(retained),
        "filtered_count": filtered_count,
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "y_distribution": y_distribution,
        "counts_by_stage1_selection_group": dict(
            sorted(selection_group_counts.items())
        ),
        "original_filtering_logic_reused": {
            "get_exclusion_reasons": "convert_tau2_results_to_l2t_pkl.get_exclusion_reasons",
            "extract_features": "convert_tau2_results_to_l2t_pkl.extract_features",
            "encode_trajectory": "convert_tau2_results_to_l2t_pkl.encode_trajectory",
            "label_rule": "y = 1 only when reward == 1.0, else y = 0",
        },
        "schema_fields": SCHEMA_FIELDS,
    }
    return retained, summary


def build_outputs(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    retained_jsonl: Path = DEFAULT_RETAINED_JSONL,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    write_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained, summary = ingest_raw_results(raw_dir=raw_dir, manifest_path=manifest_path)
    if write_outputs:
        write_jsonl(retained, retained_jsonl)
        write_json(summary, summary_json)
    return retained, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--retained-jsonl", type=Path, default=DEFAULT_RETAINED_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retained, summary = build_outputs(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest_path,
        retained_jsonl=args.retained_jsonl,
        summary_json=args.summary_json,
    )
    print(f"retained Stage 1 records: {len(retained)}")
    print(f"summary: {args.summary_json}")
    print(
        "Stage 1 retained records were not merged into the original 93-record tau2 "
        "file. Use scripts/build_tau2_stage1_analysis.py for the canonical merge "
        "and comparison outputs."
    )


if __name__ == "__main__":
    main()
