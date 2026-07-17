#!/usr/bin/env python3
"""Build the dry-run manifest for tau2 retail Stage 1 collection."""

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

from plan_tau2_additional_sampling import (  # noqa: E402
    DEFAULT_UNIFIED_TAU2_JSONL,
    DOMAIN_TASK_FILES,
    action_kind,
    load_json,
    load_jsonl,
    prior_simulated_ids,
    retained_outcome_ids,
    task_sort_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PLAN = REPO_ROOT / "data/processed/tau2_additional_sampling_plan.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data/processed/tau2_stage1_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "docs/tau2_stage1_manifest.md"
DEFAULT_RAW_DIR = REPO_ROOT / "data/processed/tau2_stage1_raw"
STAGE1_SEED = 20260717
AGENT_MODEL = "gpt-4o-mini"
USER_MODEL = "gpt-4o-mini"
AGENT = "llm_agent"
USER = "user_simulator"
SELECTION_GROUP_COUNTS = {
    "two_plus_writes": 8,
    "no_write": 2,
    "low_action_one_write": 2,
}


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def selected_group(task: dict[str, Any]) -> str:
    write_count = int(task["expected_write_action_count"])
    action_count = int(task["expected_action_count"])
    if write_count >= 2:
        return "two_plus_writes"
    if write_count == 0:
        return "no_write"
    if action_count <= 1 and write_count == 1:
        return "low_action_one_write"
    raise ValueError(f"Task {task['task_id']} does not fit a Stage 1 selection group")


def task_features_by_id(domain: str) -> dict[str, dict[str, Any]]:
    tasks = load_json(DOMAIN_TASK_FILES[domain])
    features = {}
    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        kinds = [action_kind(action.get("name")) for action in actions]
        task_id = str(task["id"])
        features[task_id] = {
            "task_id": task_id,
            "domain": domain,
            "expected_action_count": len(actions),
            "expected_read_action_count": kinds.count("read"),
            "expected_write_action_count": kinds.count("write"),
            "requires_db_mutation": kinds.count("write") > 0,
        }
    return features


def illustrative_batch_tau2_command(task_ids: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        "retail",
        "--agent",
        AGENT,
        "--user",
        USER,
        "--agent-llm",
        AGENT_MODEL,
        "--user-llm",
        USER_MODEL,
        "--num-trials",
        "1",
        "--task-ids",
        *task_ids,
        "--max-concurrency",
        "1",
        "--seed",
        str(STAGE1_SEED),
        "--log-level",
        "DEBUG",
        "--verbose-logs",
        "--llm-log-mode",
        "all",
        "--auto-resume",
        "--save-to",
        "tau2_stage1_raw/stage1_retail_12",
    ]


def validate_manifest(manifest: dict[str, Any]) -> None:
    tasks = manifest["tasks"]
    task_ids = [task["task_id"] for task in tasks]
    if len(tasks) != 12:
        raise ValueError(f"Expected 12 tasks, found {len(tasks)}")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 1 manifest contains duplicate task IDs")
    counts = Counter(task["selection_group"] for task in tasks)
    if dict(counts) != SELECTION_GROUP_COUNTS:
        raise ValueError(f"Unexpected Stage 1 composition: {dict(counts)}")
    if any(task["domain"] != "retail" for task in tasks):
        raise ValueError("Stage 1 manifest must contain retail tasks only")
    if any(task["previously_attempted"] for task in tasks):
        raise ValueError("Stage 1 manifest contains a previously attempted task")
    if any(task["previously_retained"] for task in tasks):
        raise ValueError("Stage 1 manifest contains a previously retained task")
    if any(task["selection_uses_outcome_label"] for task in tasks):
        raise ValueError("Stage 1 manifest selection uses outcome labels")


def build_manifest(
    *,
    input_plan: Path = DEFAULT_INPUT_PLAN,
    unified_tau2_jsonl: Path = DEFAULT_UNIFIED_TAU2_JSONL,
    raw_output_dir: Path = DEFAULT_RAW_DIR,
) -> dict[str, Any]:
    plan = load_json(input_plan)
    recommended = plan["proposed_next_batch"]["stage_1_tasks"]
    task_features = task_features_by_id("retail")
    attempted = prior_simulated_ids("retail")
    retained = retained_outcome_ids(load_jsonl(unified_tau2_jsonl), "retail")

    tasks = []
    for planned in recommended:
        task_id = str(planned["task_id"])
        if planned["domain"] != "retail":
            raise ValueError(f"Stage 1 cannot include non-retail task {task_id}")
        if task_id not in task_features:
            raise ValueError(
                f"Selected task {task_id} is missing from local retail tasks"
            )
        features = task_features[task_id]
        group = selected_group(features)
        tasks.append(
            {
                **features,
                "selection_group": group,
                "previously_attempted": task_id in attempted,
                "previously_retained": task_id in retained,
                "selection_reason": "; ".join(planned.get("selection_reasons") or []),
                "selection_uses_outcome_label": bool(planned.get("uses_y", False)),
            }
        )

    tasks = sorted(
        tasks,
        key=lambda item: (
            list(SELECTION_GROUP_COUNTS).index(item["selection_group"]),
            task_sort_key(item["task_id"]),
        ),
    )
    task_ids = [task["task_id"] for task in tasks]
    command = illustrative_batch_tau2_command(task_ids)
    max_calls = sum(max(1, task["expected_action_count"]) for task in tasks) * 2
    manifest = {
        "metadata": {
            "created_for": "tau2 retail Stage 1 collection dry-run",
            "seed": STAGE1_SEED,
            "deterministic_selection": True,
            "selection_source": str(input_plan.relative_to(REPO_ROOT)),
            "selection_policy": (
                "Selected from the existing additional-sampling plan using local task "
                "definitions and prior attempt/retained IDs only; no y, reward, or "
                "success/failure labels are used."
            ),
            "selection_uses_outcome_labels": False,
            "telecom_excluded": True,
            "num_trials_per_task": 1,
        },
        "model_configuration": {
            "agent": AGENT,
            "user": USER,
            "agent_model": AGENT_MODEL,
            "user_model": USER_MODEL,
            "agent_model_source": "existing tau2 pilot result folders",
            "user_model_source": "existing tau2 pilot result folders",
        },
        "run": {
            "illustrative_batch_tau2_command": command,
            "illustrative_batch_tau2_command_text": " ".join(command),
            "runner_dry_run_command": "uv run python scripts/run_tau2_stage1.py",
            "runner_execute_command": "uv run python scripts/run_tau2_stage1.py --execute",
            "execution_path": (
                "scripts/run_tau2_stage1.py is the authoritative execution path. "
                "It runs one selected task per subprocess and copies one raw JSON "
                "result per task for ingestion."
            ),
            "output_directory": str(raw_output_dir.relative_to(REPO_ROOT)),
            "native_tau2_output_prefix": "data/simulations/tau2_stage1_raw/",
            "max_concurrency": 1,
            "one_trial_per_task": True,
            "estimated_maximum_llm_calls": max_calls,
            "estimated_maximum_llm_calls_basis": (
                "Conservative planning bound: two LLM calls per expected action, "
                "with at least one action-equivalent per task; actual calls are logged."
            ),
            "runtime_and_cost_logging_requirements": [
                "Record start_time, end_time, runtime_seconds, status, reward, termination_reason, model, cost when available, and errors per task.",
                "Preserve verbose tau2 logs and original data/simulations results.json files.",
                "Copy one raw results JSON per task under data/processed/tau2_stage1_raw/ for ingestion.",
            ],
            "stop_conditions": [
                "Stop before execution unless --execute is explicitly supplied.",
                "Stop on any missing manifest task or task outside retail.",
                "Stop if a selected task was previously attempted or retained.",
                "Stop if cumulative observed cost reaches an operator-supplied --max-total-cost.",
                "Stop after the first subprocess failure unless --continue-on-error is supplied.",
            ],
        },
        "tasks": tasks,
        "composition": dict(Counter(task["selection_group"] for task in tasks)),
    }
    validate_manifest(manifest)
    return manifest


def write_report(manifest: dict[str, Any], path: Path) -> None:
    rows = [
        "| Task ID | Group | Actions | Reads | Writes | DB mutation | Previously attempted | Previously retained |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for task in manifest["tasks"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    task["task_id"],
                    task["selection_group"],
                    str(task["expected_action_count"]),
                    str(task["expected_read_action_count"]),
                    str(task["expected_write_action_count"]),
                    str(task["requires_db_mutation"]),
                    str(task["previously_attempted"]),
                    str(task["previously_retained"]),
                ]
            )
            + " |"
        )

    report = f"""# Tau2 Stage 1 Manifest

## Selection

This dry-run manifest selects exactly 12 unused retail tasks from `data/processed/tau2_additional_sampling_plan.json`. Selection is deterministic, records seed `{manifest["metadata"]["seed"]}`, excludes telecom, and does not use observed `y`, reward, or prior success/failure labels.

{chr(10).join(rows)}

## Composition

- `two_plus_writes`: {manifest["composition"].get("two_plus_writes", 0)}
- `no_write`: {manifest["composition"].get("no_write", 0)}
- `low_action_one_write`: {manifest["composition"].get("low_action_one_write", 0)}

## Illustrative Batch Tau2 Command

This command documents the selected task IDs and model/runtime settings in one
tau2 invocation. The executed Stage 1 path is the runner below, which launches
one task per subprocess and writes one raw JSON copy per task.

```bash
{manifest["run"]["illustrative_batch_tau2_command_text"]}
```

## Runner Commands

```bash
{manifest["run"]["runner_dry_run_command"]}
{manifest["run"]["runner_execute_command"]}
```

## Runtime And Cost Controls

- Output directory: `{manifest["run"]["output_directory"]}`
- Native tau2 output prefix: `{manifest["run"]["native_tau2_output_prefix"]}`
- One trial per task: `{manifest["run"]["one_trial_per_task"]}`
- Maximum concurrency: `{manifest["run"]["max_concurrency"]}`
- Estimated maximum LLM calls: `{manifest["run"]["estimated_maximum_llm_calls"]}`

## Stop Conditions

{chr(10).join(f"- {item}" for item in manifest["run"]["stop_conditions"])}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    input_plan: Path = DEFAULT_INPUT_PLAN,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    report_path: Path = DEFAULT_REPORT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    manifest = build_manifest(input_plan=input_plan)
    if write_outputs:
        write_json(manifest, output_json)
        write_report(manifest, report_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", type=Path, default=DEFAULT_INPUT_PLAN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_outputs(
        input_plan=args.input_plan,
        output_json=args.output_json,
        report_path=args.report_path,
        write_outputs=True,
    )
    task_ids = [task["task_id"] for task in manifest["tasks"]]
    print(f"wrote Stage 1 manifest with {len(task_ids)} tasks: {', '.join(task_ids)}")
    print(
        "illustrative batch command: "
        f"{manifest['run']['illustrative_batch_tau2_command_text']}"
    )


if __name__ == "__main__":
    main()
