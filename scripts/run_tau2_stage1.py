#!/usr/bin/env python3
"""Run the tau2 Stage 1 retail manifest, dry-run by default."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
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
    STAGE1_SEED,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_JSON = REPO_ROOT / "data/processed/tau2_stage1_run_status.json"
SIMULATION_PREFIX = "tau2_stage1_raw"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def display_path(path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        return str(path.relative_to(REPO_ROOT))
    return str(path)


def task_raw_path(raw_dir: Path, task_id: str) -> Path:
    return raw_dir / f"task_{task_id}.json"


def task_save_to(task_id: str) -> str:
    return f"{SIMULATION_PREFIX}/task_{task_id}"


def native_results_path(task_id: str) -> Path:
    return REPO_ROOT / "data/simulations" / task_save_to(task_id) / "results.json"


def task_completed(raw_dir: Path, task_id: str) -> bool:
    path = task_raw_path(raw_dir, task_id)
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except json.JSONDecodeError:
        return False
    return bool(data.get("simulations"))


def command_for_task(task: dict[str, Any], *, timeout: float | None) -> list[str]:
    command = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        "retail",
        "--agent",
        "llm_agent",
        "--user",
        "user_simulator",
        "--agent-llm",
        "gpt-4o-mini",
        "--user-llm",
        "gpt-4o-mini",
        "--num-trials",
        "1",
        "--task-ids",
        str(task["task_id"]),
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
        task_save_to(str(task["task_id"])),
    ]
    if timeout is not None:
        command.extend(["--timeout", str(timeout)])
    return command


def simulation_summary(raw_path: Path) -> dict[str, Any]:
    data = load_json(raw_path)
    simulations = data.get("simulations") or []
    if not simulations:
        return {}
    simulation = simulations[0]
    reward_info = simulation.get("reward_info") or {}
    cost = simulation.get("cost") or simulation.get("agent_cost") or reward_info.get("cost")
    return {
        "reward": reward_info.get("reward"),
        "termination_reason": simulation.get("termination_reason"),
        "model": {
            "agent_model": "gpt-4o-mini",
            "user_model": "gpt-4o-mini",
        },
        "cost": cost,
    }


def initial_status(manifest: dict[str, Any], *, execute: bool, raw_dir: Path) -> dict[str, Any]:
    return {
        "metadata": {
            "stage": "tau2_stage1",
            "dry_run": not execute,
            "execute": execute,
            "started_at": utc_now(),
            "ended_at": None,
            "manifest_task_ids": [task["task_id"] for task in manifest["tasks"]],
            "raw_output_dir": display_path(raw_dir),
        },
        "tasks": {},
    }


def select_tasks(
    manifest: dict[str, Any], *, task_id: str | None = None
) -> list[dict[str, Any]]:
    tasks = manifest["tasks"]
    if task_id is None:
        return tasks

    task_by_id = {str(task["task_id"]): task for task in tasks}
    if task_id not in task_by_id:
        valid_ids = ", ".join(str(task["task_id"]) for task in tasks)
        raise ValueError(f"Task ID {task_id} is not present in the manifest: {valid_ids}")
    return [task_by_id[task_id]]


def run_task(
    task: dict[str, Any],
    *,
    raw_dir: Path,
    timeout: float | None,
    execute: bool,
    subprocess_run: Any = subprocess.run,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    command = command_for_task(task, timeout=timeout)
    started = utc_now()
    monotonic_start = time.monotonic()
    raw_path = task_raw_path(raw_dir, task_id)
    native_path = native_results_path(task_id)
    record: dict[str, Any] = {
        "task_id": task_id,
        "status": "dry_run",
        "start_time": started,
        "end_time": None,
        "runtime_seconds": 0.0,
        "command": command,
        "raw_result_path": display_path(raw_path),
        "native_result_path": display_path(native_path),
        "reward": None,
        "termination_reason": None,
        "model": {"agent_model": "gpt-4o-mini", "user_model": "gpt-4o-mini"},
        "cost": None,
        "errors": [],
    }
    if task_completed(raw_dir, task_id):
        record.update({"status": "skipped_completed", **simulation_summary(raw_path)})
        record["end_time"] = utc_now()
        return record
    if not execute:
        record["end_time"] = utc_now()
        return record

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess_run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - exercised through mock tests
        record["status"] = "error"
        record["errors"].append(repr(exc))
    else:
        record["returncode"] = result.returncode
        record["stdout_tail"] = (result.stdout or "")[-4000:]
        record["stderr_tail"] = (result.stderr or "")[-4000:]
        if result.returncode != 0:
            record["status"] = "error"
            record["errors"].append(f"subprocess_returncode:{result.returncode}")
        elif not native_path.exists():
            record["status"] = "error"
            record["errors"].append(f"missing_native_results:{native_path}")
        else:
            shutil.copy2(native_path, raw_path)
            record.update({"status": "completed", **simulation_summary(raw_path)})

    record["end_time"] = utc_now()
    record["runtime_seconds"] = round(time.monotonic() - monotonic_start, 6)
    return record


def run_stage1(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    status_path: Path = DEFAULT_STATUS_JSON,
    raw_dir: Path = DEFAULT_RAW_DIR,
    task_id: str | None = None,
    execute: bool = False,
    continue_on_error: bool = False,
    max_total_cost: float | None = None,
    timeout: float | None = None,
    subprocess_run: Any = subprocess.run,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    validate_manifest(manifest)
    tasks = select_tasks(manifest, task_id=task_id)
    status = initial_status(manifest, execute=execute, raw_dir=raw_dir)
    status["metadata"]["selected_task_ids"] = [task["task_id"] for task in tasks]
    status["metadata"]["selected_task_id"] = task_id
    total_cost = 0.0
    raw_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_id = str(task["task_id"])
        task_status = run_task(
            task,
            raw_dir=raw_dir,
            timeout=timeout,
            execute=execute,
            subprocess_run=subprocess_run,
        )
        status["tasks"][task_id] = task_status
        cost = task_status.get("cost")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
        status["metadata"]["total_observed_cost"] = total_cost
        write_json(status, status_path)
        if max_total_cost is not None and total_cost >= max_total_cost:
            status["metadata"]["stop_reason"] = "max_total_cost_reached"
            break
        if task_status["status"] == "error" and not continue_on_error:
            status["metadata"]["stop_reason"] = f"task_{task_id}_error"
            break

    status["metadata"]["ended_at"] = utc_now()
    status["metadata"]["completed_count"] = sum(
        task["status"] in {"completed", "skipped_completed"}
        for task in status["tasks"].values()
    )
    status["metadata"]["dry_run_count"] = sum(
        task["status"] == "dry_run" for task in status["tasks"].values()
    )
    status["metadata"].setdefault("stop_reason", "finished")
    write_json(status, status_path)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--max-total-cost", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_stage1(
        manifest_path=args.manifest_path,
        status_path=args.status_path,
        raw_dir=args.raw_dir,
        task_id=args.task_id,
        execute=args.execute,
        continue_on_error=args.continue_on_error,
        max_total_cost=args.max_total_cost,
        timeout=args.timeout,
    )
    mode = "execute" if args.execute else "dry-run"
    print(f"Stage 1 {mode} wrote {args.status_path}")
    print(f"completed: {status['metadata']['completed_count']}; dry_run: {status['metadata']['dry_run_count']}")


if __name__ == "__main__":
    main()
