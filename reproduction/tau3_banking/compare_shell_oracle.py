#!/usr/bin/env python3
"""Replay official banking shell calls against one live Modal sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.knowledge.modal_sandbox_manager import ModalSandboxManager

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_REFERENCE = HERE / "artifacts" / "banking_knowledge_results.json"
RECURSIVE_FILENAME_LINE = re.compile(r"^\./[^/:\r\n]+\.md(?::|$)", re.MULTILINE)
STRICT_SCOPE_COUNTS = {("subset", "recursive-filename-lines"): 59}


class OracleError(RuntimeError):
    """Raised when an oracle fixture or live replay is invalid."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleError(f"Expected a JSON object: {path}")
    return value


def selected_keys(config: dict[str, Any], mode: str) -> set[tuple[str, int]]:
    mode_config = config["modes"][mode]
    task_ids = mode_config["task_ids"]
    if task_ids == "all":
        task_ids = list(config["reward_vectors"])
    return {(task_id, trial) for task_id in task_ids for trial in mode_config["trials"]}


def extract_shell_fixtures(
    results: dict[str, Any], keys: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Pair assistant shell calls with their recorded ToolMessage by call id."""
    fixtures: list[dict[str, Any]] = []
    for simulation in results.get("simulations") or []:
        key = (simulation.get("task_id"), simulation.get("trial"))
        if key not in keys:
            continue
        pending: dict[str, str] = {}
        for message in simulation.get("messages") or []:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    if call.get("name") != "shell":
                        continue
                    call_id = call.get("id")
                    command = (call.get("arguments") or {}).get("command")
                    if not isinstance(call_id, str) or not isinstance(command, str):
                        raise OracleError(f"Invalid shell call in {key}")
                    pending[call_id] = command
            elif message.get("role") == "tool" and message.get("id") in pending:
                call_id = message["id"]
                content = message.get("content")
                if not isinstance(content, str):
                    raise OracleError(f"Shell result {call_id} in {key} is not text")
                fixtures.append(
                    {
                        "task_id": key[0],
                        "trial": key[1],
                        "call_id": call_id,
                        "command": pending.pop(call_id),
                        "expected": content,
                    }
                )
        if pending:
            raise OracleError(f"Unpaired shell calls in {key}: {sorted(pending)}")
    return fixtures


def unique_expected_by_command(
    fixtures: list[dict[str, Any]],
) -> dict[str, str]:
    """Deduplicate commands after rejecting every official output conflict."""
    outputs_by_command: dict[str, set[str]] = defaultdict(set)
    for fixture in fixtures:
        outputs_by_command[fixture["command"]].add(fixture["expected"])
    conflicts = {
        command: len(outputs)
        for command, outputs in outputs_by_command.items()
        if len(outputs) != 1
    }
    if conflicts:
        raise OracleError(f"Official fixture has command/output conflicts: {conflicts}")
    return {
        command: next(iter(outputs)) for command, outputs in outputs_by_command.items()
    }


def has_recursive_filename_line(expected: str) -> bool:
    """Return whether output contains a complete flat ``./file.md`` line."""
    return RECURSIVE_FILENAME_LINE.search(expected) is not None


def select_oracle_scope(
    expected_by_command: dict[str, str], *, mode: str, scope: str
) -> dict[str, str]:
    """Select a declared oracle scope and enforce its pinned cardinality."""
    if scope == "all":
        return expected_by_command
    key = (mode, scope)
    if key not in STRICT_SCOPE_COUNTS:
        raise OracleError(f"Oracle scope {scope!r} is not defined for mode {mode!r}")
    selected = {
        command: expected
        for command, expected in expected_by_command.items()
        if has_recursive_filename_line(expected)
    }
    expected_count = STRICT_SCOPE_COUNTS[key]
    if len(selected) != expected_count:
        raise OracleError(
            f"Oracle scope {scope!r} expected {expected_count} unique commands, "
            f"found {len(selected)}"
        )
    return selected


def render_shell_result(
    return_code: int, stdout: str, stderr: str, command: str
) -> str:
    """Mirror ``ShellMixin.shell`` exactly for an underlying command result."""
    if return_code != 0:
        if return_code == 1 and "grep" in command and not stderr:
            return "No matches found."
        if stderr:
            return f"Error (exit code {return_code}): {stderr}"
        return f"Command failed with exit code {return_code}"
    return stdout if stdout else "(no output)"


def compact_difference(expected: str, actual: str) -> dict[str, Any]:
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    first = 0
    while (
        first < len(expected_lines)
        and first < len(actual_lines)
        and expected_lines[first] == actual_lines[first]
    ):
        first += 1
    return {
        "first_different_line": first + 1,
        "expected_line": expected_lines[first] if first < len(expected_lines) else None,
        "actual_line": actual_lines[first] if first < len(actual_lines) else None,
        "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "actual_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        "expected_bytes": len(expected.encode()),
        "actual_bytes": len(actual.encode()),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.part-",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "subset", "full"), default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-details", type=int, default=50)
    parser.add_argument(
        "--scope",
        choices=("all", "recursive-filename-lines"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_details < 0:
            raise OracleError("--max-details must be non-negative")
        config = load_object(args.config)
        reference_path = args.reference.resolve()
        expected_digest = config["artifacts"]["trajectory"]["sha256"]
        actual_digest = digest_file(reference_path)
        if actual_digest != expected_digest:
            raise OracleError(
                f"Official trajectory digest mismatch: {actual_digest} != {expected_digest}"
            )
        fixtures = extract_shell_fixtures(
            load_object(reference_path), selected_keys(config, args.mode)
        )
        expected_by_command = unique_expected_by_command(fixtures)
        selected_by_command = select_oracle_scope(
            expected_by_command, mode=args.mode, scope=args.scope
        )
        selected_recorded_call_count = sum(
            fixture["command"] in selected_by_command for fixture in fixtures
        )

        report: dict[str, Any] = {
            "schema_version": 1,
            "mode": args.mode,
            "scope": args.scope,
            "reference_sha256": actual_digest,
            "recorded_call_count": len(fixtures),
            "unique_command_count": len(expected_by_command),
            "selected_recorded_call_count": selected_recorded_call_count,
            "selected_unique_command_count": len(selected_by_command),
            "executed": args.execute,
        }
        strict_count = STRICT_SCOPE_COUNTS.get((args.mode, args.scope))
        if strict_count is not None:
            report["expected_selected_unique_command_count"] = strict_count
        if not args.execute:
            print(json.dumps(report, indent=2, sort_keys=True))
            print("Dry run only. No Modal sandbox was created.")
            return 0

        knowledge_base = get_knowledge_base()
        documents = [
            {"id": doc.id, "title": doc.title, "content": doc.content}
            for doc in knowledge_base.documents.values()
        ]
        details = []
        exact = 0
        manager = ModalSandboxManager(sandbox_id=f"{args.mode}-shell-oracle")
        try:
            manager.export_documents(documents, file_format="md")
            manager_info = manager.get_sandbox_info()
            if (
                args.scope == "recursive-filename-lines"
                and not manager_info["order_manifest_applied"]
            ):
                raise OracleError(
                    "The strict recursive shell oracle requires the subset order manifest"
                )
            report["order_manifest_applied"] = manager_info["order_manifest_applied"]
            report["order_manifest_sha256"] = manager_info["order_manifest_sha256"]
            for command, expected in selected_by_command.items():
                return_code, stdout, stderr = manager.run_command(command)
                actual = render_shell_result(return_code, stdout, stderr, command)
                if actual == expected:
                    exact += 1
                elif len(details) < args.max_details:
                    details.append(
                        {
                            "command": command,
                            **compact_difference(expected, actual),
                        }
                    )
        finally:
            manager.cleanup()

        total = len(selected_by_command)
        report.update(
            {
                "exact_command_count": exact,
                "mismatch_command_count": total - exact,
                "exact_percent": (100.0 * exact / total) if total else 100.0,
                "mismatch_details": details,
                "mismatch_details_truncated": total - exact > len(details),
            }
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output:
            write_json_atomic(args.output, report)
        return 0 if exact == total else 1
    except (OracleError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
