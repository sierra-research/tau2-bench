#!/usr/bin/env python3
"""Generate the disclosed 10-task Modal shell-order compatibility fixture.

The fixture is derived from the pinned public trajectory and the v1.0.1 banking
corpus.  It records one uniform filesystem insertion order; the runtime never
inspects a command and never replays a recorded command result.

This is deliberately scoped to the 10-task paid gate.  It is not evidence that
the remaining 87 tasks have historical shell traversal parity.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.knowledge.modal_sandbox_manager import (
    ModalSandboxManager,
    _order_digest,
    _ordered_file_digest,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_REFERENCE = HERE / "artifacts" / "banking_knowledge_results.json"
DEFAULT_OUTPUT = HERE / "subset_shell_order_manifest.json"

EXPECTED_TRACE_SHA256 = (
    "8c8191c43dfb2d21c1322cc154740e5e6044151837ab817c2cbbcd13ffeb626e"
)
EXPECTED_ORDER_SHA256 = (
    "898b4038585ab4bd10be0ed57f396c4ae5a2b46d8e339e39cc5c8d8219e1d32f"
)
EXPECTED_ENTRY_COUNT = 699
EXPECTED_SELECTED_COMMAND_COUNT = 59
EXPECTED_CONSTRAINED_FILE_COUNT = 451
EXPECTED_LEXICAL_TIEBREAK_FILE_COUNT = 248
EXPECTED_EDGE_COUNT = 767

# One compound command emits two independent traversals without a separator.
# Its public command digest makes the otherwise ambiguous output boundary
# explicit without embedding either recorded output.
COMPOUND_SPLITS = {
    "ed96802d314469467a53d8afda23424c05ff273babfb97211d0054f7ceddbf10": (3,)
}

# These six bounded pipelines had additional matching files hidden by `head`.
# The generator derives those hidden filename sets from the corpus by raising
# the limit; no expected output text is stored or returned at runtime.
TRUNCATION_CLOSURE_COMMANDS = {
    "a8c6cd29e3a5e27a6ed10748761a8b6043c037fb45f750278ff873ee2d8638ed",
    "a70a9e953c73f6d4176760697c8a094cf4827a20cc0a82c43d74b02470ff11ba",
    "668ce6b75ff978f3ab53ae168975bf2c060067b0b670e13cfe35617e6b61df48",
    "0f3880a6241fc35aae1277b1c8feea8704800b172c1097ba4680355928a47b27",
    "5092f3f94704202f7540e7383433a61cedba7279db05e1a83c36a66fe69bf6b4",
    "ebf72f92f818f26efa954f82180149b3894cec0b5ce01fd0986d3b74f6625056",
}


class ManifestError(RuntimeError):
    """Raised when the pinned inputs cannot produce the declared fixture."""


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
        raise ManifestError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Expected a JSON object: {path}")
    return value


def selected_keys(config: dict[str, Any]) -> set[tuple[str, int]]:
    subset = config["modes"]["subset"]
    return {
        (task_id, trial) for task_id in subset["task_ids"] for trial in subset["trials"]
    }


def extract_command_outputs(
    results: dict[str, Any], keys: set[tuple[str, int]]
) -> dict[str, str]:
    """Pair shell calls with ToolMessages and require stable repeated outputs."""
    outputs_by_command: dict[str, set[str]] = defaultdict(set)
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
                        raise ManifestError(f"Invalid shell call in {key}")
                    pending[call_id] = command
            elif message.get("role") == "tool" and message.get("id") in pending:
                call_id = message["id"]
                content = message.get("content")
                if not isinstance(content, str):
                    raise ManifestError(f"Non-text shell result in {key}")
                outputs_by_command[pending.pop(call_id)].add(content)
        if pending:
            raise ManifestError(f"Unpaired shell calls in {key}: {sorted(pending)}")

    conflicts = {
        command: len(outputs)
        for command, outputs in outputs_by_command.items()
        if len(outputs) != 1
    }
    if conflicts:
        raise ManifestError(f"Repeated command/output conflicts: {conflicts}")
    return {
        command: next(iter(outputs)) for command, outputs in outputs_by_command.items()
    }


def extract_filename_sequence(output: str, known_names: set[str]) -> list[str]:
    """Extract flat recursive paths, including grep context-line prefixes."""
    ordered_names = sorted(known_names, key=len, reverse=True)
    sequence: list[str] = []
    for line in output.splitlines():
        if not line.startswith("./"):
            continue
        remainder = line[2:]
        name = next(
            (
                candidate
                for candidate in ordered_names
                if remainder == candidate
                or remainder.startswith(f"{candidate}:")
                or remainder.startswith(f"{candidate}-")
            ),
            None,
        )
        if name is not None and (not sequence or sequence[-1] != name):
            sequence.append(name)
    return sequence


def add_sequence_edges(successors: dict[str, set[str]], sequence: list[str]) -> None:
    for left, right in zip(sequence, sequence[1:]):
        if left != right:
            successors[left].add(right)


def lexicographic_toposort(
    filenames: set[str], successors: dict[str, set[str]]
) -> list[str]:
    indegree = {name: 0 for name in filenames}
    for left, rights in successors.items():
        if left not in filenames or not rights <= filenames:
            raise ManifestError("Constraint references a file outside the corpus")
        for right in rights:
            indegree[right] += 1
    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for right in sorted(successors.get(name, ())):
            indegree[right] -= 1
            if indegree[right] == 0:
                heapq.heappush(ready, right)
    if len(order) != len(filenames):
        raise ManifestError(
            f"Official subset ordering constraints contain a cycle "
            f"({len(filenames) - len(order)} files remain)"
        )
    return order


def _unbounded_command(command: str) -> str:
    return re.sub(r"\bhead\s+(?:-n\s*)?-?\d+", "head -n 1000000", command)


def _run_for_membership(command: str, kb_dir: Path) -> str:
    process = subprocess.run(
        ["bash", "-c", _unbounded_command(command)],
        cwd=kb_dir,
        env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode not in (0, 1):
        raise ManifestError(
            f"Headless corpus query failed ({process.returncode}): "
            f"{process.stderr[:200]}"
        )
    return process.stdout


def derive_manifest(
    config: dict[str, Any], results: dict[str, Any], trace_sha256: str
) -> dict[str, Any]:
    """Derive a complete uniform order from public subset constraints."""
    knowledge_base = get_knowledge_base()
    documents = [
        {"id": doc.id, "title": doc.title, "content": doc.content}
        for doc in knowledge_base.documents.values()
    ]
    if len(documents) != EXPECTED_ENTRY_COUNT - 1:
        raise ManifestError(
            f"Expected {EXPECTED_ENTRY_COUNT - 1} documents, found {len(documents)}"
        )

    command_outputs = extract_command_outputs(results, selected_keys(config))
    successors: dict[str, set[str]] = defaultdict(set)
    selected: dict[str, tuple[str, list[str]]] = {}

    with tempfile.TemporaryDirectory(prefix="tau3-shell-order-") as temporary:
        with ModalSandboxManager(
            allow_writes=True,
            sandbox_id="manifest-generator",
            base_temp_dir=temporary,
        ) as manager:
            exported = manager.export_documents(documents, file_format="md")
            paths_by_name = {path.name: path for path in exported.values()}
            paths_by_name["INDEX.md"] = manager.kb_dir / "INDEX.md"
            filenames = set(paths_by_name)

            for command, output in command_outputs.items():
                sequence = extract_filename_sequence(output, filenames)
                if sequence:
                    selected[hashlib.sha256(command.encode()).hexdigest()] = (
                        command,
                        sequence,
                    )

            if len(selected) != EXPECTED_SELECTED_COMMAND_COUNT:
                raise ManifestError(
                    f"Expected {EXPECTED_SELECTED_COMMAND_COUNT} recursive commands, "
                    f"found {len(selected)}"
                )

            for command_sha256, (command, sequence) in selected.items():
                if re.search(r"\|\s*(?:sort|tac|shuf)\b", command):
                    continue
                split_points = COMPOUND_SPLITS.get(command_sha256, ())
                start = 0
                for stop in (*split_points, len(sequence)):
                    add_sequence_edges(successors, sequence[start:stop])
                    start = stop

            if not set(COMPOUND_SPLITS) <= set(selected):
                raise ManifestError("Pinned compound traversal is missing")
            if not TRUNCATION_CLOSURE_COMMANDS <= set(selected):
                raise ManifestError("Pinned truncation-closure traversal is missing")

            for command_sha256 in sorted(TRUNCATION_CLOSURE_COMMANDS):
                command, visible = selected[command_sha256]
                full_output = _run_for_membership(command, manager.kb_dir)
                full = set(extract_filename_sequence(full_output, filenames))
                if not set(visible) <= full:
                    raise ManifestError(
                        f"Corpus query lost visible files for {command_sha256}"
                    )
                for hidden_name in sorted(full - set(visible)):
                    successors[visible[-1]].add(hidden_name)

            order = lexicographic_toposort(filenames, successors)
            constrained = set(successors)
            for rights in successors.values():
                constrained.update(rights)
            edge_count = sum(len(rights) for rights in successors.values())
            corpus_sha256 = _ordered_file_digest(paths_by_name)

    order_sha256 = _order_digest(order)
    observed = {
        "entry_count": len(order),
        "selected_command_count": len(selected),
        "constrained_file_count": len(constrained),
        "lexical_tiebreak_file_count": len(order) - len(constrained),
        "edge_count": edge_count,
        "order_sha256": order_sha256,
    }
    expected = {
        "entry_count": EXPECTED_ENTRY_COUNT,
        "selected_command_count": EXPECTED_SELECTED_COMMAND_COUNT,
        "constrained_file_count": EXPECTED_CONSTRAINED_FILE_COUNT,
        "lexical_tiebreak_file_count": EXPECTED_LEXICAL_TIEBREAK_FILE_COUNT,
        "edge_count": EXPECTED_EDGE_COUNT,
        "order_sha256": EXPECTED_ORDER_SHA256,
    }
    if observed != expected:
        raise ManifestError(f"Derived fixture changed: {observed} != {expected}")

    subset = config["modes"]["subset"]
    return {
        "schema_version": 1,
        "scope": "tau3 banking_knowledge 10-task gate only",
        "entry_count": len(order),
        "document_count": len(order) - 1,
        "corpus_export_sha256": corpus_sha256,
        "order_sha256": order_sha256,
        "provenance": {
            "official_trace_sha256": trace_sha256,
            "official_git_commit": config["benchmark"]["git_commit"],
            "documents_tree_git_object": config["benchmark"]["dataset_git_objects"][
                "documents_tree"
            ],
            "task_ids": subset["task_ids"],
            "trials": subset["trials"],
            "selected_recursive_command_count": len(selected),
            "precedence_edge_count": edge_count,
            "constrained_file_count": len(constrained),
            "lexical_tiebreak_file_count": len(order) - len(constrained),
            "truncation_closure_command_sha256": sorted(TRUNCATION_CLOSURE_COMMANDS),
            "derivation": (
                "Filename precedence comes from public subset shell outputs; "
                "hidden matches for six head-truncated pipelines come from the "
                "pinned corpus; remaining files use a lexical topological tie-break."
            ),
        },
        "limitations": {
            "independent_blind_evaluation": False,
            "full_97_task_shell_parity": False,
            "runtime_replays_recorded_outputs": False,
            "note": (
                "This disclosed trace-derived compatibility fixture is sufficient "
                "only for the 10-task reproduction gate. The 248 lexical tie-break "
                "files have no subset-derived rank guarantee."
            ),
        },
        "filenames": order,
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
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_object(args.config)
        reference_path = args.reference.resolve()
        trace_sha256 = digest_file(reference_path)
        configured_sha256 = config["artifacts"]["trajectory"]["sha256"]
        if trace_sha256 != configured_sha256 or trace_sha256 != EXPECTED_TRACE_SHA256:
            raise ManifestError("Official trajectory checksum mismatch")
        manifest = derive_manifest(
            config, load_object(reference_path), trace_sha256=trace_sha256
        )

        if args.write:
            write_json_atomic(args.output, manifest)
            print(f"Wrote {args.output}")
        elif args.check:
            existing = load_object(args.output)
            if existing != manifest:
                raise ManifestError(f"Manifest is stale: {args.output}")
            print(f"Manifest is current: {args.output}")
        else:
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in (
                            "scope",
                            "entry_count",
                            "document_count",
                            "corpus_export_sha256",
                            "order_sha256",
                            "provenance",
                            "limitations",
                        )
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print("Dry run only. Pass --write to update the manifest.")
        return 0
    except (ManifestError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
