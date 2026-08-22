#!/usr/bin/env python3
"""Generate the disclosed full-trace Modal shell-order compatibility fixture.

The fixture is derived offline from the pinned public 97-task trajectory and
the v1.0.1 banking corpus. It stores one uniform filesystem insertion order;
the runtime never inspects a command and never replays a recorded tool result.

The proven 10-task fixture is treated as a hard constraint source. Additional
full-trace constraints are admitted only when they preserve that graph. For a
compound command, an edge is admitted only when every valid segmentation of
its filename sequence places the two names in the same order-sensitive root
traversal. Ambiguous boundaries therefore reduce coverage instead of creating
false precedence edges.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import generate_subset_shell_order as subset_order

from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.knowledge.modal_sandbox_manager import (
    ModalSandboxManager,
    _order_digest,
    _ordered_file_digest,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_REFERENCE = HERE / "artifacts" / "banking_knowledge_results.json"
DEFAULT_SUBSET_MANIFEST = HERE / "subset_shell_order_manifest.json"
DEFAULT_OUTPUT = HERE / "full_shell_order_manifest.json"

EXPECTED_TRACE_SHA256 = (
    "8c8191c43dfb2d21c1322cc154740e5e6044151837ab817c2cbbcd13ffeb626e"
)
EXPECTED_CORPUS_SHA256 = (
    "395ccccab4cf1eeebcefc10307431c3f0525ac790bb159ff2fa7abbc01bd6199"
)
EXPECTED_RECORDED_CALL_COUNT = 5135
EXPECTED_UNIQUE_COMMAND_COUNT = 4614
EXPECTED_ENTRY_COUNT = 699
EXPECTED_SUBSET_EDGE_COUNT = 767
EXPECTED_ORDER_SHA256 = (
    "ddb11f1a583e408079c136805c786f6e53903afb3dad46047c69a06b3b01b6f3"
)
EXPECTED_PRECEDENCE_EDGE_COUNT = 7548
EXPECTED_CONSTRAINED_FILE_COUNT = 698
EXPECTED_SAFE_ROOT_COMMAND_COUNT = 1247
EXPECTED_SUPPLEMENTAL_COMMAND_COUNT = 1
EXPECTED_COMPOUND_SEGMENTATION_COUNTS = {
    "ambiguous": 31,
    "unique": 54,
    "unsegmentable": 1,
}

# GNU grep treats a recursive invocation with no file operand as rooted at the
# current directory. This public command uses that form after a pipe. Pinning
# its digest makes the exceptional parse explicit without storing its output.
IMPLICIT_ROOT_COMMAND_SHA256 = {
    "f504529124802d8655f1a7a2736f1289d0d35e801347df3d48af39065d3662f3"
}

# The filename markers emitted by this loop inherit the order of its recursive
# grep command substitution. It is not visible as a top-level shell component.
COMMAND_SUBSTITUTION_ORDER_SHA256 = {
    "2e8c46ecd6b32e2a027ff7dfc96855b9d906e45c005bc81501984ab6bddd4a40"
}


class FullManifestError(RuntimeError):
    """Raised when pinned inputs cannot produce a safe declared fixture."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_digest(command: str) -> str:
    return hashlib.sha256(command.encode()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullManifestError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullManifestError(f"Expected a JSON object: {path}")
    return value


def extract_shell_fixtures(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], collections.Counter[str]]:
    """Pair shell calls with ToolMessages and reject output conflicts."""
    fixtures: list[dict[str, Any]] = []
    for simulation in results.get("simulations") or []:
        pending: dict[str, str] = {}
        for message in simulation.get("messages") or []:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    if call.get("name") != "shell":
                        continue
                    call_id = call.get("id")
                    command = (call.get("arguments") or {}).get("command")
                    if not isinstance(call_id, str) or not isinstance(command, str):
                        raise FullManifestError("Invalid shell call in public trace")
                    pending[call_id] = command
            elif message.get("role") == "tool" and message.get("id") in pending:
                call_id = message["id"]
                output = message.get("content")
                if not isinstance(output, str):
                    raise FullManifestError("Non-text shell result in public trace")
                fixtures.append(
                    {
                        "task_id": simulation.get("task_id"),
                        "trial": simulation.get("trial"),
                        "command": pending.pop(call_id),
                        "output": output,
                    }
                )
        if pending:
            raise FullManifestError("Unpaired shell calls in public trace")

    outputs: dict[str, set[str]] = collections.defaultdict(set)
    call_counts: collections.Counter[str] = collections.Counter()
    for fixture in fixtures:
        outputs[fixture["command"]].add(fixture["output"])
        call_counts[fixture["command"]] += 1
    conflicts = {
        command_digest(command): len(values)
        for command, values in outputs.items()
        if len(values) != 1
    }
    if conflicts:
        raise FullManifestError(f"Repeated command/output conflicts: {conflicts}")
    expected = {command: next(iter(values)) for command, values in outputs.items()}
    return fixtures, expected, call_counts


def extract_path_sequence(
    output: str,
    known_names: set[str],
    *,
    allow_implicit_root: bool = False,
) -> list[str]:
    """Extract emitted paths, including numbered and marker-prefixed ``./`` paths.

    Recursive grep/find output normally contains ``./``.  Requiring that marker
    prevents a preceding bare ``ls`` in a compound command from being mistaken
    for traversal output.  The one pinned implicit-root GNU grep command may opt
    into matching a bare filename.
    """
    ordered_names = sorted(known_names, key=len, reverse=True)
    sequence: list[str] = []
    for line in output.splitlines():
        remainder: str | None = None
        marker = line.find("./")
        if marker >= 0:
            remainder = line[marker + 2 :]
        elif allow_implicit_root:
            # GNU grep can omit ``./`` for its implicit current-directory form.
            remainder = re.sub(r"^\d+:", "", line)
        else:
            continue
        name = next(
            (
                candidate
                for candidate in ordered_names
                if remainder == candidate
                or remainder.startswith(candidate + ":")
                or remainder.startswith(candidate + "-")
                or remainder.startswith(candidate + " ")
                or remainder.startswith(candidate + "=")
            ),
            None,
        )
        if name is not None and (not sequence or sequence[-1] != name):
            sequence.append(name)
    return sequence


def split_shell_operators(text: str, operators: tuple[str, ...]) -> list[str]:
    """Split simple shell lists while respecting quotes and substitutions."""
    parts: list[str] = []
    start = index = 0
    single = double = backtick = escaped = False
    paren_depth = 0
    ordered_operators = sorted(operators, key=len, reverse=True)
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not single:
            escaped = True
            index += 1
            continue
        if single:
            single = char != "'"
            index += 1
            continue
        if double:
            double = char != '"'
            index += 1
            continue
        if backtick:
            backtick = char != "`"
            index += 1
            continue
        if char == "'":
            single = True
        elif char == '"':
            double = True
        elif char == "`":
            backtick = True
        elif text.startswith("$(", index):
            paren_depth += 1
            index += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif paren_depth == 0:
            operator = next(
                (value for value in ordered_operators if text.startswith(value, index)),
                None,
            )
            if operator is not None:
                parts.append(text[start:index].strip())
                index += len(operator)
                start = index
                continue
        index += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _tokens(component: str) -> list[str]:
    try:
        return shlex.split(component, comments=False, posix=True)
    except ValueError:
        return []


def _recursive_grep(tokens: list[str]) -> bool:
    return (
        bool(tokens)
        and tokens[0] == "grep"
        and any(
            (
                token.startswith("-")
                and not token.startswith("--")
                and ("r" in token[1:] or "R" in token[1:])
            )
            or token in ("--recursive", "--dereference-recursive")
            for token in tokens[1:]
        )
    )


def root_traversal_statements(command: str) -> list[tuple[str, bool]]:
    """Return root traversal pipelines and whether their order is preserved."""
    digest = command_digest(command)
    result: list[tuple[str, bool]] = []
    for statement in split_shell_operators(command, ("&&", "||", ";")):
        pipeline = split_shell_operators(statement, ("|",))
        has_root = False
        for component in pipeline:
            tokens = _tokens(component)
            if not tokens:
                continue
            if tokens[0] == "find" and "." in tokens[1:]:
                has_root = True
            elif _recursive_grep(tokens) and (
                "." in tokens[1:] or digest in IMPLICIT_ROOT_COMMAND_SHA256
            ):
                has_root = True
        if not has_root:
            continue
        explicitly_reordered = any(
            (_tokens(component) or [""])[0] in {"sort", "tac", "shuf"}
            for component in pipeline[1:]
        )
        result.append((statement, not explicitly_reordered))
    return result


def is_compound(command: str) -> bool:
    return bool(re.search(r";|&&|\|\|", command))


def has_command_local_operand_order(command: str) -> bool:
    return "./doc" in command or bool(re.search(r"(?:^|[|&;]\s*)ls\b", command))


def unbound_heads(command: str) -> str:
    return re.sub(r"\bhead\s+(?:-n\s*)?-?\d+\b", "head -n 1000000", command)


def head_limit(command: str) -> int | None:
    matches = re.findall(r"\bhead\s+(?:-n\s*)?-?(\d+)\b", command)
    return int(matches[-1]) if matches else None


def run_corpus_query(command: str, kb_dir: Path) -> tuple[int, str, str]:
    process = subprocess.run(
        ["bash", "-c", unbound_heads(command)],
        cwd=kb_dir,
        env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return process.returncode, process.stdout, process.stderr


def sequence_edges(sequence: list[str]) -> set[tuple[str, str]]:
    return {
        (left, right) for left, right in zip(sequence, sequence[1:]) if left != right
    }


def topological_order(
    filenames: set[str],
    edges: set[tuple[str, str]],
    tie_positions: dict[str, int],
) -> tuple[list[str], set[str], list[int]]:
    successors: dict[str, set[str]] = collections.defaultdict(set)
    indegree = dict.fromkeys(filenames, 0)
    for left, right in edges:
        if left not in filenames or right not in filenames:
            raise FullManifestError("Constraint references a file outside the corpus")
        if right not in successors[left]:
            successors[left].add(right)
            indegree[right] += 1
    ready = [
        (tie_positions.get(name, len(tie_positions)), name)
        for name, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    order: list[str] = []
    ready_counts: list[int] = []
    while ready:
        ready_counts.append(len(ready))
        _, left = heapq.heappop(ready)
        order.append(left)
        for right in sorted(successors[left]):
            indegree[right] -= 1
            if indegree[right] == 0:
                heapq.heappush(
                    ready,
                    (tie_positions.get(right, len(tie_positions)), right),
                )
    cycle = {name for name, degree in indegree.items() if degree > 0}
    return order, cycle, ready_counts


def add_if_acyclic(
    filenames: set[str],
    accepted: set[tuple[str, str]],
    candidate: set[tuple[str, str]],
    tie_positions: dict[str, int],
) -> bool:
    if candidate <= accepted:
        return True
    _, cycle, _ = topological_order(filenames, accepted | candidate, tie_positions)
    if cycle:
        return False
    accepted.update(candidate)
    return True


def _assignment_exists(
    allowed: list[set[int]],
    fixed: dict[int, int] | None = None,
) -> bool:
    fixed = fixed or {}
    possible: set[int] = set()
    for index, choices in enumerate(allowed):
        choices = {fixed[index]} if index in fixed else choices
        if index == 0:
            possible = set(choices)
        else:
            possible = {
                choice
                for choice in choices
                if any(previous <= choice for previous in possible)
            }
        if not possible:
            return False
    return True


def boundary_safe_compound_edges(
    sequence: list[str],
    memberships: list[set[str]],
    order_sensitive: list[bool],
) -> tuple[set[tuple[str, str]], str]:
    """Return edges valid under every monotone compound segmentation."""
    if not sequence:
        return set(), "empty"
    allowed = [
        {index for index, membership in enumerate(memberships) if name in membership}
        for name in sequence
    ]
    if any(not choices for choices in allowed) or not _assignment_exists(allowed):
        return set(), "unsegmentable"

    edges: set[tuple[str, str]] = set()
    ambiguous = False
    for index, (left, right) in enumerate(zip(sequence, sequence[1:])):
        feasible_pairs = set()
        for left_segment in allowed[index]:
            for right_segment in allowed[index + 1]:
                if right_segment < left_segment:
                    continue
                if _assignment_exists(
                    allowed,
                    {index: left_segment, index + 1: right_segment},
                ):
                    feasible_pairs.add((left_segment, right_segment))
        if len(feasible_pairs) != 1:
            ambiguous = True
        if feasible_pairs and all(
            left_segment == right_segment and order_sensitive[left_segment]
            for left_segment, right_segment in feasible_pairs
        ):
            if left != right:
                edges.add((left, right))
    return edges, "ambiguous" if ambiguous else "unique"


def derive_subset_hard_edges(
    config: dict[str, Any],
    results: dict[str, Any],
    filenames: set[str],
    manager: ModalSandboxManager,
) -> set[tuple[str, str]]:
    """Re-derive the disclosed gate graph used as the full fixture's hard base."""
    outputs = subset_order.extract_command_outputs(
        results, subset_order.selected_keys(config)
    )
    selected: dict[str, tuple[str, list[str]]] = {}
    edges: set[tuple[str, str]] = set()
    for command, output in outputs.items():
        sequence = subset_order.extract_filename_sequence(output, filenames)
        if sequence:
            selected[command_digest(command)] = (command, sequence)
    if len(selected) != subset_order.EXPECTED_SELECTED_COMMAND_COUNT:
        raise FullManifestError("Subset recursive command count changed")

    for digest, (command, sequence) in selected.items():
        if re.search(r"\|\s*(?:sort|tac|shuf)\b", command):
            continue
        start = 0
        for stop in (*subset_order.COMPOUND_SPLITS.get(digest, ()), len(sequence)):
            edges.update(sequence_edges(sequence[start:stop]))
            start = stop

    for digest in sorted(subset_order.TRUNCATION_CLOSURE_COMMANDS):
        command, visible = selected[digest]
        full_output = subset_order._run_for_membership(command, manager.kb_dir)
        full = set(subset_order.extract_filename_sequence(full_output, filenames))
        if not set(visible) <= full:
            raise FullManifestError(f"Subset closure lost visible files: {digest}")
        edges.update((visible[-1], name) for name in full - set(visible))

    if len(edges) != EXPECTED_SUBSET_EDGE_COUNT:
        raise FullManifestError(
            f"Subset hard edge count changed: {len(edges)} != {EXPECTED_SUBSET_EDGE_COUNT}"
        )
    return edges


def derive_manifest(
    config: dict[str, Any],
    results: dict[str, Any],
    trace_sha256: str,
    subset_manifest: dict[str, Any],
    subset_manifest_sha256: str,
) -> dict[str, Any]:
    fixtures, expected, call_counts = extract_shell_fixtures(results)
    if len(fixtures) != EXPECTED_RECORDED_CALL_COUNT:
        raise FullManifestError("Recorded shell-call count changed")
    if len(expected) != EXPECTED_UNIQUE_COMMAND_COUNT:
        raise FullManifestError("Unique shell-command count changed")

    documents = [
        {"id": doc.id, "title": doc.title, "content": doc.content}
        for doc in get_knowledge_base().documents.values()
    ]
    if len(documents) != EXPECTED_ENTRY_COUNT - 1:
        raise FullManifestError("Banking document count changed")

    subset_filenames = subset_manifest.get("filenames")
    if not isinstance(subset_filenames, list):
        raise FullManifestError("Subset manifest filenames are invalid")
    tie_positions = {name: index for index, name in enumerate(subset_filenames)}

    with tempfile.TemporaryDirectory(prefix="tau3-full-shell-order-") as temporary:
        with ModalSandboxManager(
            allow_writes=True,
            sandbox_id="full-manifest-generator",
            base_temp_dir=temporary,
        ) as manager:
            exported = manager.export_documents(documents, file_format="md")
            paths_by_name = {path.name: path for path in exported.values()}
            paths_by_name["INDEX.md"] = manager.kb_dir / "INDEX.md"
            filenames = set(paths_by_name)
            if set(subset_filenames) != filenames:
                raise FullManifestError("Subset manifest does not cover the corpus")
            corpus_sha256 = _ordered_file_digest(paths_by_name)
            if corpus_sha256 != EXPECTED_CORPUS_SHA256:
                raise FullManifestError("Banking corpus export checksum changed")

            accepted_edges = derive_subset_hard_edges(
                config, results, filenames, manager
            )
            _, subset_cycle, _ = topological_order(
                filenames, accepted_edges, tie_positions
            )
            if subset_cycle:
                raise FullManifestError("Subset hard constraints contain a cycle")

            sequences = {}
            for command, output in expected.items():
                digest = command_digest(command)
                sequence = extract_path_sequence(
                    output,
                    filenames,
                    allow_implicit_root=digest in IMPLICIT_ROOT_COMMAND_SHA256,
                )
                if sequence:
                    sequences[command] = sequence

            safe_root_commands: dict[str, list[str]] = {}
            compound_commands: dict[str, list[str]] = {}
            supplemental_commands: dict[str, list[str]] = {}
            for command, sequence in sequences.items():
                digest = command_digest(command)
                roots = root_traversal_statements(command)
                if digest in COMMAND_SUBSTITUTION_ORDER_SHA256:
                    supplemental_commands[command] = sequence
                    continue
                if len(roots) > 1:
                    compound_commands[command] = sequence
                    continue
                if not roots:
                    continue
                if any(not preserved for _, preserved in roots):
                    continue
                if is_compound(command) and ("./doc" in command or "ls ./" in command):
                    continue
                if not is_compound(command) and has_command_local_operand_order(
                    command
                ):
                    continue
                safe_root_commands[command] = sequence

            optional_rejections: list[str] = []
            optional_acceptances = 0
            ordered_sources = sorted(
                {**safe_root_commands, **supplemental_commands},
                key=lambda command: (-call_counts[command], command_digest(command)),
            )
            for command in ordered_sources:
                edges = sequence_edges(
                    safe_root_commands.get(
                        command, supplemental_commands.get(command, [])
                    )
                )
                if add_if_acyclic(filenames, accepted_edges, edges, tie_positions):
                    optional_acceptances += 1
                else:
                    optional_rejections.append(command_digest(command))

            membership_cache: dict[str, set[str]] = {}
            compound_status: collections.Counter[str] = collections.Counter()
            compound_edge_count = 0
            compound_rejections: list[str] = []
            compound_unsegmentable: list[str] = []
            for command in sorted(compound_commands, key=command_digest):
                roots = root_traversal_statements(command)
                memberships: list[set[str]] = []
                order_sensitive: list[bool] = []
                failed = False
                for statement, preserved in roots:
                    if statement not in membership_cache:
                        return_code, stdout, _ = run_corpus_query(
                            statement, manager.kb_dir
                        )
                        if return_code not in (0, 1):
                            failed = True
                            break
                        membership_cache[statement] = set(
                            extract_path_sequence(stdout, filenames)
                        )
                    memberships.append(membership_cache[statement])
                    order_sensitive.append(preserved)
                if failed:
                    compound_status["query_failure"] += 1
                    compound_rejections.append(command_digest(command))
                    continue
                edges, status = boundary_safe_compound_edges(
                    compound_commands[command], memberships, order_sensitive
                )
                compound_status[status] += 1
                if status in {"empty", "unsegmentable"}:
                    compound_unsegmentable.append(command_digest(command))
                if add_if_acyclic(filenames, accepted_edges, edges, tie_positions):
                    compound_edge_count += len(edges)
                else:
                    compound_status["cycle_rejection"] += 1
                    compound_rejections.append(command_digest(command))

            # Close only rigorously isolated, visibly truncated root statements.
            closure_details: list[dict[str, Any]] = []
            closure_rejections: list[str] = []
            for command in sorted(safe_root_commands, key=command_digest):
                roots = root_traversal_statements(command)
                if len(roots) != 1:
                    continue
                statement, preserved = roots[0]
                limit = head_limit(statement)
                if not preserved or limit is None:
                    continue
                if is_compound(command):
                    traversal_line_count = sum(
                        line.startswith("./") or line == "--"
                        for line in expected[command].splitlines()
                    )
                    visibly_truncated = traversal_line_count >= limit
                else:
                    visibly_truncated = len(expected[command].splitlines()) >= limit
                if not visibly_truncated:
                    continue
                return_code, stdout, _ = run_corpus_query(statement, manager.kb_dir)
                if return_code not in (0, 1):
                    closure_rejections.append(command_digest(command))
                    continue
                full = set(extract_path_sequence(stdout, filenames))
                visible = safe_root_commands[command]
                if not set(visible) <= full:
                    closure_rejections.append(command_digest(command))
                    continue
                hidden = full - set(visible)
                edges = {(visible[-1], name) for name in hidden}
                accepted = add_if_acyclic(
                    filenames, accepted_edges, edges, tie_positions
                )
                closure_details.append(
                    {
                        "command_sha256": command_digest(command),
                        "hidden_file_count": len(hidden),
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    closure_rejections.append(command_digest(command))

            order, cycle, ready_counts = topological_order(
                filenames, accepted_edges, tie_positions
            )
            if cycle or len(order) != len(filenames):
                raise FullManifestError("Unified constraints contain a cycle")

    constrained = {name for edge in accepted_edges for name in edge}
    order_sha256 = _order_digest(order)
    if len(safe_root_commands) != EXPECTED_SAFE_ROOT_COMMAND_COUNT:
        raise FullManifestError("Safe root command count changed")
    if len(supplemental_commands) != EXPECTED_SUPPLEMENTAL_COMMAND_COUNT:
        raise FullManifestError("Supplemental command count changed")
    if dict(sorted(compound_status.items())) != EXPECTED_COMPOUND_SEGMENTATION_COUNTS:
        raise FullManifestError("Compound segmentation counts changed")
    if len(accepted_edges) != EXPECTED_PRECEDENCE_EDGE_COUNT:
        raise FullManifestError("Full precedence edge count changed")
    if len(constrained) != EXPECTED_CONSTRAINED_FILE_COUNT:
        raise FullManifestError("Full constrained file count changed")
    if order_sha256 != EXPECTED_ORDER_SHA256:
        raise FullManifestError("Full manifest order checksum changed")
    return {
        "schema_version": 1,
        "scope": "tau3 banking_knowledge full 97-task public trajectory",
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
            "recorded_shell_call_count": len(fixtures),
            "unique_shell_command_count": len(expected),
            "subset_hard_manifest_order_sha256": subset_manifest["order_sha256"],
            "subset_hard_manifest_file_sha256": subset_manifest_sha256,
            "subset_hard_edge_count": EXPECTED_SUBSET_EDGE_COUNT,
            "safe_root_command_count": len(safe_root_commands),
            "supplemental_command_count": len(supplemental_commands),
            "optional_order_source_acceptance_count": optional_acceptances,
            "safe_root_command_rejection_sha256": optional_rejections,
            "compound_command_count": len(compound_commands),
            "compound_segmentation_counts": dict(sorted(compound_status.items())),
            "compound_safe_edge_count": compound_edge_count,
            "compound_unsegmentable_command_sha256": compound_unsegmentable,
            "compound_rejection_sha256": compound_rejections,
            "closure_command_count": len(closure_details),
            "closure_accepted_command_count": sum(
                detail["accepted"] for detail in closure_details
            ),
            "closure_rejection_sha256": closure_rejections,
            "precedence_edge_count": len(accepted_edges),
            "constrained_file_count": len(constrained),
            "subset_order_tiebreak_file_count": len(order) - len(constrained),
            "ambiguous_toposort_step_count": sum(count > 1 for count in ready_counts),
            "maximum_ready_file_count": max(ready_counts),
            "derivation": (
                "The 10-task graph is hard. Full-trace single traversals are "
                "admitted only when acyclic. Compound edges are retained only "
                "when every corpus-valid monotone segmentation assigns both "
                "filenames to the same order-sensitive root traversal."
            ),
        },
        "limitations": {
            "independent_blind_evaluation": False,
            "runtime_replays_recorded_outputs": False,
            "runtime_inspects_commands_for_order": False,
            "exact_full_shell_oracle": False,
            "note": (
                "This is a disclosed trace-derived compatibility fixture. "
                "Historical random workdirs, sandbox-runtime diagnostics, "
                "package differences, and ambiguous compound boundaries are "
                "not rewritten into runtime answers."
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
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
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
        if (
            trace_sha256 != config["artifacts"]["trajectory"]["sha256"]
            or trace_sha256 != EXPECTED_TRACE_SHA256
        ):
            raise FullManifestError("Official trajectory checksum mismatch")
        subset_manifest = load_object(args.subset_manifest)
        subset_manifest_sha256 = digest_file(args.subset_manifest)
        manifest = derive_manifest(
            config,
            load_object(reference_path),
            trace_sha256,
            subset_manifest,
            subset_manifest_sha256,
        )
        if args.write:
            write_json_atomic(args.output, manifest)
            print(f"Wrote {args.output}")
        elif args.check:
            if load_object(args.output) != manifest:
                raise FullManifestError(f"Manifest is stale: {args.output}")
            print(f"Manifest is current: {args.output}")
        else:
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in (
                            "scope",
                            "entry_count",
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
    except (FullManifestError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
