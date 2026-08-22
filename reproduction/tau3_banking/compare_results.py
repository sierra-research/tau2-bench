#!/usr/bin/env python3
"""Compare a banking reproduction with the immutable official result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run import (
    DEFAULT_MODAL_APP,
    DEFAULT_MODAL_SANDBOX_TIMEOUT,
    FULL_GATE_SCHEMA_VERSION,
    KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
    RunGuardError,
    build_command,
    endpoint_inventory_mismatches,
    expected_manifest_environment,
    expected_prompt_hashes,
)
from state_fingerprint import (
    StateFingerprintError,
    capture_reproduction_state,
    digest_checkpoint_artifact,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_REFERENCE_RESULTS = HERE / "artifacts" / "banking_knowledge_results.json"
DEFAULT_GATE = HERE / ".state" / "subset_score_parity.json"
REWARD_COMPONENT_KINDS = (
    "db_component",
    "env_assertions",
    "action_checks",
    "nl_assertions",
    "communicate_checks",
    "reward_basis",
    "reward_breakdown",
)
SCORING_KINDS = {
    "duplicate_simulation",
    "missing_simulation",
    "unexpected_simulation",
    "reward",
    "seed",
    "termination_reason",
    *REWARD_COMPONENT_KINDS,
}
TIMING_FOOTER = re.compile(r"(?:\r?\n){0,2}\[Timing:[^\r\n]*\][ \t]*\Z")
KNOWN_DENSE_DRIFT_MISMATCH_KIND = "tool_output_dense_known_drift"


class ComparisonError(RuntimeError):
    """Raised when inputs cannot be compared safely."""


def digest_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_results_path(path: Path) -> Path:
    """Accept either a monolithic results file or its containing directory."""
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise ComparisonError(f"Results file does not exist: {path}")
    return path.resolve()


def load_json(path: Path) -> Any:
    """Load JSON with a concise, path-aware error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Cannot read JSON {path}: {exc}") from exc


def load_results(path: Path) -> dict[str, Any]:
    """Load and minimally validate a tau results object."""
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("simulations"), list):
        raise ComparisonError(f"Expected an object with a simulations list: {path}")
    return data


def simulation_key(simulation: dict[str, Any]) -> tuple[str, int]:
    """Return the stable task/trial join key for one simulation."""
    task_id = simulation.get("task_id")
    trial = simulation.get("trial")
    if not isinstance(task_id, str) or not isinstance(trial, int):
        raise ComparisonError(
            f"Simulation has invalid task_id/trial: {task_id!r}/{trial!r}"
        )
    return task_id, trial


def index_simulations(
    simulations: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[tuple[str, int]]]:
    """Index simulations and return any duplicate keys separately."""
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    for simulation in simulations:
        if not isinstance(simulation, dict):
            raise ComparisonError("Every simulation must be a JSON object")
        key = simulation_key(simulation)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = simulation
    return indexed, duplicates


def parse_tool_arguments(value: Any) -> Any:
    """Normalize JSON-string function arguments without altering plain strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalize_tool_output(value: Any) -> Any:
    """Remove only the explicitly nondeterministic retrieval timing footer."""
    if isinstance(value, str):
        return TIMING_FOOTER.sub("", value)
    return value


def tool_interactions(
    simulation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair each serialized tool call with its result by call identifier."""
    messages = simulation.get("messages") or []
    outputs_by_id: dict[str, list[dict[str, Any]]] = {}
    unpaired_outputs: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = message.get("id", message.get("tool_call_id"))
        output = {
            "call_id": call_id,
            "content": normalize_tool_output(message.get("content")),
        }
        if isinstance(call_id, str):
            outputs_by_id.setdefault(call_id, []).append(output)
        else:
            unpaired_outputs.append(output)

    extracted = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")
            else:
                name = call.get("name")
                arguments = call.get("arguments")
            call_id = call.get("id")
            paired = (
                outputs_by_id[call_id].pop(0)
                if isinstance(call_id, str) and outputs_by_id.get(call_id)
                else None
            )
            extracted.append(
                {
                    "role": role,
                    "name": name,
                    "arguments": parse_tool_arguments(arguments),
                    "output_present": paired is not None,
                    "output": paired["content"] if paired is not None else None,
                }
            )

    for remaining in outputs_by_id.values():
        unpaired_outputs.extend(remaining)
    return extracted, unpaired_outputs


def participant_texts(simulation: dict[str, Any], role: str) -> list[Any]:
    """Extract ordered non-null assistant or user message content."""
    texts = []
    for message in simulation.get("messages") or []:
        if (
            isinstance(message, dict)
            and message.get("role") == role
            and message.get("content") is not None
        ):
            texts.append(message["content"])
    return texts


def text_preview(value: Any, limit: int = 500) -> Any:
    """Bound text-divergence diagnostics without changing comparisons."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def portable_path(path: Path) -> str:
    """Prefer a repository-relative receipt path that survives checkout moves."""
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def reward(simulation: dict[str, Any]) -> float | None:
    """Extract the scalar reward, preserving a missing value as None."""
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    value = reward_info.get("reward")
    return float(value) if isinstance(value, (int, float)) else None


def reward_components(simulation: dict[str, Any]) -> dict[str, Any]:
    """Project every ordered score-defining assertion/check structure."""
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        return {kind: None for kind in REWARD_COMPONENT_KINDS}
    db_check = reward_info.get("db_check")
    nl_assertions = reward_info.get("nl_assertions")
    projected_nl = None
    if isinstance(nl_assertions, list):
        projected_nl = [
            {
                "nl_assertion": assertion.get("nl_assertion"),
                "met": assertion.get("met"),
            }
            if isinstance(assertion, dict)
            else assertion
            for assertion in nl_assertions
        ]
    return {
        "db_component": db_check,
        "env_assertions": reward_info.get("env_assertions"),
        "action_checks": reward_info.get("action_checks"),
        "nl_assertions": projected_nl,
        "communicate_checks": reward_info.get("communicate_checks"),
        "reward_basis": reward_info.get("reward_basis"),
        "reward_breakdown": reward_info.get("reward_breakdown"),
    }


def expected_keys(config: dict[str, Any], mode: str) -> list[tuple[str, int]]:
    """Resolve the exact expected task/trial key set for a comparison mode."""
    mode_config = config["modes"][mode]
    task_ids = mode_config["task_ids"]
    if task_ids == "all":
        task_ids = list(config["reward_vectors"])
    return [(task_id, trial) for task_id in task_ids for trial in mode_config["trials"]]


def fallback_reference(
    config: dict[str, Any], keys: list[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Build score/seed/termination expectations from the compact config."""
    seeds = config["recorded_run"]["derived_trial_seeds"]
    reference = {}
    for task_id, trial in keys:
        vector = config["reward_vectors"].get(task_id)
        if not isinstance(vector, str) or trial >= len(vector):
            raise ComparisonError(f"Missing reward vector for {task_id} trial {trial}")
        reference[(task_id, trial)] = {
            "task_id": task_id,
            "trial": trial,
            "seed": seeds[trial],
            "termination_reason": "user_stop",
            "reward_info": {"reward": float(vector[trial])},
        }
    return reference


def add_mismatch(
    mismatch_counts: Counter[str],
    mismatch_details: list[dict[str, Any]],
    max_details: int,
    kind: str,
    key: tuple[str, int],
    **details: Any,
) -> None:
    """Count every mismatch while bounding verbose report details."""
    mismatch_counts[kind] += 1
    if len(mismatch_details) < max_details:
        mismatch_details.append(
            {"kind": kind, "task_id": key[0], "trial": key[1], **details}
        )


def compare(
    candidate: dict[str, Any],
    reference: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
    *,
    compare_tools: bool,
    max_details: int,
) -> dict[str, Any]:
    """Compare all requested keys and return a bounded machine-readable report."""
    candidate_index, duplicates = index_simulations(candidate["simulations"])
    expected_set = set(keys)
    mismatch_counts: Counter[str] = Counter()
    mismatch_details: list[dict[str, Any]] = []
    text_divergence_messages: Counter[str] = Counter()
    text_divergence_simulations: Counter[str] = Counter()
    text_divergence_details: list[dict[str, Any]] = []

    for key in duplicates:
        add_mismatch(
            mismatch_counts,
            mismatch_details,
            max_details,
            "duplicate_simulation",
            key,
        )
    for key in sorted(expected_set - set(candidate_index)):
        add_mismatch(
            mismatch_counts,
            mismatch_details,
            max_details,
            "missing_simulation",
            key,
        )
    for key in sorted(set(candidate_index) - expected_set):
        add_mismatch(
            mismatch_counts,
            mismatch_details,
            max_details,
            "unexpected_simulation",
            key,
        )

    candidate_reward_sum = 0.0
    candidate_reward_by_trial: Counter[int] = Counter()
    expected_reward_sum = 0.0
    expected_reward_by_trial: Counter[int] = Counter()
    compared_count = 0
    for key in keys:
        expected_reward = reward(reference[key])
        if expected_reward is not None:
            expected_reward_sum += expected_reward
            expected_reward_by_trial[key[1]] += expected_reward
        if key not in candidate_index:
            continue
        actual = candidate_index[key]
        expected = reference[key]
        actual_reward = reward(actual)
        if actual_reward is not None:
            candidate_reward_sum += actual_reward
            candidate_reward_by_trial[key[1]] += actual_reward
        compared_count += 1
        if (
            expected_reward is None
            or actual_reward is None
            or not math.isclose(expected_reward, actual_reward, abs_tol=1e-12)
        ):
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "reward",
                key,
                expected=expected_reward,
                actual=actual_reward,
            )
        for field in ("seed", "termination_reason"):
            if expected.get(field) != actual.get(field):
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    field,
                    key,
                    expected=expected.get(field),
                    actual=actual.get(field),
                )

        if not compare_tools:
            continue
        expected_components = reward_components(expected)
        actual_components = reward_components(actual)
        for kind in REWARD_COMPONENT_KINDS:
            if expected_components[kind] != actual_components[kind]:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    kind,
                    key,
                    expected=expected_components[kind],
                    actual=actual_components[kind],
                )
        expected_calls, expected_unpaired_outputs = tool_interactions(expected)
        actual_calls, actual_unpaired_outputs = tool_interactions(actual)
        if len(expected_calls) != len(actual_calls):
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_call_count",
                key,
                expected=len(expected_calls),
                actual=len(actual_calls),
            )
        expected_sequence = [
            f"{call['role']}:{call['name']}" for call in expected_calls
        ]
        actual_sequence = [f"{call['role']}:{call['name']}" for call in actual_calls]
        if expected_sequence != actual_sequence:
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_call_sequence",
                key,
                expected=expected_sequence,
                actual=actual_sequence,
            )
        for call_index, (expected_call, actual_call) in enumerate(
            zip(expected_calls, actual_calls, strict=False)
        ):
            if (
                expected_call["role"] == actual_call["role"]
                and expected_call["name"] == actual_call["name"]
                and expected_call["arguments"] != actual_call["arguments"]
            ):
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_call_arguments",
                    key,
                    call_index=call_index,
                    tool=f"{expected_call['role']}:{expected_call['name']}",
                    expected=expected_call["arguments"],
                    actual=actual_call["arguments"],
                )

            expected_output_present = expected_call["output_present"]
            actual_output_present = actual_call["output_present"]
            if expected_output_present and not actual_output_present:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_missing",
                    key,
                    call_index=call_index,
                    tool=f"{expected_call['role']}:{expected_call['name']}",
                )
            elif not expected_output_present and actual_output_present:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_unexpected",
                    key,
                    call_index=call_index,
                    tool=f"{actual_call['role']}:{actual_call['name']}",
                    actual=actual_call["output"],
                )
            elif (
                expected_output_present
                and actual_output_present
                and expected_call["output"] != actual_call["output"]
            ):
                mismatch_kind = (
                    KNOWN_DENSE_DRIFT_MISMATCH_KIND
                    if expected_call["role"] == actual_call["role"] == "assistant"
                    and expected_call["name"]
                    == actual_call["name"]
                    == "KB_search_dense"
                    else "tool_output"
                )
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    mismatch_kind,
                    key,
                    call_index=call_index,
                    tool=f"{expected_call['role']}:{expected_call['name']}",
                    expected=expected_call["output"],
                    actual=actual_call["output"],
                )

        for call_index in range(len(actual_calls), len(expected_calls)):
            expected_call = expected_calls[call_index]
            if expected_call["output_present"]:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_missing",
                    key,
                    call_index=call_index,
                    tool=f"{expected_call['role']}:{expected_call['name']}",
                )
        for call_index in range(len(expected_calls), len(actual_calls)):
            actual_call = actual_calls[call_index]
            if actual_call["output_present"]:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_unexpected",
                    key,
                    call_index=call_index,
                    tool=f"{actual_call['role']}:{actual_call['name']}",
                    actual=actual_call["output"],
                )
        for output in expected_unpaired_outputs:
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_missing",
                key,
                expected=output["content"],
                note="Reference ToolMessage is not paired with a reference call",
            )
        for output in actual_unpaired_outputs:
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_unpaired",
                key,
                actual=output["content"],
                call_id=output["call_id"],
            )

        for role in ("assistant", "user"):
            expected_texts = participant_texts(expected, role)
            actual_texts = participant_texts(actual, role)
            difference_indexes = [
                index
                for index in range(max(len(expected_texts), len(actual_texts)))
                if (expected_texts[index] if index < len(expected_texts) else None)
                != (actual_texts[index] if index < len(actual_texts) else None)
            ]
            if not difference_indexes:
                continue
            text_divergence_messages[role] += len(difference_indexes)
            text_divergence_simulations[role] += 1
            if len(text_divergence_details) < max_details:
                first = difference_indexes[0]
                text_divergence_details.append(
                    {
                        "task_id": key[0],
                        "trial": key[1],
                        "role": role,
                        "different_message_count": len(difference_indexes),
                        "expected_message_count": len(expected_texts),
                        "actual_message_count": len(actual_texts),
                        "first_difference_index": first,
                        "expected": text_preview(
                            expected_texts[first]
                            if first < len(expected_texts)
                            else None
                        ),
                        "actual": text_preview(
                            actual_texts[first] if first < len(actual_texts) else None
                        ),
                    }
                )

    score_mismatch_count = sum(
        count for kind, count in mismatch_counts.items() if kind in SCORING_KINDS
    )
    behavior_mismatch_count = sum(
        count for kind, count in mismatch_counts.items() if kind not in SCORING_KINDS
    )
    tool_output_mismatch_count = sum(
        count
        for kind, count in mismatch_counts.items()
        if kind.startswith("tool_output")
    )
    component_mismatch_count = sum(
        mismatch_counts[kind] for kind in REWARD_COMPONENT_KINDS
    )
    known_dense_drift_mismatch_count = mismatch_counts[KNOWN_DENSE_DRIFT_MISMATCH_KIND]
    non_waived_behavior_mismatch_count = (
        behavior_mismatch_count - known_dense_drift_mismatch_count
    )
    return {
        "score_parity": score_mismatch_count == 0,
        "behavior_parity": behavior_mismatch_count == 0 if compare_tools else None,
        "behavior_parity_with_known_dense_drift_waiver": (
            non_waived_behavior_mismatch_count == 0 if compare_tools else None
        ),
        "behavior_checked": compare_tools,
        "expected_simulation_count": len(keys),
        "candidate_simulation_count": len(candidate_index),
        "compared_simulation_count": compared_count,
        "expected_reward_sum": expected_reward_sum,
        "expected_reward_by_trial": [
            expected_reward_by_trial[trial]
            for trial in sorted(expected_reward_by_trial)
        ],
        "candidate_reward_sum": candidate_reward_sum,
        "candidate_reward_by_trial": [
            candidate_reward_by_trial[trial]
            for trial in sorted(expected_reward_by_trial)
        ],
        "candidate_pass_rate": (
            100.0 * candidate_reward_sum / len(keys) if keys else None
        ),
        "score_mismatch_count": score_mismatch_count,
        "component_parity_checked": compare_tools,
        "component_parity": component_mismatch_count == 0 if compare_tools else None,
        "component_mismatch_count": component_mismatch_count,
        "behavior_mismatch_count": behavior_mismatch_count,
        "known_dense_drift_waiver_scope": KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
        "known_dense_drift_mismatch_count": known_dense_drift_mismatch_count,
        "non_waived_behavior_mismatch_count": non_waived_behavior_mismatch_count,
        "tool_output_mismatch_count": tool_output_mismatch_count,
        "tool_call_mismatch_count": behavior_mismatch_count
        - tool_output_mismatch_count,
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "mismatch_details": mismatch_details,
        "mismatch_details_truncated": sum(mismatch_counts.values())
        > len(mismatch_details),
        "text_divergence_checked": compare_tools,
        "text_divergence_message_counts": dict(
            sorted(text_divergence_messages.items())
        ),
        "text_divergence_simulation_counts": dict(
            sorted(text_divergence_simulations.items())
        ),
        "text_divergence_details": text_divergence_details,
        "text_divergence_details_truncated": sum(text_divergence_simulations.values())
        > len(text_divergence_details),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically write JSON so a killed comparison cannot create a valid-looking gate."""
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


def metadata_projection(results: dict[str, Any]) -> dict[str, Any]:
    """Return a concise diagnostic view of serialized run configuration."""
    info = results.get("info") or {}
    return {
        "git_commit": info.get("git_commit"),
        "num_trials": info.get("num_trials"),
        "max_steps": info.get("max_steps"),
        "max_errors": info.get("max_errors"),
        "seed": info.get("seed"),
        "retrieval_config": info.get("retrieval_config"),
        "retrieval_config_kwargs": info.get("retrieval_config_kwargs"),
        "agent_implementation": (info.get("agent_info") or {}).get("implementation"),
        "agent_model": (info.get("agent_info") or {}).get("llm"),
        "agent_llm_args": (info.get("agent_info") or {}).get("llm_args"),
        "user_implementation": (info.get("user_info") or {}).get("implementation"),
        "user_model": (info.get("user_info") or {}).get("llm"),
        "user_llm_args": (info.get("user_info") or {}).get("llm_args"),
        "domain": (info.get("environment_info") or {}).get("domain_name"),
    }


def expected_reproduction_metadata(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return the serialized metadata expected from this reproduction runner."""
    recorded = config["recorded_run"]
    return {
        "git_commit": {"git_descendant_of": config["benchmark"]["git_commit"]},
        "num_trials": (
            {"one_of": [1, 4]}
            if mode == "subset"
            else len(config["modes"][mode]["trials"])
        ),
        "max_steps": recorded["max_steps"],
        "max_errors": recorded["max_errors"],
        "seed": recorded["seed"],
        "retrieval_config": recorded["retrieval_config"],
        "retrieval_config_kwargs": recorded["retrieval_config_kwargs"],
        "agent_implementation": recorded["agent"]["implementation"],
        "agent_model": config["reproduction_transport"]["agent_model"],
        "agent_llm_args": recorded["agent"]["llm_args"],
        "user_implementation": recorded["user"]["implementation"],
        "user_model": config["reproduction_transport"]["user_model"],
        "user_llm_args": config["reproduction_transport"]["user_llm_args"],
        "domain": config["benchmark"]["domain"],
    }


def metadata_mismatches(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Describe exact serialized run-configuration differences."""
    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        matches = actual_value == expected_value
        if isinstance(expected_value, dict) and set(expected_value) == {"one_of"}:
            matches = actual_value in expected_value["one_of"]
        elif isinstance(expected_value, dict) and set(expected_value) == {
            "git_descendant_of"
        }:
            matches = git_is_ancestor(expected_value["git_descendant_of"], actual_value)
        if not matches:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    return mismatches


def _route_counter_rows(counter: Counter[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Render route tuples as stable JSON rows rather than lossy string keys."""
    rows = []
    for route, count in sorted(counter.items(), key=lambda item: repr(item[0])):
        role, model, provider, service_tier = route
        rows.append(
            {
                "role": role,
                "model": model,
                "provider": provider,
                "service_tier": service_tier,
                "count": count,
            }
        )
    return rows


def _normalized_route_model(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.removeprefix("openrouter/")


def validate_raw_routes(
    candidate: dict[str, Any],
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    """Validate that every paid participant response used the pinned provider."""
    expected_keys = set(keys)
    routes: Counter[tuple[Any, ...]] = Counter()
    missing_raw_data: Counter[str] = Counter()
    for simulation in candidate["simulations"]:
        if not isinstance(simulation, dict):
            continue
        try:
            key = simulation_key(simulation)
        except ComparisonError:
            continue
        if key not in expected_keys:
            continue
        for message in simulation.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") not in {
                "assistant",
                "user",
            }:
                continue
            role = message["role"]
            raw_data = message.get("raw_data")
            generated = isinstance(message.get("usage"), dict) or (
                isinstance(message.get("cost"), (int, float))
                and not isinstance(message.get("cost"), bool)
                and float(message["cost"]) > 0
            )
            if not isinstance(raw_data, dict):
                if generated:
                    missing_raw_data[role] += 1
                continue
            routes[
                (
                    role,
                    raw_data.get("model"),
                    raw_data.get("provider"),
                    raw_data.get("service_tier"),
                )
            ] += 1

    mismatches: dict[str, dict[str, Any]] = {}
    for index, (route, count) in enumerate(
        sorted(routes.items(), key=lambda item: repr(item[0]))
    ):
        role, model, provider, service_tier = route
        normalized_model = _normalized_route_model(model)
        if role == "assistant":
            valid = (
                normalized_model == "qwen/qwen3.8-max"
                and provider == "Alibaba"
                and service_tier is None
            )
            expected = {
                "model": "qwen/qwen3.8-max",
                "provider": "Alibaba",
                "service_tier": None,
            }
        else:
            resolved_user_model = normalized_model
            if isinstance(resolved_user_model, str):
                resolved_user_model = resolved_user_model.removeprefix("openai/")
            valid = (
                (
                    resolved_user_model is None
                    or resolved_user_model == "gpt-5.2-2025-12-11"
                )
                and provider == "OpenAI"
                and service_tier == "default"
            )
            expected = {
                "model": "gpt-5.2-2025-12-11 when raw_data exposes a model",
                "provider": "OpenAI",
                "service_tier": "default",
            }
        if not valid:
            mismatches[f"raw_routes.{role}.{index}"] = {
                "expected": expected,
                "actual": {
                    "model": model,
                    "provider": provider,
                    "service_tier": service_tier,
                    "count": count,
                },
            }
    for role in ("assistant", "user"):
        role_count = sum(count for route, count in routes.items() if route[0] == role)
        if role_count == 0:
            mismatches[f"raw_routes.{role}.missing"] = {
                "expected": "at least one provider-attributed generated response",
                "actual": 0,
            }
        if missing_raw_data[role]:
            mismatches[f"raw_routes.{role}.unattributed"] = {
                "expected": 0,
                "actual": missing_raw_data[role],
            }
    return {
        "raw_route_parity": not mismatches,
        "raw_route_mismatch_count": len(mismatches),
        "raw_route_mismatches": mismatches,
        "raw_route_counters": _route_counter_rows(routes),
        "raw_route_unattributed_generated_messages": dict(
            sorted(missing_raw_data.items())
        ),
    }


def validate_judge_routes(
    candidate: dict[str, Any], keys: list[tuple[str, int]], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate and retain task 102's otherwise hidden NL-judge route."""
    task_keys = {key for key in keys if key[0] == "task_102"}
    if not task_keys:
        return {
            "judge_route_checked": False,
            "judge_route_parity": None,
            "judge_route_mismatch_count": 0,
            "judge_route_mismatches": {},
            "judge_route_observations": [],
        }
    candidate_index, _ = index_simulations(candidate["simulations"])
    expected_requested = config["reproduction_transport"]["nl_assertions_model"]
    expected_resolved = "gpt-4.1-2025-04-14"
    observations = []
    mismatches: dict[str, dict[str, Any]] = {}
    for key in sorted(task_keys):
        simulation = candidate_index.get(key)
        if simulation is None:
            continue
        reward_info = simulation.get("reward_info")
        info = reward_info.get("info") if isinstance(reward_info, dict) else None
        judge = None
        if isinstance(info, dict):
            judge = info.get("judge")
            if judge is None and isinstance(info.get("nl"), dict):
                judge = info["nl"].get("judge")
        observation = {
            "task_id": key[0],
            "trial": key[1],
            "requested_model": judge.get("requested_model")
            if isinstance(judge, dict)
            else None,
            "resolved_model": judge.get("resolved_model")
            if isinstance(judge, dict)
            else None,
            "provider": judge.get("provider") if isinstance(judge, dict) else None,
            "service_tier": judge.get("service_tier")
            if isinstance(judge, dict)
            else None,
            "response_id": judge.get("response_id")
            if isinstance(judge, dict)
            else None,
        }
        observations.append(observation)
        resolved = _normalized_route_model(observation["resolved_model"])
        if isinstance(resolved, str):
            resolved = resolved.removeprefix("openai/")
        expected = {
            "requested_model": expected_requested,
            "resolved_model": expected_resolved,
            "provider": "OpenAI",
            "service_tier": "default",
            "response_id": "non-empty string",
        }
        valid = (
            observation["requested_model"] == expected_requested
            and resolved == expected_resolved
            and observation["provider"] == "OpenAI"
            and observation["service_tier"] == "default"
            and isinstance(observation["response_id"], str)
            and bool(observation["response_id"])
        )
        if not valid:
            mismatches[f"judge_route.{key[0]}.{key[1]}"] = {
                "expected": expected,
                "actual": observation,
            }
    return {
        "judge_route_checked": True,
        "judge_route_parity": not mismatches and len(observations) == len(task_keys),
        "judge_route_mismatch_count": len(mismatches)
        + (len(task_keys) - len(observations)),
        "judge_route_mismatches": mismatches,
        "judge_route_observations": observations,
    }


def validate_execution_manifest(
    candidate_path: Path,
    candidate: dict[str, Any],
    config: dict[str, Any],
    config_digest: str,
    mode: str,
    artifact_digest: str,
) -> dict[str, Any]:
    """Bind a candidate checkpoint to the guarded command that produced it."""
    manifest_path = candidate_path.parent / f"reproduction_manifest_{mode}.json"
    mismatches: dict[str, dict[str, Any]] = {}
    if not manifest_path.is_file():
        mismatches["execution_manifest.missing"] = {
            "expected": portable_path(manifest_path),
            "actual": None,
        }
        return {
            "execution_manifest_parity": False,
            "execution_manifest_mismatch_count": 1,
            "execution_manifest_mismatches": mismatches,
            "execution_manifest": None,
            "execution_manifest_sha256": None,
        }
    manifest = load_json(manifest_path)
    candidate_head = (candidate.get("info") or {}).get("git_commit")
    runtime = (manifest.get("execution_state") or {}).get("runtime") or {}
    execution_state_digest = (manifest.get("execution_state") or {}).get("digest")
    canonical_commands = [
        build_command(config, mode, candidate_path.parent, resume=resume)
        for resume in (False, True)
    ]
    canonical_environment = expected_manifest_environment(
        config,
        modal_app=DEFAULT_MODAL_APP,
        modal_sandbox_timeout=DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )
    expected = {
        "mode": mode,
        "dry_run": False,
        "status": "completed",
        "exit_code": 0,
        "output_dir": str(candidate_path.parent),
        "reference_config_sha256": config_digest,
        "checkpoint_sha256": artifact_digest,
        "execution_state.runtime.head": candidate_head,
        "post_run_execution_state.digest": execution_state_digest,
        "environment": canonical_environment,
        "prompt_hashes": expected_prompt_hashes(config),
    }
    actual = {
        "mode": manifest.get("mode"),
        "dry_run": manifest.get("dry_run"),
        "status": manifest.get("status"),
        "exit_code": manifest.get("exit_code"),
        "output_dir": manifest.get("output_dir"),
        "reference_config_sha256": manifest.get("reference_config_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "execution_state.runtime.head": runtime.get("head"),
        "post_run_execution_state.digest": (
            manifest.get("post_run_execution_state") or {}
        ).get("digest"),
        "environment": manifest.get("environment"),
        "prompt_hashes": manifest.get("prompt_hashes"),
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            mismatches[f"execution_manifest.{field}"] = {
                "expected": expected_value,
                "actual": actual[field],
            }
    command = manifest.get("command")
    if command not in canonical_commands:
        mismatches["execution_manifest.command"] = {
            "expected": canonical_commands,
            "actual": command,
        }
    inventory_mismatches = endpoint_inventory_mismatches(
        manifest.get("openrouter_endpoint_inventory")
    )
    for field, detail in inventory_mismatches.items():
        mismatches[f"execution_manifest.endpoint_inventory.{field}"] = detail
    state_digest = execution_state_digest
    if not isinstance(state_digest, str) or len(state_digest) != 64:
        mismatches["execution_manifest.execution_state.digest"] = {
            "expected": "64-character state digest",
            "actual": state_digest,
        }
    return {
        "execution_manifest_parity": not mismatches,
        "execution_manifest_mismatch_count": len(mismatches),
        "execution_manifest_mismatches": mismatches,
        "execution_manifest": portable_path(manifest_path),
        "execution_manifest_sha256": digest_file(manifest_path),
    }


def git_is_ancestor(ancestor: Any, descendant: Any) -> bool:
    """Check a serialized candidate commit against the pinned upstream base."""
    if not isinstance(ancestor, str) or not isinstance(descendant, str):
        return False
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.returncode == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", type=Path, help="Candidate results.json or run dir"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Reference JSON path"
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=None,
        help="Verified official trajectory (defaults to artifacts/...) ",
    )
    parser.add_argument("--mode", choices=("smoke", "subset", "full"), required=True)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Compare task/trial reward, seed, and termination without loading traces",
    )
    parser.add_argument(
        "--output", type=Path, help="Also atomically write the JSON report here"
    )
    parser.add_argument(
        "--write-gate",
        nargs="?",
        const=DEFAULT_GATE,
        type=Path,
        help="Write a full-run gate after exact subset score parity",
    )
    parser.add_argument(
        "--allow-known-dense-drift",
        action="store_true",
        help=(
            "With --write-gate, waive only paired assistant KB_search_dense "
            "ToolMessage content mismatches; strict diagnostics remain in the report"
        ),
    )
    parser.add_argument(
        "--max-mismatch-details",
        type=int,
        default=100,
        help="Bound verbose mismatch objects while retaining exact counts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the result comparison."""
    args = parse_args(argv)
    try:
        if args.max_mismatch_details < 0:
            raise ComparisonError("--max-mismatch-details must be non-negative")
        if args.allow_known_dense_drift and args.write_gate is None:
            raise ComparisonError(
                "--allow-known-dense-drift is valid only with --write-gate"
            )
        config_path = args.config.resolve()
        config = load_json(config_path)
        config_digest = digest_file(config_path)
        candidate_path = resolve_results_path(args.candidate)
        candidate = load_results(candidate_path)
        candidate_artifact_digest = digest_checkpoint_artifact(candidate_path)
        keys = expected_keys(config, args.mode)

        reference_path = None
        reference_results = None
        reference = fallback_reference(config, keys)
        if not args.score_only:
            reference_path = resolve_results_path(
                args.reference_results or DEFAULT_REFERENCE_RESULTS
            )
            expected_digest = config["artifacts"]["trajectory"]["sha256"]
            actual_digest = digest_file(reference_path)
            if actual_digest != expected_digest:
                raise ComparisonError(
                    "Official trajectory digest mismatch: "
                    f"{actual_digest} != {expected_digest}"
                )
            if reference_path == candidate_path:
                reference_results = candidate
            else:
                reference_results = load_results(reference_path)
            reference_index, reference_duplicates = index_simulations(
                reference_results["simulations"]
            )
            if reference_duplicates:
                raise ComparisonError(
                    "Official trajectory has duplicate task/trial keys"
                )
            missing_reference = set(keys) - set(reference_index)
            if missing_reference:
                raise ComparisonError(
                    f"Official trajectory is missing {len(missing_reference)} expected keys"
                )
            reference = {key: reference_index[key] for key in keys}

        report = {
            "schema_version": 2,
            "mode": args.mode,
            "candidate": str(candidate_path),
            "candidate_sha256": digest_file(candidate_path),
            "candidate_artifact_sha256": candidate_artifact_digest,
            "reference_config": str(config_path),
            "reference_config_sha256": config_digest,
            "reference_results": str(reference_path) if reference_path else None,
            "reference_results_sha256": (
                config["artifacts"]["trajectory"]["sha256"] if reference_path else None
            ),
            "candidate_metadata": metadata_projection(candidate),
            "reference_metadata": (
                metadata_projection(reference_results)
                if reference_results is not None
                else config["recorded_run"]
            ),
            **compare(
                candidate,
                reference,
                keys,
                compare_tools=not args.score_only,
                max_details=args.max_mismatch_details,
            ),
        }
        raw_route_report = validate_raw_routes(candidate, keys)
        judge_route_report = validate_judge_routes(candidate, keys, config)
        execution_manifest_report = validate_execution_manifest(
            candidate_path,
            candidate,
            config,
            config_digest,
            args.mode,
            candidate_artifact_digest,
        )
        report.update(raw_route_report)
        report.update(judge_route_report)
        report.update(execution_manifest_report)
        report["known_dense_drift_waiver_requested"] = args.allow_known_dense_drift
        expected_metadata = expected_reproduction_metadata(config, args.mode)
        config_mismatches = metadata_mismatches(
            report["candidate_metadata"], expected_metadata
        )
        config_mismatches.update(raw_route_report["raw_route_mismatches"])
        config_mismatches.update(judge_route_report["judge_route_mismatches"])
        config_mismatches.update(
            execution_manifest_report["execution_manifest_mismatches"]
        )
        if (
            judge_route_report["judge_route_checked"]
            and not judge_route_report["judge_route_parity"]
            and judge_route_report["judge_route_mismatch_count"]
            > len(judge_route_report["judge_route_mismatches"])
        ):
            config_mismatches["judge_route.coverage"] = {
                "expected": len([key for key in keys if key[0] == "task_102"]),
                "actual": len(judge_route_report["judge_route_observations"]),
            }
        report["expected_reproduction_metadata"] = expected_metadata
        report["configuration_parity"] = not config_mismatches
        report["configuration_mismatch_count"] = len(config_mismatches)
        report["configuration_mismatches"] = config_mismatches

        if args.write_gate is not None:
            if args.mode != "subset":
                raise ComparisonError(
                    "A full-run gate can only be written in subset mode"
                )
            if args.score_only:
                raise ComparisonError(
                    "Refusing to write gate from --score-only: exact component, "
                    "tool-call, and ToolMessage parity must be checked"
                )
            if not report["score_parity"]:
                raise ComparisonError(
                    "Refusing to write gate: subset score parity failed"
                )
            if not report["configuration_parity"]:
                raise ComparisonError(
                    "Refusing to write gate: reproduction run configuration differs"
                )
            if not report["behavior_checked"]:
                raise ComparisonError(
                    "Refusing to write gate: tool behavior was not checked"
                )
            if not report["behavior_parity"]:
                if not args.allow_known_dense_drift:
                    raise ComparisonError(
                        "Refusing to write gate: strict tool behavior differs; pass "
                        "--allow-known-dense-drift only after confirming every mismatch "
                        "is a paired assistant KB_search_dense ToolMessage content drift"
                    )
                if not report["behavior_parity_with_known_dense_drift_waiver"]:
                    raise ComparisonError(
                        "Refusing to write gate: tool behavior contains a mismatch "
                        "outside the narrow known dense-output drift waiver"
                    )
            if not report["component_parity_checked"] or not report["component_parity"]:
                raise ComparisonError(
                    "Refusing to write gate: reward component parity differs"
                )
            gate_reference_path = resolve_results_path(
                args.reference_results or DEFAULT_REFERENCE_RESULTS
            )
            gate_reference_digest = digest_file(gate_reference_path)
            expected_reference_digest = config["artifacts"]["trajectory"]["sha256"]
            if gate_reference_digest != expected_reference_digest:
                raise ComparisonError(
                    "Refusing to write gate: official trajectory digest mismatch"
                )
            execution_state = capture_reproduction_state(
                REPO_ROOT, require_clean=True, require_cache=True
            )
            if (
                report["candidate_metadata"].get("git_commit")
                != execution_state["runtime"]["head"]
            ):
                raise ComparisonError(
                    "Refusing to write gate: candidate commit differs from the "
                    "current clean runtime HEAD"
                )
            if (
                digest_checkpoint_artifact(candidate_path)
                != report["candidate_artifact_sha256"]
            ):
                raise ComparisonError(
                    "Refusing to write gate: candidate checkpoint changed during comparison"
                )
            gate_manifest_path = (
                Path(report["execution_manifest"])
                if Path(report["execution_manifest"]).is_absolute()
                else REPO_ROOT / report["execution_manifest"]
            )
            if digest_file(gate_manifest_path) != report["execution_manifest_sha256"]:
                raise ComparisonError(
                    "Refusing to write gate: execution manifest changed during comparison"
                )
            gate_manifest = load_json(gate_manifest_path)
            if (gate_manifest.get("post_run_execution_state") or {}).get(
                "digest"
            ) != execution_state["digest"]:
                raise ComparisonError(
                    "Refusing to write gate: candidate post-run cache/runtime state "
                    "differs from the current state"
                )
            gate = {
                "schema_version": FULL_GATE_SCHEMA_VERSION,
                "kind": "tau3_banking_subset_score_parity",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": "subset",
                "expected_simulation_count": report["expected_simulation_count"],
                "expected_reward_sum": report["expected_reward_sum"],
                "expected_reward_by_trial": report["expected_reward_by_trial"],
                "candidate_reward_sum": report["candidate_reward_sum"],
                "candidate_reward_by_trial": report["candidate_reward_by_trial"],
                "candidate_sha256": report["candidate_sha256"],
                "candidate_artifact": portable_path(candidate_path),
                "candidate_artifact_sha256": report["candidate_artifact_sha256"],
                "execution_manifest": report["execution_manifest"],
                "execution_manifest_sha256": report["execution_manifest_sha256"],
                "reference_config_sha256": report["reference_config_sha256"],
                "reference_results_sha256": config["artifacts"]["trajectory"]["sha256"],
                "score_parity": True,
                "score_mismatch_count": 0,
                "configuration_parity": True,
                "configuration_mismatch_count": 0,
                "candidate_metadata": report["candidate_metadata"],
                "behavior_checked": report["behavior_checked"],
                "behavior_parity": report["behavior_parity"],
                "behavior_mismatch_count": report["behavior_mismatch_count"],
                "behavior_parity_with_known_dense_drift_waiver": report[
                    "behavior_parity_with_known_dense_drift_waiver"
                ],
                "known_dense_drift_waiver_requested": (args.allow_known_dense_drift),
                "known_dense_drift_waiver_applied": (
                    args.allow_known_dense_drift and not report["behavior_parity"]
                ),
                "known_dense_drift_waiver_scope": report[
                    "known_dense_drift_waiver_scope"
                ],
                "known_dense_drift_mismatch_count": report[
                    "known_dense_drift_mismatch_count"
                ],
                "non_waived_behavior_mismatch_count": report[
                    "non_waived_behavior_mismatch_count"
                ],
                "component_parity_checked": report["component_parity_checked"],
                "component_parity": report["component_parity"],
                "component_mismatch_count": report["component_mismatch_count"],
                "execution_manifest_parity": report["execution_manifest_parity"],
                "execution_manifest_mismatch_count": report[
                    "execution_manifest_mismatch_count"
                ],
                "raw_route_parity": report["raw_route_parity"],
                "raw_route_mismatch_count": report["raw_route_mismatch_count"],
                "raw_route_counters": report["raw_route_counters"],
                "judge_route_parity": report["judge_route_parity"],
                "judge_route_mismatch_count": report["judge_route_mismatch_count"],
                "judge_route_observations": report["judge_route_observations"],
                "reference_verified_from": str(gate_reference_path),
                "execution_state": execution_state,
            }
            write_json_atomic(args.write_gate, gate)
            report["gate_written"] = str(args.write_gate.resolve())

        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            write_json_atomic(args.output, report)
        if not report["score_parity"]:
            return 1
        if not report["configuration_parity"]:
            return 1
        behavior_accepted_by_written_gate = (
            args.write_gate is not None
            and args.allow_known_dense_drift
            and report["behavior_parity_with_known_dense_drift_waiver"]
            and report.get("gate_written") is not None
        )
        if (
            report["behavior_checked"]
            and not report["behavior_parity"]
            and not behavior_accepted_by_written_gate
        ):
            return 1
        return 0
    except (
        ComparisonError,
        RunGuardError,
        StateFingerprintError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
