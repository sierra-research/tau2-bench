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
    GPT52_ALIAS_MODEL,
    KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
    MODEL_SAMPLING_DRIFT_WAIVER_SCOPE,
    RunGuardError,
    build_command,
    endpoint_inventory_mismatches,
    expected_manifest_environment,
    expected_prompt_hashes,
    gpt52_alias_inventory_mismatches,
    verify_canonical_tau2_runtime,
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
    "message_schema_reference",
    "message_schema_candidate",
    "message_protocol_reference",
    "message_protocol_candidate",
    *REWARD_COMPONENT_KINDS,
}
STRUCTURAL_KINDS = {
    "duplicate_simulation",
    "missing_simulation",
    "unexpected_simulation",
    "seed",
    "termination_reason",
    "message_schema_reference",
    "message_schema_candidate",
    "message_protocol_reference",
    "message_protocol_candidate",
}
TIMING_FOOTER = re.compile(r"(?:\r?\n){0,2}\[Timing:[^\r\n]*\][ \t]*\Z")
KNOWN_DENSE_DRIFT_MISMATCH_KIND = "tool_output_dense_known_drift"
MAX_TOOL_OUTPUT_PREVIEW = 500
INITIAL_ASSISTANT_GREETING = "Hi! How can I help you today?"
REWARD_BASIS_COMPONENT_FIELDS = {
    "DB": "db_component",
    "ENV_ASSERTION": "env_assertions",
    "ACTION": "action_checks",
    "NL_ASSERTION": "nl_assertions",
    "COMMUNICATE": "communicate_checks",
}


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
            "requestor": message.get("requestor", "assistant"),
            "error": message.get("error", False),
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
                    # Match ToolCall's Pydantic default exactly.  The outer
                    # message role is not a safe fallback: a user-authored
                    # message with an omitted requestor replays as an
                    # assistant-requested call.
                    "requestor": call.get("requestor", "assistant"),
                    "name": name,
                    "arguments": parse_tool_arguments(arguments),
                    "output_present": paired is not None,
                    "output": paired["content"] if paired is not None else None,
                    "output_requestor": (
                        paired["requestor"] if paired is not None else None
                    ),
                    "output_error": paired["error"] if paired is not None else None,
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


def message_protocol_issues(
    messages: list[Any], *, require_user_terminal: bool = False
) -> list[dict[str, Any]]:
    """Validate the half-duplex ordering that produced a text trajectory."""
    issues: list[dict[str, Any]] = []
    expected_participant = "assistant"
    pending_calls: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    participant_messages: list[Any] = []

    if require_user_terminal:
        first = messages[0] if messages else None
        valid_fixed_greeting = (
            first is not None
            and first.role == "assistant"
            and first.content == INITIAL_ASSISTANT_GREETING
            and first.tool_calls is None
            and first.raw_data is None
            and first.usage is None
            and first.cost == 0.0
            and first.is_audio is False
            and first.audio_content is None
        )
        if not valid_fixed_greeting:
            issues.append(
                {
                    "message_index": 0,
                    "reason": "missing_fixed_initial_assistant_greeting",
                }
            )

    for index, message in enumerate(messages):
        role = message.role
        if role in {"assistant", "user"}:
            participant_messages.append(message)
            audio_fields = (
                "audio_content",
                "audio_format",
                "audio_path",
                "audio_script_gold",
                "speech_effects",
                "source_effects",
                "channel_effects",
            )
            present_audio_fields = [
                field
                for field in audio_fields
                if getattr(message, field, None) is not None
            ]
            if message.is_audio or present_audio_fields:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "audio_not_allowed_in_text_benchmark",
                        "is_audio": message.is_audio,
                        "present_audio_fields": present_audio_fields,
                    }
                )
            if message.tool_calls == []:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "empty_tool_call_list",
                    }
                )
            if not message.has_content() and not (message.tool_calls or []):
                issues.append(
                    {
                        "message_index": index,
                        "reason": "empty_participant_message",
                    }
                )
            if pending_calls:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "participant_message_before_all_tool_outputs",
                        "pending_call_ids": sorted(pending_calls),
                    }
                )
            if role != expected_participant:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "participant_out_of_turn",
                        "expected": expected_participant,
                        "actual": role,
                    }
                )
            tool_calls = message.tool_calls or []
            if tool_calls:
                for call in tool_calls:
                    if call.id in seen_call_ids:
                        issues.append(
                            {
                                "message_index": index,
                                "reason": "duplicate_tool_call_id",
                                "call_id": call.id,
                            }
                        )
                    seen_call_ids.add(call.id)
                    if call.requestor != role:
                        issues.append(
                            {
                                "message_index": index,
                                "reason": "tool_call_requestor_mismatch",
                                "call_id": call.id,
                                "expected": role,
                                "actual": call.requestor,
                            }
                        )
                    pending_calls[call.id] = call.requestor
            else:
                expected_participant = "user" if role == "assistant" else "assistant"
        elif role == "tool":
            requestor = pending_calls.pop(message.id, None)
            if requestor is None:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "tool_output_without_pending_call",
                        "call_id": message.id,
                    }
                )
            elif message.requestor != requestor:
                issues.append(
                    {
                        "message_index": index,
                        "reason": "tool_output_requestor_mismatch",
                        "call_id": message.id,
                        "expected": requestor,
                        "actual": message.requestor,
                    }
                )
        else:
            issues.append(
                {
                    "message_index": index,
                    "reason": "unsupported_half_duplex_role",
                    "actual": role,
                }
            )

    if pending_calls:
        issues.append(
            {
                "message_index": len(messages),
                "reason": "missing_tool_outputs_at_end",
                "pending_call_ids": sorted(pending_calls),
            }
        )
    if not participant_messages:
        issues.append(
            {
                "message_index": 0,
                "reason": "missing_participant_messages",
            }
        )
    elif require_user_terminal:
        terminal = participant_messages[-1]
        content = terminal.content or ""
        stop_tokens = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
        if (
            terminal.role != "user"
            or terminal.tool_calls is not None
            or not any(token in content for token in stop_tokens)
        ):
            issues.append(
                {
                    "message_index": len(messages) - 1,
                    "reason": "missing_final_user_stop_token",
                    "actual_role": terminal.role,
                    "actual_tool_calls": terminal.tool_calls,
                }
            )
    return issues


def _participant_text_transcript(simulation: dict[str, Any]) -> dict[str, list[Any]]:
    """Project only text that can affect communication/NL grading."""
    return {role: participant_texts(simulation, role) for role in ("assistant", "user")}


def load_authoritative_banking_tasks() -> dict[str, Any]:
    """Load the current committed banking task definitions by stable ID."""
    verify_canonical_tau2_runtime()
    from tau2.domains.banking_knowledge.environment import get_tasks

    tasks: dict[str, Any] = {}
    for task in get_tasks():
        if task.id in tasks:
            raise ComparisonError(f"Duplicate authoritative banking task: {task.id}")
        tasks[task.id] = task
    return tasks


def _serialized_deterministic_component(
    reward_info: dict[str, Any], component: str
) -> dict[str, Any]:
    """Project only deterministic evaluator outputs for one component."""
    basis = reward_info.get("reward_basis")
    breakdown = reward_info.get("reward_breakdown")
    projected: dict[str, Any]
    if component == "DB":
        db_check = reward_info.get("db_check")
        projected = {
            "db_match": db_check.get("db_match")
            if isinstance(db_check, dict)
            else None,
            "db_reward": db_check.get("db_reward")
            if isinstance(db_check, dict)
            else None,
        }
    else:
        specs = {
            "ENV_ASSERTION": (
                "env_assertions",
                ("env_assertion", "met", "reward"),
            ),
            "ACTION": (
                "action_checks",
                ("action", "action_match", "action_reward"),
            ),
            "COMMUNICATE": (
                "communicate_checks",
                ("info", "met"),
            ),
        }
        field, keys = specs[component]
        records = reward_info.get(field)
        projected = {
            "records": [
                {key: record.get(key) for key in keys}
                if isinstance(record, dict)
                else record
                for record in records
            ]
            if isinstance(records, list)
            else records
        }
    if isinstance(basis, list) and component in basis and isinstance(breakdown, dict):
        projected["basis_reward"] = breakdown.get(component)
    return projected


def _recompute_deterministic_components(
    simulation: dict[str, Any], task: Any, components: set[str]
) -> dict[str, dict[str, Any]]:
    """Re-run deterministic official evaluators without retrieval or remote calls."""
    from pydantic import TypeAdapter

    from tau2.data_model.message import Message
    from tau2.domains.banking_knowledge.environment import get_environment
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.runner.build import _derive_read_log_allowlist

    messages = TypeAdapter(list[Message]).validate_python(
        simulation.get("messages") or []
    )
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, dict):
        raise ComparisonError("candidate reward_info is not an object")
    recomputed: dict[str, dict[str, Any]] = {}

    if components & {"DB", "ENV_ASSERTION"}:
        allowlist = _derive_read_log_allowlist(task)

        def environment_constructor(**kwargs: Any) -> Any:
            return get_environment(
                retrieval_variant="no_knowledge",
                task=task,
                read_log_allowlist=allowlist,
                **kwargs,
            )

        env_reward = EnvironmentEvaluator.calculate_reward(
            environment_constructor=environment_constructor,
            task=task,
            full_trajectory=messages,
            solo_mode=False,
            env_kwargs={},
            strict_replay=False,
        )
        env_dump = env_reward.model_dump(mode="json")
        for component in components & {"DB", "ENV_ASSERTION"}:
            recomputed[component] = _serialized_deterministic_component(
                {
                    **env_dump,
                    "reward_basis": reward_info.get("reward_basis"),
                },
                component,
            )

    if "ACTION" in components:
        action_reward = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
            tool_types=None,
        ).model_dump(mode="json")
        recomputed["ACTION"] = _serialized_deterministic_component(
            {
                **action_reward,
                "reward_basis": reward_info.get("reward_basis"),
            },
            "ACTION",
        )

    if "COMMUNICATE" in components:
        communicate_reward = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
        ).model_dump(mode="json")
        recomputed["COMMUNICATE"] = _serialized_deterministic_component(
            {
                **communicate_reward,
                "reward_basis": reward_info.get("reward_basis"),
            },
            "COMMUNICATE",
        )
    return recomputed


def sampling_score_attribution_issues(
    expected: dict[str, Any],
    actual: dict[str, Any],
    key: tuple[str, int],
    config: dict[str, Any] | None,
    task: Any = None,
) -> list[dict[str, Any]]:
    """Reject component drift not reproduced from its exact causal inputs."""
    expected_info = expected.get("reward_info")
    actual_info = actual.get("reward_info")
    if not isinstance(expected_info, dict) or not isinstance(actual_info, dict):
        return []
    expected_components = reward_components(expected)
    actual_components = reward_components(actual)
    participant_text_drift = _participant_text_transcript(
        expected
    ) != _participant_text_transcript(actual)
    issues: list[dict[str, Any]] = []
    differing_components = {
        component
        for component, field in REWARD_BASIS_COMPONENT_FIELDS.items()
        if expected_components[field] != actual_components[field]
    }
    # A copied or stale serialized outcome is just as dangerous as an outcome
    # that differs from the official run. Recompute every deterministic basis
    # component, plus any non-basis diagnostic that actually differs. RewardInfo
    # serializes unrelated components as empty/None placeholders, which are not
    # evaluator claims and must not expand the validation scope.
    basis_components = {
        component
        for basis in (
            expected_info.get("reward_basis"),
            actual_info.get("reward_basis"),
        )
        if isinstance(basis, list)
        for component in basis
        if isinstance(component, str)
    }
    deterministic_components = (differing_components | basis_components) & {
        "DB",
        "ENV_ASSERTION",
        "ACTION",
        "COMMUNICATE",
    }
    recomputed_expected: dict[str, dict[str, Any]] = {}
    recomputed_actual: dict[str, dict[str, Any]] = {}
    expected_recomputation_error = None
    actual_recomputation_error = None
    if deterministic_components:
        if task is None:
            expected_recomputation_error = "authoritative task unavailable"
            actual_recomputation_error = "authoritative task unavailable"
        else:
            try:
                recomputed_expected = _recompute_deterministic_components(
                    expected, task, deterministic_components
                )
            except Exception as exc:
                expected_recomputation_error = type(exc).__name__
            try:
                recomputed_actual = _recompute_deterministic_components(
                    actual, task, deterministic_components
                )
            except Exception as exc:
                actual_recomputation_error = type(exc).__name__

    for component in sorted(deterministic_components | differing_components):
        field = REWARD_BASIS_COMPONENT_FIELDS.get(component)
        issue = {
            "task_id": key[0],
            "trial": key[1],
            "component": component,
            "component_field": field,
        }
        if component != "NL_ASSERTION":
            expected_projection = _serialized_deterministic_component(
                expected_info, component
            )
            actual_projection = _serialized_deterministic_component(
                actual_info, component
            )
            static_identity_mismatch = False
            expected_static_identity = None
            actual_static_identity = None
            if component == "ACTION":
                expected_records = expected_info.get("action_checks")
                actual_records = actual_info.get("action_checks")
                expected_static_identity = (
                    [
                        {
                            "action": record.get("action"),
                            "tool_type": record.get("tool_type"),
                        }
                        if isinstance(record, dict)
                        else record
                        for record in expected_records
                    ]
                    if isinstance(expected_records, list)
                    else expected_records
                )
                actual_static_identity = (
                    [
                        {
                            "action": record.get("action"),
                            "tool_type": record.get("tool_type"),
                        }
                        if isinstance(record, dict)
                        else record
                        for record in actual_records
                    ]
                    if isinstance(actual_records, list)
                    else actual_records
                )
                static_identity_mismatch = (
                    expected_static_identity != actual_static_identity
                )
            if (
                static_identity_mismatch
                or (
                    component in differing_components
                    and component == "COMMUNICATE"
                    and not participant_text_drift
                )
                or expected_recomputation_error is not None
                or actual_recomputation_error is not None
                or recomputed_expected.get(component) != expected_projection
                or recomputed_actual.get(component) != actual_projection
            ):
                issues.append(
                    {
                        **issue,
                        "required_causal_input": (
                            "official and candidate component outcomes both exactly "
                            "reproduced by the offline evaluator from their trajectories"
                        ),
                        "expected_recomputation_error": (expected_recomputation_error),
                        "candidate_recomputation_error": actual_recomputation_error,
                        "serialized_official_outcome": expected_projection,
                        "recomputed_official_outcome": recomputed_expected.get(
                            component
                        ),
                        "serialized_candidate_outcome": actual_projection,
                        "recomputed_candidate_outcome": recomputed_actual.get(
                            component
                        ),
                        "official_static_identity": expected_static_identity,
                        "candidate_static_identity": actual_static_identity,
                    }
                )
            continue

        expected_nl = expected_info.get("nl_assertions")
        actual_nl = actual_info.get("nl_assertions")
        expected_nl_identity = (
            [record.get("nl_assertion") for record in expected_nl]
            if isinstance(expected_nl, list)
            and all(isinstance(record, dict) for record in expected_nl)
            else expected_nl
        )
        actual_nl_identity = (
            [record.get("nl_assertion") for record in actual_nl]
            if isinstance(actual_nl, list)
            and all(isinstance(record, dict) for record in actual_nl)
            else actual_nl
        )
        if component == "NL_ASSERTION" and (
            expected_nl_identity != actual_nl_identity
            or not valid_dated_task102_judge_route(actual, key, config)
        ):
            issues.append(
                {
                    **issue,
                    "required_causal_input": (
                        "validated dated task_102 NL-judge route for every NL outcome "
                        "change"
                    ),
                    "participant_text_drift": participant_text_drift,
                    "official_static_identity": expected_nl_identity,
                    "candidate_static_identity": actual_nl_identity,
                }
            )
    return issues


def text_preview(value: Any, limit: int = 500) -> Any:
    """Bound text-divergence diagnostics without changing comparisons."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def tool_output_diagnostic(value: Any) -> dict[str, Any]:
    """Describe a ToolMessage without dumping arbitrarily large content."""
    if isinstance(value, str):
        rendered = value
        value_type = "string"
    else:
        rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
        value_type = type(value).__name__
    encoded = rendered.encode("utf-8")
    return {
        "type": value_type,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preview": (
            rendered
            if len(rendered) <= MAX_TOOL_OUTPUT_PREVIEW
            else rendered[:MAX_TOOL_OUTPUT_PREVIEW] + "..."
        ),
    }


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


def _binary_reward(value: Any) -> float | None:
    """Return a canonical binary reward, rejecting bools and non-finite values."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        return None
    return float(value)


def _canonical_component_reward(
    reward_info: dict[str, Any],
    expected_info: dict[str, Any],
    component: str,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Recompute one reward component from its canonical evaluator records."""
    issues: list[dict[str, Any]] = []
    record_specs = {
        "ENV_ASSERTION": ("env_assertions", "met", "reward", "env_assertion"),
        "ACTION": ("action_checks", "action_match", "action_reward", "action"),
        "COMMUNICATE": ("communicate_checks", "met", None, "info"),
        "NL_ASSERTION": ("nl_assertions", "met", None, "nl_assertion"),
    }
    if component == "DB":
        db_check = reward_info.get("db_check")
        if not isinstance(db_check, dict):
            return None, [
                {
                    "field": "db_check",
                    "expected": "object with consistent db_match/db_reward",
                    "actual": db_check,
                }
            ]
        db_match = db_check.get("db_match")
        db_reward = _binary_reward(db_check.get("db_reward"))
        if not isinstance(db_match, bool) or db_reward is None:
            issues.append(
                {
                    "field": "db_check",
                    "expected": "boolean db_match and binary finite db_reward",
                    "actual": db_check,
                }
            )
            return None, issues
        canonical = 1.0 if db_match else 0.0
        if db_reward != canonical:
            issues.append(
                {
                    "field": "db_check.db_reward",
                    "expected_recombined": canonical,
                    "actual": db_check.get("db_reward"),
                }
            )
        return canonical, issues

    spec = record_specs.get(component)
    if spec is None:
        return None, [
            {
                "field": "reward_basis",
                "expected": "recognized reward component",
                "actual": component,
            }
        ]
    field, met_field, numeric_field, identity_field = spec
    records = reward_info.get(field)
    expected_records = expected_info.get(field)
    if not isinstance(expected_records, list):
        expected_records = [] if expected_records is None else expected_records
    if not isinstance(records, list):
        return None, [
            {
                "field": field,
                "expected": f"list with {len(expected_records)} evaluator record(s)",
                "actual": records,
            }
        ]
    if not isinstance(expected_records, list) or len(records) != len(expected_records):
        issues.append(
            {
                "field": f"{field}.length",
                "expected": len(expected_records)
                if isinstance(expected_records, list)
                else "list",
                "actual": len(records),
            }
        )

    component_reward = 1.0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(
                {
                    "field": f"{field}[{index}]",
                    "expected": "evaluator record object",
                    "actual": record,
                }
            )
            continue
        expected_record = (
            expected_records[index]
            if isinstance(expected_records, list)
            and index < len(expected_records)
            and isinstance(expected_records[index], dict)
            else None
        )
        if expected_record is None or record.get(identity_field) != expected_record.get(
            identity_field
        ):
            issues.append(
                {
                    "field": f"{field}[{index}].{identity_field}",
                    "expected": expected_record.get(identity_field)
                    if expected_record is not None
                    else "matching official evaluator record",
                    "actual": record.get(identity_field),
                }
            )
        if component == "ACTION" and expected_record is not None:
            for static_field in ("tool_type",):
                if record.get(static_field) != expected_record.get(static_field):
                    issues.append(
                        {
                            "field": f"{field}[{index}].{static_field}",
                            "expected": expected_record.get(static_field),
                            "actual": record.get(static_field),
                        }
                    )
        met = record.get(met_field)
        if not isinstance(met, bool):
            issues.append(
                {
                    "field": f"{field}[{index}].{met_field}",
                    "expected": "boolean",
                    "actual": met,
                }
            )
            continue
        canonical = 1.0 if met else 0.0
        if numeric_field is not None:
            serialized = _binary_reward(record.get(numeric_field))
            if serialized is None or serialized != canonical:
                issues.append(
                    {
                        "field": f"{field}[{index}].{numeric_field}",
                        "expected_recombined": canonical,
                        "actual": record.get(numeric_field),
                    }
                )
        component_reward *= canonical
    return component_reward, issues


def grading_integrity_issues(
    simulation: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check candidate reward serialization without requiring the same outcome."""
    issues: list[dict[str, Any]] = []
    reward_info = simulation.get("reward_info")
    expected_info = expected.get("reward_info")
    if not isinstance(reward_info, dict):
        return [{"field": "reward_info", "expected": "object", "actual": reward_info}]
    value = reward_info.get("reward")
    canonical_value = _binary_reward(value)
    if canonical_value is None:
        issues.append(
            {"field": "reward", "expected": "binary finite number", "actual": value}
        )
    basis = reward_info.get("reward_basis")
    expected_basis = (
        expected_info.get("reward_basis") if isinstance(expected_info, dict) else None
    )
    basis_is_valid = (
        isinstance(basis, list)
        and bool(basis)
        and all(isinstance(component, str) for component in basis)
    )
    if basis != expected_basis or not basis_is_valid:
        issues.append(
            {
                "field": "reward_basis",
                "expected": expected_basis,
                "actual": basis,
            }
        )
    breakdown = reward_info.get("reward_breakdown")
    if not isinstance(breakdown, dict):
        issues.append(
            {
                "field": "reward_breakdown",
                "expected": "object keyed by reward_basis",
                "actual": breakdown,
            }
        )
        return issues
    if basis_is_valid and set(breakdown) != set(basis):
        issues.append(
            {
                "field": "reward_breakdown.keys",
                "expected": sorted(basis),
                "actual": sorted(breakdown),
            }
        )
    canonical_breakdown: dict[str, float] = {}
    if basis_is_valid:
        for component in basis:
            component_reward, component_issues = _canonical_component_reward(
                reward_info,
                expected_info if isinstance(expected_info, dict) else {},
                component,
            )
            issues.extend(component_issues)
            if component_reward is not None:
                canonical_breakdown[component] = component_reward
                serialized_component = _binary_reward(breakdown.get(component))
                if serialized_component != component_reward:
                    issues.append(
                        {
                            "field": f"reward_breakdown.{component}",
                            "expected_recombined": component_reward,
                            "actual": breakdown.get(component),
                        }
                    )
    if basis_is_valid and len(canonical_breakdown) == len(basis):
        recombined = math.prod(canonical_breakdown[component] for component in basis)
        if canonical_value != recombined:
            issues.append(
                {
                    "field": "reward",
                    "expected_recombined": recombined,
                    "actual": value,
                }
            )
    return issues


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


def _tool_call_signature(call: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    """Return the exact call identity used to align outputs across reordering."""
    arguments = json.dumps(
        call.get("arguments"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return (
        call.get("role"),
        call.get("requestor"),
        call.get("name"),
        arguments,
    )


def _same_tool_outcome(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return expected["output_present"] == actual["output_present"] and (
        not expected["output_present"]
        or (
            expected["output"] == actual["output"]
            and expected["output_requestor"] == actual["output_requestor"]
            and expected["output_error"] == actual["output_error"]
        )
    )


def compare_aligned_tool_outputs(
    expected_calls: list[dict[str, Any]],
    actual_calls: list[dict[str, Any]],
    key: tuple[str, int],
    mismatch_counts: Counter[str],
    model_sampling_drift_counts: Counter[str],
    mismatch_details: list[dict[str, Any]],
    max_details: int,
) -> None:
    """Compare tool results by exact call identity, independent of call position.

    Calls added, removed, or reordered by model sampling may move positional
    indexes. Exact call identities are therefore aligned first. Equal-size
    groups of duplicate identities retain occurrence order because stateful
    repeated calls can legitimately return a sequence of different results.
    Duplicate identities with ambiguous non-matching results fail conservatively.
    """
    expected_groups: dict[
        tuple[Any, Any, Any, str], list[tuple[int, dict[str, Any]]]
    ] = {}
    actual_groups: dict[
        tuple[Any, Any, Any, str], list[tuple[int, dict[str, Any]]]
    ] = {}
    for index, call in enumerate(expected_calls):
        expected_groups.setdefault(_tool_call_signature(call), []).append((index, call))
    for index, call in enumerate(actual_calls):
        actual_groups.setdefault(_tool_call_signature(call), []).append((index, call))

    signatures = sorted(
        set(expected_groups) | set(actual_groups), key=lambda signature: repr(signature)
    )
    for signature in signatures:
        expected_group = expected_groups.get(signature, [])
        actual_group = actual_groups.get(signature, [])
        missing_suffix_after_exact_observed_prefix = len(expected_group) > len(
            actual_group
        ) and all(
            _same_tool_outcome(expected_call, actual_call)
            for (_, expected_call), (_, actual_call) in zip(
                expected_group, actual_group, strict=False
            )
        )
        if len(expected_group) == len(actual_group):
            # With no inserted/removed occurrence, positional order is causal.
            # Multiset matching would hide swapped stateful outputs.
            unmatched_expected = list(expected_group)
            unmatched_actual = list(actual_group)
        else:
            unmatched_actual = list(actual_group)
            unmatched_expected = []

            # When the model changed the occurrence count, remove exact outcomes
            # first so a clearly inserted/removed call can retain its narrow
            # downstream-output waiver. Any ambiguous remainder stays fatal.
            for expected_entry in expected_group:
                match_index = next(
                    (
                        index
                        for index, (_, actual_call) in enumerate(unmatched_actual)
                        if _same_tool_outcome(expected_entry[1], actual_call)
                    ),
                    None,
                )
                if match_index is None:
                    unmatched_expected.append(expected_entry)
                else:
                    unmatched_actual.pop(match_index)

        paired_count = min(len(unmatched_expected), len(unmatched_actual))
        for offset in range(paired_count):
            expected_index, expected_call = unmatched_expected[offset]
            actual_index, actual_call = unmatched_actual[offset]
            tool = f"{expected_call['role']}:{expected_call['name']}"
            if expected_call["output_present"] and not actual_call["output_present"]:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_missing",
                    key,
                    expected_call_index=expected_index,
                    actual_call_index=actual_index,
                    tool=tool,
                )
            elif not expected_call["output_present"] and actual_call["output_present"]:
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_output_unexpected",
                    key,
                    expected_call_index=expected_index,
                    actual_call_index=actual_index,
                    tool=tool,
                    actual=tool_output_diagnostic(actual_call["output"]),
                )
            elif expected_call["output_present"] and actual_call["output_present"]:
                if expected_call["output_requestor"] != actual_call["output_requestor"]:
                    add_mismatch(
                        mismatch_counts,
                        mismatch_details,
                        max_details,
                        "tool_output_requestor",
                        key,
                        expected_call_index=expected_index,
                        actual_call_index=actual_index,
                        tool=tool,
                        expected=expected_call["output_requestor"],
                        actual=actual_call["output_requestor"],
                    )
                if expected_call["output_error"] != actual_call["output_error"]:
                    add_mismatch(
                        mismatch_counts,
                        mismatch_details,
                        max_details,
                        "tool_output_error",
                        key,
                        expected_call_index=expected_index,
                        actual_call_index=actual_index,
                        tool=tool,
                        expected=expected_call["output_error"],
                        actual=actual_call["output_error"],
                    )
                if expected_call["output"] != actual_call["output"]:
                    mismatch_kind = (
                        KNOWN_DENSE_DRIFT_MISMATCH_KIND
                        if expected_call["role"] == "assistant"
                        and expected_call["name"] == "KB_search_dense"
                        else "tool_output"
                    )
                    add_mismatch(
                        mismatch_counts,
                        mismatch_details,
                        max_details,
                        mismatch_kind,
                        key,
                        expected_call_index=expected_index,
                        actual_call_index=actual_index,
                        tool=tool,
                        expected=tool_output_diagnostic(expected_call["output"]),
                        actual=tool_output_diagnostic(actual_call["output"]),
                    )

        remaining_expected = unmatched_expected[paired_count:]
        remaining_actual = unmatched_actual[paired_count:]
        for expected_index, expected_call in remaining_expected:
            if not expected_call["output_present"]:
                continue
            ambiguous_duplicate = (
                not missing_suffix_after_exact_observed_prefix
                and bool(actual_group)
                and not any(
                    _same_tool_outcome(expected_call, actual_call)
                    for _, actual_call in actual_group
                )
            )
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_missing",
                key,
                expected_call_index=expected_index,
                tool=f"{expected_call['role']}:{expected_call['name']}",
                note=(
                    "Ambiguous duplicate call has no matching result"
                    if ambiguous_duplicate
                    else "Model-selected call is absent from the candidate"
                ),
            )
            if not ambiguous_duplicate:
                model_sampling_drift_counts["tool_output_missing"] += 1
        for actual_index, actual_call in remaining_actual:
            if not actual_call["output_present"]:
                continue
            ambiguous_duplicate = bool(expected_group) and not any(
                _same_tool_outcome(expected_call, actual_call)
                for _, expected_call in expected_group
            )
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_unexpected",
                key,
                actual_call_index=actual_index,
                tool=f"{actual_call['role']}:{actual_call['name']}",
                actual=tool_output_diagnostic(actual_call["output"]),
                note=(
                    "Ambiguous duplicate call has no matching result"
                    if ambiguous_duplicate
                    else "Model selected an additional call"
                ),
            )
            if not ambiguous_duplicate:
                model_sampling_drift_counts["tool_output_unexpected"] += 1


def compare(
    candidate: dict[str, Any],
    reference: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
    *,
    compare_tools: bool,
    max_details: int,
    config: dict[str, Any] | None = None,
    tasks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare all requested keys and return a bounded machine-readable report."""
    verify_canonical_tau2_runtime()
    from pydantic import TypeAdapter

    from tau2.data_model.message import Message

    message_adapter = TypeAdapter(list[Message])
    candidate_index, duplicates = index_simulations(candidate["simulations"])
    expected_set = set(keys)
    mismatch_counts: Counter[str] = Counter()
    model_sampling_drift_counts: Counter[str] = Counter()
    mismatch_details: list[dict[str, Any]] = []
    grading_integrity_issue_count = 0
    grading_integrity_details: list[dict[str, Any]] = []
    text_divergence_messages: Counter[str] = Counter()
    text_divergence_simulations: Counter[str] = Counter()
    text_divergence_details: list[dict[str, Any]] = []
    attribution_issues: list[dict[str, Any]] = []

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
        integrity_issues = grading_integrity_issues(actual, expected)
        for source, simulation in (("reference", expected), ("candidate", actual)):
            try:
                parsed_messages = message_adapter.validate_python(
                    simulation.get("messages") or []
                )
            except Exception as exc:
                kind = f"message_schema_{source}"
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    kind,
                    key,
                    expected="official list[Message] schema",
                    actual_error_type=type(exc).__name__,
                )
                if source == "candidate":
                    integrity_issues.append(
                        {
                            "field": "messages",
                            "expected": "official list[Message] schema",
                            "actual_error_type": type(exc).__name__,
                        }
                    )
            else:
                require_user_terminal = simulation.get(
                    "termination_reason"
                ) == "user_stop" and any(
                    message.role in {"assistant", "user"}
                    and isinstance(getattr(message, "raw_data", None), dict)
                    for message in parsed_messages
                )
                protocol_issues = message_protocol_issues(
                    parsed_messages, require_user_terminal=require_user_terminal
                )
                if protocol_issues:
                    kind = f"message_protocol_{source}"
                    add_mismatch(
                        mismatch_counts,
                        mismatch_details,
                        max_details,
                        kind,
                        key,
                        expected="valid half-duplex participant/tool chronology",
                        actual=protocol_issues[:10],
                    )
                    if source == "candidate":
                        integrity_issues.append(
                            {
                                "field": "messages",
                                "expected": (
                                    "valid half-duplex participant/tool chronology"
                                ),
                                "actual": protocol_issues[:10],
                            }
                        )
        grading_integrity_issue_count += len(integrity_issues)
        for issue in integrity_issues:
            if len(grading_integrity_details) >= max_details:
                break
            grading_integrity_details.append(
                {"task_id": key[0], "trial": key[1], **issue}
            )
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
        attribution_issues.extend(
            sampling_score_attribution_issues(
                expected,
                actual,
                key,
                config,
                task=tasks.get(key[0]) if isinstance(tasks, dict) else None,
            )
        )
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
            model_sampling_drift_counts["tool_call_count"] += 1
        expected_sequence = [
            f"{call['role']}:{call['requestor']}:{call['name']}"
            for call in expected_calls
        ]
        actual_sequence = [
            f"{call['role']}:{call['requestor']}:{call['name']}"
            for call in actual_calls
        ]
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
            model_sampling_drift_counts["tool_call_sequence"] += 1
        for call_index, (expected_call, actual_call) in enumerate(
            zip(expected_calls, actual_calls, strict=False)
        ):
            if (
                expected_call["role"] == actual_call["role"]
                and expected_call["name"] == actual_call["name"]
                and expected_call["requestor"] != actual_call["requestor"]
            ):
                add_mismatch(
                    mismatch_counts,
                    mismatch_details,
                    max_details,
                    "tool_call_requestor",
                    key,
                    call_index=call_index,
                    tool=f"{expected_call['role']}:{expected_call['name']}",
                    expected=expected_call["requestor"],
                    actual=actual_call["requestor"],
                )
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
                model_sampling_drift_counts["tool_call_arguments"] += 1
        compare_aligned_tool_outputs(
            expected_calls,
            actual_calls,
            key,
            mismatch_counts,
            model_sampling_drift_counts,
            mismatch_details,
            max_details,
        )
        for output in expected_unpaired_outputs:
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_missing",
                key,
                expected=tool_output_diagnostic(output["content"]),
                note="Reference ToolMessage is not paired with a reference call",
            )
        for output in actual_unpaired_outputs:
            add_mismatch(
                mismatch_counts,
                mismatch_details,
                max_details,
                "tool_output_unpaired",
                key,
                actual=tool_output_diagnostic(output["content"]),
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
    model_sampling_drift_mismatch_count = sum(model_sampling_drift_counts.values())
    remaining_behavior_mismatch_count = (
        behavior_mismatch_count
        - known_dense_drift_mismatch_count
        - model_sampling_drift_mismatch_count
    )
    text_divergence_message_count = sum(text_divergence_messages.values())
    structural_mismatch_count = sum(mismatch_counts[kind] for kind in STRUCTURAL_KINDS)
    reward_vector_mismatch_count = mismatch_counts["reward"]
    aggregate_reward_parity = math.isclose(
        expected_reward_sum, candidate_reward_sum, abs_tol=1e-12
    )
    candidate_reward_by_trial_values = [
        candidate_reward_by_trial[trial] for trial in sorted(expected_reward_by_trial)
    ]
    expected_reward_by_trial_values = [
        expected_reward_by_trial[trial] for trial in sorted(expected_reward_by_trial)
    ]
    grading_integrity = grading_integrity_issue_count == 0
    aggregate_score_parity = (
        structural_mismatch_count == 0 and aggregate_reward_parity and grading_integrity
    )
    behavior_mismatch_counts = {
        kind: count
        for kind, count in sorted(mismatch_counts.items())
        if kind not in SCORING_KINDS
    }
    return {
        "score_parity": score_mismatch_count == 0,
        "strict_reproduction_parity": (
            score_mismatch_count == 0
            and behavior_mismatch_count == 0
            and text_divergence_message_count == 0
        ),
        "aggregate_score_parity": aggregate_score_parity,
        "aggregate_reward_parity": aggregate_reward_parity,
        "structural_parity": structural_mismatch_count == 0,
        "structural_mismatch_count": structural_mismatch_count,
        "reward_vector_parity": reward_vector_mismatch_count == 0,
        "reward_vector_mismatch_count": reward_vector_mismatch_count,
        "behavior_parity": behavior_mismatch_count == 0 if compare_tools else None,
        "strict_trace_parity": (
            behavior_mismatch_count == 0 and text_divergence_message_count == 0
            if compare_tools
            else None
        ),
        "behavior_parity_with_known_dense_drift_waiver": (
            non_waived_behavior_mismatch_count == 0 if compare_tools else None
        ),
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": (
            remaining_behavior_mismatch_count == 0 if compare_tools else None
        ),
        "behavior_checked": compare_tools,
        "expected_simulation_count": len(keys),
        "candidate_simulation_count": len(candidate_index),
        "compared_simulation_count": compared_count,
        "expected_reward_sum": expected_reward_sum,
        "expected_reward_by_trial": expected_reward_by_trial_values,
        "candidate_reward_sum": candidate_reward_sum,
        "candidate_reward_by_trial": candidate_reward_by_trial_values,
        "reward_by_trial_parity": (
            candidate_reward_by_trial_values == expected_reward_by_trial_values
        ),
        "candidate_pass_rate": (
            100.0 * candidate_reward_sum / len(keys) if keys else None
        ),
        "score_mismatch_count": score_mismatch_count,
        "component_parity_checked": compare_tools,
        "component_parity": component_mismatch_count == 0 if compare_tools else None,
        "component_mismatch_count": component_mismatch_count,
        "candidate_grading_integrity_checked": compare_tools,
        "candidate_grading_integrity": grading_integrity if compare_tools else None,
        "candidate_grading_integrity_issue_count": (
            grading_integrity_issue_count if compare_tools else 0
        ),
        "candidate_grading_integrity_details": (
            grading_integrity_details if compare_tools else []
        ),
        "candidate_grading_integrity_details_truncated": (
            grading_integrity_issue_count > len(grading_integrity_details)
            if compare_tools
            else False
        ),
        "sampling_score_attribution_checked": compare_tools,
        "sampling_score_attribution_valid": (
            not attribution_issues if compare_tools else None
        ),
        "sampling_score_attribution_issue_count": (
            len(attribution_issues) if compare_tools else 0
        ),
        "sampling_score_attribution_issues": (
            attribution_issues if compare_tools else []
        ),
        "behavior_mismatch_count": behavior_mismatch_count,
        "behavior_mismatch_counts": behavior_mismatch_counts,
        "known_dense_drift_waiver_scope": KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
        "known_dense_drift_mismatch_count": known_dense_drift_mismatch_count,
        "non_waived_behavior_mismatch_count": non_waived_behavior_mismatch_count,
        "model_sampling_drift_waiver_scope": MODEL_SAMPLING_DRIFT_WAIVER_SCOPE,
        "model_sampling_drift_mismatch_count": model_sampling_drift_mismatch_count,
        "model_sampling_drift_mismatch_counts": dict(
            sorted(model_sampling_drift_counts.items())
        ),
        "remaining_behavior_mismatch_count_after_waiver_scopes": (
            remaining_behavior_mismatch_count
        ),
        "tool_output_mismatch_count": tool_output_mismatch_count,
        "tool_call_mismatch_count": behavior_mismatch_count
        - tool_output_mismatch_count,
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "mismatch_details": mismatch_details,
        "mismatch_details_truncated": sum(mismatch_counts.values())
        > len(mismatch_details),
        "text_divergence_checked": compare_tools,
        "text_divergence_message_count": text_divergence_message_count,
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


def _is_fixed_initial_participant_message(message: dict[str, Any], index: int) -> bool:
    """Recognize the sole rawless participant message injected by the runner."""
    return (
        index == 0
        and message.get("role") == "assistant"
        and message.get("content") == INITIAL_ASSISTANT_GREETING
        and message.get("tool_calls") in (None, [])
        and message.get("raw_data") is None
        and message.get("usage") is None
        and message.get("cost", 0.0) == 0.0
    )


def participant_raw_response_binding_issues(
    message: dict[str, Any], *, response_key: str
) -> list[dict[str, Any]]:
    """Bind serialized participant output and usage to its provider response."""
    raw_data = message.get("raw_data")
    issues: list[dict[str, Any]] = []

    def issue(reason: str) -> None:
        issues.append({"response_key": response_key, "reason": reason})

    if not isinstance(raw_data, dict):
        issue("raw_data_not_object")
        return issues
    choices = raw_data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        issue("expected_exactly_one_raw_choice")
        return issues
    choice = choices[0]
    raw_message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(choice, dict) or not isinstance(raw_message, dict):
        issue("raw_choice_message_not_object")
        return issues
    if choice.get("index") != 0 or raw_message.get("role") != "assistant":
        issue("raw_choice_identity")
    if raw_message.get("content") != message.get("content"):
        issue("raw_choice_content")

    outer_calls = message.get("tool_calls")
    raw_calls = raw_message.get("tool_calls")
    expected_finish_reason = "tool_calls" if outer_calls else "stop"
    if choice.get("finish_reason") != expected_finish_reason:
        issue("raw_choice_finish_reason")
    if outer_calls is None:
        if raw_calls is not None:
            issue("raw_choice_tool_calls_presence")
    elif not isinstance(outer_calls, list) or not outer_calls:
        issue("serialized_tool_calls_shape")
    elif not isinstance(raw_calls, list) or len(raw_calls) != len(outer_calls):
        issue("raw_choice_tool_call_count")
    else:
        canonical_raw_calls = []
        malformed_raw_call = False
        for raw_call in raw_calls:
            function = raw_call.get("function") if isinstance(raw_call, dict) else None
            arguments = (
                function.get("arguments") if isinstance(function, dict) else None
            )
            parsed_arguments = parse_tool_arguments(arguments)
            if (
                not isinstance(raw_call, dict)
                or raw_call.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(raw_call.get("id"), str)
                or not isinstance(function.get("name"), str)
                or not isinstance(arguments, str)
                or not isinstance(parsed_arguments, dict)
            ):
                malformed_raw_call = True
                break
            canonical_raw_calls.append(
                {
                    "id": raw_call["id"],
                    "name": function["name"],
                    "arguments": parsed_arguments,
                }
            )
        canonical_outer_calls = [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("arguments"),
            }
            if isinstance(call, dict)
            else call
            for call in outer_calls
        ]
        if malformed_raw_call:
            issue("raw_choice_tool_call_shape")
        elif canonical_raw_calls != canonical_outer_calls:
            issue("raw_choice_tool_calls")

    raw_usage = raw_data.get("usage")
    outer_usage = message.get("usage")
    if not isinstance(raw_usage, dict) or not isinstance(outer_usage, dict):
        issue("usage_not_object")
    else:
        for field in ("prompt_tokens", "completion_tokens"):
            if raw_usage.get(field) != outer_usage.get(field):
                issue(f"usage_{field}")
        raw_cost = raw_usage.get("cost")
        outer_cost = message.get("cost")
        if (
            not isinstance(raw_cost, (int, float))
            or isinstance(raw_cost, bool)
            or not math.isfinite(float(raw_cost))
            or float(raw_cost) < 0.0
            or not isinstance(outer_cost, (int, float))
            or isinstance(outer_cost, bool)
            or not math.isfinite(float(outer_cost))
            or not math.isclose(float(raw_cost), float(outer_cost), abs_tol=1e-12)
        ):
            issue("usage_cost")
    return issues


def validate_raw_routes(
    candidate: dict[str, Any],
    keys: list[tuple[str, int]],
    bound_endpoint_inventory: Any = None,
) -> dict[str, Any]:
    """Validate that every paid participant response used the pinned provider."""
    expected_keys = set(keys)
    routes: Counter[tuple[Any, ...]] = Counter()
    missing_raw_data: Counter[str] = Counter()
    invalid_response_ids: Counter[str] = Counter()
    response_id_records: list[dict[str, Any]] = []
    response_id_keys: dict[str, list[str]] = {}
    response_id_counts_by_simulation = {
        key: Counter({"assistant": 0, "user": 0}) for key in expected_keys
    }
    usage_costs: dict[str, list[float]] = {"assistant": [], "user": []}
    invalid_usage_costs: Counter[str] = Counter()
    raw_response_binding_issues: list[dict[str, Any]] = []
    gpt52_alias_inventory_proof_mismatches = gpt52_alias_inventory_mismatches(
        bound_endpoint_inventory
    )
    gpt52_alias_inventory_proven = not gpt52_alias_inventory_proof_mismatches
    for simulation in candidate["simulations"]:
        if not isinstance(simulation, dict):
            continue
        try:
            key = simulation_key(simulation)
        except ComparisonError:
            continue
        if key not in expected_keys:
            continue
        for message_index, message in enumerate(simulation.get("messages") or []):
            if not isinstance(message, dict) or message.get("role") not in {
                "assistant",
                "user",
            }:
                continue
            role = message["role"]
            raw_data = message.get("raw_data")
            if not isinstance(raw_data, dict):
                if not _is_fixed_initial_participant_message(message, message_index):
                    missing_raw_data[role] += 1
                continue
            response_id = raw_data.get("id")
            response_key = f"{key[0]}:{key[1]}:{message_index}:{role}"
            raw_response_binding_issues.extend(
                participant_raw_response_binding_issues(
                    message, response_key=response_key
                )
            )
            if not isinstance(response_id, str) or not response_id.strip():
                invalid_response_ids[role] += 1
            else:
                response_id_records.append(
                    {
                        "task_id": key[0],
                        "trial": key[1],
                        "message_index": message_index,
                        "role": role,
                        "response_id": response_id,
                    }
                )
                response_id_keys.setdefault(response_id, []).append(response_key)
                response_id_counts_by_simulation[key][role] += 1
            routes[
                (
                    role,
                    raw_data.get("model"),
                    raw_data.get("provider"),
                    raw_data.get("service_tier"),
                )
            ] += 1
            usage = raw_data.get("usage")
            raw_cost = usage.get("cost") if isinstance(usage, dict) else None
            if (
                isinstance(raw_cost, (int, float))
                and not isinstance(raw_cost, bool)
                and math.isfinite(float(raw_cost))
                and float(raw_cost) >= 0.0
            ):
                usage_costs[role].append(float(raw_cost))
            else:
                invalid_usage_costs[role] += 1

    mismatches: dict[str, dict[str, Any]] = {}
    gpt52_alias_observed = False
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
            alias_route = normalized_model == GPT52_ALIAS_MODEL
            gpt52_alias_observed = gpt52_alias_observed or alias_route
            valid = (
                (
                    resolved_user_model in {"gpt-5.2-2025-12-11", "gpt-5.2-20251211"}
                    or (alias_route and gpt52_alias_inventory_proven)
                )
                and provider == "OpenAI"
                and service_tier == "default"
            )
            expected = {
                "model": (
                    "a dated GPT-5.2 response, or openai/gpt-5.2 only when the "
                    "bound endpoint inventory has the exact OpenAI dated-route proof"
                ),
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
        if invalid_response_ids[role]:
            mismatches[f"raw_routes.{role}.response_id"] = {
                "expected": "one nonblank response ID per attributed message",
                "actual": invalid_response_ids[role],
            }
        if invalid_usage_costs[role]:
            mismatches[f"raw_routes.{role}.usage_cost"] = {
                "expected": "one finite non-negative provider cost per attributed response",
                "actual_invalid_count": invalid_usage_costs[role],
            }
    duplicate_response_ids = [
        response_keys
        for response_keys in response_id_keys.values()
        if len(response_keys) > 1
    ]
    if duplicate_response_ids:
        mismatches["raw_routes.response_id_uniqueness"] = {
            "expected": "globally unique participant response IDs",
            "actual": duplicate_response_ids,
        }
    if raw_response_binding_issues:
        mismatches["raw_routes.response_binding"] = {
            "expected": "every serialized response exactly bound to raw choice/usage",
            "actual": raw_response_binding_issues,
        }
    response_id_simulation_rows = [
        {
            "task_id": task_id,
            "trial": trial,
            "assistant": response_id_counts_by_simulation[(task_id, trial)][
                "assistant"
            ],
            "user": response_id_counts_by_simulation[(task_id, trial)]["user"],
        }
        for task_id, trial in sorted(expected_keys)
    ]
    missing_simulation_coverage = [
        row
        for row in response_id_simulation_rows
        if row["assistant"] == 0 or row["user"] == 0
    ]
    if missing_simulation_coverage:
        mismatches["raw_routes.simulation_coverage"] = {
            "expected": "at least one attributed assistant and user response per simulation",
            "actual": missing_simulation_coverage,
        }
    response_id_records.sort(
        key=lambda record: (
            record["task_id"],
            record["trial"],
            record["message_index"],
            record["role"],
            record["response_id"],
        )
    )
    response_id_payload = json.dumps(
        response_id_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "raw_route_parity": not mismatches,
        "raw_route_mismatch_count": len(mismatches),
        "raw_route_mismatches": mismatches,
        "raw_route_counters": _route_counter_rows(routes),
        "raw_route_unattributed_generated_messages": dict(
            sorted(missing_raw_data.items())
        ),
        "raw_route_response_id_count": len(response_id_records),
        "raw_route_response_id_counts_by_role": dict(
            sorted(Counter(record["role"] for record in response_id_records).items())
        ),
        "raw_route_response_id_counts_by_simulation": response_id_simulation_rows,
        "raw_route_response_id_simulation_coverage_count": (
            len(response_id_simulation_rows) - len(missing_simulation_coverage)
        ),
        "raw_route_response_id_sha256": hashlib.sha256(response_id_payload).hexdigest(),
        "raw_response_binding_issue_count": len(raw_response_binding_issues),
        "raw_response_binding_issues": raw_response_binding_issues,
        "raw_route_gpt52_alias_observed": gpt52_alias_observed,
        "raw_route_gpt52_alias_inventory_proven": (
            gpt52_alias_inventory_proven if gpt52_alias_observed else None
        ),
        "raw_route_gpt52_alias_inventory_proof_mismatches": (
            gpt52_alias_inventory_proof_mismatches if gpt52_alias_observed else {}
        ),
        "raw_usage_cost_usd_by_role": {
            role: math.fsum(costs) for role, costs in sorted(usage_costs.items())
        },
        "raw_usage_cost_usd_total": math.fsum(
            cost for costs in usage_costs.values() for cost in costs
        ),
        "raw_usage_cost_message_counts": {
            role: len(costs) for role, costs in sorted(usage_costs.items())
        },
    }


def judge_route_observation(
    simulation: dict[str, Any], key: tuple[str, int]
) -> dict[str, Any]:
    """Extract the persisted NL-judge route without trusting evaluator output."""
    reward_info = simulation.get("reward_info")
    info = reward_info.get("info") if isinstance(reward_info, dict) else None
    judge = None
    if isinstance(info, dict):
        judge = info.get("judge")
        if judge is None and isinstance(info.get("nl"), dict):
            judge = info["nl"].get("judge")
    binding = judge_raw_response_binding(simulation, key)
    return {
        "task_id": key[0],
        "trial": key[1],
        "requested_model": (
            judge.get("requested_model") if isinstance(judge, dict) else None
        ),
        "resolved_model": (
            judge.get("resolved_model") if isinstance(judge, dict) else None
        ),
        "response_model": (
            judge.get("response_model") if isinstance(judge, dict) else None
        ),
        "provider": judge.get("provider") if isinstance(judge, dict) else None,
        "service_tier": (
            judge.get("service_tier") if isinstance(judge, dict) else None
        ),
        "response_id": judge.get("response_id") if isinstance(judge, dict) else None,
        "response_content_sha256": binding["response_content_sha256"],
        "raw_response_sha256": binding["raw_response_sha256"],
    }


def judge_raw_response_binding(
    simulation: dict[str, Any], key: tuple[str, int]
) -> dict[str, Any]:
    """Bind task 102's serialized NL checks to the retained provider response."""
    reward_info = simulation.get("reward_info")
    info = reward_info.get("info") if isinstance(reward_info, dict) else None
    judge = info.get("judge") if isinstance(info, dict) else None
    if judge is None and isinstance(info, dict) and isinstance(info.get("nl"), dict):
        judge = info["nl"].get("judge")
    issues: list[str] = []
    if not isinstance(judge, dict):
        return {
            "valid": False,
            "issues": ["judge_not_object"],
            "response_content_sha256": None,
            "raw_response_sha256": None,
        }
    raw_response = judge.get("raw_response")
    if not isinstance(raw_response, dict):
        return {
            "valid": False,
            "issues": ["raw_response_not_object"],
            "response_content_sha256": None,
            "raw_response_sha256": None,
        }
    raw_payload = json.dumps(
        raw_response,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    raw_response_sha256 = hashlib.sha256(raw_payload).hexdigest()
    if judge.get("response_id") != raw_response.get("id"):
        issues.append("response_id")
    if judge.get("response_model") != raw_response.get("model"):
        issues.append("response_model")
    if judge.get("provider") != raw_response.get("provider"):
        issues.append("provider")
    if judge.get("service_tier") != raw_response.get("service_tier"):
        issues.append("service_tier")
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        issues.append("choice_count")
        return {
            "valid": False,
            "issues": issues,
            "response_content_sha256": None,
            "raw_response_sha256": raw_response_sha256,
        }
    choice = choices[0]
    raw_message = choice.get("message") if isinstance(choice, dict) else None
    content = raw_message.get("content") if isinstance(raw_message, dict) else None
    if (
        not isinstance(choice, dict)
        or choice.get("index") != 0
        or choice.get("finish_reason") != "stop"
        or not isinstance(raw_message, dict)
        or raw_message.get("role") != "assistant"
        or raw_message.get("tool_calls") is not None
        or not isinstance(content, str)
        or not content.strip()
    ):
        issues.append("choice_shape")
    content_sha256 = (
        hashlib.sha256(content.encode("utf-8")).hexdigest()
        if isinstance(content, str)
        else None
    )
    try:
        result_data = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        result_data = None
    results = result_data.get("results") if isinstance(result_data, dict) else None
    if not isinstance(results, list):
        issues.append("choice_json")
    else:
        reconstructed = [
            {
                "nl_assertion": result.get("expectedOutcome"),
                "met": result.get("metExpectation"),
                "justification": result.get("reasoning"),
            }
            if isinstance(result, dict)
            else result
            for result in results
        ]
        serialized = (
            reward_info.get("nl_assertions") if isinstance(reward_info, dict) else None
        )
        if reconstructed != serialized:
            issues.append("serialized_nl_checks")
    usage = raw_response.get("usage")
    raw_cost = usage.get("cost") if isinstance(usage, dict) else None
    if (
        not isinstance(raw_cost, (int, float))
        or isinstance(raw_cost, bool)
        or not math.isfinite(float(raw_cost))
        or float(raw_cost) < 0.0
    ):
        issues.append("usage_cost")
    return {
        "valid": not issues,
        "issues": issues,
        "response_content_sha256": content_sha256,
        "raw_response_sha256": raw_response_sha256,
    }


def expected_judge_route(config: dict[str, Any]) -> dict[str, Any]:
    """Return the exact dated route required for task 102 NL scoring."""
    return {
        "requested_model": config["reproduction_transport"]["nl_assertions_model"],
        "resolved_model": "gpt-4.1-2025-04-14",
        "response_model": "openai/gpt-4.1 or the exact dated response model",
        "provider": "OpenAI",
        "service_tier": "default",
        "response_id": "non-empty string",
        "response_content_sha256": "bound raw judge choice",
        "raw_response_sha256": "bound raw judge response",
    }


def valid_dated_task102_judge_route(
    simulation: dict[str, Any],
    key: tuple[str, int],
    config: dict[str, Any] | None,
) -> bool:
    """Return whether task 102 records the exact allowed dated NL-judge route."""
    if key[0] != "task_102" or not isinstance(config, dict):
        return False
    try:
        expected = expected_judge_route(config)
    except (KeyError, TypeError):
        return False
    observation = judge_route_observation(simulation, key)
    binding = judge_raw_response_binding(simulation, key)
    resolved = _normalized_route_model(observation["resolved_model"])
    if isinstance(resolved, str):
        resolved = resolved.removeprefix("openai/")
    return (
        observation["requested_model"] == expected["requested_model"]
        and resolved == expected["resolved_model"]
        and _normalized_route_model(observation["response_model"])
        in {"openai/gpt-4.1", "openai/gpt-4.1-2025-04-14"}
        and observation["provider"] == expected["provider"]
        and observation["service_tier"] == expected["service_tier"]
        and isinstance(observation["response_id"], str)
        and bool(observation["response_id"].strip())
        and binding["valid"]
    )


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
    expected = expected_judge_route(config)
    observations = []
    mismatches: dict[str, dict[str, Any]] = {}
    response_id_keys: dict[str, list[tuple[str, int]]] = {}
    participant_response_id_keys: dict[str, list[str]] = {}
    for simulation in candidate["simulations"]:
        if not isinstance(simulation, dict):
            continue
        try:
            simulation_id = simulation_key(simulation)
        except ComparisonError:
            continue
        if simulation_id not in set(keys):
            continue
        for message_index, message in enumerate(simulation.get("messages") or []):
            if not isinstance(message, dict) or message.get("role") not in {
                "assistant",
                "user",
            }:
                continue
            raw_data = message.get("raw_data")
            response_id = raw_data.get("id") if isinstance(raw_data, dict) else None
            if isinstance(response_id, str) and response_id.strip():
                participant_response_id_keys.setdefault(response_id, []).append(
                    f"{simulation_id[0]}:{simulation_id[1]}:{message_index}:"
                    f"{message['role']}"
                )
    for key in sorted(task_keys):
        simulation = candidate_index.get(key)
        if simulation is None:
            continue
        observation = judge_route_observation(simulation, key)
        observations.append(observation)
        response_id = observation.get("response_id")
        if isinstance(response_id, str) and response_id.strip():
            response_id_keys.setdefault(response_id, []).append(key)
        if not valid_dated_task102_judge_route(simulation, key, config):
            mismatches[f"judge_route.{key[0]}.{key[1]}"] = {
                "expected": expected,
                "actual": observation,
            }
    duplicate_response_keys = [
        keys for keys in response_id_keys.values() if len(keys) > 1
    ]
    if duplicate_response_keys:
        mismatches["judge_route.response_id_uniqueness"] = {
            "expected": "a different nonblank response_id for every task_102 trial",
            "actual": [
                [f"{task_id}:{trial}" for task_id, trial in keys]
                for keys in duplicate_response_keys
            ],
        }
    cross_route_collisions = {
        response_id: {
            "judge": [f"{task_id}:{trial}" for task_id, trial in judge_keys],
            "participant": participant_response_id_keys[response_id],
        }
        for response_id, judge_keys in response_id_keys.items()
        if response_id in participant_response_id_keys
    }
    if cross_route_collisions:
        mismatches["judge_route.response_id_global_uniqueness"] = {
            "expected": "judge response IDs disjoint from participant response IDs",
            "actual": cross_route_collisions,
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
            "bound_openrouter_endpoint_inventory": None,
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
    if mode == "full":
        shell_receipt = config["reproduction_transport"][
            "full_shell_oracle_receipt_integrity"
        ]
        expected.update(
            {
                "known_full_shell_drift_acknowledged": bool(
                    shell_receipt["mismatch_command_count"]
                ),
                "full_shell_oracle_receipt": shell_receipt["file"],
                "full_shell_oracle_receipt_sha256": shell_receipt["file_sha256"],
                "full_shell_oracle_mismatch_count": shell_receipt[
                    "mismatch_command_count"
                ],
            }
        )
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
    if mode == "full":
        actual.update(
            {
                "known_full_shell_drift_acknowledged": manifest.get(
                    "known_full_shell_drift_acknowledged"
                ),
                "full_shell_oracle_receipt": manifest.get("full_shell_oracle_receipt"),
                "full_shell_oracle_receipt_sha256": manifest.get(
                    "full_shell_oracle_receipt_sha256"
                ),
                "full_shell_oracle_mismatch_count": manifest.get(
                    "full_shell_oracle_mismatch_count"
                ),
            }
        )
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
        "bound_openrouter_endpoint_inventory": manifest.get(
            "openrouter_endpoint_inventory"
        ),
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
    parser.add_argument(
        "--mode",
        choices=("smoke", "subset_trial0", "subset", "full"),
        required=True,
    )
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
        "--allow-model-sampling-drift",
        action="store_true",
        help=(
            "With --write-gate, accept exact aggregate subset score despite "
            "seed-replay trajectory drift; strict vector/component/behavior/text "
            "diagnostics remain in the report and backend drift remains fatal"
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
        if args.allow_model_sampling_drift and args.write_gate is None:
            raise ComparisonError(
                "--allow-model-sampling-drift is valid only with --write-gate"
            )
        config_path = args.config.resolve()
        if args.write_gate is not None and config_path != DEFAULT_CONFIG.resolve():
            raise ComparisonError(
                "Gate writing requires the committed canonical reference.json"
            )
        config = load_json(config_path)
        config_digest = digest_file(config_path)
        candidate_path = resolve_results_path(args.candidate)
        candidate = load_results(candidate_path)
        candidate_artifact_digest = digest_checkpoint_artifact(candidate_path)
        keys = expected_keys(config, args.mode)

        reference_path = None
        reference_results = None
        authoritative_tasks = None
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
            authoritative_tasks = load_authoritative_banking_tasks()
            missing_tasks = sorted(
                {task_id for task_id, _ in keys} - set(authoritative_tasks)
            )
            if missing_tasks:
                raise ComparisonError(
                    "Authoritative banking task definitions are missing: "
                    f"{missing_tasks}"
                )

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
                config=config,
                tasks=authoritative_tasks,
            ),
        }
        execution_manifest_report = validate_execution_manifest(
            candidate_path,
            candidate,
            config,
            config_digest,
            args.mode,
            candidate_artifact_digest,
        )
        bound_inventory = (
            execution_manifest_report["bound_openrouter_endpoint_inventory"]
            if execution_manifest_report["execution_manifest_parity"]
            else None
        )
        raw_route_report = validate_raw_routes(candidate, keys, bound_inventory)
        judge_route_report = validate_judge_routes(candidate, keys, config)
        report.update(raw_route_report)
        report.update(judge_route_report)
        report.update(execution_manifest_report)
        report["known_dense_drift_waiver_requested"] = args.allow_known_dense_drift
        report["model_sampling_drift_waiver_requested"] = (
            args.allow_model_sampling_drift
        )
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
            if args.allow_model_sampling_drift:
                if not report["aggregate_score_parity"]:
                    raise ComparisonError(
                        "Refusing to write gate: aggregate sampling-drift mode still "
                        "requires the exact official reward sum, exact task/trial keys, "
                        "seeds, user_stop terminations, and internally valid grading"
                    )
            elif not report["score_parity"]:
                raise ComparisonError(
                    "Refusing to write gate: strict subset reward/component vector parity failed"
                )
            if not report["configuration_parity"]:
                raise ComparisonError(
                    "Refusing to write gate: reproduction run configuration differs"
                )
            if not report["behavior_checked"]:
                raise ComparisonError(
                    "Refusing to write gate: tool behavior was not checked"
                )
            allowed_behavior_mismatch_count = 0
            if args.allow_known_dense_drift:
                allowed_behavior_mismatch_count += report[
                    "known_dense_drift_mismatch_count"
                ]
            if args.allow_model_sampling_drift:
                allowed_behavior_mismatch_count += report[
                    "model_sampling_drift_mismatch_count"
                ]
            if report["behavior_mismatch_count"] != allowed_behavior_mismatch_count:
                raise ComparisonError(
                    "Refusing to write gate: tool behavior contains backend or "
                    "serialization drift outside the explicitly requested dense/model "
                    "sampling scopes"
                )
            if (
                report["text_divergence_message_count"] > 0
                and not args.allow_model_sampling_drift
            ):
                raise ComparisonError(
                    "Refusing to write gate: generated participant text differs; pass "
                    "--allow-model-sampling-drift only for a proven nondeterministic replay"
                )
            if (
                not report["candidate_grading_integrity_checked"]
                or not report["candidate_grading_integrity"]
            ):
                raise ComparisonError(
                    "Refusing to write gate: candidate grading serialization is internally invalid"
                )
            if (
                not report["sampling_score_attribution_checked"]
                or not report["sampling_score_attribution_valid"]
                or report["sampling_score_attribution_issue_count"] != 0
                or report["sampling_score_attribution_issues"]
            ):
                raise ComparisonError(
                    "Refusing to write gate: a deterministic component differs from "
                    "its exact offline evaluator result or NL judge provenance is invalid"
                )
            if not args.allow_model_sampling_drift and (
                not report["component_parity_checked"] or not report["component_parity"]
            ):
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
                "candidate_simulation_count": report["candidate_simulation_count"],
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
                "score_parity": report["score_parity"],
                "strict_reproduction_parity": report["strict_reproduction_parity"],
                "aggregate_score_parity": report["aggregate_score_parity"],
                "aggregate_reward_parity": report["aggregate_reward_parity"],
                "structural_parity": report["structural_parity"],
                "structural_mismatch_count": report["structural_mismatch_count"],
                "reward_vector_parity": report["reward_vector_parity"],
                "reward_vector_mismatch_count": report["reward_vector_mismatch_count"],
                "reward_by_trial_parity": report["reward_by_trial_parity"],
                "score_mismatch_count": report["score_mismatch_count"],
                "mismatch_counts": report["mismatch_counts"],
                "configuration_parity": True,
                "configuration_mismatch_count": 0,
                "candidate_metadata": report["candidate_metadata"],
                "behavior_checked": report["behavior_checked"],
                "behavior_parity": report["behavior_parity"],
                "strict_trace_parity": report["strict_trace_parity"],
                "behavior_mismatch_count": report["behavior_mismatch_count"],
                "behavior_mismatch_counts": report["behavior_mismatch_counts"],
                "behavior_parity_with_known_dense_drift_waiver": report[
                    "behavior_parity_with_known_dense_drift_waiver"
                ],
                "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": report[
                    "behavior_parity_with_known_dense_and_model_sampling_drift_waivers"
                ],
                "known_dense_drift_waiver_requested": (args.allow_known_dense_drift),
                "known_dense_drift_waiver_applied": (
                    args.allow_known_dense_drift
                    and report["known_dense_drift_mismatch_count"] > 0
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
                "model_sampling_drift_waiver_requested": (
                    args.allow_model_sampling_drift
                ),
                "model_sampling_drift_waiver_applied": (
                    args.allow_model_sampling_drift
                    and (
                        not report["score_parity"]
                        or report["model_sampling_drift_mismatch_count"] > 0
                        or report["text_divergence_message_count"] > 0
                    )
                ),
                "model_sampling_aggregate_score_waiver_applied": (
                    args.allow_model_sampling_drift and not report["score_parity"]
                ),
                "model_sampling_drift_waiver_scope": report[
                    "model_sampling_drift_waiver_scope"
                ],
                "model_sampling_drift_mismatch_count": report[
                    "model_sampling_drift_mismatch_count"
                ],
                "model_sampling_drift_mismatch_counts": report[
                    "model_sampling_drift_mismatch_counts"
                ],
                "remaining_behavior_mismatch_count_after_waiver_scopes": report[
                    "remaining_behavior_mismatch_count_after_waiver_scopes"
                ],
                "text_divergence_message_count": report[
                    "text_divergence_message_count"
                ],
                "text_divergence_message_counts": report[
                    "text_divergence_message_counts"
                ],
                "component_parity_checked": report["component_parity_checked"],
                "component_parity": report["component_parity"],
                "component_mismatch_count": report["component_mismatch_count"],
                "candidate_grading_integrity_checked": report[
                    "candidate_grading_integrity_checked"
                ],
                "candidate_grading_integrity": report["candidate_grading_integrity"],
                "candidate_grading_integrity_issue_count": report[
                    "candidate_grading_integrity_issue_count"
                ],
                "sampling_score_attribution_checked": report[
                    "sampling_score_attribution_checked"
                ],
                "sampling_score_attribution_valid": report[
                    "sampling_score_attribution_valid"
                ],
                "sampling_score_attribution_issue_count": report[
                    "sampling_score_attribution_issue_count"
                ],
                "sampling_score_attribution_issues": report[
                    "sampling_score_attribution_issues"
                ],
                "execution_manifest_parity": report["execution_manifest_parity"],
                "execution_manifest_mismatch_count": report[
                    "execution_manifest_mismatch_count"
                ],
                "raw_route_parity": report["raw_route_parity"],
                "raw_route_mismatch_count": report["raw_route_mismatch_count"],
                "raw_route_counters": report["raw_route_counters"],
                "raw_route_unattributed_generated_messages": report[
                    "raw_route_unattributed_generated_messages"
                ],
                "raw_route_response_id_count": report["raw_route_response_id_count"],
                "raw_route_response_id_counts_by_role": report[
                    "raw_route_response_id_counts_by_role"
                ],
                "raw_route_response_id_counts_by_simulation": report[
                    "raw_route_response_id_counts_by_simulation"
                ],
                "raw_route_response_id_simulation_coverage_count": report[
                    "raw_route_response_id_simulation_coverage_count"
                ],
                "raw_route_response_id_sha256": report["raw_route_response_id_sha256"],
                "raw_response_binding_issue_count": report[
                    "raw_response_binding_issue_count"
                ],
                "raw_response_binding_issues": report["raw_response_binding_issues"],
                "raw_route_gpt52_alias_observed": report[
                    "raw_route_gpt52_alias_observed"
                ],
                "raw_route_gpt52_alias_inventory_proven": report[
                    "raw_route_gpt52_alias_inventory_proven"
                ],
                "raw_usage_cost_usd_by_role": report["raw_usage_cost_usd_by_role"],
                "raw_usage_cost_usd_total": report["raw_usage_cost_usd_total"],
                "raw_usage_cost_message_counts": report[
                    "raw_usage_cost_message_counts"
                ],
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
        if not report["configuration_parity"]:
            return 1
        if report.get("gate_written") is not None:
            return 0
        if not report["score_parity"]:
            return 1
        if report["behavior_checked"] and not report["strict_trace_parity"]:
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
