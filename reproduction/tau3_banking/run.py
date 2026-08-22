#!/usr/bin/env python3
"""Plan or launch guarded tau3 banking reproduction runs."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from state_fingerprint import (
    StateFingerprintError,
    capture_committed_runtime,
    capture_embedding_cache,
    capture_reproduction_state,
    digest_checkpoint_artifact,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_CREDENTIAL_CONFIG = Path.home() / ".rllm" / "config.json"
DEFAULT_GATE = HERE / ".state" / "subset_score_parity.json"
DEFAULT_RUNS_DIR = HERE / "runs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FULL_RUN_ENV = "ALLOW_FULL_RUN"
DEFAULT_MODAL_APP = "tau3-banking-sandboxes"
DEFAULT_MODAL_SANDBOX_TIMEOUT = 3600
MODAL_ORDER_MANIFEST_RELATIVE = (
    "reproduction/tau3_banking/subset_shell_order_manifest.json"
)
KNOWN_DENSE_DRIFT_WAIVER_SCOPE = (
    "paired assistant KB_search_dense ToolMessage content only"
)
ENDPOINT_INVENTORY_SCHEMA_VERSION = 2
FULL_GATE_SCHEMA_VERSION = 3
ENDPOINT_INVENTORY_SPECS = (
    {
        "requested_model": "qwen/qwen3.8-max",
        "response_model_id": "qwen/qwen3.8-max",
        "provider": "Alibaba",
        "resolved_model": "qwen/qwen3.8-max-20260803",
        "require_sole_active_endpoint": True,
    },
    {
        "requested_model": "openai/gpt-5.2",
        "response_model_id": "openai/gpt-5.2",
        "provider": "OpenAI",
        "resolved_model": "openai/gpt-5.2-20251211",
        "require_sole_active_endpoint": False,
    },
    {
        "requested_model": "openai/gpt-4.1-2025-04-14",
        "response_model_id": "openai/gpt-4.1",
        "provider": "OpenAI",
        "resolved_model": "openai/gpt-4.1-2025-04-14",
        "require_sole_active_endpoint": False,
    },
)
PREWARM_SCRIPT = (
    "from tau2.knowledge.embeddings_cache import warm_kb_cache; "
    "warm_kb_cache([('openai', {'model': 'text-embedding-3-large'})])"
)


class RunGuardError(RuntimeError):
    """Raised when a safety or parity precondition is not satisfied."""


def digest_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    """Return SHA-256 over a stable UTF-8 JSON representation."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with path-aware errors."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunGuardError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunGuardError(f"Expected a JSON object: {path}")
    return value


def load_openrouter_key(path: Path, environment: dict[str, str]) -> str:
    """Load the authoritative file key and reject a conflicting ambient key."""
    config = load_json(path)
    candidates = [
        (config.get("api_keys") or {}).get("openrouter")
        if isinstance(config.get("api_keys"), dict)
        else None,
        config.get("OPENROUTER_API_KEY"),
        config.get("openrouter_api_key"),
        (config.get("openrouter") or {}).get("api_key")
        if isinstance(config.get("openrouter"), dict)
        else None,
    ]
    keys = [candidate for candidate in candidates if isinstance(candidate, str)]
    if len(set(keys)) > 1:
        raise RunGuardError(
            f"Multiple different OpenRouter credentials found in {path}"
        )
    if not keys:
        raise RunGuardError(
            f"No OpenRouter credential found in {path}:api_keys.openrouter"
        )
    key = keys[0]
    if len(key) < 20 or any(character.isspace() for character in key):
        raise RunGuardError("OpenRouter credential has an invalid shape")
    ambient = environment.get("OPENROUTER_API_KEY")
    if ambient is not None and not hmac.compare_digest(ambient, key):
        raise RunGuardError(
            "Ambient OPENROUTER_API_KEY conflicts with the authoritative credential file"
        )
    return key


def expected_manifest_environment(
    config: dict[str, Any],
    *,
    modal_app: str = DEFAULT_MODAL_APP,
    modal_sandbox_timeout: int = DEFAULT_MODAL_SANDBOX_TIMEOUT,
) -> dict[str, str]:
    """Return the complete non-secret execution environment recorded in manifests."""
    configured_order_manifest = config["reproduction_transport"].get(
        "sandbox_order_manifest"
    )
    if configured_order_manifest != MODAL_ORDER_MANIFEST_RELATIVE:
        raise RunGuardError(
            "Reference config must bind the committed Modal shell-order manifest"
        )
    return {
        "OPENROUTER_API_KEY": "<loaded without logging>",
        "OPENAI_API_KEY": "<same OpenRouter key for SDK compatibility>",
        "OPENAI_BASE_URL": OPENROUTER_BASE_URL,
        "TAU2_SANDBOX_BACKEND": "modal",
        "TAU2_MODAL_APP": modal_app,
        "TAU2_MODAL_SANDBOX_TIMEOUT": str(modal_sandbox_timeout),
        "TAU2_MODAL_ORDER_MANIFEST": MODAL_ORDER_MANIFEST_RELATIVE,
        "TAU2_NL_ASSERTIONS_MODEL": config["reproduction_transport"][
            "nl_assertions_model"
        ],
        "TAU2_NL_ASSERTIONS_ARGS": json.dumps(
            config["reproduction_transport"]["nl_assertions_llm_args"],
            separators=(",", ":"),
        ),
    }


def expected_prompt_hashes(config: dict[str, Any]) -> dict[str, str]:
    """Hash the exact committed user guidelines and expanded environment policy."""
    recorded = config["recorded_run"]
    if recorded.get("retrieval_config") != "alltools" or recorded.get(
        "retrieval_config_kwargs"
    ) not in (None, {}):
        raise RunGuardError("Prompt hashing only supports the pinned alltools config")
    guidelines = (
        REPO_ROOT / "data/tau2/user_simulator/simulation_guidelines.md"
    ).read_text(encoding="utf-8")
    prompts = REPO_ROOT / "data/tau2/domains/banking_knowledge/prompts"
    policy = (prompts / "all_tools.md").read_text(encoding="utf-8")
    component_pattern = re.compile(r"\{\{component:(\w+)\}\}")

    def replace_component(match: re.Match[str]) -> str:
        return (prompts / "components" / f"{match.group(1)}.md").read_text(
            encoding="utf-8"
        )

    policy = component_pattern.sub(replace_component, policy)
    dense_instructions = (
        "The `KB_search_dense` tool uses **OpenAI API** with embedding model "
        "`text-embedding-3-large` for dense retrieval."
    )
    policy = policy.replace("{{all_tools_dense_instructions}}", dense_instructions)
    if "{{" in policy:
        raise RunGuardError("Expanded alltools policy still contains a placeholder")
    return {
        "user_global_simulation_guidelines_sha256": hashlib.sha256(
            guidelines.encode("utf-8")
        ).hexdigest(),
        "environment_policy_sha256": hashlib.sha256(policy.encode("utf-8")).hexdigest(),
    }


def endpoint_inventory_mismatches(inventory: Any) -> dict[str, Any]:
    """Validate the hashed, unauthenticated OpenRouter endpoint inventory."""
    mismatches: dict[str, Any] = {}
    if not isinstance(inventory, dict):
        return {"inventory": {"expected": "object", "actual": type(inventory).__name__}}
    if inventory.get("schema_version") != ENDPOINT_INVENTORY_SCHEMA_VERSION:
        mismatches["schema_version"] = {
            "expected": ENDPOINT_INVENTORY_SCHEMA_VERSION,
            "actual": inventory.get("schema_version"),
        }
    if not isinstance(inventory.get("fetched_at"), str) or not inventory["fetched_at"]:
        mismatches["fetched_at"] = {
            "expected": "non-empty UTC timestamp",
            "actual": inventory.get("fetched_at"),
        }
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        return {"entries": {"expected": "list", "actual": type(entries).__name__}}
    expected_by_model = {
        spec["requested_model"]: spec for spec in ENDPOINT_INVENTORY_SPECS
    }
    actual_by_model = {
        entry.get("requested_model"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("requested_model"), str)
    }
    if set(actual_by_model) != set(expected_by_model) or len(entries) != len(
        actual_by_model
    ):
        mismatches["models"] = {
            "expected": sorted(expected_by_model),
            "actual": sorted(actual_by_model),
        }
    for requested_model, spec in expected_by_model.items():
        entry = actual_by_model.get(requested_model)
        if not isinstance(entry, dict):
            continue
        expected_entry_fields = {
            "requested_model",
            "response_model_id",
            "provider",
            "resolved_model",
            "status",
            "active_endpoint_count",
            "matching_active_endpoint_count",
            "raw_sha256",
        }
        if set(entry) != expected_entry_fields:
            mismatches[f"{requested_model}.fields"] = {
                "expected": sorted(expected_entry_fields),
                "actual": sorted(entry),
            }
        expected_fields = {
            "response_model_id": spec["response_model_id"],
            "provider": spec["provider"],
            "resolved_model": spec["resolved_model"],
            "status": 0,
        }
        for field, expected in expected_fields.items():
            if entry.get(field) != expected:
                mismatches[f"{requested_model}.{field}"] = {
                    "expected": expected,
                    "actual": entry.get(field),
                }
        active_count = entry.get("active_endpoint_count")
        matching_count = entry.get("matching_active_endpoint_count")
        if spec["require_sole_active_endpoint"]:
            if active_count != 1:
                mismatches[f"{requested_model}.active_endpoint_count"] = {
                    "expected": 1,
                    "actual": active_count,
                }
            if matching_count != 1:
                mismatches[f"{requested_model}.matching_active_endpoint_count"] = {
                    "expected": 1,
                    "actual": matching_count,
                }
        else:
            if not _is_int(active_count) or active_count < 1:
                mismatches[f"{requested_model}.active_endpoint_count"] = {
                    "expected": "positive integer",
                    "actual": active_count,
                }
            if not _is_int(matching_count) or matching_count < 1:
                mismatches[f"{requested_model}.matching_active_endpoint_count"] = {
                    "expected": "positive integer",
                    "actual": matching_count,
                }
        raw_digest = entry.get("raw_sha256")
        if not isinstance(raw_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", raw_digest
        ):
            mismatches[f"{requested_model}.raw_sha256"] = {
                "expected": "64 lowercase hexadecimal characters",
                "actual": raw_digest,
            }
    expected_digest = canonical_digest(entries)
    if inventory.get("digest") != expected_digest:
        mismatches["digest"] = {
            "expected": expected_digest,
            "actual": inventory.get("digest"),
        }
    if inventory.get("unauthenticated") is not True:
        mismatches["unauthenticated"] = {
            "expected": True,
            "actual": inventory.get("unauthenticated"),
        }
    return mismatches


def validate_endpoint_inventory(inventory: Any) -> None:
    """Raise when an endpoint inventory cannot prove the pinned routes exist."""
    mismatches = endpoint_inventory_mismatches(inventory)
    if mismatches:
        raise RunGuardError(f"OpenRouter endpoint inventory is invalid: {mismatches}")


def fetch_openrouter_endpoint_inventory(
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Fetch the free public endpoint catalog without sending a credential."""
    entries = []
    for spec in ENDPOINT_INVENTORY_SPECS:
        url = f"{OPENROUTER_BASE_URL}/models/{spec['requested_model']}/endpoints"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "tau3-banking-parity-harness/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(10 * 1024 * 1024 + 1)
                status_code = getattr(response, "status", 200)
        except (OSError, urllib.error.URLError) as exc:
            raise RunGuardError(
                f"Cannot fetch public OpenRouter endpoint inventory for {spec['requested_model']}"
            ) from exc
        if status_code != 200 or len(raw) > 10 * 1024 * 1024:
            raise RunGuardError(
                f"Invalid public endpoint inventory response for {spec['requested_model']}"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunGuardError(
                f"Public endpoint inventory is not JSON for {spec['requested_model']}"
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list):
            raise RunGuardError(
                f"Public endpoint inventory has no endpoint list for {spec['requested_model']}"
            )
        active_endpoints = [
            endpoint
            for endpoint in endpoints
            if isinstance(endpoint, dict) and endpoint.get("status") == 0
        ]
        matches = [
            endpoint
            for endpoint in active_endpoints
            if endpoint.get("provider_name") == spec["provider"]
            and endpoint.get("name") == f"{spec['provider']} | {spec['resolved_model']}"
        ]
        if not matches:
            raise RunGuardError(
                f"Pinned provider endpoint is unavailable for {spec['requested_model']}"
            )
        if spec["require_sole_active_endpoint"] and (
            len(active_endpoints) != 1 or len(matches) != 1
        ):
            raise RunGuardError(
                "The unpinned official Qwen request is safe only while its sole "
                "active OpenRouter endpoint is the exact Alibaba snapshot; found "
                f"{len(active_endpoints)} active endpoint(s), {len(matches)} matching"
            )
        entries.append(
            {
                "requested_model": spec["requested_model"],
                "response_model_id": data.get("id"),
                "provider": spec["provider"],
                "resolved_model": spec["resolved_model"],
                "status": 0,
                "active_endpoint_count": len(active_endpoints),
                "matching_active_endpoint_count": len(matches),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    inventory = {
        "schema_version": ENDPOINT_INVENTORY_SCHEMA_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "unauthenticated": True,
        "entries": entries,
        "digest": canonical_digest(entries),
    }
    validate_endpoint_inventory(inventory)
    return inventory


def git_output(*args: str) -> str:
    """Run a read-only git command in the benchmark checkout."""
    process = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RunGuardError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return whether *ancestor* is in *descendant*'s history."""
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        return True
    if process.returncode == 1:
        return False
    raise RunGuardError(process.stderr.strip() or "git merge-base failed")


def verify_checkout(config: dict[str, Any]) -> dict[str, Any]:
    """Verify the immutable upstream base and banking data objects."""
    expected_commit = config["benchmark"]["git_commit"]
    actual_commit = git_output("rev-parse", "HEAD")
    if not git_is_ancestor(expected_commit, actual_commit):
        raise RunGuardError(
            f"Official base {expected_commit} is not an ancestor of HEAD {actual_commit}"
        )

    object_paths = {
        "tasks_tree": "data/tau2/domains/banking_knowledge/tasks",
        "documents_tree": "data/tau2/domains/banking_knowledge/documents",
        "prompts_tree": "data/tau2/domains/banking_knowledge/prompts",
        "database_blob": "data/tau2/domains/banking_knowledge/db.json",
    }
    actual_objects = {}
    for name, relative_path in object_paths.items():
        actual_objects[name] = git_output("rev-parse", f"HEAD:{relative_path}")
        expected_object = config["benchmark"]["dataset_git_objects"][name]
        if actual_objects[name] != expected_object:
            raise RunGuardError(
                f"Git object mismatch for {relative_path}: "
                f"{actual_objects[name]} != {expected_object}"
            )

    data_status = git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "data/tau2/domains/banking_knowledge",
    )
    if data_status:
        raise RunGuardError(
            "Banking data has uncommitted changes; restore exact v1.0.1 data first:\n"
            f"{data_status}"
        )
    return {
        "head": actual_commit,
        "official_base_commit": expected_commit,
        "official_base_is_ancestor": True,
        "dataset_git_objects": actual_objects,
        "worktree_status": git_output("status", "--short", "--untracked-files=all"),
    }


def verify_runtime_patches() -> None:
    """Refuse execution if OpenRouter embeddings or Modal selection are not wired."""
    retrieval_path = REPO_ROOT / "src/tau2/domains/banking_knowledge/retrieval.py"
    embedder_path = REPO_ROOT / "src/tau2/knowledge/embedders/openai_embedder.py"
    modal_path = REPO_ROOT / "src/tau2/knowledge/modal_sandbox_manager.py"
    modal_order_manifest_path = REPO_ROOT / MODAL_ORDER_MANIFEST_RELATIVE
    nl_judge_path = REPO_ROOT / "src/tau2/evaluator/evaluator_nl_assertions.py"
    if "TAU2_SANDBOX_BACKEND" not in retrieval_path.read_text(encoding="utf-8"):
        raise RunGuardError(
            "This checkout does not wire TAU2_SANDBOX_BACKEND; refusing to fall back "
            "to a local shell sandbox"
        )
    if "OPENAI_BASE_URL" not in embedder_path.read_text(encoding="utf-8"):
        raise RunGuardError(
            "This checkout does not route dense OpenAI embeddings through the "
            "configured base URL"
        )
    if not modal_path.is_file():
        raise RunGuardError("Modal sandbox implementation is missing")
    if not modal_order_manifest_path.is_file():
        raise RunGuardError(
            "The committed Modal shell-order manifest is missing: "
            f"{MODAL_ORDER_MANIFEST_RELATIVE}"
        )
    judge_source = nl_judge_path.read_text(encoding="utf-8")
    if not all(
        name in judge_source
        for name in ("TAU2_NL_ASSERTIONS_MODEL", "TAU2_NL_ASSERTIONS_ARGS")
    ):
        raise RunGuardError("This checkout does not wire the NL-judge environment")


def verify_modal_credentials(environment: dict[str, str]) -> None:
    """Check only for the presence, never the contents, of Modal credentials."""
    token_environment = bool(
        environment.get("MODAL_TOKEN_ID") and environment.get("MODAL_TOKEN_SECRET")
    )
    known_config = any(
        path.is_file()
        for path in (
            Path.home() / ".modal.toml",
            Path.home() / ".modal" / "config.json",
        )
    )
    if not token_environment and not known_config:
        raise RunGuardError(
            "No Modal credential source detected; run `modal setup` before execution"
        )


def full_gate_behavior_mismatches(gate: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict-or-narrow-waiver behavior fields in a gate receipt."""
    mismatches: dict[str, Any] = {}
    boolean_fields = (
        "behavior_parity",
        "known_dense_drift_waiver_requested",
        "known_dense_drift_waiver_applied",
    )
    invalid_boolean_fields = [
        field for field in boolean_fields if not isinstance(gate.get(field), bool)
    ]
    if invalid_boolean_fields:
        mismatches["behavior_waiver_boolean_fields"] = {
            "expected": "boolean",
            "actual": {field: gate.get(field) for field in invalid_boolean_fields},
        }

    strict_behavior_parity = gate.get("behavior_parity") is True
    waiver_requested = gate.get("known_dense_drift_waiver_requested") is True
    waiver_applied = gate.get("known_dense_drift_waiver_applied") is True
    behavior_mismatch_count = gate.get("behavior_mismatch_count")
    dense_mismatch_count = gate.get("known_dense_drift_mismatch_count")

    def non_negative_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    counts_are_valid = non_negative_integer(
        behavior_mismatch_count
    ) and non_negative_integer(dense_mismatch_count)
    if not counts_are_valid:
        mismatches["behavior_mismatch_counts"] = {
            "expected": "non-negative integer behavior and known-dense counts",
            "actual": {
                "behavior": behavior_mismatch_count,
                "known_dense": dense_mismatch_count,
            },
        }
    elif strict_behavior_parity:
        if behavior_mismatch_count != 0 or dense_mismatch_count != 0 or waiver_applied:
            mismatches["strict_behavior_receipt"] = {
                "expected": {
                    "behavior_mismatch_count": 0,
                    "known_dense_drift_mismatch_count": 0,
                    "known_dense_drift_waiver_applied": False,
                },
                "actual": {
                    "behavior_mismatch_count": behavior_mismatch_count,
                    "known_dense_drift_mismatch_count": dense_mismatch_count,
                    "known_dense_drift_waiver_applied": waiver_applied,
                },
            }
    elif not (
        waiver_requested
        and waiver_applied
        and dense_mismatch_count > 0
        and behavior_mismatch_count == dense_mismatch_count
    ):
        mismatches["known_dense_drift_waiver"] = {
            "expected": (
                "explicitly requested and applied, with every behavior mismatch "
                "classified as known dense ToolMessage content drift"
            ),
            "actual": {
                "behavior_parity": gate.get("behavior_parity"),
                "known_dense_drift_waiver_requested": waiver_requested,
                "known_dense_drift_waiver_applied": waiver_applied,
                "behavior_mismatch_count": behavior_mismatch_count,
                "known_dense_drift_mismatch_count": dense_mismatch_count,
            },
        }
    return mismatches


def verify_full_gate(
    gate_path: Path, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Require a current exact-score gate with no unwaived behavior drift."""
    if os.environ.get(FULL_RUN_ENV) != "1":
        raise RunGuardError(f"Full mode requires {FULL_RUN_ENV}=1")
    gate = load_json(gate_path)
    required = {
        "schema_version": FULL_GATE_SCHEMA_VERSION,
        "kind": "tau3_banking_subset_score_parity",
        "mode": "subset",
        "expected_simulation_count": config["modes"]["subset"][
            "expected_simulation_count"
        ],
        "expected_reward_sum": float(config["modes"]["subset"]["expected_reward_sum"]),
        "expected_reward_by_trial": [
            float(value)
            for value in config["modes"]["subset"]["expected_reward_by_trial"]
        ],
        "candidate_reward_sum": float(config["modes"]["subset"]["expected_reward_sum"]),
        "candidate_reward_by_trial": [
            float(value)
            for value in config["modes"]["subset"]["expected_reward_by_trial"]
        ],
        "score_parity": True,
        "score_mismatch_count": 0,
        "configuration_parity": True,
        "configuration_mismatch_count": 0,
        "behavior_checked": True,
        "behavior_parity_with_known_dense_drift_waiver": True,
        "known_dense_drift_waiver_scope": KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
        "non_waived_behavior_mismatch_count": 0,
        "component_parity_checked": True,
        "component_parity": True,
        "component_mismatch_count": 0,
        "execution_manifest_parity": True,
        "execution_manifest_mismatch_count": 0,
        "raw_route_parity": True,
        "raw_route_mismatch_count": 0,
        "judge_route_parity": True,
        "judge_route_mismatch_count": 0,
        "reference_config_sha256": digest_file(config_path),
        "reference_results_sha256": config["artifacts"]["trajectory"]["sha256"],
    }
    mismatches = {
        key: {"expected": expected, "actual": gate.get(key)}
        for key, expected in required.items()
        if gate.get(key) != expected
    }
    mismatches.update(full_gate_behavior_mismatches(gate))
    if mismatches:
        raise RunGuardError(f"Full-run gate is invalid or stale: {mismatches}")
    current_state = capture_reproduction_state(
        REPO_ROOT, require_clean=True, require_cache=True
    )
    gate_state = gate.get("execution_state")
    if (
        not isinstance(gate_state, dict)
        or gate_state.get("digest") != current_state["digest"]
    ):
        raise RunGuardError(
            "Full-run gate runtime/cache state is stale. Re-run the validated "
            "subset and write a new gate from the current clean commit and selected cache."
        )

    def bound_path(field: str) -> Path:
        value = gate.get(field)
        if not isinstance(value, str) or not value:
            raise RunGuardError(f"Full-run gate lacks bound path: {field}")
        path = Path(value)
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

    candidate_path = bound_path("candidate_artifact")
    candidate_digest = digest_checkpoint_artifact(candidate_path)
    if candidate_digest != gate.get("candidate_artifact_sha256"):
        raise RunGuardError(
            "Full-run gate candidate checkpoint was changed after comparison"
        )
    manifest_path = bound_path("execution_manifest")
    manifest_digest = digest_file(manifest_path)
    if manifest_digest != gate.get("execution_manifest_sha256"):
        raise RunGuardError(
            "Full-run gate execution manifest was changed after comparison"
        )
    manifest = load_json(manifest_path)
    if manifest.get("checkpoint_sha256") != candidate_digest:
        raise RunGuardError(
            "Bound execution manifest does not name the bound candidate checkpoint"
        )
    canonical_commands = {
        tuple(build_command(config, "subset", candidate_path.parent, resume=resume))
        for resume in (False, True)
    }
    manifest_requirements = {
        "mode": manifest.get("mode") == "subset",
        "dry_run": manifest.get("dry_run") is False,
        "status": manifest.get("status") == "completed",
        "exit_code": manifest.get("exit_code") == 0,
        "output_dir": manifest.get("output_dir") == str(candidate_path.parent),
        "reference_config": manifest.get("reference_config_sha256")
        == digest_file(config_path),
        "environment": manifest.get("environment")
        == expected_manifest_environment(config),
        "command": tuple(manifest.get("command") or ()) in canonical_commands,
        "prompt_hashes": manifest.get("prompt_hashes")
        == expected_prompt_hashes(config),
        "endpoint_inventory": not endpoint_inventory_mismatches(
            manifest.get("openrouter_endpoint_inventory")
        ),
    }
    failed_manifest_requirements = [
        name for name, passed in manifest_requirements.items() if not passed
    ]
    if failed_manifest_requirements:
        raise RunGuardError(
            "Bound execution manifest is not canonical: "
            + ", ".join(failed_manifest_requirements)
        )
    if (manifest.get("post_run_execution_state") or {}).get("digest") != current_state[
        "digest"
    ]:
        raise RunGuardError(
            "Bound execution manifest cache/runtime state differs from the current state"
        )
    return current_state


def target_task_ids(config: dict[str, Any], mode: str) -> list[str]:
    """Resolve the ordered task set for one guarded mode."""
    task_ids = config["modes"][mode]["task_ids"]
    return list(config["reward_vectors"] if task_ids == "all" else task_ids)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _checkpoint_metadata_mismatches(
    info: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    current_head: str,
) -> dict[str, dict[str, Any]]:
    """Return every serialized run-setting mismatch that auto-resume would merge."""
    recorded = config["recorded_run"]
    allowed_trials: Any = len(config["modes"][mode]["trials"])
    if mode == "subset":
        allowed_trials = {1, 4}
    expected = {
        "git_commit": current_head,
        "num_trials": allowed_trials,
        "max_steps": recorded["max_steps"],
        "max_errors": recorded["max_errors"],
        "seed": recorded["seed"],
        "text_streaming_config": {"chunk_by": "words", "chunk_size": 1},
        "speech_complexity": None,
        "audio_native_config": None,
        "retrieval_config": recorded["retrieval_config"],
        "retrieval_config_kwargs": recorded["retrieval_config_kwargs"],
        "user_info.implementation": recorded["user"]["implementation"],
        "user_info.llm": config["reproduction_transport"]["user_model"],
        "user_info.llm_args": config["reproduction_transport"]["user_llm_args"],
        "user_info.voice_settings": None,
        "user_info.persona_config": None,
        "agent_info.implementation": recorded["agent"]["implementation"],
        "agent_info.llm": config["reproduction_transport"]["agent_model"],
        "agent_info.llm_args": recorded["agent"]["llm_args"],
        "agent_info.voice_settings": None,
        "environment_info.domain_name": config["benchmark"]["domain"],
        "environment_info.tool_defs": None,
    }

    def nested(path: str) -> Any:
        value: Any = info
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    mismatches = {}
    for field, expected_value in expected.items():
        actual = nested(field)
        matches = (
            actual in expected_value
            if isinstance(expected_value, set)
            else actual == expected_value
        )
        if not matches:
            rendered_expected: Any = (
                sorted(expected_value)
                if isinstance(expected_value, set)
                else expected_value
            )
            mismatches[field] = {"expected": rendered_expected, "actual": actual}

    prompt_hashes = expected_prompt_hashes(config)
    for field, expected_digest in (
        (
            "user_info.global_simulation_guidelines",
            prompt_hashes["user_global_simulation_guidelines_sha256"],
        ),
        (
            "environment_info.policy",
            prompt_hashes["environment_policy_sha256"],
        ),
    ):
        actual = nested(field)
        actual_digest = (
            hashlib.sha256(actual.encode("utf-8")).hexdigest()
            if isinstance(actual, str)
            else None
        )
        if actual_digest != expected_digest:
            mismatches[field] = {
                "expected_sha256": expected_digest,
                "actual_sha256": actual_digest,
            }
    return mismatches


def _load_checkpoint(results_path: Path) -> dict[str, Any]:
    """Load monolithic or directory-format checkpoint simulations safely."""
    checkpoint = load_json(results_path)
    simulations = checkpoint.get("simulations")
    simulations_dir = results_path.parent / "simulations"
    if simulations_dir.is_dir():
        if simulations not in (None, []):
            raise RunGuardError(
                "Checkpoint mixes embedded simulations with a simulations directory"
            )
        files = sorted(simulations_dir.glob("*.json"))
        non_json = [
            path for path in simulations_dir.iterdir() if path.suffix != ".json"
        ]
        if non_json:
            raise RunGuardError(
                f"Checkpoint simulations directory contains unrelated files: {non_json}"
            )
        simulations = [load_json(path) for path in files]
        checkpoint["simulations"] = simulations
    if not isinstance(checkpoint.get("info"), dict):
        raise RunGuardError("Checkpoint info must be a JSON object")
    if not isinstance(checkpoint.get("tasks"), list):
        raise RunGuardError("Checkpoint tasks must be a JSON list")
    if not isinstance(checkpoint.get("simulations"), list):
        raise RunGuardError("Checkpoint simulations must be a JSON list")
    return checkpoint


def _validate_simulation_index(
    checkpoint: dict[str, Any], simulations_by_id: dict[str, dict[str, Any]]
) -> None:
    index = checkpoint.get("simulation_index")
    if index is None:
        return
    if not isinstance(index, list):
        raise RunGuardError("Checkpoint simulation_index must be null or a list")
    index_by_id: dict[str, dict[str, Any]] = {}
    index_keys: set[tuple[str, int]] = set()
    for entry in index:
        if not isinstance(entry, dict):
            raise RunGuardError("Every simulation_index entry must be an object")
        simulation_id = entry.get("id")
        task_id = entry.get("task_id")
        trial = entry.get("trial")
        if (
            not isinstance(simulation_id, str)
            or not isinstance(task_id, str)
            or not _is_int(trial)
        ):
            raise RunGuardError("simulation_index contains an invalid id/task/trial")
        if simulation_id in index_by_id or (task_id, trial) in index_keys:
            raise RunGuardError(
                "simulation_index contains duplicate IDs or task/trials"
            )
        index_by_id[simulation_id] = entry
        index_keys.add((task_id, trial))
    if set(index_by_id) != set(simulations_by_id):
        raise RunGuardError(
            "simulation_index IDs do not exactly match checkpoint simulations"
        )
    for simulation_id, simulation in simulations_by_id.items():
        entry = index_by_id[simulation_id]
        expected_reward = (
            simulation.get("reward_info", {}).get("reward")
            if isinstance(simulation.get("reward_info"), dict)
            else None
        )
        for field, expected in (
            ("task_id", simulation.get("task_id")),
            ("trial", simulation.get("trial")),
            ("termination_reason", simulation.get("termination_reason")),
            ("reward", expected_reward),
        ):
            actual = entry.get(field)
            matches = actual == expected
            if (
                field == "reward"
                and _is_finite_number(actual)
                and _is_finite_number(expected)
            ):
                matches = math.isclose(float(actual), float(expected), abs_tol=1e-12)
            if not matches:
                raise RunGuardError(
                    f"simulation_index {simulation_id} has stale {field}: "
                    f"{actual!r} != {expected!r}"
                )


def _validate_resume_manifest(
    results_path: Path,
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    current_state: dict[str, Any],
    modal_app: str,
    modal_sandbox_timeout: int,
) -> str:
    """Require provenance from an earlier guarded execution of this checkpoint."""
    allowed_source_modes = {mode}
    if mode == "subset":
        allowed_source_modes.add("smoke")
    expected_environment = expected_manifest_environment(
        config,
        modal_app=modal_app,
        modal_sandbox_timeout=modal_sandbox_timeout,
    )
    prompt_hashes = expected_prompt_hashes(config)
    expected_config_digest = digest_file(config_path)
    expected_checkpoint_digest = digest_checkpoint_artifact(results_path)
    failures = []
    manifests = sorted(output_dir.glob("reproduction_manifest_*.json"))
    for manifest_path in manifests:
        try:
            manifest = load_json(manifest_path)
            source_mode = manifest.get("mode")
            valid_commands = (
                {
                    tuple(build_command(config, source_mode, output_dir, resume=resume))
                    for resume in (False, True)
                }
                if source_mode in allowed_source_modes
                else set()
            )
            checks = {
                "source_mode": source_mode in allowed_source_modes,
                "executed": manifest.get("dry_run") is False,
                "output_dir": manifest.get("output_dir") == str(output_dir),
                "reference_config": manifest.get("reference_config_sha256")
                == expected_config_digest,
                "runtime_cache_state": (manifest.get("execution_state") or {}).get(
                    "digest"
                )
                == current_state["digest"],
                "post_run_runtime_cache_state": (
                    manifest.get("post_run_execution_state") or {}
                ).get("digest")
                == current_state["digest"],
                "checkpoint": manifest.get("checkpoint_sha256")
                == expected_checkpoint_digest,
                "environment": manifest.get("environment") == expected_environment,
                "command": tuple(manifest.get("command") or ()) in valid_commands,
                "prompt_hashes": manifest.get("prompt_hashes") == prompt_hashes,
                "endpoint_inventory": not endpoint_inventory_mismatches(
                    manifest.get("openrouter_endpoint_inventory")
                ),
            }
        except (RunGuardError, TypeError, KeyError, ValueError) as exc:
            failures.append(f"{manifest_path.name}: {exc}")
            continue
        failed = [name for name, passed in checks.items() if not passed]
        if not failed:
            return str(manifest_path)
        failures.append(f"{manifest_path.name}: {', '.join(failed)}")
    detail = "; ".join(failures) if failures else "no execution manifests found"
    raise RunGuardError(
        "Resume requires a prior guarded execution manifest with the same committed "
        f"runtime, config, command, and environment: {detail}"
    )


def validate_resume_checkpoint(
    results_path: Path,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    current_state: dict[str, Any],
    modal_app: str,
    modal_sandbox_timeout: int,
) -> dict[str, Any]:
    """Reject every checkpoint merge not proven compatible with the target run."""
    checkpoint = _load_checkpoint(results_path)
    info = checkpoint["info"]
    metadata_mismatches = _checkpoint_metadata_mismatches(
        info, config, mode, current_state["runtime"]["head"]
    )
    if metadata_mismatches:
        raise RunGuardError(
            f"Checkpoint metadata is incompatible with {mode}: {metadata_mismatches}"
        )

    target_tasks = set(target_task_ids(config, mode))
    target_trials = set(config["modes"][mode]["trials"])
    checkpoint_num_trials = info.get("num_trials")
    checkpoint_trials = (
        set(range(checkpoint_num_trials)) if _is_int(checkpoint_num_trials) else set()
    )
    task_ids = []
    for task in checkpoint["tasks"]:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise RunGuardError(
                "Every checkpoint task must be an object with a string id"
            )
        task_ids.append(task["id"])
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RunGuardError("Checkpoint tasks are empty or contain duplicate IDs")
    unrelated_tasks = set(task_ids) - target_tasks
    if unrelated_tasks:
        raise RunGuardError(
            f"Checkpoint contains tasks outside {mode}: {sorted(unrelated_tasks)}"
        )

    allowed_termination_reasons = {
        "user_stop",
        "agent_stop",
        "max_steps",
        "timeout",
        "too_many_errors",
        "agent_error",
        "user_error",
        "infrastructure_error",
        "context_window_exceeded",
        "unexpected_error",
    }
    derived_seeds = config["recorded_run"]["derived_trial_seeds"]
    simulations_by_id: dict[str, dict[str, Any]] = {}
    simulation_keys: set[tuple[str, int]] = set()
    for simulation in checkpoint["simulations"]:
        if not isinstance(simulation, dict):
            raise RunGuardError("Every checkpoint simulation must be an object")
        simulation_id = simulation.get("id")
        task_id = simulation.get("task_id")
        trial = simulation.get("trial")
        if (
            not isinstance(simulation_id, str)
            or not isinstance(task_id, str)
            or not _is_int(trial)
        ):
            raise RunGuardError("Checkpoint simulation has invalid id/task/trial")
        key = (task_id, trial)
        if simulation_id in simulations_by_id or key in simulation_keys:
            raise RunGuardError("Checkpoint contains duplicate simulation IDs or keys")
        if task_id not in target_tasks or task_id not in task_ids:
            raise RunGuardError(f"Checkpoint simulation has unrelated task: {task_id}")
        if trial not in target_trials:
            raise RunGuardError(f"Checkpoint simulation has unrelated trial: {key}")
        if trial not in checkpoint_trials:
            raise RunGuardError(
                f"Checkpoint simulation {key} is outside serialized num_trials="
                f"{checkpoint_num_trials}"
            )
        if simulation.get("seed") != derived_seeds[trial]:
            raise RunGuardError(
                f"Checkpoint simulation {key} has seed {simulation.get('seed')!r}; "
                f"expected {derived_seeds[trial]}"
            )
        termination = simulation.get("termination_reason")
        if termination not in allowed_termination_reasons:
            raise RunGuardError(
                f"Checkpoint simulation {key} has invalid termination_reason"
            )
        reward_info = simulation.get("reward_info")
        if termination == "infrastructure_error":
            if reward_info is not None:
                raise RunGuardError(
                    f"Infrastructure-error simulation {key} must have null reward_info"
                )
        else:
            if termination != "user_stop":
                raise RunGuardError(
                    f"Checkpoint simulation {key} completed with {termination}; "
                    "auto-resume would silently treat it as done"
                )
            if (
                not isinstance(reward_info, dict)
                or not _is_finite_number(reward_info.get("reward"))
                or not 0.0 <= float(reward_info["reward"]) <= 1.0
            ):
                raise RunGuardError(
                    f"Completed simulation {key} lacks a finite reward in [0, 1]"
                )
        if simulation.get("mode", "half_duplex") != "half_duplex":
            raise RunGuardError(f"Checkpoint simulation {key} is not half_duplex")
        simulations_by_id[simulation_id] = simulation
        simulation_keys.add(key)

    _validate_simulation_index(checkpoint, simulations_by_id)
    manifest = _validate_resume_manifest(
        results_path,
        results_path.parent,
        config_path,
        config,
        mode,
        current_state,
        modal_app,
        modal_sandbox_timeout,
    )
    return {
        "manifest": manifest,
        "task_count": len(task_ids),
        "simulation_count": len(simulations_by_id),
        "infrastructure_error_count": sum(
            simulation.get("termination_reason") == "infrastructure_error"
            for simulation in simulations_by_id.values()
        ),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a non-secret run manifest."""
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


def finalize_execution_manifest(
    plan: dict[str, Any],
    manifest_path: Path,
    results_path: Path,
    *,
    status: str,
    exit_code: int,
    require_cache: bool = True,
    exception_type: str | None = None,
) -> bool:
    """Persist checkpoint and post-state provenance even when a runner is interrupted."""
    plan["completed_at"] = datetime.now(timezone.utc).isoformat()
    plan["status"] = status
    plan["exit_code"] = exit_code
    if exception_type is not None:
        plan["runner_exception_type"] = exception_type
    finalization_errors: dict[str, str] = {}
    try:
        plan["checkpoint_sha256"] = (
            digest_checkpoint_artifact(results_path) if results_path.exists() else None
        )
    except (OSError, StateFingerprintError) as exc:
        plan["checkpoint_sha256"] = None
        finalization_errors["checkpoint"] = type(exc).__name__
    try:
        plan["post_run_execution_state"] = capture_reproduction_state(
            REPO_ROOT, require_clean=True, require_cache=require_cache
        )
    except (OSError, StateFingerprintError) as exc:
        plan["post_run_execution_state"] = None
        finalization_errors["post_run_execution_state"] = type(exc).__name__
    if finalization_errors:
        plan["finalization_errors"] = finalization_errors
        if status == "completed":
            plan["status"] = "finalization_failed"
            plan["exit_code"] = 2
    else:
        plan.pop("finalization_errors", None)
    write_json_atomic(manifest_path, plan)
    return not finalization_errors


def build_command(
    config: dict[str, Any], mode: str, output_dir: Path, *, resume: bool
) -> list[str]:
    """Build an argument-vector invocation with no shell interpolation."""
    recorded = config["recorded_run"]
    mode_config = config["modes"][mode]
    command = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "knowledge",
        "tau2",
        "run",
        "--domain",
        config["benchmark"]["domain"],
        "--agent",
        recorded["agent"]["implementation"],
        "--agent-llm",
        config["reproduction_transport"]["agent_model"],
        "--agent-llm-args",
        json.dumps(recorded["agent"]["llm_args"], separators=(",", ":")),
        "--user",
        recorded["user"]["implementation"],
        "--user-llm",
        config["reproduction_transport"]["user_model"],
        "--user-llm-args",
        json.dumps(
            config["reproduction_transport"]["user_llm_args"],
            separators=(",", ":"),
        ),
        "--retrieval-config",
        recorded["retrieval_config"],
        "--num-trials",
        str(len(mode_config["trials"])),
        "--max-steps",
        str(recorded["max_steps"]),
        "--max-errors",
        str(recorded["max_errors"]),
        "--max-concurrency",
        "1" if mode == "smoke" else str(recorded["max_concurrency"]),
        "--seed",
        str(recorded["seed"]),
        "--log-level",
        "ERROR",
        "--save-to",
        str(output_dir),
    ]
    if resume:
        command.append("--auto-resume")
    task_ids = mode_config["task_ids"]
    if task_ids == "all":
        task_ids = list(config["reward_vectors"])
    command.extend(["--task-ids", *task_ids])
    return command


def build_prewarm_command() -> list[str]:
    """Build the document-only cache warmup that must precede paid chat."""
    return [
        "uv",
        "run",
        "--offline",
        "--frozen",
        "--extra",
        "knowledge",
        "python",
        "-c",
        PREWARM_SCRIPT,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "subset", "full"))
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Reference JSON path"
    )
    parser.add_argument(
        "--credential-config",
        type=Path,
        default=DEFAULT_CREDENTIAL_CONFIG,
        help="JSON credential source; the key is never printed or passed on argv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory; defaults to runs/<mode>_<UTC timestamp>",
    )
    parser.add_argument(
        "--modal-app",
        default=DEFAULT_MODAL_APP,
        help="Modal app name used by the remote shell sandboxes",
    )
    parser.add_argument(
        "--modal-sandbox-timeout",
        type=int,
        default=DEFAULT_MODAL_SANDBOX_TIMEOUT,
        help="Lifetime in seconds for each per-simulation Modal sandbox",
    )
    parser.add_argument(
        "--cost-ceiling-usd",
        type=float,
        help="Preflight ceiling checked against historical chat cost; not a hard API cap",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch paid calls (default is a non-mutating dry run)",
    )
    parser.add_argument(
        "--confirm-paid-api-calls",
        action="store_true",
        help="Required with --execute to acknowledge paid OpenRouter and Modal calls",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=DEFAULT_GATE,
        help="Subset score-parity receipt required by full mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output results file after all normal guards",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Plan a run or launch it after all guards pass."""
    args = parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_json(config_path)
        checkout = verify_checkout(config)
        if args.modal_sandbox_timeout < 1:
            raise RunGuardError("--modal-sandbox-timeout must be positive")
        historical_cost = float(config["modes"][args.mode]["historical_chat_cost_usd"])
        if args.execute:
            if args.credential_config.expanduser().resolve() != (
                DEFAULT_CREDENTIAL_CONFIG.resolve()
            ):
                raise RunGuardError(
                    "Paid execution requires the authoritative ~/.rllm/config.json credential"
                )
            if not args.confirm_paid_api_calls:
                raise RunGuardError("--execute requires --confirm-paid-api-calls")
            if args.cost_ceiling_usd is None:
                raise RunGuardError("--execute requires an explicit --cost-ceiling-usd")
            if args.cost_ceiling_usd < historical_cost:
                raise RunGuardError(
                    f"Cost ceiling ${args.cost_ceiling_usd:.2f} is below the historical "
                    f"chat cost ${historical_cost:.2f}"
                )
            verify_runtime_patches()
            verify_modal_credentials(os.environ)
        execution_state = None
        if args.mode == "full":
            execution_state = verify_full_gate(args.gate.resolve(), config_path, config)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (DEFAULT_RUNS_DIR / f"{args.mode}_{timestamp}").resolve()
        )
        results_path = output_dir / "results.json"
        if results_path.exists() and not args.resume:
            raise RunGuardError(
                f"Results already exist at {results_path}; use --resume explicitly"
            )
        if args.resume and not results_path.exists():
            raise RunGuardError(f"Cannot resume missing results: {results_path}")

        cache_prewarm_required = False
        if args.execute and execution_state is None:
            capture_committed_runtime(REPO_ROOT, require_clean=True)
            if args.resume:
                execution_state = capture_reproduction_state(
                    REPO_ROOT, require_clean=True, require_cache=True
                )
            else:
                cache_state = capture_embedding_cache(REPO_ROOT, require_nonempty=False)
                cache_prewarm_required = cache_state["selected"] is None
                if not cache_prewarm_required:
                    execution_state = capture_reproduction_state(
                        REPO_ROOT, require_clean=True, require_cache=True
                    )
        resume_preflight = None
        if args.resume:
            current_state = (
                execution_state
                if execution_state is not None
                else capture_reproduction_state(
                    REPO_ROOT, require_clean=True, require_cache=True
                )
            )
            resume_preflight = validate_resume_checkpoint(
                results_path,
                config_path,
                config,
                args.mode,
                current_state,
                args.modal_app,
                args.modal_sandbox_timeout,
            )

        command = build_command(config, args.mode, output_dir, resume=args.resume)
        prewarm_command = build_prewarm_command()
        manifest_environment = expected_manifest_environment(
            config,
            modal_app=args.modal_app,
            modal_sandbox_timeout=args.modal_sandbox_timeout,
        )
        endpoint_inventory = (
            fetch_openrouter_endpoint_inventory() if args.execute else None
        )
        plan = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "dry_run": not args.execute,
            "historical_chat_cost_usd": historical_cost,
            "cost_ceiling_usd": args.cost_ceiling_usd,
            "cost_scope_caveat": config["historical_cost_observations_usd"]["scope"],
            "known_non_parity": config["reproduction_transport"]["known_non_parity"],
            "reference_config_sha256": digest_file(config_path),
            "prompt_hashes": expected_prompt_hashes(config),
            "checkout": checkout,
            "execution_state": execution_state,
            "cache_prewarm_required": cache_prewarm_required,
            "cache_prewarm_command": prewarm_command,
            "resume_preflight": resume_preflight,
            "output_dir": str(output_dir),
            "command": command,
            "environment": manifest_environment,
            "openrouter_endpoint_inventory": endpoint_inventory,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(f"\nargv: {shlex.join(command)}")
        if not args.execute:
            print("\nDry run only. No model, embedding, or Modal calls were made.")
            return 0

        key = load_openrouter_key(args.credential_config, os.environ)
        environment = os.environ.copy()
        environment.pop("OPENAI_API_BASE", None)
        environment.update(
            {
                "OPENROUTER_API_KEY": key,
                "OPENAI_API_KEY": key,
                **{
                    name: value
                    for name, value in manifest_environment.items()
                    if name not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}
                },
                "PYTHONUNBUFFERED": "1",
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"reproduction_manifest_{args.mode}.json"
        if cache_prewarm_required:
            plan["dry_run"] = False
            plan["status"] = "prewarming_embedding_cache"
            write_json_atomic(manifest_path, plan)
            try:
                prewarm = subprocess.run(
                    prewarm_command, cwd=REPO_ROOT, env=environment, check=False
                )
            except KeyboardInterrupt:
                finalize_execution_manifest(
                    plan,
                    manifest_path,
                    results_path,
                    status="embedding_cache_prewarm_interrupted",
                    exit_code=130,
                    require_cache=False,
                    exception_type="KeyboardInterrupt",
                )
                return 130
            except Exception as exc:
                finalize_execution_manifest(
                    plan,
                    manifest_path,
                    results_path,
                    status="embedding_cache_prewarm_exception",
                    exit_code=2,
                    require_cache=False,
                    exception_type=type(exc).__name__,
                )
                raise RunGuardError(
                    f"Embedding-cache prewarm subprocess raised {type(exc).__name__}"
                ) from exc
            if prewarm.returncode:
                finalize_execution_manifest(
                    plan,
                    manifest_path,
                    results_path,
                    status="embedding_cache_prewarm_failed",
                    exit_code=prewarm.returncode,
                    require_cache=False,
                )
                return prewarm.returncode
            try:
                execution_state = capture_reproduction_state(
                    REPO_ROOT, require_clean=True, require_cache=True
                )
            except (OSError, StateFingerprintError) as exc:
                finalize_execution_manifest(
                    plan,
                    manifest_path,
                    results_path,
                    status="embedding_cache_validation_failed",
                    exit_code=2,
                    require_cache=False,
                    exception_type=type(exc).__name__,
                )
                raise
            plan["execution_state"] = execution_state
            plan["cache_prewarm_completed"] = True
        plan["dry_run"] = False
        plan["status"] = "running"
        plan["checkpoint_sha256_before_run"] = (
            digest_checkpoint_artifact(results_path) if results_path.exists() else None
        )
        write_json_atomic(manifest_path, plan)
        try:
            process = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, check=False
            )
        except KeyboardInterrupt:
            finalize_execution_manifest(
                plan,
                manifest_path,
                results_path,
                status="interrupted",
                exit_code=130,
                exception_type="KeyboardInterrupt",
            )
            return 130
        except Exception as exc:
            finalize_execution_manifest(
                plan,
                manifest_path,
                results_path,
                status="runner_exception",
                exit_code=2,
                exception_type=type(exc).__name__,
            )
            raise RunGuardError(
                f"Benchmark runner subprocess raised {type(exc).__name__}"
            ) from exc
        finalized = finalize_execution_manifest(
            plan,
            manifest_path,
            results_path,
            status="completed" if process.returncode == 0 else "failed",
            exit_code=process.returncode,
        )
        if not finalized and process.returncode == 0:
            return 2
        if process.returncode == 0:
            print(
                "Compare with:\n"
                f"  {shlex.join([sys.executable, str(HERE / 'compare_results.py'), str(results_path), '--mode', args.mode])}"
            )
        return process.returncode
    except (
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
