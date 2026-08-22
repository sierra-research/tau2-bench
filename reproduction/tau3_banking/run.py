#!/usr/bin/env python3
"""Plan or launch guarded tau3 banking reproduction runs."""

from __future__ import annotations

import argparse
import fcntl
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
from collections import Counter
from contextlib import contextmanager
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
CANONICAL_TAU2_SRC = (REPO_ROOT / "src").resolve()
CANONICAL_TAU2_DATA = (REPO_ROOT / "data").resolve()


def pin_harness_python_environment() -> None:
    """Pin offline guards and paid children to this checkout's tau2 inputs."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["TAU2_DATA_DIR"] = str(CANONICAL_TAU2_DATA)
    canonical_src = str(CANONICAL_TAU2_SRC)
    sys.path[:] = [entry for entry in sys.path if entry != canonical_src]
    sys.path.insert(0, canonical_src)


pin_harness_python_environment()

DEFAULT_CONFIG = HERE / "reference.json"
DEFAULT_CREDENTIAL_CONFIG = Path.home() / ".rllm" / "config.json"
DEFAULT_GATE = HERE / ".state" / "subset_score_parity.json"
DEFAULT_RUNS_DIR = HERE / "runs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CREDITS_URL = f"{OPENROUTER_BASE_URL}/credits"
OPENROUTER_CREDIT_RECEIPT_SCHEMA_VERSION = 1
FULL_RUN_ENV = "ALLOW_FULL_RUN"
DEFAULT_MODAL_APP = "tau3-banking-sandboxes"
DEFAULT_MODAL_SANDBOX_TIMEOUT = 3600
MODAL_ORDER_MANIFEST_RELATIVE = (
    "reproduction/tau3_banking/full_shell_order_manifest.json"
)
MODAL_EXPECTED_IMAGE_ID_ENV = "TAU2_MODAL_EXPECTED_IMAGE_ID"
FULL_SHELL_ORACLE_RECEIPT_RELATIVE = (
    "reproduction/tau3_banking/artifacts/shell_oracle_full_active_manifest.json"
)
MODAL_ORDER_MANIFEST_SCHEMA_VERSION = 1
FULL_SHELL_ORACLE_SCHEMA_VERSION = 1
KNOWN_DENSE_DRIFT_WAIVER_SCOPE = (
    "paired assistant KB_search_dense ToolMessage content only"
)
MODEL_SAMPLING_DRIFT_WAIVER_SCOPE = (
    "generated assistant/user text and model-selected tool-call count, sequence, "
    "arguments, their downstream paired outputs, and resulting per-task reward/"
    "component differences only when every differing deterministic component is "
    "exactly reproduced by the official offline evaluator and every NL change has "
    "the validated dated judge route; exact task/trial coverage, seeds, user_stop, "
    "internally valid grading, and aggregate reward remain mandatory"
)
MODEL_SAMPLING_MISMATCH_KINDS = {
    "tool_call_count",
    "tool_call_sequence",
    "tool_call_arguments",
    "tool_output_missing",
    "tool_output_unexpected",
}
ENDPOINT_INVENTORY_SCHEMA_VERSION = 3
FULL_GATE_SCHEMA_VERSION = 7
INITIAL_ASSISTANT_GREETING = "Hi! How can I help you today?"
GPT52_ALIAS_MODEL = "openai/gpt-5.2"
GPT52_RESOLVED_MODEL = "openai/gpt-5.2-20251211"
GPT52_ALIAS_ACTIVE_ENDPOINT_COUNT = 4
GPT52_ALIAS_ELIGIBLE_ACTIVE_ENDPOINT_COUNT = 3
GPT52_ALIAS_MATCHING_ENDPOINT_COUNT = 3
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


def verify_canonical_tau2_runtime() -> None:
    """Fail closed if grading imports or task data escaped this checkout."""
    import tau2

    invalid_modules: dict[str, str | None] = {}
    tau2_file = getattr(tau2, "__file__", None)
    if tau2_file is None or not Path(tau2_file).resolve().is_relative_to(
        CANONICAL_TAU2_SRC
    ):
        invalid_modules["tau2"] = (
            str(Path(tau2_file).resolve()) if tau2_file is not None else None
        )
    for name, module in tuple(sys.modules.items()):
        if name != "tau2" and not name.startswith("tau2."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(CANONICAL_TAU2_SRC):
            invalid_modules[name] = str(resolved)
    if invalid_modules:
        raise RunGuardError(
            "Offline grading runtime is not pinned to the canonical checkout: "
            f"modules={invalid_modules}"
        )

    from tau2.utils import utils as tau2_utils

    data_dir = Path(tau2_utils.DATA_DIR).resolve()
    if data_dir != CANONICAL_TAU2_DATA:
        raise RunGuardError(
            "Offline grading runtime is not pinned to the canonical checkout: "
            f"data_dir={data_dir}"
        )


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


def fetch_openrouter_credit_state(
    key: str,
    required_usd: float,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Authenticate and prove enough OpenRouter credit before any paid child."""
    if not _is_finite_number(required_usd) or float(required_usd) < 0:
        raise RunGuardError("Required OpenRouter credit must be finite and nonnegative")
    request = urllib.request.Request(
        OPENROUTER_CREDITS_URL,
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "tau3-banking-parity-harness/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(1024 * 1024 + 1)
            status_code = getattr(response, "status", 200)
    except (OSError, urllib.error.URLError):
        raise RunGuardError("OpenRouter credit preflight is unavailable") from None
    if status_code != 200 or len(raw) > 1024 * 1024:
        raise RunGuardError("OpenRouter credit preflight returned an invalid response")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RunGuardError(
            "OpenRouter credit preflight returned invalid JSON"
        ) from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise RunGuardError("OpenRouter credit preflight returned malformed data")
    total_credits = data.get("total_credits")
    total_usage = data.get("total_usage")
    if (
        not _is_finite_number(total_credits)
        or not _is_finite_number(total_usage)
        or float(total_credits) < 0
        or float(total_usage) < 0
    ):
        raise RunGuardError("OpenRouter credit preflight returned invalid totals")
    total_credits_usd = float(total_credits)
    total_usage_usd = float(total_usage)
    remaining_usd = total_credits_usd - total_usage_usd
    if not math.isfinite(remaining_usd) or remaining_usd < float(required_usd):
        raise RunGuardError(
            "OpenRouter credit is below this mode's historical chat cost "
            f"(${remaining_usd:.6f} available; ${float(required_usd):.6f} required)"
        )
    # Only this explicit numeric allowlist is persisted. The credential, raw body,
    # response headers, and any account/key labels are intentionally discarded.
    return {
        "schema_version": OPENROUTER_CREDIT_RECEIPT_SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_credits_usd": total_credits_usd,
        "total_usage_usd": total_usage_usd,
        "remaining_usd": remaining_usd,
        "required_usd": float(required_usd),
        "sufficient": True,
    }


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
    modal_image_object_id = config["reproduction_transport"].get(
        "modal_image_object_id"
    )
    if (
        not isinstance(modal_image_object_id, str)
        or re.fullmatch(r"im-[A-Za-z0-9_-]+", modal_image_object_id) is None
    ):
        raise RunGuardError(
            "Reference config must bind a hydrated Modal image object ID"
        )
    return {
        "OPENROUTER_API_KEY": "<loaded without logging>",
        "OPENROUTER_API_BASE": OPENROUTER_BASE_URL,
        "OPENAI_API_KEY": "<same OpenRouter key for SDK compatibility>",
        "OPENAI_BASE_URL": OPENROUTER_BASE_URL,
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "TAU2_DATA_DIR": str(REPO_ROOT / "data"),
        "TAU2_SANDBOX_BACKEND": "modal",
        "TAU2_MODAL_APP": modal_app,
        "TAU2_MODAL_SANDBOX_TIMEOUT": str(modal_sandbox_timeout),
        "TAU2_MODAL_ORDER_MANIFEST": MODAL_ORDER_MANIFEST_RELATIVE,
        MODAL_EXPECTED_IMAGE_ID_ENV: modal_image_object_id,
        "TAU2_NL_ASSERTIONS_MODEL": config["reproduction_transport"][
            "nl_assertions_model"
        ],
        "TAU2_NL_ASSERTIONS_ARGS": json.dumps(
            config["reproduction_transport"]["nl_assertions_llm_args"],
            separators=(",", ":"),
        ),
    }


def build_paid_environment(
    key: str,
    manifest_environment: dict[str, str],
    ambient_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a paid child environment with both OpenRouter transports pinned."""
    if manifest_environment.get("OPENROUTER_API_BASE") != OPENROUTER_BASE_URL:
        raise RunGuardError("Manifest must pin OPENROUTER_API_BASE to OpenRouter")
    if manifest_environment.get("OPENAI_BASE_URL") != OPENROUTER_BASE_URL:
        raise RunGuardError("Manifest must pin OPENAI_BASE_URL to OpenRouter")
    ambient = os.environ if ambient_environment is None else ambient_environment
    safe_ambient_names = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "PATH",
        "TERM",
        "TMPDIR",
        "TZ",
    }
    environment = {
        name: value
        for name, value in ambient.items()
        if name in safe_ambient_names or name.startswith("LC_")
    }
    environment.update(
        {
            "OPENROUTER_API_KEY": key,
            "OPENAI_API_KEY": key,
            **{
                name: value
                for name, value in manifest_environment.items()
                if name not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}
            },
        }
    )
    return environment


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


def verify_canonical_paid_inputs(
    config_path: Path,
    config: dict[str, Any],
    *,
    modal_app: str,
    modal_sandbox_timeout: int,
) -> None:
    """Reject paid runs whose harness or Modal knobs are outside provenance."""
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RunGuardError(
            "Paid execution requires the committed canonical reference.json"
        )
    if modal_app != DEFAULT_MODAL_APP:
        raise RunGuardError(
            f"Paid execution requires the canonical Modal app {DEFAULT_MODAL_APP!r}"
        )
    expected_timeout = config["reproduction_transport"]["sandbox_timeout_seconds"]
    if (
        not _is_int(expected_timeout)
        or expected_timeout != DEFAULT_MODAL_SANDBOX_TIMEOUT
        or modal_sandbox_timeout != expected_timeout
    ):
        raise RunGuardError(
            "Paid execution requires the canonical Modal sandbox timeout"
        )


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
            "eligible_active_endpoint_count",
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
        eligible_count = entry.get("eligible_active_endpoint_count")
        matching_count = entry.get("matching_active_endpoint_count")
        counts_are_ordered = (
            _is_int(active_count)
            and _is_int(eligible_count)
            and _is_int(matching_count)
            and 0 <= matching_count <= eligible_count <= active_count
        )
        if not counts_are_ordered:
            mismatches[f"{requested_model}.endpoint_count_relationship"] = {
                "expected": (
                    "non-boolean integers satisfying 0 <= matching <= eligible <= active"
                ),
                "actual": {
                    "active": active_count,
                    "eligible": eligible_count,
                    "matching": matching_count,
                },
            }
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
            if eligible_count != 1:
                mismatches[f"{requested_model}.eligible_active_endpoint_count"] = {
                    "expected": 1,
                    "actual": eligible_count,
                }
        elif requested_model == GPT52_ALIAS_MODEL:
            if active_count != GPT52_ALIAS_ACTIVE_ENDPOINT_COUNT:
                mismatches[f"{requested_model}.active_endpoint_count"] = {
                    "expected": GPT52_ALIAS_ACTIVE_ENDPOINT_COUNT,
                    "actual": active_count,
                }
            if matching_count != GPT52_ALIAS_MATCHING_ENDPOINT_COUNT:
                mismatches[f"{requested_model}.matching_active_endpoint_count"] = {
                    "expected": GPT52_ALIAS_MATCHING_ENDPOINT_COUNT,
                    "actual": matching_count,
                }
            if eligible_count != GPT52_ALIAS_ELIGIBLE_ACTIVE_ENDPOINT_COUNT:
                mismatches[f"{requested_model}.eligible_active_endpoint_count"] = {
                    "expected": GPT52_ALIAS_ELIGIBLE_ACTIVE_ENDPOINT_COUNT,
                    "actual": eligible_count,
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
            if (
                not _is_int(eligible_count)
                or eligible_count < 1
                or eligible_count != matching_count
            ):
                mismatches[f"{requested_model}.eligible_active_endpoint_count"] = {
                    "expected": (
                        "positive integer equal to matching_active_endpoint_count"
                    ),
                    "actual": eligible_count,
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


def gpt52_alias_inventory_mismatches(inventory: Any) -> dict[str, Any]:
    """Require the exact catalog evidence that binds the GPT-5.2 moving alias."""
    mismatches = {
        f"inventory.{field}": detail
        for field, detail in endpoint_inventory_mismatches(inventory).items()
    }
    entries = inventory.get("entries") if isinstance(inventory, dict) else None
    matching_entries = [
        entry
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict) and entry.get("requested_model") == GPT52_ALIAS_MODEL
    ]
    if len(matching_entries) != 1:
        mismatches["gpt52_alias.entry_count"] = {
            "expected": 1,
            "actual": len(matching_entries),
        }
        return mismatches
    entry = matching_entries[0]
    expected = {
        "response_model_id": GPT52_ALIAS_MODEL,
        "provider": "OpenAI",
        "resolved_model": GPT52_RESOLVED_MODEL,
        "status": 0,
        "active_endpoint_count": GPT52_ALIAS_ACTIVE_ENDPOINT_COUNT,
        "eligible_active_endpoint_count": (GPT52_ALIAS_ELIGIBLE_ACTIVE_ENDPOINT_COUNT),
        "matching_active_endpoint_count": GPT52_ALIAS_MATCHING_ENDPOINT_COUNT,
    }
    for field, expected_value in expected.items():
        if entry.get(field) != expected_value:
            mismatches[f"gpt52_alias.{field}"] = {
                "expected": expected_value,
                "actual": entry.get(field),
            }
    return mismatches


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
        eligible_endpoints = [
            endpoint
            for endpoint in active_endpoints
            if endpoint.get("provider_name") == spec["provider"]
        ]
        matches = [
            endpoint
            for endpoint in eligible_endpoints
            if endpoint.get("name") == f"{spec['provider']} | {spec['resolved_model']}"
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
                "eligible_active_endpoint_count": len(eligible_endpoints),
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


def verify_modal_order_manifest(
    config: dict[str, Any], manifest_path: Path | None = None
) -> dict[str, Any]:
    """Bind the active Modal order fixture to its pinned config integrity fields."""
    transport = config.get("reproduction_transport")
    configured_path = (
        transport.get("sandbox_order_manifest") if isinstance(transport, dict) else None
    )
    fixture = (
        transport.get("sandbox_order_manifest_integrity")
        if isinstance(transport, dict)
        else None
    )
    if not isinstance(fixture, dict):
        raise RunGuardError(
            "Reference config lacks Modal order manifest integrity metadata"
        )
    if (
        configured_path != MODAL_ORDER_MANIFEST_RELATIVE
        or fixture.get("file") != MODAL_ORDER_MANIFEST_RELATIVE
    ):
        raise RunGuardError(
            "Reference config does not bind the active full Modal order manifest"
        )

    expected_sha_fields = {
        "file_sha256": fixture.get("file_sha256"),
        "order_sha256": fixture.get("order_sha256"),
        "corpus_export_sha256": fixture.get("corpus_export_sha256"),
    }
    malformed_sha_fields = [
        name
        for name, value in expected_sha_fields.items()
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ]
    expected_schema_version = fixture.get("schema_version")
    expected_entry_count = fixture.get("entry_count")
    valid_entry_count = (
        isinstance(expected_entry_count, int)
        and not isinstance(expected_entry_count, bool)
        and expected_entry_count > 0
    )
    if (
        malformed_sha_fields
        or expected_schema_version != MODAL_ORDER_MANIFEST_SCHEMA_VERSION
        or not valid_entry_count
    ):
        invalid = [*malformed_sha_fields]
        if expected_schema_version != MODAL_ORDER_MANIFEST_SCHEMA_VERSION:
            invalid.append("schema_version")
        if not valid_entry_count:
            invalid.append("entry_count")
        raise RunGuardError(
            "Active full Modal order manifest integrity metadata is malformed: "
            + ", ".join(invalid)
        )

    path = manifest_path or (REPO_ROOT / MODAL_ORDER_MANIFEST_RELATIVE)
    if not path.is_file():
        raise RunGuardError(f"The committed Modal order manifest is missing: {path}")
    actual_file_sha256 = digest_file(path)
    if actual_file_sha256 != expected_sha_fields["file_sha256"]:
        raise RunGuardError(
            "Active full Modal order manifest file SHA-256 does not match reference.json"
        )

    manifest = load_json(path)
    filenames = manifest.get("filenames")
    if manifest.get("schema_version") != expected_schema_version:
        raise RunGuardError("Active full Modal order manifest schema is invalid")
    if (
        not isinstance(filenames, list)
        or not filenames
        or not all(isinstance(name, str) and name for name in filenames)
        or len(filenames) != len(set(filenames))
    ):
        raise RunGuardError(
            "Active full Modal order manifest filenames are malformed or duplicated"
        )
    entry_count = manifest.get("entry_count")
    if entry_count != len(filenames) or entry_count != expected_entry_count:
        raise RunGuardError(
            "Active full Modal order manifest entry count does not match reference.json"
        )
    if manifest.get("document_count") != entry_count - 1:
        raise RunGuardError(
            "Active full Modal order manifest document count is invalid"
        )

    calculated_order_sha256 = hashlib.sha256(
        ("\n".join(filenames) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("order_sha256") != calculated_order_sha256
        or manifest.get("order_sha256") != expected_sha_fields["order_sha256"]
    ):
        raise RunGuardError(
            "Active full Modal order manifest order SHA-256 does not match reference.json"
        )
    if (
        manifest.get("corpus_export_sha256")
        != expected_sha_fields["corpus_export_sha256"]
    ):
        raise RunGuardError(
            "Active full Modal order manifest corpus checksum does not match reference.json"
        )
    return manifest


def verify_full_shell_oracle_receipt(
    config: dict[str, Any],
    *,
    allow_known_drift: bool,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Bind full-run authorization to the reviewed active Modal oracle."""
    transport = config.get("reproduction_transport")
    integrity = (
        transport.get("full_shell_oracle_receipt_integrity")
        if isinstance(transport, dict)
        else None
    )
    if not isinstance(integrity, dict):
        raise RunGuardError(
            "Reference config lacks full Modal shell-oracle receipt integrity metadata"
        )
    if integrity.get("file") != FULL_SHELL_ORACLE_RECEIPT_RELATIVE:
        raise RunGuardError(
            "Reference config does not bind the canonical full shell-oracle receipt"
        )
    expected_file_sha256 = integrity.get("file_sha256")
    if (
        not isinstance(expected_file_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None
    ):
        raise RunGuardError("Full shell-oracle receipt SHA-256 metadata is malformed")
    expected_modal_image_recipe_sha256 = integrity.get("modal_image_recipe_sha256")
    if (
        not isinstance(expected_modal_image_recipe_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_modal_image_recipe_sha256) is None
    ):
        raise RunGuardError(
            "Full shell-oracle Modal image recipe SHA-256 metadata is malformed"
        )
    expected_modal_image_object_id = integrity.get("modal_image_object_id")
    configured_modal_image_object_id = transport.get("modal_image_object_id")
    if (
        not isinstance(expected_modal_image_object_id, str)
        or re.fullmatch(r"im-[A-Za-z0-9_-]+", expected_modal_image_object_id) is None
        or configured_modal_image_object_id != expected_modal_image_object_id
    ):
        raise RunGuardError(
            "Full shell-oracle hydrated Modal image object ID metadata is malformed"
        )
    verify_canonical_tau2_runtime()
    from tau2.knowledge.modal_sandbox_manager import MODAL_IMAGE_RECIPE_SHA256

    if expected_modal_image_recipe_sha256 != MODAL_IMAGE_RECIPE_SHA256:
        raise RunGuardError(
            "Full shell-oracle receipt was produced by a different Modal image recipe"
        )

    order_integrity = transport.get("sandbox_order_manifest_integrity")
    expected_order_sha256 = (
        order_integrity.get("order_sha256")
        if isinstance(order_integrity, dict)
        else None
    )
    expected_reference_sha256 = (
        config.get("artifacts", {}).get("trajectory", {}).get("sha256")
    )
    expected_fields = {
        "schema_version": FULL_SHELL_ORACLE_SCHEMA_VERSION,
        "mode": "full",
        "scope": "all",
        "reference_sha256": expected_reference_sha256,
        "recorded_call_count": integrity.get("recorded_call_count"),
        "unique_command_count": integrity.get("unique_command_count"),
        "selected_recorded_call_count": integrity.get("recorded_call_count"),
        "selected_unique_command_count": integrity.get("unique_command_count"),
        "executed": True,
        "order_manifest_applied": True,
        "order_manifest_sha256": expected_order_sha256,
        "modal_image_recipe_sha256": expected_modal_image_recipe_sha256,
        "modal_image_object_id": expected_modal_image_object_id,
        "exact_command_count": integrity.get("exact_command_count"),
        "mismatch_command_count": integrity.get("mismatch_command_count"),
    }
    count_fields = (
        "recorded_call_count",
        "unique_command_count",
        "exact_command_count",
        "mismatch_command_count",
    )
    malformed_counts = [
        field
        for field in count_fields
        if not _is_int(expected_fields[field]) or expected_fields[field] < 0
    ]
    if (
        malformed_counts
        or not isinstance(expected_reference_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_reference_sha256) is None
        or not isinstance(expected_order_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_order_sha256) is None
        or expected_fields["recorded_call_count"] <= 0
        or expected_fields["unique_command_count"] <= 0
        or expected_fields["recorded_call_count"]
        < expected_fields["unique_command_count"]
        or expected_fields["exact_command_count"]
        + expected_fields["mismatch_command_count"]
        != expected_fields["unique_command_count"]
    ):
        raise RunGuardError(
            "Full shell-oracle receipt integrity counts or digests are malformed"
        )
    classification = integrity.get("mismatch_classification")
    reviewed_and_accepted = integrity.get("mismatches_reviewed_and_explicitly_accepted")
    score_impact_assessment = integrity.get("score_impact_assessment")
    mismatch_count = expected_fields["mismatch_command_count"]
    if mismatch_count and (
        not isinstance(classification, str)
        or not classification.strip()
        or reviewed_and_accepted is not True
        or not isinstance(score_impact_assessment, str)
        or not score_impact_assessment.strip()
    ):
        raise RunGuardError(
            "Known full shell-oracle drift lacks a committed review and acceptance"
        )

    path = receipt_path or (REPO_ROOT / FULL_SHELL_ORACLE_RECEIPT_RELATIVE)
    if not path.is_file() or path.is_symlink():
        raise RunGuardError(
            f"Full Modal shell-oracle receipt is missing or unsafe: {path}"
        )
    if digest_file(path) != expected_file_sha256:
        raise RunGuardError(
            "Full Modal shell-oracle receipt SHA-256 does not match reference.json"
        )
    receipt = load_json(path)
    mismatches = {
        field: {"expected": expected, "actual": receipt.get(field)}
        for field, expected in expected_fields.items()
        if receipt.get(field) != expected
    }
    exact_percent = receipt.get("exact_percent")
    expected_percent = (
        100.0
        * expected_fields["exact_command_count"]
        / expected_fields["unique_command_count"]
        if expected_fields["unique_command_count"]
        else 100.0
    )
    if not _is_finite_number(exact_percent) or not math.isclose(
        float(exact_percent), expected_percent, abs_tol=1e-12
    ):
        mismatches["exact_percent"] = {
            "expected": expected_percent,
            "actual": exact_percent,
        }
    details = receipt.get("mismatch_details")
    details_truncated = receipt.get("mismatch_details_truncated")
    if (
        not isinstance(details, list)
        or details_truncated is not False
        or len(details) != mismatch_count
    ):
        mismatches["mismatch_details"] = {
            "expected": f"all {mismatch_count} details, not truncated",
            "actual": {
                "count": len(details) if isinstance(details, list) else None,
                "truncated": details_truncated,
            },
        }
    if mismatches:
        raise RunGuardError(f"Full Modal shell-oracle receipt is invalid: {mismatches}")
    if mismatch_count and not allow_known_drift:
        raise RunGuardError(
            "The reviewed full shell oracle has known drift; pass "
            "--allow-known-full-shell-drift to acknowledge it explicitly"
        )
    return receipt


def verify_runtime_patches(config: dict[str, Any]) -> None:
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
    verify_modal_order_manifest(config, modal_order_manifest_path)
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
    """Validate strict diagnostics and the two explicit behavior-waiver scopes."""
    mismatches: dict[str, Any] = {}
    boolean_fields = (
        "score_parity",
        "strict_reproduction_parity",
        "aggregate_score_parity",
        "aggregate_reward_parity",
        "structural_parity",
        "reward_vector_parity",
        "reward_by_trial_parity",
        "component_parity",
        "candidate_grading_integrity",
        "sampling_score_attribution_checked",
        "sampling_score_attribution_valid",
        "behavior_parity",
        "strict_trace_parity",
        "behavior_parity_with_known_dense_drift_waiver",
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers",
        "known_dense_drift_waiver_requested",
        "known_dense_drift_waiver_applied",
        "model_sampling_drift_waiver_requested",
        "model_sampling_drift_waiver_applied",
        "model_sampling_aggregate_score_waiver_applied",
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
    strict_trace_parity = gate.get("strict_trace_parity") is True
    dense_requested = gate.get("known_dense_drift_waiver_requested") is True
    dense_applied = gate.get("known_dense_drift_waiver_applied") is True
    sampling_requested = gate.get("model_sampling_drift_waiver_requested") is True
    sampling_applied = gate.get("model_sampling_drift_waiver_applied") is True
    aggregate_score_waiver_applied = (
        gate.get("model_sampling_aggregate_score_waiver_applied") is True
    )
    behavior_mismatch_count = gate.get("behavior_mismatch_count")
    dense_mismatch_count = gate.get("known_dense_drift_mismatch_count")
    sampling_mismatch_count = gate.get("model_sampling_drift_mismatch_count")
    remaining_mismatch_count = gate.get(
        "remaining_behavior_mismatch_count_after_waiver_scopes"
    )
    text_mismatch_count = gate.get("text_divergence_message_count")
    attribution_issue_count = gate.get("sampling_score_attribution_issue_count")

    def non_negative_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    count_fields = {
        "behavior": behavior_mismatch_count,
        "known_dense": dense_mismatch_count,
        "model_sampling": sampling_mismatch_count,
        "remaining": remaining_mismatch_count,
        "text": text_mismatch_count,
        "score_attribution": attribution_issue_count,
    }
    counts_are_valid = all(
        non_negative_integer(value) for value in count_fields.values()
    )
    if not counts_are_valid:
        mismatches["behavior_mismatch_counts"] = {
            "expected": (
                "non-negative integer behavior, waiver, remaining, text, and score-"
                "attribution counts"
            ),
            "actual": count_fields,
        }
        return mismatches

    attribution_issues = gate.get("sampling_score_attribution_issues")
    if (
        not isinstance(attribution_issues, list)
        or len(attribution_issues) != attribution_issue_count
        or not all(isinstance(issue, dict) for issue in attribution_issues)
    ):
        mismatches["sampling_score_attribution_issues"] = {
            "expected": (
                f"exact list of {attribution_issue_count} score-attribution issue(s)"
            ),
            "actual": attribution_issues,
        }

    behavior_counts = gate.get("behavior_mismatch_counts")
    sampling_counts = gate.get("model_sampling_drift_mismatch_counts")
    text_counts = gate.get("text_divergence_message_counts")
    for field, value, expected_total in (
        ("behavior_mismatch_counts", behavior_counts, behavior_mismatch_count),
        (
            "model_sampling_drift_mismatch_counts",
            sampling_counts,
            sampling_mismatch_count,
        ),
        ("text_divergence_message_counts", text_counts, text_mismatch_count),
    ):
        valid_mapping = isinstance(value, dict) and all(
            isinstance(kind, str) and non_negative_integer(count)
            for kind, count in (value.items() if isinstance(value, dict) else ())
        )
        if field == "text_divergence_message_counts" and isinstance(value, dict):
            valid_mapping = valid_mapping and set(value) <= {"assistant", "user"}
        actual_total = sum(value.values()) if valid_mapping else None
        if not valid_mapping or actual_total != expected_total:
            mismatches[field] = {
                "expected": f"non-negative integer counts summing to {expected_total}",
                "actual": value,
            }

    if isinstance(behavior_counts, dict) and isinstance(sampling_counts, dict):
        invalid_sampling_types = {
            kind: count
            for kind, count in sampling_counts.items()
            if not non_negative_integer(count)
            or count > behavior_counts.get(kind, 0)
            or kind == "tool_output_dense_known_drift"
            or kind not in MODEL_SAMPLING_MISMATCH_KINDS
        }
        if invalid_sampling_types:
            mismatches["model_sampling_drift_type_scope"] = {
                "expected": (
                    "a subset of behavior mismatch counts excluding the separately "
                    "classified known-dense output kind"
                ),
                "actual": invalid_sampling_types,
            }

    all_counts = gate.get("mismatch_counts")
    valid_all_counts = isinstance(all_counts, dict) and all(
        isinstance(kind, str) and non_negative_integer(count)
        for kind, count in (all_counts.items() if isinstance(all_counts, dict) else ())
    )
    behavior_kinds = {
        "tool_call_count",
        "tool_call_sequence",
        "tool_call_requestor",
        "tool_call_arguments",
        "tool_output",
        "tool_output_dense_known_drift",
        "tool_output_missing",
        "tool_output_unexpected",
        "tool_output_unpaired",
        "tool_output_requestor",
        "tool_output_error",
    }
    scoring_kinds = {
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
        "db_component",
        "env_assertions",
        "action_checks",
        "nl_assertions",
        "communicate_checks",
        "reward_basis",
        "reward_breakdown",
    }
    if valid_all_counts and isinstance(behavior_counts, dict):
        unknown_kinds = set(all_counts) - behavior_kinds - scoring_kinds
        expected_behavior_counts = {
            kind: count for kind, count in all_counts.items() if kind in behavior_kinds
        }
        if unknown_kinds or behavior_counts != expected_behavior_counts:
            mismatches["behavior_mismatch_type_counts"] = {
                "expected": expected_behavior_counts,
                "actual": behavior_counts,
                "unknown": sorted(unknown_kinds),
            }
        score_count = sum(
            count for kind, count in all_counts.items() if kind in scoring_kinds
        )
        structural_count = sum(
            all_counts.get(kind, 0)
            for kind in (
                "duplicate_simulation",
                "missing_simulation",
                "unexpected_simulation",
                "seed",
                "termination_reason",
                "message_schema_reference",
                "message_schema_candidate",
                "message_protocol_reference",
                "message_protocol_candidate",
            )
        )
        component_count = sum(
            all_counts.get(kind, 0)
            for kind in (
                "db_component",
                "env_assertions",
                "action_checks",
                "nl_assertions",
                "communicate_checks",
                "reward_basis",
                "reward_breakdown",
            )
        )
        count_invariants = {
            "score_mismatch_count": score_count,
            "structural_mismatch_count": structural_count,
            "reward_vector_mismatch_count": all_counts.get("reward", 0),
            "component_mismatch_count": component_count,
        }
        invalid_count_invariants = {
            field: {"expected": expected, "actual": gate.get(field)}
            for field, expected in count_invariants.items()
            if gate.get(field) != expected
        }
        if invalid_count_invariants:
            mismatches["score_mismatch_count_invariants"] = invalid_count_invariants
    else:
        mismatches["mismatch_counts"] = {
            "expected": "non-negative integer mismatch counts",
            "actual": all_counts,
        }

    expected_dense_count = (
        behavior_counts.get("tool_output_dense_known_drift", 0)
        if isinstance(behavior_counts, dict)
        else None
    )
    expected_remaining = (
        behavior_mismatch_count - dense_mismatch_count - sampling_mismatch_count
    )
    invariants = {
        "score_parity": gate.get("score_parity")
        is (gate.get("score_mismatch_count") == 0),
        "strict_reproduction_parity": gate.get("strict_reproduction_parity")
        is (gate.get("score_parity") is True and strict_trace_parity),
        "structural_parity": gate.get("structural_parity")
        is (gate.get("structural_mismatch_count") == 0),
        "reward_vector_parity": gate.get("reward_vector_parity")
        is (gate.get("reward_vector_mismatch_count") == 0),
        "component_parity": gate.get("component_parity")
        is (gate.get("component_mismatch_count") == 0),
        "behavior_parity": strict_behavior_parity == (behavior_mismatch_count == 0),
        "strict_trace_parity": strict_trace_parity
        == (behavior_mismatch_count == 0 and text_mismatch_count == 0),
        "known_dense_count": dense_mismatch_count == expected_dense_count,
        "remaining_count": remaining_mismatch_count == expected_remaining,
        "known_dense_only_parity": gate.get(
            "behavior_parity_with_known_dense_drift_waiver"
        )
        is (behavior_mismatch_count - dense_mismatch_count == 0),
        "combined_scope_parity": gate.get(
            "behavior_parity_with_known_dense_and_model_sampling_drift_waivers"
        )
        is (expected_remaining == 0),
        "dense_applied": dense_applied
        == (dense_requested and dense_mismatch_count > 0),
        "sampling_applied": sampling_applied
        == (
            sampling_requested
            and (
                gate.get("score_parity") is not True
                or sampling_mismatch_count > 0
                or text_mismatch_count > 0
            )
        ),
        "aggregate_score_waiver_applied": aggregate_score_waiver_applied
        == (sampling_requested and gate.get("score_parity") is not True),
        "sampling_score_attribution": (
            gate.get("sampling_score_attribution_checked") is True
            and gate.get("sampling_score_attribution_valid")
            is (attribution_issue_count == 0)
        ),
    }
    failed_invariants = [name for name, valid in invariants.items() if not valid]
    if failed_invariants:
        mismatches["behavior_receipt_invariants"] = {
            "expected": "internally consistent strict and waiver fields",
            "actual": failed_invariants,
        }

    uncovered = []
    if dense_mismatch_count > 0 and not (dense_requested and dense_applied):
        uncovered.append("known_dense")
    if (sampling_mismatch_count > 0 or text_mismatch_count > 0) and not (
        sampling_requested and sampling_applied
    ):
        uncovered.append("model_sampling")
    if remaining_mismatch_count != 0:
        uncovered.append("outside_waiver_scopes")
    if (
        gate.get("sampling_score_attribution_checked") is not True
        or gate.get("sampling_score_attribution_valid") is not True
        or attribution_issue_count != 0
        or attribution_issues != []
    ):
        uncovered.append("sampling_score_attribution")
    if gate.get("score_parity") is not True:
        if not (
            sampling_requested
            and sampling_applied
            and aggregate_score_waiver_applied
            and gate.get("aggregate_score_parity") is True
            and gate.get("aggregate_reward_parity") is True
            and gate.get("structural_parity") is True
            and gate.get("candidate_grading_integrity") is True
        ):
            uncovered.append("aggregate_score")
    if uncovered:
        mismatches["behavior_waiver_coverage"] = {
            "expected": "every non-strict mismatch covered by its explicitly requested scope",
            "actual": uncovered,
        }
    return mismatches


def _candidate_raw_route_projection(
    candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Project raw participant provenance directly from the bound checkpoint."""
    expected_keys = {
        (task_id, trial)
        for task_id in target_task_ids(config, "subset")
        for trial in config["modes"]["subset"]["trials"]
    }
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
    for simulation in candidate.get("simulations") or []:
        if not isinstance(simulation, dict):
            continue
        key = (simulation.get("task_id"), simulation.get("trial"))
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
            fixed_initial = (
                message_index == 0
                and role == "assistant"
                and message.get("content") == INITIAL_ASSISTANT_GREETING
                and message.get("tool_calls") in (None, [])
                and raw_data is None
                and message.get("usage") is None
                and message.get("cost", 0.0) == 0.0
            )
            if not isinstance(raw_data, dict):
                if not fixed_initial:
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
            response_id = raw_data.get("id")
            response_key = f"{key[0]}:{key[1]}:{message_index}:{role}"
            if not isinstance(response_id, str) or not response_id.strip():
                invalid_response_ids[role] += 1
                continue
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
            usage = raw_data.get("usage")
            raw_cost = usage.get("cost") if isinstance(usage, dict) else None
            if _is_finite_number(raw_cost) and float(raw_cost) >= 0.0:
                usage_costs[role].append(float(raw_cost))
            else:
                invalid_usage_costs[role] += 1
    response_id_records.sort(
        key=lambda record: (
            record["task_id"],
            record["trial"],
            record["message_index"],
            record["role"],
            record["response_id"],
        )
    )
    counters = [
        {
            "role": role,
            "model": model,
            "provider": provider,
            "service_tier": service_tier,
            "count": count,
        }
        for (role, model, provider, service_tier), count in sorted(
            routes.items(), key=lambda item: repr(item[0])
        )
    ]
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
    return {
        "raw_route_counters": counters,
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
        "raw_route_response_id_sha256": canonical_digest(response_id_records),
        "raw_usage_cost_usd_by_role": {
            role: math.fsum(costs) for role, costs in sorted(usage_costs.items())
        },
        "raw_usage_cost_usd_total": math.fsum(
            cost for costs in usage_costs.values() for cost in costs
        ),
        "raw_usage_cost_message_counts": {
            role: len(costs) for role, costs in sorted(usage_costs.items())
        },
        "invalid_usage_costs": dict(sorted(invalid_usage_costs.items())),
        "invalid_response_ids": dict(sorted(invalid_response_ids.items())),
        "duplicate_response_id_keys": [
            keys for keys in response_id_keys.values() if len(keys) > 1
        ],
        "response_id_keys": response_id_keys,
        "missing_simulation_coverage": missing_simulation_coverage,
    }


def full_gate_raw_route_mismatches(
    gate: dict[str, Any],
    endpoint_inventory: Any,
    candidate: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate serialized route counters against the bound endpoint catalog."""
    mismatches: dict[str, Any] = {}
    counters = gate.get("raw_route_counters")
    if not isinstance(counters, list):
        return {
            "raw_route_counters": {
                "expected": "list",
                "actual": type(counters).__name__,
            }
        }
    role_totals = {"assistant": 0, "user": 0}
    alias_observed = False
    invalid_rows = []
    for row in counters:
        if not isinstance(row, dict):
            invalid_rows.append(row)
            continue
        role = row.get("role")
        model = row.get("model")
        provider = row.get("provider")
        service_tier = row.get("service_tier")
        count = row.get("count")
        normalized = (
            model.removeprefix("openrouter/") if isinstance(model, str) else model
        )
        if not _is_int(count) or count < 1 or role not in role_totals:
            invalid_rows.append(row)
            continue
        if role == "assistant":
            valid = (
                normalized == "qwen/qwen3.8-max"
                and provider == "Alibaba"
                and service_tier is None
            )
        else:
            alias_route = normalized == GPT52_ALIAS_MODEL
            alias_observed = alias_observed or alias_route
            resolved = (
                normalized.removeprefix("openai/")
                if isinstance(normalized, str)
                else normalized
            )
            valid = (
                provider == "OpenAI"
                and service_tier == "default"
                and (
                    resolved in {"gpt-5.2-2025-12-11", "gpt-5.2-20251211"}
                    or (
                        alias_route
                        and not gpt52_alias_inventory_mismatches(endpoint_inventory)
                    )
                )
            )
        if not valid:
            invalid_rows.append(row)
            continue
        role_totals[role] += count
    if invalid_rows:
        mismatches["raw_route_counters"] = {
            "expected": "only pinned Qwen and proven OpenAI GPT-5.2 routes",
            "actual": invalid_rows,
        }
    if any(count == 0 for count in role_totals.values()):
        mismatches["raw_route_coverage"] = {
            "expected": "at least one attributed assistant and user response",
            "actual": role_totals,
        }
    expected_alias_proven = (
        not gpt52_alias_inventory_mismatches(endpoint_inventory)
        if alias_observed
        else None
    )
    if gate.get("raw_route_gpt52_alias_observed") is not alias_observed:
        mismatches["raw_route_gpt52_alias_observed"] = {
            "expected": alias_observed,
            "actual": gate.get("raw_route_gpt52_alias_observed"),
        }
    if gate.get("raw_route_gpt52_alias_inventory_proven") is not expected_alias_proven:
        mismatches["raw_route_gpt52_alias_inventory_proven"] = {
            "expected": expected_alias_proven,
            "actual": gate.get("raw_route_gpt52_alias_inventory_proven"),
        }
    unattributed = gate.get("raw_route_unattributed_generated_messages")
    if unattributed != {}:
        mismatches["raw_route_unattributed_generated_messages"] = {
            "expected": {},
            "actual": unattributed,
        }
    response_id_count = gate.get("raw_route_response_id_count")
    response_id_counts = gate.get("raw_route_response_id_counts_by_role")
    response_id_simulation_coverage_count = gate.get(
        "raw_route_response_id_simulation_coverage_count"
    )
    response_id_sha256 = gate.get("raw_route_response_id_sha256")
    valid_response_id_counts = (
        _is_int(response_id_count)
        and response_id_count > 0
        and isinstance(response_id_counts, dict)
        and set(response_id_counts) == {"assistant", "user"}
        and all(_is_int(count) and count > 0 for count in response_id_counts.values())
        and sum(response_id_counts.values()) == response_id_count
    )
    if not valid_response_id_counts:
        mismatches["raw_route_response_id_counts"] = {
            "expected": "positive assistant/user counts summing to the total",
            "actual": {
                "count": response_id_count,
                "by_role": response_id_counts,
            },
        }
    if (
        not _is_int(response_id_simulation_coverage_count)
        or response_id_simulation_coverage_count <= 0
    ):
        mismatches["raw_route_response_id_simulation_coverage_count"] = {
            "expected": "positive integer",
            "actual": response_id_simulation_coverage_count,
        }
    if (
        not isinstance(response_id_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", response_id_sha256) is None
    ):
        mismatches["raw_route_response_id_sha256"] = {
            "expected": "64 lowercase hexadecimal characters",
            "actual": response_id_sha256,
        }
    usage_costs_by_role = gate.get("raw_usage_cost_usd_by_role")
    usage_cost_total = gate.get("raw_usage_cost_usd_total")
    usage_cost_counts = gate.get("raw_usage_cost_message_counts")
    valid_usage_costs = (
        isinstance(usage_costs_by_role, dict)
        and set(usage_costs_by_role) == {"assistant", "user"}
        and all(
            _is_finite_number(value) and float(value) >= 0.0
            for value in usage_costs_by_role.values()
        )
        and _is_finite_number(usage_cost_total)
        and float(usage_cost_total) >= 0.0
        and math.isclose(
            math.fsum(float(value) for value in usage_costs_by_role.values()),
            float(usage_cost_total),
            abs_tol=1e-12,
        )
        and isinstance(usage_cost_counts, dict)
        and usage_cost_counts == response_id_counts
    )
    if not valid_usage_costs:
        mismatches["raw_usage_cost_coverage"] = {
            "expected": "finite non-negative costs for every attributed response",
            "actual": {
                "by_role": usage_costs_by_role,
                "total": usage_cost_total,
                "message_counts": usage_cost_counts,
                "response_id_counts": response_id_counts,
            },
        }
    if (
        gate.get("raw_response_binding_issue_count") != 0
        or gate.get("raw_response_binding_issues") != []
    ):
        mismatches["raw_response_binding"] = {
            "expected": {"issue_count": 0, "issues": []},
            "actual": {
                "issue_count": gate.get("raw_response_binding_issue_count"),
                "issues": gate.get("raw_response_binding_issues"),
            },
        }
    if candidate is not None:
        if not isinstance(config, dict):
            mismatches["raw_route_candidate_binding"] = {
                "expected": "config for bound candidate projection",
                "actual": None,
            }
        else:
            from compare_results import validate_raw_routes

            validated_routes = validate_raw_routes(
                candidate,
                [
                    (task_id, trial)
                    for task_id in target_task_ids(config, "subset")
                    for trial in config["modes"]["subset"]["trials"]
                ],
                endpoint_inventory,
            )
            if validated_routes.get("raw_route_parity") is not True:
                mismatches["raw_route_candidate_validation"] = {
                    "expected": "exact provider/raw-choice/usage provenance",
                    "actual": validated_routes.get("raw_route_mismatches"),
                }
            for field in (
                "raw_response_binding_issue_count",
                "raw_response_binding_issues",
            ):
                if gate.get(field) != validated_routes.get(field):
                    mismatches[f"{field}.candidate_binding"] = {
                        "expected": validated_routes.get(field),
                        "actual": gate.get(field),
                    }
            projection = _candidate_raw_route_projection(candidate, config)
            for field in (
                "raw_route_counters",
                "raw_route_unattributed_generated_messages",
                "raw_route_response_id_count",
                "raw_route_response_id_counts_by_role",
                "raw_route_response_id_counts_by_simulation",
                "raw_route_response_id_simulation_coverage_count",
                "raw_route_response_id_sha256",
                "raw_usage_cost_usd_by_role",
                "raw_usage_cost_usd_total",
                "raw_usage_cost_message_counts",
            ):
                if gate.get(field) != projection[field]:
                    mismatches[f"{field}.candidate_binding"] = {
                        "expected": projection[field],
                        "actual": gate.get(field),
                    }
            if projection["invalid_response_ids"]:
                mismatches["raw_route_response_ids.invalid"] = {
                    "expected": {},
                    "actual": projection["invalid_response_ids"],
                }
            if projection["duplicate_response_id_keys"]:
                mismatches["raw_route_response_ids.duplicate"] = {
                    "expected": [],
                    "actual": projection["duplicate_response_id_keys"],
                }
            if projection["missing_simulation_coverage"]:
                mismatches["raw_route_response_ids.simulation_coverage"] = {
                    "expected": "at least one assistant and user response per simulation",
                    "actual": projection["missing_simulation_coverage"],
                }
            if projection["invalid_usage_costs"]:
                mismatches["raw_usage_costs.invalid"] = {
                    "expected": {},
                    "actual": projection["invalid_usage_costs"],
                }
    return mismatches


def full_gate_judge_route_mismatches(
    gate: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Re-bind task 102 judge observations to the exact checkpoint messages."""
    from compare_results import validate_judge_routes

    expected_keys = [
        (task_id, trial)
        for task_id in target_task_ids(config, "subset")
        for trial in config["modes"]["subset"]["trials"]
    ]
    validation = validate_judge_routes(candidate, expected_keys, config)
    observations = validation["judge_route_observations"]
    mismatches: dict[str, Any] = {}
    if validation["judge_route_checked"] and not validation["judge_route_parity"]:
        mismatches["judge_route_observations.invalid"] = {
            "expected": "raw-response-bound dated OpenAI judge routes",
            "actual": validation["judge_route_mismatches"],
        }
    if gate.get("judge_route_observations") != observations:
        mismatches["judge_route_observations.binding"] = {
            "expected": observations,
            "actual": gate.get("judge_route_observations"),
        }
    return mismatches


def full_gate_candidate_comparison_mismatches(
    gate: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    reference_path: Path | None = None,
    tasks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the complete offline subset receipt from the bound checkpoint."""
    try:
        from compare_results import (
            compare,
            index_simulations,
            load_authoritative_banking_tasks,
        )

        trajectory = config["artifacts"]["trajectory"]
        if reference_path is None:
            filename = trajectory.get("filename")
            if filename != "banking_knowledge_results.json":
                raise RunGuardError(
                    "Reference config does not name the canonical official trajectory"
                )
            reference_path = HERE / "artifacts" / filename
        if not reference_path.is_file() or reference_path.is_symlink():
            raise RunGuardError(
                f"Official trajectory is missing or unsafe: {reference_path}"
            )
        reference_digest = digest_file(reference_path)
        if reference_digest != trajectory["sha256"]:
            raise RunGuardError("Official trajectory SHA-256 mismatch")
        reference_results = load_json(reference_path)
        reference_index, reference_duplicates = index_simulations(
            reference_results.get("simulations") or []
        )
        if reference_duplicates:
            raise RunGuardError("Official trajectory contains duplicate simulations")
        keys = [
            (task_id, trial)
            for task_id in target_task_ids(config, "subset")
            for trial in config["modes"]["subset"]["trials"]
        ]
        missing_reference = set(keys) - set(reference_index)
        if missing_reference:
            raise RunGuardError(
                f"Official trajectory lacks {len(missing_reference)} subset keys"
            )
        reference = {key: reference_index[key] for key in keys}
        authoritative_tasks = tasks or load_authoritative_banking_tasks()
        missing_tasks = sorted(
            {task_id for task_id, _ in keys} - set(authoritative_tasks)
        )
        if missing_tasks:
            raise RunGuardError(
                f"Authoritative banking tasks are missing: {missing_tasks}"
            )
        report = compare(
            candidate,
            reference,
            keys,
            compare_tools=True,
            max_details=0,
            config=config,
            tasks=authoritative_tasks,
        )
    except Exception as exc:
        return {
            "candidate_offline_recomputation": {
                "expected": "successful exact offline subset comparison",
                "actual_error_type": type(exc).__name__,
            }
        }

    receipt_fields = (
        "expected_simulation_count",
        "candidate_simulation_count",
        "expected_reward_sum",
        "expected_reward_by_trial",
        "candidate_reward_sum",
        "candidate_reward_by_trial",
        "score_parity",
        "strict_reproduction_parity",
        "aggregate_score_parity",
        "aggregate_reward_parity",
        "structural_parity",
        "structural_mismatch_count",
        "reward_vector_parity",
        "reward_vector_mismatch_count",
        "reward_by_trial_parity",
        "score_mismatch_count",
        "mismatch_counts",
        "behavior_checked",
        "behavior_parity",
        "strict_trace_parity",
        "behavior_mismatch_count",
        "behavior_mismatch_counts",
        "behavior_parity_with_known_dense_drift_waiver",
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers",
        "known_dense_drift_mismatch_count",
        "non_waived_behavior_mismatch_count",
        "model_sampling_drift_mismatch_count",
        "model_sampling_drift_mismatch_counts",
        "remaining_behavior_mismatch_count_after_waiver_scopes",
        "text_divergence_message_count",
        "text_divergence_message_counts",
        "component_parity_checked",
        "component_parity",
        "component_mismatch_count",
        "candidate_grading_integrity_checked",
        "candidate_grading_integrity",
        "candidate_grading_integrity_issue_count",
        "sampling_score_attribution_checked",
        "sampling_score_attribution_valid",
        "sampling_score_attribution_issue_count",
        "sampling_score_attribution_issues",
    )
    mismatches = {
        f"candidate_receipt.{field}": {
            "expected": report.get(field),
            "actual": gate.get(field),
        }
        for field in receipt_fields
        if gate.get(field) != report.get(field)
    }
    exact_requirements = {
        "candidate_simulation_count": config["modes"]["subset"][
            "expected_simulation_count"
        ],
        "candidate_reward_sum": float(config["modes"]["subset"]["expected_reward_sum"]),
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "candidate_grading_integrity": True,
        "sampling_score_attribution_valid": True,
    }
    for field, expected in exact_requirements.items():
        if report.get(field) != expected:
            mismatches[f"candidate_recomputed.{field}"] = {
                "expected": expected,
                "actual": report.get(field),
            }
    return mismatches


def verify_full_gate(
    gate_path: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    allow_known_full_shell_drift: bool = False,
    shell_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Require a current exact-score gate with no unwaived behavior drift."""
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RunGuardError("Full mode requires the canonical reference.json")
    if os.environ.get(FULL_RUN_ENV) != "1":
        raise RunGuardError(f"Full mode requires {FULL_RUN_ENV}=1")
    verify_full_shell_oracle_receipt(
        config,
        allow_known_drift=allow_known_full_shell_drift,
        receipt_path=shell_receipt_path,
    )
    gate = load_json(gate_path)
    required = {
        "schema_version": FULL_GATE_SCHEMA_VERSION,
        "kind": "tau3_banking_subset_score_parity",
        "mode": "subset",
        "expected_simulation_count": config["modes"]["subset"][
            "expected_simulation_count"
        ],
        "candidate_simulation_count": config["modes"]["subset"][
            "expected_simulation_count"
        ],
        "expected_reward_sum": float(config["modes"]["subset"]["expected_reward_sum"]),
        "expected_reward_by_trial": [
            float(value)
            for value in config["modes"]["subset"]["expected_reward_by_trial"]
        ],
        "candidate_reward_sum": float(config["modes"]["subset"]["expected_reward_sum"]),
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "structural_mismatch_count": 0,
        "configuration_parity": True,
        "configuration_mismatch_count": 0,
        "behavior_checked": True,
        "known_dense_drift_waiver_scope": KNOWN_DENSE_DRIFT_WAIVER_SCOPE,
        "model_sampling_drift_waiver_scope": MODEL_SAMPLING_DRIFT_WAIVER_SCOPE,
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": True,
        "remaining_behavior_mismatch_count_after_waiver_scopes": 0,
        "component_parity_checked": True,
        "candidate_grading_integrity_checked": True,
        "candidate_grading_integrity": True,
        "candidate_grading_integrity_issue_count": 0,
        "sampling_score_attribution_checked": True,
        "sampling_score_attribution_valid": True,
        "sampling_score_attribution_issue_count": 0,
        "sampling_score_attribution_issues": [],
        "execution_manifest_parity": True,
        "execution_manifest_mismatch_count": 0,
        "raw_route_parity": True,
        "raw_route_mismatch_count": 0,
        "raw_route_unattributed_generated_messages": {},
        "raw_route_response_id_simulation_coverage_count": config["modes"]["subset"][
            "expected_simulation_count"
        ],
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
    candidate_by_trial = gate.get("candidate_reward_by_trial")
    expected_trial_count = len(config["modes"]["subset"]["trials"])
    valid_candidate_by_trial = (
        isinstance(candidate_by_trial, list)
        and len(candidate_by_trial) == expected_trial_count
        and all(_is_finite_number(value) for value in candidate_by_trial)
        and math.isclose(
            math.fsum(float(value) for value in candidate_by_trial),
            float(config["modes"]["subset"]["expected_reward_sum"]),
            abs_tol=1e-12,
        )
    )
    if not valid_candidate_by_trial:
        mismatches["candidate_reward_by_trial"] = {
            "expected": (
                f"{expected_trial_count} finite trial totals summing to the exact "
                "official aggregate"
            ),
            "actual": candidate_by_trial,
        }
    elif gate.get("reward_by_trial_parity") is not (
        candidate_by_trial == required["expected_reward_by_trial"]
    ):
        mismatches["reward_by_trial_parity"] = {
            "expected": candidate_by_trial == required["expected_reward_by_trial"],
            "actual": gate.get("reward_by_trial_parity"),
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
    candidate_checkpoint = _load_checkpoint(candidate_path)
    candidate_comparison_mismatches = full_gate_candidate_comparison_mismatches(
        gate, candidate_checkpoint, config
    )
    if candidate_comparison_mismatches:
        raise RunGuardError(
            "Full-run gate candidate receipt is invalid: "
            f"{candidate_comparison_mismatches}"
        )
    resume_validation = validate_resume_checkpoint(
        candidate_path,
        config_path,
        config,
        "subset",
        current_state,
        DEFAULT_MODAL_APP,
        DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )
    if (
        resume_validation["simulation_count"]
        != config["modes"]["subset"]["expected_simulation_count"]
        or resume_validation["infrastructure_error_count"] != 0
    ):
        raise RunGuardError(
            "Bound subset checkpoint does not contain every completed simulation"
        )
    judge_receipt_mismatches = full_gate_judge_route_mismatches(
        gate, candidate_checkpoint, config
    )
    if judge_receipt_mismatches:
        raise RunGuardError(
            f"Full-run gate judge route receipt is invalid: {judge_receipt_mismatches}"
        )
    manifest_path = bound_path("execution_manifest")
    manifest_digest = digest_file(manifest_path)
    if manifest_digest != gate.get("execution_manifest_sha256"):
        raise RunGuardError(
            "Full-run gate execution manifest was changed after comparison"
        )
    manifest = load_json(manifest_path)
    route_receipt_mismatches = full_gate_raw_route_mismatches(
        gate,
        manifest.get("openrouter_endpoint_inventory"),
        candidate_checkpoint,
        config,
    )
    if route_receipt_mismatches:
        raise RunGuardError(
            f"Full-run gate route receipt is invalid: {route_receipt_mismatches}"
        )
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


def _checkpoint_trial_is_admissible(
    mode: str,
    serialized_num_trials: Any,
    trial: int,
    target_trials: set[int],
) -> bool:
    """Accept the one known stale-num-trials artifact from staged subset resume.

    Upstream auto-resume preserves the original ``Results.info`` object. A
    subset expanded from the one-trial checkpoint therefore continues to say
    ``num_trials=1`` even after trials 1-3 have been checkpointed. The exact
    task, seed, state, command, and prior-manifest provenance are validated
    independently before any such checkpoint is resumed.
    """
    if not _is_int(serialized_num_trials):
        return False
    if trial in range(serialized_num_trials):
        return True
    return mode == "subset" and serialized_num_trials == 1 and trial in target_trials


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
    if mode == "subset_trial0":
        allowed_source_modes.add("smoke")
    elif mode == "subset":
        allowed_source_modes.add("smoke")
        allowed_source_modes.add("subset_trial0")
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
            command = tuple(manifest.get("command") or ())
            status = manifest.get("status")
            is_running = status == "running"
            checkpoint_before_run = manifest.get("checkpoint_sha256_before_run")
            resumed_command = "--auto-resume" in command
            resume_preflight = manifest.get("resume_preflight")
            running_checkpoint_provenance = (
                is_running
                and manifest.get("checkpoint_sha256") is None
                and manifest.get("post_run_execution_state") is None
                and manifest.get("completed_at") is None
                and manifest.get("exit_code") is None
                and manifest.get("finalization_errors") is None
                and (
                    (
                        resumed_command
                        and isinstance(checkpoint_before_run, str)
                        and re.fullmatch(r"[0-9a-f]{64}", checkpoint_before_run)
                        is not None
                        and isinstance(resume_preflight, dict)
                        and resume_preflight.get("checkpoint_sha256")
                        == checkpoint_before_run
                    )
                    or (
                        not resumed_command
                        and checkpoint_before_run is None
                        and resume_preflight is None
                    )
                )
            )
            finalized_checkpoint_provenance = (
                status in {"completed", "failed", "interrupted", "runner_exception"}
                and manifest.get("checkpoint_sha256") == expected_checkpoint_digest
                and (manifest.get("post_run_execution_state") or {}).get("digest")
                == current_state["digest"]
            )
            retryable_infrastructure_provenance = False
            if (
                status == "post_run_validation_failed"
                and source_mode in allowed_source_modes
            ):
                post_validation = manifest.get("post_run_validation")
                checkpoint = _load_checkpoint(results_path)
                simulations = checkpoint["simulations"]
                actual_keys = {
                    (simulation.get("task_id"), simulation.get("trial"))
                    for simulation in simulations
                    if isinstance(simulation, dict)
                }
                expected_keys = {
                    (task_id, trial)
                    for task_id in target_task_ids(config, source_mode)
                    for trial in config["modes"][source_mode]["trials"]
                }
                infrastructure_error_count = sum(
                    isinstance(simulation, dict)
                    and simulation.get("termination_reason") == "infrastructure_error"
                    for simulation in simulations
                )
                completed_validation = (
                    post_validation.get("completed_simulation_validation")
                    if isinstance(post_validation, dict)
                    else None
                )
                retryable_infrastructure_provenance = (
                    isinstance(post_validation, dict)
                    and post_validation.get("passed") is False
                    and post_validation.get("retryable_infrastructure_error") is True
                    and post_validation.get("runner_exit_code") == 0
                    and post_validation.get("expected_simulation_count")
                    == len(expected_keys)
                    and post_validation.get("actual_simulation_count")
                    == len(simulations)
                    and post_validation.get("infrastructure_error_count")
                    == infrastructure_error_count
                    and infrastructure_error_count > 0
                    and post_validation.get("task_trial_coverage_sha256")
                    == canonical_digest(
                        sorted([task_id, trial] for task_id, trial in actual_keys)
                    )
                    and post_validation.get("checkpoint_sha256")
                    == expected_checkpoint_digest
                    and actual_keys == expected_keys
                    and isinstance(completed_validation, dict)
                    and completed_validation.get("completed_simulation_count")
                    == len(simulations) - infrastructure_error_count
                    and completed_validation.get("grading_protocol_route_validation")
                    is True
                    and manifest.get("exit_code") == 2
                    and manifest.get("checkpoint_sha256") == expected_checkpoint_digest
                    and (manifest.get("post_run_execution_state") or {}).get("digest")
                    == current_state["digest"]
                    and manifest.get("finalization_errors") is None
                )
            finalization_errors = manifest.get("finalization_errors")
            stored_post_digest = (manifest.get("post_run_execution_state") or {}).get(
                "digest"
            )
            recoverable_finalization_failure = (
                isinstance(finalization_errors, dict)
                and bool(finalization_errors)
                and set(finalization_errors)
                <= {"checkpoint", "post_run_execution_state"}
                and all(
                    isinstance(error_type, str) and bool(error_type)
                    for error_type in finalization_errors.values()
                )
                and manifest.get("checkpoint_sha256")
                in {None, expected_checkpoint_digest}
                and stored_post_digest in {None, current_state["digest"]}
                and (
                    (
                        status == "finalization_failed"
                        and manifest.get("runner_exit_code") == 0
                        and manifest.get("exit_code") == 2
                    )
                    or status in {"failed", "interrupted", "runner_exception"}
                )
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
                "checkpoint_provenance": running_checkpoint_provenance
                or finalized_checkpoint_provenance
                or retryable_infrastructure_provenance
                or recoverable_finalization_failure,
                "environment": manifest.get("environment") == expected_environment,
                "command": command in valid_commands,
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


def _validate_completed_resume_simulations(
    simulations: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_path: Path,
    *,
    reference_path: Path | None = None,
    tasks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove every simulation auto-resume would skip is complete and authentic."""
    completed = [
        simulation
        for simulation in simulations
        if simulation.get("termination_reason") == "user_stop"
    ]
    if not completed:
        return {
            "completed_simulation_count": 0,
            "grading_protocol_route_validation": True,
        }

    try:
        from compare_results import (
            compare,
            index_simulations,
            load_authoritative_banking_tasks,
            validate_judge_routes,
            validate_raw_routes,
        )

        trajectory = config["artifacts"]["trajectory"]
        if reference_path is None:
            if trajectory.get("filename") != "banking_knowledge_results.json":
                raise RunGuardError(
                    "Reference config does not name the canonical official trajectory"
                )
            reference_path = HERE / "artifacts" / trajectory["filename"]
        if not reference_path.is_file() or reference_path.is_symlink():
            raise RunGuardError(
                f"Official trajectory is missing or unsafe: {reference_path}"
            )
        if digest_file(reference_path) != trajectory["sha256"]:
            raise RunGuardError("Official trajectory SHA-256 mismatch")
        reference_results = load_json(reference_path)
        reference_index, duplicates = index_simulations(
            reference_results.get("simulations") or []
        )
        if duplicates:
            raise RunGuardError("Official trajectory contains duplicate simulations")

        keys = sorted(
            (simulation["task_id"], simulation["trial"]) for simulation in completed
        )
        missing_reference = set(keys) - set(reference_index)
        if missing_reference:
            raise RunGuardError(
                f"Official trajectory lacks completed checkpoint keys: "
                f"{sorted(missing_reference)}"
            )
        reference = {key: reference_index[key] for key in keys}
        authoritative_tasks = tasks or load_authoritative_banking_tasks()
        missing_tasks = sorted(
            {task_id for task_id, _ in keys} - set(authoritative_tasks)
        )
        if missing_tasks:
            raise RunGuardError(
                f"Authoritative banking tasks are missing: {missing_tasks}"
            )
        candidate = {"simulations": completed}
        report = compare(
            candidate,
            reference,
            keys,
            compare_tools=True,
            max_details=10,
            config=config,
            tasks=authoritative_tasks,
        )
        invalid_comparison = {
            "structural_parity": report.get("structural_parity"),
            "candidate_grading_integrity": report.get("candidate_grading_integrity"),
            "sampling_score_attribution_valid": report.get(
                "sampling_score_attribution_valid"
            ),
        }
        if not all(invalid_comparison.values()):
            raise RunGuardError(
                "Completed checkpoint simulations fail offline grading/protocol "
                f"validation: {invalid_comparison}"
            )

        manifest = load_json(manifest_path)
        raw_routes = validate_raw_routes(
            candidate,
            keys,
            manifest.get("openrouter_endpoint_inventory"),
        )
        if raw_routes.get("raw_route_parity") is not True:
            raise RunGuardError(
                "Completed checkpoint simulations fail provider/response-ID/cost "
                f"validation: {raw_routes.get('raw_route_mismatches')}"
            )
        judge_routes = validate_judge_routes(candidate, keys, config)
        if judge_routes.get("judge_route_checked") and not judge_routes.get(
            "judge_route_parity"
        ):
            raise RunGuardError(
                "Completed checkpoint simulations fail NL-judge route validation: "
                f"{judge_routes.get('judge_route_mismatches')}"
            )
    except RunGuardError:
        raise
    except Exception as exc:
        raise RunGuardError(
            "Completed checkpoint simulation validation failed closed "
            f"({type(exc).__name__})"
        ) from exc

    return {
        "completed_simulation_count": len(completed),
        "grading_protocol_route_validation": True,
        "participant_response_id_count": raw_routes["raw_route_response_id_count"],
        "raw_usage_cost_message_counts": raw_routes["raw_usage_cost_message_counts"],
        "judge_route_checked": judge_routes["judge_route_checked"],
    }


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
        if not _checkpoint_trial_is_admissible(
            mode, checkpoint_num_trials, trial, target_trials
        ):
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
    completed_validation = _validate_completed_resume_simulations(
        list(simulations_by_id.values()),
        config,
        Path(manifest),
    )
    checkpoint_sha256 = digest_checkpoint_artifact(results_path)
    return {
        "manifest": manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "task_count": len(task_ids),
        "simulation_count": len(simulations_by_id),
        "serialized_num_trials": checkpoint_num_trials,
        "completed_simulation_validation": completed_validation,
        "stale_subset_num_trials_metadata": (
            mode == "subset"
            and checkpoint_num_trials == 1
            and any(trial > 0 for _, trial in simulation_keys)
        ),
        "infrastructure_error_count": sum(
            simulation.get("termination_reason") == "infrastructure_error"
            for simulation in simulations_by_id.values()
        ),
    }


def validate_completed_run_checkpoint(
    results_path: Path,
    config_path: Path,
    config: dict[str, Any],
    mode: str,
    current_state: dict[str, Any],
    modal_app: str,
    modal_sandbox_timeout: int,
) -> dict[str, Any]:
    """Require a complete, authentic task-by-trial product after runner success."""
    validation = validate_resume_checkpoint(
        results_path,
        config_path,
        config,
        mode,
        current_state,
        modal_app,
        modal_sandbox_timeout,
    )
    checkpoint = _load_checkpoint(results_path)
    expected_tasks = set(target_task_ids(config, mode))
    expected_trials = set(config["modes"][mode]["trials"])
    expected_keys = {
        (task_id, trial) for task_id in expected_tasks for trial in expected_trials
    }
    actual_tasks = {
        task.get("id")
        for task in checkpoint["tasks"]
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }
    actual_keys = {
        (simulation.get("task_id"), simulation.get("trial"))
        for simulation in checkpoint["simulations"]
        if isinstance(simulation, dict)
    }
    configured_count = config["modes"][mode].get("expected_simulation_count")
    if (
        not _is_int(configured_count)
        or configured_count != len(expected_keys)
        or validation["simulation_count"] != configured_count
        or actual_tasks != expected_tasks
        or actual_keys != expected_keys
    ):
        raise RunGuardError(
            "Runner returned zero without exact expected task-by-trial coverage"
        )
    infrastructure_error_count = validation["infrastructure_error_count"]
    completed_validation = validation["completed_simulation_validation"]
    if (
        completed_validation.get("completed_simulation_count")
        != configured_count - infrastructure_error_count
        or completed_validation.get("grading_protocol_route_validation") is not True
    ):
        raise RunGuardError(
            "Runner returned zero without complete structural, grading, and route validation"
        )
    ordered_keys = sorted([task_id, trial] for task_id, trial in actual_keys)
    receipt = {
        "passed": infrastructure_error_count == 0,
        "retryable_infrastructure_error": infrastructure_error_count > 0,
        "expected_simulation_count": configured_count,
        "actual_simulation_count": validation["simulation_count"],
        "infrastructure_error_count": infrastructure_error_count,
        "task_trial_coverage_sha256": canonical_digest(ordered_keys),
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "completed_simulation_validation": completed_validation,
    }
    return receipt


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


@contextmanager
def hold_output_run_lock(output_dir: Path):
    """Hold one cross-process lease for every writer/paid child in an output dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".tau3_banking_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunGuardError(
                f"Another guarded run still owns output directory {output_dir}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "owner_pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
            plan["runner_exit_code"] = exit_code
            plan["status"] = "finalization_failed"
            plan["exit_code"] = 2
    else:
        plan.pop("finalization_errors", None)
        plan.pop("runner_exit_code", None)
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
    parser.add_argument("mode", choices=("smoke", "subset_trial0", "subset", "full"))
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
        "--allow-known-full-shell-drift",
        action="store_true",
        help=(
            "Full mode only: explicitly acknowledge the pinned, reviewed, "
            "potentially score-affecting full Modal shell-oracle mismatches"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output results file after all normal guards",
    )
    return parser.parse_args(argv)


def execute_paid_plan(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    command: list[str],
    prewarm_command: list[str],
    manifest_environment: dict[str, str],
    output_dir: Path,
    results_path: Path,
    cache_prewarm_required: bool,
) -> int:
    """Execute one paid plan while one inherited output-directory lease is held."""
    key = load_openrouter_key(args.credential_config, os.environ)
    required_credit_usd = config["modes"][args.mode]["historical_chat_cost_usd"]
    if (
        not _is_finite_number(required_credit_usd)
        or not _is_finite_number(plan.get("historical_chat_cost_usd"))
        or float(plan["historical_chat_cost_usd"]) != float(required_credit_usd)
    ):
        raise RunGuardError("Paid plan has a stale historical chat cost")
    plan["openrouter_credit_state"] = fetch_openrouter_credit_state(
        key, float(required_credit_usd)
    )
    environment = build_paid_environment(key, manifest_environment)
    with hold_output_run_lock(output_dir) as lock_handle:
        inherited_lock = (lock_handle.fileno(),)
        planned_state = plan.get("execution_state")
        planned_runtime = plan.get("preflight_committed_runtime")
        locked_state = None
        if isinstance(planned_state, dict):
            locked_state = capture_reproduction_state(
                REPO_ROOT, require_clean=True, require_cache=True
            )
            if planned_state.get("digest") != locked_state["digest"]:
                raise RunGuardError(
                    "Reproduction state changed while waiting for the output lock"
                )
        elif cache_prewarm_required:
            if not isinstance(planned_runtime, dict):
                raise RunGuardError(
                    "Embedding-cache prewarm plan lacks its committed runtime"
                )
            locked_runtime = capture_committed_runtime(REPO_ROOT, require_clean=True)
            if planned_runtime.get("digest") != locked_runtime["digest"]:
                raise RunGuardError(
                    "Committed runtime changed while waiting for the output lock"
                )
        if not args.resume:
            if results_path.exists():
                raise RunGuardError(
                    f"Results appeared before the paid launch at {results_path}; "
                    "use --resume explicitly"
                )
        else:
            if not results_path.exists():
                raise RunGuardError(
                    f"Resume checkpoint disappeared before launch: {results_path}"
                )
            if locked_state is None:
                raise RunGuardError(
                    "Resume plan lacks a locked canonical reproduction state"
                )
            locked_resume_preflight = validate_resume_checkpoint(
                results_path,
                config_path,
                config,
                args.mode,
                locked_state,
                args.modal_app,
                args.modal_sandbox_timeout,
            )
            if locked_resume_preflight != plan.get("resume_preflight"):
                raise RunGuardError(
                    "Resume checkpoint changed while waiting for the output lock"
                )
        manifest_path = output_dir / f"reproduction_manifest_{args.mode}.json"
        if cache_prewarm_required:
            plan["dry_run"] = False
            plan["status"] = "prewarming_embedding_cache"
            write_json_atomic(manifest_path, plan)
            try:
                prewarm = subprocess.run(
                    prewarm_command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=False,
                    pass_fds=inherited_lock,
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
            if not isinstance(planned_runtime, dict) or execution_state["runtime"].get(
                "digest"
            ) != planned_runtime.get("digest"):
                raise RunGuardError(
                    "Committed runtime changed during embedding-cache prewarm"
                )
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
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                pass_fds=inherited_lock,
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
        if process.returncode == 0:
            try:
                post_run_state = capture_reproduction_state(
                    REPO_ROOT, require_clean=True, require_cache=True
                )
                plan["post_run_validation"] = validate_completed_run_checkpoint(
                    results_path,
                    config_path,
                    config,
                    args.mode,
                    post_run_state,
                    args.modal_app,
                    args.modal_sandbox_timeout,
                )
                if plan["post_run_validation"].get("passed") is not True:
                    plan["post_run_validation"]["runner_exit_code"] = 0
                    finalize_execution_manifest(
                        plan,
                        manifest_path,
                        results_path,
                        status="post_run_validation_failed",
                        exit_code=2,
                    )
                    print(
                        "error: post-run checkpoint validation found retryable "
                        "infrastructure-error simulations",
                        file=sys.stderr,
                    )
                    return 2
            except Exception as exc:
                plan["post_run_validation"] = {
                    "passed": False,
                    "retryable_infrastructure_error": False,
                    "runner_exit_code": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                finalize_execution_manifest(
                    plan,
                    manifest_path,
                    results_path,
                    status="post_run_validation_failed",
                    exit_code=2,
                )
                print(
                    f"error: post-run checkpoint validation failed: {exc}",
                    file=sys.stderr,
                )
                return 2
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
            verify_canonical_paid_inputs(
                config_path,
                config,
                modal_app=args.modal_app,
                modal_sandbox_timeout=args.modal_sandbox_timeout,
            )
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
            if not math.isfinite(args.cost_ceiling_usd) or args.cost_ceiling_usd <= 0:
                raise RunGuardError(
                    "--cost-ceiling-usd must be a positive finite number"
                )
            if args.cost_ceiling_usd < historical_cost:
                raise RunGuardError(
                    f"Cost ceiling ${args.cost_ceiling_usd:.2f} is below the historical "
                    f"chat cost ${historical_cost:.2f}"
                )
            verify_runtime_patches(config)
            verify_modal_credentials(os.environ)
        execution_state = None
        if args.mode == "full":
            execution_state = verify_full_gate(
                args.gate.resolve(),
                config_path,
                config,
                allow_known_full_shell_drift=args.allow_known_full_shell_drift,
            )
        elif args.allow_known_full_shell_drift:
            raise RunGuardError(
                "--allow-known-full-shell-drift is valid only for full mode"
            )

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
        preflight_committed_runtime = (
            execution_state.get("runtime")
            if isinstance(execution_state, dict)
            else None
        )
        if args.execute and execution_state is None:
            preflight_committed_runtime = capture_committed_runtime(
                REPO_ROOT, require_clean=True
            )
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
            "preflight_committed_runtime": preflight_committed_runtime,
            "cache_prewarm_required": cache_prewarm_required,
            "cache_prewarm_command": prewarm_command,
            "resume_preflight": resume_preflight,
            "output_dir": str(output_dir),
            "command": command,
            "environment": manifest_environment,
            "openrouter_endpoint_inventory": endpoint_inventory,
            "known_full_shell_drift_acknowledged": (
                args.mode == "full"
                and bool(
                    config["reproduction_transport"][
                        "full_shell_oracle_receipt_integrity"
                    ]["mismatch_command_count"]
                )
                and args.allow_known_full_shell_drift
            ),
            "full_shell_oracle_receipt": (
                config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
                    "file"
                ]
                if args.mode == "full"
                else None
            ),
            "full_shell_oracle_receipt_sha256": (
                config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
                    "file_sha256"
                ]
                if args.mode == "full"
                else None
            ),
            "full_shell_oracle_modal_image_recipe_sha256": (
                config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
                    "modal_image_recipe_sha256"
                ]
                if args.mode == "full"
                else None
            ),
            "full_shell_oracle_modal_image_object_id": (
                config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
                    "modal_image_object_id"
                ]
                if args.mode == "full"
                else None
            ),
            "full_shell_oracle_mismatch_count": (
                config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
                    "mismatch_command_count"
                ]
                if args.mode == "full"
                else None
            ),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(f"\nargv: {shlex.join(command)}")
        if not args.execute:
            print("\nDry run only. No model, embedding, or Modal calls were made.")
            return 0

        return execute_paid_plan(
            args=args,
            plan=plan,
            config_path=config_path,
            config=config,
            command=command,
            prewarm_command=prewarm_command,
            manifest_environment=manifest_environment,
            output_dir=output_dir,
            results_path=results_path,
            cache_prewarm_required=cache_prewarm_required,
        )
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
