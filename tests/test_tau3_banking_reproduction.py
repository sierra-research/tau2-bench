"""Offline safety tests for the tau3 banking reproduction harness."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "reproduction" / "tau3_banking"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import compare_results  # noqa: E402
import run as reproduction_run  # noqa: E402


def _tool_simulation(*, tool: str, arguments: dict, output: str) -> dict:
    return {
        "task_id": "task_001",
        "trial": 0,
        "seed": 626729,
        "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": tool,
                        "arguments": arguments,
                    }
                ],
            },
            {"role": "tool", "id": "call-1", "content": output},
        ],
    }


def _compare_tool_output(
    *,
    tool: str = "KB_search_dense",
    expected_arguments: dict | None = None,
    actual_arguments: dict | None = None,
) -> dict:
    expected_arguments = expected_arguments or {"query": "cash back", "k": 10}
    actual_arguments = actual_arguments or copy.deepcopy(expected_arguments)
    expected = _tool_simulation(
        tool=tool, arguments=expected_arguments, output="official output"
    )
    actual = _tool_simulation(
        tool=tool, arguments=actual_arguments, output="OpenRouter output"
    )
    return compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=10,
    )


def test_dense_output_drift_is_reported_strictly_but_narrowly_waivable():
    report = _compare_tool_output()

    assert report["behavior_parity"] is False
    assert report["behavior_mismatch_count"] == 1
    assert report["known_dense_drift_mismatch_count"] == 1
    assert report["non_waived_behavior_mismatch_count"] == 0
    assert report["behavior_parity_with_known_dense_drift_waiver"] is True
    assert report["mismatch_counts"] == {
        compare_results.KNOWN_DENSE_DRIFT_MISMATCH_KIND: 1
    }


def test_dense_waiver_rejects_argument_drift_even_when_output_also_differs():
    report = _compare_tool_output(actual_arguments={"query": "different", "k": 10})

    assert report["known_dense_drift_mismatch_count"] == 1
    assert report["non_waived_behavior_mismatch_count"] == 1
    assert report["behavior_parity_with_known_dense_drift_waiver"] is False
    assert report["mismatch_counts"]["tool_call_arguments"] == 1


def test_dense_waiver_never_covers_another_tool_output():
    report = _compare_tool_output(tool="KB_search_bm25")

    assert report["known_dense_drift_mismatch_count"] == 0
    assert report["non_waived_behavior_mismatch_count"] == 1
    assert report["behavior_parity_with_known_dense_drift_waiver"] is False
    assert report["mismatch_counts"]["tool_output"] == 1


def test_dense_waiver_never_covers_a_missing_dense_toolmessage():
    expected = _tool_simulation(
        tool="KB_search_dense",
        arguments={"query": "cash back", "k": 10},
        output="official output",
    )
    actual = copy.deepcopy(expected)
    actual["messages"] = actual["messages"][:1]

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=10,
    )

    assert report["known_dense_drift_mismatch_count"] == 0
    assert report["non_waived_behavior_mismatch_count"] == 1
    assert report["behavior_parity_with_known_dense_drift_waiver"] is False
    assert report["mismatch_counts"]["tool_output_missing"] == 1


def test_full_gate_behavior_receipt_accepts_only_strict_or_explicit_dense_drift():
    strict = {
        "behavior_parity": True,
        "behavior_mismatch_count": 0,
        "known_dense_drift_waiver_requested": False,
        "known_dense_drift_waiver_applied": False,
        "known_dense_drift_mismatch_count": 0,
    }
    waived = {
        "behavior_parity": False,
        "behavior_mismatch_count": 7,
        "known_dense_drift_waiver_requested": True,
        "known_dense_drift_waiver_applied": True,
        "known_dense_drift_mismatch_count": 7,
    }

    assert reproduction_run.full_gate_behavior_mismatches(strict) == {}
    assert reproduction_run.full_gate_behavior_mismatches(waived) == {}

    unrequested = {**waived, "known_dense_drift_waiver_requested": False}
    mixed = {**waived, "behavior_mismatch_count": 8}
    assert reproduction_run.full_gate_behavior_mismatches(unrequested)
    assert reproduction_run.full_gate_behavior_mismatches(mixed)


def test_dense_waiver_flag_cannot_change_a_diagnostic_only_comparison():
    assert (
        compare_results.main(
            [
                "unused-results.json",
                "--mode",
                "subset",
                "--allow-known-dense-drift",
            ]
        )
        == 2
    )


def test_explicit_dense_waiver_writes_schema_v3_gate_and_keeps_strict_failure(
    monkeypatch, tmp_path
):
    expected = _tool_simulation(
        tool="KB_search_dense",
        arguments={"query": "cash back", "k": 10},
        output="official output",
    )
    actual = _tool_simulation(
        tool="KB_search_dense",
        arguments={"query": "cash back", "k": 10},
        output="OpenRouter output",
    )
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "results.json"
    reference_path.write_text(
        json.dumps({"info": {"git_commit": "test-head"}, "simulations": [expected]})
    )
    candidate_path.write_text(
        json.dumps({"info": {"git_commit": "test-head"}, "simulations": [actual]})
    )

    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    config["modes"]["subset"].update(
        {
            "task_ids": ["task_001"],
            "trials": [0],
            "expected_simulation_count": 1,
            "expected_reward_sum": 1,
            "expected_reward_by_trial": [1],
        }
    )
    config["artifacts"]["trajectory"]["sha256"] = hashlib.sha256(
        reference_path.read_bytes()
    ).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    execution_state = {"digest": "state-digest", "runtime": {"head": "test-head"}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"post_run_execution_state": execution_state}))
    monkeypatch.setattr(
        compare_results, "metadata_mismatches", lambda actual, expected: {}
    )
    monkeypatch.setattr(
        compare_results,
        "validate_raw_routes",
        lambda candidate, keys: {
            "raw_route_parity": True,
            "raw_route_mismatch_count": 0,
            "raw_route_mismatches": {},
            "raw_route_counters": [],
            "raw_route_unattributed_generated_messages": {},
        },
    )
    monkeypatch.setattr(
        compare_results,
        "validate_judge_routes",
        lambda candidate, keys, config: {
            "judge_route_checked": True,
            "judge_route_parity": True,
            "judge_route_mismatch_count": 0,
            "judge_route_mismatches": {},
            "judge_route_observations": [],
        },
    )
    monkeypatch.setattr(
        compare_results,
        "validate_execution_manifest",
        lambda *args, **kwargs: {
            "execution_manifest_parity": True,
            "execution_manifest_mismatch_count": 0,
            "execution_manifest_mismatches": {},
            "execution_manifest": str(manifest_path),
            "execution_manifest_sha256": compare_results.digest_file(manifest_path),
        },
    )
    monkeypatch.setattr(
        compare_results,
        "capture_reproduction_state",
        lambda *args, **kwargs: execution_state,
    )
    rejected_gate_path = tmp_path / "rejected-gate.json"
    rejected_exit_code = compare_results.main(
        [
            str(candidate_path),
            "--config",
            str(config_path),
            "--reference-results",
            str(reference_path),
            "--mode",
            "subset",
            "--write-gate",
            str(rejected_gate_path),
        ]
    )
    assert rejected_exit_code == 2
    assert not rejected_gate_path.exists()

    gate_path = tmp_path / "gate.json"

    exit_code = compare_results.main(
        [
            str(candidate_path),
            "--config",
            str(config_path),
            "--reference-results",
            str(reference_path),
            "--mode",
            "subset",
            "--write-gate",
            str(gate_path),
            "--allow-known-dense-drift",
        ]
    )

    assert exit_code == 0
    gate = json.loads(gate_path.read_text())
    assert gate["schema_version"] == reproduction_run.FULL_GATE_SCHEMA_VERSION
    assert gate["behavior_parity"] is False
    assert gate["known_dense_drift_waiver_requested"] is True
    assert gate["known_dense_drift_waiver_applied"] is True
    assert gate["known_dense_drift_mismatch_count"] == 1
    assert gate["non_waived_behavior_mismatch_count"] == 0


def test_manifest_environment_overrides_ambient_modal_order_fixture(monkeypatch):
    monkeypatch.setenv("TAU2_MODAL_ORDER_MANIFEST", "/tmp/untrusted.json")
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))

    environment = reproduction_run.expected_manifest_environment(config)

    assert environment["TAU2_MODAL_ORDER_MANIFEST"] == (
        reproduction_run.MODAL_ORDER_MANIFEST_RELATIVE
    )


def test_qwen_request_keeps_the_official_xhigh_only_arguments(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    command = reproduction_run.build_command(
        config, "smoke", tmp_path / "run", resume=False
    )
    serialized = command[command.index("--agent-llm-args") + 1]

    assert json.loads(serialized) == {"extra_body": {"reasoning": {"effort": "xhigh"}}}


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status = 200
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, _limit: int) -> bytes:
        return self._payload


def _endpoint_payload(spec: dict, *, extra_active: bool = False) -> dict:
    endpoints = [
        {
            "provider_name": spec["provider"],
            "name": f"{spec['provider']} | {spec['resolved_model']}",
            "status": 0,
        }
    ]
    if extra_active:
        endpoints.append(
            {
                "provider_name": "AnotherProvider",
                "name": "AnotherProvider | moving-alias",
                "status": 0,
            }
        )
    return {"data": {"id": spec["response_model_id"], "endpoints": endpoints}}


def test_endpoint_inventory_records_and_validates_active_endpoint_counts(monkeypatch):
    responses = iter(
        [
            _FakeResponse(_endpoint_payload(spec))
            for spec in reproduction_run.ENDPOINT_INVENTORY_SPECS
        ]
    )
    monkeypatch.setattr(
        reproduction_run.urllib.request,
        "urlopen",
        lambda request, timeout: next(responses),
    )

    inventory = reproduction_run.fetch_openrouter_endpoint_inventory()

    assert inventory["schema_version"] == 2
    assert reproduction_run.endpoint_inventory_mismatches(inventory) == {}
    qwen = inventory["entries"][0]
    assert qwen["active_endpoint_count"] == 1
    assert qwen["matching_active_endpoint_count"] == 1

    stale = copy.deepcopy(inventory)
    stale["entries"][0]["active_endpoint_count"] = 2
    stale["digest"] = reproduction_run.canonical_digest(stale["entries"])
    mismatches = reproduction_run.endpoint_inventory_mismatches(stale)
    assert "qwen/qwen3.8-max.active_endpoint_count" in mismatches


def test_endpoint_preflight_rejects_a_second_active_qwen_route(monkeypatch):
    qwen_spec = reproduction_run.ENDPOINT_INVENTORY_SPECS[0]
    monkeypatch.setattr(
        reproduction_run.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            _endpoint_payload(qwen_spec, extra_active=True)
        ),
    )

    with pytest.raises(reproduction_run.RunGuardError, match="sole active"):
        reproduction_run.fetch_openrouter_endpoint_inventory()
