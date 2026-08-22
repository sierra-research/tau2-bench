"""Offline safety tests for the tau3 banking reproduction harness."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "reproduction" / "tau3_banking"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import compare_results  # noqa: E402
import run as reproduction_run  # noqa: E402


def _attributed_message(
    role: str,
    response_id: str,
    content: str | None,
    *,
    cost: float,
    tool_calls: list[dict] | None = None,
) -> dict:
    model = "qwen/qwen3.8-max" if role == "assistant" else "openai/gpt-5.2"
    provider = "Alibaba" if role == "assistant" else "OpenAI"
    service_tier = None if role == "assistant" else "default"
    raw_calls = (
        [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], separators=(",", ":")),
                },
            }
            for call in tool_calls
        ]
        if tool_calls is not None
        else None
    )
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "cost": cost,
        "usage": usage,
        "raw_data": {
            "id": response_id,
            "model": model,
            "provider": provider,
            "service_tier": service_tier,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": raw_calls,
                    },
                }
            ],
            "usage": {**usage, "cost": cost},
        },
    }


def _judge_provenance(config: dict, response_id: str, checks: list[dict]) -> dict:
    results = [
        {
            "expectedOutcome": check["nl_assertion"],
            "metExpectation": check["met"],
            "reasoning": check["justification"],
        }
        for check in checks
    ]
    content = json.dumps({"results": results}, separators=(",", ":"))
    raw_response = {
        "id": response_id,
        "model": "openai/gpt-4.1",
        "provider": "OpenAI",
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
    }
    return {
        "requested_model": config["reproduction_transport"]["nl_assertions_model"],
        "resolved_model": "openai/gpt-4.1-2025-04-14",
        "response_model": "openai/gpt-4.1",
        "provider": "OpenAI",
        "service_tier": "default",
        "response_id": response_id,
        "raw_response": raw_response,
    }


def _tool_simulation(*, tool: str, arguments: dict, output: str) -> dict:
    return {
        "task_id": "task_001",
        "trial": 0,
        "seed": 626729,
        "termination_reason": "user_stop",
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
            "env_assertions": [],
            "action_checks": [],
            "nl_assertions": None,
            "communicate_checks": None,
            "reward_basis": ["DB"],
            "reward_breakdown": {"DB": 1.0},
        },
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


def _tool_sequence_simulation(calls: list[tuple[str, dict, str]]) -> dict:
    simulation = _tool_simulation(
        tool=calls[0][0], arguments=calls[0][1], output=calls[0][2]
    )
    messages = []
    for index, (tool, arguments, output) in enumerate(calls):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": call_id, "name": tool, "arguments": arguments}
                    ],
                },
                {"role": "tool", "id": call_id, "content": output},
            ]
        )
    simulation["messages"] = messages
    return simulation


def _synthetic_task(
    *, actions: list[dict] | None = None, communicate_info: list[str] | None = None
):
    from tau2.data_model.tasks import Task

    return Task.model_validate(
        {
            "id": "task_001",
            "user_scenario": {"instructions": "offline evaluator fixture"},
            "evaluation_criteria": {
                "actions": actions or [],
                "communicate_info": communicate_info or [],
                "reward_basis": ["ACTION"] if actions else ["COMMUNICATE"],
            },
        }
    )


def _task_001():
    from tau2.data_model.tasks import Task

    return Task.model_validate_json(
        (
            REPO_ROOT / "data/tau2/domains/banking_knowledge/tasks/task_001.json"
        ).read_text(encoding="utf-8")
    )


def _call_messages(
    role: str, name: str, arguments: dict, *, call_id: str
) -> list[dict]:
    return [
        {
            "role": role,
            "tool_calls": [
                {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "requestor": role,
                }
            ],
        },
        {
            "role": "tool",
            "id": call_id,
            "content": "offline recorded output",
            "requestor": role,
        },
    ]


def _task_001_db_simulation(*, include_application: bool, trial: int) -> dict:
    simulation = _tool_simulation(
        tool="KB_search_dense",
        arguments={"query": f"retrieval trial {trial}"},
        output="retrieval output",
    )
    simulation["trial"] = trial
    simulation["messages"].append(
        {"role": "assistant", "content": "Retrieval complete."}
    )
    if include_application:
        simulation["messages"].extend(
            _call_messages(
                "user",
                "apply_for_credit_card",
                {
                    "card_type": "Gold Rewards Card",
                    "customer_name": "Sarah Bosch",
                    "annual_income": 100000,
                    "rho_bank_subscription": True,
                },
                call_id=f"application-{trial}",
            )
        )
        simulation["messages"].append({"role": "user", "content": "Done."})
        outcome = 1.0
    else:
        outcome = 0.0
    simulation["reward_info"].update(
        {
            "reward": outcome,
            "db_check": {"db_match": bool(outcome), "db_reward": outcome},
            "reward_breakdown": {"DB": outcome},
        }
    )
    return simulation


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

    assert report["known_dense_drift_mismatch_count"] == 0
    assert report["non_waived_behavior_mismatch_count"] == 3
    assert report["behavior_parity_with_known_dense_drift_waiver"] is False
    assert report["mismatch_counts"]["tool_call_arguments"] == 1
    assert report["mismatch_counts"]["tool_output_missing"] == 1
    assert report["mismatch_counts"]["tool_output_unexpected"] == 1
    assert report["model_sampling_drift_mismatch_count"] == 3


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


def test_model_sampling_scope_covers_changed_call_and_its_downstream_output():
    report = _compare_tool_output(
        tool="KB_search_shell",
        expected_arguments={"command": "printf official"},
        actual_arguments={"command": "printf sampled"},
    )

    assert report["behavior_mismatch_count"] == 3
    assert report["model_sampling_drift_mismatch_count"] == 3
    assert report["model_sampling_drift_mismatch_counts"] == {
        "tool_call_arguments": 1,
        "tool_output_missing": 1,
        "tool_output_unexpected": 1,
    }
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] == 0


def test_model_sampling_scope_never_covers_same_shell_call_backend_drift():
    report = _compare_tool_output(tool="KB_search_shell")

    assert report["behavior_mismatch_count"] == 1
    assert report["model_sampling_drift_mismatch_count"] == 0
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] == 1


def test_strict_trace_detects_requestor_only_tool_call_drift():
    expected = _tool_simulation(
        tool="submit_transaction", arguments={"amount": 1}, output="same"
    )
    actual = copy.deepcopy(expected)
    actual["messages"][0]["tool_calls"][0]["requestor"] = "user"

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["behavior_parity"] is False
    assert report["mismatch_counts"]["tool_call_requestor"] == 1
    assert report["mismatch_counts"]["tool_call_sequence"] == 1
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] >= 1
    assert (
        report["behavior_parity_with_known_dense_and_model_sampling_drift_waivers"]
        is False
    )


def test_strict_trace_uses_tool_call_default_when_user_requestor_is_omitted():
    expected = _tool_simulation(
        tool="submit_transaction", arguments={"amount": 1}, output="same"
    )
    expected["messages"][0]["role"] = "user"
    expected["messages"][0]["tool_calls"][0]["requestor"] = "user"
    expected["messages"][1]["requestor"] = "user"
    expected["messages"].insert(0, {"role": "assistant", "content": "How can I help?"})
    actual = copy.deepcopy(expected)
    del actual["messages"][1]["tool_calls"][0]["requestor"]

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["behavior_parity"] is False
    assert report["mismatch_counts"]["tool_call_requestor"] == 1
    assert report["mismatch_counts"]["tool_call_sequence"] == 1
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] >= 1


def test_invalid_tool_call_argument_serialization_cannot_strict_pass():
    expected = _tool_simulation(
        tool="submit_transaction", arguments={"amount": 1}, output="same"
    )
    actual = copy.deepcopy(expected)
    actual["messages"][0]["tool_calls"][0]["arguments"] = '{"amount":1}'

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["strict_reproduction_parity"] is False
    assert report["structural_parity"] is False
    assert report["mismatch_counts"]["message_schema_candidate"] == 1
    assert report["candidate_grading_integrity"] is False


@pytest.mark.parametrize("left,right", [(0, 1), (2, 3), (3, 4)])
def test_message_protocol_rejects_reordered_participants_and_tool_outputs(left, right):
    expected = _task_001_db_simulation(include_application=True, trial=0)
    actual = copy.deepcopy(expected)
    actual["messages"][left], actual["messages"][right] = (
        actual["messages"][right],
        actual["messages"][left],
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": _task_001()},
    )

    assert report["mismatch_counts"]["message_protocol_candidate"] == 1
    assert report["structural_parity"] is False
    assert report["strict_reproduction_parity"] is False
    assert report["candidate_grading_integrity"] is False


@pytest.mark.parametrize("message_index", [1, 2, 3])
def test_message_protocol_rejects_empty_generated_messages_and_missing_stop(
    message_index,
):
    expected = _tool_simulation(
        tool="KB_search_dense", arguments={"query": "unused"}, output="unused"
    )
    expected["messages"] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": "Please help.",
            "raw_data": {"id": "user-1"},
        },
        {
            "role": "assistant",
            "content": "Certainly.",
            "raw_data": {"id": "assistant-1"},
        },
        {
            "role": "user",
            "content": "Thanks. ###STOP###",
            "raw_data": {"id": "user-2"},
        },
    ]
    actual = copy.deepcopy(expected)
    actual["messages"][message_index]["content"] = None

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["message_protocol_candidate"] == 1
    assert report["structural_parity"] is False
    assert report["candidate_grading_integrity"] is False


def test_message_protocol_rejects_hidden_audio_in_text_only_benchmark():
    expected = _tool_simulation(
        tool="KB_search_dense", arguments={"query": "unused"}, output="unused"
    )
    expected["messages"] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Please help.", "raw_data": {"id": "user-1"}},
        {
            "role": "assistant",
            "content": "Certainly.",
            "raw_data": {"id": "assistant-1"},
        },
        {
            "role": "user",
            "content": "Thanks. ###STOP###",
            "raw_data": {"id": "user-2"},
        },
    ]
    actual = copy.deepcopy(expected)
    actual["messages"][1]["content"] = None
    actual["messages"][1]["audio_content"] = "YQ=="

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["message_protocol_candidate"] == 1
    assert report["structural_parity"] is False
    assert report["candidate_grading_integrity"] is False


def test_message_protocol_rejects_empty_tool_call_list_on_final_stop():
    expected = _tool_simulation(
        tool="KB_search_dense", arguments={"query": "unused"}, output="unused"
    )
    expected["messages"] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Please help.", "raw_data": {"id": "user-1"}},
        {
            "role": "assistant",
            "content": "Certainly.",
            "raw_data": {"id": "assistant-1"},
        },
        {
            "role": "user",
            "content": "Thanks. ###STOP###",
            "raw_data": {"id": "user-2"},
        },
    ]
    actual = copy.deepcopy(expected)
    actual["messages"][-1]["tool_calls"] = []

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["message_protocol_candidate"] == 1
    assert report["structural_parity"] is False
    assert report["candidate_grading_integrity"] is False


def test_message_protocol_requires_the_fixed_rawless_initial_greeting():
    expected = _tool_simulation(
        tool="KB_search_dense", arguments={"query": "unused"}, output="unused"
    )
    expected["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you today?",
            "cost": 0.0,
        },
        {
            "role": "user",
            "content": "Please help.",
            "raw_data": {"id": "user-1"},
        },
        {
            "role": "assistant",
            "content": "Certainly.",
            "raw_data": {"id": "assistant-1"},
        },
        {
            "role": "user",
            "content": "Thanks. ###STOP###",
            "raw_data": {"id": "user-2"},
        },
    ]
    actual = copy.deepcopy(expected)
    actual["messages"][0].update(
        {
            "content": "Sampled greeting",
            "raw_data": {"id": "assistant-extra"},
            "cost": 0.01,
        }
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["message_protocol_candidate"] == 1
    assert report["structural_parity"] is False
    assert report["candidate_grading_integrity"] is False


@pytest.mark.parametrize(
    ("field", "value", "mismatch_kind"),
    [
        ("requestor", "user", "tool_output_requestor"),
        ("error", True, "tool_output_error"),
    ],
)
def test_strict_trace_detects_tool_message_semantic_drift(
    field: str, value: object, mismatch_kind: str
):
    expected = _tool_simulation(
        tool="submit_transaction", arguments={"amount": 1}, output="same"
    )
    actual = copy.deepcopy(expected)
    actual["messages"][1][field] = value

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["strict_reproduction_parity"] is False
    assert report["mismatch_counts"][mismatch_kind] == 1
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] >= 1


def test_model_sampling_scope_aligns_same_call_after_an_insertion():
    expected = _tool_sequence_simulation(
        [
            ("KB_search_shell", {"command": "A"}, "A official"),
            ("KB_search_shell", {"command": "B"}, "B official"),
        ]
    )
    actual = _tool_sequence_simulation(
        [
            ("KB_search_shell", {"command": "X"}, "X sampled"),
            ("KB_search_shell", {"command": "A"}, "A backend drift"),
            ("KB_search_shell", {"command": "B"}, "B official"),
        ]
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["tool_output"] == 1
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] == 1
    assert (
        report["behavior_parity_with_known_dense_and_model_sampling_drift_waivers"]
        is False
    )


def test_duplicate_stateful_tool_outputs_retain_occurrence_order():
    expected = _tool_sequence_simulation(
        [
            ("request_human_agent_transfer", {}, "request #1"),
            ("request_human_agent_transfer", {}, "request #2"),
        ]
    )
    actual = copy.deepcopy(expected)
    actual["messages"][1]["content"], actual["messages"][3]["content"] = (
        actual["messages"][3]["content"],
        actual["messages"][1]["content"],
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["tool_output"] == 2
    assert report["strict_reproduction_parity"] is False
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] == 2


def test_model_sampling_scope_fails_ambiguous_duplicate_output_drift():
    expected = _tool_sequence_simulation(
        [("KB_search_shell", {"command": "A"}, "stable")]
    )
    actual = _tool_sequence_simulation(
        [
            ("KB_search_shell", {"command": "A"}, "stable"),
            ("KB_search_shell", {"command": "A"}, "ambiguous drift"),
        ]
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["mismatch_counts"]["tool_output_unexpected"] == 1
    assert report["model_sampling_drift_mismatch_counts"] == {
        "tool_call_count": 1,
        "tool_call_sequence": 1,
    }
    assert report["remaining_behavior_mismatch_count_after_waiver_scopes"] == 1


@pytest.mark.parametrize(
    ("basis", "field", "expected_record", "actual_record"),
    [
        (
            "DB",
            "db_check",
            {"db_match": True, "db_reward": 1.0},
            {"db_match": False, "db_reward": 0.0},
        ),
        (
            "ENV_ASSERTION",
            "env_assertions",
            [{"env_assertion": {"name": "x"}, "met": True, "reward": 1.0}],
            [{"env_assertion": {"name": "x"}, "met": False, "reward": 0.0}],
        ),
        (
            "ACTION",
            "action_checks",
            [
                {
                    "action": {"name": "write", "arguments": {}},
                    "action_match": True,
                    "action_reward": 1.0,
                    "tool_type": "write",
                }
            ],
            [
                {
                    "action": {"name": "write", "arguments": {}},
                    "action_match": False,
                    "action_reward": 0.0,
                    "tool_type": "write",
                }
            ],
        ),
        (
            "COMMUNICATE",
            "communicate_checks",
            [{"info": "say x", "met": True, "justification": "yes"}],
            [{"info": "say x", "met": False, "justification": "no"}],
        ),
        (
            "NL_ASSERTION",
            "nl_assertions",
            [{"nl_assertion": "x is true", "met": True, "justification": "yes"}],
            [{"nl_assertion": "x is true", "met": False, "justification": "no"}],
        ),
    ],
)
def test_grading_integrity_recomputes_every_reward_basis_component(
    basis: str, field: str, expected_record: object, actual_record: object
):
    expected = _tool_simulation(tool="KB_search_shell", arguments={}, output="same")
    actual = copy.deepcopy(expected)
    for simulation, record in ((expected, expected_record), (actual, actual_record)):
        reward_info = simulation["reward_info"]
        reward_info["reward_basis"] = [basis]
        reward_info[field] = record
        reward_info["reward_breakdown"] = {basis: 1.0}
        reward_info["reward"] = 1.0

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
    )

    assert report["candidate_grading_integrity"] is False
    assert report["candidate_grading_integrity_issue_count"] > 0
    assert report["aggregate_score_parity"] is False


def test_grading_integrity_allows_recombined_db_nl_sampling_outcome():
    expected = _tool_simulation(tool="KB_search_shell", arguments={}, output="same")
    actual = copy.deepcopy(expected)
    nl_identity = {
        "nl_assertion": "recommend the eligible company",
        "met": True,
        "justification": "met",
    }
    expected["reward_info"].update(
        {
            "reward_basis": ["DB", "NL_ASSERTION"],
            "nl_assertions": [nl_identity],
            "reward_breakdown": {"DB": 1.0, "NL_ASSERTION": 1.0},
            "reward": 1.0,
        }
    )
    actual["reward_info"].update(
        {
            "db_check": {"db_match": False, "db_reward": 0.0},
            "reward_basis": ["DB", "NL_ASSERTION"],
            "nl_assertions": [
                {**nl_identity, "met": False, "justification": "not met"}
            ],
            "reward_breakdown": {"DB": 0.0, "NL_ASSERTION": 0.0},
            "reward": 0.0,
        }
    )

    assert compare_results.grading_integrity_issues(actual, expected) == []


def test_sampling_attribution_rejects_compensating_db_flips_with_retrieval_drift():
    task = _task_001()
    expected_pass = _task_001_db_simulation(include_application=True, trial=0)
    expected_fail = _task_001_db_simulation(include_application=False, trial=1)
    actual_fail = copy.deepcopy(expected_pass)
    actual_fail["messages"][0]["tool_calls"][0]["arguments"] = {"query": "sampled fail"}
    actual_fail["reward_info"].update(
        {
            "reward": 0.0,
            "db_check": {"db_match": False, "db_reward": 0.0},
            "reward_breakdown": {"DB": 0.0},
        }
    )
    actual_pass = copy.deepcopy(expected_fail)
    actual_pass["messages"][0]["tool_calls"][0]["arguments"] = {"query": "sampled pass"}
    actual_pass["reward_info"].update(
        {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
            "reward_breakdown": {"DB": 1.0},
        }
    )

    report = compare_results.compare(
        {"simulations": [actual_fail, actual_pass]},
        {
            ("task_001", 0): expected_pass,
            ("task_001", 1): expected_fail,
        },
        [("task_001", 0), ("task_001", 1)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )

    assert report["aggregate_score_parity"] is True
    assert report["candidate_grading_integrity"] is True
    assert report["sampling_score_attribution_valid"] is False
    assert report["sampling_score_attribution_issue_count"] == 2
    assert [
        (issue["task_id"], issue["trial"], issue["component"])
        for issue in report["sampling_score_attribution_issues"]
    ] == [("task_001", 0, "DB"), ("task_001", 1, "DB")]


def test_sampling_attribution_rejects_copied_stale_db_outcome():
    task = _task_001()
    expected = _task_001_db_simulation(include_application=True, trial=0)
    actual = copy.deepcopy(expected)
    application = next(
        call
        for message in actual["messages"]
        for call in message.get("tool_calls") or []
        if call["name"] == "apply_for_credit_card"
    )
    application["arguments"]["card_type"] = "Diamond Elite Card"

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )

    assert report["aggregate_score_parity"] is True
    assert report["candidate_grading_integrity"] is True
    assert report["sampling_score_attribution_valid"] is False
    assert report["sampling_score_attribution_issue_count"] == 1
    issue = report["sampling_score_attribution_issues"][0]
    assert issue["component"] == "DB"
    assert issue["recomputed_candidate_outcome"]["db_reward"] == 0.0


def test_sampling_attribution_replays_failed_and_generic_mutating_writes_exactly():
    task = _task_001()
    expected = _task_001_db_simulation(include_application=True, trial=0)

    failed_write = copy.deepcopy(expected)
    failed_write["messages"].extend(
        _call_messages(
            "assistant",
            "change_user_email",
            {"user_id": "missing-user", "new_email": "nobody@example.com"},
            call_id="failed-write",
        )
    )
    failed_write["reward_info"].update(
        {
            "reward": 0.0,
            "db_check": {"db_match": False, "db_reward": 0.0},
            "reward_breakdown": {"DB": 0.0},
        }
    )
    rejected = compare_results.compare(
        {"simulations": [failed_write]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert rejected["candidate_grading_integrity"] is True
    assert rejected["sampling_score_attribution_issue_count"] == 1
    assert (
        rejected["sampling_score_attribution_issues"][0][
            "recomputed_candidate_outcome"
        ]["db_match"]
        is True
    )

    generic_mutation = copy.deepcopy(expected)
    generic_mutation["messages"].extend(
        _call_messages(
            "assistant",
            "give_discoverable_user_tool",
            {"discoverable_tool_name": "get_referral_link", "arguments": "{}"},
            call_id="generic-mutation",
        )
    )
    generic_mutation["reward_info"].update(
        {
            "reward": 0.0,
            "db_check": {"db_match": False, "db_reward": 0.0},
            "reward_breakdown": {"DB": 0.0},
        }
    )
    attributed = compare_results.compare(
        {"simulations": [generic_mutation]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert attributed["sampling_score_attribution_valid"] is True
    assert attributed["sampling_score_attribution_issues"] == []


def test_sampling_attribution_keys_action_to_exact_gold_compare_arguments():
    expected = _tool_sequence_simulation(
        [
            (
                "submit_referral",
                {"user_id": "u1", "account_type": "Gold Years Account"},
                "ok",
            ),
            ("change_user_email", {"user_id": "u1", "email": "old@example.com"}, "ok"),
        ]
    )
    action = {
        "action_id": "gold-1",
        "requestor": "user",
        "name": "submit_referral",
        "arguments": {"user_id": "u1", "account_type": "Gold Years Account"},
        "info": None,
        "compare_args": ["user_id"],
    }
    task = _synthetic_task(actions=[action])
    expected["reward_info"].update(
        {
            "reward_basis": ["ACTION"],
            "action_checks": [
                {
                    "action": action,
                    "action_match": True,
                    "action_reward": 1.0,
                    "tool_type": "write",
                }
            ],
            "reward_breakdown": {"ACTION": 1.0},
            "reward": 1.0,
        }
    )
    actual = copy.deepcopy(expected)
    actual["messages"][0]["tool_calls"][0]["arguments"]["account_type"] = (
        "Sky Blue Account"
    )
    actual["messages"][2]["tool_calls"][0]["arguments"]["email"] = "new@example.com"
    actual["reward_info"].update(
        {
            "action_checks": [
                {
                    "action": action,
                    "action_match": False,
                    "action_reward": 0.0,
                    "tool_type": "write",
                }
            ],
            "reward_breakdown": {"ACTION": 0.0},
            "reward": 0.0,
        }
    )

    rejected = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert rejected["candidate_grading_integrity"] is True
    assert rejected["sampling_score_attribution_issue_count"] == 1
    recomputed_action = rejected["sampling_score_attribution_issues"][0][
        "recomputed_candidate_outcome"
    ]["records"][0]
    assert recomputed_action["action"] == action
    assert recomputed_action["action_match"] is True
    assert recomputed_action["action_reward"] == 1.0

    actual["messages"][0]["tool_calls"][0]["arguments"]["user_id"] = "u2"
    attributed = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert attributed["sampling_score_attribution_valid"] is True
    assert attributed["sampling_score_attribution_issues"] == []

    stale_official = copy.deepcopy(expected)
    stale_official["reward_info"].update(
        {
            "action_checks": [
                {
                    "action": action,
                    "action_match": False,
                    "action_reward": 0.0,
                    "tool_type": "write",
                }
            ],
            "reward_breakdown": {"ACTION": 0.0},
            "reward": 0.0,
        }
    )
    identical_trajectory_flip = compare_results.compare(
        {"simulations": [expected]},
        {("task_001", 0): stale_official},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert identical_trajectory_flip["candidate_grading_integrity"] is True
    assert identical_trajectory_flip["sampling_score_attribution_valid"] is False
    issue = identical_trajectory_flip["sampling_score_attribution_issues"][0]
    assert issue["serialized_official_outcome"] != issue["recomputed_official_outcome"]


def test_sampling_attribution_rejects_diagnostic_action_tool_type_mutation():
    task = _task_001()
    expected = _task_001_db_simulation(include_application=True, trial=0)
    expected["reward_info"]["action_checks"] = [
        {
            "action": task.evaluation_criteria.actions[0].model_dump(mode="json"),
            "action_match": True,
            "action_reward": 1.0,
            "tool_type": "write",
        }
    ]
    actual = copy.deepcopy(expected)
    actual["reward_info"]["action_checks"][0]["tool_type"] = "read"

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )

    assert report["sampling_score_attribution_valid"] is False
    assert report["sampling_score_attribution_issue_count"] == 1


def test_sampling_attribution_communication_requires_participant_text_drift():
    expected = _tool_simulation(tool="KB_search_dense", arguments={}, output="same")
    expected["messages"].append({"role": "assistant", "content": "say x official"})
    task = _synthetic_task(communicate_info=["say x"])
    expected["reward_info"].update(
        {
            "reward_basis": ["COMMUNICATE"],
            "communicate_checks": [{"info": "say x", "met": True}],
            "reward_breakdown": {"COMMUNICATE": 1.0},
            "reward": 1.0,
        }
    )
    actual = copy.deepcopy(expected)
    actual["reward_info"].update(
        {
            "communicate_checks": [{"info": "say x", "met": False}],
            "reward_breakdown": {"COMMUNICATE": 0.0},
            "reward": 0.0,
        }
    )

    rejected = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert rejected["sampling_score_attribution_issue_count"] == 1

    actual["messages"][-1]["content"] = "sampled"
    attributed = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=20,
        tasks={"task_001": task},
    )
    assert attributed["sampling_score_attribution_valid"] is True


def test_sampling_attribution_nl_allows_only_valid_dated_task102_judge_route():
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    expected = _tool_simulation(tool="KB_search_dense", arguments={}, output="same")
    expected["task_id"] = "task_102"
    expected["reward_info"].update(
        {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": [
                {"nl_assertion": "say x", "met": True, "justification": "met"}
            ],
            "reward_breakdown": {"NL_ASSERTION": 1.0},
            "reward": 1.0,
        }
    )
    actual = copy.deepcopy(expected)
    actual_checks = [
        {"nl_assertion": "say x", "met": False, "justification": "not met"}
    ]
    actual["reward_info"].update(
        {
            "nl_assertions": actual_checks,
            "reward_breakdown": {"NL_ASSERTION": 0.0},
            "reward": 0.0,
            "info": {"judge": _judge_provenance(config, "response-1", actual_checks)},
        }
    )

    attributed = compare_results.compare(
        {"simulations": [actual]},
        {("task_102", 0): expected},
        [("task_102", 0)],
        compare_tools=True,
        max_details=20,
        config=config,
    )
    assert attributed["sampling_score_attribution_valid"] is True

    forged_raw = copy.deepcopy(actual)
    forged_raw["reward_info"]["info"]["judge"]["raw_response"]["choices"][0]["message"][
        "content"
    ] = json.dumps(
        {
            "results": [
                {
                    "expectedOutcome": "say x",
                    "metExpectation": True,
                    "reasoning": "met",
                }
            ]
        }
    )
    forged = compare_results.compare(
        {"simulations": [forged_raw]},
        {("task_102", 0): expected},
        [("task_102", 0)],
        compare_tools=True,
        max_details=20,
        config=config,
    )
    assert forged["sampling_score_attribution_valid"] is False

    actual["reward_info"]["info"]["judge"]["resolved_model"] = "openai/gpt-4.1"
    rejected = compare_results.compare(
        {"simulations": [actual]},
        {("task_102", 0): expected},
        [("task_102", 0)],
        compare_tools=True,
        max_details=20,
        config=config,
    )
    assert rejected["sampling_score_attribution_issue_count"] == 1

    actual["messages"].append({"role": "assistant", "content": "sampled NL text"})
    still_rejected = compare_results.compare(
        {"simulations": [actual]},
        {("task_102", 0): expected},
        [("task_102", 0)],
        compare_tools=True,
        max_details=20,
        config=config,
    )
    assert still_rejected["sampling_score_attribution_issue_count"] == 1
    assert (
        still_rejected["sampling_score_attribution_issues"][0]["participant_text_drift"]
        is True
    )

    actual["reward_info"]["info"]["judge"].update(
        {
            "resolved_model": "openai/gpt-4.1-2025-04-14",
            "response_id": "   ",
        }
    )
    assert not compare_results.valid_dated_task102_judge_route(
        actual, ("task_102", 0), config
    )


def test_judge_routes_require_unique_response_ids_and_gate_binds_observations():
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    simulations = []
    for trial in (0, 1):
        simulation = _tool_simulation(
            tool="KB_search_dense", arguments={}, output="same"
        )
        simulation["task_id"] = "task_102"
        simulation["trial"] = trial
        simulation["reward_info"]["nl_assertions"] = []
        simulation["reward_info"]["info"] = {
            "judge": _judge_provenance(config, "duplicate-response", [])
        }
        simulations.append(simulation)
    candidate = {"simulations": simulations}
    report = compare_results.validate_judge_routes(
        candidate, [("task_102", 0), ("task_102", 1)], config
    )
    assert report["judge_route_parity"] is False
    assert "judge_route.response_id_uniqueness" in report["judge_route_mismatches"]

    config["modes"]["subset"]["task_ids"] = ["task_102"]
    config["modes"]["subset"]["trials"] = [0, 1]
    gate = {"judge_route_observations": report["judge_route_observations"]}
    assert "judge_route_observations.invalid" in (
        reproduction_run.full_gate_judge_route_mismatches(gate, candidate, config)
    )

    simulations[1]["reward_info"]["info"]["judge"] = _judge_provenance(
        config, "response-2", []
    )
    valid_report = compare_results.validate_judge_routes(
        candidate, [("task_102", 0), ("task_102", 1)], config
    )
    gate["judge_route_observations"] = valid_report["judge_route_observations"]
    assert (
        reproduction_run.full_gate_judge_route_mismatches(gate, candidate, config) == {}
    )

    participant_response_id = "participant-response"
    simulations[0]["messages"][0]["raw_data"] = {
        "id": participant_response_id,
        "model": "qwen/qwen3.8-max",
        "provider": "Alibaba",
        "service_tier": None,
    }
    simulations[0]["reward_info"]["info"]["judge"] = _judge_provenance(
        config, participant_response_id, []
    )
    collision_report = compare_results.validate_judge_routes(
        candidate, [("task_102", 0), ("task_102", 1)], config
    )
    assert (
        "judge_route.response_id_global_uniqueness"
        in collision_report["judge_route_mismatches"]
    )
    gate["judge_route_observations"] = collision_report["judge_route_observations"]
    assert "judge_route_observations.invalid" in (
        reproduction_run.full_gate_judge_route_mismatches(gate, candidate, config)
    )

    simulations[0]["reward_info"]["info"]["judge"] = _judge_provenance(
        config, "response-1", []
    )
    gate["judge_route_observations"] = compare_results.validate_judge_routes(
        candidate, [("task_102", 0), ("task_102", 1)], config
    )["judge_route_observations"]
    gate["judge_route_observations"][0]["response_id"] = "forged"
    assert "judge_route_observations.binding" in (
        reproduction_run.full_gate_judge_route_mismatches(gate, candidate, config)
    )


def test_tool_output_diagnostics_are_bounded_and_retain_hash_and_length():
    expected = _tool_simulation(
        tool="KB_search_shell", arguments={"command": "echo x"}, output="A" * 50_000
    )
    actual = _tool_simulation(
        tool="KB_search_shell", arguments={"command": "echo x"}, output="B" * 50_000
    )

    report = compare_results.compare(
        {"simulations": [actual]},
        {("task_001", 0): expected},
        [("task_001", 0)],
        compare_tools=True,
        max_details=10,
    )

    detail = report["mismatch_details"][0]
    assert detail["expected"]["utf8_bytes"] == 50_000
    assert detail["actual"]["utf8_bytes"] == 50_000
    assert len(detail["expected"]["preview"]) <= 503
    assert len(json.dumps(report)) < 10_000


def test_full_gate_behavior_receipt_accepts_only_strict_or_explicit_dense_drift():
    strict = {
        "score_parity": True,
        "strict_reproduction_parity": True,
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "reward_vector_parity": True,
        "reward_by_trial_parity": True,
        "component_parity": True,
        "candidate_grading_integrity": True,
        "sampling_score_attribution_checked": True,
        "sampling_score_attribution_valid": True,
        "sampling_score_attribution_issue_count": 0,
        "sampling_score_attribution_issues": [],
        "behavior_parity": True,
        "strict_trace_parity": True,
        "behavior_parity_with_known_dense_drift_waiver": True,
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": True,
        "behavior_mismatch_count": 0,
        "behavior_mismatch_counts": {},
        "known_dense_drift_waiver_requested": False,
        "known_dense_drift_waiver_applied": False,
        "known_dense_drift_mismatch_count": 0,
        "model_sampling_drift_waiver_requested": False,
        "model_sampling_drift_waiver_applied": False,
        "model_sampling_aggregate_score_waiver_applied": False,
        "model_sampling_drift_mismatch_count": 0,
        "model_sampling_drift_mismatch_counts": {},
        "remaining_behavior_mismatch_count_after_waiver_scopes": 0,
        "text_divergence_message_count": 0,
        "text_divergence_message_counts": {},
        "mismatch_counts": {},
        "score_mismatch_count": 0,
        "structural_mismatch_count": 0,
        "reward_vector_mismatch_count": 0,
        "component_mismatch_count": 0,
    }
    waived = {
        **strict,
        "behavior_parity": False,
        "strict_trace_parity": False,
        "strict_reproduction_parity": False,
        "behavior_mismatch_count": 7,
        "behavior_mismatch_counts": {
            compare_results.KNOWN_DENSE_DRIFT_MISMATCH_KIND: 7
        },
        "known_dense_drift_waiver_requested": True,
        "known_dense_drift_waiver_applied": True,
        "known_dense_drift_mismatch_count": 7,
        "mismatch_counts": {compare_results.KNOWN_DENSE_DRIFT_MISMATCH_KIND: 7},
    }

    assert reproduction_run.full_gate_behavior_mismatches(strict) == {}
    assert reproduction_run.full_gate_behavior_mismatches(waived) == {}

    unrequested = {**waived, "known_dense_drift_waiver_requested": False}
    mixed = {**waived, "behavior_mismatch_count": 8}
    assert reproduction_run.full_gate_behavior_mismatches(unrequested)
    assert reproduction_run.full_gate_behavior_mismatches(mixed)


def test_full_gate_behavior_receipt_accepts_explicit_aggregate_sampling_drift():
    receipt = {
        "score_parity": False,
        "strict_reproduction_parity": False,
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "reward_vector_parity": False,
        "reward_by_trial_parity": False,
        "component_parity": False,
        "candidate_grading_integrity": True,
        "sampling_score_attribution_checked": True,
        "sampling_score_attribution_valid": True,
        "sampling_score_attribution_issue_count": 0,
        "sampling_score_attribution_issues": [],
        "behavior_parity": False,
        "strict_trace_parity": False,
        "behavior_parity_with_known_dense_drift_waiver": False,
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": True,
        "behavior_mismatch_count": 1,
        "behavior_mismatch_counts": {"tool_call_arguments": 1},
        "known_dense_drift_waiver_requested": False,
        "known_dense_drift_waiver_applied": False,
        "known_dense_drift_mismatch_count": 0,
        "model_sampling_drift_waiver_requested": True,
        "model_sampling_drift_waiver_applied": True,
        "model_sampling_aggregate_score_waiver_applied": True,
        "model_sampling_drift_mismatch_count": 1,
        "model_sampling_drift_mismatch_counts": {"tool_call_arguments": 1},
        "remaining_behavior_mismatch_count_after_waiver_scopes": 0,
        "text_divergence_message_count": 1,
        "text_divergence_message_counts": {"user": 1},
        "mismatch_counts": {
            "reward": 2,
            "action_checks": 2,
            "tool_call_arguments": 1,
        },
        "score_mismatch_count": 4,
        "structural_mismatch_count": 0,
        "reward_vector_mismatch_count": 2,
        "component_mismatch_count": 2,
    }

    assert reproduction_run.full_gate_behavior_mismatches(receipt) == {}

    receipt["aggregate_reward_parity"] = False
    assert reproduction_run.full_gate_behavior_mismatches(receipt)


def test_full_gate_never_classifies_backend_output_as_sampling_drift():
    receipt = {
        "score_parity": True,
        "strict_reproduction_parity": False,
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "reward_vector_parity": True,
        "reward_by_trial_parity": True,
        "component_parity": True,
        "candidate_grading_integrity": True,
        "sampling_score_attribution_checked": True,
        "sampling_score_attribution_valid": True,
        "sampling_score_attribution_issue_count": 0,
        "sampling_score_attribution_issues": [],
        "behavior_parity": False,
        "strict_trace_parity": False,
        "behavior_parity_with_known_dense_drift_waiver": False,
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": True,
        "behavior_mismatch_count": 1,
        "behavior_mismatch_counts": {"tool_output": 1},
        "known_dense_drift_waiver_requested": False,
        "known_dense_drift_waiver_applied": False,
        "known_dense_drift_mismatch_count": 0,
        "model_sampling_drift_waiver_requested": True,
        "model_sampling_drift_waiver_applied": True,
        "model_sampling_aggregate_score_waiver_applied": False,
        "model_sampling_drift_mismatch_count": 1,
        "model_sampling_drift_mismatch_counts": {"tool_output": 1},
        "remaining_behavior_mismatch_count_after_waiver_scopes": 0,
        "text_divergence_message_count": 0,
        "text_divergence_message_counts": {},
        "mismatch_counts": {"tool_output": 1},
        "score_mismatch_count": 0,
        "structural_mismatch_count": 0,
        "reward_vector_mismatch_count": 0,
        "component_mismatch_count": 0,
    }

    mismatches = reproduction_run.full_gate_behavior_mismatches(receipt)

    assert "model_sampling_drift_type_scope" in mismatches
    receipt["text_divergence_message_counts"] = {"system": 0}
    mismatches = reproduction_run.full_gate_behavior_mismatches(receipt)
    assert "text_divergence_message_counts" in mismatches


def test_full_gate_behavior_receipt_rejects_unattributed_score_flip():
    receipt = {
        "score_parity": False,
        "strict_reproduction_parity": False,
        "aggregate_score_parity": True,
        "aggregate_reward_parity": True,
        "structural_parity": True,
        "reward_vector_parity": False,
        "reward_by_trial_parity": False,
        "component_parity": False,
        "candidate_grading_integrity": True,
        "sampling_score_attribution_checked": True,
        "sampling_score_attribution_valid": False,
        "sampling_score_attribution_issue_count": 1,
        "sampling_score_attribution_issues": [
            {"task_id": "task_001", "trial": 0, "component": "DB"}
        ],
        "behavior_parity": False,
        "strict_trace_parity": False,
        "behavior_parity_with_known_dense_drift_waiver": False,
        "behavior_parity_with_known_dense_and_model_sampling_drift_waivers": True,
        "behavior_mismatch_count": 1,
        "behavior_mismatch_counts": {"tool_call_arguments": 1},
        "known_dense_drift_waiver_requested": False,
        "known_dense_drift_waiver_applied": False,
        "known_dense_drift_mismatch_count": 0,
        "model_sampling_drift_waiver_requested": True,
        "model_sampling_drift_waiver_applied": True,
        "model_sampling_aggregate_score_waiver_applied": True,
        "model_sampling_drift_mismatch_count": 1,
        "model_sampling_drift_mismatch_counts": {"tool_call_arguments": 1},
        "remaining_behavior_mismatch_count_after_waiver_scopes": 0,
        "text_divergence_message_count": 0,
        "text_divergence_message_counts": {},
        "mismatch_counts": {
            "reward": 2,
            "db_component": 2,
            "tool_call_arguments": 1,
        },
        "score_mismatch_count": 4,
        "structural_mismatch_count": 0,
        "reward_vector_mismatch_count": 2,
        "component_mismatch_count": 2,
    }

    mismatches = reproduction_run.full_gate_behavior_mismatches(receipt)

    assert "behavior_waiver_coverage" in mismatches
    assert (
        "sampling_score_attribution" in mismatches["behavior_waiver_coverage"]["actual"]
    )


def test_full_gate_recomputes_score_and_grading_from_bound_candidate(tmp_path):
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
    simulation = _task_001_db_simulation(include_application=True, trial=0)
    candidate = {"simulations": [simulation]}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(candidate))
    config["artifacts"]["trajectory"]["sha256"] = reproduction_run.digest_file(
        reference_path
    )
    task = _task_001()
    report = compare_results.compare(
        candidate,
        {("task_001", 0): simulation},
        [("task_001", 0)],
        compare_tools=True,
        max_details=0,
        config=config,
        tasks={"task_001": task},
    )
    gate = copy.deepcopy(report)

    assert (
        reproduction_run.full_gate_candidate_comparison_mismatches(
            gate,
            candidate,
            config,
            reference_path=reference_path,
            tasks={"task_001": task},
        )
        == {}
    )

    gate["candidate_reward_sum"] = 0.0
    mismatches = reproduction_run.full_gate_candidate_comparison_mismatches(
        gate,
        candidate,
        config,
        reference_path=reference_path,
        tasks={"task_001": task},
    )
    assert "candidate_receipt.candidate_reward_sum" in mismatches


def test_full_gate_rebind_preserves_explicit_aggregate_sampling_waiver(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    config["modes"]["subset"].update(
        {
            "task_ids": ["task_001"],
            "trials": [0, 1],
            "expected_simulation_count": 2,
            "expected_reward_sum": 1,
            "expected_reward_by_trial": [1, 0],
        }
    )
    expected = [
        _task_001_db_simulation(include_application=True, trial=0),
        _task_001_db_simulation(include_application=False, trial=1),
    ]
    candidate_simulations = [
        _task_001_db_simulation(include_application=False, trial=0),
        _task_001_db_simulation(include_application=True, trial=1),
    ]
    candidate = {"simulations": candidate_simulations}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({"simulations": expected}))
    config["artifacts"]["trajectory"]["sha256"] = reproduction_run.digest_file(
        reference_path
    )
    task = _task_001()
    report = compare_results.compare(
        candidate,
        {
            (simulation["task_id"], simulation["trial"]): simulation
            for simulation in expected
        },
        [("task_001", 0), ("task_001", 1)],
        compare_tools=True,
        max_details=0,
        config=config,
        tasks={"task_001": task},
    )

    assert report["candidate_reward_sum"] == 1
    assert report["candidate_reward_by_trial"] == [0.0, 1.0]
    assert report["reward_by_trial_parity"] is False
    assert (
        reproduction_run.full_gate_candidate_comparison_mismatches(
            report,
            candidate,
            config,
            reference_path=reference_path,
            tasks={"task_001": task},
        )
        == {}
    )


def test_offline_grading_pins_checkout_runtime_and_rejects_preloaded_tau2(tmp_path):
    hostile = tmp_path / "hostile"
    hostile_tau2 = hostile / "tau2"
    hostile_tau2.mkdir(parents=True)
    (hostile_tau2 / "__init__.py").write_text("HOSTILE = True\n")
    hostile_data = tmp_path / "hostile-data"
    hostile_data.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": str(hostile),
        "TAU2_DATA_DIR": str(hostile_data),
    }
    bootstrap = (
        "import os,sys; "
        f"sys.path.insert(0, {str(HARNESS_DIR)!r}); "
        "import run; run.verify_canonical_tau2_runtime(); "
        "import tau2; "
        "print(tau2.__file__); "
        "print(os.environ['TAU2_DATA_DIR']); "
        "print(os.environ['PYTHON_DOTENV_DISABLED'])"
    )
    pinned = subprocess.run(
        [sys.executable, "-c", bootstrap],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert pinned.returncode == 0, pinned.stderr
    assert str(REPO_ROOT / "src" / "tau2") in pinned.stdout
    assert str(REPO_ROOT / "data") in pinned.stdout
    assert pinned.stdout.rstrip().endswith("1")

    preloaded = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tau2,sys; "
                f"sys.path.insert(0, {str(HARNESS_DIR)!r}); "
                "import run; run.verify_canonical_tau2_runtime()"
            ),
        ],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert preloaded.returncode != 0
    assert "not pinned to the canonical checkout" in preloaded.stderr


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


def test_sampling_waiver_flag_cannot_change_a_diagnostic_only_comparison():
    assert (
        compare_results.main(
            [
                "unused-results.json",
                "--mode",
                "subset",
                "--allow-model-sampling-drift",
            ]
        )
        == 2
    )


def test_explicit_waivers_write_current_gate_schema_and_keep_strict_failure(
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
    for simulation in (expected, actual):
        simulation["messages"].append(
            {"role": "assistant", "content": "I can submit that application."}
        )
        simulation["messages"].extend(
            _call_messages(
                "user",
                "apply_for_credit_card",
                {
                    "card_type": "Gold Rewards Card",
                    "customer_name": "Sarah Bosch",
                    "annual_income": 100000,
                    "rho_bank_subscription": True,
                },
                call_id="application-0",
            )
        )
    gold_action = _task_001().evaluation_criteria.actions[0].model_dump(mode="json")
    for simulation in (expected, actual):
        simulation["reward_info"]["action_checks"] = [
            {
                "action": gold_action,
                "action_match": True,
                "action_reward": 1.0,
                "tool_type": "write",
            }
        ]
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
    monkeypatch.setattr(compare_results, "DEFAULT_CONFIG", config_path)

    execution_state = {"digest": "state-digest", "runtime": {"head": "test-head"}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"post_run_execution_state": execution_state}))
    monkeypatch.setattr(
        compare_results, "metadata_mismatches", lambda actual, expected: {}
    )
    monkeypatch.setattr(
        compare_results,
        "validate_raw_routes",
        lambda candidate, keys, inventory: {
            "raw_route_parity": True,
            "raw_route_mismatch_count": 0,
            "raw_route_mismatches": {},
            "raw_route_counters": [],
            "raw_route_unattributed_generated_messages": {},
            "raw_route_response_id_count": 2,
            "raw_route_response_id_counts_by_role": {
                "assistant": 1,
                "user": 1,
            },
            "raw_route_response_id_counts_by_simulation": [
                {
                    "task_id": "task_001",
                    "trial": 0,
                    "assistant": 1,
                    "user": 1,
                }
            ],
            "raw_route_response_id_simulation_coverage_count": 1,
            "raw_route_response_id_sha256": "0" * 64,
            "raw_response_binding_issue_count": 0,
            "raw_response_binding_issues": [],
            "raw_route_gpt52_alias_observed": False,
            "raw_route_gpt52_alias_inventory_proven": None,
            "raw_route_gpt52_alias_inventory_proof_mismatches": {},
            "raw_usage_cost_usd_by_role": {"assistant": 0.0, "user": 0.0},
            "raw_usage_cost_usd_total": 0.0,
            "raw_usage_cost_message_counts": {"assistant": 1, "user": 1},
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
            "bound_openrouter_endpoint_inventory": {},
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
    assert gate["schema_version"] == reproduction_run.FULL_GATE_SCHEMA_VERSION == 7
    assert gate["behavior_parity"] is False
    assert gate["known_dense_drift_waiver_requested"] is True
    assert gate["known_dense_drift_waiver_applied"] is True
    assert gate["known_dense_drift_mismatch_count"] == 1
    assert gate["non_waived_behavior_mismatch_count"] == 0

    sampling_expected = [
        _task_001_db_simulation(include_application=True, trial=0),
        _task_001_db_simulation(include_application=False, trial=1),
    ]
    sampling_actual = [
        _task_001_db_simulation(include_application=False, trial=0),
        _task_001_db_simulation(include_application=True, trial=1),
    ]
    config["modes"]["subset"].update(
        {
            "task_ids": ["task_001"],
            "trials": [0, 1],
            "expected_simulation_count": 2,
            "expected_reward_sum": 1,
            "expected_reward_by_trial": [1, 0],
        }
    )
    reference_path.write_text(
        json.dumps(
            {"info": {"git_commit": "test-head"}, "simulations": sampling_expected}
        )
    )
    config["artifacts"]["trajectory"]["sha256"] = hashlib.sha256(
        reference_path.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config))
    candidate_path.write_text(
        json.dumps(
            {"info": {"git_commit": "test-head"}, "simulations": sampling_actual}
        )
    )
    sampling_gate_path = tmp_path / "sampling-gate.json"
    sampling_exit_code = compare_results.main(
        [
            str(candidate_path),
            "--config",
            str(config_path),
            "--reference-results",
            str(reference_path),
            "--mode",
            "subset",
            "--write-gate",
            str(sampling_gate_path),
            "--allow-model-sampling-drift",
        ]
    )

    assert sampling_exit_code == 0
    sampling_gate = json.loads(sampling_gate_path.read_text())
    assert sampling_gate["score_parity"] is False
    assert sampling_gate["aggregate_score_parity"] is True
    assert sampling_gate["component_parity"] is False
    assert sampling_gate["model_sampling_drift_waiver_requested"] is True
    assert sampling_gate["model_sampling_drift_waiver_applied"] is True
    assert sampling_gate["model_sampling_aggregate_score_waiver_applied"] is True
    assert sampling_gate["remaining_behavior_mismatch_count_after_waiver_scopes"] == 0
    assert sampling_gate["sampling_score_attribution_checked"] is True
    assert sampling_gate["sampling_score_attribution_valid"] is True
    assert sampling_gate["sampling_score_attribution_issue_count"] == 0
    assert sampling_gate["sampling_score_attribution_issues"] == []


def test_manifest_environment_overrides_ambient_modal_order_fixture(monkeypatch):
    monkeypatch.setenv("TAU2_MODAL_ORDER_MANIFEST", "/tmp/untrusted.json")
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))

    environment = reproduction_run.expected_manifest_environment(config)

    assert environment["TAU2_MODAL_ORDER_MANIFEST"] == (
        reproduction_run.MODAL_ORDER_MANIFEST_RELATIVE
    )


def test_paid_environment_cannot_be_redirected_by_ambient_api_bases():
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    manifest = reproduction_run.expected_manifest_environment(config)

    environment = reproduction_run.build_paid_environment(
        "not-a-real-secret-key-value",
        manifest,
        {
            "OPENROUTER_API_BASE": "https://attacker.invalid/v1",
            "OPENAI_API_BASE": "https://attacker.invalid/legacy",
            "OPENAI_BASE_URL": "https://attacker.invalid/openai",
            "PYTHONPATH": "/tmp/attacker",
            "PYTHONHOME": "/tmp/attacker-python",
            "UV_PROJECT": "/tmp/attacker-project",
            "UV_PROJECT_ENVIRONMENT": "/tmp/attacker-venv",
            "VIRTUAL_ENV": "/tmp/attacker-venv",
            "TAU2_DATA_DIR": "/tmp/attacker-data",
            "PYTHON_DOTENV_DISABLED": "0",
            "AWS_SECRET_ACCESS_KEY": "unrelated-secret",
            "GITHUB_TOKEN": "unrelated-token",
            "HTTPS_PROXY": "https://attacker.invalid:8443",
            "LITELLM_LOG": "DEBUG",
            "OPENAI_ORGANIZATION": "attacker-org",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "UNRELATED": "preserved",
        },
    )

    assert manifest["OPENROUTER_API_BASE"] == reproduction_run.OPENROUTER_BASE_URL
    assert environment["OPENROUTER_API_BASE"] == reproduction_run.OPENROUTER_BASE_URL
    assert environment["OPENAI_BASE_URL"] == reproduction_run.OPENROUTER_BASE_URL
    assert "OPENAI_API_BASE" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "UV_PROJECT" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert environment["TAU2_DATA_DIR"] == str(REPO_ROOT / "data")
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["MODAL_TOKEN_ID"] == "modal-id"
    assert environment["MODAL_TOKEN_SECRET"] == "modal-secret"
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "HTTPS_PROXY",
        "LITELLM_LOG",
        "OPENAI_ORGANIZATION",
        "UNRELATED",
    ):
        assert name not in environment


def test_paid_environment_rejects_unpinned_manifest_api_base():
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    manifest = reproduction_run.expected_manifest_environment(config)
    manifest["OPENROUTER_API_BASE"] = "https://attacker.invalid/v1"

    with pytest.raises(reproduction_run.RunGuardError, match="OPENROUTER_API_BASE"):
        reproduction_run.build_paid_environment(
            "not-a-real-secret-key-value", manifest, {}
        )


def test_paid_execution_requires_canonical_config_and_modal_knobs(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))

    with pytest.raises(reproduction_run.RunGuardError, match="reference.json"):
        reproduction_run.verify_canonical_paid_inputs(
            tmp_path / "reference.json",
            config,
            modal_app=reproduction_run.DEFAULT_MODAL_APP,
            modal_sandbox_timeout=reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
        )
    with pytest.raises(reproduction_run.RunGuardError, match="Modal app"):
        reproduction_run.verify_canonical_paid_inputs(
            HARNESS_DIR / "reference.json",
            config,
            modal_app="other-app",
            modal_sandbox_timeout=reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
        )
    with pytest.raises(reproduction_run.RunGuardError, match="sandbox timeout"):
        reproduction_run.verify_canonical_paid_inputs(
            HARNESS_DIR / "reference.json",
            config,
            modal_app=reproduction_run.DEFAULT_MODAL_APP,
            modal_sandbox_timeout=1,
        )


def test_qwen_request_keeps_the_official_xhigh_only_arguments(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    command = reproduction_run.build_command(
        config, "smoke", tmp_path / "run", resume=False
    )
    serialized = command[command.index("--agent-llm-args") + 1]

    assert json.loads(serialized) == {"extra_body": {"reasoning": {"effort": "xhigh"}}}


def test_subset_trial0_is_a_ten_task_one_trial_intermediate(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    mode = config["modes"]["subset_trial0"]
    command = reproduction_run.build_command(
        config, "subset_trial0", tmp_path / "run", resume=True
    )

    assert len(mode["task_ids"]) == 10
    assert mode["trials"] == [0]
    assert mode["expected_simulation_count"] == 10
    assert mode["expected_reward_sum"] == 6
    assert command[command.index("--num-trials") + 1] == "1"
    assert command[command.index("--max-concurrency") + 1] == "10"
    assert "--auto-resume" in command


def test_subset_partial_resume_accepts_stale_one_trial_metadata_only_in_subset():
    target_trials = {0, 1, 2, 3}

    assert reproduction_run._checkpoint_trial_is_admissible(
        "subset", 1, 3, target_trials
    )
    assert not reproduction_run._checkpoint_trial_is_admissible(
        "subset_trial0", 1, 3, target_trials
    )
    assert not reproduction_run._checkpoint_trial_is_admissible(
        "full", 1, 3, target_trials
    )
    assert not reproduction_run._checkpoint_trial_is_admissible(
        "subset", 1, 4, target_trials
    )


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


def test_openrouter_credit_preflight_authenticates_and_persists_only_allowlisted_state(
    monkeypatch,
):
    observed = {}

    def open_credit(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return _FakeResponse(
            {
                "data": {
                    "total_credits": 20.0,
                    "total_usage": 7.5,
                    "label": "must-not-be-persisted",
                }
            }
        )

    monkeypatch.setattr(reproduction_run.urllib.request, "urlopen", open_credit)
    receipt = reproduction_run.fetch_openrouter_credit_state(
        "not-a-real-secret-key-value", 12.0
    )

    assert observed == {
        "url": reproduction_run.OPENROUTER_CREDITS_URL,
        "authorization": "Bearer not-a-real-secret-key-value",
        "timeout": 20.0,
    }
    assert receipt["remaining_usd"] == 12.5
    assert receipt["required_usd"] == 12.0
    assert receipt["sufficient"] is True
    serialized = json.dumps(receipt)
    assert "not-a-real-secret-key-value" not in serialized
    assert "must-not-be-persisted" not in serialized
    assert "label" not in serialized


def test_openrouter_credit_preflight_rejects_insufficient_credit(monkeypatch):
    monkeypatch.setattr(
        reproduction_run.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {"data": {"total_credits": 10.0, "total_usage": 9.0}}
        ),
    )

    with pytest.raises(reproduction_run.RunGuardError, match="below"):
        reproduction_run.fetch_openrouter_credit_state("k" * 20, 2.0)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"data": None},
        {"data": {"total_credits": "10", "total_usage": 1}},
        {"data": {"total_credits": float("nan"), "total_usage": 1}},
        {"data": {"total_credits": 10, "total_usage": -1}},
    ),
)
def test_openrouter_credit_preflight_rejects_malformed_api(monkeypatch, payload):
    monkeypatch.setattr(
        reproduction_run.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(payload),
    )

    with pytest.raises(reproduction_run.RunGuardError, match="malformed|invalid"):
        reproduction_run.fetch_openrouter_credit_state("k" * 20, 1.0)


def _endpoint_payload(spec: dict, *, extra_active: bool = False) -> dict:
    endpoints = [
        {
            "provider_name": spec["provider"],
            "name": f"{spec['provider']} | {spec['resolved_model']}",
            "status": 0,
        }
    ]
    if spec["requested_model"] == reproduction_run.GPT52_ALIAS_MODEL:
        endpoints.extend(copy.deepcopy(endpoints[0]) for _ in range(2))
        endpoints.append(
            {
                "provider_name": "Azure",
                "name": "Azure | openai/gpt-5.2-other-snapshot",
                "status": 0,
            }
        )
    if extra_active:
        endpoints.append(
            {
                "provider_name": "AnotherProvider",
                "name": "AnotherProvider | moving-alias",
                "status": 0,
            }
        )
    return {"data": {"id": spec["response_model_id"], "endpoints": endpoints}}


def _bound_endpoint_inventory() -> dict:
    entries = []
    for spec in reproduction_run.ENDPOINT_INVENTORY_SPECS:
        active_count = 1
        eligible_count = 1
        matching_count = 1
        if spec["requested_model"] == reproduction_run.GPT52_ALIAS_MODEL:
            active_count = reproduction_run.GPT52_ALIAS_ACTIVE_ENDPOINT_COUNT
            eligible_count = reproduction_run.GPT52_ALIAS_ELIGIBLE_ACTIVE_ENDPOINT_COUNT
            matching_count = reproduction_run.GPT52_ALIAS_MATCHING_ENDPOINT_COUNT
        elif spec["requested_model"] == "openai/gpt-4.1-2025-04-14":
            active_count = 3
        entries.append(
            {
                "requested_model": spec["requested_model"],
                "response_model_id": spec["response_model_id"],
                "provider": spec["provider"],
                "resolved_model": spec["resolved_model"],
                "status": 0,
                "active_endpoint_count": active_count,
                "eligible_active_endpoint_count": eligible_count,
                "matching_active_endpoint_count": matching_count,
                "raw_sha256": "0" * 64,
            }
        )
    return {
        "schema_version": reproduction_run.ENDPOINT_INVENTORY_SCHEMA_VERSION,
        "fetched_at": "2026-08-22T00:00:00+00:00",
        "entries": entries,
        "digest": reproduction_run.canonical_digest(entries),
        "unauthenticated": True,
    }


def _valid_completed_resume_simulation() -> dict:
    return {
        "id": "completed-task-001-trial-0",
        "task_id": "task_001",
        "trial": 0,
        "seed": 626729,
        "termination_reason": "user_stop",
        "mode": "half_duplex",
        "messages": [
            {
                "role": "assistant",
                "content": "Hi! How can I help you today?",
                "cost": 0.0,
            },
            _attributed_message("user", "resume-user-1", "Please help me.", cost=0.01),
            _attributed_message(
                "assistant",
                "resume-assistant-1",
                "I could not complete the application.",
                cost=0.02,
            ),
            _attributed_message("user", "resume-user-2", "Okay. ###STOP###", cost=0.01),
        ],
        "reward_info": {
            "reward": 0.0,
            "db_check": {"db_match": False, "db_reward": 0.0},
            "env_assertions": [],
            "action_checks": [
                {
                    "action": {
                        "action_id": "001_0",
                        "requestor": "user",
                        "name": "apply_for_credit_card",
                        "arguments": {
                            "card_type": "Gold Rewards Card",
                            "customer_name": "Sarah Bosch",
                            "annual_income": 100000,
                            "rho_bank_subscription": True,
                        },
                        "info": None,
                        "compare_args": None,
                    },
                    "action_match": True,
                    "action_reward": 1.0,
                    "tool_type": "write",
                }
            ],
            "nl_assertions": None,
            "communicate_checks": None,
            "reward_basis": ["DB"],
            "reward_breakdown": {"DB": 0.0},
        },
    }


@pytest.mark.parametrize(
    "mutation",
    ("fractional_reward", "empty_stop", "wrong_provider", "missing_cost"),
)
def test_completed_resume_simulations_fail_closed_before_paid_launch(
    tmp_path, mutation
):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    expected = _valid_completed_resume_simulation()
    reference_path = tmp_path / "official.json"
    reference_path.write_text(json.dumps({"simulations": [expected]}))
    config["artifacts"]["trajectory"].update(
        {
            "filename": "banking_knowledge_results.json",
            "sha256": reproduction_run.digest_file(reference_path),
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"openrouter_endpoint_inventory": _bound_endpoint_inventory()})
    )

    valid = reproduction_run._validate_completed_resume_simulations(
        [copy.deepcopy(expected)],
        config,
        manifest_path,
        reference_path=reference_path,
    )
    assert valid["grading_protocol_route_validation"] is True

    candidate = copy.deepcopy(expected)
    if mutation == "fractional_reward":
        candidate["reward_info"]["reward"] = 0.5
    elif mutation == "empty_stop":
        candidate["messages"][-1]["content"] = None
    elif mutation == "wrong_provider":
        candidate["messages"][1]["raw_data"]["provider"] = "Azure"
    else:
        del candidate["messages"][1]["raw_data"]["usage"]["cost"]

    with pytest.raises(reproduction_run.RunGuardError):
        reproduction_run._validate_completed_resume_simulations(
            [candidate],
            config,
            manifest_path,
            reference_path=reference_path,
        )


def _write_running_resume_manifest(
    tmp_path: Path,
    config: dict,
    *,
    resume: bool,
    checkpoint_before_run: str | None,
) -> tuple[Path, dict]:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps({"checkpoint": "progress-after-launch"}))
    state = {"digest": "runtime-cache-state"}
    manifest = {
        "mode": "smoke",
        "dry_run": False,
        "status": "running",
        "output_dir": str(tmp_path),
        "reference_config_sha256": reproduction_run.digest_file(
            HARNESS_DIR / "reference.json"
        ),
        "execution_state": state,
        "environment": reproduction_run.expected_manifest_environment(config),
        "command": reproduction_run.build_command(
            config, "smoke", tmp_path, resume=resume
        ),
        "prompt_hashes": reproduction_run.expected_prompt_hashes(config),
        "openrouter_endpoint_inventory": _bound_endpoint_inventory(),
        "checkpoint_sha256_before_run": checkpoint_before_run,
        "resume_preflight": (
            {
                "manifest": "prior-manifest.json",
                "checkpoint_sha256": checkpoint_before_run,
            }
            if resume
            else None
        ),
    }
    (tmp_path / "reproduction_manifest_smoke.json").write_text(json.dumps(manifest))
    return results_path, state


def test_abruptly_terminated_running_manifest_authorizes_structural_resume(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    results_path, state = _write_running_resume_manifest(
        tmp_path,
        config,
        resume=False,
        checkpoint_before_run=None,
    )

    provenance = reproduction_run._validate_resume_manifest(
        results_path,
        tmp_path,
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        state,
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )

    assert provenance.endswith("reproduction_manifest_smoke.json")


def test_running_resume_manifest_requires_a_bound_validated_base_checkpoint(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    results_path, state = _write_running_resume_manifest(
        tmp_path,
        config,
        resume=True,
        checkpoint_before_run="1" * 64,
    )
    manifest_path = tmp_path / "reproduction_manifest_smoke.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resume_preflight"]["checkpoint_sha256"] = "2" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        reproduction_run.RunGuardError,
        match="checkpoint_provenance",
    ):
        reproduction_run._validate_resume_manifest(
            results_path,
            tmp_path,
            HARNESS_DIR / "reference.json",
            config,
            "smoke",
            state,
            reproduction_run.DEFAULT_MODAL_APP,
            reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
        )


def test_transient_successful_finalization_failure_can_be_revalidated(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    results_path, state = _write_running_resume_manifest(
        tmp_path,
        config,
        resume=False,
        checkpoint_before_run=None,
    )
    manifest_path = tmp_path / "reproduction_manifest_smoke.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "finalization_failed",
            "runner_exit_code": 0,
            "exit_code": 2,
            "completed_at": "2026-08-22T00:00:00+00:00",
            "checkpoint_sha256": None,
            "post_run_execution_state": None,
            "finalization_errors": {
                "checkpoint": "OSError",
                "post_run_execution_state": "StateFingerprintError",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    assert reproduction_run._validate_resume_manifest(
        results_path,
        tmp_path,
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        state,
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    ).endswith("reproduction_manifest_smoke.json")

    del manifest["runner_exit_code"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(reproduction_run.RunGuardError, match="checkpoint_provenance"):
        reproduction_run._validate_resume_manifest(
            results_path,
            tmp_path,
            HARNESS_DIR / "reference.json",
            config,
            "smoke",
            state,
            reproduction_run.DEFAULT_MODAL_APP,
            reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
        )


def test_output_run_lock_rejects_a_second_writer(tmp_path):
    with reproduction_run.hold_output_run_lock(tmp_path) as handle:
        assert handle.fileno() >= 0
        with pytest.raises(reproduction_run.RunGuardError, match="still owns"):
            with reproduction_run.hold_output_run_lock(tmp_path):
                pass

    with reproduction_run.hold_output_run_lock(tmp_path) as handle:
        assert handle.fileno() >= 0


def test_paid_launch_rechecks_results_and_resume_checkpoint_under_lock(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.json"
    launched = False

    monkeypatch.setattr(reproduction_run, "load_openrouter_key", lambda *args: "k" * 20)
    monkeypatch.setattr(
        reproduction_run,
        "fetch_openrouter_credit_state",
        lambda *args: {"sufficient": True},
    )
    monkeypatch.setattr(
        reproduction_run, "build_paid_environment", lambda *args: {"PINNED": "1"}
    )
    config = {"modes": {"smoke": {"historical_chat_cost_usd": 1.0}}}

    def unexpected_launch(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("paid child must not launch")

    monkeypatch.setattr(reproduction_run.subprocess, "run", unexpected_launch)
    results_path.write_text("{}")
    fresh_args = reproduction_run.parse_args(["smoke", "--execute"])
    with pytest.raises(reproduction_run.RunGuardError, match="Results appeared"):
        reproduction_run.execute_paid_plan(
            args=fresh_args,
            plan={"execution_state": None, "historical_chat_cost_usd": 1.0},
            config_path=HARNESS_DIR / "reference.json",
            config=config,
            command=["paid-child"],
            prewarm_command=["prewarm"],
            manifest_environment={},
            output_dir=output_dir,
            results_path=results_path,
            cache_prewarm_required=False,
        )
    assert launched is False

    state = {"digest": "state"}
    monkeypatch.setattr(
        reproduction_run, "capture_reproduction_state", lambda *args, **kwargs: state
    )
    monkeypatch.setattr(
        reproduction_run,
        "validate_resume_checkpoint",
        lambda *args, **kwargs: {"checkpoint_sha256": "new"},
    )
    resume_args = reproduction_run.parse_args(["smoke", "--execute", "--resume"])
    with pytest.raises(
        reproduction_run.RunGuardError, match="checkpoint changed while waiting"
    ):
        reproduction_run.execute_paid_plan(
            args=resume_args,
            plan={
                "execution_state": state,
                "resume_preflight": {"checkpoint_sha256": "old"},
                "historical_chat_cost_usd": 1.0,
            },
            config_path=HARNESS_DIR / "reference.json",
            config=config,
            command=["paid-child"],
            prewarm_command=["prewarm"],
            manifest_environment={},
            output_dir=output_dir,
            results_path=results_path,
            cache_prewarm_required=False,
        )
    assert launched is False


def test_cache_prewarm_rejects_commit_change_before_output_lock(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.json"
    monkeypatch.setattr(reproduction_run, "load_openrouter_key", lambda *args: "k" * 20)
    monkeypatch.setattr(
        reproduction_run,
        "fetch_openrouter_credit_state",
        lambda *args: {"sufficient": True},
    )
    monkeypatch.setattr(
        reproduction_run, "build_paid_environment", lambda *args: {"PINNED": "1"}
    )
    monkeypatch.setattr(
        reproduction_run,
        "capture_committed_runtime",
        lambda *args, **kwargs: {"digest": "runtime-B"},
    )

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("prewarm and paid child must not launch")

    monkeypatch.setattr(reproduction_run.subprocess, "run", unexpected_launch)
    with pytest.raises(
        reproduction_run.RunGuardError,
        match="Committed runtime changed while waiting",
    ):
        reproduction_run.execute_paid_plan(
            args=reproduction_run.parse_args(["smoke", "--execute"]),
            plan={
                "execution_state": None,
                "preflight_committed_runtime": {"digest": "runtime-A"},
                "historical_chat_cost_usd": 1.0,
            },
            config_path=HARNESS_DIR / "reference.json",
            config={"modes": {"smoke": {"historical_chat_cost_usd": 1.0}}},
            command=["paid-child"],
            prewarm_command=["prewarm"],
            manifest_environment={},
            output_dir=output_dir,
            results_path=results_path,
            cache_prewarm_required=True,
        )


def test_cache_prewarm_rejects_commit_change_during_prewarm(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    results_path = output_dir / "results.json"
    monkeypatch.setattr(reproduction_run, "load_openrouter_key", lambda *args: "k" * 20)
    monkeypatch.setattr(
        reproduction_run,
        "fetch_openrouter_credit_state",
        lambda *args: {"sufficient": True},
    )
    monkeypatch.setattr(
        reproduction_run, "build_paid_environment", lambda *args: {"PINNED": "1"}
    )
    monkeypatch.setattr(
        reproduction_run,
        "capture_committed_runtime",
        lambda *args, **kwargs: {"digest": "runtime-A"},
    )
    monkeypatch.setattr(
        reproduction_run,
        "capture_reproduction_state",
        lambda *args, **kwargs: {
            "digest": "state-B",
            "runtime": {"digest": "runtime-B"},
        },
    )
    subprocess_calls = []

    def prewarm_only(*args, **kwargs):
        subprocess_calls.append(args[0])
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(reproduction_run.subprocess, "run", prewarm_only)
    with pytest.raises(
        reproduction_run.RunGuardError,
        match="Committed runtime changed during embedding-cache prewarm",
    ):
        reproduction_run.execute_paid_plan(
            args=reproduction_run.parse_args(["smoke", "--execute"]),
            plan={
                "execution_state": None,
                "preflight_committed_runtime": {"digest": "runtime-A"},
                "historical_chat_cost_usd": 1.0,
            },
            config_path=HARNESS_DIR / "reference.json",
            config={"modes": {"smoke": {"historical_chat_cost_usd": 1.0}}},
            command=["paid-child"],
            prewarm_command=["prewarm"],
            manifest_environment={},
            output_dir=output_dir,
            results_path=results_path,
            cache_prewarm_required=True,
        )
    assert subprocess_calls == [["prewarm"]]


def test_credit_preflight_fails_before_output_directory_or_paid_child(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "not-created"
    monkeypatch.setattr(reproduction_run, "load_openrouter_key", lambda *args: "k" * 20)

    def insufficient(*args, **kwargs):
        raise reproduction_run.RunGuardError("insufficient test credit")

    monkeypatch.setattr(reproduction_run, "fetch_openrouter_credit_state", insufficient)
    monkeypatch.setattr(
        reproduction_run.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paid child must not launch")
        ),
    )

    with pytest.raises(reproduction_run.RunGuardError, match="insufficient"):
        reproduction_run.execute_paid_plan(
            args=reproduction_run.parse_args(["smoke", "--execute"]),
            plan={"execution_state": None, "historical_chat_cost_usd": 1.0},
            config_path=HARNESS_DIR / "reference.json",
            config={"modes": {"smoke": {"historical_chat_cost_usd": 1.0}}},
            command=["paid-child"],
            prewarm_command=["prewarm"],
            manifest_environment={},
            output_dir=output_dir,
            results_path=output_dir / "results.json",
            cache_prewarm_required=False,
        )

    assert not output_dir.exists()


def test_resume_preserves_valid_work_and_retries_infrastructure_error(
    monkeypatch, tmp_path
):
    checkpoint = {
        "info": {"num_trials": 1},
        "tasks": [{"id": "task_001"}],
        "simulations": [
            {
                "id": "task-001-trial-0",
                "task_id": "task_001",
                "trial": 0,
                "seed": 626729,
                "termination_reason": "infrastructure_error",
                "reward_info": None,
                "mode": "half_duplex",
            }
        ],
    }
    monkeypatch.setattr(reproduction_run, "_load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(
        reproduction_run, "_checkpoint_metadata_mismatches", lambda *args: {}
    )
    monkeypatch.setattr(
        reproduction_run, "_validate_simulation_index", lambda *args: None
    )
    monkeypatch.setattr(
        reproduction_run,
        "_validate_resume_manifest",
        lambda *args: str(tmp_path / "reproduction_manifest_smoke.json"),
    )
    monkeypatch.setattr(
        reproduction_run,
        "_validate_completed_resume_simulations",
        lambda *args, **kwargs: {
            "completed_simulation_count": 0,
            "grading_protocol_route_validation": True,
        },
    )
    monkeypatch.setattr(
        reproduction_run, "digest_checkpoint_artifact", lambda *args: "1" * 64
    )
    config = {
        "modes": {"smoke": {"task_ids": ["task_001"], "trials": [0]}},
        "recorded_run": {"derived_trial_seeds": [626729]},
    }

    validation = reproduction_run.validate_resume_checkpoint(
        tmp_path / "results.json",
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        {"runtime": {"head": "test-head"}},
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )

    assert validation["simulation_count"] == 1
    assert validation["infrastructure_error_count"] == 1
    assert validation["completed_simulation_validation"] == {
        "completed_simulation_count": 0,
        "grading_protocol_route_validation": True,
    }


def test_post_run_infrastructure_failure_manifest_authorizes_exact_retry(
    monkeypatch, tmp_path
):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    results_path = tmp_path / "results.json"
    checkpoint = {
        "info": {"num_trials": 1},
        "tasks": [{"id": "task_001"}],
        "simulations": [
            {
                "id": "task-001-trial-0",
                "task_id": "task_001",
                "trial": 0,
                "seed": 626729,
                "termination_reason": "infrastructure_error",
                "reward_info": None,
                "mode": "half_duplex",
            }
        ],
    }
    results_path.write_text(json.dumps(checkpoint))
    checkpoint_sha256 = reproduction_run.digest_checkpoint_artifact(results_path)
    state = {"digest": "runtime-cache-state", "runtime": {"head": "test-head"}}
    manifest_path = tmp_path / "reproduction_manifest_smoke.json"
    manifest = {
        "mode": "smoke",
        "dry_run": False,
        "status": "post_run_validation_failed",
        "exit_code": 2,
        "output_dir": str(tmp_path),
        "reference_config_sha256": reproduction_run.digest_file(
            HARNESS_DIR / "reference.json"
        ),
        "execution_state": state,
        "post_run_execution_state": state,
        "checkpoint_sha256": checkpoint_sha256,
        "environment": reproduction_run.expected_manifest_environment(config),
        "command": reproduction_run.build_command(
            config, "smoke", tmp_path, resume=False
        ),
        "prompt_hashes": reproduction_run.expected_prompt_hashes(config),
        "openrouter_endpoint_inventory": _bound_endpoint_inventory(),
        "post_run_validation": {
            "passed": False,
            "retryable_infrastructure_error": True,
            "runner_exit_code": 0,
            "expected_simulation_count": 1,
            "actual_simulation_count": 1,
            "infrastructure_error_count": 1,
            "task_trial_coverage_sha256": reproduction_run.canonical_digest(
                [["task_001", 0]]
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "completed_simulation_validation": {
                "completed_simulation_count": 0,
                "grading_protocol_route_validation": True,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        reproduction_run, "_checkpoint_metadata_mismatches", lambda *args: {}
    )

    validation = reproduction_run.validate_resume_checkpoint(
        results_path,
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        state,
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )
    assert validation["manifest"] == str(manifest_path)
    assert validation["infrastructure_error_count"] == 1
    assert (
        validation["completed_simulation_validation"]["completed_simulation_count"] == 0
    )

    for field, invalid_value in (
        ("retryable_infrastructure_error", False),
        ("checkpoint_sha256", "0" * 64),
    ):
        tampered = copy.deepcopy(manifest)
        tampered["post_run_validation"][field] = invalid_value
        manifest_path.write_text(json.dumps(tampered))
        with pytest.raises(
            reproduction_run.RunGuardError, match="checkpoint_provenance"
        ):
            reproduction_run.validate_resume_checkpoint(
                results_path,
                HARNESS_DIR / "reference.json",
                config,
                "smoke",
                state,
                reproduction_run.DEFAULT_MODAL_APP,
                reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
            )


def test_completed_run_validator_requires_exact_coverage(monkeypatch, tmp_path):
    config = {
        "modes": {
            "smoke": {
                "task_ids": ["task_001"],
                "trials": [0],
                "expected_simulation_count": 1,
            }
        }
    }
    checkpoint = {
        "info": {},
        "tasks": [{"id": "task_001"}],
        "simulations": [{"task_id": "task_001", "trial": 0}],
    }
    monkeypatch.setattr(reproduction_run, "_load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(
        reproduction_run,
        "validate_resume_checkpoint",
        lambda *args: {
            "simulation_count": 1,
            "infrastructure_error_count": 0,
            "checkpoint_sha256": "1" * 64,
            "completed_simulation_validation": {
                "completed_simulation_count": 1,
                "grading_protocol_route_validation": True,
            },
        },
    )

    receipt = reproduction_run.validate_completed_run_checkpoint(
        tmp_path / "results.json",
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        {"digest": "state"},
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )
    assert receipt["passed"] is True
    assert receipt["actual_simulation_count"] == 1

    monkeypatch.setattr(
        reproduction_run,
        "validate_resume_checkpoint",
        lambda *args: {
            "simulation_count": 1,
            "infrastructure_error_count": 1,
            "checkpoint_sha256": "1" * 64,
            "completed_simulation_validation": {
                "completed_simulation_count": 0,
                "grading_protocol_route_validation": True,
            },
        },
    )
    retryable = reproduction_run.validate_completed_run_checkpoint(
        tmp_path / "results.json",
        HARNESS_DIR / "reference.json",
        config,
        "smoke",
        {"digest": "state"},
        reproduction_run.DEFAULT_MODAL_APP,
        reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
    )
    assert retryable["passed"] is False
    assert retryable["retryable_infrastructure_error"] is True
    assert retryable["infrastructure_error_count"] == 1

    checkpoint["simulations"] = []
    with pytest.raises(reproduction_run.RunGuardError, match="exact expected"):
        reproduction_run.validate_completed_run_checkpoint(
            tmp_path / "results.json",
            HARNESS_DIR / "reference.json",
            config,
            "smoke",
            {"digest": "state"},
            reproduction_run.DEFAULT_MODAL_APP,
            reproduction_run.DEFAULT_MODAL_SANDBOX_TIMEOUT,
        )


@pytest.mark.parametrize(
    ("termination_reason", "expected_status", "expected_exit_code"),
    (
        ("infrastructure_error", "post_run_validation_failed", 2),
        ("user_stop", "completed", 0),
    ),
)
def test_zero_exit_runner_is_finalized_only_after_checkpoint_validation(
    monkeypatch,
    tmp_path,
    termination_reason,
    expected_status,
    expected_exit_code,
):
    output_dir = tmp_path / termination_reason
    results_path = output_dir / "results.json"
    state = {"digest": "state", "runtime": {"digest": "runtime"}}
    events = []
    credit_receipt = {
        "schema_version": 1,
        "remaining_usd": 10.0,
        "required_usd": 1.0,
        "sufficient": True,
    }
    monkeypatch.setattr(reproduction_run, "load_openrouter_key", lambda *args: "k" * 20)

    def credit_check(*args):
        events.append("credit")
        return credit_receipt

    monkeypatch.setattr(reproduction_run, "fetch_openrouter_credit_state", credit_check)
    monkeypatch.setattr(
        reproduction_run, "build_paid_environment", lambda *args: {"PINNED": "1"}
    )
    monkeypatch.setattr(
        reproduction_run, "capture_reproduction_state", lambda *args, **kwargs: state
    )

    def runner(*args, **kwargs):
        assert events == ["credit"]
        events.append("runner")
        results_path.write_text(
            json.dumps({"simulations": [{"termination_reason": termination_reason}]})
        )
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(reproduction_run.subprocess, "run", runner)

    def validate_checkpoint(*args):
        checkpoint = json.loads(results_path.read_text())
        if checkpoint["simulations"][0]["termination_reason"] == "infrastructure_error":
            return {
                "passed": False,
                "retryable_infrastructure_error": True,
                "infrastructure_error_count": 1,
                "expected_simulation_count": 1,
                "actual_simulation_count": 1,
                "task_trial_coverage_sha256": "1" * 64,
                "checkpoint_sha256": "2" * 64,
                "completed_simulation_validation": {
                    "completed_simulation_count": 0,
                    "grading_protocol_route_validation": True,
                },
            }
        return {"passed": True, "actual_simulation_count": 1}

    monkeypatch.setattr(
        reproduction_run, "validate_completed_run_checkpoint", validate_checkpoint
    )
    exit_code = reproduction_run.execute_paid_plan(
        args=reproduction_run.parse_args(["smoke", "--execute"]),
        plan={"execution_state": state, "historical_chat_cost_usd": 1.0},
        config_path=HARNESS_DIR / "reference.json",
        config={"modes": {"smoke": {"historical_chat_cost_usd": 1.0}}},
        command=["paid-child"],
        prewarm_command=["prewarm"],
        manifest_environment={},
        output_dir=output_dir,
        results_path=results_path,
        cache_prewarm_required=False,
    )

    assert exit_code == expected_exit_code
    manifest = json.loads((output_dir / "reproduction_manifest_smoke.json").read_text())
    assert manifest["status"] == expected_status
    assert manifest["exit_code"] == expected_exit_code
    assert manifest["openrouter_credit_state"] == credit_receipt
    assert manifest["post_run_validation"]["passed"] is (
        termination_reason == "user_stop"
    )
    assert "k" * 20 not in json.dumps(manifest)


def test_full_shell_oracle_receipt_is_pinned_reviewed_and_acknowledged(tmp_path):
    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "mode": "full",
        "scope": "all",
        "reference_sha256": config["artifacts"]["trajectory"]["sha256"],
        "recorded_call_count": 2,
        "unique_command_count": 2,
        "selected_recorded_call_count": 2,
        "selected_unique_command_count": 2,
        "executed": True,
        "order_manifest_applied": True,
        "order_manifest_sha256": config["reproduction_transport"][
            "sandbox_order_manifest_integrity"
        ]["order_sha256"],
        "modal_image_recipe_sha256": (
            __import__(
                "tau2.knowledge.modal_sandbox_manager",
                fromlist=["MODAL_IMAGE_RECIPE_SHA256"],
            ).MODAL_IMAGE_RECIPE_SHA256
        ),
        "modal_image_object_id": config["reproduction_transport"][
            "modal_image_object_id"
        ],
        "exact_command_count": 1,
        "mismatch_command_count": 1,
        "exact_percent": 50.0,
        "mismatch_details": [{"command": "pwd", "actual_sha256": "0" * 64}],
        "mismatch_details_truncated": False,
    }
    report_path = tmp_path / "shell-oracle.json"
    report_path.write_text(json.dumps(report))
    config["reproduction_transport"]["full_shell_oracle_receipt_integrity"] = {
        "file": reproduction_run.FULL_SHELL_ORACLE_RECEIPT_RELATIVE,
        "file_sha256": reproduction_run.digest_file(report_path),
        "modal_image_recipe_sha256": report["modal_image_recipe_sha256"],
        "modal_image_object_id": report["modal_image_object_id"],
        "recorded_call_count": 2,
        "unique_command_count": 2,
        "exact_command_count": 1,
        "mismatch_command_count": 1,
        "mismatch_classification": "pwd differs only in randomized sandbox path",
        "mismatches_reviewed_and_explicitly_accepted": True,
        "score_impact_assessment": (
            "Shell output can affect model behavior and benchmark score."
        ),
    }

    with pytest.raises(reproduction_run.RunGuardError, match="acknowledge"):
        reproduction_run.verify_full_shell_oracle_receipt(
            config,
            allow_known_drift=False,
            receipt_path=report_path,
        )
    assert (
        reproduction_run.verify_full_shell_oracle_receipt(
            config,
            allow_known_drift=True,
            receipt_path=report_path,
        )["mismatch_command_count"]
        == 1
    )

    config["reproduction_transport"]["full_shell_oracle_receipt_integrity"][
        "mismatches_reviewed_and_explicitly_accepted"
    ] = False
    with pytest.raises(reproduction_run.RunGuardError, match="review and acceptance"):
        reproduction_run.verify_full_shell_oracle_receipt(
            config,
            allow_known_drift=True,
            receipt_path=report_path,
        )


def test_raw_gpt52_alias_requires_exact_bound_inventory_and_reports_raw_costs():
    candidate = {
        "simulations": [
            {
                "task_id": "task_001",
                "trial": 0,
                "messages": [
                    _attributed_message(
                        "user", "user-response-1", "user text", cost=0.0082551
                    ),
                    _attributed_message(
                        "assistant",
                        "assistant-response-1",
                        "assistant text",
                        cost=0.154672,
                    ),
                ],
            }
        ]
    }
    keys = [("task_001", 0)]

    without_proof = compare_results.validate_raw_routes(candidate, keys)
    assert without_proof["raw_route_parity"] is False

    inventory = _bound_endpoint_inventory()
    report = compare_results.validate_raw_routes(candidate, keys, inventory)
    assert report["raw_route_parity"] is True
    assert report["raw_route_gpt52_alias_inventory_proven"] is True
    assert report["raw_usage_cost_usd_by_role"] == {
        "assistant": 0.154672,
        "user": 0.0082551,
    }
    assert report["raw_usage_cost_usd_total"] == pytest.approx(0.1629271)

    swapped_raw = copy.deepcopy(candidate)
    (
        swapped_raw["simulations"][0]["messages"][0]["raw_data"],
        swapped_raw["simulations"][0]["messages"][1]["raw_data"],
    ) = (
        swapped_raw["simulations"][0]["messages"][1]["raw_data"],
        swapped_raw["simulations"][0]["messages"][0]["raw_data"],
    )
    swapped_report = compare_results.validate_raw_routes(swapped_raw, keys, inventory)
    assert swapped_report["raw_route_parity"] is False
    assert swapped_report["raw_response_binding_issue_count"] > 0
    assert "raw_routes.response_binding" in swapped_report["raw_route_mismatches"]

    missing_cost = copy.deepcopy(candidate)
    del missing_cost["simulations"][0]["messages"][0]["raw_data"]["usage"]["cost"]
    missing_cost_report = compare_results.validate_raw_routes(
        missing_cost, keys, inventory
    )
    assert missing_cost_report["raw_route_parity"] is False
    assert "raw_routes.user.usage_cost" in missing_cost_report["raw_route_mismatches"]

    second_simulation = copy.deepcopy(candidate["simulations"][0])
    second_simulation["task_id"] = "task_002"
    missing_per_simulation = {
        "simulations": [
            {"task_id": "task_001", "trial": 0, "messages": []},
            second_simulation,
        ]
    }
    missing_coverage_report = compare_results.validate_raw_routes(
        missing_per_simulation,
        [("task_001", 0), ("task_002", 0)],
        inventory,
    )
    assert missing_coverage_report["raw_route_parity"] is False
    assert (
        "raw_routes.simulation_coverage"
        in missing_coverage_report["raw_route_mismatches"]
    )

    stale = copy.deepcopy(inventory)
    stale["entries"][1]["active_endpoint_count"] += 1
    stale["digest"] = reproduction_run.canonical_digest(stale["entries"])
    stale_report = compare_results.validate_raw_routes(candidate, keys, stale)
    assert stale_report["raw_route_parity"] is False
    assert stale_report["raw_route_gpt52_alias_inventory_proven"] is False

    gate = {
        "raw_route_counters": report["raw_route_counters"],
        "raw_route_unattributed_generated_messages": {},
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
        "raw_response_binding_issue_count": 0,
        "raw_response_binding_issues": [],
        "raw_usage_cost_usd_by_role": report["raw_usage_cost_usd_by_role"],
        "raw_usage_cost_usd_total": report["raw_usage_cost_usd_total"],
        "raw_usage_cost_message_counts": report["raw_usage_cost_message_counts"],
        "raw_route_gpt52_alias_observed": True,
        "raw_route_gpt52_alias_inventory_proven": True,
    }
    assert reproduction_run.full_gate_raw_route_mismatches(gate, inventory) == {}
    assert reproduction_run.full_gate_raw_route_mismatches(gate, stale)

    config = json.loads((HARNESS_DIR / "reference.json").read_text(encoding="utf-8"))
    config["modes"]["subset"]["task_ids"] = ["task_001"]
    config["modes"]["subset"]["trials"] = [0]
    assert (
        reproduction_run.full_gate_raw_route_mismatches(
            gate, inventory, candidate, config
        )
        == {}
    )
    forged_gate = copy.deepcopy(gate)
    forged_gate["raw_route_response_id_sha256"] = "f" * 64
    assert "raw_route_response_id_sha256.candidate_binding" in (
        reproduction_run.full_gate_raw_route_mismatches(
            forged_gate, inventory, candidate, config
        )
    )

    missing_model = copy.deepcopy(candidate)
    missing_model["simulations"][0]["messages"][0]["raw_data"]["model"] = None
    missing_model_report = compare_results.validate_raw_routes(
        missing_model, keys, inventory
    )
    assert missing_model_report["raw_route_parity"] is False

    stripped = copy.deepcopy(candidate)
    stripped["simulations"][0]["messages"].extend(
        [
            {"role": "assistant", "content": "unattributed sampled response"},
            {"role": "user", "content": "unattributed sampled response"},
        ]
    )
    stripped_report = compare_results.validate_raw_routes(stripped, keys, inventory)
    assert stripped_report["raw_route_parity"] is False
    assert stripped_report["raw_route_unattributed_generated_messages"] == {
        "assistant": 1,
        "user": 1,
    }

    second = copy.deepcopy(candidate["simulations"][0])
    second["trial"] = 1
    second["messages"][0]["raw_data"]["id"] = "user-response-2"
    second["messages"][1]["raw_data"]["id"] = "assistant-response-2"
    ordered = {"simulations": [candidate["simulations"][0], second]}
    reversed_order = {"simulations": list(reversed(ordered["simulations"]))}
    ordered_report = compare_results.validate_raw_routes(
        ordered, [("task_001", 0), ("task_001", 1)], inventory
    )
    reversed_report = compare_results.validate_raw_routes(
        reversed_order, [("task_001", 0), ("task_001", 1)], inventory
    )
    assert (
        ordered_report["raw_route_response_id_sha256"]
        == reversed_report["raw_route_response_id_sha256"]
    )


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

    assert inventory["schema_version"] == 3
    assert reproduction_run.endpoint_inventory_mismatches(inventory) == {}
    qwen = inventory["entries"][0]
    assert qwen["active_endpoint_count"] == 1
    assert qwen["eligible_active_endpoint_count"] == 1
    assert qwen["matching_active_endpoint_count"] == 1

    stale = copy.deepcopy(inventory)
    stale["entries"][0]["active_endpoint_count"] = 2
    stale["digest"] = reproduction_run.canonical_digest(stale["entries"])
    mismatches = reproduction_run.endpoint_inventory_mismatches(stale)
    assert "qwen/qwen3.8-max.active_endpoint_count" in mismatches

    stale_alias = copy.deepcopy(inventory)
    stale_alias["entries"][1]["active_endpoint_count"] += 1
    stale_alias["digest"] = reproduction_run.canonical_digest(stale_alias["entries"])
    alias_mismatches = reproduction_run.endpoint_inventory_mismatches(stale_alias)
    assert "openai/gpt-5.2.active_endpoint_count" in alias_mismatches

    ineligible_snapshot = copy.deepcopy(inventory)
    ineligible_snapshot["entries"][1]["eligible_active_endpoint_count"] += 1
    ineligible_snapshot["digest"] = reproduction_run.canonical_digest(
        ineligible_snapshot["entries"]
    )
    route_mismatches = reproduction_run.endpoint_inventory_mismatches(
        ineligible_snapshot
    )
    assert "openai/gpt-5.2.eligible_active_endpoint_count" in route_mismatches

    impossible = _bound_endpoint_inventory()
    impossible["entries"][2]["active_endpoint_count"] = 1
    impossible["entries"][2]["eligible_active_endpoint_count"] = 2
    impossible["entries"][2]["matching_active_endpoint_count"] = 2
    impossible["digest"] = reproduction_run.canonical_digest(impossible["entries"])
    count_mismatches = reproduction_run.endpoint_inventory_mismatches(impossible)
    assert "openai/gpt-4.1-2025-04-14.endpoint_count_relationship" in count_mismatches


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


@pytest.mark.parametrize("ceiling", ["nan", "inf", "-inf", "0", "-1"])
def test_paid_run_rejects_non_positive_or_non_finite_cost_ceiling(
    ceiling: str, capsys: pytest.CaptureFixture[str]
):
    exit_code = reproduction_run.main(
        [
            "smoke",
            "--execute",
            "--confirm-paid-api-calls",
            f"--cost-ceiling-usd={ceiling}",
        ]
    )

    assert exit_code == 2
    assert "positive finite number" in capsys.readouterr().err
