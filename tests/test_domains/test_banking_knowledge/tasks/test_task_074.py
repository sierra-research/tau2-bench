"""Tests for task_074."""

import json

from tau2.utils import DATA_DIR

TASK_ID = "task_074"
LIGHT_BLUE_ACCOUNT_ID = "chk_ar72c5d8e3_2"
EXPECTED_LIGHT_BLUE_CREDIT = 14.50


def _credit_amount_for_account(task: dict, account_id: str) -> float:
    for action in task["evaluation_criteria"]["actions"]:
        if action["name"] != "call_discoverable_agent_tool":
            continue
        arguments = action["arguments"]
        if arguments["agent_tool_name"] != "apply_checking_account_credit_5829":
            continue
        tool_arguments = json.loads(arguments["arguments"])
        if tool_arguments["account_id"] == account_id:
            return tool_arguments["amount"]
    raise AssertionError(f"Missing checking account credit for {account_id}")


def test_task_074_light_blue_credit_includes_both_free_atm_allowances():
    task_dir = DATA_DIR / "tau2" / "domains" / "banking_knowledge" / "tasks"
    task_file = task_dir / f"{TASK_ID}.json"
    with open(task_file) as fp:
        task = json.load(fp)

    assert _credit_amount_for_account(task, LIGHT_BLUE_ACCOUNT_ID) == (
        EXPECTED_LIGHT_BLUE_CREDIT
    )


def test_task_074_combined_tasks_json_matches_task_file():
    tasks_file = DATA_DIR / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
    with open(tasks_file) as fp:
        tasks = json.load(fp)

    task = next(task for task in tasks if task["id"] == TASK_ID)
    assert _credit_amount_for_account(task, LIGHT_BLUE_ACCOUNT_ID) == (
        EXPECTED_LIGHT_BLUE_CREDIT
    )
