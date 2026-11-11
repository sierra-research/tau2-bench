"""Regression tests for banking_knowledge task_082."""

import json
from pathlib import Path

from tau2.data_model.message import ToolCall
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.environment.environment import Environment

from .conftest import create_environment

TASK_PATH = (
    Path(__file__).parents[4]
    / "data/tau2/domains/banking_knowledge/tasks/task_082.json"
)


def _load_task_082() -> dict:
    with TASK_PATH.open() as fp:
        return json.load(fp)


def _create_task_082_environment() -> Environment:
    task = _load_task_082()
    db_data = task["initial_state"]["initialization_data"]["agent_data"]
    return create_environment(TransactionalDB.model_validate(db_data))


def _run_gold_actions(environment: Environment, task: dict) -> None:
    for action in task["evaluation_criteria"]["actions"]:
        response = environment.get_response(
            ToolCall(
                id=action["action_id"],
                name=action["name"],
                arguments=action["arguments"],
                requestor=action.get("requestor", "assistant"),
            )
        )
        assert not response.error, f"{action['action_id']} failed: {response.content}"


class TestTask082:
    """Tests for T082's multi-card debit dispute remediation."""

    def test_close_and_reissue_disputes_secure_each_affected_debit_card(self):
        task = _load_task_082()
        environment = _create_task_082_environment()

        _run_gold_actions(environment, task)

        db = environment.tools.db
        close_and_reissue_card_ids = {
            dispute["card_id"]
            for dispute in db.debit_card_disputes.data.values()
            if dispute["card_action"] == "close_and_reissue"
        }
        closed_card_ids = {
            card_id
            for card_id, card in db.debit_cards.data.items()
            if card.get("status") == "CLOSED"
        }
        ordered_account_ids = {
            order["account_id"] for order in db.debit_card_orders.data.values()
        }
        reissued_card_account_ids = {
            db.debit_cards.data[card_id]["account_id"]
            for card_id in close_and_reissue_card_ids
        }

        assert close_and_reissue_card_ids == {
            "dbc_mc47a2b9e1_blue",
            "dbc_mc47a2b9e1_gff",
        }
        assert close_and_reissue_card_ids <= closed_card_ids
        assert reissued_card_account_ids <= ordered_account_ids
