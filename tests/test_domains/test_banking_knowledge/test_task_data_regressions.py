import json
from pathlib import Path

BANKING_DATA_DIR = Path("data/tau2/domains/banking_knowledge")


def _load_task(task_id: str, source: Path) -> dict:
    with source.open() as fp:
        data = json.load(fp)

    if isinstance(data, list):
        return next(task for task in data if task["id"] == task_id)
    return data


def _duplicate_dispute_transaction_id(task: dict) -> str:
    for action in task["evaluation_criteria"]["actions"]:
        if action["name"] != "call_discoverable_agent_tool":
            continue
        args = action["arguments"]
        if args["agent_tool_name"] != "file_debit_card_transaction_dispute_6281":
            continue
        dispute_args = json.loads(args["arguments"])
        if dispute_args["dispute_category"] == "duplicate_charge":
            return dispute_args["transaction_id"]

    raise AssertionError("Task has no duplicate_charge dispute action")


def _transactions(task: dict) -> dict:
    return task["initial_state"]["initialization_data"]["agent_data"][
        "bank_account_transaction_history"
    ]["data"]


def _matching_duplicate_transactions(
    task: dict, *, description: str, date: str, amount: float
) -> list[dict]:
    return [
        transaction
        for transaction in _transactions(task).values()
        if transaction["description"] == description
        and transaction["date"] == date
        and transaction["amount"] == amount
    ]


def test_duplicate_charge_gold_transactions_have_observable_ordering():
    cases = [
        (
            "task_083",
            "AUSTIN COFFEE ROASTERS",
            "11/11/2025",
            -67.25,
        ),
        (
            "task_084",
            "BELLAS BISTRO CHICAGO IL",
            "11/06/2025",
            -47.5,
        ),
    ]
    sources = [
        BANKING_DATA_DIR / "tasks.json",
        BANKING_DATA_DIR / "tasks" / "task_083.json",
        BANKING_DATA_DIR / "tasks" / "task_084.json",
    ]

    for task_id, description, date, amount in cases:
        for source in sources:
            if source.name.startswith("task_") and source.stem != task_id:
                continue

            task = _load_task(task_id, source)
            duplicate_transactions = _matching_duplicate_transactions(
                task, description=description, date=date, amount=amount
            )

            assert len(duplicate_transactions) == 2
            assert all(
                "posting_sequence" in transaction
                for transaction in duplicate_transactions
            )

            sequences = {
                transaction["posting_sequence"] for transaction in duplicate_transactions
            }
            assert len(sequences) == len(duplicate_transactions)

            earliest_transaction = min(
                duplicate_transactions,
                key=lambda transaction: transaction["posting_sequence"],
            )
            assert _duplicate_dispute_transaction_id(task) == earliest_transaction[
                "transaction_id"
            ]
