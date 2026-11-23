"""Keep aggregate banking tasks in sync with the files used for evaluation.

These data-only checks can also run with ``pytest --noconftest`` without
importing the framework or optional retrieval dependencies.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "tau2"
    / "domains"
    / "banking_knowledge"
)
TASK_FILES = sorted((DATA_DIR / "tasks").glob("task_*.json"))


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_task_ids_are_unique():
    """Do not let dictionary construction silently discard duplicate IDs."""
    for tasks in (
        _load_json(DATA_DIR / "tasks.json"),
        [_load_json(path) for path in TASK_FILES],
    ):
        counts = Counter(task["id"] for task in tasks)
        duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
        assert not duplicates, f"Duplicate task IDs: {duplicates}"


def test_task_ids_match():
    aggregate_ids = {task["id"] for task in _load_json(DATA_DIR / "tasks.json")}
    individual_ids = {_load_json(path)["id"] for path in TASK_FILES}
    assert individual_ids, "No individual task files found"
    assert aggregate_ids == individual_ids, (
        f"Missing from aggregate: {sorted(individual_ids - aggregate_ids)}; "
        f"missing from individual files: {sorted(aggregate_ids - individual_ids)}"
    )


@pytest.mark.parametrize("task_file", TASK_FILES, ids=lambda path: path.stem)
def test_task_definitions_match(task_file):
    individual = _load_json(task_file)
    aggregate = {task["id"]: task for task in _load_json(DATA_DIR / "tasks.json")}
    assert aggregate.get(individual["id"]) == individual
