# Copyright Sierra
"""Tests for the Hindi-localized airline task subset (``airline_hi``).

Localization invariants:
- Exactly 10 tasks; each id is ``<source_id>_hi`` for an existing airline task.
- ``evaluation_criteria`` is byte-for-byte identical to the English source.
- Every concrete value (emails, user ids, reservation/flight codes, numbers)
  appearing in the English instructions appears verbatim in the Hindi
  instructions (values must stay in Latin script for tool lookups/ASR tests).
- Prose is actually localized (contains Devanagari).
"""

import json
import re

import pytest

from tau2.data_model.tasks import Task
from tau2.domains.airline.utils import AIRLINE_TASK_SET_HI_PATH, AIRLINE_TASK_SET_PATH
from tau2.registry import registry

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Concrete-value extractors for the English source instructions.
VALUE_PATTERNS = [
    # Emails (none in current selection, but guard against future edits).
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # User ids, e.g. anya_garcia_5901 (also matches quoted daiki_muller_1116).
    re.compile(r"\b[a-z]+_[a-z]+_\d+\b"),
    # Uppercase alphanumeric codes containing a digit: reservation ids
    # (JMO1MG, 1N99U6), flight numbers (HAT169), etc.
    re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{2,}\b|\b\d[A-Z0-9]{2,}\b"),
    # Numbers: amounts, dates, last-4 digits, DOBs (each numeric run must
    # survive verbatim, e.g. 2001-04-12 -> 2001, 04, 12).
    re.compile(r"\d+"),
]


def _load_raw(path):
    with open(path) as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def hi_tasks_raw() -> list[dict]:
    return _load_raw(AIRLINE_TASK_SET_HI_PATH)


@pytest.fixture(scope="module")
def source_tasks_raw() -> dict[str, dict]:
    return {task["id"]: task for task in _load_raw(AIRLINE_TASK_SET_PATH)}


def _instruction_text(task: dict) -> str:
    instructions = task["user_scenario"]["instructions"]
    fields = ["task_instructions", "reason_for_call", "known_info", "unknown_info"]
    return "\n".join(instructions[field] or "" for field in fields)


def test_airline_hi_task_set_loads_via_registry():
    loader = registry.get_tasks_loader("airline_hi")
    tasks = loader(task_split_name="base")
    assert len(tasks) == 10
    assert all(isinstance(task, Task) for task in tasks)
    # Default split (None) behaves the same; unknown splits are rejected.
    assert len(loader(task_split_name=None)) == 10
    with pytest.raises(ValueError):
        loader(task_split_name="full")


def test_airline_hi_registered_in_task_sets():
    assert "airline_hi" in registry.get_task_sets()


def test_exactly_ten_tasks_with_hi_suffix_mapping_to_source(
    hi_tasks_raw, source_tasks_raw
):
    assert len(hi_tasks_raw) == 10
    ids = [task["id"] for task in hi_tasks_raw]
    assert len(set(ids)) == 10
    for task_id in ids:
        assert task_id.endswith("_hi")
        assert task_id.removesuffix("_hi") in source_tasks_raw


def test_evaluation_criteria_identical_to_source(hi_tasks_raw, source_tasks_raw):
    for task in hi_tasks_raw:
        source = source_tasks_raw[task["id"].removesuffix("_hi")]
        assert task["evaluation_criteria"] == source["evaluation_criteria"], (
            f"evaluation_criteria modified for task {task['id']}"
        )


def test_concrete_values_preserved_verbatim(hi_tasks_raw, source_tasks_raw):
    for task in hi_tasks_raw:
        source = source_tasks_raw[task["id"].removesuffix("_hi")]
        source_text = _instruction_text(source)
        hi_text = _instruction_text(task)
        values = set()
        for pattern in VALUE_PATTERNS:
            values.update(pattern.findall(source_text))
        # Drop pure-English words accidentally caught by the code pattern
        # (e.g. "DOB", "NOT"): keep only tokens containing a digit or '@'.
        values = {v for v in values if any(ch.isdigit() for ch in v) or "@" in v}
        for value in sorted(values):
            assert value in hi_text, (
                f"Value '{value}' from source task "
                f"{source['id']} missing in Hindi instructions of {task['id']}"
            )


def test_tasks_validate_against_task_model(hi_tasks_raw):
    for task in hi_tasks_raw:
        Task.model_validate(task)


def test_prose_contains_devanagari(hi_tasks_raw):
    for task in hi_tasks_raw:
        purpose = task["description"]["purpose"] or ""
        assert DEVANAGARI_RE.search(purpose), (
            f"description.purpose of {task['id']} not localized"
        )
        instructions = task["user_scenario"]["instructions"]
        for field in ["task_instructions", "reason_for_call", "known_info"]:
            text = instructions[field] or ""
            assert DEVANAGARI_RE.search(text), f"{field} of {task['id']} not localized"


def test_values_stay_latin_script(hi_tasks_raw):
    """IDs/codes must not be transliterated into Devanagari digits/letters."""
    for task in hi_tasks_raw:
        text = _instruction_text(task)
        # No Devanagari digits anywhere (०-९).
        assert not re.search(r"[०-९]", text), f"Devanagari digits found in {task['id']}"
