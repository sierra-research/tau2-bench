"""Validation and repair helpers for task reward specifications.

The functions are deliberately pure: callers can use them in a migration or
CI check without mutating the source task list.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

AIRLINE_NL_ASSERTION_TASKS = frozenset(
    {0, 1, 2, 4, 5, 6, 9, 10, 12, 13, 26, 27, 28, 29, 31, 34, 36, 38, 41, 43, 44, 45, 46, 47, 48, 49}
)
AIRLINE_NONEMPTY_COMMUNICATE_TASKS = frozenset({3, 7, 11, 14, 18, 23})
RETAIL_NO_CRITERIA_TASKS = frozenset({10, 12, 25, 50, 57, 65})


def _task_number(task: dict[str, Any]) -> int | None:
    """Extract a numeric TNN id from the common task id fields."""
    value = task.get("id", task.get("task_id"))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.upper().startswith("T"):
        try:
            return int(value[1:])
        except ValueError:
            return None
    return None


def _basis_names(task: dict[str, Any]) -> list[str]:
    basis = task.get("reward_basis") or []
    return [item.get("type", item) if isinstance(item, dict) else item for item in basis]


def repair_reward_basis(tasks: Iterable[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    """Return repaired task copies for the known airline reward defects.

    Retail tasks with no criteria are reported by :func:`find_no_criteria`;
    they are not guessed or assigned fabricated assertions.
    """
    result = deepcopy(list(tasks))
    if domain.lower() != "airline":
        return result
    for task in result:
        number = _task_number(task)
        basis = task.get("reward_basis")
        if not isinstance(basis, list):
            continue
        if number in AIRLINE_NL_ASSERTION_TASKS and "NL_ASSERTION" not in _basis_names(task):
            basis.append("NL_ASSERTION")
        if number not in AIRLINE_NONEMPTY_COMMUNICATE_TASKS:
            task["reward_basis"] = [
                item for item in basis
                if (item.get("type", item) if isinstance(item, dict) else item) != "COMMUNICATE"
            ]
    return result


def find_no_criteria(tasks: Iterable[dict[str, Any]], domain: str) -> list[str]:
    """Return IDs that have no populated reward criterion at all."""
    expected = RETAIL_NO_CRITERIA_TASKS if domain.lower() == "retail" else frozenset()
    found: list[str] = []
    for task in tasks:
        number = _task_number(task)
        if number not in expected:
            continue
        basis = task.get("reward_basis") or []
        populated = any(
            isinstance(item, dict) and any(item.get(key) not in (None, "", [], {}) for key in ("criteria", "value"))
            for item in basis
        )
        assertions = task.get("nl_assertions") or task.get("nl_assertion")
        if not populated and not assertions:
            found.append(str(task.get("id", task.get("task_id", f"T{number}"))))
    return found
