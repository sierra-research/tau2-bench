"""Shipped English replacement tasks may differ from the registry only as reviewed.

English retail and telecom bind reviewed same-ID replacements of registry tasks.
Each known difference is normalized exactly; anything else is drift and fails
with the differing JSON paths.
"""

import json
import re
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.telecom.data_model import TelecomDB
from tau2.registry import registry
from tau2.utils import DATA_DIR

ENGLISH = DATA_DIR / "tau2" / "multilingual" / "en"
# The retail caller block appended to known_info: one line, always last.
NAME_DETAILS = re.compile(r"\n\nName details:\n[^\n]+\Z")
TELECOM_TARGET = ("C1001", "John Smith")


def _json_diff(actual, expected, path: str = "") -> list[str]:
    """Readable leaf-level differences between two JSON values."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [
            line
            for key in sorted(actual.keys() | expected.keys())
            for line in (
                _json_diff(actual[key], expected[key], f"{path}/{key}")
                if key in actual and key in expected
                else [
                    f"{path}/{key}: only in {'shipped' if key in actual else 'registry'}"
                ]
            )
        ]
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: {len(actual)} items != {len(expected)} items"]
        return [
            line
            for index, (left, right) in enumerate(zip(actual, expected))
            for line in _json_diff(left, right, f"{path}[{index}]")
        ]
    if actual == expected:
        return []
    if isinstance(actual, str) and isinstance(expected, str):
        # Show both strings from a little before their first difference.
        start = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        start = max(0, start - 20)
        actual, expected = actual[start : start + 120], expected[start : start + 120]
        path = f"{path} (from char {start})"
    return [f"{path}: shipped {actual!r} != registry {expected!r}"]


def _shipped_and_registry(domain: str) -> list[tuple[dict, dict]]:
    shipped = json.loads(Path(ENGLISH / f"{domain}_tasks_en.json").read_text())
    registry_tasks = {
        task.id: task
        for task in registry.get_tasks_loader(domain)(task_split_name=None)
    }
    assert {row["id"] for row in shipped} <= registry_tasks.keys()
    return [
        (
            Task.model_validate(row).model_dump(mode="json"),
            registry_tasks[row["id"]].model_dump(mode="json"),
        )
        for row in shipped
    ]


def _assert_no_drift(pairs: list[tuple[dict, dict]]) -> None:
    drift = [
        f"task {shipped['id']}: {line}"
        for shipped, expected in pairs
        for line in _json_diff(shipped, expected)
    ]
    assert not drift, "Shipped English tasks drifted from the registry:\n" + "\n".join(
        drift
    )


def test_english_retail_differs_from_registry_only_by_name_details():
    pairs = []
    for shipped, expected in _shipped_and_registry("retail"):
        instructions = shipped["user_scenario"]["instructions"]
        stripped, count = NAME_DETAILS.subn("", instructions["known_info"])
        assert count == 1, f"task {shipped['id']} lacks one final Name details block"
        instructions["known_info"] = stripped
        pairs.append((shipped, expected))
    assert len(pairs) == 50
    _assert_no_drift(pairs)


def _replace_once(text: str, old: str, new: str, task_id: str) -> str:
    assert text.count(old) == 1, f"task {task_id}: expected one {old!r}"
    return text.replace(old, new)


def test_english_telecom_differs_from_registry_only_by_caller_swap():
    target_id, target_name = TELECOM_TARGET
    canonical = TelecomDB.load(
        DATA_DIR / "tau2" / "domains" / "telecom" / "db.toml"
    ).model_dump(mode="json")["customers"]
    pairs = []
    for shipped, expected in _shipped_and_registry("telecom"):
        task_id = shipped["id"]
        state = shipped["initial_state"]
        [caller] = [
            call
            for call in state["initialization_actions"]
            if call["func_name"] == "set_user_info"
        ]
        name = caller["arguments"]["name"]
        # The only initialization data is the canonical customer list with the
        # target account renamed to the task's caller.
        renamed = [
            {**row, "full_name": name} if row["customer_id"] == target_id else row
            for row in canonical
        ]
        assert state["initialization_data"] == {
            "agent_data": {"customers": renamed},
            "user_data": None,
        }, f"task {task_id}: initialization_data is not the caller swap"
        state["initialization_data"] = None
        caller["arguments"]["name"] = target_name
        shipped["ticket"] = _replace_once(
            shipped["ticket"],
            f"Customer name: {name}, ",
            f"Customer name: {target_name}, ",
            task_id,
        )
        instructions = shipped["user_scenario"]["instructions"]
        instructions["known_info"] = _replace_once(
            instructions["known_info"],
            f"You are {name} with ",
            f"You are {target_name} with ",
            task_id,
        )
        pairs.append((shipped, expected))
    assert len(pairs) == 50
    _assert_no_drift(pairs)
