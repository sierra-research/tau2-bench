# Copyright Sierra
"""Tests for the extracted localization engine (tau2.multilingual.localize_lib).

The library is the engine behind the retired translator CSV round-trip (and the factory's
``translate`` stage). These tests check both layers: the pure extract/inject
round trip on a tiny in-memory task, and that the tasks-csv verbs produce
exactly the same artifacts as the library calls.
"""

import copy

import pytest

from tau2.multilingual.localize_lib import (
    inject_translations,
    iter_rows,
    read_filled_csv,
    resolve_script_code,
    write_translation_csv,
)

# A tiny task with one of everything the flattener handles: instruction
# fields (one empty), metadata fields, and concrete values (code + user id).
TOY_TASK = {
    "id": "0",
    "description": {
        "purpose": "Caller cancels a reservation.",
        "notes": "Keep it short.",
    },
    "user_scenario": {
        "instructions": {
            "task_instructions": "Cancel reservation ABC123 politely.",
            "reason_for_call": "You want to cancel a flight.",
            "known_info": "You are mia_li_3668.",
            "unknown_info": None,
        }
    },
    "evaluation_criteria": {"actions": [], "communicate_info": ["ABC123"]},
}


def localize(text: str) -> str:
    """A fake Cyrillic 'translation' that keeps every concrete value."""
    return f"Тест: {text}"


def filled_translations() -> dict[tuple[str, str], str]:
    return {
        (TOY_TASK["id"], label): localize(text) for label, text in iter_rows(TOY_TASK)
    }


class TestRoundTrip:
    def test_iter_rows_skips_empty_fields(self):
        labels = [label for label, _ in iter_rows(TOY_TASK)]
        assert labels == [
            "task_instructions",
            "reason_for_call",
            "known_info",
            "purpose (metadata)",
            "notes (metadata)",
        ]

    def test_extract_then_read_round_trips(self, tmp_path):
        csv_path = tmp_path / "toy_tasks_xv_translation.csv"
        rows = write_translation_csv([TOY_TASK], csv_path)
        assert rows == 5
        # An untouched extract has empty translations for every row.
        translations = read_filled_csv(csv_path)
        assert set(translations) == {
            (TOY_TASK["id"], label) for label, _ in iter_rows(TOY_TASK)
        }
        assert all(value == "" for value in translations.values())

    def test_inject_applies_translations_and_passes_invariants(self):
        localized, problems = inject_translations(
            [copy.deepcopy(TOY_TASK)],
            filled_translations(),
            lang="xv",
            script_code="cyrl",
        )
        assert problems == []
        (task,) = localized
        assert task["id"] == "0_xv"
        instructions = task["user_scenario"]["instructions"]
        assert instructions["task_instructions"] == localize(
            "Cancel reservation ABC123 politely."
        )
        assert instructions["unknown_info"] is None
        assert task["description"]["purpose"] == localize(
            "Caller cancels a reservation."
        )
        # Invariant: evaluation criteria byte-identical to the source.
        assert task["evaluation_criteria"] == TOY_TASK["evaluation_criteria"]

    def test_inject_reports_missing_translations(self):
        translations = filled_translations()
        translations[("0", "known_info")] = ""
        _, problems = inject_translations(
            [copy.deepcopy(TOY_TASK)], translations, lang="xv", script_code="cyrl"
        )
        assert any("field 'known_info' not translated" in p for p in problems)

    def test_inject_reports_dropped_values(self):
        translations = filled_translations()
        translations[("0", "task_instructions")] = "Тест: без кода брони."
        _, problems = inject_translations(
            [copy.deepcopy(TOY_TASK)], translations, lang="xv", script_code="cyrl"
        )
        assert any("'ABC123'" in p for p in problems)


class TestResolveScriptCode:
    def test_override_wins(self):
        assert resolve_script_code("anything", "thai") == "thai"

    def test_pack_script_used(self):
        assert resolve_script_code("hi") == "deva"

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="pass\\s+--script-code"):
            resolve_script_code("nope")
