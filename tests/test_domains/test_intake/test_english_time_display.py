"""English caller clock phrases leave frozen tasks and tool answers intact."""

import hashlib
import random

import pytest

from tau2.data_model.elicitation import ComplicationKind
from tau2.domains.intake import complications, environment
from tau2.domains.intake.scoring import project_scoring_folds
from tau2.domains.intake.time_display import (
    TIME_CALLER_GUIDANCE,
    display_time_payload,
)
from tau2.domains.intake.utils import spoken_form


def _calls(task):
    return {
        action.func_name: action.arguments
        for action in task.initial_state.initialization_actions
    }


def _time_task(value="2:15 PM"):
    task = next(
        task
        for task in environment.load_frozen_tasks()
        if task.id.startswith("intake_times_easy_")
    ).model_copy(deep=True)
    calls = _calls(task)
    calls["set_entities"]["entities"]["appointment_time"] = value
    field = next(
        field
        for field in calls["seed_record"]["record"]["fields"]
        if field["name"] == "appointment_time"
    )
    field["value"] = value
    task.evaluation_criteria.actions[0].arguments["fields"]["appointment_time"] = value
    return task


def _force_kind(monkeypatch, kind):
    monkeypatch.setattr(
        complications,
        "effective_kind_rates",
        lambda *_: {item: float(item is kind) for item in ComplicationKind},
    )


@pytest.mark.parametrize("strategy", ["scaffolded", "agent_directed"])
def test_both_english_loaders_project_callers_but_preserve_frozen_gold(strategy):
    frozen = {task.id: task for task in environment.load_frozen_tasks()}
    before = {key: task.model_dump_json() for key, task in frozen.items()}
    loaded = environment.get_tasks(strategy=strategy)
    changed = 0
    for task in loaded:
        original = frozen[task.id]
        original_calls, calls = _calls(original), _calls(task)
        assert (
            calls["seed_record"]
            == _calls(project_scoring_folds(original))["seed_record"]
        )
        assert (
            task.evaluation_criteria.actions[0].arguments["fields"]
            == original.evaluation_criteria.actions[0].arguments["fields"]
        )
        original_entities = original_calls["set_entities"]["entities"]
        entities = calls["set_entities"]["entities"]
        if "appointment_time" in entities:
            assert entities["appointment_time"] == display_time_payload(
                original_entities["appointment_time"], "en"
            )
            assert (
                TIME_CALLER_GUIDANCE
                in task.user_scenario.instructions.task_instructions
            )
            changed += entities != original_entities
        else:
            assert entities == original_entities
            assert (
                TIME_CALLER_GUIDANCE
                not in task.user_scenario.instructions.task_instructions
            )
    assert changed > 0
    assert {key: task.model_dump_json() for key, task in frozen.items()} == before


@pytest.mark.parametrize(
    ("value", "spoken"),
    [
        ("2:15 PM", "quarter past two"),
        ("2:30 PM", "half past two"),
        ("1:45 PM", "quarter to two"),
    ],
)
def test_english_caller_knows_period_but_begins_with_bare_idiom(value, spoken):
    original = _time_task(value)
    before = original.model_dump_json()
    task = environment.english_time_display_task(original)
    payload = _calls(task)["set_entities"]["entities"]["appointment_time"]
    assert spoken_form(payload) == spoken
    assert value in payload
    assert original.model_dump_json() == before
    assert environment.english_time_display_task(task) == task


def test_meridiem_flip_slip_remains_audibly_different(monkeypatch):
    _force_kind(monkeypatch, ComplicationKind.SELF_CORRECTION)
    monkeypatch.setattr(
        complications, "_pool_values", lambda *_: ["2:15 PM", "2:15 AM"]
    )
    task = environment.english_time_display_task(_time_task())
    sample = complications.sample_complication(
        42, task, rate_override=1, channel="voice"
    )
    assert sample.kind is ComplicationKind.SELF_CORRECTION
    assert sample.params.decoy_value == "2:15 AM"
    assert sample.params.decoy_value in sample.line
    assert "records:" not in sample.line
    payload = _calls(task)["set_entities"]["entities"]["appointment_time"]
    assert sample.params.decoy_value != spoken_form(payload)
    assert task.evaluation_criteria.actions[0].arguments["fields"] == {
        "appointment_time": "2:15 PM"
    }


def test_time_omission_and_restart_use_the_correct_side_of_dual_form(monkeypatch):
    task = environment.english_time_display_task(_time_task())
    _force_kind(monkeypatch, ComplicationKind.LAZY_OMISSION)
    sample = complications.sample_complication(
        42, task, rate_override=1, channel="voice"
    )
    assert sample.kind is ComplicationKind.LAZY_OMISSION
    assert '"quarter past two"' in sample.line
    assert "2:15 PM" not in sample.line
    assert sample.params.omitted_component == "am_pm"

    _force_kind(monkeypatch, ComplicationKind.SPELL_CORRECTION)
    sample = complications.sample_complication(
        42, task, rate_override=1, channel="voice"
    )
    assert sample.kind is ComplicationKind.SPELL_CORRECTION
    assert sample.params.sequence_length == len("215PM")
    payload = _calls(task)["set_entities"]["entities"]["appointment_time"]
    assert complications._spell_units(payload) == list("215PM")


def test_wrong_slot_time_is_a_complete_value_not_an_annotation(monkeypatch):
    task = next(
        task
        for task in environment.load_frozen_tasks()
        if task.id.startswith("intake_person_names_easy_")
    ).model_copy(deep=True)
    calls = _calls(task)
    gold_field = next(iter(calls["set_entities"]["entities"]))
    record = calls["seed_record"]["record"]
    record["fields"] = [
        field for field in record["fields"] if field["name"] == gold_field
    ] + [
        {
            "name": "appointment_time",
            "fold": "time",
            "value": "2:15 PM",
            "description": "Appointment time",
        }
    ]
    _force_kind(monkeypatch, ComplicationKind.WRONG_SLOT)
    sample = complications.sample_complication(
        42, task, rate_override=1, channel="voice"
    )
    assert sample.kind is ComplicationKind.WRONG_SLOT
    assert sample.params.off_slot_value == "2:15 PM"
    assert "records:" not in sample.line


def test_updated_render_catalog_preserves_previous_seed_stream():
    task_id = "intake_times_easy_01"
    old_seed = hashlib.sha256(f"2.4.0:42:{task_id}".encode()).digest()
    old_rng = random.Random(int.from_bytes(old_seed[:8], "big"))
    current_rng = complications._rng(42, task_id)
    assert [current_rng.random() for _ in range(10)] == [
        old_rng.random() for _ in range(10)
    ]
    assert complications.COMPLICATION_CATALOG_VERSION == "2.5.0"
