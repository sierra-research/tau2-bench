"""Caller dates follow local order while agent tool submissions stay ISO."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from tau2.data_model.tasks import Task
from tau2.domains.intake.date_display import (
    display_date,
    display_date_payload,
    locale_date_format,
    matches_date_payload,
)
from tau2.domains.intake.localization import validate_intake_tasks
from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.domains.intake.utils import render_dual_form, spoken_form
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.utils.utils import DATA_DIR


@pytest.mark.parametrize(
    "style,expected",
    [
        ("dmy-hyphen", "19-08-2025"),
        ("dmy-slash", "19/08/2025"),
        ("ymd-hanzi", "2025年8月19日"),
        ("ymd-hangul", "2025년 8월 19일"),
        ("ymd-hyphen", "2025-08-19"),
    ],
)
def test_plain_and_relative_dates_keep_same_calendar_day(style, expected):
    assert display_date("2025-08-19", style) == expected
    value = display_date_payload(render_dual_form("next Tuesday", "2025-08-19"), style)
    assert spoken_form(value) == "next Tuesday"
    assert expected in value
    assert matches_date_payload(value, "2025-08-19", style)
    assert not matches_date_payload(value, "2025-09-19", style)


def test_hindi_runtime_dates_decoys_omissions_and_iso_gold_agree():
    mapping = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )
    documents = {d.id: d for d in compile_localization(mapping)}
    tasks = TypeAdapter(list[Task]).validate_python(documents["tasks"].content)
    selected = IntakeLocalizationData.model_validate(
        documents["intake_localization"].content
    )
    dates = [s for s in selected.stimuli if s.family == "dates"]
    assert len(dates) == 20
    for stimulus in dates:
        task = next(t for t in tasks if t.id == stimulus.canonical_task_id)
        actions = {
            a.func_name: a.arguments for a in task.initial_state.initialization_actions
        }
        caller_value = actions["set_entities"]["entities"][stimulus.field_name]
        gold = task.evaluation_criteria.actions[0].arguments["fields"][
            stimulus.field_name
        ]
        assert gold == stimulus.written
        assert matches_date_payload(caller_value, gold, "dmy-hyphen")
        assert all(
            len(value) == 10 and value[2] == value[5] == "-"
            for value in stimulus.decoys
        )
        if stimulus.omission_instruction:
            assert (
                display_date(gold, "dmy-hyphen", include_year=False)
                in stimulus.omission_instruction
            )
            assert "month and day" not in stimulus.omission_instruction
    validate_intake_tasks(tasks, "hi")
    task = next(t for t in tasks if t.id == dates[0].canonical_task_id)
    entity_call = next(
        a
        for a in task.initial_state.initialization_actions
        if a.func_name == "set_entities"
    )
    entity_call.arguments["entities"][dates[0].field_name] = "01-01-2000"
    with pytest.raises(ValueError, match="caller value"):
        validate_intake_tasks(tasks, "hi")


@pytest.mark.parametrize("language", ["zh", "ko"])
def test_romanized_dates_preserve_local_order_without_native_script(language):
    style = locale_date_format(language, "romanized")
    assert display_date("2025-08-19", style) == "2025-08-19"
    assert display_date("2025-08-19", style, include_year=False) == "08-19"
