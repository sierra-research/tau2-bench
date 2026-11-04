"""Caller time expressions retain private day-period knowledge and canonical gold."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from tau2.data_model.elicitation import ComplicationKind
from tau2.data_model.tasks import Task
from tau2.domains.intake.localization import validate_intake_tasks
from tau2.domains.intake.localized import (
    IntakeLocalizationData,
    sample_localized_complication,
)
from tau2.domains.intake.spell_events import entity_match_texts
from tau2.domains.intake.time_display import (
    TIME_CALLER_GUIDANCE,
    TIME_DISPLAY_VERSION,
    display_time,
    display_time_payload,
    locale_time_format,
    time_expression,
)
from tau2.domains.intake.utils import render_time_dual_form, spoken_form
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.multilingual.factory.localization_entity_csv import entity_review_rows
from tau2.utils.utils import DATA_DIR


@pytest.mark.parametrize(
    "value,style,expected",
    [
        ("1:45 PM", "hi", "पौने दो बजे"),
        ("2:15 PM", "hi", "सवा दो बजे"),
        ("1:30 AM", "hi", "डेढ़ बजे"),
        ("2:30 PM", "hi", "ढाई बजे"),
        ("3:30 PM", "hi", "साढ़े तीन बजे"),
        ("11:45 PM", "hi", "पौने बारह बजे"),
        ("12:45 AM", "hi", "पौने एक बजे"),
        ("2:15 PM", "en", "quarter past two"),
        ("2:30 PM", "en", "half past two"),
        ("2:45 PM", "en", "quarter to three"),
        ("11:45 AM", "en", "quarter to twelve"),
        ("11:45 PM", "en", "quarter to twelve"),
        ("1:00 AM", "es", "la una"),
        ("2:00 PM", "es", "las dos"),
        ("4:00 PM", "es", "las cuatro"),
        ("11:00 AM", "es", "las once"),
        ("11:00 PM", "es", "las once"),
        ("12:00 AM", "es", "las doce"),
        ("12:00 PM", "es", "las doce"),
        ("2:15 PM", "es", "las dos y cuarto"),
        ("2:30 PM", "es", "las dos y media"),
        ("12:45 AM", "es", "la una menos cuarto"),
        ("2:15 PM", "pt", "duas e quinze"),
        ("2:30 PM", "pt", "duas e meia"),
        ("2:45 PM", "pt", "quinze para as três"),
        ("12:45 PM", "pt", "quinze para a uma"),
        ("2:15 PM", "zh", "2点一刻"),
        ("2:30 PM", "zh", "2点半"),
        ("2:45 PM", "zh", "2点三刻"),
        ("2:30 PM", "ko", "2시 반"),
        ("2:45 PM", "ko", "3시 15분 전"),
        ("12:45 AM", "ko", "1시 15분 전"),
    ],
)
def test_idioms_start_without_period_but_preserve_known_complete_time(
    value, style, expected
):
    payload = display_time_payload(value, style)
    assert spoken_form(payload) == expected
    assert entity_match_texts(payload) == [expected, display_time(value, style)]
    assert "records" not in " ".join(entity_match_texts(payload))


@pytest.mark.parametrize(
    "style", ["en", "hi", "hi-latn", "es", "pt", "zh", "zh-latn", "ko", "ko-latn"]
)
def test_every_instant_has_distinct_full_anchor_and_period_omission_is_ambiguous(style):
    anchors = {
        display_time(f"{hour}:{minute:02} {period}", style)
        for hour in range(1, 13)
        for minute in range(60)
        for period in ("AM", "PM")
    }
    assert len(anchors) == 1440
    for hour in range(1, 13):
        for minute in (0, 15, 30, 45):
            am, pm = (f"{hour}:{minute:02} {p}" for p in ("AM", "PM"))
            assert display_time(am, style, include_period=False) == display_time(
                pm, style, include_period=False
            )
            assert time_expression(am, style) == time_expression(pm, style)


def test_complete_boundary_forms_and_plain_times_are_unambiguous():
    assert display_time_payload("9:40 PM", "ko") == "오후 9시 40분"
    assert display_time("12:00 AM", "ko") == "밤 12시"
    assert display_time("12:00 PM", "ko") == "낮 12시"
    assert display_time("12:00 AM", "zh") == "午夜12点整"
    assert display_time("12:00 PM", "zh") == "中午12点整"
    assert display_time("12:00 PM", "es") == "12:00 del mediodía"
    assert time_expression("2:15 PM", "ko") is None  # no literal Korean quarter calque
    assert display_time_payload("not a time", "hi") == "not a time"
    with pytest.raises(ValueError, match="canonical"):
        display_time("13:60 PM", "hi")


@pytest.fixture(
    scope="module",
    params=[
        (language, representation)
        for language in ("hi", "es", "pt", "zh", "ko")
        for representation in ("native", "romanized")
    ],
)
def compiled_times(request):
    language, representation = request.param
    mapping = read_localization_map(
        Path(DATA_DIR) / f"tau2/multilingual/{language}/localization/intake.json"
    )
    docs = {d.id: d for d in compile_localization(mapping, representation)}
    tasks = TypeAdapter(list[Task]).validate_python(docs["tasks"].content)
    selected = IntakeLocalizationData.model_validate(
        docs["intake_localization"].content
    )
    return language, representation, mapping, tasks, selected


def test_frozen_runtime_times_and_csv_follow_same_projection(compiled_times):
    language, representation, mapping, tasks, selected = compiled_times
    style = locale_time_format(language, representation)
    stimuli = [s for s in selected.stimuli if s.family == "times"]
    authored = {
        surface.canonical_task_id: surface for surface in mapping.intake_caller_surfaces
    }
    assert len(stimuli) == 20
    assert all(s.time_display_format == style for s in selected.stimuli)
    for stimulus in stimuli:
        task = next(t for t in tasks if t.id == stimulus.canonical_task_id)
        actions = {
            a.func_name: a.arguments for a in task.initial_state.initialization_actions
        }
        gold = task.evaluation_criteria.actions[0].arguments["fields"][
            stimulus.field_name
        ]
        payload = actions["set_entities"]["entities"][stimulus.field_name]
        assert gold == stimulus.written
        surface = authored.get(stimulus.canonical_task_id)
        if surface is not None:
            spoken = getattr(surface, f"{representation}_spoken")
            assert stimulus.caller_spoken == spoken
            assert payload == render_time_dual_form(spoken, display_time(gold, style))
        else:
            assert stimulus.caller_spoken is None
            assert payload == display_time_payload(gold, style)
        assert TIME_CALLER_GUIDANCE in task.user_scenario.instructions.task_instructions
        assert all(d != display_time(gold, style) for d in stimulus.decoys)
        if stimulus.omission_instruction:
            omitted = (
                stimulus.caller_spoken
                or time_expression(gold, style)
                or display_time(gold, style, include_period=False)
            )
            assert f'"{omitted}"' in stimulus.omission_instruction
        if representation == "romanized":
            assert not any(
                "\u0900" <= c <= "\u097f"
                or "\u4e00" <= c <= "\u9fff"
                or "\uac00" <= c <= "\ud7af"
                for c in payload
            )
    validate_intake_tasks(tasks, language, representation)
    if representation == "native":
        rows = entity_review_rows(mapping)
        time_rows = [
            r
            for r in rows
            if json.loads(r.lineage).get("display_projection") == TIME_DISPLAY_VERSION
        ]
        assert len(time_rows) >= 20
        entries = {entry.id: entry for entry in mapping.entries}
        for row in time_rows:
            lineage = json.loads(row.lineage)
            if overrides := lineage.get("intake_caller_surfaces"):
                expected = authored[overrides[0]["canonical_task_id"]].native_spoken
            else:
                current = entries[lineage["source_entry_id"]].forms.native
                for component in lineage["value_path"]:
                    current = current[component]
                expected = (
                    display_time(current, style)
                    if row.review_type == "Decoy"
                    else spoken_form(display_time_payload(current, style))
                )
            assert row.suggested_entity == expected
            assert row.decision == row.proposed_entity == row.reason == ""
    damaged = [t.model_copy(deep=True) for t in tasks]
    task = next(t for t in damaged if t.id == stimuli[0].canonical_task_id)
    call = next(
        a
        for a in task.initial_state.initialization_actions
        if a.func_name == "set_entities"
    )
    call.arguments["entities"][stimuli[0].field_name] = display_time_payload(
        "1:30 AM", style
    )
    with pytest.raises(ValueError, match="caller value"):
        validate_intake_tasks(damaged, language, representation)


@pytest.mark.parametrize(
    "kind",
    [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.LAZY_OMISSION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.WRONG_SLOT,
    ],
)
def test_localized_complications_use_time_surfaces_without_annotation_leak(
    compiled_times, kind, monkeypatch
):
    from tau2.domains.intake import complications

    language, representation, _, tasks, selected = compiled_times
    style = locale_time_format(language, representation)
    monkeypatch.setattr(
        complications, "effective_kind_rates", lambda *args: {kind: 1.0}
    )
    task = next(t for t in tasks if t.id == "intake_times_hard_09").model_copy(
        deep=True
    )
    stimulus = selected.for_task(task.id)
    if kind == ComplicationKind.WRONG_SLOT:
        record = next(
            a.arguments["record"]
            for a in task.initial_state.initialization_actions
            if a.func_name == "seed_record"
        )
        record["fields"] = [
            f for f in record["fields"] if f["name"] == stimulus.field_name
        ] + [{"name": "other_time", "fold": "time", "value": "2:30 AM"}]
    result = sample_localized_complication(
        2, task, stimulus, rate_override=1, channel="voice"
    )
    assert result is not None
    assert "the real time" not in result.line
    if kind == ComplicationKind.SELF_CORRECTION:
        assert result.params.decoy_value in stimulus.decoys
    if kind == ComplicationKind.WRONG_SLOT:
        assert result.params.off_slot_value == display_time("2:30 AM", style)
    if representation == "native":
        assert "AM" not in result.line and "PM" not in result.line


def test_spanish_whole_hours_omit_only_the_period():
    from tau2.domains.intake.time_display import omission_instruction

    for canonical, spoken in [
        ("2:00 PM", "las dos"),
        ("4:00 PM", "las cuatro"),
        ("11:00 AM", "las once"),
    ]:
        payload = display_time_payload(canonical, "es")
        assert spoken_form(payload) == spoken
        assert display_time(canonical, "es") in entity_match_texts(payload)
        assert f'give only "{spoken}"' in omission_instruction(canonical, "es")
