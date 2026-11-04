"""Source-bound hard time surfaces preserve exact private anchors and tool truth."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.domains.intake.spell_events import entity_match_texts
from tau2.multilingual.factory.localization_artifact import (
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.multilingual.factory.localization_entity_csv import entity_review_rows
from tau2.multilingual.localization import IntakeCallerSurface, LocalizationMap
from tau2.utils.utils import DATA_DIR


@pytest.fixture
def candidate():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )


def override(candidate):
    return IntakeCallerSurface(
        canonical_task_id="intake_times_hard_09",
        canonical_value="9:40 PM",
        native_spoken="열 시 이십 분 전",
        romanized_spoken="yeol si isip bun jeon",
        rationale="Interpret minutes before the next hour; clarify the private period.",
        provenance=candidate.provenance,
    )


def with_override(candidate):
    data = candidate.model_dump()
    data["intake_caller_surfaces"] = [override(candidate).model_dump()]
    return LocalizationMap.model_validate(data)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"canonical_value": "9:40 AM"}, "canonical"),
        ({"canonical_task_id": "intake_times_easy_01"}, "hard"),
        ({"canonical_task_id": "intake_dates_hard_01"}, "hard"),
        ({"canonical_task_id": "intake_times_hard_99"}, "selected"),
        ({"native_spoken": "yeol si"}, "script"),
        ({"native_spoken": "   "}, "blank"),
    ],
)
def test_rejects_unbound_or_non_native_surface(candidate, change, expected):
    data = candidate.model_dump()
    surface = override(candidate).model_dump()
    surface.update(change)
    data["intake_caller_surfaces"] = [surface]
    with pytest.raises((ValidationError, ValueError), match=expected):
        LocalizationMap.model_validate(data)


def test_rejects_duplicate_and_cross_domain_surfaces(candidate):
    data = with_override(candidate).model_dump()
    data["intake_caller_surfaces"] *= 2
    with pytest.raises(ValueError, match="unique"):
        LocalizationMap.model_validate(data)
    data["intake_caller_surfaces"] = data["intake_caller_surfaces"][:1]
    data["source"]["domain"] = "retail"
    with pytest.raises(ValueError, match="intake"):
        LocalizationMap.model_validate(data)


def test_rejects_unsupported_language_instead_of_ignoring_override(candidate):
    data = with_override(candidate).model_dump()
    data["settings"]["language"] = "en"
    with pytest.raises(ValueError, match="English overrides are unsupported"):
        LocalizationMap.model_validate(data)


def test_csv_rejects_differing_surfaces_for_shared_primary_entry(candidate):
    candidate = with_override(candidate)
    document = next(
        d for d in candidate.source.documents if d.id == "intake_localization"
    )
    selected = document.content["stimuli"]
    first = next(
        i
        for i, s in enumerate(selected)
        if s["canonical_task_id"] == "intake_times_hard_09"
    )
    second = next(
        i
        for i, s in enumerate(selected)
        if s["canonical_task_id"] == "intake_times_hard_08"
    )
    entry = next(
        e
        for e in candidate.entries
        if any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", first, "written"]
            for o in e.occurrences
        )
    )
    occurrence = next(
        o
        for o in entry.occurrences
        if o.document_id == "intake_localization"
        and o.path == ["stimuli", first, "written"]
    )
    entry.occurrences.append(
        occurrence.model_copy(update={"path": ["stimuli", second, "written"]})
    )
    # The projection must fail even before compiler/source checks when a shared
    # source row links an authored task and another task with the ordinary form.
    with pytest.raises(ValueError, match="differing task caller surfaces"):
        entity_review_rows(candidate)


def test_compact_artifact_round_trip_preserves_override(candidate, tmp_path):
    candidate = with_override(candidate)
    path = tmp_path / "map.json"
    write_localization_map(candidate, path)
    restored = read_localization_map(path)
    assert restored.intake_caller_surfaces == candidate.intake_caller_surfaces
    assert (
        "intake_caller_surfaces"
        not in candidate.model_copy(update={"intake_caller_surfaces": []}).model_dump()
    )


@pytest.mark.parametrize("representation", ["native", "romanized"])
def test_compile_surface_keeps_gold_full_anchor_and_complication_truth(
    candidate, representation
):
    candidate = with_override(candidate)
    documents = {d.id: d for d in compile_localization(candidate, representation)}
    task = next(
        t for t in documents["tasks"].content if t["id"] == "intake_times_hard_09"
    )
    calls = {
        a["func_name"]: a["arguments"]
        for a in task["initial_state"]["initialization_actions"]
    }
    stimulus = IntakeLocalizationData.model_validate(
        documents["intake_localization"].content
    ).for_task(task["id"])
    phrase = getattr(override(candidate), f"{representation}_spoken")
    assert stimulus.caller_spoken == phrase
    assert stimulus.written == "9:40 PM"
    assert phrase in stimulus.omission_instruction
    payload = calls["set_entities"]["entities"][stimulus.field_name]
    assert entity_match_texts(payload) == [
        phrase,
        "오후 9시 40분" if representation == "native" else "ohu 9si 40bun",
    ]
    assert (
        task["evaluation_criteria"]["actions"][0]["arguments"]["fields"][
            stimulus.field_name
        ]
        == "9:40 PM"
    )
    assert all(phrase != decoy for decoy in stimulus.decoys)


def test_csv_uses_initial_phrase_but_preserves_anchor_and_generation(candidate):
    candidate = with_override(candidate)
    rows = entity_review_rows(candidate)
    row = next(
        r
        for r in rows
        if r.original_entity == "9:40 PM" and r.review_type == "Date/time"
    )
    assert row.suggested_entity == "열 시 이십 분 전"
    lineage = json.loads(row.lineage)
    assert "오후 9시 40분" in lineage["caller_time_payload"]
    assert lineage["intake_caller_surfaces"][0][
        "provenance"
    ] == candidate.provenance.model_dump(mode="json")
    assert all(
        r.suggested_entity != row.suggested_entity
        for r in rows
        if r.review_type == "Decoy" and r.original_entity == "9:40 PM"
    )


def test_reviewed_caller_surface_binds_exact_wording_and_preserves_lineage(candidate):
    from tau2.utils.artifacts import ReviewApproval, content_digest

    before = override(candidate)
    after = before.model_copy(
        update={"rationale": before.rationale + " Reviewed wording."}
    )
    corrected = after.model_copy(
        update={
            "review": ReviewApproval(
                content_sha256=content_digest(after), reviewed_by="test reviewer"
            )
        }
    )
    data = candidate.model_dump()
    data["intake_caller_surfaces"] = [corrected.model_dump()]
    result = LocalizationMap.model_validate(data)
    assert result.source == candidate.source
    assert result.entries == candidate.entries
    surface = result.intake_caller_surfaces[0]
    assert surface.provenance == before.provenance
    assert surface.canonical_value == before.canonical_value
    assert surface.review is not None
    for field, value in [("native_spoken", "아홉 시"), ("canonical_value", "9:40 AM")]:
        tampered = surface.model_dump()
        tampered[field] = value
        with pytest.raises(ValueError, match="review does not match"):
            IntakeCallerSurface.model_validate(tampered)
