"""Canonical temporal dependencies are checked before generated batches are cached."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.intake_authoring import validate_intake_forms
from tau2.multilingual.localization import LocalizedForms, SourceEntry, SourceOccurrence


def temporal_entry(family="times"):
    return SourceEntry(
        id=f"test-{family}-decoys",
        category="other",
        source=["9:40 AM", "9:04 PM"]
        if family == "times"
        else ["2025-06-13", "2025-06-14"],
        occurrences=[
            SourceOccurrence(
                document_id="intake_localization", path=["stimuli", 90, "decoys"]
            )
        ],
        context=f"intake stimulus:intake_{family}_hard_01:decoys; entity=intake_{family}_hard_01; dependency=decoys; selected={{}}; Source test.",
    )


@pytest.mark.parametrize("representation", ["native", "romanized"])
@pytest.mark.parametrize(
    "wrong", ["오전 9시 40분", "9:40 pm", "21:40", "13:40 PM", "9:60 PM", "09:40 AM"]
)
def test_time_decoys_reject_localized_or_noncanonical_clock_notation(
    representation, wrong
):
    entry = temporal_entry()
    forms = LocalizedForms(native=entry.source, romanized=entry.source, english="Test")
    setattr(forms, representation, [wrong, "9:04 PM"])
    with pytest.raises(ValueError, match="H:MM AM/PM"):
        validate_intake_forms(entry, forms, language="ko")


@pytest.mark.parametrize(
    "wrong",
    ["13-06-2025", "2025年6月13日", "2025년 6월 13일", "20250613", "2025-02-30"],
)
def test_date_decoys_reject_local_notation_and_impossible_calendar_dates(wrong):
    entry = temporal_entry("dates")
    forms = LocalizedForms(
        native=[wrong, "2025-06-14"], romanized=entry.source, english="Test"
    )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_intake_forms(entry, forms, language="hi")


@pytest.mark.parametrize("family", ["dates", "times"])
def test_canonical_temporal_decoys_are_valid_in_both_representations(family):
    entry = temporal_entry(family)
    validate_intake_forms(
        entry,
        LocalizedForms(native=entry.source, romanized=entry.source, english="Test"),
        language="ko",
    )


def test_temporal_identity_is_not_inferred_from_a_property_value():
    entry = temporal_entry()
    entry.context = "intake stimulus:intake_properties_hard_01:decoys; entity=intake_properties_hard_01; dependency=decoys; selected={};"
    entry.source = ["9:40 AM", "9:04 PM"]
    validate_intake_forms(
        entry,
        LocalizedForms(
            native=["오전 숙소", "저녁 숙소"],
            romanized=["Morning Stay", "Evening Stay"],
            english="Local properties",
        ),
        language="ko",
    )


def test_decoy_matching_correct_time_is_rejected_before_compilation():
    from tau2.multilingual.localization import LocalizationEntry

    entry = temporal_entry()
    written = LocalizationEntry(
        id="test-written",
        category="other",
        source="9:40 PM",
        occurrences=[
            SourceOccurrence(
                document_id="intake_localization", path=["stimuli", 90, "written"]
            )
        ],
        context="intake field:appointment_time;",
        forms=LocalizedForms(native="9:40 PM", romanized="9:40 PM", english="Time"),
        operation="preserve",
        notes="Test",
    )
    with pytest.raises(ValueError, match="collapses"):
        validate_intake_forms(
            entry,
            LocalizedForms(
                native=["9:40 PM", "9:04 PM"], romanized=entry.source, english="Test"
            ),
            [written],
            language="ko",
        )


def test_invalid_time_generation_is_repaired_before_checkpoint(monkeypatch, tmp_path):
    from tau2.multilingual.factory import database_localization as database
    from tau2.multilingual.factory.intake_authoring import prompt_version
    from tests.test_domains.test_intake.test_localization_intake import identity_map

    proposal = identity_map()
    entry = next(
        entry
        for entry in proposal.source.entries
        if entry.context.startswith("intake stimulus:intake_times_hard_01:decoys;")
    )
    request = database.RegionalDraftRequest(
        source=proposal.source,
        settings=proposal.settings,
        revision="test-repair",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    batch = database.BatchInput(
        request_sha256="0" * 64,
        entries=[entry],
        completed=[],
        prompt_version=prompt_version(
            "intake", database.DATABASE_LOCALIZATION_REVISION_VERSION
        ),
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        values = (
            ["오전 9시 40분" for _ in entry.source] if len(calls) == 1 else entry.source
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": entry.id,
                            "forms": {
                                "native": values,
                                "romanized": values,
                                "english": "Test",
                            },
                            "operation": "adapt",
                            "notes": "Test",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(database, "generate", generate)
    result = database._generate_batch(request, batch, tmp_path)
    assert len(calls) == 2
    assert "H:MM AM/PM" in calls[1]["messages"][-1].content
    assert result[0].forms.native == entry.source
    checkpoint = database.BatchArtifact.model_validate_json(
        next(tmp_path.glob("*.json")).read_text()
    )
    assert len(checkpoint.rejected_candidates) == 1
    assert checkpoint.response.entries[0].forms.native == entry.source


def medication_entry(source):
    entry = temporal_entry()
    entry.id = "test-medication-decoys"
    entry.source = source
    entry.context = "intake stimulus:intake_medications_hard_01:decoys; entity=intake_medications_hard_01; dependency=decoys; selected={};"
    return entry


@pytest.mark.parametrize(
    "source,changed",
    [
        ("Olmesartan 20 mg tablet", "Telmisartán 80 mg comprimido"),
        ("Lamotrigine 100 mg tablet", "Sitagliptina 50 mg comprimido"),
        ("Clonidine 0.1 mg tablet", "Clonidina 0,1 mcg comprimido"),
    ],
)
@pytest.mark.parametrize("representation", ["native", "romanized"])
def test_medication_decoys_preserve_each_source_strength_and_unit(
    source, changed, representation
):
    entry = medication_entry([source])
    forms = LocalizedForms(native=[source], romanized=[source], english="Test")
    setattr(forms, representation, [changed])
    with pytest.raises(ValueError, match="medication decoy.*strength or units"):
        validate_intake_forms(entry, forms, language="es")


def test_medication_decoy_decimal_comma_is_not_dose_drift():
    entry = medication_entry(["Clonidine 0.1 mg tablet"])
    validate_intake_forms(
        entry,
        LocalizedForms(
            native=["Clonidina 0,1 mg comprimido"],
            romanized=["Clonidina 0,1 mg comprimido"],
            english="Test",
        ),
        language="pt",
    )


def test_medication_strengths_are_bound_per_decoy_not_pooled_or_reordered():
    entry = medication_entry(["Medicine A 20 mg", "Medicine B 80 mg"])
    with pytest.raises(ValueError, match="medication decoy.*strength or units"):
        validate_intake_forms(
            entry,
            LocalizedForms(
                native=["Medicine A 80 mg", "Medicine B 20 mg"],
                romanized=entry.source,
                english="Test",
            ),
            language="es",
        )


def test_other_families_are_not_classified_as_medications_from_their_text():
    entry = medication_entry(["20 mg"])
    entry.context = "intake stimulus:intake_coined_hard_01:decoys; entity=intake_coined_hard_01; dependency=decoys; selected={};"
    validate_intake_forms(
        entry,
        LocalizedForms(
            native=["80 mg"], romanized=["80 mg"], english="Synthetic brand"
        ),
        language="es",
    )
