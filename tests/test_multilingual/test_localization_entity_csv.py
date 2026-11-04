"""Entity CSVs classify literal pairs without scripts or opaque tool payloads."""

import csv

import pytest

from tau2.multilingual.factory.localization_entity_csv import (
    EntityCsvManifest,
    EntityCsvSource,
    entity_review_rows,
    export_entity_csv,
)
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizedForms,
    ProseEntityBinding,
    SourceOccurrence,
)
from tests.test_multilingual.test_localization_review import sample_map


def add(mapping, identifier, source, target, category="other", path=None):
    mapping.entries.append(
        LocalizationEntry(
            id=identifier,
            source=source,
            category=category,
            occurrences=[SourceOccurrence(document_id="db", path=path or [identifier])],
            context="Synthetic source entry for review tests.",
            forms=LocalizedForms(
                native=target, romanized=target, english="Fixture entity."
            ),
            operation="adapt",
            notes="Pending fixture review.",
        )
    )


def test_projection_excludes_scripts_and_instructions_but_retains_used_entities():
    mapping = sample_map("retail")
    add(
        mapping, "script", "Full customer script" * 500, "Full customer script", "prose"
    )
    add(mapping, "arguments", '{"user_id":"123"}', '{"user_id":"123"}', "prose")
    add(
        mapping,
        "style",
        ["Long spelling instructions"],
        ["Localized instructions"],
        path=["stimuli", 0, "spelling_styles"],
    )
    add(
        mapping,
        "omission",
        "Omit middle name",
        "Omit middle name",
        path=["stimuli", 0, "omission_instruction"],
    )
    add(mapping, "decoys", ["Name A", "Name B"], ["नाम अ", "नाम ब"])
    add(
        mapping,
        "digits",
        {"0": "zero", "1": "one"},
        {"0": "शून्य", "1": "एक"},
        path=["digit_words"],
    )
    rows = entity_review_rows(mapping)
    assert [row.original_entity for row in rows] == [
        "Name A",
        "Name B",
        "zero",
        "one",
        "Source name",
    ]
    assert all(row.decision == row.reason == row.proposed_entity == "" for row in rows)
    assert [row.review_type for row in rows] == [
        "Decoy",
        "Decoy",
        "Spoken digit",
        "Spoken digit",
        "Entity",
    ]
    assert len({row.key for row in rows}) == len(rows)
    assert '"value_path": [0]' in rows[0].lineage


def test_pronunciation_pairs_exclude_machine_rules_and_preserve_source_alignment():
    mapping = sample_map("retail")
    before = {
        "token": "Name",
        "reference": "NAYM",
        "distorted": "nime",
        "boundary": "word",
        "reason": "vowel_swap",
    }
    after = {
        "token": "नाम",
        "reference": "नाम",
        "distorted": "नीम",
        "boundary": "word",
        "reason": "vowel_swap",
    }
    add(mapping, "pronunciation", [before], [after])
    mapping.entries[-1].forms.romanized = [
        {
            "token": "Naam",
            "reference": "naam",
            "distorted": "neem",
            "boundary": "word",
            "reason": "vowel_swap",
        }
    ]
    rows = [
        row
        for row in entity_review_rows(mapping)
        if row.entry_id.startswith("pronunciation")
    ]
    assert [(row.original_entity, row.suggested_entity) for row in rows] == [
        ("Name", "नाम"),
        ("nime", "नीम"),
    ]
    assert [row.entry_id for row in rows] == [
        "pronunciation/0/token",
        "pronunciation/0/distorted",
    ]
    assert [row.romanized for row in rows] == ["Naam", "neem"]
    assert [row.review_type for row in rows] == [
        "Pronunciation token",
        "Mispronunciation",
    ]
    mapping.entries[-1].forms.native = []
    with pytest.raises(ValueError, match="alignment"):
        entity_review_rows(mapping)


def test_incidental_entities_are_reviewed_without_the_containing_script():
    mapping = sample_map("banking_knowledge")
    mapping.prose_bindings = [
        ProseEntityBinding(
            entry_id=mapping.entries[0].id,
            source="Incidental name",
            forms=mapping.entries[0].forms,
            kind="reuse",
            source_entry_id=mapping.entries[0].id,
            reason="Explicit identity reference.",
        ),
        ProseEntityBinding(
            entry_id=mapping.entries[0].id,
            source="PROTOCOL",
            forms=LocalizedForms(
                native="PROTOCOL", romanized="PROTOCOL", english="Protected token."
            ),
            kind="protected",
            reason="Keep protocol.",
        ),
    ]
    rows = entity_review_rows(mapping)
    assert len(rows) == 2
    assert rows[1].original_entity == "Incidental name"
    assert rows[1].origin == "Reused"
    assert rows[1].review_type == "Entity reference"


def test_export_is_literal_seven_columns_and_does_not_overwrite_review_edits(tmp_path):
    mapping = sample_map("retail", value='=Literal, "quoted"\nname')
    mapping.settings.language = "hi"
    mapping.entries[0].forms.native = "नाम, पता\nदूसरी पंक्ति"
    sources = [EntityCsvSource(mapping=mapping, artifact_sha256="a" * 64)]
    manifests = export_entity_csv(sources, tmp_path)
    path = tmp_path / "localization-hi.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [
        "Type",
        "Difficulty",
        "Original entity",
        "Suggested localized entity",
        "Approved / Disapproved",
        "Proposed entity",
        "Reason",
    ]
    assert rows[1] == [
        "Entity",
        "Not applicable",
        mapping.entries[0].source,
        mapping.entries[0].forms.native,
        "",
        "",
        "",
    ]
    assert all(len(row) == 7 for row in rows)
    sidecar = EntityCsvManifest.model_validate_json(
        path.with_suffix(".manifest.json").read_text()
    )
    assert sidecar == manifests[0]
    assert sidecar.scope == "entity_pairs_only"
    before = path.read_bytes()
    export_entity_csv(sources, tmp_path)
    assert path.read_bytes() == before
    rows[1][4] = "Approved"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(FileExistsError, match="review edits"):
        export_entity_csv(sources, tmp_path)


def test_unknown_structured_payload_is_not_silently_dumped_into_a_cell():
    mapping = sample_map("retail")
    add(
        mapping,
        "unknown",
        {"instructions": "Long script"},
        {"instructions": "Long script"},
    )
    with pytest.raises(ValueError, match="opaque JSON"):
        entity_review_rows(mapping)


def test_cli_registers_entity_export():
    import argparse

    from tau2.multilingual.factory.cli import add_factory_args

    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    args = parser.parse_args(
        ["export-localization-entities-csv", "--map", "hi.json", "--output", "review"]
    )
    assert args.func.__name__ == "run_entity_csv_export"


def test_identical_banking_references_merge_but_keep_every_scope():
    mapping = sample_map("banking_knowledge")
    base = mapping.entries[0]
    mapping.prose_bindings = [
        ProseEntityBinding(
            entry_id=identifier,
            source="Same source",
            forms=base.forms,
            kind="reuse",
            source_entry_id=base.id,
            reason="Scoped test.",
        )
        for identifier in ("task1", "task2")
    ]
    rows = entity_review_rows(mapping)
    refs = [r for r in rows if r.review_type == "Entity reference"]
    assert len(refs) == 1
    assert len(refs[0].reference_occurrences) == 2
    assert len({r.entry_id for r in refs[0].reference_occurrences}) == 2
    mapping.prose_bindings[1].forms = base.forms.model_copy(
        update={"native": "Different"}
    )
    assert (
        len(
            [
                r
                for r in entity_review_rows(mapping)
                if r.review_type == "Entity reference"
            ]
        )
        == 2
    )


def test_intake_difficulty_comes_from_measured_selection_not_name_guessing():
    from tau2.multilingual.localization import SourceDocument

    mapping = sample_map("intake")
    mapping.source.documents.append(
        SourceDocument(
            id="intake_localization",
            kind="dependency",
            source_path="test",
            sha256="a" * 64,
            content={"stimuli": [{"tier": "easy"}, {"tier": "hard"}]},
        )
    )
    entry = mapping.entries[0]
    entry.occurrences = [
        SourceOccurrence(
            document_id="intake_localization", path=["stimuli", 1, "written"]
        )
    ]
    assert entity_review_rows(mapping)[0].difficulty == "Hard"
    entry.occurrences.append(
        SourceOccurrence(
            document_id="intake_localization", path=["stimuli", 0, "written"]
        )
    )
    assert entity_review_rows(mapping)[0].difficulty == "Easy + Hard"
    entry.occurrences = [SourceOccurrence(document_id="tasks", path=[0, "callee"])]
    assert entity_review_rows(mapping)[0].difficulty == "Context only"


def test_relative_date_surface_and_shared_digits_retain_actual_usage_tiers():
    from tau2.multilingual.factory.localization_entity_csv import _difficulty
    from tau2.multilingual.localization import SourceDocument

    mapping = sample_map("intake")
    for identifier, content in {
        "tasks": [{"id": "intake_dates_hard_01"}],
        "intake_localization": {
            "stimuli": [
                {
                    "canonical_task_id": "intake_dates_hard_01",
                    "field_name": "drop_off_date",
                    "tier": "hard",
                },
                {
                    "canonical_task_id": "intake_dates_easy_01",
                    "field_name": "drop_off_date",
                    "tier": "easy",
                },
            ]
        },
    }.items():
        mapping.source.documents.append(
            SourceDocument(
                id=identifier,
                kind="dependency",
                source_path="test",
                sha256="a" * 64,
                content=content,
            )
        )
    entry = mapping.entries[0]
    entry.occurrences = [
        SourceOccurrence(
            document_id="tasks",
            path=[
                0,
                "initial_state",
                "initialization_actions",
                3,
                "arguments",
                "entities",
                "drop_off_date",
            ],
        )
    ]
    assert _difficulty(mapping, entry) == "Hard"
    entry.occurrences[0].path[-1] = "callee_full_name"
    assert _difficulty(mapping, entry) == "Context only"
    entry.occurrences = [
        SourceOccurrence(document_id="intake_localization", path=["digit_words"])
    ]
    assert _difficulty(mapping, entry) == "Easy + Hard"
