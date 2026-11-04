"""Review decisions follow exact map entries across tabs, sorting and additions."""

import pytest

openpyxl = pytest.importorskip("openpyxl")

from tau2.multilingual.factory.localization_artifact import (  # noqa: E402
    AuthoredLocalization,
    RegionalMapArtifact,
    SourceDescriptor,
    SourceManifest,
)
from tau2.multilingual.factory.localization_spreadsheet import (  # noqa: E402
    export_localization_spreadsheet,
    ingest_localization_spreadsheet,
)
from tau2.utils.artifacts import content_digest  # noqa: E402
from tests.test_multilingual.test_localization_review import sample_map  # noqa: E402


def artifact(domain="retail", language="ko"):
    source = sample_map(domain, value="=Source name")
    source.settings.language = language
    return RegionalMapArtifact(
        revision=source.revision,
        settings=source.settings,
        source=SourceManifest(
            domain=domain,
            documents=[
                SourceDescriptor.model_validate(doc.model_dump(exclude={"content"}))
                for doc in source.source.documents
            ],
        ),
        entries=[
            AuthoredLocalization.model_validate(
                entry.model_dump(include=set(AuthoredLocalization.model_fields))
            )
            for entry in source.entries[:1]
        ],
        provenance=source.provenance,
    )


def edit(path, cell, value, sheet="ko"):
    workbook = openpyxl.load_workbook(path)
    workbook[sheet][cell] = value
    workbook.save(path)
    workbook.close()


def test_export_has_language_tabs_editable_columns_and_literal_entities(tmp_path):
    maps = [artifact(), artifact("telecom", "hi")]
    path = tmp_path / "entities.xlsx"
    export_localization_spreadsheet(maps, path)
    workbook = openpyxl.load_workbook(path)
    assert workbook.sheetnames == ["hi", "ko"]
    sheet = workbook["ko"]
    assert [sheet.cell(1, i).value for i in range(1, 6)] == [
        "Original entity",
        "Suggested localized entity",
        "Approved / Disapproved",
        "Proposed entity",
        "Reason",
    ]
    assert sheet["A2"].value == "=Source name"
    assert sheet["A2"].data_type == "s"
    assert sheet["A2"].protection.locked
    assert not sheet["C2"].protection.locked
    assert not sheet["D2"].protection.locked
    assert sheet.freeze_panes == "C2"
    assert sheet.data_validations.count == 1
    workbook.close()
    first = path.read_bytes()
    export_localization_spreadsheet(maps, path)
    assert path.read_bytes() == first
    result = ingest_localization_spreadsheet(maps, path, "Native reviewer")
    assert not result.approvals
    assert all(row.decision == "" for row in result.rows)


def test_decisions_survive_reordering_and_adding_domain(tmp_path):
    original = [artifact(), artifact("airline")]
    path = tmp_path / "first.xlsx"
    export_localization_spreadsheet(original, path)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["ko"]
    for row in range(2, 4):
        sheet.cell(row, 3, "Approved")
    values = [[cell.value for cell in row] for row in sheet.iter_rows(min_row=2)]
    for row_number, row in enumerate(reversed(values), start=2):
        for col, value in enumerate(row, start=1):
            cell = sheet.cell(row_number, col, value)
            if isinstance(value, str):
                cell.data_type = "s"
    workbook.save(path)
    workbook.close()
    result = ingest_localization_spreadsheet(original, path, "Reviewer A")
    assert {approval.content_sha256 for approval in result.approvals} == {
        content_digest(mapping) for mapping in original
    }
    revised = tmp_path / "with-telecom.xlsx"
    export_localization_spreadsheet(
        [*original, artifact("telecom")], revised, previous=path
    )
    result = ingest_localization_spreadsheet(
        [*original, artifact("telecom")], revised, "Reviewer A"
    )
    assert len(result.approvals) == 2
    assert next(row for row in result.rows if row.domain == "telecom").decision == ""


@pytest.mark.parametrize(
    ("cell", "value", "message"),
    [
        ("A2", "tampered original", "immutable"),
        ("C2", "Yes", "decision"),
        ("C2", "Disapproved", "reason"),
        ("D2", "Replacement", "reason"),
        ("C2", '=IF(TRUE,"Approved","")', "formula"),
    ],
)
def test_ingest_rejects_tampering_or_incomplete_feedback(
    tmp_path, cell, value, message
):
    maps = [artifact()]
    path = tmp_path / "entities.xlsx"
    export_localization_spreadsheet(maps, path)
    edit(path, cell, value)
    with pytest.raises(ValueError, match=message):
        ingest_localization_spreadsheet(maps, path, "Reviewer A")


def test_replacement_is_feedback_not_approval_and_stale_maps_rejected(tmp_path):
    maps = [artifact()]
    path = tmp_path / "entities.xlsx"
    export_localization_spreadsheet(maps, path)
    edit(path, "C2", "Disapproved")
    edit(path, "D2", "제안한 이름")
    edit(path, "E2", "Use the locally appropriate spelling.")
    result = ingest_localization_spreadsheet(maps, path, "Reviewer A")
    assert result.rows[0].proposed_entity == "제안한 이름"
    assert not result.approvals
    edit(path, "C2", "Approved")
    with pytest.raises(ValueError, match="replacement"):
        ingest_localization_spreadsheet(maps, path, "Reviewer A")
    maps[0].revision = "changed"
    with pytest.raises(ValueError, match="stale"):
        ingest_localization_spreadsheet(maps, path, "Reviewer A")


def test_refresh_resets_changed_candidate_and_does_not_drop_completed_maps(tmp_path):
    maps = [artifact(), artifact("airline")]
    path = tmp_path / "first.xlsx"
    export_localization_spreadsheet(maps, path)
    edit(path, "C2", "Approved")
    edit(path, "C3", "Approved")
    changed = [mapping.model_copy(deep=True) for mapping in maps]
    changed[0].entries[0].forms.native = "새 이름"
    output = tmp_path / "updated.xlsx"
    export_localization_spreadsheet(changed, output, previous=path)
    result = ingest_localization_spreadsheet(changed, output, "Reviewer A")
    assert len(result.approvals) == 1
    assert result.approvals[0].domain == "airline"
    with pytest.raises(ValueError, match="remove"):
        export_localization_spreadsheet(
            changed[:1], tmp_path / "lost.xlsx", previous=path
        )


def test_ingest_rejects_missing_and_duplicate_rows(tmp_path):
    maps = [artifact()]
    path = tmp_path / "entities.xlsx"
    export_localization_spreadsheet(maps, path)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["ko"]
    sheet.append([cell.value for cell in sheet[2]])
    sheet["A3"].data_type = "s"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="duplicate"):
        ingest_localization_spreadsheet(maps, path, "Reviewer A")
    workbook = openpyxl.load_workbook(path)
    workbook["ko"].delete_rows(2, 2)
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="missing"):
        ingest_localization_spreadsheet(maps, path, "Reviewer A")


def test_export_rejects_excel_silent_text_truncation(tmp_path):
    mapping = artifact()
    mapping.entries[0].notes = "x" * 32768
    path = tmp_path / "entities.xlsx"
    with pytest.raises(ValueError, match="cell limit"):
        export_localization_spreadsheet([mapping], path)
    assert not path.exists()
    assert not path.with_suffix(".manifest.json").exists()


def test_prose_references_require_individual_approval_and_reject_stale_edits(tmp_path):
    from tau2.multilingual.localization import ProseEntityBinding

    mapping = artifact()
    mapping.prose_bindings = [
        ProseEntityBinding(
            entry_id=mapping.entries[0].id,
            source="Source name",
            forms=mapping.entries[0].forms.model_copy(deep=True),
            kind="generated",
            reason="Synthetic fixture for an incidental reference.",
            provenance=mapping.provenance,
        )
    ]
    path = tmp_path / "entities.xlsx"
    manifest = export_localization_spreadsheet([mapping], path)
    assert len(manifest.rows) == 2
    reference = manifest.rows[1]
    assert reference.category == "prose_reference"
    assert reference.entry_id.startswith("prose-reference:")
    assert reference.original_entity == "Source name"
    assert reference.decision == ""
    edit(path, "C2", "Approved")
    assert not ingest_localization_spreadsheet([mapping], path, "Reviewer").approvals
    edit(path, "C3", "Approved")
    result = ingest_localization_spreadsheet([mapping], path, "Reviewer")
    assert len(result.approvals) == 1
    mapping.prose_bindings[0].forms.native = "Changed reference"
    with pytest.raises(ValueError, match="stale"):
        ingest_localization_spreadsheet([mapping], path, "Reviewer")
    refreshed = tmp_path / "refreshed.xlsx"
    refreshed_manifest = export_localization_spreadsheet(
        [mapping], refreshed, previous=path
    )
    assert refreshed_manifest.rows[1].key == reference.key
    result = ingest_localization_spreadsheet([mapping], refreshed, "Reviewer")
    assert result.rows[0].decision == "Approved"
    assert result.rows[1].decision == ""
    assert not result.approvals


def test_factory_commands_export_and_ingest_review_evidence(tmp_path, monkeypatch):
    import argparse

    from tau2.multilingual.factory import localization_spreadsheet as module
    from tau2.multilingual.factory.cli import add_factory_args
    from tau2.utils.artifacts import ReviewApproval

    maps = [artifact()]
    monkeypatch.setattr(module, "_source_checked_maps", lambda paths: maps)
    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    path = tmp_path / "entities.xlsx"
    args = parser.parse_args(
        [
            "export-localization-spreadsheet",
            "--map",
            "source.json",
            "--output",
            str(path),
        ]
    )
    args.func(args)
    edit(path, "C2", "Approved")
    output = tmp_path / "decisions.json"
    approvals = tmp_path / "approvals"
    args = parser.parse_args(
        [
            "ingest-localization-spreadsheet",
            "--map",
            "source.json",
            "--workbook",
            str(path),
            "--reviewed-by",
            "Reviewer A",
            "--output",
            str(output),
            "--approval-dir",
            str(approvals),
        ]
    )
    args.func(args)
    result = module.SpreadsheetReview.model_validate_json(output.read_text())
    assert len(result.approvals) == 1
    approval = ReviewApproval.model_validate_json(
        (approvals / "ko/retail.approval.json").read_text()
    )
    approval.require(maps[0])
