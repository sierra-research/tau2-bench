# Copyright Sierra
"""Sheet families: manifest atomicity, checksums, header validation, batch ids."""

import json

import pytest

from tau2.annotation.artifacts import (
    ArtifactManifest,
    payload_from_cold_sheet,
    payload_from_rows,
    read_artifact,
    write_sheet_family,
)
from tau2.annotation.models import (
    ColdSheet,
    ColdSheetRow,
    ColdSidecarRow,
    FactorKeyRow,
    JudgePrecisionRow,
    TranslationReviewRow,
)
from test_annotation.conftest import fill_csv, read_family_csv, write_family_csv


def _translation_rows(n=2) -> list[TranslationReviewRow]:
    return [
        TranslationReviewRow(
            row_id=i,
            task_id=f"{i}_tl",
            field="task_instructions",
            english_original=f"English {i}",
            translation=f"Vertaling {i}",
        )
        for i in range(1, n + 1)
    ]


def _write_translation_family(out_dir, name="airline_translation_review"):
    return write_sheet_family(
        out_dir / name,
        kind="translation_review",
        payloads={
            "sheet": payload_from_rows(TranslationReviewRow, _translation_rows())
        },
        provenance={"translator_model": "model-a"},
        language="tl",
        domain="airline",
    )


def _write_cold_family(out_dir, name="hi_cold"):
    key = FactorKeyRow(
        language="hi",
        factor_id="register_formality",
        nuance="Formal address",
        native_does="aap",
        non_native_ref="tum",
    )
    sheet = ColdSheet(
        rows=[
            ColdSheetRow(
                transcript="tx",
                sim_id="s1",
                task_id="t1",
                language="hi",
                labels={"Formal address": None},
            )
        ],
        factors=[key],
    )
    sidecar = [
        ColdSidecarRow(
            sim_id="s1",
            factor_id="register_formality",
            language="hi",
            judge_outcome="fail",
            judge_reasoning="slip",
            judge_quote="tum",
        )
    ]
    return write_sheet_family(
        out_dir / name,
        kind="nativeness_cold",
        payloads={
            "sheet": payload_from_cold_sheet(sheet),
            "judge_sidecar": payload_from_rows(ColdSidecarRow, sidecar, editable=False),
            "factors_key": payload_from_rows(
                FactorKeyRow, sheet.factors, editable=False
            ),
        },
        provenance={},
        language="hi",
    )


def test_family_writes_manifest_and_csvs_atomically(tmp_path):
    manifest_path = _write_translation_family(tmp_path)
    assert manifest_path.name == "airline_translation_review.manifest.json"
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == "translation_review"
    assert manifest.language == "tl" and manifest.domain == "airline"
    assert manifest.provenance["translator_model"] == "model-a"
    assert [f.name for f in manifest.files] == ["airline_translation_review.csv"]
    assert manifest.git_sha and manifest.git_sha != ""
    # No stray .tmp staging files remain.
    assert not list(tmp_path.glob("*.tmp"))
    # Headers come from the model.
    headers, rows = read_family_csv(tmp_path / "airline_translation_review.csv")
    assert headers == TranslationReviewRow.headers()
    assert len(rows) == 2


def test_batch_id_is_content_derived(tmp_path):
    a = _write_translation_family(tmp_path / "a")
    b = _write_translation_family(tmp_path / "b")
    ma = ArtifactManifest.model_validate_json(a.read_text())
    mb = ArtifactManifest.model_validate_json(b.read_text())
    # Identical inputs -> identical batch_id; created_at is the only
    # nondeterministic field.
    assert ma.batch_id == mb.batch_id
    assert ma.model_dump(exclude={"created_at"}) == mb.model_dump(
        exclude={"created_at"}
    )
    # Different content -> different id.
    c = write_sheet_family(
        tmp_path / "c" / "airline_translation_review",
        kind="translation_review",
        payloads={
            "sheet": payload_from_rows(TranslationReviewRow, _translation_rows(3))
        },
        provenance={},
        language="tl",
    )
    mc = ArtifactManifest.model_validate_json(c.read_text())
    assert mc.batch_id != ma.batch_id


def test_ingest_hard_fails_without_manifest(tmp_path):
    csv_path = tmp_path / "orphan.csv"
    write_family_csv(csv_path, TranslationReviewRow.headers(), [])
    with pytest.raises(FileNotFoundError, match="manifest"):
        read_artifact(csv_path)


def test_read_artifact_returns_typed_rows(tmp_path):
    _write_translation_family(tmp_path)
    csv_path = tmp_path / "airline_translation_review.csv"
    fill_csv(csv_path, lambda i, row: row.update({"meaning_ok (Y/N)": "Y"}))
    artifact = read_artifact(csv_path)
    assert artifact.manifest.kind == "translation_review"
    assert artifact.role == "sheet"
    rows = artifact.rows("sheet")
    assert all(isinstance(r, TranslationReviewRow) for r in rows)
    assert rows[0].native_ok is True


def test_formula_cells_are_armored_and_round_trip(tmp_path):
    """LLM/transcript content that would execute as a spreadsheet formula is
    written with a leading apostrophe and comes back verbatim on ingest."""
    rows = [
        TranslationReviewRow(
            row_id=1,
            task_id="1_tl",
            field="task_instructions",
            english_original="=SUM(A1:A9)",
            translation="@cmd |' /C calc'!A0",
            comments="'already apostrophed",
        ),
        TranslationReviewRow(
            row_id=2,
            task_id="2_tl",
            field="task_instructions",
            english_original="-2 degrees",
            translation="+1 more",
        ),
    ]
    write_sheet_family(
        tmp_path / "airline_translation_review",
        kind="translation_review",
        payloads={"sheet": payload_from_rows(TranslationReviewRow, rows)},
        provenance={},
        language="tl",
        domain="airline",
    )
    csv_path = tmp_path / "airline_translation_review.csv"
    text = csv_path.read_text()
    # On disk every formula-leading cell is apostrophe-armored...
    assert "'=SUM(A1:A9)" in text
    assert "'-2 degrees" in text
    assert "'+1 more" in text
    assert "\n=" not in text and ",=" not in text
    # ...and ingest strips exactly one armoring apostrophe back off.
    loaded = read_artifact(csv_path).rows("sheet")
    assert loaded[0].english_original == "=SUM(A1:A9)"
    assert loaded[0].translation == "@cmd |' /C calc'!A0"
    assert loaded[0].comments == "'already apostrophed"  # untouched: not armor
    assert loaded[1].english_original == "-2 degrees"
    assert loaded[1].translation == "+1 more"


def test_missing_non_editable_member_is_loud(tmp_path):
    _write_cold_family(tmp_path)
    (tmp_path / "hi_cold_judge_sidecar.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        read_artifact(tmp_path / "hi_cold.csv")


def test_deleted_factor_column_is_loud(tmp_path):
    _write_cold_family(tmp_path)
    grid = tmp_path / "hi_cold.csv"
    headers, rows = read_family_csv(grid)
    headers.remove("Formal address")
    write_family_csv(grid, headers, [{k: r.get(k, "") for k in headers} for r in rows])
    with pytest.raises(ValueError, match="missing factor columns"):
        read_artifact(grid)


def test_failed_export_leaves_no_tmp_litter(tmp_path):
    """A write that dies mid-staging must clean up its *.tmp files and never
    land a manifest."""
    # A directory squatting on the second member's staging path makes its
    # write_bytes raise after the first member was already staged.
    (tmp_path / "hi_cold_judge_sidecar.csv.tmp").mkdir(parents=True)
    with pytest.raises(OSError):
        _write_cold_family(tmp_path)
    assert [p for p in tmp_path.glob("*.tmp") if p.is_file()] == []
    assert not (tmp_path / "hi_cold.manifest.json").exists()


def test_lone_manifest_does_not_claim_unrelated_csv(tmp_path):
    """A manifest never claims a CSV it does not list by exact name — an
    unrelated CSV dropped next to it must not ingest as the family."""
    _write_translation_family(tmp_path)
    stray = tmp_path / "stray.csv"
    write_family_csv(stray, TranslationReviewRow.headers(), [])
    with pytest.raises(FileNotFoundError, match="manifest"):
        read_artifact(stray)


def test_checksum_enforced_on_non_editable_members(tmp_path):
    _write_cold_family(tmp_path)
    # Tampering with the answer key (judge sidecar) must fail ingest loudly.
    sidecar = tmp_path / "hi_cold_judge_sidecar.csv"
    fill_csv(sidecar, lambda i, row: row.update({"judge_outcome": "pass"}))
    with pytest.raises(ValueError, match="checksum"):
        read_artifact(tmp_path / "hi_cold.csv")


def test_editable_sheet_may_change_but_headers_must_match(tmp_path):
    _write_cold_family(tmp_path)
    grid = tmp_path / "hi_cold.csv"
    # Filling annotator cells is fine.
    fill_csv(grid, lambda i, row: row.update({"Formal address": "yes"}))
    artifact = read_artifact(grid)
    sheet = artifact.cold_sheet()
    assert list(sheet.melt())[0][3] is not None
    # But an unknown factor column is a loud error.
    headers, rows = read_family_csv(grid)
    headers.append("Invented column")
    write_family_csv(grid, headers, rows)
    with pytest.raises(ValueError, match="factor columns"):
        read_artifact(grid)


def test_fixed_shape_header_mismatch_is_loud(tmp_path):
    _write_translation_family(tmp_path)
    csv_path = tmp_path / "airline_translation_review.csv"
    headers, rows = read_family_csv(csv_path)
    headers.remove("task_id")
    write_family_csv(
        csv_path, headers, [{k: r.get(k, "") for k in headers} for r in rows]
    )
    with pytest.raises(ValueError, match="header set"):
        read_artifact(csv_path)


def test_renamed_filled_csv_with_explicit_manifest(tmp_path):
    manifest_path = _write_translation_family(tmp_path)
    src = tmp_path / "airline_translation_review.csv"
    moved = tmp_path / "elsewhere" / "FILLED.csv"
    moved.parent.mkdir()
    headers, rows = read_family_csv(src)
    for row in rows:
        row["meaning_ok (Y/N)"] = "N"
        row["natural_ok (Y/N)"] = "N"
    write_family_csv(moved, headers, rows)
    artifact = read_artifact(moved, manifest_path)
    assert artifact.role == "sheet"
    assert artifact.rows("sheet")[0].native_ok is False


def test_derived_precision_role_in_cold_family(tmp_path):
    # A `both`-mode family carries an extra editable precision sheet whose
    # ingest role is resolved by header set.
    key = FactorKeyRow(language="hi", factor_id="f", nuance="N")
    sheet = ColdSheet(
        rows=[
            ColdSheetRow(
                transcript="tx", sim_id="s1", language="hi", labels={"N": None}
            )
        ],
        factors=[key],
    )
    write_sheet_family(
        tmp_path / "hi_cold",
        kind="nativeness_cold",
        payloads={
            "sheet": payload_from_cold_sheet(sheet),
            "factors_key": payload_from_rows(FactorKeyRow, [key], editable=False),
            "precision": payload_from_rows(
                JudgePrecisionRow,
                [JudgePrecisionRow(sim_id="s1", factor_id="f", language="hi")],
            ),
        },
        provenance={},
        language="hi",
    )
    artifact = read_artifact(tmp_path / "hi_cold_precision.csv")
    assert artifact.role == "precision"
    assert isinstance(artifact.rows("precision")[0], JudgePrecisionRow)


def test_unknown_role_rejected(tmp_path):
    with pytest.raises(ValueError, match="not part of"):
        write_sheet_family(
            tmp_path / "x",
            kind="translation_review",
            payloads={
                "mystery": payload_from_rows(TranslationReviewRow, _translation_rows())
            },
            provenance={},
        )


def test_manifest_json_shape_is_stable(tmp_path):
    manifest_path = _write_translation_family(tmp_path)
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == 1
    assert set(payload["files"][0]) == {"name", "sha256", "role", "editable"}
