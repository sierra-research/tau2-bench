# Copyright Sierra
"""Nativeness calibration sheet builders over judges.export records (no LLM —
fixture sims carry complete stored verdicts, so reuse-existing always holds)."""

from tau2.annotation.artifacts import ArtifactManifest, read_artifact
from tau2.annotation.metrics import analyze_cold, analyze_precision
from tau2.annotation.models import (
    COLD_TRANSCRIPT_LABEL,
    JudgePrecisionRow,
)
from tau2.annotation.nativeness_sheets import (
    build_cold_sheet,
    build_precision_rows,
    derive_precision_rows,
    export_nativeness_sheets,
)
from tau2.data_model.simulation import JudgeOutcome
from tau2.judges.export import judge_factor_ids
from test_annotation.conftest import (
    FAIL_FACTOR,
    fill_csv,
    hi_sim,
    read_family_csv,
)


def test_precision_rows_only_judge_fails():
    sim = hi_sim("s1", "t1")  # one LLM-factor FAIL, rest PASS
    rows = build_precision_rows([sim])
    assert len(rows) == 1
    row = rows[0]
    assert row.factor_id == FAIL_FACTOR
    assert row.language == "hi" and row.sim_id == "s1" and row.task_id == "t1"
    assert "singular agreement" in row.why_flagged  # judge reasoning surfaced
    assert row.quote == "आप तैयार है"  # exact offending span
    assert row.transcript  # transcript included
    # The rubric's readable nuance (not the slug) heads the row.
    assert row.what_checked and row.what_checked != FAIL_FACTOR
    assert row.verdict is None  # blank for the annotator


def test_precision_rows_empty_when_no_violations():
    assert build_precision_rows([hi_sim("s1", "t1", fail_factor=None)]) == []


def _rubric(factor_id: str, nuance: str):
    from tau2.judges.export import NativenessAnnotationRubric

    return NativenessAnnotationRubric(
        language="hi",
        factor_id=factor_id,
        category="register",
        severity=2,
        annotator_label=f"Violates {factor_id}",
        description="What this axis measures.",
        nuance=nuance,
        native_does="x",
        ai_likely_does="y",
    )


def test_cold_headers_duplicate_nuance_is_loud():
    import pytest

    from tau2.annotation.nativeness_sheets import validate_cold_headers

    with pytest.raises(ValueError, match="collision.*Formal address"):
        validate_cold_headers(
            "hi",
            [
                _rubric("factor_a", "Formal address"),
                _rubric("factor_b", "Formal address"),
            ],
        )


def test_cold_headers_reserved_meta_name_is_loud():
    import pytest

    from tau2.annotation.models import COLD_TRANSCRIPT_LABEL as META
    from tau2.annotation.nativeness_sheets import validate_cold_headers

    with pytest.raises(ValueError, match="reserved"):
        validate_cold_headers("hi", [_rubric("factor_a", META)])


def test_cold_headers_distinct_nuances_pass():
    from tau2.annotation.nativeness_sheets import validate_cold_headers

    validate_cold_headers(
        "hi", [_rubric("factor_a", "Formal address"), _rubric("factor_b", "Politeness")]
    )


def test_precision_transcript_truncation():
    rows = build_precision_rows([hi_sim("s1", "t1")], max_trans_chars=5)
    assert len(rows[0].transcript) == 5


def test_cold_sheet_wide_with_sidecar_and_key():
    sim = hi_sim("s1", "t1", fail_factor=None)  # all PASS
    sheet, sidecar = build_cold_sheet([sim])
    assert len(sheet.rows) == 1  # one row per CALL, not per factor
    row = sheet.rows[0]
    expected_factors = judge_factor_ids("hi")
    assert len(row.labels) == len(expected_factors)
    assert all(label is None for label in row.labels.values())  # blank cells
    # Factor columns are headed by readable nuance titles, not slugs.
    assert FAIL_FACTOR not in row.labels
    reg_key = next(k for k in sheet.factors if k.factor_id == FAIL_FACTOR)
    assert reg_key.nuance in row.labels
    assert reg_key.native_does and reg_key.non_native_ref
    # No judge verdict leaks into the visible grid.
    cells = sheet.to_cells()[0]
    assert "judge" not in " ".join(cells).lower()
    assert COLD_TRANSCRIPT_LABEL in cells
    assert {"sim_id", "task_id", "language"} <= set(cells)
    # The sidecar (long, hidden) carries the verdicts per factor.
    assert {s.factor_id for s in sidecar} == expected_factors
    reg = next(s for s in sidecar if s.factor_id == FAIL_FACTOR)
    assert reg.judge_outcome is JudgeOutcome.PASS


def test_cold_sheet_missing_check_is_blank_outcome():
    # Factor exists for the language but no check stored on the sim -> the
    # sidecar records no verdict (None), never a fake outcome.
    sim = hi_sim("s1", "t1")
    sim.nativeness_info.factor_checks = []
    sheet, sidecar = build_cold_sheet([sim])
    assert sidecar and all(s.judge_outcome is None for s in sidecar)
    # ...which analyzers bucket as judge_unscored, keeping gaps visible.
    filled = sheet.model_copy(deep=True)
    header = filled.factors[0].nuance
    filled.rows[0].labels[header] = "yes"
    metrics = analyze_cold(filled, sidecar)
    assert metrics[("hi", filled.factors[0].factor_id)].counts.judge_unscored == 1


def test_derive_precision_from_cold():
    sim_fail = hi_sim("s1", "t1")  # honorific-agreement FAIL
    sim_pass = hi_sim("s2", "t2", fail_factor=None)
    sheet, sidecar = build_cold_sheet([sim_fail, sim_pass])
    rows = derive_precision_rows(sheet, sidecar)
    assert len(rows) == 1 and rows[0].sim_id == "s1"  # only the FAIL
    assert rows[0].factor_id == FAIL_FACTOR
    assert rows[0].why_flagged == "used singular agreement with आप"
    assert rows[0].quote == "आप तैयार है"
    assert rows[0].transcript == sheet.rows[0].transcript  # pulled by sim
    assert rows[0].task_id == "t1"
    # The rubric's nuance heads the derived row too.
    reg_key = next(k for k in sheet.factors if k.factor_id == FAIL_FACTOR)
    assert rows[0].what_checked == reg_key.nuance


def test_export_both_family_end_to_end(hi_results_dir, tmp_path):
    out_stem = tmp_path / "hi_calibration"
    manifest_path = export_nativeness_sheets(
        [hi_results_dir], "both", out_stem, reuse_existing=True
    )
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == "nativeness_cold"
    assert manifest.language == "hi"
    assert manifest.provenance["judge_models"] == ["fake-judge"]
    assert manifest.provenance["reuse_existing"] is True
    roles = {f.role: f for f in manifest.files}
    assert set(roles) == {"sheet", "judge_sidecar", "factors_key", "precision"}
    assert roles["judge_sidecar"].editable is False
    assert roles["factors_key"].editable is False

    # The derived precision sheet has exactly the fixture's one FAIL (s1).
    _, precision_rows = read_family_csv(tmp_path / "hi_calibration_precision.csv")
    assert len(precision_rows) == 1
    headers, grid_rows = read_family_csv(tmp_path / "hi_calibration.csv")
    assert len(grid_rows) == 2  # one row per sim

    # Fill + ingest the precision sheet through the artifact layer.
    fill_csv(
        tmp_path / "hi_calibration_precision.csv",
        lambda i, row: row.update({"Real violation?": "✅ yes"}),
    )
    artifact = read_artifact(tmp_path / "hi_calibration_precision.csv")
    assert artifact.role == "precision"
    rows = [JudgePrecisionRow.model_validate(r) for r in artifact.rows("precision")]
    metrics = analyze_precision(rows)
    assert metrics[("hi", FAIL_FACTOR)].precision == 1.0

    # Fill + ingest the cold grid: label s1's fail-factor column yes, s2's no.
    reg_nuance = next(
        k for k in artifact.rows("factors_key") if k.factor_id == FAIL_FACTOR
    ).nuance

    def label(i, row):
        row[reg_nuance] = "yes" if row["sim_id"] == "s1" else "no"

    fill_csv(tmp_path / "hi_calibration.csv", label)
    cold_artifact = read_artifact(tmp_path / "hi_calibration.csv")
    sheet = cold_artifact.cold_sheet()
    metrics = analyze_cold(sheet, cold_artifact.rows("judge_sidecar"))
    m = metrics[("hi", FAIL_FACTOR)]
    assert m.counts.tp == 1 and m.counts.tn == 1  # judge FAIL on s1, PASS on s2
    assert m.precision == 1.0 and m.recall == 1.0


def test_export_precision_family(hi_results_dir, tmp_path):
    manifest_path = export_nativeness_sheets(
        [hi_results_dir], "precision", tmp_path / "hi_precision", reuse_existing=True
    )
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == "nativeness_precision"
    headers, rows = read_family_csv(tmp_path / "hi_precision.csv")
    assert headers == JudgePrecisionRow.headers()
    assert len(rows) == 1  # only s1's FAIL
