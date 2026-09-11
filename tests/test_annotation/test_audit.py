# Copyright Sierra
"""Audit tab bodies: schema, dropdowns, blind deltas, file outputs."""

import json

import pytest

from tau2.annotation.models import AuditNuanceRow, AuditStatus
from tau2.annotation.sheets.audit import (
    AUDIT_TABS,
    HEADERS,
    SEVERITY_DISPLAY,
    build_audit_bodies,
    build_blind_leaderboard_body,
    build_blind_tab_body,
    build_leaderboard_body,
    build_tab_body,
    tab_categories,
)


def _rows():
    return [
        AuditNuanceRow(
            category="Register & formality",
            nuance="aap vs tum drift",
            ai_does="drops to tum mid-call",
            native_does="keeps aap with a stranger agent",
            severity=3,
        ),
        AuditNuanceRow(
            category="Numbers (word-forms, gender, dates)",
            nuance="digit-by-digit codes",
            ai_does="reads codes as whole numbers",
            native_does="spells codes digit by digit",
            severity=2,
        ),
    ]


def test_registry_covers_the_kept_languages_with_unique_tabs():
    assert set(AUDIT_TABS) == {"es", "hi", "ko", "pt", "zh"}
    sheet_ids = [cfg.sheet_id for cfg in AUDIT_TABS.values()]
    assert len(set(sheet_ids)) == len(sheet_ids)
    names = [cfg.tab_name for cfg in AUDIT_TABS.values()]
    assert len(set(names)) == len(names)


def test_tab_body_writes_schema_and_rows():
    cfg = AUDIT_TABS["hi"]
    body = build_tab_body(cfg, _rows())
    reqs = body["requests"]
    updates = [r["updateCells"] for r in reqs if "updateCells" in r]
    # Header row at index 3, with {L} filled with the tab name.
    header_update = next(u for u in updates if u["start"].get("rowIndex") == 3)
    header_values = [
        v["userEnteredValue"]["stringValue"] for v in header_update["rows"][0]["values"]
    ]
    assert header_values == [h.replace("{L}", "Hindi") for h in HEADERS]
    # Data rows start at index 4; status/severity render display labels.
    data_update = next(u for u in updates if u["start"].get("rowIndex") == 4)
    first = [
        v["userEnteredValue"]["stringValue"] for v in data_update["rows"][0]["values"]
    ]
    assert first[1] == AuditStatus.AUTO.display
    assert first[3] == "aap vs tum drift"
    assert first[6] == "" and first[7] == ""  # example/ref left for annotators
    assert first[8] == SEVERITY_DISPLAY[3]
    # Dropdown validations: Status, Category (with conditional cats), Severity.
    validations = [r["setDataValidation"] for r in reqs if "setDataValidation" in r]
    option_lists = [
        [v["userEnteredValue"] for v in rule["rule"]["condition"]["values"]]
        for rule in validations
        if rule.get("rule")
    ]
    assert AuditStatus.options() in option_lists
    assert tab_categories(cfg) in option_lists
    assert list(SEVERITY_DISPLAY.values()) in option_lists
    # Frozen header rows preserved.
    assert any("updateSheetProperties" in r for r in reqs)


def test_tab_categories_dedupe_and_conditionals():
    cats = tab_categories(AUDIT_TABS["zh"])
    assert "Measure words (量词)" in cats
    assert cats.count("Tones (meaning/word choice)") == 1  # deduped
    assert cats[-1] == "Other"


def test_blind_tab_deletes_status_column_last():
    body = build_blind_tab_body(AUDIT_TABS["es"])
    reqs = body["requests"]
    # The column delete MUST be the final request: everything before it uses
    # pre-delete column indices.
    assert "deleteDimension" in reqs[-1]
    delete = reqs[-1]["deleteDimension"]["range"]
    assert (delete["startIndex"], delete["endIndex"]) == (1, 2)
    assert not any("deleteDimension" in r for r in reqs[:-1])


def test_leaderboards_count_correct_columns():
    round2 = build_leaderboard_body()
    formulas = [
        row["values"][0]["userEnteredValue"]["formulaValue"]
        for row in round2["requests"][0]["updateCells"]["rows"]
    ]
    assert len(formulas) == 5  # one per audit tab
    # Round 2 counts the Confirmed dropdown value with COUNTIF (never COUNTA).
    assert all(f.startswith("=COUNTIF(") and "✅ Confirmed" in f for f in formulas)
    blind = build_blind_leaderboard_body()
    blind_formulas = [
        row["values"][0]["userEnteredValue"]["formulaValue"]
        for row in blind["requests"][0]["updateCells"]["rows"]
    ]
    # Round 1 counts free-text findings with a <>"" test (COUNTA would count
    # the template's pre-filled empty strings), over col C rows 7+.
    assert all("SUMPRODUCT" in f and "C7:C1000<>" in f for f in blind_formulas)


def test_build_audit_bodies_reads_candidates_and_writes_manifest(tmp_path):
    audit_dir = tmp_path / "nuance_audit"
    audit_dir.mkdir()
    (audit_dir / "hi.json").write_text(
        json.dumps(
            {
                "language": "hi",
                "model": "test-model",
                "prompt_version": "v1",
                "prompt_sha256": "a" * 64,
                "created_at": "2026-01-01T00:00:00Z",
                "git_sha": "test-sha",
                "nuances": [
                    {
                        "category": "Register & formality",
                        "title": "aap vs tum drift",
                        "ai_likely_does": "drops to tum",
                        "native_does": "keeps aap",
                        "severity": 3,
                    }
                ],
            }
        )
    )
    written = build_audit_bodies(["hi"], audit_dir=audit_dir)
    out_dir = audit_dir / "bodies"
    assert (out_dir / "hi.json").exists()
    assert (out_dir / "leaderboard.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["files"] == ["hi.json", "leaderboard.json"]
    assert manifest["created_at"] and manifest["git_sha"]
    assert len(written) == 2
    body = json.loads((out_dir / "hi.json").read_text())
    assert body["requests"]


def test_build_audit_bodies_blind_needs_no_candidates(tmp_path):
    audit_dir = tmp_path / "nuance_audit"
    audit_dir.mkdir()
    build_audit_bodies(["es", "hi"], blind=True, audit_dir=audit_dir)
    out_dir = audit_dir / "bodies_blind"
    assert (out_dir / "es.json").exists()
    assert (out_dir / "start.json").exists()
    assert (out_dir / "leaderboard.json").exists()


def test_build_audit_bodies_missing_candidates_is_loud(tmp_path):
    audit_dir = tmp_path / "nuance_audit"
    audit_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="annotate audit --generate"):
        build_audit_bodies(["hi"], audit_dir=audit_dir)
    with pytest.raises(ValueError, match="unknown audit language"):
        build_audit_bodies(["xx"], audit_dir=audit_dir)
