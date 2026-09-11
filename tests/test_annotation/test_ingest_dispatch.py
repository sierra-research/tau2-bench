# Copyright Sierra
"""ONE ingest verb: dispatch on the artifact manifest's kind (+ role).

Browser-exported packet CSVs are the one manifest-less path: they dispatch by
exact header match against the packet form row models."""

import argparse
import json

import pytest
from pydantic import ValidationError

from tau2.annotation.cli import add_annotate_args
from tau2.annotation.models import ErrorAnalysisRow, VoiceReviewRow
from tau2.annotation.nativeness_sheets import export_nativeness_sheets
from tau2.multilingual.factory.paths import calibration_dir
from test_annotation.conftest import fill_csv, make_hi_results_dir, write_family_csv


@pytest.fixture
def annotate(tmp_path):
    parser = argparse.ArgumentParser(prog="tau2 annotate")
    add_annotate_args(parser)

    def run(*argv: str):
        args = parser.parse_args(list(argv))
        args.func(args)

    return run


@pytest.fixture
def both_family(tmp_path):
    run_dir = make_hi_results_dir(tmp_path)
    out_stem = tmp_path / "hi_cal"
    export_nativeness_sheets([run_dir], "both", out_stem, reuse_existing=True)
    return tmp_path


def test_ingest_dispatches_precision_role(annotate, both_family, capsys):
    fill_csv(
        both_family / "hi_cal_precision.csv",
        lambda i, row: row.update({"Real violation?": "❌ no"}),
    )
    annotate("ingest", str(both_family / "hi_cal_precision.csv"))
    out = capsys.readouterr().out
    assert "kind: nativeness_cold | role: precision" in out
    assert "precision" in out and "shadow" in out  # 0.0 precision gates shadow


def test_ingest_dispatches_cold_grid_with_second_annotator(
    annotate, both_family, capsys
):
    from tau2.annotation.artifacts import read_artifact

    keys = read_artifact(both_family / "hi_cal.csv").rows("factors_key")
    nuance = keys[0].nuance

    def label_yes(i, row):
        row[nuance] = "yes"

    fill_csv(both_family / "hi_cal.csv", label_yes)
    second = both_family / "second.csv"
    second.write_bytes((both_family / "hi_cal.csv").read_bytes())
    annotate(
        "ingest",
        str(both_family / "hi_cal.csv"),
        "--second",
        str(second),
        "--precision-bar",
        "0.5",
    )
    out = capsys.readouterr().out
    assert "kind: nativeness_cold | role: sheet" in out
    assert "hkappa" in out  # inter-annotator column present


def test_ingest_dispatches_translation_and_judge_sample(
    annotate, isolated_pack_env, tmp_path, capsys
):
    from test_multilingual.factory_testing.toy_language import TOY_DOMAIN, TOY_LANGUAGE

    from tau2.annotation.communicate_judge import export_communicate_judge_sample
    from tau2.annotation.translation import export_translation_review
    from test_annotation.conftest import judged_run_dir, seed_pipeline_state

    # Translation review (export is the `tau2 annotate translation-review`
    # verb; ingest of the filled sheet stays here).
    seed_pipeline_state(n_verified=1, n_flagged=1)
    export_translation_review(TOY_LANGUAGE, TOY_DOMAIN)
    csv_path = calibration_dir(TOY_LANGUAGE) / f"{TOY_DOMAIN}_translation_review.csv"
    fill_csv(
        csv_path,
        lambda i, row: row.update({"meaning_ok (Y/N)": "Y", "natural_ok (Y/N)": "Y"}),
    )
    annotate("ingest", str(csv_path))
    agreement = (
        calibration_dir(TOY_LANGUAGE) / f"{TOY_DOMAIN}_translation_agreement.json"
    )
    assert agreement.exists()
    assert json.loads(agreement.read_text())["summary"]["n_compared"] == 2

    # Communicate-judge sample (export verb retired with its calibration
    # wave; the library export and the ingest dispatch remain).
    run_dir = judged_run_dir(tmp_path)
    export_communicate_judge_sample(run_dir, seed=0)
    sample_csv = calibration_dir(TOY_LANGUAGE) / "communicate_judge_sample.csv"
    fill_csv(sample_csv, lambda i, row: row.update({"native_verdict (Y/N)": "Y"}))
    annotate("ingest", str(sample_csv))
    judge_agreement = (
        calibration_dir(TOY_LANGUAGE) / f"{TOY_DOMAIN}_communicate_judge_agreement.json"
    )
    assert judge_agreement.exists()
    out = capsys.readouterr().out
    assert "kind: translation_review" in out
    assert "kind: communicate_judge" in out


def _browser_row(headers: list[str], **values) -> dict:
    row = dict.fromkeys(headers, "")
    row.update(values)
    return row


def test_ingest_browser_packet_csv_by_header_match(annotate, tmp_path, capsys):
    headers = VoiceReviewRow.headers()
    rows = [
        _browser_row(
            headers,
            batch="round1",
            rater="alice",
            task_id="t1",
            simulation_id="s1",
            trial="0",
            error_source="user",
            error_type="logical",
            notes="asked twice",
            speech_accuracy="3",
            completed="true",
            created_at="2026-01-01T00:00:00Z",
        ),
        _browser_row(headers, task_id="t2", simulation_id="s2", rater="bob"),
    ]
    csv_path = tmp_path / "round1_alice_export.csv"
    write_family_csv(csv_path, headers, rows)
    annotate("ingest", str(csv_path))
    out = capsys.readouterr().out
    assert "kind: packet_voice_review" in out
    assert "rows: 2" in out
    assert "completed: 1" in out
    assert "alice" in out and "bob" in out


def test_ingest_browser_csv_next_to_unrelated_manifest(annotate, both_family, capsys):
    """A browser packet CSV dropped into a directory holding an UNRELATED
    sheet-family manifest must fall back to header-match dispatch — the lone
    manifest never claims a CSV it does not list."""
    headers = VoiceReviewRow.headers()
    csv_path = both_family / "round1_alice_export.csv"
    write_family_csv(
        csv_path,
        headers,
        [_browser_row(headers, task_id="t1", simulation_id="s1", rater="alice")],
    )
    annotate("ingest", str(csv_path))
    out = capsys.readouterr().out
    assert "kind: packet_voice_review" in out
    assert "nativeness_cold" not in out


def test_ingest_browser_error_analysis_csv(annotate, tmp_path, capsys):
    headers = ErrorAnalysisRow.headers()
    csv_path = tmp_path / "errors.csv"
    write_family_csv(
        csv_path,
        headers,
        [_browser_row(headers, task_id="t1", simulation_id="s1", error_source="agent")],
    )
    annotate("ingest", str(csv_path))
    assert "kind: packet_error_analysis" in capsys.readouterr().out


def test_ingest_browser_csv_junk_cell_is_loud(annotate, tmp_path):
    headers = ErrorAnalysisRow.headers()
    csv_path = tmp_path / "junk.csv"
    write_family_csv(
        csv_path,
        headers,
        [
            _browser_row(
                headers, task_id="t1", simulation_id="s1", error_source="gremlin"
            )
        ],
    )
    with pytest.raises(ValidationError):
        annotate("ingest", str(csv_path))


def test_ingest_unmatched_headers_is_loud(annotate, tmp_path):
    csv_path = tmp_path / "mystery.csv"
    write_family_csv(csv_path, ["foo", "bar"], [{"foo": "1", "bar": "2"}])
    with pytest.raises(ValueError, match="packet's CSV contract"):
        annotate("ingest", str(csv_path))


def test_ingest_metrics_csv_out(annotate, both_family, tmp_path):
    fill_csv(
        both_family / "hi_cal_precision.csv",
        lambda i, row: row.update({"Real violation?": "✅ yes"}),
    )
    metrics_out = tmp_path / "metrics.csv"
    annotate(
        "ingest", str(both_family / "hi_cal_precision.csv"), "--out", str(metrics_out)
    )
    assert metrics_out.exists()
    assert "precision" in metrics_out.read_text()
