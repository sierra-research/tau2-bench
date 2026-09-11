# Copyright Sierra
"""Translation-review + communicate-judge calibration round trips (no LLM).

Rewrites the old factory-calibration suite on the annotation package: export
with typed pipeline-verdict columns → programmatic native fill → ingest via
the artifact layer → agreement report with a hand-computed Cohen's kappa;
plus the verdict-less ``--from-existing`` export, manifest provenance, the
seeded communicate-judge sample, and the parity-probe surfacing.
"""

import hashlib
import json

import pytest
from test_multilingual.factory_testing.results_factory import make_run_dir
from test_multilingual.factory_testing.toy_language import (
    TOY_DOMAIN,
    TOY_LANGUAGE,
    toy_tasks_localized,
)

from tau2.annotation.artifacts import ArtifactManifest, read_artifact
from tau2.annotation.communicate_judge import (
    export_communicate_judge_sample,
    ingest_communicate_judge,
)
from tau2.annotation.models import CommunicateJudgeRow, TranslationReviewRow
from tau2.annotation.reports import agreement_report, summarize_parity_probe
from tau2.annotation.translation import (
    export_translation_review,
    ingest_translation_review,
)
from tau2.multilingual.factory.paths import calibration_dir
from tau2.multilingual.factory.state import factory_root_dir
from tau2.multilingual.factory.translation_rows import VERIFIER_SYSTEM_PROMPT
from test_annotation.conftest import (
    fill_csv,
    judged_run_dir,
    read_family_csv,
    seed_pipeline_state,
)

# ---------------------------------------------------------------------------
# Translation review: export -> fill -> ingest -> report
# ---------------------------------------------------------------------------


def test_export_fill_ingest_report_round_trip(isolated_pack_env):
    seed_pipeline_state(n_verified=5, n_flagged=5)
    manifest_path = export_translation_review(TOY_LANGUAGE, TOY_DOMAIN)
    csv_path = manifest_path.with_name(
        manifest_path.name.replace(".manifest.json", ".csv")
    )
    header, rows = read_family_csv(csv_path)
    assert header == TranslationReviewRow.headers()
    assert len(rows) == 10
    # Typed pipeline columns replace the old "pipeline: ..." comment hack;
    # annotation task ids carry the language suffix like the shipped packs.
    assert rows[0]["pipeline_verdict"] == "verified"
    assert rows[0]["pipeline_detail"] == "attempt 1"
    assert rows[0]["task_id"] == f"1_{TOY_LANGUAGE}"
    assert rows[0]["comments"] == ""  # comments are the annotator's now
    assert rows[5]["pipeline_verdict"] == "flagged"
    assert rows[5]["pipeline_detail"] == "meaning drift"
    # Flagged rows get their best attempt prefilled as the suggested rewrite.
    assert rows[5]["suggested_rewrite (only if N)"] == rows[5]["translation"]
    assert rows[0]["suggested_rewrite (only if N)"] == ""

    # Native fill with a known confusion matrix: a=4, b=1, c=2, d=3.
    def fill(i, row):
        pipeline_ok = i < 5
        native_ok = i in (0, 1, 2, 3, 5, 6)
        row["meaning_ok (Y/N)"] = "Y" if native_ok else "N"
        row["natural_ok (Y/N)"] = "Y" if native_ok else "N"
        if not native_ok:
            row["issue_type"] = "meaning" if pipeline_ok else "naturalness"

    fill_csv(csv_path, fill)

    artifact = read_artifact(csv_path)
    agreement = ingest_translation_review(artifact, csv_path)
    assert (
        agreement
        == calibration_dir(TOY_LANGUAGE) / f"{TOY_DOMAIN}_translation_agreement.json"
    )
    payload = json.loads(agreement.read_text())
    assert payload["kind"] == "translation"
    assert payload["domain"] == TOY_DOMAIN
    assert len(payload["records"]) == 10

    report = agreement_report(TOY_LANGUAGE)
    summary = report["translation"][TOY_DOMAIN]
    assert summary["n_compared"] == 10
    assert summary["percent_agreement"] == pytest.approx(0.7)
    assert summary["cohens_kappa"] == pytest.approx(0.4)
    assert summary["confusion"] == {
        "both_ok": 4,
        "pipeline_only": 1,
        "native_only": 2,
        "neither": 3,
    }
    assert "task_instructions" in summary["per_field"]
    assert summary["per_issue_type"]["meaning"]["pipeline_only"] == 1
    assert summary["per_issue_type"]["naturalness"]["neither"] == 3
    assert report["communicate_judge"] is None
    assert report["parity_probe"] is None


def test_agreement_files_are_per_domain_and_report_lists_all(isolated_pack_env):
    """Ingesting two domains must persist two files (no clobber) and the
    report must surface both keyed by domain."""
    from tau2.annotation.metrics import AgreementRecord
    from tau2.annotation.translation import (
        TRANSLATION_AGREEMENT_SUFFIX,
        persist_agreement,
    )

    def record(task_id: str) -> AgreementRecord:
        return AgreementRecord(
            task_id=task_id,
            field="task_instructions",
            pipeline_ok=True,
            native_ok=True,
        )

    a = persist_agreement(
        TOY_LANGUAGE,
        "airline",
        TRANSLATION_AGREEMENT_SUFFIX,
        "translation",
        calibration_dir(TOY_LANGUAGE) / "airline.csv",
        [record("1")],
    )
    b = persist_agreement(
        TOY_LANGUAGE,
        "retail",
        TRANSLATION_AGREEMENT_SUFFIX,
        "translation",
        calibration_dir(TOY_LANGUAGE) / "retail.csv",
        [record("2"), record("3")],
    )
    assert a != b and a.exists() and b.exists()
    report = agreement_report(TOY_LANGUAGE)["translation"]
    assert set(report) == {"airline", "retail"}
    assert report["airline"]["n_records"] == 1
    assert report["retail"]["n_records"] == 2


def test_export_requires_rows_or_from_existing(isolated_pack_env):
    with pytest.raises(FileNotFoundError, match="from-existing"):
        export_translation_review(TOY_LANGUAGE, TOY_DOMAIN)


def test_from_existing_export_is_verdict_less(isolated_pack_env):
    manifest_path = export_translation_review(
        TOY_LANGUAGE,
        TOY_DOMAIN,
        from_existing=isolated_pack_env.localized_tasks_fixture,
    )
    csv_path = manifest_path.with_name(
        manifest_path.name.replace(".manifest.json", ".csv")
    )
    header, rows = read_family_csv(csv_path)
    assert header == TranslationReviewRow.headers()
    assert len(rows) == 19  # all extract rows of the 4 toyair tasks
    assert all(row["pipeline_verdict"] == "" for row in rows)
    localized_first = toy_tasks_localized()[0]
    assert rows[0]["task_id"] == localized_first["id"]
    assert (
        rows[0]["translation"]
        == localized_first["user_scenario"]["instructions"]["task_instructions"]
    )

    # Verdict-less rows ingest fine but are excluded from the comparison.
    fill_csv(
        csv_path,
        lambda i, row: row.update({"meaning_ok (Y/N)": "Y", "natural_ok (Y/N)": "Y"}),
    )
    artifact = read_artifact(csv_path)  # language resolved from the manifest
    ingest_translation_review(artifact, csv_path)
    summary = agreement_report(TOY_LANGUAGE)["translation"][TOY_DOMAIN]
    assert summary["n_records"] == 19
    assert summary["n_compared"] == 0
    assert summary["percent_agreement"] is None
    assert summary["cohens_kappa"] is None


def test_export_manifest_provenance(isolated_pack_env):
    seed_pipeline_state()
    project_dir = factory_root_dir() / TOY_LANGUAGE
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(
        json.dumps({"language": TOY_LANGUAGE, "pack_revision": 3})
    )

    manifest_path = export_translation_review(TOY_LANGUAGE, TOY_DOMAIN)
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.language == TOY_LANGUAGE
    assert manifest.domain == TOY_DOMAIN
    assert manifest.provenance["translator_model"] == "model-a"
    assert manifest.provenance["verifier_model"] == "model-b"
    assert manifest.provenance["pack_revision"] == 3
    assert (
        manifest.provenance["verifier_prompt_sha256"]
        == hashlib.sha256(VERIFIER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    )


def test_agreement_report_surfaces_parity_probe(isolated_pack_env):
    """The parity-probe verdict sink (written by factory.parity) must be read
    back by agreement_report."""
    from tau2.multilingual.factory.paths import PARITY_PROBE_FILENAME

    assert summarize_parity_probe([]) == {"n_runs": 0, "latest": None}
    assert agreement_report(TOY_LANGUAGE)["parity_probe"] is None

    out_dir = calibration_dir(TOY_LANGUAGE)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "domain": "toyair",
            "aggregate_gap": 0.05,
            "total_tasks": 4,
            "flagged": [],
            "tasks": {},
        },
        {
            "domain": "toyair",
            "aggregate_gap": 0.04,
            "total_tasks": 4,
            "flagged": ["3"],
            "tasks": {},
        },
    ]
    (out_dir / PARITY_PROBE_FILENAME).write_text(json.dumps(records))

    parity_summary = agreement_report(TOY_LANGUAGE)["parity_probe"]
    assert parity_summary["n_runs"] == 2
    assert parity_summary["latest"]["n_flagged"] == 1
    assert parity_summary["latest"]["flagged"] == ["3"]


# ---------------------------------------------------------------------------
# Communicate-judge calibration
# ---------------------------------------------------------------------------


def test_judge_sample_export_ingest_and_agreement(isolated_pack_env, tmp_path):
    run_dir = judged_run_dir(tmp_path)
    manifest_path = export_communicate_judge_sample(run_dir, n=50, seed=1)
    csv_path = manifest_path.with_name(
        manifest_path.name.replace(".manifest.json", ".csv")
    )
    header, rows = read_family_csv(csv_path)
    assert header == CommunicateJudgeRow.headers()
    assert len(rows) == 2
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["1"]["judge_verdict (Y/N)"] == "Y"
    assert by_task["3"]["judge_verdict (Y/N)"] == "N"
    assert by_task["1"]["criterion"] == "$250 refund to the original payment method"
    assert "refund van $250" in by_task["1"]["agent_turn_excerpt"]
    assert all(row["native_verdict (Y/N)"] == "" for row in rows)
    assert by_task["3"]["comments"].startswith("[llm_judge]")
    assert by_task["3"]["llm_judged"] == "Y"

    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    # Results record no judge model, so provenance says so instead of
    # parroting the current config default (which may not be what judged).
    assert manifest.provenance["judge_model"] == "unknown"
    assert manifest.language == TOY_LANGUAGE  # inferred from the run
    assert manifest.provenance["sample_seed"] == 1

    # Native annotator agrees on task 1, disagrees on task 3.
    def fill(i, row):
        row["native_verdict (Y/N)"] = "Y"

    fill_csv(csv_path, fill)
    ingest_communicate_judge(read_artifact(csv_path), csv_path)
    summary = agreement_report(TOY_LANGUAGE)["communicate_judge"][TOY_DOMAIN]
    assert summary["n_compared"] == 2
    assert summary["percent_agreement"] == pytest.approx(0.5)
    assert summary["confusion"] == {
        "both_ok": 1,
        "pipeline_only": 0,
        "native_only": 1,
        "neither": 0,
    }


def test_agent_excerpt_full_transcript_capped_with_note(isolated_pack_env, tmp_path):
    """The excerpt is the FULL agent-side transcript (all agent turns), capped
    at the configured max with an in-cell truncation note."""
    from tau2.annotation.communicate_judge import TRUNCATION_NOTE
    from tau2.config import DEFAULT_ANNOTATION_AGENT_EXCERPT_MAX_CHARS

    long_turn = "refund verwerkt. " * 400  # ~6800 chars > 4000 cap
    run_dir = make_run_dir(
        tmp_path,
        "toylang",
        rewards={"1": [1.0]},
        communicate_checks_by_task={
            "1": [{"info": "$250 refund", "met": True, "justification": "ok"}]
        },
        agent_messages_by_task={"1": ["Eerste zin over de refund.", long_turn]},
    )
    export_communicate_judge_sample(run_dir, n=5, seed=0, out_stem=tmp_path / "sample")
    _, rows = read_family_csv(tmp_path / "sample.csv")
    excerpt = rows[0]["agent_turn_excerpt"]
    # Both agent turns present (not just the first ~chars of one turn)...
    assert excerpt.startswith("Eerste zin over de refund.")
    assert "refund verwerkt." in excerpt
    # ...capped at the configured max plus the visible truncation note.
    assert excerpt.endswith(TRUNCATION_NOTE)
    assert len(excerpt) == DEFAULT_ANNOTATION_AGENT_EXCERPT_MAX_CHARS + len(
        TRUNCATION_NOTE
    )


def test_judge_sample_is_seed_deterministic(isolated_pack_env, tmp_path):
    checks = {
        str(i): [
            {"info": f"fact {i}", "met": i % 2 == 0, "justification": "[llm_judge] x"}
        ]
        for i in range(8)
    }
    run_dir = make_run_dir(
        tmp_path,
        "toylang",
        rewards={str(i): [1.0] for i in range(8)},
        communicate_checks_by_task=checks,
    )
    out_dir = calibration_dir(TOY_LANGUAGE)
    first = export_communicate_judge_sample(
        run_dir, n=3, seed=7, out_stem=out_dir / "sample_a"
    )
    ma = ArtifactManifest.model_validate_json(first.read_text())
    first_csv = (out_dir / "sample_a.csv").read_bytes()
    # Same seed + different stem: identical sampled content.
    export_communicate_judge_sample(run_dir, n=3, seed=7, out_stem=out_dir / "sample_b")
    assert first_csv == (out_dir / "sample_b.csv").read_bytes()
    # Content-derived batch id: RE-exporting the same family reproduces it
    # (batch_name participates in the hash, so a different stem differs).
    again = export_communicate_judge_sample(
        run_dir, n=3, seed=7, out_stem=out_dir / "sample_a"
    )
    assert (
        ArtifactManifest.model_validate_json(again.read_text()).batch_id == ma.batch_id
    )
    _, rows = read_family_csv(out_dir / "sample_a.csv")
    assert len(rows) == 3


def test_judge_sample_lang_must_be_explicit_when_ambiguous(isolated_pack_env, tmp_path):
    run_dir = make_run_dir(tmp_path, "english_arm", rewards={"1": [1.0]}, language=None)
    with pytest.raises(ValueError, match="pass lang explicitly"):
        export_communicate_judge_sample(run_dir, n=5, seed=0)


def test_mixed_run_exports_only_requested_language(isolated_pack_env, tmp_path):
    """export_communicate_judge_sample must filter rows by language when lang
    is explicitly passed."""
    from tau2.data_model.simulation import Results

    checks_a = [{"info": "criterion A", "met": True, "justification": "ok"}]
    checks_b = [{"info": "criterion B", "met": False, "justification": "no"}]
    run_dir = make_run_dir(
        tmp_path,
        arm_name="mixed_run",
        rewards={"1": [1.0], "2": [1.0], "3": [1.0], "4": [1.0]},
        communicate_checks_by_task={
            "1": checks_a,
            "2": checks_a,
            "3": checks_b,
            "4": checks_b,
        },
        language="hi",
    )
    # Rewrite tasks 3+4 as a second language.
    results = Results.load(run_dir)
    for sim in results.simulations:
        if str(sim.task_id) in ("3", "4") and sim.speech_environment is not None:
            sim.speech_environment.language = "sw"
    results.save(run_dir, format="dir")

    out_stem = tmp_path / "sample"
    export_communicate_judge_sample(
        run_dir, n=100, seed=0, lang="hi", out_stem=out_stem
    )
    _, rows = read_family_csv(tmp_path / "sample.csv")
    exported_task_ids = {row["task_id"] for row in rows}
    assert exported_task_ids == {"1", "2"}
