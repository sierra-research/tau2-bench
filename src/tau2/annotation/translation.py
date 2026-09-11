# Copyright Sierra
"""Translation-review export/ingest: native annotators vs the factory pipeline.

Export is LEGACY (task translation is retired; the paper flow is English-prose
seed tasks) and is invoked via ``tau2 annotate translation-review``;
filled sheets still ingest through ``tau2 annotate ingest``.

Export reads the factory's typed translation state (``load_translation_state``)
and emits a ``TranslationReviewRow`` sheet family with the pipeline verdict in
typed hidden columns. With ``from_existing`` an already-translated task-set
JSON is exported instead, verdict-less — retro-calibration of sets translated
before the factory existed.

Ingest compares the native verdicts against the pipeline's row-by-row and
persists the agreement summary under ``_factory/<lang>/calibration/``, one
file per (language, domain) — exports are per-(lang, domain), so persisted
agreements must be too (a second domain must never clobber the first).
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from tau2.annotation.artifacts import (
    LoadedArtifact,
    payload_from_rows,
    provenance_stamp,
    write_json_artifact,
    write_sheet_family,
)
from tau2.annotation.metrics import (
    AgreementRecord,
    AgreementSummary,
    summarize_agreement,
)
from tau2.annotation.models import PipelineVerdict, TranslationReviewRow
from tau2.multilingual.factory.paths import calibration_dir
from tau2.multilingual.factory.translation_rows import (
    RowState,
    RowStatus,
    judging_provenance,
    load_translation_state,
    rows_path,
)
from tau2.multilingual.localize_lib import iter_rows, load_domain_tasks

# Persisted agreement files are keyed by domain: <domain> + this suffix.
TRANSLATION_AGREEMENT_SUFFIX = "_translation_agreement.json"


class AgreementFile(BaseModel):
    """The persisted agreement artifact for one (language, domain) ingest.

    Written by ``persist_agreement`` (shared by the translation and
    communicate-judge ingests) and read back by
    ``reports.agreement_report`` — the typed envelope both ends validate
    against.
    """

    kind: str
    domain: str
    source_csv: str
    created_at: str = ""
    git_sha: str = ""
    records: list[AgreementRecord]
    summary: AgreementSummary


_STATUS_TO_VERDICT = {
    RowStatus.VERIFIED: PipelineVerdict.VERIFIED,
    RowStatus.FLAGGED: PipelineVerdict.FLAGGED,
    RowStatus.UNTRANSLATED: PipelineVerdict.UNTRANSLATED,
}


def _pipeline_columns(row: RowState) -> tuple[Optional[PipelineVerdict], str]:
    """(verdict, detail) for one translation row — typed, not a comment hack."""
    verdict = _STATUS_TO_VERDICT.get(row.status)
    if row.status is RowStatus.VERIFIED:
        return verdict, f"attempt {len(row.attempts)}"
    if row.status in (RowStatus.FLAGGED, RowStatus.UNTRANSLATED):
        return verdict, "; ".join(row.last_issues)
    return None, row.status.value


def _rows_from_state(lang: str, domain: str) -> tuple[list[TranslationReviewRow], dict]:
    state = load_translation_state(lang, domain)
    if state is None:
        raise FileNotFoundError(
            f"No translation rows at {rows_path(lang, domain)} — run "
            "the retired task-translation loop first — it is deleted, so only "
            "packs from the translation era have rows. Pass --from-existing "
            "for a set that was never run through it."
        )
    models = {
        (row.translator_model, row.verifier_model)
        for row in state.rows
        if row.translator_model
    }
    rows: list[TranslationReviewRow] = []
    for row in sorted(state.rows, key=lambda r: r.row_id):
        translation = row.final_translation or (
            row.attempts[-1].translation if row.attempts else ""
        )
        verdict, detail = _pipeline_columns(row)
        rows.append(
            TranslationReviewRow(
                row_id=row.row_id,
                task_id=f"{row.task_id}_{lang}",
                field=row.field,
                english_original=row.english,
                translation=translation,
                # Flagged rows get their best attempt prefilled as the
                # suggested rewrite for the annotator to correct.
                suggested_rewrite=(
                    row.final_translation or ""
                    if row.status is RowStatus.FLAGGED
                    else ""
                ),
                pipeline_verdict=verdict,
                pipeline_detail=detail,
            )
        )
    pair = next(iter(models), (None, None))
    extra = {"mode": "pipeline", "translator_model": pair[0], "verifier_model": pair[1]}
    return rows, extra


def _rows_from_existing(
    lang: str, domain: str, from_existing: Path
) -> tuple[list[TranslationReviewRow], dict]:
    """Verdict-less rows from an already-translated task-set JSON.

    Tasks without an English source and fields without a source counterpart
    cannot be reviewed side-by-side; they are dropped from the sheet, and the
    drop counts are recorded in the export provenance (a silent drop would
    hide a coverage hole from the calibration round).
    """
    source_by_id = {t["id"]: t for t in load_domain_tasks(domain)}
    localized = json.loads(Path(from_existing).read_text())
    rows: list[TranslationReviewRow] = []
    row_id = 0
    dropped_tasks: list[str] = []
    dropped_fields = 0
    for task in localized:
        source = source_by_id.get(task["id"].removesuffix(f"_{lang}"))
        if source is None:
            dropped_tasks.append(task["id"])
            continue
        source_fields = dict(iter_rows(source))
        for label, text in iter_rows(task):
            if label not in source_fields:
                dropped_fields += 1
                continue
            row_id += 1
            rows.append(
                TranslationReviewRow(
                    row_id=row_id,
                    task_id=task["id"],
                    field=label,
                    english_original=source_fields[label],
                    translation=text,
                )
            )
    if dropped_tasks or dropped_fields:
        logger.warning(
            f"translation review export dropped {len(dropped_tasks)} task(s) "
            f"without an English source and {dropped_fields} field(s) without "
            "a source counterpart"
        )
    extra = {
        "mode": "from_existing",
        "source": str(from_existing),
        "dropped_tasks_without_source": dropped_tasks,
        "dropped_fields_without_source": dropped_fields,
    }
    return rows, extra


def export_translation_review(
    lang: str,
    domain: str,
    from_existing: Optional[Path] = None,
    out_stem: Optional[Path] = None,
) -> Path:
    """Write the native-review sheet family for one language x domain.

    Returns the manifest path; the annotator CSV sits beside it.
    """
    if from_existing is not None:
        rows, extra = _rows_from_existing(lang, domain, from_existing)
    else:
        rows, extra = _rows_from_state(lang, domain)
    provenance = {
        **judging_provenance(lang).model_dump(),
        "domain": domain,
        **extra,
    }
    out_stem = out_stem or calibration_dir(lang) / f"{domain}_translation_review"
    return write_sheet_family(
        out_stem,
        kind="translation_review",
        payloads={"sheet": payload_from_rows(TranslationReviewRow, rows)},
        provenance=provenance,
        language=lang,
        domain=domain,
    )


def agreement_records(rows: list[TranslationReviewRow]) -> list[AgreementRecord]:
    """Annotated rows as agreement records (unlabeled rows are skipped)."""
    records: list[AgreementRecord] = []
    for row in rows:
        native_ok = row.native_ok
        if native_ok is None:
            continue  # not annotated
        records.append(
            AgreementRecord(
                task_id=row.task_id,
                field=row.field,
                pipeline_ok=row.pipeline_ok,
                native_ok=native_ok,
                issue_type=row.issue_type,
                comments=row.comments,
            )
        )
    return records


def persist_agreement(
    lang: str,
    domain: str,
    suffix: str,
    kind: str,
    source: Path,
    records: list[AgreementRecord],
) -> Path:
    """Persist agreement records + summary under the language's calibration
    dir, keyed by domain (shared by the translation and communicate-judge
    ingests). File name = ``<domain><suffix>``."""
    payload = AgreementFile(
        kind=kind,
        domain=domain,
        source_csv=str(source),
        **provenance_stamp(),
        records=records,
        summary=summarize_agreement(records),
    )
    path = write_json_artifact(calibration_dir(lang) / f"{domain}{suffix}", payload)
    logger.info(f"persisted {len(records)} {kind} agreement record(s) to {path}")
    return path


def ingest_translation_review(artifact: LoadedArtifact, source: Path) -> Path:
    """Compare native verdicts vs pipeline verdicts; persist the agreement."""
    lang = artifact.manifest.language
    if not lang:
        raise ValueError(f"manifest {artifact.manifest.batch_name} carries no language")
    domain = artifact.manifest.domain
    if not domain:
        raise ValueError(f"manifest {artifact.manifest.batch_name} carries no domain")
    rows = artifact.rows("sheet")
    return persist_agreement(
        lang,
        domain,
        TRANSLATION_AGREEMENT_SUFFIX,
        "translation",
        source,
        agreement_records(rows),
    )
