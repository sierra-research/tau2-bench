# Copyright Sierra
"""Nativeness-judge calibration sheets over typed ``tau2.judges.export`` records.

Export modes:
  precision  One row per judge FAIL (the violations it flagged); the annotator
             marks each real / false → PRECISION.
  cold       WIDE blind sheet — one row per call, one dropdown column per
             judge factor, verdicts hidden in a sidecar; the annotator labels
             blind → PRECISION *and* RECALL, unbiased. The per-factor rubric
             rides the ``factors_key`` sidecar (it can't sit inline when
             factors are columns).
  both       One judge pass: the cold family plus a precision sheet derived
             from the sidecar's FAILs (no re-judging).

Analysis ingests the annotator-filled CSV (via ``tau2.annotation.artifacts``)
and reports, per (language, factor), how well the judge agrees with native
speakers, gating each factor LIVE vs shadow on a precision bar.
"""

from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

from loguru import logger

from tau2.annotation.artifacts import (
    LoadedArtifact,
    SheetPayload,
    payload_from_cold_sheet,
    payload_from_rows,
    write_sheet_family,
)
from tau2.annotation.models import (
    COLD_META_LABELS,
    ColdSheet,
    ColdSheetRow,
    ColdSidecarRow,
    FactorKeyRow,
    JudgePrecisionRow,
)
from tau2.config import (
    DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
    DEFAULT_JUDGE_STREAM_CONCURRENCY,
)
from tau2.data_model.simulation import JudgeOutcome, SimulationRun
from tau2.judges.export import (
    NativenessAnnotationRubric,
    factor_rubrics_for,
    iter_judged_sims_detailed,
    nativeness_verdicts,
)

NativenessSheetMode = Literal["precision", "cold", "both"]


def stream_judged_sims(
    results: Iterable[Path | str], *, reuse_existing: bool, concurrency: int
) -> Iterator[SimulationRun]:
    """Lazily stream judged sims across all result inputs (no accumulation)."""
    for path in results:
        logger.info(f"loading {path} ...")
        for judged in iter_judged_sims_detailed(
            Path(path), reuse_existing=reuse_existing, concurrency=concurrency
        ):
            yield judged.sim


def _rubrics(language: str) -> dict[str, NativenessAnnotationRubric]:
    return {r.factor_id: r for r in factor_rubrics_for(language)}


def factor_key_row(rubric: NativenessAnnotationRubric) -> FactorKeyRow:
    return FactorKeyRow(
        language=rubric.language,
        factor_id=rubric.factor_id,
        category=rubric.category,
        nuance=rubric.nuance,
        native_does=rubric.native_does,
        non_native_ref=rubric.ai_likely_does,
    )


def validate_cold_headers(
    language: str, rubrics: Iterable[NativenessAnnotationRubric]
) -> None:
    """Loud sheet-build-time guard against the two silent cold-grid collisions.

    The wide grid heads each factor column with the rubric's nuance text, so:
    (a) two factors sharing one nuance would collapse into one column,
        silently dropping a factor from calibration; and
    (b) a nuance named like a reserved meta column (transcript/Notes/trace
        ids) would corrupt the grid's render/ingest round trip.
    Both are pack-authoring bugs — fail naming the factor ids.
    """
    by_header: dict[str, list[str]] = {}
    for rubric in rubrics:
        header = rubric.nuance or rubric.factor_id
        by_header.setdefault(header, []).append(rubric.factor_id)
    duplicates = {h: ids for h, ids in by_header.items() if len(ids) > 1}
    if duplicates:
        detail = "; ".join(
            f"{h!r} <- {sorted(ids)}" for h, ids in sorted(duplicates.items())
        )
        raise ValueError(
            f"[{language}] cold-sheet column collision: factors share a "
            f"nuance header, which would silently drop factors from "
            f"calibration: {detail}"
        )
    reserved = {h: ids for h, ids in by_header.items() if h in COLD_META_LABELS}
    if reserved:
        detail = "; ".join(
            f"{h!r} <- {sorted(ids)}" for h, ids in sorted(reserved.items())
        )
        raise ValueError(
            f"[{language}] cold-sheet column collision: factor nuance headers "
            f"collide with reserved meta columns {sorted(COLD_META_LABELS)}: "
            f"{detail}"
        )


def build_precision_rows(
    sims: Iterable[SimulationRun],
    max_trans_chars: int = DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
) -> list[JudgePrecisionRow]:
    """One adjudication row per judge FAIL, from stored verdict records."""
    rows: list[JudgePrecisionRow] = []
    for sim in sims:
        rubric_cache: dict[str, NativenessAnnotationRubric] = {}
        for verdict in nativeness_verdicts(sim):
            if verdict.outcome is not JudgeOutcome.FAIL:
                continue
            if not rubric_cache:
                rubric_cache = _rubrics(verdict.language)
            rubric = rubric_cache.get(verdict.factor_id)
            rows.append(
                JudgePrecisionRow(
                    what_checked=rubric.nuance if rubric else "",
                    quote=verdict.quote or "",
                    why_flagged=verdict.reasoning or "",
                    transcript=verdict.transcript[:max_trans_chars],
                    factor_id=verdict.factor_id,
                    sim_id=verdict.sim_id,
                    task_id=verdict.task_id,
                    language=verdict.language,
                )
            )
    return rows


def build_cold_sheet(
    sims: Iterable[SimulationRun],
    max_trans_chars: int = DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
) -> tuple[ColdSheet, list[ColdSidecarRow]]:
    """The WIDE blind sheet + its hidden judge sidecar, from verdict records.

    One row per call (the annotator reads each transcript once and labels all
    factors in that row); the sidecar keeps the per-(sim, factor) verdicts for
    the confusion matrix — a factor with no stored check is a blank outcome.
    """
    rows: list[ColdSheetRow] = []
    sidecar: list[ColdSidecarRow] = []
    key: dict[tuple[str, str], FactorKeyRow] = {}
    validated_languages: set[str] = set()
    for sim in sims:
        verdicts = nativeness_verdicts(sim)
        if not verdicts:
            continue
        language = verdicts[0].language
        rubrics = _rubrics(language)
        if language not in validated_languages:
            validate_cold_headers(language, rubrics.values())
            validated_languages.add(language)
        labels: dict[str, None] = {}
        for verdict in verdicts:
            rubric = rubrics.get(verdict.factor_id)
            header = (rubric.nuance if rubric else "") or verdict.factor_id
            labels[header] = None  # blank annotator dropdown cell
            if rubric is not None:
                key.setdefault((language, verdict.factor_id), factor_key_row(rubric))
            sidecar.append(
                ColdSidecarRow(
                    sim_id=verdict.sim_id,
                    factor_id=verdict.factor_id,
                    language=language,
                    judge_outcome=verdict.outcome,
                    judge_reasoning=verdict.reasoning or "",
                    judge_quote=verdict.quote or "",
                )
            )
        rows.append(
            ColdSheetRow(
                transcript=verdicts[0].transcript[:max_trans_chars],
                sim_id=sim.id,
                task_id=str(sim.task_id),
                language=language,
                labels=labels,
            )
        )
    return ColdSheet(rows=rows, factors=list(key.values())), sidecar


def derive_precision_rows(
    sheet: ColdSheet, sidecar: list[ColdSidecarRow]
) -> list[JudgePrecisionRow]:
    """Precision rows from a finished cold pass — the judge's FAILs, no
    re-judging. Transcript pulled from the wide rows (by sim), the rubric from
    the factors key (by factor), joined to the sidecar's FAIL verdicts."""
    transcript_by_sim = {r.sim_id: r.transcript for r in sheet.rows}
    task_by_sim = {r.sim_id: r.task_id for r in sheet.rows}
    key_by_factor = {(k.language, k.factor_id): k for k in sheet.factors}
    rows: list[JudgePrecisionRow] = []
    for entry in sidecar:
        if entry.judge_outcome is not JudgeOutcome.FAIL:
            continue
        key = key_by_factor.get((entry.language, entry.factor_id))
        rows.append(
            JudgePrecisionRow(
                what_checked=key.nuance if key else "",
                quote=entry.judge_quote,
                why_flagged=entry.judge_reasoning,
                transcript=transcript_by_sim.get(entry.sim_id, ""),
                factor_id=entry.factor_id,
                sim_id=entry.sim_id,
                task_id=task_by_sim.get(entry.sim_id, ""),
                language=entry.language,
            )
        )
    return rows


def _family_provenance(
    results: list[str],
    sims_meta: dict,
    *,
    reuse_existing: bool,
    max_trans_chars: int,
) -> dict:
    return {
        "results": results,
        "reuse_existing": reuse_existing,
        "max_trans_chars": max_trans_chars,
        "judge_models": sorted(sims_meta.get("judge_models", set())),
        "languages": sorted(sims_meta.get("languages", set())),
    }


def export_nativeness_sheets(
    results: list[Path | str],
    mode: NativenessSheetMode,
    out_stem: Path,
    *,
    reuse_existing: bool = False,
    max_trans_chars: int = DEFAULT_ANNOTATION_MAX_TRANS_CHARS,
    concurrency: int = DEFAULT_JUDGE_STREAM_CONCURRENCY,
) -> Path:
    """Build one nativeness calibration sheet family; returns the manifest.

    Streams sims (large audio-bearing runs stay memory-bounded); the judge is
    re-invoked only when ``reuse_existing`` is off or stored verdicts are
    incomplete.
    """
    meta: dict = {"judge_models": set(), "languages": set()}

    def tracked() -> Iterator[SimulationRun]:
        for sim in stream_judged_sims(
            results, reuse_existing=reuse_existing, concurrency=concurrency
        ):
            info = sim.nativeness_info
            if info is not None:
                if info.judge_model:
                    meta["judge_models"].add(info.judge_model)
                if info.language:
                    meta["languages"].add(info.language)
            yield sim

    payloads: dict[str, SheetPayload]
    if mode == "precision":
        rows = build_precision_rows(tracked(), max_trans_chars)
        kind = "nativeness_precision"
        payloads = {"sheet": payload_from_rows(JudgePrecisionRow, rows)}
    else:
        sheet, sidecar = build_cold_sheet(tracked(), max_trans_chars)
        kind = "nativeness_cold"
        payloads = {
            "sheet": payload_from_cold_sheet(sheet),
            "judge_sidecar": payload_from_rows(ColdSidecarRow, sidecar, editable=False),
            "factors_key": payload_from_rows(
                FactorKeyRow, sheet.factors, editable=False
            ),
        }
        if mode == "both":
            payloads["precision"] = payload_from_rows(
                JudgePrecisionRow, derive_precision_rows(sheet, sidecar)
            )
    languages = meta["languages"]
    return write_sheet_family(
        Path(out_stem),
        kind=kind,
        payloads=payloads,
        provenance=_family_provenance(
            [str(r) for r in results],
            meta,
            reuse_existing=reuse_existing,
            max_trans_chars=max_trans_chars,
        ),
        language=next(iter(languages)) if len(languages) == 1 else None,
    )


def precision_rows_from_artifact(
    artifact: LoadedArtifact, role: Optional[str] = None
) -> list[JudgePrecisionRow]:
    """The filled precision rows of an artifact (either family kind)."""
    role = role or (
        "precision" if artifact.manifest.kind == "nativeness_cold" else "sheet"
    )
    return artifact.rows(role)


def cold_sheet_from_artifact(
    artifact: LoadedArtifact,
) -> tuple[ColdSheet, list[ColdSidecarRow]]:
    """The filled cold grid + verified judge sidecar of a cold artifact."""
    return artifact.cold_sheet(), artifact.rows("judge_sidecar")
