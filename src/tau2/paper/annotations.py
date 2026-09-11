# Copyright Sierra
"""Reviewer-safe export of final τ-Multilingual annotation results."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.annotation.metrics import ColdCounts, accumulate_cold_cell, cold_metrics
from tau2.annotation.models import ColdLabel
from tau2.data_model.simulation import JudgeOutcome

ANNOTATION_EXPORT_VERSION = "tau-multilingual-annotations-v1"
ANNOTATION_LANGUAGES = ("es", "pt", "hi", "ko", "zh")


class AnnotationCounts(BaseModel):
    """Confusion counts retained in a final recall file."""

    model_config = ConfigDict(frozen=True)

    tp: Annotated[int, Field(description="True-positive count.")]
    fn: Annotated[int, Field(description="False-negative count.")]
    fp_opp: Annotated[int, Field(description="False positives with opportunity.")]
    tn: Annotated[int, Field(description="True-negative count.")]
    fa_noopp: Annotated[
        int, Field(description="Positive predictions without human opportunity.")
    ]
    noopp_opp: Annotated[
        int, Field(description="Human labels without a judge-scored opportunity.")
    ]
    unsure: Annotated[int, Field(description="Uncertain human labels.")]
    unlabeled: Annotated[int, Field(description="Rows without a final human label.")]
    judge_unscored: Annotated[
        int, Field(description="Rows for which the judge emitted no score.")
    ]


class AnnotationMetric(BaseModel):
    """Final per-factor precision or recall evidence."""

    model_config = ConfigDict(frozen=True)

    judge: Annotated[str, Field(description="Judge family.")]
    factor_id: Annotated[str, Field(description="Stable evaluated factor identifier.")]
    precision: Annotated[
        Optional[float], Field(description="Final precision, when estimable.")
    ] = None
    recall: Annotated[
        Optional[float], Field(description="Final recall, when estimable.")
    ] = None
    f1: Annotated[Optional[float], Field(description="Final F1, when estimable.")] = (
        None
    )
    n_opportunity: Annotated[
        Optional[int], Field(description="Number of recall opportunities.")
    ] = None
    n_confirmed: Annotated[
        Optional[int], Field(description="Confirmed positive predictions.")
    ] = None
    n_rejected: Annotated[
        Optional[int], Field(description="Rejected positive predictions.")
    ] = None
    counts: Annotated[
        Optional[AnnotationCounts], Field(description="Recall confusion counts.")
    ] = None


class AnnotationResultFile(BaseModel):
    """One language's final annotation-derived result file."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        int, Field(description="Serialized result schema version.")
    ] = 1
    export_version: Annotated[
        str, Field(description="Versioned normalization contract.")
    ] = ANNOTATION_EXPORT_VERSION
    annotation_type: Annotated[
        Literal["precision", "recall"], Field(description="Annotation result type.")
    ]
    language: Annotated[str, Field(description="Language code.")]
    n_calls: Annotated[int, Field(description="Number of annotated calls.")]
    judge_versions: Annotated[
        dict[str, list[str]], Field(description="Judge model and prompt versions.")
    ]
    source_sha256: Annotated[
        str, Field(description="SHA-256 of the archived source report.")
    ]
    metrics: Annotated[
        list[AnnotationMetric], Field(description="Final per-factor results.")
    ]


class AnnotationExportEntry(BaseModel):
    """One file in the portable annotation export."""

    model_config = ConfigDict(frozen=True)

    annotation_type: Annotated[
        Literal["precision", "recall"], Field(description="Annotation result type.")
    ]
    language: Annotated[str, Field(description="Language code.")]
    file: Annotated[str, Field(description="Path relative to the export directory.")]
    sha256: Annotated[str, Field(description="SHA-256 of the exported file.")]
    metrics: Annotated[int, Field(description="Number of per-factor metric rows.")]


class AnnotationExportManifest(BaseModel):
    """Portable annotation export inventory."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        int, Field(description="Serialized manifest schema version.")
    ] = 1
    export_version: Annotated[
        str, Field(description="Versioned normalization contract.")
    ] = ANNOTATION_EXPORT_VERSION
    files: Annotated[
        list[AnnotationExportEntry], Field(description="Exported annotation files.")
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_simulations(root: Path, language: str) -> dict[str, dict]:
    simulations: dict[str, dict] = {}
    for path in sorted((root / language).glob("*/simulations/*.json")):
        payload = json.loads(path.read_text())
        sim_id = str(payload["id"])
        if sim_id in simulations:
            raise ValueError(f"duplicate simulation id {sim_id} under {root}")
        simulations[sim_id] = payload
    return simulations


def _conversation_calls(root: Path, language: str) -> dict[str, dict]:
    payload = json.loads(
        (root / language / f"conversation_{language}.json").read_text()
    )
    return {str(call["sim_id"]): call for call in payload["calls"]}


def _factor_outcome(
    judge: str, factor_id: str, sim: dict, conversation: Optional[dict]
) -> Optional[JudgeOutcome]:
    if judge == "nativeness":
        checks = (sim.get("nativeness_info") or {}).get("factor_checks", [])
    elif judge == "quality":
        quality = sim.get("quality_info")
        if quality is None and conversation is not None:
            quality = (conversation.get("shadow_scores") or {}).get("ours")
        checks = (quality or {}).get("factor_checks", [])
    elif judge == "delivery":
        delivery = sim.get("delivery_info") or {}
        if factor_id in {"fidelity", "intonation"}:
            failed = any(
                finding.get("axis") == factor_id
                for utterance in delivery.get("utterance_results", [])
                for finding in utterance.get("findings", [])
            )
            return JudgeOutcome.FAIL if failed else JudgeOutcome.PASS
        checks = delivery.get("factor_checks", [])
    elif judge == "semantic" and conversation is not None:
        if factor_id == "conciseness":
            turns = conversation.get("conciseness_turns", [])
            if not turns:
                return JudgeOutcome.NO_OPPORTUNITY
            score = sum((int(turn["rating"]) - 1) / 2 for turn in turns) / len(turns)
            return JudgeOutcome.FAIL if score < 0.5 else JudgeOutcome.PASS
        for dimension in conversation.get("progression_dimensions", []):
            if dimension.get("name") == factor_id:
                return (
                    JudgeOutcome.FAIL if dimension.get("flagged") else JudgeOutcome.PASS
                )
        return None
    else:
        return None
    for check in checks:
        if check.get("id") == factor_id:
            return JudgeOutcome(check["outcome"])
    return None


def _final_metrics(
    labels_path: Path, evidence_root: Path
) -> dict[str, list[AnnotationMetric]]:
    payload = json.loads(labels_path.read_text())
    records = payload["records"]
    by_language: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["lang"] in ANNOTATION_LANGUAGES:
            by_language[record["lang"]].append(record)

    result: dict[str, list[AnnotationMetric]] = {}
    label_map = {
        "yes": ColdLabel.YES,
        "no": ColdLabel.NO,
        "no_opportunity": ColdLabel.NO_OPPORTUNITY,
        "unsure": ColdLabel.UNSURE,
    }
    for language in ANNOTATION_LANGUAGES:
        simulations = _find_simulations(evidence_root, language)
        conversations = _conversation_calls(evidence_root, language)
        counts_by_factor: dict[tuple[str, str], ColdCounts] = {}
        for record in by_language[language]:
            sim_id = str(record["sim_id"])
            sim = simulations.get(sim_id)
            if sim is None:
                raise ValueError(f"{language}: no frozen simulation for {sim_id}")
            key = (str(record["judge"]), str(record["factor_id"]))
            counts = counts_by_factor.setdefault(key, ColdCounts())
            accumulate_cold_cell(
                counts,
                label_map[str(record["label"])],
                _factor_outcome(key[0], key[1], sim, conversations.get(sim_id)),
            )
        result[language] = []
        for (judge, factor_id), counts in sorted(counts_by_factor.items()):
            metric = cold_metrics(counts)
            result[language].append(
                AnnotationMetric(
                    judge=judge,
                    factor_id=factor_id,
                    precision=metric.precision,
                    recall=metric.recall,
                    f1=metric.f1,
                    n_opportunity=metric.n_opportunity,
                    n_confirmed=counts.tp,
                    n_rejected=counts.fp_opp + counts.fa_noopp,
                    counts=AnnotationCounts.model_validate(counts.model_dump()),
                )
            )
    return result


def export_annotations(
    labels: Path, evidence_root: Path, out: Path
) -> AnnotationExportManifest:
    """Write normalized final precision and recall results from final labels."""
    labels = labels.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics_by_language = _final_metrics(labels, evidence_root)
    entries: list[AnnotationExportEntry] = []
    for annotation_type in ("precision", "recall"):
        for language in ANNOTATION_LANGUAGES:
            source_metrics = metrics_by_language[language]
            metrics = []
            for metric in source_metrics:
                if annotation_type == "precision":
                    metrics.append(
                        AnnotationMetric(
                            judge=metric.judge,
                            factor_id=metric.factor_id,
                            precision=metric.precision,
                            n_confirmed=metric.n_confirmed,
                            n_rejected=metric.n_rejected,
                        )
                    )
                else:
                    metrics.append(
                        AnnotationMetric(
                            judge=metric.judge,
                            factor_id=metric.factor_id,
                            recall=metric.recall,
                            f1=metric.f1,
                            n_opportunity=metric.n_opportunity,
                            counts=metric.counts,
                        )
                    )
            result = AnnotationResultFile(
                annotation_type=annotation_type,
                language=language,
                n_calls=len(
                    {
                        record["sim_id"]
                        for record in json.loads(labels.read_text())["records"]
                        if record["lang"] == language
                    }
                ),
                judge_versions={},
                source_sha256=_sha256(labels),
                metrics=metrics,
            )
            relative = Path(annotation_type) / f"{language}.json"
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(result.model_dump_json(indent=2) + "\n")
            entries.append(
                AnnotationExportEntry(
                    annotation_type=annotation_type,
                    language=language,
                    file=relative.as_posix(),
                    sha256=_sha256(destination),
                    metrics=len(result.metrics),
                )
            )
    manifest = AnnotationExportManifest(files=entries)
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest
