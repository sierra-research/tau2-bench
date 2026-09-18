# Copyright Sierra
"""Build the sparse repository overlay used by corrected paper analyses.

The overlay exposes the original ninety τ-Multilingual voice cells at their
canonical repository-relative paths.  Eighty paths point at the unchanged
paper corpus; the Korean and Mandarin retail paths point at the ten verified
trial-0 split-composition cells.  Fixed-name analysis sidecars are installed
alongside those links so the frozen Experience analysis can consume the
overlay as an ordinary repository root.

This command is deliberately read-only with respect to every input root.  It
creates a new sparse tree atomically and re-verifies an existing tree on an
idempotent rerun.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import Results, TaskSubsetInfo
from tau2.judges.corrected_paper_analysis import (
    DOMAINS,
    LANGUAGES,
    REPEATED_SYSTEM_SLUGS,
    SYSTEM_SLUGS,
    SYSTEMS,
    CorrectedAnalysisConfig,
    _canonical_results_path,
    _canonical_task_id,
    _load_final_window,
    _validate_system_metadata,
    _verified_composition,
)
from tau2.judges.corrected_paper_suite import (
    CANONICAL_GENERIC_COHORT,
    sha256_file,
)

OVERLAY_SCHEMA_VERSION = "tau-multi-corrected-analysis-overlay-v1"
OVERLAY_MANIFEST_FILENAME = "corrected_paper_analysis_overlay.json"
CANONICAL_MAIN_RUNS = Path("data/simulations/paper_runs/tau-multi/main_runs")
GENDER_SIDECARS = {
    language: Path(f"data/analysis/{language}_gender_v3_male_full_2026-09-03.csv")
    for language in ("pt", "hi")
}
FINAL_WINDOW_STABLE_CSV = Path(
    "data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.csv"
)
FINAL_WINDOW_STABLE_JSON = FINAL_WINDOW_STABLE_CSV.with_suffix(".json")
TaskSubsetProvenance = Literal["pinned", "legacy_null_verified_against_pinned_xai"]
LEGACY_NULL_TASK_SUBSET_KEYS = frozenset(
    ("en", "telecom", system_slug) for system_slug in REPEATED_SYSTEM_SLUGS
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sidecar_identity_sha256(csv_path: Path) -> str:
    """Hash a sidecar's CSV and adjacent typed manifest as one identity."""
    return _sha256_value(
        {
            "csv": sha256_file(csv_path),
            "manifest": sha256_file(csv_path.with_suffix(".json")),
        }
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorrectedAnalysisOverlayConfig(BaseModel):
    """Explicit immutable inputs for one corrected analysis repository."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_repo_root: Annotated[
        Path, Field(description="Repository containing the canonical paper corpus.")
    ]
    quality_workspace: Annotated[
        Path, Field(description="Verified 7+3 quality replay workspace.")
    ]
    delivery_workspace: Annotated[
        Path, Field(description="Verified frozen delivery replay workspace.")
    ]
    modal_workspace: Annotated[
        Path, Field(description="Verified Mandarin modal-particle workspace.")
    ]
    composed_workspace: Annotated[
        Path, Field(description="Existing verified split-composition workspace.")
    ]
    final_window_sidecar: Annotated[
        Path,
        Field(description="Hybrid final-window CSV with an adjacent typed JSON."),
    ]
    output_repo_root: Annotated[
        Path, Field(description="New sparse repository overlay root.")
    ]


class OverlayCellIdentity(BaseModel):
    """One source cell exposed at one canonical logical path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    domain: str
    system_slug: str
    source_kind: Literal["canonical", "corrected_composition"]
    source_cell_path: str
    source_results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    canonical_results_path: str
    canonical_results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    target_cell_relative_path: str
    trial_0_calls: Literal[50]
    source_trials: tuple[int, ...]
    task_frame_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    simulation_ids_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    nonjudge_hashes_verified: bool
    task_subset_provenance: Annotated[
        TaskSubsetProvenance,
        Field(
            description="Pinned subset metadata, or the closed legacy exception "
            "verified against the pinned English telecom xAI frame."
        ),
    ]

    @model_validator(mode="after")
    def _corrected_cells_are_nonjudge_verified(self) -> "OverlayCellIdentity":
        if self.source_kind == "corrected_composition" and not (
            self.nonjudge_hashes_verified
        ):
            raise ValueError("corrected cells require reproduced non-judge hashes")
        if (
            self.task_subset_provenance == "legacy_null_verified_against_pinned_xai"
            and (
                self.source_kind != "canonical"
                or (self.language, self.domain, self.system_slug)
                not in LEGACY_NULL_TASK_SUBSET_KEYS
            )
        ):
            raise ValueError(
                "legacy null subset provenance is restricted to the four "
                "canonical English telecom repeated-provider cells"
            )
        return self


class OverlayFileIdentity(BaseModel):
    """One copied analysis sidecar with source and installed byte identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal[
        "pt_gender",
        "hi_gender",
        "hybrid_final_window",
        "hybrid_final_window_manifest",
    ]
    source_path: str
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    target_relative_path: str
    installed_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rows: Annotated[int, Field(ge=0)] | None = None
    transformation: Literal["byte_copy", "stable_artifact_name"]


class CorrectedAnalysisOverlayManifest(BaseModel):
    """Complete reproducible identity for a sparse corrected analysis tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-analysis-overlay-v1"]
    created_at: str
    canonical_repo_root: str
    output_repo_root: str
    compose_report_path: str
    compose_report_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    composition_cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    cells: tuple[OverlayCellIdentity, ...]
    files: tuple[OverlayFileIdentity, ...]
    cells_total: Literal[90]
    canonical_cells: Literal[80]
    corrected_cells: Literal[10]
    trial_0_calls: Literal[4500]
    corrected_nonjudge_verified_calls: Literal[500]
    legacy_null_task_subset_cells: Annotated[
        tuple[str, ...],
        Field(
            description="Exact canonical cells whose legacy headers omit subset "
            "metadata and whose task frame was verified against pinned xAI."
        ),
    ]
    cohort_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _closed_inventory(self) -> "CorrectedAnalysisOverlayManifest":
        if len(self.cells) != self.cells_total:
            raise ValueError("overlay manifest does not contain ninety cells")
        canonical = sum(row.source_kind == "canonical" for row in self.cells)
        corrected = sum(
            row.source_kind == "corrected_composition" for row in self.cells
        )
        if (canonical, corrected) != (self.canonical_cells, self.corrected_cells):
            raise ValueError("overlay source-kind counts do not close")
        keys = {(row.language, row.domain, row.system_slug) for row in self.cells}
        targets = {row.target_cell_relative_path for row in self.cells}
        if len(keys) != self.cells_total or len(targets) != self.cells_total:
            raise ValueError("overlay contains duplicate logical cells or targets")
        if sum(row.trial_0_calls for row in self.cells) != self.trial_0_calls:
            raise ValueError("overlay trial-0 call count does not close")
        observed_legacy = tuple(
            sorted(
                f"{row.language}/{row.domain}/{row.system_slug}"
                for row in self.cells
                if row.task_subset_provenance
                == "legacy_null_verified_against_pinned_xai"
            )
        )
        expected_legacy = tuple(
            sorted("/".join(key) for key in LEGACY_NULL_TASK_SUBSET_KEYS)
        )
        if (
            observed_legacy != expected_legacy
            or self.legacy_null_task_subset_cells != expected_legacy
        ):
            raise ValueError("overlay legacy null task-subset inventory drifted")
        if {row.role for row in self.files} != {
            "pt_gender",
            "hi_gender",
            "hybrid_final_window",
            "hybrid_final_window_manifest",
        }:
            raise ValueError("overlay sidecar inventory is incomplete")
        return self


@dataclass(frozen=True)
class _OverlayPlan:
    manifest: CorrectedAnalysisOverlayManifest
    links: tuple[tuple[Path, Path], ...]
    files: tuple[tuple[Path, bytes], ...]


@dataclass(frozen=True)
class _CellEvidence:
    trials: tuple[int, ...]
    trial_0_ids: frozenset[str]
    trial_0_task_ids: frozenset[str]
    trial_0_tasks: frozenset[str]
    task_frame_sha256: str
    simulation_ids_sha256: str
    task_subset_provenance: TaskSubsetProvenance


def _safe_relative_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe overlay-relative path: {path}")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_output_root(config: CorrectedAnalysisOverlayConfig) -> tuple[Path, ...]:
    inputs = tuple(
        path.expanduser().resolve()
        for path in (
            config.canonical_repo_root,
            config.quality_workspace,
            config.delivery_workspace,
            config.modal_workspace,
            config.composed_workspace,
        )
    )
    output = config.output_repo_root.expanduser().resolve()
    for source in inputs:
        if _paths_overlap(output, source):
            raise ValueError(
                f"overlay output must not overlap an input root: {output} / {source}"
            )
    final_window = config.final_window_sidecar.expanduser().resolve()
    if _paths_overlap(output, final_window):
        raise ValueError("overlay output must not contain its final-window input")
    return inputs


def _task_subset_provenance(
    subset: TaskSubsetInfo | None,
    *,
    path: Path,
    language: str,
    domain: str,
    system_slug: str,
    corrected: bool,
    canonical_results_path: Path,
) -> TaskSubsetProvenance:
    if subset is not None:
        if (
            subset.name != f"{domain}_50"
            or subset.size != 50
            or subset.tasks_scored != 50
        ):
            raise ValueError(f"cell task-subset metadata drifted: {path}")
        return "pinned"
    legacy_key = (language, domain, system_slug)
    if (
        corrected
        or legacy_key not in LEGACY_NULL_TASK_SUBSET_KEYS
        or path.resolve() != canonical_results_path.resolve()
    ):
        raise ValueError(f"cell task-subset metadata drifted: {path}")
    return "legacy_null_verified_against_pinned_xai"


def _validate_legacy_null_frame(
    evidence: _CellEvidence,
    *,
    pinned_xai_task_ids: frozenset[str],
    path: Path,
) -> None:
    if (
        evidence.task_subset_provenance == "legacy_null_verified_against_pinned_xai"
        and evidence.trial_0_task_ids != pinned_xai_task_ids
    ):
        raise ValueError(
            f"legacy English telecom frame differs from pinned xAI: {path}"
        )


def _validate_cell(
    path: Path,
    *,
    language: str,
    domain: str,
    system_slug: str,
    corrected: bool,
    canonical_results_path: Path,
) -> _CellEvidence:
    if not path.is_file():
        raise ValueError(f"results file is missing: {path}")
    results = Results.load_metadata(path)
    entries = results.simulation_index
    if entries is None:
        raise ValueError(f"results file has no simulation index: {path}")
    expected_trials = (
        {0} if corrected or system_slug not in REPEATED_SYSTEM_SLUGS else {0, 1}
    )
    trials = {entry.trial for entry in entries}
    if trials != expected_trials:
        raise ValueError(f"cell trial inventory drifted at {path}: {sorted(trials)}")
    for trial in expected_trials:
        if sum(entry.trial == trial for entry in entries) != 50:
            raise ValueError(
                f"cell does not contain 50 calls for trial {trial}: {path}"
            )
    if any(entry.reward not in {0.0, 1.0} for entry in entries):
        raise ValueError(f"cell has a non-binary or missing reward: {path}")
    _validate_system_metadata(results, system_slug, path)
    if results.info.environment_info.domain_name != domain:
        raise ValueError(f"cell domain metadata drifted: {path}")
    subset_provenance = _task_subset_provenance(
        results.info.task_subset,
        path=path,
        language=language,
        domain=domain,
        system_slug=system_slug,
        corrected=corrected,
        canonical_results_path=canonical_results_path,
    )
    trial_0 = [entry for entry in entries if entry.trial == 0]
    ids = [entry.id for entry in trial_0]
    tasks = [str(entry.task_id) for entry in trial_0]
    normalized_tasks = [_canonical_task_id(task, language) for task in tasks]
    if len(set(ids)) != 50 or len(set(normalized_tasks)) != 50:
        raise ValueError(f"cell identity or task inventory is not unique: {path}")
    for entry in trial_0:
        simulation = path.parent / "simulations" / f"{entry.id}.json"
        if not simulation.is_file():
            raise ValueError(f"indexed simulation is missing: {simulation}")
    return _CellEvidence(
        trials=tuple(sorted(trials)),
        trial_0_ids=frozenset(ids),
        trial_0_task_ids=frozenset(tasks),
        trial_0_tasks=frozenset(normalized_tasks),
        task_frame_sha256=_sha256_value(sorted(normalized_tasks)),
        simulation_ids_sha256=_sha256_value(sorted(ids)),
        task_subset_provenance=subset_provenance,
    )


def _gender_file(
    path: Path,
    *,
    language: str,
    expected_ids: set[str],
) -> tuple[bytes, int]:
    if not path.is_file():
        raise ValueError(f"required {language} gender sidecar is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row.get("sim_id") or "") for row in rows]
    if len(rows) != 750 or len(set(ids)) != len(ids):
        raise ValueError(f"{language} gender sidecar inventory drifted: {path}")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))[:3]
        extra = sorted(set(ids) - expected_ids)[:3]
        raise ValueError(
            f"{language} gender sidecar does not match the overlay: "
            f"missing={missing}, extra={extra}"
        )
    if any(
        row.get("outcome") not in {"pass", "fail", "no_opportunity"} for row in rows
    ):
        raise ValueError(f"{language} gender sidecar has an invalid outcome")
    return path.read_bytes(), len(rows)


def _json_artifact_bytes(value: BaseModel) -> bytes:
    return (value.model_dump_json(indent=2) + "\n").encode()


def _plan_overlay(
    config: CorrectedAnalysisOverlayConfig,
    *,
    created_at: str,
) -> _OverlayPlan:
    canonical_repo, quality, delivery, modal, composed = _validate_output_root(config)
    output = config.output_repo_root.expanduser().resolve()
    analysis_config = CorrectedAnalysisConfig(
        quality_workspace=quality,
        delivery_workspace=delivery,
        modal_workspace=modal,
        composed_workspace=composed,
        output=output / OVERLAY_MANIFEST_FILENAME,
    )
    report, judge_manifest, composition = _verified_composition(analysis_config)
    if judge_manifest.cohort != CANONICAL_GENERIC_COHORT:
        raise ValueError("composition does not use the exact ten-cell paper contract")
    if (
        report.cells != 10
        or report.calls != 500
        or report.nonjudge_hash_mismatches != 0
        or judge_manifest.counts.infrastructure_errors != 0
    ):
        raise ValueError("composition count or non-judge provenance drifted")
    corrected_by_key = {
        (cell.language, cell.system_slug): cell for cell in judge_manifest.cells
    }
    if len(corrected_by_key) != 10:
        raise ValueError("composition manifest does not contain ten unique cells")

    canonical_main = canonical_repo / CANONICAL_MAIN_RUNS
    if not canonical_main.is_dir():
        raise ValueError(f"canonical main-runs root is missing: {canonical_main}")
    xai_system = next(
        system for system in SYSTEMS if SYSTEM_SLUGS[system] == "xai_provider_default"
    )
    pinned_xai_results = _canonical_results_path(
        canonical_main, "en", "telecom", xai_system
    )
    pinned_xai_evidence = _validate_cell(
        pinned_xai_results,
        language="en",
        domain="telecom",
        system_slug="xai_provider_default",
        corrected=False,
        canonical_results_path=pinned_xai_results,
    )
    if pinned_xai_evidence.task_subset_provenance != "pinned":
        raise ValueError("English telecom xAI task frame is not pinned")
    pinned_xai_task_ids = pinned_xai_evidence.trial_0_task_ids
    links: list[tuple[Path, Path]] = []
    cells: list[OverlayCellIdentity] = []
    frames: dict[tuple[str, str], frozenset[str]] = {}
    output_simulation_ids: set[str] = set()
    ids_by_language: dict[str, set[str]] = defaultdict(set)
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in SYSTEMS:
                system_slug = SYSTEM_SLUGS[system]
                canonical_results = _canonical_results_path(
                    canonical_main, language, domain, system
                )
                canonical_evidence = _validate_cell(
                    canonical_results,
                    language=language,
                    domain=domain,
                    system_slug=system_slug,
                    corrected=False,
                    canonical_results_path=canonical_results,
                )
                _validate_legacy_null_frame(
                    canonical_evidence,
                    pinned_xai_task_ids=pinned_xai_task_ids,
                    path=canonical_results,
                )
                corrected_cell = corrected_by_key.get((language, system_slug))
                use_corrected = domain == "retail" and corrected_cell is not None
                if corrected_cell is not None and domain != "retail":
                    use_corrected = False
                if use_corrected:
                    source_results = (
                        Path(report.merged_root)
                        / corrected_cell.source_relative_path
                        / "results.json"
                    ).resolve()
                    evidence = _validate_cell(
                        source_results,
                        language=language,
                        domain=domain,
                        system_slug=system_slug,
                        corrected=True,
                        canonical_results_path=canonical_results,
                    )
                    locked_ids = {call.simulation_id for call in corrected_cell.calls}
                    if evidence.trial_0_ids != locked_ids:
                        raise ValueError(
                            f"corrected composition identities drifted: {source_results}"
                        )
                    source_kind: Literal["canonical", "corrected_composition"] = (
                        "corrected_composition"
                    )
                    nonjudge_verified = True
                else:
                    source_results = canonical_results
                    evidence = canonical_evidence
                    source_kind = "canonical"
                    nonjudge_verified = False
                frame_key = (language, domain)
                expected_frame = frames.setdefault(frame_key, evidence.trial_0_tasks)
                if evidence.trial_0_tasks != expected_frame:
                    raise ValueError(
                        f"systems do not share one task frame: {language}/{domain}"
                    )
                overlap = output_simulation_ids & set(evidence.trial_0_ids)
                if overlap:
                    raise ValueError(
                        f"duplicate trial-0 simulation ids across cells: {sorted(overlap)[:3]}"
                    )
                output_simulation_ids.update(evidence.trial_0_ids)
                ids_by_language[language].update(evidence.trial_0_ids)
                target_cell = canonical_results.parent.relative_to(canonical_repo)
                _safe_relative_path(target_cell)
                links.append((target_cell, source_results.parent))
                cells.append(
                    OverlayCellIdentity(
                        language=language,
                        domain=domain,
                        system_slug=system_slug,
                        source_kind=source_kind,
                        source_cell_path=str(source_results.parent),
                        source_results_sha256=sha256_file(source_results),
                        canonical_results_path=str(canonical_results),
                        canonical_results_sha256=sha256_file(canonical_results),
                        target_cell_relative_path=target_cell.as_posix(),
                        trial_0_calls=50,
                        source_trials=evidence.trials,
                        task_frame_sha256=evidence.task_frame_sha256,
                        simulation_ids_sha256=evidence.simulation_ids_sha256,
                        nonjudge_hashes_verified=nonjudge_verified,
                        task_subset_provenance=evidence.task_subset_provenance,
                    )
                )
    if len(cells) != 90 or len(output_simulation_ids) != 4_500:
        raise ValueError("overlay does not contain the exact 90 x 50 paper cohort")
    if sum(row.source_kind == "corrected_composition" for row in cells) != 10:
        raise ValueError("overlay does not contain exactly ten corrected cells")

    files: list[tuple[Path, bytes]] = []
    file_identities: list[OverlayFileIdentity] = []
    for language, relative in GENDER_SIDECARS.items():
        source = (canonical_repo / relative).resolve()
        content, rows = _gender_file(
            source, language=language, expected_ids=ids_by_language[language]
        )
        files.append((relative, content))
        file_identities.append(
            OverlayFileIdentity(
                role=f"{language}_gender",
                source_path=str(source),
                source_sha256=sha256_file(source),
                target_relative_path=relative.as_posix(),
                installed_sha256=_sha256_bytes(content),
                rows=rows,
                transformation="byte_copy",
            )
        )

    final_source = config.final_window_sidecar.expanduser().resolve()
    final_rows, final_manifest, _final_by_key = _load_final_window(final_source)
    if (
        final_manifest.replacement_cohort_fingerprint_sha256
        != report.cohort_fingerprint_sha256
        or final_manifest.replacement_calls != report.calls
        or final_manifest.canonical_sidecar_sha256 is None
    ):
        raise ValueError("final-window sidecar is not the hybrid for this composition")
    canonical_final = (canonical_repo / FINAL_WINDOW_STABLE_CSV).resolve()
    if (
        _sidecar_identity_sha256(canonical_final)
        != final_manifest.canonical_sidecar_sha256
    ):
        raise ValueError("hybrid final-window sidecar names another canonical source")
    unknown_final_ids = {row.sim_id for row in final_rows} - output_simulation_ids
    if unknown_final_ids:
        raise ValueError(
            "hybrid final-window sidecar contains calls outside the overlay: "
            f"{sorted(unknown_final_ids)[:3]}"
        )
    final_csv_bytes = final_source.read_bytes()
    installed_final_manifest = final_manifest.model_copy(
        update={"artifact": FINAL_WINDOW_STABLE_CSV.name}
    )
    final_json_bytes = _json_artifact_bytes(installed_final_manifest)
    files.extend(
        (
            (FINAL_WINDOW_STABLE_CSV, final_csv_bytes),
            (FINAL_WINDOW_STABLE_JSON, final_json_bytes),
        )
    )
    final_manifest_source = final_source.with_suffix(".json")
    file_identities.extend(
        (
            OverlayFileIdentity(
                role="hybrid_final_window",
                source_path=str(final_source),
                source_sha256=sha256_file(final_source),
                target_relative_path=FINAL_WINDOW_STABLE_CSV.as_posix(),
                installed_sha256=_sha256_bytes(final_csv_bytes),
                rows=len(final_rows),
                transformation="byte_copy",
            ),
            OverlayFileIdentity(
                role="hybrid_final_window_manifest",
                source_path=str(final_manifest_source),
                source_sha256=sha256_file(final_manifest_source),
                target_relative_path=FINAL_WINDOW_STABLE_JSON.as_posix(),
                installed_sha256=_sha256_bytes(final_json_bytes),
                rows=len(final_rows),
                transformation="stable_artifact_name",
            ),
        )
    )
    cells.sort(key=lambda row: (row.language, row.domain, row.system_slug))
    file_identities.sort(key=lambda row: row.role)
    cohort_payload = {
        "composition": composition.model_dump(mode="json"),
        "cells": [row.model_dump(mode="json") for row in cells],
        "files": [row.model_dump(mode="json") for row in file_identities],
    }
    manifest = CorrectedAnalysisOverlayManifest(
        schema_version=OVERLAY_SCHEMA_VERSION,
        created_at=created_at,
        canonical_repo_root=str(canonical_repo),
        output_repo_root=str(output),
        compose_report_path=composition.compose_report_path,
        compose_report_sha256=composition.compose_report_sha256,
        composition_cohort_fingerprint_sha256=(composition.cohort_fingerprint_sha256),
        cells=tuple(cells),
        files=tuple(file_identities),
        cells_total=90,
        canonical_cells=80,
        corrected_cells=10,
        trial_0_calls=4_500,
        corrected_nonjudge_verified_calls=500,
        legacy_null_task_subset_cells=tuple(
            sorted("/".join(key) for key in LEGACY_NULL_TASK_SUBSET_KEYS)
        ),
        cohort_fingerprint_sha256=_sha256_value(cohort_payload),
    )
    return _OverlayPlan(
        manifest=manifest,
        links=tuple(sorted(links, key=lambda row: row[0].as_posix())),
        files=tuple(sorted(files, key=lambda row: row[0].as_posix())),
    )


def _write_plan(root: Path, plan: _OverlayPlan) -> None:
    for relative, source in plan.links:
        target = root / _safe_relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    for relative, content in plan.files:
        target = root / _safe_relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / OVERLAY_MANIFEST_FILENAME).write_bytes(_json_artifact_bytes(plan.manifest))


def _tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    links: set[str] = set()
    files: set[str] = set()
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                links.add(path.relative_to(root).as_posix())
            else:
                retained.append(name)
        directories[:] = retained
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                links.add(relative)
            else:
                files.add(relative)
    return links, files


def _verify_plan(root: Path, plan: _OverlayPlan) -> None:
    expected_links = {relative.as_posix() for relative, _source in plan.links}
    expected_files = {
        *(relative.as_posix() for relative, _content in plan.files),
        OVERLAY_MANIFEST_FILENAME,
    }
    observed_links, observed_files = _tree_inventory(root)
    if observed_links != expected_links or observed_files != expected_files:
        raise ValueError(
            "analysis overlay inventory drifted: "
            f"links={len(observed_links)}/{len(expected_links)}, "
            f"files={len(observed_files)}/{len(expected_files)}"
        )
    for relative, source in plan.links:
        target = root / relative
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError(f"analysis overlay link drifted: {target}")
    for relative, content in plan.files:
        target = root / relative
        if not target.is_file() or target.read_bytes() != content:
            raise ValueError(f"analysis overlay sidecar drifted: {target}")
    manifest_path = root / OVERLAY_MANIFEST_FILENAME
    expected_manifest = _json_artifact_bytes(plan.manifest)
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError("analysis overlay manifest bytes drifted")
    observed = CorrectedAnalysisOverlayManifest.model_validate_json(
        manifest_path.read_text()
    )
    if observed != plan.manifest:
        raise ValueError("analysis overlay manifest does not reproduce")


def build_corrected_analysis_overlay(
    config: CorrectedAnalysisOverlayConfig,
) -> CorrectedAnalysisOverlayManifest:
    """Build or fully re-verify the sparse corrected analysis repository."""
    output = config.output_repo_root.expanduser().resolve()
    existing: CorrectedAnalysisOverlayManifest | None = None
    if output.exists():
        manifest_path = output / OVERLAY_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError("analysis overlay exists without its typed manifest")
        existing = CorrectedAnalysisOverlayManifest.model_validate_json(
            manifest_path.read_text()
        )
    plan = _plan_overlay(
        config,
        created_at=existing.created_at if existing is not None else _now(),
    )
    if existing is not None:
        if existing != plan.manifest:
            raise ValueError("existing analysis overlay was built from other inputs")
        _verify_plan(output, plan)
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_plan(temporary, plan)
        _verify_plan(temporary, plan)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_plan(output, plan)
    return plan.manifest


__all__ = [
    "CorrectedAnalysisOverlayConfig",
    "CorrectedAnalysisOverlayManifest",
    "OVERLAY_MANIFEST_FILENAME",
    "OverlayCellIdentity",
    "OverlayFileIdentity",
    "build_corrected_analysis_overlay",
]
