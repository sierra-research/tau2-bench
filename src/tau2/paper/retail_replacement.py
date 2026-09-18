# Copyright Sierra
"""Typed contract for the corrected Korean/Mandarin retail release swap."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ArtifactReference(BaseModel):
    """One stable repository-relative artifact reference."""

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: Sha256


class AuditReference(ArtifactReference):
    """A frozen multilingual audit and its short artifact identifier."""

    artifact_id: str


class ResultSnapshot(BaseModel):
    """Identity and outcome fields needed to archive or install one cell."""

    model_config = ConfigDict(frozen=True)

    results_sha256: Sha256
    calls: Annotated[int, Field(ge=1)]
    trials: tuple[Annotated[int, Field(ge=0)], ...]
    trial_success: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    task_ids_sha256: Sha256
    run_git_commit: str


class PromptTreatmentEvidence(BaseModel):
    """Backfilled proof that every corrected caller saw the v1 treatment."""

    model_config = ConfigDict(frozen=True)

    nonprompt_payload_sha256: Sha256
    simulation_payload_sha256: Sha256
    prompt_evidence_sha256: Sha256
    template_sha256: Sha256
    calls_verified: Annotated[int, Field(ge=1)]
    prompt_requests_verified: Annotated[int, Field(ge=1)]


class ReplacementCell(BaseModel):
    """Old archive entry and corrected replacement for one active result cell."""

    model_config = ConfigDict(frozen=True)

    modality: Literal["voice", "text"]
    language: Literal["ko", "zh"]
    system: Literal[
        "openai_minimal",
        "openai_xhigh",
        "gemini_minimal",
        "gemini_high",
        "xai_provider_default",
        "gpt55_xhigh",
        "gemini31pro_high",
    ]
    canonical_path: str
    archive_path: str
    staged_source_path: str
    provider: str
    model: str
    reasoning_effort: str
    previous: ResultSnapshot
    replacement: ResultSnapshot
    prompt_treatment: PromptTreatmentEvidence

    @model_validator(mode="after")
    def validate_paths_and_counts(self) -> "ReplacementCell":
        for label, value in (
            ("canonical_path", self.canonical_path),
            ("archive_path", self.archive_path),
            ("staged_source_path", self.staged_source_path),
        ):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be a safe relative path")
        expected_archive = f"validation_runs/{self.language}/{self.canonical_path}"
        if self.archive_path != expected_archive:
            raise ValueError("archive path must preserve the canonical relative path")
        if not self.canonical_path.endswith("/results.json"):
            raise ValueError("canonical path must identify results.json")
        if not self.staged_source_path.startswith("data/simulations/"):
            raise ValueError(
                "staged source must use a repository-relative logical path"
            )
        if self.previous.calls != self.replacement.calls:
            raise ValueError("replacement must preserve the indexed call count")
        if self.previous.trials != self.replacement.trials:
            raise ValueError("replacement must preserve the trial frame")
        if self.previous.task_ids_sha256 != self.replacement.task_ids_sha256:
            raise ValueError("replacement must preserve the task frame")
        if self.replacement.calls != self.prompt_treatment.calls_verified:
            raise ValueError("every replacement call must have prompt-treatment proof")
        if self.replacement.calls != self.prompt_treatment.prompt_requests_verified:
            raise ValueError("every replacement call must have one verified prompt")
        return self


class CohortFingerprints(BaseModel):
    """Before/after cohort digests and the unchanged-cell proof."""

    model_config = ConfigDict(frozen=True)

    previous_repeated_voice: Sha256
    replacement_repeated_voice: Sha256
    previous_xai_voice: Sha256
    replacement_xai_voice: Sha256
    previous_text: Sha256
    replacement_text: Sha256
    previous_replacement_cells: Sha256
    replacement_cells: Sha256
    unchanged_cells: Sha256
    previous_full_cohort: Sha256
    replacement_full_cohort: Sha256


class ExclusionSummary(BaseModel):
    """Raw artifacts intentionally omitted from the canonical release cells."""

    model_config = ConfigDict(frozen=True)

    hallucination_discard_records: Annotated[int, Field(ge=0)]
    unindexed_artifact_directories: Annotated[int, Field(ge=0)]
    orphan_simulation_files: Annotated[int, Field(ge=0)]
    included_infrastructure_failures: Annotated[int, Field(ge=0)]


class AblationAttemptExclusions(BaseModel):
    """Attempt artifacts excluded from one corrected ablation result cell."""

    model_config = ConfigDict(frozen=True)

    hallucination_discard_records: Annotated[
        int,
        Field(
            ge=0,
            description="Retry attempts rejected by the caller-hallucination guard.",
        ),
    ]
    infrastructure_error_artifacts: Annotated[
        int,
        Field(ge=0, description="Failed retry attempts caused by infrastructure."),
    ]
    superseded_complete_artifacts: Annotated[
        int,
        Field(ge=0, description="Complete retry attempts superseded by the index."),
    ]
    incomplete_retry_artifacts: Annotated[
        int,
        Field(ge=0, description="Interrupted retry attempts with no final status."),
    ]
    unindexed_artifact_directories: Annotated[
        int,
        Field(ge=0, description="All attempt directories absent from the index."),
    ]
    orphan_simulation_files: Annotated[
        int,
        Field(ge=0, description="Simulation JSON files absent from the index."),
    ]
    included_infrastructure_failures: Annotated[
        int,
        Field(ge=0, description="Infrastructure failures retained in scored calls."),
    ]

    @model_validator(mode="after")
    def validate_attempt_partition(self) -> "AblationAttemptExclusions":
        classified = (
            self.hallucination_discard_records
            + self.infrastructure_error_artifacts
            + self.superseded_complete_artifacts
            + self.incomplete_retry_artifacts
        )
        if classified != self.unindexed_artifact_directories:
            raise ValueError("unindexed attempt classifications must be exhaustive")
        return self


class PromptArchiveReference(ArtifactReference):
    """Integrity summary for the frozen prompt snapshot archive."""

    artifact_id: Annotated[
        Sha256, Field(description="Content-derived prompt archive identifier.")
    ]
    cells: Annotated[int, Field(ge=1, description="Result cells in the archive.")]
    simulations: Annotated[
        int, Field(ge=1, description="Indexed simulations in the archive.")
    ]
    objects: Annotated[
        int, Field(ge=1, description="Referenced content-addressed objects.")
    ]
    retained_cells: Annotated[
        int, Field(ge=0, description="Prompt cells preserved byte-for-byte.")
    ]
    replaced_cells: Annotated[
        int, Field(ge=1, description="Prompt cells rebuilt for this correction.")
    ]
    previous_objects: Annotated[
        int, Field(ge=1, description="Object count before the incremental rebuild.")
    ]
    objects_added: Annotated[
        int, Field(ge=0, description="New content-addressed objects installed.")
    ]
    objects_removed: Annotated[
        int, Field(ge=0, description="Unreferenced prior objects removed.")
    ]
    source_verification_sha256: Annotated[
        Sha256, Field(description="Digest of the retained source-verification report.")
    ]
    source_verified: Annotated[
        bool, Field(description="Whether the archive passed offline integrity checks.")
    ]

    @model_validator(mode="after")
    def validate_incremental_counts(self) -> "PromptArchiveReference":
        if self.retained_cells + self.replaced_cells != self.cells:
            raise ValueError(
                "retained and replaced prompt cells must cover the archive"
            )
        if (
            self.previous_objects + self.objects_added - self.objects_removed
            != self.objects
        ):
            raise ValueError(
                "prompt object delta does not produce the final object count"
            )
        return self


class TranscriptArchiveReference(ArtifactReference):
    """Integrity summary for the compact retail-ablation transcript archive."""

    rows: Annotated[int, Field(ge=1, description="Calls in the transcript archive.")]
    delivered_turns: Annotated[
        int, Field(ge=1, description="Delivered turns represented by all calls.")
    ]
    source_result_hashes: Annotated[
        int,
        Field(ge=1, description="Distinct source result files bound by the archive."),
    ]
    source_simulation_hashes: Annotated[
        int,
        Field(ge=1, description="Distinct source simulations bound by the archive."),
    ]
    transcript_sha256: Annotated[
        Sha256, Field(description="Digest of the companion transcript JSONL.")
    ]


class AblationReplacementCell(BaseModel):
    """Old and corrected snapshots for one Mandarin localization-ablation cell."""

    model_config = ConfigDict(frozen=True)

    condition: Annotated[
        Literal["source_english_identity", "native_script_database"],
        Field(
            description="Entity-representation intervention represented by the cell."
        ),
    ]
    language: Annotated[
        Literal["zh"], Field(description="Language code for the corrected cohort.")
    ]
    system: Annotated[
        Literal["openai_xhigh", "gemini_high"],
        Field(description="Voice system represented by the result cell."),
    ]
    canonical_path: Annotated[
        str, Field(description="Active path relative to the tau-multi evidence root.")
    ]
    archive_path: Annotated[
        str, Field(description="Preserved old path relative to the evidence root.")
    ]
    staged_source_path: Annotated[
        str, Field(description="Repository-relative logical path of the corrected run.")
    ]
    provider: Annotated[str, Field(description="Audio-native provider.")]
    model: Annotated[str, Field(description="Audio-native model.")]
    reasoning_effort: Annotated[str, Field(description="Pinned provider effort.")]
    previous: Annotated[
        ResultSnapshot, Field(description="Archived pre-correction result snapshot.")
    ]
    replacement: Annotated[
        ResultSnapshot, Field(description="Active corrected result snapshot.")
    ]
    prompt_treatment: Annotated[
        PromptTreatmentEvidence,
        Field(description="Proof that every corrected caller received the treatment."),
    ]
    prompt_profile_sha256: Annotated[
        Sha256,
        Field(description="Profile digest in the corrected prompt snapshot archive."),
    ]
    exclusions: Annotated[
        AblationAttemptExclusions,
        Field(description="Noncanonical attempt artifacts excluded from this cell."),
    ]

    @model_validator(mode="after")
    def validate_paths_frame_and_prompt(self) -> "AblationReplacementCell":
        for label, value in (
            ("canonical_path", self.canonical_path),
            ("archive_path", self.archive_path),
            ("staged_source_path", self.staged_source_path),
        ):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be a safe relative path")

        expected_archive = f"validation_runs/zh/{self.canonical_path}"
        if self.archive_path != expected_archive:
            raise ValueError("archive path must preserve the canonical relative path")

        root_name = {
            "source_english_identity": "retail_unlocalized",
            "native_script_database": "retail_dbscript",
        }[self.condition]
        expected_canonical = (
            "ablations/localization/"
            f"{root_name}_v1_mandarin_retail/zh_retail_{self.system}/results.json"
        )
        if self.canonical_path != expected_canonical:
            raise ValueError("canonical path disagrees with condition or system")

        staged_root = {
            "source_english_identity": "retail_unlocalized_name_roles",
            "native_script_database": "retail_dbscript_name_roles",
        }[self.condition]
        expected_staged = (
            f"data/simulations/{staged_root}_v1_mandarin_retail/"
            f"zh_retail_{self.system}/results.json"
        )
        if self.staged_source_path != expected_staged:
            raise ValueError("staged path disagrees with condition or system")

        expected_runtime = {
            "openai_xhigh": ("openai", "gpt-realtime-2", "xhigh"),
            "gemini_high": ("gemini", "gemini-3.1-flash-live-preview", "high"),
        }[self.system]
        if (self.provider, self.model, self.reasoning_effort) != expected_runtime:
            raise ValueError("provider, model, or effort disagrees with the system")

        if self.previous.calls != 30 or self.replacement.calls != 30:
            raise ValueError("each corrected ablation cell must contain 30 calls")
        if self.previous.trials != (0,) or self.replacement.trials != (0,):
            raise ValueError("corrected ablation cells must use trial zero only")
        if set(self.previous.trial_success) != {"0"} or set(
            self.replacement.trial_success
        ) != {"0"}:
            raise ValueError("trial-success maps must contain trial zero only")
        if self.previous.task_ids_sha256 != self.replacement.task_ids_sha256:
            raise ValueError("replacement must preserve the 30-task frame")
        if self.prompt_treatment.calls_verified != self.replacement.calls:
            raise ValueError("every replacement call must have prompt proof")
        if self.prompt_treatment.prompt_requests_verified != self.replacement.calls:
            raise ValueError("every replacement call must have one prompt request")
        if self.exclusions.included_infrastructure_failures:
            raise ValueError("scored calls must exclude infrastructure failures")
        return self


class RetailAblationReplacementManifest(BaseModel):
    """Portable evidence for the four corrected Mandarin ablation cells."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        Literal["tau-multi-retail-ablation-replacement-v1"],
        Field(description="Serialized contract version."),
    ]
    replacement_id: Annotated[
        Literal["retail-ablation-name-roles-v1-zh-2026-09-18"],
        Field(description="Stable identifier for this four-cell replacement."),
    ]
    target_branch: Annotated[
        Literal["soham/tau-multilingual-review-base"],
        Field(description="Release branch receiving the corrected artifacts."),
    ]
    base_commit: Annotated[
        str, Field(description="Release branch commit before the swap.")
    ]
    prompt_treatment_version: Annotated[
        Literal["v1"], Field(description="Retail caller name-role treatment version.")
    ]
    prompt_template_sha256: Annotated[
        Sha256, Field(description="Digest of the applied name-role prompt template.")
    ]
    active_replacement_cells: Annotated[
        int, Field(ge=1, description="Corrected ablation cells installed as active.")
    ]
    unchanged_active_ablation_cells: Annotated[
        int, Field(ge=0, description="Ablation cells intentionally left unchanged.")
    ]
    accepted_calls: Annotated[
        int, Field(ge=1, description="Indexed calls across corrected cells.")
    ]
    previous_cells_archived: Annotated[
        bool,
        Field(description="Whether every prior cell is preserved in validation_runs."),
    ]
    old_archive_uploaded: Annotated[
        bool, Field(description="Whether the preserved raw archive is on Drive.")
    ]
    human_validation_preserved_unchanged: Annotated[
        bool,
        Field(description="Whether the frozen human-validation cohort is unchanged."),
    ]
    exclusions: Annotated[
        AblationAttemptExclusions,
        Field(description="Attempt exclusions summed over all corrected cells."),
    ]
    prompt_archive: Annotated[
        PromptArchiveReference,
        Field(description="Prompt archive after the four-cell replacement."),
    ]
    transcript_archive: Annotated[
        TranscriptArchiveReference,
        Field(description="Compact transcript archive after replacement."),
    ]
    cells: Annotated[
        tuple[AblationReplacementCell, ...],
        Field(
            min_length=4, max_length=4, description="Exact corrected cell inventory."
        ),
    ]

    @model_validator(mode="after")
    def validate_replacement_scope(self) -> "RetailAblationReplacementManifest":
        expected = {
            (condition, system)
            for condition in ("source_english_identity", "native_script_database")
            for system in ("openai_xhigh", "gemini_high")
        }
        keys = {(cell.condition, cell.system) for cell in self.cells}
        if keys != expected or len(keys) != len(self.cells):
            raise ValueError("manifest must contain the exact four Mandarin cells")
        if self.active_replacement_cells != len(self.cells):
            raise ValueError("active replacement count disagrees with cell inventory")
        if self.unchanged_active_ablation_cells != 4:
            raise ValueError("the four Hindi ablation cells must remain unchanged")
        if self.accepted_calls != sum(cell.replacement.calls for cell in self.cells):
            raise ValueError("accepted-call total disagrees with cell inventory")
        if not self.previous_cells_archived:
            raise ValueError("every previous cell must remain in validation_runs")
        if self.old_archive_uploaded:
            raise ValueError("preparation manifest must not claim a Drive upload")
        if not self.human_validation_preserved_unchanged:
            raise ValueError("frozen human validation must remain unchanged")
        if not self.prompt_archive.source_verified:
            raise ValueError("prompt archive must pass offline verification")
        if any(
            cell.prompt_treatment.template_sha256 != self.prompt_template_sha256
            for cell in self.cells
        ):
            raise ValueError("cell prompt hashes disagree with the treatment template")
        if len({cell.replacement.task_ids_sha256 for cell in self.cells}) != 1:
            raise ValueError("all cells must preserve one shared 30-task frame")

        for field_name in AblationAttemptExclusions.model_fields:
            observed = getattr(self.exclusions, field_name)
            expected_total = sum(
                getattr(cell.exclusions, field_name) for cell in self.cells
            )
            if observed != expected_total:
                raise ValueError(f"aggregate exclusion mismatch: {field_name}")
        return self


class JudgeEvidence(BaseModel):
    """Judge composition contracts for corrected trial-zero voice calls."""

    model_config = ConfigDict(frozen=True)

    scope: Literal["corrected_voice_trial_0"]
    calls: Annotated[int, Field(ge=1)]
    composition_cohort_fingerprint: Sha256
    composition_report_sha256: Sha256
    quality_manifest_sha256: Sha256
    delivery_manifest_sha256: Sha256
    naturalness_manifest_sha256: Sha256
    naturalness_identity_sha256: Sha256
    quality_model: str
    quality_prompt_version: str
    quality_rubric_version: str
    quality_metrics_version: str
    delivery_model: str
    delivery_prompt_version: str
    delivery_sample_rate: Annotated[float, Field(gt=0.0, le=1.0)]
    delivery_max_segments: Annotated[int, Field(ge=1)]
    excluded_quality_factor_ids: tuple[str, ...]


class ReleaseArtifacts(BaseModel):
    """Portable evidence regenerated for the replacement cohort."""

    model_config = ConfigDict(frozen=True)

    prompt_archive: ArtifactReference
    experience_analysis: ArtifactReference
    latency_analysis: ArtifactReference
    naturalness_bootstrap: ArtifactReference
    previous_experience_analysis: ArtifactReference
    human_validation: ArtifactReference


class RetailReplacementManifest(BaseModel):
    """Complete, non-destructive plan for the 14-cell active-data swap."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["tau-multi-retail-replacement-v1"]
    replacement_id: Literal["retail-name-roles-v1-ko-zh-2026-09-18"]
    target_branch: Literal["soham/tau-multilingual-review-base"]
    base_commit: str
    previous_audit: AuditReference
    replacement_audit: AuditReference
    prompt_treatment_version: Literal["v1"]
    prompt_template_sha256: Sha256
    active_replacement_cells: Annotated[int, Field(ge=1)]
    unchanged_active_cells: Annotated[int, Field(ge=0)]
    old_archive_uploaded: bool
    human_validation_preserved_unchanged: bool
    exclusions: ExclusionSummary
    judges: JudgeEvidence
    artifacts: ReleaseArtifacts
    fingerprints: CohortFingerprints
    cells: Annotated[tuple[ReplacementCell, ...], Field(min_length=14, max_length=14)]

    @model_validator(mode="after")
    def validate_replacement_scope(self) -> "RetailReplacementManifest":
        expected = {
            (modality, language, system)
            for modality, systems in (
                (
                    "voice",
                    (
                        "openai_minimal",
                        "openai_xhigh",
                        "gemini_minimal",
                        "gemini_high",
                        "xai_provider_default",
                    ),
                ),
                ("text", ("gpt55_xhigh", "gemini31pro_high")),
            )
            for language in ("ko", "zh")
            for system in systems
        }
        keys = {(cell.modality, cell.language, cell.system) for cell in self.cells}
        if keys != expected or len(keys) != len(self.cells):
            raise ValueError("manifest must contain exactly the 14 approved cells")
        if self.active_replacement_cells != len(self.cells):
            raise ValueError("active_replacement_cells does not match cell inventory")
        if sum(cell.replacement.calls for cell in self.cells) != 1_100:
            raise ValueError("approved replacement cohort must contain 1,100 calls")
        if self.unchanged_active_cells != 112:
            raise ValueError("the other 112 active cells must remain unchanged")
        if self.old_archive_uploaded:
            raise ValueError("preparation manifest must not claim a Drive upload")
        if not self.human_validation_preserved_unchanged:
            raise ValueError("frozen human validation must remain unchanged")
        if (
            self.prompt_template_sha256
            != next(iter(self.cells)).prompt_treatment.template_sha256
        ):
            raise ValueError("top-level prompt hash disagrees with cell evidence")
        if any(
            cell.prompt_treatment.template_sha256 != self.prompt_template_sha256
            for cell in self.cells
        ):
            raise ValueError("all cells must use the same prompt template")
        return self


def load_retail_replacement_manifest(path: Path) -> RetailReplacementManifest:
    """Load and validate a checked-in retail replacement manifest."""

    return RetailReplacementManifest.model_validate_json(path.read_text())


def dump_retail_replacement_manifest(
    manifest: RetailReplacementManifest, path: Path
) -> None:
    """Write a canonical, human-reviewable manifest."""

    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n"
    )


def load_retail_ablation_replacement_manifest(
    path: Path,
) -> RetailAblationReplacementManifest:
    """Load and validate a checked-in retail-ablation replacement manifest."""

    return RetailAblationReplacementManifest.model_validate_json(path.read_text())


def dump_retail_ablation_replacement_manifest(
    manifest: RetailAblationReplacementManifest, path: Path
) -> None:
    """Write a canonical, human-reviewable ablation replacement manifest."""

    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n"
    )
