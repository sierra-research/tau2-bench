# Copyright Sierra
"""Promote a completed corrected slice into a full paper naturalness sidecar.

The corrected Korean and Mandarin retail replay is intentionally runnable before
the large canonical sidecar is available.  This module later composes those 500
completed calls with the 3,250 unchanged canonical calls without invoking a
judge.  Replacement artifacts are accepted only when their complete semantic
input identity matches the hybrid cohort; their source namespace and input
digests are then re-keyed for the canonical paper paths while judge outputs are
preserved verbatim.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import JudgeOutcome
from tau2.judges.nativeness.corrected_paper_trial import (
    CANONICAL_CORRECTED_COHORT_CONTRACT,
    REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT,
    CorrectedCohortContract,
    CorrectedRunManifest,
    _prepare_bundle,
    _simulation_key,
    _validate_bootstrap_call,
)
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EXPERIENCE_PATH,
    CANONICAL_VALIDATION_PATH,
    TRIAL_ARTIFACT_SCHEMA_VERSION,
    ExperienceCohort,
    ExperienceManifest,
    ExperienceSource,
    FrozenJudgeSpec,
    TrialCodeProvenance,
    TrialCounts,
    TrialRunIdentity,
    TrialRunManifest,
    TrialSimulationArtifact,
    TrialSimulationPlan,
    TrialSourceIdentity,
    TrialUtteranceArtifact,
    TrialUtterancePlan,
    ValidationEvidenceIdentity,
    _aggregate_simulation,
    _aggregate_trial,
    _atomic_write,
    _load_experience,
    _sha256_value,
    _simulation_path,
    _unit_digest,
)
from tau2.judges.nativeness.validation import VALIDATION_LANGUAGES, sha256_file

PROMOTION_SCHEMA_VERSION = "tau-multi-naturalness-promotion-v2"
PROMOTED_EXPERIENCE_FILENAME = "hybrid_experience.json"
PROMOTION_FILENAME = "promotion.json"
VALIDATION_BRIDGE_SCHEMA_VERSION = "tau-multi-naturalness-validation-bridge-v1"
CODE_BRIDGE_SCHEMA_VERSION = "tau-multi-naturalness-code-bridge-v1"
LEGACY_VALIDATION_PATH = (
    "data/simulations/paper_runs/tau-multi/validations/utterance_naturalness"
)
NATURALNESS_VALIDATION_REVISION = "naturalness-v16-rubric-v20"


class NaturalnessPromotionConfig(BaseModel):
    """Validated paths for one API-free corrected-sidecar promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Annotated[
        Path,
        Field(description="Repository containing the canonical Experience."),
    ]
    validation_repo_root: Annotated[
        Path | None,
        Field(
            description="Repository containing current validation evidence; "
            "defaults to repo_root."
        ),
    ] = None
    experience_manifest: Annotated[
        Path,
        Field(description="Full 75-root corrected cohort lock."),
    ]
    evidence_root: Annotated[
        Path,
        Field(description="Read-only hybrid transcript evidence root."),
    ]
    bootstrap_experience: Annotated[
        Path,
        Field(description="Exact Experience artifact named by the bootstrap."),
    ]
    bootstrap_from: Annotated[
        Path,
        Field(description="Completed canonical 3,750-call naturalness sidecar."),
    ]
    replacement_from: Annotated[
        Path,
        Field(description="Completed replacement-only 500-call sidecar."),
    ]
    output_root: Annotated[
        Path,
        Field(description="New full canonical-consumable sidecar root."),
    ]


class PromotedExperienceProvenance(BaseModel):
    """Exact source inventory represented by the promoted sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_fingerprint_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Full 90-root cohort hash."),
    ]
    results_files: Annotated[
        list[ExperienceSource], Field(description="Full hybrid source inventory.")
    ]


class PromotedExperienceManifest(BaseModel):
    """Minimal Experience projection needed to reproduce the promoted identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: Annotated[str, Field(description="Experience instrument id.")]
    instrument_version: Annotated[
        str, Field(description="Experience instrument version.")
    ]
    cohort: Annotated[ExperienceCohort, Field(description="Full paper cohort.")]
    provenance: Annotated[
        PromotedExperienceProvenance,
        Field(description="Hybrid source hashes and cohort fingerprint."),
    ]


class NaturalnessPromotionInput(BaseModel):
    """One immutable input manifest accepted by the promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Resolved input path.")]
    manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Input manifest hash.")
    ]
    identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Input run identity.")
    ]


class NaturalnessValidationSourceFile(BaseModel):
    """One source artifact named by current naturalness provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Annotated[str, Field(min_length=1, description="Source artifact role.")]
    source_path: Annotated[
        str, Field(min_length=1, description="Archived source path.")
    ]
    sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source byte hash.")
    ]
    revision: Annotated[
        str, Field(min_length=1, description="Source contract revision.")
    ]


class NaturalnessValidationProvenance(BaseModel):
    """Typed current-archive provenance for one language's naturalness judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validation-provenance-v1"]
    language: Annotated[str, Field(min_length=1, description="Language code.")]
    measure_id: Literal["naturalness"]
    evaluation_level: Literal["utterance"]
    source_revision: Annotated[
        str, Field(min_length=1, description="Archive source revision.")
    ]
    source_files: Annotated[
        list[NaturalnessValidationSourceFile],
        Field(description="Hash-pinned upstream source inventory."),
    ]


class NaturalnessValidationLanguageProof(BaseModel):
    """Hash bridge from one legacy validation package into the current archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Annotated[str, Field(min_length=1, description="Language code.")]
    legacy_dev_result_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy dev result hash.")
    ]
    legacy_test_result_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy test result hash.")
    ]
    legacy_prompt_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy prompt hash.")
    ]
    current_prompt_path: Annotated[
        str, Field(description="Current archive prompt path.")
    ]
    current_prompt_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Current prompt hash.")
    ]
    current_provenance_path: Annotated[
        str, Field(description="Current archive provenance path.")
    ]
    current_provenance_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Provenance file hash.")
    ]
    current_result_sources: Annotated[
        list[NaturalnessValidationSourceFile],
        Field(min_length=2, max_length=2, description="Pinned dev/test sources."),
    ]


class NaturalnessValidationMigrationBridge(BaseModel):
    """Fail-closed proof connecting legacy and current validation identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-validation-bridge-v1"]
    bootstrap_manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy run manifest.")
    ]
    bootstrap_identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy run identity.")
    ]
    replacement_manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Replacement manifest.")
    ]
    replacement_identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Replacement identity.")
    ]
    legacy_evidence: Annotated[
        ValidationEvidenceIdentity,
        Field(description="Exact evidence identity embedded in the bootstrap."),
    ]
    current_evidence: Annotated[
        ValidationEvidenceIdentity,
        Field(description="Validated current archive used by the replacement."),
    ]
    languages: Annotated[
        list[NaturalnessValidationLanguageProof],
        Field(description="One exact hash bridge per non-English language."),
    ]


class NaturalnessCodeMigrationBridge(BaseModel):
    """Exact legacy and current code identities for a semantic replay bridge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-code-bridge-v1"]
    bootstrap_manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy run manifest.")
    ]
    bootstrap_identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Legacy run identity.")
    ]
    replacement_manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Replacement manifest.")
    ]
    replacement_identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Replacement identity.")
    ]
    bootstrap_invocations: Literal[5]
    legacy_code: Annotated[
        TrialCodeProvenance,
        Field(description="One exact provenance shared by all bootstrap invocations."),
    ]
    replacement_code: Annotated[
        TrialCodeProvenance,
        Field(description="Frozen code provenance embedded in the corrected replay."),
    ]
    current_code: Annotated[
        TrialCodeProvenance,
        Field(description="Current frozen scorer provenance used for verification."),
    ]


class NaturalnessPromotedArtifact(BaseModel):
    """Byte identity for one call artifact copied or re-keyed into the output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: Annotated[
        Literal["canonical", "replacement"],
        Field(description="Whether the judge output was reused or re-keyed."),
    ]
    language: Annotated[str, Field(description="Call language.")]
    domain: Annotated[str, Field(description="Call domain.")]
    system_slug: Annotated[str, Field(description="Canonical system slug.")]
    simulation_id: Annotated[str, Field(description="Stable simulation id.")]
    utterances: Annotated[int, Field(ge=0, description="Embedded utterance results.")]
    input_path: Annotated[str, Field(description="Resolved source artifact path.")]
    input_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source artifact hash.")
    ]
    output_path: Annotated[
        str, Field(description="Artifact path relative to the promoted root.")
    ]
    output_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Promoted artifact hash.")
    ]


class NaturalnessPromotionCounts(BaseModel):
    """Closed promotion sizes and utterance totals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    roots: Annotated[int, Field(ge=0, description="Selected hybrid roots.")]
    calls: Annotated[int, Field(ge=0, description="Selected hybrid calls.")]
    utterances: Annotated[int, Field(ge=0, description="All embedded utterances.")]
    canonical_calls: Annotated[
        int, Field(ge=0, description="Unchanged calls copied byte-for-byte.")
    ]
    replacement_calls: Annotated[
        int, Field(ge=0, description="Corrected calls re-keyed without judging.")
    ]
    canonical_utterances: Annotated[
        int, Field(ge=0, description="Utterances reused from the bootstrap.")
    ]
    replacement_utterances: Annotated[
        int, Field(ge=0, description="Utterances reused from corrected output.")
    ]
    zero_utterance_calls: Annotated[
        int, Field(ge=0, description="Calls with no delivered agent utterance.")
    ]


class NaturalnessPromotionReport(BaseModel):
    """Auditable, deterministic receipt for the API-free composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-promotion-v2"]
    hybrid_cohort_manifest: Annotated[
        NaturalnessPromotionInput,
        Field(description="Full corrected cohort lock used for re-keying."),
    ]
    canonical_bootstrap: Annotated[
        NaturalnessPromotionInput,
        Field(description="Full original naturalness sidecar."),
    ]
    bootstrap_experience: Annotated[
        NaturalnessPromotionInput,
        Field(description="Original Experience artifact bound by the bootstrap."),
    ]
    replacement_sidecar: Annotated[
        NaturalnessPromotionInput,
        Field(description="Completed corrected-only naturalness sidecar."),
    ]
    validation_bridge: Annotated[
        NaturalnessValidationMigrationBridge | None,
        Field(description="Typed legacy-to-current validation migration proof."),
    ] = None
    validation_bridge_sha256: Annotated[
        str | None,
        Field(description="Hash of the complete typed validation bridge."),
    ] = None
    code_bridge: Annotated[
        NaturalnessCodeMigrationBridge | None,
        Field(description="Typed legacy-to-current scorer-code migration proof."),
    ] = None
    code_bridge_sha256: Annotated[
        str | None,
        Field(description="Hash of the complete typed code bridge."),
    ] = None
    output_manifest_path: Annotated[
        str, Field(description="Promoted TrialRunManifest path.")
    ]
    output_manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Promoted manifest hash.")
    ]
    output_identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Promoted run identity.")
    ]
    promoted_experience_path: Annotated[
        str, Field(description="Minimal hybrid Experience projection path.")
    ]
    promoted_experience_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Hybrid Experience hash."),
    ]
    artifact_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Ordered artifact hash.")
    ]
    counts: Annotated[NaturalnessPromotionCounts, Field(description="Work counts.")]
    artifacts: Annotated[
        list[NaturalnessPromotedArtifact],
        Field(description="Complete promoted call-artifact inventory."),
    ]

    @model_validator(mode="after")
    def _validation_bridge_is_bound(self) -> "NaturalnessPromotionReport":
        if self.validation_bridge is None:
            if self.validation_bridge_sha256 is not None:
                raise ValueError("validation bridge hash exists without a bridge")
            return self
        expected = _sha256_value(self.validation_bridge.model_dump(mode="json"))
        if self.validation_bridge_sha256 != expected:
            raise ValueError("validation bridge hash does not reproduce")
        if (
            self.validation_bridge.bootstrap_manifest_sha256
            != self.canonical_bootstrap.manifest_sha256
            or self.validation_bridge.bootstrap_identity_sha256
            != self.canonical_bootstrap.identity_sha256
            or self.validation_bridge.replacement_manifest_sha256
            != self.replacement_sidecar.manifest_sha256
            or self.validation_bridge.replacement_identity_sha256
            != self.replacement_sidecar.identity_sha256
        ):
            raise ValueError("validation bridge is not bound to both input manifests")
        return self

    @model_validator(mode="after")
    def _code_bridge_is_bound(self) -> "NaturalnessPromotionReport":
        if self.code_bridge is None:
            if self.code_bridge_sha256 is not None:
                raise ValueError("code bridge hash exists without a bridge")
            return self
        expected = _sha256_value(self.code_bridge.model_dump(mode="json"))
        if self.code_bridge_sha256 != expected:
            raise ValueError("code bridge hash does not reproduce")
        if (
            self.code_bridge.bootstrap_manifest_sha256
            != self.canonical_bootstrap.manifest_sha256
            or self.code_bridge.bootstrap_identity_sha256
            != self.canonical_bootstrap.identity_sha256
            or self.code_bridge.replacement_manifest_sha256
            != self.replacement_sidecar.manifest_sha256
            or self.code_bridge.replacement_identity_sha256
            != self.replacement_sidecar.identity_sha256
        ):
            raise ValueError("code bridge is not bound to both input manifests")
        return self


def _source_key(source: TrialSourceIdentity) -> tuple[str, str, str]:
    return source.language, source.domain, source.system_slug


def _cohort_fingerprint(sources: list[ExperienceSource]) -> str:
    payload = "\n".join(
        f"{row.path}\t{row.sha256}\t{row.trial_0_calls}" for row in sources
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_code_provenance(code: TrialCodeProvenance, *, label: str) -> None:
    if code.source_fingerprint_sha256 != _sha256_value(code.source_files):
        raise ValueError(f"{label} code fingerprint does not reproduce")


def _build_code_migration_bridge(
    *,
    bootstrap_invocation_codes: list[TrialCodeProvenance],
    bootstrap_runtime_source_fingerprint_sha256: str,
    replacement_code: TrialCodeProvenance,
    current_code: TrialCodeProvenance,
    bootstrap_manifest_sha256: str,
    bootstrap_identity_sha256: str,
    replacement_manifest_sha256: str,
    replacement_identity_sha256: str,
) -> NaturalnessCodeMigrationBridge | None:
    """Prove one semantic replay across exact historical code identities."""
    if not bootstrap_invocation_codes:
        raise ValueError("bootstrap has no invocation code provenance")
    for code in bootstrap_invocation_codes:
        _validate_code_provenance(code, label="bootstrap invocation")
    legacy_code = bootstrap_invocation_codes[0]
    if any(code != legacy_code for code in bootstrap_invocation_codes[1:]):
        raise ValueError("bootstrap invocations disagree on legacy code provenance")
    if (
        legacy_code.source_fingerprint_sha256
        != bootstrap_runtime_source_fingerprint_sha256
    ):
        raise ValueError("bootstrap identity and invocation code fingerprints differ")
    _validate_code_provenance(replacement_code, label="replacement")
    _validate_code_provenance(current_code, label="current")
    if (
        replacement_code.source_fingerprint_sha256
        != current_code.source_fingerprint_sha256
        or replacement_code.source_files != current_code.source_files
    ):
        raise ValueError("replacement and current frozen scorer code differ")
    if (
        bootstrap_runtime_source_fingerprint_sha256
        == current_code.source_fingerprint_sha256
    ):
        if legacy_code.source_files != current_code.source_files:
            raise ValueError("equal code fingerprints hide a source inventory drift")
        return None
    if len(bootstrap_invocation_codes) != 5:
        raise ValueError("legacy code migration requires exactly five invocations")
    return NaturalnessCodeMigrationBridge(
        schema_version=CODE_BRIDGE_SCHEMA_VERSION,
        bootstrap_manifest_sha256=bootstrap_manifest_sha256,
        bootstrap_identity_sha256=bootstrap_identity_sha256,
        replacement_manifest_sha256=replacement_manifest_sha256,
        replacement_identity_sha256=replacement_identity_sha256,
        bootstrap_invocations=5,
        legacy_code=legacy_code,
        replacement_code=replacement_code,
        current_code=current_code,
    )


def _validate_paid_judge_specs(
    *,
    current_judges: dict[str, FrozenJudgeSpec],
    bootstrap_judges: dict[str, FrozenJudgeSpec],
    replacement_judges: dict[str, FrozenJudgeSpec],
    replacement_languages: tuple[str, ...],
) -> None:
    for language, judge in current_judges.items():
        if bootstrap_judges.get(language) != judge:
            raise ValueError(f"bootstrap judge identity drifted for {language}")
        if language in replacement_languages and (
            replacement_judges.get(language) != judge
        ):
            raise ValueError(f"replacement judge identity drifted for {language}")


def _build_validation_migration_bridge(
    *,
    validation_repo_root: Path,
    bootstrap_evidence: ValidationEvidenceIdentity | None,
    replacement_evidence: ValidationEvidenceIdentity | None,
    current_evidence: ValidationEvidenceIdentity | None,
    bootstrap_manifest_sha256: str,
    bootstrap_identity_sha256: str,
    replacement_manifest_sha256: str,
    replacement_identity_sha256: str,
) -> NaturalnessValidationMigrationBridge | None:
    """Prove the one supported legacy-to-current validation migration."""
    evidence = (bootstrap_evidence, replacement_evidence, current_evidence)
    if all(row is None for row in evidence):
        return None
    if any(row is None for row in evidence):
        raise ValueError("validation evidence is missing from one promotion input")
    assert bootstrap_evidence is not None
    assert replacement_evidence is not None
    assert current_evidence is not None
    if replacement_evidence != current_evidence:
        raise ValueError(
            "replacement validation evidence differs from the current archive"
        )
    if bootstrap_evidence == current_evidence:
        return None
    if bootstrap_evidence.path != LEGACY_VALIDATION_PATH:
        raise ValueError("unsupported legacy validation evidence path")
    if current_evidence.path != CANONICAL_VALIDATION_PATH.as_posix():
        raise ValueError("unsupported current validation evidence path")

    expected_legacy_files = {
        f"{language}/{relative}"
        for language in VALIDATION_LANGUAGES
        for relative in (
            "dev/dataset.json",
            "dev/results.json",
            "prompt.json",
            "test/dataset.json",
            "test/results.json",
        )
    }
    if set(bootstrap_evidence.files) != expected_legacy_files:
        raise ValueError("legacy validation evidence inventory drifted")

    validation_repo_root = validation_repo_root.expanduser().resolve()
    current_root = (validation_repo_root / CANONICAL_VALIDATION_PATH).resolve()
    if not current_root.is_relative_to(validation_repo_root):
        raise ValueError("current validation evidence escapes its repository")
    languages: list[NaturalnessValidationLanguageProof] = []
    for language in VALIDATION_LANGUAGES:
        legacy_dev = bootstrap_evidence.files[f"{language}/dev/results.json"]
        legacy_test = bootstrap_evidence.files[f"{language}/test/results.json"]
        legacy_prompt = bootstrap_evidence.files[f"{language}/prompt.json"]

        prompt_relative = f"prompt_sources/naturalness/{language}.json"
        prompt_path = current_root / prompt_relative
        prompt_sha = current_evidence.files.get(prompt_relative)
        if (
            prompt_sha != legacy_prompt
            or not prompt_path.is_file()
            or sha256_file(prompt_path) != legacy_prompt
        ):
            raise ValueError(
                f"current naturalness prompt does not pin legacy hash: {language}"
            )

        provenance_relative = f"provenance/utterance/naturalness/{language}.json"
        provenance_path = current_root / provenance_relative
        provenance_sha = current_evidence.files.get(provenance_relative)
        if (
            provenance_sha is None
            or not provenance_path.is_file()
            or sha256_file(provenance_path) != provenance_sha
        ):
            raise ValueError(
                f"current naturalness provenance is not hash-bound: {language}"
            )
        provenance = NaturalnessValidationProvenance.model_validate_json(
            provenance_path.read_text()
        )
        if provenance.language != language:
            raise ValueError(
                f"current naturalness provenance language drifted: {language}"
            )
        result_sources = sorted(
            (
                source
                for source in provenance.source_files
                if source.role == "naturalness_validation_results"
                and source.revision == NATURALNESS_VALIDATION_REVISION
            ),
            key=lambda source: source.source_path,
        )
        if len(result_sources) != 2 or {source.sha256 for source in result_sources} != {
            legacy_dev,
            legacy_test,
        }:
            raise ValueError(
                "current naturalness provenance does not pin the legacy dev/test "
                f"result hashes: {language}"
            )
        languages.append(
            NaturalnessValidationLanguageProof(
                language=language,
                legacy_dev_result_sha256=legacy_dev,
                legacy_test_result_sha256=legacy_test,
                legacy_prompt_sha256=legacy_prompt,
                current_prompt_path=prompt_relative,
                current_prompt_sha256=prompt_sha,
                current_provenance_path=provenance_relative,
                current_provenance_sha256=provenance_sha,
                current_result_sources=result_sources,
            )
        )
    return NaturalnessValidationMigrationBridge(
        schema_version=VALIDATION_BRIDGE_SCHEMA_VERSION,
        bootstrap_manifest_sha256=bootstrap_manifest_sha256,
        bootstrap_identity_sha256=bootstrap_identity_sha256,
        replacement_manifest_sha256=replacement_manifest_sha256,
        replacement_identity_sha256=replacement_identity_sha256,
        legacy_evidence=bootstrap_evidence,
        current_evidence=current_evidence,
        languages=languages,
    )


def _canonical_work_fingerprint(plans: list[TrialSimulationPlan]) -> str:
    return _sha256_value(
        {
            "simulations": [
                (
                    row.source.results_path,
                    row.simulation_id,
                    row.source_simulation_sha256,
                )
                for row in plans
            ],
            "utterances": [
                _unit_digest(row, turn) for row in plans for turn in row.turns
            ],
        }
    )


def _load_complete_replacement(
    root: Path,
    contract: CorrectedCohortContract,
) -> tuple[
    CorrectedRunManifest,
    dict[tuple[str, str, str, str], tuple[TrialSimulationArtifact, Path]],
]:
    manifest_path = root / "manifest.json"
    manifest = CorrectedRunManifest.model_validate_json(manifest_path.read_text())
    if manifest.identity_sha256 != _sha256_value(
        manifest.identity.model_dump(mode="json")
    ):
        raise ValueError("replacement sidecar identity hash does not reproduce")
    counts = manifest.counts
    if (
        manifest.status != "complete"
        or manifest.identity.bootstrap is not None
        or counts.roots != contract.roots
        or counts.calls != contract.calls
        or counts.reused_roots
        or counts.reused_calls
        or counts.pending_roots != contract.pending_roots
        or counts.pending_calls != contract.pending_calls
        or counts.complete_calls != contract.calls
        or counts.error_calls
        or counts.error_utterances
        or counts.judged_utterances != counts.utterances
        or manifest.aggregate is None
    ):
        raise ValueError("replacement sidecar is not a complete replacement-only run")
    expected_grid = {
        (language, contract.replacement_domain, system)
        for language in contract.replacement_languages
        for system in contract.system_slugs
    }
    sources = {_source_key(source): source for source in manifest.identity.sources}
    if set(sources) != expected_grid or len(sources) != len(manifest.identity.sources):
        raise ValueError("replacement sidecar source grid drifted")
    if any(source.origin != "corrected" for source in manifest.identity.sources):
        raise ValueError("replacement sidecar contains a non-corrected source")

    artifact_paths = sorted((root / "simulations").rglob("*.json"))
    if len(artifact_paths) != contract.calls:
        raise ValueError("replacement sidecar call-artifact inventory drifted")
    artifacts: dict[
        tuple[str, str, str, str], tuple[TrialSimulationArtifact, Path]
    ] = {}
    rows: list[TrialSimulationArtifact] = []
    for path in artifact_paths:
        artifact = TrialSimulationArtifact.model_validate_json(path.read_text())
        source = sources.get(_source_key(artifact.source))
        if source is None:
            raise ValueError(f"replacement artifact has an unknown source: {path}")
        if (
            artifact.source.results_path != source.path
            or artifact.source.results_sha256 != source.results_sha256
            or artifact.source.trial_0_calls != len(source.calls)
        ):
            raise ValueError(f"replacement artifact source identity drifted: {path}")
        old_plan = TrialSimulationPlan(
            source=artifact.source,
            simulation_id=artifact.simulation_id,
            task_id=artifact.task_id,
            trial=0,
            language=artifact.language,
            domain=artifact.domain,
            source_simulation_sha256=artifact.source_simulation_sha256,
            judge=artifact.judge,
            turns=[row.turn for row in artifact.utterances],
        )
        _validate_bootstrap_call(old_plan, artifact, path)
        key = (*_source_key(artifact.source), artifact.simulation_id)
        if key in artifacts:
            raise ValueError(f"duplicate replacement call artifact: {key}")
        artifacts[key] = (artifact, path)
        rows.append(artifact)
    if manifest.aggregate != _aggregate_trial(rows):
        raise ValueError("replacement sidecar aggregate drifted from call artifacts")
    return manifest, artifacts


def _rekey_replacement(
    plan: TrialSimulationPlan,
    artifact: TrialSimulationArtifact,
    path: Path,
) -> TrialSimulationArtifact:
    # The standalone replay is deliberately allowed to finish before the
    # independent quality, delivery, and modal-particle overlays are composed.
    # Those overlays change the serialized results/simulation hashes but do not
    # change the naturalness judge's input.  Bind semantic identity here and
    # validate every ordered AgentTurn below; the promoted artifact is then
    # re-keyed to the final hybrid hashes.  Requiring the pre-overlay byte hashes
    # would discard 500 completed judgments merely because sibling judge fields
    # were added later.
    if (
        artifact.source.language != plan.source.language
        or artifact.source.domain != plan.source.domain
        or artifact.source.system_slug != plan.source.system_slug
        or artifact.source.trial_0_calls != plan.source.trial_0_calls
        or artifact.simulation_id != plan.simulation_id
        or artifact.task_id != plan.task_id
        or artifact.trial != plan.trial
        or artifact.language != plan.language
        or artifact.domain != plan.domain
        or artifact.judge != plan.judge
    ):
        raise ValueError(f"replacement call does not match hybrid input: {path}")
    by_index = {row.turn.index: row for row in artifact.utterances}
    if len(by_index) != len(artifact.utterances) or set(by_index) != {
        turn.index for turn in plan.turns
    }:
        raise ValueError(f"replacement utterance inventory drifted: {path}")
    promoted: list[TrialUtteranceArtifact] = []
    for turn in plan.turns:
        source = by_index[turn.index]
        if source.turn != turn or source.check.outcome in {
            JudgeOutcome.ERROR,
            JudgeOutcome.DEFERRED,
        }:
            raise ValueError(f"replacement utterance input is incomplete: {path}")
        expected_plan = TrialUtterancePlan(
            source=plan.source,
            simulation_id=plan.simulation_id,
            task_id=plan.task_id,
            trial=0,
            language=plan.language,
            domain=plan.domain,
            source_simulation_sha256=plan.source_simulation_sha256,
            judge=plan.judge,
            turn=turn,
            input_sha256=_unit_digest(plan, turn),
        )
        promoted.append(
            TrialUtteranceArtifact(
                schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
                created_at=source.created_at,
                input_sha256=expected_plan.input_sha256,
                source=expected_plan.source,
                simulation_id=expected_plan.simulation_id,
                task_id=expected_plan.task_id,
                trial=0,
                language=expected_plan.language,
                domain=expected_plan.domain,
                source_simulation_sha256=expected_plan.source_simulation_sha256,
                judge=expected_plan.judge,
                turn=expected_plan.turn,
                check=source.check,
            )
        )
    result = _aggregate_simulation(plan, promoted)
    if result.check != artifact.check:
        raise ValueError(f"replacement call verdict changed during re-keying: {path}")
    return result


def _selected_sources(plans: list[TrialSimulationPlan]) -> list[TrialSourceIdentity]:
    selected: list[TrialSourceIdentity] = []
    seen: set[str] = set()
    for plan in plans:
        if plan.source.results_path not in seen:
            seen.add(plan.source.results_path)
            selected.append(plan.source)
    return selected


def _hybrid_all_sources(
    original: ExperienceManifest,
    selected: list[TrialSourceIdentity],
) -> list[ExperienceSource]:
    updates = {
        source.results_path: ExperienceSource(
            path=source.results_path,
            sha256=source.results_sha256,
            trial_0_calls=source.trial_0_calls,
        )
        for source in selected
    }
    original_paths = {source.path for source in original.results_files}
    if not set(updates).issubset(original_paths):
        raise ValueError("hybrid selected sources are outside the Experience cohort")
    return [updates.get(source.path, source) for source in original.results_files]


def _assert_distinct_output(config: NaturalnessPromotionConfig) -> None:
    output = config.output_root.expanduser().resolve()
    inputs = (
        config.evidence_root.expanduser().resolve(),
        config.bootstrap_from.expanduser().resolve(),
        config.replacement_from.expanduser().resolve(),
    )
    for source in inputs:
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError("promotion output must be separate from every input tree")


def _verify_existing_promotion(
    output_root: Path,
    *,
    hybrid_manifest_sha256: str,
    bootstrap_manifest_sha256: str,
    bootstrap_experience_sha256: str,
    replacement_manifest_sha256: str,
) -> NaturalnessPromotionReport:
    report_path = output_root / PROMOTION_FILENAME
    if not report_path.is_file():
        raise ValueError("existing promotion output has no typed receipt")
    report = NaturalnessPromotionReport.model_validate_json(report_path.read_text())
    if (
        report.hybrid_cohort_manifest.manifest_sha256 != hybrid_manifest_sha256
        or report.canonical_bootstrap.manifest_sha256 != bootstrap_manifest_sha256
        or report.bootstrap_experience.manifest_sha256 != bootstrap_experience_sha256
        or report.replacement_sidecar.manifest_sha256 != replacement_manifest_sha256
        or sha256_file(output_root / "manifest.json") != report.output_manifest_sha256
        or sha256_file(output_root / PROMOTED_EXPERIENCE_FILENAME)
        != report.promoted_experience_sha256
    ):
        raise ValueError("existing promotion output identity drifted")
    for artifact in report.artifacts:
        if sha256_file(output_root / artifact.output_path) != artifact.output_sha256:
            raise ValueError(
                f"existing promoted artifact drifted: {artifact.output_path}"
            )
    return report


def promote_replacement_only_trial0_naturalness(
    config: NaturalnessPromotionConfig,
    *,
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
    replacement_contract: CorrectedCohortContract = (
        REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT
    ),
) -> NaturalnessPromotionReport:
    """Compose a full canonical-schema sidecar without any model calls."""
    _assert_distinct_output(config)
    repo_root = config.repo_root.expanduser().resolve()
    validation_repo_root = (
        (config.validation_repo_root or config.repo_root).expanduser().resolve()
    )
    bootstrap_root = config.bootstrap_from.expanduser().resolve()
    replacement_root = config.replacement_from.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    bundle = _prepare_bundle(
        repo_root=validation_repo_root,
        experience_manifest=config.experience_manifest,
        evidence_root=config.evidence_root,
        bootstrap_from=bootstrap_root,
        contract=contract,
    )
    bootstrap_manifest_path = bootstrap_root / "manifest.json"
    bootstrap_manifest = TrialRunManifest.model_validate_json(
        bootstrap_manifest_path.read_text()
    )
    if bootstrap_manifest.identity_sha256 != _sha256_value(
        bootstrap_manifest.identity.model_dump(mode="json")
    ):
        raise ValueError("bootstrap run identity hash does not reproduce")
    bootstrap_experience_path = config.bootstrap_experience.expanduser().resolve()
    bootstrap_experience_sha = sha256_file(bootstrap_experience_path)
    if bootstrap_experience_sha != bootstrap_manifest.identity.experience_sha256:
        raise ValueError("bootstrap Experience hash differs from its run identity")
    bootstrap_experience = _load_experience(bootstrap_experience_path)
    if bootstrap_experience.results_files != bootstrap_manifest.identity.all_sources:
        raise ValueError("bootstrap Experience source inventory drifted")
    replacement_manifest, replacement_artifacts = _load_complete_replacement(
        replacement_root, replacement_contract
    )
    replacement_manifest_path = replacement_root / "manifest.json"
    hybrid_manifest_path = config.experience_manifest.expanduser().resolve()
    hybrid_manifest_sha = sha256_file(hybrid_manifest_path)
    bootstrap_manifest_sha = sha256_file(bootstrap_manifest_path)
    replacement_manifest_sha = sha256_file(replacement_manifest_path)
    if output_root.exists():
        return _verify_existing_promotion(
            output_root,
            hybrid_manifest_sha256=hybrid_manifest_sha,
            bootstrap_manifest_sha256=bootstrap_manifest_sha,
            bootstrap_experience_sha256=bootstrap_experience_sha,
            replacement_manifest_sha256=replacement_manifest_sha,
        )

    frozen_code = bundle.report.code.frozen_judge
    _validate_paid_judge_specs(
        current_judges=bundle.report.judges,
        bootstrap_judges=bootstrap_manifest.identity.judges,
        replacement_judges=replacement_manifest.identity.judges,
        replacement_languages=replacement_contract.languages,
    )
    code_bridge = _build_code_migration_bridge(
        bootstrap_invocation_codes=[
            invocation.code for invocation in bootstrap_manifest.invocations
        ],
        bootstrap_runtime_source_fingerprint_sha256=(
            bootstrap_manifest.identity.runtime_source_fingerprint_sha256
        ),
        replacement_code=replacement_manifest.identity.code.frozen_judge,
        current_code=frozen_code,
        bootstrap_manifest_sha256=bootstrap_manifest_sha,
        bootstrap_identity_sha256=bootstrap_manifest.identity_sha256,
        replacement_manifest_sha256=replacement_manifest_sha,
        replacement_identity_sha256=replacement_manifest.identity_sha256,
    )
    validation_bridge = _build_validation_migration_bridge(
        validation_repo_root=validation_repo_root,
        bootstrap_evidence=bootstrap_manifest.identity.validation_evidence,
        replacement_evidence=replacement_manifest.identity.validation_evidence,
        current_evidence=bundle.plan.validation_evidence,
        bootstrap_manifest_sha256=bootstrap_manifest_sha,
        bootstrap_identity_sha256=bootstrap_manifest.identity_sha256,
        replacement_manifest_sha256=replacement_manifest_sha,
        replacement_identity_sha256=replacement_manifest.identity_sha256,
    )

    replacement_plans = [
        plan
        for plan in bundle.plan.simulations
        if contract.is_replacement(plan.language, plan.domain)
    ]
    replacement_keys = {
        (*_source_key(plan.source), plan.simulation_id) for plan in replacement_plans
    }
    if replacement_keys != set(replacement_artifacts):
        missing = sorted(replacement_keys - set(replacement_artifacts))[:3]
        extra = sorted(set(replacement_artifacts) - replacement_keys)[:3]
        raise ValueError(
            "replacement/hybrid call inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )

    source_experience = _load_experience(repo_root / CANONICAL_EXPERIENCE_PATH)
    if source_experience.results_files != bootstrap_manifest.identity.all_sources:
        raise ValueError("canonical bootstrap and Experience source inventories differ")
    selected_sources = _selected_sources(bundle.plan.simulations)
    all_sources = _hybrid_all_sources(source_experience, selected_sources)
    cohort_fingerprint = _cohort_fingerprint(all_sources)
    experience = PromotedExperienceManifest(
        instrument=source_experience.instrument,
        instrument_version=source_experience.instrument_version,
        cohort=source_experience.cohort,
        provenance=PromotedExperienceProvenance(
            cohort_fingerprint_sha256=cohort_fingerprint,
            results_files=all_sources,
        ),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        experience_path = temporary / PROMOTED_EXPERIENCE_FILENAME
        _atomic_write(experience_path, experience)
        experience_sha = sha256_file(experience_path)
        artifacts: list[TrialSimulationArtifact] = []
        inventory: list[NaturalnessPromotedArtifact] = []
        canonical_utterances = 0
        replacement_utterances = 0
        for plan in bundle.plan.simulations:
            is_replacement = contract.is_replacement(plan.language, plan.domain)
            if is_replacement:
                source_artifact, source_path = replacement_artifacts[
                    (*_source_key(plan.source), plan.simulation_id)
                ]
                artifact = _rekey_replacement(plan, source_artifact, source_path)
                replacement_utterances += len(artifact.utterances)
                origin: Literal["canonical", "replacement"] = "replacement"
            else:
                artifact = bundle.bootstrap_artifacts[_simulation_key(plan)]
                source_path = _simulation_path(bootstrap_root, plan)
                canonical_utterances += len(artifact.utterances)
                origin = "canonical"
            output_path = _simulation_path(temporary, plan)
            if origin == "canonical":
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, output_path)
            else:
                _atomic_write(output_path, artifact)
            artifacts.append(artifact)
            inventory.append(
                NaturalnessPromotedArtifact(
                    origin=origin,
                    language=plan.language,
                    domain=plan.domain,
                    system_slug=plan.source.system_slug,
                    simulation_id=plan.simulation_id,
                    utterances=len(artifact.utterances),
                    input_path=str(source_path.resolve()),
                    input_sha256=sha256_file(source_path),
                    output_path=output_path.relative_to(temporary).as_posix(),
                    output_sha256=sha256_file(output_path),
                )
            )

        identity = TrialRunIdentity(
            schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
            experience_path=CANONICAL_EXPERIENCE_PATH.as_posix(),
            experience_sha256=experience_sha,
            cohort_fingerprint_sha256=cohort_fingerprint,
            work_fingerprint_sha256=_canonical_work_fingerprint(
                bundle.plan.simulations
            ),
            validation_evidence=bundle.plan.validation_evidence,
            runtime_source_fingerprint_sha256=frozen_code.source_fingerprint_sha256,
            all_sources=all_sources,
            selected_sources=selected_sources,
            judges=bundle.report.judges,
            trial=0,
        )
        identity_sha = _sha256_value(identity.model_dump(mode="json"))
        utterances = sum(len(row.utterances) for row in artifacts)
        zero_utterance_calls = sum(not row.utterances for row in artifacts)
        manifest = TrialRunManifest(
            schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
            created_at=min(
                bootstrap_manifest.created_at, replacement_manifest.created_at
            ),
            updated_at=max(
                bootstrap_manifest.updated_at, replacement_manifest.updated_at
            ),
            status="complete",
            identity_sha256=identity_sha,
            identity=identity,
            counts=TrialCounts(
                verified_roots=len(all_sources),
                selected_roots=len(selected_sources),
                simulations=len(artifacts),
                utterances=utterances,
                zero_utterance_simulations=zero_utterance_calls,
                complete_utterances=utterances,
                error_utterances=0,
                complete_simulations=len(artifacts),
                error_simulations=0,
            ),
            aggregate=_aggregate_trial(artifacts),
            invocations=[],
        )
        manifest_path = temporary / "manifest.json"
        _atomic_write(manifest_path, manifest)
        report = NaturalnessPromotionReport(
            schema_version=PROMOTION_SCHEMA_VERSION,
            hybrid_cohort_manifest=NaturalnessPromotionInput(
                path=str(hybrid_manifest_path),
                manifest_sha256=hybrid_manifest_sha,
                identity_sha256=bundle.report.work_fingerprint_sha256,
            ),
            canonical_bootstrap=NaturalnessPromotionInput(
                path=str(bootstrap_root),
                manifest_sha256=bootstrap_manifest_sha,
                identity_sha256=bootstrap_manifest.identity_sha256,
            ),
            bootstrap_experience=NaturalnessPromotionInput(
                path=str(bootstrap_experience_path),
                manifest_sha256=bootstrap_experience_sha,
                identity_sha256=(bootstrap_experience.cohort_fingerprint_sha256),
            ),
            replacement_sidecar=NaturalnessPromotionInput(
                path=str(replacement_root),
                manifest_sha256=replacement_manifest_sha,
                identity_sha256=replacement_manifest.identity_sha256,
            ),
            validation_bridge=validation_bridge,
            validation_bridge_sha256=(
                _sha256_value(validation_bridge.model_dump(mode="json"))
                if validation_bridge is not None
                else None
            ),
            code_bridge=code_bridge,
            code_bridge_sha256=(
                _sha256_value(code_bridge.model_dump(mode="json"))
                if code_bridge is not None
                else None
            ),
            output_manifest_path="manifest.json",
            output_manifest_sha256=sha256_file(manifest_path),
            output_identity_sha256=identity_sha,
            promoted_experience_path=PROMOTED_EXPERIENCE_FILENAME,
            promoted_experience_sha256=experience_sha,
            artifact_fingerprint_sha256=_sha256_value(
                [row.model_dump(mode="json") for row in inventory]
            ),
            counts=NaturalnessPromotionCounts(
                roots=contract.roots,
                calls=contract.calls,
                utterances=utterances,
                canonical_calls=contract.reused_calls,
                replacement_calls=contract.pending_calls,
                canonical_utterances=canonical_utterances,
                replacement_utterances=replacement_utterances,
                zero_utterance_calls=zero_utterance_calls,
            ),
            artifacts=inventory,
        )
        _atomic_write(temporary / PROMOTION_FILENAME, report)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
