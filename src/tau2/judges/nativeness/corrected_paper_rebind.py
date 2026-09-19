# Copyright Sierra
"""Rebind a promoted naturalness sidecar to canonical active result headers.

The corrected retail naturalness replay was promoted from ten temporary
trial-zero-only result headers.  The canonical paper archive subsequently
installed the same 500 simulations in full result files (two trials for the
repeated systems).  This module proves that every judged trial-zero simulation
is byte-identical, re-keys only the source identities and derived input hashes,
and copies every other judgment byte-for-byte.  It never invokes a judge and it
does not read the other 65 non-English result roots.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import Results, SimulationRun
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.nativeness.corrected_paper_promotion import (
    PROMOTED_EXPERIENCE_FILENAME,
    PROMOTION_FILENAME,
    NaturalnessPromotionReport,
    PromotedExperienceManifest,
    PromotedExperienceProvenance,
    _canonical_work_fingerprint,
    _cohort_fingerprint,
    _rekey_replacement,
)
from tau2.judges.nativeness.corrected_paper_trial import _validate_bootstrap_call
from tau2.judges.nativeness.harness import build_agent_turns
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EVIDENCE_PREFIX,
    ExperienceSource,
    TrialCounts,
    TrialRunIdentity,
    TrialRunManifest,
    TrialSimulationArtifact,
    TrialSimulationPlan,
    TrialSourceIdentity,
    ValidationEvidenceIdentity,
    _aggregate_trial,
    _atomic_write,
    _sha256_value,
)
from tau2.judges.nativeness.validation import sha256_file
from tau2.judges.validation_archive import (
    ArchiveFile,
    ValidationLeafManifest,
    ValidationRootManifest,
    ValidationRow,
    ValidationSourceArtifact,
    read_validation_archive,
)
from tau2.paper.multilingual import MultilingualAudit
from tau2.paper.retail_replacement import load_retail_replacement_manifest

REBIND_SCHEMA_VERSION = "tau-multi-naturalness-canonical-rebind-v2"
REBIND_FILENAME = "rebind.json"
DEFAULT_ACTIVE_AUDIT = Path("papers/tau-multilingual/reproduction/audit.json")
DEFAULT_REPLACEMENT_MANIFEST = Path(
    "papers/tau-multilingual/reproduction/retail_name_role_replacement.json"
)
VALIDATION_PROJECTION_SCHEMA_VERSION = (
    "tau-multi-naturalness-validation-evidence-projection-v1"
)
VALIDATION_DROPPED_FIELDS = (
    "source_row_ids",
    "source_artifacts",
    "human_note",
)


class NaturalnessRebindContract(BaseModel):
    """Closed cohort dimensions for one source-header rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    languages: tuple[str, ...]
    domains: tuple[str, ...]
    system_slugs: tuple[str, ...]
    affected_languages: tuple[str, ...]
    affected_domain: str
    trial: Literal[0] = 0
    calls_per_root: Annotated[int, Field(ge=1)]

    @property
    def all_roots(self) -> int:
        return len(self.languages) * len(self.domains) * len(self.system_slugs)

    @property
    def selected_roots(self) -> int:
        return (len(self.languages) - 1) * len(self.domains) * len(self.system_slugs)

    @property
    def affected_roots(self) -> int:
        return len(self.affected_languages) * len(self.system_slugs)

    @property
    def selected_calls(self) -> int:
        return self.selected_roots * self.calls_per_root

    @property
    def affected_calls(self) -> int:
        return self.affected_roots * self.calls_per_root


CANONICAL_REBIND_CONTRACT = NaturalnessRebindContract(
    languages=("en", "es", "pt", "hi", "ko", "zh"),
    domains=("airline", "retail", "telecom"),
    system_slugs=(
        "openai_minimal",
        "openai_xhigh",
        "gemini_minimal",
        "gemini_high",
        "xai_provider_default",
    ),
    affected_languages=("ko", "zh"),
    affected_domain="retail",
    calls_per_root=50,
)


class ValidationEvidenceProjectionContract(BaseModel):
    """Closed identity and count contract for the approved public projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sanitized_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sanitized_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    public_projection_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    partitions: Annotated[int, Field(ge=1)]
    rows: Annotated[int, Field(ge=1)]
    metric_rows: Annotated[int, Field(ge=1)]
    prompt_source_files: Annotated[int, Field(ge=0)]
    invariant_files: Annotated[int, Field(ge=1)]
    removed_provenance_files: Annotated[int, Field(ge=1)]
    source_files: Annotated[int, Field(ge=1)]
    sanitized_files: Annotated[int, Field(ge=1)]


CANONICAL_VALIDATION_PROJECTION_CONTRACT = ValidationEvidenceProjectionContract(
    source_manifest_sha256=(
        "641e831bffb0ddafc07c31ce0a671ec7b74ea6613308499e73467a767d3675aa"
    ),
    sanitized_manifest_sha256=(
        "4f6bd97c3d9a1d8ae4179ae5a1b3990b13a055f3922e86c3957fd7da4b4c86a5"
    ),
    source_inventory_sha256=(
        "dba1fa3404fa7e89ed65d16e471d3efae16e195ca7d2c49d9c3a35984cc03738"
    ),
    sanitized_inventory_sha256=(
        "4481fd57c77aa211e8da9e78430723227e51e374796fb5cb0aa38cc4312e76a5"
    ),
    public_projection_sha256=(
        "ceeb35d675304b90a3efb9e21c75bd7bc5462f9d52938012eb3a192687506602"
    ),
    partitions=29,
    rows=3601,
    metric_rows=80,
    prompt_source_files=7,
    invariant_files=38,
    removed_provenance_files=29,
    source_files=126,
    sanitized_files=97,
)


class ValidationEvidenceProjectionConfig(BaseModel):
    """Explicit roots used to prove and reference a sanitized evidence archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_repo_root: Annotated[
        Path,
        Field(description="Repository root for the sanitized output identity path."),
    ]
    source_root: Annotated[
        Path,
        Field(description="Archived v1 evidence root named by the source sidecar."),
    ]
    sanitized_root: Annotated[
        Path,
        Field(description="Approved v2 public projection to bind into the output."),
    ]


class NaturalnessRebindConfig(BaseModel):
    """Validated paths for an API-free canonical source rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Annotated[Path, Field(description="Release repository root.")]
    active_audit: Annotated[
        Path,
        Field(description="Strict audit containing current active result hashes."),
    ] = DEFAULT_ACTIVE_AUDIT
    replacement_manifest: Annotated[
        Path,
        Field(description="Typed Korean/Mandarin retail replacement receipt."),
    ] = DEFAULT_REPLACEMENT_MANIFEST
    active_results_root: Annotated[
        Path,
        Field(description="Read-only root resolving canonical result paths."),
    ]
    sidecar_from: Annotated[
        Path,
        Field(description="Completed promoted naturalness sidecar to rebind."),
    ]
    output_root: Annotated[
        Path,
        Field(description="New canonical-bound naturalness sidecar root."),
    ]
    validation_evidence_projection: Annotated[
        ValidationEvidenceProjectionConfig | None,
        Field(
            description=(
                "Optional explicit source/public evidence pair whose equivalence "
                "must be proven before replacing inherited validation evidence."
            )
        ),
    ] = None


class RebindInputReference(BaseModel):
    """Hash-pinned external input to a rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    identity: str

    @model_validator(mode="after")
    def _path_is_portable(self) -> "RebindInputReference":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or self.path in {"", "."}:
            raise ValueError("rebind input references must use safe relative paths")
        return self


class _LegacyValidationPrivateFields(BaseModel):
    """The three review-only columns deliberately removed from the public archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_row_ids: Annotated[list[str], Field(min_length=1)]
    source_artifacts: Annotated[list[ValidationSourceArtifact], Field(min_length=1)]
    human_note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_csv_values(cls, value):
        if not isinstance(value, dict):
            return value
        parsed = dict(value)
        for field in ("source_row_ids", "source_artifacts"):
            raw = parsed.get(field)
            if isinstance(raw, str):
                parsed[field] = json.loads(raw)
        note = parsed.get("human_note")
        if isinstance(note, str) and not note.strip():
            parsed["human_note"] = None
        return parsed


class _LegacyValidationLeafManifest(BaseModel):
    """Historical private validation-leaf contract used only for projection proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validation-leaf-v1"]
    evaluation_level: str
    measure_id: str
    language: str
    validation: ArchiveFile
    metrics: ArchiveFile
    rows: Annotated[int, Field(ge=0)]
    metric_rows: Annotated[int, Field(ge=0)]


class _LegacyValidationPartitionRecord(BaseModel):
    """One historical validation leaf referenced by the v1 root manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_level: str
    measure_id: str
    language: str
    manifest: ArchiveFile


class _LegacyValidationRootManifest(BaseModel):
    """Historical private validation-root contract used only for projection proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validations-v1"]
    index: ArchiveFile
    readme: ArchiveFile
    partitions: tuple[_LegacyValidationPartitionRecord, ...]
    rows: Annotated[int, Field(ge=0)]
    metric_rows: Annotated[int, Field(ge=0)]


class ValidationEvidenceProjectionCounts(BaseModel):
    """Closed counts proving the private-to-public archive projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    partitions: Annotated[int, Field(ge=1)]
    rows: Annotated[int, Field(ge=1)]
    metric_rows: Annotated[int, Field(ge=1)]
    prompt_source_files: Annotated[int, Field(ge=0)]
    invariant_files: Annotated[int, Field(ge=1)]
    removed_provenance_files: Annotated[int, Field(ge=1)]
    source_files: Annotated[int, Field(ge=1)]
    sanitized_files: Annotated[int, Field(ge=1)]


class ValidationEvidenceProjectionProof(BaseModel):
    """Typed proof that a public archive changes no judge-facing evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-validation-evidence-projection-v1"]
    prior_evidence: ValidationEvidenceIdentity
    sanitized_evidence: ValidationEvidenceIdentity
    dropped_fields: tuple[
        Literal["source_row_ids", "source_artifacts", "human_note"], ...
    ]
    source_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sanitized_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_projection_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sanitized_projection_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    invariant_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    removed_provenance_inventory_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    counts: ValidationEvidenceProjectionCounts

    @model_validator(mode="after")
    def _projection_is_coherent(self) -> "ValidationEvidenceProjectionProof":
        for label, raw in (
            ("prior evidence", self.prior_evidence.path),
            ("sanitized evidence", self.sanitized_evidence.path),
        ):
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or raw in {"", "."}:
                raise ValueError(f"{label} path must be safe and relative")
        if self.dropped_fields != VALIDATION_DROPPED_FIELDS:
            raise ValueError("validation projection dropped-field contract drifted")
        if self.source_projection_sha256 != self.sanitized_projection_sha256:
            raise ValueError("validation public projections differ")
        return self


class NaturalnessActiveInventory(BaseModel):
    """Current source hashes and the exact approved source-difference set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit: RebindInputReference
    replacement: RebindInputReference
    sources: tuple[ExperienceSource, ...]
    affected_paths: tuple[str, ...]

    @model_validator(mode="after")
    def _paths_are_unique_and_closed(self) -> "NaturalnessActiveInventory":
        source_paths = [source.path for source in self.sources]
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("active inventory contains duplicate source paths")
        if len(self.affected_paths) != len(set(self.affected_paths)):
            raise ValueError("active inventory contains duplicate affected paths")
        if not set(self.affected_paths).issubset(source_paths):
            raise ValueError("affected paths are outside the active source inventory")
        return self


class NaturalnessReboundSource(BaseModel):
    """Exact old-to-current proof for one rebound result source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    domain: str
    system_slug: str
    results_path: str
    source_key: str
    previous_results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    current_results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    simulations: Annotated[int, Field(ge=1)]
    utterances: Annotated[int, Field(ge=0)]
    simulation_inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    previous_artifact_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    current_artifact_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]


class NaturalnessReboundArtifact(BaseModel):
    """Byte provenance for one copied or source-rebound call artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["copied", "rebound"]
    language: str
    domain: str
    system_slug: str
    simulation_id: str
    utterances: Annotated[int, Field(ge=0)]
    input_path: str
    input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_path: str
    output_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class NaturalnessRebindCounts(BaseModel):
    """Closed work counts for the source-header rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    all_roots: Annotated[int, Field(ge=1)]
    selected_roots: Annotated[int, Field(ge=1)]
    calls: Annotated[int, Field(ge=1)]
    utterances: Annotated[int, Field(ge=0)]
    rebound_roots: Annotated[int, Field(ge=1)]
    rebound_calls: Annotated[int, Field(ge=1)]
    rebound_utterances: Annotated[int, Field(ge=0)]
    copied_roots: Annotated[int, Field(ge=0)]
    copied_calls: Annotated[int, Field(ge=0)]
    copied_utterances: Annotated[int, Field(ge=0)]
    zero_utterance_calls: Annotated[int, Field(ge=0)]


class NaturalnessRebindReport(BaseModel):
    """Deterministic receipt for a no-judge canonical source rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-canonical-rebind-v2"]
    source_sidecar: RebindInputReference
    source_promotion: RebindInputReference
    source_experience: RebindInputReference
    active_audit: RebindInputReference
    replacement_manifest: RebindInputReference
    output_manifest_path: Literal["manifest.json"]
    output_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_identity_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_experience_path: Literal["hybrid_experience.json"]
    output_experience_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    validation_evidence_projection: ValidationEvidenceProjectionProof | None = None
    artifact_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    counts: NaturalnessRebindCounts
    sources: tuple[NaturalnessReboundSource, ...]
    artifacts: tuple[NaturalnessReboundArtifact, ...]

    @model_validator(mode="after")
    def _artifact_fingerprint_reproduces(self) -> "NaturalnessRebindReport":
        observed = _sha256_value(
            [artifact.model_dump(mode="json") for artifact in self.artifacts]
        )
        if observed != self.artifact_fingerprint_sha256:
            raise ValueError("rebind artifact fingerprint does not reproduce")
        if len(self.artifacts) != self.counts.calls:
            raise ValueError("rebind artifact inventory does not equal call count")
        if len(self.sources) != self.counts.rebound_roots:
            raise ValueError("rebind source inventory does not equal root count")
        return self


class _LoadedPromotedSidecar(BaseModel):
    """Validated promoted sidecar held with machine-local artifact paths."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    manifest: TrialRunManifest
    experience: PromotedExperienceManifest
    promotion: NaturalnessPromotionReport
    artifacts: tuple[TrialSimulationArtifact, ...]
    artifact_paths: tuple[Path, ...]
    manifest_sha256: str
    experience_sha256: str
    promotion_sha256: str


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (repo_root / expanded).resolve()
    )


def _portable_relative(path: Path, root: Path, *, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be inside its portable reference root")
    relative = resolved.relative_to(root).as_posix()
    if relative in {"", "."}:
        raise ValueError(f"{label} must name a file or child directory")
    return relative


def _evidence_inventory(root: Path, *, label: str) -> dict[str, str]:
    """Return the exact non-manifest file inventory for one evidence root."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a real directory, not a symlink")
    entries = list(root.rglob("*"))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        relative = [path.relative_to(root).as_posix() for path in symlinks]
        raise ValueError(f"{label} contains unsafe symlinks: {sorted(relative)}")
    inventory = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(entries)
        if path.is_file() and path != root / "manifest.json"
    }
    for relative in inventory:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
            raise ValueError(f"{label} contains an unsafe file path: {relative}")
    return inventory


def _verify_archive_record(root: Path, record: ArchiveFile, *, label: str) -> Path:
    """Resolve and byte-verify one manifest-owned archive record."""
    relative = PurePosixPath(record.path)
    if relative.is_absolute() or ".." in relative.parts or record.path in {"", "."}:
        raise ValueError(f"{label} contains an unsafe manifest path")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} does not resolve to a safe regular file")
    if path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
        raise ValueError(f"{label} hash or byte count drifted")
    return path


def _read_projection_rows(
    root: Path, *, legacy: bool
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Validate typed rows and retain their exact CSV-string public projection."""
    projected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    public_fields = list(ValidationRow.model_fields)
    dropped = set(VALIDATION_DROPPED_FIELDS)
    paths = sorted(root.rglob("validation.csv"))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)):
                raise ValueError(f"{relative}: missing or duplicate CSV columns")
            if legacy:
                if set(fields) != set(public_fields) | dropped:
                    raise ValueError(
                        f"{relative}: private validation columns differ from contract"
                    )
                if [field for field in fields if field not in dropped] != public_fields:
                    raise ValueError(
                        f"{relative}: public validation column order drifted"
                    )
            elif fields != public_fields:
                raise ValueError(f"{relative}: public validation columns drifted")

            row_count = 0
            for raw in reader:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"{relative}: malformed CSV row")
                public = {field: raw[field] for field in public_fields}
                ValidationRow.model_validate(public)
                if legacy:
                    _LegacyValidationPrivateFields.model_validate(
                        {field: raw[field] for field in VALIDATION_DROPPED_FIELDS}
                    )
                projected.append({"partition": relative, "row": public})
                row_count += 1
            counts[relative] = row_count
    return projected, counts


def prove_sanitized_validation_evidence(
    config: ValidationEvidenceProjectionConfig,
    *,
    prior_evidence: ValidationEvidenceIdentity,
    contract: ValidationEvidenceProjectionContract = (
        CANONICAL_VALIDATION_PROJECTION_CONTRACT
    ),
) -> ValidationEvidenceProjectionProof:
    """Prove that v2 removes only private provenance from the inherited archive."""
    reference_root = config.validation_repo_root.expanduser().resolve()
    source_root = config.source_root.expanduser()
    sanitized_root = config.sanitized_root.expanduser()
    if source_root.is_symlink() or sanitized_root.is_symlink():
        raise ValueError("validation evidence roots may not be symlinks")
    source_root = source_root.resolve()
    sanitized_root = sanitized_root.resolve()
    sanitized_relative = _portable_relative(
        sanitized_root, reference_root, label="sanitized validation evidence"
    )
    if source_root == sanitized_root:
        raise ValueError("source and sanitized validation evidence must be separate")

    prior_path = PurePosixPath(prior_evidence.path)
    if (
        prior_path.is_absolute()
        or ".." in prior_path.parts
        or prior_evidence.path in {"", "."}
    ):
        raise ValueError("inherited validation evidence path is unsafe")

    source_manifest_path = source_root / "manifest.json"
    sanitized_manifest_path = sanitized_root / "manifest.json"
    if not source_manifest_path.is_file() or not sanitized_manifest_path.is_file():
        raise ValueError("validation evidence root is missing manifest.json")
    source_manifest_sha = sha256_file(source_manifest_path)
    sanitized_manifest_sha = sha256_file(sanitized_manifest_path)
    source_inventory = _evidence_inventory(source_root, label="source evidence")
    sanitized_inventory = _evidence_inventory(
        sanitized_root, label="sanitized evidence"
    )
    source_inventory_sha = _sha256_value(source_inventory)
    sanitized_inventory_sha = _sha256_value(sanitized_inventory)
    if (
        prior_evidence.manifest_sha256 != source_manifest_sha
        or prior_evidence.files != source_inventory
    ):
        raise ValueError("source archive does not equal inherited validation evidence")
    if source_manifest_sha != contract.source_manifest_sha256:
        raise ValueError("source validation manifest is not the approved artifact")
    if sanitized_manifest_sha != contract.sanitized_manifest_sha256:
        raise ValueError("sanitized validation manifest is not the approved artifact")
    if source_inventory_sha != contract.source_inventory_sha256:
        raise ValueError("source validation inventory fingerprint drifted")
    if sanitized_inventory_sha != contract.sanitized_inventory_sha256:
        raise ValueError("sanitized validation inventory fingerprint drifted")

    source_manifest = _LegacyValidationRootManifest.model_validate_json(
        source_manifest_path.read_text(encoding="utf-8")
    )
    sanitized_manifest = ValidationRootManifest.model_validate_json(
        sanitized_manifest_path.read_text(encoding="utf-8")
    )
    source_keys = [
        (row.evaluation_level, row.measure_id, row.language)
        for row in source_manifest.partitions
    ]
    sanitized_keys = [
        (row.evaluation_level.value, row.measure_id.value, row.language)
        for row in sanitized_manifest.partitions
    ]
    if source_keys != sanitized_keys or len(source_keys) != len(set(source_keys)):
        raise ValueError("validation partition identities or order changed")
    if (
        len(source_keys) != contract.partitions
        or source_manifest.rows != contract.rows
        or sanitized_manifest.rows != contract.rows
        or source_manifest.metric_rows != contract.metric_rows
        or sanitized_manifest.metric_rows != contract.metric_rows
    ):
        raise ValueError("validation root counts differ from the approved contract")

    _verify_archive_record(
        source_root, source_manifest.index, label="source validation index"
    )
    _verify_archive_record(
        sanitized_root, sanitized_manifest.index, label="sanitized validation index"
    )
    source_readme_path = _verify_archive_record(
        source_root, source_manifest.readme, label="source validation README"
    )
    sanitized_readme_path = _verify_archive_record(
        sanitized_root, sanitized_manifest.readme, label="sanitized validation README"
    )
    expected_readme = source_readme_path.read_text(encoding="utf-8").replace(
        "are row provenance", "identify sampling strata"
    )
    if expected_readme == source_readme_path.read_text(encoding="utf-8"):
        raise ValueError("source README lacks the expected private-provenance wording")
    if sanitized_readme_path.read_text(encoding="utf-8") != expected_readme:
        raise ValueError("sanitized README differs beyond the approved wording change")

    source_rows, source_row_counts = _read_projection_rows(source_root, legacy=True)
    sanitized_rows, sanitized_row_counts = _read_projection_rows(
        sanitized_root, legacy=False
    )
    if source_rows != sanitized_rows or source_row_counts != sanitized_row_counts:
        raise ValueError("sanitized validation rows change a public field or row order")
    source_projection_sha = _sha256_value(source_rows)
    sanitized_projection_sha = _sha256_value(sanitized_rows)
    if (
        len(source_rows) != contract.rows
        or source_projection_sha != contract.public_projection_sha256
        or sanitized_projection_sha != contract.public_projection_sha256
    ):
        raise ValueError("validation public projection fingerprint drifted")

    sanitized_by_key = {
        (row.evaluation_level.value, row.measure_id.value, row.language): row
        for row in sanitized_manifest.partitions
    }
    metrics_paths: set[str] = set()
    leaf_manifest_paths: set[str] = set()
    validation_paths: set[str] = set()
    total_metric_rows = 0
    for source_partition in source_manifest.partitions:
        key = (
            source_partition.evaluation_level,
            source_partition.measure_id,
            source_partition.language,
        )
        sanitized_partition = sanitized_by_key[key]
        if source_partition.manifest.path != sanitized_partition.manifest.path:
            raise ValueError("validation leaf manifest path changed")
        source_leaf_path = _verify_archive_record(
            source_root,
            source_partition.manifest,
            label=f"source validation leaf {key}",
        )
        sanitized_leaf_path = _verify_archive_record(
            sanitized_root,
            sanitized_partition.manifest,
            label=f"sanitized validation leaf {key}",
        )
        source_leaf = _LegacyValidationLeafManifest.model_validate_json(
            source_leaf_path.read_text(encoding="utf-8")
        )
        sanitized_leaf = ValidationLeafManifest.model_validate_json(
            sanitized_leaf_path.read_text(encoding="utf-8")
        )
        if (
            (
                source_leaf.evaluation_level,
                source_leaf.measure_id,
                source_leaf.language,
            )
            != (
                sanitized_leaf.evaluation_level.value,
                sanitized_leaf.measure_id.value,
                sanitized_leaf.language,
            )
            or source_leaf.rows != sanitized_leaf.rows
            or source_leaf.metric_rows != sanitized_leaf.metric_rows
            or source_leaf.metrics != sanitized_leaf.metrics
            or source_leaf.validation.path != sanitized_leaf.validation.path
            or source_leaf.validation.rows != sanitized_leaf.validation.rows
        ):
            raise ValueError(f"validation leaf changed beyond public projection: {key}")
        source_validation = _verify_archive_record(
            source_leaf_path.parent,
            source_leaf.validation,
            label=f"source validation rows {key}",
        )
        sanitized_validation = _verify_archive_record(
            sanitized_leaf_path.parent,
            sanitized_leaf.validation,
            label=f"sanitized validation rows {key}",
        )
        _verify_archive_record(
            source_leaf_path.parent,
            source_leaf.metrics,
            label=f"source validation metrics {key}",
        )
        _verify_archive_record(
            sanitized_leaf_path.parent,
            sanitized_leaf.metrics,
            label=f"sanitized validation metrics {key}",
        )
        source_validation_relative = source_validation.relative_to(
            source_root
        ).as_posix()
        sanitized_validation_relative = sanitized_validation.relative_to(
            sanitized_root
        ).as_posix()
        if source_validation_relative != sanitized_validation_relative:
            raise ValueError("validation CSV path changed during projection")
        if (
            source_row_counts.get(source_validation_relative) != source_leaf.rows
            or sanitized_row_counts.get(sanitized_validation_relative)
            != sanitized_leaf.rows
        ):
            raise ValueError(f"validation leaf row count drifted: {key}")
        total_metric_rows += source_leaf.metric_rows
        leaf_manifest_paths.add(source_partition.manifest.path)
        validation_paths.add(source_validation_relative)
        metrics_paths.add(
            (source_leaf_path.parent / source_leaf.metrics.path)
            .relative_to(source_root)
            .as_posix()
        )
    if total_metric_rows != contract.metric_rows:
        raise ValueError("validation leaf metric-row total drifted")

    public_rows, metric_rows, index_rows = read_validation_archive(
        sanitized_root.parent
    )
    if (
        len(public_rows) != contract.rows
        or len(metric_rows) != contract.metric_rows
        or len(index_rows) != contract.partitions
    ):
        raise ValueError("sanitized archive semantic validation counts drifted")

    source_paths = set(source_inventory)
    sanitized_paths = set(sanitized_inventory)
    removed_paths = source_paths - sanitized_paths
    if sanitized_paths - source_paths:
        raise ValueError("sanitized evidence introduced an extra file")
    if (
        len(removed_paths) != contract.removed_provenance_files
        or any(
            not path.startswith("provenance/") or not path.endswith(".json")
            for path in removed_paths
        )
        or any(path.startswith("provenance/") for path in sanitized_paths)
    ):
        raise ValueError("removed validation files are not exactly provenance JSON")

    prompt_source_paths = {
        path for path in sanitized_paths if path.startswith("prompt_sources/")
    }
    invariant_paths = metrics_paths | {
        "index.csv",
        "prompts.json",
        *prompt_source_paths,
    }
    if (
        len(prompt_source_paths) != contract.prompt_source_files
        or len(invariant_paths) != contract.invariant_files
        or not invariant_paths.issubset(source_paths & sanitized_paths)
        or any(
            source_inventory[path] != sanitized_inventory[path]
            for path in invariant_paths
        )
    ):
        raise ValueError("validation invariant-file inventory changed")
    allowed_changed = leaf_manifest_paths | validation_paths | {"README.md"}
    changed_common = {
        path
        for path in source_paths & sanitized_paths
        if source_inventory[path] != sanitized_inventory[path]
    }
    if changed_common != allowed_changed:
        raise ValueError("validation files changed outside the approved projection")
    if (
        len(source_inventory) != contract.source_files
        or len(sanitized_inventory) != contract.sanitized_files
    ):
        raise ValueError("validation evidence file count drifted")

    sanitized_evidence = ValidationEvidenceIdentity(
        path=sanitized_relative,
        manifest_sha256=sanitized_manifest_sha,
        files=sanitized_inventory,
    )
    return ValidationEvidenceProjectionProof(
        schema_version=VALIDATION_PROJECTION_SCHEMA_VERSION,
        prior_evidence=prior_evidence,
        sanitized_evidence=sanitized_evidence,
        dropped_fields=VALIDATION_DROPPED_FIELDS,
        source_inventory_sha256=source_inventory_sha,
        sanitized_inventory_sha256=sanitized_inventory_sha,
        source_projection_sha256=source_projection_sha,
        sanitized_projection_sha256=sanitized_projection_sha,
        invariant_inventory_sha256=_sha256_value(
            {path: sanitized_inventory[path] for path in sorted(invariant_paths)}
        ),
        removed_provenance_inventory_sha256=_sha256_value(
            {path: source_inventory[path] for path in sorted(removed_paths)}
        ),
        counts=ValidationEvidenceProjectionCounts(
            partitions=len(source_keys),
            rows=len(source_rows),
            metric_rows=total_metric_rows,
            prompt_source_files=len(prompt_source_paths),
            invariant_files=len(invariant_paths),
            removed_provenance_files=len(removed_paths),
            source_files=len(source_inventory),
            sanitized_files=len(sanitized_inventory),
        ),
    )


def load_active_naturalness_inventory(
    config: NaturalnessRebindConfig,
    *,
    contract: NaturalnessRebindContract = CANONICAL_REBIND_CONTRACT,
) -> NaturalnessActiveInventory:
    """Bind the current audit to the approved ten-cell replacement receipt."""
    repo_root = config.repo_root.expanduser().resolve()
    audit_path = _resolve_repo_path(repo_root, config.active_audit)
    replacement_path = _resolve_repo_path(repo_root, config.replacement_manifest)
    audit = MultilingualAudit.model_validate_json(audit_path.read_text())
    replacement = load_retail_replacement_manifest(replacement_path)
    audit_sha = sha256_file(audit_path)
    replacement_sha = sha256_file(replacement_path)
    if not audit.ok:
        raise ValueError("active audit contains a failing finding")
    if (
        replacement.replacement_audit.sha256 != audit_sha
        or replacement.replacement_audit.artifact_id != audit.artifact_id
    ):
        raise ValueError("replacement receipt is not bound to the active audit")

    cells = {(row.language, row.domain, row.system): row for row in audit.voice_cells}
    expected = {
        (language, domain, system)
        for language in contract.languages
        for domain in contract.domains
        for system in contract.system_slugs
    }
    if set(cells) != expected or len(cells) != len(expected):
        raise ValueError("active audit does not contain the complete voice grid")

    sources: list[ExperienceSource] = []
    for language in contract.languages:
        for domain in contract.domains:
            for system in contract.system_slugs:
                cell = cells[(language, domain, system)]
                logical = (CANONICAL_EVIDENCE_PREFIX / cell.results_path).as_posix()
                sources.append(
                    ExperienceSource(
                        path=logical,
                        sha256=cell.sha256,
                        trial_0_calls=contract.calls_per_root,
                    )
                )

    voice_replacements = [
        cell for cell in replacement.cells if cell.modality == "voice"
    ]
    affected: dict[str, str] = {}
    for cell in voice_replacements:
        logical = (CANONICAL_EVIDENCE_PREFIX / cell.canonical_path).as_posix()
        audit_cell = cells.get((cell.language, contract.affected_domain, cell.system))
        if audit_cell is None:
            raise ValueError(f"replacement cell is outside the active audit: {logical}")
        if audit_cell.sha256 != cell.replacement.results_sha256:
            raise ValueError(f"replacement and audit hashes differ: {logical}")
        affected[logical] = cell.replacement.results_sha256
    expected_affected = {
        (language, contract.affected_domain, system)
        for language in contract.affected_languages
        for system in contract.system_slugs
    }
    actual_affected = {
        (cell.language, contract.affected_domain, cell.system)
        for cell in voice_replacements
    }
    if actual_affected != expected_affected or len(affected) != contract.affected_roots:
        raise ValueError("replacement receipt does not name the exact rebind scope")
    return NaturalnessActiveInventory(
        audit=RebindInputReference(
            path=_portable_relative(audit_path, repo_root, label="active audit"),
            sha256=audit_sha,
            identity=audit.artifact_id,
        ),
        replacement=RebindInputReference(
            path=_portable_relative(
                replacement_path, repo_root, label="replacement manifest"
            ),
            sha256=replacement_sha,
            identity=replacement.replacement_id,
        ),
        sources=tuple(sources),
        affected_paths=tuple(sorted(affected)),
    )


def _artifact_plan(artifact: TrialSimulationArtifact) -> TrialSimulationPlan:
    return TrialSimulationPlan(
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


def _load_promoted_sidecar(
    root: Path, contract: NaturalnessRebindContract
) -> _LoadedPromotedSidecar:
    manifest_path = root / "manifest.json"
    experience_path = root / PROMOTED_EXPERIENCE_FILENAME
    promotion_path = root / PROMOTION_FILENAME
    for path in (manifest_path, experience_path, promotion_path):
        if not path.is_file():
            raise ValueError(f"promoted sidecar input is missing {path.name}")
    manifest = TrialRunManifest.model_validate_json(manifest_path.read_text())
    experience = PromotedExperienceManifest.model_validate_json(
        experience_path.read_text()
    )
    promotion = NaturalnessPromotionReport.model_validate_json(
        promotion_path.read_text()
    )
    manifest_sha = sha256_file(manifest_path)
    experience_sha = sha256_file(experience_path)
    promotion_sha = sha256_file(promotion_path)
    if manifest.identity_sha256 != _sha256_value(
        manifest.identity.model_dump(mode="json")
    ):
        raise ValueError("source sidecar identity hash does not reproduce")
    if (
        manifest.status != "complete"
        or manifest.counts.verified_roots != contract.all_roots
        or manifest.counts.selected_roots != contract.selected_roots
        or manifest.counts.simulations != contract.selected_calls
        or manifest.counts.complete_simulations != contract.selected_calls
        or manifest.counts.error_simulations
        or manifest.counts.error_utterances
        or manifest.aggregate is None
    ):
        raise ValueError("source sidecar is not the complete expected cohort")
    if (
        promotion.output_manifest_sha256 != manifest_sha
        or promotion.output_identity_sha256 != manifest.identity_sha256
        or promotion.promoted_experience_sha256 != experience_sha
        or promotion.counts.roots != contract.selected_roots
        or promotion.counts.calls != contract.selected_calls
    ):
        raise ValueError("promotion receipt is not bound to the source sidecar")
    if (
        experience.provenance.results_files != manifest.identity.all_sources
        or experience.provenance.cohort_fingerprint_sha256
        != manifest.identity.cohort_fingerprint_sha256
        or manifest.identity.experience_sha256 != experience_sha
    ):
        raise ValueError("source hybrid Experience differs from the run identity")
    if promotion.artifact_fingerprint_sha256 != _sha256_value(
        [row.model_dump(mode="json") for row in promotion.artifacts]
    ):
        raise ValueError("source promotion artifact fingerprint does not reproduce")
    if len(promotion.artifacts) != contract.selected_calls:
        raise ValueError("source promotion artifact count drifted")

    artifacts: list[TrialSimulationArtifact] = []
    artifact_paths: list[Path] = []
    expected_paths: set[Path] = set()
    for receipt in promotion.artifacts:
        relative = PurePosixPath(receipt.output_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source promotion contains an unsafe artifact path")
        path = (root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(root) or path in expected_paths:
            raise ValueError("source promotion artifact paths are not unique and safe")
        expected_paths.add(path)
        if not path.is_file() or sha256_file(path) != receipt.output_sha256:
            raise ValueError(f"source promotion artifact drifted: {relative}")
        artifact = TrialSimulationArtifact.model_validate_json(path.read_text())
        if (
            artifact.language != receipt.language
            or artifact.domain != receipt.domain
            or artifact.source.system_slug != receipt.system_slug
            or artifact.simulation_id != receipt.simulation_id
            or len(artifact.utterances) != receipt.utterances
        ):
            raise ValueError(f"source promotion receipt mismatch: {relative}")
        _validate_bootstrap_call(_artifact_plan(artifact), artifact, path)
        artifacts.append(artifact)
        artifact_paths.append(path)
    actual_paths = {path.resolve() for path in (root / "simulations").rglob("*.json")}
    if actual_paths != expected_paths:
        raise ValueError("source sidecar simulation inventory differs from receipt")
    if _aggregate_trial(artifacts) != manifest.aggregate:
        raise ValueError("source sidecar aggregate differs from artifacts")
    if sum(len(row.utterances) for row in artifacts) != manifest.counts.utterances:
        raise ValueError("source sidecar utterance count differs from artifacts")
    if (
        _canonical_work_fingerprint([_artifact_plan(row) for row in artifacts])
        != manifest.identity.work_fingerprint_sha256
    ):
        raise ValueError("source sidecar artifact order differs from work fingerprint")
    return _LoadedPromotedSidecar(
        manifest=manifest,
        experience=experience,
        promotion=promotion,
        artifacts=tuple(artifacts),
        artifact_paths=tuple(artifact_paths),
        manifest_sha256=manifest_sha,
        experience_sha256=experience_sha,
        promotion_sha256=promotion_sha,
    )


def _validate_inventory_diff(
    loaded: _LoadedPromotedSidecar,
    inventory: NaturalnessActiveInventory,
    contract: NaturalnessRebindContract,
) -> tuple[list[ExperienceSource], dict[str, ExperienceSource]]:
    current = {source.path: source for source in inventory.sources}
    previous = {source.path: source for source in loaded.manifest.identity.all_sources}
    if len(current) != contract.all_roots or set(current) != set(previous):
        raise ValueError("active and source-sidecar inventories cover different roots")
    affected = set(inventory.affected_paths)
    if len(affected) != contract.affected_roots:
        raise ValueError("active inventory does not name the expected affected roots")
    selected = {
        source.results_path: source
        for source in loaded.manifest.identity.selected_sources
    }
    expected_affected = {
        source.results_path
        for source in selected.values()
        if source.language in contract.affected_languages
        and source.domain == contract.affected_domain
        and source.system_slug in contract.system_slugs
    }
    if affected != expected_affected:
        raise ValueError("affected paths do not equal Korean/Mandarin retail")
    differences = {
        path for path in current if current[path].sha256 != previous[path].sha256
    }
    if differences != affected:
        raise ValueError("source-hash differences do not equal the approved scope")
    for path, source in current.items():
        if source.trial_0_calls != contract.calls_per_root:
            raise ValueError(f"active trial-zero count drifted: {path}")
    ordered = [current[source.path] for source in loaded.manifest.identity.all_sources]
    return ordered, current


def _current_source_identity(
    previous: TrialSourceIdentity, current: ExperienceSource
) -> TrialSourceIdentity:
    if (
        previous.results_path != current.path
        or previous.trial_0_calls != current.trial_0_calls
    ):
        raise ValueError("current result source changed path or trial-zero count")
    return previous.model_copy(update={"results_sha256": current.sha256})


def _active_results_path(root: Path, logical: str) -> Path:
    relative = PurePosixPath(logical)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.is_relative_to(PurePosixPath(CANONICAL_EVIDENCE_PREFIX))
    ):
        raise ValueError(f"unsafe active result path: {logical}")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"active result escapes its read-only root: {logical}")
    return path


def _rebind_one_source(
    *,
    active_results_root: Path,
    current: ExperienceSource,
    previous_source: TrialSourceIdentity,
    artifacts: list[tuple[TrialSimulationArtifact, Path]],
    contract: NaturalnessRebindContract,
) -> tuple[list[TrialSimulationArtifact], NaturalnessReboundSource]:
    results_path = _active_results_path(active_results_root, current.path)
    if not results_path.is_file() or sha256_file(results_path) != current.sha256:
        raise ValueError(f"current canonical results hash mismatch: {current.path}")
    metadata = Results.load_metadata(results_path)
    if metadata.simulation_index is None:
        raise ValueError(f"current canonical source has no index: {current.path}")
    entries = [row for row in metadata.simulation_index if row.trial == contract.trial]
    entry_ids = [row.id for row in entries]
    artifact_by_id = {row.simulation_id: (row, path) for row, path in artifacts}
    artifact_ids = list(artifact_by_id)
    if (
        len(entries) != contract.calls_per_root
        or len(entry_ids) != len(set(entry_ids))
        or set(entry_ids) != set(artifact_ids)
        or len(artifact_ids) != len(set(artifact_ids))
    ):
        raise ValueError(f"current trial-zero inventory drifted: {current.path}")
    task_ids = {str(task.id) for task in metadata.tasks}
    new_source = _current_source_identity(previous_source, current)
    rebound: list[TrialSimulationArtifact] = []
    simulation_inventory: list[tuple[str, str, str]] = []
    for simulation_id in entry_ids:
        artifact, artifact_path = artifact_by_id[simulation_id]
        sim_path = (
            results_path.parent / "simulations" / f"{artifact.simulation_id}.json"
        )
        raw = sim_path.read_bytes()
        sim_sha = hashlib.sha256(raw).hexdigest()
        if sim_sha != artifact.source_simulation_sha256:
            raise ValueError(
                "current simulation bytes differ from judged artifact: "
                f"{artifact.simulation_id}"
            )
        simulation = SimulationRun.model_validate_json(raw)
        language, _script = get_simulation_language_info(simulation)
        if (
            simulation.id != artifact.simulation_id
            or str(simulation.task_id) != artifact.task_id
            or simulation.trial != contract.trial
            or str(simulation.task_id) not in task_ids
            or language != artifact.language
        ):
            raise ValueError(
                f"current simulation identity drifted: {artifact.simulation_id}"
            )
        plan = TrialSimulationPlan(
            source=new_source,
            simulation_id=simulation.id,
            task_id=str(simulation.task_id),
            trial=0,
            language=language,
            domain=artifact.domain,
            source_simulation_sha256=sim_sha,
            judge=artifact.judge,
            turns=build_agent_turns(simulation),
        )
        rebound.append(_rekey_replacement(plan, artifact, artifact_path))
        simulation_inventory.append((simulation.id, str(simulation.task_id), sim_sha))
    old_fingerprint = _sha256_value(
        [row.model_dump(mode="json") for row, _path in artifacts]
    )
    new_fingerprint = _sha256_value([row.model_dump(mode="json") for row in rebound])
    return rebound, NaturalnessReboundSource(
        language=previous_source.language,
        domain=previous_source.domain,
        system_slug=previous_source.system_slug,
        results_path=current.path,
        source_key=previous_source.source_key,
        previous_results_sha256=previous_source.results_sha256,
        current_results_sha256=current.sha256,
        simulations=len(rebound),
        utterances=sum(len(row.utterances) for row in rebound),
        simulation_inventory_sha256=_sha256_value(simulation_inventory),
        previous_artifact_fingerprint_sha256=old_fingerprint,
        current_artifact_fingerprint_sha256=new_fingerprint,
    )


def _verify_existing_rebind(
    output_root: Path,
    *,
    loaded: _LoadedPromotedSidecar,
    inventory: NaturalnessActiveInventory,
    validation_evidence: ValidationEvidenceIdentity | None,
    validation_projection: ValidationEvidenceProjectionProof | None,
) -> NaturalnessRebindReport:
    receipt_path = output_root / REBIND_FILENAME
    if not receipt_path.is_file():
        raise ValueError("existing rebind output has no typed receipt")
    report = NaturalnessRebindReport.model_validate_json(receipt_path.read_text())
    output_manifest = TrialRunManifest.model_validate_json(
        (output_root / "manifest.json").read_text()
    )
    if (
        report.source_sidecar.sha256 != loaded.manifest_sha256
        or report.source_sidecar.identity != loaded.manifest.identity_sha256
        or report.source_promotion.sha256 != loaded.promotion_sha256
        or report.source_experience.sha256 != loaded.experience_sha256
        or report.active_audit != inventory.audit
        or report.replacement_manifest != inventory.replacement
        or report.validation_evidence_projection != validation_projection
        or output_manifest.identity.validation_evidence != validation_evidence
        or sha256_file(output_root / report.output_manifest_path)
        != report.output_manifest_sha256
        or sha256_file(output_root / report.output_experience_path)
        != report.output_experience_sha256
    ):
        raise ValueError("existing rebind output identity drifted")
    for artifact in report.artifacts:
        path = output_root / artifact.output_path
        if not path.is_file() or sha256_file(path) != artifact.output_sha256:
            raise ValueError(
                f"existing rebound artifact drifted: {artifact.output_path}"
            )
    return report


def rebind_promoted_trial0_naturalness(
    config: NaturalnessRebindConfig,
    *,
    contract: NaturalnessRebindContract = CANONICAL_REBIND_CONTRACT,
    inventory: NaturalnessActiveInventory | None = None,
    validation_projection_contract: ValidationEvidenceProjectionContract = (
        CANONICAL_VALIDATION_PROJECTION_CONTRACT
    ),
) -> NaturalnessRebindReport:
    """Re-key exact existing verdicts to current canonical result headers."""
    repo_root = config.repo_root.expanduser().resolve()
    active_results_root = _resolve_repo_path(repo_root, config.active_results_root)
    source_root = _resolve_repo_path(repo_root, config.sidecar_from)
    output_root = _resolve_repo_path(repo_root, config.output_root)
    if not active_results_root.is_dir():
        raise ValueError("active results root is not a directory")
    source_sidecar_relative = _portable_relative(
        source_root, active_results_root, label="source sidecar"
    )
    for input_root in (active_results_root, source_root):
        if (
            output_root == input_root
            or output_root.is_relative_to(input_root)
            or input_root.is_relative_to(output_root)
        ):
            raise ValueError("rebind output must be separate from every input tree")
    loaded = _load_promoted_sidecar(source_root, contract)
    validation_projection = None
    validation_evidence = loaded.manifest.identity.validation_evidence
    if config.validation_evidence_projection is not None:
        if validation_evidence is None:
            raise ValueError(
                "source sidecar has no inherited validation evidence to project"
            )
        validation_projection = prove_sanitized_validation_evidence(
            config.validation_evidence_projection,
            prior_evidence=validation_evidence,
            contract=validation_projection_contract,
        )
        validation_evidence = validation_projection.sanitized_evidence
    active = inventory or load_active_naturalness_inventory(config, contract=contract)
    audit_path = _resolve_repo_path(repo_root, Path(active.audit.path))
    replacement_path = _resolve_repo_path(repo_root, Path(active.replacement.path))
    if (
        not audit_path.is_file()
        or sha256_file(audit_path) != active.audit.sha256
        or not replacement_path.is_file()
        or sha256_file(replacement_path) != active.replacement.sha256
    ):
        raise ValueError("active inventory input hash drifted")
    ordered_all_sources, current_by_path = _validate_inventory_diff(
        loaded, active, contract
    )
    if output_root.exists():
        return _verify_existing_rebind(
            output_root,
            loaded=loaded,
            inventory=active,
            validation_evidence=validation_evidence,
            validation_projection=validation_projection,
        )

    old_sources = {
        source.results_path: source
        for source in loaded.manifest.identity.selected_sources
    }
    grouped: dict[str, list[tuple[TrialSimulationArtifact, Path]]] = {
        path: [] for path in old_sources
    }
    for artifact, path in zip(loaded.artifacts, loaded.artifact_paths, strict=True):
        grouped[artifact.source.results_path].append((artifact, path))
    if any(len(rows) != contract.calls_per_root for rows in grouped.values()):
        raise ValueError("source sidecar call count drifted within a result root")
    old_canonical_order = [
        row
        for source in loaded.manifest.identity.selected_sources
        for row, _path in grouped[source.results_path]
    ]
    if (
        _canonical_work_fingerprint(
            [_artifact_plan(row) for row in old_canonical_order]
        )
        != loaded.manifest.identity.work_fingerprint_sha256
    ):
        raise ValueError("source sidecar selected-source ordering drifted")

    rebound_by_id: dict[tuple[str, str], TrialSimulationArtifact] = {}
    rebound_by_source: dict[str, list[TrialSimulationArtifact]] = {}
    source_proofs: list[NaturalnessReboundSource] = []
    for logical in active.affected_paths:
        rebound, proof = _rebind_one_source(
            active_results_root=active_results_root,
            current=current_by_path[logical],
            previous_source=old_sources[logical],
            artifacts=grouped[logical],
            contract=contract,
        )
        source_proofs.append(proof)
        rebound_by_source[logical] = rebound
        rebound_by_id.update(
            {(row.source.results_path, row.simulation_id): row for row in rebound}
        )
    if len(rebound_by_id) != contract.affected_calls:
        raise ValueError("rebound call inventory does not equal the approved scope")

    current_selected = [
        _current_source_identity(source, current_by_path[source.results_path])
        for source in loaded.manifest.identity.selected_sources
    ]
    cohort_fingerprint = _cohort_fingerprint(ordered_all_sources)
    experience = PromotedExperienceManifest(
        instrument=loaded.experience.instrument,
        instrument_version=loaded.experience.instrument_version,
        cohort=loaded.experience.cohort,
        provenance=PromotedExperienceProvenance(
            cohort_fingerprint_sha256=cohort_fingerprint,
            results_files=ordered_all_sources,
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
        artifact_receipts: list[NaturalnessReboundArtifact] = []
        for old_artifact, old_path, old_receipt in zip(
            loaded.artifacts,
            loaded.artifact_paths,
            loaded.promotion.artifacts,
            strict=True,
        ):
            key = (old_artifact.source.results_path, old_artifact.simulation_id)
            artifact = rebound_by_id.get(key, old_artifact)
            output_path = temporary / old_receipt.output_path
            if key in rebound_by_id:
                _atomic_write(output_path, artifact)
                operation: Literal["copied", "rebound"] = "rebound"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old_path, output_path)
                operation = "copied"
            artifacts.append(artifact)
            artifact_receipts.append(
                NaturalnessReboundArtifact(
                    operation=operation,
                    language=artifact.language,
                    domain=artifact.domain,
                    system_slug=artifact.source.system_slug,
                    simulation_id=artifact.simulation_id,
                    utterances=len(artifact.utterances),
                    input_path=old_receipt.output_path,
                    input_sha256=sha256_file(old_path),
                    output_path=old_receipt.output_path,
                    output_sha256=sha256_file(output_path),
                )
            )

        canonical_work_artifacts: list[TrialSimulationArtifact] = []
        for source in current_selected:
            canonical_work_artifacts.extend(
                rebound_by_source.get(
                    source.results_path,
                    [row for row, _path in grouped[source.results_path]],
                )
            )
        if len(canonical_work_artifacts) != contract.selected_calls:
            raise ValueError("canonical work-order call inventory drifted")
        plans = [_artifact_plan(row) for row in canonical_work_artifacts]
        identity = TrialRunIdentity(
            schema_version=loaded.manifest.identity.schema_version,
            experience_path=loaded.manifest.identity.experience_path,
            experience_sha256=experience_sha,
            cohort_fingerprint_sha256=cohort_fingerprint,
            work_fingerprint_sha256=_canonical_work_fingerprint(plans),
            validation_evidence=validation_evidence,
            runtime_source_fingerprint_sha256=(
                loaded.manifest.identity.runtime_source_fingerprint_sha256
            ),
            all_sources=ordered_all_sources,
            selected_sources=current_selected,
            judges=loaded.manifest.identity.judges,
            trial=0,
        )
        identity_sha = _sha256_value(identity.model_dump(mode="json"))
        utterances = sum(len(row.utterances) for row in artifacts)
        zero_utterance_calls = sum(not row.utterances for row in artifacts)
        aggregate = _aggregate_trial(artifacts)
        if aggregate != loaded.manifest.aggregate:
            raise ValueError("naturalness verdicts changed during canonical rebind")
        manifest = TrialRunManifest(
            schema_version=loaded.manifest.schema_version,
            created_at=loaded.manifest.created_at,
            updated_at=loaded.manifest.updated_at,
            status="complete",
            identity_sha256=identity_sha,
            identity=identity,
            counts=TrialCounts(
                verified_roots=contract.all_roots,
                selected_roots=contract.selected_roots,
                simulations=len(artifacts),
                utterances=utterances,
                zero_utterance_simulations=zero_utterance_calls,
                complete_utterances=utterances,
                error_utterances=0,
                complete_simulations=len(artifacts),
                error_simulations=0,
            ),
            aggregate=aggregate,
            invocations=[],
        )
        manifest_path = temporary / "manifest.json"
        _atomic_write(manifest_path, manifest)
        rebound_utterances = sum(row.utterances for row in source_proofs)
        counts = NaturalnessRebindCounts(
            all_roots=contract.all_roots,
            selected_roots=contract.selected_roots,
            calls=contract.selected_calls,
            utterances=utterances,
            rebound_roots=contract.affected_roots,
            rebound_calls=contract.affected_calls,
            rebound_utterances=rebound_utterances,
            copied_roots=contract.selected_roots - contract.affected_roots,
            copied_calls=contract.selected_calls - contract.affected_calls,
            copied_utterances=utterances - rebound_utterances,
            zero_utterance_calls=zero_utterance_calls,
        )
        artifact_fingerprint = _sha256_value(
            [row.model_dump(mode="json") for row in artifact_receipts]
        )
        report = NaturalnessRebindReport(
            schema_version=REBIND_SCHEMA_VERSION,
            source_sidecar=RebindInputReference(
                path=source_sidecar_relative,
                sha256=loaded.manifest_sha256,
                identity=loaded.manifest.identity_sha256,
            ),
            source_promotion=RebindInputReference(
                path=(
                    PurePosixPath(source_sidecar_relative) / PROMOTION_FILENAME
                ).as_posix(),
                sha256=loaded.promotion_sha256,
                identity=loaded.promotion.output_identity_sha256,
            ),
            source_experience=RebindInputReference(
                path=(
                    PurePosixPath(source_sidecar_relative)
                    / PROMOTED_EXPERIENCE_FILENAME
                ).as_posix(),
                sha256=loaded.experience_sha256,
                identity=loaded.experience.provenance.cohort_fingerprint_sha256,
            ),
            active_audit=active.audit,
            replacement_manifest=active.replacement,
            output_manifest_path="manifest.json",
            output_manifest_sha256=sha256_file(manifest_path),
            output_identity_sha256=identity_sha,
            output_experience_path=PROMOTED_EXPERIENCE_FILENAME,
            output_experience_sha256=experience_sha,
            validation_evidence_projection=validation_projection,
            artifact_fingerprint_sha256=artifact_fingerprint,
            counts=counts,
            sources=tuple(sorted(source_proofs, key=lambda row: row.results_path)),
            artifacts=tuple(artifact_receipts),
        )
        _atomic_write(temporary / REBIND_FILENAME, report)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
