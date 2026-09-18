# Copyright Sierra
"""Corrected-cohort adapter for the frozen τ-Multilingual naturalness judge.

The canonical paper replay in :mod:`tau2.judges.nativeness.paper_trial` remains
the immutable source contract.  This module selects a separately locked hybrid
cohort, proves that every unchanged call has an exact completed canonical call
artifact, and pays only for the corrected Korean and Mandarin retail calls.
It never writes to a results root or to the canonical bootstrap sidecar.

Build the lock only after every source-mutating generic-judge merge has
finished.  The lock deliberately hashes complete results and simulation JSON
bytes; no source may change between locking and the corrected replay.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY
from tau2.data_model.simulation import JudgeOutcome, Results, SimulationRun
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.nativeness.harness import AgentTurn, build_agent_turns
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EXPERIENCE_INSTRUMENT_VERSION,
    CANONICAL_EXPERIENCE_SHA256,
    RUN_LOCK_FILENAME,
    ExperienceManifest,
    ExperienceSource,
    FrozenJudgeSpec,
    TrialAggregateSummary,
    TrialCodeProvenance,
    TrialSimulationArtifact,
    TrialSimulationPlan,
    TrialSourceIdentity,
    TrialUtteranceArtifact,
    TrialUtterancePlan,
    ValidationEvidenceIdentity,
    _aggregate_simulation,
    _aggregate_trial,
    _atomic_write,
    _bounded_map,
    _code_provenance,
    _load_existing_simulation,
    _load_experience,
    _load_matching_utterance,
    _score_trial_utterance,
    _sha256_value,
    _simulation_path,
    _unit_digest,
    _utterance_path,
    _validated_evidence_identity,
    frozen_judge_spec,
)
from tau2.judges.nativeness.validation import sha256_file
from tau2.runner.run_lock import claim_run_directories

CORRECTED_COHORT_SCHEMA_VERSION = "tau-multi-naturalness-corrected-cohort-v1"
CORRECTED_RUN_SCHEMA_VERSION = "tau-multi-naturalness-corrected-trial0-v1"
CORRECTED_INSTRUMENT = "tau-multilingual-corrected-naturalness-cohort"
CORRECTED_LANGUAGES = ("es", "pt", "hi", "ko", "zh")
CORRECTED_DOMAINS = ("airline", "retail", "telecom")
CORRECTED_SYSTEM_SLUGS = (
    "openai_minimal",
    "openai_xhigh",
    "gemini_minimal",
    "gemini_high",
    "xai_provider_default",
)
CORRECTED_REPLACEMENT_LANGUAGES = ("ko", "zh")
CORRECTED_REPLACEMENT_DOMAIN = "retail"
CORRECTED_EXPECTED_ROOTS = 75
CORRECTED_EXPECTED_CALLS = 3_750
CORRECTED_EXPECTED_REUSED_ROOTS = 65
CORRECTED_EXPECTED_REUSED_CALLS = 3_250
CORRECTED_EXPECTED_PENDING_ROOTS = 10
CORRECTED_EXPECTED_PENDING_CALLS = 500
REPLACEMENT_ONLY_EXPECTED_ROOTS = 10
REPLACEMENT_ONLY_EXPECTED_CALLS = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorrectedCohortContract(BaseModel):
    """Closed hybrid-cohort dimensions (injectable only by offline tests)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    languages: Annotated[
        tuple[str, ...], Field(description="Non-English reporting languages.")
    ]
    domains: Annotated[tuple[str, ...], Field(description="Benchmark domains.")]
    system_slugs: Annotated[
        tuple[str, ...], Field(description="Canonical system slugs.")
    ]
    replacement_languages: Annotated[
        tuple[str, ...], Field(description="Languages whose retail roots changed.")
    ]
    replacement_domain: Annotated[
        str, Field(description="The single corrected benchmark domain.")
    ]
    trial: Annotated[int, Field(ge=0, description="Selected trial index.")]
    calls_per_root: Annotated[
        int, Field(ge=1, description="Required calls in every results root.")
    ]

    @property
    def roots(self) -> int:
        return len(self.languages) * len(self.domains) * len(self.system_slugs)

    @property
    def calls(self) -> int:
        return self.roots * self.calls_per_root

    @property
    def pending_roots(self) -> int:
        return len(self.replacement_languages) * len(self.system_slugs)

    @property
    def pending_calls(self) -> int:
        return self.pending_roots * self.calls_per_root

    @property
    def reused_roots(self) -> int:
        return self.roots - self.pending_roots

    @property
    def reused_calls(self) -> int:
        return self.calls - self.pending_calls

    def is_replacement(self, language: str, domain: str) -> bool:
        """Whether one grid cell belongs to the explicitly corrected slice."""
        return (
            language in self.replacement_languages and domain == self.replacement_domain
        )


CANONICAL_CORRECTED_COHORT_CONTRACT = CorrectedCohortContract(
    languages=CORRECTED_LANGUAGES,
    domains=CORRECTED_DOMAINS,
    system_slugs=CORRECTED_SYSTEM_SLUGS,
    replacement_languages=CORRECTED_REPLACEMENT_LANGUAGES,
    replacement_domain=CORRECTED_REPLACEMENT_DOMAIN,
    trial=0,
    calls_per_root=50,
)

REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT = CorrectedCohortContract(
    languages=CORRECTED_REPLACEMENT_LANGUAGES,
    domains=(CORRECTED_REPLACEMENT_DOMAIN,),
    system_slugs=CORRECTED_SYSTEM_SLUGS,
    replacement_languages=CORRECTED_REPLACEMENT_LANGUAGES,
    replacement_domain=CORRECTED_REPLACEMENT_DOMAIN,
    trial=0,
    calls_per_root=50,
)


def _uses_canonical_corrected_contract(contract: CorrectedCohortContract) -> bool:
    return contract == CANONICAL_CORRECTED_COHORT_CONTRACT


def _uses_replacement_only_contract(contract: CorrectedCohortContract) -> bool:
    return contract == REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT


def _uses_production_contract(contract: CorrectedCohortContract) -> bool:
    return _uses_canonical_corrected_contract(
        contract
    ) or _uses_replacement_only_contract(contract)


class CorrectedCohortDimensions(BaseModel):
    """Shape declared by the explicit corrected Experience manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: Annotated[int, Field(ge=0, description="Total selected calls.")]
    roots: Annotated[int, Field(ge=0, description="Total selected results roots.")]
    languages: Annotated[list[str], Field(description="Selected languages.")]
    domains: Annotated[list[str], Field(description="Selected domains.")]
    system_slugs: Annotated[list[str], Field(description="Selected systems.")]
    trial: Annotated[int, Field(ge=0, description="Selected trial index.")]
    calls_per_root: Annotated[
        int, Field(ge=1, description="Selected calls in every results root.")
    ]


class CorrectedCallLock(BaseModel):
    """Byte- and transcript-locked identity for one selected call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: Annotated[str, Field(min_length=1, description="Simulation id.")]
    task_id: Annotated[str, Field(min_length=1, description="Task id.")]
    trial: Literal[0]
    simulation_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Simulation JSON hash.")
    ]
    transcript_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Hash of the exact delivered agent turns judged.",
        ),
    ]


class CorrectedSourceLock(BaseModel):
    """One explicitly located and hashed results root in the hybrid cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[
        str, Field(min_length=1, description="Path relative to the evidence root.")
    ]
    results_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="results.json hash.")
    ]
    language: Annotated[str, Field(description="ISO language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    system_slug: Annotated[str, Field(description="Canonical system slug.")]
    origin: Annotated[
        Literal["canonical", "corrected"],
        Field(description="Whether this root is reused or newly corrected."),
    ]
    calls: Annotated[list[CorrectedCallLock], Field(description="Selected calls.")]


class CorrectedExperienceManifest(BaseModel):
    """Complete external lock for the hybrid trial-0 naturalness cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-corrected-cohort-v1"]
    instrument: Literal["tau-multilingual-corrected-naturalness-cohort"]
    cohort: Annotated[
        CorrectedCohortDimensions, Field(description="Closed cohort dimensions.")
    ]
    cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Manifest work hash.")
    ]
    sources: Annotated[
        list[CorrectedSourceLock], Field(description="Exact source inventory.")
    ]


class ResolvedCorrectedSource(BaseModel):
    """A locked source plus its machine-local resolution."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    lock: Annotated[CorrectedSourceLock, Field(description="Serialized source lock.")]
    identity: Annotated[
        TrialSourceIdentity, Field(description="Artifact source identity.")
    ]
    resolved_path: Annotated[Path, Field(description="Resolved results.json path.")]


class CorrectedTrialPlan(BaseModel):
    """Verified judge plan for all 3,750 non-English calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experience_manifest_path: Annotated[
        str, Field(description="Resolved explicit manifest path.")
    ]
    experience_manifest_sha256: Annotated[
        str, Field(description="Explicit manifest byte hash.")
    ]
    cohort_fingerprint_sha256: Annotated[
        str, Field(description="Reproduced manifest fingerprint.")
    ]
    validation_evidence: Annotated[
        Optional[ValidationEvidenceIdentity],
        Field(description="Frozen v16/rubric-v20 validation evidence."),
    ]
    sources: Annotated[list[CorrectedSourceLock], Field(description="Source locks.")]
    simulations: Annotated[
        list[TrialSimulationPlan], Field(description="All selected call plans.")
    ]
    utterances: Annotated[
        list[TrialUtterancePlan], Field(description="All selected paid units.")
    ]


class BootstrapCallArtifact(BaseModel):
    """Hash and identity of one exact canonical call artifact being reused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results_path: Annotated[str, Field(description="Locked source results path.")]
    simulation_id: Annotated[str, Field(description="Reused simulation id.")]
    artifact_path: Annotated[
        str, Field(description="Path relative to the canonical bootstrap root.")
    ]
    artifact_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Artifact byte hash.")
    ]
    utterances: Annotated[int, Field(ge=0, description="Reused paid units.")]


class BootstrapIdentity(BaseModel):
    """Canonical sidecar and exact subset accepted for corrected reuse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Resolved canonical sidecar root.")]
    manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Canonical manifest hash.")
    ]
    identity_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Canonical run identity.")
    ]
    call_artifact_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Accepted-artifact hash.")
    ]
    call_artifacts: Annotated[
        list[BootstrapCallArtifact], Field(description="Accepted call artifacts.")
    ]


class CorrectedPreparationCounts(BaseModel):
    """No-cost corrected-cohort work counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    roots: Annotated[int, Field(ge=0, description="All locked roots.")]
    calls: Annotated[int, Field(ge=0, description="All locked calls.")]
    utterances: Annotated[int, Field(ge=0, description="All utterance units.")]
    reused_roots: Annotated[int, Field(ge=0, description="Canonical roots reused.")]
    reused_calls: Annotated[int, Field(ge=0, description="Canonical calls reused.")]
    reused_utterances: Annotated[
        int, Field(ge=0, description="Canonical utterance artifacts reused.")
    ]
    pending_roots: Annotated[int, Field(ge=0, description="Corrected roots to judge.")]
    pending_calls: Annotated[int, Field(ge=0, description="Corrected calls to judge.")]
    pending_utterances: Annotated[
        int, Field(ge=0, description="Exact new paid judge-call count.")
    ]
    zero_utterance_pending_calls: Annotated[
        int, Field(ge=0, description="Pending calls with no delivered agent turn.")
    ]


class CorrectedCodeProvenance(BaseModel):
    """Frozen judge implementation plus this cohort adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frozen_judge: Annotated[
        TrialCodeProvenance, Field(description="Canonical judge provenance.")
    ]
    adapter_path: Annotated[str, Field(description="Corrected adapter source path.")]
    adapter_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Adapter source hash.")
    ]
    source_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Combined source hash.")
    ]


class CorrectedPreparationReport(BaseModel):
    """Typed, API-free corrected-cohort preflight report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-corrected-trial0-v1"]
    experience_manifest_path: Annotated[str, Field(description="Manifest path.")]
    experience_manifest_sha256: Annotated[str, Field(description="Manifest hash.")]
    cohort_fingerprint_sha256: Annotated[str, Field(description="Cohort hash.")]
    work_fingerprint_sha256: Annotated[str, Field(description="Judge work hash.")]
    validation_evidence: Annotated[
        Optional[ValidationEvidenceIdentity], Field(description="Validation evidence.")
    ]
    code: Annotated[CorrectedCodeProvenance, Field(description="Runtime code.")]
    bootstrap: Annotated[
        Optional[BootstrapIdentity],
        Field(description="Canonical reuse, absent for replacement-only output."),
    ]
    counts: Annotated[
        CorrectedPreparationCounts, Field(description="Verified work counts.")
    ]
    judges: Annotated[
        dict[str, FrozenJudgeSpec], Field(description="Frozen judge by language.")
    ]
    pending_calls_by_language_system: Annotated[
        dict[str, dict[str, int]],
        Field(description="Pending calls by replacement language and system."),
    ]
    pending_utterances_by_language_system: Annotated[
        dict[str, dict[str, int]],
        Field(description="Exact paid units by replacement language and system."),
    ]


class CorrectedRunIdentity(BaseModel):
    """Static identity that must match before corrected work resumes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-corrected-trial0-v1"]
    experience_manifest_path: str
    experience_manifest_sha256: str
    cohort_fingerprint_sha256: str
    work_fingerprint_sha256: str
    validation_evidence: Optional[ValidationEvidenceIdentity]
    code: CorrectedCodeProvenance
    bootstrap: Optional[BootstrapIdentity]
    sources: list[CorrectedSourceLock]
    judges: dict[str, FrozenJudgeSpec]
    trial: Literal[0]


class CorrectedInvocation(BaseModel):
    """One resumable corrected-run invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    finished_at: str
    max_concurrency: int
    processes: int = 1
    judged_utterances: int
    reused_utterances: int
    error_utterances: int


class CorrectedRunCounts(BaseModel):
    """Fixed cohort and current corrected-output completion counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    roots: int
    calls: int
    utterances: int
    reused_roots: int
    reused_calls: int
    reused_utterances: int
    pending_roots: int
    pending_calls: int
    pending_utterances: int
    judged_utterances: int
    error_utterances: int
    complete_calls: int
    error_calls: int


class CorrectedRunManifest(BaseModel):
    """Standalone corrected sidecar provenance and aggregate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-corrected-trial0-v1"]
    created_at: str
    updated_at: str
    status: Literal["running", "complete", "complete_with_errors", "source_drift"]
    identity_sha256: str
    identity: CorrectedRunIdentity
    counts: CorrectedRunCounts
    aggregate: Optional[TrialAggregateSummary]
    invocations: list[CorrectedInvocation]

    @model_validator(mode="after")
    def _completed_has_aggregate(self) -> "CorrectedRunManifest":
        if self.status != "running" and self.aggregate is None:
            raise ValueError("completed corrected run must include an aggregate")
        return self


class CorrectedTrialRunConfig(BaseModel):
    """Validated configuration for a corrected trial-0 replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Annotated[Path, Field(description="Judge-code repository root.")]
    experience_manifest: Annotated[
        Path, Field(description="Explicit typed corrected Experience manifest.")
    ]
    evidence_root: Annotated[
        Path, Field(description="Read-only root resolving every manifest source.")
    ]
    bootstrap_from: Annotated[
        Path, Field(description="Completed canonical naturalness sidecar root.")
    ]
    output_root: Annotated[Path, Field(description="Standalone corrected output.")]
    max_concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
            description="Maximum in-flight frozen judge calls.",
        ),
    ]
    processes: Annotated[
        int, Field(ge=1, default=1, description="Isolated scoring worker processes.")
    ]


class ReplacementOnlyTrialRunConfig(BaseModel):
    """Validated configuration for the standalone 500-call replacement replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Annotated[Path, Field(description="Judge-code repository root.")]
    experience_manifest: Annotated[
        Path, Field(description="Explicit typed replacement-only manifest.")
    ]
    evidence_root: Annotated[
        Path, Field(description="Read-only root resolving the ten locked sources.")
    ]
    output_root: Annotated[
        Path, Field(description="Standalone replacement-only output.")
    ]
    max_concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
            description="Maximum in-flight frozen judge calls.",
        ),
    ]
    processes: Annotated[
        int, Field(ge=1, default=1, description="Isolated scoring worker processes.")
    ]


class _PreparedBundle(BaseModel):
    """Internal validated plan plus loaded canonical call artifacts."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plan: CorrectedTrialPlan
    bootstrap: Optional[BootstrapIdentity]
    bootstrap_artifacts: dict[str, TrialSimulationArtifact]
    report: CorrectedPreparationReport


def corrected_manifest_fingerprint(cohort: object, sources: object) -> str:
    """Hash exactly the typed cohort dimensions and ordered source inventory."""
    cohort_model = (
        cohort
        if isinstance(cohort, CorrectedCohortDimensions)
        else CorrectedCohortDimensions.model_validate(cohort)
    )
    source_models = [
        row
        if isinstance(row, CorrectedSourceLock)
        else CorrectedSourceLock.model_validate(row)
        for row in sources  # type: ignore[union-attr]
    ]
    return _sha256_value(
        {
            "cohort": cohort_model.model_dump(mode="json"),
            "sources": [row.model_dump(mode="json") for row in source_models],
        }
    )


def transcript_sha256(turns: list[AgentTurn]) -> str:
    """Hash the exact delivered agent-turn payload consumed by the judge."""
    return _sha256_value([turn.model_dump(mode="json") for turn in turns])


def _source_for_cell(
    experience: ExperienceManifest,
    language: str,
    domain: str,
    system_slug: str,
) -> ExperienceSource:
    expected_leaf = f"{language}_{domain}_{system_slug}"
    matches = [
        source
        for source in experience.results_files
        if Path(source.path).parent.name == expected_leaf
    ]
    if len(matches) != 1:
        raise ValueError(f"canonical Experience source grid mismatch: {expected_leaf}")
    _safe_relative_path(matches[0].path, label="canonical source")
    return matches[0]


def _replacement_cell(
    path: Path, contract: CorrectedCohortContract
) -> tuple[str, str, str]:
    if path.name != "results.json":
        raise ValueError(f"replacement source is not results.json: {path}")
    leaf = path.parent.name
    matches = [
        (language, contract.replacement_domain, system)
        for language in contract.replacement_languages
        for system in contract.system_slugs
        if leaf == f"{language}_{contract.replacement_domain}_{system}"
    ]
    if len(matches) != 1:
        raise ValueError(f"replacement source is outside the closed grid: {path}")
    return matches[0]


def _validate_production_replacement_root(
    path: Path,
    language: str,
    system_slug: str,
) -> None:
    language_name = {"ko": "korean", "zh": "mandarin"}[language]
    wrapper = path.parent.parent.name
    expected_wrapper = (
        f"retail_name_roles_xai_v1_{language_name}_retail"
        if system_slug == "xai_provider_default"
        else f"retail_name_roles_v1_{language_name}_retail"
    )
    if wrapper != expected_wrapper:
        raise ValueError(
            "replacement source must be a completed corrected main-run root, "
            f"not smoke/text/ablation output: {path}"
        )


def _validate_production_results_metadata(
    metadata: Results,
    *,
    domain: str,
    system_slug: str,
) -> None:
    info = metadata.info
    if info.environment_info.domain_name != domain:
        raise ValueError("replacement source domain metadata drifted")
    if info.task_subset is None or (
        info.task_subset.name != f"{domain}_50"
        or info.task_subset.size != 50
        or info.task_subset.tasks_scored != 50
    ):
        raise ValueError("replacement source does not use the frozen 50-task subset")
    audio = info.audio_native_config
    expected_arms = {
        "openai_minimal": ("openai", "gpt-realtime-2", "minimal"),
        "openai_xhigh": ("openai", "gpt-realtime-2", "xhigh"),
        "gemini_minimal": (
            "gemini",
            "gemini-3.1-flash-live-preview",
            "minimal",
        ),
        "gemini_high": ("gemini", "gemini-3.1-flash-live-preview", "high"),
        "xai_provider_default": (
            "xai",
            "grok-voice-think-fast-1.0",
            "provider_default",
        ),
    }
    observed = (
        audio.provider if audio is not None else None,
        audio.model if audio is not None else None,
        audio.reasoning_effort if audio is not None else None,
    )
    if observed != expected_arms[system_slug]:
        raise ValueError(
            f"replacement source is not the frozen {system_slug} voice arm: {observed}"
        )


def _lock_one_source(
    actual_results_path: Path,
    *,
    logical_path: str,
    language: str,
    domain: str,
    system_slug: str,
    origin: Literal["canonical", "corrected"],
    contract: CorrectedCohortContract,
) -> CorrectedSourceLock:
    actual_results_path = actual_results_path.expanduser().resolve()
    if not actual_results_path.is_file():
        raise ValueError(f"cohort results source is missing: {actual_results_path}")
    metadata = Results.load_metadata(actual_results_path)
    if origin == "corrected" and _uses_production_contract(contract):
        _validate_production_replacement_root(
            actual_results_path, language, system_slug
        )
        _validate_production_results_metadata(
            metadata, domain=domain, system_slug=system_slug
        )
    if metadata.simulation_index is None:
        raise ValueError(f"cohort results source has no index: {actual_results_path}")
    entries = [
        entry for entry in metadata.simulation_index if entry.trial == contract.trial
    ]
    if len(entries) != contract.calls_per_root:
        raise ValueError(
            f"cohort results source has {len(entries)} trial-0 calls, expected "
            f"{contract.calls_per_root}: {actual_results_path}"
        )
    tasks = {str(task.id) for task in metadata.tasks}
    calls: list[CorrectedCallLock] = []
    for entry in entries:
        sim_path = actual_results_path.parent / "simulations" / f"{entry.id}.json"
        sim_path = sim_path.resolve()
        if not sim_path.is_relative_to(
            (actual_results_path.parent / "simulations").resolve()
        ):
            raise ValueError(f"cohort simulation path escapes source: {sim_path}")
        raw = sim_path.read_bytes()
        simulation = SimulationRun.model_validate_json(raw)
        if (
            simulation.id != entry.id
            or simulation.trial != contract.trial
            or str(simulation.task_id) not in tasks
        ):
            raise ValueError(f"cohort simulation identity mismatch: {sim_path}")
        simulation_language, _script = get_simulation_language_info(simulation)
        if simulation_language != language:
            raise ValueError(f"cohort simulation language mismatch: {sim_path}")
        turns = build_agent_turns(simulation)
        calls.append(
            CorrectedCallLock(
                simulation_id=simulation.id,
                task_id=str(simulation.task_id),
                trial=0,
                simulation_sha256=hashlib.sha256(raw).hexdigest(),
                transcript_sha256=transcript_sha256(turns),
            )
        )
    return CorrectedSourceLock(
        path=logical_path,
        results_sha256=sha256_file(actual_results_path),
        language=language,
        domain=domain,
        system_slug=system_slug,
        origin=origin,
        calls=calls,
    )


def plan_corrected_cohort_lock(
    *,
    canonical_experience: Path,
    canonical_evidence_root: Path,
    replacement_results: list[Path],
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
) -> tuple[CorrectedExperienceManifest, dict[str, Path]]:
    """Plan the exact hybrid lock and source mapping without writing artifacts."""
    canonical_experience = canonical_experience.expanduser().resolve()
    canonical_evidence_root = canonical_evidence_root.expanduser().resolve()
    experience = _load_experience(canonical_experience)
    if _uses_canonical_corrected_contract(contract) and (
        sha256_file(canonical_experience) != CANONICAL_EXPERIENCE_SHA256
        or experience.instrument_version != CANONICAL_EXPERIENCE_INSTRUMENT_VERSION
    ):
        raise ValueError("canonical Experience artifact drifted")
    replacement_by_cell: dict[tuple[str, str, str], Path] = {}
    for raw_path in replacement_results:
        path = raw_path.expanduser().resolve()
        cell = _replacement_cell(path, contract)
        if cell in replacement_by_cell:
            raise ValueError(f"duplicate replacement source cell: {cell}")
        replacement_by_cell[cell] = path
    expected_replacements = {
        (language, contract.replacement_domain, system)
        for language in contract.replacement_languages
        for system in contract.system_slugs
    }
    if set(replacement_by_cell) != expected_replacements:
        missing = sorted(expected_replacements - set(replacement_by_cell))
        extra = sorted(set(replacement_by_cell) - expected_replacements)
        raise ValueError(
            f"replacement source grid mismatch; missing={missing}, extra={extra}"
        )

    # Validate the paid replacement slice first.  An incomplete live run must
    # fail before we spend time hashing the 3,250 unchanged calls.
    replacement_locks: dict[tuple[str, str, str], CorrectedSourceLock] = {}
    for cell, actual in replacement_by_cell.items():
        language, domain, system_slug = cell
        canonical_source = _source_for_cell(experience, language, domain, system_slug)
        replacement_locks[cell] = _lock_one_source(
            actual,
            logical_path=canonical_source.path,
            language=language,
            domain=domain,
            system_slug=system_slug,
            origin="corrected",
            contract=contract,
        )

    sources: list[CorrectedSourceLock] = []
    actual_by_logical_path: dict[str, Path] = {}
    for language in contract.languages:
        for domain in contract.domains:
            for system_slug in contract.system_slugs:
                canonical_source = _source_for_cell(
                    experience, language, domain, system_slug
                )
                logical = canonical_source.path
                cell = (language, domain, system_slug)
                if contract.is_replacement(language, domain):
                    actual = replacement_by_cell[cell]
                    source = replacement_locks[cell]
                else:
                    relative = _safe_relative_path(logical, label="canonical source")
                    actual = (canonical_evidence_root / relative).resolve()
                    if not actual.is_relative_to(canonical_evidence_root):
                        raise ValueError(
                            f"canonical source escapes evidence root: {logical}"
                        )
                    if (
                        not actual.is_file()
                        or sha256_file(actual) != canonical_source.sha256
                    ):
                        raise ValueError(f"canonical results hash mismatch: {logical}")
                    source = _lock_one_source(
                        actual,
                        logical_path=logical,
                        language=language,
                        domain=domain,
                        system_slug=system_slug,
                        origin="canonical",
                        contract=contract,
                    )
                sources.append(source)
                actual_by_logical_path[logical] = actual
    cohort = CorrectedCohortDimensions(
        calls=contract.calls,
        roots=contract.roots,
        languages=list(contract.languages),
        domains=list(contract.domains),
        system_slugs=list(contract.system_slugs),
        trial=contract.trial,
        calls_per_root=contract.calls_per_root,
    )
    manifest = CorrectedExperienceManifest(
        schema_version=CORRECTED_COHORT_SCHEMA_VERSION,
        instrument=CORRECTED_INSTRUMENT,
        cohort=cohort,
        cohort_fingerprint_sha256=corrected_manifest_fingerprint(cohort, sources),
        sources=sources,
    )
    _validate_manifest_shape(manifest, contract)
    return manifest, actual_by_logical_path


def plan_replacement_only_cohort_lock(
    *,
    replacement_results: list[Path],
    contract: CorrectedCohortContract = REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT,
) -> tuple[CorrectedExperienceManifest, dict[str, Path]]:
    """Lock exactly the ten corrected retail roots without canonical inputs."""
    if contract.reused_calls != 0 or contract.reused_roots != 0:
        raise ValueError("replacement-only locking requires an all-corrected contract")
    replacement_by_cell: dict[tuple[str, str, str], Path] = {}
    for raw_path in replacement_results:
        path = raw_path.expanduser().resolve()
        cell = _replacement_cell(path, contract)
        if cell in replacement_by_cell:
            raise ValueError(f"duplicate replacement source cell: {cell}")
        replacement_by_cell[cell] = path
    expected = {
        (language, contract.replacement_domain, system)
        for language in contract.replacement_languages
        for system in contract.system_slugs
    }
    if set(replacement_by_cell) != expected:
        missing = sorted(expected - set(replacement_by_cell))
        extra = sorted(set(replacement_by_cell) - expected)
        raise ValueError(
            f"replacement source grid mismatch; missing={missing}, extra={extra}"
        )

    sources: list[CorrectedSourceLock] = []
    actual_by_logical_path: dict[str, Path] = {}
    for language in contract.languages:
        for system_slug in contract.system_slugs:
            cell = (language, contract.replacement_domain, system_slug)
            actual = replacement_by_cell[cell]
            logical = (
                "replacement_only/"
                f"{language}_{contract.replacement_domain}_{system_slug}/results.json"
            )
            source = _lock_one_source(
                actual,
                logical_path=logical,
                language=language,
                domain=contract.replacement_domain,
                system_slug=system_slug,
                origin="corrected",
                contract=contract,
            )
            sources.append(source)
            actual_by_logical_path[logical] = actual
    cohort = CorrectedCohortDimensions(
        calls=contract.calls,
        roots=contract.roots,
        languages=list(contract.languages),
        domains=list(contract.domains),
        system_slugs=list(contract.system_slugs),
        trial=contract.trial,
        calls_per_root=contract.calls_per_root,
    )
    manifest = CorrectedExperienceManifest(
        schema_version=CORRECTED_COHORT_SCHEMA_VERSION,
        instrument=CORRECTED_INSTRUMENT,
        cohort=cohort,
        cohort_fingerprint_sha256=corrected_manifest_fingerprint(cohort, sources),
        sources=sources,
    )
    _validate_manifest_shape(manifest, contract)
    return manifest, actual_by_logical_path


def _stage_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)


def _verify_staged_evidence(
    evidence_root: Path, manifest: CorrectedExperienceManifest
) -> None:
    expected_files: set[Path] = set()
    for source in manifest.sources:
        results_path = evidence_root / source.path
        expected_files.add(results_path)
        if (
            not results_path.is_file()
            or sha256_file(results_path) != source.results_sha256
        ):
            raise ValueError(f"existing staged results mismatch: {source.path}")
        for call in source.calls:
            sim_path = (
                results_path.parent / "simulations" / f"{call.simulation_id}.json"
            )
            expected_files.add(sim_path)
            if (
                not sim_path.is_file()
                or sha256_file(sim_path) != call.simulation_sha256
            ):
                raise ValueError(
                    f"existing staged simulation mismatch: {call.simulation_id}"
                )
    actual_files = {path for path in evidence_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "existing staged evidence contains unexpected or missing files"
        )


def build_corrected_cohort_lock(
    *,
    canonical_experience: Path,
    canonical_evidence_root: Path,
    replacement_results: list[Path],
    evidence_root: Path,
    experience_manifest: Path,
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
) -> CorrectedExperienceManifest:
    """Atomically stage transcript evidence and write its complete typed lock."""
    manifest, actual_by_logical = plan_corrected_cohort_lock(
        canonical_experience=canonical_experience,
        canonical_evidence_root=canonical_evidence_root,
        replacement_results=replacement_results,
        contract=contract,
    )
    evidence_root = evidence_root.expanduser().resolve()
    experience_manifest = experience_manifest.expanduser().resolve()
    if experience_manifest == evidence_root or experience_manifest.is_relative_to(
        evidence_root
    ):
        raise ValueError("corrected Experience manifest must be outside evidence root")
    source_roots = {path.parent.resolve() for path in actual_by_logical.values()}
    if any(
        evidence_root == source_root
        or evidence_root.is_relative_to(source_root)
        or source_root.is_relative_to(evidence_root)
        for source_root in source_roots
    ):
        raise ValueError("staged evidence must be separate from every source root")
    if evidence_root.exists():
        _verify_staged_evidence(evidence_root, manifest)
    else:
        evidence_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{evidence_root.name}.", dir=evidence_root.parent)
        )
        try:
            for source in manifest.sources:
                actual_results = actual_by_logical[source.path]
                staged_results = temporary / source.path
                _stage_file(actual_results, staged_results)
                for call in source.calls:
                    actual_sim = (
                        actual_results.parent
                        / "simulations"
                        / f"{call.simulation_id}.json"
                    )
                    staged_sim = (
                        staged_results.parent
                        / "simulations"
                        / f"{call.simulation_id}.json"
                    )
                    _stage_file(actual_sim, staged_sim)
            temporary.replace(evidence_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    if experience_manifest.exists():
        existing = CorrectedExperienceManifest.model_validate_json(
            experience_manifest.read_text()
        )
        if existing != manifest:
            raise ValueError("existing corrected Experience manifest drifted")
    else:
        _atomic_write(experience_manifest, manifest)
    return manifest


def build_replacement_only_cohort_lock(
    *,
    replacement_results: list[Path],
    evidence_root: Path,
    experience_manifest: Path,
    contract: CorrectedCohortContract = REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT,
) -> CorrectedExperienceManifest:
    """Atomically stage only the ten corrected roots and their 500 calls."""
    manifest, actual_by_logical = plan_replacement_only_cohort_lock(
        replacement_results=replacement_results,
        contract=contract,
    )
    evidence_root = evidence_root.expanduser().resolve()
    experience_manifest = experience_manifest.expanduser().resolve()
    if experience_manifest == evidence_root or experience_manifest.is_relative_to(
        evidence_root
    ):
        raise ValueError("corrected Experience manifest must be outside evidence root")
    source_roots = {path.parent.resolve() for path in actual_by_logical.values()}
    if any(
        evidence_root == source_root
        or evidence_root.is_relative_to(source_root)
        or source_root.is_relative_to(evidence_root)
        for source_root in source_roots
    ):
        raise ValueError("staged evidence must be separate from every source root")
    if evidence_root.exists():
        _verify_staged_evidence(evidence_root, manifest)
    else:
        evidence_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{evidence_root.name}.", dir=evidence_root.parent)
        )
        try:
            for source in manifest.sources:
                actual_results = actual_by_logical[source.path]
                staged_results = temporary / source.path
                _stage_file(actual_results, staged_results)
                for call in source.calls:
                    actual_sim = (
                        actual_results.parent
                        / "simulations"
                        / f"{call.simulation_id}.json"
                    )
                    staged_sim = (
                        staged_results.parent
                        / "simulations"
                        / f"{call.simulation_id}.json"
                    )
                    _stage_file(actual_sim, staged_sim)
            temporary.replace(evidence_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    if experience_manifest.exists():
        existing = CorrectedExperienceManifest.model_validate_json(
            experience_manifest.read_text()
        )
        if existing != manifest:
            raise ValueError("existing corrected Experience manifest drifted")
    else:
        _atomic_write(experience_manifest, manifest)
    return manifest


def _corrected_code_provenance() -> CorrectedCodeProvenance:
    frozen = _code_provenance()
    path = Path(__file__).resolve()
    adapter_sha = sha256_file(path)
    fingerprint = _sha256_value(
        {
            "frozen_judge": frozen.source_fingerprint_sha256,
            "adapter": adapter_sha,
        }
    )
    return CorrectedCodeProvenance(
        frozen_judge=frozen,
        adapter_path=str(path),
        adapter_sha256=adapter_sha,
        source_fingerprint_sha256=fingerprint,
    )


def _load_manifest(path: Path) -> tuple[CorrectedExperienceManifest, str]:
    path = path.expanduser().resolve()
    manifest = CorrectedExperienceManifest.model_validate_json(path.read_text())
    fingerprint = corrected_manifest_fingerprint(manifest.cohort, manifest.sources)
    if fingerprint != manifest.cohort_fingerprint_sha256:
        raise ValueError("corrected Experience cohort fingerprint does not reproduce")
    return manifest, sha256_file(path)


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != "results.json":
        raise ValueError(f"unsafe {label} path: {value}")
    return path


def _validate_manifest_shape(
    manifest: CorrectedExperienceManifest,
    contract: CorrectedCohortContract,
) -> None:
    cohort = manifest.cohort
    if (
        tuple(cohort.languages) != contract.languages
        or tuple(cohort.domains) != contract.domains
        or tuple(cohort.system_slugs) != contract.system_slugs
        or cohort.trial != contract.trial
        or cohort.calls_per_root != contract.calls_per_root
        or cohort.roots != contract.roots
        or cohort.calls != contract.calls
        or len(manifest.sources) != contract.roots
    ):
        raise ValueError("corrected Experience cohort dimensions drifted")
    if contract.trial != 0:
        raise ValueError("corrected naturalness supports trial 0 only")
    if len({source.path for source in manifest.sources}) != contract.roots:
        raise ValueError("corrected source paths are not unique")
    expected_grid = {
        (language, domain, system)
        for language in contract.languages
        for domain in contract.domains
        for system in contract.system_slugs
    }
    observed_grid = {
        (source.language, source.domain, source.system_slug)
        for source in manifest.sources
    }
    if observed_grid != expected_grid or len(observed_grid) != len(manifest.sources):
        raise ValueError("corrected source grid is incomplete or duplicated")
    seen_simulations: set[str] = set()
    for source in manifest.sources:
        expected_origin = (
            "corrected"
            if contract.is_replacement(source.language, source.domain)
            else "canonical"
        )
        if source.origin != expected_origin:
            raise ValueError(
                f"source origin disagrees with corrected slice: {source.path}"
            )
        if len(source.calls) != contract.calls_per_root:
            raise ValueError(f"source call count drifted: {source.path}")
        if len({call.task_id for call in source.calls}) != contract.calls_per_root:
            raise ValueError(f"source task ids are not unique: {source.path}")
        for call in source.calls:
            if call.simulation_id in seen_simulations:
                raise ValueError(
                    f"duplicate corrected simulation id: {call.simulation_id}"
                )
            seen_simulations.add(call.simulation_id)
    if len(seen_simulations) != contract.calls:
        raise ValueError("corrected selected simulation count drifted")
    if _uses_canonical_corrected_contract(contract):
        expected = (
            CORRECTED_EXPECTED_ROOTS,
            CORRECTED_EXPECTED_CALLS,
            CORRECTED_EXPECTED_REUSED_ROOTS,
            CORRECTED_EXPECTED_REUSED_CALLS,
            CORRECTED_EXPECTED_PENDING_ROOTS,
            CORRECTED_EXPECTED_PENDING_CALLS,
        )
        observed = (
            contract.roots,
            contract.calls,
            contract.reused_roots,
            contract.reused_calls,
            contract.pending_roots,
            contract.pending_calls,
        )
        if observed != expected:
            raise ValueError("canonical corrected cohort constants drifted")
    if _uses_replacement_only_contract(contract) and (
        contract.roots != REPLACEMENT_ONLY_EXPECTED_ROOTS
        or contract.calls != REPLACEMENT_ONLY_EXPECTED_CALLS
        or contract.reused_roots != 0
        or contract.reused_calls != 0
        or contract.pending_roots != REPLACEMENT_ONLY_EXPECTED_ROOTS
        or contract.pending_calls != REPLACEMENT_ONLY_EXPECTED_CALLS
    ):
        raise ValueError("replacement-only cohort constants drifted")


def prepare_corrected_trial_plan(
    *,
    repo_root: Path,
    experience_manifest: Path,
    evidence_root: Path,
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
) -> CorrectedTrialPlan:
    """Verify every source/call/transcript lock without making model calls."""
    repo_root = repo_root.expanduser().resolve()
    manifest_path = experience_manifest.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    manifest, manifest_sha = _load_manifest(manifest_path)
    _validate_manifest_shape(manifest, contract)
    simulations: list[TrialSimulationPlan] = []
    utterances: list[TrialUtterancePlan] = []
    for source in manifest.sources:
        relative = _safe_relative_path(source.path, label="corrected source")
        results_path = (evidence_root / relative).resolve()
        if not results_path.is_relative_to(evidence_root):
            raise ValueError(f"corrected source escapes evidence root: {source.path}")
        if (
            not results_path.is_file()
            or sha256_file(results_path) != source.results_sha256
        ):
            raise ValueError(f"corrected results hash mismatch: {source.path}")
        metadata = Results.load_metadata(results_path)
        if metadata.simulation_index is None:
            raise ValueError(f"corrected source has no simulation index: {source.path}")
        entries = [
            entry
            for entry in metadata.simulation_index
            if entry.trial == contract.trial
        ]
        locked_ids = [call.simulation_id for call in source.calls]
        if [entry.id for entry in entries] != locked_ids:
            raise ValueError(f"corrected simulation index mismatch: {source.path}")
        tasks = {str(task.id) for task in metadata.tasks}
        identity = TrialSourceIdentity(
            results_path=source.path,
            results_sha256=source.results_sha256,
            source_key=hashlib.sha256(source.path.encode()).hexdigest()[:16],
            language=source.language,
            domain=source.domain,
            system_slug=source.system_slug,
            trial_0_calls=len(source.calls),
        )
        simulations_dir = results_path.parent / "simulations"
        if not simulations_dir.is_dir():
            raise ValueError(f"corrected source is not dir-format: {source.path}")
        for entry, call in zip(entries, source.calls, strict=True):
            sim_path = (simulations_dir / f"{call.simulation_id}.json").resolve()
            if not sim_path.is_relative_to(
                simulations_dir.resolve()
            ) or not sim_path.is_relative_to(evidence_root):
                raise ValueError(f"corrected simulation path escapes: {sim_path}")
            raw = sim_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != call.simulation_sha256:
                raise ValueError(
                    f"corrected simulation hash mismatch: {call.simulation_id}"
                )
            simulation = SimulationRun.model_validate_json(raw)
            if (
                simulation.id != call.simulation_id
                or str(simulation.task_id) != call.task_id
                or simulation.trial != call.trial
                or str(simulation.task_id) not in tasks
            ):
                raise ValueError(
                    f"corrected simulation identity mismatch: {call.simulation_id}"
                )
            language, _script = get_simulation_language_info(simulation)
            if language != source.language:
                raise ValueError(
                    f"corrected simulation language mismatch: {call.simulation_id}"
                )
            turns = build_agent_turns(simulation)
            if transcript_sha256(turns) != call.transcript_sha256:
                raise ValueError(
                    f"corrected transcript hash mismatch: {call.simulation_id}"
                )
            judge = frozen_judge_spec(language)
            sim_plan = TrialSimulationPlan(
                source=identity,
                simulation_id=simulation.id,
                task_id=str(simulation.task_id),
                trial=0,
                language=language,
                domain=source.domain,
                source_simulation_sha256=call.simulation_sha256,
                judge=judge,
                turns=turns,
            )
            simulations.append(sim_plan)
            utterances.extend(
                TrialUtterancePlan(
                    source=identity,
                    simulation_id=simulation.id,
                    task_id=str(simulation.task_id),
                    trial=0,
                    language=language,
                    domain=source.domain,
                    source_simulation_sha256=call.simulation_sha256,
                    judge=judge,
                    turn=turn,
                    input_sha256=_unit_digest(sim_plan, turn),
                )
                for turn in turns
            )
    validation = (
        _validated_evidence_identity(repo_root)
        if _uses_production_contract(contract)
        else None
    )
    return CorrectedTrialPlan(
        experience_manifest_path=str(manifest_path),
        experience_manifest_sha256=manifest_sha,
        cohort_fingerprint_sha256=manifest.cohort_fingerprint_sha256,
        validation_evidence=validation,
        sources=manifest.sources,
        simulations=simulations,
        utterances=utterances,
    )


def _corrected_work_fingerprint(plan: CorrectedTrialPlan) -> str:
    return _sha256_value(
        {
            "manifest": plan.experience_manifest_sha256,
            "simulations": [
                (
                    row.source.results_path,
                    row.simulation_id,
                    row.source_simulation_sha256,
                    transcript_sha256(row.turns),
                )
                for row in plan.simulations
            ],
            "utterances": [row.input_sha256 for row in plan.utterances],
        }
    )


def _simulation_key(plan: TrialSimulationPlan) -> str:
    return f"{plan.source.results_path}\0{plan.simulation_id}"


def _validate_bootstrap_call(
    plan: TrialSimulationPlan,
    artifact: TrialSimulationArtifact,
    artifact_path: Path,
) -> None:
    prefix = f"bootstrap artifact {plan.simulation_id}"
    if (
        artifact.source != plan.source
        or artifact.simulation_id != plan.simulation_id
        or artifact.task_id != plan.task_id
        or artifact.trial != plan.trial
        or artifact.language != plan.language
        or artifact.domain != plan.domain
        or artifact.source_simulation_sha256 != plan.source_simulation_sha256
        or artifact.judge != plan.judge
    ):
        raise ValueError(f"{prefix} identity mismatch at {artifact_path}")
    planned_turns = {turn.index: turn for turn in plan.turns}
    if len(artifact.utterances) != len(plan.turns):
        raise ValueError(f"{prefix} utterance-count mismatch at {artifact_path}")
    for row in artifact.utterances:
        turn = planned_turns.get(row.turn.index)
        if turn is None:
            raise ValueError(f"{prefix} turn mismatch at {artifact_path}")
        expected_input = _unit_digest(plan, turn)
        if (
            row.input_sha256 != expected_input
            or row.source != plan.source
            or row.simulation_id != plan.simulation_id
            or row.task_id != plan.task_id
            or row.trial != plan.trial
            or row.language != plan.language
            or row.domain != plan.domain
            or row.source_simulation_sha256 != plan.source_simulation_sha256
            or row.judge != plan.judge
            or row.turn != turn
            or row.check.outcome is JudgeOutcome.ERROR
        ):
            raise ValueError(f"{prefix} utterance mismatch at {artifact_path}")
    reproduced = _aggregate_simulation(plan, artifact.utterances)
    if reproduced != artifact:
        raise ValueError(f"{prefix} aggregate mismatch at {artifact_path}")


def _verify_bootstrap(
    plan: CorrectedTrialPlan,
    bootstrap_root: Path,
    contract: CorrectedCohortContract,
) -> tuple[BootstrapIdentity, dict[str, TrialSimulationArtifact]]:
    from tau2.judges.nativeness.paper_trial import TrialRunManifest

    bootstrap_root = bootstrap_root.expanduser().resolve()
    manifest_path = bootstrap_root / "manifest.json"
    manifest = TrialRunManifest.model_validate_json(manifest_path.read_text())
    if manifest.status != "complete":
        raise ValueError("bootstrap canonical naturalness run is not complete")
    if (
        manifest.counts.simulations != contract.calls
        or manifest.counts.complete_simulations != contract.calls
        or manifest.counts.error_simulations != 0
        or manifest.counts.error_utterances != 0
    ):
        raise ValueError("bootstrap canonical cohort shape/status mismatch")
    source_by_cell = {
        (source.language, source.domain, source.system_slug): source
        for source in manifest.identity.selected_sources
    }
    expected_cells = {
        (language, domain, system)
        for language in contract.languages
        for domain in contract.domains
        for system in contract.system_slugs
    }
    if set(source_by_cell) != expected_cells:
        raise ValueError("bootstrap canonical source grid mismatch")
    judges = {language: frozen_judge_spec(language) for language in contract.languages}
    if any(
        manifest.identity.judges.get(language) != judge
        for language, judge in judges.items()
    ):
        raise ValueError("bootstrap frozen judge identity mismatch")

    artifacts: dict[str, TrialSimulationArtifact] = {}
    identities: list[BootstrapCallArtifact] = []
    reused_plans = [
        row
        for row in plan.simulations
        if not contract.is_replacement(row.language, row.domain)
    ]
    for sim_plan in reused_plans:
        cell = (sim_plan.language, sim_plan.domain, sim_plan.source.system_slug)
        if source_by_cell[cell] != sim_plan.source:
            raise ValueError(
                f"bootstrap source identity mismatch: {sim_plan.source.results_path}"
            )
        artifact_path = _simulation_path(bootstrap_root, sim_plan)
        try:
            artifact = _load_existing_simulation(artifact_path, sim_plan)
        except ValueError as exc:
            raise ValueError(
                f"bootstrap artifact identity mismatch at {artifact_path}"
            ) from exc
        if artifact is None:
            raise ValueError(f"bootstrap call artifact missing: {artifact_path}")
        _validate_bootstrap_call(sim_plan, artifact, artifact_path)
        key = _simulation_key(sim_plan)
        artifacts[key] = artifact
        identities.append(
            BootstrapCallArtifact(
                results_path=sim_plan.source.results_path,
                simulation_id=sim_plan.simulation_id,
                artifact_path=artifact_path.relative_to(bootstrap_root).as_posix(),
                artifact_sha256=sha256_file(artifact_path),
                utterances=len(artifact.utterances),
            )
        )
    if len(artifacts) != contract.reused_calls:
        raise ValueError("bootstrap reusable call count mismatch")
    identities.sort(key=lambda row: (row.results_path, row.simulation_id))
    identity = BootstrapIdentity(
        path=str(bootstrap_root),
        manifest_sha256=sha256_file(manifest_path),
        identity_sha256=manifest.identity_sha256,
        call_artifact_fingerprint_sha256=_sha256_value(
            [row.model_dump(mode="json") for row in identities]
        ),
        call_artifacts=identities,
    )
    return identity, artifacts


def _prepare_bundle(
    *,
    repo_root: Path,
    experience_manifest: Path,
    evidence_root: Path,
    bootstrap_from: Optional[Path],
    contract: CorrectedCohortContract,
) -> _PreparedBundle:
    plan = prepare_corrected_trial_plan(
        repo_root=repo_root,
        experience_manifest=experience_manifest,
        evidence_root=evidence_root,
        contract=contract,
    )
    if contract.reused_calls:
        if bootstrap_from is None:
            raise ValueError("hybrid corrected replay requires canonical bootstrap")
        bootstrap, bootstrap_artifacts = _verify_bootstrap(
            plan, bootstrap_from, contract
        )
    else:
        if bootstrap_from is not None:
            raise ValueError("replacement-only replay does not accept a bootstrap")
        bootstrap = None
        bootstrap_artifacts = {}
    pending = [
        row
        for row in plan.simulations
        if contract.is_replacement(row.language, row.domain)
    ]
    reused = [
        row
        for row in plan.simulations
        if not contract.is_replacement(row.language, row.domain)
    ]
    if len(pending) != contract.pending_calls or len(reused) != contract.reused_calls:
        raise ValueError("corrected reuse/pending partition drifted")
    pending_calls: dict[str, Counter[str]] = defaultdict(Counter)
    pending_units: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pending:
        pending_calls[row.language][row.source.system_slug] += 1
        pending_units[row.language][row.source.system_slug] += len(row.turns)
    counts = CorrectedPreparationCounts(
        roots=contract.roots,
        calls=len(plan.simulations),
        utterances=len(plan.utterances),
        reused_roots=contract.reused_roots,
        reused_calls=len(reused),
        reused_utterances=sum(len(row.turns) for row in reused),
        pending_roots=contract.pending_roots,
        pending_calls=len(pending),
        pending_utterances=sum(len(row.turns) for row in pending),
        zero_utterance_pending_calls=sum(not row.turns for row in pending),
    )
    if _uses_canonical_corrected_contract(contract) and (
        counts.roots != CORRECTED_EXPECTED_ROOTS
        or counts.calls != CORRECTED_EXPECTED_CALLS
        or counts.reused_roots != CORRECTED_EXPECTED_REUSED_ROOTS
        or counts.reused_calls != CORRECTED_EXPECTED_REUSED_CALLS
        or counts.pending_roots != CORRECTED_EXPECTED_PENDING_ROOTS
        or counts.pending_calls != CORRECTED_EXPECTED_PENDING_CALLS
    ):
        raise ValueError("corrected production reuse/pending counts drifted")
    if _uses_replacement_only_contract(contract) and (
        counts.roots != REPLACEMENT_ONLY_EXPECTED_ROOTS
        or counts.calls != REPLACEMENT_ONLY_EXPECTED_CALLS
        or counts.reused_roots != 0
        or counts.reused_calls != 0
        or counts.pending_roots != REPLACEMENT_ONLY_EXPECTED_ROOTS
        or counts.pending_calls != REPLACEMENT_ONLY_EXPECTED_CALLS
    ):
        raise ValueError("replacement-only production counts drifted")
    report = CorrectedPreparationReport(
        schema_version=CORRECTED_RUN_SCHEMA_VERSION,
        experience_manifest_path=plan.experience_manifest_path,
        experience_manifest_sha256=plan.experience_manifest_sha256,
        cohort_fingerprint_sha256=plan.cohort_fingerprint_sha256,
        work_fingerprint_sha256=_corrected_work_fingerprint(plan),
        validation_evidence=plan.validation_evidence,
        code=_corrected_code_provenance(),
        bootstrap=bootstrap,
        counts=counts,
        judges={
            language: frozen_judge_spec(language) for language in contract.languages
        },
        pending_calls_by_language_system={
            language: dict(sorted(values.items()))
            for language, values in sorted(pending_calls.items())
        },
        pending_utterances_by_language_system={
            language: dict(sorted(values.items()))
            for language, values in sorted(pending_units.items())
        },
    )
    return _PreparedBundle(
        plan=plan,
        bootstrap=bootstrap,
        bootstrap_artifacts=bootstrap_artifacts,
        report=report,
    )


def prepare_corrected_trial0_naturalness(
    *,
    repo_root: Path,
    experience_manifest: Path,
    evidence_root: Path,
    bootstrap_from: Path,
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
) -> CorrectedPreparationReport:
    """Verify the hybrid cohort and exact canonical reuse without model calls."""
    return _prepare_bundle(
        repo_root=repo_root,
        experience_manifest=experience_manifest,
        evidence_root=evidence_root,
        bootstrap_from=bootstrap_from,
        contract=contract,
    ).report


def prepare_replacement_only_trial0_naturalness(
    *,
    repo_root: Path,
    experience_manifest: Path,
    evidence_root: Path,
    contract: CorrectedCohortContract = REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT,
) -> CorrectedPreparationReport:
    """Verify the closed ten-root replacement cohort without model calls."""
    return _prepare_bundle(
        repo_root=repo_root,
        experience_manifest=experience_manifest,
        evidence_root=evidence_root,
        bootstrap_from=None,
        contract=contract,
    ).report


def _run_identity(bundle: _PreparedBundle) -> CorrectedRunIdentity:
    report = bundle.report
    return CorrectedRunIdentity(
        schema_version=CORRECTED_RUN_SCHEMA_VERSION,
        experience_manifest_path=report.experience_manifest_path,
        experience_manifest_sha256=report.experience_manifest_sha256,
        cohort_fingerprint_sha256=report.cohort_fingerprint_sha256,
        work_fingerprint_sha256=report.work_fingerprint_sha256,
        validation_evidence=report.validation_evidence,
        code=report.code,
        bootstrap=report.bootstrap,
        sources=bundle.plan.sources,
        judges=report.judges,
        trial=0,
    )


def _initial_run_manifest(
    bundle: _PreparedBundle, identity: CorrectedRunIdentity
) -> CorrectedRunManifest:
    now = _now()
    counts = bundle.report.counts
    return CorrectedRunManifest(
        schema_version=CORRECTED_RUN_SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        status="running",
        identity_sha256=_sha256_value(identity.model_dump(mode="json")),
        identity=identity,
        counts=CorrectedRunCounts(
            roots=counts.roots,
            calls=counts.calls,
            utterances=counts.utterances,
            reused_roots=counts.reused_roots,
            reused_calls=counts.reused_calls,
            reused_utterances=counts.reused_utterances,
            pending_roots=counts.pending_roots,
            pending_calls=counts.pending_calls,
            pending_utterances=counts.pending_utterances,
            judged_utterances=0,
            error_utterances=0,
            complete_calls=0,
            error_calls=0,
        ),
        aggregate=None,
        invocations=[],
    )


def _ensure_output_root(
    output_root: Path,
    bundle: _PreparedBundle,
    identity: CorrectedRunIdentity,
) -> CorrectedRunManifest:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = CorrectedRunManifest.model_validate_json(manifest_path.read_text())
        expected = _sha256_value(identity.model_dump(mode="json"))
        if manifest.identity_sha256 != expected or manifest.identity != identity:
            raise ValueError("corrected output identity drifted; use a new output root")
        return manifest
    if output_root.exists() and any(
        path.name != RUN_LOCK_FILENAME for path in output_root.iterdir()
    ):
        raise ValueError("non-empty corrected output has no typed manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _initial_run_manifest(bundle, identity)
    _atomic_write(manifest_path, manifest)
    return manifest


def _write_bootstrap_call(
    output_root: Path,
    plan: TrialSimulationPlan,
    artifact: TrialSimulationArtifact,
) -> None:
    path = _simulation_path(output_root, plan)
    existing = _load_existing_simulation(path, plan)
    if existing is None:
        _atomic_write(path, artifact)
    elif existing != artifact:
        raise ValueError(f"bootstrapped corrected output drifted at {path}")


def _score_replacement_worker(
    item: tuple[TrialUtterancePlan, Path],
) -> TrialUtteranceArtifact:
    """Score one uniquely addressed utterance in an isolated worker process."""
    plan, output_root = item
    return _score_trial_utterance(plan, output_root)


def _bounded_process_score(
    items: list[TrialUtterancePlan],
    output_root: Path,
    processes: int,
) -> list[TrialUtteranceArtifact]:
    """Score with bounded process submissions while preserving atomic resume files."""
    iterator = iter(items)
    completed = 0
    artifacts: list[TrialUtteranceArtifact] = []
    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = {
            executor.submit(_score_replacement_worker, (item, output_root))
            for item in islice(iterator, processes)
        }
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                artifacts.append(future.result())
            futures |= {
                executor.submit(_score_replacement_worker, (item, output_root))
                for item in islice(iterator, len(done))
            }
    return artifacts


def _provenance_still_matches(
    bundle: _PreparedBundle,
    evidence_root: Path,
    bootstrap_root: Optional[Path],
) -> bool:
    try:
        plan = bundle.plan
        if sha256_file(Path(plan.experience_manifest_path)) != (
            plan.experience_manifest_sha256
        ):
            return False
        if bundle.bootstrap is not None:
            if bootstrap_root is None:
                return False
            if sha256_file(bootstrap_root / "manifest.json") != (
                bundle.bootstrap.manifest_sha256
            ):
                return False
            for artifact in bundle.bootstrap.call_artifacts:
                if sha256_file(bootstrap_root / artifact.artifact_path) != (
                    artifact.artifact_sha256
                ):
                    return False
        elif bootstrap_root is not None:
            return False
        for source in plan.sources:
            results_path = evidence_root / _safe_relative_path(
                source.path, label="corrected source"
            )
            if sha256_file(results_path) != source.results_sha256:
                return False
            for call in source.calls:
                path = (
                    results_path.parent / "simulations" / f"{call.simulation_id}.json"
                )
                if sha256_file(path) != call.simulation_sha256:
                    return False
        return True
    except (OSError, ValueError):
        return False


def run_corrected_trial0_naturalness(
    config: CorrectedTrialRunConfig,
    *,
    contract: CorrectedCohortContract = CANONICAL_CORRECTED_COHORT_CONTRACT,
) -> CorrectedRunManifest:
    """Run/resume only corrected calls after exact canonical bootstrapping."""
    output_root = config.output_root.expanduser().resolve()
    evidence_root = config.evidence_root.expanduser().resolve()
    bootstrap_root = config.bootstrap_from.expanduser().resolve()
    if output_root == evidence_root or output_root.is_relative_to(evidence_root):
        raise ValueError("corrected output must be outside the evidence tree")
    if output_root == bootstrap_root or output_root.is_relative_to(bootstrap_root):
        raise ValueError("corrected output must be outside the bootstrap tree")
    with claim_run_directories([output_root]):
        return _run_corrected_claimed(
            repo_root=config.repo_root,
            experience_manifest=config.experience_manifest,
            evidence_root=config.evidence_root,
            bootstrap_from=config.bootstrap_from,
            output_root=config.output_root,
            max_concurrency=config.max_concurrency,
            processes=config.processes,
            contract=contract,
        )


def run_replacement_only_trial0_naturalness(
    config: ReplacementOnlyTrialRunConfig,
    *,
    contract: CorrectedCohortContract = REPLACEMENT_ONLY_CORRECTED_COHORT_CONTRACT,
) -> CorrectedRunManifest:
    """Run/resume the standalone ten-root corrected-call replay."""
    output_root = config.output_root.expanduser().resolve()
    evidence_root = config.evidence_root.expanduser().resolve()
    if output_root == evidence_root or output_root.is_relative_to(evidence_root):
        raise ValueError("corrected output must be outside the evidence tree")
    with claim_run_directories([output_root]):
        return _run_corrected_claimed(
            repo_root=config.repo_root,
            experience_manifest=config.experience_manifest,
            evidence_root=config.evidence_root,
            bootstrap_from=None,
            output_root=config.output_root,
            max_concurrency=config.max_concurrency,
            processes=config.processes,
            contract=contract,
        )


def _run_corrected_claimed(
    *,
    repo_root: Path,
    experience_manifest: Path,
    evidence_root: Path,
    bootstrap_from: Optional[Path],
    output_root: Path,
    max_concurrency: int,
    processes: int,
    contract: CorrectedCohortContract,
) -> CorrectedRunManifest:
    started = _now()
    output_root = output_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    bundle = _prepare_bundle(
        repo_root=repo_root,
        experience_manifest=experience_manifest,
        evidence_root=evidence_root,
        bootstrap_from=bootstrap_from,
        contract=contract,
    )
    identity = _run_identity(bundle)
    prior = _ensure_output_root(output_root, bundle, identity)
    pending_plans = [
        row
        for row in bundle.plan.simulations
        if contract.is_replacement(row.language, row.domain)
    ]
    reused_plans = [
        row
        for row in bundle.plan.simulations
        if not contract.is_replacement(row.language, row.domain)
    ]
    expected_sim_paths = {
        _simulation_path(output_root, row) for row in bundle.plan.simulations
    }
    existing_sim_paths = (
        set((output_root / "simulations").rglob("*.json"))
        if (output_root / "simulations").exists()
        else set()
    )
    if existing_sim_paths - expected_sim_paths:
        raise ValueError("corrected output contains non-cohort simulations")
    pending_keys = {(row.source.source_key, row.simulation_id) for row in pending_plans}
    expected_utterance_paths = {
        _utterance_path(output_root, utterance)
        for utterance in bundle.plan.utterances
        if (utterance.source.source_key, utterance.simulation_id) in pending_keys
    }
    existing_utterance_paths = (
        set((output_root / "utterances").rglob("*.json"))
        if (output_root / "utterances").exists()
        else set()
    )
    if existing_utterance_paths - expected_utterance_paths:
        raise ValueError("corrected output contains non-cohort utterances")

    simulation_artifacts: list[TrialSimulationArtifact] = []
    for sim_plan in reused_plans:
        artifact = bundle.bootstrap_artifacts[_simulation_key(sim_plan)]
        _write_bootstrap_call(output_root, sim_plan, artifact)
        simulation_artifacts.append(artifact)

    cached: dict[tuple[str, str], list[TrialUtteranceArtifact]] = defaultdict(list)
    pending_units: list[TrialUtterancePlan] = []
    utterances_by_sim = defaultdict(list)
    for utterance in bundle.plan.utterances:
        utterances_by_sim[
            (utterance.source.source_key, utterance.simulation_id)
        ].append(utterance)
    for sim_plan in pending_plans:
        _load_existing_simulation(_simulation_path(output_root, sim_plan), sim_plan)
        for item in utterances_by_sim[
            (sim_plan.source.source_key, sim_plan.simulation_id)
        ]:
            artifact = _load_matching_utterance(
                _utterance_path(output_root, item), item
            )
            if artifact is None:
                pending_units.append(item)
            else:
                cached[(item.source.source_key, item.simulation_id)].append(artifact)
    if processes > max_concurrency:
        raise ValueError("worker processes cannot exceed maximum concurrency")
    if processes == 1:
        scored = _bounded_map(
            lambda item: _score_trial_utterance(item, output_root),
            pending_units,
            max_concurrency,
        )
    else:
        scored = iter(_bounded_process_score(pending_units, output_root, processes))
    for artifact in scored:
        cached[(artifact.source.source_key, artifact.simulation_id)].append(artifact)

    utterance_errors = 0
    simulation_errors = 0
    for sim_plan in pending_plans:
        records = cached[(sim_plan.source.source_key, sim_plan.simulation_id)]
        if len(records) != len(sim_plan.turns):
            raise ValueError(
                f"incomplete corrected utterance set for {sim_plan.simulation_id}"
            )
        artifact = _aggregate_simulation(sim_plan, records)
        existing = _load_existing_simulation(
            _simulation_path(output_root, sim_plan), sim_plan
        )
        if existing is None or existing.check.outcome is JudgeOutcome.ERROR:
            _atomic_write(_simulation_path(output_root, sim_plan), artifact)
        elif existing != artifact:
            raise ValueError(
                f"complete corrected simulation drifted: {sim_plan.simulation_id}"
            )
        simulation_artifacts.append(artifact)
        utterance_errors += sum(
            row.check.outcome is JudgeOutcome.ERROR for row in records
        )
        simulation_errors += artifact.check.outcome is JudgeOutcome.ERROR

    source_drift = not _provenance_still_matches(
        bundle,
        evidence_root,
        bootstrap_from.expanduser().resolve() if bootstrap_from is not None else None,
    )
    status: Literal["complete", "complete_with_errors", "source_drift"] = (
        "source_drift"
        if source_drift
        else ("complete_with_errors" if utterance_errors else "complete")
    )
    invocation = CorrectedInvocation(
        started_at=started,
        finished_at=_now(),
        max_concurrency=max_concurrency,
        processes=processes,
        judged_utterances=len(pending_units),
        reused_utterances=bundle.report.counts.pending_utterances - len(pending_units),
        error_utterances=utterance_errors,
    )
    counts = bundle.report.counts
    manifest = CorrectedRunManifest(
        schema_version=CORRECTED_RUN_SCHEMA_VERSION,
        created_at=prior.created_at,
        updated_at=invocation.finished_at,
        status=status,
        identity_sha256=prior.identity_sha256,
        identity=identity,
        counts=CorrectedRunCounts(
            roots=counts.roots,
            calls=counts.calls,
            utterances=counts.utterances,
            reused_roots=counts.reused_roots,
            reused_calls=counts.reused_calls,
            reused_utterances=counts.reused_utterances,
            pending_roots=counts.pending_roots,
            pending_calls=counts.pending_calls,
            pending_utterances=counts.pending_utterances,
            judged_utterances=counts.pending_utterances - utterance_errors,
            error_utterances=utterance_errors,
            complete_calls=counts.calls - simulation_errors,
            error_calls=simulation_errors,
        ),
        aggregate=_aggregate_trial(simulation_artifacts),
        invocations=[*prior.invocations, invocation],
    )
    _atomic_write(output_root / "manifest.json", manifest)
    if source_drift:
        raise ValueError("corrected evidence changed during replay")
    return manifest
