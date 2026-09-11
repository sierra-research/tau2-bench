# Copyright Sierra
"""Frozen combined-naturalness replay for the τ-Multilingual paper cohort.

This runner has two paid modes with one execution seam:

* validation replay judges every checked-in labeled utterance and rebuilds its
  metrics; and
* trial-0 replay reads only the 90 roots named by the frozen Experience
  manifest, selects the 75 non-English roots, and writes sidecar artifacts.

Neither mode mutates a source simulation.  Each paid utterance is written
atomically as soon as it completes and is reused only when its full typed input
identity matches, so an interrupted invocation safely fills just the gaps.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
    DEFAULT_TAU_MULTI_NATURALNESS_JUDGE,
    DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS,
)
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeSettings,
    Results,
    SimulationRun,
)
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.nativeness.harness import (
    AgentTurn,
    aggregate_utterance_results,
    build_agent_turns,
    evaluate_pack_judge_factors,
)
from tau2.judges.nativeness.judge import (
    NATIVENESS_JUDGE_PROMPT_VERSION,
    NativenessJudgeCriterion,
    NativenessJudgeResult,
)
from tau2.judges.nativeness.validation import (
    COMBINED_NATURALNESS_RUBRIC_VERSION,
    FIXED_VALIDATION_ACCEPTANCE_POLICY,
    VALIDATION_FACTOR_ID,
    VALIDATION_FACTOR_NAME,
    VALIDATION_LANGUAGES,
    BinaryMetrics,
    FileReference,
    ManifestResultSummary,
    ValidationDataset,
    ValidationDatasetRow,
    ValidationJudgeConfig,
    ValidationLabel,
    ValidationManifest,
    ValidationPrompt,
    ValidationResultRow,
    ValidationResults,
    ValidationSplit,
    compute_binary_metrics,
    passes_acceptance_policy,
    render_validation_readme,
    runtime_combined_factor,
    runtime_validation_prompt,
    sha256_file,
    validate_validation_package,
)
from tau2.judges.validation_archive import (
    MeasureId as ArchiveMeasureId,
)
from tau2.judges.validation_archive import (
    ValidationRootManifest,
    validate_archive,
)
from tau2.runner.run_lock import RUN_LOCK_FILENAME, claim_run_directories
from tau2.utils.utils import DATA_DIR

TRIAL_ARTIFACT_SCHEMA_VERSION = "tau-multi-naturalness-trial0-v1"
VALIDATION_CACHE_SCHEMA_VERSION = "tau-multi-naturalness-validation-cache-v1"
CANONICAL_EXPERIENCE_PATH = Path("papers/tau-multilingual/reproduction/experience.json")
CANONICAL_VALIDATION_PATH = Path(
    "data/simulations/paper_runs/tau-multi/human_annotations/validations"
)
CANONICAL_EVIDENCE_PREFIX = Path("data/simulations/paper_runs/tau-multi")
CANONICAL_EXPERIENCE_INSTRUMENT_VERSION = "2.2.0"
CANONICAL_EXPERIENCE_SHA256 = (
    "2087ea23bcb9c14cd73cb945e70f53ffd0685d2c36c5492d3c7a325b44f1369f"
)
CANONICAL_WORK_FINGERPRINT_SHA256 = (
    "1cf1258a9f8e624c0334059b62f5e3fd7e9b2b1bc29fc3feef6235d523347e59"
)
CANONICAL_UTTERANCES = 59_680
CANONICAL_ZERO_UTTERANCE_SIMULATIONS = 14
CODE_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_LANGUAGES = ("en", "es", "pt", "hi", "ko", "zh")
CANONICAL_DOMAINS = ("airline", "retail", "telecom")
CANONICAL_SYSTEMS = (
    "OpenAI minimal",
    "OpenAI xhigh",
    "Gemini minimal",
    "Gemini high",
    "xAI",
)
CANONICAL_SYSTEM_SLUGS = (
    "openai_minimal",
    "openai_xhigh",
    "gemini_minimal",
    "gemini_high",
    "xai_provider_default",
)

WorkItemT = TypeVar("WorkItemT")
ResultT = TypeVar("ResultT")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
            handle.write("\n")
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _safe_name(value: str) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{stem[:64]}-{digest}"


def _code_provenance() -> TrialCodeProvenance:
    code_relatives = (
        "src/tau2/config.py",
        "src/tau2/judges/nativeness/factors.py",
        "src/tau2/judges/nativeness/harness.py",
        "src/tau2/judges/nativeness/judge.py",
        "src/tau2/judges/nativeness/paper_trial.py",
        "src/tau2/judges/nativeness/validation.py",
        "src/tau2/multilingual/nativeness_catalog.py",
    )
    source_files: dict[str, str] = {}
    for relative in code_relatives:
        path = CODE_ROOT / relative
        if not path.is_file():
            raise ValueError(f"runtime judge source file is missing: {path}")
        source_files[f"code:{relative}"] = sha256_file(path)
    data_root = DATA_DIR.resolve()
    for language in VALIDATION_LANGUAGES:
        relative = Path("tau2/multilingual") / language / "pack.yaml"
        path = data_root / relative
        if not path.is_file():
            raise ValueError(f"runtime language pack is missing: {path}")
        source_files[f"data:{relative.as_posix()}"] = sha256_file(path)

    try:
        commit = subprocess.run(
            ["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(CODE_ROOT),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = None
    return TrialCodeProvenance(
        code_root=str(CODE_ROOT),
        data_root=str(data_root),
        git_commit=commit,
        git_dirty=dirty,
        source_fingerprint_sha256=_sha256_value(source_files),
        source_files=dict(sorted(source_files.items())),
    )


class FrozenJudgeSpec(BaseModel):
    """Every prompt-affecting input to one frozen judge call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: Literal["natural_word_choice"]
    factor_name: Literal["combined_utterance_naturalness"]
    model: Annotated[str, Field(description="LLM judge model id.")]
    model_args: Annotated[dict, Field(description="Arguments passed to the model.")]
    prompt_version: Annotated[str, Field(description="Judge wrapper prompt version.")]
    rubric_version: Annotated[str, Field(description="Nativeness rubric version.")]
    criterion: Annotated[
        NativenessJudgeCriterion, Field(description="Complete language criterion.")
    ]


def frozen_judge_spec(language: str) -> FrozenJudgeSpec:
    """Build the frozen spec from the production pack, with wiring guards."""
    factor = runtime_combined_factor(language)
    assert factor.params is not None
    return FrozenJudgeSpec(
        factor_id=VALIDATION_FACTOR_ID,
        factor_name=VALIDATION_FACTOR_NAME,
        model=DEFAULT_TAU_MULTI_NATURALNESS_JUDGE,
        model_args=dict(DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS),
        prompt_version=NATIVENESS_JUDGE_PROMPT_VERSION,
        rubric_version=COMBINED_NATURALNESS_RUBRIC_VERSION,
        criterion=NativenessJudgeCriterion.from_rubric(factor.id, factor.params),
    )


class ExperienceSource(BaseModel):
    """One frozen results root recorded by the Experience artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Repository-relative results.json path.")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="File hash.")]
    trial_0_calls: Annotated[int, Field(ge=0, description="Trial-0 index rows.")]


class ExperienceCohort(BaseModel):
    """Closed paper cohort dimensions needed by this runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: Annotated[int, Field(ge=0, description="Total trial-0 calls.")]
    calls_per_system: Annotated[int, Field(ge=0, description="Calls per system.")]
    languages: Annotated[list[str], Field(description="Cohort languages.")]
    domains: Annotated[list[str], Field(description="Cohort domains.")]
    systems: Annotated[list[str], Field(description="Cohort systems.")]
    trial: Annotated[int, Field(ge=0, description="Selected trial index.")]


class ExperienceManifest(BaseModel):
    """Projection of the Experience artifact consumed by the judge runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: Annotated[str, Field(description="Experience instrument id.")]
    instrument_version: Annotated[str, Field(description="Instrument version.")]
    cohort: Annotated[ExperienceCohort, Field(description="Closed cohort dimensions.")]
    cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Cohort fingerprint.")
    ]
    results_files: Annotated[
        list[ExperienceSource], Field(description="Exact results root inventory.")
    ]


class CohortContract(BaseModel):
    """Expected shape of a canonical cohort (injectable only for tests)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    languages: tuple[str, ...]
    domains: tuple[str, ...]
    systems: tuple[str, ...]
    system_slugs: tuple[str, ...]
    trial: int
    calls_per_root: int

    @property
    def roots(self) -> int:
        return len(self.languages) * len(self.domains) * len(self.systems)

    @property
    def calls(self) -> int:
        return self.roots * self.calls_per_root


CANONICAL_COHORT_CONTRACT = CohortContract(
    languages=CANONICAL_LANGUAGES,
    domains=CANONICAL_DOMAINS,
    systems=CANONICAL_SYSTEMS,
    system_slugs=CANONICAL_SYSTEM_SLUGS,
    trial=0,
    calls_per_root=50,
)


def _uses_canonical_contract(contract: CohortContract) -> bool:
    return contract == CANONICAL_COHORT_CONTRACT


class TrialSourceIdentity(BaseModel):
    """Move-stable identity for one selected non-English results root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results_path: Annotated[str, Field(description="Repository-relative source path.")]
    results_sha256: Annotated[str, Field(description="Frozen results.json SHA-256.")]
    source_key: Annotated[str, Field(description="Filesystem-safe source namespace.")]
    language: Annotated[str, Field(description="ISO language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    system_slug: Annotated[
        str, Field(description="Canonical Experience-system filesystem slug.")
    ]
    trial_0_calls: Annotated[int, Field(ge=0, description="Selected calls.")]


class PreparedSource(BaseModel):
    """A verified source plus machine-local resolved path."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    identity: Annotated[
        TrialSourceIdentity, Field(description="Stable source identity.")
    ]
    resolved_path: Annotated[Path, Field(description="Machine-local results path.")]
    simulation_ids: Annotated[list[str], Field(description="Selected index ids.")]


class TrialSimulationPlan(BaseModel):
    """Judge-relevant projection of one frozen trial-0 simulation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Annotated[TrialSourceIdentity, Field(description="Source results root.")]
    simulation_id: Annotated[str, Field(description="Frozen simulation id.")]
    task_id: Annotated[str, Field(description="Frozen task id.")]
    trial: Literal[0]
    language: Annotated[str, Field(description="ISO language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    source_simulation_sha256: Annotated[
        str, Field(description="SHA-256 of the source simulation JSON.")
    ]
    judge: Annotated[FrozenJudgeSpec, Field(description="Frozen judge configuration.")]
    turns: Annotated[list[AgentTurn], Field(description="Delivered agent utterances.")]


class TrialUtterancePlan(BaseModel):
    """One independently resumable paid judge input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: TrialSourceIdentity
    simulation_id: str
    task_id: str
    trial: Literal[0]
    language: str
    domain: str
    source_simulation_sha256: str
    judge: FrozenJudgeSpec
    turn: AgentTurn
    input_sha256: str


class ValidationEvidenceIdentity(BaseModel):
    """Validated evidence package that pins the runtime judge rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Repository-relative evidence root.")]
    manifest_sha256: Annotated[
        str, Field(description="Byte hash of the validated evidence manifest.")
    ]
    files: Annotated[
        dict[str, str], Field(description="Validated package file inventory.")
    ]


class TrialPlan(BaseModel):
    """Complete verified replay plan held without source audio payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experience_path: str
    experience_sha256: str
    cohort_fingerprint_sha256: str
    validation_evidence: Optional[ValidationEvidenceIdentity]
    all_sources: list[ExperienceSource]
    selected_sources: list[TrialSourceIdentity]
    simulations: list[TrialSimulationPlan]
    utterances: list[TrialUtterancePlan]


class TrialUtteranceArtifact(BaseModel):
    """Atomic paid result for one delivered agent utterance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-trial0-v1"]
    created_at: str
    input_sha256: str
    source: TrialSourceIdentity
    simulation_id: str
    task_id: str
    trial: Literal[0]
    language: str
    domain: str
    source_simulation_sha256: str
    judge: FrozenJudgeSpec
    turn: AgentTurn
    check: NativenessFactorCheck

    @model_validator(mode="after")
    def _single_unit(self) -> "TrialUtteranceArtifact":
        if self.check.id != VALIDATION_FACTOR_ID:
            raise ValueError("trial artifact contains the wrong factor")
        if self.check.outcome is not JudgeOutcome.ERROR:
            if len(self.check.unit_results) != 1:
                raise ValueError("complete utterance artifact needs one unit result")
            if self.check.unit_results[0].unit_index != self.turn.index:
                raise ValueError("unit result index differs from source turn")
        return self


class TrialSimulationArtifact(BaseModel):
    """Call-level any-fail rollup over atomic utterance artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-trial0-v1"]
    created_at: Annotated[
        Optional[str],
        Field(description="Latest child timestamp, or null for a zero-utterance call."),
    ]
    input_sha256: str
    source: TrialSourceIdentity
    simulation_id: str
    task_id: str
    trial: Literal[0]
    language: str
    domain: str
    source_simulation_sha256: str
    judge: FrozenJudgeSpec
    utterances: list[TrialUtteranceArtifact]
    check: NativenessFactorCheck


class TrialCodeProvenance(BaseModel):
    """Actual code/data checkout supplying the production judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_root: Annotated[str, Field(description="Resolved source checkout root.")]
    data_root: Annotated[str, Field(description="Resolved runtime DATA_DIR.")]
    git_commit: Annotated[str, Field(description="Commit resolved with git -C.")]
    git_dirty: Annotated[
        Optional[bool], Field(description="Checkout dirty state, null outside git.")
    ]
    source_fingerprint_sha256: Annotated[
        str, Field(description="Hash of every runtime-relevant source-file hash.")
    ]
    source_files: Annotated[
        dict[str, str], Field(description="Runtime-relevant source files and hashes.")
    ]


class TrialOutcomeSummary(BaseModel):
    """Call counts and rates for one paper reporting cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: Annotated[int, Field(ge=0, description="Calls in the reporting cell.")]
    scored_calls: Annotated[
        int, Field(ge=0, description="PASS plus FAIL calls used by paper rates.")
    ]
    pass_calls: Annotated[int, Field(ge=0, description="PASS call count.")]
    fail_calls: Annotated[int, Field(ge=0, description="FAIL call count.")]
    no_opportunity_calls: Annotated[
        int, Field(ge=0, description="NO_OPPORTUNITY call count.")
    ]
    error_calls: Annotated[int, Field(ge=0, description="ERROR call count.")]
    pass_rate: Annotated[float, Field(ge=0, le=1, description="PASS / scored calls.")]
    fail_rate: Annotated[float, Field(ge=0, le=1, description="FAIL / scored calls.")]
    coverage_rate: Annotated[
        float, Field(ge=0, le=1, description="Scored calls / all calls.")
    ]
    no_opportunity_rate: Annotated[
        float, Field(ge=0, le=1, description="NO_OPPORTUNITY / calls.")
    ]
    error_rate: Annotated[float, Field(ge=0, le=1, description="ERROR / calls.")]

    @model_validator(mode="after")
    def _counts_and_rates_agree(self) -> "TrialOutcomeSummary":
        counts = (
            self.pass_calls,
            self.fail_calls,
            self.no_opportunity_calls,
            self.error_calls,
        )
        if sum(counts) != self.calls:
            raise ValueError("trial outcome counts do not sum to calls")
        if self.scored_calls != self.pass_calls + self.fail_calls:
            raise ValueError("trial scored_calls does not equal PASS plus FAIL")
        scored_denominator = self.scored_calls or 1
        all_denominator = self.calls or 1
        expected_rates = (
            self.pass_calls / scored_denominator,
            self.fail_calls / scored_denominator,
            self.scored_calls / all_denominator,
            self.no_opportunity_calls / all_denominator,
            self.error_calls / all_denominator,
        )
        observed_rates = (
            self.pass_rate,
            self.fail_rate,
            self.coverage_rate,
            self.no_opportunity_rate,
            self.error_rate,
        )
        if any(
            abs(left - right) > 1e-12
            for left, right in zip(observed_rates, expected_rates, strict=True)
        ):
            raise ValueError("trial outcome rates do not reproduce from counts")
        return self


class TrialAggregateSummary(BaseModel):
    """Paper-ready call outcomes at every relevant cohort grain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall: Annotated[TrialOutcomeSummary, Field(description="All selected calls.")]
    by_language: Annotated[
        dict[str, TrialOutcomeSummary], Field(description="Outcomes by language.")
    ]
    by_language_system: Annotated[
        dict[str, dict[str, TrialOutcomeSummary]],
        Field(description="Outcomes by language then canonical system slug."),
    ]
    by_language_domain_system: Annotated[
        dict[str, dict[str, dict[str, TrialOutcomeSummary]]],
        Field(description="Outcomes by language, domain, then canonical system slug."),
    ]


class TrialInvocation(BaseModel):
    """One invocation appended to the output-root provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    finished_at: str
    code: TrialCodeProvenance
    max_concurrency: int
    judged_utterances: int
    reused_utterances: int
    error_utterances: int


class TrialCounts(BaseModel):
    """Expected and current artifact counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_roots: int
    selected_roots: int
    simulations: int
    utterances: int
    zero_utterance_simulations: int
    complete_utterances: int
    error_utterances: int
    complete_simulations: int
    error_simulations: int


class TrialRunIdentity(BaseModel):
    """Static identity that must match before any result can be resumed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-trial0-v1"]
    experience_path: str
    experience_sha256: str
    cohort_fingerprint_sha256: str
    work_fingerprint_sha256: Annotated[
        str,
        Field(description="All selected simulation bytes and utterance input hashes."),
    ]
    validation_evidence: Optional[ValidationEvidenceIdentity]
    runtime_source_fingerprint_sha256: Annotated[
        str, Field(description="Runtime judge implementation source fingerprint.")
    ]
    all_sources: list[ExperienceSource]
    selected_sources: list[TrialSourceIdentity]
    judges: dict[str, FrozenJudgeSpec]
    trial: Literal[0]


class TrialRunManifest(BaseModel):
    """Output-root provenance and resumability contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-trial0-v1"]
    created_at: str
    updated_at: str
    status: Literal["running", "complete", "complete_with_errors", "source_drift"]
    identity_sha256: str
    identity: TrialRunIdentity
    counts: TrialCounts
    aggregate: Annotated[
        Optional[TrialAggregateSummary],
        Field(description="Call-level paper metrics; null while no rollup exists."),
    ]
    invocations: list[TrialInvocation]

    @model_validator(mode="after")
    def _completed_run_has_aggregate(self) -> "TrialRunManifest":
        if self.status != "running" and self.aggregate is None:
            raise ValueError("completed trial run must include an aggregate summary")
        return self


class TrialPreparationCounts(BaseModel):
    """No-cost cohort counts verified before a paid replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_roots: Annotated[int, Field(ge=0, description="Hashed Experience roots.")]
    selected_roots: Annotated[int, Field(ge=0, description="Non-English roots.")]
    simulations: Annotated[int, Field(ge=0, description="Selected trial-0 calls.")]
    utterances: Annotated[int, Field(ge=0, description="Paid utterance units.")]
    zero_utterance_simulations: Annotated[
        int,
        Field(ge=0, description="Calls producing NO_OPPORTUNITY without an LLM call."),
    ]


class TrialPreparationReport(BaseModel):
    """Typed preflight report with no paid judge calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-trial0-v1"]
    experience_path: Annotated[str, Field(description="Frozen Experience path.")]
    experience_sha256: Annotated[str, Field(description="Experience byte hash.")]
    cohort_fingerprint_sha256: Annotated[
        str, Field(description="Experience cohort fingerprint.")
    ]
    work_fingerprint_sha256: Annotated[
        str, Field(description="Exact selected simulation and utterance work hash.")
    ]
    validation_evidence: Annotated[
        Optional[ValidationEvidenceIdentity],
        Field(description="Validated prompt/results evidence for canonical mode."),
    ]
    code: Annotated[
        TrialCodeProvenance, Field(description="Runtime code and pack provenance.")
    ]
    counts: Annotated[TrialPreparationCounts, Field(description="Verified work size.")]
    judges: Annotated[
        dict[str, FrozenJudgeSpec], Field(description="Frozen judge spec by language.")
    ]
    simulations_by_language: Annotated[
        dict[str, int], Field(description="Selected calls by language.")
    ]
    utterances_by_language: Annotated[
        dict[str, int], Field(description="Paid units by language.")
    ]
    simulations_by_language_system: Annotated[
        dict[str, dict[str, int]],
        Field(description="Selected calls by language and canonical system slug."),
    ]
    utterances_by_language_system: Annotated[
        dict[str, dict[str, int]],
        Field(description="Paid units by language and canonical system slug."),
    ]


class TrialRunConfig(BaseModel):
    """Validated CLI configuration for a canonical trial-0 replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Annotated[Path, Field(description="Reviewer-repository root.")]
    evidence_root: Annotated[
        Optional[Path],
        Field(
            description="Detached frozen tau-multi result bundle; when omitted, "
            "results are read below the reviewer repository."
        ),
    ] = None
    output_root: Annotated[Path, Field(description="Standalone artifact root.")]
    max_concurrency: Annotated[
        int,
        Field(
            ge=1,
            default=DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
            description="Maximum in-flight utterance judge calls.",
        ),
    ]


def _load_experience(path: Path) -> ExperienceManifest:
    raw = json.loads(path.read_text())
    projection = {
        "instrument": raw.get("instrument"),
        "instrument_version": raw.get("instrument_version"),
        "cohort": raw.get("cohort"),
        "cohort_fingerprint_sha256": (raw.get("provenance") or {}).get(
            "cohort_fingerprint_sha256"
        ),
        "results_files": (raw.get("provenance") or {}).get("results_files"),
    }
    return ExperienceManifest.model_validate(projection)


def _cohort_fingerprint(sources: list[ExperienceSource]) -> str:
    payload = "\n".join(
        f"{row.path}\t{row.sha256}\t{row.trial_0_calls}" for row in sources
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _path_cell(path: str, contract: CohortContract) -> tuple[str, str, str]:
    parent = Path(path).parent.name
    match = re.fullmatch(
        rf"({'|'.join(map(re.escape, contract.languages))})_"
        rf"({'|'.join(map(re.escape, contract.domains))})_(.+)",
        parent,
    )
    if match is None or match.group(3) not in contract.system_slugs:
        raise ValueError(f"non-canonical Experience results path: {path}")
    return match.group(1), match.group(2), match.group(3)


def _validate_simulation_id(simulation_id: str) -> None:
    if (
        not simulation_id
        or simulation_id in {".", ".."}
        or "/" in simulation_id
        or "\\" in simulation_id
        or Path(simulation_id).name != simulation_id
    ):
        raise ValueError(f"unsafe simulation id in frozen index: {simulation_id!r}")


def verify_experience_cohort(
    repo_root: Path,
    *,
    contract: CohortContract = CANONICAL_COHORT_CONTRACT,
    evidence_root: Optional[Path] = None,
) -> tuple[ExperienceManifest, list[PreparedSource], str]:
    """Verify all manifest/header hashes and return selected non-English roots."""
    repo_root = repo_root.resolve()
    experience_path = repo_root / CANONICAL_EXPERIENCE_PATH
    experience_sha256 = sha256_file(experience_path)
    experience = _load_experience(experience_path)
    cohort = experience.cohort
    if experience.instrument != "tau-multilingual-utterance-experience":
        raise ValueError("unexpected Experience instrument")
    if _uses_canonical_contract(contract):
        if experience.instrument_version != CANONICAL_EXPERIENCE_INSTRUMENT_VERSION:
            raise ValueError("canonical Experience instrument version drifted")
        if experience_sha256 != CANONICAL_EXPERIENCE_SHA256:
            raise ValueError("canonical Experience artifact hash drifted")
    if (
        tuple(cohort.languages) != contract.languages
        or tuple(cohort.domains) != contract.domains
        or tuple(cohort.systems) != contract.systems
        or cohort.trial != contract.trial
        or cohort.calls != contract.calls
        or cohort.calls_per_system
        != len(contract.languages) * len(contract.domains) * contract.calls_per_root
        or len(experience.results_files) != contract.roots
    ):
        raise ValueError("Experience cohort dimensions drifted")
    if len({source.path for source in experience.results_files}) != contract.roots:
        raise ValueError("Experience source paths are not unique")
    if any(
        source.trial_0_calls != contract.calls_per_root
        for source in experience.results_files
    ):
        raise ValueError("Experience source call counts drifted")
    fingerprint = _cohort_fingerprint(experience.results_files)
    if fingerprint != experience.cohort_fingerprint_sha256:
        raise ValueError("Experience cohort fingerprint does not reproduce")

    grid: set[tuple[str, str, str]] = set()
    selected: list[PreparedSource] = []
    for source in experience.results_files:
        relative = Path(source.path)
        expected_prefix = Path("data/simulations/paper_runs/tau-multi/main_runs")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe Experience source path: {source.path}")
        if not relative.is_relative_to(expected_prefix):
            raise ValueError(f"source is outside canonical main_runs: {source.path}")
        if evidence_root is None:
            source_root = repo_root
            path = (source_root / relative).resolve()
        else:
            source_root = evidence_root.expanduser().resolve()
            path = (
                source_root / relative.relative_to(CANONICAL_EVIDENCE_PREFIX)
            ).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Experience source escapes evidence root: {source.path}")
        if not path.is_file() or sha256_file(path) != source.sha256:
            raise ValueError(f"frozen results hash mismatch: {source.path}")
        language, domain, system = _path_cell(source.path, contract)
        grid.add((language, domain, system))
        meta = Results.load_metadata(path)
        if meta.simulation_index is None:
            raise ValueError(f"source has no simulation index: {source.path}")
        entries = [
            entry for entry in meta.simulation_index if entry.trial == contract.trial
        ]
        if len(entries) != source.trial_0_calls:
            raise ValueError(f"trial-0 index count mismatch: {source.path}")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError(f"duplicate trial-0 simulation id: {source.path}")
        for entry in entries:
            _validate_simulation_id(entry.id)
        if language == "en":
            continue
        identity = TrialSourceIdentity(
            results_path=source.path,
            results_sha256=source.sha256,
            source_key=hashlib.sha256(source.path.encode()).hexdigest()[:16],
            language=language,
            domain=domain,
            system_slug=system,
            trial_0_calls=source.trial_0_calls,
        )
        selected.append(
            PreparedSource(
                identity=identity,
                resolved_path=path,
                simulation_ids=[entry.id for entry in entries],
            )
        )
    expected_grid = {
        (language, domain, system)
        for language in contract.languages
        for domain in contract.domains
        for system in contract.system_slugs
    }
    if grid != expected_grid:
        raise ValueError("Experience source grid is incomplete or duplicated")
    expected_selected = (
        (len(contract.languages) - 1) * len(contract.domains) * len(contract.systems)
    )
    if len(selected) != expected_selected:
        raise ValueError("non-English source count drifted")
    return experience, selected, experience_sha256


def _unit_digest(plan: TrialSimulationPlan, turn: AgentTurn) -> str:
    return _sha256_value(
        {
            "schema_version": TRIAL_ARTIFACT_SCHEMA_VERSION,
            "source": plan.source.model_dump(mode="json"),
            "simulation_id": plan.simulation_id,
            "task_id": plan.task_id,
            "trial": plan.trial,
            "language": plan.language,
            "domain": plan.domain,
            "source_simulation_sha256": plan.source_simulation_sha256,
            "judge": plan.judge.model_dump(mode="json"),
            "turn": turn.model_dump(mode="json"),
        }
    )


def _validated_evidence_identity(repo_root: Path) -> ValidationEvidenceIdentity:
    root = (repo_root / CANONICAL_VALIDATION_PATH).resolve()
    if not root.is_relative_to(repo_root.resolve()):
        raise ValueError("canonical validation evidence escapes repository")
    validate_archive(root.parent)
    manifest_path = root / "manifest.json"
    manifest = ValidationRootManifest.model_validate_json(manifest_path.read_text())
    naturalness_languages = {
        partition.language
        for partition in manifest.partitions
        if partition.measure_id is ArchiveMeasureId.NATURALNESS
    }
    if naturalness_languages != set(VALIDATION_LANGUAGES):
        raise ValueError("canonical validation naturalness coverage drifted")
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    return ValidationEvidenceIdentity(
        path=CANONICAL_VALIDATION_PATH.as_posix(),
        manifest_sha256=sha256_file(manifest_path),
        files=dict(sorted(files.items())),
    )


def prepare_trial_plan(
    repo_root: Path,
    *,
    contract: CohortContract = CANONICAL_COHORT_CONTRACT,
    evidence_root: Optional[Path] = None,
) -> TrialPlan:
    """Verify and project the canonical cohort before any paid call starts."""
    repo_root = repo_root.resolve()
    experience, sources, experience_sha = verify_experience_cohort(
        repo_root, contract=contract, evidence_root=evidence_root
    )
    validation_evidence = (
        _validated_evidence_identity(repo_root)
        if _uses_canonical_contract(contract)
        else None
    )
    simulations: list[TrialSimulationPlan] = []
    utterances: list[TrialUtterancePlan] = []
    seen: set[tuple[str, str]] = set()
    for prepared in sources:
        meta = Results.load_metadata(prepared.resolved_path)
        tasks = {str(task.id) for task in meta.tasks}
        simulations_dir = prepared.resolved_path.parent / "simulations"
        if not simulations_dir.is_dir():
            raise ValueError(
                f"canonical source is not dir-format: {prepared.resolved_path}"
            )
        for simulation_id in prepared.simulation_ids:
            _validate_simulation_id(simulation_id)
            key = (prepared.identity.results_path, simulation_id)
            if key in seen:
                raise ValueError(f"duplicate selected simulation identity: {key}")
            seen.add(key)
            sim_path = (simulations_dir / f"{simulation_id}.json").resolve()
            if not sim_path.is_relative_to(simulations_dir.resolve()):
                raise ValueError(f"simulation path escapes frozen source: {sim_path}")
            raw = sim_path.read_bytes()
            simulation = SimulationRun.model_validate_json(raw)
            if simulation.id != simulation_id or simulation.trial != contract.trial:
                raise ValueError(f"simulation/index identity mismatch: {sim_path}")
            if str(simulation.task_id) not in tasks:
                raise ValueError(f"simulation references missing task: {sim_path}")
            language, _script = get_simulation_language_info(simulation)
            if language != prepared.identity.language:
                raise ValueError(f"simulation language mismatch: {sim_path}")
            judge = frozen_judge_spec(language)
            sim_plan = TrialSimulationPlan(
                source=prepared.identity,
                simulation_id=simulation.id,
                task_id=str(simulation.task_id),
                trial=0,
                language=language,
                domain=prepared.identity.domain,
                source_simulation_sha256=hashlib.sha256(raw).hexdigest(),
                judge=judge,
                turns=build_agent_turns(simulation),
            )
            simulations.append(sim_plan)
            utterances.extend(
                TrialUtterancePlan(
                    source=sim_plan.source,
                    simulation_id=sim_plan.simulation_id,
                    task_id=sim_plan.task_id,
                    trial=0,
                    language=sim_plan.language,
                    domain=sim_plan.domain,
                    source_simulation_sha256=sim_plan.source_simulation_sha256,
                    judge=sim_plan.judge,
                    turn=turn,
                    input_sha256=_unit_digest(sim_plan, turn),
                )
                for turn in sim_plan.turns
            )
    expected_sims = len(sources) * contract.calls_per_root
    if len(simulations) != expected_sims:
        raise ValueError("selected simulation count drifted")
    plan = TrialPlan(
        experience_path=CANONICAL_EXPERIENCE_PATH.as_posix(),
        experience_sha256=experience_sha,
        cohort_fingerprint_sha256=experience.cohort_fingerprint_sha256,
        validation_evidence=validation_evidence,
        all_sources=experience.results_files,
        selected_sources=[source.identity for source in sources],
        simulations=simulations,
        utterances=utterances,
    )
    if _uses_canonical_contract(contract) and (
        _work_fingerprint(plan) != CANONICAL_WORK_FINGERPRINT_SHA256
    ):
        raise ValueError("canonical transcript work fingerprint drifted")
    if _uses_canonical_contract(contract) and (
        len(plan.utterances) != CANONICAL_UTTERANCES
        or sum(not row.turns for row in plan.simulations)
        != CANONICAL_ZERO_UTTERANCE_SIMULATIONS
    ):
        raise ValueError("canonical transcript work counts drifted")
    return plan


def _work_fingerprint(plan: TrialPlan) -> str:
    return _sha256_value(
        {
            "simulations": [
                (
                    row.source.results_path,
                    row.simulation_id,
                    row.source_simulation_sha256,
                )
                for row in plan.simulations
            ],
            "utterances": [row.input_sha256 for row in plan.utterances],
        }
    )


def prepare_trial0_naturalness(
    repo_root: Path,
    *,
    contract: CohortContract = CANONICAL_COHORT_CONTRACT,
    evidence_root: Optional[Path] = None,
) -> TrialPreparationReport:
    """Hash and project the full cohort without making a paid judge call."""
    code = _code_provenance()
    plan = prepare_trial_plan(repo_root, contract=contract, evidence_root=evidence_root)
    simulations_by_language = Counter(row.language for row in plan.simulations)
    utterances_by_language = Counter(row.language for row in plan.utterances)
    simulations_by_language_system: dict[str, Counter[str]] = defaultdict(Counter)
    utterances_by_language_system: dict[str, Counter[str]] = defaultdict(Counter)
    for row in plan.simulations:
        simulations_by_language_system[row.language][row.source.system_slug] += 1
    for row in plan.utterances:
        utterances_by_language_system[row.language][row.source.system_slug] += 1
    return TrialPreparationReport(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        experience_path=plan.experience_path,
        experience_sha256=plan.experience_sha256,
        cohort_fingerprint_sha256=plan.cohort_fingerprint_sha256,
        work_fingerprint_sha256=_work_fingerprint(plan),
        validation_evidence=plan.validation_evidence,
        code=code,
        counts=TrialPreparationCounts(
            verified_roots=len(plan.all_sources),
            selected_roots=len(plan.selected_sources),
            simulations=len(plan.simulations),
            utterances=len(plan.utterances),
            zero_utterance_simulations=sum(not row.turns for row in plan.simulations),
        ),
        judges={
            language: frozen_judge_spec(language)
            for language in sorted({row.language for row in plan.simulations})
        },
        simulations_by_language=dict(sorted(simulations_by_language.items())),
        utterances_by_language=dict(sorted(utterances_by_language.items())),
        simulations_by_language_system={
            language: dict(sorted(counts.items()))
            for language, counts in sorted(simulations_by_language_system.items())
        },
        utterances_by_language_system={
            language: dict(sorted(counts.items()))
            for language, counts in sorted(utterances_by_language_system.items())
        },
    )


def _utterance_path(root: Path, plan: TrialUtterancePlan) -> Path:
    return (
        root
        / "utterances"
        / plan.source.source_key
        / _safe_name(plan.simulation_id)
        / f"turn-{plan.turn.index:04d}.json"
    )


def _simulation_path(root: Path, plan: TrialSimulationPlan) -> Path:
    return (
        root
        / "simulations"
        / plan.source.source_key
        / f"{_safe_name(plan.simulation_id)}.json"
    )


def _error_check(language: str, message: str) -> NativenessFactorCheck:
    factor = runtime_combined_factor(language)
    return NativenessFactorCheck(
        id=factor.id,
        category=factor.category,
        severity=factor.severity,
        outcome=JudgeOutcome.ERROR,
        evidence=message,
        shadow=factor.shadow,
        evaluation_level="utterance",
        observed_severity=0,
        violation_count=0,
        unit_results=[],
    )


def _judge_turn(language: str, turn: AgentTurn) -> NativenessFactorCheck:
    factor = runtime_combined_factor(language)
    evaluation = evaluate_pack_judge_factors(
        [turn],
        language,
        settings=NativenessJudgeSettings(
            llm_judge=True,
            model=DEFAULT_TAU_MULTI_NATURALNESS_JUDGE,
            model_args=dict(DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS),
        ),
        factors=[factor],
    )
    if len(evaluation.checks) != 1:
        raise ValueError("combined judge did not return exactly one factor check")
    return evaluation.checks[0]


def _score_trial_utterance(
    plan: TrialUtterancePlan, output_root: Path
) -> TrialUtteranceArtifact:
    try:
        check = _judge_turn(plan.language, plan.turn)
    except Exception as exc:  # noqa: BLE001 - each paid unit has an error artifact
        logger.warning(
            f"combined naturalness failed for {plan.simulation_id}/"
            f"{plan.turn.index}: {exc}"
        )
        check = _error_check(plan.language, str(exc))
    artifact = TrialUtteranceArtifact(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        created_at=_now(),
        input_sha256=plan.input_sha256,
        source=plan.source,
        simulation_id=plan.simulation_id,
        task_id=plan.task_id,
        trial=0,
        language=plan.language,
        domain=plan.domain,
        source_simulation_sha256=plan.source_simulation_sha256,
        judge=plan.judge,
        turn=plan.turn,
        check=check,
    )
    _atomic_write(_utterance_path(output_root, plan), artifact)
    return artifact


def _load_matching_utterance(
    path: Path, plan: TrialUtterancePlan
) -> Optional[TrialUtteranceArtifact]:
    if not path.exists():
        return None
    try:
        artifact = TrialUtteranceArtifact.model_validate_json(path.read_text())
    except Exception as exc:
        raise ValueError(f"invalid resumable artifact {path}: {exc}") from exc
    if (
        artifact.input_sha256 != plan.input_sha256
        or artifact.source != plan.source
        or artifact.simulation_id != plan.simulation_id
        or artifact.turn != plan.turn
        or artifact.judge != plan.judge
    ):
        raise ValueError(f"resume identity mismatch at {path}; use a new output root")
    return artifact if artifact.check.outcome is not JudgeOutcome.ERROR else None


def _bounded_map(
    fn: Callable[[WorkItemT], ResultT],
    items: Iterable[WorkItemT],
    workers: int,
) -> Iterator[ResultT]:
    iterator = iter(items)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, item) for item in islice(iterator, workers)}
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                if completed % 100 == 0:
                    logger.info(
                        f"combined naturalness: {completed} utterances completed"
                    )
                yield future.result()
            futures |= {
                executor.submit(fn, item) for item in islice(iterator, len(done))
            }


def _aggregate_simulation(
    plan: TrialSimulationPlan, utterances: list[TrialUtteranceArtifact]
) -> TrialSimulationArtifact:
    utterances = sorted(utterances, key=lambda row: row.turn.index)
    errors = [row for row in utterances if row.check.outcome is JudgeOutcome.ERROR]
    units = [unit for row in utterances for unit in row.check.unit_results]
    factor = runtime_combined_factor(plan.language)
    if errors:
        check = NativenessFactorCheck(
            id=factor.id,
            category=factor.category,
            severity=factor.severity,
            outcome=JudgeOutcome.ERROR,
            evidence="; ".join(
                row.check.evidence or "unknown judge error" for row in errors
            ),
            shadow=factor.shadow,
            evaluation_level="utterance",
            observed_severity=0,
            violation_count=0,
            unit_results=units,
        )
    else:
        replies = [
            NativenessJudgeResult(
                factor_id=VALIDATION_FACTOR_ID,
                opportunity=unit.opportunity,
                violated=unit.violated,
                severity=unit.severity,
                reasoning=unit.reasoning,
                quote=unit.quote,
            )
            for unit in units
        ]
        aggregate = aggregate_utterance_results(replies, strategy="any")
        check = NativenessFactorCheck(
            id=factor.id,
            category=factor.category,
            severity=factor.severity,
            outcome=aggregate.outcome,
            evidence=aggregate.evidence,
            quote=aggregate.quote,
            shadow=factor.shadow,
            evaluation_level="utterance",
            observed_severity=aggregate.severity,
            violation_count=aggregate.violation_count,
            unit_results=units,
        )
    input_sha = _sha256_value([row.input_sha256 for row in utterances])
    return TrialSimulationArtifact(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        created_at=max((row.created_at for row in utterances), default=None),
        input_sha256=input_sha,
        source=plan.source,
        simulation_id=plan.simulation_id,
        task_id=plan.task_id,
        trial=0,
        language=plan.language,
        domain=plan.domain,
        source_simulation_sha256=plan.source_simulation_sha256,
        judge=plan.judge,
        utterances=utterances,
        check=check,
    )


def _load_existing_simulation(
    path: Path, plan: TrialSimulationPlan
) -> Optional[TrialSimulationArtifact]:
    if not path.exists():
        return None
    try:
        artifact = TrialSimulationArtifact.model_validate_json(path.read_text())
    except Exception as exc:
        raise ValueError(f"invalid simulation artifact {path}: {exc}") from exc
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
        raise ValueError(f"simulation resume identity mismatch at {path}")
    return artifact


def _write_simulation_if_changed(
    path: Path, plan: TrialSimulationPlan, artifact: TrialSimulationArtifact
) -> None:
    existing = _load_existing_simulation(path, plan)
    if existing is None:
        _atomic_write(path, artifact)
        return
    if existing == artifact:
        return
    existing_had_error = existing.check.outcome is JudgeOutcome.ERROR or any(
        row.check.outcome is JudgeOutcome.ERROR for row in existing.utterances
    )
    if not existing_had_error:
        raise ValueError(f"complete simulation artifact drifted at {path}")
    _atomic_write(path, artifact)


def _outcome_summary(
    artifacts: Iterable[TrialSimulationArtifact],
) -> TrialOutcomeSummary:
    counts = Counter(row.check.outcome for row in artifacts)
    allowed = {
        JudgeOutcome.PASS,
        JudgeOutcome.FAIL,
        JudgeOutcome.NO_OPPORTUNITY,
        JudgeOutcome.ERROR,
    }
    if set(counts) - allowed:
        raise ValueError(f"unexpected trial outcome(s): {set(counts) - allowed}")
    calls = sum(counts.values())
    pass_calls = counts[JudgeOutcome.PASS]
    fail_calls = counts[JudgeOutcome.FAIL]
    scored_calls = pass_calls + fail_calls
    no_opportunity_calls = counts[JudgeOutcome.NO_OPPORTUNITY]
    error_calls = counts[JudgeOutcome.ERROR]
    scored_denominator = scored_calls or 1
    all_denominator = calls or 1
    return TrialOutcomeSummary(
        calls=calls,
        scored_calls=scored_calls,
        pass_calls=pass_calls,
        fail_calls=fail_calls,
        no_opportunity_calls=no_opportunity_calls,
        error_calls=error_calls,
        pass_rate=pass_calls / scored_denominator,
        fail_rate=fail_calls / scored_denominator,
        coverage_rate=scored_calls / all_denominator,
        no_opportunity_rate=no_opportunity_calls / all_denominator,
        error_rate=error_calls / all_denominator,
    )


def _aggregate_trial(
    artifacts: list[TrialSimulationArtifact],
) -> TrialAggregateSummary:
    by_language: dict[str, list[TrialSimulationArtifact]] = defaultdict(list)
    by_language_system: dict[str, dict[str, list[TrialSimulationArtifact]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    by_language_domain_system: dict[
        str, dict[str, dict[str, list[TrialSimulationArtifact]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for artifact in artifacts:
        language = artifact.language
        domain = artifact.domain
        system = artifact.source.system_slug
        by_language[language].append(artifact)
        by_language_system[language][system].append(artifact)
        by_language_domain_system[language][domain][system].append(artifact)
    return TrialAggregateSummary(
        overall=_outcome_summary(artifacts),
        by_language={
            language: _outcome_summary(rows)
            for language, rows in sorted(by_language.items())
        },
        by_language_system={
            language: {
                system: _outcome_summary(rows)
                for system, rows in sorted(systems.items())
            }
            for language, systems in sorted(by_language_system.items())
        },
        by_language_domain_system={
            language: {
                domain: {
                    system: _outcome_summary(rows)
                    for system, rows in sorted(systems.items())
                }
                for domain, systems in sorted(domains.items())
            }
            for language, domains in sorted(by_language_domain_system.items())
        },
    )


def _run_identity(plan: TrialPlan, code: TrialCodeProvenance) -> TrialRunIdentity:
    judges = {
        language: frozen_judge_spec(language) for language in VALIDATION_LANGUAGES
    }
    return TrialRunIdentity(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        experience_path=plan.experience_path,
        experience_sha256=plan.experience_sha256,
        cohort_fingerprint_sha256=plan.cohort_fingerprint_sha256,
        work_fingerprint_sha256=_work_fingerprint(plan),
        validation_evidence=plan.validation_evidence,
        runtime_source_fingerprint_sha256=code.source_fingerprint_sha256,
        all_sources=plan.all_sources,
        selected_sources=plan.selected_sources,
        judges=judges,
        trial=0,
    )


def _initial_manifest(plan: TrialPlan, identity: TrialRunIdentity) -> TrialRunManifest:
    now = _now()
    return TrialRunManifest(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        status="running",
        identity_sha256=_sha256_value(identity.model_dump(mode="json")),
        identity=identity,
        counts=TrialCounts(
            verified_roots=len(plan.all_sources),
            selected_roots=len(plan.selected_sources),
            simulations=len(plan.simulations),
            utterances=len(plan.utterances),
            zero_utterance_simulations=sum(not row.turns for row in plan.simulations),
            complete_utterances=0,
            error_utterances=0,
            complete_simulations=0,
            error_simulations=0,
        ),
        aggregate=None,
        invocations=[],
    )


def _ensure_output_root(
    output_root: Path, plan: TrialPlan, identity: TrialRunIdentity
) -> TrialRunManifest:
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = TrialRunManifest.model_validate_json(manifest_path.read_text())
        expected = _sha256_value(identity.model_dump(mode="json"))
        if manifest.identity_sha256 != expected or manifest.identity != identity:
            raise ValueError("trial output identity drifted; use a new output root")
        return manifest
    if output_root.exists() and any(
        path.name != RUN_LOCK_FILENAME for path in output_root.iterdir()
    ):
        raise ValueError("non-empty trial output has no typed manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _initial_manifest(plan, identity)
    _atomic_write(manifest_path, manifest)
    return manifest


def run_trial0_naturalness(
    config: TrialRunConfig,
    *,
    contract: CohortContract = CANONICAL_COHORT_CONTRACT,
) -> TrialRunManifest:
    """Run/resume the frozen judge over canonical non-English trial-0 calls."""
    repo_root = config.repo_root.resolve()
    output_root = config.output_root.resolve()
    evidence_root = (
        config.evidence_root.expanduser().resolve()
        if config.evidence_root is not None
        else (repo_root / CANONICAL_EVIDENCE_PREFIX).resolve()
    )
    main_runs = (evidence_root / "main_runs").resolve()
    if (
        output_root == repo_root
        or repo_root.is_relative_to(output_root)
        or output_root == main_runs
        or output_root.is_relative_to(main_runs)
    ):
        raise ValueError("output must be outside the frozen main_runs tree")
    with claim_run_directories([output_root]):
        return _run_trial0_naturalness_claimed(config, contract=contract)


def _run_trial0_naturalness_claimed(
    config: TrialRunConfig,
    *,
    contract: CohortContract,
) -> TrialRunManifest:
    started = _now()
    code = _code_provenance()
    repo_root = config.repo_root.resolve()
    output_root = config.output_root.resolve()
    plan = prepare_trial_plan(
        repo_root,
        contract=contract,
        evidence_root=config.evidence_root,
    )
    identity = _run_identity(plan, code)
    prior = _ensure_output_root(output_root, plan, identity)

    cached: dict[tuple[str, str], list[TrialUtteranceArtifact]] = defaultdict(list)
    pending: list[TrialUtterancePlan] = []
    expected_paths = {_utterance_path(output_root, item) for item in plan.utterances}
    existing_paths = (
        set((output_root / "utterances").rglob("*.json"))
        if (output_root / "utterances").exists()
        else set()
    )
    unexpected = existing_paths - expected_paths
    if unexpected:
        raise ValueError(
            f"trial output contains non-cohort utterances: {sorted(unexpected)[:3]}"
        )
    expected_sim_paths = {
        _simulation_path(output_root, item) for item in plan.simulations
    }
    existing_sim_paths = (
        set((output_root / "simulations").rglob("*.json"))
        if (output_root / "simulations").exists()
        else set()
    )
    unexpected_sims = existing_sim_paths - expected_sim_paths
    if unexpected_sims:
        raise ValueError(
            f"trial output contains non-cohort simulations: {sorted(unexpected_sims)[:3]}"
        )
    for sim_plan in plan.simulations:
        _load_existing_simulation(_simulation_path(output_root, sim_plan), sim_plan)
    for item in plan.utterances:
        artifact = _load_matching_utterance(_utterance_path(output_root, item), item)
        if artifact is None:
            pending.append(item)
        else:
            cached[(item.source.source_key, item.simulation_id)].append(artifact)

    records_by_sim = cached
    for artifact in _bounded_map(
        lambda item: _score_trial_utterance(item, output_root),
        pending,
        config.max_concurrency,
    ):
        records_by_sim[(artifact.source.source_key, artifact.simulation_id)].append(
            artifact
        )

    simulation_errors = 0
    utterance_errors = 0
    simulation_artifacts: list[TrialSimulationArtifact] = []
    for sim_plan in plan.simulations:
        records = records_by_sim[(sim_plan.source.source_key, sim_plan.simulation_id)]
        if len(records) != len(sim_plan.turns):
            raise ValueError(f"incomplete utterance set for {sim_plan.simulation_id}")
        sim_artifact = _aggregate_simulation(sim_plan, records)
        _write_simulation_if_changed(
            _simulation_path(output_root, sim_plan), sim_plan, sim_artifact
        )
        simulation_artifacts.append(sim_artifact)
        if sim_artifact.check.outcome is JudgeOutcome.ERROR:
            simulation_errors += 1
        utterance_errors += sum(
            row.check.outcome is JudgeOutcome.ERROR for row in records
        )

    # Detect a concurrent source-header edit before declaring the replay complete.
    drift = [
        source.path
        for source in plan.all_sources
        if sha256_file(repo_root / source.path) != source.sha256
    ]
    status: Literal["complete", "complete_with_errors", "source_drift"]
    status = (
        "source_drift"
        if drift
        else ("complete_with_errors" if utterance_errors else "complete")
    )
    complete_utterances = len(plan.utterances) - utterance_errors
    invocation = TrialInvocation(
        started_at=started,
        finished_at=_now(),
        code=code,
        max_concurrency=config.max_concurrency,
        judged_utterances=len(pending),
        reused_utterances=len(plan.utterances) - len(pending),
        error_utterances=utterance_errors,
    )
    manifest = TrialRunManifest(
        schema_version=TRIAL_ARTIFACT_SCHEMA_VERSION,
        created_at=prior.created_at,
        updated_at=invocation.finished_at,
        status=status,
        identity_sha256=prior.identity_sha256,
        identity=identity,
        counts=TrialCounts(
            verified_roots=len(plan.all_sources),
            selected_roots=len(plan.selected_sources),
            simulations=len(plan.simulations),
            utterances=len(plan.utterances),
            zero_utterance_simulations=sum(not row.turns for row in plan.simulations),
            complete_utterances=complete_utterances,
            error_utterances=utterance_errors,
            complete_simulations=len(plan.simulations) - simulation_errors,
            error_simulations=simulation_errors,
        ),
        aggregate=_aggregate_trial(simulation_artifacts),
        invocations=[*prior.invocations, invocation],
    )
    _atomic_write(output_root / "manifest.json", manifest)
    if drift:
        raise ValueError(f"frozen source headers changed during replay: {drift[:3]}")
    return manifest


class ValidationWorkItem(BaseModel):
    """One labeled row projected into the shared utterance judge seam."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    split: ValidationSplit
    row: ValidationDatasetRow
    turn: AgentTurn
    judge: FrozenJudgeSpec
    input_sha256: str


class ValidationCacheRecord(BaseModel):
    """Atomic raw judge response keyed only by the exact judge input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-validation-cache-v1"]
    created_at: str
    input_sha256: str
    language: str
    turn: AgentTurn
    judge: FrozenJudgeSpec
    check: NativenessFactorCheck


class ValidationCacheIdentity(BaseModel):
    """All inputs and execution settings shared by cached validation rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judges: Annotated[
        dict[str, FrozenJudgeSpec], Field(description="Exact frozen judge by language.")
    ]
    max_concurrency: Annotated[
        int, Field(ge=1, description="Concurrency used for every cached prediction.")
    ]


class ValidationCacheManifest(BaseModel):
    """Strict provenance preventing mixed validation-cache invocations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-naturalness-validation-cache-manifest-v1"]
    identity_sha256: Annotated[str, Field(description="Hash of the typed identity.")]
    identity: Annotated[
        ValidationCacheIdentity, Field(description="Exact cache invocation identity.")
    ]


class ValidationEvaluationReport(BaseModel):
    """Summary of one paid validation replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str
    rows: int
    judged: int
    reused: int
    errors: int
    metrics: dict[str, dict[str, BinaryMetrics]]


def _validation_input_digest(
    language: str, turn: AgentTurn, judge: FrozenJudgeSpec
) -> str:
    return _sha256_value(
        {
            "schema_version": VALIDATION_CACHE_SCHEMA_VERSION,
            "language": language,
            "judge": judge.model_dump(mode="json"),
            "turn": turn.model_dump(mode="json"),
        }
    )


def _validation_digest(row: ValidationDatasetRow, language: str) -> str:
    turn = AgentTurn(
        index=row.agent_turn_index,
        text=row.agent_text,
        preceding_user_text=row.preceding_customer_text,
        interrupted=row.source_interrupted,
    )
    return _validation_input_digest(language, turn, frozen_judge_spec(language))


def _validation_cache_path(cache_root: Path, item: ValidationWorkItem) -> Path:
    return cache_root / "records" / item.language / f"{item.input_sha256}.json"


def _prepare_validation_cache(
    cache_root: Path,
    max_concurrency: int,
) -> None:
    identity = ValidationCacheIdentity(
        judges={
            language: frozen_judge_spec(language) for language in VALIDATION_LANGUAGES
        },
        max_concurrency=max_concurrency,
    )
    expected = ValidationCacheManifest(
        schema_version="tau-multi-naturalness-validation-cache-manifest-v1",
        identity_sha256=_sha256_value(identity.model_dump(mode="json")),
        identity=identity,
    )
    manifest_path = cache_root / "cache-manifest.json"
    existing_json = set(cache_root.rglob("*.json")) if cache_root.exists() else set()
    existing_records = existing_json - {manifest_path}
    if manifest_path.exists():
        observed = ValidationCacheManifest.model_validate_json(
            manifest_path.read_text()
        )
        if observed.identity_sha256 != _sha256_value(
            observed.identity.model_dump(mode="json")
        ):
            raise ValueError("validation cache manifest identity hash drifted")
        if existing_records and observed != expected:
            raise ValueError(
                "validation cache identity or concurrency drifted; use a new cache root"
            )
        if not existing_records and observed != expected:
            _atomic_write(manifest_path, expected)
        for path in existing_records:
            record = ValidationCacheRecord.model_validate_json(path.read_text())
            expected_judge = identity.judges.get(record.language)
            if expected_judge is None or record.judge != expected_judge:
                raise ValueError(f"validation cache record judge drifted: {path}")
            expected_digest = _validation_input_digest(
                record.language, record.turn, record.judge
            )
            expected_path = (
                cache_root / "records" / record.language / f"{expected_digest}.json"
            )
            if record.input_sha256 != expected_digest or path != expected_path:
                raise ValueError(f"validation cache record input drifted: {path}")
        return
    if existing_records:
        raise ValueError("validation cache records have no typed cache manifest")
    _atomic_write(manifest_path, expected)


def _score_validation_item(
    item: ValidationWorkItem, cache_root: Path
) -> ValidationCacheRecord:
    try:
        check = _judge_turn(item.language, item.turn)
    except Exception as exc:  # noqa: BLE001 - row error is serialized and retried
        logger.warning(
            f"validation judge failed for {item.language}/{item.split.value}/"
            f"{item.row.row_index}: {exc}"
        )
        check = _error_check(item.language, str(exc))
    record = ValidationCacheRecord(
        schema_version=VALIDATION_CACHE_SCHEMA_VERSION,
        created_at=_now(),
        input_sha256=item.input_sha256,
        language=item.language,
        turn=item.turn,
        judge=item.judge,
        check=check,
    )
    _atomic_write(_validation_cache_path(cache_root, item), record)
    return record


def _load_validation_cache(
    path: Path, item: ValidationWorkItem
) -> Optional[ValidationCacheRecord]:
    if not path.exists():
        return None
    record = ValidationCacheRecord.model_validate_json(path.read_text())
    if (
        record.input_sha256 != item.input_sha256
        or record.language != item.language
        or record.turn != item.turn
        or record.judge != item.judge
    ):
        raise ValueError(f"validation cache identity mismatch: {path}")
    return record if record.check.outcome is not JudgeOutcome.ERROR else None


def _materialize_validation_result(
    row: ValidationDatasetRow, check: NativenessFactorCheck
) -> ValidationResultRow:
    return ValidationResultRow(
        row_index=row.row_index,
        simulation_id=row.simulation_id,
        agent_turn_index=row.agent_turn_index,
        label_positive=row.label is ValidationLabel.VIOLATION,
        judge_positive=check.outcome is JudgeOutcome.FAIL,
        judge_outcome=check.outcome,
        judge_check=check,
    )


def evaluate_validation_corpus(
    root: Path,
    *,
    max_concurrency: int = DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY,
    cache_root: Optional[Path] = None,
) -> ValidationEvaluationReport:
    """Run/resume all labeled rows and rebuild result files plus manifest."""
    root = root.resolve()
    cache_root = (cache_root or root / ".work").resolve()
    manifest_path = root / "manifest.json"
    manifest = ValidationManifest.model_validate_json(manifest_path.read_text())
    if manifest.acceptance_policy != FIXED_VALIDATION_ACCEPTANCE_POLICY:
        raise ValueError("validation acceptance policy drifted")
    items: list[ValidationWorkItem] = []
    datasets: dict[tuple[str, ValidationSplit], ValidationDataset] = {}
    paths: dict[tuple[str, ValidationSplit], tuple[Path, Path, Path]] = {}
    for language in VALIDATION_LANGUAGES:
        prompt_summary = manifest.languages[language].prompt
        prompt_path = (root / prompt_summary.path).resolve()
        if not prompt_path.is_relative_to(root):
            raise ValueError(f"{language}: prompt path escapes validation package")
        if prompt_path != (root / language / "prompt.json").resolve():
            raise ValueError(f"{language}: prompt path is not canonical")
        prompt = runtime_validation_prompt(language)
        packaged_prompt = ValidationPrompt.model_validate_json(prompt_path.read_text())
        if packaged_prompt != prompt:
            raise ValueError(f"{language}: packaged prompt differs from production")
        prompt_sha = sha256_file(prompt_path)
        if prompt_summary.sha256 != prompt_sha:
            raise ValueError(f"{language}: prompt hash mismatch")
        for split in manifest.split_roles:
            summary = manifest.languages[language].splits[split]
            dataset_path = (root / summary.dataset.path).resolve()
            results_path = (root / summary.results.path).resolve()
            if not dataset_path.is_relative_to(root) or not results_path.is_relative_to(
                root
            ):
                raise ValueError(
                    f"{language}/{split.value}: artifact path escapes validation package"
                )
            expected_split_root = (root / language / split.value).resolve()
            if dataset_path != expected_split_root / "dataset.json":
                raise ValueError(
                    f"{language}/{split.value}: dataset path is not canonical"
                )
            if results_path != expected_split_root / "results.json":
                raise ValueError(
                    f"{language}/{split.value}: results path is not canonical"
                )
            dataset_sha = sha256_file(dataset_path)
            if dataset_sha != summary.dataset.sha256:
                raise ValueError(f"{language}/{split.value}: dataset hash mismatch")
            dataset = ValidationDataset.model_validate_json(dataset_path.read_text())
            if dataset.language != language or dataset.split is not split:
                raise ValueError(f"{language}/{split.value}: dataset identity mismatch")
            datasets[(language, split)] = dataset
            paths[(language, split)] = (dataset_path, prompt_path, results_path)
            judge = frozen_judge_spec(language)
            for row in dataset.rows:
                turn = AgentTurn(
                    index=row.agent_turn_index,
                    text=row.agent_text,
                    preceding_user_text=row.preceding_customer_text,
                    interrupted=row.source_interrupted,
                )
                items.append(
                    ValidationWorkItem(
                        language=language,
                        split=split,
                        row=row,
                        turn=turn,
                        judge=judge,
                        input_sha256=_validation_digest(row, language),
                    )
                )

    cached: dict[str, ValidationCacheRecord] = {}
    pending_by_input: dict[str, ValidationWorkItem] = {}
    _prepare_validation_cache(cache_root, max_concurrency)
    for item in items:
        record = _load_validation_cache(_validation_cache_path(cache_root, item), item)
        if record is None:
            pending_by_input.setdefault(item.input_sha256, item)
        else:
            cached[item.input_sha256] = record
    reused_rows = sum(item.input_sha256 in cached for item in items)
    pending = list(pending_by_input.values())
    for record in _bounded_map(
        lambda item: _score_validation_item(item, cache_root),
        pending,
        max_concurrency,
    ):
        cached[record.input_sha256] = record

    manifest_data = manifest.model_dump(mode="json")
    metrics_report: dict[str, dict[str, BinaryMetrics]] = defaultdict(dict)
    error_count = 0
    item_by_row = {
        (item.language, item.split, item.row.row_index): item for item in items
    }
    for language in VALIDATION_LANGUAGES:
        for split in manifest.split_roles:
            dataset = datasets[(language, split)]
            dataset_path, prompt_path, results_path = paths[(language, split)]
            rows = [
                _materialize_validation_result(
                    row,
                    cached[
                        item_by_row[(language, split, row.row_index)].input_sha256
                    ].check,
                )
                for row in dataset.rows
            ]
            metrics = compute_binary_metrics(rows)
            error_count += metrics.n_errors
            results = ValidationResults(
                schema_version="tau-multi-utterance-naturalness-results-v1",
                language=language,
                split=split,
                role=dataset.role,
                evaluation_level="utterance",
                factor_id=VALIDATION_FACTOR_ID,
                factor_name=VALIDATION_FACTOR_NAME,
                dataset=FileReference(
                    path=os.path.relpath(dataset_path, results_path.parent),
                    sha256=sha256_file(dataset_path),
                ),
                prompt=FileReference(
                    path=os.path.relpath(prompt_path, results_path.parent),
                    sha256=sha256_file(prompt_path),
                ),
                judge=ValidationJudgeConfig(
                    model=DEFAULT_TAU_MULTI_NATURALNESS_JUDGE,
                    model_args=dict(DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS),
                    prompt_version=NATIVENESS_JUDGE_PROMPT_VERSION,
                    rubric_version=COMBINED_NATURALNESS_RUBRIC_VERSION,
                    max_concurrency=max_concurrency,
                ),
                n_rows=len(rows),
                metrics=metrics,
                rows=rows,
            )
            _atomic_write(results_path, results)
            result_sha = sha256_file(results_path)
            relative_result = results_path.relative_to(root).as_posix()
            manifest_data["files"][relative_result] = result_sha
            result_summary = ManifestResultSummary(
                path=relative_result,
                sha256=result_sha,
                max_concurrency=max_concurrency,
                metrics=metrics,
                passes_acceptance_policy=passes_acceptance_policy(
                    metrics, FIXED_VALIDATION_ACCEPTANCE_POLICY
                ),
            )
            manifest_data["languages"][language]["splits"][split.value]["results"] = (
                result_summary.model_dump(mode="json")
            )
            metrics_report[language][split.value] = metrics
    updated_manifest = ValidationManifest.model_validate(manifest_data)
    _atomic_write(manifest_path, updated_manifest)
    _atomic_write_text(root / "README.md", render_validation_readme(updated_manifest))
    if not error_count:
        validate_validation_package(root)
    return ValidationEvaluationReport(
        root=str(root),
        rows=len(items),
        judged=len(pending),
        reused=reused_rows,
        errors=error_count,
        metrics=dict(metrics_report),
    )
