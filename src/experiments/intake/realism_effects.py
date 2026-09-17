"""Reproduce the tau-Intake caller-realism assignment analysis.

The frozen agent-directed and scaffolded grids assign at most one caller
realism to each task/environment unit. Assignment probabilities are known from
the intake complication catalog and vary with the entity bank and value-level
feasibility. This module estimates assignment effects with normalized
inverse-probability (Hajek) means within each acoustic environment, then
averages the three environment-specific contrasts.

The script intentionally analyzes *assignment*, not observed application.
Conditional realisms can remain latent when the interaction that would expose
them never occurs.  ``spelling_opportunity`` reports that first-stage behavior
for the two spelling-dependent realisms without changing the primary
randomized estimand.

The checked-in reviewer archive is the source of truth. The result manifest,
run configurations, prompt manifest, and content-addressed task snapshots
validate each compact transcript before it enters the analysis. The immutable
scoring-correction ledger is then applied by simulation id without modifying
the transcripts. Run from the repository root::

    .venv/bin/python src/experiments/intake/realism_effects.py \
        --arm scaffolded \
        --output papers/tau-intake/v1/reproduction/analysis_inputs/realism_effects.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.data_model.simulation import ComplicationProfile
from tau2.data_model.tasks import Task
from tau2.domains.intake.complications import (
    BANK_COMPLICATIONS,
    COMPLICATION_CATALOG_VERSION,
    ComplicationKind,
    _init_action_args,
    _kind_feasible,
    _parse_task_id,
    effective_kind_rates,
)
from tau2.domains.intake.utils import spoken_form
from tau2.paper.elicitation_scoring import (
    DEFAULT_ARTIFACT,
    ScoringCorrectionArtifact,
)
from tau2.utils.utils import get_now

ANALYSIS_VERSION = "2.1.0"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_RANDOMIZATION_DRAWS = 100_000
DEFAULT_SEED = 42


class Environment(str, Enum):
    """The three linked acoustic realizations in the paper grid."""

    REGULAR = "regular"
    CHANNEL_HEAVY = "chanheavy"
    SPEECH_HEAVY = "speechheavy"


class Arm(str, Enum):
    """Prompt strategy used by the frozen paper grid."""

    AGENT_DIRECTED = "agent_directed"
    SCAFFOLDED = "scaffolded"


class System(str, Enum):
    """Systems included in the caller-realism analysis."""

    GPT_MINIMAL = "gpt_minimal"
    GPT_XHIGH = "gpt_xhigh"
    GEMINI_HIGH = "gemini_high"
    GROK = "grok"


AGENT_DIRECTED_RESULT_PATHS: dict[System, dict[Environment, str]] = {
    System.GPT_MINIMAL: {
        Environment.REGULAR: "main_runs/modeb_openai_minimal_regular_2026-09-02/results.json",
        Environment.CHANNEL_HEAVY: "main_runs/modeb_openai_minimal_chanheavy_2026-09-05/results.json",
        Environment.SPEECH_HEAVY: "main_runs/modeb_openai_minimal_speechheavy_2026-09-05/results.json",
    },
    System.GPT_XHIGH: {
        environment: f"main_runs/modeb_openai_xhigh_{environment.value}_2026-09-02/results.json"
        for environment in Environment
    },
    System.GEMINI_HIGH: {
        environment: f"main_runs/modeb_gemini_high_{environment.value}_2026-09-02/results.json"
        for environment in Environment
    },
    System.GROK: {
        environment: f"main_runs/modeb_xai_10_{environment.value}_2026-09-02/results.json"
        for environment in Environment
    },
}

SCAFFOLDED_RESULT_PATHS: dict[System, dict[Environment, str]] = {
    System.GPT_MINIMAL: {
        Environment.REGULAR: "main_runs/intake_m_openai_minimal_regular/results.json",
        Environment.CHANNEL_HEAVY: "ablations/scaffolded/modea_openai_minimal_chanheavy_2026-09-02/results.json",
        Environment.SPEECH_HEAVY: "main_runs/intake_m_openai_minimal_chanlight_speechheavy/results.json",
    },
    System.GPT_XHIGH: {
        Environment.REGULAR: "main_runs/intake_m_openai_xhigh_regular/results.json",
        Environment.CHANNEL_HEAVY: "ablations/scaffolded/modea_openai_xhigh_chanheavy_2026-09-02/results.json",
        Environment.SPEECH_HEAVY: "main_runs/intake_m_openai_xhigh_chanlight_speechheavy/results.json",
    },
    System.GEMINI_HIGH: {
        Environment.REGULAR: "main_runs/intake_m_gemini_high_regular/results.json",
        Environment.CHANNEL_HEAVY: "ablations/scaffolded/modea_gemini_high_chanheavy_2026-09-02/results.json",
        Environment.SPEECH_HEAVY: "main_runs/intake_m_gemini_high_chanlight_speechheavy/results.json",
    },
    System.GROK: {
        Environment.REGULAR: "main_runs/intake_m_xai_10_regular/results.json",
        Environment.CHANNEL_HEAVY: "ablations/scaffolded/modea_xai_10_chanheavy_2026-09-02/results.json",
        Environment.SPEECH_HEAVY: "main_runs/intake_m_xai_10_chanlight_speechheavy/results.json",
    },
}

RESULT_PATHS_BY_ARM = {
    Arm.AGENT_DIRECTED: AGENT_DIRECTED_RESULT_PATHS,
    Arm.SCAFFOLDED: SCAFFOLDED_RESULT_PATHS,
}

CATALOG_VERSION_BY_ARM_ENVIRONMENT: dict[Arm, dict[Environment, str]] = {
    Arm.AGENT_DIRECTED: {
        environment: COMPLICATION_CATALOG_VERSION for environment in Environment
    },
    Arm.SCAFFOLDED: {
        Environment.REGULAR: "2.3.0",
        Environment.CHANNEL_HEAVY: "2.4.0",
        Environment.SPEECH_HEAVY: "2.3.0",
    },
}

COHORT_BY_ARM = {
    Arm.AGENT_DIRECTED: "paper_agent_directed",
    Arm.SCAFFOLDED: "paper_scaffolded",
}

CONDITION_BY_ENVIRONMENT = {
    Environment.REGULAR: "regular",
    Environment.CHANNEL_HEAVY: "noise_heavy",
    Environment.SPEECH_HEAVY: "speech_heavy",
}

RELEASE_ROOT_PATH = Path("papers/tau-intake/v1/reproduction")
ARCHIVE_MANIFEST_PATH = Path("manifest.json")
RESULT_MANIFEST_PATH = Path("results/manifest.csv")
PROMPT_MANIFEST_PATH = Path("prompts/manifest.json")
REALISM_EVENT_LEDGER_PATH = (
    "papers/tau-intake/v1/reproduction/analysis_inputs/realism_event_ledger.jsonl"
)

REPORTED_KINDS: tuple[ComplicationKind, ...] = (
    ComplicationKind.WRONG_SLOT,
    ComplicationKind.SPELLING_STYLE,
    ComplicationKind.SELF_CORRECTION,
    ComplicationKind.MISPRONOUNCED_TERM,
    ComplicationKind.SPELL_CORRECTION,
)


class ArchiveArtifact(BaseModel):
    """One content-addressed file in the compact reviewer archive."""

    model_config = ConfigDict(extra="ignore")

    path: str
    sha256: str
    rows: Optional[int] = None


class ArchiveManifest(BaseModel):
    """Projection of the compact archive manifest used for hash validation."""

    model_config = ConfigDict(extra="ignore")

    artifacts: list[ArchiveArtifact]


class ResultManifestCell(BaseModel):
    """Frozen cell metadata required by the realism analysis."""

    model_config = ConfigDict(extra="ignore")

    results_path: str
    source_results_sha256: str
    cohort: str
    system: str
    condition: str
    rows: Annotated[int, Field(gt=0)]
    tasks: Annotated[int, Field(gt=0)]
    trials: list[int]
    passes: Annotated[int, Field(ge=0)]
    pass_rate: Annotated[float, Field(ge=0, le=1)]
    git_commit: str
    seed: int
    complication_profile: ComplicationProfile
    complication_rate: Optional[float] = None
    channel_effects_mode: str
    speech_effects_mode: str
    config_file: str
    transcript_file: str
    transcript_sha256: str

    @field_validator("trials", mode="before")
    @classmethod
    def _parse_trials(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("complication_rate", mode="before")
    @classmethod
    def _parse_optional_float(cls, value: object) -> object:
        return None if value in (None, "") else value


class RunInfo(BaseModel):
    """Fields cross-checked against one frozen run configuration."""

    model_config = ConfigDict(extra="ignore")

    git_commit: str
    seed: int
    complication_profile: ComplicationProfile
    complication_rate: Optional[float] = None
    channel_effects_mode: str
    speech_effects_mode: str
    num_trials: Annotated[int, Field(gt=0)]


class RunConfig(BaseModel):
    """Validated wrapper for a frozen run configuration."""

    model_config = ConfigDict(extra="ignore")

    info: RunInfo


class TaskSnapshotRef(BaseModel):
    """Content-addressed task snapshot used by one frozen cell."""

    model_config = ConfigDict(extra="ignore")

    archive_path: str
    sha256: str


class PromptCell(BaseModel):
    """Task-snapshot projection of one prompt-manifest entry."""

    model_config = ConfigDict(extra="ignore")

    cell: str
    task_snapshots: list[TaskSnapshotRef]


class PromptManifest(BaseModel):
    """Validated prompt manifest used to locate historical task contracts."""

    model_config = ConfigDict(extra="ignore")

    cells: list[PromptCell]


class CompactComplication(BaseModel):
    """Assignment fields retained in a compact transcript."""

    model_config = ConfigDict(extra="ignore")

    kind: ComplicationKind
    catalog_version: str


class CompactCall(BaseModel):
    """Compact call fields required by the assignment analysis."""

    model_config = ConfigDict(extra="ignore")

    cell: str
    cohort: str
    simulation_id: str
    task_id: str
    trial: int
    reward: float
    complication: Optional[CompactComplication] = None


class SourceRun(BaseModel):
    """Provenance for one frozen input run."""

    arm: Annotated[Arm, Field(description="Prompt-strategy arm.")]
    system: Annotated[System, Field(description="Paper system label.")]
    environment: Annotated[
        Environment, Field(description="Acoustic realization label.")
    ]
    results_path: Annotated[
        str, Field(description="Detached source-results path from the manifest.")
    ]
    source_results_sha256: Annotated[
        str, Field(description="SHA-256 of the detached source results.json.")
    ]
    transcript_path: Annotated[
        str, Field(description="Repository-relative compact-transcript path.")
    ]
    transcript_sha256: Annotated[
        str, Field(description="SHA-256 of the compact transcript.")
    ]
    run_config_path: Annotated[
        str, Field(description="Repository-relative frozen run configuration.")
    ]
    run_config_sha256: Annotated[
        str, Field(description="SHA-256 of the frozen run configuration.")
    ]
    task_snapshots: Annotated[
        list[TaskSnapshotRef],
        Field(description="Validated task snapshots referenced by the cell."),
    ]
    git_commit: Annotated[str, Field(description="Commit recorded by the run.")]
    run_seed: Annotated[int, Field(description="Complication-assignment seed.")]
    complication_catalog_version: Annotated[
        str, Field(description="Complication catalog used by the frozen run.")
    ]
    complication_rate: Annotated[
        Optional[float],
        Field(description="Uniform trigger override; None uses catalog-native rates."),
    ] = None


class ArchiveSource(BaseModel):
    """Provenance for the compact manifests that gate all source runs."""

    archive_manifest_path: str
    results_manifest_path: str
    results_manifest_sha256: str
    prompt_manifest_path: str
    prompt_manifest_sha256: str


class WeightedGroup(BaseModel):
    """One side of a normalized inverse-probability contrast."""

    units: Annotated[int, Field(ge=0, description="Eligible assignment units.")]
    effective_n: Annotated[
        float, Field(ge=0, description="Kish effective sample size of the weights.")
    ]
    weighted_success: Annotated[
        float, Field(ge=0, le=1, description="Normalized weighted success mean.")
    ]


class EffectEstimate(BaseModel):
    """One environment-stratified Hajek assignment contrast."""

    label: Annotated[str, Field(description="Human-facing realism label.")]
    kind: Annotated[
        Optional[ComplicationKind],
        Field(description="Complication kind; None denotes any realism."),
    ] = None
    resampling_seed: Annotated[
        int,
        Field(description="Seed used for this row's bootstrap and randomization."),
    ]
    effect: Annotated[
        float, Field(description="Weighted realism-minus-clean success contrast.")
    ]
    effect_points: Annotated[
        float, Field(description="Effect expressed in percentage points.")
    ]
    interval_95_points: Annotated[
        tuple[float, float],
        Field(description="Task-clustered 95% bootstrap interval in points."),
    ]
    randomization_p: Annotated[
        float, Field(ge=0, le=1, description="Two-sided randomization p-value.")
    ]
    randomization_p_holm: Annotated[
        Optional[float],
        Field(
            ge=0,
            le=1,
            description="Holm-adjusted p-value across all reported contrasts.",
        ),
    ] = None
    assigned: Annotated[
        WeightedGroup, Field(description="Realism-assigned side of the contrast.")
    ]
    clean: Annotated[
        WeightedGroup, Field(description="Clean-assigned side of the contrast.")
    ]
    environment_effects: Annotated[
        dict[Environment, float],
        Field(description="Hajek effect within each acoustic environment."),
    ]


class SpellingOpportunityRow(BaseModel):
    """Observed spelling opportunity for one spelling-dependent assignment."""

    kind: Annotated[ComplicationKind, Field(description="Assigned realism kind.")]
    assigned_calls: Annotated[int, Field(ge=0, description="Assigned system calls.")]
    calls_with_spelling_event: Annotated[
        int, Field(ge=0, description="Assigned calls containing a spell-out event.")
    ]
    spelling_event_rate: Annotated[
        float, Field(ge=0, le=1, description="Share containing a spell-out event.")
    ]
    calls_with_realism_event: Annotated[
        Optional[int],
        Field(
            ge=0,
            description="Calls with the assigned falter/restart event, when defined.",
        ),
    ] = None


class RealismEventRow(BaseModel):
    """One compact call-level realism-event ledger row."""

    schema_version: Annotated[str, Field(description="Ledger schema version.")]
    cell: Annotated[str, Field(description="Frozen result cell.")]
    system: Annotated[System, Field(description="Paper system key.")]
    environment: Annotated[Environment, Field(description="Acoustic realization.")]
    simulation_id: Annotated[str, Field(description="Frozen simulation id.")]
    source_simulation_sha256: Annotated[
        str, Field(description="Hash of the complete source simulation.")
    ]
    task_id: Annotated[str, Field(description="Frozen task id.")]
    complication_kind: Annotated[
        Optional[ComplicationKind], Field(description="Assigned caller realism.")
    ] = None
    spell_requests: Annotated[int, Field(ge=0, description="Agent spelling requests.")]
    spell_events: Annotated[int, Field(ge=0, description="Caller spell-out events.")]
    spell_restarts: Annotated[int, Field(ge=0, description="Caller spelling restarts.")]
    duration_seconds: Annotated[float, Field(ge=0, description="Call duration.")]


class EventLedgerSource(BaseModel):
    """Provenance for the compact realism-event ledger."""

    path: Annotated[str, Field(description="Repository-relative ledger path.")]
    sha256: Annotated[str, Field(description="SHA-256 of the ledger.")]
    calls: Annotated[int, Field(gt=0, description="Validated ledger rows.")]


class CorrectionSource(BaseModel):
    """Provenance for the immutable reward-correction layer."""

    path: str
    sha256: str
    schema_version: str
    total_corrected_calls: Annotated[int, Field(ge=0)]
    applied_corrected_calls: Annotated[int, Field(ge=0)]


class RepairCostDiagnostic(BaseModel):
    """Weighted descriptive repair-cost contrast for mispronunciation."""

    kind: Annotated[ComplicationKind, Field(description="Assigned realism kind.")]
    spelling_request_effect: Annotated[
        float, Field(description="Weighted change in probability of any request.")
    ]
    spelling_request_effect_points: Annotated[
        float, Field(description="Request-probability change in percentage points.")
    ]
    duration_effect_seconds: Annotated[
        float, Field(description="Weighted change in call duration, seconds.")
    ]
    spelling_request_environment_effects: Annotated[
        dict[Environment, float], Field(description="Request effect by environment.")
    ]
    duration_environment_effects_seconds: Annotated[
        dict[Environment, float], Field(description="Duration effect by environment.")
    ]


class RealismEffectsArtifact(BaseModel):
    """Versioned, provenance-bearing caller-realism analysis artifact."""

    instrument: str = "tau-intake-realism-effects"
    instrument_version: str = ANALYSIS_VERSION
    arm: Annotated[Arm, Field(description="Prompt-strategy arm analyzed.")]
    created_at: Annotated[str, Field(description="Artifact creation timestamp.")]
    complication_catalog_versions: Annotated[
        list[str],
        Field(description="Assignment catalogs used for propensity recovery."),
    ]
    seed: Annotated[int, Field(description="Analysis RNG seed.")]
    bootstrap_resamples: Annotated[int, Field(gt=0)]
    randomization_draws: Annotated[int, Field(gt=0)]
    estimand: Annotated[str, Field(description="Plain-language estimand definition.")]
    interval_method: Annotated[str, Field(description="Confidence-interval method.")]
    significance_method: Annotated[str, Field(description="Hypothesis-test method.")]
    archive: ArchiveSource
    inputs: list[SourceRun]
    scoring_correction: CorrectionSource
    event_ledger: Optional[EventLedgerSource] = None
    effects: list[EffectEstimate]
    spelling_opportunity: list[SpellingOpportunityRow]
    repair_cost_diagnostic: Optional[RepairCostDiagnostic] = None


class AnalysisMatrix(BaseModel):
    """Validated numeric inputs passed between loading and estimation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_ids: list[str]
    tasks: list[Task]
    outcomes: np.ndarray
    assignments: np.ndarray
    kind_probabilities: np.ndarray
    clean_probabilities: np.ndarray
    sources: list[SourceRun]
    archive: ArchiveSource
    scoring_correction: CorrectionSource


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_kind_probabilities(
    task: Task, *, catalog_version: str, complication_rate: Optional[float]
) -> tuple[np.ndarray, float]:
    task_id = str(task.id)
    bank, tier = _parse_task_id(task_id)
    entities = _init_action_args(task, "set_entities")["entities"]
    if len(entities) != 1:
        raise ValueError(f"Expected one entity in {task_id}, got {len(entities)}")
    raw_value = next(iter(entities.values()))
    value = spoken_form(raw_value)
    rates = effective_kind_rates(ComplicationProfile.DEFAULT, complication_rate, bank)
    if catalog_version == "2.3.0":
        rates[ComplicationKind.SPELL_CORRECTION] = 0.0
    elif catalog_version != "2.4.0":
        raise ValueError(f"Unsupported complication catalog {catalog_version}")
    probabilities = np.asarray(
        [
            rates[kind]
            if kind in BANK_COMPLICATIONS[bank]
            and _kind_feasible(kind, bank, tier, value, "voice", raw_value)
            else 0.0
            for kind in ComplicationKind
        ],
        dtype=float,
    )
    clean_probability = 1.0 - float(probabilities.sum())
    if clean_probability < -1e-12:
        raise ValueError(f"Negative clean probability for {task_id}")
    return probabilities, max(0.0, clean_probability)


def _archive_index(
    release_root: Path,
) -> tuple[Path, dict[str, ArchiveArtifact]]:
    manifest_path = release_root / ARCHIVE_MANIFEST_PATH
    manifest = ArchiveManifest.model_validate_json(manifest_path.read_text())
    index = {artifact.path: artifact for artifact in manifest.artifacts}
    if len(index) != len(manifest.artifacts):
        raise ValueError("Compact archive manifest contains duplicate paths")
    return manifest_path, index


def _validated_archive_file(
    release_root: Path,
    relative_path: str | Path,
    archive_index: dict[str, ArchiveArtifact],
) -> tuple[Path, str]:
    key = Path(relative_path).as_posix()
    artifact = archive_index.get(key)
    if artifact is None:
        raise ValueError(f"Compact archive manifest does not contain {key}")
    path = release_root / key
    digest = _sha256(path)
    if digest != artifact.sha256:
        raise ValueError(f"Compact archive hash mismatch: {key}")
    return path, digest


def _result_manifest_cells(path: Path) -> dict[str, ResultManifestCell]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            ResultManifestCell.model_validate(row) for row in csv.DictReader(handle)
        ]
    index = {row.results_path: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Result manifest contains duplicate results paths")
    return index


def _prompt_cells(path: Path) -> dict[str, PromptCell]:
    manifest = PromptManifest.model_validate_json(path.read_text())
    index = {cell.cell: cell for cell in manifest.cells}
    if len(index) != len(manifest.cells):
        raise ValueError("Prompt manifest contains duplicate cells")
    return index


def _tasks_for_cell(
    release_root: Path,
    prompt_cell: PromptCell,
    archive_index: dict[str, ArchiveArtifact],
) -> tuple[list[Task], list[TaskSnapshotRef]]:
    tasks: dict[str, Task] = {}
    if not prompt_cell.task_snapshots:
        raise ValueError(f"Prompt cell has no task snapshot: {prompt_cell.cell}")
    for reference in prompt_cell.task_snapshots:
        path, digest = _validated_archive_file(
            release_root, reference.archive_path, archive_index
        )
        if digest != reference.sha256:
            raise ValueError(
                f"Prompt-manifest task hash mismatch: {reference.archive_path}"
            )
        document = json.loads(path.read_text())
        raw_tasks = document.get("tasks") if isinstance(document, dict) else document
        if not isinstance(raw_tasks, list):
            raise ValueError(
                f"Task snapshot has no task list: {reference.archive_path}"
            )
        for raw_task in raw_tasks:
            task = Task.model_validate(raw_task)
            task_id = str(task.id)
            previous = tasks.get(task_id)
            if previous is not None and previous != task:
                raise ValueError(f"Task snapshot conflict for {task_id}")
            tasks[task_id] = task
    return [tasks[task_id] for task_id in sorted(tasks)], prompt_cell.task_snapshots


def _validate_run_config(
    release_root: Path,
    cell: ResultManifestCell,
    archive_index: dict[str, ArchiveArtifact],
) -> tuple[str, str]:
    path, digest = _validated_archive_file(
        release_root, cell.config_file, archive_index
    )
    config = RunConfig.model_validate_json(path.read_text())
    expected = (
        cell.git_commit,
        cell.seed,
        cell.complication_profile,
        cell.complication_rate,
        cell.channel_effects_mode,
        cell.speech_effects_mode,
        len(cell.trials),
    )
    observed = (
        config.info.git_commit,
        config.info.seed,
        config.info.complication_profile,
        config.info.complication_rate,
        config.info.channel_effects_mode,
        config.info.speech_effects_mode,
        config.info.num_trials,
    )
    if observed != expected:
        raise ValueError(
            f"Run configuration disagrees with manifest: {cell.config_file}"
        )
    return str(RELEASE_ROOT_PATH / cell.config_file), digest


def _load_compact_calls(path: Path) -> list[CompactCall]:
    return [
        CompactCall.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line
    ]


def _load_corrections(
    release_root: Path,
    *,
    results_manifest_sha256: str,
    prompt_manifest_sha256: str,
) -> tuple[ScoringCorrectionArtifact, dict[str, float], Path]:
    path = release_root / DEFAULT_ARTIFACT
    artifact = ScoringCorrectionArtifact.model_validate_json(path.read_text())
    if artifact.corrected_calls != 80 or len(artifact.calls) != 80:
        raise ValueError("Expected the immutable 80-call scoring-correction ledger")
    if artifact.results_manifest_path != RESULT_MANIFEST_PATH.as_posix():
        raise ValueError("Correction ledger references a different result manifest")
    if artifact.results_manifest_sha256 != results_manifest_sha256:
        raise ValueError("Correction ledger result-manifest hash mismatch")
    if artifact.prompt_manifest_path != PROMPT_MANIFEST_PATH.as_posix():
        raise ValueError("Correction ledger references a different prompt manifest")
    if artifact.prompt_manifest_sha256 != prompt_manifest_sha256:
        raise ValueError("Correction ledger prompt-manifest hash mismatch")
    corrections = {row.simulation_id: row.corrected_reward for row in artifact.calls}
    if len(corrections) != artifact.corrected_calls:
        raise ValueError("Correction ledger contains duplicate simulation ids")
    return artifact, corrections, path


def load_matrix(repo_root: Path, *, arm: Arm = Arm.AGENT_DIRECTED) -> AnalysisMatrix:
    """Load and validate one archived four-system by three-environment grid."""
    environments = list(Environment)
    systems = list(System)
    outcomes: Optional[np.ndarray] = None
    assignments: Optional[np.ndarray] = None
    kind_probabilities: Optional[np.ndarray] = None
    clean_probabilities: Optional[np.ndarray] = None
    task_ids: Optional[list[str]] = None
    tasks: Optional[list[Task]] = None
    sources: list[SourceRun] = []
    release_root = repo_root / RELEASE_ROOT_PATH
    archive_manifest_path, archive_index = _archive_index(release_root)
    results_manifest_path, results_manifest_sha256 = _validated_archive_file(
        release_root, RESULT_MANIFEST_PATH, archive_index
    )
    prompt_manifest_path, prompt_manifest_sha256 = _validated_archive_file(
        release_root, PROMPT_MANIFEST_PATH, archive_index
    )
    result_cells = _result_manifest_cells(results_manifest_path)
    prompt_cells = _prompt_cells(prompt_manifest_path)
    correction_artifact, corrections, correction_path = _load_corrections(
        release_root,
        results_manifest_sha256=results_manifest_sha256,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )
    all_simulation_ids: set[str] = set()
    applied_corrections: set[str] = set()
    result_paths = RESULT_PATHS_BY_ARM[arm]
    cohort = COHORT_BY_ARM[arm]

    for environment_index, environment in enumerate(environments):
        reference_assignments: Optional[list[int]] = None
        environment_seed: Optional[int] = None
        environment_rate: Optional[float] = None
        environment_tasks: Optional[list[Task]] = None
        catalog_version = CATALOG_VERSION_BY_ARM_ENVIRONMENT[arm][environment]
        for system_index, system in enumerate(systems):
            results_path = result_paths[system][environment]
            cell = result_cells.get(results_path)
            if cell is None:
                raise ValueError(f"Result manifest does not contain {results_path}")
            if (
                cell.cohort != cohort
                or cell.system != system.value
                or cell.condition != CONDITION_BY_ENVIRONMENT[environment]
            ):
                raise ValueError(f"Unexpected cell labels for {results_path}")
            transcript_path, transcript_sha256 = _validated_archive_file(
                release_root, cell.transcript_file, archive_index
            )
            if transcript_sha256 != cell.transcript_sha256:
                raise ValueError(
                    f"Transcript hash disagrees with manifest: {results_path}"
                )
            rows = sorted(
                _load_compact_calls(transcript_path), key=lambda row: row.task_id
            )
            current_ids = [row.task_id for row in rows]
            current_trials = sorted({row.trial for row in rows})
            current_simulation_ids = {row.simulation_id for row in rows}
            if len(current_simulation_ids) != len(rows):
                raise ValueError(f"Duplicate simulation ids in {cell.transcript_file}")
            if all_simulation_ids & current_simulation_ids:
                raise ValueError("Simulation ids overlap across selected cells")
            all_simulation_ids.update(current_simulation_ids)
            if len(rows) != cell.rows:
                raise ValueError(f"Row count differs from manifest: {results_path}")
            if len(set(current_ids)) != cell.tasks:
                raise ValueError(f"Task count differs from manifest: {results_path}")
            if current_trials != cell.trials:
                raise ValueError(f"Trial set differs from manifest: {results_path}")
            if sum(row.reward for row in rows) != cell.passes:
                raise ValueError(
                    f"Original pass count differs from manifest: {results_path}"
                )
            if not np.isclose(cell.pass_rate, cell.passes / cell.rows):
                raise ValueError(f"Pass rate differs from manifest: {results_path}")
            if any(row.cell != results_path or row.cohort != cohort for row in rows):
                raise ValueError(
                    f"Compact row identity differs from manifest: {results_path}"
                )

            prompt_cell = prompt_cells.get(results_path)
            if prompt_cell is None:
                raise ValueError(f"Prompt manifest does not contain {results_path}")
            current_tasks, task_snapshots = _tasks_for_cell(
                release_root, prompt_cell, archive_index
            )
            snapshot_ids = [str(task.id) for task in current_tasks]
            if current_ids != snapshot_ids:
                raise ValueError(
                    f"Task snapshot differs from transcript: {results_path}"
                )
            if task_ids is None:
                task_ids = current_ids
                tasks = current_tasks
                outcomes = np.zeros(
                    (len(task_ids), len(environments), len(systems)), dtype=float
                )
                assignments = np.full((len(task_ids), len(environments)), -1, dtype=int)
                kind_probabilities = np.zeros(
                    (len(task_ids), len(environments), len(ComplicationKind)),
                    dtype=float,
                )
                clean_probabilities = np.zeros(
                    (len(task_ids), len(environments)), dtype=float
                )
            elif current_ids != task_ids or current_tasks != tasks:
                raise ValueError(f"Task alignment differs in {cell.transcript_file}")
            if len(current_ids) != 200:
                raise ValueError(f"Expected 200 tasks in {cell.transcript_file}")

            run_config_path, run_config_sha256 = _validate_run_config(
                release_root, cell, archive_index
            )
            if environment_seed is None:
                environment_seed = cell.seed
                environment_rate = cell.complication_rate
                environment_tasks = current_tasks
            elif (
                cell.seed != environment_seed
                or cell.complication_rate != environment_rate
            ):
                raise ValueError(
                    f"Assignment configuration differs within {environment.value}"
                )

            observed_assignments: list[int] = []
            for task_index, row in enumerate(rows):
                if row.complication is None:
                    observed_index = -1
                else:
                    if row.complication.catalog_version != catalog_version:
                        raise ValueError(
                            "Recorded complication catalog differs from frozen run: "
                            f"{row.simulation_id}"
                        )
                    observed_index = list(ComplicationKind).index(row.complication.kind)
                observed_assignments.append(observed_index)
                reward = float(corrections.get(row.simulation_id, row.reward))
                if row.simulation_id in corrections:
                    if row.reward != 0.0 or reward != 1.0:
                        raise ValueError(
                            f"Invalid correction target: {row.simulation_id}"
                        )
                    applied_corrections.add(row.simulation_id)
                if reward not in (0.0, 1.0):
                    raise ValueError(
                        f"Non-binary reward for {row.task_id} in {cell.transcript_file}"
                    )
                assert outcomes is not None
                outcomes[task_index, environment_index, system_index] = reward
            if reference_assignments is None:
                reference_assignments = observed_assignments
                assert assignments is not None
                assignments[:, environment_index] = observed_assignments
            elif observed_assignments != reference_assignments:
                differing_task = next(
                    task_id
                    for task_id, observed, expected in zip(
                        current_ids,
                        observed_assignments,
                        reference_assignments,
                        strict=True,
                    )
                    if observed != expected
                )
                raise ValueError(
                    "Recorded complication differs across systems for "
                    f"{differing_task} in {environment.value}"
                )
            sources.append(
                SourceRun(
                    arm=arm,
                    system=system,
                    environment=environment,
                    results_path=results_path,
                    source_results_sha256=cell.source_results_sha256,
                    transcript_path=str(RELEASE_ROOT_PATH / cell.transcript_file),
                    transcript_sha256=transcript_sha256,
                    run_config_path=run_config_path,
                    run_config_sha256=run_config_sha256,
                    task_snapshots=task_snapshots,
                    git_commit=cell.git_commit,
                    run_seed=cell.seed,
                    complication_catalog_version=catalog_version,
                    complication_rate=cell.complication_rate,
                )
            )

        assert environment_seed is not None
        assert environment_tasks is not None
        assert assignments is not None
        assert kind_probabilities is not None
        assert clean_probabilities is not None
        for task_index, task in enumerate(environment_tasks):
            probabilities, clean_probability = _task_kind_probabilities(
                task,
                catalog_version=catalog_version,
                complication_rate=environment_rate,
            )
            kind_probabilities[task_index, environment_index] = probabilities
            clean_probabilities[task_index, environment_index] = clean_probability

    selected_corrections = {
        row.simulation_id for row in correction_artifact.calls if row.cohort == cohort
    }
    if applied_corrections != selected_corrections:
        missing = sorted(selected_corrections - applied_corrections)
        extra = sorted(applied_corrections - selected_corrections)
        raise ValueError(
            "Correction coverage differs for selected arm; "
            f"missing={missing}, extra={extra}"
        )
    expected_applied = correction_artifact.corrected_calls_by_cohort.get(cohort, 0)
    if len(applied_corrections) != expected_applied:
        raise ValueError(
            f"Expected {expected_applied} corrections for {arm.value}, "
            f"applied {len(applied_corrections)}"
        )

    assert task_ids is not None
    assert tasks is not None
    assert outcomes is not None
    assert assignments is not None
    assert kind_probabilities is not None
    assert clean_probabilities is not None
    return AnalysisMatrix(
        task_ids=task_ids,
        tasks=tasks,
        outcomes=outcomes,
        assignments=assignments,
        kind_probabilities=kind_probabilities,
        clean_probabilities=clean_probabilities,
        sources=sources,
        archive=ArchiveSource(
            archive_manifest_path=str(RELEASE_ROOT_PATH / ARCHIVE_MANIFEST_PATH),
            results_manifest_path=str(RELEASE_ROOT_PATH / RESULT_MANIFEST_PATH),
            results_manifest_sha256=results_manifest_sha256,
            prompt_manifest_path=str(RELEASE_ROOT_PATH / PROMPT_MANIFEST_PATH),
            prompt_manifest_sha256=prompt_manifest_sha256,
        ),
        scoring_correction=CorrectionSource(
            path=str(RELEASE_ROOT_PATH / DEFAULT_ARTIFACT),
            sha256=_sha256(correction_path),
            schema_version=correction_artifact.schema_version,
            total_corrected_calls=correction_artifact.corrected_calls,
            applied_corrected_calls=len(applied_corrections),
        ),
    )


def _kish_effective_n(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _effect_for_indices(
    matrix: AnalysisMatrix,
    indices: np.ndarray,
    kind: Optional[ComplicationKind],
) -> tuple[float, dict[Environment, float]]:
    outcomes = matrix.outcomes[indices].mean(axis=2)
    assignments = matrix.assignments[indices]
    probabilities = matrix.kind_probabilities[indices]
    clean_probabilities = matrix.clean_probabilities[indices]
    kind_index = None if kind is None else list(ComplicationKind).index(kind)
    environment_effects: dict[Environment, float] = {}
    for environment_index, environment in enumerate(Environment):
        treatment_probability = (
            probabilities[:, environment_index].sum(axis=1)
            if kind_index is None
            else probabilities[:, environment_index, kind_index]
        )
        treated = (
            assignments[:, environment_index] >= 0
            if kind_index is None
            else assignments[:, environment_index] == kind_index
        )
        clean = assignments[:, environment_index] < 0
        positive = (treatment_probability > 0) & (
            clean_probabilities[:, environment_index] > 0
        )
        treated &= positive
        clean &= positive
        if not treated.any() or not clean.any():
            return float("nan"), {}
        treated_weights = 1.0 / treatment_probability[treated]
        clean_weights = 1.0 / clean_probabilities[:, environment_index][clean]
        treated_mean = np.average(
            outcomes[:, environment_index][treated], weights=treated_weights
        )
        clean_mean = np.average(
            outcomes[:, environment_index][clean], weights=clean_weights
        )
        environment_effects[environment] = float(treated_mean - clean_mean)
    return float(np.mean(list(environment_effects.values()))), environment_effects


def _groups(
    matrix: AnalysisMatrix, kind: Optional[ComplicationKind]
) -> tuple[WeightedGroup, WeightedGroup]:
    outcomes = matrix.outcomes.mean(axis=2)
    assignments = matrix.assignments
    kind_index = None if kind is None else list(ComplicationKind).index(kind)
    treatment_probability = (
        matrix.kind_probabilities.sum(axis=2)
        if kind_index is None
        else matrix.kind_probabilities[:, :, kind_index]
    )
    treated = assignments >= 0 if kind_index is None else assignments == kind_index
    clean = assignments < 0
    positive = (treatment_probability > 0) & (matrix.clean_probabilities > 0)
    treated &= positive
    clean &= positive
    treated_weights = 1.0 / treatment_probability[treated]
    clean_weights = 1.0 / matrix.clean_probabilities[clean]
    return (
        WeightedGroup(
            units=int(treated.sum()),
            effective_n=_kish_effective_n(treated_weights),
            weighted_success=float(
                np.average(outcomes[treated], weights=treated_weights)
            ),
        ),
        WeightedGroup(
            units=int(clean.sum()),
            effective_n=_kish_effective_n(clean_weights),
            weighted_success=float(np.average(outcomes[clean], weights=clean_weights)),
        ),
    )


def _bootstrap_effects(
    matrix: AnalysisMatrix,
    kind: Optional[ComplicationKind],
    *,
    num_resamples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(matrix.task_ids), size=(num_resamples, len(matrix.task_ids))
    )
    estimates = np.asarray(
        [_effect_for_indices(matrix, indices, kind)[0] for indices in draws]
    )
    return estimates[np.isfinite(estimates)]


def _randomization_p(
    matrix: AnalysisMatrix,
    kind: Optional[ComplicationKind],
    *,
    num_draws: int,
    seed: int,
) -> float:
    """Sharp-null test by redrawing the catalog's categorical assignment."""
    observed, _ = _effect_for_indices(matrix, np.arange(len(matrix.task_ids)), kind)
    rng = np.random.default_rng(seed)
    outcomes = matrix.outcomes.mean(axis=2)
    kind_index = None if kind is None else list(ComplicationKind).index(kind)
    exceedances = 0
    completed = 0
    batch_size = 2_000
    probabilities = np.concatenate(
        [matrix.clean_probabilities[..., None], matrix.kind_probabilities], axis=2
    )
    cumulative = np.cumsum(probabilities, axis=2)
    while completed < num_draws:
        batch = min(batch_size, num_draws - completed)
        rolls = rng.random((batch, *matrix.assignments.shape))
        redrawn = (rolls[..., None] > cumulative[None, ...]).sum(axis=3) - 1
        effects = np.zeros(batch, dtype=float)
        valid = np.ones(batch, dtype=bool)
        for environment_index in range(len(Environment)):
            treatment_probability = (
                matrix.kind_probabilities[:, environment_index].sum(axis=1)
                if kind_index is None
                else matrix.kind_probabilities[:, environment_index, kind_index]
            )
            positive = (treatment_probability > 0) & (
                matrix.clean_probabilities[:, environment_index] > 0
            )
            treated = (
                redrawn[:, :, environment_index] >= 0
                if kind_index is None
                else redrawn[:, :, environment_index] == kind_index
            ) & positive[None, :]
            clean = (redrawn[:, :, environment_index] < 0) & positive[None, :]
            treated_weight = np.zeros_like(treated, dtype=float)
            np.divide(
                treated,
                treatment_probability[None, :],
                out=treated_weight,
                where=treatment_probability[None, :] > 0,
            )
            clean_weight = np.zeros_like(clean, dtype=float)
            np.divide(
                clean,
                matrix.clean_probabilities[None, :, environment_index],
                out=clean_weight,
                where=matrix.clean_probabilities[None, :, environment_index] > 0,
            )
            treated_denominator = treated_weight.sum(axis=1)
            clean_denominator = clean_weight.sum(axis=1)
            valid &= (treated_denominator > 0) & (clean_denominator > 0)
            treated_mean = np.zeros(batch, dtype=float)
            np.divide(
                treated_weight @ outcomes[:, environment_index],
                treated_denominator,
                out=treated_mean,
                where=treated_denominator > 0,
            )
            clean_mean = np.zeros(batch, dtype=float)
            np.divide(
                clean_weight @ outcomes[:, environment_index],
                clean_denominator,
                out=clean_mean,
                where=clean_denominator > 0,
            )
            effects += (treated_mean - clean_mean) / len(Environment)
        exceedances += int((np.abs(effects[valid]) >= abs(observed) - 1e-15).sum())
        completed += int(valid.sum())
    return (exceedances + 1) / (completed + 1)


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (len(p_values) - rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def _label(kind: Optional[ComplicationKind]) -> str:
    labels = {
        None: "Any caller realism",
        ComplicationKind.WRONG_SLOT: "Wrong-field answer",
        ComplicationKind.SPELLING_STYLE: "Spelling variation",
        ComplicationKind.SELF_CORRECTION: "Self-correction",
        ComplicationKind.MISPRONOUNCED_TERM: "Mispronunciation",
        ComplicationKind.SPELL_CORRECTION: "Falter and restart",
    }
    return labels[kind]


def _estimate(
    matrix: AnalysisMatrix,
    kind: Optional[ComplicationKind],
    *,
    bootstrap_resamples: int,
    randomization_draws: int,
    seed: int,
) -> EffectEstimate:
    indices = np.arange(len(matrix.task_ids))
    effect, environment_effects = _effect_for_indices(matrix, indices, kind)
    bootstrap = _bootstrap_effects(
        matrix, kind, num_resamples=bootstrap_resamples, seed=seed
    )
    interval = np.quantile(bootstrap, (0.025, 0.975))
    assigned, clean = _groups(matrix, kind)
    return EffectEstimate(
        label=_label(kind),
        kind=kind,
        resampling_seed=seed,
        effect=effect,
        effect_points=100 * effect,
        interval_95_points=(100 * float(interval[0]), 100 * float(interval[1])),
        randomization_p=_randomization_p(
            matrix, kind, num_draws=randomization_draws, seed=seed
        ),
        assigned=assigned,
        clean=clean,
        environment_effects=environment_effects,
    )


def _load_event_ledger(repo_root: Path) -> tuple[list[RealismEventRow], Path]:
    path = repo_root / REALISM_EVENT_LEDGER_PATH
    rows = [
        RealismEventRow.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line
    ]
    identities = {(row.task_id, row.environment, row.system) for row in rows}
    if len(rows) != 2_400 or len(identities) != len(rows):
        raise ValueError(
            "Expected 2,400 unique task/environment/system realism-event rows; "
            f"found {len(rows)} rows and {len(identities)} identities"
        )
    return rows, path


def _spelling_opportunity(rows: list[RealismEventRow]) -> list[SpellingOpportunityRow]:
    counters = {
        kind: {"assigned": 0, "spelling": 0, "realism": 0}
        for kind in (
            ComplicationKind.SPELLING_STYLE,
            ComplicationKind.SPELL_CORRECTION,
        )
    }
    for row in rows:
        kind = row.complication_kind
        if kind not in counters:
            continue
        counters[kind]["assigned"] += 1
        counters[kind]["spelling"] += int(row.spell_events > 0)
        counters[kind]["realism"] += int(row.spell_restarts > 0)
    output = []
    for kind, counts in counters.items():
        assigned = counts["assigned"]
        output.append(
            SpellingOpportunityRow(
                kind=kind,
                assigned_calls=assigned,
                calls_with_spelling_event=counts["spelling"],
                spelling_event_rate=counts["spelling"] / assigned if assigned else 0.0,
                calls_with_realism_event=(
                    counts["realism"]
                    if kind == ComplicationKind.SPELL_CORRECTION
                    else None
                ),
            )
        )
    return output


def _repair_cost_diagnostic(
    matrix: AnalysisMatrix, rows: list[RealismEventRow]
) -> RepairCostDiagnostic:
    task_indices = {task_id: index for index, task_id in enumerate(matrix.task_ids)}
    request_outcomes = np.zeros_like(matrix.outcomes)
    duration_outcomes = np.zeros_like(matrix.outcomes)
    for row in rows:
        task_index = task_indices[row.task_id]
        environment_index = list(Environment).index(row.environment)
        system_index = list(System).index(row.system)
        request_outcomes[task_index, environment_index, system_index] = int(
            row.spell_requests > 0
        )
        duration_outcomes[task_index, environment_index, system_index] = (
            row.duration_seconds
        )
    indices = np.arange(len(matrix.task_ids))
    request_effect, request_by_environment = _effect_for_indices(
        matrix.model_copy(update={"outcomes": request_outcomes}),
        indices,
        ComplicationKind.MISPRONOUNCED_TERM,
    )
    duration_effect, duration_by_environment = _effect_for_indices(
        matrix.model_copy(update={"outcomes": duration_outcomes}),
        indices,
        ComplicationKind.MISPRONOUNCED_TERM,
    )
    return RepairCostDiagnostic(
        kind=ComplicationKind.MISPRONOUNCED_TERM,
        spelling_request_effect=request_effect,
        spelling_request_effect_points=100 * request_effect,
        duration_effect_seconds=duration_effect,
        spelling_request_environment_effects=request_by_environment,
        duration_environment_effects_seconds=duration_by_environment,
    )


def analyze(
    repo_root: Path,
    *,
    arm: Arm = Arm.AGENT_DIRECTED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    randomization_draws: int = DEFAULT_RANDOMIZATION_DRAWS,
    seed: int = DEFAULT_SEED,
    include_spelling_opportunity: bool = True,
) -> RealismEffectsArtifact:
    """Run one arm's assignment analysis and available event diagnostics."""
    matrix = load_matrix(repo_root, arm=arm)
    reported_kinds = (
        REPORTED_KINDS
        if arm is Arm.AGENT_DIRECTED
        else tuple(
            kind
            for kind in REPORTED_KINDS
            if kind is not ComplicationKind.SPELL_CORRECTION
        )
    )
    reported = (None, *reported_kinds)
    effects = [
        _estimate(
            matrix,
            kind,
            bootstrap_resamples=bootstrap_resamples,
            randomization_draws=randomization_draws,
            seed=seed + effect_index,
        )
        for effect_index, kind in enumerate(reported)
    ]
    adjusted = _holm([effect.randomization_p for effect in effects])
    for effect, p_value in zip(effects, adjusted, strict=True):
        effect.randomization_p_holm = p_value
    event_rows: list[RealismEventRow] = []
    event_source: Optional[EventLedgerSource] = None
    repair_cost: Optional[RepairCostDiagnostic] = None
    if arm is Arm.AGENT_DIRECTED and include_spelling_opportunity:
        event_rows, event_path = _load_event_ledger(repo_root)
        event_source = EventLedgerSource(
            path=REALISM_EVENT_LEDGER_PATH,
            sha256=_sha256(event_path),
            calls=len(event_rows),
        )
        repair_cost = _repair_cost_diagnostic(matrix, event_rows)
    return RealismEffectsArtifact(
        arm=arm,
        created_at=get_now(),
        complication_catalog_versions=sorted(
            set(CATALOG_VERSION_BY_ARM_ENVIRONMENT[arm].values())
        ),
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        randomization_draws=randomization_draws,
        estimand=(
            "Intention-to-treat effect of assigning a caller realism versus a "
            "clean assignment among task/environment units with positive "
            "probability of either assignment under each frozen run's catalog. "
            "Outcomes are averaged over four systems; Hajek contrasts are "
            "computed within each of three acoustic environments and then "
            "averaged."
        ),
        interval_method=(
            "Percentile bootstrap over task ids; all systems and the three "
            "linked environments of a sampled task stay together. Contrast "
            "seeds are the analysis seed plus the displayed row index."
        ),
        significance_method=(
            "Two-sided sharp-null randomization test using the catalog's known "
            "categorical assignment probabilities; +1 correction; Holm across "
            "the overall contrast and the estimable displayed subtypes. Contrast "
            "seeds are the analysis seed plus the displayed row index."
        ),
        archive=matrix.archive,
        inputs=matrix.sources,
        scoring_correction=matrix.scoring_correction,
        event_ledger=event_source,
        effects=effects,
        spelling_opportunity=(_spelling_opportunity(event_rows) if event_rows else []),
        repair_cost_diagnostic=repair_cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce caller-realism assignment effects and tests."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--arm", choices=[arm.value for arm in Arm], default=Arm.AGENT_DIRECTED.value
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument(
        "--randomization-draws", type=int, default=DEFAULT_RANDOMIZATION_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-spelling-opportunity",
        action="store_true",
        help=(
            "Skip the agent-directed observed-event diagnostic. The scaffolded "
            "archive has no corresponding compact event ledger."
        ),
    )
    args = parser.parse_args()
    artifact = analyze(
        args.repo_root.resolve(),
        arm=Arm(args.arm),
        bootstrap_resamples=args.bootstrap_resamples,
        randomization_draws=args.randomization_draws,
        seed=args.seed,
        include_spelling_opportunity=not args.skip_spelling_opportunity,
    )
    rendered = artifact.model_dump_json(indent=2, exclude_none=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return
    output = args.output
    if not output.is_absolute():
        output = args.repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


if __name__ == "__main__":
    main()
