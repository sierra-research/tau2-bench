"""Reproduce the tau-Intake caller-realism assignment analysis.

The frozen agent-directed grid assigns at most one caller realism to each
task/environment unit.  Assignment probabilities are known from the intake
complication catalog and vary with the entity bank and value-level
feasibility.  This module estimates assignment effects with normalized
inverse-probability (Hajek) means within each acoustic environment, then
averages the three environment-specific contrasts.

The script intentionally analyzes *assignment*, not observed application.
Conditional realisms can remain latent when the interaction that would expose
them never occurs.  ``spelling_opportunity`` reports that first-stage behavior
for the two spelling-dependent realisms without changing the primary
randomized estimand.

The checked-in reviewer archive is the source of truth: compact transcripts
provide outcomes and assignments, the compact realism-event ledger provides
observed application and repair-cost inputs, and frozen expected-draw maps
validate each environment's assignment. Run from the repository root::

    .venv/bin/python src/experiments/intake/realism_effects.py \
        --output papers/tau-intake/v1/reproduction/analysis_inputs/realism_effects.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
from tau2.utils.utils import get_now

ANALYSIS_VERSION = "1.1.0"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_RANDOMIZATION_DRAWS = 100_000
DEFAULT_SEED = 42


class Environment(str, Enum):
    """The three linked acoustic realizations in the paper grid."""

    REGULAR = "regular"
    CHANNEL_HEAVY = "chanheavy"
    SPEECH_HEAVY = "speechheavy"


class System(str, Enum):
    """Agent-directed systems included in the caller-realism analysis."""

    GPT_MINIMAL = "gpt_minimal"
    GPT_XHIGH = "gpt_xhigh"
    GEMINI_HIGH = "gemini_high"
    GROK = "grok"


TRANSCRIPT_PATHS: dict[System, dict[Environment, str]] = {
    System.GPT_MINIMAL: {
        Environment.REGULAR: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            "main_runs__modeb_openai_minimal_regular_2026-09-02.jsonl"
        ),
        Environment.CHANNEL_HEAVY: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            "main_runs__modeb_openai_minimal_chanheavy_2026-09-05.jsonl"
        ),
        Environment.SPEECH_HEAVY: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            "main_runs__modeb_openai_minimal_speechheavy_2026-09-05.jsonl"
        ),
    },
    System.GPT_XHIGH: {
        environment: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            f"main_runs__modeb_openai_xhigh_{environment.value}_2026-09-02.jsonl"
        )
        for environment in Environment
    },
    System.GEMINI_HIGH: {
        environment: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            f"main_runs__modeb_gemini_high_{environment.value}_2026-09-02.jsonl"
        )
        for environment in Environment
    },
    System.GROK: {
        environment: (
            "papers/tau-intake/v1/reproduction/transcripts/"
            f"main_runs__modeb_xai_10_{environment.value}_2026-09-02.jsonl"
        )
        for environment in Environment
    },
}

EXPECTED_DRAW_PATHS: dict[Environment, tuple[int, str]] = {
    Environment.REGULAR: (
        9401,
        "papers/tau-intake/v1/reproduction/analysis_inputs/expected_draws_9401.json",
    ),
    Environment.CHANNEL_HEAVY: (
        9402,
        "papers/tau-intake/v1/reproduction/analysis_inputs/expected_draws_9402.json",
    ),
    Environment.SPEECH_HEAVY: (
        9403,
        "papers/tau-intake/v1/reproduction/analysis_inputs/expected_draws_9403.json",
    ),
}

TASK_SNAPSHOT_PATH = (
    "papers/tau-intake/v1/reproduction/prompts/task_snapshots/"
    "a0c6cd32dea7f871fd4f017c64a6769df1ea43c5b994183e25ef5a263fc45210.json"
)
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


class SourceRun(BaseModel):
    """Provenance for one frozen input run."""

    system: Annotated[System, Field(description="Paper system label.")]
    environment: Annotated[
        Environment, Field(description="Acoustic realization label.")
    ]
    transcript_path: Annotated[
        str, Field(description="Repository-relative compact-transcript path.")
    ]
    sha256: Annotated[str, Field(description="SHA-256 of the compact transcript.")]
    run_seed: Annotated[int, Field(description="Complication-assignment seed.")]


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
    created_at: Annotated[str, Field(description="Artifact creation timestamp.")]
    complication_catalog_version: Annotated[
        str, Field(description="Assignment catalog used for propensity recovery.")
    ]
    seed: Annotated[int, Field(description="Analysis RNG seed.")]
    bootstrap_resamples: Annotated[int, Field(gt=0)]
    randomization_draws: Annotated[int, Field(gt=0)]
    estimand: Annotated[str, Field(description="Plain-language estimand definition.")]
    interval_method: Annotated[str, Field(description="Confidence-interval method.")]
    significance_method: Annotated[str, Field(description="Hypothesis-test method.")]
    inputs: list[SourceRun]
    event_ledger: EventLedgerSource
    effects: list[EffectEstimate]
    spelling_opportunity: list[SpellingOpportunityRow]
    repair_cost_diagnostic: RepairCostDiagnostic


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_kind_probabilities(task: Task) -> tuple[np.ndarray, float]:
    task_id = str(task.id)
    bank, tier = _parse_task_id(task_id)
    entities = _init_action_args(task, "set_entities")["entities"]
    if len(entities) != 1:
        raise ValueError(f"Expected one entity in {task_id}, got {len(entities)}")
    raw_value = next(iter(entities.values()))
    value = spoken_form(raw_value)
    rates = effective_kind_rates(ComplicationProfile.DEFAULT, 1.0, bank)
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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_matrix(repo_root: Path) -> AnalysisMatrix:
    """Load and validate the archived four-system by three-environment grid."""
    environments = list(Environment)
    systems = list(System)
    outcomes: Optional[np.ndarray] = None
    assignments: Optional[np.ndarray] = None
    kind_probabilities: Optional[np.ndarray] = None
    clean_probabilities: Optional[np.ndarray] = None
    task_ids: Optional[list[str]] = None
    snapshot = json.loads((repo_root / TASK_SNAPSHOT_PATH).read_text())
    snapshot_tasks = sorted(
        [Task.model_validate(task) for task in snapshot["tasks"]],
        key=lambda task: str(task.id),
    )
    snapshot_ids = [str(task.id) for task in snapshot_tasks]
    sources: list[SourceRun] = []

    for environment_index, environment in enumerate(environments):
        environment_seed, draw_path = EXPECTED_DRAW_PATHS[environment]
        expected_draws = json.loads((repo_root / draw_path).read_text())
        reference_rows: Optional[list[dict]] = None
        for system_index, system in enumerate(systems):
            relative_path = TRANSCRIPT_PATHS[system][environment]
            path = repo_root / relative_path
            rows = sorted(_load_jsonl(path), key=lambda row: row["task_id"])
            current_ids = [row["task_id"] for row in rows]
            if task_ids is None:
                task_ids = current_ids
                if task_ids != snapshot_ids:
                    raise ValueError("Task snapshot does not match compact transcripts")
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
            elif current_ids != task_ids:
                raise ValueError(f"Task alignment differs in {relative_path}")
            if len(current_ids) != 200:
                raise ValueError(f"Expected 200 tasks in {relative_path}")
            if reference_rows is None:
                reference_rows = rows
            for task_index, row in enumerate(rows):
                reward = row["reward"]
                if reward not in (0.0, 1.0):
                    raise ValueError(
                        f"Non-binary reward for {row['task_id']} in {relative_path}"
                    )
                assert outcomes is not None
                outcomes[task_index, environment_index, system_index] = reward
                observed = (
                    None if row["complication"] is None else row["complication"]["kind"]
                )
                if observed != expected_draws[row["task_id"]]:
                    raise ValueError(
                        "Recorded complication differs from frozen draw for "
                        f"{row['task_id']} in {relative_path}"
                    )
            sources.append(
                SourceRun(
                    system=system,
                    environment=environment,
                    transcript_path=relative_path,
                    sha256=_sha256(path),
                    run_seed=environment_seed,
                )
            )
        assert reference_rows is not None
        assert assignments is not None
        assert kind_probabilities is not None
        assert clean_probabilities is not None
        for task_index, task in enumerate(snapshot_tasks):
            probabilities, clean_probability = _task_kind_probabilities(task)
            kind_probabilities[task_index, environment_index] = probabilities
            clean_probabilities[task_index, environment_index] = clean_probability
            expected = expected_draws[str(task.id)]
            if expected is not None:
                assignments[task_index, environment_index] = list(
                    ComplicationKind
                ).index(ComplicationKind(expected))

    assert task_ids is not None
    assert outcomes is not None
    assert assignments is not None
    assert kind_probabilities is not None
    assert clean_probabilities is not None
    matrix = AnalysisMatrix(
        task_ids=task_ids,
        tasks=snapshot_tasks,
        outcomes=outcomes,
        assignments=assignments,
        kind_probabilities=kind_probabilities,
        clean_probabilities=clean_probabilities,
        sources=sources,
    )
    return matrix


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
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    randomization_draws: int = DEFAULT_RANDOMIZATION_DRAWS,
    seed: int = DEFAULT_SEED,
) -> RealismEffectsArtifact:
    """Run the assignment analysis and observed-event diagnostics."""
    matrix = load_matrix(repo_root)
    reported = (None, *REPORTED_KINDS)
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
    event_rows, event_path = _load_event_ledger(repo_root)
    return RealismEffectsArtifact(
        created_at=get_now(),
        complication_catalog_version=COMPLICATION_CATALOG_VERSION,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        randomization_draws=randomization_draws,
        estimand=(
            "Intention-to-treat effect of assigning a caller realism versus a "
            "clean assignment among task/environment units with positive "
            "probability of either assignment. Outcomes are averaged over four "
            "systems; Hajek contrasts are computed within each of three acoustic "
            "environments and then averaged."
        ),
        interval_method=(
            "Percentile bootstrap over task ids; all systems and the three "
            "linked environments of a sampled task stay together. Contrast "
            "seeds are the analysis seed plus the displayed row index."
        ),
        significance_method=(
            "Two-sided sharp-null randomization test using the catalog's known "
            "categorical assignment probabilities; +1 correction; Holm across "
            "the overall contrast and five estimable realism subtypes. Contrast "
            "seeds are the analysis seed plus the displayed row index."
        ),
        inputs=matrix.sources,
        event_ledger=EventLedgerSource(
            path=REALISM_EVENT_LEDGER_PATH,
            sha256=_sha256(event_path),
            calls=len(event_rows),
        ),
        effects=effects,
        spelling_opportunity=_spelling_opportunity(event_rows),
        repair_cost_diagnostic=_repair_cost_diagnostic(matrix, event_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce caller-realism assignment effects and tests."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument(
        "--randomization-draws", type=int, default=DEFAULT_RANDOMIZATION_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    artifact = analyze(
        args.repo_root.resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        randomization_draws=args.randomization_draws,
        seed=args.seed,
    )
    rendered = artifact.model_dump_json(indent=2) + "\n"
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
