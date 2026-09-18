# Copyright Sierra
"""Deterministic turn-taking latency artifact for the τ-Multilingual paper."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.paper.multilingual import (
    DOMAINS,
    LANGUAGES,
    VOICE_SYSTEMS,
    MultilingualAudit,
)

Sha256 = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hexadecimal SHA-256 digest.",
    ),
]

RESULT_PATH_PREFIX = PurePosixPath("data/simulations/paper_runs/tau-multi")
SYSTEM_LABELS = {
    "openai_minimal": "OpenAI minimal",
    "openai_xhigh": "OpenAI xhigh",
    "gemini_minimal": "Gemini minimal",
    "gemini_high": "Gemini high",
    "xai_provider_default": "xAI",
}
EXPECTED_REPLACEMENT_KEYS = tuple(
    f"{language}/retail/{system}"
    for language in ("ko", "zh")
    for system in VOICE_SYSTEMS
)
LATENCY_FORMULA = (
    "For each reporting cell, compute event-weighted response and yield latency "
    "separately, then take their equal-weight mean. Provider values are the "
    "six-language macro mean of language-system values."
)


class _LatencyMetrics(BaseModel):
    """Latency fields retained from one deterministic quality-factor check."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    response_total: Annotated[float | None, Field(ge=0)] = None
    response_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    response_latency_mean: Annotated[float | None, Field(ge=0)] = None
    yield_total: Annotated[float | None, Field(ge=0)] = None
    yield_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    yield_latency_mean: Annotated[float | None, Field(ge=0)] = None


class _LatencyFactorCheck(BaseModel):
    """Minimal typed view of one stored quality-factor check."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: Annotated[str, Field(description="Stable quality-factor identifier.")]
    metrics: Annotated[
        _LatencyMetrics,
        Field(description="Deterministic metrics recorded by the factor."),
    ] = Field(default_factory=_LatencyMetrics)


class _QualityInfo(BaseModel):
    """Minimal typed view of stored call-quality output."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    factor_checks: Annotated[
        tuple[_LatencyFactorCheck, ...],
        Field(description="Quality-factor checks attached to the call."),
    ] = ()


class _SimulationPrefix(BaseModel):
    """Top-level simulation fields needed for latency analysis."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: Annotated[str, Field(description="Simulation identifier.")]
    task_id: Annotated[str, Field(description="Task identifier.")]
    quality_info: Annotated[
        _QualityInfo | None,
        Field(description="Stored deterministic and judged quality output."),
    ] = None


class _SimulationIndexRow(BaseModel):
    """One result-index entry used to resolve a stored simulation."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: Annotated[str, Field(description="Simulation identifier.")]
    task_id: Annotated[str, Field(description="Task identifier.")]
    trial: Annotated[int, Field(ge=0, description="Trial index.")]


class _ResultsIndex(BaseModel):
    """Typed index from one result cell."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    simulation_index: Annotated[
        tuple[_SimulationIndexRow, ...],
        Field(description="Simulation rows stored beside this results file."),
    ]


class LatencyCounts(BaseModel):
    """Sufficient statistics for event-weighted response and yield latency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_seconds_sum: Annotated[
        float, Field(ge=0, description="Sum of response latency times event count.")
    ] = 0.0
    response_events: Annotated[
        float, Field(ge=0, description="Number of scored response events.")
    ] = 0.0
    yield_seconds_sum: Annotated[
        float, Field(ge=0, description="Sum of yield latency times event count.")
    ] = 0.0
    yield_events: Annotated[
        float, Field(ge=0, description="Number of scored yield events.")
    ] = 0.0

    @property
    def latency_seconds(self) -> float:
        """Return the equal-weight mean of available event-weighted channels."""
        channel_means = []
        if self.response_events:
            channel_means.append(self.response_seconds_sum / self.response_events)
        if self.yield_events:
            channel_means.append(self.yield_seconds_sum / self.yield_events)
        if not channel_means:
            raise ValueError("latency counts contain no response or yield events")
        return sum(channel_means) / len(channel_means)

    def plus(self, other: LatencyCounts) -> LatencyCounts:
        """Combine independent sufficient statistics without reweighting calls."""
        return LatencyCounts(
            response_seconds_sum=(
                self.response_seconds_sum + other.response_seconds_sum
            ),
            response_events=self.response_events + other.response_events,
            yield_seconds_sum=self.yield_seconds_sum + other.yield_seconds_sum,
            yield_events=self.yield_events + other.yield_events,
        )


class LatencyInputSnapshot(BaseModel):
    """One result cell and the exact simulation bytes used for its latency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_path: Annotated[
        str,
        Field(
            description=(
                "Canonical repository-relative result path used in cohort identity."
            )
        ),
    ]
    storage_path: Annotated[
        str,
        Field(
            description=(
                "Repository-relative path from which bytes were read; historical "
                "replacement inputs live under validation_runs."
            )
        ),
    ]
    results_sha256: Sha256
    simulations_sha256: Annotated[
        Sha256,
        Field(
            description=(
                "Digest of ordered simulation identifiers, task identifiers, and "
                "individual simulation hashes."
            )
        ),
    ]
    calls: Annotated[int, Field(ge=1, description="Trial-zero calls in the cell.")]
    counts: Annotated[
        LatencyCounts,
        Field(description="Event-level sufficient statistics across the cell."),
    ]
    latency_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Equal-weight mean of response and yield event-weighted means.",
        ),
    ]

    @model_validator(mode="after")
    def validate_paths_and_value(self) -> LatencyInputSnapshot:
        """Reject unsafe paths and latency values detached from their counts."""
        for field_name in ("logical_path", "storage_path"):
            path = PurePosixPath(getattr(self, field_name))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{field_name} must be a safe relative path")
        if not math.isclose(
            self.latency_seconds,
            self.counts.latency_seconds,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("latency value disagrees with sufficient statistics")
        return self


class LatencyCell(BaseModel):
    """Previous and corrected latency inputs for one language/domain/system cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Annotated[str, Field(description="Benchmark language code.")]
    language_name: Annotated[str, Field(description="Paper display language.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    system: Annotated[str, Field(description="Stable voice-system slug.")]
    system_label: Annotated[str, Field(description="Paper display system.")]
    previous: Annotated[
        LatencyInputSnapshot,
        Field(
            description=(
                "Reconstructed pre-replacement input. Nonreplaced cells reuse the "
                "same shared canonical bytes; replaced cells read the archived bytes."
            )
        ),
    ]
    active: Annotated[
        LatencyInputSnapshot,
        Field(description="Corrected active input and latency sufficient statistics."),
    ]
    input_changed: Annotated[
        bool,
        Field(description="Whether result or simulation bytes changed."),
    ]
    latency_changed: Annotated[
        bool,
        Field(description="Whether the recomputed cell latency changed."),
    ]

    @property
    def key(self) -> str:
        """Return a stable serialized identity for comparison summaries."""
        return f"{self.language}/{self.domain}/{self.system}"

    @model_validator(mode="after")
    def validate_identity_and_change_flags(self) -> LatencyCell:
        """Keep display identities and replacement flags derived from evidence."""
        if LANGUAGES.get(self.language) != self.language_name:
            raise ValueError("language display name disagrees with language code")
        if SYSTEM_LABELS.get(self.system) != self.system_label:
            raise ValueError("system display label disagrees with system slug")
        if self.previous.logical_path != self.active.logical_path:
            raise ValueError("previous and active cells must share one logical path")
        input_changed = (
            self.previous.results_sha256 != self.active.results_sha256
            or self.previous.simulations_sha256 != self.active.simulations_sha256
        )
        if self.input_changed != input_changed:
            raise ValueError("input_changed disagrees with source hashes")
        latency_changed = not math.isclose(
            self.previous.latency_seconds,
            self.active.latency_seconds,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        if self.latency_changed != latency_changed:
            raise ValueError("latency_changed disagrees with recomputed values")
        if not input_changed and self.previous != self.active:
            raise ValueError("unchanged input hashes must retain an identical snapshot")
        return self


class LatencyEventRollup(BaseModel):
    """One event-pooled latency value and its sufficient statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_calls: Annotated[int, Field(ge=1, description="Calls represented.")]
    latency_seconds: Annotated[
        float, Field(ge=0, description="Turn-taking latency in seconds.")
    ]
    counts: Annotated[
        LatencyCounts,
        Field(description="Sufficient statistics used to compute this value."),
    ]


class LatencyMacroRollup(BaseModel):
    """One equal-weight macro latency value across lower-level reporting cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_calls: Annotated[int, Field(ge=1, description="Calls represented.")]
    latency_seconds: Annotated[
        float, Field(ge=0, description="Equal-weight macro latency in seconds.")
    ]


class LatencyDescriptiveSummary(BaseModel):
    """Latency at the language-system, provider, and overall paper rollups."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_system: Annotated[
        dict[str, dict[str, LatencyEventRollup]],
        Field(description="Three-domain latency keyed by language and system."),
    ]
    language: Annotated[
        dict[str, LatencyMacroRollup],
        Field(description="Five-system macro latency keyed by language."),
    ]
    provider: Annotated[
        dict[str, LatencyMacroRollup],
        Field(description="Six-language macro latency keyed by system."),
    ]
    overall: Annotated[
        LatencyMacroRollup,
        Field(description="Five-provider macro latency."),
    ]


class LatencyCohort(BaseModel):
    """Dimensions of the complete trial-zero voice cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: Annotated[int, Field(ge=1, description="Voice result cells.")]
    calls: Annotated[int, Field(ge=1, description="Trial-zero calls.")]
    calls_per_cell: Annotated[int, Field(ge=1, description="Calls per voice cell.")]
    languages: Annotated[tuple[str, ...], Field(description="Ordered languages.")]
    domains: Annotated[tuple[str, ...], Field(description="Ordered domains.")]
    systems: Annotated[tuple[str, ...], Field(description="Ordered voice systems.")]
    trial: Annotated[Literal[0], Field(description="Included trial index.")]


class LatencyProvenance(BaseModel):
    """Hashes binding the latency artifact to active and historical evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_artifact_id: Annotated[str, Field(description="Active audit identifier.")]
    audit_sha256: Sha256
    previous_experience_sha256: Sha256
    previous_results_fingerprint_sha256: Sha256
    active_results_fingerprint_sha256: Sha256
    previous_simulations_fingerprint_sha256: Sha256
    active_simulations_fingerprint_sha256: Sha256


class LatencyReplacementComparison(BaseModel):
    """Exact scope of source and latency differences after replacement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: Annotated[int, Field(ge=1, description="Compared result cells.")]
    unchanged_input_cells: Annotated[
        int,
        Field(
            ge=0,
            description="Nonreplaced cells sharing the same canonical input bytes.",
        ),
    ]
    changed_input_cells: Annotated[
        int, Field(ge=0, description="Cells with changed input hashes.")
    ]
    changed_latency_cells: Annotated[
        int, Field(ge=0, description="Cells whose recomputed latency changed.")
    ]
    changed_input_keys: Annotated[
        tuple[str, ...], Field(description="Stable identities of changed input cells.")
    ]
    changed_latency_keys: Annotated[
        tuple[str, ...],
        Field(description="Stable identities of changed latency cells."),
    ]


class MultilingualLatencyArtifact(BaseModel):
    """Typed, reproducible corrected latency artifact for the paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multilingual-latency-v1"]
    instrument: Literal["tau-multilingual-turn-taking-latency"]
    instrument_version: Literal["1.0.0"]
    formula: Annotated[str, Field(description="Complete aggregation definition.")]
    cohort: Annotated[LatencyCohort, Field(description="Analyzed cohort dimensions.")]
    provenance: Annotated[
        LatencyProvenance, Field(description="Active and historical input hashes.")
    ]
    comparison: Annotated[
        LatencyReplacementComparison,
        Field(description="Scope of the corrected retail replacement."),
    ]
    cells: Annotated[
        tuple[LatencyCell, ...],
        Field(description="All previous and active cell-level sufficient statistics."),
    ]
    previous_descriptive_complete_cohort: Annotated[
        LatencyDescriptiveSummary,
        Field(description="Recomputed pre-replacement latency rollups."),
    ]
    descriptive_complete_cohort: Annotated[
        LatencyDescriptiveSummary,
        Field(description="Recomputed corrected active latency rollups."),
    ]

    @model_validator(mode="after")
    def validate_complete_cohort_and_rollups(self) -> MultilingualLatencyArtifact:
        """Validate the full cross-product, comparison, hashes, and summaries."""
        expected_order = tuple(
            (language, domain, system)
            for language in LANGUAGES
            for domain in DOMAINS
            for system in VOICE_SYSTEMS
        )
        observed_order = tuple(
            (cell.language, cell.domain, cell.system) for cell in self.cells
        )
        if observed_order != expected_order:
            raise ValueError(
                "latency cells must follow the canonical 90-cell voice ordering"
            )
        if self.formula != LATENCY_FORMULA:
            raise ValueError("latency formula does not match the versioned contract")
        if self.cohort != LatencyCohort(
            cells=90,
            calls=4_500,
            calls_per_cell=50,
            languages=tuple(LANGUAGES),
            domains=tuple(DOMAINS),
            systems=tuple(VOICE_SYSTEMS),
            trial=0,
        ):
            raise ValueError(
                "latency cohort dimensions are not the complete voice grid"
            )
        if any(
            cell.previous.calls != 50 or cell.active.calls != 50 for cell in self.cells
        ):
            raise ValueError(
                "every previous and active latency cell must have 50 calls"
            )

        changed_inputs = tuple(cell.key for cell in self.cells if cell.input_changed)
        changed_latencies = tuple(
            cell.key for cell in self.cells if cell.latency_changed
        )
        comparison = LatencyReplacementComparison(
            cells=len(self.cells),
            unchanged_input_cells=len(self.cells) - len(changed_inputs),
            changed_input_cells=len(changed_inputs),
            changed_latency_cells=len(changed_latencies),
            changed_input_keys=changed_inputs,
            changed_latency_keys=changed_latencies,
        )
        if self.comparison != comparison:
            raise ValueError("replacement comparison disagrees with cell evidence")
        if changed_inputs != EXPECTED_REPLACEMENT_KEYS:
            raise ValueError(
                "changed latency inputs must be exactly the ten approved KO/ZH "
                "retail replacements"
            )
        if not set(changed_latencies).issubset(EXPECTED_REPLACEMENT_KEYS):
            raise ValueError("latency changes escape the approved replacement scope")

        previous_summary = _summarize(self.cells, snapshot="previous")
        active_summary = _summarize(self.cells, snapshot="active")
        if self.previous_descriptive_complete_cohort != previous_summary:
            raise ValueError("previous latency rollups disagree with cell counts")
        if self.descriptive_complete_cohort != active_summary:
            raise ValueError("active latency rollups disagree with cell counts")

        previous_results = _results_fingerprint(self.cells, snapshot="previous")
        active_results = _results_fingerprint(self.cells, snapshot="active")
        previous_simulations = _simulations_fingerprint(self.cells, snapshot="previous")
        active_simulations = _simulations_fingerprint(self.cells, snapshot="active")
        if (
            self.provenance.previous_results_fingerprint_sha256 != previous_results
            or self.provenance.active_results_fingerprint_sha256 != active_results
            or self.provenance.previous_simulations_fingerprint_sha256
            != previous_simulations
            or self.provenance.active_simulations_fingerprint_sha256
            != active_simulations
        ):
            raise ValueError("provenance fingerprints disagree with cell evidence")
        return self


class _PreviousSource(BaseModel):
    """Source identity read from the frozen pre-replacement experience artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    path: Annotated[str, Field(description="Canonical repository-relative path.")]
    sha256: Sha256
    trial_0_calls: Annotated[int, Field(ge=1)]


class _PreviousProvenance(BaseModel):
    """Relevant pre-replacement provenance fields."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cohort_fingerprint_sha256: Sha256
    results_files: Annotated[tuple[_PreviousSource, ...], Field(min_length=1)]


class _PreviousLatencyValue(BaseModel):
    """Relevant latency value from a prior paper rollup."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    latency_seconds: Annotated[float, Field(ge=0)]


class _PreviousDescriptiveSummary(BaseModel):
    """Relevant rollups from the prior experience artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    language_system: dict[str, dict[str, _PreviousLatencyValue]]
    provider: dict[str, _PreviousLatencyValue]
    overall: _PreviousLatencyValue


class _PreviousExperienceArtifact(BaseModel):
    """Typed latency-facing view of the pre-replacement experience artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    provenance: _PreviousProvenance
    descriptive_complete_cohort: _PreviousDescriptiveSummary


def _sha256_file(path: Path) -> str:
    """Hash one file without loading large simulations into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_simulation_prefix(path: Path) -> _SimulationPrefix:
    """Validate top-level simulation metadata without loading its tick stream."""
    marker = b'\n  "ticks": '
    payload = bytearray()
    with path.open("rb") as handle:
        while marker not in payload:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"No ticks field found in {path}")
            payload.extend(chunk)
    prefix = bytes(payload).split(marker, 1)[0].rstrip()
    if prefix.endswith(b","):
        prefix = prefix[:-1]
    return _SimulationPrefix.model_validate_json(prefix + b"\n}")


def _latency_counts(simulation: _SimulationPrefix) -> LatencyCounts:
    """Extract response/yield sufficient statistics from one typed simulation."""
    checks = (
        {check.id: check for check in simulation.quality_info.factor_checks}
        if (simulation.quality_info is not None)
        else {}
    )
    if simulation.quality_info is not None and len(checks) != len(
        simulation.quality_info.factor_checks
    ):
        raise ValueError(f"duplicate quality factor in simulation {simulation.id}")

    response = checks.get("responsiveness")
    yielding = checks.get("yielding")
    response_total = response.metrics.response_total if response else None
    response_latency = response.metrics.response_latency_mean if response else None
    yield_total = yielding.metrics.yield_total if yielding else None
    yield_latency = yielding.metrics.yield_latency_mean if yielding else None
    response_total, response_latency = _validated_metric_pair(
        response_total,
        response_latency,
        rate=response.metrics.response_rate if response else None,
        simulation_id=simulation.id,
        channel="response",
    )
    yield_total, yield_latency = _validated_metric_pair(
        yield_total,
        yield_latency,
        rate=yielding.metrics.yield_rate if yielding else None,
        simulation_id=simulation.id,
        channel="yield",
    )
    return LatencyCounts(
        response_seconds_sum=(
            float(response_total) * float(response_latency)
            if response_total is not None and response_latency is not None
            else 0.0
        ),
        response_events=(
            float(response_total)
            if response_total is not None and response_latency is not None
            else 0.0
        ),
        yield_seconds_sum=(
            float(yield_total) * float(yield_latency)
            if yield_total is not None and yield_latency is not None
            else 0.0
        ),
        yield_events=(
            float(yield_total)
            if yield_total is not None and yield_latency is not None
            else 0.0
        ),
    )


def _validated_metric_pair(
    total: float | None,
    latency: float | None,
    *,
    rate: float | None,
    simulation_id: str,
    channel: str,
) -> tuple[float | None, float | None]:
    """Require complete pairs, except zero-rate channels with no latency events."""
    if latency is None and (total == 0 or (total is not None and rate == 0)):
        return None, None
    if (total is None) != (latency is None):
        raise ValueError(
            f"partial {channel} latency metrics in simulation {simulation_id}"
        )
    if total is not None and not float(total).is_integer():
        raise ValueError(
            f"non-integral {channel} event count in simulation {simulation_id}"
        )
    return total, latency


def _snapshot(
    results_path: Path,
    *,
    logical_path: str,
    storage_path: str,
    expected_results_sha256: str,
) -> LatencyInputSnapshot:
    """Recompute one cell from its exact trial-zero simulation files."""
    observed_results_sha256 = _sha256_file(results_path)
    if observed_results_sha256 != expected_results_sha256:
        raise ValueError(
            f"results hash mismatch for {results_path}: "
            f"{observed_results_sha256} != {expected_results_sha256}"
        )
    results = _ResultsIndex.model_validate_json(results_path.read_text())
    rows = tuple(row for row in results.simulation_index if row.trial == 0)
    if len(rows) != 50:
        raise ValueError(
            f"expected 50 trial-zero rows in {results_path}, got {len(rows)}"
        )
    if len({row.id for row in rows}) != len(rows):
        raise ValueError(f"duplicate trial-zero simulation ids in {results_path}")

    counts = LatencyCounts()
    simulation_identities = []
    for row in rows:
        simulation_path = results_path.parent / "simulations" / f"{row.id}.json"
        simulation_sha256 = _sha256_file(simulation_path)
        simulation = _read_simulation_prefix(simulation_path)
        if simulation.id != row.id or simulation.task_id != row.task_id:
            raise ValueError(f"simulation identity drift for {simulation_path}")
        counts = counts.plus(_latency_counts(simulation))
        simulation_identities.append(f"{row.id}\t{row.task_id}\t{simulation_sha256}")
    simulations_sha256 = hashlib.sha256(
        "\n".join(simulation_identities).encode()
    ).hexdigest()
    return LatencyInputSnapshot(
        logical_path=logical_path,
        storage_path=storage_path,
        results_sha256=observed_results_sha256,
        simulations_sha256=simulations_sha256,
        calls=len(rows),
        counts=counts,
        latency_seconds=counts.latency_seconds,
    )


def _combine_counts(snapshots: list[LatencyInputSnapshot]) -> LatencyCounts:
    """Combine cell sufficient statistics in their serialized order."""
    counts = LatencyCounts()
    for snapshot in snapshots:
        counts = counts.plus(snapshot.counts)
    return counts


def _summarize(
    cells: tuple[LatencyCell, ...],
    *,
    snapshot: Literal["previous", "active"],
) -> LatencyDescriptiveSummary:
    """Build the exact paper rollups from cell sufficient statistics."""
    by_language_system: dict[tuple[str, str], list[LatencyInputSnapshot]] = defaultdict(
        list
    )
    for cell in cells:
        by_language_system[(cell.language, cell.system)].append(getattr(cell, snapshot))

    language_system: dict[str, dict[str, LatencyEventRollup]] = {}
    for language in LANGUAGES:
        language_name = LANGUAGES[language]
        language_system[language_name] = {}
        for system in VOICE_SYSTEMS:
            snapshots = by_language_system[(language, system)]
            counts = _combine_counts(snapshots)
            language_system[language_name][SYSTEM_LABELS[system]] = LatencyEventRollup(
                n_calls=sum(item.calls for item in snapshots),
                latency_seconds=counts.latency_seconds,
                counts=counts,
            )

    language: dict[str, LatencyMacroRollup] = {}
    for language_name in LANGUAGES.values():
        rollups = list(language_system[language_name].values())
        language[language_name] = LatencyMacroRollup(
            n_calls=sum(item.n_calls for item in rollups),
            latency_seconds=(
                sum(item.latency_seconds for item in rollups) / len(rollups)
            ),
        )

    provider: dict[str, LatencyMacroRollup] = {}
    for system in VOICE_SYSTEMS:
        system_label = SYSTEM_LABELS[system]
        rollups = [
            language_system[LANGUAGES[language]][system_label] for language in LANGUAGES
        ]
        provider[system_label] = LatencyMacroRollup(
            n_calls=sum(item.n_calls for item in rollups),
            latency_seconds=(
                sum(item.latency_seconds for item in rollups) / len(rollups)
            ),
        )

    provider_rollups = list(provider.values())
    return LatencyDescriptiveSummary(
        language_system=language_system,
        language=language,
        provider=provider,
        overall=LatencyMacroRollup(
            n_calls=sum(item.n_calls for item in provider_rollups),
            latency_seconds=(
                sum(item.latency_seconds for item in provider_rollups)
                / len(provider_rollups)
            ),
        ),
    )


def _results_fingerprint(
    cells: tuple[LatencyCell, ...],
    *,
    snapshot: Literal["previous", "active"],
) -> str:
    """Match the experience artifact's ordered result-source fingerprint."""
    payload = "\n".join(
        f"{item.logical_path}\t{item.results_sha256}\t{item.calls}"
        for item in (getattr(cell, snapshot) for cell in cells)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _simulations_fingerprint(
    cells: tuple[LatencyCell, ...],
    *,
    snapshot: Literal["previous", "active"],
) -> str:
    """Bind every result cell to its ordered trial-zero simulation bytes."""
    payload = "\n".join(
        f"{item.logical_path}\t{item.simulations_sha256}\t{item.calls}"
        for item in (getattr(cell, snapshot) for cell in cells)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _assert_matches_previous_artifact(
    computed: LatencyDescriptiveSummary,
    previous: _PreviousDescriptiveSummary,
) -> None:
    """Prove that the historical inputs reproduce every frozen old rollup."""
    comparisons = []
    for language_name in LANGUAGES.values():
        for system_label in SYSTEM_LABELS.values():
            comparisons.append(
                (
                    f"{language_name}/{system_label}",
                    computed.language_system[language_name][
                        system_label
                    ].latency_seconds,
                    previous.language_system[language_name][
                        system_label
                    ].latency_seconds,
                )
            )
    for system_label in SYSTEM_LABELS.values():
        comparisons.append(
            (
                system_label,
                computed.provider[system_label].latency_seconds,
                previous.provider[system_label].latency_seconds,
            )
        )
    comparisons.append(
        (
            "overall",
            computed.overall.latency_seconds,
            previous.overall.latency_seconds,
        )
    )
    for label, observed, expected in comparisons:
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"historical latency mismatch for {label}: {observed} != {expected}"
            )


def build_multilingual_latency_artifact(
    evidence_root: Path,
    audit_path: Path,
    previous_experience_path: Path,
) -> MultilingualLatencyArtifact:
    """Recompute corrected latency and compare it with the frozen prior cohort."""
    evidence_root = evidence_root.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    previous_experience_path = previous_experience_path.expanduser().resolve()
    audit = MultilingualAudit.model_validate_json(audit_path.read_text())
    previous = _PreviousExperienceArtifact.model_validate_json(
        previous_experience_path.read_text()
    )

    audit_cells = {
        (cell.language, cell.domain, cell.system): cell for cell in audit.voice_cells
    }
    expected_keys = {
        (language, domain, system)
        for language in LANGUAGES
        for domain in DOMAINS
        for system in VOICE_SYSTEMS
    }
    if set(audit_cells) != expected_keys or len(audit.voice_cells) != len(
        expected_keys
    ):
        raise ValueError("audit voice cells do not equal the complete 90-cell cohort")

    previous_sources = {
        source.path: source for source in previous.provenance.results_files
    }
    if len(previous_sources) != 90:
        raise ValueError("previous experience artifact must bind exactly 90 sources")

    cells = []
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in VOICE_SYSTEMS:
                audit_cell = audit_cells[(language, domain, system)]
                logical_path = (RESULT_PATH_PREFIX / audit_cell.results_path).as_posix()
                previous_source = previous_sources.get(logical_path)
                if previous_source is None:
                    raise ValueError(f"previous artifact is missing {logical_path}")

                active_path = evidence_root / audit_cell.results_path
                active = _snapshot(
                    active_path,
                    logical_path=logical_path,
                    storage_path=logical_path,
                    expected_results_sha256=audit_cell.sha256,
                )
                if previous_source.sha256 == active.results_sha256:
                    previous_snapshot = active
                else:
                    archived_relative = (
                        PurePosixPath("validation_runs")
                        / language
                        / audit_cell.results_path
                    )
                    archived_storage_path = (
                        RESULT_PATH_PREFIX / archived_relative
                    ).as_posix()
                    previous_snapshot = _snapshot(
                        evidence_root / archived_relative,
                        logical_path=logical_path,
                        storage_path=archived_storage_path,
                        expected_results_sha256=previous_source.sha256,
                    )
                if previous_snapshot.calls != previous_source.trial_0_calls:
                    raise ValueError(f"previous call count drift for {logical_path}")

                input_changed = (
                    previous_snapshot.results_sha256 != active.results_sha256
                    or previous_snapshot.simulations_sha256 != active.simulations_sha256
                )
                cells.append(
                    LatencyCell(
                        language=language,
                        language_name=LANGUAGES[language],
                        domain=domain,
                        system=system,
                        system_label=SYSTEM_LABELS[system],
                        previous=previous_snapshot,
                        active=active,
                        input_changed=input_changed,
                        latency_changed=not math.isclose(
                            previous_snapshot.latency_seconds,
                            active.latency_seconds,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ),
                    )
                )
    cell_tuple = tuple(cells)
    previous_summary = _summarize(cell_tuple, snapshot="previous")
    _assert_matches_previous_artifact(
        previous_summary, previous.descriptive_complete_cohort
    )
    active_summary = _summarize(cell_tuple, snapshot="active")
    changed_inputs = tuple(cell.key for cell in cell_tuple if cell.input_changed)
    changed_latencies = tuple(cell.key for cell in cell_tuple if cell.latency_changed)
    artifact = MultilingualLatencyArtifact(
        schema_version="tau-multilingual-latency-v1",
        instrument="tau-multilingual-turn-taking-latency",
        instrument_version="1.0.0",
        formula=LATENCY_FORMULA,
        cohort=LatencyCohort(
            cells=90,
            calls=4_500,
            calls_per_cell=50,
            languages=tuple(LANGUAGES),
            domains=tuple(DOMAINS),
            systems=tuple(VOICE_SYSTEMS),
            trial=0,
        ),
        provenance=LatencyProvenance(
            audit_artifact_id=audit.artifact_id,
            audit_sha256=_sha256_file(audit_path),
            previous_experience_sha256=_sha256_file(previous_experience_path),
            previous_results_fingerprint_sha256=_results_fingerprint(
                cell_tuple, snapshot="previous"
            ),
            active_results_fingerprint_sha256=_results_fingerprint(
                cell_tuple, snapshot="active"
            ),
            previous_simulations_fingerprint_sha256=_simulations_fingerprint(
                cell_tuple, snapshot="previous"
            ),
            active_simulations_fingerprint_sha256=_simulations_fingerprint(
                cell_tuple, snapshot="active"
            ),
        ),
        comparison=LatencyReplacementComparison(
            cells=90,
            unchanged_input_cells=90 - len(changed_inputs),
            changed_input_cells=len(changed_inputs),
            changed_latency_cells=len(changed_latencies),
            changed_input_keys=changed_inputs,
            changed_latency_keys=changed_latencies,
        ),
        cells=cell_tuple,
        previous_descriptive_complete_cohort=previous_summary,
        descriptive_complete_cohort=active_summary,
    )
    if artifact.provenance.previous_results_fingerprint_sha256 != (
        previous.provenance.cohort_fingerprint_sha256
    ):
        raise ValueError(
            "reconstructed prior cohort fingerprint does not match artifact"
        )
    return artifact


def write_multilingual_latency_artifact(
    evidence_root: Path,
    audit_path: Path,
    previous_experience_path: Path,
    output_path: Path,
) -> MultilingualLatencyArtifact:
    """Recompute and canonically serialize the corrected latency artifact."""
    artifact = build_multilingual_latency_artifact(
        evidence_root, audit_path, previous_experience_path
    )
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2) + "\n")
    return artifact


def load_multilingual_latency_artifact(path: Path) -> MultilingualLatencyArtifact:
    """Load and fully validate a serialized latency artifact."""
    return MultilingualLatencyArtifact.model_validate_json(path.read_text())
