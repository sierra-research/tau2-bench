# Copyright Sierra
"""Bind a generated final-window CSV to the active canonical paper headers.

This is deliberately a manifest-only operation.  It verifies the strict active
audit, all 90 ``results.json`` headers, and every exclusion row against the
stored delivery finding, then copies the CSV byte-for-byte beside a newly bound
manifest.  It reads neither audio nor tick timelines and makes no external
calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.judges.corrected_paper_suite import (
    FinalWindowExclusionManifest,
    FinalWindowExclusionRow,
    GeneratedFinalWindowExclusionManifest,
    canonical_cohort_fingerprint_sha256,
    canonical_results_files_sha256,
    sha256_file,
)
from tau2.judges.delivery.postprocess import (
    is_in_final_utterance_window,
    time_range_bounds_seconds,
)
from tau2.paper.multilingual import (
    DOMAINS,
    LANGUAGES,
    VOICE_SYSTEMS,
    MultilingualAudit,
)

ACTIVE_EVIDENCE_PREFIX = Path("data/simulations/paper_runs/tau-multi")
DEFAULT_ACTIVE_AUDIT = Path("papers/tau-multilingual/reproduction/audit.json")
CORRECTED_FINAL_WINDOW_CSV_SHA256 = (
    "2813e0dc25a80fd3a60775dd1533326523fe0b569220f1010bdf78c91dd52841"
)
CORRECTED_GENERATION_RESULTS_FILES_SHA256 = (
    "e5fca92e1affdebf15100e2b7fae2fb7f1b5a0a0210058299a755f10a9998e75"
)
CORRECTED_REPLACEMENT_COHORT_SHA256 = (
    "bdf7bdc623ae2107c77704dfa2f974b857152b71e08db0e75eb6e09ba1e49974"
)
CORRECTED_CANONICAL_SIDECAR_SHA256 = (
    "426c6742b42a4a6267f683d90404b57c87616297d10546f9e95cfdd2c52b3e7a"
)
SYSTEM_LABELS = {
    "openai_minimal": "OpenAI minimal",
    "openai_xhigh": "OpenAI xhigh",
    "gemini_minimal": "Gemini minimal",
    "gemini_high": "Gemini high",
    "xai_provider_default": "xAI",
}


class FinalWindowRebindContract(BaseModel):
    """Closed cohort and immutable CSV identity for one canonical rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    languages: tuple[str, ...]
    domains: tuple[str, ...]
    system_slugs: tuple[str, ...]
    system_labels: dict[str, str]
    calls_per_cell: Annotated[int, Field(ge=1)]
    expected_csv_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_generation_results_files_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    expected_replacement_cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    expected_replacement_calls: Annotated[int, Field(ge=1)]
    expected_canonical_sidecar_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @property
    def results_files(self) -> int:
        """Return the exact number of result headers in the contract."""
        return len(self.languages) * len(self.domains) * len(self.system_slugs)

    @property
    def trial0_calls(self) -> int:
        """Return the exact number of canonical trial-zero calls."""
        return self.results_files * self.calls_per_cell


CANONICAL_REBIND_CONTRACT = FinalWindowRebindContract(
    languages=tuple(LANGUAGES),
    domains=DOMAINS,
    system_slugs=VOICE_SYSTEMS,
    system_labels=SYSTEM_LABELS,
    calls_per_cell=50,
    expected_csv_sha256=CORRECTED_FINAL_WINDOW_CSV_SHA256,
    expected_generation_results_files_sha256=(
        CORRECTED_GENERATION_RESULTS_FILES_SHA256
    ),
    expected_replacement_cohort_fingerprint_sha256=(
        CORRECTED_REPLACEMENT_COHORT_SHA256
    ),
    expected_replacement_calls=500,
    expected_canonical_sidecar_sha256=CORRECTED_CANONICAL_SIDECAR_SHA256,
)


class FinalWindowRebindConfig(BaseModel):
    """Filesystem inputs for the offline, manifest-only rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Path
    sidecar_from: Path
    output_root: Path
    active_audit: Path = DEFAULT_ACTIVE_AUDIT


class _LegacyGeneratedFinalWindowManifest(BaseModel):
    """Exact frozen schema emitted before the canonical manifest rebind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    filter_version: Literal["v1"]
    window_seconds: Literal[1.0]
    cohort: str
    excluded_findings: int
    excluded_severity_2plus_findings: int
    by_language: dict[str, int]
    by_system: dict[str, int]
    results_files_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    replacement_cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    replacement_calls: Annotated[int, Field(ge=1)]
    canonical_sidecar_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _IndexHeader(BaseModel):
    """Fields needed from one ``results.json`` index row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    trial: int


class _ResultsHeader(BaseModel):
    """The only ``results.json`` payload required by this rebind."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    simulation_index: list[_IndexHeader]


class _FindingHeader(BaseModel):
    """Stored delivery-finding fields named by a sidecar row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    axis: str
    severity: int
    time_range: str | None


class _UtteranceHeader(BaseModel):
    """Stored utterance identity and ordered findings."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    utterance_idx: int
    findings: list[_FindingHeader]


class _DeliveryHeader(BaseModel):
    """Stored delivery inventory required for metadata verification."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    utterance_results: list[_UtteranceHeader]


class _SimulationHeader(BaseModel):
    """Prefix of a simulation file before its large tick timeline."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    delivery_info: _DeliveryHeader | None


class _CanonicalCall(BaseModel):
    """Canonical cell dimensions and path for one trial-zero call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    language: str
    domain: str
    system: str


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (repo_root / expanded).resolve()
    )


def _read_simulation_prefix(path: Path) -> _SimulationHeader:
    """Read top-level judged fields without loading the large tick timeline."""
    marker = b'\n  "ticks": '
    payload = bytearray()
    with path.open("rb") as handle:
        while marker not in payload:
            chunk = handle.read(64 * 1024)
            if not chunk:
                raise ValueError(f"simulation has no top-level ticks field: {path}")
            payload.extend(chunk)
    prefix = bytes(payload).split(marker, 1)[0].rstrip()
    if prefix.endswith(b","):
        prefix = prefix[:-1]
    return _SimulationHeader.model_validate_json(prefix + b"\n}")


def _load_generated_sidecar(
    path: Path,
    contract: FinalWindowRebindContract,
) -> tuple[list[FinalWindowExclusionRow], GeneratedFinalWindowExclusionManifest]:
    if not path.is_file():
        raise ValueError(f"final-window source CSV is missing: {path}")
    observed_csv_sha256 = sha256_file(path)
    if observed_csv_sha256 != contract.expected_csv_sha256:
        raise ValueError(
            "generated final-window CSV identity drifted: "
            f"{observed_csv_sha256} != {contract.expected_csv_sha256}"
        )
    with path.open(newline="") as handle:
        rows = [
            FinalWindowExclusionRow.model_validate(row)
            for row in csv.DictReader(handle)
        ]
    keys = [(row.sim_id, row.utterance_idx, row.finding_index) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("generated final-window CSV has duplicate exclusion rows")

    manifest_path = path.with_suffix(".json")
    legacy_manifest = _LegacyGeneratedFinalWindowManifest.model_validate_json(
        manifest_path.read_text()
    )
    manifest = GeneratedFinalWindowExclusionManifest(
        artifact=legacy_manifest.artifact,
        filter_version=legacy_manifest.filter_version,
        window_seconds=legacy_manifest.window_seconds,
        cohort=legacy_manifest.cohort,
        excluded_findings=legacy_manifest.excluded_findings,
        excluded_severity_2plus_findings=(
            legacy_manifest.excluded_severity_2plus_findings
        ),
        by_language=legacy_manifest.by_language,
        by_system=legacy_manifest.by_system,
        generation_results_files_sha256=legacy_manifest.results_files_sha256,
        replacement_cohort_fingerprint_sha256=(
            legacy_manifest.replacement_cohort_fingerprint_sha256
        ),
        replacement_calls=legacy_manifest.replacement_calls,
        canonical_sidecar_sha256=legacy_manifest.canonical_sidecar_sha256,
    )
    if manifest.artifact != path.name:
        raise ValueError("generated final-window manifest names another CSV")
    if manifest.excluded_findings != len(rows):
        raise ValueError("generated final-window manifest row count drifted")
    severity_rows = sum(row.severity >= 2 for row in rows)
    if manifest.excluded_severity_2plus_findings != severity_rows:
        raise ValueError("generated final-window severity count drifted")
    if manifest.by_language != dict(
        sorted(Counter(row.language for row in rows).items())
    ):
        raise ValueError("generated final-window language counts drifted")
    if manifest.by_system != dict(sorted(Counter(row.system for row in rows).items())):
        raise ValueError("generated final-window system counts drifted")
    observed_generation = (
        manifest.generation_results_files_sha256,
        manifest.replacement_cohort_fingerprint_sha256,
        manifest.replacement_calls,
        manifest.canonical_sidecar_sha256,
    )
    expected_generation = (
        contract.expected_generation_results_files_sha256,
        contract.expected_replacement_cohort_fingerprint_sha256,
        contract.expected_replacement_calls,
        contract.expected_canonical_sidecar_sha256,
    )
    if observed_generation != expected_generation:
        raise ValueError("generated final-window provenance drifted")
    return rows, manifest


def _load_active_headers(
    config: FinalWindowRebindConfig,
    contract: FinalWindowRebindContract,
) -> tuple[list[dict[str, object]], dict[str, _CanonicalCall]]:
    repo_root = config.repo_root.expanduser().resolve()
    audit_path = _resolve_repo_path(repo_root, config.active_audit)
    audit = MultilingualAudit.model_validate_json(audit_path.read_text())
    if not audit.ok:
        raise ValueError("active audit contains a failing finding")
    audit_content = audit.model_dump(
        mode="json",
        exclude={"artifact_id", "repository_commit", "evidence_root"},
    )
    audit_artifact_id = hashlib.sha256(
        json.dumps(audit_content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if audit.artifact_id != audit_artifact_id:
        raise ValueError("active audit artifact identity does not reproduce")

    cells = {
        (cell.language, cell.domain, cell.system): cell for cell in audit.voice_cells
    }
    expected = {
        (language, domain, system)
        for language in contract.languages
        for domain in contract.domains
        for system in contract.system_slugs
    }
    if len(cells) != len(audit.voice_cells) or set(cells) != expected:
        raise ValueError("active audit does not contain the exact canonical voice grid")
    if set(contract.system_labels) != set(contract.system_slugs):
        raise ValueError("final-window system labels do not cover the voice grid")

    evidence_root = (repo_root / ACTIVE_EVIDENCE_PREFIX).resolve()
    sources: list[dict[str, object]] = []
    calls: dict[str, _CanonicalCall] = {}
    for language in contract.languages:
        for domain in contract.domains:
            for system in contract.system_slugs:
                cell = cells[(language, domain, system)]
                results_path = (evidence_root / cell.results_path).resolve()
                if not results_path.is_relative_to(evidence_root):
                    raise ValueError(
                        f"audit results path escapes evidence root: {cell.results_path}"
                    )
                observed_sha256 = sha256_file(results_path)
                if observed_sha256 != cell.sha256:
                    raise ValueError(
                        f"active results header hash drifted: {results_path}"
                    )
                results = _ResultsHeader.model_validate_json(results_path.read_text())
                if len(results.simulation_index) != cell.rows:
                    raise ValueError(
                        f"active results header row count drifted: {results_path}"
                    )
                trial0 = [row for row in results.simulation_index if row.trial == 0]
                if len(trial0) != contract.calls_per_cell:
                    raise ValueError(
                        f"active results header trial-zero count drifted: {results_path}"
                    )
                source_path = results_path.relative_to(repo_root).as_posix()
                sources.append(
                    {
                        "path": source_path,
                        "sha256": observed_sha256,
                        "trial_0_calls": len(trial0),
                    }
                )
                for entry in trial0:
                    simulation_path = (
                        results_path.parent / "simulations" / f"{entry.id}.json"
                    )
                    if entry.id in calls:
                        raise ValueError(
                            f"duplicate canonical trial-zero call id: {entry.id}"
                        )
                    calls[entry.id] = _CanonicalCall(
                        path=simulation_path,
                        language=language,
                        domain=domain,
                        system=contract.system_labels[system],
                    )

    if len(sources) != contract.results_files:
        raise ValueError("canonical results-header count drifted")
    if len(calls) != contract.trial0_calls:
        raise ValueError("canonical trial-zero call count drifted")
    return sources, calls


def _verify_rows_against_cohort(
    rows: list[FinalWindowExclusionRow],
    calls: dict[str, _CanonicalCall],
) -> None:
    by_sim: dict[str, list[FinalWindowExclusionRow]] = defaultdict(list)
    for row in rows:
        if row.sim_id not in calls:
            raise ValueError(
                f"final-window row names a missing canonical call: {row.sim_id}"
            )
        by_sim[row.sim_id].append(row)

    for sim_id, call_rows in by_sim.items():
        canonical = calls[sim_id]
        simulation = _read_simulation_prefix(canonical.path)
        if simulation.id != sim_id:
            raise ValueError(f"simulation/header identity drifted: {canonical.path}")
        if simulation.delivery_info is None:
            raise ValueError(f"final-window call has no delivery judgment: {sim_id}")
        utterances = {
            utterance.utterance_idx: utterance
            for utterance in simulation.delivery_info.utterance_results
        }
        if len(utterances) != len(simulation.delivery_info.utterance_results):
            raise ValueError(
                f"delivery judgment has duplicate utterance indices: {sim_id}"
            )
        for row in call_rows:
            if (
                row.language != canonical.language
                or row.domain != canonical.domain
                or row.system != canonical.system
            ):
                raise ValueError(f"final-window row dimensions drifted: {sim_id}")
            utterance = utterances.get(row.utterance_idx)
            if utterance is None or not 0 <= row.finding_index < len(
                utterance.findings
            ):
                raise ValueError(f"final-window row names a missing finding: {sim_id}")
            finding = utterance.findings[row.finding_index]
            if (
                finding.axis != "fidelity"
                or finding.severity != row.severity
                or finding.time_range != row.time_range
            ):
                raise ValueError(f"final-window row finding metadata drifted: {sim_id}")
            bounds = time_range_bounds_seconds(row.time_range)
            if bounds is None or not (
                math.isclose(bounds[0], row.span_start_seconds, abs_tol=1e-9)
                and math.isclose(bounds[1], row.span_end_seconds, abs_tol=1e-9)
            ):
                raise ValueError(f"final-window row span metadata drifted: {sim_id}")
            if row.clip_duration_seconds <= 0 or not is_in_final_utterance_window(
                row.time_range, row.clip_duration_seconds
            ):
                raise ValueError(
                    f"final-window row no longer satisfies its rule: {sim_id}"
                )


def rebind_final_window_manifest(
    config: FinalWindowRebindConfig,
    *,
    contract: FinalWindowRebindContract = CANONICAL_REBIND_CONTRACT,
) -> FinalWindowExclusionManifest:
    """Verify and atomically emit a canonical-bound manifest plus unchanged CSV."""
    source = config.sidecar_from.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"final-window rebind output already exists: {output_root}")
    rows, generated = _load_generated_sidecar(source, contract)
    sources, calls = _load_active_headers(config, contract)
    _verify_rows_against_cohort(rows, calls)

    canonical_results_hash = canonical_results_files_sha256(sources)
    canonical_cohort_hash = canonical_cohort_fingerprint_sha256(sources)
    rebound = FinalWindowExclusionManifest(
        **generated.model_dump(mode="json"),
        final_window_csv_sha256=contract.expected_csv_sha256,
        canonical_results_files_sha256=canonical_results_hash,
        canonical_cohort_fingerprint_sha256=canonical_cohort_hash,
        canonical_results_files=len(sources),
        canonical_trial0_calls=sum(int(row["trial_0_calls"]) for row in sources),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        csv_output = temporary / source.name
        shutil.copyfile(source, csv_output)
        if sha256_file(csv_output) != contract.expected_csv_sha256:
            raise ValueError("copied final-window CSV does not match the pinned hash")
        manifest_output = csv_output.with_suffix(".json")
        manifest_output.write_text(rebound.model_dump_json(indent=2) + "\n")
        if (
            FinalWindowExclusionManifest.model_validate_json(
                manifest_output.read_text()
            )
            != rebound
        ):
            raise ValueError("written final-window manifest did not round-trip")
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return rebound


__all__ = [
    "CANONICAL_REBIND_CONTRACT",
    "CORRECTED_CANONICAL_SIDECAR_SHA256",
    "CORRECTED_FINAL_WINDOW_CSV_SHA256",
    "CORRECTED_GENERATION_RESULTS_FILES_SHA256",
    "CORRECTED_REPLACEMENT_COHORT_SHA256",
    "FinalWindowRebindConfig",
    "FinalWindowRebindContract",
    "rebind_final_window_manifest",
]
