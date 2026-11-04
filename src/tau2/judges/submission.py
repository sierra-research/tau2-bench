"""Judge diagnostics carried beside a benchmark submission, never inside rewards.

A Voice V2 submitter runs ``tau2 judges suite`` on the saved matrix. Preparation
reads the saved verdicts (no model calls), requires every applicable measure in
every cell to be judged, and records per-measure summaries plus content
fingerprints of each verdict. Validation recomputes the same export from the
original saved matrix, so maintainers can re-judge any sample and compare it to
the fingerprinted identity.

The completeness requirement belongs to preparation, not to the export format:
the frozen 2026-10-02 release has a few partial verdicts (unit errors), and its
export reports them as coverage gaps instead of hiding them.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from tau2.data_model.benchmark import BenchmarkFrame
from tau2.data_model.judges import (
    ContractModel,
    DiagnosticId,
    ImplementationFamily,
    InputExclusions,
    JudgeCatalog,
    PlanAction,
    StoredVerdict,
    SuitePlan,
    is_fidelity_output_limit,
)
from tau2.judges.summary import DiagnosticSummary, summarize_suite

# Benchmarks whose official package must carry complete judge diagnostics.
JUDGED_BENCHMARKS = frozenset({"voice-v2"})
DIAGNOSTICS_FILE_NAME = "benchmark-diagnostics.json"
# At most 1/OUTPUT_LIMIT_DIVISOR (0.1%) of a package's speech-fidelity clips may
# be left out because the judge reply hit its output limit.
OUTPUT_LIMIT_DIVISOR = 1000


def _sha256(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class JudgeIdentity(ContractModel):
    """The frozen judge contract a measure was evaluated with."""

    contract_id: Annotated[str, Field(description="Frozen judge contract.")]
    implementation: Annotated[
        ImplementationFamily, Field(description="Calibrated implementation family.")
    ]
    input_exclusions: Annotated[
        InputExclusions, Field(description="Benchmark speech removed before judging.")
    ]
    model: Annotated[str, Field(description="Exact judge model.")]
    model_args: Annotated[
        dict[str, JsonValue], Field(description="Exact judge model arguments.")
    ]
    prompt_version: Annotated[str, Field(description="Calibrated prompt version.")]
    rubric_version: Annotated[str, Field(description="Calibrated rubric version.")]


class VerdictFingerprint(ContractModel):
    """One judged call/measure: identity, inputs, unit counts and content hash."""

    simulation_id: Annotated[str, Field(description="Saved simulation identifier.")]
    trial: Annotated[int, Field(ge=0, description="Saved trial index.")]
    measure: Annotated[DiagnosticId, Field(description="Judged measure.")]
    contract_id: Annotated[str, Field(description="Judge contract used.")]
    input_fingerprint: Annotated[
        str, Field(description="SHA-256 of the judged conversation/task evidence.")
    ]
    audio_fingerprint: Annotated[
        str | None, Field(description="SHA-256 of the judged audio, when used.")
    ]
    verdict_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="SHA-256 of the complete stored verdict record.",
        ),
    ]
    pass_units: Annotated[int, Field(ge=0, description="Units without violation.")]
    fail_units: Annotated[int, Field(ge=0, description="Units with a violation.")]
    no_opportunity_units: Annotated[
        int, Field(ge=0, description="Units without a judged opportunity.")
    ]
    error_units: Annotated[
        int, Field(ge=0, description="Units the judge failed on (partial verdicts).")
    ] = 0
    input_error_units: Annotated[
        int,
        Field(
            ge=0,
            description="Units (within error_units) blocked by missing Live input.",
        ),
    ] = 0
    output_limit_units: Annotated[
        int,
        Field(
            ge=0,
            exclude_if=lambda count: count == 0,
            description="Speech-fidelity clips (within error_units) whose judge reply hit its output limit.",
        ),
    ] = 0


class CellDiagnostics(ContractModel):
    """Complete judge coverage for one released language/domain cell."""

    language: Annotated[str, Field(description="Language identifier.")]
    domain: Annotated[str, Field(description="Domain identifier.")]
    summary: Annotated[
        list[DiagnosticSummary],
        Field(description="Per-measure coverage and unit pass fractions."),
    ]
    verdicts: Annotated[
        list[VerdictFingerprint],
        Field(description="Fingerprint of every judged call/measure."),
    ]

    @model_validator(mode="after")
    def _summary_matches_fingerprints(self) -> "CellDiagnostics":
        """Every judged call is fingerprinted and its units add up to the summary."""
        for summary in self.summary:
            rows = [row for row in self.verdicts if row.measure == summary.measure]
            if (
                # Only missing Live input or a fidelity output limit may leave
                # a judged call's units unjudged.
                sum(
                    row.error_units == row.input_error_units + row.output_limit_units
                    for row in rows
                )
                != summary.judged_calls
                or sum(
                    bool(row.error_units)
                    and bool(row.pass_units + row.fail_units + row.no_opportunity_units)
                    for row in rows
                )
                != summary.partial_calls
                or sum(row.output_limit_units for row in rows)
                != summary.output_limit_units
                or sum(row.pass_units for row in rows) != summary.pass_units
                or sum(row.fail_units for row in rows) != summary.fail_units
                or sum(row.no_opportunity_units for row in rows)
                != summary.no_opportunity_units
            ):
                raise ValueError(
                    f"{self.language}-{self.domain} {summary.measure}: summary "
                    "does not match verdict fingerprints"
                )
        if {row.measure for row in self.verdicts} - {
            summary.measure for summary in self.summary
        }:
            raise ValueError("Verdict fingerprint for an unsummarized measure")
        return self


class BenchmarkDiagnostics(ContractModel):
    """Judge diagnostics for a complete matrix; task rewards stay separate."""

    schema_version: Literal["benchmark-diagnostics-v1"] = "benchmark-diagnostics-v1"
    benchmark_version: Annotated[str, Field(description="Released benchmark.")]
    frame_revision: Annotated[str, Field(description="Frozen frame revision.")]
    catalog_revision: Annotated[
        str, Field(description="Production judge catalog revision.")
    ]
    transport_revision: Annotated[
        str | None,
        Field(description="Judge request transport revision; null if none judged."),
    ]
    judges: Annotated[
        list[JudgeIdentity], Field(description="Every judge contract used.")
    ]
    cells: Annotated[
        list[CellDiagnostics], Field(description="One entry per released cell.")
    ]

    @model_validator(mode="after")
    def _known_contracts(self) -> "BenchmarkDiagnostics":
        contracts = {judge.contract_id for judge in self.judges}
        if len(contracts) != len(self.judges):
            raise ValueError("Duplicate judge contract identities")
        for cell in self.cells:
            if any(row.contract_id not in contracts for row in cell.verdicts):
                raise ValueError("Verdict references an undeclared judge contract")
        return self


class CellJudgeCoverage(ContractModel):
    """Per-cell judge summaries, or null when the cell has no saved results."""

    language: str
    domain: str
    summary: list[DiagnosticSummary] | None

    @property
    def complete(self) -> bool:
        return self.summary is not None and all(s.complete for s in self.summary)


def _cell_rows(
    plan_items, stored: list[StoredVerdict]
) -> tuple[list[VerdictFingerprint], list[StoredVerdict]]:
    records = {
        (
            row.identity.input.run_id,
            row.identity.input.simulation_id,
            row.identity.input.trial,
            row.identity.measure,
        ): row
        for row in stored
    }
    fingerprints = []
    used = []
    for item in plan_items:
        if item.action in {PlanAction.NOT_APPLICABLE, PlanAction.MISSING_INPUT}:
            continue
        row = records.get(
            (
                item.input.run_id,
                item.input.simulation_id,
                item.input.trial,
                item.measure,
            )
        )
        if (
            row is None
            or row.identity != item.identity
            or row.error is not None
            or row.verdict is None
            # A partial verdict without units adds nothing to the summary,
            # unless it is judged because only fidelity output limits failed.
            or (
                row.verdict.unit_errors
                and not row.verdict.units
                and not any(
                    is_fidelity_output_limit(item.measure, error)
                    for error in row.verdict.unit_errors
                )
            )
        ):
            continue
        units = row.verdict.units
        fingerprints.append(
            VerdictFingerprint(
                simulation_id=item.input.simulation_id,
                trial=item.input.trial,
                measure=item.measure,
                contract_id=row.identity.contract_id,
                input_fingerprint=row.identity.input.input_fingerprint,
                audio_fingerprint=row.identity.input.audio_fingerprint,
                verdict_sha256=_sha256(row.model_dump(mode="json")),
                pass_units=sum(u.opportunity and not u.violation for u in units),
                fail_units=sum(u.opportunity and u.violation for u in units),
                no_opportunity_units=sum(not u.opportunity for u in units),
                error_units=len(row.verdict.unit_errors),
                input_error_units=sum(
                    error.kind == "input" for error in row.verdict.unit_errors
                ),
                output_limit_units=sum(
                    is_fidelity_output_limit(item.measure, error)
                    for error in row.verdict.unit_errors
                ),
            )
        )
        used.append(row)
    return fingerprints, used


def collect_diagnostics(
    root: Path, frame: BenchmarkFrame, catalog: JudgeCatalog
) -> tuple[list[CellJudgeCoverage], BenchmarkDiagnostics | None]:
    """Read saved verdicts for every cell without model calls.

    Returns per-cell coverage (for incomplete-run reporting) and, only when every
    cell is fully judged, the portable diagnostics export.
    """
    from tau2.judges.suite import judge_saved_run

    coverage = []
    cells = []
    for cell in frame.cells:
        directory = root / f"{cell.language}-{cell.domain}"
        if not (directory / "results.json").exists():
            coverage.append(
                CellJudgeCoverage(
                    language=cell.language, domain=cell.domain, summary=None
                )
            )
            continue
        result = judge_saved_run(directory, catalog)
        coverage.append(
            CellJudgeCoverage(
                language=cell.language,
                domain=cell.domain,
                summary=summarize_suite(result.plan, result.verdicts),
            )
        )
        # Incomplete cells are only reported: preparation refuses them.
        if coverage[-1].complete:
            cells.append((cell.language, cell.domain, result.plan, result.verdicts))
    if not all(cell.complete for cell in coverage):
        return coverage, None
    return coverage, export_diagnostics(frame, catalog.revision, cells)


def export_diagnostics(
    frame: BenchmarkFrame,
    catalog_revision: str,
    cells: list[tuple[str, str, SuitePlan, list[StoredVerdict]]],
) -> BenchmarkDiagnostics:
    """Summaries, fingerprints and judge identities for each judged cell."""
    exported = []
    judges: dict[str, JudgeIdentity] = {}
    transports = set()
    for language, domain, plan, stored in cells:
        fingerprints, used = _cell_rows(plan.items, stored)
        for row in used:
            identity = row.identity
            transports.add(identity.transport_revision)
            judges.setdefault(
                identity.contract_id,
                JudgeIdentity(
                    contract_id=identity.contract_id,
                    implementation=identity.implementation,
                    input_exclusions=identity.input_exclusions,
                    model=identity.model,
                    model_args=identity.model_args,
                    prompt_version=identity.prompt_version,
                    rubric_version=identity.rubric_version,
                ),
            )
        exported.append(
            CellDiagnostics(
                language=language,
                domain=domain,
                summary=summarize_suite(plan, stored),
                verdicts=fingerprints,
            )
        )
    if len(transports) > 1:
        raise ValueError("Saved verdicts mix judge transport revisions")
    return BenchmarkDiagnostics(
        benchmark_version=frame.benchmark_version,
        frame_revision=frame.revision,
        catalog_revision=catalog_revision,
        transport_revision=next(iter(transports), None),
        judges=sorted(judges.values(), key=lambda judge: judge.contract_id),
        cells=exported,
    )


def output_limit_overage(diagnostics: BenchmarkDiagnostics) -> list[str]:
    """Lines describing an over-cap share of fidelity output-limit clips, if any.

    Such clips are left out of scoring rather than blocking their call. A judge
    failure that rare is tolerated; a model whose speech keeps sending the judge
    into a runaway reply needs review instead of silently losing those clips.
    """
    limited = judged = 0
    calls = []
    for cell in diagnostics.cells:
        for summary in cell.summary:
            if summary.measure == "speech_fidelity":
                limited += summary.output_limit_units
                judged += (
                    summary.pass_units + summary.fail_units + summary.output_limit_units
                )
        calls.extend(
            f"  {cell.language}-{cell.domain} {row.simulation_id} trial "
            f"{row.trial}: {row.output_limit_units} clip(s)"
            for row in cell.verdicts
            if row.output_limit_units
        )
    if OUTPUT_LIMIT_DIVISOR * limited <= judged:
        return []
    return [
        f"Judges: {limited} of {judged} speech-fidelity clips hit the judge's "
        f"output limit, above the 1/{OUTPUT_LIMIT_DIVISOR} allowed. A maintainer "
        "must review these calls:",
        *calls,
    ]
