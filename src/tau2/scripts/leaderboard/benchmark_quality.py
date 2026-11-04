"""Call-level quality carried beside a voice-v2 package: ``benchmark-quality.json``.

Every scored call gets an Interaction and an Experience (displayed as
Generation quality) verdict under the rules in ``tau2.metrics.call_quality``.
Both are recomputed from the saved calls and their saved judge verdicts, so
``tau2 submit validate`` rejects any sidecar that differs. Neither is part of
the task reward. Infrastructure-error calls are not scored and carry none.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tau2.data_model.benchmark import BenchmarkFrame
from tau2.data_model.judges import (
    ContractModel,
    JudgeCatalog,
    PlanAction,
    StoredVerdict,
)
from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.metrics import call_quality
from tau2.metrics.call_quality import (
    EXPERIENCE_VERSION,
    INTERACTION_COMPONENTS,
    INTERACTION_VERSION,
    TOOL_USE_MEASURE,
)
from tau2.metrics.interaction_quality import measure_interaction

QUALITY_FILE_NAME = "benchmark-quality.json"
# Judged benchmarks whose package must carry the quality sidecar.
QUALITY_BENCHMARKS = frozenset({"voice-v2"})
Flag = Literal[0, 1] | None


class OutcomeCounts(ContractModel):
    """Calls per outcome; rates use passed / (passed + failed)."""

    passed: Annotated[int, Field(ge=0, description="Calls that passed.")]
    failed: Annotated[int, Field(ge=0, description="Calls that failed.")]
    unscored: Annotated[
        int, Field(ge=0, description="Calls whose inputs cannot decide the outcome.")
    ]
    excluded: Annotated[
        int,
        Field(ge=0, description="Calls excluded by a documented rule; none in v1."),
    ] = 0


class ComponentCoverage(ContractModel):
    """One Interaction component over a cell's scored calls."""

    flagged: Annotated[int, Field(ge=0, description="Calls with the flag set.")]
    known: Annotated[int, Field(ge=0, description="Calls where the flag is known.")]


class InteractionComponents(ContractModel):
    nonresponse: ComponentCoverage
    interruption: ComponentCoverage
    selectivity: ComponentCoverage
    monologue: ComponentCoverage
    tool_misuse: ComponentCoverage


class InteractionFlags(ContractModel):
    """The five component flags; null when the call cannot decide one."""

    nonresponse: Annotated[Flag, Field(description="Missed, late or unyielded turn.")]
    interruption: Annotated[Flag, Field(description="More than 3 agent onsets.")]
    selectivity: Annotated[Flag, Field(description="More than 3 selectivity errors.")]
    monologue: Annotated[Flag, Field(description="Agent floor hold over 45 s.")]
    tool_misuse: Annotated[Flag, Field(description="Saved tool_use judge verdict.")]


class CallQuality(ContractModel):
    """Both verdicts for one scored call, with the inputs that decide them."""

    simulation_id: Annotated[str, Field(description="Saved simulation identifier.")]
    interaction: Annotated[
        Literal["pass", "fail", "unscored", "excluded"],
        Field(description="Call-level Interaction outcome."),
    ]
    flags: InteractionFlags
    experience: Annotated[
        Literal["pass", "fail", "unscored"],
        Field(description="Call-level Experience (Generation quality) outcome."),
    ]
    experience_utterances: Annotated[
        int, Field(ge=0, description="Eligible agent utterances (naturalness units).")
    ]
    experience_issues: Annotated[
        int | None,
        Field(
            ge=0,
            description="Utterances either judge flagged; null when the paired "
            "judgments are incomplete.",
        ),
    ]

    @model_validator(mode="after")
    def _outcomes_follow_the_rules(self) -> "CallQuality":
        flags = self.flags.model_dump()
        if self.interaction != call_quality.interaction_outcome(flags):
            raise ValueError(f"{self.simulation_id}: Interaction outcome != flags")
        if self.experience_issues is not None and (
            self.experience_issues > self.experience_utterances
        ):
            raise ValueError(f"{self.simulation_id}: more issues than utterances")
        if self.experience != call_quality.experience_outcome(
            self.experience_utterances, self.experience_issues
        ):
            raise ValueError(f"{self.simulation_id}: Experience outcome != counts")
        return self


def _counts(outcomes: Iterable[str]) -> OutcomeCounts:
    outcomes = list(outcomes)
    return OutcomeCounts(
        passed=outcomes.count("pass"),
        failed=outcomes.count("fail"),
        unscored=outcomes.count("unscored"),
        excluded=outcomes.count("excluded"),
    )


def _components(calls: list[CallQuality]) -> InteractionComponents:
    return InteractionComponents(
        **{
            name: ComponentCoverage(
                flagged=sum(getattr(c.flags, name) == 1 for c in calls),
                known=sum(getattr(c.flags, name) is not None for c in calls),
            )
            for name in INTERACTION_COMPONENTS
        }
    )


class CellQuality(ContractModel):
    """Both measures for one language/domain cell, summed from its calls."""

    language: Annotated[str, Field(description="Language identifier.")]
    domain: Annotated[str, Field(description="Domain identifier.")]
    interaction: OutcomeCounts
    experience: OutcomeCounts
    interaction_components: InteractionComponents
    calls: Annotated[
        list[CallQuality], Field(description="Every scored call, by simulation ID.")
    ]

    @model_validator(mode="after")
    def _totals_match_calls(self) -> "CellQuality":
        ids = [call.simulation_id for call in self.calls]
        if ids != sorted(set(ids)):
            raise ValueError(f"{self.language}-{self.domain}: calls unsorted or duped")
        if (
            self.interaction != _counts(c.interaction for c in self.calls)
            or self.experience != _counts(c.experience for c in self.calls)
            or self.interaction_components != _components(self.calls)
        ):
            raise ValueError(
                f"{self.language}-{self.domain}: totals do not match the calls"
            )
        return self


class BenchmarkQuality(ContractModel):
    """Call-level Interaction and Experience for every frame cell."""

    schema_version: Literal["benchmark-quality-v1"] = "benchmark-quality-v1"
    benchmark_version: Annotated[str, Field(description="Released benchmark.")]
    frame_revision: Annotated[str, Field(description="Frozen frame revision.")]
    interaction_version: Literal["voice-v2-interaction-call-v1"] = INTERACTION_VERSION
    experience_version: Literal["voice-v2-experience-call-v2"] = EXPERIENCE_VERSION
    cells: Annotated[
        list[CellQuality], Field(description="One entry per frame cell, frame order.")
    ]


def score_call(
    simulation: SimulationRun, verdicts: dict[str, StoredVerdict]
) -> CallQuality:
    """Both verdicts for one scored call from its ticks and saved judge rows."""
    flags = call_quality.interaction_flags(
        measure_interaction(simulation), verdicts.get(TOOL_USE_MEASURE)
    )
    utterances, issues = call_quality.experience_counts(simulation, verdicts)
    return CallQuality(
        simulation_id=simulation.id,
        interaction=call_quality.interaction_outcome(flags).value,
        flags=InteractionFlags(**flags),
        experience=call_quality.experience_outcome(utterances, issues).value,
        experience_utterances=utterances,
        experience_issues=issues,
    )


def cell_quality(language: str, domain: str, calls: list[CallQuality]) -> CellQuality:
    calls = sorted(calls, key=lambda call: call.simulation_id)
    return CellQuality(
        language=language,
        domain=domain,
        interaction=_counts(c.interaction for c in calls),
        experience=_counts(c.experience for c in calls),
        interaction_components=_components(calls),
        calls=calls,
    )


def build_quality(frame: BenchmarkFrame, cells: list[CellQuality]) -> BenchmarkQuality:
    order = [(cell.language, cell.domain) for cell in frame.cells]
    if [(cell.language, cell.domain) for cell in cells] != order:
        raise ValueError("Quality cells must follow the frame's cells")
    return BenchmarkQuality(
        benchmark_version=frame.benchmark_version,
        frame_revision=frame.revision,
        cells=cells,
    )


def check_quality(quality: BenchmarkQuality, frame: BenchmarkFrame) -> None:
    """The sidecar cites the package's frame and covers each of its cells once."""
    if (
        quality.benchmark_version != frame.benchmark_version
        or quality.frame_revision != frame.revision
    ):
        raise ValueError("Quality and rewards cite different frames")
    if [(c.language, c.domain) for c in quality.cells] != [
        (c.language, c.domain) for c in frame.cells
    ]:
        raise ValueError("Quality must cover every frame cell once, in frame order")


def _scored(simulation: SimulationRun) -> bool:
    """Calls with a task reward, as ``read_task_trials`` scores them."""
    return simulation.reward_info is not None and simulation.termination_reason not in {
        TerminationReason.INFRASTRUCTURE_ERROR,
        TerminationReason.UNEXPECTED_ERROR,
    }


def collect_quality(
    root: Path, frame: BenchmarkFrame, catalog: JudgeCatalog
) -> BenchmarkQuality:
    """Score every scored call of a complete saved matrix; no model calls.

    A call's verdict for a measure counts only when the judge plan reuses it as
    current; anything else is missing input, which leaves components unknown.
    """
    from tau2.judges.suite import judge_saved_run

    cells = []
    for cell in frame.cells:
        directory = root / f"{cell.language}-{cell.domain}"
        result = judge_saved_run(directory, catalog)
        rows = {
            (
                row.identity.input.simulation_id,
                row.identity.input.trial,
                row.identity.measure,
            ): row
            for row in result.verdicts
        }
        current: dict[str, dict[str, StoredVerdict]] = {}
        for item in result.plan.items:
            key = (item.input.simulation_id, item.input.trial, item.measure)
            row = rows.get(key)
            if (
                item.action == PlanAction.REUSE
                and row is not None
                and row.identity == item.identity
            ):
                current.setdefault(item.input.simulation_id, {})[item.measure] = row
        calls = [
            score_call(simulation, current.get(simulation.id, {}))
            for simulation in Results.iter_simulations(directory / "results.json")
            if _scored(simulation)
        ]
        cells.append(cell_quality(cell.language, cell.domain, calls))
    return build_quality(frame, cells)
