"""Crossed Pass^3 carried beside an entity package: ``benchmark-pass3.json``.

The entity (intake) leaderboard score is crossed Pass^3 (``tau2.runner.pass3``):
every frame task runs once in each of three pinned voice conditions (regular,
heavy channel, heavy speech) and passes only if it passes in all three. The
package's ``benchmark-submission.json`` keeps the regular condition's rewards;
this sidecar holds each task's three outcomes and the Pass^3 summary, and is
recomputed from the saved calls on review.
"""

from typing import Annotated, Literal

from pydantic import Field

from tau2.data_model.benchmark import BenchmarkFrame, BenchmarkModel, TaskTrial
from tau2.runner.pass3 import CROSSED_CONDITIONS, CROSSED_PASS3_VERSION

PASS3_FILE_NAME = "benchmark-pass3.json"
# Frames whose leaderboard score is crossed Pass^3.
PASS3_BENCHMARKS = frozenset({"voice-v2-entity"})
CONDITION_LABELS = tuple(condition.label for condition in CROSSED_CONDITIONS)


class Pass3Task(BenchmarkModel):
    """One frame task's outcome in each condition, in R/C/S order."""

    canonical_task_id: Annotated[str, Field(min_length=1)]
    passes: Annotated[
        tuple[bool | None, bool | None, bool | None],
        Field(description="Reward 1.0 per condition; null is an infrastructure error."),
    ]


class Pass3Cell(BenchmarkModel):
    """Crossed Pass^3 over one language-domain cell."""

    language: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    sources: Annotated[
        dict[str, str],
        Field(description="SHA-256 of each condition's results.json, by label."),
    ]
    tasks: list[Pass3Task]
    scored_tasks: Annotated[
        int, Field(ge=0, description="Tasks with a reward in every condition.")
    ]
    passed_all: Annotated[int, Field(ge=0, description="Tasks passed in all three.")]
    score: Annotated[
        float | None,
        Field(description="passed_all / scored_tasks; null with no scored task."),
    ]


class BenchmarkPass3(BenchmarkModel):
    """The entity package's crossed Pass^3 sidecar."""

    schema_version: Literal["benchmark-pass3-v1"] = "benchmark-pass3-v1"
    benchmark_version: str
    frame_revision: str
    pass3_version: Literal[CROSSED_PASS3_VERSION] = CROSSED_PASS3_VERSION
    conditions: tuple[str, str, str] = CONDITION_LABELS
    cells: list[Pass3Cell]
    scored_tasks: int = Field(ge=0)
    passed_all: int = Field(ge=0)
    score: Annotated[
        float | None,
        Field(description="Equal-cell mean of cell scores; null unless all scored."),
    ]


def summarize_cell(
    language: str, domain: str, sources: dict[str, str], tasks: list[Pass3Task]
) -> Pass3Cell:
    """Count a cell's tasks: a task is scored when every condition has a reward."""
    scored = [task for task in tasks if None not in task.passes]
    passed = sum(all(task.passes) for task in scored)
    return Pass3Cell(
        language=language,
        domain=domain,
        sources=sources,
        tasks=tasks,
        scored_tasks=len(scored),
        passed_all=passed,
        score=passed / len(scored) if scored else None,
    )


def summarize_pass3(frame: BenchmarkFrame, cells: list[Pass3Cell]) -> BenchmarkPass3:
    scores = [cell.score for cell in cells]
    return BenchmarkPass3(
        benchmark_version=frame.benchmark_version,
        frame_revision=frame.revision,
        cells=cells,
        scored_tasks=sum(cell.scored_tasks for cell in cells),
        passed_all=sum(cell.passed_all for cell in cells),
        score=None if None in scores else sum(scores) / len(scores),
    )


def check_pass3(
    pass3: BenchmarkPass3, frame: BenchmarkFrame, trials: list[TaskTrial]
) -> None:
    """Recompute the counts; the regular column must match the submission."""
    if (
        pass3.benchmark_version != frame.benchmark_version
        or pass3.frame_revision != frame.revision
    ):
        raise ValueError("Pass^3 and rewards cite different frames")
    if pass3.conditions != CONDITION_LABELS:
        raise ValueError("Pass^3 conditions differ from the crossed catalog")
    frame_cells = {(cell.language, cell.domain): cell for cell in frame.cells}
    seen = [(cell.language, cell.domain) for cell in pass3.cells]
    if len(seen) != len(frame_cells) or set(seen) != set(frame_cells):
        raise ValueError("Pass^3 must cover every frame cell once")
    regular = {
        (trial.language, trial.domain, trial.canonical_task_id): (
            None if trial.reward is None else trial.reward == 1.0
        )
        for trial in trials
    }
    for cell in pass3.cells:
        task_ids = [task.canonical_task_id for task in cell.tasks]
        if sorted(task_ids) != sorted(
            frame_cells[(cell.language, cell.domain)].canonical_task_ids
        ) or len(set(task_ids)) != len(task_ids):
            raise ValueError(f"{cell.language}/{cell.domain}: Pass^3 task set differs")
        if any(
            task.passes[0]
            != regular[(cell.language, cell.domain, task.canonical_task_id)]
            for task in cell.tasks
        ):
            raise ValueError(
                f"{cell.language}/{cell.domain}: Pass^3 regular outcomes differ "
                "from the submission"
            )
        if summarize_cell(cell.language, cell.domain, cell.sources, cell.tasks) != cell:
            raise ValueError(f"{cell.language}/{cell.domain}: Pass^3 counts differ")
    if summarize_pass3(frame, pass3.cells) != pass3:
        raise ValueError("Pass^3 summary differs from its cells")
