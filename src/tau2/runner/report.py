"""Compact terminal views of benchmark plans and score summaries."""

from tau2.data_model.benchmark import BenchmarkPlan, BenchmarkSummary
from tau2.utils.table import format_fraction, format_table


def render_plan(plan: BenchmarkPlan) -> str:
    """One row per selected cell config; the full plan stays in benchmark-plan.json."""
    return format_table(
        ["language", "domain", "tasks", "trials", "calls"],
        [
            [
                config.language,
                config.domain,
                len(config.task_ids),
                config.num_trials,
                len(config.task_ids) * config.num_trials,
            ]
            for config in plan.configs
        ],
    )


def render_summary(summary: BenchmarkSummary) -> str:
    """Per-cell coverage and score; partial cells show their observed score.

    Cells owing no trials (outside a requested scope) are omitted.
    """
    rows = [
        [
            cell.language,
            cell.domain,
            f"{cell.scored_trials}/{cell.expected_trials}",
            cell.error_trials,
            cell.unscored_trials,
            cell.missing_trials,
            format_fraction(cell.score),
            format_fraction(cell.observed_score),
            "yes" if cell.complete else "no",
        ]
        for cell in summary.cells
        if cell.expected_trials
    ]
    return format_table(
        [
            "language",
            "domain",
            "scored",
            "errors",
            "unscored",
            "missing",
            "score",
            "observed",
            "complete",
        ],
        rows,
    )
