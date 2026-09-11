# Copyright Sierra
"""Thin ``tau2 pool`` adapter over the shared multiprocess runner."""

from __future__ import annotations

from typing import Optional

from tau2.runner.pools import CellCensus, PoolArm, PoolSpec, census_pool
from tau2.utils.display import ConsoleDisplay


def format_matrix(spec: PoolSpec, censuses: dict[str, CellCensus]) -> str:
    """Render the pool's usable-result and task-coverage snapshot."""
    width = 17
    header = " " * 4 + "".join(
        f"{arm.provider[:3] + '/' + arm.reasoning_effort:>{width}}" for arm in spec.arms
    )
    lines = [header + f"{'row':>12}"]

    grand = complete = 0
    for language in spec.languages:
        row, row_total = f"{language:4}", 0
        for arm in spec.arms:
            census = censuses[spec.cell_name(language, arm)]
            flag = (
                f"!{census.infrastructure_errors}"
                if census.infrastructure_errors
                else ""
            )
            row += f"{census.usable:>4}/{spec.target_per_cell} {census.tasks:>2}t{flag:>3} "
            row_total += census.usable
            if census.usable >= spec.target_per_cell and census.is_complete:
                complete += 1
        grand += row_total
        lines.append(row + f"{row_total:>7}/{spec.target_per_cell * len(spec.arms)}")

    total_cells = len(spec.languages) * len(spec.arms)
    pct = 100 * grand / spec.total_target if spec.total_target else 0
    lines.extend(
        [
            "",
            f"{grand} / {spec.total_target} usable  ({pct:.0f}%)   "
            f"{complete}/{total_cells} cells clean",
        ]
    )

    infra = sum(c.infrastructure_errors for c in censuses.values())
    duplicates = sum(c.duplicates for c in censuses.values())
    lines.append(
        f"recorded exceptions: {infra} infrastructure errors, "
        f"{duplicates} duplicate simulations"
    )
    short = [
        f"{name}:{spec.target_per_cell - census.usable}"
        for name, census in censuses.items()
        if census.usable < spec.target_per_cell
    ]
    if short:
        lines.append(
            f"short: {spec.total_target - grand} usable results across "
            f"{len(short)} cells — " + " ".join(short)
        )
    return "\n".join(lines)


def _selected_cells(
    spec: PoolSpec,
    only: Optional[list[str]],
    languages: Optional[list[str]],
) -> list[tuple[str, PoolArm]]:
    selected = [
        (language, arm)
        for language, arm in spec.cells()
        if (not languages or language in languages)
        and (not only or spec.cell_name(language, arm) in only)
    ]
    if not selected:
        raise SystemExit(
            "no cells selected — check --only / --langs against `tau2 pool list`"
        )
    return selected


def run_pool(
    spec: PoolSpec,
    *,
    only: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
    workers: int,
    provider_limits: Optional[dict[str, int]] = None,
    dry_run: bool = False,
    redo_stale_tasks: bool = False,
) -> int:
    """Run every selected cell once through one shared controller."""
    cells = _selected_cells(spec, only, languages)
    configs = [
        spec.cell_config(language, arm, redo_stale_tasks=redo_stale_tasks)
        for language, arm in cells
    ]

    ConsoleDisplay.console.print(
        f"[bold]POOL {spec.name}[/bold] — {len(cells)} run(s), "
        f"{workers} workers x {spec.max_concurrency} slots"
    )
    if dry_run:
        for (language, arm), config in zip(cells, configs):
            ConsoleDisplay.console.print(
                f"\n[bold]{spec.cell_name(language, arm)}[/bold]\n"
                f"{config.model_dump_json(indent=2)}"
            )
        return 0

    from tau2.runner.batch import run_domains

    run_domains(
        configs,
        workers=workers,
        provider_limits=provider_limits,
    )
    ConsoleDisplay.console.print("\n" + format_matrix(spec, census_pool(spec)))
    return 0
