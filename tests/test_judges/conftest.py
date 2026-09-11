# Copyright Sierra
"""Fixtures for the judges suite.

The run/sim/check builders live in ``tests/fixtures_runs.py`` (shared with
the annotation suite); this conftest pins the judges-suite shape: complete
stored verdicts from a "stored-judge" (severity 3, evidence recorded), with
an optional DEFERRED/ERROR gap so rejudge/export paths can exercise
reuse-gating end-to-end on disk without any LLM calls.
"""

from typing import Optional

import fixtures_runs
from fixtures_runs import (  # noqa: F401
    HI_TRANSCRIPT,
    make_hi_results,
    make_task,
)

from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    SimulationRun,
)


def hi_factor_checks(
    gap_factor: Optional[str] = None,
    gap_outcome: JudgeOutcome = JudgeOutcome.DEFERRED,
) -> list[NativenessFactorCheck]:
    """Stored hi judge-factor checks: every factor PASSes except ``gap_factor``
    (recorded with ``gap_outcome``, e.g. DEFERRED/ERROR, to open a gap)."""
    return fixtures_runs.hi_factor_checks(
        gap_factor=gap_factor,
        gap_outcome=gap_outcome,
        pass_severity=3,
        pass_evidence="stored evidence",
    )


def hi_sim(
    sim_id: str,
    task_id: str,
    *,
    language: str = "hi",
    checks: Optional[list[NativenessFactorCheck]] = None,
    trial: int = 0,
) -> SimulationRun:
    return fixtures_runs.hi_sim(
        sim_id,
        task_id,
        checks=checks if checks is not None else hi_factor_checks(),
        judge_model="stored-judge",
        language=language,
        trial=trial,
    )
