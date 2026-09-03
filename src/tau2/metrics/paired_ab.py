"""Paired A/B scoring for two-arm comparisons (e.g. fp8 vs fp4 serving).

The single-arm metrics in agent_metrics.py exclude episodes asymmetrically:
infrastructure_error simulations are dropped from all metrics, and premature
terminations (max_steps loops) get reward 0.0 but contribute no action checks.
When the two arms fail in different modes, their metrics are averaged over
differently-selected survivor subsets and are not comparable (T19 audit,
2026-09-03).

This module compares two Results task by task and only scores episode pairs
where BOTH arms produced a scored episode. Everything excluded is counted and
reported, never silently dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tau2.data_model.simulation import Results, SimulationRun, TerminationReason

SCORED_TERMINATIONS = {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}


def episode_status(sim: SimulationRun) -> str:
    """scored | infra_error | unscored_premature"""
    if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
        return "infra_error"
    if sim.termination_reason not in SCORED_TERMINATIONS:
        return "unscored_premature"
    return "scored"


def _action_counts(sim: SimulationRun) -> tuple[int, int]:
    """(matched, total) action checks, (0, 0) when absent."""
    ri = sim.reward_info
    if ri is None or not ri.action_checks:
        return 0, 0
    return sum(1 for ac in ri.action_checks if ac.action_match), len(ri.action_checks)


@dataclass
class ArmPairedMetrics:
    name: str
    pass_count: int = 0
    action_matched: int = 0
    action_total: int = 0

    @property
    def action_match_rate(self) -> Optional[float]:
        return self.action_matched / self.action_total if self.action_total else None


@dataclass
class PairedReport:
    arm_a: ArmPairedMetrics
    arm_b: ArmPairedMetrics
    paired_task_ids: list[str] = field(default_factory=list)
    exclusions: dict[str, Counter] = field(default_factory=dict)
    tasks_only_in_one_arm: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.paired_task_ids)

    def render(self) -> str:
        lines = [
            f"Paired A/B report: {self.arm_a.name} vs {self.arm_b.name}",
            f"  paired scored tasks: {self.n_pairs}",
        ]
        for arm in (self.arm_a, self.arm_b):
            amr = arm.action_match_rate
            lines.append(
                f"  {arm.name}: pass {arm.pass_count}/{self.n_pairs}, "
                f"action match {arm.action_matched}/{arm.action_total}"
                + (f" = {amr:.1%}" if amr is not None else "")
            )
        for name, counts in self.exclusions.items():
            if counts:
                lines.append(f"  excluded [{name}]: {dict(counts)}")
        for name, tids in self.tasks_only_in_one_arm.items():
            if tids:
                lines.append(f"  tasks only in {name}: {sorted(tids)}")
        if self.n_pairs < 10:
            lines.append(
                f"  WARNING: only {self.n_pairs} paired tasks; "
                "differences at this n are not statistically meaningful."
            )
        return "\n".join(lines)


def _scored_by_task(results: Results, name: str, exclusions: dict[str, Counter]) -> dict[str, SimulationRun]:
    """First scored episode per task; excluded episodes are tallied, not dropped silently."""
    scored: dict[str, SimulationRun] = {}
    exclusions[name] = Counter()
    for sim in results.simulations:
        status = episode_status(sim)
        if status != "scored":
            exclusions[name][f"{status}:{sim.termination_reason.value}"] += 1
            continue
        scored.setdefault(sim.task_id, sim)
    return scored


def paired_compare(results_a: Results, results_b: Results,
                   name_a: str = "arm_a", name_b: str = "arm_b") -> PairedReport:
    exclusions: dict[str, Counter] = {}
    scored_a = _scored_by_task(results_a, name_a, exclusions)
    scored_b = _scored_by_task(results_b, name_b, exclusions)

    paired = sorted(set(scored_a) & set(scored_b))
    report = PairedReport(
        arm_a=ArmPairedMetrics(name=name_a),
        arm_b=ArmPairedMetrics(name=name_b),
        paired_task_ids=paired,
        exclusions=exclusions,
        tasks_only_in_one_arm={
            name_a: sorted(set(scored_a) - set(scored_b)),
            name_b: sorted(set(scored_b) - set(scored_a)),
        },
    )
    for tid in paired:
        for arm, sim in ((report.arm_a, scored_a[tid]), (report.arm_b, scored_b[tid])):
            if sim.reward_info is not None and (sim.reward_info.reward or 0) >= 1.0:
                arm.pass_count += 1
            matched, total = _action_counts(sim)
            arm.action_matched += matched
            arm.action_total += total
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Paired A/B scoring of two tau2 Results")
    ap.add_argument("results_a", type=Path)
    ap.add_argument("results_b", type=Path)
    ap.add_argument("--name-a", default="arm_a")
    ap.add_argument("--name-b", default="arm_b")
    a = ap.parse_args()
    report = paired_compare(Results.load(a.results_a), Results.load(a.results_b),
                            a.name_a, a.name_b)
    print(report.render())


if __name__ == "__main__":
    main()
