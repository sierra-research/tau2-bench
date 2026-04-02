"""Report generation: markdown, CSV, and structured output."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tau2_reliability.models import ReliabilityReport


def generate_markdown_report(report: ReliabilityReport) -> str:
    """Generate a human-readable markdown reliability report."""
    lines = []
    lines.append(f"# Reliability Report: {report.domain} ({report.agent_model})")
    lines.append("")
    lines.append(
        f"{report.num_tasks} tasks x {report.num_trials} trials | "
        f"Accuracy: {report.accuracy:.1%}"
    )
    lines.append("")

    # Overall scores
    lines.append("## Overall Reliability")
    lines.append("")
    lines.append("| Dimension | Score | SE |")
    lines.append("|-----------|-------|----|")
    _add_dim_row(lines, "Consistency (R_Con)", report.r_con, report.bootstrap_se.get("r_con"))
    _add_dim_row(lines, "Predictability (R_Pred)", report.r_pred, report.bootstrap_se.get("r_pred"))
    _add_dim_row(lines, "Robustness (R_Rob)", report.r_rob, report.bootstrap_se.get("r_rob"))
    _add_dim_row(lines, "**Overall (R)**", report.r_overall, None)
    lines.append("")

    # Consistency detail
    if report.consistency:
        c = report.consistency
        lines.append("## Consistency Metrics")
        lines.append("")
        lines.append("| Metric | Score | SE |")
        lines.append("|--------|-------|----|")
        _add_metric_row(lines, "Outcome (C_out)", c.c_out, report.bootstrap_se.get("c_out"))
        _add_metric_row(lines, "Traj. Distribution (C_traj_d)", c.c_traj_d, report.bootstrap_se.get("c_traj_d"))
        _add_metric_row(lines, "Traj. Sequence (C_traj_s)", c.c_traj_s, report.bootstrap_se.get("c_traj_s"))
        _add_metric_row(lines, "Resource (C_res)", c.c_res, report.bootstrap_se.get("c_res"))
        lines.append("")

        # Per-task breakdown
        if c.per_task:
            lines.append("## Per-Task Consistency")
            lines.append("")
            lines.append("| Task | C_out | C_traj_d | C_traj_s | C_res |")
            lines.append("|------|-------|----------|----------|-------|")
            for tid in sorted(c.per_task.keys()):
                m = c.per_task[tid]
                lines.append(
                    f"| {tid} | {_fmt(m.get('c_out'))} | {_fmt(m.get('c_traj_d'))} | "
                    f"{_fmt(m.get('c_traj_s'))} | {_fmt(m.get('c_res'))} |"
                )
            lines.append("")

    # Predictability detail
    if report.predictability:
        p = report.predictability
        lines.append("## Predictability Metrics")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Calibration (P_cal) | {_fmt(p.p_cal)} |")
        lines.append(f"| Discrimination (P_auroc) | {_fmt(p.p_auroc)} |")
        lines.append(f"| Brier Score (P_brier) | {_fmt(p.p_brier)} |")
        lines.append("")

    # Task taxonomy
    if report.task_taxonomy:
        t = report.task_taxonomy
        lines.append("## Task Taxonomy")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|-------|-------|")
        for cls, count in sorted(t.counts.items()):
            lines.append(f"| {cls} | {count} |")
        lines.append("")
        if t.bimodal_tasks:
            lines.append(
                f"**Bimodal tasks** (most informative for reliability): "
                f"{', '.join(t.bimodal_tasks[:20])}"
            )
            if len(t.bimodal_tasks) > 20:
                lines.append(f"  ... and {len(t.bimodal_tasks) - 20} more")
            lines.append("")

    # Mutation analysis
    if report.mutation_analysis:
        m = report.mutation_analysis
        lines.append("## Mutation-Aware Analysis")
        lines.append("")
        lines.append(f"- **Write action fraction**: {m.write_action_fraction:.1%}")
        if m.mutation_risk_ratio is not None:
            lines.append(f"- **Mutation risk ratio** (WRITE/READ importance): {m.mutation_risk_ratio:.2f}")
        if m.verification_gap is not None:
            lines.append(f"- **Verification gap** (success - failure READ-before-WRITE rate): {m.verification_gap:+.1%}")
        if m.verification_rate_success is not None:
            lines.append(f"  - Success trials: {m.verification_rate_success:.1%} of WRITEs preceded by READ")
        if m.verification_rate_failure is not None:
            lines.append(f"  - Failure trials: {m.verification_rate_failure:.1%} of WRITEs preceded by READ")
        if m.decisive_mutations:
            lines.append("")
            lines.append("**Decisive mutations** (WRITE actions most correlated with failure):")
            for tid, action in sorted(m.decisive_mutations.items()):
                lines.append(f"  - Task {tid}: `{action}`")
        lines.append("")

    # Policy adherence
    if report.policy_adherence:
        lines.append("## Policy Adherence (SOP-as-DAG)")
        lines.append("")
        avg = sum(p.adherence_score for p in report.policy_adherence) / len(report.policy_adherence)
        lines.append(f"Average adherence score: **{avg:.3f}**")
        lines.append("")
        # Show tasks with violations
        violated = [p for p in report.policy_adherence if p.violations]
        if violated:
            lines.append(f"Tasks with policy violations: **{len(violated)}** / {len(report.policy_adherence)}")
            lines.append("")
            lines.append("| Task | Score | Violations |")
            lines.append("|------|-------|------------|")
            for p in sorted(violated, key=lambda x: x.adherence_score)[:20]:
                v_summary = "; ".join(v.description[:60] for v in p.violations[:3])
                lines.append(f"| {p.task_id} | {p.adherence_score:.2f} | {v_summary} |")
            lines.append("")

    # Divergence summary
    if report.divergence_profiles:
        bimodal_profiles = [
            p for p in report.divergence_profiles
            if p.divergence_turn is not None
        ]
        if bimodal_profiles:
            lines.append("## Divergence Analysis")
            lines.append("")
            lines.append("| Task | Diverge Turn | Prefix | Type | Success Path | Failure Path |")
            lines.append("|------|-------------|--------|------|-------------|-------------|")
            for p in sorted(bimodal_profiles, key=lambda x: x.divergence_turn or 999)[:20]:
                prefix = "→".join(p.consensus_prefix[:3]) or "-"
                s_path = "→".join(p.success_path[:4]) or "-"
                f_path = "→".join(p.failure_path[:4]) or "-"
                dtype = p.divergence_type.value if p.divergence_type else "-"
                lines.append(f"| {p.task_id} | {p.divergence_turn} | {prefix} | {dtype} | {s_path} | {f_path} |")
            lines.append("")

    # Bootstrap info
    if report.bootstrap_se:
        lines.append("## Bootstrap Standard Errors (200 resamples)")
        lines.append("")
        lines.append("| Metric | SE |")
        lines.append("|--------|----|")
        for k, v in sorted(report.bootstrap_se.items()):
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated: {report.timestamp}*")
    return "\n".join(lines)


def generate_csv(report: ReliabilityReport, output_path: Path) -> None:
    """Write per-task metrics to CSV."""
    if not report.consistency or not report.consistency.per_task:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    rows = []
    for tid, metrics in report.consistency.per_task.items():
        row = {"task_id": tid}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("task_id")
    df.to_csv(output_path, index=False)


def _fmt(value: float | None) -> str:
    """Format a metric value for display."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.3f}"


def _add_dim_row(
    lines: list[str], name: str, value: float | None, se: float | None
) -> None:
    se_str = f"+-{se:.3f}" if se else ""
    lines.append(f"| {name} | {_fmt(value)} | {se_str} |")


def _add_metric_row(
    lines: list[str], name: str, value: float, se: float | None
) -> None:
    se_str = f"+-{se:.3f}" if se else ""
    lines.append(f"| {name} | {_fmt(value)} | {se_str} |")
