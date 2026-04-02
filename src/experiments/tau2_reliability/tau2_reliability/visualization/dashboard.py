"""Aggregate reliability dashboard visualizations.

Publication-quality plots showing reliability metrics at the benchmark level.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tau2_reliability.models import ReliabilityReport
from tau2_reliability.visualization.styles import (
    DIMENSION_COLORS,
    apply_publication_style,
    get_metric_color,
)


def plot_reliability_dashboard(
    report: ReliabilityReport,
    figsize: tuple[float, float] = (16, 12),
) -> plt.Figure:
    """Single-page reliability summary dashboard.

    Top: dimension score gauges with overall R.
    Middle: per-metric bar chart.
    Bottom: per-task heatmap.
    """
    apply_publication_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 2], hspace=0.35)

    # --- Top row: dimension gauges ---
    ax_top = fig.add_subplot(gs[0])
    dims = {
        "Consistency\n(R_Con)": report.r_con,
        "Predictability\n(R_Pred)": report.r_pred,
        "Robustness\n(R_Rob)": report.r_rob,
        "Overall\n(R)": report.r_overall,
    }
    dim_colors = [
        DIMENSION_COLORS["consistency"],
        DIMENSION_COLORS["predictability"],
        DIMENSION_COLORS["robustness"],
        DIMENSION_COLORS["overall"],
    ]

    x_positions = range(len(dims))
    values = [v if v is not None else 0.0 for v in dims.values()]
    bars = ax_top.bar(x_positions, values, color=dim_colors, alpha=0.85, width=0.6)
    for bar, v, orig in zip(bars, values, dims.values()):
        label = f"{v:.2f}" if orig is not None else "N/A"
        ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    label, ha="center", va="bottom", fontweight="bold")
    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels(list(dims.keys()))
    ax_top.set_ylim(0, 1.15)
    ax_top.set_ylabel("Score")
    ax_top.set_title(f"Reliability Dashboard — {report.domain} ({report.agent_model})")

    # --- Middle row: sub-metric bar chart ---
    ax_mid = fig.add_subplot(gs[1])
    metrics = {}
    if report.consistency:
        c = report.consistency
        metrics.update({"c_out": c.c_out, "c_traj_d": c.c_traj_d,
                        "c_traj_s": c.c_traj_s, "c_res": c.c_res})
    if report.predictability:
        p = report.predictability
        metrics.update({"p_cal": p.p_cal, "p_auroc": p.p_auroc, "p_brier": p.p_brier})

    if metrics:
        names = list(metrics.keys())
        vals = list(metrics.values())
        colors = [get_metric_color(n) for n in names]
        bars = ax_mid.bar(range(len(names)), vals, color=colors, alpha=0.8)
        ax_mid.set_xticks(range(len(names)))
        ax_mid.set_xticklabels(names, rotation=30, ha="right")
        ax_mid.set_ylim(0, 1.15)
        ax_mid.set_ylabel("Score")
        ax_mid.set_title("Sub-Metric Breakdown")
        for bar, v in zip(bars, vals):
            ax_mid.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # --- Bottom row: per-task heatmap ---
    ax_bot = fig.add_subplot(gs[2])
    if report.consistency and report.consistency.per_task:
        import pandas as pd
        per_task = report.consistency.per_task
        df = pd.DataFrame(per_task).T
        df = df.sort_values("c_out", ascending=False)
        sns.heatmap(
            df, ax=ax_bot, cmap="RdYlGn", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "Score"},
        )
        ax_bot.set_title("Per-Task Consistency Metrics")
        ax_bot.set_ylabel("Task ID")
    else:
        ax_bot.text(0.5, 0.5, "No per-task data available", ha="center", va="center")
        ax_bot.set_axis_off()

    return fig


def plot_consistency_detail(
    report: ReliabilityReport,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Detailed consistency analysis with scatter and distributions."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if not report.consistency or not report.consistency.per_task:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
        return fig

    per_task = report.consistency.per_task
    task_ids = sorted(per_task.keys())
    c_outs = [per_task[t]["c_out"] for t in task_ids]
    c_traj_ds = [per_task[t].get("c_traj_d", 0) for t in task_ids]
    c_traj_ss = [per_task[t].get("c_traj_s", 0) for t in task_ids]

    # 1. c_out vs accuracy (pass_rate proxy not available here — use c_out distribution)
    axes[0].hist(c_outs, bins=10, color=DIMENSION_COLORS["consistency"], alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("C_out")
    axes[0].set_ylabel("Number of Tasks")
    axes[0].set_title("Outcome Consistency Distribution")
    axes[0].axvline(np.mean(c_outs), color="black", linestyle="--", label=f"mean={np.mean(c_outs):.2f}")
    axes[0].legend()

    # 2. c_traj_d vs c_traj_s scatter
    axes[1].scatter(c_traj_ds, c_traj_ss, c=c_outs, cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolors="black", linewidths=0.5, alpha=0.8)
    axes[1].set_xlabel("C_traj_d (Distribution)")
    axes[1].set_ylabel("C_traj_s (Sequence)")
    axes[1].set_title("Trajectory Consistency")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

    # 3. Box plots of all consistency sub-metrics
    data = [c_outs, c_traj_ds, c_traj_ss,
            [per_task[t].get("c_res", 0) for t in task_ids]]
    bp = axes[2].boxplot(data, labels=["C_out", "C_traj_d", "C_traj_s", "C_res"],
                         patch_artist=True)
    colors = [DIMENSION_COLORS["consistency"]] * 4
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[2].set_ylabel("Score")
    axes[2].set_title("Sub-Metric Distributions")
    axes[2].set_ylim(-0.05, 1.15)

    fig.suptitle("Consistency Analysis Detail", fontsize=13)
    plt.tight_layout()
    return fig


def plot_per_task_heatmap(
    report: ReliabilityReport,
    figsize: tuple[float, float] = (10, 12),
) -> plt.Figure:
    """Large heatmap: tasks (rows) x all available metrics (columns)."""
    apply_publication_style()
    import pandas as pd

    data = {}
    if report.consistency and report.consistency.per_task:
        for tid, metrics in report.consistency.per_task.items():
            data.setdefault(tid, {}).update(metrics)

    if not data:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No per-task data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df = pd.DataFrame(data).T.sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Per-Task Reliability Metrics")
    ax.set_ylabel("Task ID")
    plt.tight_layout()
    return fig
