"""Visualization for cross-trial trajectory divergence analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt

from tau2_reliability.models import DivergenceProfile, TaskTrialData
from tau2_reliability.visualization.styles import (
    DIVERGENCE_COLORS,
    apply_publication_style,
)


def plot_divergence_summary(
    profiles: list[DivergenceProfile],
    task_data: list[TaskTrialData],
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Summary scatter: divergence turn vs pass rate per task.

    Shows which tasks diverge early (high unreliability) vs late.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    td_map = {td.task_id: td for td in task_data}

    for profile in profiles:
        td = td_map.get(profile.task_id)
        if td is None:
            continue
        pass_rate = td.pass_rate
        div_turn = profile.divergence_turn
        if div_turn is None:
            continue

        color = DIVERGENCE_COLORS["agree"] if pass_rate > 0.7 else (
            DIVERGENCE_COLORS["diverge"] if pass_rate < 0.3 else "#f39c12"
        )
        ax.scatter(div_turn, pass_rate, c=color, s=80, edgecolors="black",
                   linewidths=0.5, alpha=0.8, zorder=3)
        ax.annotate(profile.task_id, (div_turn, pass_rate),
                    fontsize=7, ha="left", va="bottom", alpha=0.7)

    ax.set_xlabel("Divergence Turn (earliest)")
    ax.set_ylabel("Pass Rate")
    ax.set_title("When Do Trials Diverge?")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    return fig


def plot_divergence_tree(
    profile: DivergenceProfile,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Tree diagram showing consensus prefix branching into success/failure paths."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    prefix = profile.consensus_prefix
    success = profile.success_path
    failure = profile.failure_path

    y_center = 0.5
    x = 0
    step = 1.2

    # Draw consensus prefix
    for i, action in enumerate(prefix):
        ax.add_patch(plt.Rectangle((x, y_center - 0.15), 1, 0.3,
                                    facecolor=DIVERGENCE_COLORS["agree"],
                                    edgecolor="black", lw=0.5))
        label = action[:10]
        ax.text(x + 0.5, y_center, label, ha="center", va="center", fontsize=7)
        if i < len(prefix) - 1:
            ax.annotate("", xy=(x + step, y_center), xytext=(x + 1, y_center),
                        arrowprops=dict(arrowstyle="->", color="black"))
        x += step

    # Branching point
    branch_x = x
    if success or failure:
        # Success branch (top)
        y_top = y_center + 0.4
        for i, action in enumerate(success[len(prefix):len(prefix) + 4]):
            ax.add_patch(plt.Rectangle((branch_x + i * step, y_top - 0.12), 0.9, 0.24,
                                        facecolor="#27ae60", edgecolor="black", lw=0.5, alpha=0.7))
            ax.text(branch_x + i * step + 0.45, y_top, action[:8],
                    ha="center", va="center", fontsize=6)

        # Failure branch (bottom)
        y_bot = y_center - 0.4
        for i, action in enumerate(failure[len(prefix):len(prefix) + 4]):
            ax.add_patch(plt.Rectangle((branch_x + i * step, y_bot - 0.12), 0.9, 0.24,
                                        facecolor="#e74c3c", edgecolor="black", lw=0.5, alpha=0.7))
            ax.text(branch_x + i * step + 0.45, y_bot, action[:8],
                    ha="center", va="center", fontsize=6)

        ax.text(branch_x - 0.3, y_top, "PASS", fontsize=8, color="#27ae60", fontweight="bold")
        ax.text(branch_x - 0.3, y_bot, "FAIL", fontsize=8, color="#e74c3c", fontweight="bold")

    ax.set_xlim(-0.5, max(branch_x + 5 * step, 5))
    ax.set_ylim(-0.2, 1.2)
    ax.set_axis_off()
    ax.set_title(f"Divergence Tree — Task {profile.task_id}")

    return fig
