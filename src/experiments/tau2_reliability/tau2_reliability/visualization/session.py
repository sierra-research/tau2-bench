"""Session-level (single trace) visualizations.

Deep-dive views into individual simulations and cross-trial comparisons.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from tau2_reliability.models import TaskTrialData
from tau2_reliability.visualization.styles import (
    DIVERGENCE_COLORS,
    apply_publication_style,
)


def plot_action_sequence_alignment(
    task_data: TaskTrialData,
    figsize: tuple[float, float] = (14, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """Multiple sequence alignment view across K trials.

    Rows = trials, columns = action step index.
    Cells colored by action name, showing where trials agree/diverge.
    """
    apply_publication_style()
    sequences = task_data.action_sequences
    outcomes = task_data.outcomes
    max_len = max((len(s) for s in sequences), default=0)

    if max_len == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No actions recorded", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Assign colors to unique actions
    all_actions = sorted(set(a for seq in sequences for a in seq))
    cmap = plt.cm.get_cmap("tab20", max(len(all_actions), 1))
    action_color = {a: cmap(i) for i, a in enumerate(all_actions)}

    n_trials = len(sequences)
    fig, ax = plt.subplots(figsize=figsize)

    for row, (seq, outcome) in enumerate(zip(sequences, outcomes)):
        for col, action in enumerate(seq):
            color = action_color[action]
            rect = plt.Rectangle(
                (col, n_trials - 1 - row), 1, 0.8, facecolor=color, edgecolor="white", lw=0.5
            )
            ax.add_patch(rect)
            # Abbreviate long names
            label = action[:12] + ".." if len(action) > 14 else action
            ax.text(
                col + 0.5, n_trials - 1 - row + 0.4, label,
                ha="center", va="center", fontsize=7, rotation=45,
            )

    # Highlight divergence columns
    for col in range(max_len):
        actions_at_col = set()
        for seq in sequences:
            if col < len(seq):
                actions_at_col.add(seq[col])
        if len(actions_at_col) > 1:
            rect = plt.Rectangle(
                (col, -0.15), 1, n_trials + 0.1,
                facecolor="none", edgecolor=DIVERGENCE_COLORS["diverge"],
                lw=2, linestyle="--",
            )
            ax.add_patch(rect)

    ax.set_xlim(0, max_len)
    ax.set_ylim(-0.2, n_trials)
    ax.set_xlabel("Action Step")
    ax.set_ylabel("Trial")
    ax.set_yticks([n_trials - 1 - i + 0.4 for i in range(n_trials)])
    outcome_labels = ["PASS" if o else "FAIL" for o in outcomes]
    ax.set_yticklabels([f"Trial {i} ({outcome_labels[i]})" for i in range(n_trials)])
    ax.set_title(title or f"Action Sequence Alignment — Task {task_data.task_id}")

    return fig


def plot_trace_comparison(
    task_data: TaskTrialData,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Side-by-side comparison of action sequences across trials.

    Shows agreement (green) and divergence (red) per action step.
    """
    apply_publication_style()
    sequences = task_data.action_sequences
    outcomes = task_data.outcomes
    max_len = max((len(s) for s in sequences), default=0)
    n_trials = len(sequences)

    if max_len == 0 or n_trials == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Compute agreement at each step
    agreement = []
    for col in range(max_len):
        actions_at_col = []
        for seq in sequences:
            if col < len(seq):
                actions_at_col.append(seq[col])
        if not actions_at_col:
            agreement.append(0.0)
        else:
            from collections import Counter
            counts = Counter(actions_at_col)
            agreement.append(counts.most_common(1)[0][1] / n_trials)

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)

    # Top: action grid
    ax = axes[0]
    for row, (seq, outcome) in enumerate(zip(sequences, outcomes)):
        for col, action in enumerate(seq):
            color = DIVERGENCE_COLORS["agree"] if agreement[col] >= 0.8 else DIVERGENCE_COLORS["diverge"]
            ax.barh(
                n_trials - 1 - row, 1, left=col, height=0.7,
                color=color, edgecolor="white", alpha=0.7,
            )
            label = action[:10]
            ax.text(col + 0.5, n_trials - 1 - row, label, ha="center", va="center", fontsize=6)

    outcome_labels = ["P" if o else "F" for o in outcomes]
    ax.set_yticks(range(n_trials))
    ax.set_yticklabels([f"T{n_trials - 1 - i} ({outcome_labels[n_trials - 1 - i]})" for i in range(n_trials)])
    ax.set_ylabel("Trial")
    ax.set_title(f"Trace Comparison — Task {task_data.task_id}")

    # Bottom: agreement bar
    axes[1].bar(
        range(max_len), agreement,
        color=[DIVERGENCE_COLORS["agree"] if a >= 0.8 else DIVERGENCE_COLORS["diverge"] for a in agreement],
        alpha=0.8,
    )
    axes[1].set_ylabel("Agreement")
    axes[1].set_xlabel("Action Step")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.8, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_resource_distribution(
    task_data: TaskTrialData,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Distribution of cost, duration, and action count across trials."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, values, label in zip(
        axes,
        [task_data.costs, task_data.durations, [float(n) for n in task_data.num_actions]],
        ["Cost ($)", "Duration (s)", "Action Count"],
    ):
        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        outcomes = task_data.outcomes
        colors = [DIVERGENCE_COLORS["agree"] if o else DIVERGENCE_COLORS["diverge"] for o in outcomes]
        ax.bar(range(len(values)), values, color=colors, alpha=0.8)
        ax.set_xlabel("Trial")
        ax.set_ylabel(label)
        if values:
            mean_val = sum(values) / len(values)
            ax.axhline(mean_val, color="black", linestyle="--", alpha=0.5, label=f"mean={mean_val:.3f}")
            ax.legend(fontsize=7)

    fig.suptitle(f"Resource Distribution — Task {task_data.task_id}")
    plt.tight_layout()
    return fig
