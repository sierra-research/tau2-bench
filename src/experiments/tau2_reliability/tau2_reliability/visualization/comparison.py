"""Cross-model and cross-domain comparison visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tau2_reliability.models import ReliabilityReport
from tau2_reliability.visualization.styles import (
    apply_publication_style,
)


def plot_radar_comparison(
    reports: list[ReliabilityReport],
    figsize: tuple[float, float] = (8, 8),
) -> plt.Figure:
    """Radar chart comparing multiple models on reliability dimensions."""
    apply_publication_style()

    dimensions = ["R_Con", "R_Pred", "R_Rob"]
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for report in reports:
        values = [
            report.r_con or 0,
            report.r_pred or 0,
            report.r_rob or 0,
        ]
        values += values[:1]
        label = report.agent_model or "unknown"
        ax.plot(angles, values, "-o", linewidth=2, label=label, markersize=6)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Profile Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return fig


def plot_reliability_vs_accuracy(
    reports: list[ReliabilityReport],
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Scatter: overall reliability vs accuracy for each model."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for report in reports:
        r = report.r_overall or report.r_con
        acc = report.accuracy
        if r is None:
            continue
        ax.scatter(
            acc, r, s=100, edgecolors="black", linewidths=0.5, alpha=0.8, zorder=3,
        )
        ax.annotate(
            report.agent_model or "", (acc, r),
            fontsize=8, ha="left", va="bottom", alpha=0.7,
        )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Reliability (R)")
    ax.set_title("Reliability vs Accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2, label="y=x")
    ax.legend()

    return fig


def plot_dimension_heatmap(
    reports: list[ReliabilityReport],
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Heatmap: models (rows) x dimensions (columns)."""
    apply_publication_style()
    import pandas as pd
    import seaborn as sns

    data = {}
    for report in reports:
        name = report.agent_model or "unknown"
        data[name] = {
            "Accuracy": report.accuracy,
            "R_Con": report.r_con or float("nan"),
            "R_Pred": report.r_pred or float("nan"),
            "R_Rob": report.r_rob or float("nan"),
            "R_Overall": report.r_overall or float("nan"),
        }

    if not data:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Model Comparison — Reliability Dimensions")
    plt.tight_layout()
    return fig
