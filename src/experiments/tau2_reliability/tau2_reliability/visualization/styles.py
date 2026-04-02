"""Shared visual language for reliability plots.

Provider color palette.
"""

from __future__ import annotations

import matplotlib as mpl

# Provider color palette
PROVIDER_COLORS = {
    "openai": "#10a37f",
    "anthropic": "#d4a574",
    "google": "#4285f4",
    "meta": "#0668E1",
    "deepseek": "#5B6EE1",
    "open_source": "#9b59b6",
    "default": "#7f8c8d",
}

PROVIDER_MARKERS = {
    "openai": "o",
    "anthropic": "^",
    "google": "D",
    "meta": "s",
    "deepseek": "P",
    "open_source": "X",
    "default": "o",
}

# Metric dimension colors
DIMENSION_COLORS = {
    "consistency": "#2ecc71",
    "predictability": "#3498db",
    "robustness": "#e67e22",
    "safety": "#e74c3c",
    "overall": "#2c3e50",
}

# Task reliability class colors
TAXONOMY_COLORS = {
    "stable_pass": "#27ae60",
    "stable_fail": "#c0392b",
    "bimodal": "#f39c12",
    "fragile": "#8e44ad",
    "model_discriminating": "#2980b9",
}

# Tool type colors
TOOL_TYPE_COLORS = {
    "READ": "#3498db",
    "WRITE": "#e74c3c",
    "THINK": "#9b59b6",
    "GENERIC": "#95a5a6",
    "UNKNOWN": "#bdc3c7",
}

# Role colors for session timelines
ROLE_COLORS = {
    "assistant": "#3498db",
    "user": "#27ae60",
    "tool": "#95a5a6",
    "system": "#7f8c8d",
}

# Divergence colors
DIVERGENCE_COLORS = {
    "agree": "#27ae60",
    "diverge": "#e74c3c",
    "gap": "#ecf0f1",
}


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def apply_interactive_style() -> None:
    """Apply clean interactive matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_metric_color(metric_name: str) -> str:
    """Get color for a specific metric based on its dimension."""
    if metric_name.startswith("c_"):
        return DIMENSION_COLORS["consistency"]
    elif metric_name.startswith("p_"):
        return DIMENSION_COLORS["predictability"]
    elif metric_name.startswith("r_") and metric_name != "r_con":
        return DIMENSION_COLORS["robustness"]
    elif metric_name.startswith("s_"):
        return DIMENSION_COLORS["safety"]
    return DIMENSION_COLORS["overall"]
