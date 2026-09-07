"""Render the tau-Elicitation compositional-accuracy figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

FIGURE_DIR = Path(__file__).resolve().parent


def render() -> None:
    fields = np.asarray([1, 2, 3])
    task_success = np.asarray([0.77, 0.69, 0.53])
    task_error = np.asarray([0.03, 0.07, 0.10])
    field_success = np.asarray([0.77, 0.81, 0.75])
    field_error = np.asarray([0.03, 0.04, 0.05])
    multiplicative = task_success[0] ** fields

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(3.35, 1.92))

    reference = axis.plot(
        fields,
        multiplicative,
        color="#A8B0BA",
        linewidth=1.2,
        linestyle=(0, (2, 2)),
        marker="o",
        markersize=3.5,
        label="77% per field",
        zorder=1,
    )[0]
    task = axis.errorbar(
        fields,
        task_success,
        yerr=task_error,
        color="#2563EB",
        linewidth=1.8,
        marker="o",
        markersize=4.8,
        capsize=3,
        elinewidth=1.1,
        label="Task Pass@1",
        zorder=3,
    )
    field = axis.errorbar(
        fields,
        field_success,
        yerr=field_error,
        color="#F97316",
        linewidth=1.8,
        marker="o",
        markersize=4.8,
        capsize=3,
        elinewidth=1.1,
        label="Field Pass@1",
        zorder=2,
    )

    labels = (
        (1, 0.77, "77%", (0, 10)),
        (2, 0.69, "69%", (-14, -10)),
        (3, 0.53, "53%", (-14, -10)),
        (2, 0.81, "81%", (16, 0)),
        (3, 0.75, "75%", (0, 12)),
    )
    for x_value, y_value, label, offset in labels:
        axis.annotate(
            label,
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#26313D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.8},
            zorder=5,
        )
    axis.annotate(
        "59%",
        xy=(2, multiplicative[1]),
        xytext=(11, -7),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=9,
        color="#7D8793",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.8},
    )
    axis.annotate(
        "46%",
        xy=(3, multiplicative[2]),
        xytext=(-11, -8),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=9,
        color="#7D8793",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.8},
    )

    axis.set_xlim(0.82, 3.18)
    axis.set_ylim(0.38, 0.91)
    axis.set_xticks(fields)
    axis.set_xlabel("Fields per call", fontweight="bold")
    axis.set_ylabel("Pass@1", fontweight="bold")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", color="#E4E8ED", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#CBD2DA")
        spine.set_linewidth(0.8)
    axis.tick_params(length=0, color="#66717E")
    axis.legend(
        [task, field, reference],
        ["Task", "Field", "77% per field"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.20, top=0.91)
    fig.savefig(FIGURE_DIR / "composition_results.pdf")
    fig.savefig(
        FIGURE_DIR / "composition_results.png",
        dpi=240,
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
