#!/usr/bin/env python3
"""Draw a stacked percentage bar chart from the failure-coding worksheet.

Reads failures_to_code.csv, uses final_code where set and proposed_code
otherwise, and writes a PDF (vector) and PNG. Category order is F1-F6.

Usage:
    python failure_distribution_chart.py \
        --csv failure_distribution/failures_to_code.csv \
        --out ../latex/media/failure_distribution
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODES = ["F1", "F2", "F3", "F4", "F5", "F6"]
COLORS = {
    "F1": "#2b6cb0",
    "F2": "#63a4d4",
    "F3": "#a6cee3",
    "F4": "#c0392b",
    "F5": "#e67e22",
    "F6": "#7b6a94",
}
ORDER = [
    "Qwen 3.5 9B",
    "GPT-OSS 20B",
    "DeepSeek chat",
    "Qwen 3.5 4B",
    "Phi-4-mini 3.8B",
]


def text_color(hex_color: str) -> str:
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "black" if luminance > 0.6 else "white"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("failure_distribution/failures_to_code.csv"))
    parser.add_argument("--out", type=Path, default=Path("failure_distribution/failure_distribution"))
    args = parser.parse_args()

    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    with args.csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("final_code") or "").strip().upper() or (row.get("proposed_code") or "").strip().upper()
            if code not in CODES:
                continue
            model = row["model"]
            counts[model][code] += 1
            totals[model] += 1

    models = [m for m in ORDER if m in totals] + sorted(m for m in totals if m not in ORDER)
    models = [m for m in models if totals[m]]
    present = [c for c in CODES if any(counts[m][c] for m in models)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
        }
    )
    height = 0.6 * len(models) + 1.35
    fig, ax = plt.subplots(figsize=(7.0, height))

    left = [0.0] * len(models)
    for code in present:
        values = [100 * counts[m][code] / totals[m] for m in models]
        bars = ax.barh(
            models,
            values,
            left=left,
            color=COLORS[code],
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
            label=code,
        )
        for bar, value in zip(bars, values):
            if value >= 7:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color=text_color(COLORS[code]),
                    fontsize=8,
                )
        left = [x + v for x, v in zip(left, values)]

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Share of failed trials (%)")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} (n={totals[m]})" for m in models])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in present],
        labels=present,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(present),
        frameon=False,
        fontsize=9,
        handlelength=1.0,
        columnspacing=1.4,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out.with_suffix(".pdf")
    png_path = args.out.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
