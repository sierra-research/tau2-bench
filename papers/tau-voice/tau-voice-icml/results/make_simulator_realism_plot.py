#!/usr/bin/env python3
"""Generate the simulator-realism rating distribution figure.

Reads the two rater CSVs in papers/tau-voice/reviews/user_realism_annotations/
and produces a grouped stacked-bar chart showing the distribution of 1-4
ratings per dimension, split by rater (anonymized as A / B).
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[2]  # papers/tau-voice/
ANNOT = PAPER_ROOT / "reviews" / "user_realism_annotations"
OUT = Path(__file__).resolve().parent / "simulator_realism_distribution.pdf"

DIMS = [
    ("voice_prosody_quality", "Voice prosody"),
    ("audio_environment_realism", "Audio environment realism"),
    ("turn_taking_naturalness", "Turn-taking naturalness"),
    ("backchannel_naturalness", "Backchannel naturalness"),
    ("interruption_behavior", "Interruption behavior"),
    ("behavioral_plausibility", "Behavioral plausibility"),
]

RATERS = [
    ("tau voice user realism human annotation - michael.csv", "Rater A"),
    ("tau voice user realism human annotation - niko.csv", "Rater B"),
]

# Colorblind-safe Likert palette: 1 (dark red) -> 4 (dark green)
COLORS = {
    1: "#b2182b",  # dark red
    2: "#f4a582",  # light red/orange
    3: "#92c5de",  # light blue
    4: "#2166ac",  # dark blue
}
RATING_LABELS = {
    1: "1: clearly unrealistic",
    2: "2",
    3: "3",
    4: "4: indistinguishable from real",
}


def load_counts(path):
    counts = {dim_key: Counter() for dim_key, _ in DIMS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("completed") != "TRUE":
                continue
            for dim_key, _ in DIMS:
                val = row.get(dim_key, "")
                if val and val not in ("None",):
                    counts[dim_key][int(val)] += 1
    return counts


def main():
    rater_counts = []
    for fname, label in RATERS:
        rater_counts.append((label, load_counts(ANNOT / fname)))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    n_dims = len(DIMS)
    n_raters = len(RATERS)
    bar_height = 0.36
    group_spacing = 1.0
    y_positions = []
    y_labels = []

    for i, (_, dim_label) in enumerate(DIMS):
        center = i * group_spacing
        # y-axis is inverted below to put dimensions top-to-bottom; lower y == top.
        # Place A above B in display order.
        a_y = center - bar_height / 2 - 0.02
        b_y = center + bar_height / 2 + 0.02
        y_positions.append((a_y, b_y))
        y_labels.append(dim_label)

    for r_idx, (rater_label, counts) in enumerate(rater_counts):
        for d_idx, (dim_key, _) in enumerate(DIMS):
            y = y_positions[d_idx][r_idx]
            left = 0
            for rating in (1, 2, 3, 4):
                c = counts[dim_key].get(rating, 0)
                if c == 0:
                    left += 0
                    continue
                ax.barh(
                    y,
                    c,
                    height=bar_height,
                    left=left,
                    color=COLORS[rating],
                    edgecolor="white",
                    linewidth=0.6,
                )
                if c >= 4:
                    ax.text(
                        left + c / 2,
                        y,
                        str(c),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if rating in (1, 4) else "black",
                    )
                left += c

    # Major y-ticks: dimension labels centered on each pair
    centers = [i * group_spacing for i in range(n_dims)]
    ax.set_yticks(centers)
    ax.set_yticklabels(y_labels, fontsize=9)

    # Minor y-ticks: per-rater A/B labels at the individual bar positions
    minor_positions = []
    minor_labels = []
    for a_y, b_y in y_positions:
        minor_positions.extend([a_y, b_y])
        minor_labels.extend(["A", "B"])
    ax.set_yticks(minor_positions, minor=True)
    ax.set_yticklabels(minor_labels, minor=True, fontsize=7)

    # Push dimension labels further left so A/B labels have their own column
    ax.tick_params(axis="y", which="major", pad=22, length=0)
    ax.tick_params(axis="y", which="minor", length=0)

    ax.invert_yaxis()

    ax.set_xlim(0, 60)
    ax.set_xlabel("Number of simulations (n=60 per rater)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in (1, 2, 3, 4)
    ]
    legend_labels = [RATING_LABELS[k] for k in (1, 2, 3, 4)]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    plt.subplots_adjust(left=0.32, right=0.98, top=0.97, bottom=0.22)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # Sanity-check printout
    print("\nPer-rater per-dimension counts (1/2/3/4):")
    for rater_label, counts in rater_counts:
        print(f"  {rater_label}:")
        for dim_key, dim_label in DIMS:
            c = counts[dim_key]
            print(
                f"    {dim_label:30s} "
                f"1={c.get(1,0):2d}  2={c.get(2,0):2d}  3={c.get(3,0):2d}  4={c.get(4,0):2d}"
            )


if __name__ == "__main__":
    main()
