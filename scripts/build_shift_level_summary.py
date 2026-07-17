#!/usr/bin/env python3
"""Build a shift-level success summary from a tau2 task-level L2T pickle."""

from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    REPO_ROOT
    / "data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl"
)
CSV_OUTPUT_PATH = (
    REPO_ROOT / "data/processed/tau2_shift_level_summary_20260714.csv"
)
MD_OUTPUT_PATH = REPO_ROOT / "notes/tau2_shift_level_summary_20260714.md"

OUTPUT_COLUMNS = [
    "shift_name",
    "shift_type",
    "source_group",
    "target_group",
    "source_n",
    "target_n",
    "source_positive",
    "target_positive",
    "source_success_rate",
    "target_success_rate",
    "drop_pp",
    "harmful_candidate",
    "notes",
]


@dataclass(frozen=True)
class GroupStats:
    name: str
    count: int
    positive: int
    success_rate: float


@dataclass(frozen=True)
class ShiftRow:
    shift_name: str
    shift_type: str
    source: GroupStats
    target: GroupStats
    notes: str

    @property
    def drop_pp(self) -> float:
        if math.isnan(self.source.success_rate) or math.isnan(
            self.target.success_rate
        ):
            return float("nan")
        return 100.0 * (self.source.success_rate - self.target.success_rate)

    @property
    def harmful_candidate(self) -> bool:
        return not math.isnan(self.drop_pp) and self.drop_pp > 10.0

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "shift_name": self.shift_name,
            "shift_type": self.shift_type,
            "source_group": self.source.name,
            "target_group": self.target.name,
            "source_n": self.source.count,
            "target_n": self.target.count,
            "source_positive": self.source.positive,
            "target_positive": self.target.positive,
            "source_success_rate": round_float(self.source.success_rate),
            "target_success_rate": round_float(self.target.success_rate),
            "drop_pp": round_float(self.drop_pp),
            "harmful_candidate": str(self.harmful_candidate).lower(),
            "notes": self.notes,
        }


def round_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    return round(value, 6)


def success_rate(positive: int, count: int) -> float:
    if count == 0:
        return float("nan")
    return positive / count


def stats_for(name: str, mask: np.ndarray, y: np.ndarray) -> GroupStats:
    count = int(mask.sum())
    positive = int(((y == 1) & mask).sum())
    return GroupStats(
        name=name,
        count=count,
        positive=positive,
        success_rate=success_rate(positive, count),
    )


def values_for(
    metadata: list[dict[str, Any]],
    feature_names: list[str],
    x: np.ndarray,
    field_name: str,
) -> np.ndarray:
    if metadata and all(field_name in row for row in metadata):
        return np.asarray([row[field_name] for row in metadata])

    if field_name in feature_names:
        index = feature_names.index(field_name)
        return x[:, index]

    raise KeyError(f"Missing field in metadata and features: {field_name}")


def median_shift(
    *,
    shift_name: str,
    shift_type: str,
    source_group: str,
    target_group: str,
    values: np.ndarray,
    value_name: str,
    y: np.ndarray,
) -> ShiftRow:
    numeric_values = values.astype(float)
    threshold = float(np.median(numeric_values))
    source = stats_for(source_group, numeric_values <= threshold, y)
    target = stats_for(target_group, numeric_values > threshold, y)
    return ShiftRow(
        shift_name=shift_name,
        shift_type=shift_type,
        source=source,
        target=target,
        notes=(
            f"median {value_name}={threshold:g}; "
            "source <= median, target > median"
        ),
    )


def sort_key(row: ShiftRow) -> float:
    if math.isnan(row.drop_pp):
        return float("-inf")
    return row.drop_pp


def format_value(value: str | int | float) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_plain_table(rows: list[ShiftRow]) -> str:
    dict_rows = [row.as_dict() for row in rows]
    widths = [
        max(
            len(column),
            *(len(format_value(row[column])) for row in dict_rows),
        )
        for column in OUTPUT_COLUMNS
    ]
    lines = [
        " | ".join(
            column.ljust(widths[index])
            for index, column in enumerate(OUTPUT_COLUMNS)
        ),
        "-+-".join("-" * width for width in widths),
    ]
    for row in dict_rows:
        lines.append(
            " | ".join(
                format_value(row[column]).ljust(widths[index])
                for index, column in enumerate(OUTPUT_COLUMNS)
            )
        )
    return "\n".join(lines)


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def render_markdown_table(rows: list[ShiftRow]) -> str:
    numeric_columns = {
        column
        for column in OUTPUT_COLUMNS
        if column.endswith(("_n", "_positive", "_rate")) or column == "drop_pp"
    }
    lines = [
        "| "
        + " | ".join(markdown_escape(column) for column in OUTPUT_COLUMNS)
        + " |",
        "| "
        + " | ".join(
            "---:" if column in numeric_columns else "---"
            for column in OUTPUT_COLUMNS
        )
        + " |",
    ]
    for row in rows:
        row_dict = row.as_dict()
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(format_value(row_dict[column]))
                for column in OUTPUT_COLUMNS
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown_report(
    *,
    rows: list[ShiftRow],
    input_path: Path,
    total_n: int,
    total_positive: int,
    overall_success_rate: float,
) -> str:
    return "\n\n".join(
        [
            "# tau2 Shift-Level Summary - 2026-07-14",
            "## Dataset\n"
            f"- Input: `{input_path}`\n"
            f"- Total N: {total_n}\n"
            f"- Positive count (`y == 1`): {total_positive}\n"
            f"- Overall success rate: {overall_success_rate:.6f}\n"
            "- Rates are proportions. `drop_pp` is percentage points, computed as "
            "`100 * (source_success_rate - target_success_rate)`.\n"
            "- `harmful_candidate` is true when `drop_pp > 10`.",
            "## Shift Summary\n" + render_markdown_table(rows),
            "",
        ]
    )


def write_csv(path: Path, rows: list[ShiftRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_markdown(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def validate_lengths(x: np.ndarray, y: np.ndarray, metadata: list[Any]) -> None:
    metadata_count = len(metadata)
    if len(x) != len(y) or metadata_count != len(y):
        raise ValueError(
            "Mismatched row counts: "
            f"len(X)={len(x)}, len(y)={len(y)}, len(metadata)={metadata_count}"
        )


def main() -> None:
    with INPUT_PATH.open("rb") as f:
        dataset = pickle.load(f)

    x = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"])
    metadata = list(dataset["metadata"])
    feature_names = list(dataset["feature_names"])
    validate_lengths(x, y, metadata)

    domains = values_for(metadata, feature_names, x, "domain").astype(str)
    expected_write = values_for(
        metadata, feature_names, x, "expected_write_action_count"
    ).astype(float)
    num_messages = values_for(metadata, feature_names, x, "num_messages")
    num_tool_calls = values_for(metadata, feature_names, x, "num_tool_calls")
    expected_actions = values_for(metadata, feature_names, x, "expected_action_count")
    expected_reads = values_for(
        metadata, feature_names, x, "expected_read_action_count"
    )

    rows = [
        ShiftRow(
            shift_name="retail -> airline",
            shift_type="domain",
            source=stats_for("retail", domains == "retail", y),
            target=stats_for("airline", domains == "airline", y),
            notes="domain comparison",
        ),
        ShiftRow(
            shift_name=(
                "no expected write actions -> expected write actions > 0"
            ),
            shift_type="action_type",
            source=stats_for(
                "no expected write actions",
                expected_write == 0,
                y,
            ),
            target=stats_for(
                "expected write actions > 0",
                expected_write > 0,
                y,
            ),
            notes="source expected_write_action_count == 0; target > 0",
        ),
        median_shift(
            shift_name="short messages -> long messages",
            shift_type="trajectory_complexity",
            source_group="short messages",
            target_group="long messages",
            values=num_messages,
            value_name="num_messages",
            y=y,
        ),
        median_shift(
            shift_name="few tool calls -> many tool calls",
            shift_type="tool_complexity",
            source_group="few tool calls",
            target_group="many tool calls",
            values=num_tool_calls,
            value_name="num_tool_calls",
            y=y,
        ),
        ShiftRow(
            shift_name=(
                "zero or one expected write -> two or more expected writes"
            ),
            shift_type="action_type",
            source=stats_for(
                "zero or one expected write",
                expected_write <= 1,
                y,
            ),
            target=stats_for(
                "two or more expected writes",
                expected_write >= 2,
                y,
            ),
            notes="source expected_write_action_count <= 1; target >= 2",
        ),
        median_shift(
            shift_name="low expected actions -> high expected actions",
            shift_type="action_type",
            source_group="low expected actions",
            target_group="high expected actions",
            values=expected_actions,
            value_name="expected_action_count",
            y=y,
        ),
        median_shift(
            shift_name=(
                "low expected read actions -> high expected read actions"
            ),
            shift_type="action_type",
            source_group="low expected read actions",
            target_group="high expected read actions",
            values=expected_reads,
            value_name="expected_read_action_count",
            y=y,
        ),
    ]

    rows = sorted(rows, key=sort_key, reverse=True)
    total_positive = int((y == 1).sum())
    overall_success_rate = success_rate(total_positive, len(y))

    write_csv(CSV_OUTPUT_PATH, rows)
    write_markdown(
        MD_OUTPUT_PATH,
        render_markdown_report(
            rows=rows,
            input_path=INPUT_PATH,
            total_n=len(y),
            total_positive=total_positive,
            overall_success_rate=overall_success_rate,
        ),
    )

    print(render_plain_table(rows))
    print(f"\nWrote CSV: {CSV_OUTPUT_PATH}")
    print(f"Wrote Markdown: {MD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
