#!/usr/bin/env python3
"""Build an explicit shift-level dataset from filtered tau2 task results."""

from __future__ import annotations

import csv
import json
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
    REPO_ROOT / "data/processed/tau2_shift_level_dataset_20260714.csv"
)
MD_OUTPUT_PATH = REPO_ROOT / "notes/tau2_shift_level_dataset_20260714.md"

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
    "source_task_ids",
    "target_task_ids",
    "source_domains",
    "target_domains",
    "source_mean_num_messages",
    "target_mean_num_messages",
    "source_mean_num_tool_calls",
    "target_mean_num_tool_calls",
    "source_mean_expected_action_count",
    "target_mean_expected_action_count",
    "source_mean_expected_write_action_count",
    "target_mean_expected_write_action_count",
]


@dataclass(frozen=True)
class TaskData:
    y: np.ndarray
    domains: np.ndarray
    task_ids: list[str]
    num_messages: np.ndarray
    num_tool_calls: np.ndarray
    expected_action_count: np.ndarray
    expected_read_action_count: np.ndarray
    expected_write_action_count: np.ndarray


def round_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    return round(value, 6)


def success_rate(positive: int, count: int) -> float:
    if count == 0:
        return float("nan")
    return positive / count


def mean_for(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.sum()):
        return float("nan")
    return float(np.mean(values[mask].astype(float)))


def list_for(values: list[str] | np.ndarray, mask: np.ndarray) -> list[str]:
    return [str(value) for value, keep in zip(values, mask, strict=True) if keep]


def domains_for(domains: np.ndarray, mask: np.ndarray) -> list[str]:
    selected = list_for(domains, mask)
    return sorted(set(selected))


def task_ids_for(data: TaskData, mask: np.ndarray) -> list[str]:
    return [
        f"{domain}:{task_id}"
        for domain, task_id, keep in zip(
            data.domains, data.task_ids, mask, strict=True
        )
        if keep
    ]


def json_list(values: list[str]) -> str:
    return json.dumps(values, separators=(",", ":"))


def format_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


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


def validate_lengths(x: np.ndarray, y: np.ndarray, metadata: list[Any]) -> None:
    metadata_count = len(metadata)
    if len(x) != len(y) or metadata_count != len(y):
        raise ValueError(
            "Mismatched row counts: "
            f"len(X)={len(x)}, len(y)={len(y)}, len(metadata)={metadata_count}"
        )


def load_task_data(path: Path) -> TaskData:
    with path.open("rb") as f:
        dataset = pickle.load(f)

    x = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"])
    metadata = list(dataset["metadata"])
    feature_names = list(dataset["feature_names"])
    validate_lengths(x, y, metadata)

    task_ids = [str(row["task_id"]) for row in metadata]

    return TaskData(
        y=y,
        domains=values_for(metadata, feature_names, x, "domain").astype(str),
        task_ids=task_ids,
        num_messages=values_for(metadata, feature_names, x, "num_messages").astype(
            float
        ),
        num_tool_calls=values_for(
            metadata, feature_names, x, "num_tool_calls"
        ).astype(float),
        expected_action_count=values_for(
            metadata, feature_names, x, "expected_action_count"
        ).astype(float),
        expected_read_action_count=values_for(
            metadata, feature_names, x, "expected_read_action_count"
        ).astype(float),
        expected_write_action_count=values_for(
            metadata, feature_names, x, "expected_write_action_count"
        ).astype(float),
    )


def build_row(
    *,
    data: TaskData,
    shift_name: str,
    shift_type: str,
    source_group: str,
    target_group: str,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, str | int | float | bool]:
    source_n = int(source_mask.sum())
    target_n = int(target_mask.sum())
    source_positive = int(((data.y == 1) & source_mask).sum())
    target_positive = int(((data.y == 1) & target_mask).sum())
    source_success_rate = success_rate(source_positive, source_n)
    target_success_rate = success_rate(target_positive, target_n)
    drop_pp = 100.0 * (source_success_rate - target_success_rate)

    return {
        "shift_name": shift_name,
        "shift_type": shift_type,
        "source_group": source_group,
        "target_group": target_group,
        "source_n": source_n,
        "target_n": target_n,
        "source_positive": source_positive,
        "target_positive": target_positive,
        "source_success_rate": round_float(source_success_rate),
        "target_success_rate": round_float(target_success_rate),
        "drop_pp": round_float(drop_pp),
        "harmful_candidate": str(
            not math.isnan(drop_pp) and drop_pp > 10.0
        ).lower(),
        "source_task_ids": json_list(task_ids_for(data, source_mask)),
        "target_task_ids": json_list(task_ids_for(data, target_mask)),
        "source_domains": json_list(domains_for(data.domains, source_mask)),
        "target_domains": json_list(domains_for(data.domains, target_mask)),
        "source_mean_num_messages": round_float(
            mean_for(data.num_messages, source_mask)
        ),
        "target_mean_num_messages": round_float(
            mean_for(data.num_messages, target_mask)
        ),
        "source_mean_num_tool_calls": round_float(
            mean_for(data.num_tool_calls, source_mask)
        ),
        "target_mean_num_tool_calls": round_float(
            mean_for(data.num_tool_calls, target_mask)
        ),
        "source_mean_expected_action_count": round_float(
            mean_for(data.expected_action_count, source_mask)
        ),
        "target_mean_expected_action_count": round_float(
            mean_for(data.expected_action_count, target_mask)
        ),
        "source_mean_expected_write_action_count": round_float(
            mean_for(data.expected_write_action_count, source_mask)
        ),
        "target_mean_expected_write_action_count": round_float(
            mean_for(data.expected_write_action_count, target_mask)
        ),
    }


def median_masks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.median(values.astype(float)))
    return values <= threshold, values > threshold, threshold


def build_rows(
    data: TaskData,
) -> tuple[list[dict[str, str | int | float | bool]], dict[str, float]]:
    short_messages, long_messages, message_median = median_masks(data.num_messages)
    few_tools, many_tools, tool_median = median_masks(data.num_tool_calls)
    low_actions, high_actions, action_median = median_masks(
        data.expected_action_count
    )
    low_reads, high_reads, read_median = median_masks(
        data.expected_read_action_count
    )

    rows = [
        build_row(
            data=data,
            shift_name="retail -> airline",
            shift_type="domain",
            source_group="retail",
            target_group="airline",
            source_mask=data.domains == "retail",
            target_mask=data.domains == "airline",
        ),
        build_row(
            data=data,
            shift_name="no expected write actions -> expected write actions > 0",
            shift_type="action_type",
            source_group="no expected write actions",
            target_group="expected write actions > 0",
            source_mask=data.expected_write_action_count == 0,
            target_mask=data.expected_write_action_count > 0,
        ),
        build_row(
            data=data,
            shift_name="zero or one expected write -> two or more expected writes",
            shift_type="action_type",
            source_group="zero or one expected write",
            target_group="two or more expected writes",
            source_mask=data.expected_write_action_count <= 1,
            target_mask=data.expected_write_action_count >= 2,
        ),
        build_row(
            data=data,
            shift_name="short messages -> long messages",
            shift_type="trajectory_complexity",
            source_group="short messages",
            target_group="long messages",
            source_mask=short_messages,
            target_mask=long_messages,
        ),
        build_row(
            data=data,
            shift_name="few tool calls -> many tool calls",
            shift_type="tool_complexity",
            source_group="few tool calls",
            target_group="many tool calls",
            source_mask=few_tools,
            target_mask=many_tools,
        ),
        build_row(
            data=data,
            shift_name="low expected actions -> high expected actions",
            shift_type="action_count",
            source_group="low expected actions",
            target_group="high expected actions",
            source_mask=low_actions,
            target_mask=high_actions,
        ),
        build_row(
            data=data,
            shift_name="low expected read actions -> high expected read actions",
            shift_type="read_action_count",
            source_group="low expected read actions",
            target_group="high expected read actions",
            source_mask=low_reads,
            target_mask=high_reads,
        ),
    ]
    medians = {
        "num_messages": message_median,
        "num_tool_calls": tool_median,
        "expected_action_count": action_median,
        "expected_read_action_count": read_median,
    }
    return rows, medians


def render_plain_table(rows: list[dict[str, str | int | float | bool]]) -> str:
    widths = [
        max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in OUTPUT_COLUMNS
    ]
    lines = [
        " | ".join(
            column.ljust(widths[index])
            for index, column in enumerate(OUTPUT_COLUMNS)
        ),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            " | ".join(
                format_value(row[column]).ljust(widths[index])
                for index, column in enumerate(OUTPUT_COLUMNS)
            )
        )
    return "\n".join(lines)


def render_markdown_table(rows: list[dict[str, str | int | float | bool]]) -> str:
    numeric_columns = {
        column
        for column in OUTPUT_COLUMNS
        if column.endswith(("_n", "_positive"))
        or column.endswith(("_rate", "_messages", "_calls", "_count"))
        or column == "drop_pp"
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
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(format_value(row[column]))
                for column in OUTPUT_COLUMNS
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown_report(
    *,
    rows: list[dict[str, str | int | float | bool]],
    medians: dict[str, float],
    input_path: Path,
    total_n: int,
    total_positive: int,
    overall_success_rate: float,
) -> str:
    median_lines = "\n".join(
        f"- Median `{name}`: {value:g}" for name, value in medians.items()
    )
    return "\n\n".join(
        [
            "# tau2 Shift-Level Dataset - 2026-07-14",
            "## Dataset\n"
            f"- Input: `{input_path}`\n"
            f"- Total N: {total_n}\n"
            f"- Positive count (`y == 1`): {total_positive}\n"
            f"- Overall success rate: {overall_success_rate:.6f}\n"
            "- Rates are proportions. `drop_pp` is percentage points, computed as "
            "`100 * (source_success_rate - target_success_rate)`.\n"
            "- `harmful_candidate` is true when `drop_pp > 10`.\n"
            "- Task IDs are domain-qualified as `domain:task_id` because raw "
            "`task_id` values repeat across domains.",
            "## Median Splits\n" + median_lines,
            "## Shift Rows\n" + render_markdown_table(rows),
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    data = load_task_data(INPUT_PATH)
    rows, medians = build_rows(data)
    total_positive = int((data.y == 1).sum())
    overall_success_rate = success_rate(total_positive, len(data.y))

    write_csv(CSV_OUTPUT_PATH, rows)
    write_markdown(
        MD_OUTPUT_PATH,
        render_markdown_report(
            rows=rows,
            medians=medians,
            input_path=INPUT_PATH,
            total_n=len(data.y),
            total_positive=total_positive,
            overall_success_rate=overall_success_rate,
        ),
    )

    print(render_plain_table(rows))
    print(f"\nWrote CSV: {CSV_OUTPUT_PATH}")
    print(f"Wrote Markdown: {MD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
