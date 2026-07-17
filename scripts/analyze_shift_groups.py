#!/usr/bin/env python3
"""Analyze candidate distribution-shift groups in a tau2 L2T pickle."""

from __future__ import annotations

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
OUTPUT_PATH = REPO_ROOT / "notes/shift_group_analysis_20260714.md"


@dataclass(frozen=True)
class GroupStats:
    name: str
    count: int
    positives: int
    success_rate: float


@dataclass(frozen=True)
class BinaryShift:
    name: str
    source: GroupStats
    target: GroupStats

    @property
    def drop(self) -> float:
        return self.source.success_rate - self.target.success_rate


def success_rate(positives: int, count: int) -> float:
    if count == 0:
        return float("nan")
    return positives / count


def fmt_rate(rate: float) -> str:
    if np.isnan(rate):
        return "n/a"
    return f"{rate:.3f} ({rate * 100:.1f}%)"


def fmt_pp(drop: float) -> str:
    if np.isnan(drop):
        return "n/a"
    return f"{drop * 100:.1f} pp"


def stats_for(name: str, mask: np.ndarray, y: np.ndarray) -> GroupStats:
    count = int(mask.sum())
    positives = int(y[mask].sum())
    return GroupStats(
        name=name,
        count=count,
        positives=positives,
        success_rate=success_rate(positives, count),
    )


def print_group(stats: GroupStats, label_width: int = 28) -> None:
    print(
        f"{stats.name:<{label_width}} "
        f"N={stats.count:>3}  "
        f"positive={stats.positives:>3}  "
        f"success_rate={fmt_rate(stats.success_rate)}"
    )


def metadata_values(
    metadata: list[dict[str, Any]],
    feature_names: list[str],
    X: np.ndarray,
    field_name: str,
) -> np.ndarray:
    if metadata and all(field_name in row for row in metadata):
        return np.asarray([row[field_name] for row in metadata])

    if field_name in feature_names:
        index = feature_names.index(field_name)
        return X[:, index]

    raise KeyError(f"Missing field in metadata and features: {field_name}")


def median_binary_shift(
    shift_name: str,
    source_name: str,
    target_name: str,
    values: np.ndarray,
    y: np.ndarray,
) -> tuple[float, GroupStats, GroupStats, BinaryShift]:
    numeric_values = values.astype(float)
    threshold = float(np.median(numeric_values))
    source = stats_for(source_name, numeric_values <= threshold, y)
    target = stats_for(target_name, numeric_values > threshold, y)
    return threshold, source, target, BinaryShift(shift_name, source, target)


def print_median_section(
    title: str,
    threshold_label: str,
    threshold: float,
    source: GroupStats,
    target: GroupStats,
    drop_label: str,
) -> None:
    print(f"\n{title}")
    print(f"{threshold_label}: {threshold:g}")
    print_group(source)
    print_group(target)
    print(f"{drop_label}: {fmt_pp(source.success_rate - target.success_rate)}")


def print_ranking(shifts: list[BinaryShift]) -> None:
    print("\n9. Candidate harmful-shift ranking")
    ranked = sorted(shifts, key=lambda shift: shift.drop, reverse=True)
    headers = [
        "shift",
        "source group",
        "target group",
        "source N",
        "target N",
        "source success",
        "target success",
        "drop",
    ]
    rows = [
        [
            shift.name,
            shift.source.name,
            shift.target.name,
            str(shift.source.count),
            str(shift.target.count),
            f"{shift.source.success_rate * 100:.1f}%"
            if not np.isnan(shift.source.success_rate)
            else "n/a",
            f"{shift.target.success_rate * 100:.1f}%"
            if not np.isnan(shift.target.success_rate)
            else "n/a",
            fmt_pp(shift.drop),
        ]
        for shift in ranked
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def markdown_rate(stats: GroupStats) -> str:
    if np.isnan(stats.success_rate):
        return "n/a"
    return f"{stats.success_rate * 100:.1f}%"


def markdown_group_table(groups: list[GroupStats]) -> str:
    lines = [
        "| Group | N | Positive | Success rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group.name} | {group.count} | {group.positives} | "
            f"{markdown_rate(group)} |"
        )
    return "\n".join(lines)


def markdown_ranking_table(shifts: list[BinaryShift]) -> str:
    ranked = sorted(shifts, key=lambda shift: shift.drop, reverse=True)
    lines = [
        "| Rank | Shift | Source group | Target group | Source N | Target N | "
        "Source success | Target success | Drop |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, shift in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {shift.name} | {shift.source.name} | "
            f"{shift.target.name} | {shift.source.count} | {shift.target.count} | "
            f"{markdown_rate(shift.source)} | {markdown_rate(shift.target)} | "
            f"{fmt_pp(shift.drop)} |"
        )
    return "\n".join(lines)


def render_report(
    *,
    x_shape: tuple[int, ...],
    traj_summary: str,
    feature_names: list[str],
    metadata_fields: list[str],
    total_count: int,
    total_positive: int,
    overall_rate: float,
    domain_shift: BinaryShift,
    write_required_shift: BinaryShift,
    write_count_groups: list[GroupStats],
    message_length_threshold: float,
    message_length_shift: BinaryShift,
    tool_call_threshold: float,
    tool_call_shift: BinaryShift,
    shifts: list[BinaryShift],
) -> str:
    ranked = sorted(shifts, key=lambda shift: shift.drop, reverse=True)
    top_shift = ranked[0]
    interpretation = (
        f"The largest candidate harmful shift is `{top_shift.name}`, where success "
        f"falls from {markdown_rate(top_shift.source)} in `{top_shift.source.name}` "
        f"to {markdown_rate(top_shift.target)} in `{top_shift.target.name}` "
        f"({fmt_pp(top_shift.drop)}). These comparisons are descriptive groupings "
        "over the existing dataset and should be treated as candidate shift signals "
        "rather than causal explanations."
    )

    return "\n\n".join(
        [
            "# Shift Group Analysis - 2026-07-14",
            "## 1. Dataset summary\n"
            f"- Input path: `{INPUT_PATH}`\n"
            f"- Total N: {total_count}\n"
            f"- X shape: `{x_shape}`\n"
            f"- {traj_summary}\n"
            f"- Feature names: `{feature_names}`\n"
            f"- Metadata fields: `{metadata_fields}`\n"
            f"- Total positive y count: {total_positive}\n"
            f"- Overall success rate: {fmt_rate(overall_rate)}",
            "## 2. Success rates by domain\n"
            f"{markdown_group_table([domain_shift.source, domain_shift.target])}\n\n"
            f"Retail minus airline drop: {fmt_pp(domain_shift.drop)}",
            "## 3. Success rates by write requirement\n"
            f"{markdown_group_table([write_required_shift.source, write_required_shift.target])}\n\n"
            f"No-write minus write-required drop: {fmt_pp(write_required_shift.drop)}\n\n"
            "Expected write count detail:\n\n"
            f"{markdown_group_table(write_count_groups)}",
            "## 4. Success rates by message length\n"
            f"Threshold/median: {message_length_threshold:g}\n\n"
            f"{markdown_group_table([message_length_shift.source, message_length_shift.target])}\n\n"
            f"Short minus long drop: {fmt_pp(message_length_shift.drop)}",
            "## 5. Success rates by tool-call count\n"
            f"Threshold/median: {tool_call_threshold:g}\n\n"
            f"{markdown_group_table([tool_call_shift.source, tool_call_shift.target])}\n\n"
            f"Few minus many drop: {fmt_pp(tool_call_shift.drop)}",
            "## 6. Candidate harmful-shift ranking table\n"
            f"{markdown_ranking_table(shifts)}",
            f"## 7. Short interpretation\n{interpretation}",
            "",
        ]
    )


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    with INPUT_PATH.open("rb") as f:
        dataset = pickle.load(f)

    X = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"]).astype(int)
    traj = dataset["traj"]
    metadata = list(dataset["metadata"])
    feature_names = list(dataset["feature_names"])

    if len(X) != len(y) or len(metadata) != len(y):
        raise ValueError(
            "Mismatched row counts: "
            f"len(X)={len(X)}, len(y)={len(y)}, len(metadata)={len(metadata)}"
        )

    metadata_fields = sorted({key for row in metadata for key in row})
    total_positive = int(y.sum())
    overall_rate = success_rate(total_positive, len(y))

    print(f"input path: {INPUT_PATH}")
    print("\n1. Basic dataset summary")
    print(f"total N: {len(y)}")
    print(f"X shape: {X.shape}")
    if isinstance(traj, dict) and "s" in traj:
        print(f"traj['s'] shape: {np.asarray(traj['s']).shape}")
    else:
        print(f"traj type: {type(traj).__name__}")
    print(f"feature names: {feature_names}")
    print(f"metadata fields: {metadata_fields}")
    print(f"total positive y count: {total_positive}")
    print(f"overall success rate: {fmt_rate(overall_rate)}")
    if isinstance(traj, dict) and "s" in traj:
        traj_summary = f"traj['s'] shape: `{np.asarray(traj['s']).shape}`"
    else:
        traj_summary = f"traj type: `{type(traj).__name__}`"

    shifts: list[BinaryShift] = []

    domains = metadata_values(metadata, feature_names, X, "domain").astype(str)
    retail = stats_for("retail", domains == "retail", y)
    airline = stats_for("airline", domains == "airline", y)
    domain_shift = BinaryShift("retail -> airline", retail, airline)
    shifts.append(domain_shift)

    print("\n2. Domain shift")
    print_group(retail)
    print_group(airline)
    print(f"retail minus airline drop: {fmt_pp(domain_shift.drop)}")

    expected_write = metadata_values(
        metadata, feature_names, X, "expected_write_action_count"
    ).astype(float)
    no_write = stats_for("no expected write actions", expected_write == 0, y)
    write_required = stats_for("expected write actions > 0", expected_write > 0, y)
    write_required_shift = BinaryShift(
        "no-write -> write-required", no_write, write_required
    )
    shifts.append(write_required_shift)

    print("\n3. Write-required shift")
    print_group(no_write)
    print_group(write_required)
    print(f"no-write minus write-required drop: {fmt_pp(write_required_shift.drop)}")

    expected_actions = metadata_values(
        metadata, feature_names, X, "expected_action_count"
    )
    threshold, low_actions, high_actions, action_count_shift = median_binary_shift(
        "low expected actions -> high expected actions",
        "low expected actions",
        "high expected actions",
        expected_actions,
        y,
    )
    shifts.append(action_count_shift)
    print_median_section(
        "\n4. Expected action count shift".strip(),
        "threshold/median",
        threshold,
        low_actions,
        high_actions,
        "low minus high drop",
    )

    expected_reads = metadata_values(
        metadata, feature_names, X, "expected_read_action_count"
    )
    threshold, low_reads, high_reads, read_count_shift = median_binary_shift(
        "low read count -> high read count",
        "low read count",
        "high read count",
        expected_reads,
        y,
    )
    shifts.append(read_count_shift)
    print_median_section(
        "5. Expected read count shift",
        "threshold/median",
        threshold,
        low_reads,
        high_reads,
        "low minus high drop",
    )

    zero_write = stats_for("zero write", expected_write == 0, y)
    one_write = stats_for("one write", expected_write == 1, y)
    two_plus_write = stats_for("two or more writes", expected_write >= 2, y)

    print("\n6. Expected write count shift")
    print_group(zero_write)
    print_group(one_write)
    print_group(two_plus_write)

    num_messages = metadata_values(metadata, feature_names, X, "num_messages")
    threshold, short_messages, long_messages, message_length_shift = median_binary_shift(
        "short messages -> long messages",
        "short messages",
        "long messages",
        num_messages,
        y,
    )
    shifts.append(message_length_shift)
    message_length_threshold = threshold
    print_median_section(
        "7. Trajectory/message length shift",
        "threshold/median",
        threshold,
        short_messages,
        long_messages,
        "short minus long drop",
    )

    num_tool_calls = metadata_values(metadata, feature_names, X, "num_tool_calls")
    threshold, few_tool_calls, many_tool_calls, tool_call_shift = median_binary_shift(
        "few tool calls -> many tool calls",
        "few tool calls",
        "many tool calls",
        num_tool_calls,
        y,
    )
    shifts.append(tool_call_shift)
    tool_call_threshold = threshold
    print_median_section(
        "8. Tool-call count shift",
        "threshold/median",
        threshold,
        few_tool_calls,
        many_tool_calls,
        "few minus many drop",
    )

    print_ranking(shifts)

    report = render_report(
        x_shape=X.shape,
        traj_summary=traj_summary,
        feature_names=feature_names,
        metadata_fields=metadata_fields,
        total_count=len(y),
        total_positive=total_positive,
        overall_rate=overall_rate,
        domain_shift=domain_shift,
        write_required_shift=write_required_shift,
        write_count_groups=[zero_write, one_write, two_plus_write],
        message_length_threshold=message_length_threshold,
        message_length_shift=message_length_shift,
        tool_call_threshold=tool_call_threshold,
        tool_call_shift=tool_call_shift,
        shifts=shifts,
    )
    write_report(OUTPUT_PATH, report)


if __name__ == "__main__":
    main()
