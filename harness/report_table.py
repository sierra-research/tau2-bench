#!/usr/bin/env python3
"""Build the paper task-level table from harness run logs.

Usage:
    python harness/report_table.py --runs harness/runs_v2
    python harness/report_table.py --runs harness/runs_v2 --out task_table.md

Reads every */task_*/trial_*/simulation.json, computes per-task successes,
pass^1/pass^4/pass^8 and failure reasons. Prints Markdown and, with --out,
writes it to a file.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from summarize import SEED, bootstrap_ci, mean_pass_k, pass_k  # noqa: E402

PAPER = {
    "0": ("T1", "Skeptical owner, new lead",
          "Person linked to TechStart; opportunity 'TechStart CRM Evaluation'"),
    "1": ("T2", "Stubborn buyer, trust",
          "Person linked to Enterprise; task 'Trust-building follow-up'"),
    "2": ("T3", "Analytical manager, pilot",
          "Person linked to Growth; 'Growth Pilot' at SCREENING"),
    "3": ("T4", "Evidence-seeking buyer, trial",
          "Person linked to ScaleUp; 'ScaleUp Trial'"),
    "4": ("T5", "Cautious tester, pilot + follow-up",
          "Person linked to Midway; 'Midway Small Pilot'; 'Follow up in 7 days'"),
    "5": ("T6", "Strategic buyer, partnership",
          "Person linked to FutureTech; 'FutureTech Strategic Partnership' at MEETING"),
    "6": ("T7", "Challenged customer, complaint",
          "Move 'Innovate Onboarding' to PROPOSAL; 'Complaint follow-up'"),
    "7": ("T8", "Advocate, referral",
          "'Success LLC Referral' for existing Alex Thompson"),
    "8": ("T9", "Loyal customer, renewal",
          "'Enterprise Co Renewal' for existing Rachel Brown"),
    "9": ("T10", "At-risk customer, recovery",
          "'Transition Corp Recovery' + 'Recovery follow-up'"),
}


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def reasons_text(reasons: collections.Counter) -> str:
    parts = [f"{n} {r}" for r, n in reasons.most_common() if r]
    return ", ".join(parts) if parts else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE / "runs_v2")
    parser.add_argument("--model", default=None, help="only this model directory")
    parser.add_argument("--out", type=Path, default=None, help="write Markdown here")
    args = parser.parse_args()

    records: dict[str, dict[str, list[tuple[int, str]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    paths = sorted(args.runs.glob("*/*/trial_*/simulation.json"))
    if not paths:
        raise SystemExit(f"No simulation.json under {args.runs}")

    for path in paths:
        model = path.parts[-4]
        if args.model and model != args.model:
            continue
        task = path.parts[-3].replace("task_", "")
        data = json.loads(path.read_text())
        reward_info = data.get("reward_info") or {}
        reward = 1 if (reward_info.get("reward") or 0) >= 1.0 else 0
        reason = data.get("termination_reason") or "unknown"
        records[model][task].append((reward, reason))

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    rng = random.Random(SEED)
    for model in sorted(records):
        task_map = records[model]
        tasks = sorted(task_map, key=lambda x: int(x) if x.isdigit() else x)
        counts = [sum(r for r, _ in task_map[t]) for t in tasks]
        trials = [len(task_map[t]) for t in tasks]

        emit(f"### {model}")
        emit()
        emit("| Task | Scenario | Required CRM outcome | Success | pass^1 | pass^4 | pass^8 | Failed trials |")
        emit("|---|---|---|---:|---:|---:|---:|---|")
        for index, task in enumerate(tasks):
            label, scenario, outcome = PAPER.get(task, (f"T{task}", "", ""))
            s, n = counts[index], trials[index]
            reasons = collections.Counter(
                r for reward, r in task_map[task] if reward == 0
            )
            emit(
                f"| {label} | {scenario} | {outcome} | {s}/{n} | "
                f"{pct(s / n, 1) if n else '-'} | "
                f"{pct(pass_k(s, n, 4))} | {pct(pass_k(s, n, 8))} | "
                f"{reasons_text(reasons)} |"
            )

        total_s, total_n = sum(counts), sum(trials)
        emit(
            f"| **All tasks** | | | **{total_s}/{total_n}** | "
            f"**{pct(mean_pass_k(counts, trials, 1), 1)}** | "
            f"**{pct(mean_pass_k(counts, trials, 4))}** | "
            f"**{pct(mean_pass_k(counts, trials, 8))}** | |"
        )

        emit()
        emit("Model-level 95% task-cluster bootstrap CIs:")
        for k in (1, 4, 8):
            value = mean_pass_k(counts, trials, k)
            lo, hi = bootstrap_ci(counts, trials, k, rng)
            emit(f"- pass^{k}: {pct(value)}  CI [{100 * lo:.2f}, {100 * hi:.2f}]")

        incomplete = {
            t: len(task_map[t]) for t in tasks if len(task_map[t]) < max(trials)
        }
        if incomplete:
            emit()
            emit(f"WARNING: incomplete tasks: {incomplete}")
        emit()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines).rstrip() + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
