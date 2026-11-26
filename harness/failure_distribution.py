#!/usr/bin/env python3
"""Build a review worksheet and first-pass failure distribution from run logs.

Table 6 (F1-F6) is a codebook, not a tau2 metric. This script reads every
*/task_*/trial_*/simulation.json, extracts the evidence a coder needs
(failed assertions, tool errors, created-versus-expected records, termination
reason) and proposes a provisional category. It writes a per-trial CSV with
blank final_code/notes columns and a Markdown summary. Edit final_code by hand,
then rerun to get the final table.

Usage:
    python failure_distribution.py --runs runs --out failure_distribution
    python failure_distribution.py --review-csv failure_distribution/failures_to_code.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
KEY_ERROR = re.compile(r"error:\s*'", re.I)

REQUIRED_ARGS = {
    "create_person": ["first_name", "last_name"],
    "create_company": ["name"],
    "create_opportunity": ["name", "person_id"],
    "create_task": ["subject"],
    "update_person_trust_commitment": ["person_id"],
    "update_person_lead_score": ["person_id", "lead_score"],
    "update_opportunity_stage": ["opportunity_id", "stage"],
    "update_task_status": ["task_id", "status"],
    "get_person_details": ["person_id"],
    "get_company_details": ["company_id"],
    "get_opportunity_details": ["opportunity_id"],
    "get_task_details": ["task_id"],
    "get_relationship_metrics": ["person_id"],
}

VALID_TOOLS = {
    "find_people",
    "find_companies",
    "find_opportunities",
    "find_tasks",
    "create_person",
    "create_company",
    "create_opportunity",
    "create_task",
    "update_person_trust_commitment",
    "update_person_lead_score",
    "update_opportunity_stage",
    "update_task_status",
    "get_person_details",
    "get_company_details",
    "get_opportunity_details",
    "get_task_details",
    "get_relationship_metrics",
}

ENTITY_TOOL = {"person": "create_person", "opportunity": "create_opportunity", "task": "create_task"}
ENTITY_ASSERTION = {
    "verify_people_total": "person",
    "verify_opportunities_total": "opportunity",
    "verify_tasks_total": "task",
}

WRONG_LINK = {"verify_person_named_company", "verify_opportunity_named", "verify_task_named"}
STAGE = {"verify_opportunity_stage"}
MISSING = {"verify_person_exists"}
TOTALS = set(ENTITY_ASSERTION)

F3_HINTS = ["policy", "blocked", "block", "denied", "forbidden", "not allowed", "unauthor", "permission"]
F1_HINTS = ["schema", "invalid", "malformed", "unknown tool", "unexpected", "not a valid", "type error"]
F2_HINTS = ["missing", "required", "null"]

MODEL_LABELS = {
    "qwen9b_q4": "Qwen 3.5 9B",
    "qwen4b_q4": "Qwen 3.5 4B",
    "gptoss20b_q4": "GPT-OSS 20B",
    "deepseek_chat": "DeepSeek chat",
    "phi4mini_instruct_q4km": "Phi-4-mini 3.8B",
}

TASK_LABELS = {str(i): f"T{i + 1}" for i in range(10)}


def find_model_dir(path: Path) -> str:
    task_dir = path.parent.parent
    candidates = [task_dir.parent.name, task_dir.parent.parent.name]
    for name in candidates:
        if name in MODEL_LABELS:
            return name
    return task_dir.parent.name


def model_label(directory: str) -> str:
    return MODEL_LABELS.get(directory, directory)


def extract_evidence(data: dict) -> dict:
    reward_info = data.get("reward_info") or {}
    assertions = reward_info.get("env_assertions") or []
    failed = {a["env_assertion"].get("func_name") for a in assertions if not a.get("met")}
    expected = {}
    for a in assertions:
        name = a["env_assertion"].get("func_name")
        if name in ENTITY_ASSERTION:
            count = (a["env_assertion"].get("arguments") or {}).get("count")
            if isinstance(count, int):
                expected[ENTITY_ASSERTION[name]] = count

    calls: dict[str, str] = {}
    results: dict[str, bool] = {}
    tool_errors: list[str] = []
    missing_args = 0
    bad_refs = 0
    bad_schema = 0
    for message in data.get("messages") or []:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                calls[call.get("id")] = call.get("name")
                name = call.get("name")
                args = call.get("arguments")
                if not isinstance(args, dict) or name not in VALID_TOOLS:
                    bad_schema += 1
                    continue
                for key in REQUIRED_ARGS.get(name, []):
                    if args.get(key) in (None, ""):
                        missing_args += 1
                if name == "create_person":
                    company = args.get("company_id")
                    if company and not UUID.match(str(company)):
                        bad_refs += 1
        elif message.get("role") == "tool":
            results[message.get("id")] = bool(message.get("error"))
            if message.get("error"):
                tool_errors.append(str(message.get("content") or ""))

    created = Counter()
    for call_id, name in calls.items():
        if results.get(call_id) is False:
            for entity, tool in ENTITY_TOOL.items():
                if name == tool:
                    created[entity] += 1
    over = sorted(e for e, n in created.items() if e in expected and n > expected[e])
    under = sorted(e for e, n in created.items() if e in expected and n < expected[e])

    return {
        "failed": failed,
        "expected": expected,
        "created": dict(created),
        "over": over,
        "under": under,
        "tool_errors": tool_errors,
        "tool_error_text": " ".join(tool_errors).lower(),
        "missing_args": missing_args,
        "bad_refs": bad_refs,
        "bad_schema": bad_schema,
        "has_calls": any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in data.get("messages") or []
        ),
        "termination": data.get("termination_reason") or "unknown",
    }


def mismatch_label(ev: dict) -> str:
    failed = ev["failed"]
    parts = []
    if failed & WRONG_LINK:
        parts.append("wrong_link")
    if failed & STAGE:
        parts.append("wrong_stage")
    if failed & MISSING:
        parts.append("missing_person")
    if ev["over"]:
        parts.append("extra_" + "+".join(ev["over"]))
    if ev["under"]:
        parts.append("short_" + "+".join(ev["under"]))
    if failed & TOTALS and not parts:
        parts.append("totals_only")
    return "|".join(parts) if parts else "none"


def propose(ev: dict) -> tuple[str, str]:
    failed = ev["failed"]
    text = ev["tool_error_text"]
    mismatch = mismatch_label(ev)
    if any(h in text for h in F3_HINTS):
        return "F3", "tool error text mentions policy/block/permission"
    if ev["bad_schema"] or any(h in text for h in F1_HINTS):
        return "F1", "malformed or rejected tool call"
    if ev["missing_args"] or any(h in text for h in F2_HINTS):
        return "F2", f"{ev['missing_args']} missing required argument(s)"
    if ev["termination"] == "max_steps" or (ev["termination"] == "too_many_errors" and not failed):
        return "F6", f"terminated with {ev['termination']}"
    if not ev["has_calls"]:
        return "F6", "no tool call issued"
    if mismatch.startswith("wrong_link") or ev["bad_refs"] or any(KEY_ERROR.search(e) for e in ev["tool_errors"]):
        return "F4", f"entity mismatch ({mismatch}); {ev['bad_refs']} non-UUID company refs"
    if ev["over"] or ev["under"] or (failed & (STAGE | MISSING)):
        return "F5", f"mutation completeness ({mismatch}; created {ev['created']} vs expected {ev['expected']})"
    if failed & TOTALS:
        return "F5", f"mutation completeness (record-count mismatch; created {ev['created']} vs expected {ev['expected']})"
    return "F6", "no concrete database mismatch; dialogue ended early"


def load_review(path: Path | None) -> dict[tuple[str, str, str], dict]:
    if not path or not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            (row["model_dir"], row["task"], row["trial"]): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "failure_distribution")
    parser.add_argument("--review-csv", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    review_path = args.review_csv or (args.out / "failures_to_code.csv")
    review = load_review(review_path)

    rows = []
    counts: dict[str, Counter] = defaultdict(Counter)
    failures: Counter = Counter()
    reviewed: Counter = Counter()

    for path in sorted(args.runs.glob("*/*/task_*/trial_*/simulation.json")):
        data = json.loads(path.read_text())
        reward_info = data.get("reward_info") or {}
        if (reward_info.get("reward") or 0) >= 1.0:
            continue
        model_dir = find_model_dir(path)
        task = path.parent.parent.name.replace("task_", "")
        trial = path.parent.name.replace("trial_", "")
        ev = extract_evidence(data)
        code, basis = propose(ev)
        previous = review.get((model_dir, task, trial), {})
        final_code = (previous.get("final_code") or "").strip().upper()
        used = final_code if final_code else code
        failures[model_dir] += 1
        if final_code:
            reviewed[model_dir] += 1
        counts[model_dir][used] += 1
        rows.append(
            {
                "model_dir": model_dir,
                "model": model_label(model_dir),
                "task": TASK_LABELS.get(task, task),
                "trial": trial,
                "termination": ev["termination"],
                "failed_assertions": ";".join(sorted(ev["failed"])),
                "mismatch": mismatch_label(ev),
                "created": json.dumps(ev["created"], sort_keys=True),
                "expected": json.dumps(ev["expected"], sort_keys=True),
                "over": "+".join(ev["over"]),
                "under": "+".join(ev["under"]),
                "tool_errors": len(ev["tool_errors"]),
                "bad_company_refs": ev["bad_refs"],
                "missing_args": ev["missing_args"],
                "has_tool_calls": ev["has_calls"],
                "proposed_code": code,
                "proposed_basis": basis,
                "final_code": final_code,
                "notes": previous.get("notes", ""),
                "path": str(path),
            }
        )

    csv_path = args.out / "failures_to_code.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)

    codes = ["F1", "F2", "F3", "F4", "F5", "F6"]
    lines = [
        "# Failure distribution (F1-F6)",
        "",
        "Counts and percentage of failed trials. Uses final_code where set, otherwise the first-pass proposal.",
        "Extra and duplicate records are coded under F5, together with missing, partial, and wrong-stage mutations.",
        "",
        "| Model | Failed | " + " | ".join(codes) + " | Reviewed |",
        "|---|---:|" + "---:|" * len(codes) + "---:|",
    ]
    totals = Counter()
    for model_dir in sorted(failures):
        total = failures[model_dir]
        totals.update(counts[model_dir])
        cells = [
            f"{counts[model_dir][c]} ({100 * counts[model_dir][c] / total:.1f}%)"
            for c in codes
        ]
        lines.append(
            f"| {model_label(model_dir)} | {total} | " + " | ".join(cells) + f" | {reviewed[model_dir]}/{total} |"
        )
    grand = sum(failures.values())
    cells = [f"{totals[c]} ({100 * totals[c] / grand:.1f}%)" for c in codes]
    lines.append(f"| **All models** | **{grand}** | " + " | ".join(cells) + f" | {sum(reviewed.values())}/{grand} |")
    lines.append("")
    lines.append(f"Worksheet: `{csv_path}`. Set `final_code` to F1-F6 and rerun with `--review-csv`.")

    md_path = args.out / "failure_distribution.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
