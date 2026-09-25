"""Build grounded, gated, tau2-compliant retail tasks (PLAN_v2 Phase 2 + gates).

Gates per candidate:
  D4(a,b) execute-validate   : gold actions run cleanly + move DB as expected   [always]
  D7      DB-diff decontam    : write-set signature not shared with the 114 test
                               tasks, and unique among synthetic tasks           [always]
  D4(c)   policy compliance   : gold actions legal per retail_policy_validator    [if provided]
  D4(d)   intent alignment    : LLM judge confirms gold actions fulfil intent      [if --judge]

Output: a JSON list[Task] validated by tau2's own pydantic model, ready to drop
into a task set and run with `tau2 run`. Run:  uv run python synth/build_tasks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib import diff_signature, policy_check, test_task_signatures, validate_task
from tau2.data_model.tasks import Task

import generators as G

OUT = Path(__file__).with_name("tasks_synth.json")


def run(judge: bool = False):
    G.reset_state()
    test_sigs = test_task_signatures()
    seen_sigs: dict[str, str] = {}
    kept, report = [], []

    for idx, builder in enumerate(G.BUILDERS, start=1):
        b = builder(idx)
        if b is None:
            report.append(("---", builder.__name__.replace("b_", ""), "NO-MATCH",
                           "no grounding found", ""))
            continue
        tid = b.task["id"]

        ok, msg, diff = validate_task(b.task, expect_db_change=b.db_changes)
        if not ok:
            report.append((tid, b.cell, "INVALID", msg, b.grounding))
            continue

        # D7 decontamination (write-set; refusal tasks have empty diff -> skip)
        sig = diff_signature(diff)
        if diff:
            if sig in test_sigs:
                report.append((tid, b.cell, "CONTAM", "DB-diff matches a test task", b.grounding))
                continue
            if sig in seen_sigs:
                report.append((tid, b.cell, "DUP", f"same write-set as {seen_sigs[sig]}", b.grounding))
                continue
            seen_sigs[sig] = tid

        # D4(c) policy compliance of the gold solution (pluggable)
        pstatus, pviol = policy_check(b.task)
        if pstatus == "violation":
            report.append((tid, b.cell, "ILLEGAL-GOLD", str(pviol), b.grounding))
            continue

        # D4(d) intent alignment (opt-in; needs an LLM judge + API key)
        if judge:
            from judge import alignment_ok  # local, optional
            aok, why = alignment_ok(b.task, diff)
            if not aok:
                report.append((tid, b.cell, "MISALIGNED", why, b.grounding))
                continue

        # Final compliance check against tau2's own schema
        try:
            Task.model_validate(b.task)
        except Exception as e:  # noqa: BLE001
            report.append((tid, b.cell, "SCHEMA-ERR", str(e)[:80], b.grounding))
            continue

        kept.append(b.task)
        tag = "VALID" + ("" if pstatus != "ok" else "+legal")
        report.append((tid, b.cell, tag, f"{b.grounding}  [{len(diff)} db-changes]", ""))

    OUT.write_text(json.dumps(kept, indent=2))

    print(f"\n{'id':<18} {'cell':<34} {'status':<12} detail")
    print("-" * 120)
    for tid, cell, status, detail, grounding in report:
        line = detail if status.startswith("VALID") else f"{detail}"
        extra = f"  ({grounding})" if grounding and not status.startswith("VALID") else ""
        print(f"{tid:<18} {cell:<34} {status:<12} {line}{extra}")

    n_valid = len(kept)
    print(f"\n{n_valid}/{len(G.BUILDERS)} tasks valid & decontaminated → {OUT}")
    if policy_check({"evaluation_criteria": {"actions": []}})[0] == "skipped":
        print("note: legality gate (D11) SKIPPED — retail_policy_validator.py not on path yet.")
    if not judge:
        print("note: intent-alignment gate (D4d) OFF — pass --judge (needs API key) to enable.")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="store_true", help="Enable LLM intent-alignment gate (D4d).")
    args = ap.parse_args()
    run(judge=args.judge)
