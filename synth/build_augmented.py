"""Augment the 74 train seeds with the 3 failure-mode patterns (force-222 mode).

For each resolved seed x each pattern: build the deterministic transform spec
(gold write actions + change summary), write the scenario text (template by
default, LLM with --llm), then gate: execute-validate -> decontaminate vs the
test split -> schema-check. Emits tasks_augmented.json + a coverage report.

Dry-run validation (no API key):  uv run python synth/build_augmented.py
With natural-language rewrite:     uv run python synth/build_augmented.py --llm
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from lib import (
    DB,
    TEST_ORDERS,
    TEST_USERS,
    diff_signature,
    test_task_signatures,
    validate_task,
)
from patterns import TRANSFORMS, TransformSpec
from seeds import Seed, load_seeds
from tau2.data_model.tasks import Task

OUT = Path(__file__).with_name("tasks_augmented.json")

_ORDER_RE = re.compile(r"#W\d+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_EMAIL2UID = {u.email.lower(): u.user_id for u in DB.users.values()}


def _touches_test_entity(task: dict) -> bool:
    """True if the task references any test-split user or order. Train seeds can
    share entities with the test split (shared db), so strict policy drops these."""
    blob = json.dumps(task)
    for oid in set(_ORDER_RE.findall(blob)):
        if oid in TEST_ORDERS:
            return True
        if oid in DB.orders and DB.orders[oid].user_id in TEST_USERS:
            return True
    return any(_EMAIL2UID.get(e.lower()) in TEST_USERS for e in _EMAIL_RE.findall(blob))

FAILURE_NOTE = {
    "conditional_fallback": "Targets the conditional-fallback failure mode (model ignores the stated fallback when the first choice is unavailable).",
    "multi_goal": "Targets the multi-goal-phrasing failure mode (model drops one goal when several are stated together).",
    "mid_call_mind_change": "Targets the mid-call-mind-change failure mode (model acts on the abandoned first request).",
}


def _template_scenario(seed: Seed, spec: TransformSpec) -> dict:
    """Deterministic prose: append the change summary (and mind-change script) to
    the seed's own reason_for_call. Plain but faithful; --llm makes it natural."""
    instr = copy.deepcopy(seed.instructions)
    base = seed.reason_for_call.rstrip()
    instr["reason_for_call"] = (base + " " + spec.change_summary).strip()
    if spec.behavioral_script:
        ti = instr.get("task_instructions", "") or ""
        instr["task_instructions"] = (ti + " " + spec.behavioral_script).strip()
    return instr


def _assemble(seed: Seed, spec: TransformSpec, use_llm: bool, prefix: str = "retail_aug") -> dict:
    if use_llm:
        from rewrite import rewrite_scenario  # lazy: needs an API key
        instr = rewrite_scenario(seed, spec)
    else:
        instr = _template_scenario(seed, spec)

    crit = {"actions": spec.actions, "reward_basis": spec.reward_basis}
    if spec.communicate:
        crit["communicate_info"] = spec.communicate

    return {
        "id": f"{prefix}_{seed.id}_{spec.pattern}",
        "description": {
            "purpose": f"Augmented from train task {seed.id}: inject '{spec.pattern}'.",
            "relevant_policies": FAILURE_NOTE[spec.pattern],
            "notes": spec.change_summary
            + (f" Mind-change script: {spec.behavioral_script}" if spec.behavioral_script else ""),
        },
        "user_scenario": {
            "persona": seed.task["user_scenario"].get("persona"),
            "instructions": instr,
        },
        "evaluation_criteria": crit,
    }


def run(use_llm: bool = False):
    seeds = load_seeds()
    resolved = [s for s in seeds if s.resolved()]
    test_sigs = test_task_signatures()
    seen_sigs: dict[str, str] = {}
    kept, report = [], []
    status = Counter()

    for seed in resolved:
        for pattern, fn in TRANSFORMS.items():
            spec = fn(seed)
            tag = f"{seed.id}/{pattern}"
            if spec is None:
                status[f"{pattern}:not-constructible"] += 1
                report.append((tag, "SKIP", "user can't host this pattern"))
                continue
            task = _assemble(seed, spec, use_llm)
            ok, msg, diff = validate_task(task, expect_db_change=spec.db_changes)
            if not ok:
                status[f"{pattern}:invalid"] += 1
                report.append((tag, "INVALID", msg))
                continue
            # strict decontam: drop tasks whose seed shares a user/order with the
            # test split (the eval set), even though their solutions are disjoint.
            if _touches_test_entity(task):
                status[f"{pattern}:test-entity"] += 1
                report.append((tag, "TEST-ENTITY", "seed shares a user/order with test split"))
                continue
            sig = diff_signature(diff)
            # CONTAM (vs TEST) is write-set-based: same solution as an eval task is
            # leakage regardless of pattern. DUP (among synthetic) is keyed by
            # (pattern, write-set): two patterns reaching the same DB state are
            # different training tasks (different dynamics), not duplicates — only
            # the same pattern from duplicate seeds is a true duplicate.
            dup_key = (pattern, sig)
            if diff and sig in test_sigs:
                status[f"{pattern}:contam"] += 1
                report.append((tag, "CONTAM", "write-set matches a TEST task"))
                continue
            if diff and dup_key in seen_sigs:
                status[f"{pattern}:dup"] += 1
                report.append((tag, "DUP", f"same {pattern} write-set as {seen_sigs[dup_key]}"))
                continue
            if diff:
                seen_sigs[dup_key] = task["id"]
            try:
                Task.model_validate(task)
            except Exception as e:  # noqa: BLE001
                status[f"{pattern}:schema"] += 1
                report.append((tag, "SCHEMA-ERR", str(e)[:80]))
                continue
            kept.append(task)
            status[f"{pattern}:kept"] += 1
            report.append((tag, "VALID", f"{len(spec.actions)} action(s), {len(diff)} db-changes"))

    OUT.write_text(json.dumps(kept, indent=2))

    print(f"\nseeds: {len(seeds)} total, {len(resolved)} resolved (write-bearing)")
    print("per-pattern outcomes:")
    for pat in TRANSFORMS:
        kept_n = status[f"{pat}:kept"]
        skipped = status[f"{pat}:not-constructible"]
        bad = sum(status[f"{pat}:{k}"] for k in ("invalid", "contam", "dup", "schema", "test-entity"))
        print(f"  {pat:<22} kept={kept_n:>3}  not-constructible={skipped:>3}  rejected={bad:>3}")
    print(f"\nTOTAL kept: {len(kept)} / {len(resolved)*3} attempted ({len(resolved)} seeds x 3) "
          f"→ {OUT}")
    if not use_llm:
        print("note: scenario text is TEMPLATE (deterministic). Re-run with --llm for "
              "natural phrasing (needs an API key); gold actions/validation are identical.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="Use the LLM scenario rewriter (needs API key).")
    args = ap.parse_args()
    run(use_llm=args.llm)
