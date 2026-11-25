"""From-scratch top-up of failure-mode tasks, to complement the train-seed
augmentation (build_augmented.py) up to a per-pattern target.

Same 3 patterns (patterns.py), but grounded on free_users() — which already
excludes the test split — so every top-up task is test-clean by construction.
A synthetic single-write "base seed" is built per free user, then the pattern
transforms are applied exactly as in the augmentation path.

Run:  uv run python synth/build_topup.py [--per-pattern 74]
Combined output: tasks_failuremode.json = augmented (train-seed) + top-up.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_augmented import _assemble, _touches_test_entity
from lib import (
    available_alternates,
    diff_signature,
    free_users,
    money,
    orders_of,
    original_payment_id,
    product_of,
    replay_gold,
    test_task_signatures,
    validate_task,
)
from patterns import TRANSFORMS
from seeds import Seed
from tau2.data_model.tasks import Task

AUG = Path(__file__).with_name("tasks_augmented.json")
TOPUP = Path(__file__).with_name("tasks_topup.json")
COMBINED = Path(__file__).with_name("tasks_failuremode.json")


def _base_write(order):
    """A simple single base write for a free user's order; prefer a variant write
    (so conditional_fallback applies natively). Returns (action, reason)."""
    pay = original_payment_id(order)
    if order.status in ("delivered", "pending"):
        for it in order.items:
            p = product_of(it.item_id)
            alts = available_alternates(p, it.item_id) if p else []
            if alts:
                tool = ("exchange_delivered_order_items" if order.status == "delivered"
                        else "modify_pending_order_items")
                return ({"action_id": f"{tool}_1", "requestor": "assistant", "name": tool,
                         "arguments": {"order_id": order.order_id, "item_ids": [it.item_id],
                                       "new_item_ids": [alts[0].item_id], "payment_method_id": pay}},
                        f"You want to change the {it.name} in order {order.order_id} to a "
                        f"different variant of the same product.")
    if order.status == "delivered":
        it = order.items[0]
        return ({"action_id": "return_delivered_order_items_1", "requestor": "assistant",
                 "name": "return_delivered_order_items",
                 "arguments": {"order_id": order.order_id, "item_ids": [it.item_id],
                               "payment_method_id": pay}},
                f"You want to return the {it.name} from order {order.order_id}.")
    if order.status == "pending":
        return ({"action_id": "cancel_pending_order_1", "requestor": "assistant",
                 "name": "cancel_pending_order",
                 "arguments": {"order_id": order.order_id, "reason": "no longer needed"}},
                f"You want to cancel pending order {order.order_id} (no longer needed).")
    return None


def _base_seed(user, order):
    built = _base_write(order)
    if not built:
        return None
    write, reason = built
    task = {
        "id": f"syn_{user.user_id}",
        "user_scenario": {"persona": None, "instructions": {
            "domain": "retail", "reason_for_call": reason,
            "known_info": f"You are {user.name.first_name} {user.name.last_name}. "
                          f"Your email is {user.email}.",
            "unknown_info": None, "task_instructions": ""}},
        "evaluation_criteria": {"actions": [write], "reward_basis": ["DB"]},
    }
    return Seed(task=task, write_actions=[write], user_id=user.user_id,
               target_order_id=order.order_id)


def run(per_pattern: int = 74):
    augmented = json.loads(AUG.read_text()) if AUG.exists() else []
    aug_counts = Counter(t["id"].split("retail_aug_")[1].split("_", 1)[1] for t in augmented)
    need = {pat: max(0, per_pattern - aug_counts.get(pat, 0)) for pat in TRANSFORMS}
    print(f"augmented per-pattern: {dict(aug_counts)} | top-up needed: {need}")

    test_sigs = test_task_signatures()
    # seed dedup keys from the augmented set so top-up doesn't repeat solutions
    seen = set()
    for t in augmented:
        ok, _m, diff = replay_gold(t)
        if ok and diff:
            pat = t["id"].split("retail_aug_")[1].split("_", 1)[1]
            seen.add((pat, diff_signature(diff)))

    kept = []
    for u in free_users():
        if all(v <= 0 for v in need.values()):
            break
        orders = orders_of(u)
        if not orders:
            continue
        seed = _base_seed(u, orders[0])
        if seed is None:
            continue
        for pattern, fn in TRANSFORMS.items():
            if need[pattern] <= 0:
                continue
            spec = fn(seed)
            if spec is None:
                continue
            task = _assemble(seed, spec, use_llm=False, prefix="retail_topup")
            ok, _msg, diff = validate_task(task, expect_db_change=spec.db_changes)
            if not ok or _touches_test_entity(task):
                continue
            sig = diff_signature(diff)
            if diff and (sig in test_sigs or (pattern, sig) in seen):
                continue
            try:
                Task.model_validate(task)
            except Exception:  # noqa: BLE001
                continue
            if diff:
                seen.add((pattern, sig))
            kept.append(task)
            need[pattern] -= 1

    TOPUP.write_text(json.dumps(kept, indent=2))
    combined = augmented + kept
    COMBINED.write_text(json.dumps(combined, indent=2))

    tc = Counter(t["id"].split("_aug_" if "_aug_" in t["id"] else "_topup_")[1].split("_", 1)[1]
                 for t in combined)
    print(f"top-up kept: {len(kept)} {dict(Counter(t['id'].split('retail_topup_')[1].split('_',1)[1] for t in kept))}")
    print(f"COMBINED failure-mode set: {len(combined)} tasks → {COMBINED.name}")
    print(f"  per pattern (aug+topup): {dict(tc)}")
    print(f"  remaining shortfall: {dict({p: v for p, v in need.items() if v > 0}) or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-pattern", type=int, default=74, help="Target tasks per failure mode (aug+topup).")
    args = ap.parse_args()
    run(per_pattern=args.per_pattern)
