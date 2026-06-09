"""Grounded retail task synthesis: shared helpers.

Tasks are grounded in real entities sampled from the retail db, so every id is
valid by construction. Reference actions (`evaluation_criteria.actions`) are
derived in code mirroring the tool semantics, so the target DB end-state is
exactly what the scenario intends. `validate_task` proves it by replaying the
actions on a fresh env.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from tau2.domains.retail.data_model import CreditCard, GiftCard, Paypal, RetailDB
from tau2.domains.retail.environment import get_environment, get_tasks_split
from tau2.domains.retail.utils import RETAIL_DB_PATH, RETAIL_TASK_SET_PATH
from tau2.utils import load_file

DB = RetailDB.load(RETAIL_DB_PATH)

# The retail benchmark splits its tasks (data/.../split_tasks.json):
#   train: 74 · test: 40 · base: 114 (all).
# Only the TEST split is the held-out post-training eval set, so only its
# entities/solutions are forbidden as synthesis seeds. The train split is
# usable seed material. Override to ("train", "test") to exclude both.
PROTECTED_SPLITS = ("test",)


def _split_task_ids(splits=PROTECTED_SPLITS) -> set[str]:
    table = get_tasks_split()
    ids: set[str] = set()
    for s in splits:
        ids |= set(table.get(s, []))
    return ids


def protected_task_entities(splits=PROTECTED_SPLITS) -> tuple[set[str], set[str]]:
    """User/order ids referenced by the protected (held-out test) split — these
    are excluded as synthesis seeds so we never train on eval entities."""
    keep = _split_task_ids(splits)
    users, orders = set(), set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "user_id" and isinstance(v, str):
                    users.add(v)
                if k == "order_id" and isinstance(v, str):
                    orders.add(v)
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    for t in load_file(RETAIL_TASK_SET_PATH):
        if isinstance(t, dict) and t.get("id") in keep:
            walk(t)
    return users, orders


TEST_USERS, TEST_ORDERS = protected_task_entities()


def free_users():
    """Users not referenced by the held-out test split, sorted for determinism."""
    return [DB.users[u] for u in sorted(DB.users) if u not in TEST_USERS]


def orders_of(user, status=None):
    objs = [DB.orders[o] for o in user.orders if o in DB.orders]
    objs = [o for o in objs if o.order_id not in TEST_ORDERS]
    if status:
        objs = [o for o in objs if o.status == status]
    return objs


def product_of(item_id):
    for p in DB.products.values():
        if item_id in p.variants:
            return p
    return None


def available_alternates(product, current_item_id):
    """Available variants of the product other than the current one, cheapest first."""
    return sorted(
        (v for v in product.variants.values() if v.available and v.item_id != current_item_id),
        key=lambda v: v.price,
    )


def original_payment_id(order):
    return order.payment_history[0].payment_method_id


def gift_cards(user):
    return [(pid, pm) for pid, pm in user.payment_methods.items() if isinstance(pm, GiftCard)]


def non_gift_payment(user, exclude=None):
    for pid, pm in user.payment_methods.items():
        if not isinstance(pm, GiftCard) and pid != exclude:
            return pid, pm
    return None, None


def money(x: float) -> str:
    return f"{round(x, 2):.2f}"


@dataclass
class Built:
    """A candidate task plus generation metadata for the review summary."""

    task: dict
    cell: str
    grounding: str
    db_changes: bool = True  # False for refusal/transfer tasks (DB must stay unchanged)


def action(action_id, name, arguments, info=None):
    return {
        "action_id": action_id,
        "requestor": "assistant",
        "name": name,
        "arguments": arguments,
        "info": info,
    }


def _fresh_db_dump() -> dict:
    return get_environment().tools.db.model_dump()


_BASE_DB = _fresh_db_dump()


def _diff_dicts(before: dict, after: dict, path="") -> list[tuple[str, object, object]]:
    """Recursive leaf-level diff of two JSON-like dicts → [(path, before, after)]."""
    out = []
    keys = set(before) | set(after)
    for k in keys:
        bp = f"{path}.{k}" if path else str(k)
        bv, av = before.get(k), after.get(k)
        if isinstance(bv, dict) and isinstance(av, dict):
            out.extend(_diff_dicts(bv, av, bp))
        elif bv != av:
            out.append((bp, bv, av))
    return out


def replay_gold(task: dict) -> tuple[bool, str, list]:
    """Replay evaluation_criteria.actions on a fresh env.
    Returns (no_error, message, realized_db_diff). Mirrors the evaluator's gold
    replay; the diff is the task's *write-set* (target DB end state)."""
    env = get_environment()
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    for a in actions:
        try:
            env.make_tool_call(
                tool_name=a["name"], requestor=a.get("requestor", "assistant"), **a["arguments"]
            )
        except Exception as e:  # noqa: BLE001 - any tool error means the task is invalid
            return False, f"action {a['name']} raised: {e}", []
    diff = _diff_dicts(_BASE_DB, env.tools.db.model_dump())
    return True, "ok", diff


def diff_signature(diff: list) -> str:
    """Canonical, order-independent signature of a realized DB diff (D7 key).
    Two tasks with the same signature produce the same DB end state → duplicates
    / contamination, regardless of which surface ids they touch."""
    return "|".join(sorted(f"{p}={a!r}" for p, _b, a in diff))


def validate_task(task: dict, expect_db_change: bool) -> tuple[bool, str, list]:
    """D4(a,b) gate: gold actions execute cleanly and move the DB in the expected
    direction. Returns (ok, message, realized_db_diff)."""
    ok, msg, diff = replay_gold(task)
    if not ok:
        return False, msg, diff
    changed = len(diff) > 0
    if expect_db_change and not changed:
        return False, "expected DB to change but it did not (no-op task)", diff
    if not expect_db_change and changed:
        return False, "expected DB unchanged (refusal/transfer) but it changed", diff
    return True, "ok", diff


def test_task_signatures(splits=PROTECTED_SPLITS) -> set[str]:
    """DB-diff (write-set) signatures of the held-out test-split tasks, for
    solution-level decontamination (D7)."""
    keep = _split_task_ids(splits)
    sigs = set()
    for t in load_file(RETAIL_TASK_SET_PATH):
        if not (isinstance(t, dict) and t.get("id") in keep):
            continue
        ok, _msg, diff = replay_gold(t)
        if ok and diff:
            sigs.add(diff_signature(diff))
    return sigs


def policy_check(task: dict):
    """D11/D4(c) gate on the GOLD solution: run the user-provided legality
    validator over the gold actions, if present. Pluggable: returns
    ('skipped', None) until `retail_policy_validator.py` is delivered."""
    try:
        from retail_policy_validator import check_gold_actions  # type: ignore
    except Exception:  # noqa: BLE001 - validator not yet provided
        return "skipped", None
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    violations = check_gold_actions(actions, db_path=str(RETAIL_DB_PATH))
    return ("ok" if not violations else "violation"), violations
