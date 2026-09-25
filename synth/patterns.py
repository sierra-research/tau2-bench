"""Deterministic transform-spec builders for the three target failure modes.

Each `apply_<pattern>(seed)` returns a `TransformSpec` (or None if the seed's
user can't host it). The spec carries the GOLD write actions (the verifiable DB
target, derived in code) and a structured `change_summary` that the LLM rewriter
turns into natural `reason_for_call` text. The LLM never decides the answer —
only how to phrase it.

Failure modes (from baseline-Qwen error analysis):
  conditional_fallback  — user states a preference + a fallback if unavailable
  multi_goal_phrasing   — two goals stated together in one turn
  mid_call_mind_change  — user asks for one thing, then switches mid-conversation
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from lib import (
    DB,
    available_alternates,
    money,
    orders_of,
    original_payment_id,
    product_of,
)
from seeds import Seed

NEW_ADDR = dict(address1="742 Evergreen Terrace", address2="Apt 5",
                city="Springfield", state="OR", country="USA", zip="97403")


@dataclass
class TransformSpec:
    pattern: str
    actions: list[dict]            # gold WRITE actions = DB target
    change_summary: str            # structured description for the LLM rewriter
    behavioral_script: str = ""    # mind_change: instruction for the user simulator
    communicate: list[str] = field(default_factory=list)
    reward_basis: list[str] = field(default_factory=lambda: ["DB"])
    db_changes: bool = True


def _act(name, arguments, i=1):
    return {"action_id": f"{name}_{i}", "requestor": "assistant",
            "name": name, "arguments": arguments}


# --- grounded add-on writes (reused by multi_goal / fallback-compose) ------

def _fallback_variant_write(order, idx=1):
    """Build a variant write on `order` whose preferred option is unavailable, so
    the gold targets the cheapest available alternate. Returns (action, summary)
    or None. Tool depends on order status (exchange=delivered, modify=pending)."""
    for it in order.items:
        p = product_of(it.item_id)
        if not p:
            continue
        opt = next(iter(it.options), None)
        if not opt:
            continue
        unavail = sorted(
            {v.options.get(opt) for v in p.variants.values()
             if not v.available and v.options.get(opt) and v.options.get(opt) != it.options.get(opt)}
        )
        alts = available_alternates(p, it.item_id)
        if not unavail or not alts:
            continue
        preferred, target = unavail[0], alts[0]
        tool = ("exchange_delivered_order_items" if order.status == "delivered"
                else "modify_pending_order_items")
        action = _act(tool, {
            "order_id": order.order_id, "item_ids": [it.item_id],
            "new_item_ids": [target.item_id],
            "payment_method_id": original_payment_id(order),
        }, idx)
        summary = (f"On order {order.order_id}, the {it.name}: the customer prefers "
                   f"{opt} '{preferred}', but if that is unavailable they accept the "
                   f"cheapest available variant of the same product "
                   f"(gold target {target.item_id}, {target.options}, {money(target.price)}). "
                   f"'{preferred}' is in fact unavailable, so the fallback applies.")
        return action, summary, money(target.price)
    return None


def _seed_order_ids(seed) -> set:
    return {a.get("arguments", {}).get("order_id") for a in seed.write_actions}


def _addon_write(seed, idx=2):
    """An independent second goal on the seed's user. Prefers a different eligible
    order; else a compatible same-(pending)-order edit. Returns (action, summary).
    Excludes ALL orders the seed already writes to (multi-write seeds touch >1)."""
    u = seed.user
    elig = orders_of(u)
    seed_orders = _seed_order_ids(seed)
    target = seed.target_order_id
    others = [o for o in elig if o.order_id not in seed_orders]

    for o in others:
        if o.status == "pending":
            return (_act("cancel_pending_order",
                         {"order_id": o.order_id, "reason": "no longer needed"}, idx),
                    f"Also cancel the separate pending order {o.order_id} (no longer needed).")
        if o.status == "delivered":
            it = o.items[0]
            return (_act("return_delivered_order_items",
                         {"order_id": o.order_id, "item_ids": [it.item_id],
                          "payment_method_id": original_payment_id(o)}, idx),
                    f"Also return the {it.name} from the separate delivered order {o.order_id}.")

    # same-order compatible add-on: a pending target can also take an address change
    to = DB.orders.get(target)
    seed_tool = seed.primary_write["name"] if seed.primary_write else ""
    if to and to.status == "pending" and seed_tool != "modify_pending_order_address":
        return (_act("modify_pending_order_address", {"order_id": target, **NEW_ADDR}, idx),
                f"Also change the shipping address of order {target} to "
                f"{NEW_ADDR['address1']}, {NEW_ADDR['city']} {NEW_ADDR['zip']}.")
    return None


# --- presence detectors (for reporting; force-222 applies regardless) ------

def present_multi_goal(seed: Seed) -> bool:
    return len(seed.write_actions) >= 2


def present_conditional_fallback(seed: Seed) -> bool:
    return bool(re.search(r"if .*(not available|unavailable|no )|otherwise|if there (is|are) no",
                          seed.reason_for_call, re.I))


def present_mind_change(seed: Seed) -> bool:
    return bool(re.search(r"chang(e|ed) (your |my )?mind|actually,? (instead|i'd)", seed.reason_for_call, re.I))


# --- the three transforms ---------------------------------------------------

def apply_conditional_fallback(seed: Seed):
    """Native (variant seed): inject a fallback into the seed's OWN variant write
    in place (preserve intent). Compose (non-variant): add a fallback variant goal
    on an order the seed doesn't already touch."""
    seed_orders = _seed_order_ids(seed)

    # native: modify one item of the seed's variant write to a fallback target
    if seed.is_variant_seed:
        actions = copy.deepcopy(seed.write_actions)
        pw = actions[-1]
        for j, old in enumerate(pw["arguments"]["item_ids"]):
            p = product_of(old)
            if not p:
                continue
            opt = next(iter(p.variants[old].options), None) if old in p.variants else None
            if not opt:
                continue
            unavail = sorted(
                {v.options.get(opt) for v in p.variants.values()
                 if not v.available and v.options.get(opt)
                 and v.options.get(opt) != p.variants[old].options.get(opt)}
            )
            alts = available_alternates(p, old)
            if not unavail or not alts:
                continue
            preferred, target = unavail[0], alts[0]
            pw["arguments"]["new_item_ids"][j] = target.item_id
            summary = (f"For the {p.name} in order {pw['arguments']['order_id']}, the "
                       f"customer prefers {opt} '{preferred}', but if unavailable accepts "
                       f"the cheapest available variant (gold {target.item_id}, "
                       f"{target.options}, {money(target.price)}); '{preferred}' is "
                       f"unavailable, so the fallback applies.")
            return TransformSpec("conditional_fallback", actions, summary,
                                 communicate=[money(target.price)])

    # compose: add a fallback variant goal on an order the seed doesn't touch
    for o in orders_of(seed.user):
        if o.order_id in seed_orders or o.status not in ("delivered", "pending"):
            continue
        built = _fallback_variant_write(o, idx=len(seed.write_actions) + 1)
        if built:
            action, summary, price = built
            actions = list(seed.write_actions) + [action]
            return TransformSpec("conditional_fallback", actions,
                                 "Keep the original request, AND additionally: " + summary,
                                 communicate=[price])
    return None


def apply_multi_goal(seed: Seed):
    addon = _addon_write(seed, idx=len(seed.write_actions) + 1)
    if not addon:
        return None
    action, summary = addon
    actions = list(seed.write_actions) + [action]
    return TransformSpec("multi_goal", actions,
                         "Two goals stated together in one turn: (1) the original "
                         "request, and (2) " + summary)


def apply_mid_call_mind_change(seed: Seed):
    """Keep the seed's write as the FINAL gold; script a decoy (different at the DB
    level) the user asks for first, then abandons."""
    pw = seed.primary_write
    if not pw:
        return None
    args = pw["arguments"]
    name = pw["name"]
    decoy = None

    if name in ("exchange_delivered_order_items", "modify_pending_order_items"):
        order = DB.orders.get(args["order_id"])
        old = args["item_ids"][0]
        final_new = args["new_item_ids"][0]
        p = product_of(old)
        alts = [v for v in (available_alternates(p, old) if p else []) if v.item_id != final_new]
        if alts:
            d = alts[0]
            decoy = (f"first ask to change the item to variant {d.item_id} "
                     f"({d.options}, {money(d.price)})")
    elif name == "cancel_pending_order":
        other = "ordered by mistake" if args.get("reason") == "no longer needed" else "no longer needed"
        decoy = f"first give the cancellation reason '{other}'"
    elif name == "return_delivered_order_items":
        order = DB.orders.get(args["order_id"])
        chosen = set(args["item_ids"])
        extra = next((it for it in (order.items if order else []) if it.item_id not in chosen), None)
        if extra:
            decoy = f"first ask to return the {extra.name} ({extra.item_id}) instead"
    elif name in ("modify_pending_order_address", "modify_user_address"):
        decoy = "first give a different, wrong address (123 Old Road, Lasttown CA 90001)"
    elif name == "modify_pending_order_payment":
        decoy = "first ask to switch to a different payment method on file"

    if not decoy:
        return None
    script = (f"Mid-call mind change: {decoy}. After the agent looks it up / starts to "
              f"act, change your mind and ask for what you actually want (the original "
              f"goal) instead. Only confirm the final choice; do not let the first "
              f"request go through.")
    return TransformSpec("mid_call_mind_change", list(seed.write_actions),
                         "Final goal is unchanged from the original task; the customer "
                         "first asks for something else, then switches.",
                         behavioral_script=script)


TRANSFORMS = {
    "conditional_fallback": apply_conditional_fallback,
    "multi_goal": apply_multi_goal,
    "mid_call_mind_change": apply_mid_call_mind_change,
}
