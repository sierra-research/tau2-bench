"""Grounded per-cell task builders (PLAN_v2 Phase 2).

Each builder mines real entities from the decontaminated user pool and derives
the reference `evaluation_criteria.actions` in code, so the target DB end-state
is correct by construction. The build orchestrator (`build_tasks.py`) runs the
gates (execute-validate, DB-diff decontam, policy, alignment).
"""

from __future__ import annotations

from collections import defaultdict

from lib import (
    Built,
    action,
    available_alternates,
    free_users,
    gift_cards,
    money,
    orders_of,
    original_payment_id,
    product_of,
)

NEW_ADDR = dict(address1="742 Evergreen Terrace", address2="Apt 5",
                city="Springfield", state="OR", country="USA", zip="97403")

taken_users: set[str] = set()
taken_orders: set[str] = set()


def reset_state() -> None:
    taken_users.clear()
    taken_orders.clear()


def claim(user, *orders) -> None:
    taken_users.add(user.user_id)
    for o in orders:
        taken_orders.add(o.order_id)


def fullname(user) -> str:
    return f"{user.name.first_name} {user.name.last_name}"


def scenario(user, reason, instructions, persona, by="email", extra_known=""):
    if by == "email":
        ident = f"Your email is {user.email}."
    else:
        ident = (f"You don't remember the email on file; identify yourself by name "
                 f"({fullname(user)}) and zip code {user.address.zip}.")
    known = f"You are {fullname(user)}. {ident} {extra_known}".strip()
    return {
        "persona": persona,
        "instructions": {
            "domain": "retail",
            "reason_for_call": reason,
            "known_info": known,
            "unknown_info": None,
            "task_instructions": instructions,
        },
    }


def make(task_id, desc, user_scenario, actions, communicate=None, reward_basis=None):
    crit = {"actions": actions}
    if communicate is not None:
        crit["communicate_info"] = communicate
    if reward_basis is not None:
        crit["reward_basis"] = reward_basis
    return {
        "id": task_id,
        "description": desc,
        "user_scenario": user_scenario,
        "evaluation_criteria": crit,
    }


# --- builders: each returns a Built or None -------------------------------

def b_cancel_mistake(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            amt = sum(p.amount for p in o.payment_history if p.transaction_type == "payment")
            claim(u, o)
            sc = scenario(u,
                f"You want to cancel order {o.order_id} because you ordered it by mistake.",
                f"Ask to cancel order {o.order_id}. The reason is that you ordered it by mistake. "
                f"Confirm when the agent explains the refund.",
                "Brief and a little embarrassed; you just want it undone.")
            acts = [action("cancel_1", "cancel_pending_order",
                           {"order_id": o.order_id, "reason": "ordered by mistake"})]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Cancel a single pending order with reason 'ordered by mistake'.",
                "relevant_policies": "Cancel pending order; valid reasons; refund routing.",
                "notes": f"Order {o.order_id}; refund {money(amt)} to original method(s)."},
                sc, acts, communicate=[money(amt)]),
                "cancel pending", f"{u.user_id} / {o.order_id} (refund {money(amt)})")
    return None


def b_cancel_plus_return(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        pend = orders_of(u, "pending")
        deliv = orders_of(u, "delivered")
        if pend and deliv:
            po, do = pend[0], deliv[0]
            it = do.items[0]
            pay = original_payment_id(do)
            claim(u, po, do)
            sc = scenario(u,
                f"Two things: cancel pending order {po.order_id}, and return one item from delivered order {do.order_id}.",
                f"First, cancel pending order {po.order_id} (no longer needed). "
                f"Second, return the {it.name} from delivered order {do.order_id}, refunded to the original payment method. "
                f"Handle both before you hang up.",
                "Efficient; you have two unrelated requests and want both done.")
            acts = [
                action("cancel_1", "cancel_pending_order",
                       {"order_id": po.order_id, "reason": "no longer needed"}),
                action("return_1", "return_delivered_order_items",
                       {"order_id": do.order_id, "item_ids": [it.item_id], "payment_method_id": pay}),
            ]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Multi-intent: cancel a pending order AND return an item from a different delivered order.",
                "relevant_policies": "Cancel pending order; return delivered order; refund to original method.",
                "notes": f"Cancel {po.order_id}; return item {it.item_id} from {do.order_id} to {pay}."},
                sc, acts, communicate=[do.order_id, po.order_id]),
                "multi-intent (cancel+return)", f"{u.user_id} / {po.order_id}+{do.order_id}")
    return None


def b_modify_items_cheapest(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            for it in o.items:
                p = product_of(it.item_id)
                alts = available_alternates(p, it.item_id) if p else []
                if alts:
                    new = alts[0]
                    pay = original_payment_id(o)
                    diff = round(new.price - it.price, 2)
                    claim(u, o)
                    sc = scenario(u,
                        f"On pending order {o.order_id} you want to switch the {it.name} to the cheapest available {p.name} variant.",
                        f"For pending order {o.order_id}, change the {it.name} to the cheapest available variant of the same product. "
                        f"Any price difference goes to your original payment method. Confirm once the agent states the new price.",
                        "Budget-conscious; you only care about getting the cheapest option.")
                    acts = [action("modify_1", "modify_pending_order_items",
                                   {"order_id": o.order_id, "item_ids": [it.item_id],
                                    "new_item_ids": [new.item_id], "payment_method_id": pay})]
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Modify a pending order item to the cheapest available same-product variant (requires variant search).",
                        "relevant_policies": "Modify pending items; same product type; pay/refund difference.",
                        "notes": f"{it.item_id}->{new.item_id} ({new.options}); price diff {money(diff)} via {pay}."},
                        sc, acts, communicate=[money(new.price)]),
                        "modify pending items (cheapest)",
                        f"{u.user_id} / {o.order_id} {it.item_id}->{new.item_id} diff {money(diff)}")
    return None


def b_modify_items_fallback(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            for it in o.items:
                p = product_of(it.item_id)
                if not p:
                    continue
                opt_key = next(iter(it.options), None)
                if not opt_key:
                    continue
                unavailable_vals = {v.options.get(opt_key) for v in p.variants.values()
                                    if not v.available and v.options.get(opt_key) != it.options.get(opt_key)}
                alts = available_alternates(p, it.item_id)
                if unavailable_vals and alts:
                    pref = sorted(x for x in unavailable_vals if x)[0]
                    new = alts[0]
                    pay = original_payment_id(o)
                    claim(u, o)
                    sc = scenario(u,
                        f"On pending order {o.order_id} you'd like to change the {it.name}: ideally {opt_key} '{pref}', "
                        f"but if that isn't available take the cheapest available variant instead.",
                        f"For pending order {o.order_id}, change the {it.name} to one with {opt_key} '{pref}'. "
                        f"If {opt_key} '{pref}' is not available, take the cheapest available variant of the same product. "
                        f"Price difference to your original payment method.",
                        "Flexible; you state a preference but accept a clear fallback.")
                    acts = [action("modify_1", "modify_pending_order_items",
                                   {"order_id": o.order_id, "item_ids": [it.item_id],
                                    "new_item_ids": [new.item_id], "payment_method_id": pay})]
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Conditional fallback: preferred option unavailable, agent must apply the stated fallback rule.",
                        "relevant_policies": "Modify pending items; check availability; same product type.",
                        "notes": f"Preferred {opt_key}='{pref}' unavailable -> fallback cheapest available {new.item_id} ({new.options})."},
                        sc, acts, communicate=[money(new.price)]),
                        "modify pending items (fallback)",
                        f"{u.user_id} / {o.order_id} pref {opt_key}={pref} -> {new.item_id}")
    return None


def b_modify_address_new(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            claim(u, o)
            sc = scenario(u,
                f"You moved and need the shipping address on pending order {o.order_id} updated.",
                f"Update the shipping address of pending order {o.order_id} to "
                f"{NEW_ADDR['address1']}, {NEW_ADDR['address2']}, {NEW_ADDR['city']}, "
                f"{NEW_ADDR['state']} {NEW_ADDR['zip']}, {NEW_ADDR['country']}.",
                "Recently relocated; precise about the new address.")
            acts = [action("addr_1", "modify_pending_order_address",
                           {"order_id": o.order_id, **NEW_ADDR})]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Change the shipping address of a pending order to a new address.",
                "relevant_policies": "Modify pending order address; explicit confirmation.",
                "notes": f"Order {o.order_id} -> {NEW_ADDR['address1']}, {NEW_ADDR['city']} {NEW_ADDR['zip']}."},
                sc, acts, communicate=[NEW_ADDR["zip"]]),
                "modify pending address (new)", f"{u.user_id} / {o.order_id}")
    return None


def b_modify_address_to_default(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            a = u.address
            if (o.address.zip, o.address.address1) != (a.zip, a.address1):
                claim(u, o)
                addr = dict(address1=a.address1, address2=a.address2, city=a.city,
                            state=a.state, country=a.country, zip=a.zip)
                sc = scenario(u,
                    f"The pending order {o.order_id} is going to an old address; ship it to your default address on file instead.",
                    f"Change the shipping address of pending order {o.order_id} to match your account's default address on file.",
                    "Assumes the agent can look up your default address; doesn't recite it.")
                acts = [action("addr_1", "modify_pending_order_address",
                               {"order_id": o.order_id, **addr})]
                return Built(make(f"retail_synth_{i:03d}", {
                    "purpose": "Set a pending order's shipping address to the user's account default (cross-entity lookup).",
                    "relevant_policies": "Modify pending order address; get user details.",
                    "notes": f"Order {o.order_id} address -> user default {a.zip} ({a.city})."},
                    sc, acts, communicate=[a.zip]),
                    "modify pending address (to default)", f"{u.user_id} / {o.order_id} -> {a.zip}")
    return None


def b_modify_payment_giftcard(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            if len(o.payment_history) != 1 or o.payment_history[0].transaction_type != "payment":
                continue
            amt = o.payment_history[0].amount
            cur = original_payment_id(o)
            for pid, gc in gift_cards(u):
                if pid != cur and gc.balance >= amt:
                    claim(u, o)
                    sc = scenario(u,
                        f"You want to pay for pending order {o.order_id} with your gift card instead of the current method.",
                        f"Change the payment method on pending order {o.order_id} to your gift card. Confirm when explained.",
                        "Wants to use up a gift card balance.")
                    acts = [action("pay_1", "modify_pending_order_payment",
                                   {"order_id": o.order_id, "payment_method_id": pid})]
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Switch a pending order's payment method to a gift card with sufficient balance.",
                        "relevant_policies": "Modify pending order payment; one payment only; gift-card balance check.",
                        "notes": f"Order {o.order_id} amount {money(amt)} -> {pid} (balance {money(gc.balance)})."},
                        sc, acts, communicate=[money(amt)]),
                        "modify pending payment (gift card)",
                        f"{u.user_id} / {o.order_id} -> {pid} (bal {money(gc.balance)} >= {money(amt)})")
    return None


def b_modify_payment_giftcard_insufficient(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            if len(o.payment_history) != 1:
                continue
            amt = o.payment_history[0].amount
            cur = original_payment_id(o)
            for pid, gc in gift_cards(u):
                if pid != cur and gc.balance < amt:
                    claim(u, o)
                    sc = scenario(u,
                        f"You want to switch pending order {o.order_id} to your gift card.",
                        f"Ask to change the payment method on pending order {o.order_id} to your gift card. "
                        f"You have no other instructions if that can't be done.",
                        "Hopeful but with no backup plan if the gift card can't cover it.")
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Refusal: gift card balance is insufficient to cover the order; agent must decline and leave the order unchanged.",
                        "relevant_policies": "Modify pending order payment; gift-card balance must cover the order.",
                        "notes": f"Order {o.order_id} amount {money(amt)} > gift card {pid} balance {money(gc.balance)} -> no change."},
                        sc, actions=[], reward_basis=["DB"]),
                        "refuse (insufficient gift card)",
                        f"{u.user_id} / {o.order_id} amt {money(amt)} > bal {money(gc.balance)}",
                        db_changes=False)
    return None


def b_exchange_single(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            for it in o.items:
                p = product_of(it.item_id)
                alts = available_alternates(p, it.item_id) if p else []
                if alts:
                    new = alts[0]
                    pay = original_payment_id(o)
                    diff = round(new.price - it.price, 2)
                    claim(u, o)
                    sc = scenario(u,
                        f"You received order {o.order_id} and want to exchange the {it.name} for a different variant.",
                        f"Exchange the {it.name} in delivered order {o.order_id} for the cheapest available variant of the same product. "
                        f"Settle any difference on your original payment method.",
                        "Decisive; confirms once the swap and price are clear.")
                    acts = [action("exch_1", "exchange_delivered_order_items",
                                   {"order_id": o.order_id, "item_ids": [it.item_id],
                                    "new_item_ids": [new.item_id], "payment_method_id": pay})]
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Exchange one item in a delivered order for an available same-product variant.",
                        "relevant_policies": "Exchange delivered order; once per order; same product type.",
                        "notes": f"{it.item_id}->{new.item_id} ({new.options}); diff {money(diff)} via {pay}."},
                        sc, acts, communicate=[money(new.price)]),
                        "exchange delivered (single)",
                        f"{u.user_id} / {o.order_id} {it.item_id}->{new.item_id}")
    return None


def b_exchange_two_fallback(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            picks = []
            for it in o.items:
                p = product_of(it.item_id)
                alts = available_alternates(p, it.item_id) if p else []
                if alts:
                    picks.append((it, p, alts[0]))
                if len(picks) == 2:
                    break
            if len(picks) == 2:
                pay = original_payment_id(o)
                old_ids = [pk[0].item_id for pk in picks]
                new_ids = [pk[2].item_id for pk in picks]
                names = " and ".join(pk[0].name for pk in picks)
                claim(u, o)
                sc = scenario(u,
                    f"You want to exchange two items from delivered order {o.order_id}: the {names}.",
                    f"In a single exchange on delivered order {o.order_id}, swap both the {picks[0][0].name} and the "
                    f"{picks[1][0].name} for the cheapest available variant of each (same products). "
                    f"Only confirm once both are handled together; differences on your original payment method.",
                    "Insists both items be handled in one exchange, not separately.",
                    by="name_zip")
                acts = [action("exch_1", "exchange_delivered_order_items",
                               {"order_id": o.order_id, "item_ids": old_ids,
                                "new_item_ids": new_ids, "payment_method_id": pay})]
                return Built(make(f"retail_synth_{i:03d}", {
                    "purpose": "Exchange two items in one call (single-call constraint), each to its cheapest available variant.",
                    "relevant_policies": "Exchange delivered order; one exchange call per order; same product type only.",
                    "notes": f"{old_ids}->{new_ids} via {pay}; both must be in one exchange call."},
                    sc, acts, communicate=[o.order_id]),
                    "exchange delivered (two, single-call)",
                    f"{u.user_id} / {o.order_id} {old_ids}->{new_ids}")
    return None


def b_exchange_disambiguation(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        deliv = orders_of(u, "delivered")
        if len(deliv) < 2:
            continue
        for o in deliv:
            for it in o.items:
                p = product_of(it.item_id)
                alts = available_alternates(p, it.item_id) if p else []
                if alts:
                    other = next(x for x in deliv if x.order_id != o.order_id)
                    new = alts[0]
                    pay = original_payment_id(o)
                    claim(u, *deliv)
                    sc = scenario(u,
                        f"You have two delivered orders ({o.order_id} and {other.order_id}); exchange the {it.name} from {o.order_id}.",
                        f"You have more than one delivered order. Exchange the {it.name} that is in order {o.order_id} "
                        f"(not your other order {other.order_id}) for the cheapest available variant of the same product. "
                        f"Original payment method for any difference.",
                        "Has several orders and can be vague at first; agent must confirm the right one.")
                    acts = [action("exch_1", "exchange_delivered_order_items",
                                   {"order_id": o.order_id, "item_ids": [it.item_id],
                                    "new_item_ids": [new.item_id], "payment_method_id": pay})]
                    return Built(make(f"retail_synth_{i:03d}", {
                        "purpose": "Disambiguate which of two delivered orders holds the item to exchange.",
                        "relevant_policies": "Exchange delivered order; identify correct order before acting.",
                        "notes": f"Exchange {it.item_id} in {o.order_id}; distractor order {other.order_id}."},
                        sc, acts, communicate=[o.order_id]),
                        "exchange delivered (disambiguation)",
                        f"{u.user_id} / target {o.order_id} vs {other.order_id}")
    return None


def b_return_single(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            it = o.items[0]
            pay = original_payment_id(o)
            claim(u, o)
            sc = scenario(u,
                f"You want to return the {it.name} from delivered order {o.order_id}.",
                f"Return the {it.name} from delivered order {o.order_id}; refund to the original payment method.",
                "Straightforward return; expects the refund to go back the way they paid.")
            acts = [action("return_1", "return_delivered_order_items",
                           {"order_id": o.order_id, "item_ids": [it.item_id], "payment_method_id": pay})]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Return a single item from a delivered order to the original payment method.",
                "relevant_policies": "Return delivered order; refund to original method or gift card.",
                "notes": f"Return {it.item_id} from {o.order_id} to {pay}."},
                sc, acts, communicate=[money(it.price)]),
                "return delivered (single)", f"{u.user_id} / {o.order_id} {it.item_id}")
    return None


def b_return_subset(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            if len(o.items) >= 3:
                items = o.items[:2]
                ids = [x.item_id for x in items]
                pay = original_payment_id(o)
                names = " and ".join(x.name for x in items)
                claim(u, o)
                sc = scenario(u,
                    f"From delivered order {o.order_id} you want to return two of the items, keeping the rest.",
                    f"Return only the {names} from delivered order {o.order_id} (keep everything else). "
                    f"Refund to the original payment method.",
                    "Keeping most of the order; only two items didn't work out.")
                acts = [action("return_1", "return_delivered_order_items",
                               {"order_id": o.order_id, "item_ids": ids, "payment_method_id": pay})]
                return Built(make(f"retail_synth_{i:03d}", {
                    "purpose": "Return a subset of items from a multi-item delivered order, keeping the others.",
                    "relevant_policies": "Return delivered order; only specified items.",
                    "notes": f"Return {ids} of {len(o.items)} items from {o.order_id} to {pay}."},
                    sc, acts, communicate=[o.order_id]),
                    "return delivered (subset)", f"{u.user_id} / {o.order_id} {ids}")
    return None


def b_return_to_giftcard(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        gcs = gift_cards(u)
        if not gcs:
            continue
        for o in orders_of(u, "delivered"):
            it = o.items[0]
            gid = gcs[0][0]
            claim(u, o)
            sc = scenario(u,
                f"Return the {it.name} from delivered order {o.order_id}, but refund to your gift card.",
                f"Return the {it.name} from delivered order {o.order_id}. Put the refund on your gift card rather than the original method.",
                "Prefers store credit; explicitly wants the gift card refund.")
            acts = [action("return_1", "return_delivered_order_items",
                           {"order_id": o.order_id, "item_ids": [it.item_id], "payment_method_id": gid})]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Return a delivered item with refund routed to the gift card (allowed alternative to original method).",
                "relevant_policies": "Return delivered order; refund to original method OR gift card.",
                "notes": f"Return {it.item_id} from {o.order_id} -> refund to {gid}."},
                sc, acts, communicate=[money(it.price)]),
                "return delivered (to gift card)", f"{u.user_id} / {o.order_id} -> {gid}")
    return None


def b_exchange_dup_item(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        deliv = orders_of(u, "delivered")
        item2orders = defaultdict(list)
        for o in deliv:
            for it in o.items:
                item2orders[it.item_id].append(o)
        for iid, ords in item2orders.items():
            if len({o.order_id for o in ords}) >= 2:
                p = product_of(iid)
                alts = available_alternates(p, iid) if p else []
                if not alts:
                    continue
                o = ords[0]
                other = next(x for x in ords if x.order_id != o.order_id)
                new = alts[0]
                pay = original_payment_id(o)
                name = next(x.name for x in o.items if x.item_id == iid)
                claim(u, o, other)
                sc = scenario(u,
                    f"The same {name} appears in two of your delivered orders; exchange only the one in {o.order_id}.",
                    f"You ordered the same {name} twice (in orders {o.order_id} and {other.order_id}). "
                    f"Exchange only the one in order {o.order_id} for the cheapest available variant; leave {other.order_id} alone. "
                    f"Original payment method for any difference.",
                    "Knows it's confusing that the item is duplicated; insists only one order changes.")
                acts = [action("exch_1", "exchange_delivered_order_items",
                               {"order_id": o.order_id, "item_ids": [iid],
                                "new_item_ids": [new.item_id], "payment_method_id": pay})]
                return Built(make(f"retail_synth_{i:03d}", {
                    "purpose": "Disambiguate a duplicated item id present in two delivered orders; act on only one.",
                    "relevant_policies": "Exchange delivered order; correct order identification.",
                    "notes": f"Item {iid} in {o.order_id} and {other.order_id}; exchange only in {o.order_id} -> {new.item_id}."},
                    sc, acts, communicate=[o.order_id]),
                    "exchange delivered (duplicate item)",
                    f"{u.user_id} / item {iid} in {o.order_id}&{other.order_id}")
    return None


def b_refuse_modify_delivered(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            it = o.items[0]
            claim(u, o)
            sc = scenario(u,
                f"You want to change the variant of the {it.name} on order {o.order_id} (which is already delivered).",
                f"Ask to modify the items on order {o.order_id} to a different variant — note you think it's still editable. "
                f"You do NOT want to exchange or return; you want it 'modified' like a pending order. Accept the answer either way.",
                "Mistakenly believes a delivered order can be edited like a pending one.")
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Refusal: 'modify items' is only valid for pending orders; the order is delivered, so the agent must decline (and may suggest exchange/return).",
                "relevant_policies": "Modify pending items applies to pending orders only; delivered -> exchange/return.",
                "notes": f"Order {o.order_id} is delivered -> no modify_pending_order_items; DB must stay unchanged."},
                sc, actions=[], reward_basis=["DB"]),
                "refuse (modify delivered)", f"{u.user_id} / {o.order_id} (delivered)",
                db_changes=False)
    return None


def b_refuse_cancel_delivered(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            claim(u, o)
            sc = scenario(u,
                f"You want to cancel order {o.order_id}.",
                f"Ask to cancel order {o.order_id}. (It has already been delivered.) Accept the agent's explanation.",
                "Thinks any order can be cancelled; not aware delivered orders can't.")
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Refusal: a delivered order cannot be cancelled; agent must decline and may suggest return/exchange.",
                "relevant_policies": "Cancel pending order only; delivered orders cannot be cancelled.",
                "notes": f"Order {o.order_id} delivered -> cancel not allowed; DB unchanged."},
                sc, actions=[], reward_basis=["DB"]),
                "refuse (cancel delivered)", f"{u.user_id} / {o.order_id} (delivered)",
                db_changes=False)
    return None


def b_transfer_out_of_scope(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            it = o.items[0]
            claim(u, o)
            sc = scenario(u,
                f"You saw the {it.name} from order {o.order_id} is cheaper elsewhere and want a price-match refund of the difference.",
                f"Demand a partial cash refund as a price match on the {it.name} in order {o.order_id}. "
                f"There is no tool/policy for price matching. Push once, then accept the outcome.",
                "A little pushy; wants money back for a price difference.")
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Out-of-policy request (price match / partial cash refund) with no supporting tool; agent must refuse or transfer, never fabricate a refund.",
                "relevant_policies": "No price-match tool; agent must not invent refunds; transfer_to_human if needed.",
                "notes": f"No DB write is correct; DB must stay unchanged for order {o.order_id}."},
                sc, actions=[], reward_basis=["DB"]),
                "refuse/transfer (out of scope)", f"{u.user_id} / {o.order_id} price-match",
                db_changes=False)
    return None


def b_multi_intent_items_plus_address(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "pending"):
            target = None
            for it in o.items:
                p = product_of(it.item_id)
                alts = available_alternates(p, it.item_id) if p else []
                if alts:
                    target = (it, alts[0])
                    break
            if target:
                it, new = target
                pay = original_payment_id(o)
                claim(u, o)
                sc = scenario(u,
                    f"On pending order {o.order_id}: change the {it.name} to a different variant AND update the shipping address.",
                    f"Two changes to pending order {o.order_id}: (1) change the {it.name} to the cheapest available variant of the same product, "
                    f"and (2) ship it to {NEW_ADDR['address1']}, {NEW_ADDR['city']}, {NEW_ADDR['state']} {NEW_ADDR['zip']}. "
                    f"Original payment method for any difference.",
                    "Has two edits for the same order and wants both applied.")
                acts = [
                    action("modify_1", "modify_pending_order_items",
                           {"order_id": o.order_id, "item_ids": [it.item_id],
                            "new_item_ids": [new.item_id], "payment_method_id": pay}),
                    action("addr_1", "modify_pending_order_address",
                           {"order_id": o.order_id, **NEW_ADDR}),
                ]
                return Built(make(f"retail_synth_{i:03d}", {
                    "purpose": "Multi-intent on one pending order: modify an item variant and change the shipping address.",
                    "relevant_policies": "Modify pending items (once); modify pending address; same product type.",
                    "notes": f"{it.item_id}->{new.item_id}; address -> {NEW_ADDR['zip']}; both on {o.order_id}."},
                    sc, acts, communicate=[NEW_ADDR["zip"]]),
                    "multi-intent (items+address)", f"{u.user_id} / {o.order_id}")
    return None


def b_auth_namezip_then_return(i):
    for u in free_users():
        if u.user_id in taken_users:
            continue
        for o in orders_of(u, "delivered"):
            it = o.items[0]
            pay = original_payment_id(o)
            claim(u, o)
            sc = scenario(u,
                f"You can't recall the email on the account; you want to return the {it.name} from order {o.order_id}.",
                f"You do not remember your account email. Have the agent find you by name and zip, then return the "
                f"{it.name} from delivered order {o.order_id} to the original payment method.",
                "Forgot which email is on file; can give name and zip.",
                by="name_zip")
            acts = [action("return_1", "return_delivered_order_items",
                           {"order_id": o.order_id, "item_ids": [it.item_id], "payment_method_id": pay})]
            return Built(make(f"retail_synth_{i:03d}", {
                "purpose": "Authenticate by name+zip (email forgotten), then return a delivered item.",
                "relevant_policies": "Authenticate by name+zip when email unavailable; return delivered order.",
                "notes": f"Auth {u.name.first_name} {u.name.last_name}/{u.address.zip}; return {it.item_id} from {o.order_id}."},
                sc, acts, communicate=[money(it.price)]),
                "auth name+zip then return", f"{u.user_id} / {o.order_id}")
    return None


BUILDERS = [
    b_cancel_mistake, b_cancel_plus_return, b_modify_items_cheapest, b_modify_items_fallback,
    b_modify_address_new, b_modify_address_to_default, b_modify_payment_giftcard,
    b_modify_payment_giftcard_insufficient, b_exchange_single, b_exchange_two_fallback,
    b_exchange_disambiguation, b_return_single, b_return_subset, b_return_to_giftcard,
    b_exchange_dup_item, b_refuse_modify_delivered, b_refuse_cancel_delivered,
    b_transfer_out_of_scope, b_multi_intent_items_plus_address, b_auth_namezip_then_return,
]