"""Canonical form of order line items, so equivalent orders hash equal."""

import json
from typing import Dict, List, Optional

from tau2.domains.food.data_model import (
    MenuItem,
    ModifierGroup,
    ModifierOption,
    OrderLineItem,
    SelectedModifier,
    Store,
)
from tau2.domains.food.modifiers import _PORTION_OPT_NAME_RE

# Menu naming convention for paired half-pizza modifier groups ('Left Half <X>' / 'Right Half <X>').
# Grading treats a pizza and its left/right mirror image as the same order.
_HALF_GROUP_SWAPS = (("Left Half ", "Right Half "), ("Right Half ", "Left Half "))


def _option_named(grp: ModifierGroup, name: str) -> Optional[ModifierOption]:
    """The first option of `grp` whose display name is exactly `name`."""
    return next((o for o in grp.options.values() if o.name == name), None)


def _mirror_group_id(store: Store, gid: str) -> Optional[str]:
    """The 'Right Half X' group for a 'Left Half X' group and vice versa, if the store has one."""
    name = store.modifier_groups[gid].name
    for this, other in _HALF_GROUP_SWAPS:
        if name.startswith(this):
            target = name.replace(this, other, 1)
            return next(
                (
                    cgid
                    for cgid, cgrp in store.modifier_groups.items()
                    if cgrp.name == target
                ),
                None,
            )
    return None


def _is_half_modifier(m: SelectedModifier, store: Optional[Store] = None) -> bool:
    if m.placement in ("left", "right"):
        return True
    parsed = _PORTION_OPT_NAME_RE.match(m.name)
    if parsed is not None and parsed.group(2) is not None:
        return True
    if store is not None and m.group_id in store.modifier_groups:
        gname = store.modifier_groups[m.group_id].name
        if gname.startswith(tuple(this for this, _ in _HALF_GROUP_SWAPS)):
            return True
    return False


def _swap_half_modifier(
    m: SelectedModifier,
    store: Optional[Store] = None,
) -> SelectedModifier:
    """Swap First Half (left) <-> Second Half (right) on a pizza modifier."""
    mc = m.model_copy(deep=True)
    parsed = _PORTION_OPT_NAME_RE.match(mc.name)
    if parsed is not None and parsed.group(2) is not None:
        other_half = "Second" if parsed.group(2).lower() == "first" else "First"
        mc.name = f"{parsed.group(1)} - {other_half} Half ({parsed.group(3)})"
        if store is not None and mc.group_id in store.modifier_groups:
            opt = _option_named(store.modifier_groups[mc.group_id], mc.name)
            if opt is not None:
                mc.option_id = opt.option_id
    if mc.placement == "left":
        mc.placement = "right"
    elif mc.placement == "right":
        mc.placement = "left"
    if store is not None and mc.group_id in store.modifier_groups:
        mirror_gid = _mirror_group_id(store, mc.group_id)
        if mirror_gid is not None:
            mc.group_id = mirror_gid
            mirror = store.modifier_groups[mirror_gid]
            if mc.option_id not in mirror.options:
                opt = _option_named(mirror, mc.name)
                if opt is not None:
                    mc.option_id = opt.option_id
    return mc


def _effective_group_defaults(
    store: Store, item: MenuItem, li: OrderLineItem, gid: str
) -> List[str]:
    """Effective default option IDs for `gid` on `item` given the parent selections on `li`."""
    grp = store.modifier_groups.get(gid)
    if grp is None:
        return []
    for m in li.modifiers:
        pgrp = store.modifier_groups.get(m.group_id)
        popt = pgrp.options.get(m.option_id) if pgrp is not None else None
        if popt is not None and gid in popt.child_group_defaults:
            return list(popt.child_group_defaults[gid])
    if gid in item.group_defaults:
        return list(item.group_defaults[gid])
    return list(grp.default_option_ids)


def _collapse_split_halves(
    mods: List[SelectedModifier],
    store: Optional[Store],
) -> List[SelectedModifier]:
    """Collapse matching '<Base> - First Half (<Amt>)' + '<Base> - Second Half (<Amt>)' into '<Base> (<Amt>)'."""
    if store is None:
        return mods
    by_half: Dict[tuple[str, str, str, str], tuple[int, SelectedModifier]] = {}
    for idx, m in enumerate(mods):
        if m.placement != "whole" or m.volume != "normal":
            continue
        parsed = _PORTION_OPT_NAME_RE.match(m.name)
        if parsed is None or parsed.group(2) is None:
            continue
        base, half, amt = parsed.group(1), parsed.group(2).lower(), parsed.group(3)
        by_half[(m.group_id, base.lower(), half, amt.lower())] = (idx, m)

    skip: set[int] = set()
    replacements: Dict[int, SelectedModifier] = {}
    for (gid, base_l, half, amt_l), (idx1, m1) in by_half.items():
        if half != "first":
            continue
        mate = by_half.get((gid, base_l, "second", amt_l))
        if mate is None:
            continue
        idx2, m2 = mate
        if m1.quantity != m2.quantity:
            continue
        grp = store.modifier_groups.get(gid)
        if grp is None:
            continue
        parsed1 = _PORTION_OPT_NAME_RE.match(m1.name)
        if parsed1 is None:
            continue
        whole_opt = _option_named(grp, f"{parsed1.group(1)} ({parsed1.group(3)})")
        if whole_opt is None:
            continue
        replacements[idx1] = SelectedModifier(
            group_id=gid,
            option_id=whole_opt.option_id,
            name=whole_opt.name,
            quantity=m1.quantity,
            placement="whole",
            volume="normal",
            unit_price_cents=whole_opt.price_cents,
        )
        skip.add(idx2)

    if not replacements:
        return mods
    return [replacements.get(idx, m) for idx, m in enumerate(mods) if idx not in skip]


def _canonicalize_line_item_modifiers(
    li: OrderLineItem,
    store: Optional[Store] = None,
) -> List[SelectedModifier]:
    """Canonicalize a line item's modifiers (half mirror symmetry and structural default equivalences).

    Entries for the same (group, option, placement, volume) are merged by summing their quantities,
    so a default applied twice (e.g. 2 espresso shots) equals an explicit selection of quantity 2.
    """
    item = store.items.get(li.item_id) if store is not None else None
    positive_oids_by_gid: Dict[str, set[str]] = {}
    for m in li.modifiers:
        if not m.name.lower().startswith("no "):
            positive_oids_by_gid.setdefault(m.group_id, set()).add(m.option_id)

    mods: List[SelectedModifier] = []
    by_key: Dict[tuple[str, str, str, str], SelectedModifier] = {}
    for m in li.modifiers:
        mc = m.model_copy(deep=True)
        if store is not None and mc.unit_price_cents == 0:
            grp = store.modifier_groups.get(mc.group_id)
            opt = grp.options.get(mc.option_id) if grp is not None else None
            eff_defs = (
                _effective_group_defaults(store, item, li, mc.group_id)
                if item is not None
                else (list(grp.default_option_ids) if grp is not None else [])
            )
            if (
                mc.quantity == 1
                and not mc.name.lower().startswith("no ")
                and grp is not None
                and grp.min_select == 0
                and not eff_defs
                and opt is not None
                and not opt.child_group_ids
                and not store.models_default_ingredients
                and any(
                    o.name.lower() == f"no {mc.name.lower()}"
                    for o in grp.options.values()
                )
            ):
                # On stores that do not expand default ingredients into the cart, a $0 option paired
                # with a same-group 'No <ingredient>' option is already part of the standard item.
                continue
            elif (
                mc.name.lower().startswith("no ")
                and grp is not None
                and grp.min_select == 0
                and store.models_default_ingredients
                and eff_defs
            ):
                target_lower = mc.name[3:].strip().lower()
                target_oid = next(
                    (
                        oid
                        for oid, o in grp.options.items()
                        if o.name.lower() == target_lower
                    ),
                    None,
                )
                if (
                    target_oid in eff_defs
                    and target_oid not in positive_oids_by_gid.get(mc.group_id, set())
                ):
                    continue
        key = (mc.group_id, mc.option_id, mc.placement, mc.volume)
        if key in by_key:
            if not mc.name.lower().startswith("no "):
                by_key[key].quantity += mc.quantity
            continue
        by_key[key] = mc
        mods.append(mc)

    mods = _collapse_split_halves(mods, store)

    def sort_key(m: SelectedModifier) -> tuple[str, str, str, str, int]:
        return (m.group_id, m.option_id, m.placement, m.volume, m.quantity)

    mods_sorted = sorted(mods, key=sort_key)

    if any(_is_half_modifier(m, store) for m in mods_sorted):
        swapped = sorted(
            [_swap_half_modifier(m, store=store) for m in mods_sorted],
            key=sort_key,
        )
        orig_json = json.dumps([m.model_dump() for m in mods_sorted], sort_keys=True)
        swap_json = json.dumps([m.model_dump() for m in swapped], sort_keys=True)
        if swap_json < orig_json:
            return swapped

    return mods_sorted


def _canonicalize_line_items(
    items: List[OrderLineItem],
    store: Optional[Store] = None,
) -> List[OrderLineItem]:
    """Merge identical line items (same item_id + canonical modifiers) and sort deterministically."""
    merged_map: Dict[tuple[str, str], OrderLineItem] = {}
    for li in items:
        li_copy = li.model_copy(deep=True)
        li_copy.modifiers = _canonicalize_line_item_modifiers(li_copy, store=store)
        mods_key = json.dumps(
            [m.model_dump() for m in li_copy.modifiers], sort_keys=True
        )
        key = (li_copy.item_id, mods_key)
        if key in merged_map:
            existing = merged_map[key]
            unit_total = li_copy.line_total_cents // max(1, li_copy.quantity)
            existing.quantity += li_copy.quantity
            existing.line_total_cents = existing.quantity * unit_total
        else:
            merged_map[key] = li_copy

    sorted_keys = sorted(merged_map.keys())
    canonical_items: List[OrderLineItem] = []
    for idx, key in enumerate(sorted_keys, start=1):
        item_copy = merged_map[key]
        item_copy.line_item_id = f"item_{idx}"
        canonical_items.append(item_copy)
    return canonical_items
