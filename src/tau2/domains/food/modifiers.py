"""Modifier resolution and line-item validation for the food domain."""

import re
from typing import Any, Callable, Dict, List, Optional

from tau2.domains.food.data_model import (
    MenuItem,
    ModifierGroup,
    ModifierOption,
    ModifierSelection,
    OrderLineItem,
    SelectedModifier,
    Store,
)
from tau2.domains.food.normalize import _slugify

_PORTION_OPT_NAME_RE = re.compile(
    r"^(.+?)(?: - (First|Second) Half)? \((Light|Normal|Extra)\)$",
    re.I,
)


def _parse_portion_option_name(name: str) -> Optional[tuple[str, str]]:
    """Return (placement, volume) in ({'whole', 'left', 'right'}, {'light', 'normal', 'extra'}) for a portioned option name."""
    m = _PORTION_OPT_NAME_RE.match(name or "")
    if not m:
        return None
    half = (m.group(2) or "").lower()
    vol = m.group(3).lower()
    part = "left" if half == "first" else ("right" if half == "second" else "whole")
    amt = "light" if vol == "light" else ("extra" if vol == "extra" else "normal")
    return part, amt


_AMOUNT_SUFFIX_RE = re.compile(r"\s*\((light|normal|extra)\)$", re.I)
_HALF_SUFFIX_RE = re.compile(r"- (first|second) half$", re.I)


def _split_portion_suffix(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """Splits '<base>[ - First|Second Half][ (Light|Normal|Extra)]' into (base, part, amount).

    `part` is 'left' or 'right' and `amount` is 'light', 'normal' or 'extra' when present.
    """
    part: Optional[str] = None
    amount: Optional[str] = None
    m = _AMOUNT_SUFFIX_RE.search(text)
    if m:
        amount = m.group(1).lower()
        text = text[: m.start()].strip()
    m = _HALF_SUFFIX_RE.search(text)
    if m:
        part = "left" if m.group(1).lower() == "first" else "right"
        text = text[: m.start()].rstrip(" -").strip()
    return text, part, amount


def _is_portion_group(grp: ModifierGroup) -> bool:
    """Return True if every option in `grp` is a portioned topping/sauce option."""
    return bool(grp.options) and all(
        _PORTION_OPT_NAME_RE.match(o.name) is not None for o in grp.options.values()
    )


def _reachable_groups(store: Store, item: MenuItem) -> List[ModifierGroup]:
    """Return reachable modifier groups from item in BFS order."""
    visited: List[str] = []
    queue: List[str] = list(item.modifier_group_ids)
    while queue:
        gid = queue.pop(0)
        if gid in visited or gid not in store.modifier_groups:
            continue
        visited.append(gid)
        for opt in store.modifier_groups[gid].options.values():
            for child_gid in opt.child_group_ids:
                if child_gid not in visited and child_gid not in queue:
                    queue.append(child_gid)
    return [store.modifier_groups[gid] for gid in visited]


def _option_matches_query(opt: ModifierOption, q_clean: str, q_slug: str) -> bool:
    if opt.option_id == q_clean or _slugify(opt.option_id) == q_slug:
        return True
    if opt.name.lower() == q_clean.lower() or _slugify(opt.name) == q_slug:
        return True
    return any(
        a.lower() == q_clean.lower() or _slugify(a) == q_slug for a in opt.aliases
    )


def _group_matches_portion_name(grp: ModifierGroup, target: str) -> bool:
    t_clean = target.strip().lower()
    t_slug = _slugify(t_clean)
    return (
        grp.group_id.lower() == t_clean
        or _slugify(grp.group_id) == t_slug
        or grp.name.lower() == t_clean
        or _slugify(grp.name) == t_slug
    )


def _named_groups(
    candidate_groups: Dict[str, ModifierGroup], raw_gid: str
) -> Optional[Dict[str, ModifierGroup]]:
    """The candidate groups a group_id names (an id, or a group name without its number), or None."""
    gid = (raw_gid or "").strip()
    if not gid:
        return None
    if gid in candidate_groups:
        return {gid: candidate_groups[gid]}
    slug = _slugify(gid)
    named = {
        g: grp for g, grp in candidate_groups.items() if _slugify(grp.name) == slug
    }
    return named or None


def _best_option_match(
    grp: ModifierGroup, q_clean: str, q_slug: str
) -> Optional[ModifierOption]:
    """The option of `grp` a query names: an exact name first, then an id, slug, or alias."""
    for opt in grp.options.values():
        if opt.name.lower() == q_clean.lower():
            return opt
    for opt in grp.options.values():
        if _option_matches_query(opt, q_clean, q_slug):
            return opt
    return None


def _match_option_in_groups(
    raw_oid: str,
    raw_gid: str,
    candidate_groups: Dict[str, ModifierGroup],
) -> tuple[List[tuple[str, str]], str]:
    """(group_id, option_id) for each candidate group that offers the option, and the scope used."""
    oid_clean = raw_oid.strip()
    oid_slug = _slugify(oid_clean)

    def offered(groups: Dict[str, ModifierGroup]) -> List[tuple[str, str]]:
        found: List[tuple[str, str]] = []
        for gid, grp in groups.items():
            opt = _best_option_match(grp, oid_clean, oid_slug)
            if opt is not None:
                found.append((gid, opt.option_id))
        return found

    named = _named_groups(candidate_groups, raw_gid)
    if named is not None:
        return offered(named), "group"
    return offered(candidate_groups), "name"


def _ingredient_key(name: str) -> str:
    """An option name without amount or half wording ('Extra Pickles' -> 'pickles')."""
    key = name.strip().lower()
    key = re.sub(r"\s*-\s*(first|second) half\b", "", key)
    key = _AMOUNT_SUFFIX_RE.sub("", key)
    key = re.sub(r"^(extra|light|add|regular|normal)\s+", "", key)
    return key.strip()


class _LineItemBuilder:
    """Resolves one menu item's modifier selections and builds its priced line item.

    Use one instance per line item: the selections, removed defaults, and active groups
    found while resolving live on the instance.
    """

    def __init__(self, store: Store, item: MenuItem, display: Dict[str, str]) -> None:
        self.store = store
        self.item = item
        self.display = display
        self.reachable = {g.group_id: g for g in _reachable_groups(store, item)}
        self.selected_by_gid: Dict[str, List[str]] = {}
        self.cleared: Dict[str, set[str]] = {}
        self.active_group_ids: List[str] = []
        self.effective_defaults: Dict[str, List[str]] = {}

    _ALLOWED_MODIFIER_KEYS = frozenset(
        {
            "option_id",
            "group_id",
            "quantity",
            "placement",
            "volume",
            # Tolerate SelectedModifier output fields when an agent echoes cart state back.
            "name",
            "unit_price_cents",
        }
    )

    def build(
        self,
        line_item_id: str,
        quantity: int,
        raw_modifiers: Optional[List[ModifierSelection | Dict[str, Any]]],
    ) -> OrderLineItem:
        """Validates the selections and builds the order line item."""
        entries = self._normalize_entries(raw_modifiers)
        resolved, ambiguous, by_name_only = self._resolve_entries(entries)
        self._refresh_active_groups()
        self._raise_if_ambiguous(entries, resolved, ambiguous, by_name_only)
        self._raise_if_unresolved(entries, resolved)
        selections = [
            r for r in resolved if r is not None and r.get("action") == "select"
        ]
        self._check_contradictions(selections)
        by_group = self._group_selections(selections)
        if self.store.models_default_ingredients:
            self._drop_noop_removals(by_group)
        return self._priced_line(
            line_item_id, quantity, self._apply_defaults_and_limits(by_group)
        )

    def _normalize_entries(
        self, raw_modifiers: Optional[List[ModifierSelection | Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Normalizes each modifier entry's fields and resolves display group names to group IDs."""
        if not raw_modifiers:
            return []
        by_display = {name.lower(): gid for gid, name in self.display.items()}
        entries: List[Dict[str, Any]] = []
        for idx, item_raw in enumerate(raw_modifiers, start=1):
            if isinstance(item_raw, ModifierSelection):
                raw: Dict[str, Any] = item_raw.model_dump()
            elif isinstance(item_raw, dict):
                unknown = sorted(set(item_raw.keys()) - self._ALLOWED_MODIFIER_KEYS)
                if unknown:
                    raise ValueError(
                        f"Modifier {idx}: unknown key(s) {unknown}; use option_id, group_id, quantity, placement, volume."
                    )
                raw = item_raw
            else:
                raise ValueError(
                    f"Modifier {idx} must be an object with 'option_id', got {type(item_raw).__name__}."
                )
            raw_qty = raw.get("quantity", 1)
            if isinstance(raw_qty, bool) or not isinstance(raw_qty, int):
                try:
                    if isinstance(raw_qty, float) and not raw_qty.is_integer():
                        raise ValueError
                    mod_qty = int(raw_qty)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Modifier {idx} 'quantity' must be an integer, got {raw_qty!r}."
                    ) from None
            else:
                mod_qty = raw_qty
            gid = str(raw.get("group_id", "") or "").strip()
            if (
                gid
                and gid not in self.store.modifier_groups
                and gid.lower() in by_display
            ):
                gid = by_display[gid.lower()]
            option_text = str(raw.get("option_id", "") or "").strip()
            if mod_qty < 0:
                raise ValueError(
                    f"Modifier quantity must be at least 0 for option '{option_text}', got {mod_qty}."
                )
            placement = str(raw.get("placement") or "").strip().lower() or "whole"
            if placement not in ("whole", "left", "right"):
                raise ValueError(
                    f"Invalid placement '{placement}' for option '{option_text}'. Must be 'whole', 'left', or 'right'."
                )
            volume = str(raw.get("volume") or "").strip().lower() or "normal"
            if volume not in ("light", "normal", "extra"):
                raise ValueError(
                    f"Invalid volume '{volume}' for option '{option_text}'. Must be 'light', 'normal', or 'extra'."
                )
            entries.append(
                {
                    "group_id": gid,
                    "option_id": option_text,
                    "quantity": mod_qty,
                    "placement": placement,
                    "volume": volume,
                }
            )
        return entries

    def _chosen_option_ids(
        self, gid: str, effective_defaults: Dict[str, List[str]]
    ) -> List[str]:
        """A group's selected options, or, with no selection, its defaults minus removed ones."""
        return self.selected_by_gid.get(gid) or [
            o
            for o in effective_defaults.get(gid, [])
            if o not in self.cleared.get(gid, set())
        ]

    def _refresh_active_groups(self) -> None:
        """Recomputes the active modifier groups and effective defaults from the selections."""
        store, item = self.store, self.item
        active_group_ids: List[str] = [
            gid for gid in item.modifier_group_ids if gid in store.modifier_groups
        ]
        effective_defaults: Dict[str, List[str]] = {}
        for gid in active_group_ids:
            if gid in item.group_defaults:
                effective_defaults[gid] = list(item.group_defaults[gid])
            elif store.modifier_groups[gid].default_option_ids:
                effective_defaults[gid] = list(
                    store.modifier_groups[gid].default_option_ids
                )

        default_writer: Dict[str, tuple[str, str]] = {}
        changed = True
        while changed:
            changed = False
            for gid in list(active_group_ids):
                grp = store.modifier_groups.get(gid)
                if not grp:
                    continue
                chosen = set(self._chosen_option_ids(gid, effective_defaults))

                for oid in [o for o in grp.options if o in chosen]:
                    opt = grp.options[oid]
                    for child_gid, child_defs in opt.child_group_defaults.items():
                        if (
                            child_gid not in store.modifier_groups
                            or child_gid in default_writer
                        ):
                            continue
                        default_writer[child_gid] = (gid, oid)
                        if effective_defaults.get(child_gid) != list(child_defs):
                            effective_defaults[child_gid] = list(child_defs)
                            changed = True
                    for child_gid in opt.child_group_ids:
                        if child_gid not in store.modifier_groups:
                            continue
                        if child_gid not in active_group_ids:
                            active_group_ids.append(child_gid)
                            if (
                                child_gid not in effective_defaults
                                and store.modifier_groups[child_gid].default_option_ids
                            ):
                                effective_defaults[child_gid] = list(
                                    store.modifier_groups[child_gid].default_option_ids
                                )
                            changed = True

        self.active_group_ids = active_group_ids
        self.effective_defaults = effective_defaults

    def _resolve_entries(
        self, entries: List[Dict[str, Any]]
    ) -> tuple[List[Optional[Dict[str, Any]]], Dict[int, List[str]], Dict[int, str]]:
        """Resolves each entry to a selection or a removed default, parents before children.

        Returns the resolution per entry (None if unresolved), the entries whose option is
        offered by several groups with no group_id picking one, and the entries resolved by
        option name alone ('portion' or 'option'), which are rechecked once all are resolved.
        """
        store, item = self.store, self.item
        resolved: List[Optional[Dict[str, Any]]] = [None] * len(entries)
        ambiguous: Dict[int, List[str]] = {}
        by_name_only: Dict[int, str] = {}
        uses_portions = any(_is_portion_group(g) for g in self.reachable.values())

        # Pass 0 resolves against the item's own groups; later passes against every group the
        # selections so far make active. A later pass that resolves nothing new ends the loop.
        pass_idx = 0
        while True:
            self._refresh_active_groups()
            effective_defaults = self.effective_defaults
            if pass_idx == 0:
                candidate_gids = [
                    gid
                    for gid in item.modifier_group_ids
                    if gid in store.modifier_groups
                ]
            else:
                candidate_gids = self.active_group_ids

            candidate_groups = {
                gid: store.modifier_groups[gid]
                for gid in candidate_gids
                if gid in store.modifier_groups
            }

            any_new = False
            for idx, raw in enumerate(entries):
                if resolved[idx] is not None:
                    continue
                opt_id_raw = raw["option_id"]
                grp_id_raw = raw["group_id"]
                if not opt_id_raw and not grp_id_raw:
                    raise ValueError("Each modifier entry must include an 'option_id'.")
                reachable_named = _named_groups(self.reachable, grp_id_raw)
                if reachable_named is not None and not (
                    reachable_named.keys() & candidate_groups.keys()
                ):
                    continue

                if uses_portions:
                    dom_res = self._resolve_portion(raw, candidate_groups)
                    if dom_res is not None:
                        if dom_res["action"] == "ambiguous":
                            ambiguous[idx] = dom_res["group_ids"]
                            continue
                        ambiguous.pop(idx, None)
                        if not dom_res.get("qualified"):
                            by_name_only[idx] = "portion"
                        if dom_res["action"] == "clear_default":
                            self.cleared.setdefault(dom_res["group_id"], set()).update(
                                effective_defaults.get(dom_res["group_id"], [])
                            )
                        else:
                            self.selected_by_gid.setdefault(
                                dom_res["group_id"], []
                            ).append(dom_res["option_id"])
                        resolved[idx] = dom_res
                        any_new = True
                        continue

                matches, scope = _match_option_in_groups(
                    opt_id_raw, grp_id_raw, candidate_groups
                )
                if raw["quantity"] == 0 and len(matches) != 1:
                    # Removing an ingredient that is only listed as its 'No ...' option, or
                    # whose name several groups offer while one group offers its 'No ...'.
                    no_matches, no_scope = _match_option_in_groups(
                        f"No {opt_id_raw}", grp_id_raw, candidate_groups
                    )
                    if len(no_matches) == 1 or (no_matches and not matches):
                        matches, scope = no_matches, no_scope
                        raw = {**raw, "option_id": f"No {opt_id_raw}", "quantity": 1}
                        entries[idx] = raw
                if len(matches) > 1:
                    ambiguous[idx] = [gid for gid, _ in matches]
                    continue
                if not matches:
                    continue
                ambiguous.pop(idx, None)
                grp_id, opt_id = matches[0]
                if scope == "name":
                    by_name_only[idx] = "option"
                resolved[idx] = self._resolve_option(
                    raw, grp_id, opt_id, effective_defaults
                )
                any_new = True

            if all(r is not None for r in resolved):
                break
            if pass_idx >= 1 and not any_new:
                break
            pass_idx += 1
        return resolved, ambiguous, by_name_only

    def _resolve_option(
        self,
        raw: Dict[str, Any],
        grp_id: str,
        opt_id: str,
        effective_defaults: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Validates an entry matched to one option, records it, and returns its resolution."""
        item = self.item
        placement = raw["placement"]
        grp_obj = self.store.modifier_groups[grp_id]
        opt_obj = grp_obj.options[opt_id]
        volume = raw["volume"]
        if volume != "normal" and not grp_obj.supports_half_and_half:
            raise ValueError(
                f"A '{volume}' amount of '{opt_obj.name}' is not offered on '{item.name}'. "
                "Only the listed modifiers can be ordered."
            )
        mod_qty = raw["quantity"]
        if mod_qty == 0:
            return self._resolve_removal(grp_id, opt_id, effective_defaults)

        self.selected_by_gid.setdefault(grp_id, []).append(opt_id)
        return {
            "action": "select",
            "group_id": grp_id,
            "option_id": opt_id,
            "quantity": mod_qty,
            "placement": placement,
            "volume": volume,
        }

    def _resolve_removal(
        self, grp_id: str, opt_id: str, effective_defaults: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Resolves quantity 0 on an option the same way as choosing its 'No ...' option.

        That is the 'No <name>' option in the option's group or, failing that, in another active
        group; for the only default of a pick-one group, the group's single 'No ...' option.
        Without one, quantity 0 clears a default of an optional group.
        """
        item = self.item
        grp_obj = self.store.modifier_groups[grp_id]
        opt_obj = grp_obj.options[opt_id]
        defaults = effective_defaults.get(grp_id, [])
        no_name = f"no {opt_obj.name.lower()}"

        def no_options(gid: str) -> List[ModifierOption]:
            return [
                o
                for o in self.store.modifier_groups[gid].options.values()
                if o.available and o.name.lower().startswith("no ")
            ]

        target: Optional[tuple[str, ModifierOption]] = next(
            ((grp_id, o) for o in no_options(grp_id) if o.name.lower() == no_name), None
        )
        if target is None:
            elsewhere = [
                (gid, o)
                for gid in self.active_group_ids
                if gid != grp_id
                for o in no_options(gid)
                if o.name.lower() == no_name
            ]
            if len(elsewhere) == 1:
                target = elsewhere[0]
        if (
            target is None
            and grp_obj.max_select == 1
            and defaults == [opt_id]
            and len(no_options(grp_id)) == 1
        ):
            target = (grp_id, no_options(grp_id)[0])
        if target is not None:
            no_gid, no_opt = target
            self.selected_by_gid.setdefault(no_gid, []).append(no_opt.option_id)
            return {
                "action": "select",
                "group_id": no_gid,
                "option_id": no_opt.option_id,
                "quantity": 1,
                "placement": "whole",
                "volume": "normal",
            }
        if opt_id not in defaults:
            raise ValueError(
                f"'{opt_obj.name}' is not on '{item.name}' by default and has no "
                f"'No {opt_obj.name}' option, so there is nothing to remove."
            )
        if grp_obj.min_select == 0:
            self.cleared.setdefault(grp_id, set()).add(opt_id)
            return {"action": "clear_default", "group_id": grp_id}
        raise ValueError(
            f"'{opt_obj.name}' is a required choice on '{item.name}' and cannot be removed; choose another option instead."
        )

    def _resolve_portion(
        self,
        raw: Dict[str, Any],
        candidate_groups: Dict[str, ModifierGroup],
    ) -> Optional[Dict[str, Any]]:
        """Resolve portioned group removals ('No <Group>') and part/volume option selection.

        `raw` is an entry normalized by `_normalize_entries`.
        """
        item = self.item
        effective_defaults = self.effective_defaults
        raw_oid = raw["option_id"]
        raw_gid = raw["group_id"]
        placement = raw["placement"]
        volume = raw["volume"]
        qty = raw["quantity"]
        oid_lower = raw_oid.lower()
        oid_slug = _slugify(raw_oid)
        named = _named_groups(candidate_groups, raw_gid)
        display = self.display

        def scoped(
            match: Callable[[ModifierGroup], bool],
        ) -> tuple[List[ModifierGroup], bool]:
            if named is not None:
                return [g for g in named.values() if match(g)], True
            return [g for g in candidate_groups.values() if match(g)], False

        def ambiguous(groups: List[ModifierGroup]) -> Dict[str, Any]:
            return {"action": "ambiguous", "group_ids": [g.group_id for g in groups]}

        explicit_option = qty >= 1 and any(
            _option_matches_query(opt, raw_oid, oid_slug)
            for grp in candidate_groups.values()
            for opt in grp.options.values()
        )
        if not explicit_option and (qty == 0 or oid_lower.startswith("no ")):
            target_name = (
                raw_oid[3:].strip() if oid_lower.startswith("no ") else raw_oid
            )
            target_name, half_part, _ = _split_portion_suffix(target_name)
            t_slug = _slugify(target_name)
            if target_name:
                groups, qualified = scoped(
                    lambda g: _is_portion_group(g)
                    and (
                        _group_matches_portion_name(g, target_name)
                        or any(
                            _option_matches_query(o, target_name, t_slug)
                            for o in g.options.values()
                        )
                    )
                )
            elif named is not None:
                groups, qualified = (
                    [g for g in named.values() if _is_portion_group(g)],
                    True,
                )
            else:
                groups, qualified = [], False
            if len(groups) > 1:
                return ambiguous(groups)
            if groups:
                matched_grp = groups[0]
                group_label = display.get(matched_grp.group_id, matched_grp.name)
                if matched_grp.min_select == 0 and effective_defaults.get(
                    matched_grp.group_id
                ):
                    matched_opts = (
                        [
                            o
                            for o in matched_grp.options.values()
                            if _option_matches_query(o, target_name, t_slug)
                        ]
                        if target_name
                        and not _group_matches_portion_name(matched_grp, target_name)
                        else []
                    )
                    matched_half_only = bool(matched_opts) and all(
                        (_parse_portion_option_name(o.name) or ("whole",))[0]
                        in ("left", "right")
                        for o in matched_opts
                    )
                    if (
                        half_part in ("left", "right")
                        or placement in ("left", "right")
                        or matched_half_only
                    ):
                        raise ValueError(
                            f"Cannot remove '{raw_oid}' from one half of '{item.name}'; "
                            f"select '{group_label}' on the half you want to keep instead."
                        )
                    return {
                        "action": "clear_default",
                        "group_id": matched_grp.group_id,
                        "qualified": qualified,
                    }
                raise ValueError(
                    f"Cannot remove '{raw_oid}' from '{item.name}': modifier group '{group_label}' is not a removable default topping/sauce on this item."
                )

        explicit_part: Optional[str] = None
        explicit_amt: Optional[str] = None
        base_query = raw_oid
        target_grp: Optional[ModifierGroup] = None

        direct, qualified = scoped(
            lambda g: _is_portion_group(g) and raw_oid in g.options
        )
        if len(direct) > 1:
            return ambiguous(direct)
        if direct:
            parsed = _parse_portion_option_name(direct[0].options[raw_oid].name)
            if parsed is not None:
                target_grp, qualified = direct[0], True
                explicit_part, explicit_amt = parsed

        if target_grp is None:
            base_query, explicit_part, explicit_amt = _split_portion_suffix(base_query)
            b_slug = _slugify(base_query)

            def offers(g: ModifierGroup) -> bool:
                return _is_portion_group(g) and (
                    _group_matches_portion_name(g, base_query)
                    or any(
                        _option_matches_query(o, raw_oid, oid_slug)
                        or _option_matches_query(o, base_query, b_slug)
                        for o in g.options.values()
                    )
                )

            groups, qualified = scoped(offers)
            if len(groups) > 1:
                return ambiguous(groups)
            target_grp = groups[0] if groups else None
            if (
                target_grp is not None
                and explicit_part is None
                and explicit_amt is None
            ):
                matched_opt = _best_option_match(target_grp, raw_oid, oid_slug)
                if matched_opt is not None:
                    parsed = _parse_portion_option_name(matched_opt.name)
                    if parsed is not None:
                        explicit_part, explicit_amt = parsed

        if target_grp is not None and _is_portion_group(target_grp):
            if explicit_part == "whole" and placement in ("left", "right"):
                explicit_part = None
            if explicit_amt == "normal" and volume in ("light", "extra"):
                explicit_amt = None
            by_part_amt = {
                parsed: o.option_id
                for o in target_grp.options.values()
                if (parsed := _parse_portion_option_name(o.name)) is not None
            }
            group_label = display.get(target_grp.group_id, target_grp.name)
            part = explicit_part or (
                placement if placement in ("left", "right") else "whole"
            )
            amt = explicit_amt or (volume if volume in ("light", "extra") else "normal")
            if part in ("left", "right") and not any(p == part for p, _ in by_part_amt):
                raise ValueError(
                    f"Modifier group '{group_label}' on '{item.name}' does not support half-and-half placement ('{part}')."
                )
            if amt in ("light", "extra") and not any(a == amt for _, a in by_part_amt):
                raise ValueError(
                    f"Modifier group '{group_label}' on '{item.name}' does not support '{amt}' portion volume."
                )
            target_oid = by_part_amt.get((part, amt))
            if target_oid is not None and qty >= 1:
                return {
                    "action": "select",
                    "group_id": target_grp.group_id,
                    "option_id": target_oid,
                    "quantity": qty,
                    "placement": "whole",
                    "volume": "normal",
                    "qualified": qualified,
                }

        return None

    def _chosen_parents(self) -> Dict[str, List[str]]:
        """Maps each nested group to the names of the chosen options that open it."""
        parents: Dict[str, List[str]] = {}
        for gid in self.active_group_ids:
            grp = self.store.modifier_groups[gid]
            chosen = self._chosen_option_ids(gid, self.effective_defaults)
            for oid in dict.fromkeys(chosen):
                opt = grp.options.get(oid)
                if opt is None:
                    continue
                for cgid in opt.child_group_ids:
                    names = parents.setdefault(cgid, [])
                    if opt.name not in names:
                        names.append(opt.name)
        return parents

    def _ambiguity_error(
        self,
        option_text: str,
        group_ids: List[str],
        parents: Dict[str, List[str]],
    ) -> ValueError:
        """Names every group offering the option, with the chosen option that opened it."""
        labels: List[str] = []
        for gid in dict.fromkeys(group_ids):
            label = f"'{self.display.get(gid, self.store.modifier_groups[gid].name)}'"
            if parents.get(gid):
                label += " (for " + ", ".join(f"'{p}'" for p in parents[gid][:2]) + ")"
            labels.append(label)
        return ValueError(
            f"'{option_text}' is offered in more than one group on '{self.item.name}': "
            f"{', '.join(labels)}. Pass the intended group name as group_id."
        )

    def _raise_if_ambiguous(
        self,
        entries: List[Dict[str, Any]],
        resolved: List[Optional[Dict[str, Any]]],
        ambiguous: Dict[int, List[str]],
        by_name_only: Dict[int, str],
    ) -> None:
        """An option offered by more than one applicable group needs a group_id; never pick one."""
        parents = self._chosen_parents()
        final_groups = {
            gid: self.store.modifier_groups[gid] for gid in self.active_group_ids
        }
        for idx, raw in enumerate(entries):
            opt_text = raw["option_id"]
            if resolved[idx] is None and idx in ambiguous:
                live = [g for g in ambiguous[idx] if g in final_groups]
                raise self._ambiguity_error(opt_text, live or ambiguous[idx], parents)
            if idx not in by_name_only:
                continue
            if by_name_only[idx] == "portion":
                again = self._resolve_portion({**raw, "group_id": ""}, final_groups)
                others = (
                    again["group_ids"]
                    if again is not None and again["action"] == "ambiguous"
                    else []
                )
            else:
                found, _scope = _match_option_in_groups(opt_text, "", final_groups)
                others = [gid for gid, _ in found] if len(found) > 1 else []
            if others:
                raise self._ambiguity_error(opt_text, others, parents)

    def _raise_if_unresolved(
        self,
        entries: List[Dict[str, Any]],
        resolved: List[Optional[Dict[str, Any]]],
    ) -> None:
        """Raises for the first entry that no applicable group offers."""
        final_groups = {
            gid: self.store.modifier_groups[gid] for gid in self.active_group_ids
        }
        for raw, res_entry in zip(entries, resolved):
            if res_entry is not None:
                continue
            opt_id_raw = raw["option_id"]
            named = _named_groups(final_groups, raw["group_id"])
            if named is not None:
                labels = ", ".join(f"'{self.display.get(g, g)}'" for g in named)
                raise ValueError(
                    f"'{opt_id_raw}' is not in {labels} on '{self.item.name}'."
                )
            raise ValueError(
                f"Option '{opt_id_raw}' is not a valid modifier option for item '{self.item.name}'."
            )

    def _check_contradictions(self, selections: List[Dict[str, Any]]) -> None:
        """Passing both 'X' and 'No X' on the same item or sub-item is a contradiction."""
        store = self.store
        group_scopes: Dict[str, set[str]] = {
            gid: {"item"} for gid in self.item.modifier_group_ids
        }
        for pgid in self.active_group_ids:
            pgrp = store.modifier_groups[pgid]
            for poid in self._chosen_option_ids(pgid, self.effective_defaults):
                popt = pgrp.options.get(poid)
                if popt is not None:
                    for cgid in popt.child_group_ids:
                        group_scopes.setdefault(cgid, set()).add(f"{pgid}:{poid}")

        chosen_opts = [
            (
                e["group_id"],
                store.modifier_groups[e["group_id"]].options[e["option_id"]],
            )
            for e in selections
        ]
        for _gid, no_opt in chosen_opts:
            if not no_opt.name.lower().startswith("no "):
                continue
            target = _ingredient_key(no_opt.name[3:])
            no_scope = group_scopes.get(_gid, {_gid})
            clash = next(
                (
                    o
                    for _cgid, o in chosen_opts
                    if (
                        _cgid == _gid
                        or bool(no_scope & group_scopes.get(_cgid, {_cgid}))
                    )
                    and not o.name.lower().startswith("no ")
                    and not o.child_group_ids
                    and (
                        _ingredient_key(o.name) == target
                        or (
                            _cgid == _gid
                            and target
                            in ("dipping sauce", "sauce", "dressing", "syrup")
                            and (
                                target in _ingredient_key(o.name)
                                or any(
                                    k in store.modifier_groups[_gid].name.lower()
                                    for k in ("sauce", "dressing", "syrup", "flavor")
                                )
                            )
                        )
                    )
                ),
                None,
            )
            if clash is not None:
                raise ValueError(
                    f"'{clash.name}' and '{no_opt.name}' cannot both be selected on '{self.item.name}'."
                )

    def _group_selections(
        self, selections: List[Dict[str, Any]]
    ) -> Dict[str, List[SelectedModifier]]:
        """Groups selections by group_id, merging identical (option_id, placement, volume) entries."""
        store, item = self.store, self.item
        selections_by_group: Dict[str, List[SelectedModifier]] = {
            gid: [] for gid in self.active_group_ids
        }
        for entry in selections:
            gid = entry["group_id"]
            oid = entry["option_id"]
            placement = entry["placement"]
            volume = entry["volume"]
            mod_qty = entry["quantity"]
            grp = store.modifier_groups[gid]
            group_label = self.display.get(gid, grp.name)
            if gid not in selections_by_group:
                raise ValueError(
                    f"Modifier group '{group_label}' is only applicable when its parent option is selected."
                )
            opt: ModifierOption = grp.options[oid]
            if not opt.available:
                raise ValueError(
                    f"Modifier option '{opt.name}' in group '{group_label}' is currently unavailable / out of stock."
                )
            if placement in ("left", "right") and not grp.supports_half_and_half:
                raise ValueError(
                    f"Modifier group '{group_label}' on '{item.name}' does not support left/right half placement ('{placement}')."
                )

            existing = next(
                (
                    m
                    for m in selections_by_group[gid]
                    if m.option_id == oid
                    and m.placement == placement
                    and m.volume == volume
                ),
                None,
            )
            if existing is not None:
                existing.quantity += mod_qty
            else:
                selections_by_group[gid].append(
                    SelectedModifier(
                        group_id=gid,
                        option_id=oid,
                        name=opt.name,
                        quantity=mod_qty,
                        placement=placement,  # type: ignore[arg-type]
                        volume=volume,  # type: ignore[arg-type]
                        unit_price_cents=opt.price_cents,
                    )
                )
        return selections_by_group

    def _drop_noop_removals(
        self, selections_by_group: Dict[str, List[SelectedModifier]]
    ) -> None:
        """Drops each $0 'No X' that changes nothing on stores that model recipe defaults.

        That is when X is neither selected nor a default on this line, or when another
        selection in X's group already replaces the group's defaults.
        """
        for gid, group_mods in selections_by_group.items():
            grp = self.store.modifier_groups[gid]
            by_name = {o.name.lower(): oid for oid, o in grp.options.items()}
            chosen = {m.option_id for m in group_mods}
            defaults = set(self.effective_defaults.get(gid, []))
            has_no_mod = any(
                grp.options[m.option_id].name.lower().startswith("no ")
                for m in group_mods
            ) or bool(self.cleared.get(gid))
            replaces_defaults = any(
                not grp.options[m.option_id].name.lower().startswith("no ")
                for m in group_mods
            ) and not (len(defaults) > 1 and has_no_mod)
            kept = []
            for m in group_mods:
                opt = grp.options[m.option_id]
                target = (
                    by_name.get(opt.name[3:].lower())
                    if opt.name.lower().startswith("no ")
                    else None
                )
                if (
                    opt.price_cents == 0
                    and grp.min_select == 0
                    and target is not None
                    and target not in chosen
                    and (target not in defaults or replaces_defaults)
                ):
                    continue
                kept.append(m)
            selections_by_group[gid] = kept

    def _apply_defaults_and_limits(
        self, selections_by_group: Dict[str, List[SelectedModifier]]
    ) -> List[SelectedModifier]:
        """Adds remaining defaults to groups without a positive selection and checks limits."""
        store, item = self.store, self.item
        all_selected: List[SelectedModifier] = []
        for gid in self.active_group_ids:
            grp = store.modifier_groups[gid]
            group_mods = selections_by_group[gid]
            raw_defaults = self.effective_defaults.get(gid, [])
            removed_here = self.cleared.get(gid, set())
            has_no_mods = any(
                grp.options[m.option_id].name.lower().startswith("no ")
                for m in group_mods
            )
            only_no_mods = bool(group_mods) and all(
                grp.options[m.option_id].name.lower().startswith("no ")
                for m in group_mods
            )
            preserve_multi_defaults = (
                len(set(raw_defaults)) > 1
                and (has_no_mods or bool(removed_here))
                and grp.max_select > 1
            )
            # In a pick-one group a 'No ...' choice (e.g. 'No Ice') replaces the default.
            if raw_defaults and (
                not group_mods
                or (
                    store.models_default_ingredients
                    and (only_no_mods or preserve_multi_defaults)
                    and grp.max_select > 1
                )
            ):
                already_selected = {m.option_id for m in group_mods}
                chosen_no_names = {
                    grp.options[m.option_id].name.lower() for m in group_mods
                }
                for def_oid in raw_defaults:
                    if (
                        def_oid in removed_here
                        or def_oid not in grp.options
                        or (group_mods and def_oid in already_selected)
                    ):
                        continue
                    def_opt = grp.options[def_oid]
                    if f"no {def_opt.name.lower()}" in chosen_no_names:
                        continue
                    # A repeated default id is a recipe quantity (e.g. 2 shots): one entry, qty n.
                    repeated = next(
                        (m for m in group_mods if m.option_id == def_oid), None
                    )
                    if repeated is not None:
                        repeated.quantity += 1
                        continue
                    group_mods.append(
                        SelectedModifier(
                            group_id=gid,
                            option_id=def_oid,
                            name=def_opt.name,
                            quantity=1,
                            placement="whole",
                            volume="normal",
                            unit_price_cents=def_opt.price_cents,
                        )
                    )

            qty_by_option: Dict[str, int] = {}
            for m in group_mods:
                qty_by_option[m.option_id] = (
                    qty_by_option.get(m.option_id, 0) + m.quantity
                )
            group_label = self.display.get(gid, grp.name)
            if _is_portion_group(grp):
                parts: List[str] = []
                for m in group_mods:
                    parsed = _parse_portion_option_name(grp.options[m.option_id].name)
                    p = parsed[0] if parsed is not None else m.placement
                    parts.extend([p] * m.quantity)
                if len(parts) != len(set(parts)) or (
                    "whole" in parts and len(parts) > 1
                ):
                    raise ValueError(
                        f"Modifier group '{group_label}' on '{item.name}' can be chosen at most once for the whole pizza or once per half."
                    )
            for oid, opt_qty in qty_by_option.items():
                if opt_qty > grp.max_per_option:
                    opt_name = grp.options[oid].name
                    raise ValueError(
                        f"Modifier option '{opt_name}' quantity {opt_qty} exceeds max_per_option={grp.max_per_option} in group '{group_label}'."
                    )

            total_selected = sum(m.quantity for m in group_mods)
            if total_selected < grp.min_select:
                raise ValueError(
                    f"Modifier group '{group_label}' on '{item.name}' requires at least {grp.min_select} selection(s), got {total_selected}."
                )
            if total_selected > grp.max_select:
                raise ValueError(
                    f"Modifier group '{group_label}' on '{item.name}' allows at most {grp.max_select} total selection(s), got {total_selected}."
                )
            all_selected.extend(group_mods)
        return all_selected

    def _priced_line(
        self, line_item_id: str, quantity: int, modifiers: List[SelectedModifier]
    ) -> OrderLineItem:
        """The line item with its modifiers in canonical order and its total price."""
        item = self.item
        modifiers.sort(
            key=lambda m: (
                m.group_id,
                m.option_id,
                m.placement,
                m.volume,
                m.quantity,
            )
        )
        modifiers_unit_cents = sum(m.unit_price_cents * m.quantity for m in modifiers)
        line_total_cents = quantity * (item.base_price_cents + modifiers_unit_cents)
        return OrderLineItem(
            line_item_id=line_item_id,
            item_id=item.item_id,
            name=item.name,
            quantity=quantity,
            unit_base_price_cents=item.base_price_cents,
            modifiers=modifiers,
            line_total_cents=line_total_cents,
        )
