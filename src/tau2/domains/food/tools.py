"""Toolkit for the food domain (Corner Cafe, City Pizza, and Sunset Burger)."""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from tau2.domains.food.canonical import _canonicalize_line_items
from tau2.domains.food.data_model import (
    CartState,
    FoodDB,
    MenuItem,
    ModifierGroup,
    ModifierOption,
    ModifierSelection,
    OrderLineItem,
    Reservation,
    ReservationPolicy,
    Store,
    SubmittedOrder,
)
from tau2.domains.food.menu_prompt import (
    group_display_names,
    not_available_for,
    schedule_summary,
    starting_price_cents,
)
from tau2.domains.food.modifiers import (
    _LineItemBuilder,
    _reachable_groups,
)
from tau2.domains.food.normalize import (
    _collapse,
    _date_key,
    _hhmm,
    _minutes,
    _normalize_address,
    _normalize_name,
    _normalize_order_type,
    _normalize_phone,
    _parse_clock,
    _phone_digits,
    _slugify,
    _truncate,
)
from tau2.domains.food.utils import FOOD_DEFAULT_NOW, format_cents
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.utils.utils import get_dict_hash

_LOOKUP_PAGE_SIZE = 20
_LOOKUP_GROUP_PAGE_SIZE = 25
_LOOKUP_INLINE_OPTION_CAP = 25
_LOOKUP_INLINE_GROUP_CAP = 12
_LOOKUP_NAME_CAP = 25
_LOOKUP_DESCRIPTION_CAP = 160
_LOOKUP_MODIFIER_MATCH_CAP = 10
_LOOKUP_MODIFIER_GROUP_CAP = 3


def _whole_number(value: Any, name: str) -> int:
    """An integer tool argument; whole-number floats such as 2.0 are accepted as integers."""
    if isinstance(value, bool) or not (
        isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    ):
        raise ValueError(f"'{name}' must be a whole number, got {value!r}.")
    return int(value)


class FoodTools(ToolKitBase):
    """All tools for the restaurant food domain (Corner Cafe, City Pizza, Sunset Burger)."""

    db: FoodDB

    def __init__(self, db: FoodDB) -> None:
        super().__init__(db)
        self._display_name_cache: Dict[str, Dict[str, str]] = {}

    def get_db_hash(self) -> str:
        """Hashes the session outcome that grading compares.

        The hash covers the active store, every submitted order in order (canonical items,
        order type, time, channel, and for mobile orders the name, phone digits, and delivery
        address), the cart (empty once transferred), confirmed reservations by content, and
        whether the session was transferred to a human. Line items are canonicalized, so
        equivalent ways of building the same items hash equal, while a missing, extra, or
        duplicate order, or items left in the cart, change the hash.
        """
        active_store_id = self.db.cart.store_id
        if not active_store_id and self.db.orders:
            active_store_id = list(self.db.orders.values())[-1].store_id
        active_store = self.db.stores.get(active_store_id or "")

        def canon(
            items: List[OrderLineItem], store: Optional[Store] = active_store
        ) -> List[Dict[str, Any]]:
            return [
                li.model_dump() for li in _canonicalize_line_items(items, store=store)
            ]

        def order_state(order: SubmittedOrder) -> Dict[str, Any]:
            ord_store = self.db.stores.get(order.store_id) or active_store
            state: Dict[str, Any] = {
                "items": canon(order.items, store=ord_store),
                "order_type": order.order_type,
                "scheduled_time": order.scheduled_time,
                "channel": order.channel,
            }
            # Who and where only matter when ordering ahead; at the drive-thru they are optional.
            if order.channel == "mobile":
                state["customer_name"] = _collapse(order.customer_name)
                state["customer_phone"] = _phone_digits(order.customer_phone) or re.sub(
                    r"\D", "", order.customer_phone
                )
                if order.order_type == "delivery":
                    state["delivery_address"] = _collapse(
                        _normalize_address(order.delivery_address)
                        if (order.delivery_address or "").strip()
                        else ""
                    )
            return state

        # Confirmed reservations by content: a booking made and cancelled in the same call counts
        # as never made, a cancelled existing booking counts as removed.
        reservations = sorted(
            (
                {
                    "store_id": r.store_id,
                    "date": _date_key(r.date),
                    "time": r.time,
                    "party_size": r.party_size,
                    "customer_name": _collapse(r.customer_name),
                    "customer_phone": _phone_digits(r.customer_phone)
                    or re.sub(r"\D", "", r.customer_phone),
                    "seating": r.seating,
                    "high_chairs": r.high_chairs,
                }
                for r in self.db.reservations.values()
                if r.status == "confirmed"
            ),
            key=lambda r: json.dumps(r, sort_keys=True),
        )
        transferred = bool(self.db.cart.transferred_to_human)
        normalized_state = {
            "store_id": active_store_id or "",
            "orders": [order_state(order) for order in self.db.orders.values()],
            "cart": [] if transferred else canon(self.db.cart.items),
            "reservations": reservations,
            "transferred": transferred,
        }
        return get_dict_hash(normalized_state)

    def update_db(self, update_data: Optional[dict[str, Any]] = None) -> None:
        """Merges 'cart' and 'reservations' updates into session state without touching menus."""
        if update_data is None:
            return
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        unknown = set(update_data) - {"cart", "reservations"}
        if unknown:
            raise ValueError(f"Unknown food DB keys: {sorted(unknown)}")
        if "cart" in update_data and isinstance(update_data["cart"], dict):
            cart_dict = self.db.cart.model_dump()
            cart_dict.update(update_data["cart"])
            self.db.cart = CartState.model_validate(cart_dict)
        if "reservations" in update_data and isinstance(
            update_data["reservations"], dict
        ):
            merged = dict(self.db.reservations)
            for k, v in update_data["reservations"].items():
                merged[k] = Reservation.model_validate(v)
            self.db.reservations = merged

    def _get_store(self, store_id: str) -> Store:
        key = _slugify(store_id or "").replace("-", "_")
        if key not in self.db.stores:
            for sid, s in self.db.stores.items():
                if _slugify(s.name).replace("-", "_") == key:
                    key = sid
                    break
        if key not in self.db.stores:
            available = sorted(self.db.stores.keys())
            raise ValueError(
                f"Store '{store_id}' not found. Available stores: {available}"
            )
        if self.db.cart.store_id is not None and key != self.db.cart.store_id:
            active = self.db.stores[self.db.cart.store_id]
            raise ValueError(
                f"Active session store is '{active.store_id}' ({active.name}); cannot query or order from '{store_id}'. Please pass store_id='{active.store_id}'."
            )
        return self.db.stores[key]

    def _now(self) -> datetime:
        try:
            return datetime.strptime(self.db.cart.current_time or "", "%Y-%m-%d %H:%M")
        except ValueError:
            return datetime.strptime(FOOD_DEFAULT_NOW, "%Y-%m-%d %H:%M")

    def _effective_time(self, scheduled_time: Optional[str] = None) -> str:
        """'YYYY-MM-DD HH:MM' the order is for: the scheduled time today, else now."""
        scheduled = (
            self.db.cart.scheduled_time if scheduled_time is None else scheduled_time
        )
        now = self._now()
        if scheduled and scheduled != "ASAP":
            return f"{now:%Y-%m-%d} {scheduled}"
        return f"{now:%Y-%m-%d %H:%M}"

    def _empty_cart(self) -> CartState:
        cart = self.db.cart
        return CartState(
            store_id=cart.store_id,
            current_time=cart.current_time,
            order_type=cart.order_type,
            channel=cart.channel,
            scheduled_time=cart.scheduled_time,
            transferred_to_human=cart.transferred_to_human,
        )

    def _check_item_availability(
        self,
        store: Store,
        item: MenuItem,
        order_type: Optional[str] = None,
        at_time: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Evaluate whether `item` is available given stock, schedule, and fulfillment method.

        The time is `at_time` ('YYYY-MM-DD HH:MM'), else the cart's scheduled time, else now.
        """
        if not item.available:
            return (
                False,
                f"Menu item '{item.name}' is currently unavailable / out of stock.",
            )

        eff_order_type = _normalize_order_type(
            order_type or self.db.cart.order_type or "pickup"
        )
        if (
            item.allowed_fulfillment_methods
            and eff_order_type not in item.allowed_fulfillment_methods
        ):
            allowed = " or ".join(
                m.replace("_", "-") for m in item.allowed_fulfillment_methods
            )
            return (
                False,
                f"Menu item '{item.name}' is not available for {eff_order_type.replace('_', '-')} orders (only {allowed}).",
            )

        if item.schedule_ids:
            cur_time_str = at_time or self._effective_time()
            try:
                if cur_time_str.endswith(" 24:00"):
                    dt = datetime.strptime(cur_time_str[:-6], "%Y-%m-%d")
                    time_str = "24:00"
                else:
                    dt = datetime.strptime(cur_time_str, "%Y-%m-%d %H:%M")
                    time_str = dt.strftime("%H:%M")
            except ValueError:
                dt = self._now()
                time_str = dt.strftime("%H:%M")
            day_name = dt.strftime("%A").upper()
            date_str = dt.strftime("%Y-%m-%d")

            matched_schedule = False
            for sid in item.schedule_ids:
                sched = store.schedules.get(sid)
                if not sched:
                    continue
                for rule in sched.rules:
                    if rule.days_of_week and day_name not in rule.days_of_week:
                        continue
                    if rule.dates and not any(
                        d.start_date <= date_str <= d.end_date for d in rule.dates
                    ):
                        continue
                    if rule.time_ranges and not any(
                        tr.start_time <= time_str < tr.end_time
                        for tr in rule.time_ranges
                    ):
                        continue
                    matched_schedule = True
                    break
                if matched_schedule:
                    break

            if not matched_schedule:
                sched_desc = schedule_summary(store, item) or "Unrestricted"
                when = (
                    f"current time {cur_time_str}"
                    if cur_time_str == f"{self._now():%Y-%m-%d %H:%M}"
                    else f"{time_str} on {dt:%A}"
                )
                return (
                    False,
                    f"Menu item '{item.name}' is not available at {when}. Scheduled availability: {sched_desc}.",
                )

        return True, "Available"

    def _get_item(
        self,
        store: Store,
        item_id_or_name: str,
    ) -> MenuItem:
        """Resolve an item by ID, display name, or alias."""
        raw = (item_id_or_name or "").strip()
        slug = _slugify(raw)

        exact_item = store.items.get(raw) or store.items.get(slug)
        if exact_item is not None:
            return exact_item

        for it in store.items.values():
            if (
                it.name.lower() == raw.lower()
                or _slugify(it.name) == slug
                or any(
                    a.lower() == raw.lower() or _slugify(a) == slug for a in it.aliases
                )
            ):
                return it

        raise ValueError(
            f"Item '{item_id_or_name}' not found in store '{store.store_id}'. Use the exact item name from the menu."
        )

    def _expand_item_modifier_groups(
        self,
        store: Store,
        item: MenuItem,
    ) -> List[ModifierGroup]:
        """Return reachable modifier groups from item in BFS order."""
        return _reachable_groups(store, item)

    def _group_display_names(self, store: Store) -> Dict[str, str]:
        if store.store_id not in self._display_name_cache:
            self._display_name_cache[store.store_id] = group_display_names(store)
        return self._display_name_cache[store.store_id]

    def _validate_and_build_line_item(
        self,
        store: Store,
        item: MenuItem,
        line_item_id: str,
        quantity: int,
        raw_modifiers: Optional[List[Dict[str, Any]]],
    ) -> OrderLineItem:
        """Validates the item and its modifier selections and builds the order line item."""
        if quantity < 1:
            raise ValueError(f"Quantity must be at least 1, got {quantity}.")
        is_avail, avail_reason = self._check_item_availability(store, item)
        if not is_avail:
            raise ValueError(avail_reason)
        builder = _LineItemBuilder(store, item, self._group_display_names(store))
        return builder.build(line_item_id, quantity, raw_modifiers)

    def _menu_entries(self, store: Store) -> List[tuple[MenuItem, str]]:
        """Each menu item paired with '' if available now or why it cannot be ordered."""
        entries: List[tuple[MenuItem, str]] = []
        for item in store.items.values():
            ok, why = self._check_item_availability(store, item)
            entries.append((item, "" if ok else why))
        return entries

    def _default_labels(self, grp: ModifierGroup, option_ids: List[str]) -> List[Any]:
        """Default choices as names, or {name, quantity} when a default is applied more than once."""
        counts: Dict[str, int] = {}
        for oid in option_ids:
            if oid in grp.options:
                counts[oid] = counts.get(oid, 0) + 1
        return [
            grp.options[oid].name
            if n == 1
            else {"name": grp.options[oid].name, "quantity": n}
            for oid, n in counts.items()
        ]

    def _option_entry(
        self,
        store: Store,
        opt: ModifierOption,
    ) -> Any:
        """A modifier as its bare name, or a dict when it has a price or opens nested groups."""
        display = self._group_display_names(store)
        entry: Dict[str, Any] = {"name": opt.name}
        if opt.price_cents:
            sign = "-" if opt.price_cents < 0 else "+"
            entry["price"] = f"{sign}{format_cents(abs(opt.price_cents))}"
        opens = [display[c] for c in opt.child_group_ids if c in display]
        if opens:
            entry["opens"] = opens
        nested_defaults: Dict[str, List[Any]] = {}
        for cgid, cdefs in opt.child_group_defaults.items():
            child = store.modifier_groups.get(cgid)
            if cgid in display and child is not None and cdefs:
                nested_defaults[display[cgid]] = self._default_labels(
                    child, list(cdefs)
                )
        if nested_defaults:
            entry["defaults_when_selected"] = nested_defaults
        return entry if len(entry) > 1 else opt.name

    def _group_entry(
        self,
        store: Store,
        gid: str,
        defaults: Optional[List[str]],
        options: Optional[List[ModifierOption]] = None,
        cap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Renders a modifier group; `cap` limits the listed modifiers."""
        grp = store.modifier_groups[gid]
        out: Dict[str, Any] = {"name": self._group_display_names(store)[gid]}
        if grp.min_select > 0:
            out["min_count"] = grp.min_select
        out["max_count"] = grp.max_select
        if grp.max_per_option > 1:
            out["max_per_option"] = grp.max_per_option
        if grp.supports_half_and_half:
            out["supports_half_and_half"] = True
        if defaults:
            out["default"] = self._default_labels(grp, defaults)
        if options is None:
            options = [o for o in grp.options.values() if o.available]
        shown = options[:cap] if cap is not None else options
        out["modifiers"] = [self._option_entry(store, o) for o in shown]
        if len(shown) < len(options):
            out["more_modifiers"] = len(options) - len(shown)
        unavailable = [o.name for o in grp.options.values() if not o.available]
        if unavailable:
            out["unavailable_modifiers"] = unavailable[:_LOOKUP_NAME_CAP]
        return out

    @is_tool(ToolType.READ)
    def get_store_info(self, store_id: str) -> dict[str, Any]:
        """Get metadata, operating hours, fulfillment rules, reservation policy, and categories for a store.

        Args:
            store_id: The store identifier given under Current Store.

        Returns:
            A dictionary with store metadata, current session state, and menu categories.
        """
        store = self._get_store(store_id)
        return {
            "store_id": store.store_id,
            "name": store.name,
            "concept": store.concept,
            "cuisine": store.cuisine,
            "open_hours": store.open_hours,
            "pickup_lead_minutes": store.pickup_lead_minutes,
            "delivery_lead_minutes": store.delivery_lead_minutes,
            "delivery_fee_cents": store.delivery_fee_cents,
            "min_delivery_subtotal_cents": store.min_delivery_subtotal_cents,
            "reservation_policy": (
                store.reservation_policy.model_dump()
                if store.reservation_policy is not None
                else None
            ),
            "current_time": self.db.cart.current_time,
            "channel": self.db.cart.channel,
            "order_type": self.db.cart.order_type,
            "scheduled_time": self.db.cart.scheduled_time,
            "categories": store.categories,
        }

    @is_tool(ToolType.READ)
    def list_menu_categories(self, store_id: str) -> dict[str, Any]:
        """List the store's menu categories and how many items each one has.

        Args:
            store_id: The store identifier given under Current Store.

        Returns:
            The categories in menu order with item counts.
        """
        store = self._get_store(store_id)
        counts: Dict[str, List[int]] = {}
        for item, reason in self._menu_entries(store):
            orderable, unavailable = counts.setdefault(item.category, [0, 0])
            counts[item.category] = [
                orderable + (not reason),
                unavailable + bool(reason),
            ]
        categories: List[Dict[str, Any]] = []
        for category in store.categories + sorted(set(counts) - set(store.categories)):
            if category not in counts:
                continue
            orderable, unavailable = counts.pop(category)
            entry: Dict[str, Any] = {"category": category, "items": orderable}
            if unavailable:
                entry["unavailable_items"] = unavailable
            categories.append(entry)
        return {"store": store.name, "categories": categories}

    @is_tool(ToolType.READ)
    def search_menu(
        self,
        store_id: str,
        query: str = "",
        category: str = "",
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search the menu by keyword and/or category; results are paged, best matches first.

        Each item shows its exact name, category, base price (plus `starting_price` when required
        choices raise the minimum build above `base_price` or when `base_price` is '$0.00'), a
        short description, and `not_available` with the reason if it can't be ordered right now.
        With a keyword, the first page also has `matching_modifiers`: modifiers (such as a milk,
        syrup, sauce, or topping) whose name contains the keyword, each with up to three of the
        numbered groups that offer it (`more_groups` counts the rest). A modifier can be ordered
        only on items that have one of those groups.

        Args:
            store_id: The store identifier given under Current Store.
            query: Keyword or phrase to match against item names, descriptions, and categories.
            category: Category name to restrict the search to (see list_menu_categories).
            offset: Index of the first result to return; use `next_offset` from the previous page.

        Returns:
            `total_matches`, the page of `items`, `next_offset` when more results exist, and
            `matching_modifiers` on the first page of a keyword search when any modifier matches.
        """
        store = self._get_store(store_id)
        q_norm = (query or "").strip().lower()
        q_tokens = [t for t in re.split(r"\W+", q_norm) if t]
        cat_norm = (category or "").strip().lower()
        exact_category = any(c.lower() == cat_norm for c in store.categories)

        scored: List[tuple[int, int, MenuItem, str]] = []
        for idx, (item, reason) in enumerate(self._menu_entries(store)):
            item_cat = item.category.lower()
            if cat_norm and (
                item_cat != cat_norm if exact_category else cat_norm not in item_cat
            ):
                continue
            score = 1
            if q_norm:
                name_lower = item.name.lower()
                aliases = [a.lower() for a in item.aliases]
                searchable = " ".join(
                    [name_lower, item_cat, item.description.lower(), *aliases]
                )
                if q_norm == name_lower or q_norm in aliases:
                    score = 200
                elif q_norm in name_lower or any(q_norm in a for a in aliases):
                    score = 150
                elif q_norm in searchable:
                    score = 100
                else:
                    in_name = sum(
                        25
                        for t in q_tokens
                        if t in name_lower or any(t in a for a in aliases)
                    )
                    score = in_name + sum(10 for t in q_tokens if t in searchable)
            if score > 0:
                scored.append((-score, idx, item, reason))
        scored.sort(key=lambda s: (s[0], s[1]))

        offset = max(0, int(offset or 0))
        page = scored[offset : offset + _LOOKUP_PAGE_SIZE]
        items: List[Dict[str, Any]] = []
        for _, _, item, reason in page:
            entry: Dict[str, Any] = {
                "name": item.name,
                "category": item.category,
                "base_price": format_cents(item.base_price_cents),
            }
            start_cents = starting_price_cents(store, item)
            if (
                item.base_price_cents == 0 or start_cents > item.base_price_cents
            ) and start_cents > 0:
                entry["starting_price"] = format_cents(start_cents)
            if item.description:
                entry["description"] = _truncate(
                    item.description, _LOOKUP_DESCRIPTION_CAP
                )
            if reason:
                entry["not_available"] = reason
            items.append(entry)
        out: Dict[str, Any] = {"total_matches": len(scored), "items": items}
        if offset + len(page) < len(scored):
            out["next_offset"] = offset + len(page)
        if q_norm and offset == 0:
            modifiers = self._matching_modifiers(store, q_norm, q_tokens)
            if modifiers:
                out["matching_modifiers"] = modifiers
        return out

    def _matching_modifiers(
        self, store: Store, q_norm: str, q_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """Available modifier names containing the query, with the groups that offer them."""
        groups_by_name: Dict[str, List[str]] = {}
        for gid, display_name in self._group_display_names(store).items():
            for opt in store.modifier_groups[gid].options.values():
                name_lower = opt.name.lower()
                if not opt.available:
                    continue
                if q_norm in name_lower or (
                    q_tokens and all(t in name_lower for t in q_tokens)
                ):
                    groups_by_name.setdefault(opt.name, []).append(display_name)
        out: List[Dict[str, Any]] = []
        for name, groups in list(groups_by_name.items())[:_LOOKUP_MODIFIER_MATCH_CAP]:
            entry: Dict[str, Any] = {
                "name": name,
                "groups": groups[:_LOOKUP_MODIFIER_GROUP_CAP],
            }
            if len(groups) > _LOOKUP_MODIFIER_GROUP_CAP:
                entry["more_groups"] = len(groups) - _LOOKUP_MODIFIER_GROUP_CAP
            out.append(entry)
        return out

    @is_tool(ToolType.READ)
    def get_item_details(self, store_id: str, item_id: str) -> dict[str, Any]:
        """Get one menu item's description, base price, availability, and modifier groups.

        The item has `name`, `category`, and `base_price` (plus `starting_price` when required
        choices raise the minimum build above `base_price` or when `base_price` is '$0.00'), and
        when they apply `description`,
        `not_available` (why it can't be ordered for the current time and fulfillment),
        `schedule` (when it is offered), and `not_available_for` (fulfillment methods it can't
        be ordered for).

        `modifier_groups` lists the item's own groups; `nested_groups` lists groups that apply
        only once a modifier that opens them is selected. Each group has:
            - `name`: the numbered group name that ordering tools accept as `group_id`.
            - `min_count`: the minimum number of choices; absent means the group is optional.
            - `max_count`: the maximum number of choices.
            - `max_per_option`: how many times one modifier can be chosen; absent means once.
            - `supports_half_and_half`: present when modifiers can go on the left or right half.
            - `default`: what the item gets when the group is left out; a default applied
              more than once is shown as {name, quantity}.
            - `modifiers`: each modifier as its name, or as a dict with `name`, `price` (the
              extra charge, e.g. '+$0.75'), `opens` (nested groups that apply once it is
              selected), and `defaults_when_selected` (the defaults it sets in those nested
              groups, used instead of the nested group's own `default`).
            - `more_modifiers`: how many modifiers are not listed.
            - `unavailable_modifiers`: modifiers that can't be ordered right now.
        `other_nested_groups` names nested groups that are not expanded (`more_nested_groups`
        counts any beyond that list), and `note` appears when anything is left out; use
        get_modifier_group to see a group in full.

        Args:
            store_id: The store identifier given under Current Store.
            item_id: The exact item name, as shown in Menu Data or search_menu.

        Returns:
            The item and its modifier groups.
        """
        store = self._get_store(store_id)
        item = self._get_item(store, item_id)
        display = self._group_display_names(store)
        ok, reason = self._check_item_availability(store, item)

        top = [g for g in item.modifier_group_ids if g in display]
        all_nested = [
            g.group_id
            for g in self._expand_item_modifier_groups(store, item)
            if g.group_id in display and g.group_id not in top
        ]
        inline = all_nested[:_LOOKUP_INLINE_GROUP_CAP]
        other_nested = all_nested[_LOOKUP_INLINE_GROUP_CAP:]

        out: Dict[str, Any] = {
            "name": item.name,
            "category": item.category,
            "base_price": format_cents(item.base_price_cents),
        }
        start_cents = starting_price_cents(store, item)
        if (
            item.base_price_cents == 0 or start_cents > item.base_price_cents
        ) and start_cents > 0:
            out["starting_price"] = format_cents(start_cents)
        if item.description:
            out["description"] = item.description
        if not ok:
            out["not_available"] = reason
        if item.schedule_ids:
            out["schedule"] = schedule_summary(store, item) or "Unrestricted"
        if not_available_for(item):
            out["not_available_for"] = not_available_for(item)

        out["modifier_groups"] = [
            self._group_entry(
                store,
                g,
                list(
                    item.group_defaults.get(
                        g, store.modifier_groups[g].default_option_ids
                    )
                ),
                cap=_LOOKUP_INLINE_OPTION_CAP,
            )
            for g in top
        ]
        if inline:
            out["nested_groups"] = [
                self._group_entry(
                    store,
                    g,
                    list(
                        item.group_defaults.get(
                            g, store.modifier_groups[g].default_option_ids
                        )
                    ),
                    cap=_LOOKUP_INLINE_OPTION_CAP,
                )
                for g in inline
            ]
        if other_nested:
            out["other_nested_groups"] = [
                display[g] for g in other_nested[:_LOOKUP_NAME_CAP]
            ]
            if len(other_nested) > _LOOKUP_NAME_CAP:
                out["more_nested_groups"] = len(other_nested) - _LOOKUP_NAME_CAP
        if other_nested or any(
            "more_modifiers" in g
            for g in out["modifier_groups"] + out.get("nested_groups", [])
        ):
            out["note"] = (
                "Some groups or modifiers are not listed here; call get_modifier_group "
                "with the group name to see them."
            )
        return out

    def _item_group_defaults(self, store: Store, item: MenuItem, gid: str) -> List[str]:
        """Default option IDs for `gid` on `item`, including single-choice parent overrides."""
        if gid in item.group_defaults:
            return list(item.group_defaults[gid])
        for pgid in item.modifier_group_ids:
            pgrp = store.modifier_groups.get(pgid)
            if pgrp is None or pgrp.min_select < 1:
                continue
            avail = [o for o in pgrp.options.values() if o.available]
            if len(avail) == 1 and gid in avail[0].child_group_defaults:
                return list(avail[0].child_group_defaults[gid])
        return list(store.modifier_groups[gid].default_option_ids)

    @is_tool(ToolType.READ)
    def get_modifier_group(
        self,
        store_id: str,
        group_name: str,
        query: str = "",
        offset: int = 0,
        item_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """List the modifiers of one modifier group, paged and optionally filtered by keyword.

        Args:
            store_id: The store identifier given under Current Store.
            group_name: The exact group name, as shown in Menu Data or get_item_details (for example 'Drink (3)').
            query: Optional keyword; only modifiers whose name contains it are returned.
            offset: Index of the first modifier to return; use `next_offset` from the previous page.
            item_id: Optional item name; when given, `default` reflects that item's override for this group.

        Returns:
            The group's limits, defaults, `total_modifiers`, the page of `modifiers`, and
            `next_offset` when more exist.
        """
        store = self._get_store(store_id)
        display = self._group_display_names(store)
        by_display = {name.lower(): gid for gid, name in display.items()}
        key = (group_name or "").strip().lower()
        gid = by_display.get(key)
        if gid is None:
            raise ValueError(
                f"No modifier group named '{group_name}' in store '{store.store_id}'."
            )
        grp = store.modifier_groups[gid]
        defaults = list(grp.default_option_ids)
        if item_id is not None and str(item_id).strip():
            item = self._get_item(store, str(item_id).strip())
            reachable = {
                g.group_id for g in self._expand_item_modifier_groups(store, item)
            }
            if gid not in reachable:
                raise ValueError(
                    f"Modifier group '{display[gid]}' does not apply to '{item.name}'."
                )
            defaults = self._item_group_defaults(store, item, gid)
        q_norm = (query or "").strip().lower()
        q_tokens = [t for t in re.split(r"\W+", q_norm) if t]
        options = [
            o
            for o in grp.options.values()
            if o.available
            and (
                not q_norm
                or q_norm in o.name.lower()
                or all(t in o.name.lower() for t in q_tokens)
            )
        ]
        offset = max(0, int(offset or 0))
        page = options[offset : offset + _LOOKUP_GROUP_PAGE_SIZE]
        out = self._group_entry(store, gid, defaults, options=page)
        out.pop("more_modifiers", None)
        out["total_modifiers"] = len(options)
        if offset + len(page) < len(options):
            out["next_offset"] = offset + len(page)
        if q_norm and not options:
            out["note"] = f"No modifiers in this group match '{query}'."
        return out

    @is_tool(ToolType.READ)
    def get_cart(self, store_id: Optional[str] = None) -> CartState:
        """Get the current contents of the working shopping cart.

        Args:
            store_id: Ignored; the active cart is returned.

        Returns:
            The current CartState including store_id and all line items.
        """
        _ = store_id
        return self.db.cart

    def _cart_line_index(self, line_item_id: str) -> int:
        idx = next(
            (
                i
                for i, li in enumerate(self.db.cart.items)
                if li.line_item_id == line_item_id
            ),
            None,
        )
        if idx is None:
            valid_ids = [li.line_item_id for li in self.db.cart.items]
            raise ValueError(
                f"Line item '{line_item_id}' not found in cart. Current line_item_ids: {valid_ids}"
            )
        return idx

    @is_tool(ToolType.WRITE)
    def add_to_cart(
        self,
        store_id: str,
        item_id: str,
        quantity: int = 1,
        modifiers: Optional[List[ModifierSelection | Dict[str, Any]]] = None,
    ) -> CartState:
        """Validate and add a menu item with its selected modifiers to the working cart.

        Groups without a selection get their default choices. A selection for a group replaces that
        group's defaults. Enforces availability, required groups, selection limits, and nested
        combo/deal groups.

        Args:
            store_id: The store identifier given under Current Store.
            item_id: The exact item name, as shown in Menu Data or search_menu.
            quantity: Number of units of this item to add (default 1).
            modifiers: Flat list of modifier selection dicts, each with:
                - 'group_id': str, the numbered group name (e.g. 'Options (3)'); required when the option name appears in more than one of the item's groups, optional otherwise
                - 'option_id': str (the exact modifier name without price suffixes; to remove a default, choose the group's 'No <name>' option, or send 'No <name>' for a pizza topping/sauce with portions)
                - 'quantity': int, optional (default 1; for counted options, the full amount; 0 removes a default, the same as its 'No <name>' option)
                - 'placement': str, optional ('whole', 'left', or 'right'; default 'whole'); only for pizza toppings that offer halves
                - 'volume': str, optional ('light', 'normal', or 'extra'; default 'normal'); only for pizza toppings that offer light/extra portions

        Returns:
            The updated CartState after adding the item.
        """
        store = self._get_store(store_id)
        item = self._get_item(store, item_id)
        line_id = f"line_{self.db.cart.next_line_id}"
        line_item = self._validate_and_build_line_item(
            store=store,
            item=item,
            line_item_id=line_id,
            quantity=_whole_number(quantity, "quantity"),
            raw_modifiers=modifiers,
        )

        self.db.cart.store_id = store.store_id
        self.db.cart.items.append(line_item)
        self.db.cart.next_line_id += 1
        return self.db.cart

    @is_tool(ToolType.WRITE)
    def update_cart_item(
        self,
        line_item_id: str,
        quantity: Optional[int] = None,
        modifiers: Optional[List[ModifierSelection | Dict[str, Any]]] = None,
        store_id: Optional[str] = None,
    ) -> CartState:
        """Update the quantity or modifiers of an existing line item in the cart.

        If quantity is set to 0, the line item is removed from the cart.
        If modifiers is provided, it completely replaces the line item's modifier selections and
        re-validates all constraints from scratch; any group or removal ('No ...') omitted from
        modifiers reverts to the item's standard defaults.

        Args:
            line_item_id: The line_item_id in the cart (e.g., 'line_1').
            quantity: Optional new quantity for the line item (set to 0 to remove).
            modifiers: Optional complete replacement list of modifier selection dicts, using the same format as add_to_cart.
            store_id: Ignored; the active cart store is used.

        Returns:
            The updated CartState.
        """
        _ = store_id
        if self.db.cart.store_id is None or not self.db.cart.items:
            raise ValueError("Cart is currently empty.")

        idx = self._cart_line_index(line_item_id)
        if quantity is not None:
            quantity = _whole_number(quantity, "quantity")
        if quantity is not None and quantity == 0:
            return self.remove_from_cart(line_item_id=line_item_id)

        existing = self.db.cart.items[idx]
        new_qty = existing.quantity if quantity is None else quantity
        if new_qty < 1:
            raise ValueError(f"Quantity must be at least 1, got {new_qty}.")
        if modifiers is None:
            updated_line = existing.model_copy(deep=True)
            unit_total = existing.line_total_cents // max(1, existing.quantity)
            updated_line.quantity = new_qty
            updated_line.line_total_cents = unit_total * new_qty
        else:
            store = self._get_store(self.db.cart.store_id)
            item = self._get_item(store, existing.item_id)
            updated_line = self._validate_and_build_line_item(
                store=store,
                item=item,
                line_item_id=existing.line_item_id,
                quantity=new_qty,
                raw_modifiers=modifiers,
            )
        self.db.cart.items[idx] = updated_line
        return self.db.cart

    @is_tool(ToolType.WRITE)
    def remove_from_cart(
        self, line_item_id: str, store_id: Optional[str] = None
    ) -> CartState:
        """Remove a line item from the working cart by its line_item_id.

        Args:
            line_item_id: The line_item_id to remove (e.g., 'line_1').
            store_id: Ignored; the active cart store is used.

        Returns:
            The updated CartState.
        """
        _ = store_id
        idx = self._cart_line_index(line_item_id)
        self.db.cart.items.pop(idx)
        if not self.db.cart.items:
            self.db.cart = self._empty_cart()
        return self.db.cart

    @is_tool(ToolType.WRITE)
    def clear_cart(self, store_id: Optional[str] = None) -> CartState:
        """Clear all items from the working cart and reset it to empty.

        Args:
            store_id: Ignored; the active cart store is used.

        Returns:
            The empty CartState.
        """
        _ = store_id
        self.db.cart = self._empty_cart()
        return self.db.cart

    def _validate_fulfillment(
        self, store: Store, order_type: str, scheduled_time: Optional[str]
    ) -> tuple[str, str]:
        """Checks an order type and time against the channel, hours, and lead times."""
        norm_type = _normalize_order_type(order_type)
        raw_time = (scheduled_time or "ASAP").strip()
        scheduled = (
            "ASAP"
            if raw_time.lower() in ("asap", "now", "")
            else _parse_clock(raw_time)
        )
        if self.db.cart.channel == "drive_thru":
            if norm_type != "pickup":
                raise ValueError(
                    "At the drive-thru the order is picked up now; delivery and dine-in can't be ordered here."
                )
            if scheduled != "ASAP":
                raise ValueError(
                    "At the drive-thru the order is for right now; a later time can't be scheduled."
                )
            return norm_type, scheduled
        if norm_type not in ("pickup", "delivery"):
            raise ValueError("Orders placed ahead are for pickup or delivery.")
        open_str, close_str = store.open_hours.split("-")
        opens, closes = _minutes(open_str), _minutes(close_str)
        now = self._now()
        now_min = now.hour * 60 + now.minute
        lead = (
            store.pickup_lead_minutes
            if norm_type == "pickup"
            else store.delivery_lead_minutes
        )
        if scheduled == "ASAP":
            if not opens <= now_min < closes:
                raise ValueError(
                    f"The store is closed right now (hours {store.open_hours}); choose a time within store hours."
                )
            if now_min + lead > closes:
                raise ValueError(
                    f"It's too late for {norm_type} today: it needs {lead} minutes and the store closes at {close_str}."
                )
            return norm_type, scheduled
        when = _minutes(scheduled)
        if not opens <= when < closes:
            raise ValueError(
                f"{scheduled} is outside store hours ({store.open_hours})."
            )
        if when < now_min + lead:
            raise ValueError(
                f"{scheduled} is too soon: {norm_type} needs at least {lead} minutes."
            )
        return norm_type, scheduled

    def _unorderable_lines(
        self, store: Store, order_type: str, scheduled_time: str
    ) -> List[str]:
        """Reasons why cart lines can't be ordered for this method and time."""
        at_time = self._effective_time(scheduled_time)
        reasons: List[str] = []
        for line in self.db.cart.items:
            item = store.items.get(line.item_id)
            if item is None:
                continue
            ok, why = self._check_item_availability(
                store, item, order_type=order_type, at_time=at_time
            )
            if not ok:
                reasons.append(why)
        return reasons

    @is_tool(ToolType.WRITE)
    def set_fulfillment(
        self,
        order_type: str,
        scheduled_time: str = "ASAP",
        store_id: Optional[str] = None,
    ) -> CartState:
        """Set how and when the order is fulfilled: pickup or delivery, ASAP or a later time today.

        Only needed when the customer orders ahead (mobile channel); at the drive-thru the order is
        always pickup, ASAP. The time must be within store hours and at least the store's lead time
        from now. Items in the cart must be orderable at that time and for that method. Item
        availability checks then use this method and time.

        Args:
            order_type: 'pickup' or 'delivery'.
            scheduled_time: 'ASAP' or a time later today in 'HH:MM' (24-hour) format.
            store_id: Ignored; the active cart store is used.

        Returns:
            The updated CartState.
        """
        _ = store_id
        if self.db.cart.store_id is None:
            raise ValueError("No active store for this session.")
        store = self._get_store(self.db.cart.store_id)
        norm_type, scheduled = self._validate_fulfillment(
            store, order_type, scheduled_time
        )
        blocked = self._unorderable_lines(store, norm_type, scheduled)
        if blocked:
            raise ValueError(
                "Can't change the fulfillment while the cart has items that would not be available: "
                + " ".join(blocked)
            )
        self.db.cart.order_type = norm_type  # type: ignore[assignment]
        self.db.cart.scheduled_time = scheduled
        return self.db.cart

    @is_tool(ToolType.WRITE)
    def place_order(
        self,
        order_type: Optional[str] = None,
        scheduled_time: Optional[str] = None,
        customer_name: str = "",
        customer_phone: str = "",
        delivery_address: Optional[str] = None,
        store_id: Optional[str] = None,
    ) -> SubmittedOrder:
        """Place the current cart as the customer's final order.

        Placing is final: a placed order can't be changed with these tools, and the cart is emptied.
        Call it once, after the customer says they are done ordering.
        At the drive-thru the order is pickup, ASAP, and name and phone are optional. When the
        customer orders ahead (mobile channel), the customer's full name and a 10-digit phone
        number are required; the order is pickup or delivery, ASAP or a later time today within
        store hours and at least the lead time from now; delivery needs an address and the store's
        minimum subtotal. Every item must be orderable at that time and for that method.

        Args:
            order_type: 'pickup' or 'delivery' (default: the cart's, initially 'pickup').
            scheduled_time: 'ASAP' or 'HH:MM' later today (default: the cart's, initially 'ASAP').
            customer_name: Customer full name (required when ordering ahead).
            customer_phone: Customer 10-digit phone number (required when ordering ahead).
            delivery_address: Street address (required for delivery; ignored for pickup).
            store_id: Ignored; the active cart store is used.

        Returns:
            The finalized SubmittedOrder with calculated subtotal, delivery fee, and total_cents.
        """
        _ = store_id
        if self.db.cart.store_id is None or not self.db.cart.items:
            raise ValueError("Cannot place an order with an empty cart.")

        store = self._get_store(self.db.cart.store_id)
        norm_type, scheduled = self._validate_fulfillment(
            store,
            order_type or self.db.cart.order_type,
            scheduled_time
            if scheduled_time is not None
            else self.db.cart.scheduled_time,
        )
        mobile = self.db.cart.channel == "mobile"
        if mobile:
            if not (customer_name or "").strip():
                raise ValueError(
                    "The customer's full name is required for orders placed ahead."
                )
            if _phone_digits(customer_phone) is None:
                raise ValueError(
                    "A 10-digit phone number is required for orders placed ahead."
                )
        norm_name = _normalize_name(customer_name, allow_default=not mobile)
        norm_phone = _normalize_phone(customer_phone, allow_default=not mobile)

        blocked = self._unorderable_lines(store, norm_type, scheduled)
        if blocked:
            raise ValueError(
                "Some items can't be ordered this way: " + " ".join(blocked)
            )

        # The order keeps the cart lines as built; get_db_hash canonicalizes them for grading.
        items = [li.model_copy(deep=True) for li in self.db.cart.items]
        subtotal_cents = sum(li.line_total_cents for li in items)
        if norm_type == "delivery":
            if not (delivery_address or "").strip():
                raise ValueError("A delivery address is required for delivery orders.")
            norm_address: Optional[str] = _normalize_address(delivery_address)
            if subtotal_cents < store.min_delivery_subtotal_cents:
                raise ValueError(
                    f"Delivery orders at '{store.name}' need a subtotal of at least "
                    f"{format_cents(store.min_delivery_subtotal_cents)}; the cart is {format_cents(subtotal_cents)}."
                )
            delivery_fee_cents = store.delivery_fee_cents
        else:
            norm_address = None
            delivery_fee_cents = 0

        total_cents = subtotal_cents + delivery_fee_cents
        order_id = f"#ORD{len(self.db.orders) + 1:03d}"

        order = SubmittedOrder(
            order_id=order_id,
            store_id=store.store_id,
            order_type=norm_type,  # type: ignore[arg-type]
            channel=self.db.cart.channel,
            scheduled_time=scheduled,
            customer_name=norm_name,
            customer_phone=norm_phone,
            delivery_address=norm_address,
            items=items,
            subtotal_cents=subtotal_cents,
            delivery_fee_cents=delivery_fee_cents,
            total_cents=total_cents,
            status="confirmed",
        )
        self.db.orders[order_id] = order
        self.db.cart = self._empty_cart()
        return order

    def _reservation_policy(
        self, store_id: Optional[str]
    ) -> tuple[Store, ReservationPolicy]:
        store = self._get_store(store_id or self.db.cart.store_id or "")
        if store.reservation_policy is None:
            raise ValueError(f"{store.name} does not take table reservations.")
        return store, store.reservation_policy

    def _slot_problem(
        self,
        store: Store,
        policy: ReservationPolicy,
        date: str,
        time: str,
        party_size: int,
    ) -> Optional[tuple[str, str]]:
        """(kind, reason) why a booking can't be made at all, or None; kind is 'party', 'date' or 'time'."""
        if not policy.min_party_size <= party_size <= policy.max_party_size:
            return (
                "party",
                f"Tables are for parties of {policy.min_party_size} to {policy.max_party_size}; "
                f"a party of {party_size} can't be booked.",
            )
        try:
            day = datetime.strptime((date or "").strip(), "%Y-%m-%d")
        except ValueError:
            return "date", f"Use a date like '2026-06-26' (got '{date}')."
        now = self._now()
        today = now.replace(hour=0, minute=0)
        if day < today:
            return "date", "That date has already passed."
        last_day = today + timedelta(days=policy.max_days_ahead)
        if day > last_day:
            return (
                "date",
                f"Tables can be booked up to {policy.max_days_ahead} days ahead (until {last_day:%Y-%m-%d}).",
            )
        open_str, close_str = store.open_hours.split("-")
        first = _minutes(open_str) + policy.first_seating_after_open_minutes
        last = _minutes(close_str) - policy.last_seating_before_close_minutes
        when = _minutes(time)
        if when % policy.slot_minutes:
            return (
                "time",
                f"Bookings start every {policy.slot_minutes} minutes (for example 18:00 or 18:30).",
            )
        if not first <= when <= last:
            return "time", f"Tables can be booked from {_hhmm(first)} to {_hhmm(last)}."
        if day == today and when <= now.hour * 60 + now.minute:
            return "time", "That time has already passed today."
        return None

    def _slot_usage(
        self, store_id: str, date: str, time: str, exclude_id: Optional[str] = None
    ) -> tuple[Dict[str, int], int]:
        tables: Dict[str, int] = {}
        chairs = 0
        day = _date_key(date)
        for r in self.db.reservations.values():
            if (
                r.status == "confirmed"
                and r.store_id == store_id
                and _date_key(r.date) == day
                and r.time == time
                and r.reservation_id != exclude_id
            ):
                tables[r.seating] = tables.get(r.seating, 0) + 1
                chairs += r.high_chairs
        return tables, chairs

    def _free_seating(
        self,
        store: Store,
        policy: ReservationPolicy,
        date: str,
        time: str,
        high_chairs: int,
        exclude_id: Optional[str] = None,
    ) -> List[str]:
        """Seating areas with a free table (and enough high chairs) in this slot."""
        tables, chairs = self._slot_usage(store.store_id, date, time, exclude_id)
        if chairs + high_chairs > policy.high_chairs_per_slot:
            return []
        return [
            area
            for area, capacity in policy.tables_per_slot.items()
            if tables.get(area, 0) < capacity
        ]

    def _normalize_seating(
        self, policy: ReservationPolicy, seating: Optional[str]
    ) -> str:
        value = (seating or "any").strip().lower()
        if value in ("any", "", "no preference", "either"):
            return "any"
        if value not in policy.tables_per_slot:
            raise ValueError(
                f"Seating options here: {', '.join(policy.tables_per_slot)} (or 'any')."
            )
        return value

    def _alternatives(
        self,
        store: Store,
        policy: ReservationPolicy,
        date: str,
        time: str,
        party_size: int,
        seating: str,
        high_chairs: int,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Up to 4 nearest bookable slots on the same day with room for the request."""
        open_str, close_str = store.open_hours.split("-")
        first = _minutes(open_str) + policy.first_seating_after_open_minutes
        last = _minutes(close_str) - policy.last_seating_before_close_minutes
        target = _minutes(time)
        found: List[tuple[int, int, Dict[str, Any]]] = []
        for slot in range(first, last + 1, policy.slot_minutes):
            hhmm = _hhmm(slot)
            if hhmm == time or self._slot_problem(
                store, policy, date, hhmm, party_size
            ):
                continue
            areas = self._free_seating(
                store, policy, date, hhmm, high_chairs, exclude_id
            )
            if seating != "any":
                areas = [a for a in areas if a == seating]
            if areas:
                found.append(
                    (abs(slot - target), slot, {"time": hhmm, "seating": areas})
                )
        found.sort(key=lambda f: (f[0], f[1]))
        return [f[2] for f in found[:4]]

    @is_tool(ToolType.READ)
    def check_table_availability(
        self,
        store_id: str,
        date: str,
        time: str,
        party_size: int,
        seating: str = "any",
        high_chairs: int = 0,
    ) -> dict[str, Any]:
        """Check whether a table can be booked, and list the nearest available times if not.

        Args:
            store_id: The store identifier given under Current Store.
            date: Date in 'YYYY-MM-DD' format.
            time: Time in 'HH:MM' (24-hour) format; bookings start every 30 minutes.
            party_size: Number of guests.
            seating: Seating area (e.g. 'indoor' or 'outdoor'), or 'any'.
            high_chairs: Number of high chairs needed.

        Returns:
            Whether the slot is available, the seating areas with room, a reason if it can't be
            booked, and up to 4 nearest available times on the same day.
        """
        store, policy = self._reservation_policy(store_id)
        party_size = _whole_number(party_size, "party_size")
        high_chairs = _whole_number(high_chairs, "high_chairs")
        if high_chairs < 0:
            raise ValueError("high_chairs can't be negative.")
        date = _date_key(date)
        clock = _parse_clock(time)
        area = self._normalize_seating(policy, seating)
        result: Dict[str, Any] = {
            "date": date,
            "time": clock,
            "party_size": party_size,
            "seating": area,
            "high_chairs": high_chairs,
        }
        slot_problem = self._slot_problem(store, policy, date, clock, party_size)
        if slot_problem and slot_problem[0] != "time":
            result.update(available=False, reason=slot_problem[1], alternatives=[])
            return result
        areas = (
            []
            if slot_problem
            else self._free_seating(store, policy, date, clock, high_chairs)
        )
        if area != "any":
            areas = [a for a in areas if a == area]
        if areas:
            result.update(available=True, available_seating=areas)
            return result
        if slot_problem:
            problem = slot_problem[1]
        else:
            _, chairs = self._slot_usage(store.store_id, date, clock)
            problem = (
                "Not enough high chairs are free at that time."
                if chairs + high_chairs > policy.high_chairs_per_slot
                else "No table is free at that time"
                + ("" if area == "any" else f" for {area} seating")
                + "."
            )
        result.update(
            available=False,
            reason=problem,
            alternatives=self._alternatives(
                store, policy, date, clock, party_size, area, high_chairs
            ),
        )
        return result

    def _book(
        self,
        store: Store,
        policy: ReservationPolicy,
        date: str,
        time: str,
        party_size: int,
        seating: str,
        high_chairs: int,
        exclude_id: Optional[str] = None,
    ) -> str:
        """Validates a booking and returns the seating area it gets."""
        problem = self._slot_problem(store, policy, date, time, party_size)
        if problem:
            raise ValueError(problem[1])
        if high_chairs < 0:
            raise ValueError("high_chairs can't be negative.")
        areas = self._free_seating(store, policy, date, time, high_chairs, exclude_id)
        if seating != "any":
            areas = [a for a in areas if a == seating]
        if not areas:
            raise ValueError(
                "That time is not available for this booking; check availability for the nearest open times."
            )
        return areas[0]

    @is_tool(ToolType.WRITE)
    def make_reservation(
        self,
        store_id: str,
        date: str,
        time: str,
        party_size: int,
        customer_name: str,
        customer_phone: str,
        seating: str = "any",
        high_chairs: int = 0,
    ) -> Reservation:
        """Book a table. Confirm the details with the customer before calling this.

        Args:
            store_id: The store identifier given under Current Store.
            date: Date in 'YYYY-MM-DD' format.
            time: Time in 'HH:MM' (24-hour) format; bookings start every 30 minutes.
            party_size: Number of guests.
            customer_name: Name for the booking.
            customer_phone: Customer 10-digit phone number.
            seating: Seating area (e.g. 'indoor' or 'outdoor'), or 'any' for the first area with room.
            high_chairs: Number of high chairs needed.

        Returns:
            The confirmed Reservation.
        """
        store, policy = self._reservation_policy(store_id)
        party_size = _whole_number(party_size, "party_size")
        high_chairs = _whole_number(high_chairs, "high_chairs")
        if not (customer_name or "").strip():
            raise ValueError("A name is required for the reservation.")
        if _phone_digits(customer_phone) is None:
            raise ValueError("A 10-digit phone number is required for the reservation.")
        date = _date_key(date)
        clock = _parse_clock(time)
        area = self._book(
            store,
            policy,
            date,
            clock,
            party_size,
            self._normalize_seating(policy, seating),
            high_chairs,
        )
        n = len(self.db.reservations) + 1
        while f"RES-{n:04d}" in self.db.reservations:
            n += 1
        reservation = Reservation(
            reservation_id=f"RES-{n:04d}",
            store_id=store.store_id,
            date=date,
            time=clock,
            party_size=party_size,
            customer_name=_normalize_name(customer_name, allow_default=False),
            customer_phone=_normalize_phone(customer_phone, allow_default=False),
            seating=area,
            high_chairs=high_chairs,
        )
        self.db.reservations[reservation.reservation_id] = reservation
        return reservation

    @is_tool(ToolType.READ)
    def find_reservations(
        self, customer_phone: str, store_id: Optional[str] = None
    ) -> list[Reservation]:
        """Find the reservations at this store made under a phone number.

        Args:
            customer_phone: The customer's phone number.
            store_id: Optional store identifier; defaults to the active store.

        Returns:
            The matching reservations (confirmed and cancelled), earliest first.
        """
        store, _ = self._reservation_policy(store_id)
        digits = _phone_digits(customer_phone)
        if digits is None:
            raise ValueError("Use the customer's 10-digit phone number.")
        found = [
            r
            for r in self.db.reservations.values()
            if r.store_id == store.store_id
            and _phone_digits(r.customer_phone) == digits
        ]
        return sorted(found, key=lambda r: (r.date, r.time, r.reservation_id))

    def _get_reservation(
        self, reservation_id: str, store_id: Optional[str] = None
    ) -> tuple[Store, ReservationPolicy, Reservation]:
        reservation = self.db.reservations.get((reservation_id or "").strip().upper())
        eff_store_id = store_id or (
            self.db.cart.store_id
            or (reservation.store_id if reservation is not None else "")
        )
        store, policy = self._reservation_policy(eff_store_id)
        if reservation is None or reservation.store_id != store.store_id:
            raise ValueError(
                f"No reservation '{reservation_id}' was found at {store.name}."
            )
        if reservation.status != "confirmed":
            raise ValueError(f"Reservation '{reservation_id}' is already cancelled.")
        return store, policy, reservation

    @is_tool(ToolType.WRITE)
    def update_reservation(
        self,
        reservation_id: str,
        date: Optional[str] = None,
        time: Optional[str] = None,
        party_size: Optional[int] = None,
        seating: Optional[str] = None,
        high_chairs: Optional[int] = None,
        customer_name: Optional[str] = None,
        customer_phone: Optional[str] = None,
        store_id: Optional[str] = None,
    ) -> Reservation:
        """Change an existing reservation. Confirm the new details with the customer first.

        Only the fields you pass change; the booking keeps its seating area unless you pass one.

        Args:
            reservation_id: The reservation ID (e.g. 'RES-0021').
            date: New date in 'YYYY-MM-DD' format.
            time: New time in 'HH:MM' (24-hour) format.
            party_size: New number of guests.
            seating: New seating area (e.g. 'indoor' or 'outdoor'), or 'any'.
            high_chairs: New number of high chairs.
            customer_name: New name for the booking.
            customer_phone: New 10-digit phone number.
            store_id: Optional store identifier; defaults to the active store.

        Returns:
            The updated Reservation.
        """
        store, policy, reservation = self._get_reservation(
            reservation_id, store_id=store_id
        )
        new_date = _date_key(date) if date else reservation.date
        new_time = _parse_clock(time) if time else reservation.time
        new_party = (
            reservation.party_size
            if party_size is None
            else _whole_number(party_size, "party_size")
        )
        new_chairs = (
            reservation.high_chairs
            if high_chairs is None
            else _whole_number(high_chairs, "high_chairs")
        )
        area = (
            reservation.seating
            if seating is None
            else self._normalize_seating(policy, seating)
        )
        if customer_phone is not None and _phone_digits(customer_phone) is None:
            raise ValueError("A 10-digit phone number is required for the reservation.")
        assigned = self._book(
            store,
            policy,
            new_date,
            new_time,
            new_party,
            area,
            new_chairs,
            exclude_id=reservation.reservation_id,
        )
        updated = reservation.model_copy(
            update={
                "date": new_date,
                "time": new_time,
                "party_size": new_party,
                "high_chairs": new_chairs,
                "seating": assigned,
            }
        )
        if customer_name is not None:
            updated.customer_name = _normalize_name(customer_name, allow_default=False)
        if customer_phone is not None:
            updated.customer_phone = _normalize_phone(
                customer_phone, allow_default=False
            )
        self.db.reservations[updated.reservation_id] = updated
        return updated

    @is_tool(ToolType.WRITE)
    def cancel_reservation(
        self, reservation_id: str, store_id: Optional[str] = None
    ) -> Reservation:
        """Cancel an existing reservation. Confirm with the customer first.

        Args:
            reservation_id: The reservation ID (e.g. 'RES-0021').
            store_id: Optional store identifier; defaults to the active store.

        Returns:
            The cancelled Reservation.
        """
        _, _, reservation = self._get_reservation(reservation_id, store_id=store_id)
        cancelled = reservation.model_copy(update={"status": "cancelled"})
        self.db.reservations[cancelled.reservation_id] = cancelled
        return cancelled

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the caller to a human restaurant staff member.

        Only call this tool if the customer asks for a person, or raises a restaurant issue the
        ordering tools can't handle (for example a problem with a past order). After calling it,
        tell the customer they are being transferred.

        Args:
            summary: A concise summary of the customer's request or issue.

        Returns:
            Confirmation message that the transfer succeeded.
        """
        _ = summary
        self.db.cart.transferred_to_human = True
        return "Transfer successful"
