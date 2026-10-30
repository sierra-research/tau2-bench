"""Renders a store menu as standard Markdown sections for the agent prompt."""

from typing import Callable, Dict, List, Tuple

from tau2.domains.food.data_model import MenuItem, Store
from tau2.domains.food.utils import format_cents

AvailabilityFn = Callable[[Store, MenuItem], Tuple[bool, str]]


def group_display_names(store: Store) -> Dict[str, str]:
    """Maps group_id to a unique display name, numbering repeated names like 'Options (3)'."""
    counts: Dict[str, int] = {}
    names: Dict[str, str] = {}
    for gid, grp in store.modifier_groups.items():
        base = grp.name or gid
        counts[base] = counts.get(base, 0) + 1
        names[gid] = f"{base} ({counts[base]})"
    return names


def schedule_summary(store: Store, item: MenuItem) -> str:
    """The item's availability windows, e.g. 'daily 07:00-11:30', or '' if unrestricted."""
    parts: List[str] = []
    for sid in item.schedule_ids:
        sched = store.schedules.get(sid)
        if not sched:
            continue
        for rule in sched.rules:
            if len(rule.days_of_week) == 7:
                day_part = "daily"
            elif rule.days_of_week:
                day_part = ", ".join(d.title() for d in rule.days_of_week)
            else:
                day_part = ", ".join(
                    f"{d.start_date}..{d.end_date}" for d in rule.dates
                )
            time_part = ", ".join(
                f"{tr.start_time}-{tr.end_time}" for tr in rule.time_ranges
            )
            parts.append(f"{day_part} {time_part}".strip())
    return "; ".join(parts)


def not_available_for(item: MenuItem) -> List[str]:
    """Fulfillment methods (pickup, delivery) the item can't be ordered for."""
    if not item.allowed_fulfillment_methods:
        return []
    return [
        m for m in ("pickup", "delivery") if m not in item.allowed_fulfillment_methods
    ]


def starting_price_cents(store: Store, item: MenuItem) -> int:
    """Minimum valid build price in cents for an item (base price plus cheapest required picks)."""
    total = item.base_price_cents
    for gid in item.modifier_group_ids:
        grp = store.modifier_groups.get(gid)
        if grp is None or grp.min_select <= 0:
            continue
        avail_prices = sorted(
            o.price_cents for o in grp.options.values() if o.available
        )
        if len(avail_prices) >= grp.min_select:
            total += sum(avail_prices[: grp.min_select])
    return total


def render_menu(store: Store, is_available: AvailabilityFn) -> str:
    """Renders a concise store menu in standard Markdown."""
    display = group_display_names(store)

    items_by_category: Dict[str, List[str]] = {}
    unavailable_lines: List[str] = []
    for item in store.items.values():
        ok, why = is_available(store, item)
        if not ok:
            unavailable_lines.append(f"- **{item.name}**: {why}")
            continue

        start_cents = starting_price_cents(store, item)
        if item.base_price_cents == 0 or start_cents > item.base_price_cents:
            price_label = (
                f"from {format_cents(start_cents)}"
                if start_cents > 0
                else format_cents(0)
            )
        else:
            price_label = format_cents(item.base_price_cents)
        meta: List[str] = [price_label]
        sched = schedule_summary(store, item)
        if sched:
            meta.append(f"schedule: {sched}")
        restricted = not_available_for(item)
        if restricted:
            meta.append(f"not available for: {', '.join(restricted)}")

        groups = [display[g] for g in item.modifier_group_ids if g in display]
        line = f"- **{item.name}** ({'; '.join(meta)})"
        if groups:
            line += f" — modifier groups: {', '.join(groups)}"
        category = item.category or "Other"
        items_by_category.setdefault(category, []).append(line)

    group_lines: List[str] = []
    for gid, grp in store.modifier_groups.items():
        options: List[str] = []
        for opt in grp.options.values():
            if not opt.available:
                continue
            children = [display[c] for c in opt.child_group_ids if c in display]
            if children:
                options.append(f"{opt.name} (→ {', '.join(children)})")
            else:
                options.append(opt.name)

        rules: List[str] = []
        if grp.min_select > 0:
            rules.append(f"min: {grp.min_select}")
        if grp.max_select > 0:
            rules.append(f"max: {grp.max_select}")
        if grp.supports_half_and_half:
            rules.append("supports half-and-half")
        rule_str = f" ({', '.join(rules)})" if rules else ""
        opts_str = ", ".join(options)
        group_lines.append(f"- **{display[gid]}**{rule_str}: {opts_str}")

    sections: List[str] = ["### Menu Items"]
    ordered_cats = list(store.categories) + [
        c for c in items_by_category if c not in store.categories
    ]
    for cat in ordered_cats:
        if cat in items_by_category:
            sections.append(f"#### {cat}\n" + "\n".join(items_by_category[cat]))

    sections.append("### Modifier Groups\n" + "\n".join(group_lines))

    if unavailable_lines:
        sections.append("### Unavailable Items\n" + "\n".join(unavailable_lines))
    else:
        sections.append("### Unavailable Items\nNone")

    return "\n\n".join(sections) + "\n"
