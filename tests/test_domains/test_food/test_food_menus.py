"""Tests for the food menu data: synthetic ids, references, categories, and orderability."""

import json
import re
from pathlib import Path

import pytest

from tau2.domains.food.menu_prompt import group_display_names
from tau2.domains.food.utils import FOOD_DATA_DIR, FOOD_DB_PATH, FOOD_MENUS_DIR
from tau2.environment.environment import Environment

from .conftest import CAFE, FICTIONAL_PHONE_RE, PIZZA, UUID_RE, store_env

STORE_PREFIXES = {
    "burger": "bg",
    "cafe": "cf",
    "pizza": "pz",
}

SYNTHETIC_ID_RE = re.compile(r"^(bg|cf|pz)-[igo]\d{4}$")
_REQUIRED_RE = re.compile(
    r"Modifier group '(.+)' on '.*' requires at least (\d+) selection"
)
_MAX_FILL_ROUNDS = 12


def _menu(store: str) -> dict:
    return json.loads((FOOD_MENUS_DIR / f"{store}.json").read_text(encoding="utf-8"))


def test_each_menu_file_loads_as_a_store(environment: Environment):
    menu_files = {p.stem for p in FOOD_MENUS_DIR.glob("*.json")}
    stores = environment.tools.db.stores
    assert set(stores) == menu_files == set(STORE_PREFIXES)
    assert {s.name for s in stores.values()} == {
        "Corner Cafe",
        "City Pizza",
        "Sunset Burger",
    }


@pytest.mark.parametrize("store,prefix", STORE_PREFIXES.items())
def test_menu_ids_follow_synthetic_schema(store: str, prefix: str) -> None:
    data = _menu(store)
    item_re = re.compile(rf"^{prefix}-i\d{{4}}$")
    group_re = re.compile(rf"^{prefix}-g\d{{4}}$")
    opt_re = re.compile(rf"^{prefix}-o\d{{4}}$")
    items = data["items"]
    groups = data["modifier_groups"]

    for iid, item in items.items():
        assert item_re.match(iid), f"Invalid item key {iid}"
        assert item["item_id"] == iid
    for gid, grp in groups.items():
        assert group_re.match(gid), f"Invalid group key {gid}"
        assert grp["group_id"] == gid
        for oid, opt in grp["options"].items():
            assert opt_re.match(oid), f"Invalid option id {oid} in {gid}"
            assert opt["option_id"] == oid

    item_nums = [int(iid[-4:]) for iid in items]
    group_nums = [int(gid[-4:]) for gid in groups]
    assert len(item_nums) == len(set(item_nums))
    assert len(group_nums) == len(set(group_nums))


def test_no_uuids_or_raw_ids_in_food_domain() -> None:
    code_dir = Path(__file__).resolve().parents[3] / "src" / "tau2" / "domains" / "food"
    files_to_scan = [
        *sorted(FOOD_DATA_DIR.rglob("*.json")),
        *sorted(FOOD_DATA_DIR.rglob("*.md")),
        *sorted(code_dir.rglob("*.py")),
    ]
    assert files_to_scan

    id_field_re = re.compile(r'"(?:item_id|group_id|option_id)"\s*:\s*"([^"]+)"')
    for path in files_to_scan:
        text = path.read_text(encoding="utf-8")
        uuid_match = UUID_RE.search(text)
        assert uuid_match is None, f"UUID {uuid_match.group(0)} found in {path}"
        if path.suffix == ".json" and path.parent == FOOD_MENUS_DIR:
            for m in id_field_re.finditer(text):
                val = m.group(1)
                assert not val.isdigit(), f"All-digit ID {val!r} found in {path}"
                assert SYNTHETIC_ID_RE.match(val), f"Non-synthetic ID {val!r} in {path}"


def test_seed_reservation_phones_are_fictional() -> None:
    db = json.loads(Path(FOOD_DB_PATH).read_text(encoding="utf-8"))
    for rid, res in db["reservations"].items():
        assert FICTIONAL_PHONE_RE.match(res["customer_phone"]), (
            rid,
            res["customer_phone"],
        )


@pytest.mark.parametrize("store", list(STORE_PREFIXES))
def test_menu_references_resolve(store: str) -> None:
    data = _menu(store)
    items = data["items"]
    groups = data["modifier_groups"]

    assert len(data["categories"]) == len(set(data["categories"]))
    assert {item["category"] for item in items.values()} == set(data["categories"])
    for iid, item in items.items():
        for sid in item.get("schedule_ids", []):
            assert sid in data["schedules"], f"Dangling schedule {sid} in {iid}"
        for gid in item.get("modifier_group_ids", []):
            assert gid in groups, f"Dangling modifier_group_id {gid} in {iid}"
        for gid, defaults in item.get("group_defaults", {}).items():
            assert gid in item["modifier_group_ids"], (
                f"{iid}: default for unattached {gid}"
            )
            for oid in defaults:
                assert oid in groups[gid]["options"], f"{iid}: unknown default {oid}"
    for gid, grp in groups.items():
        for oid in grp.get("default_option_ids", []):
            assert oid in grp["options"], f"{gid}: unknown default {oid}"
        for oid, opt in grp["options"].items():
            for cgid in opt.get("child_group_ids", []):
                assert cgid in groups, f"Dangling child_group_id {cgid} in {oid}"
            for cgid, defaults in opt.get("child_group_defaults", {}).items():
                assert cgid in opt["child_group_ids"], (
                    f"{oid}: default for non-child {cgid}"
                )
                for doid in defaults:
                    assert doid in groups[cgid]["options"], (
                        f"{oid}: unknown default {doid}"
                    )

    # Every group must be reachable from an item; unreachable groups would still be rendered.
    reachable: set[str] = set()
    stack = [gid for item in items.values() for gid in item["modifier_group_ids"]]
    while stack:
        gid = stack.pop()
        if gid in reachable:
            continue
        reachable.add(gid)
        for opt in groups[gid]["options"].values():
            stack.extend(opt.get("child_group_ids", []))
    assert reachable == set(groups), sorted(set(groups) - reachable)
    used = {sid for item in items.values() for sid in item.get("schedule_ids", [])}
    assert used == set(data["schedules"]), sorted(set(data["schedules"]) - used)


def test_burger_meals_are_not_filed_with_single_items() -> None:
    meal_re = re.compile(r"(Combo|Boardwalk Box|Snack Duo|Meal)$")
    for item in _menu("burger")["items"].values():
        if item["category"] in ("Burgers", "Chicken & Fish", "Salads"):
            assert not meal_re.search(item["name"]), item["name"]
            assert not item["name"].startswith("Kids' "), item["name"]


@pytest.mark.parametrize("store", list(STORE_PREFIXES))
def test_item_names_are_unique(store: str) -> None:
    names = [item["name"].lower() for item in _menu(store)["items"].values()]
    assert len(names) == len(set(names))


def test_cafe_items_have_real_base_prices(environment: Environment):
    cafe = environment.tools.db.stores["cafe"]
    free = {i.name for i in cafe.items.values() if i.base_price_cents <= 0}
    assert free == set()


def test_counted_groups_keep_their_limits():
    from tau2.domains.food.modifiers import _is_portion_group

    stores = store_env(CAFE).tools.db.stores
    cafe = stores[CAFE].modifier_groups.values()
    for name in ("Espresso Shots", "Syrups", "Sweetener"):
        assert any(g.max_per_option > 1 for g in cafe if g.name == name), name
    pizza = stores[PIZZA].modifier_groups
    assert any(
        g.max_per_option > 1 for g in pizza.values() if g.name == "Block Party Picks"
    )
    # Portion groups keep the limit derived from their selection count.
    assert all(
        g.max_per_option == g.max_select for g in pizza.values() if _is_portion_group(g)
    )


def _add_filling_required_choices(
    env: Environment, store_id: str, item_id: str, mods: list
) -> list:
    """Adds the item; each required group the tool reports as unchosen gets its first choices."""
    tools = env.tools
    store = tools.db.stores[store_id]
    by_display = {name: gid for gid, name in group_display_names(store).items()}
    mods = list(mods)
    for _ in range(_MAX_FILL_ROUNDS):
        try:
            tools.add_to_cart(
                store_id=store_id, item_id=item_id, modifiers=mods or None
            )
            return mods
        except ValueError as e:
            match = _REQUIRED_RE.search(str(e))
            if not match:
                raise
            label, needed = match.group(1), int(match.group(2))
            grp = store.modifier_groups[by_display.get(label, label)]
            choices = [o for o in grp.options.values() if o.available][:needed]
            mods += [{"option_id": o.name, "group_id": label} for o in choices]
    raise AssertionError(
        f"{item_id}: required choices still missing after {_MAX_FILL_ROUNDS} rounds"
    )


@pytest.mark.parametrize("store_id", ["cafe", "pizza", "burger"])
def test_every_item_can_be_added_at_every_size(store_id: str):
    env = store_env(store_id)
    tools = env.tools
    store = tools.db.stores[store_id]
    unavailable: set[str] = set()
    offset = 0
    while offset is not None:
        page = tools.search_menu(store_id, offset=offset)
        unavailable |= {i["name"] for i in page["items"] if "not_available" in i}
        offset = page.get("next_offset")
    failures: list[str] = []
    for item in store.items.values():
        if item.name in unavailable:
            continue
        size_group = next(
            (
                store.modifier_groups[gid]
                for gid in item.modifier_group_ids
                if gid in store.modifier_groups
                and store.modifier_groups[gid].name.endswith("Size")
            ),
            None,
        )
        sizes = (
            [None]
            if size_group is None
            else [o.name for o in size_group.options.values() if o.available]
        )
        # Size names like "Medium" can repeat in other groups (e.g. cooking temperature).
        size_label = (
            None
            if size_group is None
            else group_display_names(store)[size_group.group_id]
        )
        for size in sizes:
            mods = [] if size is None else [{"option_id": size, "group_id": size_label}]
            try:
                _add_filling_required_choices(env, store_id, item.item_id, mods)
            except (ValueError, AssertionError) as e:
                failures.append(f"{item.name} / {size}: {e}")
            tools.clear_cart()
    assert not failures, failures[:10]


def test_repeated_default_quantities_fit_the_group_limits():
    env = store_env("cafe")
    for size in ("Medium", "Large"):
        cart = env.tools.add_to_cart(
            store_id="cafe",
            item_id="Latte",
            modifiers=[{"option_id": size}],
        )
        shots = [m for m in cart.items[-1].modifiers if m.name == "Espresso Shot"]
        assert sum(m.quantity for m in shots) == 2
