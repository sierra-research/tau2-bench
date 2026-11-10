"""Tests for the food domain tools: menu lookups, cart edits, fulfillment, ordering, reservations."""

import json
import re

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.food.environment import get_environment
from tau2.domains.food.menu_prompt import group_display_names
from tau2.domains.food.tools import _normalize_address
from tau2.environment.environment import Environment

from .conftest import (
    BURGER,
    CAFE,
    DEFAULT_NOW,
    PIZZA,
    STORES,
    UUID_RE,
    line_names,
    store_env,
    task_for,
)

MIX_MATCH = "Block Party Deal"
LOADED_PEPPERONI = "Late Night - Medium Classic"
# Upper bound for a single lookup result, in JSON characters (~4k tokens).
_MAX_LOOKUP_CHARS = 16000
# Half Sandwich Lunch choices; its Matcha Latte opens a two-option Sweetness Level group.
_LUNCH_WITH_MATCHA = [
    {"option_id": "Half Turkey Club Sandwich"},
    {"option_id": "Multigrain Wheat Bread"},
    {"option_id": "Cup of Chicken Tortilla Soup"},
    {"option_id": "Matcha Latte"},
]


def _group(details: dict, prefix: str) -> dict:
    groups = details["modifier_groups"] + details.get("nested_groups", [])
    return next(g for g in groups if g["name"].startswith(prefix))


def _modifier_names(group: dict) -> list[str]:
    return [m if isinstance(m, str) else m["name"] for m in group["modifiers"]]


def _all_groups(details: dict) -> list[dict]:
    return details["modifier_groups"] + details.get("nested_groups", [])


def _labels(env: Environment) -> dict[str, str]:
    """group_id -> numbered display name for the session's store."""
    return group_display_names(env.tools.db.stores[env.tools.db.cart.store_id])


def _first_option(env: Environment, item_name: str, group_name: str) -> str:
    details = env.tools.get_item_details(env.tools.db.cart.store_id, item_name)
    return next(
        _modifier_names(g)[0]
        for g in _all_groups(details)
        if g["name"].rsplit(" (", 1)[0] == group_name
    )


def _label_opened_by(env: Environment, item_name: str, parent: str, option: str) -> str:
    """Display name of the item's group that `parent` opens and that offers `option`."""
    store_id = env.tools.db.cart.store_id
    details = env.tools.get_item_details(store_id, item_name)
    for group in _all_groups(details):
        for entry in group["modifiers"]:
            if isinstance(entry, str) or entry["name"] != parent:
                continue
            opens = entry.get("opens", [])
            if isinstance(opens, str):
                opens = group["nested_group_sets"][opens]
            for name in opens:
                found = env.tools.get_modifier_group(store_id, name, query=option)
                if option in _modifier_names(found):
                    return name
    raise AssertionError(f"no group of {parent!r} offers {option!r}")


def _line_modifiers(env: Environment, line: int = -1) -> list[tuple[str, str]]:
    labels = _labels(env)
    return [
        (labels.get(m.group_id, m.group_id), m.name)
        for m in env.tools.db.cart.items[line].modifiers
    ]


def test_store_info_and_negative_high_chairs_validation():
    env = store_env(CAFE, channel="mobile")
    info = env.tools.get_store_info(CAFE)
    assert info["pickup_lead_minutes"] == 20
    assert info["delivery_lead_minutes"] == 45
    assert info["channel"] == "mobile"
    assert info["scheduled_time"] == "ASAP"
    assert info["reservation_policy"]["max_party_size"] == 8

    with pytest.raises(ValueError, match="high_chairs"):
        env.tools.check_table_availability(
            CAFE, "2026-06-27", "18:00", 2, high_chairs=-1
        )


@pytest.mark.parametrize(
    "store_id, item_name, group_prefix, expected_default",
    [
        ("cafe", "Latte", "Size", "Medium"),
        (
            "pizza",
            "Boulevard - Extra Large Sicilian",
            "Pepperoni",
            "Pepperoni (Normal)",
        ),
        (
            "burger",
            "Double Cheeseburger Boardwalk Box",
            "Bundle Side",
            "Snack Fries",
        ),
    ],
)
def test_item_details_use_display_names_and_show_defaults(
    store_id: str, item_name: str, group_prefix: str, expected_default: str
):
    env = store_env(store_id)
    details = env.tools.get_item_details(store_id, item_name)

    assert details["name"] == item_name
    assert details["base_price"].startswith("$")
    group = _group(details, group_prefix)
    assert re.fullmatch(rf"{group_prefix}.* \(\d+\)", group["name"])
    assert expected_default in group["default"]
    assert expected_default in _modifier_names(group)
    serialized = json.dumps(details)
    assert len(serialized) < _MAX_LOOKUP_CHARS
    assert not UUID_RE.search(serialized)
    assert "group_id" not in serialized and "option_id" not in serialized


def test_item_details_show_modifier_prices_and_nested_groups():
    env = store_env("burger")
    details = env.tools.get_item_details("burger", "Double Cheeseburger Boardwalk Box")

    sandwich = _group(details, "Double Cheeseburger")
    assert sandwich["min_count"] == 1
    (choice,) = sandwich["modifiers"]
    assert choice["price"] == "+$3.59"
    (opened,) = choice["opens"]
    nested_names = [g["name"] for g in details["nested_groups"]]
    assert opened in nested_names


def test_long_modifier_lists_are_capped_with_a_count():
    env = store_env("burger")
    details = env.tools.get_item_details("burger", "Double Cheeseburger Boardwalk Box")

    drink = _group(details, "Drink")
    total = env.tools.get_modifier_group("burger", drink["name"])["total_modifiers"]
    assert drink["more_modifiers"] > 0
    assert len(drink["modifiers"]) + drink["more_modifiers"] == total
    assert "get_modifier_group" in details["note"]


def test_get_modifier_group_pages_and_filters():
    env = store_env("burger")
    drink_name = _group(
        env.tools.get_item_details("burger", "Double Cheeseburger Boardwalk Box"),
        "Drink",
    )["name"]

    names: list[str] = []
    pages = 0
    offset = 0
    while offset is not None:
        page = env.tools.get_modifier_group("burger", drink_name, offset=offset)
        assert len(json.dumps(page)) < _MAX_LOOKUP_CHARS
        names += _modifier_names(page)
        pages += 1
        offset = page.get("next_offset")
    assert pages > 1
    assert len(names) == len(set(names)) == page["total_modifiers"]

    filtered = env.tools.get_modifier_group(
        "burger", drink_name, query="diet spiced cherry"
    )
    assert _modifier_names(filtered) == [
        "Small Diet Spiced Cherry Soda",
        "Medium Diet Spiced Cherry Soda",
        "Large Diet Spiced Cherry Soda",
    ]


def test_search_menu_pages_and_bounds_results():
    env = store_env("burger")
    first = env.tools.search_menu("burger")
    second = env.tools.search_menu("burger", offset=first["next_offset"])

    assert first["total_matches"] > len(first["items"]) + len(second["items"])
    assert {i["name"] for i in first["items"]}.isdisjoint(
        i["name"] for i in second["items"]
    )
    for page in (first, second):
        assert len(json.dumps(page)) < _MAX_LOOKUP_CHARS
        assert not UUID_RE.search(json.dumps(page))


def test_search_menu_ranks_name_matches():
    env = store_env("cafe")
    result = env.tools.search_menu("cafe", query="hazelnut")

    assert "Hazelnut" in result["items"][0]["name"]


def test_search_menu_lists_matching_modifiers_with_their_groups():
    env = store_env("cafe")
    display_names = set(_labels(env).values())

    # A modifier that is not a menu item is still found, with the groups offering it.
    lentil = env.tools.search_menu("cafe", query="lentil")
    assert lentil["total_matches"] == 0
    assert [m["name"] for m in lentil["matching_modifiers"]] == ["Cup of Lentil Soup"]

    oat = env.tools.search_menu("cafe", query="oat milk")
    match = next(m for m in oat["matching_modifiers"] if m["name"] == "Oat Milk")
    assert match["groups"] and set(match["groups"]) <= display_names
    assert len(match["groups"]) <= 3

    # Only the first page of a keyword search lists modifiers.
    assert "matching_modifiers" not in env.tools.search_menu("cafe")
    assert "matching_modifiers" not in env.tools.search_menu(
        "cafe", query="oat milk", offset=1
    )


def test_search_menu_filters_by_category():
    env = store_env("pizza")
    categories = env.tools.list_menu_categories("pizza")["categories"]
    specialty = next(c for c in categories if c["category"] == "Specialty Pizzas")

    result = env.tools.search_menu("pizza", category="Specialty Pizzas")
    assert result["total_matches"] == specialty["items"] + specialty.get(
        "unavailable_items", 0
    )
    assert {i["category"] for i in result["items"]} == {"Specialty Pizzas"}


def test_unavailable_reasons_hide_internal_item_ids():
    reasons: list[str] = []
    for store_id in STORES:
        env = store_env(store_id)
        item_ids = env.tools.db.stores[store_id].items
        texts = [
            get_environment(task=task_for(store_id))
            .get_policy()
            .rsplit("### Unavailable Items", 1)[1]
        ]
        offset = 0
        while offset is not None:
            page = env.tools.search_menu(store_id, offset=offset)
            texts += [i["not_available"] for i in page["items"] if "not_available" in i]
            offset = page.get("next_offset")
        reasons += texts[1:]
        for text in texts:
            assert not any(iid in text for iid in item_ids), (store_id, text)
    assert reasons


def test_unknown_names_raise_errors():
    env = store_env("burger")
    with pytest.raises(ValueError, match="not found in store 'burger'"):
        env.tools.get_item_details("burger", "chocolate shake please")
    with pytest.raises(ValueError, match="No modifier group named 'Drink'"):
        env.tools.get_modifier_group("burger", "Drink")


def test_lookup_names_are_accepted_by_add_to_cart_in_every_store():
    burger = store_env("burger")
    details = burger.tools.get_item_details(
        "burger", "Double Cheeseburger Boardwalk Box"
    )
    pickles_group = _group(details, "Double Cheeseburger")["modifiers"][0]["opens"][0]
    nugget_sauce_group = next(
        m["opens"][0]
        for m in _group(details, "Boardwalk Box Snack")["modifiers"]
        if m["name"] == "4 Piece Chicken Nuggets"
    )
    drink_group = _group(details, "Drink")["name"]
    drink = _modifier_names(
        burger.tools.get_modifier_group(
            "burger", drink_group, query="medium diet spiced"
        )
    )[0]
    burger.tools.add_to_cart(
        store_id="burger",
        item_id=details["name"],
        modifiers=[
            {"option_id": "No Pickles", "group_id": pickles_group},
            {"option_id": "Habanero Dipping Sauce", "group_id": nugget_sauce_group},
            {"option_id": drink, "group_id": drink_group},
        ],
    )

    pizza = store_env("pizza")
    pizza_name = "Meatpacking - Extra Large Sicilian"
    mushrooms = _group(pizza.tools.get_item_details("pizza", pizza_name), "Mushrooms")
    pizza.tools.add_to_cart(
        store_id="pizza",
        item_id=pizza_name,
        modifiers=[{"option_id": "Mushrooms (Extra)", "group_id": mushrooms["name"]}],
    )

    cafe = store_env("cafe")
    latte = cafe.tools.get_item_details("cafe", "Latte")
    milk = _group(latte, "Milk")
    assert "Oat Milk" in _modifier_names(milk)
    cafe.tools.add_to_cart(
        store_id="cafe",
        item_id="Latte",
        modifiers=[
            {
                "option_id": "Small",
                "group_id": _group(latte, "Size")["name"],
            },
            {"option_id": "Oat Milk", "group_id": milk["name"]},
        ],
    )

    for env, expected in (
        (burger, {"No Pickles", "Medium Diet Spiced Cherry Soda"}),
        (pizza, {"Mushrooms (Extra)"}),
        (cafe, {"Small", "Oat Milk"}),
    ):
        (line,) = env.tools.db.cart.items
        assert expected <= {m.name for m in line.modifiers}


def _label_counts(labels) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels or []:
        name, qty = (
            (label, 1) if isinstance(label, str) else (label["name"], label["quantity"])
        )
        counts[name] = counts.get(name, 0) + qty
    return counts


def _defaults_from_lookup(
    env: Environment, store_id: str, details: dict, size: str
) -> dict:
    """Per-group default counts the lookup output promises once `size` is chosen."""
    shown = {
        g["name"]: g
        for g in details["modifier_groups"] + details.get("nested_groups", [])
    }
    size_group = next(
        g for g in details["modifier_groups"] if size in _modifier_names(g)
    )
    expected = {size_group["name"]: {size: 1}}
    for group in details["modifier_groups"]:
        if group is not size_group:
            expected[group["name"]] = _label_counts(group.get("default"))
    entry = next(
        m
        for m in size_group["modifiers"]
        if (m if isinstance(m, str) else m["name"]) == size
    )
    if isinstance(entry, dict):
        opens = entry.get("opens", [])
        if isinstance(opens, str):
            opens = size_group["nested_group_sets"][opens]
        overrides = entry.get("defaults_when_selected", {})
        for name in opens:
            if name in overrides:
                expected[name] = _label_counts(overrides[name])
            elif name in shown:
                expected[name] = _label_counts(shown[name].get("default"))
            else:
                group = env.tools.get_modifier_group(
                    store_id, name, item_id=details["name"]
                )
                expected[name] = _label_counts(group.get("default"))
    return {k: v for k, v in expected.items() if v}


def _cart_counts(env: Environment, store_id: str) -> dict:
    display = group_display_names(env.tools.db.stores[store_id])
    counts: dict = {}
    for m in env.tools.db.cart.items[-1].modifiers:
        group = counts.setdefault(display.get(m.group_id, m.group_id), {})
        group[m.name] = group.get(m.name, 0) + m.quantity
    return counts


@pytest.mark.parametrize(
    "item_name, size, group_prefix, default_name, quantity",
    [
        ("Vanilla Bean Shake", "Large", "Ice Cream Scoops", "Ice Cream Scoop", 4),
        ("Flat White", "Medium", "Espresso Shots", "Espresso Shot", 3),
        (
            "Vanilla Caramel Latte",
            "Medium",
            "Syrups",
            "Vanilla Syrup",
            3,
        ),
        (
            "Blended Latte",
            "Large",
            "Espresso Shots",
            "Espresso Shot",
            4,
        ),
    ],
)
def test_lookup_default_quantities_match_the_cart(
    item_name: str, size: str, group_prefix: str, default_name: str, quantity: int
):
    env = store_env("cafe")
    details = env.tools.get_item_details("cafe", item_name)
    promised = _defaults_from_lookup(env, "cafe", details, size)
    group_name = next(g for g in promised if g.startswith(group_prefix))
    assert promised[group_name] == {default_name: quantity}

    env.tools.add_to_cart(
        store_id="cafe", item_id=item_name, modifiers=[{"option_id": size}]
    )
    assert _cart_counts(env, "cafe") == promised


@pytest.mark.parametrize("store_id", ["cafe", "pizza"])
def test_lookup_defaults_match_the_cart_for_every_size(store_id: str):
    env = store_env(store_id)
    store = env.tools.db.stores[store_id]
    checked, mismatches = 0, []
    for name in sorted({i.name for i in store.items.values()}):
        details = env.tools.get_item_details(store_id, name)
        size_group = next(
            (
                g
                for g in details["modifier_groups"]
                if g["name"].split(" (")[0].endswith("Size")
            ),
            None,
        )
        if "not_available" in details or size_group is None:
            continue
        for size in _modifier_names(size_group):
            promised = _defaults_from_lookup(env, store_id, details, size)
            try:
                env.tools.add_to_cart(
                    store_id=store_id,
                    item_id=name,
                    modifiers=[{"group_id": size_group["name"], "option_id": size}],
                )
            except ValueError as e:
                assert "requires at least" in str(e), (name, size, str(e))
                continue
            checked += 1
            if _cart_counts(env, store_id) != promised:
                mismatches.append((name, size))
            env.tools.clear_cart()
    assert checked > 0
    assert not mismatches, mismatches[:10]


def test_option_in_two_applicable_groups_needs_a_group():
    env = store_env(BURGER)
    item = "Pier Burger Combo"
    drink = _first_option(env, item, "Drink")
    burger_label = _label_opened_by(env, item, "Pier Burger", "Ketchup")
    fries_label = _label_opened_by(env, item, "Medium Fries", "Ketchup")
    with pytest.raises(ValueError, match="more than one group") as err:
        env.tools.add_to_cart(
            BURGER, item, modifiers=[{"option_id": drink}, {"option_id": "Ketchup"}]
        )
    # Each numbered group is named with the chosen option that opened it.
    assert f"'{burger_label}' (for 'Pier Burger')" in str(err.value)
    assert f"'{fries_label}' (for 'Medium Fries')" in str(err.value)
    assert env.tools.db.cart.items == []

    env.tools.add_to_cart(
        BURGER,
        item,
        modifiers=[
            {"option_id": drink},
            {"group_id": fries_label, "option_id": "Ketchup"},
        ],
    )
    assert (fries_label, "Ketchup") in _line_modifiers(env)
    assert (burger_label, "Ketchup") not in _line_modifiers(env)


def test_combo_size_and_doneness_named_medium_need_a_group():
    env = store_env(BURGER)
    item = "Sunset Burger Combo"
    drink = _first_option(env, item, "Combo Drink Choices")
    details = env.tools.get_item_details(BURGER, item)
    doneness = next(
        g["name"]
        for g in _all_groups(details)
        if g["name"].startswith("Cooking Temperature")
    )
    with pytest.raises(ValueError, match="more than one group"):
        env.tools.add_to_cart(
            BURGER, item, modifiers=[{"option_id": drink}, {"option_id": "Medium"}]
        )
    env.tools.add_to_cart(
        BURGER,
        item,
        modifiers=[
            {"option_id": drink},
            {"group_id": doneness, "option_id": "Well Done"},
        ],
    )
    assert (doneness, "Well Done") in _line_modifiers(env)


def test_named_group_without_the_option_is_an_error_not_a_reroute():
    env = store_env(BURGER)
    item = "Pier Burger Combo"
    drink = _first_option(env, item, "Drink")
    fries_label = _label_opened_by(env, item, "Medium Fries", "Ketchup")
    with pytest.raises(ValueError, match="is not in"):
        env.tools.add_to_cart(
            BURGER,
            item,
            modifiers=[
                {"option_id": drink},
                {"group_id": fries_label, "option_id": "No Ketchup"},
            ],
        )


def test_explicit_group_id_picks_its_own_portion_group():
    env = store_env(PIZZA)
    sandwich_label = _label_opened_by(
        env, MIX_MATCH, "Chicken Parm Hero", "Parmesan (Light)"
    )
    pizza_label = _label_opened_by(env, MIX_MATCH, LOADED_PEPPERONI, "Parmesan (Light)")
    assert sandwich_label != pizza_label
    env.tools.add_to_cart(
        PIZZA,
        MIX_MATCH,
        modifiers=[
            {"option_id": LOADED_PEPPERONI},
            {"option_id": "Chicken Parm Hero"},
            {"group_id": sandwich_label, "option_id": "Parmesan (Light)"},
        ],
    )
    mods = _line_modifiers(env)
    assert (sandwich_label, "Parmesan (Light)") in mods
    assert (pizza_label, "Parmesan (Normal)") in mods


@pytest.mark.parametrize(
    "entry",
    [
        {"option_id": "Parmesan (Light)"},
        {
            "group_id": "Parmesan",
            "option_id": "Parmesan (Light)",
        },
    ],
    ids=["no-group", "bare-group-name"],
)
def test_portion_shared_by_two_deal_choices_needs_a_numbered_group(entry):
    env = store_env(PIZZA)
    with pytest.raises(ValueError, match="more than one group"):
        env.tools.add_to_cart(
            PIZZA,
            MIX_MATCH,
            modifiers=[
                {"option_id": LOADED_PEPPERONI},
                {"option_id": "Chicken Parm Hero"},
                entry,
            ],
        )


def test_portion_removal_with_mismatched_group_id_is_rejected():
    env = store_env(PIZZA)
    with pytest.raises(ValueError, match="is not in"):
        env.tools.add_to_cart(
            PIZZA,
            "Large Classic Pie",
            modifiers=[{"group_id": "Tomato Sauce (1)", "option_id": "No Anchovies"}],
        )


def test_add_to_cart_accepts_menu_group_display_name():
    env = store_env(BURGER)
    env.tools.add_to_cart(
        store_id="burger",
        item_id="Double Cheeseburger Boardwalk Box",
        modifiers=[
            {"option_id": "Double Cheeseburger"},
            {
                "option_id": "No Pickles",
                "group_id": "Double Cheeseburger Toppings (1)",
            },
            {"option_id": "4 Piece Spicy Chicken Nuggets"},
            {
                "option_id": "Habanero Dipping Sauce",
                "group_id": "Nugget Dipping Sauce (1)",
            },
            {"option_id": "Snack Fries"},
            {"option_id": "Medium Diet Spiced Cherry Soda"},
        ],
    )
    (line,) = env.tools.db.cart.items
    assert {
        "No Pickles",
        "4 Piece Spicy Chicken Nuggets",
        "Habanero Dipping Sauce",
        "Medium Diet Spiced Cherry Soda",
    } <= {m.name for m in line.modifiers}


def _pizza_size_group(env: Environment, item_name: str) -> str:
    details = env.tools.get_item_details(PIZZA, item_name)
    return next(
        g["name"] for g in details["modifier_groups"] if g["name"].startswith("Size (")
    )


def test_explicit_nested_group_id_is_honored():
    env = store_env(PIZZA)
    size = _pizza_size_group(env, "Cheese Pizza")
    env.tools.add_to_cart(
        store_id=PIZZA,
        item_id="Cheese Pizza",
        modifiers=[
            {"group_id": size, "option_id": "Large"},
            {"option_id": "City Thin"},
            {"group_id": "Combo (1)", "option_id": "Yes"},
            {"option_id": "Diet Cola"},
            {"option_id": "Large", "group_id": "Drink Size (1)"},
        ],
    )
    mods = _line_modifiers(env)
    assert ("Drink Size (1)", "Large") in mods
    assert (size, "Large") in mods


def test_inactive_nested_group_error_is_rejected():
    env = store_env(PIZZA)
    size = _pizza_size_group(env, "Cheese Pizza")
    with pytest.raises(ValueError, match="not a valid modifier option"):
        env.tools.add_to_cart(
            store_id=PIZZA,
            item_id="Cheese Pizza",
            modifiers=[
                {"group_id": size, "option_id": "Large"},
                {"option_id": "City Thin"},
                {"option_id": "Large", "group_id": "Drink Size (1)"},
            ],
        )


def test_half_and_half_supreme_half_keeps_onions_in_pizza_store(
    environment: Environment,
):
    # The Half-and-Half Pizza lives next to portioned pizzas; its Supreme half still cannot drop onions.
    with pytest.raises(ValueError, match="'No Onions' is not in"):
        environment.tools.add_to_cart(
            store_id="pizza",
            item_id="Half-and-Half Pizza",
            modifiers=[
                {"option_id": "Medium"},
                {"option_id": "City Thin"},
                {"group_id": "Left Half Base (1)", "option_id": "Pepperoni"},
                {"group_id": "Right Half Base (1)", "option_id": "Supreme"},
                {"group_id": "Right Half Edits (1)", "option_id": "No Onions"},
            ],
        )


@pytest.mark.parametrize(
    "item, option",
    [("Crispy Chicken Sandwich", "Pickles"), ("Bacon Avocado Burger", "Chipotle Mayo")],
)
def test_plain_add_ons_can_be_chosen_once(item, option):
    with pytest.raises(ValueError, match="exceeds max_per_option=1"):
        store_env(BURGER).tools.add_to_cart(
            store_id=BURGER,
            item_id=item,
            modifiers=[{"option_id": option, "quantity": 2}],
        )


@pytest.mark.parametrize(
    "store_id, item, option",
    [
        (BURGER, "Crispy Chicken Sandwich", "Pickles"),
        (BURGER, "Bacon Avocado Burger", "Chipotle Mayo"),
        (BURGER, "Crispy Chicken Combo", "Mayo"),
        (BURGER, "Pier Burger Combo", "Pickles"),
        (BURGER, "Classic Chicken Sandwich", "No Mayonnaise"),
    ],
)
def test_extra_volume_is_rejected_where_the_menu_has_no_portions(
    store_id, item, option
):
    env = store_env(store_id)
    with pytest.raises(ValueError, match="not offered"):
        env.tools.add_to_cart(
            store_id=store_id,
            item_id=item,
            modifiers=[{"option_id": option, "volume": "extra"}],
        )


def test_portion_volumes_still_work_on_pizza_toppings():
    env = store_env(PIZZA)
    env.tools.add_to_cart(
        store_id=PIZZA,
        item_id="Mushroom Pizza",
        modifiers=[
            {"option_id": "Medium"},
            {"option_id": "Feta", "volume": "extra"},
        ],
    )
    feta = next(m for m in env.tools.db.cart.items[0].modifiers if m.name == "Feta")
    assert feta.volume == "extra"


@pytest.mark.parametrize(
    "phrase", ["Half Sweet", "25% sweet", "50% sweet", "Light Sweet"]
)
def test_sweetness_phrases_are_not_mapped_to_a_different_option(phrase: str):
    env = store_env(CAFE)
    env.tools.add_to_cart(
        CAFE,
        "Half Sandwich Lunch",
        modifiers=_LUNCH_WITH_MATCHA + [{"option_id": "Less Sweet"}],
    )
    with pytest.raises(ValueError, match=f"'{phrase}' is not a valid modifier option"):
        env.tools.add_to_cart(
            CAFE,
            "Half Sandwich Lunch",
            modifiers=_LUNCH_WITH_MATCHA + [{"option_id": phrase}],
        )


def test_quantity_zero_on_a_non_default_is_rejected():
    env = store_env(CAFE)
    with pytest.raises(ValueError, match="not on 'Latte' by default"):
        env.tools.add_to_cart(
            store_id=CAFE,
            item_id="Latte",
            modifiers=[
                {"option_id": "Medium"},
                {"option_id": "Oat Milk", "quantity": 0},
            ],
        )


def test_a_new_line_with_x_and_no_x_is_an_error():
    env = store_env(BURGER)
    with pytest.raises(ValueError, match="cannot both be selected"):
        env.tools.add_to_cart(
            BURGER,
            "Classic Cheeseburger",
            modifiers=[{"option_id": "No Pickles"}, {"option_id": "Pickles"}],
        )


def test_update_replaces_modifiers_and_reverts_omitted_removals():
    env = store_env(CAFE)
    env.tools.db.cart.current_time = "2026-06-22 09:00"
    env.tools.add_to_cart(
        CAFE, "Steel-Cut Oatmeal", modifiers=[{"option_id": "No Toasted Walnuts"}]
    )
    initial_names = {name for _, name in _line_modifiers(env)}
    assert "No Toasted Walnuts" in initial_names
    assert "Brown Sugar" in initial_names
    assert "Toasted Walnuts" not in initial_names
    # Omitting No Toasted Walnuts restores Toasted Walnuts default
    env.tools.update_cart_item("line_1", modifiers=[{"option_id": "Oat Milk"}])
    names = {name for _, name in _line_modifiers(env)}
    assert "No Toasted Walnuts" not in names
    assert "Toasted Walnuts" in names
    assert "Brown Sugar" in names
    assert "Oat Milk" in names


@pytest.mark.parametrize(
    "store_id, item, removal, other",
    [
        (BURGER, "Classic Cheeseburger", "No Pickles", "Smoked Bacon"),
        (CAFE, "Mocha", "No Whipped Cream", "Oat Milk"),
    ],
)
def test_no_x_reverts_when_an_update_leaves_it_out(store_id, item, removal, other):
    env = store_env(store_id)
    env.tools.add_to_cart(store_id, item, modifiers=[{"option_id": removal}])
    env.tools.update_cart_item("line_1", modifiers=[{"option_id": other}])
    names = {name for _, name in _line_modifiers(env)}
    assert removal not in names
    assert other in names

    direct = store_env(store_id)
    direct.tools.add_to_cart(store_id, item, modifiers=[{"option_id": other}])
    assert env.tools.get_db_hash() == direct.tools.get_db_hash()


@pytest.mark.parametrize(
    "store_id, item, removal",
    [
        (BURGER, "Classic Cheeseburger", "No Pickles"),
        (CAFE, "Mocha", "No Whipped Cream"),
    ],
)
def test_empty_modifiers_on_update_restores_the_standard(store_id, item, removal):
    env = store_env(store_id)
    env.tools.add_to_cart(store_id, item, modifiers=[{"option_id": removal}])
    env.tools.update_cart_item("line_1", modifiers=[])
    assert removal not in {name for _, name in _line_modifiers(env)}

    plain = store_env(store_id)
    plain.tools.add_to_cart(store_id, item)
    assert env.tools.get_db_hash() == plain.tools.get_db_hash()


def test_choosing_x_on_update_clears_no_x():
    env = store_env(BURGER)
    env.tools.add_to_cart(
        BURGER, "Pier Burger", modifiers=[{"option_id": "No Pickles"}]
    )
    env.tools.update_cart_item("line_1", modifiers=[{"option_id": "Pickles"}])
    assert "No Pickles" not in {name for _, name in _line_modifiers(env)}
    plain = store_env(BURGER)
    plain.tools.add_to_cart(BURGER, "Pier Burger")
    assert env.tools.get_db_hash() == plain.tools.get_db_hash()

    cafe_env = store_env(CAFE)
    cafe_env.tools.add_to_cart(
        CAFE, "Mocha", modifiers=[{"option_id": "No Whipped Cream"}]
    )
    cafe_env.tools.update_cart_item(
        "line_1", modifiers=[{"option_id": "Light Whipped Cream"}]
    )
    names = {name for _, name in _line_modifiers(cafe_env)}
    assert "No Whipped Cream" not in names
    assert "Light Whipped Cream" in names


def test_quantity_zero_on_non_default_no_option_is_an_error():
    env = store_env(BURGER)
    env.tools.add_to_cart(BURGER, "Classic Cheeseburger")
    with pytest.raises(ValueError, match="not on 'Classic Cheeseburger' by default"):
        env.tools.update_cart_item(
            "line_1", modifiers=[{"option_id": "No Pickles", "quantity": 0}]
        )


def test_removed_default_without_a_no_option_requires_explicit_inclusion_on_update():
    pizza = "Meatpacking - Large Classic"
    env = store_env(PIZZA)
    env.tools.add_to_cart(PIZZA, pizza, modifiers=[{"option_id": "No Ham"}])
    env.tools.update_cart_item(
        "line_1",
        modifiers=[{"option_id": "No Ham"}, {"option_id": "Mushrooms (Normal)"}],
    )
    names = {name for _, name in _line_modifiers(env)}
    assert "Mushrooms (Normal)" in names
    assert not any(name.startswith("Ham") for name in names)

    env.tools.update_cart_item("line_1", modifiers=[{"option_id": "Ham (Light)"}])
    assert "Ham (Light)" in {name for _, name in _line_modifiers(env)}


def test_quantity_update_keeps_a_removed_default():
    pizza = "Meatpacking - Large Classic"
    env = store_env(PIZZA)
    env.tools.add_to_cart(
        store_id=PIZZA, item_id=pizza, modifiers=[{"option_id": "No Ham"}]
    )
    assert "Ham (Normal)" not in line_names(env)
    line_id = env.tools.db.cart.items[0].line_item_id

    env.tools.update_cart_item(line_item_id=line_id, quantity=2)
    assert "Ham (Normal)" not in line_names(env)
    assert env.tools.db.cart.items[0].quantity == 2

    direct = store_env(PIZZA)
    direct.tools.add_to_cart(
        store_id=PIZZA, item_id=pizza, quantity=2, modifiers=[{"option_id": "No Ham"}]
    )
    assert env.tools.get_db_hash() == direct.tools.get_db_hash()


def test_full_modifier_update_replaces_selections_completely():
    pizza = "Meatpacking - Large Classic"
    env = store_env(PIZZA)
    env.tools.add_to_cart(
        store_id=PIZZA, item_id=pizza, modifiers=[{"option_id": "No Ham"}]
    )
    line = env.tools.db.cart.items[0]

    # Including "No Ham" in the replacement list keeps Ham removed
    env.tools.update_cart_item(
        line_item_id=line.line_item_id,
        modifiers=[{"option_id": "No Ham"}, {"option_id": "Mushrooms (Normal)"}],
    )
    assert "Ham (Normal)" not in line_names(env)
    assert "Mushrooms (Normal)" in line_names(env)

    # Omitting "No Ham" from the replacement list reverts Ham to its default
    env.tools.update_cart_item(
        line_item_id=line.line_item_id,
        modifiers=[{"option_id": "Mushrooms (Normal)"}],
    )
    assert "Ham (Normal)" in line_names(env)


def test_remove_and_clear_cart():
    env = store_env(BURGER)
    env.tools.add_to_cart(store_id=BURGER, item_id="Double Griddle Burger")
    env.tools.add_to_cart(store_id=BURGER, item_id="Regular Chocolate Shake")
    first = env.tools.db.cart.items[0].line_item_id

    env.tools.remove_from_cart(line_item_id=first)
    assert [li.name for li in env.tools.db.cart.items] == ["Regular Chocolate Shake"]
    with pytest.raises(ValueError, match="not found"):
        env.tools.update_cart_item(line_item_id=first, quantity=2)

    env.tools.clear_cart()
    assert env.tools.db.cart.items == []
    assert env.tools.db.cart.store_id == BURGER


def test_set_fulfillment_validates_time_against_hours_and_lead_time():
    env = store_env(CAFE, "mobile")
    with pytest.raises(ValueError, match="pickup needs at least 20 minutes"):
        env.tools.set_fulfillment("pickup", "11:10")
    with pytest.raises(ValueError, match="outside store hours"):
        env.tools.set_fulfillment("pickup", "21:30")
    with pytest.raises(ValueError, match="delivery needs at least 45 minutes"):
        env.tools.set_fulfillment("delivery", "11:30")
    cart = env.tools.set_fulfillment("delivery", "6:30 pm")
    assert (cart.order_type, cart.scheduled_time) == ("delivery", "18:30")


def test_drive_thru_is_pickup_now_only():
    env = store_env(BURGER)
    with pytest.raises(ValueError, match="drive-thru"):
        env.tools.set_fulfillment("delivery")
    with pytest.raises(ValueError, match="drive-thru"):
        env.tools.set_fulfillment("pickup", "13:00")
    env.tools.add_to_cart(store_id=BURGER, item_id="Double Griddle Burger")
    order = env.tools.place_order(order_type="pickup")
    assert (order.customer_name, order.scheduled_time, order.channel) == (
        "Guest",
        "ASAP",
        "drive_thru",
    )


def test_availability_uses_the_scheduled_time_and_method():
    env = store_env(CAFE, "mobile")
    env.tools.set_fulfillment("pickup", "16:30")
    with pytest.raises(ValueError, match="not available at 16:30 on Monday"):
        env.tools.add_to_cart(store_id=CAFE, item_id="Half Sandwich Lunch")
    env.tools.add_to_cart(store_id=CAFE, item_id="Latte")
    with pytest.raises(ValueError, match="not available for delivery"):
        env.tools.set_fulfillment("delivery", "17:00")


def test_place_order_rechecks_items_at_the_order_time():
    env = store_env(CAFE, "mobile")
    env.tools.add_to_cart(
        store_id=CAFE, item_id="Half Sandwich Lunch", modifiers=_LUNCH_WITH_MATCHA
    )
    with pytest.raises(ValueError, match="Half Sandwich Lunch"):
        env.tools.place_order(
            scheduled_time="16:00",
            customer_name="Maria Lopez",
            customer_phone="949-555-0118",
        )


def test_mobile_orders_need_name_phone_address_and_delivery_minimum():
    env = store_env(BURGER, "mobile")
    env.tools.add_to_cart(store_id=BURGER, item_id="Small Fries")
    with pytest.raises(ValueError, match="full name"):
        env.tools.place_order()
    with pytest.raises(ValueError, match="10-digit phone"):
        env.tools.place_order(customer_name="Maria Lopez", customer_phone="555-0142")
    with pytest.raises(ValueError, match="address"):
        env.tools.place_order(
            order_type="delivery",
            customer_name="Maria Lopez",
            customer_phone="9495550118",
        )
    with pytest.raises(ValueError, match=r"at least \$10.00"):
        env.tools.place_order(
            order_type="delivery",
            customer_name="Maria Lopez",
            customer_phone="9495550118",
            delivery_address="12 Main Street",
        )


def test_asap_needs_the_store_open_long_enough():
    env = store_env(CAFE, "mobile", now="2026-06-22 20:50")
    env.tools.add_to_cart(store_id=CAFE, item_id="Cream Scone")
    with pytest.raises(ValueError, match="too late"):
        env.tools.place_order(customer_name="Maria Lopez", customer_phone="9495550118")


def test_order_totals_add_modifier_prices_quantities_and_delivery_fee():
    env = store_env(BURGER, "mobile")
    store = env.tools.db.stores[BURGER]
    burger = next(i for i in store.items.values() if i.name == "Classic Cheeseburger")
    bacon = next(
        o
        for gid in burger.modifier_group_ids
        for o in store.modifier_groups[gid].options.values()
        if o.name == "Smoked Bacon"
    )
    assert bacon.price_cents > 0
    with_bacon = burger.base_price_cents + bacon.price_cents

    cart = env.tools.add_to_cart(
        BURGER,
        "Classic Cheeseburger",
        quantity=2,
        modifiers=[{"option_id": "Smoked Bacon"}],
    )
    assert cart.items[0].line_total_cents == 2 * with_bacon
    env.tools.add_to_cart(
        BURGER, "Classic Cheeseburger", modifiers=[{"option_id": "Smoked Bacon"}]
    )
    env.tools.add_to_cart(
        BURGER, "Classic Cheeseburger", modifiers=[{"option_id": "No Pickles"}]
    )
    order = env.tools.place_order(
        order_type="delivery",
        scheduled_time="13:00",
        customer_name="Maria Lopez",
        customer_phone="9495550118",
        delivery_address="12 Main Street",
    )
    assert sorted((li.quantity, li.line_total_cents) for li in order.items) == [
        (1, burger.base_price_cents),
        (1, with_bacon),
        (2, 2 * with_bacon),
    ]
    assert order.subtotal_cents == 3 * with_bacon + burger.base_price_cents
    assert order.delivery_fee_cents == store.delivery_fee_cents > 0
    assert order.total_cents == order.subtotal_cents + order.delivery_fee_cents

    pickup = store_env(BURGER, "mobile")
    pickup.tools.add_to_cart(BURGER, "Classic Cheeseburger")
    order = pickup.tools.place_order(
        customer_name="Maria Lopez", customer_phone="9495550118"
    )
    assert order.delivery_fee_cents == 0
    assert order.total_cents == order.subtotal_cents == burger.base_price_cents


@pytest.mark.parametrize(
    "spoken, written",
    [
        ("Twelve Oak Street", "12 Oak St"),
        ("twenty Elm Street", "20 Elm St"),
        ("Four Twenty Oak Avenue", "420 Oak Ave"),
        ("Seven Forty-Two Linden Terrace", "742 Linden Ter"),
        ("One Oh Five Main Street", "105 Main St"),
        ("Twelve Hundred Pine Road", "1200 Pine Rd"),
        ("Four hundred and twenty Oak Avenue", "420 Oak Ave"),
        ("One thousand and five Pine Parkway", "1005 Pine Pkwy"),
        ("12 Fifth Avenue", "12 Fifth Ave"),
    ],
)
def test_spoken_house_numbers_are_written_as_digits(spoken, written):
    assert _normalize_address(spoken) == written
    assert (
        _normalize_address("420 Maple St #3B")
        == _normalize_address("420 Maple Street Apt 3B")
        == _normalize_address("420 Maple Street Unit 3B")
    )
    assert _normalize_address("80 Cedar Circle") == "80 Cedar Cir"
    assert _normalize_address("9 Pacific Highway") == "9 Pacific Hwy"


def test_full_slot_offers_nearest_alternatives():
    result = store_env(CAFE).tools.check_table_availability(
        CAFE, "2026-06-26", "19:00", 4
    )
    assert result["available"] is False
    assert [a["time"] for a in result["alternatives"]][:2] == ["18:30", "19:30"]
    outdoor = store_env(CAFE).tools.check_table_availability(
        CAFE, "2026-06-26", "18:30", 2, "outdoor"
    )
    assert outdoor["available"] is False and "outdoor" in outdoor["reason"]
    indoor = store_env(CAFE).tools.check_table_availability(
        CAFE, "2026-06-26", "18:30", 2, "indoor"
    )
    assert indoor["available"] is True


def test_booking_rules_and_high_chairs():
    tools = store_env(CAFE).tools
    assert (
        "parties of 1 to 8"
        in tools.check_table_availability(CAFE, "2026-06-27", "18:00", 10)["reason"]
    )
    assert (
        "high chairs"
        in tools.check_table_availability(CAFE, "2026-06-27", "12:00", 3, "any", 1)[
            "reason"
        ]
    )
    assert (
        tools.check_table_availability(CAFE, "2026-06-27", "12:00", 3)["available"]
        is True
    )
    assert (
        "from 07:30 to 20:00"
        in tools.check_table_availability(CAFE, "2026-06-27", "20:30", 2)["reason"]
    )
    assert (
        "30 days ahead"
        in tools.check_table_availability(CAFE, "2026-07-30", "12:00", 2)["reason"]
    )
    assert store_env(PIZZA).tools.check_table_availability(
        PIZZA, "2026-06-27", "18:30", 4
    )["available"]
    with pytest.raises(ValueError, match="does not take table reservations"):
        store_env(BURGER).tools.check_table_availability(
            BURGER, "2026-06-27", "18:30", 4
        )


def test_make_find_update_cancel_reservation():
    tools = store_env(CAFE).tools
    booking = tools.make_reservation(
        CAFE, "2026-06-26", "18:30", 4, "jo kim", "408 555 0132", high_chairs=1
    )
    assert (booking.reservation_id, booking.seating, booking.customer_phone) == (
        "RES-0021",
        "indoor",
        "408-555-0132",
    )
    assert [r.reservation_id for r in tools.find_reservations("(408) 555-0132")] == [
        "RES-0021"
    ]
    updated = tools.update_reservation("RES-0021", time="18:00", party_size=5)
    assert (updated.time, updated.party_size, updated.high_chairs) == ("18:00", 5, 1)
    with pytest.raises(ValueError, match="not available"):
        tools.update_reservation("RES-0021", time="19:00")
    with pytest.raises(ValueError, match="not available"):
        tools.make_reservation(CAFE, "2026-06-26", "19:00", 2, "Jo Kim", "4085550132")
    tools.cancel_reservation("RES-0021")
    with pytest.raises(ValueError, match="already cancelled"):
        tools.cancel_reservation("RES-0021")


def test_reservation_dates_are_normalized_and_malformed_ones_rejected():
    env = store_env(CAFE)
    full = env.tools.check_table_availability(CAFE, "2026-6-26", "19:00", 2)
    assert full["available"] is False and full["date"] == "2026-06-26"
    with pytest.raises(ValueError, match="not available"):
        env.tools.make_reservation(
            CAFE, "2026-6-26", "19:00", 2, "Ann Lee", "415-555-0111"
        )
    booking = env.tools.make_reservation(
        CAFE, "2026-6-27", "18:00", 2, "Ann Lee", "415-555-0111"
    )
    assert booking.date == "2026-06-27"

    iso = store_env(CAFE)
    iso.tools.make_reservation(
        CAFE, "2026-06-27", "18:00", 2, "Ann Lee", "415-555-0111"
    )
    assert env.tools.get_db_hash() == iso.tools.get_db_hash()

    with pytest.raises(ValueError, match="Use a date like"):
        env.tools.make_reservation(
            CAFE, "06/27/2026", "18:00", 2, "Ann Lee", "415-555-0111"
        )


def test_update_reservation_changes_contact_details():
    tools = store_env(CAFE).tools
    booking = tools.make_reservation(
        CAFE, "2026-06-29", "18:00", 2, "Jo Kim", "4085550132", seating="outdoor"
    )
    updated = tools.update_reservation(
        booking.reservation_id,
        party_size=3,
        customer_name="  ann   lee ",
        customer_phone="(415) 555-0111",
    )
    assert (updated.customer_name, updated.customer_phone) == (
        "Ann Lee",
        "415-555-0111",
    )
    assert (updated.party_size, updated.seating) == (3, "outdoor")
    assert tools.find_reservations("4155550111") == [updated]
    assert tools.find_reservations("4085550132") == []


def test_new_reservation_ids_skip_ids_already_taken():
    env = store_env(CAFE)
    taken = env.tools.db.reservations["RES-0001"].model_copy(
        update={"reservation_id": "RES-0022"}
    )
    env.tools.update_db({"reservations": {"RES-0022": taken.model_dump()}})
    booking = env.tools.make_reservation(
        CAFE, "2026-06-29", "18:00", 2, "Maria Lopez", "9495550118"
    )
    assert booking.reservation_id == "RES-0023"


@pytest.mark.parametrize("key", ["orders", "stores"])
def test_update_db_rejects_unknown_keys(key: str):
    env = store_env(CAFE)
    with pytest.raises(ValueError, match="Unknown food DB keys"):
        env.tools.update_db({key: {}})


def test_get_cart_returns_the_session_cart():
    env = store_env(BURGER)
    env.tools.add_to_cart(BURGER, "Double Griddle Burger")
    cart = env.tools.get_cart()
    assert cart.store_id == BURGER
    assert [li.name for li in cart.items] == ["Double Griddle Burger"]


def test_quantity_zero_removes_a_line_and_the_last_removal_keeps_the_session():
    env = store_env(CAFE, "mobile")
    env.tools.set_fulfillment("delivery", "18:00")
    env.tools.add_to_cart(CAFE, "Cream Scone")
    env.tools.add_to_cart(CAFE, "Cream Scone", quantity=2)
    cart = env.tools.update_cart_item("line_1", quantity=0)
    assert [li.line_item_id for li in cart.items] == ["line_2"]
    cart = env.tools.remove_from_cart("line_2")
    assert cart.items == []
    assert (cart.store_id, cart.channel, cart.order_type, cart.scheduled_time) == (
        CAFE,
        "mobile",
        "delivery",
        "18:00",
    )


def test_transfer_marks_the_session_as_transferred():
    env = store_env(BURGER)
    assert env.tools.transfer_to_human_agents("Refund for a past order.") == (
        "Transfer successful"
    )
    assert env.tools.db.cart.transferred_to_human


def _item(store_id: str, item: str):
    return ("add_to_cart", {"store_id": store_id, "item_id": item})


def _booking(**overrides) -> dict:
    return {
        "store_id": CAFE,
        "date": "2026-06-29",
        "time": "18:00",
        "party_size": 2,
        "customer_name": "Maria Lopez",
        "customer_phone": "9495550118",
        **overrides,
    }


@pytest.mark.parametrize(
    "store_id, channel, now, setup, name, arguments, error",
    [
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "add_to_cart",
            {"store_id": "taco", "item_id": "Taco"},
            "Store 'taco' not found",
            id="unknown-store",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "get_item_details",
            {"store_id": "pizza", "item_id": "Cheese Pizza"},
            "Active session store is 'burger'",
            id="other-store",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "add_to_cart",
            {"store_id": BURGER, "item_id": "Small Fries", "quantity": 0},
            "Quantity must be at least 1",
            id="add-quantity-zero",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "add_to_cart",
            {
                "store_id": BURGER,
                "item_id": "Small Fries",
                "modifiers": [{"quantity": 2}],
            },
            "must include an 'option_id'",
            id="modifier-without-option",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "update_cart_item",
            {"line_item_id": "line_1", "quantity": 2},
            "Cart is currently empty",
            id="update-empty-cart",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [_item(BURGER, "Small Fries")],
            "update_cart_item",
            {"line_item_id": "line_1", "quantity": -1},
            "Quantity must be at least 1",
            id="update-negative-quantity",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [_item(BURGER, "Small Fries")],
            "remove_from_cart",
            {"line_item_id": "line_9"},
            r"Line item 'line_9' not found in cart\. Current line_item_ids: \['line_1'\]",
            id="remove-unknown-line",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "place_order",
            {},
            "empty cart",
            id="place-empty-cart",
        ),
        pytest.param(
            None,
            "drive_thru",
            None,
            [],
            "set_fulfillment",
            {"order_type": "pickup"},
            "No active store",
            id="fulfillment-without-store",
        ),
        pytest.param(
            CAFE,
            "mobile",
            None,
            [],
            "set_fulfillment",
            {"order_type": "dine_in"},
            "pickup or delivery",
            id="mobile-dine-in",
        ),
        pytest.param(
            CAFE,
            "mobile",
            "2026-06-22 06:30",
            [],
            "set_fulfillment",
            {"order_type": "pickup"},
            "closed right now",
            id="asap-before-opening",
        ),
        pytest.param(
            BURGER,
            "drive_thru",
            None,
            [],
            "find_reservations",
            {"customer_phone": "9495550118"},
            "does not take table reservations",
            id="no-reservations-here",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(date="2026-06-21"),
            "already passed",
            id="past-date",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(time="18:15"),
            "every 30 minutes",
            id="off-grid-time",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(date="2026-06-22", time="10:30"),
            "already passed today",
            id="earlier-today",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(seating="patio"),
            "Seating options here",
            id="unknown-seating",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(customer_name=" "),
            "A name is required",
            id="booking-without-name",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(customer_phone="555-0142"),
            "10-digit phone",
            id="booking-short-phone",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "make_reservation",
            _booking(high_chairs=-1),
            "high_chairs can't be negative",
            id="negative-high-chairs",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "find_reservations",
            {"customer_phone": "555-0142"},
            "10-digit phone",
            id="find-short-phone",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "cancel_reservation",
            {"reservation_id": "RES-9999"},
            "No reservation 'RES-9999'",
            id="unknown-reservation",
        ),
        pytest.param(
            CAFE,
            "drive_thru",
            None,
            [],
            "update_reservation",
            {"reservation_id": "RES-0001", "customer_phone": "555"},
            "10-digit phone",
            id="update-short-phone",
        ),
    ],
)
def test_tool_errors_are_reported_to_the_agent(
    store_id, channel, now, setup, name, arguments, error
):
    env = (
        get_environment()
        if store_id is None
        else store_env(store_id, channel, now or DEFAULT_NOW)
    )
    for tool, kwargs in setup:
        getattr(env.tools, tool)(**kwargs)
    message = env.get_response(ToolCall(id="call_1", name=name, arguments=arguments))
    assert message.error
    assert re.search(error, message.content), message.content


def test_modifier_entry_validation_and_portion_overlap_rejected():
    pizza = store_env(PIZZA)
    with pytest.raises(ValueError, match="unknown key"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Large Classic Pie",
            modifiers=[{"option_id": "Pepperoni (Normal)", "option": "extra"}],
        )
    with pytest.raises(ValueError, match="must be an integer"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Large Classic Pie",
            modifiers=[{"option_id": "Pepperoni (Normal)", "quantity": "two"}],
        )
    with pytest.raises(
        ValueError,
        match="can be chosen at most once for the whole pizza or once per half",
    ):
        pizza.tools.add_to_cart(
            PIZZA,
            "Large Classic Pie",
            modifiers=[
                {"option_id": "Italian Sausage (Normal)"},
                {"option_id": "Italian Sausage - First Half (Extra)"},
            ],
        )

    burger = store_env(BURGER)
    with pytest.raises(ValueError, match="cannot both be selected"):
        burger.tools.add_to_cart(
            BURGER,
            "10 Piece Chicken Nuggets",
            modifiers=[
                {"option_id": "Honey Mustard Dipping Sauce"},
                {"option_id": "No Dipping Sauce"},
            ],
        )


def test_quantity_zero_on_a_recipe_ingredient_selects_its_no_option():
    """Burger toppings are not modeled defaults; quantity 0 still means the 'No ...' option."""
    qty0 = store_env(BURGER)
    qty0.tools.add_to_cart(
        BURGER, "Pier Burger", modifiers=[{"option_id": "Onion", "quantity": 0}]
    )
    no_option = store_env(BURGER)
    no_option.tools.add_to_cart(
        BURGER, "Pier Burger", modifiers=[{"option_id": "No Onion"}]
    )
    assert "No Onion" in line_names(qty0)
    assert qty0.tools.get_db_hash() == no_option.tools.get_db_hash()


def test_quantity_zero_on_a_sole_default_selects_the_groups_no_option():
    def cheeseburger(mods) -> str:
        env = store_env(BURGER)
        env.tools.add_to_cart(BURGER, "Classic Cheeseburger", modifiers=mods)
        return env.tools.get_db_hash()

    assert cheeseburger([{"option_id": "American Cheese", "quantity": 0}]) == (
        cheeseburger([{"option_id": "No Cheese"}])
    )


def test_quantity_zero_uses_the_one_group_offering_the_no_option():
    """Several of the meal's groups offer 'Ketchup'; only one offers 'No Ketchup'."""

    def kids_meal(mods) -> str:
        env = store_env(BURGER)
        env.tools.add_to_cart(
            BURGER,
            "Kids' Cheeseburger Meal",
            modifiers=mods + [{"option_id": "Organic Apple Juice"}],
        )
        return env.tools.get_db_hash()

    assert kids_meal([{"option_id": "Ketchup", "quantity": 0}]) == (
        kids_meal([{"option_id": "No Ketchup"}])
    )


@pytest.mark.parametrize("item", ["Cold Brew", "Iced Latte"])
def test_a_no_option_in_a_pick_one_group_replaces_the_default(item: str):
    no_ice = store_env(CAFE)
    no_ice.tools.add_to_cart(
        CAFE, item, modifiers=[{"option_id": "Medium"}, {"option_id": "No Ice"}]
    )
    assert "Regular Ice" not in line_names(no_ice)
    qty0 = store_env(CAFE)
    qty0.tools.add_to_cart(
        CAFE,
        item,
        modifiers=[
            {"option_id": "Medium"},
            {"option_id": "Regular Ice", "quantity": 0},
        ],
    )
    assert no_ice.tools.get_db_hash() == qty0.tools.get_db_hash()


def test_whole_number_floats_count_as_integers_and_fractions_are_rejected():
    ints = store_env(BURGER)
    ints.tools.add_to_cart(BURGER, "Double Griddle Burger", quantity=2)
    floats = store_env(BURGER)
    floats.tools.add_to_cart(BURGER, "Double Griddle Burger", quantity=1.0)
    floats.tools.update_cart_item("line_1", quantity=2.0)
    assert floats.tools.db.cart.items[0].quantity == 2
    assert floats.tools.get_db_hash() == ints.tools.get_db_hash()
    with pytest.raises(ValueError, match="'quantity' must be a whole number"):
        floats.tools.update_cart_item("line_1", quantity=2.5)

    booking = dict(
        store_id=CAFE,
        date="2026-06-29",
        time="18:00",
        party_size=2,
        customer_name="Maria Lopez",
        customer_phone="9495550118",
    )
    exact, spelled = store_env(CAFE), store_env(CAFE)
    res_id = exact.tools.make_reservation(**booking).reservation_id
    spelled.tools.make_reservation(**booking)
    exact.tools.update_reservation(res_id, party_size=5, high_chairs=1)
    spelled.tools.update_reservation(res_id, party_size=5.0, high_chairs=1.0)
    assert spelled.tools.db.reservations[res_id].party_size == 5
    assert spelled.tools.get_db_hash() == exact.tools.get_db_hash()
    with pytest.raises(ValueError, match="'party_size' must be a whole number"):
        spelled.tools.update_reservation(res_id, party_size=3.5)


@pytest.mark.parametrize(
    "field, value", [("placement", "first half"), ("volume", "lite")]
)
def test_an_unknown_placement_or_volume_is_an_error(field: str, value: str):
    env = store_env(PIZZA)
    with pytest.raises(ValueError, match=f"Invalid {field} '{value}'"):
        env.tools.add_to_cart(
            PIZZA,
            "Medium Classic Pie",
            modifiers=[{"option_id": "Pepperoni", field: value}],
        )


def test_placed_order_keeps_the_cart_lines_as_built():
    env = store_env(PIZZA)
    cart = env.tools.add_to_cart(
        PIZZA,
        "Medium Classic Pie",
        modifiers=[{"option_id": "Pepperoni", "placement": "right"}],
    )
    built = [li.model_copy(deep=True) for li in cart.items]
    order = env.tools.place_order()
    assert order.items == built
    assert any("Second Half" in m.name for m in order.items[0].modifiers)


def test_set_fulfillment_ignores_a_store_id():
    env = store_env(CAFE, "mobile")
    cart = env.tools.set_fulfillment("pickup", "13:00", store_id=CAFE)
    assert (cart.order_type, cart.scheduled_time) == ("pickup", "13:00")


def test_negative_and_half_portion_quantity_zero_modifiers_are_rejected():
    pizza = store_env(PIZZA)
    with pytest.raises(ValueError, match="Modifier quantity must be at least 0"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Medium Classic Pie",
            modifiers=[{"option_id": "Pepperoni", "quantity": -1}],
        )
    with pytest.raises(ValueError, match="not a removable default"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Medium Classic Pie",
            modifiers=[{"option_id": "Pepperoni - First Half (Normal)", "quantity": 0}],
        )
    with pytest.raises(ValueError, match="Cannot remove .* from one half"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Meatpacking - Large Classic",
            modifiers=[{"option_id": "Ham - First Half (Normal)", "quantity": 0}],
        )
    with pytest.raises(ValueError, match="Cannot remove .* from one half"):
        pizza.tools.add_to_cart(
            PIZZA,
            "Meatpacking - Large Classic",
            modifiers=[{"option_id": "Ham", "placement": "left", "quantity": 0}],
        )
    for half_ref in ("pz-o5407", "ham-first-half-normal"):
        with pytest.raises(ValueError, match="Cannot remove .* from one half"):
            pizza.tools.add_to_cart(
                PIZZA,
                "Meatpacking - Large Classic",
                modifiers=[{"option_id": half_ref, "quantity": 0}],
            )
    with pytest.raises(ValueError, match="Cannot remove .* from one half") as exc_info:
        pizza.tools.add_to_cart(
            PIZZA,
            "Meatpacking - Large Classic",
            modifiers=[{"option_id": "No ham-first-half-normal"}],
        )
    assert "quantity 0" not in str(exc_info.value)

    cafe = store_env(CAFE)
    with pytest.raises(ValueError, match="Modifier quantity must be at least 0"):
        cafe.tools.add_to_cart(
            CAFE,
            "Latte",
            modifiers=[
                {"option_id": "Medium"},
                {"option_id": "Oat Milk", "quantity": -1},
            ],
        )


def test_category_no_modifier_contradictions_and_non_sauce_coexistence():
    pizza = store_env(PIZZA)
    cart = pizza.tools.add_to_cart(
        PIZZA,
        "Half-and-Half Pizza",
        modifiers=[
            {"option_id": "Medium"},
            {"option_id": "City Thin"},
            {"group_id": "Left Half Base (1)", "option_id": "Pepperoni"},
            {"group_id": "Right Half Base (1)", "option_id": "Cheese"},
            {"group_id": "Left Half Edits (1)", "option_id": "No Sauce"},
            {"group_id": "Left Half Edits (1)", "option_id": "Add Mushrooms"},
        ],
    )
    left_edits = {
        m.name
        for m in cart.items[0].modifiers
        if m.name in ("No Sauce", "Add Mushrooms")
    }
    assert left_edits == {"No Sauce", "Add Mushrooms"}

    cafe = store_env(CAFE)
    with pytest.raises(ValueError, match="cannot both be selected"):
        cafe.tools.add_to_cart(
            CAFE,
            "Cold Brew",
            modifiers=[
                {"option_id": "Medium"},
                {"option_id": "No Syrup"},
                {"option_id": "Vanilla Syrup"},
            ],
        )
    with pytest.raises(ValueError, match="cannot both be selected"):
        cafe.tools.add_to_cart(
            CAFE,
            "Cold Brew",
            modifiers=[
                {"option_id": "Medium"},
                {"option_id": "No Syrup"},
                {"option_id": "Mocha Sauce"},
            ],
        )
    with pytest.raises(ValueError, match="cannot both be selected"):
        cafe.tools.add_to_cart(
            CAFE,
            "Caesar Salad",
            modifiers=[
                {"option_id": "No Dressing"},
                {"option_id": "Balsamic Vinaigrette"},
            ],
        )

    burger = store_env(BURGER)
    with pytest.raises(ValueError, match="cannot both be selected"):
        burger.tools.add_to_cart(
            BURGER,
            "Sunset Burger Combo",
            modifiers=[
                {"option_id": "Cold Brew"},
                {"option_id": "No Syrup"},
                {"option_id": "Sugar-Free Vanilla"},
            ],
        )


def test_multi_default_group_preserves_siblings_and_roundtrips_cart_echo():
    env = store_env(CAFE, now="2026-06-22 09:00")
    cart = env.tools.add_to_cart(
        CAFE,
        "Steel-Cut Oatmeal",
        modifiers=[
            {"option_id": "No Toasted Walnuts"},
            {"option_id": "Blueberries"},
        ],
    )
    names = [m.name for m in cart.items[0].modifiers]
    assert "Brown Sugar" in names
    assert "No Toasted Walnuts" in names
    assert "Blueberries" in names
    assert "Toasted Walnuts" not in names
    assert names.count("Brown Sugar") == 1
    before_hash = env.tools.get_db_hash()

    display = _labels(env)
    echoed = [
        {
            "group_id": display[m.group_id],
            "option_id": m.name,
            "quantity": m.quantity,
        }
        for m in cart.items[0].modifiers
    ]
    updated = env.tools.update_cart_item("line_1", modifiers=echoed)
    updated_names = [m.name for m in updated.items[0].modifiers]
    assert "Toasted Walnuts" not in updated_names
    assert updated_names.count("Brown Sugar") == 1
    assert env.tools.get_db_hash() == before_hash

    env_qty0 = store_env(CAFE, now="2026-06-22 09:00")
    cart_qty0 = env_qty0.tools.add_to_cart(
        CAFE,
        "Steel-Cut Oatmeal",
        modifiers=[
            {"option_id": "Toasted Walnuts", "quantity": 0},
            {"option_id": "Blueberries"},
        ],
    )
    qty0_names = [m.name for m in cart_qty0.items[0].modifiers]
    assert "Brown Sugar" in qty0_names
    assert "Blueberries" in qty0_names
    assert "Toasted Walnuts" not in qty0_names
    assert env_qty0.tools.get_db_hash() == before_hash


def test_item_specific_defaults_in_get_modifier_group_and_starting_prices():
    cafe = store_env(CAFE, now="2026-06-22 09:00")
    assert cafe.tools.get_modifier_group(CAFE, "Size (1)")["default"] == ["Medium"]
    assert cafe.tools.get_modifier_group(CAFE, "Size (1)", item_id="Flat White")[
        "default"
    ] == ["Small"]
    with pytest.raises(ValueError, match="does not apply to 'Steel-Cut Oatmeal'"):
        cafe.tools.get_modifier_group(CAFE, "Size (1)", item_id="Steel-Cut Oatmeal")

    pastry = cafe.tools.get_item_details(CAFE, "Morning Pastry Combo")
    assert pastry["base_price"] == "$6.50"
    assert pastry["starting_price"] == "$7.00"
    pastry_search = next(
        i
        for i in cafe.tools.search_menu(CAFE, query="Morning Pastry Combo")["items"]
        if i["name"] == "Morning Pastry Combo"
    )
    assert pastry_search["starting_price"] == "$7.00"
    cafe_policy = get_environment(task=task_for(CAFE)).get_policy()
    # Morning Pastry Combo is breakfast-only at 11:00 default task time, so check a 09:00 task/render.
    morning_task = task_for(CAFE).model_copy(deep=True)
    morning_task.initial_state.initialization_data.agent_data["cart"][
        "current_time"
    ] = "2026-06-22 09:00"
    morning_menu = get_environment(task=morning_task).get_policy()
    assert "**Morning Pastry Combo** (from $7.00" in morning_menu
    assert cafe_policy

    burger = store_env(BURGER)
    pier = burger.tools.get_item_details(BURGER, "Pier Burger Combo")
    assert pier["base_price"] == "$0.00"
    assert pier["starting_price"] == "$9.87"
    pier_search = next(
        i
        for i in burger.tools.search_menu(BURGER, query="Pier Burger Combo")["items"]
        if i["name"] == "Pier Burger Combo"
    )
    assert pier_search["starting_price"] == "$9.87"
    assert burger.tools.get_modifier_group(
        BURGER, "Cheese Choice (1)", item_id="Sunset Burger Combo"
    )["default"] == ["Cheddar Cheese"]


def test_address_normalization_handles_hash_units_routes_and_ordinal_streets():
    from tau2.domains.food.normalize import _normalize_address

    for variant in (
        "420 Maple St #3B",
        "420 Maple St Apt #3B",
        "420 Maple St Unit #3B",
        "Four Twenty Maple Street Apartment 3B",
    ):
        assert _normalize_address(variant) == "420 Maple St Apt 3B"
    assert _normalize_address("100 Main St Suite #200") == "100 Main St Ste 200"
    assert _normalize_address("50 Oak Ave Bldg #2") == "50 Oak Ave Bldg 2"
    assert _normalize_address("9 Highway #1") == "9 Hwy 1"
    assert (
        _normalize_address("One Hundred and Twenty First Street")
        == "100 And Twenty First St"
    )
    assert _normalize_address("Two Hundred and Ten Street") == "200 And Ten St"
