"""Tests for food domain registration, environment construction, and the agent prompt."""

import re

import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.domains.food.data_model import (
    MenuItem,
    MenuSchedule,
    MenuScheduleRule,
    MenuTimeRange,
    ModifierGroup,
    ModifierOption,
    Store,
)
from tau2.domains.food.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.food.menu_prompt import render_menu
from tau2.environment.toolkit import ToolType
from tau2.multilingual.runtime import task_context
from tau2.registry import registry
from tau2.runner.build import _build_env_kwargs, build_environment

from .conftest import BURGER, CAFE, STORES, store_of, task_for

# Tool name -> (ToolType, mutates state). Lookups are read-only; the transfer is generic but
# changes the graded state.
TOOL_TYPES = {
    "get_store_info": (ToolType.READ, False),
    "list_menu_categories": (ToolType.READ, False),
    "search_menu": (ToolType.READ, False),
    "get_item_details": (ToolType.READ, False),
    "get_modifier_group": (ToolType.READ, False),
    "get_cart": (ToolType.READ, False),
    "add_to_cart": (ToolType.WRITE, True),
    "update_cart_item": (ToolType.WRITE, True),
    "remove_from_cart": (ToolType.WRITE, True),
    "clear_cart": (ToolType.WRITE, True),
    "set_fulfillment": (ToolType.WRITE, True),
    "place_order": (ToolType.WRITE, True),
    "check_table_availability": (ToolType.READ, False),
    "make_reservation": (ToolType.WRITE, True),
    "find_reservations": (ToolType.READ, False),
    "update_reservation": (ToolType.WRITE, True),
    "cancel_reservation": (ToolType.WRITE, True),
    "transfer_to_human_agents": (ToolType.GENERIC, True),
}


def _mobile_task(store_id: str):
    task = task_for(store_id).model_copy(deep=True)
    task.initial_state.initialization_data.agent_data["cart"]["channel"] = "mobile"
    return task


def test_food_domain_is_registered():
    assert "food" in registry.get_domains()
    assert "food" in registry.get_task_sets()
    assert registry.get_task_splits_loader("food")() == get_tasks_split()
    assert registry.get_env_constructor("food") is get_environment


def test_runner_passes_the_task():
    task = task_for("cafe")
    config = TextRunConfig(domain="food")
    kwargs = _build_env_kwargs(
        config, task, language_context=task_context(config, task)
    )
    assert kwargs == {"task": task}
    env = build_environment("food", env_kwargs=kwargs)
    assert "- store_id for tool calls: cafe" in env.get_policy()


def test_environment_has_all_tools_with_their_types():
    env = get_environment(task=task_for("burger"))
    assert env.get_domain_name() == "food"
    assert {t.name for t in env.get_tools()} == set(TOOL_TYPES)
    for name, (tool_type, mutates) in TOOL_TYPES.items():
        assert env.tools.tool_type(name) == tool_type, name
        assert env.tools.tool_mutates_state(name) is mutates, name


def test_get_environment_rejects_solo_mode_and_tasks_without_a_store():
    with pytest.raises(ValueError, match="solo mode"):
        get_environment(solo_mode=True)
    task = task_for(CAFE).model_copy(deep=True)
    task.initial_state.initialization_data.agent_data["cart"]["store_id"] = "taco"
    with pytest.raises(ValueError, match="no valid cart.store_id"):
        get_environment(task=task)


def test_every_task_builds_an_environment_with_its_store_menu():
    for task in get_tasks("base"):
        policy = get_environment(task=task).get_policy()
        assert f"- store_id for tool calls: {store_of(task)}" in policy
        assert "## Menu Data" in policy


def test_policy_names_only_real_agent_tools():
    env = get_environment()
    policy = env.get_policy()
    named = set(re.findall(r"`([a-z]+(?:_[a-z]+)+)`", policy))
    assert named <= {t.name for t in env.get_tools()}
    assert "place_order" in named
    assert "<!--" not in policy
    assert "Benchmark" not in policy


def test_drive_thru_store_block():
    policy = get_environment(task=task_for(CAFE)).get_policy()
    block = policy.split("## Current Store", 1)[1].split("## Menu Data", 1)[0]
    assert "- Order channel: drive-thru" in block
    assert "- Current time: Monday, 2026-06-22 11:00" in block
    assert "- Store hours today: 07:00-21:00" in block
    assert "- Fulfillment method: pickup" in block
    assert "- Table reservations: accepted (party of 1 to 8" in block
    assert "lead time" not in block


def test_mobile_store_block():
    block = (
        get_environment(task=_mobile_task(BURGER))
        .get_policy()
        .split("## Current Store", 1)[1]
        .split("## Menu Data", 1)[0]
    )
    assert "- Order channel: mobile" in block
    assert "- Fulfillment so far: pickup, ASAP (change with set_fulfillment)" in block
    assert "- Pickup lead time: 20 minutes" in block
    assert (
        "- Delivery: $3.99 fee, $10.00 minimum subtotal, 45-minute lead time" in block
    )
    assert "- Table reservations: not accepted" in block


@pytest.mark.parametrize("store_id", STORES)
def test_policy_includes_store_block_and_concise_menu_data(store_id: str):
    task = task_for(store_id)
    policy = get_environment(task=task).get_policy()

    store_block = policy.split("## Current Store", 1)[1].split("## Menu Data", 1)[0]
    menu_block = policy.split("## Menu Data", 1)[1]
    assert f"- store_id for tool calls: {store_id}" in store_block
    assert "### Menu Items" in menu_block
    assert "### Modifier Groups" in menu_block


def test_task_policy_embeds_only_its_store_menu():
    task = next(
        t
        for t in get_tasks("base")
        if t.id == "pizza_xlarge_crust_constraint_and_half_sauce"
    )
    policy = get_environment(task=task).get_policy()
    menu = policy.rsplit("## Menu Data", 1)[1]

    assert "- store_id for tool calls: pizza" in policy
    assert "**Meatpacking - Extra Large Sicilian**" in menu
    assert "Sunset Burger" not in policy.split("## Menu Data")[0]


def test_prompt_menu_shows_schedules_and_delivery_limits():
    menu = (
        get_environment(task=task_for(CAFE)).get_policy().rsplit("## Menu Data", 1)[1]
    )
    trio = menu.split("- **Half Sandwich Lunch**", 1)[1].split("\n", 1)[0]
    assert "schedule: daily 11:00-15:00" in trio
    latte = menu.split("- **Latte**", 1)[1].split("\n", 1)[0]
    assert "not available for: delivery" in latte


def test_menu_referenced_groups_resolve_to_defined_modifier_groups():
    """Every modifier group referenced in rendered Menu Data is defined under ### Modifier Groups."""
    for store_id in STORES:
        menu = (
            get_environment(task=task_for(store_id))
            .get_policy()
            .rsplit("## Menu Data", 1)[1]
        )
        groups_sec = menu.split("### Modifier Groups\n", 1)[1].split(
            "\n\n### Unavailable Items", 1
        )[0]
        group_names = set(re.findall(r"^- \*\*(.+?)\*\*", groups_sec, flags=re.M))
        assert group_names
        for refs in re.findall(r"modifier groups: (.+)$", menu, flags=re.M):
            for name in refs.split(", "):
                assert name in group_names, (store_id, name)


@pytest.mark.parametrize("store_id", STORES)
def test_menu_data_shows_no_internal_ids(store_id: str):
    store = get_environment().tools.db.stores[store_id]
    ids = set(store.items) | set(store.modifier_groups)
    ids |= {oid for g in store.modifier_groups.values() for oid in g.options}
    policy = get_environment(task=task_for(store_id)).get_policy()
    assert not [i for i in ids if i in policy]


def test_inquiry_tasks_require_get_item_details_for_omitted_descriptions():
    store_tasks = {store: get_tasks(store)[0] for store in STORES}
    checks = [
        ("cafe", "Sourdough Grilled Cheese", "smoked Gouda"),
        ("pizza", "Baked Ziti", "ricotta, mozzarella, and Parmesan"),
        ("pizza", "Italian Hero", "lettuce, tomatoes, onions, and oil and vinegar"),
        ("burger", "Southwest Salad", "tortilla strips"),
        ("burger", "Crispy Fish Sandwich", "tartar sauce"),
    ]
    for store_id, item_name, desc_token in checks:
        env = get_environment(task=store_tasks[store_id])
        menu_block = env.get_policy().split("## Menu Data", 1)[1]
        details = env.tools.get_item_details(store_id, item_name)
        assert details["description"] and details["description"] not in menu_block, (
            store_id,
            item_name,
        )
        assert desc_token.lower() in details["description"].lower(), (
            store_id,
            item_name,
        )


def _tiny_store() -> Store:
    size = ModifierGroup(
        group_id="g1",
        name="Size",
        min_select=1,
        options={
            "o1": ModifierOption(option_id="o1", name="Small"),
            "o2": ModifierOption(option_id="o2", name="Large", child_group_ids=["g2"]),
            "o3": ModifierOption(option_id="o3", name="Huge", available=False),
        },
    )
    toppings = ModifierGroup(
        group_id="g2",
        name="Size",
        max_select=2,
        supports_half_and_half=True,
        options={"o4": ModifierOption(option_id="o4", name="Cheese")},
    )
    lunch = MenuSchedule(
        schedule_id="s1",
        rules=[
            MenuScheduleRule(
                days_of_week=[
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ],
                time_ranges=[MenuTimeRange(start_time="11:00", end_time="15:00")],
            )
        ],
    )

    def item(item_id: str, name: str, category: str, cents: int, **kw) -> MenuItem:
        return MenuItem(
            item_id=item_id,
            name=name,
            category=category,
            description="",
            base_price_cents=cents,
            **kw,
        )

    return Store(
        store_id="tiny",
        name="Tiny Pies",
        concept="pizza",
        cuisine="Pizza",
        categories=["Drinks", "Pizzas"],
        items={
            "i1": item("i1", "Plain Pie", "Pizzas", 1250, modifier_group_ids=["g1"]),
            "i2": item(
                "i2", "Soda", "Drinks", 199, allowed_fulfillment_methods=["pickup"]
            ),
            "i3": item("i3", "Lunch Slice", "Specials", 350, schedule_ids=["s1"]),
            "i4": item("i4", "Gone Pie", "Pizzas", 900),
        },
        modifier_groups={"g1": size, "g2": toppings},
        schedules={"s1": lunch},
    )


def test_render_menu_lists_items_groups_and_unavailable_items():
    store = _tiny_store()
    menu = render_menu(
        store, lambda _, i: (i.item_id != "i4", "sold out" if i.item_id == "i4" else "")
    )
    assert menu == (
        "### Menu Items\n\n"
        "#### Drinks\n"
        "- **Soda** ($1.99; not available for: delivery)\n\n"
        "#### Pizzas\n"
        "- **Plain Pie** ($12.50) — modifier groups: Size (1)\n\n"
        "#### Specials\n"
        "- **Lunch Slice** ($3.50; schedule: daily 11:00-15:00)\n\n"
        "### Modifier Groups\n"
        "- **Size (1)** (min: 1, max: 1): Small, Large (→ Size (2))\n"
        "- **Size (2)** (max: 2, supports half-and-half): Cheese\n\n"
        "### Unavailable Items\n"
        "- **Gone Pie**: sold out\n"
    )
    everything = render_menu(store, lambda *_: (True, ""))
    assert everything.endswith("### Unavailable Items\nNone\n")
