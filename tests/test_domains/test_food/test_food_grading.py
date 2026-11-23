"""Tests for food grading: golden replays, get_db_hash equivalences, and session isolation."""

import json
import signal

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.domains.food.environment import (
    get_environment,
    get_tasks,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from .conftest import BURGER, CAFE, PIZZA, line_names, store_env, task_for


def _cart_hash(task, actions) -> str:
    env = get_environment()
    env.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
        strict=True,
    )
    for action in actions:
        getattr(env.tools, action["name"])(**action["arguments"])
    return env.tools.get_db_hash()


def _gold(task):
    return [
        {"name": a.name, "arguments": a.arguments}
        for a in task.evaluation_criteria.actions or []
    ]


class _Timeout:
    """Fails the test instead of hanging when a tool call does not return."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if not hasattr(signal, "SIGALRM"):  # POSIX only; elsewhere run unguarded.
            return

        def _raise(*_):
            raise TimeoutError(f"tool call took more than {self.seconds}s")

        signal.signal(signal.SIGALRM, _raise)
        signal.alarm(self.seconds)

    def __exit__(self, *_):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda t: t.id)
def test_golden_actions_reach_full_reward(task):
    """The gold actions run without errors, change the DB, and score 1.0 on strict replay."""
    env = get_environment(task=task)
    env.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=task.initial_state.message_history or [],
        strict=True,
    )
    initial_hash = env.tools.get_db_hash()
    trajectory = []
    for idx, action in enumerate(task.evaluation_criteria.actions or [], start=1):
        call = ToolCall(
            id=f"tc_{idx}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        response = env.get_response(call)
        assert not response.error, (action.name, response.content)
        trajectory += [
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
            response,
        ]
    assert env.tools.get_db_hash() != initial_hash

    reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
        env_kwargs={"task": task},
        strict_replay=True,
    )
    assert reward.db_check.db_match is True
    assert reward.reward == 1.0


def _order_hash(*steps) -> str:
    env = store_env(BURGER)
    for step in steps:
        if step == "place":
            env.tools.place_order(order_type="pickup")
        else:
            env.tools.add_to_cart(store_id=BURGER, item_id=step)
    return env.tools.get_db_hash()


def test_order_hash_covers_every_order_and_the_cart():
    gold = _order_hash("Double Griddle Burger", "place")
    assert _order_hash("Double Griddle Burger") != gold  # place_order never called
    assert (
        _order_hash("Double Griddle Burger", "place", "Double Griddle Burger", "place")
        != gold
    )
    assert (
        _order_hash(
            "Double Griddle Burger",
            "place",
            "Regular Chocolate Shake",
        )
        != gold
    )  # extra item left in the cart
    # Equivalent ways of building the same single order still match.
    assert _order_hash("Double Griddle Burger", "place") == gold


def test_unwarranted_transfer_changes_the_db_hash():
    """An unwarranted transfer_to_human_agents call after a completed order alters get_db_hash()."""
    task = next(
        t
        for t in get_tasks("base")
        if t.id == "burger_medium_size_vs_doneness_overload"
    )
    gold_actions = _gold(task)
    gold_hash = _cart_hash(task, gold_actions)
    false_transfer_hash = _cart_hash(
        task,
        gold_actions
        + [
            {
                "name": "transfer_to_human_agents",
                "arguments": {"summary": "Customer asked about street EV chargers."},
            }
        ],
    )
    assert false_transfer_hash != gold_hash


def test_transfer_hash_accepts_unplaced_cart_or_clear_cart():
    """On cafe_transfer_past_order_double_charge, leaving the unplaced cart or clearing it before transfer hashes equal."""
    task = next(
        t for t in get_tasks("base") if t.id == "cafe_transfer_past_order_double_charge"
    )
    gold_actions = _gold(task)
    gold_hash = _cart_hash(task, gold_actions)
    with_clear = (
        gold_actions[:-1]
        + [{"name": "clear_cart", "arguments": {}}]
        + [gold_actions[-1]]
    )
    assert _cart_hash(task, with_clear) == gold_hash


def test_lookups_do_not_change_the_db_hash():
    env = store_env("pizza")
    before = env.tools.get_db_hash()
    env.tools.list_menu_categories("pizza")
    env.tools.search_menu("pizza", query="pepperoni")
    env.tools.get_item_details("pizza", "Lunch Rush Deal")
    assert env.tools.get_db_hash() == before


def _add(item: str, *mods: dict, quantity: int = 1):
    def step(tools) -> None:
        tools.add_to_cart(
            tools.db.cart.store_id, item, quantity=quantity, modifiers=list(mods)
        )

    return step


def _place(tools) -> None:
    tools.place_order(order_type="pickup")


def _db_hash(store_id: str, steps, now: str | None = None) -> str:
    env = store_env(store_id, now=now) if now else store_env(store_id)
    for step in steps:
        step(env.tools)
    return env.tools.get_db_hash()


_BURGER = "Double Griddle Burger"
_SHAKE = "Regular Chocolate Shake"
_LARGE = {"option_id": "Large"}


@pytest.mark.parametrize(
    "store_id, first, second",
    [
        pytest.param(
            BURGER,
            [_add(_BURGER), _add(_BURGER)],
            [_add(_BURGER, quantity=2)],
            id="two-lines-equal-quantity-two",
        ),
        pytest.param(
            BURGER,
            [_add(_BURGER), _add(_BURGER), _place],
            [_add(_BURGER, quantity=2), _place],
            id="placed-two-lines-equal-quantity-two",
        ),
        pytest.param(
            BURGER,
            [_add(_BURGER), _add(_SHAKE)],
            [_add(_SHAKE), _add(_BURGER)],
            id="line-order",
        ),
        pytest.param(
            BURGER,
            [_add("Chicken Caesar Salad", {"option_id": "No Dressing"})],
            [_add("Chicken Caesar Salad", {"option_id": "No Caesar Dressing"})],
            id="no-dressing-is-the-salads-own-dressing",
        ),
        pytest.param(
            PIZZA,
            [
                _add(
                    "Mushroom Pizza", _LARGE, {"option_id": "Feta", "placement": "left"}
                )
            ],
            [
                _add(
                    "Mushroom Pizza",
                    _LARGE,
                    {"option_id": "Feta", "placement": "right"},
                )
            ],
            id="a-topping-on-either-half",
        ),
        pytest.param(
            BURGER,
            [_add("Pier Burger")],
            [
                _add(
                    "Pier Burger",
                    {"option_id": "Onion"},
                    {"option_id": "Pickles"},
                )
            ],
            id="burger-same-group-standard-toppings",
        ),
        pytest.param(
            PIZZA,
            [_add("Medium Classic Pie", {"option_id": "Pepperoni (Normal)"})],
            [
                _add(
                    "Medium Classic Pie",
                    {"option_id": "Pepperoni - First Half (Normal)"},
                    {"option_id": "Pepperoni - Second Half (Normal)"},
                )
            ],
            id="both-halves-same-amount-collapse-to-whole",
        ),
        pytest.param(
            PIZZA,
            [_add("Medium Classic Pie", {"option_id": "Pepperoni (Extra)"})],
            [
                _add(
                    "Medium Classic Pie",
                    {"option_id": "Pepperoni - First Half (Extra)"},
                    {"option_id": "Pepperoni - Second Half (Extra)"},
                )
            ],
            id="both-halves-extra-collapse-to-whole-extra",
        ),
    ],
)
def test_db_hash_treats_equivalent_orders_as_equal(store_id, first, second):
    assert _db_hash(store_id, first) == _db_hash(store_id, second)


def test_multi_default_no_modifier_equals_explicit_remaining_default():
    morning = "2026-06-22 09:00"
    assert _db_hash(
        CAFE,
        [_add("Steel-Cut Oatmeal", {"option_id": "No Toasted Walnuts"})],
        now=morning,
    ) == _db_hash(
        CAFE,
        [_add("Steel-Cut Oatmeal", {"option_id": "Brown Sugar"})],
        now=morning,
    )
    assert _db_hash(
        CAFE,
        [
            _add(
                "Steel-Cut Oatmeal",
                {"option_id": "No Toasted Walnuts"},
                {"option_id": "Blueberries"},
            )
        ],
        now=morning,
    ) == _db_hash(
        CAFE,
        [
            _add(
                "Steel-Cut Oatmeal",
                {"option_id": "Brown Sugar"},
                {"option_id": "Blueberries"},
            )
        ],
        now=morning,
    )


@pytest.mark.parametrize(
    "store_id, first, second",
    [
        pytest.param(
            BURGER, [_add(_BURGER)], [_add(_BURGER, quantity=2)], id="quantity"
        ),
        pytest.param(
            CAFE,
            [_add("Latte", {"option_id": "Medium"})],
            [_add("Latte", {"option_id": "Medium"}, {"option_id": "Oat Milk"})],
            id="paid-add-on",
        ),
        pytest.param(
            CAFE,
            [
                _add(
                    "Iced Hazelnut Mocha",
                    {"option_id": "Medium"},
                    {"option_id": "Extra Cocoa Nibs"},
                )
            ],
            [
                _add(
                    "Iced Hazelnut Mocha",
                    {"option_id": "Medium"},
                    {"option_id": "Extra Cocoa Nibs"},
                    {"option_id": "Cocoa Nibs"},
                )
            ],
            id="extra-replaces-default-unless-both-selected",
        ),
        pytest.param(
            BURGER,
            [_add("Chicken Caesar Salad")],
            [_add("Chicken Caesar Salad", {"option_id": "No Dressing"})],
            id="removal",
        ),
        pytest.param(
            BURGER,
            [_add("Veggie Black Bean Burger")],
            [_add("Veggie Black Bean Burger", {"option_id": "Mayo"})],
            id="burger-non-default-sauce",
        ),
        pytest.param(
            CAFE,
            [_add("Caprese Panini")],
            [_add("Caprese Panini", {"option_id": "Red Onion"})],
            id="cafe-sandwich-non-default-topping",
        ),
        pytest.param(
            CAFE,
            [_add("Iced Hazelnut Mocha", {"option_id": "Medium"})],
            [
                _add(
                    "Iced Hazelnut Mocha",
                    {"option_id": "Medium"},
                    {"option_id": "Whipped Cream"},
                )
            ],
            id="cafe-non-default-whipped-cream-on-mocha-variant",
        ),
        pytest.param(
            PIZZA,
            [_add("Mushroom Pizza", _LARGE, {"option_id": "Feta"})],
            [_add("Mushroom Pizza", _LARGE, {"option_id": "Feta", "volume": "extra"})],
            id="volume",
        ),
        pytest.param(
            PIZZA,
            [_add("Mushroom Pizza", _LARGE, {"option_id": "Feta"})],
            [
                _add(
                    "Mushroom Pizza", _LARGE, {"option_id": "Feta", "placement": "left"}
                )
            ],
            id="whole-vs-half",
        ),
        pytest.param(
            PIZZA,
            [
                _add(
                    "Mushroom Pizza",
                    _LARGE,
                    {"option_id": "Pepperoni", "placement": "left"},
                    {"option_id": "Feta", "placement": "left"},
                )
            ],
            [
                _add(
                    "Mushroom Pizza",
                    _LARGE,
                    {"option_id": "Pepperoni", "placement": "left"},
                    {"option_id": "Feta", "placement": "right"},
                )
            ],
            id="same-half-vs-opposite-halves",
        ),
        pytest.param(
            BURGER,
            [_add("Crispy Chicken Sandwich")],
            [_add("Crispy Chicken Sandwich", {"option_id": "No Pickles"})],
            id="real-pickle-removal-on-chicken-sandwich",
        ),
        pytest.param(
            BURGER,
            [_add("Classic Cheeseburger")],
            [_add("Classic Cheeseburger", {"option_id": "No Cheese"})],
            id="real-cheese-removal-on-cheeseburger",
        ),
        pytest.param(
            CAFE,
            [_add("Cold Brew", {"option_id": "Medium"})],
            [_add("Cold Brew", {"option_id": "Medium"}, {"option_id": "No Ice"})],
            id="real-ice-removal-on-cold-brew",
        ),
        pytest.param(
            CAFE,
            [_add("Cinnamon Oat Cold Brew", {"option_id": "Medium"})],
            [
                _add(
                    "Cinnamon Oat Cold Brew",
                    {"option_id": "Medium"},
                    {"option_id": "No Ice"},
                )
            ],
            id="real-ice-removal-on-iced-cold-brew-without-modeled-default",
        ),
        pytest.param(
            CAFE,
            [_add("Roast Beef Panini")],
            [_add("Roast Beef Panini", {"option_id": "No Mayo"})],
            id="real-mayo-removal-on-horseradish-mayo-panini",
        ),
        pytest.param(
            BURGER,
            [_add("Double Griddle Burger")],
            [_add("Double Griddle Burger", {"option_id": "No Red Onion"})],
            id="real-onion-removal-on-diced-onion-burger",
        ),
        pytest.param(
            BURGER,
            [_add("Sunset Burger")],
            [_add("Sunset Burger", {"option_id": "No Red Onion"})],
            id="real-onion-removal-on-caramelized-onion-burger",
        ),
        pytest.param(
            PIZZA,
            [_add("Medium Classic Pie", {"option_id": "Pepperoni (Normal)"})],
            [
                _add(
                    "Medium Classic Pie",
                    {"option_id": "Pepperoni - First Half (Normal)"},
                    {"option_id": "Pepperoni - Second Half (Extra)"},
                )
            ],
            id="mismatched-half-amounts-do-not-collapse-to-whole",
        ),
    ],
)
def test_db_hash_tells_different_orders_apart(store_id, first, second):
    assert _db_hash(store_id, first) != _db_hash(store_id, second)


def test_portion_default_removal_and_half_mirror_hash_invariance():
    # Order 1: Pepperoni on First Half, Italian Sausage on Second Half, remove default Tomato Sauce via "No ..."
    env1 = get_environment()
    env1.tools.add_to_cart(
        store_id="pizza",
        item_id="Large Classic Pie",
        quantity=1,
        modifiers=[
            {"option_id": "No Tomato Sauce"},
            {"option_id": "Pesto (Normal)"},
            {"option_id": "Pepperoni - First Half (Normal)"},
            {"option_id": "Italian Sausage - Second Half (Extra)"},
        ],
    )
    env1.tools.place_order(order_type="pickup")

    # Order 2: Mirror halves (Pepperoni on Second Half, Italian Sausage on First Half using placement/volume args)
    # and remove default Tomato Sauce via quantity=0
    env2 = get_environment()
    env2.tools.add_to_cart(
        store_id="pizza",
        item_id="Large Classic Pie",
        quantity=1,
        modifiers=[
            {
                "option_id": "Tomato Sauce (Normal)",
                "quantity": 0,
            },
            {"option_id": "Pesto"},
            {"option_id": "Pepperoni", "placement": "right", "volume": "normal"},
            {"option_id": "Italian Sausage", "placement": "left", "volume": "extra"},
        ],
    )
    env2.tools.place_order(order_type="pickup")

    assert env1.get_db_hash() == env2.get_db_hash()


def test_classic_and_half_and_half_pizza_left_right_mirror_symmetry():
    env1 = store_env(PIZZA)
    env1.tools.add_to_cart(
        PIZZA,
        "Mushroom Pizza",
        modifiers=[
            {"option_id": "Large"},
            {"option_id": "Pepperoni", "placement": "left"},
            {"option_id": "Feta", "placement": "right"},
        ],
    )
    env2 = store_env(PIZZA)
    env2.tools.add_to_cart(
        PIZZA,
        "Mushroom Pizza",
        modifiers=[
            {"option_id": "Large"},
            {"option_id": "Feta", "placement": "left"},
            {"option_id": "Pepperoni", "placement": "right"},
        ],
    )
    assert env1.tools.get_db_hash() == env2.tools.get_db_hash()

    env3 = store_env(PIZZA)
    env3.tools.add_to_cart(
        PIZZA,
        "Half-and-Half Pizza",
        modifiers=[
            {"option_id": "Large"},
            {"group_id": "Left Half Base (1)", "option_id": "Pepperoni"},
            {"group_id": "Right Half Base (1)", "option_id": "Supreme"},
            {"option_id": "No Cheese", "group_id": "Left Half Edits (1)"},
        ],
    )
    env4 = store_env(PIZZA)
    env4.tools.add_to_cart(
        PIZZA,
        "Half-and-Half Pizza",
        modifiers=[
            {"option_id": "Large"},
            {"group_id": "Left Half Base (1)", "option_id": "Supreme"},
            {"group_id": "Right Half Base (1)", "option_id": "Pepperoni"},
            {"option_id": "No Cheese", "group_id": "Right Half Edits (1)"},
        ],
    )
    assert env3.tools.get_db_hash() == env4.tools.get_db_hash()


def test_conflicting_deal_defaults_terminate_and_ignore_input_order():
    pizzas = ["Medium City Thin Pie", "Midtown - Medium Classic"]
    hashes = []
    for order in (pizzas, pizzas[::-1]):
        env = store_env(PIZZA)
        with _Timeout(20):
            env.tools.add_to_cart(
                store_id=PIZZA,
                item_id="Block Party Deal",
                modifiers=[{"option_id": name} for name in order],
            )
        assert set(pizzas) <= line_names(env)
        hashes.append(env.tools.get_db_hash())
    assert hashes[0] == hashes[1]


def test_removing_a_default_by_quantity_zero_equals_its_no_option():
    def latte(mods) -> str:
        env = store_env(CAFE)
        env.tools.add_to_cart(
            store_id=CAFE,
            item_id="Latte",
            modifiers=[{"option_id": "Medium"}] + mods,
        )
        assert "Foam" not in line_names(env)
        return env.tools.get_db_hash()

    assert latte([{"option_id": "Foam", "quantity": 0}]) == latte(
        [{"option_id": "No Foam"}]
    )


def test_update_cart_item_requires_explicit_removals_in_replacement_list():
    env1 = store_env(CAFE)
    env1.tools.add_to_cart(
        CAFE,
        "Mocha",
        modifiers=[{"option_id": "No Whipped Cream", "group_id": "Whipped Cream (1)"}],
    )
    env1.tools.update_cart_item(
        "line_1",
        modifiers=[
            {"option_id": "No Whipped Cream", "group_id": "Whipped Cream (1)"},
            {"option_id": "Oat Milk", "group_id": "Milk (1)"},
        ],
    )

    env2 = store_env(CAFE)
    env2.tools.add_to_cart(
        CAFE,
        "Mocha",
        modifiers=[{"option_id": "No Whipped Cream"}, {"option_id": "Oat Milk"}],
    )
    assert env1.tools.get_db_hash() == env2.tools.get_db_hash()


def test_cafe_no_x_is_redundant_when_its_group_gets_another_choice():
    def latte(steps) -> str:
        env = store_env(CAFE)
        for step in steps:
            step(env.tools)
        return env.tools.get_db_hash()

    item = "Iced Hazelnut Mocha"
    removal = {"option_id": "No Hazelnut Cold Foam"}
    other = {"option_id": "Vanilla Cold Foam"}
    swap_only = latte([lambda t: t.add_to_cart(CAFE, item, modifiers=[other])])
    assert (
        latte([lambda t: t.add_to_cart(CAFE, item, modifiers=[removal, other])])
        == swap_only
    )
    assert (
        latte(
            [
                lambda t: t.add_to_cart(CAFE, item, modifiers=[removal]),
                lambda t: t.update_cart_item("line_1", modifiers=[other]),
            ]
        )
        == swap_only
    )


def test_cafe_no_option_for_an_absent_ingredient_is_a_no_op():
    def latte(mods) -> str:
        env = store_env(CAFE)
        env.tools.add_to_cart(
            store_id=CAFE, item_id="Iced Protein Latte", modifiers=mods
        )
        return env.tools.get_db_hash()

    assert latte([{"option_id": "No Caramel Drizzle"}]) == latte([])
    assert latte([{"option_id": "Caramel Drizzle"}]) != latte([])


def test_burger_no_options_stay_real_removals():
    def burger(mods) -> str:
        env = store_env(BURGER)
        env.tools.add_to_cart(
            store_id=BURGER, item_id="Classic Cheeseburger", modifiers=mods
        )
        return env.tools.get_db_hash()

    assert burger([{"option_id": "No Tomato"}]) != burger([])


def test_shot_counts_are_distinct_and_defaults_match_explicit_quantity():
    def latte_hash(shots: int | None) -> str:
        env = store_env("cafe")
        mods = [{"option_id": "Medium"}]
        if shots is not None:
            mods.append(
                {
                    "option_id": "Espresso Shot",
                    "group_id": shots_group,
                    "quantity": shots,
                }
            )
        env.tools.add_to_cart(store_id="cafe", item_id="Latte", modifiers=mods)
        return env.tools.get_db_hash()

    details = store_env("cafe").tools.get_item_details("cafe", "Latte")
    shots_group = next(
        g["name"]
        for g in details["modifier_groups"] + details.get("nested_groups", [])
        if g["name"].startswith("Espresso Shots")
    )
    by_count = {n: latte_hash(n) for n in (1, 2, 3)}
    assert len(set(by_count.values())) == 3
    assert latte_hash(None) == by_count[2]


def _mobile_order_hash(**place) -> str:
    env = store_env(BURGER, "mobile")
    env.tools.add_to_cart(store_id=BURGER, item_id="Double Griddle Burger")
    env.tools.place_order(**place)
    return env.tools.get_db_hash()


def test_mobile_order_hash_covers_time_type_and_customer():
    base = dict(
        customer_name="Maria Lopez",
        customer_phone="949-555-0118",
        scheduled_time="13:00",
    )
    gold = _mobile_order_hash(**base)
    assert _mobile_order_hash(**{**base, "customer_name": "  maria   LOPEZ "}) == gold
    assert _mobile_order_hash(**{**base, "customer_phone": "(949) 555 0118"}) == gold
    assert _mobile_order_hash(**{**base, "customer_phone": "619-555-0156"}) != gold
    assert _mobile_order_hash(**{**base, "scheduled_time": "13:30"}) != gold
    assert _mobile_order_hash(**{**base, "scheduled_time": "ASAP"}) != gold
    # A pickup order stores no address.
    assert _mobile_order_hash(**{**base, "delivery_address": "12 Main St"}) == gold
    delivery = {**base, "order_type": "delivery", "delivery_address": "12 Main Street"}
    assert _mobile_order_hash(**delivery) != gold
    assert _mobile_order_hash(**{**delivery, "delivery_address": " 12 main st "}) == (
        _mobile_order_hash(**delivery)
    )


def test_set_fulfillment_then_place_equals_place_with_arguments():
    env = store_env(BURGER, "mobile")
    env.tools.add_to_cart(store_id=BURGER, item_id="Double Griddle Burger")
    env.tools.set_fulfillment("pickup", "13:00")
    env.tools.place_order(customer_name="Maria Lopez", customer_phone="9495550118")
    direct = _mobile_order_hash(
        customer_name="Maria Lopez", customer_phone="9495550118", scheduled_time="13:00"
    )
    assert env.tools.get_db_hash() == direct


def test_drive_thru_hash_ignores_a_volunteered_name():
    def drive_thru(**place) -> str:
        env = store_env(BURGER)
        env.tools.add_to_cart(store_id=BURGER, item_id="Double Griddle Burger")
        env.tools.place_order(**place)
        return env.tools.get_db_hash()

    assert drive_thru() == drive_thru(customer_name="Sam", customer_phone="4155550101")


def test_delivery_hash_matches_a_spoken_house_number():
    def order(address: str) -> str:
        env = store_env(PIZZA, channel="mobile")
        env.tools.set_fulfillment("delivery", "18:00")
        env.tools.add_to_cart(PIZZA, "Medium Classic Pie", quantity=3)
        env.tools.place_order(
            customer_name="Priya Nair",
            customer_phone="415-555-0144",
            delivery_address=address,
        )
        return env.tools.get_db_hash()

    assert order("Twelve Oak Street") == order("12 Oak Street")
    assert order("Twelve Oak Street") != order("13 Oak Street")


def test_reservation_hash():
    def booked(*bookings, cancel=()) -> str:
        tools = store_env(CAFE).tools
        for b in bookings:
            tools.make_reservation(CAFE, *b)
        for rid in cancel:
            tools.cancel_reservation(rid)
        return tools.get_db_hash()

    untouched = booked()
    one = ("2026-06-29", "18:00", 2, "Maria Lopez", "9495550118")
    gold = booked(one)
    assert gold != untouched
    assert (
        booked(one + ("indoor",)) == gold
    )  # 'any' resolves to the first area with room
    assert booked(one + ("outdoor",)) != gold
    assert booked(one, one) != gold  # double booking
    assert booked(("2026-06-29", "18:30", 2, "Maria Lopez", "9495550118")) != gold
    assert booked(one, cancel=["RES-0021"]) == untouched
    assert booked(cancel=["RES-0001"]) != untouched


def test_task_seeded_reservation_and_isolation():
    task = task_for(CAFE).model_copy(deep=True)
    task.initial_state.initialization_data.agent_data["reservations"] = {
        "RES-T001": {
            "reservation_id": "RES-T001",
            "store_id": CAFE,
            "date": "2026-06-29",
            "time": "18:00",
            "party_size": 2,
            "customer_name": "Maria Lopez",
            "customer_phone": "949-555-0118",
            "seating": "indoor",
        }
    }
    env = get_environment(task=task)
    env.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=None,
        message_history=[],
    )
    assert [r.reservation_id for r in env.tools.find_reservations("9495550118")] == [
        "RES-T001"
    ]
    env.tools.update_reservation("RES-T001", time="19:30", party_size=3)
    fresh = store_env(CAFE)
    assert "RES-T001" not in fresh.tools.db.reservations
    assert fresh.tools.db.reservations["RES-0001"].status == "confirmed"


def test_sessions_do_not_share_state():
    untouched = store_env(CAFE).tools.get_db_hash()
    env = store_env(CAFE, "mobile")
    env.tools.cancel_reservation("RES-0001")
    env.tools.make_reservation(
        CAFE, "2026-06-29", "18:00", 2, "Maria Lopez", "9495550118"
    )
    env.tools.add_to_cart(CAFE, "Cream Scone")
    env.tools.place_order(customer_name="Maria Lopez", customer_phone="9495550118")
    env.tools.add_to_cart(CAFE, "Cream Scone")
    env.tools.transfer_to_human_agents("Customer asked for a manager.")

    fresh = store_env(CAFE)
    assert fresh.tools.db.reservations["RES-0001"].status == "confirmed"
    assert len(fresh.tools.db.reservations) == len(env.tools.db.reservations) - 1
    assert fresh.tools.db.cart.items == []
    assert fresh.tools.db.orders == {}
    assert not fresh.tools.db.cart.transferred_to_human
    assert fresh.tools.get_db_hash() == untouched


def test_tool_calls_leave_the_shared_menus_unchanged():
    def menus() -> str:
        stores = get_environment().tools.db.stores
        return json.dumps({k: v.model_dump() for k, v in stores.items()})

    before = menus()
    for store_id in (CAFE, PIZZA, BURGER):
        task = task_for(store_id)
        env = get_environment(task=task)
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=None,
            message_history=[],
        )
        for action in task.evaluation_criteria.actions or []:
            getattr(env.tools, action.name)(**action.arguments)
        env.tools.search_menu(store_id, query="large")
    assert menus() == before
