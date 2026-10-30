"""Food domain environment: the policy with the task's store context, and task loading."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.food.data_model import CartState, FoodDB, Store
from tau2.domains.food.menu_prompt import render_menu
from tau2.domains.food.tools import FoodTools
from tau2.domains.food.utils import (
    FOOD_DB_PATH,
    FOOD_DEFAULT_NOW,
    FOOD_POLICY_PATH,
    FOOD_TASK_SET_PATH,
    format_cents,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

_CACHED_FOOD_DB: Optional[FoodDB] = None


def _get_fresh_food_db() -> FoodDB:
    global _CACHED_FOOD_DB
    if _CACHED_FOOD_DB is None:
        _CACHED_FOOD_DB = FoodDB.load(FOOD_DB_PATH)
    # Store objects in `stores` are shared read-only across sessions and must never be
    # mutated by tools; `cart`, `orders`, and `reservations` are copied per session.
    return FoodDB.model_construct(
        stores=_CACHED_FOOD_DB.stores,
        cart=_CACHED_FOOD_DB.cart.model_copy(deep=True),
        orders={k: v.model_copy(deep=True) for k, v in _CACHED_FOOD_DB.orders.items()},
        reservations={
            k: v.model_copy() for k, v in _CACHED_FOOD_DB.reservations.items()
        },
    )


def _current_store_block(store: Store, cart: CartState) -> str:
    """The Current Store section: who, when, how the customer orders, and reservations."""
    now = datetime.strptime(cart.current_time or FOOD_DEFAULT_NOW, "%Y-%m-%d %H:%M")
    lines = [
        f"- Restaurant: {store.name}",
        f"- store_id for tool calls: {store.store_id}",
    ]
    if cart.channel == "mobile":
        lines.append(
            "- Order channel: mobile (the customer orders ahead by phone for a later pickup or delivery)"
        )
    else:
        lines.append(
            "- Order channel: drive-thru (the customer is at the drive-thru speaker and picks up now)"
        )
    lines += [
        f"- Current time: {now:%A}, {now:%Y-%m-%d %H:%M}",
        f"- Store hours today: {store.open_hours}",
    ]
    if cart.channel == "mobile":
        lines += [
            f"- Fulfillment so far: {cart.order_type}, {cart.scheduled_time} (change with set_fulfillment)",
            f"- Pickup lead time: {store.pickup_lead_minutes} minutes",
            f"- Delivery: {format_cents(store.delivery_fee_cents)} fee, "
            f"{format_cents(store.min_delivery_subtotal_cents)} "
            f"minimum subtotal, {store.delivery_lead_minutes}-minute lead time",
        ]
    else:
        lines.append(f"- Fulfillment method: {cart.order_type}")
    policy = store.reservation_policy
    if policy is None:
        lines.append("- Table reservations: not accepted")
    else:
        seating = " or ".join(policy.tables_per_slot)
        lines.append(
            f"- Table reservations: accepted (party of {policy.min_party_size} to {policy.max_party_size}, "
            f"up to {policy.max_days_ahead} days ahead, {seating} seating)"
        )
    return "\n\n## Current Store\n\n" + "\n".join(lines) + "\n"


def _store_context(db: FoodDB, task: Task) -> str:
    """Renders the Current Store and concise Menu Data sections for the task's store."""
    init = task.initial_state.initialization_data if task.initial_state else None
    cart_data = ((init.agent_data or {}) if init else {}).get("cart") or {}
    store_id = cart_data.get("store_id")
    if store_id not in db.stores:
        raise ValueError(f"Task {task.id} has no valid cart.store_id: {store_id!r}")
    preview_db = FoodDB.model_construct(
        stores=db.stores, cart=CartState.model_validate(cart_data), orders={}
    )
    tools = FoodTools(preview_db)
    store = db.stores[store_id]
    menu = render_menu(store, tools._check_item_availability)
    return _current_store_block(store, preview_db.cart) + f"\n## Menu Data\n\n{menu}"


def get_environment(
    db: Optional[FoodDB] = None,
    solo_mode: bool = False,
    task: Optional[Task] = None,
) -> Environment:
    """Builds the food environment.

    Args:
        db: Database to use; defaults to a fresh copy of the food db.
        solo_mode: Not supported.
        task: With a task, the policy ends with that task's Current Store and Menu Data.
    """
    if solo_mode:
        raise ValueError("Food domain does not support solo mode")
    if db is None:
        db = _get_fresh_food_db()
    tools = FoodTools(db)
    policy = FOOD_POLICY_PATH.read_text()
    if task is not None:
        policy += _store_context(db, task)
    return Environment(
        domain_name="food",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load food domain tasks, optionally filtered to a named split."""
    tasks = load_file(FOOD_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {list(task_splits.keys())}"
        )
    tasks = [task for task in tasks if task.id in task_splits[task_split_name]]
    return tasks


def get_tasks_split() -> dict[str, list[str]]:
    """Load named task splits for the food domain."""
    split_file = (
        Path(FOOD_TASK_SET_PATH).parent / f"split_{Path(FOOD_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
