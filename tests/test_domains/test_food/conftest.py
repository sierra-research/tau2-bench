"""Shared helpers for the food domain tests."""

import re

import pytest

from tau2.domains.food.environment import get_environment, get_tasks
from tau2.environment.environment import Environment

BURGER, CAFE, PIZZA = "burger", "cafe", "pizza"
STORES = (CAFE, PIZZA, BURGER)
DEFAULT_NOW = "2026-06-22 11:00"
# Reserved fictional phone block NPA-555-0100..0199.
FICTIONAL_PHONE_RE = re.compile(r"^\d{3}-555-01\d{2}$")
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


def store_of(task) -> str:
    return task.initial_state.initialization_data.agent_data["cart"]["store_id"]


def task_for(store_id: str):
    """The first task set at the store."""
    return next(t for t in get_tasks("base") if store_of(t) == store_id)


def store_env(
    store_id: str, channel: str = "drive_thru", now: str = DEFAULT_NOW
) -> Environment:
    """A fresh environment whose session is at the store, on the channel, at the time."""
    env = get_environment()
    env.tools.update_db(
        {"cart": {"store_id": store_id, "current_time": now, "channel": channel}}
    )
    return env


def line_names(env: Environment, line: int = -1) -> set[str]:
    """Modifier names on a cart line."""
    return {m.name for m in env.tools.db.cart.items[line].modifiers}


@pytest.fixture
def environment() -> Environment:
    return get_environment()
