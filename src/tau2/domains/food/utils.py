"""Data paths and currency formatting helpers for the food domain."""

from tau2.utils.utils import DATA_DIR

FOOD_DATA_DIR = DATA_DIR / "tau2" / "domains" / "food"
FOOD_DB_PATH = FOOD_DATA_DIR / "db.json"
FOOD_MENUS_DIR = FOOD_DATA_DIR / "menus"
FOOD_POLICY_PATH = FOOD_DATA_DIR / "policy.md"
FOOD_TASK_SET_PATH = FOOD_DATA_DIR / "tasks.json"

# Session time ('YYYY-MM-DD HH:MM') used when a task's cart does not set `current_time`.
FOOD_DEFAULT_NOW = "2026-06-22 11:00"


def format_cents(cents: int) -> str:
    """Formats an amount in cents as dollars, e.g. 1250 -> '$12.50'."""
    return f"${cents // 100}.{cents % 100:02d}"
