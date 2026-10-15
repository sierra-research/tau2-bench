"""Utility functions and constants for the hospitality domain."""

from datetime import date, datetime
from pathlib import Path

# Base path for hospitality domain data
HOSPITALITY_DATA_DIR = (
    Path(__file__).parents[4] / "data" / "tau2" / "domains" / "hospitality"
)

# Data file paths
HOSPITALITY_DB_PATH = HOSPITALITY_DATA_DIR / "db.json"
HOSPITALITY_USER_DB_PATH = HOSPITALITY_DATA_DIR / "user_db.json"
HOSPITALITY_POLICY_PATH = HOSPITALITY_DATA_DIR / "policy.md"
HOSPITALITY_TASK_SET_PATH = HOSPITALITY_DATA_DIR / "tasks.json"

# Default date for the simulation
DEFAULT_DATE = date(2026, 1, 14)
DEFAULT_DATETIME = datetime(2026, 1, 14, 18, 0, 0)


def get_today() -> date:
    """Get the current date for the simulation."""
    return DEFAULT_DATE


def get_now() -> datetime:
    """Get the current datetime for the simulation."""
    return DEFAULT_DATETIME


def is_federal_holiday(check_date: date) -> bool:
    """Check if a given date is a federal holiday in 2026."""
    federal_holidays_2026 = [
        date(2026, 1, 1),  # New Year's Day (Thursday)
        date(2026, 1, 19),  # MLK Day (Monday)
        date(2026, 2, 16),  # Presidents' Day (Monday)
        date(2026, 5, 25),  # Memorial Day (Monday)
        date(2026, 6, 19),  # Juneteenth (Friday)
        date(2026, 7, 3),  # Independence Day observed (Friday, since 7/4 is Saturday)
        date(2026, 7, 4),  # Independence Day (Saturday)
        date(2026, 9, 7),  # Labor Day (Monday)
        date(2026, 10, 12),  # Columbus Day (Monday)
        date(2026, 11, 11),  # Veterans Day (Wednesday)
        date(2026, 11, 26),  # Thanksgiving (Thursday)
        date(2026, 12, 25),  # Christmas (Friday)
    ]
    return check_date in federal_holidays_2026


def is_weekday(check_date: date) -> bool:
    """Check if a given date is a weekday (Mon-Fri)."""
    return check_date.weekday() < 5


def is_lunch_time(check_time: datetime) -> bool:
    """Check if the given time is before 5 PM (lunch special hours)."""
    return check_time.hour < 17
