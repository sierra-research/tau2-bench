"""Paths and shared constants for the campus domain (Tau2-ZH, v1.1 contract)."""

from tau2.utils.utils import DATA_DIR

CAMPUS_DATA_DIR = DATA_DIR / "tau2" / "domains" / "campus"
CAMPUS_DB_PATH = CAMPUS_DATA_DIR / "db.json"
CAMPUS_USER_DB_PATH = CAMPUS_DATA_DIR / "user_db.json"
CAMPUS_POLICY_PATH = CAMPUS_DATA_DIR / "policy.md"
CAMPUS_TASK_SET_PATH = CAMPUS_DATA_DIR / "tasks.json"
