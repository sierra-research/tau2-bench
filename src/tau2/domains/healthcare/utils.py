# Copyright Sierra
from pathlib import Path

HEALTHCARE_DOMAIN_PATH = Path(__file__).parent
HEALTHCARE_DATA_PATH = (
    HEALTHCARE_DOMAIN_PATH.parent.parent.parent.parent
    / "data"
    / "tau2"
    / "domains"
    / "healthcare"
)
HEALTHCARE_DB_PATH = HEALTHCARE_DATA_PATH / "db.json"
HEALTHCARE_USER_DB_PATH = HEALTHCARE_DATA_PATH / "user_db.json"
HEALTHCARE_POLICY_PATH = HEALTHCARE_DATA_PATH / "policy.md"
HEALTHCARE_TASK_SET_PATH = HEALTHCARE_DATA_PATH / "tasks.json"
HEALTHCARE_TASK_SET_FULL_PATH = HEALTHCARE_DATA_PATH / "tasks_full.json"
HEALTHCARE_TASK_SET_SMALL_PATH = HEALTHCARE_DATA_PATH / "tasks_small.json"
