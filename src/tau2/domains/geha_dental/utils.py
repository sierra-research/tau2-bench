from tau2.utils.utils import DATA_DIR

GEHA_DENTAL_DATA_DIR = DATA_DIR / "tau2" / "domains" / "geha_dental"
GEHA_DENTAL_DB_PATH = GEHA_DENTAL_DATA_DIR / "db.json"
GEHA_DENTAL_POLICY_PATH = GEHA_DENTAL_DATA_DIR / "policy.md"
GEHA_DENTAL_TASK_SET_PATH = GEHA_DENTAL_DATA_DIR / "tasks.json"

# The sandbox runs on a fixed date so tasks and grading are reproducible.
TODAY = "2026-03-16"
# Sandbox rule: an approved enrollment change takes effect on the first day of the next month.
NEXT_EFFECTIVE_DATE = "2026-04-01"
