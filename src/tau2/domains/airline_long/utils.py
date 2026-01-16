from tau2.utils.utils import DATA_DIR

# Use the regular airline data directory (not airline_long)
# The _long variant only provides different environment/tools, not different data
AIRLINE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "airline"
AIRLINE_DB_PATH = AIRLINE_DATA_DIR / "db.json"
AIRLINE_POLICY_PATH = AIRLINE_DATA_DIR / "policy.md"
AIRLINE_TASK_SET_PATH = AIRLINE_DATA_DIR / "tasks.json"
