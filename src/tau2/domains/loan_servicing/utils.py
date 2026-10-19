from tau2.utils.utils import DATA_DIR

LOAN_SERVICING_DATA_DIR = DATA_DIR / "tau2" / "domains" / "loan_servicing"
LOAN_SERVICING_DB_PATH = LOAN_SERVICING_DATA_DIR / "db.json"
LOAN_SERVICING_POLICY_PATH = LOAN_SERVICING_DATA_DIR / "policy.md"
LOAN_SERVICING_POLICY_FR_PATH = LOAN_SERVICING_DATA_DIR / "policy_fr.md"
LOAN_SERVICING_TASK_SET_PATH = LOAN_SERVICING_DATA_DIR / "tasks.json"

# The fixed "today" of the environment. The policy states the same date.
LOAN_SERVICING_TODAY = "2026-03-16"
