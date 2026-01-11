import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tau2.utils.utils import DATA_DIR

InsuranceClaims_DATA_DIR = DATA_DIR / "tau2" / "domains" / "claims"
CLAIMS_DB_PATH = InsuranceClaims_DATA_DIR / "db.json"
POLICY_PATH = InsuranceClaims_DATA_DIR/"policy.md"
TASK_SET_PATH = InsuranceClaims_DATA_DIR/"tasks.json"


