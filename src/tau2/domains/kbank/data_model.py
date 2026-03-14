from tau2.domains.banking.data_model import BankingDB
from tau2.domains.kbank.utils import KBANK_DB_PATH


def get_db():
    return BankingDB.load(KBANK_DB_PATH)
