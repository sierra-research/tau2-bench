from tau2.domains.banking.data_model import BankingDB
from tau2.domains.tossbank.utils import TOSSBANK_DB_PATH


def get_db():
    return BankingDB.load(TOSSBANK_DB_PATH)
