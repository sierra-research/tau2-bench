from tau2.domains.banking.data_model import BankingDB
from tau2.domains.kakaobank.utils import KAKAOBANK_DB_PATH


def get_db():
    return BankingDB.load(KAKAOBANK_DB_PATH)
