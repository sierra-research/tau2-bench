"""Валидатор данных домена banking_ru.

Ловит расхождения между db.json, tasks.json и спецификацией домена до того,
как на них будет потрачен платный прогон модели: конвенции идентификаторов,
ссылочную целостность и сходимость производных чисел.
"""

import json
import re
from datetime import date, timedelta

import pytest

from tau2.domains.banking_ru.data_model import BankingDB
from tau2.domains.banking_ru.environment import get_tasks
from tau2.domains.banking_ru.utils import BANKING_RU_DB_PATH, BANKING_RU_TASK_SET_PATH

ID_PATTERNS = {
    "customers": r"^[a-z]+_[a-z]_\d{4}$",
    "accounts": r"^acc_\d{4}$",
    "cards": r"^card_\d{4}$",
    "transactions": r"^txn_\d{6}$",
    "disputes": r"^dsp_\d{4}$",
    "subscriptions": r"^sub_\d{4}$",
    "autopayments": r"^ap_\d{4}$",
    "deposits": r"^dep_\d{4}$",
    "loans": r"^ln_\d{4}$",
    "cases": r"^case_\d{4}$",
    "devices": r"^dev_\d{4}$",
    "documents": r"^doc_\d{4}_\d$",
}


@pytest.fixture(scope="module")
def db() -> BankingDB:
    return BankingDB.load(BANKING_RU_DB_PATH)


@pytest.mark.parametrize("collection,pattern", ID_PATTERNS.items())
def test_id_conventions(db: BankingDB, collection: str, pattern: str):
    for key in getattr(db, collection):
        assert re.match(pattern, key), (
            f"{collection}: идентификатор {key!r} не соответствует соглашению {pattern}"
        )


def test_collection_keys_match_object_ids(db: BankingDB):
    for collection in ID_PATTERNS:
        for key, value in getattr(db, collection).items():
            assert key == value.id, f"{collection}: ключ {key} не совпадает с id {value.id}"


def test_referential_integrity(db: BankingDB):
    for account in db.accounts.values():
        assert account.customer_id in db.customers
    for card in db.cards.values():
        assert card.customer_id in db.customers
        assert card.account_id in db.accounts
        assert card.id in db.card_limits, f"У карты {card.id} нет записи лимитов"
    for transaction in db.transactions.values():
        assert transaction.customer_id in db.customers
        assert transaction.account_id in db.accounts
        if transaction.card_id is not None:
            assert transaction.card_id in db.cards
        if transaction.dispute_id is not None:
            assert transaction.dispute_id in db.disputes
    for dispute in db.disputes.values():
        assert dispute.transaction_id in db.transactions
    for deposit in db.deposits.values():
        assert deposit.payout_account_id in db.accounts
    for customer in db.customers.values():
        assert customer.tariff_id in db.tariffs


def test_blocked_cards_have_reason_and_date(db: BankingDB):
    for card in db.cards.values():
        if card.status == "blocked":
            assert card.block_reason is not None
            assert card.blocked_at is not None


def test_deposit_payout_arithmetic(db: BankingDB):
    """Выплата при досрочном расторжении выводится из ставки и дат.

    Готовых полей с процентами в модели нет намеренно: расчёт потери — работа
    агента (механизм M1)."""
    today = date.fromisoformat(db.today)
    for deposit in db.deposits.values():
        if deposit.early_rate is not None:
            days = (today - date.fromisoformat(deposit.opened_at)).days
            expected = deposit.amount * (1 + deposit.early_rate * days / 365)
            assert deposit.early_withdrawal_payout == pytest.approx(
                expected, abs=0.01
            ), f"{deposit.id}: досрочная выплата не выводится из early_rate"
        assert deposit.maturity_payout > deposit.early_withdrawal_payout, (
            f"{deposit.id}: выплата в срок должна быть больше досрочной"
        )


def test_customers_have_secrets(db: BankingDB):
    """У каждого клиента есть кодовое слово и телефон — основа идентификации."""
    words = [c.code_word for c in db.customers.values()]
    assert all(words), "пустое кодовое слово"
    assert all(c.phone for c in db.customers.values())


def test_loan_payoff_arithmetic(db: BankingDB):
    """Полная сумма погашения складывается из долга, процентов и живых штрафов."""
    loan = db.loans["ln_6630"]
    unpaid = sum(p.amount for p in loan.penalties if not p.waived and not p.paid)
    assert loan.principal + loan.accrued_interest + unpaid == pytest.approx(383500.0)
    assert loan.waivers_used == loan.max_waivers, (
        "Ловушка bank_hard_02 требует, чтобы послабление было уже исчерпано"
    )


def test_cashback_expectation(db: BankingDB):
    """Ожидаемый кешбэк из bank_easy_02 выводится из правил, а не из текста задачи."""
    transaction = db.transactions["txn_552290"]
    rules = db.cashback_rules[transaction.customer_id]
    category = next(c for c in rules.categories if transaction.mcc in c.mcc_codes)
    assert transaction.amount * category.rate == pytest.approx(60.0)
    assert rules.accrued_current_period == 0.0


def test_dispute_deadline(db: BankingDB):
    """Срок ответа по спору из bank_easy_01 приходится на 19 сентября 2026 года."""
    dispute = db.disputes["dsp_3391"]
    deadline = date.fromisoformat(dispute.filed_at) + timedelta(days=dispute.sla_days)
    assert deadline.isoformat() == "2026-09-19"


def test_subscription_savings(db: BankingDB):
    """Экономия по подпискам дороже 400 ₽ из bank_hard_01 равна 1938 ₽."""
    expensive = [
        s
        for s in db.subscriptions.values()
        if s.customer_id == "solomina_o_5214" and s.status == "active" and s.amount > 400
    ]
    assert sum(s.amount for s in expensive) == pytest.approx(1938.0)
    assert len(expensive) == 3


def test_tasks_reference_existing_entities():
    """Каждый идентификатор, упомянутый в задачах, существует в БД."""
    db = BankingDB.load(BANKING_RU_DB_PATH)
    known = set()
    for collection in ID_PATTERNS:
        known |= set(getattr(db, collection))
    known |= set(db.card_limits) | set(db.tariffs)
    known |= {p.id for loan in db.loans.values() for p in loan.penalties}

    raw = json.loads(BANKING_RU_TASK_SET_PATH.read_text(encoding="utf-8"))
    id_like = re.compile(
        r"\b(?:acc|card|txn|dsp|sub|ap|dep|ln|pen|case|dev|doc)_\d[0-9a-z_]*\b"
        r"|\b[a-z]+_[a-z]_\d{4}\b"
    )
    for task in raw:
        for match in id_like.findall(json.dumps(task, ensure_ascii=False)):
            assert match in known, (
                f"Задача {task['id']} ссылается на несуществующий объект {match}"
            )


def test_task_split_covers_all_tasks():
    from tau2.domains.banking_ru.environment import get_tasks_split

    splits = get_tasks_split()
    assert "base" in splits, "Сплит base обязателен: он используется по умолчанию"
    all_ids = {task.id for task in get_tasks(task_split_name=None)}
    assert set(splits["base"]) == all_ids
    assert set(splits["train"]) <= all_ids, "train ссылается на несуществующие задачи"
    assert len(splits["train"]) == 10, "train — фиксированные 10 открытых задач"
    assert set(splits["test"]) <= all_ids, "test ссылается на несуществующие задачи"
    assert not set(splits["test"]) & set(splits["train"]), (
        "train и test не пересекаются: test — закрытый набор"
    )
    assert set(splits["train"]) | set(splits["test"]) == all_ids, (
        "каждая задача принадлежит ровно одному из сплитов train/test"
    )


def test_wave2_dispute_window(db: BankingDB):
    """bank_011 внутри окна, bank_012 — за ним, дистракторы по разные стороны."""
    today = date.fromisoformat(db.today)
    window = db.dispute_window_days

    def age(txn_id: str) -> int:
        return (today - date.fromisoformat(db.transactions[txn_id].date)).days

    assert age("txn_771204") <= window, "bank_011: спор должен быть возможен"
    assert age("txn_768120") > window, "bank_011: мартовский дистрактор вне окна"
    assert age("txn_540117") > window, "bank_012: операция должна быть просрочена"
    assert age("txn_552140") <= window, "bank_012: майский дистрактор внутри окна"


def test_wave2_familiar_charges_match_subscriptions(db: BankingDB):
    """bank_008: обе «незнакомые» операции соответствуют активным подпискам."""
    subscriptions = {
        s.name: s for s in db.subscriptions.values()
        if s.customer_id == "smirnova_o_1123"
    }
    assert all(s.status == "active" for s in subscriptions.values())
    for txn_id, name in (("txn_055210", "Яндекс Плюс"), ("txn_055180", "Литрес Подписка")):
        transaction = db.transactions[txn_id]
        assert transaction.is_subscription
        assert transaction.amount == subscriptions[name].amount


def test_wave2_foreign_operations(db: BankingDB):
    """bank_010: мошеннические операции отличимы от легитимных зарубежных."""
    fraud = ("txn_088345", "txn_088320")
    legit = ("txn_088300", "txn_087100")
    for txn_id in fraud:
        assert db.transactions[txn_id].country == "TH"
        assert not db.transactions[txn_id].is_subscription
    assert db.transactions["txn_088300"].is_subscription
    for txn_id in fraud + legit:
        assert db.transactions[txn_id].dispute_id is None, (
            "споры по этим операциям открывает агент, а не снимок БД"
        )


def test_wave2_hold_expiry(db: BankingDB):
    """bank_017: холд снимается автоматически не позже семи дней с даты операции."""
    for txn_id in ("txn_799102", "txn_799050"):
        transaction = db.transactions[txn_id]
        assert transaction.status == "hold"
        expiry = date.fromisoformat(transaction.hold_expires_at)
        assert expiry > date.fromisoformat(db.today)
        assert (expiry - date.fromisoformat(transaction.date)).days <= 7


def test_wave2_processing_transfer(db: BankingDB):
    """bank_020: перевод в обработке и его исполненный двойник двумя неделями раньше."""
    assert db.transactions["txn_861530"].status == "processing"
    assert db.transactions["txn_860210"].status == "posted"
    assert db.transactions["txn_861530"].date == "2026-08-26"


def test_wave2_sbp_headroom(db: BankingDB):
    """bank_019: текущий лимит мал для перевода, максимум по тарифу — достаточен."""
    limits = db.card_limits["card_5581"]
    tariff = db.tariffs[db.customers["nikitin_r_5581"].tariff_id]
    assert limits.sbp_limit < 150000.0 <= tariff.max_sbp_limit


def test_wave2_starts_empty_cases(db: BankingDB):
    """Обращения создаёт только агент: заранее их в БД нет."""
    assert db.cases == {}
