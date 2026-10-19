"""Validation of the loan_servicing database (db.json)."""

import re
from datetime import date

import pytest

from tau2.domains.loan_servicing.data_model import LoanServicingDB, get_db
from tau2.domains.loan_servicing.utils import LOAN_SERVICING_TODAY

TODAY = date.fromisoformat(LOAN_SERVICING_TODAY)
ID_PATTERNS = {
    "borrower": r"^BF-\d{5}$",
    "loan": r"^LN-\d{5}$",
    "payment": r"^PM-\d{5}$",
    "account": r"^BA-\d{5}$",
    "fee": r"^FE-\d{5}$",
    "hardship": r"^HP-\d{5}$",
}


@pytest.fixture(scope="module")
def db() -> LoanServicingDB:
    return get_db()


def expected_monthly_payment(principal: float, rate: float, term: int) -> float:
    i = rate / 12
    return principal * i / (1 - (1 + i) ** -term)


def test_db_loads(db: LoanServicingDB):
    stats = db.get_statistics()
    assert stats["num_borrowers"] > 0
    assert stats["num_loans"] > 0
    assert db.document_requests == {}
    assert db.transfers == {}


def test_ids_match_keys_and_format(db: LoanServicingDB):
    for key, borrower in db.borrowers.items():
        assert key == borrower.borrower_id
        assert re.match(ID_PATTERNS["borrower"], key)
        for account in borrower.bank_accounts:
            assert re.match(ID_PATTERNS["account"], account.method_id)
    for key, loan in db.loans.items():
        assert key == loan.loan_id
        assert re.match(ID_PATTERNS["loan"], key)
        for fee in loan.fees:
            assert re.match(ID_PATTERNS["fee"], fee.fee_id)
        for plan in loan.hardship_history:
            assert re.match(ID_PATTERNS["hardship"], plan.hardship_id)
    for key, payment in db.payments.items():
        assert key == payment.payment_id
        assert re.match(ID_PATTERNS["payment"], key)


def test_ids_are_unique(db: LoanServicingDB):
    accounts = [a.method_id for b in db.borrowers.values() for a in b.bank_accounts]
    fees = [f.fee_id for loan in db.loans.values() for f in loan.fees]
    plans = [p.hardship_id for loan in db.loans.values() for p in loan.hardship_history]
    for ids in (accounts, fees, plans):
        assert len(ids) == len(set(ids))


def test_borrower_loan_links(db: LoanServicingDB):
    for loan in db.loans.values():
        assert loan.borrower_ids
        for borrower_id in loan.borrower_ids:
            assert loan.loan_id in db.borrowers[borrower_id].loan_ids
    for borrower in db.borrowers.values():
        for loan_id in borrower.loan_ids:
            assert borrower.borrower_id in db.loans[loan_id].borrower_ids


def test_payment_methods_belong_to_loan_borrowers(db: LoanServicingDB):
    def methods(loan_id: str) -> set[str]:
        loan = db.loans[loan_id]
        return {
            a.method_id
            for borrower_id in loan.borrower_ids
            for a in db.borrowers[borrower_id].bank_accounts
        }

    for payment in db.payments.values():
        assert payment.method_id in methods(payment.loan_id)
    for loan in db.loans.values():
        if loan.autopay.enabled:
            assert loan.autopay.method_id in methods(loan.loan_id)


def test_monthly_payment_matches_amortization(db: LoanServicingDB):
    for loan in db.loans.values():
        expected = expected_monthly_payment(
            loan.original_principal, loan.annual_rate, loan.term_months
        )
        assert loan.monthly_payment == pytest.approx(expected, abs=0.006)


def test_principal_balance_matches_posted_payments(db: LoanServicingDB):
    paid: dict[str, float] = {loan_id: 0.0 for loan_id in db.loans}
    for payment in db.payments.values():
        if payment.status == "posted":
            assert payment.allocation is not None
            allocation = payment.allocation
            total = allocation.fees + allocation.interest + allocation.principal
            assert payment.amount == pytest.approx(total, abs=0.006)
            paid[payment.loan_id] += allocation.principal
        else:
            assert payment.allocation is None
    for loan in db.loans.values():
        assert loan.original_principal - paid[loan.loan_id] == pytest.approx(
            loan.principal_balance, abs=0.01
        )


def test_scheduled_payments_are_in_the_future(db: LoanServicingDB):
    for payment in db.payments.values():
        payment_date = date.fromisoformat(payment.date)
        if payment.status == "scheduled":
            assert payment_date > TODAY
        else:
            assert payment_date <= TODAY


def test_status_is_consistent(db: LoanServicingDB):
    missed = {"current": 0, "in_hardship": 0, "past_due_30": 1, "past_due_60": 2}
    for loan in db.loans.values():
        if loan.status == "paid_off":
            assert loan.principal_balance == 0
            assert loan.past_due_amount == 0
        elif loan.status == "charged_off":
            assert loan.past_due_amount >= 3 * loan.monthly_payment - 0.01
        else:
            expected = missed[loan.status] * loan.monthly_payment
            assert loan.past_due_amount == pytest.approx(expected, abs=0.01)
        active = [
            p
            for p in loan.hardship_history
            if p.start_date <= LOAN_SERVICING_TODAY <= p.end_date
        ]
        assert bool(active) == (loan.status == "in_hardship")


def test_due_dates(db: LoanServicingDB):
    for loan in db.loans.values():
        assert 1 <= loan.due_day <= 28
        if loan.status in ("paid_off", "charged_off"):
            assert loan.next_due_date is None
        else:
            next_due = date.fromisoformat(loan.next_due_date)
            assert next_due > TODAY
            assert next_due.day == loan.due_day
        for change in loan.due_date_changes:
            assert date.fromisoformat(change) <= TODAY


def test_autopay_settings(db: LoanServicingDB):
    for loan in db.loans.values():
        autopay = loan.autopay
        if not autopay.enabled:
            assert autopay.method_id is None and autopay.day is None
            continue
        assert loan.status in ("current", "past_due_30")
        assert max(1, loan.due_day - 5) <= autopay.day <= loan.due_day


def test_fees(db: LoanServicingDB):
    for loan in db.loans.values():
        for fee in loan.fees:
            assert fee.amount > 0
            assert date.fromisoformat(fee.assessed_date) <= TODAY
            assert (fee.status == "waived") == (fee.waived_date is not None)


def test_borrower_fields(db: LoanServicingDB):
    for borrower in db.borrowers.values():
        date.fromisoformat(borrower.date_of_birth)
        assert re.match(r"^[A-Z]\d[A-Z] \d[A-Z]\d$", borrower.postal_code)
        assert re.match(r"^\d{3}-\d{3}-\d{4}$", borrower.phone)
        assert borrower.email == borrower.email.strip().lower()
        assert borrower.bank_accounts
