"""Tests for the shared banking data model and tools."""

import json

import pytest

from tau2.domains.banking.data_model import BankingDB
from tau2.domains.banking.tools import BankingTools


@pytest.fixture
def sample_data() -> dict:
    return {
        "customers": {
            "C001": {
                "customer_id": "C001",
                "name": "Kim Minjun",
                "email": "minjun@example.com",
                "phone": "010-1234-5678",
                "date_of_birth": "1990-05-15",
                "address": {
                    "city": "Seoul",
                    "district": "Gangnam-gu",
                    "detail": "123 Teheran-ro",
                    "postal_code": "06134",
                },
                "annual_income": 50000000,
                "credit_score": 800,
                "kyc_verified": True,
                "accounts": ["A001", "A002"],
            },
            "C002": {
                "customer_id": "C002",
                "name": "Park Suji",
                "email": "suji@example.com",
                "phone": "010-9876-5432",
                "date_of_birth": "1985-11-20",
                "address": {
                    "city": "Busan",
                    "district": "Haeundae-gu",
                    "detail": "456 Marine-ro",
                    "postal_code": "48094",
                },
                "annual_income": 40000000,
                "credit_score": 650,
                "kyc_verified": False,
                "accounts": ["A003"],
            },
        },
        "accounts": {
            "A001": {
                "account_id": "A001",
                "customer_id": "C001",
                "product_id": "P001",
                "account_type": "savings",
                "status": "active",
                "balance": 5000000,
                "currency": "KRW",
                "opened_date": "2024-01-15",
            },
            "A002": {
                "account_id": "A002",
                "customer_id": "C001",
                "product_id": "P002",
                "account_type": "checking",
                "status": "active",
                "balance": 1000000,
                "currency": "KRW",
                "opened_date": "2024-03-01",
            },
            "A003": {
                "account_id": "A003",
                "customer_id": "C002",
                "product_id": "P001",
                "account_type": "savings",
                "status": "active",
                "balance": 2000000,
                "currency": "KRW",
                "opened_date": "2024-06-10",
            },
        },
        "products": {
            "P001": {
                "product_id": "P001",
                "name": "Free Savings",
                "category": "savings",
                "interest_rate": 3.5,
                "min_balance": 10000,
                "max_balance": 100000000,
                "features": ["No minimum balance", "Free transfers"],
            },
            "P002": {
                "product_id": "P002",
                "name": "Smart Checking",
                "category": "checking",
                "interest_rate": 1.0,
                "min_balance": 0,
                "max_balance": 500000000,
                "features": ["Free debit card", "ATM fee refund"],
            },
            "P003": {
                "product_id": "P003",
                "name": "Personal Loan",
                "category": "loan",
                "interest_rate": 5.5,
                "min_balance": 1000000,
                "max_balance": 50000000,
                "features": ["Flexible repayment"],
                "terms_months": 36,
                "eligibility_credit_score": 600,
            },
            "P004": {
                "product_id": "P004",
                "name": "Fixed Deposit",
                "category": "deposit",
                "interest_rate": 4.0,
                "min_balance": 100000,
                "max_balance": 200000000,
                "features": ["Guaranteed rate"],
                "terms_months": 12,
            },
        },
        "transactions": {
            "TX001": {
                "transaction_id": "TX001",
                "account_id": "A001",
                "type": "deposit",
                "amount": 5000000,
                "balance_after": 5000000,
                "counterparty": None,
                "date": "2024-01-15T10:00:00",
                "description": "Initial deposit",
            },
            "TX002": {
                "transaction_id": "TX002",
                "account_id": "A002",
                "type": "deposit",
                "amount": 1000000,
                "balance_after": 1000000,
                "counterparty": None,
                "date": "2024-03-01T14:30:00",
                "description": "Initial deposit",
            },
        },
    }


@pytest.fixture
def db(sample_data: dict) -> BankingDB:
    return BankingDB.model_validate(sample_data)


@pytest.fixture
def tools(db: BankingDB) -> BankingTools:
    return BankingTools(db)


class TestBankingDB:
    def test_construct_from_dict(self, sample_data: dict):
        db = BankingDB.model_validate(sample_data)
        assert len(db.customers) == 2
        assert len(db.accounts) == 3
        assert len(db.products) == 4
        assert len(db.transactions) == 2

    def test_get_statistics(self, db: BankingDB):
        stats = db.get_statistics()
        assert stats == {
            "num_customers": 2,
            "num_accounts": 3,
            "num_products": 4,
            "num_transactions": 2,
        }


class TestFindCustomer:
    def test_find_customer_by_email(self, tools: BankingTools):
        result = tools.find_customer_by_email("minjun@example.com")
        assert result == "C001"

    def test_find_customer_by_email_case_insensitive(self, tools: BankingTools):
        result = tools.find_customer_by_email("MINJUN@EXAMPLE.COM")
        assert result == "C001"

    def test_find_customer_by_email_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Customer not found"):
            tools.find_customer_by_email("nobody@example.com")

    def test_find_customer_by_phone(self, tools: BankingTools):
        result = tools.find_customer_by_phone("010-1234-5678")
        assert result == "C001"

    def test_find_customer_by_phone_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Customer not found"):
            tools.find_customer_by_phone("010-0000-0000")


class TestGetCustomerInfo:
    def test_get_customer_info(self, tools: BankingTools):
        customer = tools.get_customer_info("C001")
        assert customer.name == "Kim Minjun"
        assert customer.email == "minjun@example.com"

    def test_get_customer_info_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Customer not found"):
            tools.get_customer_info("C999")


class TestGetAccount:
    def test_get_account(self, tools: BankingTools):
        account = tools.get_account("A001")
        assert account.balance == 5000000
        assert account.account_type == "savings"

    def test_get_account_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Account not found"):
            tools.get_account("A999")


class TestListProducts:
    def test_list_products_all(self, tools: BankingTools):
        result = json.loads(tools.list_products("all"))
        assert len(result) == 4

    def test_list_products_by_category(self, tools: BankingTools):
        result = json.loads(tools.list_products("savings"))
        assert len(result) == 1
        assert "P001" in result

    def test_list_products_empty_category(self, tools: BankingTools):
        result = json.loads(tools.list_products("card"))
        assert len(result) == 0


class TestGetProductDetails:
    def test_get_product_details(self, tools: BankingTools):
        product = tools.get_product_details("P001")
        assert product.name == "Free Savings"
        assert product.interest_rate == 3.5

    def test_get_product_details_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Product not found"):
            tools.get_product_details("P999")


class TestGetTransactionHistory:
    def test_get_transaction_history(self, tools: BankingTools):
        result = json.loads(tools.get_transaction_history("A001"))
        assert len(result) == 1
        assert result[0]["transaction_id"] == "TX001"

    def test_get_transaction_history_not_found(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Account not found"):
            tools.get_transaction_history("A999")


class TestOpenAccount:
    def test_open_account(self, tools: BankingTools):
        result = json.loads(
            tools.open_account("C001", "P001", initial_deposit=100000)
        )
        assert result["status"] == "opened"
        assert result["balance"] == 100000
        # Account should exist in db
        account = tools.get_account(result["account_id"])
        assert account.status == "active"

    def test_open_account_kyc_not_verified(self, tools: BankingTools):
        with pytest.raises(ValueError, match="KYC verification is not complete"):
            tools.open_account("C002", "P001", initial_deposit=100000)

    def test_open_account_below_min_balance(self, tools: BankingTools):
        with pytest.raises(ValueError, match="below minimum balance"):
            tools.open_account("C001", "P001", initial_deposit=100)

    def test_open_account_above_max_balance(self, tools: BankingTools):
        with pytest.raises(ValueError, match="exceeds maximum balance"):
            tools.open_account("C001", "P001", initial_deposit=200000000)


class TestCloseAccount:
    def test_close_account(self, tools: BankingTools):
        # First set balance to 0
        tools.db.accounts["A001"].balance = 0
        result = json.loads(tools.close_account("A001", "No longer needed"))
        assert result["status"] == "closed"
        assert tools.db.accounts["A001"].status == "closed"

    def test_close_account_nonzero_balance(self, tools: BankingTools):
        with pytest.raises(ValueError, match="balance must be zero"):
            tools.close_account("A001", "No longer needed")

    def test_close_account_already_closed(self, tools: BankingTools):
        tools.db.accounts["A001"].balance = 0
        tools.close_account("A001", "test")
        with pytest.raises(ValueError, match="already closed"):
            tools.close_account("A001", "test again")


class TestTransfer:
    def test_transfer(self, tools: BankingTools):
        result = json.loads(tools.transfer("A001", "A002", 500000))
        assert result["status"] == "completed"
        assert result["from_balance_after"] == 4500000
        assert result["to_balance_after"] == 1500000

    def test_transfer_insufficient_balance(self, tools: BankingTools):
        with pytest.raises(ValueError, match="Insufficient balance"):
            tools.transfer("A001", "A002", 99999999)

    def test_transfer_inactive_account(self, tools: BankingTools):
        tools.db.accounts["A001"].status = "frozen"
        with pytest.raises(ValueError, match="Source account is not active"):
            tools.transfer("A001", "A002", 100000)

    def test_transfer_creates_transactions(self, tools: BankingTools):
        initial_tx_count = len(tools.db.transactions)
        tools.transfer("A001", "A002", 100000)
        assert len(tools.db.transactions) == initial_tx_count + 2


class TestApplyLoan:
    def test_apply_loan(self, tools: BankingTools):
        result = json.loads(tools.apply_loan("C001", "P003", 10000000))
        assert result["status"] == "approved"
        assert result["amount"] == 10000000

    def test_apply_loan_not_loan_product(self, tools: BankingTools):
        with pytest.raises(ValueError, match="not a loan product"):
            tools.apply_loan("C001", "P001", 10000000)

    def test_apply_loan_below_min(self, tools: BankingTools):
        with pytest.raises(ValueError, match="below minimum"):
            tools.apply_loan("C001", "P003", 100)

    def test_apply_loan_above_max(self, tools: BankingTools):
        with pytest.raises(ValueError, match="exceeds maximum"):
            tools.apply_loan("C001", "P003", 999999999)


class TestUpdateCustomerInfo:
    def test_update_email(self, tools: BankingTools):
        result = json.loads(
            tools.update_customer_info("C001", "email", "new@example.com")
        )
        assert result["status"] == "updated"
        assert tools.db.customers["C001"].email == "new@example.com"

    def test_update_phone(self, tools: BankingTools):
        tools.update_customer_info("C001", "phone", "010-0000-0000")
        assert tools.db.customers["C001"].phone == "010-0000-0000"

    def test_update_invalid_field(self, tools: BankingTools):
        with pytest.raises(ValueError, match="cannot be updated"):
            tools.update_customer_info("C001", "name", "New Name")

    def test_update_address(self, tools: BankingTools):
        address_json = json.dumps({
            "city": "Incheon",
            "district": "Yeonsu-gu",
            "detail": "789 Songdo-ro",
            "postal_code": "21990",
        })
        tools.update_customer_info("C001", "address", address_json)
        assert tools.db.customers["C001"].address.city == "Incheon"


class TestGenericTools:
    def test_calculate(self, tools: BankingTools):
        assert tools.calculate("2 + 3") == "5.0"

    def test_transfer_to_human(self, tools: BankingTools):
        assert tools.transfer_to_human_agents("test") == "Transfer successful"
