"""Comprehensive tests for the banking_knowledge domain tools.

This module tests ALL tools in the banking_knowledge domain, covering:
- Core agent tools (user lookup, email, time, transfer, verification)
- Discoverable agent tools (disputes, card mgmt, account mgmt, etc.)
- User discoverable tools (referral links, card digits, deposits, etc.)
- User tools (transactions, applications, referrals)
- PIN validation helper
- db_query module operations
"""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.banking_knowledge.data_model import DatabaseTable, TransactionalDB
from tau2.domains.banking_knowledge.db_query import (
    add_to_db,
    list_databases,
    query_database_tool,
    query_db,
    remove_from_db,
    update_record_in_db,
)
from tau2.domains.banking_knowledge.tools import (
    KnowledgeTools,
    KnowledgeUserTools,
    _validate_pin,
)
from tau2.environment.environment import Environment

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_knowledge_db() -> TransactionalDB:
    """Create a comprehensive TransactionalDB with test data for all tool categories."""
    return TransactionalDB(
        users=DatabaseTable(
            data={
                "user_001": {
                    "name": "Amara Okonkwo",
                    "user_id": "user_001",
                    "address": "305 Magnolia Street, Houston, TX 77002",
                    "email": "amara@icloud.com",
                    "phone_number": "713-555-0963",
                    "date_of_birth": "08/11/1997",
                },
                "user_002": {
                    "name": "Fatima Al-Hassan",
                    "user_id": "user_002",
                    "address": "1923 Oak Park Boulevard, Detroit, MI 48226",
                    "email": "fatima@protonmail.com",
                    "phone_number": "313-555-0246",
                    "date_of_birth": "12/05/1993",
                },
            }
        ),
        accounts=DatabaseTable(
            data={
                "chk_001": {
                    "account_id": "chk_001",
                    "user_id": "user_001",
                    "account_type": "checking",
                    "account_class": "Standard Checking",
                    "class": "checking",
                    "level": "Blue Account",
                    "current_holdings": "5000.00",
                    "status": "OPEN",
                    "date_opened": "01/15/2024",
                },
                "chk_002": {
                    "account_id": "chk_002",
                    "user_id": "user_002",
                    "account_type": "checking",
                    "account_class": "Premium Checking",
                    "class": "checking",
                    "level": "Green Account",
                    "current_holdings": "200.00",
                    "status": "OPEN",
                    "date_opened": "03/10/2024",
                },
                "sav_001": {
                    "account_id": "sav_001",
                    "user_id": "user_001",
                    "account_type": "savings",
                    "account_class": "High-Yield Savings",
                    "class": "savings",
                    "level": "Silver Account",
                    "current_holdings": "10000.00",
                    "status": "OPEN",
                    "date_opened": "01/15/2024",
                },
                "chk_closed": {
                    "account_id": "chk_closed",
                    "user_id": "user_001",
                    "account_type": "checking",
                    "account_class": "Standard Checking",
                    "class": "checking",
                    "level": "Blue Account",
                    "current_holdings": "0.00",
                    "status": "CLOSED",
                    "date_opened": "01/01/2023",
                },
            }
        ),
        credit_card_accounts=DatabaseTable(
            data={
                "cc_001": {
                    "account_id": "cc_001",
                    "user_id": "user_001",
                    "card_type": "Gold Rewards Card",
                    "date_opened": "02/01/2024",
                    "current_balance": 1500.00,
                    "reward_points": 5000,
                    "credit_limit": 5000,
                    "status": "ACTIVE",
                },
                "cc_002": {
                    "account_id": "cc_002",
                    "user_id": "user_002",
                    "card_type": "Silver Rewards Card",
                    "date_opened": "04/01/2024",
                    "current_balance": 800.00,
                    "reward_points": 2000,
                    "credit_limit": 3000,
                    "status": "ACTIVE",
                },
            }
        ),
        credit_card_transaction_history=DatabaseTable(
            data={
                "txn_001": {
                    "transaction_id": "txn_001",
                    "user_id": "user_001",
                    "credit_card_type": "Gold Rewards Card",
                    "merchant_name": "Amazon",
                    "transaction_amount": "$150.00",
                    "transaction_date": "10/01/2025",
                    "category": "Shopping",
                    "status": "COMPLETED",
                    "rewards_earned": "375 points",
                },
                "txn_002": {
                    "transaction_id": "txn_002",
                    "user_id": "user_001",
                    "credit_card_type": "Gold Rewards Card",
                    "merchant_name": "Delta Airlines",
                    "transaction_amount": "$500.00",
                    "transaction_date": "10/15/2025",
                    "category": "Travel",
                    "status": "COMPLETED",
                    "rewards_earned": "1250 points",
                },
                "txn_003": {
                    "transaction_id": "txn_003",
                    "user_id": "user_002",
                    "credit_card_type": "Silver Rewards Card",
                    "merchant_name": "Trader Joe's",
                    "transaction_amount": "$47.83",
                    "transaction_date": "11/01/2025",
                    "category": "Groceries",
                    "status": "COMPLETED",
                    "rewards_earned": "47 points",
                },
            }
        ),
        debit_cards=DatabaseTable(
            data={
                "dbc_001": {
                    "card_id": "dbc_001",
                    "account_id": "chk_001",
                    "user_id": "user_001",
                    "last_4_digits": "4321",
                    "status": "PENDING",
                    "issue_reason": "new_account",
                    "date_issued": "11/01/2025",
                },
                "dbc_002": {
                    "card_id": "dbc_002",
                    "account_id": "chk_001",
                    "user_id": "user_001",
                    "last_4_digits": "8765",
                    "status": "ACTIVE",
                    "issue_reason": "new_account",
                    "date_issued": "09/01/2025",
                },
                "dbc_003": {
                    "card_id": "dbc_003",
                    "account_id": "chk_002",
                    "user_id": "user_002",
                    "last_4_digits": "1111",
                    "status": "PENDING",
                    "issue_reason": "lost",
                    "date_issued": "11/01/2025",
                },
                "dbc_frozen": {
                    "card_id": "dbc_frozen",
                    "account_id": "chk_001",
                    "user_id": "user_001",
                    "last_4_digits": "9999",
                    "status": "FROZEN",
                    "issue_reason": "new_account",
                    "date_issued": "08/01/2025",
                },
                "dbc_expired": {
                    "card_id": "dbc_expired",
                    "account_id": "chk_001",
                    "user_id": "user_001",
                    "last_4_digits": "5555",
                    "status": "PENDING",
                    "issue_reason": "expired",
                    "date_issued": "11/10/2025",
                },
            }
        ),
        referrals=DatabaseTable(
            data={
                "ref_001": {
                    "referral_id": "ref_001",
                    "referrer_id": "user_001",
                    "referred_account_type": "Gold Rewards Card",
                    "referral_status": "NO_PROGRESS",
                    "date": "10/01/2025",
                },
            }
        ),
        credit_card_applications=DatabaseTable(data={}),
        user_discoverable_tools=DatabaseTable(data={}),
        user_discoverable_tool_calls=DatabaseTable(data={}),
        verification_history=DatabaseTable(data={}),
        cash_back_disputes=DatabaseTable(data={}),
        bank_account_transaction_history=DatabaseTable(data={}),
        agent_discoverable_tools=DatabaseTable(data={}),
        task_config=DatabaseTable(
            data={"dispute_settings": {"auto_resolve_disputes": True}}
        ),
        human_transfer_requests=DatabaseTable(data={}),
        transaction_disputes=DatabaseTable(data={}),
        credit_card_orders=DatabaseTable(data={}),
        debit_card_orders=DatabaseTable(data={}),
        credit_card_closure_reasons=DatabaseTable(data={}),
        credit_card_account_flags=DatabaseTable(data={}),
        credit_limit_increase_requests=DatabaseTable(data={}),
        payment_history=DatabaseTable(data={}),
        debit_card_disputes=DatabaseTable(data={}),
    )


def _create_test_environment(db: TransactionalDB) -> Environment:
    """Create a test environment with the given database (no retrieval tools)."""
    tools = KnowledgeTools(db)
    user_tools = KnowledgeUserTools(db)
    return Environment(
        domain_name="banking_knowledge",
        policy="",
        tools=tools,
        user_tools=user_tools,
    )


@pytest.fixture
def environment(base_knowledge_db: TransactionalDB) -> Environment:
    """Standard environment for testing."""
    return _create_test_environment(base_knowledge_db)


@pytest.fixture
def env_no_auto_resolve(base_knowledge_db: TransactionalDB) -> Environment:
    """Environment with auto_resolve_disputes disabled."""
    base_knowledge_db.task_config = DatabaseTable(
        data={"dispute_settings": {"auto_resolve_disputes": False}}
    )
    return _create_test_environment(base_knowledge_db)


@pytest.fixture
def environment_auto_resolve(
    knowledge_db_auto_resolve: TransactionalDB,
) -> Environment:
    """Environment with auto_resolve_disputes enabled."""
    return _create_test_environment(knowledge_db_auto_resolve)


@pytest.fixture
def environment_no_auto_resolve(
    knowledge_db_no_auto_resolve: TransactionalDB,
) -> Environment:
    """Environment with auto_resolve_disputes disabled."""
    return _create_test_environment(knowledge_db_no_auto_resolve)


# =============================================================================
# Helper to make ToolCall + get_response easier
# =============================================================================


def call(env: Environment, name: str, arguments: dict, requestor: str = "assistant"):
    """Helper to create a ToolCall and get a response."""
    tc = ToolCall(id="test", name=name, arguments=arguments, requestor=requestor)
    return env.get_response(tc)


def call_discoverable_agent(env: Environment, tool_name: str, arguments: dict):
    """Helper to unlock then call a discoverable agent tool."""
    # Unlock
    resp = call(env, "unlock_discoverable_agent_tool", {"agent_tool_name": tool_name})
    assert not resp.error, f"Failed to unlock {tool_name}: {resp.content}"
    # Call
    resp = call(
        env,
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": tool_name,
            "arguments": json.dumps(arguments),
        },
    )
    return resp


def call_discoverable_user(env: Environment, tool_name: str, arguments: dict):
    """Helper to give then call a discoverable user tool."""
    # Give
    resp = call(
        env, "give_discoverable_user_tool", {"discoverable_tool_name": tool_name}
    )
    assert not resp.error, f"Failed to give {tool_name}: {resp.content}"
    # Call
    resp = call(
        env,
        "call_discoverable_user_tool",
        {
            "discoverable_tool_name": tool_name,
            "arguments": json.dumps(arguments),
        },
        requestor="user",
    )
    return resp


# =============================================================================
# 1. PIN Validation Helper
# =============================================================================


class TestValidatePin:
    """Tests for the _validate_pin helper function."""

    def test_valid_pin(self):
        assert _validate_pin("5823") is None

    def test_valid_pin_with_zero(self):
        assert _validate_pin("0571") is None

    def test_too_short(self):
        result = _validate_pin("123")
        assert result is not None
        assert "4 digits" in result

    def test_too_long(self):
        result = _validate_pin("12345")
        assert result is not None
        assert "4 digits" in result

    def test_non_digits(self):
        result = _validate_pin("12ab")
        assert result is not None
        assert "4 digits" in result

    def test_empty_string(self):
        result = _validate_pin("")
        assert result is not None

    def test_sequential_ascending(self):
        result = _validate_pin("1234")
        assert result is not None
        assert "sequential" in result.lower()

    def test_sequential_descending(self):
        result = _validate_pin("4321")
        assert result is not None
        assert "sequential" in result.lower()

    def test_sequential_high(self):
        result = _validate_pin("6789")
        assert result is not None
        assert "sequential" in result.lower()

    def test_all_same_digits(self):
        result = _validate_pin("1111")
        assert result is not None
        assert "same digit" in result.lower()

    def test_all_zeros(self):
        result = _validate_pin("0000")
        assert result is not None
        assert "same digit" in result.lower()

    def test_two_repeated_digits_ok(self):
        """Two repeated digits (e.g., 1122) should be allowed."""
        assert _validate_pin("1122") is None


# =============================================================================
# 2. Core Agent Tools - User Information
# =============================================================================


class TestGetUserInformation:
    """Tests for user lookup tools."""

    def test_get_user_by_id_found(self, environment: Environment):
        resp = call(environment, "get_user_information_by_id", {"user_id": "user_001"})
        assert not resp.error
        assert "Amara Okonkwo" in resp.content
        assert "amara@icloud.com" in resp.content

    def test_get_user_by_id_not_found(self, environment: Environment):
        resp = call(
            environment, "get_user_information_by_id", {"user_id": "nonexistent"}
        )
        assert "No records found" in resp.content

    def test_get_user_by_name_found(self, environment: Environment):
        resp = call(
            environment,
            "get_user_information_by_name",
            {"customer_name": "Fatima Al-Hassan"},
        )
        assert not resp.error
        assert "user_002" in resp.content
        assert "fatima@protonmail.com" in resp.content

    def test_get_user_by_name_not_found(self, environment: Environment):
        resp = call(
            environment, "get_user_information_by_name", {"customer_name": "Nobody"}
        )
        assert "No records found" in resp.content

    def test_get_user_by_name_case_sensitive(self, environment: Environment):
        """Name lookup is case-sensitive."""
        resp = call(
            environment,
            "get_user_information_by_name",
            {"customer_name": "amara okonkwo"},
        )
        assert "No records found" in resp.content

    def test_get_user_by_email_found(self, environment: Environment):
        resp = call(
            environment, "get_user_information_by_email", {"email": "amara@icloud.com"}
        )
        assert not resp.error
        assert "Amara Okonkwo" in resp.content

    def test_get_user_by_email_not_found(self, environment: Environment):
        resp = call(
            environment, "get_user_information_by_email", {"email": "nobody@test.com"}
        )
        assert "No records found" in resp.content


class TestChangeUserEmail:
    """Tests for change_user_email."""

    def test_change_email_success(self, environment: Environment):
        resp = call(
            environment,
            "change_user_email",
            {
                "user_id": "user_001",
                "new_email": "newemail@test.com",
            },
        )
        assert not resp.error
        assert "Email updated successfully" in resp.content
        assert "newemail@test.com" in resp.content
        # Verify DB state
        assert (
            environment.tools.db.users.data["user_001"]["email"] == "newemail@test.com"
        )

    def test_change_email_user_not_found(self, environment: Environment):
        resp = call(
            environment,
            "change_user_email",
            {
                "user_id": "nonexistent",
                "new_email": "test@test.com",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


# =============================================================================
# 3. Core Agent Tools - Time, Transfer, Verification
# =============================================================================


class TestGetCurrentTime:
    """Tests for get_current_time."""

    def test_returns_fixed_time(self, environment: Environment):
        resp = call(environment, "get_current_time", {})
        assert not resp.error
        assert "2025-11-14 03:40:00 EST" in resp.content


class TestTransferToHumanAgents:
    """Tests for transfer_to_human_agents."""

    def test_valid_transfer_reason(self, environment: Environment):
        resp = call(
            environment,
            "transfer_to_human_agents",
            {
                "summary": "Customer needs help with fraud",
                "reason": "fraud_or_security_concern",
            },
        )
        assert not resp.error
        assert "Transfer successful" in resp.content
        assert "fraud_or_security_concern" in resp.content

    def test_default_other_reason(self, environment: Environment):
        resp = call(
            environment,
            "transfer_to_human_agents",
            {
                "summary": "General request",
                "reason": "other",
            },
        )
        assert not resp.error
        assert "Transfer successful" in resp.content

    def test_invalid_transfer_reason(self, environment: Environment):
        resp = call(
            environment,
            "transfer_to_human_agents",
            {
                "summary": "Test",
                "reason": "totally_invalid_reason",
            },
        )
        assert "Error" in resp.content
        assert "Invalid transfer reason" in resp.content


class TestLogVerification:
    """Tests for log_verification."""

    def test_log_verification_success(self, environment: Environment):
        resp = call(
            environment,
            "log_verification",
            {
                "name": "Amara Okonkwo",
                "user_id": "user_001",
                "address": "305 Magnolia Street, Houston, TX 77002",
                "email": "amara@icloud.com",
                "phone_number": "713-555-0963",
                "date_of_birth": "08/11/1997",
                "time_verified": "2025-11-14 03:40:00 EST",
            },
        )
        assert not resp.error
        assert "Verification logged successfully" in resp.content
        assert len(environment.tools.db.verification_history.data) == 1

    def test_log_verification_duplicate(self, environment: Environment):
        """Logging same verification twice should fail (same ID)."""
        args = {
            "name": "Amara Okonkwo",
            "user_id": "user_001",
            "address": "305 Magnolia Street, Houston, TX 77002",
            "email": "amara@icloud.com",
            "phone_number": "713-555-0963",
            "date_of_birth": "08/11/1997",
            "time_verified": "2025-11-14 03:40:00 EST",
        }
        call(environment, "log_verification", args)
        resp = call(environment, "log_verification", args)
        assert "Failed" in resp.content or "already exist" in resp.content


# =============================================================================
# 4. Referral and Transaction Lookup
# =============================================================================


class TestGetReferralsByUser:
    """Tests for get_referrals_by_user."""

    def test_get_referrals_found(self, environment: Environment):
        resp = call(environment, "get_referrals_by_user", {"user_id": "user_001"})
        assert not resp.error
        assert "ref_001" in resp.content
        assert "Gold Rewards Card" in resp.content

    def test_get_referrals_none(self, environment: Environment):
        resp = call(environment, "get_referrals_by_user", {"user_id": "user_002"})
        assert "No records found" in resp.content


class TestGetCreditCardTransactionsByUser:
    """Tests for get_credit_card_transactions_by_user."""

    def test_get_transactions_found(self, environment: Environment):
        resp = call(
            environment, "get_credit_card_transactions_by_user", {"user_id": "user_001"}
        )
        assert not resp.error
        assert "txn_001" in resp.content
        assert "txn_002" in resp.content
        assert "Amazon" in resp.content
        assert "Delta Airlines" in resp.content

    def test_get_transactions_none(self, environment: Environment):
        resp = call(
            environment,
            "get_credit_card_transactions_by_user",
            {"user_id": "nonexistent"},
        )
        assert "No records found" in resp.content


class TestGetCreditCardAccountsByUser:
    """Tests for get_credit_card_accounts_by_user."""

    def test_get_accounts_found(self, environment: Environment):
        resp = call(
            environment, "get_credit_card_accounts_by_user", {"user_id": "user_001"}
        )
        assert not resp.error
        assert "cc_001" in resp.content
        assert "Gold Rewards Card" in resp.content

    def test_get_accounts_none(self, environment: Environment):
        resp = call(
            environment, "get_credit_card_accounts_by_user", {"user_id": "nonexistent"}
        )
        assert "No records found" in resp.content


# =============================================================================
# 5. Discoverable Tool Infrastructure
# =============================================================================


class TestGiveDiscoverableUserTool:
    """Tests for give_discoverable_user_tool."""

    def test_give_valid_tool(self, environment: Environment):
        resp = call(
            environment,
            "give_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
            },
        )
        assert not resp.error
        assert "Tool given to user" in resp.content
        assert len(environment.tools.db.user_discoverable_tools.data) == 1

    def test_give_unknown_tool(self, environment: Environment):
        resp = call(
            environment,
            "give_discoverable_user_tool",
            {
                "discoverable_tool_name": "nonexistent_tool",
            },
        )
        assert "Error" in resp.content
        assert "Unknown discoverable tool" in resp.content

    def test_give_tool_with_invalid_args(self, environment: Environment):
        resp = call(
            environment,
            "give_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": '{"bogus_param": "value"}',
            },
        )
        assert "Error" in resp.content
        assert "Unexpected parameter" in resp.content


class TestUnlockDiscoverableAgentTool:
    """Tests for unlock_discoverable_agent_tool."""

    def test_unlock_valid_tool(self, environment: Environment):
        resp = call(
            environment,
            "unlock_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
            },
        )
        assert not resp.error
        assert "Tool unlocked" in resp.content
        # DB should NOT have a record yet (only on call)
        assert len(environment.tools.db.agent_discoverable_tools.data) == 0

    def test_unlock_unknown_tool(self, environment: Environment):
        resp = call(
            environment,
            "unlock_discoverable_agent_tool",
            {
                "agent_tool_name": "nonexistent_tool_9999",
            },
        )
        assert "Error" in resp.content
        assert "Unknown agent tool" in resp.content


class TestCallDiscoverableAgentTool:
    """Tests for call_discoverable_agent_tool."""

    def test_call_without_unlock_fails(self, environment: Environment):
        resp = call(
            environment,
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
                "arguments": json.dumps(
                    {"transaction_id": "txn_001", "new_rewards_earned": "500 points"}
                ),
            },
        )
        assert "Error" in resp.content
        assert "has not been unlocked" in resp.content

    def test_call_unknown_tool(self, environment: Environment):
        resp = call(
            environment,
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": "fake_tool_1234",
                "arguments": "{}",
            },
        )
        assert "Error" in resp.content

    def test_call_with_invalid_json(self, environment: Environment):
        call(
            environment,
            "unlock_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
            },
        )
        resp = call(
            environment,
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
                "arguments": "not valid json {{{",
            },
        )
        assert "Error" in resp.content
        assert "Invalid JSON" in resp.content


class TestCallDiscoverableUserTool:
    """Tests for call_discoverable_user_tool."""

    def test_call_without_giving_fails(self, environment: Environment):
        resp = call(
            environment,
            "call_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "user_001", "transaction_id": "txn_001"}
                ),
            },
            requestor="user",
        )
        assert "Error" in resp.content
        assert "has not been given" in resp.content

    def test_call_unknown_user_tool(self, environment: Environment):
        resp = call(
            environment,
            "call_discoverable_user_tool",
            {
                "discoverable_tool_name": "totally_fake_tool",
                "arguments": "{}",
            },
            requestor="user",
        )
        assert "Error" in resp.content


# =============================================================================
# 6. Update Transaction Rewards (Agent Discoverable)
# =============================================================================


class TestUpdateTransactionRewards:
    """Tests for update_transaction_rewards_3847."""

    def test_update_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "update_transaction_rewards_3847",
            {
                "transaction_id": "txn_001",
                "new_rewards_earned": "500 points",
            },
        )
        assert not resp.error
        assert (
            "Transaction updated" in resp.content
            or "updated successfully" in resp.content
        )
        assert (
            environment.tools.db.credit_card_transaction_history.data["txn_001"][
                "rewards_earned"
            ]
            == "500 points"
        )
        # DB should record the agent tool call
        assert len(environment.tools.db.agent_discoverable_tools.data) == 1

    def test_update_nonexistent_transaction(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "update_transaction_rewards_3847",
            {
                "transaction_id": "txn_nonexistent",
                "new_rewards_earned": "100 points",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


# =============================================================================
# 7. Credit Card Transaction Disputes (Agent Discoverable)
# =============================================================================


class TestFileCreditCardTransactionDispute:
    """Tests for file_credit_card_transaction_dispute_4829."""

    def test_file_dispute_full_refund(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_credit_card_transaction_dispute_4829",
            {
                "transaction_id": "txn_001",
                "card_action": "keep_active",
                "card_last_4_digits": "1234",
                "full_name": "Amara Okonkwo",
                "user_id": "user_001",
                "phone": "713-555-0963",
                "email": "amara@icloud.com",
                "address": "305 Magnolia Street, Houston, TX 77002",
                "contacted_merchant": True,
                "purchase_date": "10/01/2025",
                "issue_noticed_date": "10/15/2025",
                "dispute_reason": "unauthorized_fraudulent_charge",
                "resolution_requested": "full_refund",
                "eligible_for_provisional_credit": True,
            },
        )
        assert not resp.error
        assert "filed successfully" in resp.content
        assert len(environment.tools.db.transaction_disputes.data) == 1

    def test_file_dispute_partial_refund_requires_amount(
        self, environment: Environment
    ):
        resp = call_discoverable_agent(
            environment,
            "file_credit_card_transaction_dispute_4829",
            {
                "transaction_id": "txn_001",
                "card_action": "keep_active",
                "card_last_4_digits": "1234",
                "full_name": "Amara Okonkwo",
                "user_id": "user_001",
                "phone": "713-555-0963",
                "email": "amara@icloud.com",
                "address": "305 Magnolia Street",
                "contacted_merchant": False,
                "purchase_date": "10/01/2025",
                "issue_noticed_date": "10/15/2025",
                "dispute_reason": "incorrect_amount",
                "resolution_requested": "partial_refund",
                "eligible_for_provisional_credit": False,
                # Missing partial_refund_amount
            },
        )
        assert "Error" in resp.content
        assert "partial_refund_amount" in resp.content

    def test_file_dispute_partial_refund_with_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_credit_card_transaction_dispute_4829",
            {
                "transaction_id": "txn_002",
                "card_action": "cancel_and_reissue",
                "card_last_4_digits": "5678",
                "full_name": "Amara Okonkwo",
                "user_id": "user_001",
                "phone": "713-555-0963",
                "email": "amara@icloud.com",
                "address": "305 Magnolia Street",
                "contacted_merchant": True,
                "purchase_date": "10/15/2025",
                "issue_noticed_date": "10/20/2025",
                "dispute_reason": "incorrect_amount",
                "resolution_requested": "partial_refund",
                "eligible_for_provisional_credit": True,
                "partial_refund_amount": 100.00,
            },
        )
        assert not resp.error
        assert "filed successfully" in resp.content

    def test_file_dispute_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_credit_card_transaction_dispute_4829",
            {
                "transaction_id": "txn_001",
                "card_action": "keep_active",
                "card_last_4_digits": "1234",
                "full_name": "Test User",
                "user_id": "user_001",
                "phone": "555-1234",
                "email": "test@test.com",
                "address": "123 Main St",
                "contacted_merchant": False,
                "purchase_date": "10/01/2025",
                "issue_noticed_date": "10/15/2025",
                "dispute_reason": "invalid_reason",
                "resolution_requested": "full_refund",
                "eligible_for_provisional_credit": False,
            },
        )
        assert "Error" in resp.content
        assert "Invalid dispute_reason" in resp.content

    def test_file_dispute_invalid_card_action(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_credit_card_transaction_dispute_4829",
            {
                "transaction_id": "txn_001",
                "card_action": "destroy_card",
                "card_last_4_digits": "1234",
                "full_name": "Test User",
                "user_id": "user_001",
                "phone": "555-1234",
                "email": "test@test.com",
                "address": "123 Main St",
                "contacted_merchant": False,
                "purchase_date": "10/01/2025",
                "issue_noticed_date": "10/15/2025",
                "dispute_reason": "unauthorized_fraudulent_charge",
                "resolution_requested": "full_refund",
                "eligible_for_provisional_credit": False,
            },
        )
        assert "Error" in resp.content
        assert "Invalid card_action" in resp.content


# =============================================================================
# 8. Debit Card Transaction Disputes (Agent Discoverable)
# =============================================================================


class TestFileDebitCardTransactionDispute:
    """Tests for file_debit_card_transaction_dispute_6281."""

    def test_file_debit_dispute_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_debit_card_transaction_dispute_6281",
            {
                "transaction_id": "txn_debit_001",
                "account_id": "chk_001",
                "card_id": "dbc_002",
                "user_id": "user_001",
                "dispute_category": "unauthorized_transaction",
                "transaction_date": "11/01/2025",
                "discovery_date": "11/05/2025",
                "disputed_amount": 250.00,
                "transaction_type": "pin_purchase",
                "card_in_possession": True,
                "pin_compromised": "no",
                "contacted_merchant": True,
                "police_report_filed": False,
                "written_statement_provided": True,
                "provisional_credit_eligible": True,
                "customer_max_liability_amount": 50.00,
                "card_action": "keep_active",
            },
        )
        assert not resp.error
        assert "Dispute ID:" in resp.content
        assert len(environment.tools.db.debit_card_disputes.data) == 1

    def test_file_debit_dispute_invalid_category(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "file_debit_card_transaction_dispute_6281",
            {
                "transaction_id": "txn_debit_002",
                "account_id": "chk_001",
                "card_id": "dbc_002",
                "user_id": "user_001",
                "dispute_category": "invalid_category",
                "transaction_date": "11/01/2025",
                "discovery_date": "11/05/2025",
                "disputed_amount": 100.00,
                "transaction_type": "pin_purchase",
                "card_in_possession": True,
                "pin_compromised": "no",
                "contacted_merchant": False,
                "police_report_filed": False,
                "written_statement_provided": False,
                "provisional_credit_eligible": False,
                "customer_max_liability_amount": 0,
                "card_action": "keep_active",
            },
        )
        assert "Error" in resp.content
        assert "Invalid dispute_category" in resp.content


class TestSetDebitCardRecurringBlock:
    """Tests for set_debit_card_recurring_block_7382."""

    def test_block_recurring(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "set_debit_card_recurring_block_7382",
            {
                "card_id": "dbc_002",
                "block_recurring": True,
            },
        )
        assert not resp.error
        assert "BLOCKED" in resp.content
        assert (
            environment.tools.db.debit_cards.data["dbc_002"]["recurring_blocked"]
            is True
        )

    def test_unblock_recurring(self, environment: Environment):
        environment.tools.db.debit_cards.data["dbc_002"]["recurring_blocked"] = True
        resp = call_discoverable_agent(
            environment,
            "set_debit_card_recurring_block_7382",
            {
                "card_id": "dbc_002",
                "block_recurring": False,
            },
        )
        assert not resp.error
        assert "UNBLOCKED" in resp.content

    def test_block_nonexistent_card(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "set_debit_card_recurring_block_7382",
            {
                "card_id": "dbc_nonexistent",
                "block_recurring": True,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


# =============================================================================
# 9. Credit Card Account Management (Agent Discoverable)
# =============================================================================


class TestOrderReplacementCreditCard:
    """Tests for order_replacement_credit_card_7291.

    Note: The tool writes to 'credit_card_replacement_orders' but the data model
    field is 'credit_card_orders'. When the DB table is missing, add_to_db returns
    False and the tool reports an error. Tests validate both error paths and
    validation logic.
    """

    def test_order_replacement_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "order_replacement_credit_card_7291",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "shipping_address": "123 Main St",
                "reason": "bored_of_design",
            },
        )
        assert "Error" in resp.content
        assert "Invalid reason" in resp.content

    def test_order_replacement_account_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "order_replacement_credit_card_7291",
            {
                "credit_card_account_id": "cc_nonexistent",
                "user_id": "user_001",
                "shipping_address": "123 Main St",
                "reason": "lost",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestLogCreditCardClosureReason:
    """Tests for log_credit_card_closure_reason_4521."""

    def test_log_closure_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "log_credit_card_closure_reason_4521",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "closure_reason": "annual_fee",
            },
        )
        assert not resp.error
        assert "logged successfully" in resp.content
        assert len(environment.tools.db.credit_card_closure_reasons.data) == 1

    def test_log_closure_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "log_credit_card_closure_reason_4521",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "closure_reason": "alien_invasion",
            },
        )
        assert "Error" in resp.content
        assert "Invalid closure_reason" in resp.content


class TestApplyCreditCardAccountFlag:
    """Tests for apply_credit_card_account_flag_6147."""

    def test_apply_flag_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_credit_card_account_flag_6147",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "flag_type": "annual_fee_waived",
                "expiration_date": "11/14/2026",
                "reason": "retention_offer",
            },
        )
        assert not resp.error
        assert "flag applied successfully" in resp.content
        assert len(environment.tools.db.credit_card_account_flags.data) == 1

    def test_apply_flag_invalid_type(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_credit_card_account_flag_6147",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "flag_type": "invalid_flag",
                "expiration_date": "11/14/2026",
                "reason": "retention_offer",
            },
        )
        assert "Error" in resp.content
        assert "Invalid flag_type" in resp.content

    def test_apply_flag_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_credit_card_account_flag_6147",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "flag_type": "annual_fee_waived",
                "expiration_date": "11/14/2026",
                "reason": "invalid_reason",
            },
        )
        assert "Error" in resp.content
        assert "Invalid reason" in resp.content


class TestCloseCreditCardAccount:
    """Tests for close_credit_card_account_7834."""

    def test_close_account_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_credit_card_account_7834",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
            },
        )
        assert not resp.error
        assert "closed successfully" in resp.content
        assert (
            environment.tools.db.credit_card_accounts.data["cc_001"]["status"]
            == "CLOSED"
        )

    def test_close_account_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_credit_card_account_7834",
            {
                "credit_card_account_id": "cc_nonexistent",
                "user_id": "user_001",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


# =============================================================================
# 10. Statement Credits & CC Payments
# =============================================================================


class TestApplyStatementCredit:
    """Tests for apply_statement_credit_8472."""

    def test_apply_credit_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_statement_credit_8472",
            {
                "user_id": "user_001",
                "credit_card_account_id": "cc_001",
                "amount": 25.00,
                "reason": "goodwill_adjustment",
            },
        )
        assert not resp.error
        assert "applied successfully" in resp.content
        assert "$25.00" in resp.content

    def test_apply_credit_zero_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_statement_credit_8472",
            {
                "user_id": "user_001",
                "credit_card_account_id": "cc_001",
                "amount": 0,
                "reason": "goodwill_adjustment",
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content

    def test_apply_credit_negative_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_statement_credit_8472",
            {
                "user_id": "user_001",
                "credit_card_account_id": "cc_001",
                "amount": -50.00,
                "reason": "goodwill_adjustment",
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content

    def test_apply_credit_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_statement_credit_8472",
            {
                "user_id": "user_001",
                "credit_card_account_id": "cc_001",
                "amount": 25.00,
                "reason": "because_i_want_to",
            },
        )
        assert "Error" in resp.content
        assert "Invalid reason" in resp.content

    def test_apply_credit_account_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_statement_credit_8472",
            {
                "user_id": "user_001",
                "credit_card_account_id": "cc_nonexistent",
                "amount": 25.00,
                "reason": "goodwill_adjustment",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestPayCreditCardFromChecking:
    """Tests for pay_credit_card_from_checking_9182."""

    def test_pay_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "pay_credit_card_from_checking_9182",
            {
                "user_id": "user_001",
                "checking_account_id": "chk_001",
                "credit_card_account_id": "cc_001",
                "amount": 500.00,
            },
        )
        assert not resp.error
        assert "Payment processed successfully" in resp.content
        # Verify checking balance decreased
        assert (
            environment.tools.db.accounts.data["chk_001"]["current_holdings"]
            == "4500.00"
        )

    def test_pay_insufficient_funds(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "pay_credit_card_from_checking_9182",
            {
                "user_id": "user_002",
                "checking_account_id": "chk_002",
                "credit_card_account_id": "cc_002",
                "amount": 10000.00,
            },
        )
        assert "Error" in resp.content
        assert "Insufficient funds" in resp.content

    def test_pay_checking_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "pay_credit_card_from_checking_9182",
            {
                "user_id": "user_001",
                "checking_account_id": "chk_nonexistent",
                "credit_card_account_id": "cc_001",
                "amount": 100.00,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_pay_zero_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "pay_credit_card_from_checking_9182",
            {
                "user_id": "user_001",
                "checking_account_id": "chk_001",
                "credit_card_account_id": "cc_001",
                "amount": 0,
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content


# =============================================================================
# 11. Credit Limit Increase Flow
# =============================================================================


class TestCreditLimitIncrease:
    """Tests for credit limit increase request/approve/deny tools."""

    def test_submit_request_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "submit_credit_limit_increase_request_7392",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "requested_increase_amount": 2500,
            },
        )
        assert not resp.error
        assert "submitted successfully" in resp.content
        assert len(environment.tools.db.credit_limit_increase_requests.data) == 1

    def test_submit_request_zero_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "submit_credit_limit_increase_request_7392",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "requested_increase_amount": 0,
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content

    def test_approve_increase_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "approve_credit_limit_increase_5847",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "new_credit_limit": 7500,
            },
        )
        assert not resp.error
        assert "Credit limit increase approved" in resp.content
        assert (
            environment.tools.db.credit_card_accounts.data["cc_001"]["credit_limit"]
            == "$7500.00"
        )

    def test_approve_increase_account_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "approve_credit_limit_increase_5847",
            {
                "credit_card_account_id": "cc_nonexistent",
                "user_id": "user_001",
                "new_credit_limit": 7500,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_deny_increase_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "deny_credit_limit_increase_5848",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "denial_reason": "insufficient_account_age",
            },
        )
        assert not resp.error
        assert "denied" in resp.content.lower()

    def test_deny_increase_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "deny_credit_limit_increase_5848",
            {
                "credit_card_account_id": "cc_001",
                "user_id": "user_001",
                "denial_reason": "bad_vibes",
            },
        )
        assert "Error" in resp.content
        assert "Invalid denial_reason" in resp.content


# =============================================================================
# 12. Bank Account Management (Agent Discoverable)
# =============================================================================


class TestOpenBankAccount:
    """Tests for open_bank_account_4821."""

    def test_open_checking_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "open_bank_account_4821",
            {
                "user_id": "user_002",
                "account_type": "checking",
                "account_class": "Standard Checking",
            },
        )
        assert not resp.error
        assert "opened successfully" in resp.content

    def test_open_savings_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "open_bank_account_4821",
            {
                "user_id": "user_002",
                "account_type": "savings",
                "account_class": "High-Yield Savings",
            },
        )
        assert not resp.error
        assert "opened successfully" in resp.content

    def test_open_invalid_type(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "open_bank_account_4821",
            {
                "user_id": "user_001",
                "account_type": "crypto_wallet",
                "account_class": "Test",
            },
        )
        assert "Error" in resp.content
        assert "Invalid account_type" in resp.content


class TestCloseBankAccount:
    """Tests for close_bank_account_7392."""

    def test_close_zero_balance_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_bank_account_7392",
            {
                "account_id": "chk_closed",
            },
        )
        # Account already closed might cause issues; let's re-open it first
        environment.tools.db.accounts.data["chk_closed"]["status"] = "OPEN"
        resp = call_discoverable_agent(
            environment,
            "close_bank_account_7392",
            {
                "account_id": "chk_closed",
            },
        )
        assert not resp.error
        assert "closed successfully" in resp.content

    def test_close_nonzero_balance_fails(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_bank_account_7392",
            {
                "account_id": "chk_001",
            },
        )
        assert "Error" in resp.content
        assert "balance" in resp.content.lower()

    def test_close_nonexistent_account(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_bank_account_7392",
            {
                "account_id": "acct_nonexistent",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestTransferFundsBetweenAccounts:
    """Tests for transfer_funds_between_bank_accounts_7291."""

    def test_transfer_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "transfer_funds_between_bank_accounts_7291",
            {
                "source_account_id": "chk_001",
                "destination_account_id": "chk_002",
                "amount": 1000.00,
            },
        )
        assert not resp.error
        assert "Transfer completed successfully" in resp.content
        assert (
            environment.tools.db.accounts.data["chk_001"]["current_holdings"]
            == "$4000.00"
        )
        assert (
            environment.tools.db.accounts.data["chk_002"]["current_holdings"]
            == "$1200.00"
        )

    def test_transfer_insufficient_funds(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "transfer_funds_between_bank_accounts_7291",
            {
                "source_account_id": "chk_002",
                "destination_account_id": "chk_001",
                "amount": 99999.00,
            },
        )
        assert "Error" in resp.content
        assert "Insufficient funds" in resp.content

    def test_transfer_source_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "transfer_funds_between_bank_accounts_7291",
            {
                "source_account_id": "acct_fake",
                "destination_account_id": "chk_001",
                "amount": 100.00,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_transfer_destination_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "transfer_funds_between_bank_accounts_7291",
            {
                "source_account_id": "chk_001",
                "destination_account_id": "acct_fake",
                "amount": 100.00,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_transfer_zero_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "transfer_funds_between_bank_accounts_7291",
            {
                "source_account_id": "chk_001",
                "destination_account_id": "chk_002",
                "amount": 0,
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content


class TestApplyCheckingAccountCredit:
    """Tests for apply_checking_account_credit_5829."""

    def test_apply_rebate_credit(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_001",
                "amount": 15.00,
                "credit_type": "rebate_credit",
            },
        )
        assert not resp.error
        assert "Credit applied" in resp.content
        assert (
            environment.tools.db.accounts.data["chk_001"]["current_holdings"]
            == "$5015.00"
        )

    def test_apply_fee_refund(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_001",
                "amount": 10.00,
                "credit_type": "fee_refund",
            },
        )
        assert not resp.error
        assert "Credit applied" in resp.content

    def test_apply_invalid_credit_type(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_001",
                "amount": 10.00,
                "credit_type": "magic_money",
            },
        )
        assert "Error" in resp.content
        assert "Invalid credit_type" in resp.content

    def test_apply_credit_account_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "acct_nonexistent",
                "amount": 10.00,
                "credit_type": "fee_refund",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_apply_credit_negative_amount(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_001",
                "amount": -5.00,
                "credit_type": "fee_refund",
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content


class TestApplySavingsAccountCredit:
    """Tests for apply_savings_account_credit_6831."""

    def test_apply_interest_correction(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "sav_001",
                "amount": 25.50,
                "credit_type": "interest_correction",
            },
        )
        assert not resp.error
        assert "Credit applied" in resp.content
        assert (
            environment.tools.db.accounts.data["sav_001"]["current_holdings"]
            == "10025.50"
        )

    def test_apply_goodwill_credit(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "sav_001",
                "amount": 50.00,
                "credit_type": "goodwill_credit",
            },
        )
        assert not resp.error

    def test_apply_invalid_type(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "sav_001",
                "amount": 10.00,
                "credit_type": "bonus_credit",
            },
        )
        assert "Error" in resp.content
        assert "Invalid credit_type" in resp.content


# =============================================================================
# 13. Debit Card Lifecycle (Agent Discoverable)
# =============================================================================


class TestOrderDebitCard:
    """Tests for order_debit_card_5739."""

    def test_order_card_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "order_debit_card_5739",
            {
                "account_id": "chk_002",
                "user_id": "user_002",
                "delivery_option": "STANDARD",
                "delivery_fee": 0.00,
                "card_design": "CLASSIC",
                "design_fee": 0.00,
                "shipping_address": "1923 Oak Park Boulevard, Detroit, MI 48226",
            },
        )
        assert not resp.error
        assert "Debit Card Order Confirmed" in resp.content

    def test_order_card_expedited(self, environment: Environment):
        # Deactivate the existing active card so the order can proceed
        environment.tools.db.debit_cards.data["dbc_002"]["status"] = "CLOSED"
        resp = call_discoverable_agent(
            environment,
            "order_debit_card_5739",
            {
                "account_id": "chk_001",
                "user_id": "user_001",
                "delivery_option": "RUSH",
                "delivery_fee": 25.00,
                "card_design": "PREMIUM",
                "design_fee": 10.00,
                "shipping_address": "305 Magnolia Street, Houston, TX 77002",
            },
        )
        assert not resp.error
        assert "Debit Card Order Confirmed" in resp.content
        assert "$35" in resp.content  # Total fees

    def test_order_card_invalid_delivery(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "order_debit_card_5739",
            {
                "account_id": "chk_001",
                "user_id": "user_001",
                "delivery_option": "TELEPORT",
                "delivery_fee": 0.00,
                "card_design": "CLASSIC",
                "design_fee": 0.00,
                "shipping_address": "123 Main St",
            },
        )
        assert "Error" in resp.content
        assert "Invalid delivery_option" in resp.content

    def test_order_card_invalid_design(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "order_debit_card_5739",
            {
                "account_id": "chk_001",
                "user_id": "user_001",
                "delivery_option": "STANDARD",
                "delivery_fee": 0.00,
                "card_design": "DIAMOND",
                "design_fee": 0.00,
                "shipping_address": "123 Main St",
            },
        )
        assert "Error" in resp.content
        assert "Invalid card_design" in resp.content


class TestActivateDebitCard:
    """Tests for debit card activation tools (8291, 8292, 8293)."""

    def test_activate_new_card_success(self, environment: Environment):
        """Activate a new_account card with tool 8291."""
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_001",
                "last_4_digits": "4321",
                "expiration_date": "11/27",
                "cvv": "123",
                "pin": "5823",
            },
        )
        assert not resp.error
        assert "Activation Successful" in resp.content
        assert environment.tools.db.debit_cards.data["dbc_001"]["status"] == "ACTIVE"

    def test_activate_wrong_tool_for_issue_reason(self, environment: Environment):
        """Using new-account tool for a lost card should fail."""
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_003",  # issue_reason = "lost"
                "last_4_digits": "1111",
                "expiration_date": "11/27",
                "cvv": "456",
                "pin": "7890",
            },
        )
        assert "Error" in resp.content
        assert "Wrong activation tool" in resp.content

    def test_activate_replacement_card_success(self, environment: Environment):
        """Activate a lost card with tool 8292."""
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8292",
            {
                "card_id": "dbc_003",
                "last_4_digits": "1111",
                "expiration_date": "11/27",
                "cvv": "789",
                "pin": "7290",
            },
        )
        assert not resp.error
        assert "Activation Successful" in resp.content

    def test_activate_reissued_card_success(self, environment: Environment):
        """Activate an expired card with tool 8293."""
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8293",
            {
                "card_id": "dbc_expired",
                "last_4_digits": "5555",
                "expiration_date": "11/27",
                "cvv": "321",
                "pin": "9876",
            },
        )
        # pin 9876 is sequential descending, should fail
        assert "Error" in resp.content
        assert "sequential" in resp.content.lower()

    def test_activate_with_valid_pin_reissued(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8293",
            {
                "card_id": "dbc_expired",
                "last_4_digits": "5555",
                "expiration_date": "11/27",
                "cvv": "321",
                "pin": "5823",
            },
        )
        assert not resp.error
        assert "Activation Successful" in resp.content

    def test_activate_already_active(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_002",  # Already ACTIVE
                "last_4_digits": "8765",
                "expiration_date": "12/26",
                "cvv": "456",
                "pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "already active" in resp.content

    def test_activate_wrong_last_4(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_001",
                "last_4_digits": "0000",  # Wrong digits
                "expiration_date": "11/27",
                "cvv": "123",
                "pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "do not match" in resp.content

    def test_activate_sequential_pin_rejected(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_001",
                "last_4_digits": "4321",
                "expiration_date": "11/27",
                "cvv": "123",
                "pin": "1234",  # Sequential
            },
        )
        assert "Error" in resp.content
        assert "sequential" in resp.content.lower()

    def test_activate_repeating_pin_rejected(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_001",
                "last_4_digits": "4321",
                "expiration_date": "11/27",
                "cvv": "123",
                "pin": "5555",  # All same
            },
        )
        assert "Error" in resp.content
        assert "same digit" in resp.content.lower()

    def test_activate_invalid_cvv(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_001",
                "last_4_digits": "4321",
                "expiration_date": "11/27",
                "cvv": "12",  # Too short
                "pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "CVV" in resp.content

    def test_activate_nonexistent_card(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "activate_debit_card_8291",
            {
                "card_id": "dbc_nonexistent",
                "last_4_digits": "0000",
                "expiration_date": "11/27",
                "cvv": "123",
                "pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestCloseDebitCard:
    """Tests for close_debit_card_4721."""

    def test_close_card_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_debit_card_4721",
            {
                "card_id": "dbc_002",
                "reason": "no_longer_needed",
            },
        )
        assert not resp.error
        assert "Debit Card Closed Successfully" in resp.content
        assert environment.tools.db.debit_cards.data["dbc_002"]["status"] == "CLOSED"

    def test_close_card_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_debit_card_4721",
            {
                "card_id": "dbc_002",
                "reason": "just_because",
            },
        )
        assert "Error" in resp.content
        assert "Invalid reason" in resp.content

    def test_close_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "close_debit_card_4721",
            {
                "card_id": "dbc_nonexistent",
                "reason": "lost",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestFreezeUnfreezeDebitCard:
    """Tests for freeze_debit_card_3892 and unfreeze_debit_card_3893."""

    def test_freeze_card_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "freeze_debit_card_3892",
            {
                "card_id": "dbc_002",
            },
        )
        assert not resp.error
        assert "Debit Card Frozen Successfully" in resp.content
        assert environment.tools.db.debit_cards.data["dbc_002"]["status"] == "FROZEN"

    def test_freeze_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "freeze_debit_card_3892",
            {
                "card_id": "dbc_nonexistent",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_unfreeze_card_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "unfreeze_debit_card_3893",
            {
                "card_id": "dbc_frozen",
            },
        )
        assert not resp.error
        assert "Debit Card Unfrozen Successfully" in resp.content
        assert environment.tools.db.debit_cards.data["dbc_frozen"]["status"] == "ACTIVE"

    def test_unfreeze_card_not_frozen(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "unfreeze_debit_card_3893",
            {
                "card_id": "dbc_002",  # ACTIVE, not FROZEN
            },
        )
        assert "Error" in resp.content
        assert "already active" in resp.content

    def test_unfreeze_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "unfreeze_debit_card_3893",
            {
                "card_id": "dbc_nonexistent",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestClearDebitCardFraudAlert:
    """Tests for clear_debit_card_fraud_alert_4892."""

    def test_clear_alert_success(self, environment: Environment):
        # Set up the card with an active fraud alert
        environment.tools.db.debit_cards.data["dbc_002"]["fraud_alert_active"] = True
        environment.tools.db.debit_cards.data["dbc_002"]["fraud_alert_source"] = (
            "customer"
        )
        resp = call_discoverable_agent(
            environment,
            "clear_debit_card_fraud_alert_4892",
            {
                "card_id": "dbc_002",
                "reason": "customer_verified",
            },
        )
        assert not resp.error
        assert "Fraud Alert Cleared" in resp.content

    def test_clear_alert_invalid_reason(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "clear_debit_card_fraud_alert_4892",
            {
                "card_id": "dbc_002",
                "reason": "felt_like_it",
            },
        )
        assert "Error" in resp.content
        assert "Invalid reason" in resp.content

    def test_clear_alert_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "clear_debit_card_fraud_alert_4892",
            {
                "card_id": "dbc_nonexistent",
                "reason": "customer_verified",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestDebitCardPin:
    """Tests for reset_debit_card_pin_6284 and change_debit_card_pin_6285."""

    def test_reset_pin_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "reset_debit_card_pin_6284",
            {
                "card_id": "dbc_002",
                "last_4_digits": "8765",
                "new_pin": "5823",
            },
        )
        assert not resp.error
        assert "PIN Reset Successfully" in resp.content

    def test_reset_pin_wrong_last4(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "reset_debit_card_pin_6284",
            {
                "card_id": "dbc_002",
                "last_4_digits": "0000",
                "new_pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "do not match" in resp.content

    def test_reset_pin_sequential(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "reset_debit_card_pin_6284",
            {
                "card_id": "dbc_002",
                "last_4_digits": "8765",
                "new_pin": "1234",
            },
        )
        assert "Error" in resp.content
        assert "sequential" in resp.content.lower()

    def test_reset_pin_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "reset_debit_card_pin_6284",
            {
                "card_id": "dbc_nonexistent",
                "last_4_digits": "0000",
                "new_pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_change_pin_success(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "change_debit_card_pin_6285",
            {
                "card_id": "dbc_002",
                "current_pin": "9999",
                "new_pin": "5823",
            },
        )
        assert not resp.error
        assert "PIN Changed Successfully" in resp.content

    def test_change_pin_invalid_new_pin(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "change_debit_card_pin_6285",
            {
                "card_id": "dbc_002",
                "current_pin": "9999",
                "new_pin": "0000",  # All same digits
            },
        )
        assert "Error" in resp.content
        assert "same digit" in resp.content.lower()

    def test_change_pin_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "change_debit_card_pin_6285",
            {
                "card_id": "dbc_nonexistent",
                "current_pin": "1234",
                "new_pin": "5823",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


class TestRequestTemporaryLimitIncrease:
    """Tests for request_temporary_debit_card_limit_increase_8374."""

    def test_request_atm_increase(self, environment: Environment):
        # Set up ATM limit on the card so the tool can process it
        environment.tools.db.debit_cards.data["dbc_002"]["daily_atm_limit"] = 500
        resp = call_discoverable_agent(
            environment,
            "request_temporary_debit_card_limit_increase_8374",
            {
                "card_id": "dbc_002",
                "limit_type": "atm",
                "new_limit": 700,  # Within 150% of $500 = $750 max
            },
        )
        assert not resp.error
        assert "Increase Granted Successfully" in resp.content
        assert "24 hours" in resp.content

    def test_request_purchase_increase(self, environment: Environment):
        # Set up purchase limit on the card so the tool can process it
        environment.tools.db.debit_cards.data["dbc_002"]["daily_purchase_limit"] = 5000
        resp = call_discoverable_agent(
            environment,
            "request_temporary_debit_card_limit_increase_8374",
            {
                "card_id": "dbc_002",
                "limit_type": "purchase",
                "new_limit": 7000,  # Within 150% of $5000 = $7500 max
            },
        )
        assert not resp.error
        assert "Increase Granted Successfully" in resp.content

    def test_request_invalid_limit_type(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "request_temporary_debit_card_limit_increase_8374",
            {
                "card_id": "dbc_002",
                "limit_type": "international",
                "new_limit": 1000,
            },
        )
        assert "Error" in resp.content
        assert "Invalid limit_type" in resp.content

    def test_request_card_not_found(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "request_temporary_debit_card_limit_increase_8374",
            {
                "card_id": "dbc_nonexistent",
                "limit_type": "atm",
                "new_limit": 500,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content


# =============================================================================
# 14. User Discoverable Tools
# =============================================================================


class TestSubmitCashBackDispute:
    """Tests for submit_cash_back_dispute_0589 (user discoverable)."""

    def test_submit_auto_resolve(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "submit_cash_back_dispute_0589",
            {
                "user_id": "user_001",
                "transaction_id": "txn_001",
            },
        )
        assert not resp.error
        assert "submitted successfully" in resp.content
        assert "RESOLVED" in resp.content
        assert len(environment.tools.db.cash_back_disputes.data) == 1
        dispute = list(environment.tools.db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "RESOLVED"

    def test_submit_no_auto_resolve(self, env_no_auto_resolve: Environment):
        resp = call_discoverable_user(
            env_no_auto_resolve,
            "submit_cash_back_dispute_0589",
            {
                "user_id": "user_001",
                "transaction_id": "txn_001",
            },
        )
        assert not resp.error
        assert "SUBMITTED" in resp.content
        dispute = list(env_no_auto_resolve.tools.db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "SUBMITTED"
        assert "resolution" not in dispute


class TestGetReferralLink:
    """Tests for get_referral_link (user discoverable)."""

    def test_get_referral_link_success(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "get_referral_link",
            {
                "user_id": "user_001",
                "card_name": "Gold Rewards Card",
            },
        )
        assert not resp.error
        assert "Referral link generated successfully" in resp.content
        assert "rhobank.com/refer/" in resp.content

    def test_get_referral_link_without_giving(self, environment: Environment):
        resp = call(
            environment,
            "call_discoverable_user_tool",
            {
                "discoverable_tool_name": "get_referral_link",
                "arguments": json.dumps(
                    {"user_id": "user_001", "card_name": "Gold Rewards Card"}
                ),
            },
            requestor="user",
        )
        assert "Error" in resp.content
        assert "has not been given" in resp.content


class TestGetCardLast4Digits:
    """Tests for get_card_last_4_digits (user discoverable)."""

    def test_get_card_digits_success(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "get_card_last_4_digits",
            {
                "credit_card_account_id": "cc_001",
            },
        )
        assert not resp.error
        assert "Card information retrieved" in resp.content
        assert "Last 4 digits" in resp.content

    def test_get_card_digits_not_found(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "get_card_last_4_digits",
            {
                "credit_card_account_id": "cc_nonexistent",
            },
        )
        assert "Error" in resp.content or "not found" in resp.content


class TestDepositCheck:
    """Tests for deposit_check_3847 (user discoverable)."""

    def test_deposit_check_success(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "deposit_check_3847",
            {
                "account_id": "chk_001",
                "check_amount": 500.00,
            },
        )
        assert not resp.error
        assert "deposited successfully" in resp.content
        assert "$500.00" in resp.content

    def test_deposit_check_account_not_found(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "deposit_check_3847",
            {
                "account_id": "acct_nonexistent",
                "check_amount": 100.00,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content

    def test_deposit_check_negative_amount(self, environment: Environment):
        resp = call_discoverable_user(
            environment,
            "deposit_check_3847",
            {
                "account_id": "chk_001",
                "check_amount": -100.00,
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content


# =============================================================================
# 15. User Tools (Non-discoverable)
# =============================================================================


class TestApplyForCreditCard:
    """Tests for apply_for_credit_card (user tool)."""

    def test_apply_success(self, environment: Environment):
        resp = call(
            environment,
            "apply_for_credit_card",
            {
                "card_type": "Gold Rewards Card",
                "customer_name": "Test User",
                "annual_income": 75000.00,
                "rho_bank_subscription": False,
            },
            requestor="user",
        )
        assert not resp.error
        assert "application submitted" in resp.content.lower()
        assert len(environment.user_tools.db.credit_card_applications.data) == 1

    def test_apply_with_subscription(self, environment: Environment):
        resp = call(
            environment,
            "apply_for_credit_card",
            {
                "card_type": "Platinum Rewards Card",
                "customer_name": "Premium User",
                "annual_income": 150000.00,
                "rho_bank_subscription": True,
            },
            requestor="user",
        )
        assert not resp.error
        assert "submitted" in resp.content.lower()

    def test_apply_invalid_card_type(self, environment: Environment):
        resp = call(
            environment,
            "apply_for_credit_card",
            {
                "card_type": "EcoCard (Sustainable Rewards Credit Card)",
                "customer_name": "Test User",
                "annual_income": 75000.00,
                "rho_bank_subscription": False,
            },
            requestor="user",
        )
        assert "Error" in resp.content
        assert "Invalid card_type" in resp.content
        assert "EcoCard" in resp.content
        assert len(environment.user_tools.db.credit_card_applications.data) == 0


class TestSubmitReferral:
    """Tests for submit_referral (user tool)."""

    def test_submit_referral_success(self, environment: Environment):
        resp = call(
            environment,
            "submit_referral",
            {
                "user_id": "user_001",
                "account_type": "Savings Account",
            },
            requestor="user",
        )
        assert not resp.error
        assert "Referral request submitted successfully" in resp.content
        assert "NO_PROGRESS" in resp.content


class TestRequestHumanAgentTransfer:
    """Tests for request_human_agent_transfer (user tool)."""

    def test_request_transfer_success(self, environment: Environment):
        resp = call(environment, "request_human_agent_transfer", {}, requestor="user")
        assert not resp.error
        assert "Transfer request" in resp.content
        assert len(environment.user_tools.db.human_transfer_requests.data) == 1

    def test_multiple_transfer_requests(self, environment: Environment):
        call(environment, "request_human_agent_transfer", {}, requestor="user")
        resp = call(environment, "request_human_agent_transfer", {}, requestor="user")
        assert not resp.error
        assert len(environment.user_tools.db.human_transfer_requests.data) == 2


# =============================================================================
# 16. Submit Transaction & Rewards Calculation
# =============================================================================


class TestSubmitTransaction:
    """Tests for submit_transaction and rewards calculation logic."""

    def test_gold_card_default_rate(self, environment: Environment):
        """Gold Rewards Card: 2.5% on all purchases."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Gold Rewards Card",
                "merchant_name": "Target",
                "amount": 100.00,
                "category": "Shopping",
            },
            requestor="user",
        )
        assert not resp.error
        # 100 * 2.5 = 250 points
        assert "250 points" in resp.content
        assert "2.5%" in resp.content

    def test_silver_card_travel_rate(self, environment: Environment):
        """Silver Rewards Card: 4% on travel."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Silver Rewards Card",
                "merchant_name": "United Airlines",
                "amount": 200.00,
                "category": "Travel",
            },
            requestor="user",
        )
        assert not resp.error
        # 200 * 4.0 = 800 points
        assert "800 points" in resp.content
        assert "4.0%" in resp.content

    def test_silver_card_default_rate(self, environment: Environment):
        """Silver Rewards Card: 1% on non-travel/software."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Silver Rewards Card",
                "merchant_name": "Grocery Store",
                "amount": 50.00,
                "category": "Groceries",
            },
            requestor="user",
        )
        assert not resp.error
        # 50 * 1.0 = 50 points
        assert "50 points" in resp.content

    def test_business_silver_travel_rate(self, environment: Environment):
        """Business Silver Rewards Card: 10% on travel."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Business Silver Rewards Card",
                "merchant_name": "Delta Airlines",
                "amount": 315.00,
                "category": "Travel",
            },
            requestor="user",
        )
        assert not resp.error
        # 315 * 10.0 = 3150 points
        assert "3150 points" in resp.content

    def test_platinum_card_rate(self, environment: Environment):
        """Platinum Rewards Card: 10% on all purchases."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Platinum Rewards Card",
                "merchant_name": "Luxury Store",
                "amount": 1000.00,
                "category": "Shopping",
            },
            requestor="user",
        )
        assert not resp.error
        # 1000 * 10.0 = 10000 points
        assert "10000 points" in resp.content

    def test_crypto_cash_back_rate(self, environment: Environment):
        """Crypto-Cash Back: 2% on all purchases."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_002",
                "credit_card_type": "Crypto-Cash Back",
                "merchant_name": "Trader Joe's",
                "amount": 47.83,
                "category": "Groceries",
            },
            requestor="user",
        )
        assert not resp.error
        # int(47.83 * 2.0) = 95 points
        assert "95 points" in resp.content

    def test_bronze_card_rate(self, environment: Environment):
        """Bronze Rewards Card: 1% on all purchases."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Bronze Rewards Card",
                "merchant_name": "Coffee Shop",
                "amount": 5.00,
                "category": "Dining",
            },
            requestor="user",
        )
        assert not resp.error
        # int(5.0 * 1.0) = 5 points
        assert "5 points" in resp.content

    def test_diamond_elite_card_rate(self, environment: Environment):
        """Diamond Elite Card: 5% on all purchases."""
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Diamond Elite Card",
                "merchant_name": "High End Store",
                "amount": 500.00,
                "category": "Shopping",
            },
            requestor="user",
        )
        assert not resp.error
        # 500 * 5.0 = 2500 points
        assert "2500 points" in resp.content

    def test_unknown_card_type(self, environment: Environment):
        resp = call(
            environment,
            "submit_transaction",
            {
                "user_id": "user_001",
                "credit_card_type": "Nonexistent Card",
                "merchant_name": "Store",
                "amount": 100.00,
                "category": "Shopping",
            },
            requestor="user",
        )
        assert "Error" in resp.content
        assert "Unknown credit card type" in resp.content


# =============================================================================
# 17. db_query Module Operations
# =============================================================================


class TestDbQuery:
    """Tests for the db_query module functions."""

    def test_list_databases(self, base_knowledge_db: TransactionalDB):
        dbs = list_databases(base_knowledge_db)
        assert "users" in dbs
        assert "accounts" in dbs
        assert "credit_card_accounts" in dbs
        assert "referrals" in dbs

    def test_query_db_exact_match(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, name="Amara Okonkwo")
        assert len(results) == 1
        assert results[0]["user_id"] == "user_001"

    def test_query_db_no_match(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, name="Nobody")
        assert len(results) == 0

    def test_query_db_return_ids(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, return_ids=True)
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert results[0][0] in ("user_001", "user_002")

    def test_query_db_with_limit(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, limit=1)
        assert len(results) == 1

    def test_query_db_ne_operator(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, user_id__ne="user_001")
        assert len(results) == 1
        assert results[0]["user_id"] == "user_002"

    def test_query_db_contains_operator(self, base_knowledge_db: TransactionalDB):
        results = query_db("users", db=base_knowledge_db, name__contains="Amara")
        assert len(results) == 1

    def test_query_db_nonexistent_database(self, base_knowledge_db: TransactionalDB):
        results = query_db("nonexistent_db", db=base_knowledge_db)
        assert len(results) == 0

    def test_add_to_db_success(self, base_knowledge_db: TransactionalDB):
        success = add_to_db(
            "users",
            "user_003",
            {
                "name": "New User",
                "user_id": "user_003",
            },
            db=base_knowledge_db,
        )
        assert success
        assert "user_003" in base_knowledge_db.users.data

    def test_add_to_db_duplicate(self, base_knowledge_db: TransactionalDB):
        success = add_to_db(
            "users",
            "user_001",
            {
                "name": "Duplicate",
            },
            db=base_knowledge_db,
        )
        assert not success

    def test_add_to_db_nonexistent_table(self, base_knowledge_db: TransactionalDB):
        success = add_to_db(
            "nonexistent_table", "id1", {"data": "value"}, db=base_knowledge_db
        )
        assert not success

    def test_update_record_success(self, base_knowledge_db: TransactionalDB):
        success, record = update_record_in_db(
            "users",
            db=base_knowledge_db,
            record_id="user_001",
            updates={"phone_number": "555-NEW-NUM"},
        )
        assert success
        assert record["phone_number"] == "555-NEW-NUM"

    def test_update_record_not_found(self, base_knowledge_db: TransactionalDB):
        success, record = update_record_in_db(
            "users",
            db=base_knowledge_db,
            record_id="nonexistent",
            updates={"name": "New"},
        )
        assert not success
        assert record is None

    def test_remove_from_db_success(self, base_knowledge_db: TransactionalDB):
        removed = remove_from_db("users", db=base_knowledge_db, user_id="user_002")
        assert len(removed) == 1
        assert removed[0]["name"] == "Fatima Al-Hassan"
        assert "user_002" not in base_knowledge_db.users.data

    def test_remove_from_db_no_match(self, base_knowledge_db: TransactionalDB):
        removed = remove_from_db("users", db=base_knowledge_db, user_id="nonexistent")
        assert len(removed) == 0

    def test_query_database_tool_formatted(self, base_knowledge_db: TransactionalDB):
        result = query_database_tool(
            "users", '{"user_id": "user_001"}', db=base_knowledge_db
        )
        assert "Found 1 record" in result
        assert "Amara Okonkwo" in result

    def test_query_database_tool_invalid_db(self, base_knowledge_db: TransactionalDB):
        result = query_database_tool("fake_db", "{}", db=base_knowledge_db)
        assert "Error" in result
        assert "not found" in result

    def test_query_database_tool_invalid_json(self, base_knowledge_db: TransactionalDB):
        result = query_database_tool("users", "not json {{", db=base_knowledge_db)
        assert "Error" in result
        assert "Invalid JSON" in result


# =============================================================================
# 18. Complete Dispute Flow Integration Tests
# =============================================================================


class TestCompleteDisputeFlow:
    """Integration tests for multi-step dispute flows."""

    def test_full_dispute_flow_auto_resolve(self, environment: Environment):
        """Test: verify user -> give dispute tool -> user submits -> unlock update tool -> update rewards."""
        db = environment.tools.db

        # Step 1: Log verification
        resp = call(
            environment,
            "log_verification",
            {
                "name": "Amara Okonkwo",
                "user_id": "user_001",
                "address": "305 Magnolia Street, Houston, TX 77002",
                "email": "amara@icloud.com",
                "phone_number": "713-555-0963",
                "date_of_birth": "08/11/1997",
                "time_verified": "2025-11-14 03:40:00 EST",
            },
        )
        assert not resp.error
        assert len(db.verification_history.data) == 1

        # Step 2: Agent gives dispute tool to user
        resp = call(
            environment,
            "give_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
            },
        )
        assert not resp.error
        assert len(db.user_discoverable_tools.data) == 1

        # Step 3: User submits dispute (auto-resolved)
        resp = call(
            environment,
            "call_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "user_001", "transaction_id": "txn_001"}
                ),
            },
            requestor="user",
        )
        assert not resp.error
        assert "RESOLVED" in resp.content
        assert len(db.cash_back_disputes.data) == 1

        # Step 4: Agent unlocks update tool
        resp = call(
            environment,
            "unlock_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
            },
        )
        assert not resp.error

        # Step 5: Agent updates rewards
        resp = call(
            environment,
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": "update_transaction_rewards_3847",
                "arguments": json.dumps(
                    {
                        "transaction_id": "txn_001",
                        "new_rewards_earned": "500 points",
                    }
                ),
            },
        )
        assert not resp.error
        assert (
            db.credit_card_transaction_history.data["txn_001"]["rewards_earned"]
            == "500 points"
        )
        assert len(db.agent_discoverable_tools.data) == 1

    def test_adversarial_dispute_flow_no_auto_resolve(
        self, env_no_auto_resolve: Environment
    ):
        """Test: disputes stay SUBMITTED when auto_resolve is disabled."""
        db = env_no_auto_resolve.tools.db

        # Give and submit dispute
        call(
            env_no_auto_resolve,
            "give_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
            },
        )
        resp = call(
            env_no_auto_resolve,
            "call_discoverable_user_tool",
            {
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "user_001", "transaction_id": "txn_001"}
                ),
            },
            requestor="user",
        )
        assert "SUBMITTED" in resp.content

        # Verify dispute is NOT resolved
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "SUBMITTED"
        assert "resolution" not in dispute


# =============================================================================
# 19. Read-only Discoverable Agent Tool Tests
# =============================================================================


class TestReadOnlyDiscoverableAgentTools:
    """Tests for read-only discoverable agent tools."""

    def test_get_all_user_accounts(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_all_user_accounts_by_user_id_3847",
            {
                "user_id": "user_001",
            },
        )
        assert not resp.error
        assert "accounts retrieved" in resp.content.lower()

    def test_get_all_user_accounts_empty(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_all_user_accounts_by_user_id_3847",
            {
                "user_id": "nonexistent_user",
            },
        )
        assert not resp.error

    def test_get_bank_account_transactions(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_bank_account_transactions_9173",
            {
                "account_id": "chk_001",
            },
        )
        assert not resp.error
        assert "transactions" in resp.content.lower()

    def test_get_debit_cards_by_account(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_debit_cards_by_account_id_7823",
            {
                "account_id": "chk_001",
            },
        )
        assert not resp.error
        assert "card_id" in resp.content

    def test_get_user_dispute_history(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_user_dispute_history_7291",
            {
                "user_id": "user_001",
            },
        )
        assert not resp.error

    def test_get_debit_dispute_status(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_debit_dispute_status_7483",
            {
                "user_id": "user_001",
            },
        )
        assert not resp.error

    def test_get_closure_reason_history(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_closure_reason_history_8293",
            {
                "credit_card_account_id": "cc_001",
            },
        )
        assert not resp.error

    def test_get_credit_limit_increase_history(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_credit_limit_increase_history_4829",
            {
                "credit_card_account_id": "cc_001",
            },
        )
        assert not resp.error

    def test_get_payment_history(self, environment: Environment):
        resp = call_discoverable_agent(
            environment,
            "get_payment_history_6183",
            {
                "credit_card_account_id": "cc_001",
                "months": 6,
            },
        )
        assert not resp.error


# =============================================================================
# 18. Tau-knowledge Parity Tests
#
# These tests verify behavior that exists in the tau-knowledge branch.
# They catch missing validation checks and logic that should have been
# ported from tau-knowledge to dev/tau3.
# =============================================================================


class TestApplyCheckingAccountCreditParity:
    """Verify apply_checking_account_credit_5829 matches tau-knowledge checks."""

    def test_credit_to_savings_account_rejected(self, environment: Environment):
        """tau-knowledge rejects credits to non-checking accounts."""
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "sav_001",  # savings account
                "amount": 100.0,
                "credit_type": "rebate_credit",
            },
        )
        assert "Error" in resp.content
        assert "checking" in resp.content.lower()

    def test_credit_to_closed_account_rejected(self, environment: Environment):
        """tau-knowledge rejects credits to non-active accounts."""
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_closed",
                "amount": 50.0,
                "credit_type": "fee_refund",
            },
        )
        assert "Error" in resp.content
        assert "not active" in resp.content.lower() or "CLOSED" in resp.content

    def test_credit_creates_transaction_record(self, environment: Environment):
        """tau-knowledge creates a transaction record in bank_account_transaction_history."""
        resp = call_discoverable_agent(
            environment,
            "apply_checking_account_credit_5829",
            {
                "account_id": "chk_001",
                "amount": 100.0,
                "credit_type": "rebate_credit",
            },
        )
        assert not resp.error
        # Verify transaction was recorded
        assert len(environment.tools.db.bank_account_transaction_history.data) > 0


class TestApplySavingsAccountCreditParity:
    """Verify apply_savings_account_credit_6831 matches tau-knowledge checks."""

    def test_credit_to_checking_account_rejected(self, environment: Environment):
        """tau-knowledge rejects credits to non-savings accounts."""
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "chk_001",  # checking account
                "amount": 100.0,
                "credit_type": "fee_refund",
            },
        )
        assert "Error" in resp.content
        assert "savings" in resp.content.lower()

    def test_credit_to_closed_savings_rejected(self, environment: Environment):
        """tau-knowledge rejects credits to non-active savings accounts."""
        # Change sav_001 status to CLOSED for this test
        environment.tools.db.accounts.data["sav_001"]["status"] = "CLOSED"
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "sav_001",
                "amount": 50.0,
                "credit_type": "fee_refund",
            },
        )
        assert "Error" in resp.content

    def test_credit_creates_transaction_record(self, environment: Environment):
        """tau-knowledge creates a transaction record in bank_account_transaction_history."""
        resp = call_discoverable_agent(
            environment,
            "apply_savings_account_credit_6831",
            {
                "account_id": "sav_001",
                "amount": 100.0,
                "credit_type": "fee_refund",
            },
        )
        assert not resp.error
        assert len(environment.tools.db.bank_account_transaction_history.data) > 0


class TestSetDebitCardRecurringBlockParity:
    """Verify set_debit_card_recurring_block_7382 matches tau-knowledge checks."""

    def test_block_on_non_active_card_rejected(self, environment: Environment):
        """tau-knowledge rejects recurring block changes on non-ACTIVE cards."""
        resp = call_discoverable_agent(
            environment,
            "set_debit_card_recurring_block_7382",
            {
                "card_id": "dbc_001",  # status is PENDING
                "block_recurring": True,
            },
        )
        assert "Error" in resp.content
        assert "ACTIVE" in resp.content or "status" in resp.content.lower()


class TestGetDebitCardsByAccountIdParity:
    """Verify get_debit_cards_by_account_id_7823 matches tau-knowledge checks."""

    def test_account_not_found(self, environment: Environment):
        """tau-knowledge returns error for non-existent account."""
        resp = call_discoverable_agent(
            environment,
            "get_debit_cards_by_account_id_7823",
            {"account_id": "acct_nonexistent"},
        )
        assert "Error" in resp.content or "not found" in resp.content.lower()

    def test_savings_account_rejected(self, environment: Environment):
        """tau-knowledge rejects queries for non-checking accounts."""
        resp = call_discoverable_agent(
            environment,
            "get_debit_cards_by_account_id_7823",
            {"account_id": "sav_001"},
        )
        assert "Error" in resp.content
        assert "checking" in resp.content.lower()


class TestSubmitInterestDiscrepancyReportParity:
    """Verify submit_interest_discrepancy_report_7294 matches tau-knowledge checks."""

    def test_account_not_found(self, environment: Environment):
        """tau-knowledge checks account existence."""
        resp = call_discoverable_agent(
            environment,
            "submit_interest_discrepancy_report_7294",
            {
                "account_id": "acct_nonexistent",
                "user_id": "user_001",
                "expected_apy": 2.5,
                "actual_apy": 1.5,
                "amount_difference": 50.0,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content.lower()

    def test_user_not_found(self, environment: Environment):
        """tau-knowledge checks user existence."""
        resp = call_discoverable_agent(
            environment,
            "submit_interest_discrepancy_report_7294",
            {
                "account_id": "sav_001",
                "user_id": "user_nonexistent",
                "expected_apy": 2.5,
                "actual_apy": 1.5,
                "amount_difference": 50.0,
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content.lower()


class TestGetPaymentHistoryParity:
    """Verify get_payment_history_6183 matches tau-knowledge checks."""

    def test_zero_months_rejected(self, environment: Environment):
        """tau-knowledge rejects months <= 0."""
        resp = call_discoverable_agent(
            environment,
            "get_payment_history_6183",
            {
                "credit_card_account_id": "cc_001",
                "months": 0,
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content.lower() or "must be" in resp.content.lower()

    def test_negative_months_rejected(self, environment: Environment):
        """tau-knowledge rejects negative months."""
        resp = call_discoverable_agent(
            environment,
            "get_payment_history_6183",
            {
                "credit_card_account_id": "cc_001",
                "months": -3,
            },
        )
        assert "Error" in resp.content


class TestFileDebitCardDisputeParity:
    """Verify file_debit_card_transaction_dispute_6281 matches tau-knowledge checks."""

    def test_invalid_dispute_category_rejected(self, environment: Environment):
        """tau-knowledge validates dispute_category."""
        # First add a debit card transaction
        environment.tools.db.bank_account_transaction_history.data["dtxn_001"] = {
            "transaction_id": "dtxn_001",
            "account_id": "chk_001",
            "date": "11/01/2025",
            "description": "ATM Withdrawal",
            "amount": -200.00,
            "type": "atm_withdrawal",
        }
        resp = call_discoverable_agent(
            environment,
            "file_debit_card_transaction_dispute_6281",
            {
                "transaction_id": "dtxn_001",
                "account_id": "chk_001",
                "card_id": "dbc_002",
                "user_id": "user_001",
                "dispute_category": "INVALID_CATEGORY",
                "transaction_type": "atm_withdrawal",
                "disputed_amount": 200.00,
                "customer_max_liability_amount": 50.00,
                "card_in_possession": True,
                "contacted_merchant": False,
                "police_report_filed": False,
                "written_statement_provided": True,
                "pin_compromised": "no",
                "card_action": "freeze",
                "description": "Unauthorized ATM withdrawal",
            },
        )
        assert "Error" in resp.content
        assert "dispute_category" in resp.content.lower() or "Invalid" in resp.content

    def test_negative_disputed_amount_rejected(self, environment: Environment):
        """tau-knowledge rejects non-positive disputed amounts."""
        resp = call_discoverable_agent(
            environment,
            "file_debit_card_transaction_dispute_6281",
            {
                "transaction_id": "dtxn_001",
                "account_id": "chk_001",
                "card_id": "dbc_002",
                "user_id": "user_001",
                "dispute_category": "unauthorized_transaction",
                "transaction_date": "11/01/2025",
                "discovery_date": "11/05/2025",
                "transaction_type": "pin_purchase",
                "disputed_amount": -50.00,
                "customer_max_liability_amount": 0.00,
                "card_in_possession": True,
                "contacted_merchant": False,
                "police_report_filed": False,
                "written_statement_provided": True,
                "provisional_credit_eligible": False,
                "pin_compromised": "no",
                "card_action": "keep_active",
            },
        )
        assert "Error" in resp.content
        assert "positive" in resp.content.lower() or "amount" in resp.content.lower()


class TestApplyCreditCardAccountFlagParity:
    """Verify apply_credit_card_account_flag_6147 matches tau-knowledge checks."""

    def test_nonexistent_cc_account_rejected(self, environment: Environment):
        """tau-knowledge checks CC account existence."""
        resp = call_discoverable_agent(
            environment,
            "apply_credit_card_account_flag_6147",
            {
                "credit_card_account_id": "cc_nonexistent",
                "user_id": "user_001",
                "flag_type": "annual_fee_waived",
                "expiration_date": "12/31/2025",
                "reason": "retention_offer",
            },
        )
        assert "Error" in resp.content
        assert "not found" in resp.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
