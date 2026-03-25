"""Tests for task_028.

task_028: Full end-to-end cash back dispute and resolution flow.
- User: Fatima Al-Hassan (user_id: 890389b165)
- Scenario: Agent investigates discrepancies, helps user submit disputes,
  then after disputes are resolved (auto-resolved), agent updates transaction
  rewards to correct values.
- Cards: Crypto-Cash Back, Business Platinum Rewards Card, EcoCard, Silver Rewards Card
- Transactions with incorrect rewards (6 total):
  - txn_57ecc6da56c2: Trader Joe's $47.83 -> should be 95 points (2%)
  - txn_d80aef98f532: United Airlines $347.62 -> should be 1390 points (4%)
  - txn_896ac64b98d7: Patagonia $128.47 -> should be 642 points (5% eco)
  - txn_adea68821a1d: Marriott Hotels $167.34 -> should be 669 points (4%)
  - txn_0be1ccc37761: LinkedIn Ads $512.47 -> should be 2049 points (4%)
  - txn_e647e242ce96: Google Ads $187.56 -> should be 750 points (4%)
"""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment

from .conftest import create_environment, create_fatima_db


@pytest.fixture
def environment() -> Environment:
    """Environment with Fatima's data and auto_resolve=True."""
    db = create_fatima_db(auto_resolve=True)
    return create_environment(db)


class TestTask028:
    """Tests for task_028 - Full dispute flow with auto-resolve."""

    def test_verify_user_fatima(self, environment: Environment):
        """Test that agent can verify Fatima Al-Hassan's identity."""
        call = ToolCall(
            id="1",
            name="log_verification",
            arguments={
                "name": "Fatima Al-Hassan",
                "user_id": "890389b165",
                "address": "1923 Oak Park Boulevard, Detroit, MI 48226",
                "email": "coffeelover_fati@protonmail.com",
                "phone_number": "313-555-0246",
                "date_of_birth": "12/05/1993",
                "time_verified": "2025-11-25 10:30:00 EST",
            },
        )
        response = environment.get_response(call)
        assert not response.error
        assert "Verification logged successfully" in response.content
        assert "Fatima Al-Hassan" in response.content

        # Verify logged in database
        db = environment.tools.db
        assert len(db.verification_history.data) == 1

    def test_give_dispute_tool_to_user(self, environment: Environment):
        """Test that agent gives submit_cash_back_dispute_0589 tool to user."""
        call = ToolCall(
            id="1",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        response = environment.get_response(call)
        assert not response.error
        assert "submit_cash_back_dispute_0589" in response.content
        assert "Tool given to user" in response.content

        # Verify recorded in database
        db = environment.tools.db
        assert len(db.user_discoverable_tools.data) == 1

    def test_user_submits_dispute_auto_resolved(self, environment: Environment):
        """Test that user submits dispute and it's auto-resolved."""
        # First, agent gives the tool
        give_call = ToolCall(
            id="1",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        # User submits dispute for United Airlines transaction
        submit_call = ToolCall(
            id="2",
            name="call_discoverable_user_tool",
            arguments={
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "890389b165", "transaction_id": "txn_d80aef98f532"}
                ),
            },
            requestor="user",
        )
        response = environment.get_response(submit_call)
        assert not response.error
        assert "RESOLVED" in response.content
        assert "approved" in response.content.lower()

        # Verify dispute in database with RESOLVED status
        db = environment.tools.db
        assert len(db.cash_back_disputes.data) == 1
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "RESOLVED"
        assert dispute["resolution"] == "APPROVED"
        assert dispute["transaction_id"] == "txn_d80aef98f532"

    def test_agent_unlocks_update_rewards_tool(self, environment: Environment):
        """Test that agent unlocks update_transaction_rewards_3847 tool."""
        call = ToolCall(
            id="1",
            name="unlock_discoverable_agent_tool",
            arguments={"agent_tool_name": "update_transaction_rewards_3847"},
        )
        response = environment.get_response(call)
        assert not response.error
        assert "Tool unlocked" in response.content
        assert "update_transaction_rewards_3847" in response.content

        # Verify unlock only stores in-memory state (DB write happens on call)
        db = environment.tools.db
        assert len(db.agent_discoverable_tools.data) == 0

    def test_agent_updates_transaction_rewards(self, environment: Environment):
        """Test that agent updates transaction rewards after unlocking."""
        # First unlock the tool
        unlock_call = ToolCall(
            id="1",
            name="unlock_discoverable_agent_tool",
            arguments={"agent_tool_name": "update_transaction_rewards_3847"},
        )
        environment.get_response(unlock_call)

        # Update United Airlines transaction rewards
        update_call = ToolCall(
            id="2",
            name="call_discoverable_agent_tool",
            arguments={
                "agent_tool_name": "update_transaction_rewards_3847",
                "arguments": json.dumps(
                    {
                        "transaction_id": "txn_d80aef98f532",
                        "new_rewards_earned": "1390 points",
                    }
                ),
            },
        )
        response = environment.get_response(update_call)
        assert not response.error
        assert "Transaction updated" in response.content
        assert "1390 points" in response.content

        # Verify in database
        db = environment.tools.db
        txn = db.credit_card_transaction_history.data["txn_d80aef98f532"]
        assert txn["rewards_earned"] == "1390 points"

    def test_full_dispute_flow_all_six_transactions(self, environment: Environment):
        """Test complete dispute flow for all 6 incorrect transactions."""
        db = environment.tools.db

        # Expected correct rewards for each transaction
        transactions = [
            ("txn_57ecc6da56c2", "95 points"),  # Trader Joe's, Crypto 2%
            ("txn_d80aef98f532", "1390 points"),  # United, Business Platinum 4%
            ("txn_896ac64b98d7", "642 points"),  # Patagonia, EcoCard 5%
            ("txn_adea68821a1d", "669 points"),  # Marriott, Silver 4%
            ("txn_0be1ccc37761", "2049 points"),  # LinkedIn Ads, Business Platinum 4%
            ("txn_e647e242ce96", "750 points"),  # Google Ads, Business Platinum 4%
        ]

        # Step 1: Agent gives dispute tool to user
        give_call = ToolCall(
            id="0",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        response = environment.get_response(give_call)
        assert not response.error

        # Step 2: User submits disputes for all 6 transactions
        for i, (txn_id, _) in enumerate(transactions):
            submit_call = ToolCall(
                id=str(i + 1),
                name="call_discoverable_user_tool",
                arguments={
                    "discoverable_tool_name": "submit_cash_back_dispute_0589",
                    "arguments": json.dumps(
                        {"user_id": "890389b165", "transaction_id": txn_id}
                    ),
                },
                requestor="user",
            )
            response = environment.get_response(submit_call)
            assert not response.error
            assert "RESOLVED" in response.content

        # Verify all 6 disputes are recorded
        assert len(db.cash_back_disputes.data) == 6
        assert len(db.user_discoverable_tool_calls.data) == 6

        # Step 3: Agent unlocks update tool
        unlock_call = ToolCall(
            id="10",
            name="unlock_discoverable_agent_tool",
            arguments={"agent_tool_name": "update_transaction_rewards_3847"},
        )
        response = environment.get_response(unlock_call)
        assert not response.error

        # Step 4: Agent updates rewards for all 6 transactions
        for i, (txn_id, correct_rewards) in enumerate(transactions):
            update_call = ToolCall(
                id=str(i + 20),
                name="call_discoverable_agent_tool",
                arguments={
                    "agent_tool_name": "update_transaction_rewards_3847",
                    "arguments": json.dumps(
                        {
                            "transaction_id": txn_id,
                            "new_rewards_earned": correct_rewards,
                        }
                    ),
                },
            )
            response = environment.get_response(update_call)
            assert not response.error
            assert "Transaction updated" in response.content

        # Verify all transactions have correct rewards
        for txn_id, correct_rewards in transactions:
            txn = db.credit_card_transaction_history.data[txn_id]
            assert txn["rewards_earned"] == correct_rewards

        # Verify agent tool recorded (unique by tool name, not by each call)
        assert len(db.agent_discoverable_tools.data) == 1

    def test_lookup_user_transactions(self, environment: Environment):
        """Test that agent can look up Fatima's transactions."""
        call = ToolCall(
            id="1",
            name="get_credit_card_transactions_by_user",
            arguments={"user_id": "890389b165"},
        )
        response = environment.get_response(call)
        assert not response.error
        assert "txn_57ecc6da56c2" in response.content
        assert "txn_d80aef98f532" in response.content
        assert "Trader Joe's" in response.content
        assert "United Airlines" in response.content
