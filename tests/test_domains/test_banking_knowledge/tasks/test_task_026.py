"""Tests for task_026.

task_026: Full end-to-end cash back dispute and resolution flow.
- User: Amara Okonkwo (user_id: 755bcb4d5d)
- Scenario: Agent investigates discrepancies, helps user submit disputes,
  then after disputes are resolved (auto-resolved), agent updates transaction
  rewards to correct values.
- Cards: Business Silver Rewards Card, Silver Rewards Card
- Transactions with incorrect rewards:
  - txn_a8f1c2d3e403: JetBlue Airways $315 -> should be 6300 points (10% doubled via promo)
  - txn_b7e2d4c5f506: GitHub Enterprise $255 -> should be 1020 points (4%)
  - txn_a8f1c2d3e410: Southwest Airlines $380 -> should be 3800 points (10%)
  - txn_a8f1c2d3e411: Zoom Video $149.99 -> should be 1500 points (10%)
"""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment

from .conftest import create_amara_db, create_environment


@pytest.fixture
def environment() -> Environment:
    """Environment with Amara's data and auto_resolve=True."""
    db = create_amara_db(auto_resolve=True)
    return create_environment(db)


class TestTask026:
    """Tests for task_026 - Full dispute flow with auto-resolve."""

    def test_verify_user_amara(self, environment: Environment):
        """Test that agent can verify Amara Okonkwo's identity."""
        call = ToolCall(
            id="1",
            name="log_verification",
            arguments={
                "name": "Amara Okonkwo",
                "user_id": "755bcb4d5d",
                "address": "305 Magnolia Street, Houston, TX 77002",
                "email": "dancing_star_amara@icloud.com",
                "phone_number": "713-555-0963",
                "date_of_birth": "08/11/1997",
                "time_verified": "2025-11-14 03:40:00 EST",
            },
        )
        response = environment.get_response(call)
        assert not response.error
        assert "Verification logged successfully" in response.content
        assert "Amara Okonkwo" in response.content

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

        # User submits dispute for JetBlue transaction
        submit_call = ToolCall(
            id="2",
            name="call_discoverable_user_tool",
            arguments={
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "755bcb4d5d", "transaction_id": "txn_a8f1c2d3e403"}
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
        assert dispute["transaction_id"] == "txn_a8f1c2d3e403"

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

        # Update JetBlue transaction rewards
        update_call = ToolCall(
            id="2",
            name="call_discoverable_agent_tool",
            arguments={
                "agent_tool_name": "update_transaction_rewards_3847",
                "arguments": json.dumps(
                    {
                        "transaction_id": "txn_a8f1c2d3e403",
                        "new_rewards_earned": "6300 points",
                    }
                ),
            },
        )
        response = environment.get_response(update_call)
        assert not response.error
        assert "Transaction updated" in response.content
        assert "6300 points" in response.content

        # Verify in database
        db = environment.tools.db
        txn = db.credit_card_transaction_history.data["txn_a8f1c2d3e403"]
        assert txn["rewards_earned"] == "6300 points"

    def test_full_dispute_flow_all_four_transactions(self, environment: Environment):
        """Test complete dispute flow for all 4 incorrect transactions."""
        db = environment.tools.db

        # Expected correct rewards for each transaction
        transactions = [
            ("txn_a8f1c2d3e403", "6300 points"),  # JetBlue, Business Silver 10% doubled
            ("txn_b7e2d4c5f506", "1020 points"),  # GitHub, Silver Rewards 4%
            ("txn_a8f1c2d3e410", "3800 points"),  # Southwest, Business Silver 10%
            ("txn_a8f1c2d3e411", "1500 points"),  # Zoom, Business Silver 10%
        ]

        # Step 1: Agent gives dispute tool to user
        give_call = ToolCall(
            id="0",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        response = environment.get_response(give_call)
        assert not response.error

        # Step 2: User submits disputes for all 4 transactions
        for i, (txn_id, _) in enumerate(transactions):
            submit_call = ToolCall(
                id=str(i + 1),
                name="call_discoverable_user_tool",
                arguments={
                    "discoverable_tool_name": "submit_cash_back_dispute_0589",
                    "arguments": json.dumps(
                        {"user_id": "755bcb4d5d", "transaction_id": txn_id}
                    ),
                },
                requestor="user",
            )
            response = environment.get_response(submit_call)
            assert not response.error
            assert "RESOLVED" in response.content

        # Verify all 4 disputes are recorded
        assert len(db.cash_back_disputes.data) == 4
        assert len(db.user_discoverable_tool_calls.data) == 4

        # Step 3: Agent unlocks update tool
        unlock_call = ToolCall(
            id="10",
            name="unlock_discoverable_agent_tool",
            arguments={"agent_tool_name": "update_transaction_rewards_3847"},
        )
        response = environment.get_response(unlock_call)
        assert not response.error

        # Step 4: Agent updates rewards for all 4 transactions
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
        """Test that agent can look up Amara's transactions."""
        call = ToolCall(
            id="1",
            name="get_credit_card_transactions_by_user",
            arguments={"user_id": "755bcb4d5d"},
        )
        response = environment.get_response(call)
        assert not response.error
        assert "txn_a8f1c2d3e403" in response.content
        assert "txn_b7e2d4c5f506" in response.content
        assert "JetBlue Airways" in response.content
        assert "GitHub Enterprise" in response.content
