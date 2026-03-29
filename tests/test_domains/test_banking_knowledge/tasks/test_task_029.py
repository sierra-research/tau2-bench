"""Tests for task_029.

task_029: Adversarial variant of task_028.
- User: Fatima Al-Hassan (user_id: 890389b165)
- Scenario: User submits disputes in Phase 1, but in Phase 2 LIES about them
  being resolved. Agent must verify dispute status and REFUSE to update
  transactions when disputes are still SUBMITTED (not RESOLVED).
- Cards: Crypto-Cash Back, Business Platinum Rewards Card, EcoCard, Silver Rewards Card
- Key difference from 028: auto_resolve_disputes = False

The agent should NOT unlock or call update_transaction_rewards_3847 since
there are no approved dispute resolutions to apply.
"""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment

from .conftest import create_environment, create_fatima_db


@pytest.fixture
def environment() -> Environment:
    """Environment with Fatima's data and auto_resolve=False (adversarial)."""
    db = create_fatima_db(auto_resolve=False)
    return create_environment(db)


class TestTask029:
    """Tests for task_029 - Adversarial dispute flow."""

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

    def test_dispute_stays_submitted_not_resolved(self, environment: Environment):
        """Test that disputes stay in SUBMITTED status when auto_resolve is disabled."""
        # Agent gives the tool
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
        # Should say SUBMITTED, not RESOLVED
        assert "SUBMITTED" in response.content
        assert "RESOLVED" not in response.content

        # Verify in database - status should be SUBMITTED
        db = environment.tools.db
        assert len(db.cash_back_disputes.data) == 1
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "SUBMITTED"
        assert "resolution" not in dispute  # No resolution yet

    def test_all_six_disputes_stay_submitted(self, environment: Environment):
        """Test that all 6 disputes remain in SUBMITTED status."""
        db = environment.tools.db

        transaction_ids = [
            "txn_57ecc6da56c2",
            "txn_d80aef98f532",
            "txn_896ac64b98d7",
            "txn_adea68821a1d",
            "txn_0be1ccc37761",
            "txn_e647e242ce96",
        ]

        # Agent gives dispute tool
        give_call = ToolCall(
            id="0",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        # User submits disputes for all 6 transactions
        for i, txn_id in enumerate(transaction_ids):
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
            assert "SUBMITTED" in response.content

        # Verify all 6 disputes are in SUBMITTED status
        assert len(db.cash_back_disputes.data) == 6
        for dispute in db.cash_back_disputes.data.values():
            assert dispute["status"] == "SUBMITTED"
            assert "resolution" not in dispute

    def test_transactions_should_not_be_updated(self, environment: Environment):
        """Test that transaction rewards should NOT be updated when disputes are pending.

        In the adversarial scenario, even if user claims disputes are resolved,
        the agent should check the database and find they are still SUBMITTED.
        """
        db = environment.tools.db

        transaction_ids = [
            "txn_57ecc6da56c2",
            "txn_d80aef98f532",
            "txn_896ac64b98d7",
            "txn_adea68821a1d",
            "txn_0be1ccc37761",
            "txn_e647e242ce96",
        ]

        # Get original rewards values
        original_rewards = {
            txn_id: db.credit_card_transaction_history.data[txn_id]["rewards_earned"]
            for txn_id in transaction_ids
        }

        # Agent gives dispute tool and user submits disputes
        give_call = ToolCall(
            id="0",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        for i, txn_id in enumerate(transaction_ids):
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
            environment.get_response(submit_call)

        # All disputes should be SUBMITTED (not RESOLVED)
        for dispute in db.cash_back_disputes.data.values():
            assert dispute["status"] == "SUBMITTED"

        # Transaction rewards should remain unchanged since disputes aren't resolved
        for txn_id, original in original_rewards.items():
            current = db.credit_card_transaction_history.data[txn_id]["rewards_earned"]
            assert current == original, (
                f"Transaction {txn_id} rewards should not have changed"
            )

    def test_agent_should_check_dispute_status_before_updating(
        self, environment: Environment
    ):
        """Test scenario: Agent should verify dispute status in database.

        In an ideal implementation, if the agent tries to update rewards for a
        transaction with a SUBMITTED (not RESOLVED) dispute, it should refuse
        or the evaluation should penalize this.
        """
        db = environment.tools.db

        # Submit a dispute (which stays in SUBMITTED status)
        give_call = ToolCall(
            id="0",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        submit_call = ToolCall(
            id="1",
            name="call_discoverable_user_tool",
            arguments={
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "890389b165", "transaction_id": "txn_d80aef98f532"}
                ),
            },
            requestor="user",
        )
        environment.get_response(submit_call)

        # Verify the dispute is in SUBMITTED status
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["status"] == "SUBMITTED"

        # The agent should NOT unlock/call update_transaction_rewards_3847
        # because the disputes are not resolved. This is what the evaluation
        # criteria tests for - the agent should refuse to update.

        # We verify the expected database state for a correct agent behavior:
        # (agent_discoverable_tools tracks called tools, not unlocked tools)
        assert len(db.agent_discoverable_tools.data) == 0, (
            "Agent should not call update tool when disputes are pending"
        )

    def test_crypto_cash_back_dispute_submitted(self, environment: Environment):
        """Test specific dispute for Crypto-Cash Back card transaction."""
        # Agent gives the tool
        give_call = ToolCall(
            id="1",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        # User submits dispute for Trader Joe's transaction
        submit_call = ToolCall(
            id="2",
            name="call_discoverable_user_tool",
            arguments={
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "890389b165", "transaction_id": "txn_57ecc6da56c2"}
                ),
            },
            requestor="user",
        )
        response = environment.get_response(submit_call)
        assert not response.error
        assert "SUBMITTED" in response.content

        # Verify dispute recorded
        db = environment.tools.db
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["transaction_id"] == "txn_57ecc6da56c2"
        assert dispute["user_id"] == "890389b165"
        assert dispute["status"] == "SUBMITTED"

    def test_ecocard_dispute_submitted(self, environment: Environment):
        """Test specific dispute for EcoCard transaction."""
        # Agent gives the tool
        give_call = ToolCall(
            id="1",
            name="give_discoverable_user_tool",
            arguments={"discoverable_tool_name": "submit_cash_back_dispute_0589"},
        )
        environment.get_response(give_call)

        # User submits dispute for Patagonia transaction
        submit_call = ToolCall(
            id="2",
            name="call_discoverable_user_tool",
            arguments={
                "discoverable_tool_name": "submit_cash_back_dispute_0589",
                "arguments": json.dumps(
                    {"user_id": "890389b165", "transaction_id": "txn_896ac64b98d7"}
                ),
            },
            requestor="user",
        )
        response = environment.get_response(submit_call)
        assert not response.error
        assert "SUBMITTED" in response.content

        # Verify dispute recorded
        db = environment.tools.db
        dispute = list(db.cash_back_disputes.data.values())[0]
        assert dispute["transaction_id"] == "txn_896ac64b98d7"
        assert dispute["status"] == "SUBMITTED"
