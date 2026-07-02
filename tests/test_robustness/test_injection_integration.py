"""Integration tests: verify blocking injections actually block write actions.

For each domain, load a real environment, find data matching an injection's
precondition, apply the mutation, then call the blocked write-action tool and
confirm it raises an error (for API-enforced checks) or produces the wrong
state (for policy-enforced checks).
"""

import json

import pytest

from tau2.registry import registry
from tau_robustness.injection_config import (
    InjectionConfig,
    apply_mutation,
    check_precondition,
)


# ---------------------------------------------------------------------------
# Retail integration tests
# ---------------------------------------------------------------------------


class TestRetailBlockingIntegration:
    """Retail write tools enforce status preconditions at the API level."""

    @pytest.fixture
    def env(self):
        return registry.get_env_constructor("retail")()

    def _find_order_by_status(self, env, status: str):
        """Find a real order with the given status."""
        for order_id, order in env.tools.db.orders.items():
            if order.status == status:
                return order
        pytest.skip(f"No order with status={status} in retail DB")

    def _find_user_for_order(self, env, order):
        """Find the user who owns this order."""
        return env.tools.db.users[order.user_id]

    def test_delivered_to_cancelled_blocks_exchange(self, env):
        """Flipping delivered → cancelled should make exchange_delivered_order_items fail."""
        config = InjectionConfig.from_domain("retail")
        inj = next(i for i in config.injections if i.id == "retail_block_exchange_cancelled")

        order = self._find_order_by_status(env, "delivered")
        # Verify precondition matches
        order_data = json.loads(order.model_dump_json())
        assert check_precondition(order_data, inj.precondition)

        # Apply mutation directly to the real DB object
        order.status = "cancelled"

        # Now try to call exchange — should fail with status check
        user = self._find_user_for_order(env, order)
        # Get any payment method from user
        payment_method_id = next(iter(user.payment_methods.keys()))
        # Get first item from order
        item_id = order.items[0].item_id

        with pytest.raises(ValueError, match="Non-delivered order cannot be exchanged"):
            env.tools.exchange_delivered_order_items(
                order_id=order.order_id,
                item_ids=[item_id],
                new_item_ids=[item_id],  # same item is fine, it'll fail on status first
                payment_method_id=payment_method_id,
            )

    def test_delivered_to_cancelled_blocks_return(self, env):
        """Flipping delivered → cancelled should make return_delivered_order_items fail."""
        order = self._find_order_by_status(env, "delivered")
        user = self._find_user_for_order(env, order)
        payment_method_id = next(iter(user.payment_methods.keys()))
        item_id = order.items[0].item_id

        # Apply mutation
        order.status = "cancelled"

        with pytest.raises(ValueError, match="Non-delivered order cannot be returned"):
            env.tools.return_delivered_order_items(
                order_id=order.order_id,
                item_ids=[item_id],
                payment_method_id=payment_method_id,
            )

    def test_pending_to_delivered_blocks_cancel(self, env):
        """Flipping pending → delivered should make cancel_pending_order fail."""
        order = self._find_order_by_status(env, "pending")

        # Apply mutation
        order.status = "delivered"

        with pytest.raises(ValueError, match="Non-pending order cannot be cancelled"):
            env.tools.cancel_pending_order(
                order_id=order.order_id,
                reason="no longer needed",
            )

    def test_pending_to_delivered_blocks_modify_items(self, env):
        """Flipping pending → delivered should make modify_pending_order_items fail."""
        order = self._find_order_by_status(env, "pending")
        user = self._find_user_for_order(env, order)
        payment_method_id = next(iter(user.payment_methods.keys()))
        item_id = order.items[0].item_id

        # Apply mutation
        order.status = "delivered"

        with pytest.raises(ValueError, match="Non-pending order cannot be modified"):
            env.tools.modify_pending_order_items(
                order_id=order.order_id,
                item_ids=[item_id],
                new_item_ids=[item_id],
                payment_method_id=payment_method_id,
            )

    def test_pending_to_item_modified_blocks_cancel(self, env):
        """Flipping pending → 'pending (item modified)' blocks cancel (strict check)."""
        order = self._find_order_by_status(env, "pending")

        # Apply the "already modified" mutation
        order.status = "pending (item modified)"

        with pytest.raises(ValueError, match="Non-pending order cannot be cancelled"):
            env.tools.cancel_pending_order(
                order_id=order.order_id,
                reason="no longer needed",
            )

    def test_pending_to_item_modified_does_not_block_address(self, env):
        """'pending (item modified)' should NOT block modify_address (loose check)."""
        order = self._find_order_by_status(env, "pending")

        # Apply the "already modified" mutation
        order.status = "pending (item modified)"

        # This should succeed because modify_address uses loose "pending" in status
        result = env.tools.modify_pending_order_address(
            order_id=order.order_id,
            address1="123 Test St",
            address2="",
            city="Test City",
            state="TS",
            country="USA",
            zip="00000",
        )
        assert result.address.city == "Test City"

    def test_delivered_to_processed_blocks_exchange(self, env):
        """Flipping delivered → processed should block exchange."""
        order = self._find_order_by_status(env, "delivered")
        user = self._find_user_for_order(env, order)
        payment_method_id = next(iter(user.payment_methods.keys()))
        item_id = order.items[0].item_id

        order.status = "processed"

        with pytest.raises(ValueError, match="Non-delivered order cannot be exchanged"):
            env.tools.exchange_delivered_order_items(
                order_id=order.order_id,
                item_ids=[item_id],
                new_item_ids=[item_id],
                payment_method_id=payment_method_id,
            )


# ---------------------------------------------------------------------------
# Airline integration tests
# ---------------------------------------------------------------------------


class TestAirlineBlockingIntegration:
    """Airline blocking is mostly policy-enforced by the agent, not the API.
    We verify the mutations produce data states that WOULD cause the agent
    to refuse action, and test the one API-enforced block (cancelled status
    doesn't prevent cancel_reservation since the API has no status guard,
    but we can verify the data transformation is correct).
    """

    @pytest.fixture
    def env(self):
        return registry.get_env_constructor("airline")()

    def _find_reservation_by_cabin(self, env, cabin: str):
        """Find a reservation with the given cabin class."""
        for res_id, res in env.tools.db.reservations.items():
            if res.cabin == cabin and res.status is None:
                return res
        pytest.skip(f"No active reservation with cabin={cabin} in airline DB")

    def test_cabin_flip_to_basic_economy_produces_correct_state(self, env):
        """Flipping economy → basic_economy should change the reservation data.
        The agent reads this and refuses to modify flights per policy."""
        config = InjectionConfig.from_domain("airline")
        inj = next(i for i in config.injections if i.id == "airline_block_update_cabin_flip")

        res = self._find_reservation_by_cabin(env, "economy")
        # Apply mutation to the reservation object
        res_data = json.loads(res.model_dump_json())
        assert check_precondition(res_data, inj.precondition)
        apply_mutation(res_data, inj.mutations[0])
        assert res_data["cabin"] == "basic_economy"

    def test_cancelled_status_injection_produces_cancelled_state(self, env):
        """Setting status to 'cancelled' on an active reservation should
        produce a state the agent interprets as already-cancelled."""
        config = InjectionConfig.from_domain("airline")
        inj = next(i for i in config.injections if i.id == "airline_block_all_cancelled_status")

        # Find any active reservation (status=None)
        res = None
        for res_id, r in env.tools.db.reservations.items():
            if r.status is None:
                res = r
                break
        if res is None:
            pytest.skip("No active reservation in airline DB")

        res_data = json.loads(res.model_dump_json())
        assert check_precondition(res_data, inj.precondition)
        apply_mutation(res_data, inj.mutations[0])
        assert res_data["status"] == "cancelled"

    def test_insurance_flip_produces_no_coverage_state(self, env):
        """Flipping insurance yes → no should remove cancellation grounds."""
        config = InjectionConfig.from_domain("airline")
        inj = next(i for i in config.injections if i.id == "airline_block_cancel_insurance_flip")

        # Find reservation with insurance
        res = None
        for res_id, r in env.tools.db.reservations.items():
            if r.insurance == "yes" and r.status is None:
                res = r
                break
        if res is None:
            pytest.skip("No insured active reservation in airline DB")

        res_data = json.loads(res.model_dump_json())
        assert check_precondition(res_data, inj.precondition)
        apply_mutation(res_data, inj.mutations[0])
        assert res_data["insurance"] == "no"


# ---------------------------------------------------------------------------
# Telecom integration tests
# ---------------------------------------------------------------------------


class TestTelecomBlockingIntegration:
    """Telecom write tools enforce status preconditions at the API level."""

    @pytest.fixture
    def env(self):
        return registry.get_env_constructor("telecom")()

    def _find_line_by_status(self, env, status_value: str):
        """Find a customer and line where line has the given status."""
        for line in env.tools.db.lines:
            if line.status.value == status_value:
                # Find the customer who owns this line
                for customer in env.tools.db.customers:
                    if line.line_id in customer.line_ids:
                        return customer, line
        pytest.skip(f"No line with status={status_value} in telecom DB")

    def _find_active_line(self, env):
        return self._find_line_by_status(env, "Active")

    def _find_suspended_line(self, env):
        return self._find_line_by_status(env, "Suspended")

    def test_active_to_suspended_blocks_suspend(self, env):
        """If an Active line is flipped to Suspended, suspend_line should fail
        because it requires status=Active."""
        from tau2.domains.telecom.data_model import LineStatus

        customer, line = self._find_active_line(env)

        # Apply mutation: flip Active → Suspended
        line.status = LineStatus.SUSPENDED

        with pytest.raises(ValueError, match="Line must be active to suspend"):
            env.tools.suspend_line(
                customer_id=customer.customer_id,
                line_id=line.line_id,
                reason="test suspension",
            )

    def test_active_to_closed_blocks_suspend(self, env):
        """If an Active line is flipped to Closed, suspend_line should fail
        because it requires status=Active (Closed != Active)."""
        from tau2.domains.telecom.data_model import LineStatus

        customer, line = self._find_active_line(env)

        # Apply a different mutation: Active → Closed
        line.status = LineStatus.CLOSED

        with pytest.raises(ValueError, match="Line must be active to suspend"):
            env.tools.suspend_line(
                customer_id=customer.customer_id,
                line_id=line.line_id,
                reason="test suspension",
            )

    def test_suspended_to_active_blocks_resume(self, env):
        """If a Suspended line is flipped to Active, resume_line should fail
        because it requires status=Suspended or Pending Activation."""
        from tau2.domains.telecom.data_model import LineStatus

        customer, line = self._find_suspended_line(env)

        # Apply mutation: flip Suspended → Active
        line.status = LineStatus.ACTIVE

        with pytest.raises(ValueError, match="Line must be suspended to resume"):
            env.tools.resume_line(
                customer_id=customer.customer_id,
                line_id=line.line_id,
            )
