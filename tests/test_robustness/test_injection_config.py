"""Tests for injection configuration data models and field mutations."""

import json
import pytest

from tau_robustness.injection_config import (
    FieldMutation,
    InjectionConfig,
    InjectionDef,
    InjectionType,
    apply_mutation,
    append_nested,
    check_precondition,
    delete_nested,
    get_nested,
    set_nested,
)


# --- Nested dict helpers ---


class TestGetNested:
    def test_simple_key(self):
        data = {"name": "Alice", "age": 30}
        assert get_nested(data, "name") == "Alice"

    def test_nested_key(self):
        data = {"user": {"address": {"city": "NYC"}}}
        assert get_nested(data, "user.address.city") == "NYC"

    def test_list_index(self):
        data = {"items": [{"id": 1}, {"id": 2}]}
        assert get_nested(data, "items.0.id") == 1
        assert get_nested(data, "items.1.id") == 2

    def test_missing_key_raises(self):
        data = {"a": 1}
        with pytest.raises(KeyError):
            get_nested(data, "b")

    def test_list_out_of_range(self):
        data = {"items": [1, 2]}
        with pytest.raises(IndexError):
            get_nested(data, "items.5")


class TestSetNested:
    def test_simple_set(self):
        data = {"status": "pending"}
        set_nested(data, "status", "delivered")
        assert data["status"] == "delivered"

    def test_nested_set(self):
        data = {"user": {"name": "Alice"}}
        set_nested(data, "user.name", "Bob")
        assert data["user"]["name"] == "Bob"

    def test_list_set(self):
        data = {"items": [10, 20, 30]}
        set_nested(data, "items.1", 99)
        assert data["items"][1] == 99

    def test_set_creates_overwrite(self):
        data = {"a": {"b": 1}}
        set_nested(data, "a.b", {"c": 2})
        assert data["a"]["b"] == {"c": 2}


class TestDeleteNested:
    def test_delete_simple(self):
        data = {"name": "Alice", "age": 30}
        deleted = delete_nested(data, "age")
        assert deleted == 30
        assert "age" not in data

    def test_delete_nested(self):
        data = {"user": {"name": "Alice", "email": "a@b.com"}}
        delete_nested(data, "user.email")
        assert "email" not in data["user"]

    def test_delete_from_list(self):
        data = {"items": [1, 2, 3]}
        deleted = delete_nested(data, "items.1")
        assert deleted == 2
        assert data["items"] == [1, 3]


class TestAppendNested:
    def test_append_to_list(self):
        data = {"items": [1, 2]}
        append_nested(data, "items", 3)
        assert data["items"] == [1, 2, 3]

    def test_append_to_nested_list(self):
        data = {"user": {"reservations": ["A", "B"]}}
        append_nested(data, "user.reservations", "C")
        assert data["user"]["reservations"] == ["A", "B", "C"]

    def test_append_to_non_list_raises(self):
        data = {"name": "Alice"}
        with pytest.raises(TypeError):
            append_nested(data, "name", "extra")


# --- Mutation application ---


class TestApplyMutation:
    def test_set_mutation(self):
        data = {"status": "pending"}
        mutation = FieldMutation(field_path="status", action="set", value="delivered")
        apply_mutation(data, mutation)
        assert data["status"] == "delivered"

    def test_delete_mutation(self):
        data = {"address": "123 Main St", "phone": "555-1234"}
        mutation = FieldMutation(field_path="address", action="delete")
        apply_mutation(data, mutation)
        assert "address" not in data

    def test_append_mutation(self):
        data = {"payment_methods": [{"id": "cc_1"}]}
        mutation = FieldMutation(
            field_path="payment_methods",
            action="append",
            value={"id": "gift_9999", "source": "gift_card"},
        )
        apply_mutation(data, mutation)
        assert len(data["payment_methods"]) == 2
        assert data["payment_methods"][1]["id"] == "gift_9999"

    def test_flip_mutation_string(self):
        data = {"status": "pending"}
        mutation = FieldMutation(
            field_path="status",
            action="flip",
            flip_map={"pending": "cancelled"},
        )
        apply_mutation(data, mutation)
        assert data["status"] == "cancelled"

    def test_flip_mutation_missing_value_raises(self):
        data = {"status": "delivered"}
        mutation = FieldMutation(
            field_path="status",
            action="flip",
            flip_map={"pending": "cancelled"},
        )
        with pytest.raises(ValueError, match="Flip map does not contain"):
            apply_mutation(data, mutation)

    def test_unknown_action_raises(self):
        data = {"x": 1}
        mutation = FieldMutation(field_path="x", action="explode")
        with pytest.raises(ValueError, match="Unknown mutation action"):
            apply_mutation(data, mutation)


# --- Precondition checking ---


class TestCheckPrecondition:
    def test_simple_match(self):
        data = {"status": "delivered"}
        assert check_precondition(data, {"status": "delivered"}) is True

    def test_simple_no_match(self):
        data = {"status": "pending"}
        assert check_precondition(data, {"status": "delivered"}) is False

    def test_nested_match(self):
        data = {"user": {"membership": "gold"}}
        assert check_precondition(data, {"user.membership": "gold"}) is True

    def test_missing_field(self):
        data = {"name": "Alice"}
        assert check_precondition(data, {"email": "a@b.com"}) is False

    def test_empty_precondition(self):
        data = {"anything": "here"}
        assert check_precondition(data, {}) is True


# --- InjectionConfig ---


class TestInjectionConfig:
    @pytest.fixture
    def sample_config(self):
        return InjectionConfig(
            domain="retail",
            injections=[
                InjectionDef(
                    id="test_stale",
                    type=InjectionType.STALE_DATA,
                    target_tool="get_order_details",
                    description="Test stale data",
                    difficulty=2,
                    mutations=[
                        FieldMutation(
                            field_path="status", action="set", value="pending"
                        )
                    ],
                    detection_signals=["already delivered"],
                    recovery_signals=["get_order_details"],
                ),
                InjectionDef(
                    id="test_phantom",
                    type=InjectionType.PHANTOM,
                    target_tool="get_user_details",
                    description="Test phantom payment",
                    difficulty=3,
                    mutations=[
                        FieldMutation(
                            field_path="payment_methods",
                            action="append",
                            value={"id": "fake"},
                        )
                    ],
                ),
                InjectionDef(
                    id="test_missing",
                    type=InjectionType.MISSING_DATA,
                    target_tool="get_order_details",
                    description="Test missing data",
                    difficulty=1,
                    mutations=[FieldMutation(field_path="address", action="delete")],
                ),
            ],
        )

    def test_get_injections_for_tool(self, sample_config):
        order_injs = sample_config.get_injections_for_tool("get_order_details")
        assert len(order_injs) == 2
        assert all(inj.target_tool == "get_order_details" for inj in order_injs)

    def test_get_injections_for_unknown_tool(self, sample_config):
        assert sample_config.get_injections_for_tool("nonexistent_tool") == []

    def test_get_injections_by_type(self, sample_config):
        phantom = sample_config.get_injections_by_type(InjectionType.PHANTOM)
        assert len(phantom) == 1
        assert phantom[0].id == "test_phantom"

    def test_get_injections_by_difficulty(self, sample_config):
        easy = sample_config.get_injections_by_difficulty(1)
        assert len(easy) == 1
        assert easy[0].id == "test_missing"


class TestBlocksActions:
    def test_blocks_actions_default_empty(self):
        """blocks_actions defaults to empty list when not specified."""
        inj = InjectionDef(
            id="test",
            type=InjectionType.STALE_DATA,
            target_tool="get_order_details",
            description="Test",
            mutations=[FieldMutation(field_path="status", action="set", value="x")],
        )
        assert inj.blocks_actions == []

    def test_blocks_actions_set_explicitly(self):
        """blocks_actions can be set with specific action names."""
        inj = InjectionDef(
            id="test",
            type=InjectionType.STATUS_FLIP,
            target_tool="get_order_details",
            description="Test",
            mutations=[FieldMutation(field_path="status", action="set", value="x")],
            blocks_actions=["exchange_delivered_order_items", "return_delivered_order_items"],
        )
        assert inj.blocks_actions == [
            "exchange_delivered_order_items",
            "return_delivered_order_items",
        ]

    def test_blocks_actions_loads_from_yaml(self, tmp_path):
        """blocks_actions should load correctly from YAML config files."""
        yaml_content = """
domain: test
injections:
  - id: blocking_inj
    type: status_flip
    target_tool: get_order_details
    description: "Test blocking injection"
    difficulty: 2
    mutations:
      - field_path: status
        action: flip
        flip_map:
          "delivered": "cancelled"
    precondition:
      status: "delivered"
    blocks_actions:
      - "exchange_delivered_order_items"
      - "return_delivered_order_items"
  - id: cosmetic_inj
    type: missing_data
    target_tool: get_order_details
    description: "Test cosmetic injection"
    difficulty: 1
    mutations:
      - field_path: address
        action: delete
    blocks_actions: []
"""
        yaml_file = tmp_path / "test_injections.yaml"
        yaml_file.write_text(yaml_content)

        config = InjectionConfig.from_yaml(yaml_file)
        assert len(config.injections) == 2

        blocking = config.injections[0]
        assert blocking.id == "blocking_inj"
        assert blocking.blocks_actions == [
            "exchange_delivered_order_items",
            "return_delivered_order_items",
        ]

        cosmetic = config.injections[1]
        assert cosmetic.id == "cosmetic_inj"
        assert cosmetic.blocks_actions == []

    def test_blocks_actions_omitted_in_yaml_defaults_to_empty(self, tmp_path):
        """When blocks_actions is omitted from YAML, it defaults to empty list."""
        yaml_content = """
domain: test
injections:
  - id: no_blocks
    type: stale_data
    target_tool: get_order_details
    description: "No blocks_actions field"
    mutations:
      - field_path: status
        action: set
        value: "pending"
"""
        yaml_file = tmp_path / "test_injections.yaml"
        yaml_file.write_text(yaml_content)

        config = InjectionConfig.from_yaml(yaml_file)
        assert config.injections[0].blocks_actions == []


class TestDomainYAMLIntegrity:
    """Verify all domain YAML injection files load correctly and have valid structure."""

    @pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
    def test_domain_yaml_loads(self, domain):
        """Each domain's injection YAML should parse into a valid InjectionConfig."""
        config = InjectionConfig.from_domain(domain)
        assert config.domain == domain
        assert len(config.injections) >= 1

    @pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
    def test_domain_has_blocking_injections(self, domain):
        """Each domain should have at least one blocking injection."""
        config = InjectionConfig.from_domain(domain)
        blocking = [inj for inj in config.injections if inj.blocks_actions]
        assert len(blocking) >= 1, f"{domain} has no blocking injections"

    @pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
    def test_domain_injection_ids_unique(self, domain):
        """All injection IDs within a domain should be unique."""
        config = InjectionConfig.from_domain(domain)
        ids = [inj.id for inj in config.injections]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {domain}: {ids}"

    def test_retail_blocking_targets(self):
        """Retail blocking injections should target order-related write actions."""
        config = InjectionConfig.from_domain("retail")
        blocking = [inj for inj in config.injections if inj.blocks_actions]
        all_blocked = set()
        for inj in blocking:
            all_blocked.update(inj.blocks_actions)
        # Should block both pending-order and delivered-order actions
        assert "exchange_delivered_order_items" in all_blocked
        assert "cancel_pending_order" in all_blocked
        assert "modify_pending_order_items" in all_blocked

    def test_retail_already_modified_injection_mechanics(self):
        """The 'already modified' injection should flip pending → pending (item modified)."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_block_modify_already_modified"
        )
        # Verify it targets get_order_details with the right precondition
        assert inj.target_tool == "get_order_details"
        assert inj.precondition == {"status": "pending"}
        assert "modify_pending_order_items" in inj.blocks_actions
        assert "cancel_pending_order" in inj.blocks_actions
        # Should NOT block address/payment modify (those use loose check)
        assert "modify_pending_order_address" not in inj.blocks_actions
        assert "modify_pending_order_payment" not in inj.blocks_actions

        # Verify the mutation produces the right value
        data = {"status": "pending", "items": []}
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "pending (item modified)"

    def test_retail_phantom_payment_mutation_on_dict(self):
        """Phantom payment injection should set a key in dict-type payment_methods."""
        config = InjectionConfig.from_domain("retail")
        inj = next(i for i in config.injections if i.id == "retail_phantom_payment")
        assert inj.target_tool == "get_user_details"
        assert inj.blocks_actions == []  # cosmetic, not blocking

        # Simulate a dict-type payment_methods (as in retail data model)
        data = {
            "payment_methods": {
                "credit_card_1234567": {
                    "source": "credit_card",
                    "id": "credit_card_1234567",
                    "brand": "visa",
                    "last_four": "1234",
                }
            }
        }
        for mutation in inj.mutations:
            apply_mutation(data, mutation)

        # Should have added a new key without removing the existing one
        assert "credit_card_1234567" in data["payment_methods"]
        assert "gift_card_9999999" in data["payment_methods"]
        phantom = data["payment_methods"]["gift_card_9999999"]
        assert phantom["source"] == "gift_card"
        assert phantom["balance"] == 150.0

    def test_retail_block_exchange_processed_mechanics(self):
        """The 'processed' injection should flip delivered → processed."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_block_exchange_processed"
        )
        assert inj.target_tool == "get_order_details"
        assert inj.precondition == {"status": "delivered"}
        assert "exchange_delivered_order_items" in inj.blocks_actions
        assert "return_delivered_order_items" in inj.blocks_actions

        data = {"status": "delivered", "items": []}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "processed"

    def test_retail_block_modify_processed_mechanics(self):
        """The 'processed' injection should flip pending → processed."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_block_modify_processed"
        )
        assert inj.target_tool == "get_order_details"
        assert inj.precondition == {"status": "pending"}
        assert "cancel_pending_order" in inj.blocks_actions
        assert "modify_pending_order_items" in inj.blocks_actions
        assert "modify_pending_order_address" in inj.blocks_actions
        assert "modify_pending_order_payment" in inj.blocks_actions

        data = {"status": "pending", "items": []}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "processed"

    def test_retail_block_return_already_requested_mechanics(self):
        """The 'return requested' injection should flip delivered → return requested."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_block_return_already_requested"
        )
        assert inj.target_tool == "get_order_details"
        assert inj.precondition == {"status": "delivered"}
        assert "exchange_delivered_order_items" in inj.blocks_actions
        assert "return_delivered_order_items" in inj.blocks_actions

        data = {"status": "delivered", "items": []}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "return requested"

    def test_retail_address_city_corruption_mechanics(self):
        """Address city corruption should overwrite city field."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_address_city_corruption"
        )
        assert inj.target_tool == "get_order_details"
        assert inj.blocks_actions == []  # cosmetic

        data = {
            "status": "pending",
            "address": {
                "address1": "123 Main St",
                "address2": "",
                "city": "New York",
                "state": "NY",
                "country": "USA",
                "zip": "10001",
            },
        }
        for mutation in inj.mutations:
            apply_mutation(data, mutation)
        assert data["address"]["city"] == "Springfield"
        # Other address fields should be untouched
        assert data["address"]["state"] == "NY"

    def test_retail_item_duplication_mechanics(self):
        """Item duplication should append a phantom item to the items list."""
        config = InjectionConfig.from_domain("retail")
        inj = next(
            i for i in config.injections
            if i.id == "retail_item_duplication"
        )
        assert inj.target_tool == "get_order_details"
        assert inj.blocks_actions == []  # cosmetic

        data = {
            "status": "delivered",
            "items": [
                {
                    "name": "T-Shirt",
                    "product_id": "1234567890",
                    "item_id": "9876543210",
                    "price": 29.99,
                    "options": {"color": "blue", "size": "M"},
                }
            ],
        }
        for mutation in inj.mutations:
            apply_mutation(data, mutation)
        assert len(data["items"]) == 2
        assert data["items"][1]["name"] == "Phantom Duplicate"
        assert data["items"][1]["item_id"] == "0000000000"

    def test_airline_blocking_targets(self):
        """Airline blocking injections should target reservation write actions."""
        config = InjectionConfig.from_domain("airline")
        blocking = [inj for inj in config.injections if inj.blocks_actions]
        all_blocked = set()
        for inj in blocking:
            all_blocked.update(inj.blocks_actions)
        assert "update_reservation_flights" in all_blocked
        assert "cancel_reservation" in all_blocked
        assert "update_reservation_baggages" in all_blocked
        assert "update_reservation_passengers" in all_blocked
        assert "send_certificate" in all_blocked
        assert "book_reservation" in all_blocked

    def test_airline_block_all_cancelled_status_mechanics(self):
        """Cancelled status injection should set status to 'cancelled' on active reservations."""
        config = InjectionConfig.from_domain("airline")
        inj = next(
            i for i in config.injections
            if i.id == "airline_block_all_cancelled_status"
        )
        assert inj.target_tool == "get_reservation_details"
        # Precondition: status is null (None) = active reservation
        assert inj.precondition == {"status": None}
        # Blocks all write actions
        assert "update_reservation_flights" in inj.blocks_actions
        assert "cancel_reservation" in inj.blocks_actions
        assert "update_reservation_baggages" in inj.blocks_actions
        assert "update_reservation_passengers" in inj.blocks_actions

        # Apply mutation to an active reservation (status=None)
        data = {"status": None, "cabin": "economy", "flights": []}
        assert check_precondition(data, inj.precondition)
        for mutation in inj.mutations:
            apply_mutation(data, mutation)
        assert data["status"] == "cancelled"

    def test_airline_block_update_business_cabin_flip_mechanics(self):
        """Business cabin flip should turn business → basic_economy."""
        config = InjectionConfig.from_domain("airline")
        inj = next(
            i for i in config.injections
            if i.id == "airline_block_update_business_cabin_flip"
        )
        assert inj.target_tool == "get_reservation_details"
        assert inj.precondition == {"cabin": "business"}
        assert "update_reservation_flights" in inj.blocks_actions

        data = {"cabin": "business", "flights": []}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["cabin"] == "basic_economy"

    def test_airline_block_book_phantom_payment_mechanics(self):
        """Phantom payment injection should add a fake gift card to user's payment methods."""
        config = InjectionConfig.from_domain("airline")
        inj = next(
            i for i in config.injections
            if i.id == "airline_block_book_phantom_payment"
        )
        assert inj.target_tool == "get_user_details"
        assert "book_reservation" in inj.blocks_actions
        assert inj.type == InjectionType.PHANTOM

        # Simulate user details with existing credit card
        data = {
            "payment_methods": {
                "credit_card_1234567": {
                    "source": "credit_card",
                    "id": "credit_card_1234567",
                    "brand": "visa",
                    "last_four": "7890",
                }
            },
            "reservations": [],
        }
        for mutation in inj.mutations:
            apply_mutation(data, mutation)

        # Should add phantom gift card without removing existing payment
        assert "credit_card_1234567" in data["payment_methods"]
        assert "gift_card_9999999" in data["payment_methods"]
        phantom = data["payment_methods"]["gift_card_9999999"]
        assert phantom["source"] == "gift_card"
        assert phantom["amount"] == 500.0

    def test_airline_block_flight_seats_full_mechanics(self):
        """Flight seats full injection should set available seats to 0."""
        config = InjectionConfig.from_domain("airline")
        inj = next(
            i for i in config.injections
            if i.id == "airline_block_flight_seats_full"
        )
        assert inj.target_tool == "search_direct_flight"
        assert "update_reservation_flights" in inj.blocks_actions
        assert "book_reservation" in inj.blocks_actions

        # Simulate search results (list of flights)
        data = [
            {
                "flight_number": "HAT139",
                "available_seats": {"basic_economy": 7, "economy": 5, "business": 3},
                "prices": {"basic_economy": 65, "economy": 114, "business": 395},
            }
        ]
        for mutation in inj.mutations:
            apply_mutation(data, mutation)

        assert data[0]["available_seats"]["economy"] == 0
        # Other cabin classes should be untouched
        assert data[0]["available_seats"]["basic_economy"] == 7
        assert data[0]["available_seats"]["business"] == 3

    def test_airline_block_compensation_membership_flip_mechanics(self):
        """Membership flip should turn gold → regular to block compensation."""
        config = InjectionConfig.from_domain("airline")
        inj = next(
            i for i in config.injections
            if i.id == "airline_block_compensation_membership_flip"
        )
        assert inj.target_tool == "get_user_details"
        assert inj.precondition == {"membership": "gold"}
        assert "send_certificate" in inj.blocks_actions

        data = {"membership": "gold", "reservations": []}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["membership"] == "regular"

    def test_telecom_blocking_targets(self):
        """Telecom blocking injections should target line/billing write actions."""
        config = InjectionConfig.from_domain("telecom")
        blocking = [inj for inj in config.injections if inj.blocks_actions]
        all_blocked = set()
        for inj in blocking:
            all_blocked.update(inj.blocks_actions)
        assert "resume_line" in all_blocked
        assert "refuel_data" in all_blocked
        assert "enable_roaming" in all_blocked
        assert "suspend_line" in all_blocked
        assert "send_payment_request" in all_blocked

    def test_telecom_suspend_block_mechanics(self):
        """The suspend block injection should flip Active → Suspended."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_block_suspend_already_suspended"
        )
        assert inj.target_tool == "get_details_by_id"
        assert inj.precondition == {"status": "Active"}
        assert "suspend_line" in inj.blocks_actions

        # Verify the mutation
        data = {"status": "Active", "line_id": "L001"}
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "Suspended"

    def test_telecom_bill_amount_corruption_mechanics(self):
        """Bill amount corruption should set total_due to 999.99."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_bill_amount_corruption"
        )
        assert inj.target_tool == "get_bills_for_customer"
        assert inj.blocks_actions == []  # cosmetic

        data = [{"bill_id": "B001", "total_due": 45.00, "status": "Issued"}]
        apply_mutation(data, inj.mutations[0])
        assert data[0]["total_due"] == 999.99

    def test_telecom_block_refuel_high_limit_mechanics(self):
        """High data limit injection should corrupt data_limit_gb to 100."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_block_refuel_high_limit"
        )
        assert inj.target_tool == "get_data_usage"
        assert inj.type == InjectionType.CORRUPT_FIELD
        assert "refuel_data" in inj.blocks_actions

        # Simulate data usage response with a normal 5 GB limit
        data = {"data_used_gb": 4.8, "data_limit_gb": 5.0}
        apply_mutation(data, inj.mutations[0])
        assert data["data_limit_gb"] == 100.0
        # Actual usage should be untouched
        assert data["data_used_gb"] == 4.8

    def test_telecom_block_roaming_false_speed_mechanics(self):
        """False speed test injection should corrupt download speed to 50 Mbps."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_block_roaming_false_speed"
        )
        assert inj.target_tool == "run_speed_test"
        assert inj.type == InjectionType.CORRUPT_FIELD
        assert "enable_roaming" in inj.blocks_actions

        # Simulate a speed test result (should show 0 or low for no connection)
        data = {"download_speed_mbps": 0.0, "upload_speed_mbps": 0.0, "ping_ms": 999}
        apply_mutation(data, inj.mutations[0])
        assert data["download_speed_mbps"] == 50.0
        # Upload and ping should be untouched
        assert data["upload_speed_mbps"] == 0.0
        assert data["ping_ms"] == 999

    def test_telecom_block_suspend_closed_line_mechanics(self):
        """Closed line injection should flip Active → Closed."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_block_suspend_closed_line"
        )
        assert inj.target_tool == "get_details_by_id"
        assert inj.type == InjectionType.STATUS_FLIP
        assert inj.precondition == {"status": "Active"}
        assert "suspend_line" in inj.blocks_actions

        data = {"status": "Active", "line_id": "L001"}
        assert check_precondition(data, inj.precondition)
        apply_mutation(data, inj.mutations[0])
        assert data["status"] == "Closed"

    def test_telecom_wrong_phone_number_mechanics(self):
        """Wrong phone number injection should set phone_number to placeholder."""
        config = InjectionConfig.from_domain("telecom")
        inj = next(
            i for i in config.injections
            if i.id == "telecom_wrong_phone_number"
        )
        assert inj.target_tool == "get_details_by_id"
        assert inj.blocks_actions == []  # cosmetic

        data = {"line_id": "L001", "phone_number": "555-123-4567"}
        apply_mutation(data, inj.mutations[0])
        assert data["phone_number"] == "000-000-0000"

    def test_cross_domain_unique_ids(self):
        """All injection IDs must be globally unique across all domains."""
        all_ids = []
        for domain in ["retail", "airline", "telecom"]:
            config = InjectionConfig.from_domain(domain)
            for inj in config.injections:
                all_ids.append(inj.id)
        assert len(all_ids) == len(set(all_ids)), (
            f"Duplicate IDs found: {[x for x in all_ids if all_ids.count(x) > 1]}"
        )

    def test_cross_domain_schema_consistency(self):
        """All domain YAMLs should have consistent structure."""
        for domain in ["retail", "airline", "telecom"]:
            config = InjectionConfig.from_domain(domain)
            assert config.domain == domain
            assert len(config.injections) >= 1
            for inj in config.injections:
                assert inj.id
                assert inj.type
                assert inj.target_tool
                assert inj.description
                assert len(inj.mutations) > 0
                assert 1 <= inj.difficulty <= 3
                assert len(inj.detection_signals) > 0
                assert len(inj.recovery_signals) > 0
