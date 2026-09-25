"""Tests for the hospitality domain tools."""

import pytest

from tau2.domains.hospitality.data_model import (
    HospitalityDB,
    StaffRole,
    TableStatus,
)
from tau2.domains.hospitality.tools import HospitalityTools
from tau2.domains.hospitality.environment import get_environment


class TestHospitalityTools:
    """Test suite for HospitalityTools."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        return get_environment()

    @pytest.fixture
    def tools(self, env):
        """Get tools from the environment."""
        return env.tools

    @pytest.fixture
    def db(self, env):
        """Get the database from the environment."""
        return env.tools.db

    def test_get_restaurant_info(self, tools):
        """Test getting restaurant info."""
        info = tools.get_restaurant_info()
        assert info["name"] == "Berkeley Hot Pot"
        assert "hours" in info
        assert "location" in info

    def test_get_menu_details(self, tools):
        """Test getting menu details."""
        result = tools.get_menu_details()
        assert "soup_bases" in result
        assert "menu_items" in result
        assert len(result["soup_bases"]) > 0
        assert len(result["menu_items"]) > 0

    def test_get_menu_details_by_category(self, tools):
        """Test getting menu details by category."""
        result = tools.get_menu_details(category="protein")
        assert "menu_items" in result
        assert all(item["category"] == "protein" for item in result["menu_items"])

    def test_check_table_availability(self, tools):
        """Test checking table availability."""
        result = tools.check_table_availability(
            party_size=4,
            date_str="2026-01-15",
            time_str="18:00",
        )
        assert "available_tables" in result
        assert "total_available" in result
        assert result["party_size"] == 4
        # Check that tables have the new fields
        if result["available_tables"]:
            table = result["available_tables"][0]
            assert "std_capacity" in table
            assert "std_expansion" in table
            assert "max_squeeze" in table
            assert "fit_type" in table

    def test_check_allergy_safety_plain_water(self, tools):
        """Test that Plain Water is always safe."""
        result = tools.check_allergy_safety("S08", "vinegar")
        assert result["is_safe"] is True
        assert "Plain Water" in result["item"]

    def test_check_allergy_safety_tomato_vinegar(self, tools):
        """Test that Tomato soup base warns about hidden vinegar."""
        result = tools.check_allergy_safety("S07", "vinegar")
        assert result["is_safe"] is False
        assert "hidden_ingredients" in result
        assert "vinegar" in [h.lower() for h in result["hidden_ingredients"]]
        assert "CANNOT GUARANTEE" in result["recommendation"]

    def test_check_lunch_special_availability(self, tools):
        """Test lunch special availability check."""
        result = tools.check_lunch_special_availability()
        assert "available" in result
        assert "is_federal_holiday" in result
        assert "is_weekday" in result
        assert "is_before_5pm" in result

    def test_check_item_inventory(self, tools):
        """Test checking item inventory."""
        result = tools.check_item_inventory("Fairy Wand")
        assert result["name"] == "Fairy Wand"
        assert result["stock"] == 0
        assert result["in_stock"] is False

    def test_get_customer_profile(self, tools):
        """Test getting customer profile."""
        customer = tools.get_customer_profile(customer_id="C1001")
        assert customer.name == "VIP Customer"
        assert customer.tier.value == "Diamond"
        assert customer.points == 12500

    def test_get_current_staff_authority(self, tools):
        """Test getting staff authority."""
        result = tools.get_current_staff_authority()
        assert result["role"] == "Server"
        assert result["max_round_off"] == 10.0
        assert result["max_discount_pct"] == 12.0

    def test_create_reservation(self, tools):
        """Test creating a reservation."""
        result = tools.create_reservation(
            customer_name="Test Customer",
            phone="555-999-9999",
            party_size=4,
            date_str="2026-01-20",
            time_str="19:00",
            special_occasion="birthday",
        )
        assert "reservation" in result
        assert result["message"] == "Reservation created successfully"

    def test_create_reservation_party_too_large_weekend(self, tools):
        """Test that large party reservations on weekends are rejected."""
        with pytest.raises(ValueError, match="cannot accept reservations"):
            tools.create_reservation(
                customer_name="Large Party",
                phone="555-999-0000",
                party_size=25,
                date_str="2026-01-17",  # Saturday
                time_str="18:00",
            )

    def test_apply_discount_within_authority(self, tools, db):
        """Test applying discount within server authority."""
        # First create an order to apply discount to
        order = db.orders[0]
        result = tools.apply_discount(
            order_id=order.order_id,
            discount_type="percentage",
            discount_value=10.0,
            reason="Customer service",
        )
        assert "new_total" in result
        assert result["discount_type"] == "percentage"

    def test_apply_discount_exceeds_authority(self, tools, db):
        """Test that discount exceeding server authority is rejected."""
        order = db.orders[0]
        with pytest.raises(ValueError, match="exceeds Server authority"):
            tools.apply_discount(
                order_id=order.order_id,
                discount_type="percentage",
                discount_value=50.0,
                reason="Large discount",
            )

    def test_redeem_secret_code(self, tools):
        """Test redeeming a secret code."""
        result = tools.redeem_secret_code(
            code_phrase="I like your golden bricks",
            table_id="A1",
        )
        assert result["success"] is True
        assert "Fried Steamed Buns" in result["reward"]

    def test_redeem_secret_code_invalid(self, tools):
        """Test redeeming an invalid secret code."""
        result = tools.redeem_secret_code(
            code_phrase="This is not a real code",
            table_id="A2",
        )
        assert result["success"] is False

    def test_record_service_incident(self, tools):
        """Test recording a service incident."""
        result = tools.record_service_incident(
            incident_type="slow_service",
            description="Customer waited 30 minutes for dish",
            table_id="A1",
        )
        assert "incident_id" in result
        assert result["incident_type"] == "slow_service"


class TestHospitalityEnvironment:
    """Test suite for HospitalityEnvironment."""

    def test_environment_creation(self):
        """Test creating the environment."""
        env = get_environment()
        assert env.get_domain_name() == "hospitality"
        assert len(env.get_tools()) > 0
        assert len(env.get_user_tools()) > 0

    def test_environment_policy(self):
        """Test that policy is loaded."""
        env = get_environment()
        policy = env.get_policy()
        assert "Berkeley Hot Pot" in policy
        assert "zero negative review" in policy.lower()

    def test_environment_solo_mode(self):
        """Test solo mode gives access to user tools."""
        env = get_environment(solo_mode=True)
        # In solo mode, agent has access to both agent and user tools
        assert env.solo_mode is True
