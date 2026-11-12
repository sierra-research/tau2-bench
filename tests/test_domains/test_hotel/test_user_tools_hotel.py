import pytest

from tau2.domains.hotel.environment import get_environment
from tau2.domains.hotel.user_data_model import (
    GuestContext,
    GuestPreferences,
    HotelUserDB,
)
from tau2.domains.hotel.user_tools import HotelUserTools


@pytest.fixture
def user_db() -> HotelUserDB:
    """Create a minimal user database for testing"""
    return HotelUserDB(
        guest_context=GuestContext(
            guest_id="G001",
            current_location="in_room",
            preferences=GuestPreferences(
                dietary_restrictions=["gluten-free"],
                wake_up_time="07:00",
            ),
        )
    )


@pytest.fixture
def user_tools(user_db: HotelUserDB) -> HotelUserTools:
    """Create hotel user tools for testing"""
    return HotelUserTools(user_db)


class TestHotelUserTools:
    """Test hotel user-side tools functionality"""

    def test_view_my_preferences(self, user_tools: HotelUserTools):
        """Test viewing guest preferences"""
        prefs = user_tools.view_my_preferences()
        assert prefs.dietary_restrictions == ["gluten-free"]
        assert prefs.wake_up_time == "07:00"

    def test_check_dietary_restrictions(self, user_tools: HotelUserTools):
        """Test checking dietary restrictions"""
        restrictions = user_tools.check_dietary_restrictions()
        assert restrictions == ["gluten-free"]

    def test_check_dietary_restrictions_empty(self):
        """Test checking dietary restrictions when none set"""
        empty_db = HotelUserDB()
        tools = HotelUserTools(empty_db)
        restrictions = tools.check_dietary_restrictions()
        assert restrictions == []

    def test_update_dietary_restrictions(self, user_tools: HotelUserTools):
        """Test updating dietary restrictions"""
        result = user_tools.update_dietary_restrictions(["vegan", "nut allergy"])
        assert "vegan" in result
        assert user_tools.preferences.dietary_restrictions == ["vegan", "nut allergy"]

    def test_add_dietary_restriction(self, user_tools: HotelUserTools):
        """Test adding a single dietary restriction"""
        result = user_tools.add_dietary_restriction("lactose-free")
        assert "lactose-free" in result
        assert "lactose-free" in user_tools.preferences.dietary_restrictions
        assert "gluten-free" in user_tools.preferences.dietary_restrictions

    def test_add_dietary_restriction_duplicate(self, user_tools: HotelUserTools):
        """Test adding a duplicate dietary restriction"""
        result = user_tools.add_dietary_restriction("gluten-free")
        assert "already exists" in result
        # Should still only have one instance
        assert user_tools.preferences.dietary_restrictions.count("gluten-free") == 1

    def test_set_wake_up_time(self, user_tools: HotelUserTools):
        """Test setting wake-up time"""
        result = user_tools.set_wake_up_time("06:30")
        assert "06:30" in result
        assert user_tools.preferences.wake_up_time == "06:30"

    def test_set_room_temperature_preference(self, user_tools: HotelUserTools):
        """Test setting room temperature preference"""
        result = user_tools.set_room_temperature_preference("cool")
        assert "cool" in result
        assert user_tools.preferences.room_temperature == "cool"

    def test_set_room_temperature_preference_invalid(self, user_tools: HotelUserTools):
        """Test setting invalid room temperature preference"""
        result = user_tools.set_room_temperature_preference("freezing")
        assert "Invalid" in result
        assert user_tools.preferences.room_temperature is None

    def test_set_pillow_type(self, user_tools: HotelUserTools):
        """Test setting pillow type"""
        result = user_tools.set_pillow_type("firm")
        assert "firm" in result
        assert user_tools.preferences.pillow_type == "firm"

    def test_set_pillow_type_invalid(self, user_tools: HotelUserTools):
        """Test setting invalid pillow type"""
        result = user_tools.set_pillow_type("extra-hard")
        assert "Invalid" in result
        assert user_tools.preferences.pillow_type is None

    def test_add_special_request(self, user_tools: HotelUserTools):
        """Test adding a special request"""
        result = user_tools.add_special_request("Extra hangers in closet")
        assert "Extra hangers" in result
        assert "Extra hangers in closet" in user_tools.preferences.special_requests

    def test_clear_preferences(self, user_tools: HotelUserTools):
        """Test clearing all preferences"""
        result = user_tools.clear_preferences()
        assert "cleared" in result.lower()
        assert user_tools.preferences.dietary_restrictions is None
        assert user_tools.preferences.wake_up_time is None
        assert user_tools.preferences.room_temperature is None
        assert user_tools.preferences.pillow_type is None
        assert user_tools.preferences.special_requests is None

    def test_set_location(self, user_tools: HotelUserTools):
        """Test setting guest location"""
        result = user_tools.set_location("at_pool")
        assert "at_pool" in result
        assert user_tools.guest_context.current_location == "at_pool"

    def test_check_my_location(self, user_tools: HotelUserTools):
        """Test checking guest location"""
        location = user_tools.check_my_location()
        assert location == "in_room"

    def test_check_my_location_unknown(self):
        """Test checking location when not set"""
        empty_db = HotelUserDB()
        tools = HotelUserTools(empty_db)
        location = tools.check_my_location()
        assert location == "Unknown"

    def test_set_guest_id(self, user_tools: HotelUserTools):
        """Test setting guest ID"""
        user_tools.set_guest_id("G002")
        assert user_tools.guest_context.guest_id == "G002"


class TestHotelEnvironmentIntegration:
    """Test integration of hotel environment with user tools"""

    def test_environment_has_user_tools(self):
        """Test that environment includes user tools"""
        env = get_environment()
        assert hasattr(env, "user_tools")
        assert isinstance(env.user_tools, HotelUserTools)

    def test_environment_user_tools_accessible(self):
        """Test that user tools are accessible through environment"""
        env = get_environment()
        user_tools = env.user_tools
        assert hasattr(user_tools, "view_my_preferences")
        assert hasattr(user_tools, "update_dietary_restrictions")
        assert hasattr(user_tools, "set_wake_up_time")

    def test_environment_sync_tools_valid_guest(self):
        """Test sync_tools with valid guest ID"""
        env = get_environment()
        # Set a valid guest ID from the agent DB
        env.user_tools.set_guest_id("G001")
        # Should not raise an error
        env.sync_tools()

    def test_environment_sync_tools_invalid_guest(self):
        """Test sync_tools with invalid guest ID"""
        env = get_environment()
        # Set an invalid guest ID
        env.user_tools.set_guest_id("G999")
        # Should raise ValueError
        with pytest.raises(ValueError, match="Guest .* not found"):
            env.sync_tools()

    def test_environment_sync_tools_empty_guest_id(self):
        """Test sync_tools with empty guest ID"""
        env = get_environment()
        # Empty guest ID should not raise error
        env.sync_tools()
