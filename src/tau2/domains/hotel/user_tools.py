from typing import List, Optional

from loguru import logger

from tau2.domains.hotel.user_data_model import GuestPreferences, HotelUserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HotelUserTools(ToolKitBase):
    """
    Provides methods for user-side actions in the hotel domain.
    These tools allow guests to manage their preferences and view their context.
    """

    db: HotelUserDB

    def __init__(self, db: HotelUserDB):
        """Initialize the Hotel User Tools."""
        super().__init__(db)

    # --- Properties ---
    @property
    def guest_context(self):
        """Returns the guest context."""
        return self.db.guest_context

    @property
    def preferences(self) -> GuestPreferences:
        """Returns the guest preferences."""
        return self.db.guest_context.preferences

    # --- Helper Methods ---
    def set_guest_id(self, guest_id: str):
        """Sets the guest ID for this user session."""
        self.db.guest_context.guest_id = guest_id
        logger.info(f"Guest ID set to: {guest_id}")

    # --- READ TOOLS ---
    @is_tool(ToolType.READ)
    def view_my_preferences(self) -> GuestPreferences:
        """
        View your current guest preferences including dietary restrictions,
        wake-up time, room temperature, pillow type, and special requests.

        Returns:
            Your current guest preferences
        """
        return self.preferences.model_copy(deep=True)

    @is_tool(ToolType.READ)
    def check_dietary_restrictions(self) -> List[str]:
        """
        Check your current dietary restrictions.

        Returns:
            List of dietary restrictions or empty list if none set
        """
        return self.preferences.dietary_restrictions or []

    # --- WRITE TOOLS ---
    @is_tool(ToolType.WRITE)
    def update_dietary_restrictions(self, restrictions: List[str]) -> str:
        """
        Update your dietary restrictions.

        Args:
            restrictions: List of dietary restrictions (e.g., ["vegan", "nut allergy"])

        Returns:
            Confirmation message
        """
        self.preferences.dietary_restrictions = restrictions
        logger.info(
            f"Updated dietary restrictions for guest {self.guest_context.guest_id}: {restrictions}"
        )
        return f"Dietary restrictions updated to: {', '.join(restrictions)}"

    @is_tool(ToolType.WRITE)
    def add_dietary_restriction(self, restriction: str) -> str:
        """
        Add a single dietary restriction to your existing list.

        Args:
            restriction: Dietary restriction to add (e.g., "gluten-free")

        Returns:
            Confirmation message
        """
        if self.preferences.dietary_restrictions is None:
            self.preferences.dietary_restrictions = []

        if restriction not in self.preferences.dietary_restrictions:
            self.preferences.dietary_restrictions.append(restriction)
            logger.info(
                f"Added dietary restriction for guest {self.guest_context.guest_id}: {restriction}"
            )
            return f"Added dietary restriction: {restriction}"
        else:
            return f"Dietary restriction '{restriction}' already exists"

    @is_tool(ToolType.WRITE)
    def set_wake_up_time(self, time: str) -> str:
        """
        Set your preferred wake-up time.

        Args:
            time: Wake-up time in HH:MM format (e.g., "07:30")

        Returns:
            Confirmation message
        """
        self.preferences.wake_up_time = time
        logger.info(f"Set wake-up time for guest {self.guest_context.guest_id}: {time}")
        return f"Wake-up time set to: {time}"

    @is_tool(ToolType.WRITE)
    def set_room_temperature_preference(self, temperature: str) -> str:
        """
        Set your preferred room temperature.

        Args:
            temperature: Temperature preference (e.g., "cool", "moderate", "warm")

        Returns:
            Confirmation message
        """
        valid_temps = ["cool", "moderate", "warm"]
        if temperature.lower() not in valid_temps:
            return f"Invalid temperature preference. Please choose from: {', '.join(valid_temps)}"

        self.preferences.room_temperature = temperature.lower()
        logger.info(
            f"Set room temperature preference for guest {self.guest_context.guest_id}: {temperature}"
        )
        return f"Room temperature preference set to: {temperature}"

    @is_tool(ToolType.WRITE)
    def set_pillow_type(self, pillow_type: str) -> str:
        """
        Set your preferred pillow type.

        Args:
            pillow_type: Pillow type preference (e.g., "soft", "medium", "firm")

        Returns:
            Confirmation message
        """
        valid_types = ["soft", "medium", "firm"]
        if pillow_type.lower() not in valid_types:
            return f"Invalid pillow type. Please choose from: {', '.join(valid_types)}"

        self.preferences.pillow_type = pillow_type.lower()
        logger.info(
            f"Set pillow type preference for guest {self.guest_context.guest_id}: {pillow_type}"
        )
        return f"Pillow type preference set to: {pillow_type}"

    @is_tool(ToolType.WRITE)
    def add_special_request(self, request: str) -> str:
        """
        Add a special request to your guest profile.

        Args:
            request: Special request description

        Returns:
            Confirmation message
        """
        if self.preferences.special_requests is None:
            self.preferences.special_requests = []

        self.preferences.special_requests.append(request)
        logger.info(
            f"Added special request for guest {self.guest_context.guest_id}: {request}"
        )
        return f"Special request added: {request}"

    @is_tool(ToolType.WRITE)
    def clear_preferences(self) -> str:
        """
        Clear all your preferences.

        Returns:
            Confirmation message
        """
        self.preferences.dietary_restrictions = None
        self.preferences.wake_up_time = None
        self.preferences.room_temperature = None
        self.preferences.pillow_type = None
        self.preferences.special_requests = None
        logger.info(f"Cleared all preferences for guest {self.guest_context.guest_id}")
        return "All preferences cleared"

    # --- Location Tools ---
    @is_tool(ToolType.WRITE)
    def set_location(self, location: str) -> str:
        """
        Set your current location in the hotel.

        Args:
            location: Your current location (e.g., "in_room", "at_pool", "lobby")

        Returns:
            Confirmation message
        """
        self.guest_context.current_location = location
        logger.info(f"Guest {self.guest_context.guest_id} location set to: {location}")
        return f"Location set to: {location}"

    @is_tool(ToolType.READ)
    def check_my_location(self) -> str:
        """
        Check your current location in the hotel.

        Returns:
            Your current location or "Unknown" if not set
        """
        return self.guest_context.current_location or "Unknown"
