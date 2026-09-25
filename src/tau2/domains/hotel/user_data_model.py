from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tau2.domains.hotel.utils import HOTEL_USER_DB_PATH
from tau2.environment.db import DB


class GuestPreferences(BaseModel):
    """Represents guest preferences and special requests"""

    dietary_restrictions: Optional[List[str]] = Field(
        default=None,
        description="List of dietary restrictions (e.g., vegan, gluten-free, nut allergy)",
    )
    wake_up_time: Optional[str] = Field(
        default=None, description="Preferred wake-up time in HH:MM format"
    )
    room_temperature: Optional[str] = Field(
        default=None,
        description="Preferred room temperature (e.g., cool, moderate, warm)",
    )
    pillow_type: Optional[str] = Field(
        default=None, description="Preferred pillow type (e.g., soft, medium, firm)"
    )
    special_requests: Optional[List[str]] = Field(
        default=None, description="Other special requests"
    )


class GuestContext(BaseModel):
    """Represents the user-side context for a hotel guest"""

    guest_id: str = Field(default="", description="Unique identifier for the guest")
    current_location: Optional[str] = Field(
        default=None,
        description="Current location in hotel (e.g., 'in_room', 'at_pool', 'lobby')",
    )
    preferences: GuestPreferences = Field(
        default_factory=GuestPreferences, description="Guest preferences"
    )


class HotelUserDB(DB):
    """User-side database for hotel domain containing guest-specific data"""

    guest_context: GuestContext = Field(
        default_factory=GuestContext,
        description="Current guest context including preferences",
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "guest_id": self.guest_context.guest_id,
            "has_preferences": any(
                [
                    self.guest_context.preferences.dietary_restrictions,
                    self.guest_context.preferences.wake_up_time,
                    self.guest_context.preferences.room_temperature,
                    self.guest_context.preferences.pillow_type,
                    self.guest_context.preferences.special_requests,
                ]
            ),
        }


def get_user_db() -> HotelUserDB:
    """Load and return the hotel user database"""
    return HotelUserDB.load(HOTEL_USER_DB_PATH)


if __name__ == "__main__":
    db = get_user_db()
    print(db.get_statistics())
