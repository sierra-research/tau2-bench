from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.hotel.utils import HOTEL_DB_PATH
from tau2.environment.db import DB

# Literals for type safety
LoyaltyTier = Literal["platinum", "gold", "silver", "regular"]
ServiceRequestType = Literal[
    "towels",
    "pillows",
    "maintenance",
    "housekeeping",
    "wake_up_call",
    "room_service",
    "other",
]
ServiceRequestStatus = Literal["pending", "in_progress", "completed", "cancelled"]
ExperienceCategory = Literal["dining", "spa", "tours", "activities", "transportation"]
PaymentMethodType = Literal["credit_card", "apple_pay", "google_pay", "room_charge"]
BookingStatus = Literal["confirmed", "cancelled", "completed"]


class GuestName(BaseModel):
    """Represents a guest's full name"""

    first_name: str = Field(description="Guest's first name")
    last_name: str = Field(description="Guest's last name")


class RoomBooking(BaseModel):
    """Represents a room reservation"""

    booking_id: str = Field(description="Unique booking identifier")
    room_number: str = Field(description="Room number")
    room_type: str = Field(description="Type of room (e.g., standard, deluxe, suite)")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(description="Number of guests")
    rate_per_night: float = Field(description="Nightly rate in dollars")


class Guest(BaseModel):
    """Represents a hotel guest with profile and booking information"""

    guest_id: str = Field(description="Unique identifier for the guest")
    name: GuestName = Field(description="Guest's full name")
    email: str = Field(description="Guest's email address")
    phone: str = Field(description="Guest's phone number")
    loyalty_tier: LoyaltyTier = Field(description="Guest's loyalty program tier")
    loyalty_points: int = Field(description="Current loyalty points balance")
    current_booking: RoomBooking = Field(description="Current room reservation")
    service_requests: List[str] = Field(
        default_factory=list, description="List of service request IDs"
    )
    experience_bookings: List[str] = Field(
        default_factory=list, description="List of experience booking IDs"
    )


class Amenity(BaseModel):
    """Represents a hotel amenity or facility"""

    amenity_id: str = Field(description="Unique identifier for the amenity")
    name: str = Field(description="Name of the amenity (e.g., Pool, Gym, Spa)")
    location: str = Field(
        description="Location within the hotel (e.g., 3rd Floor, Rooftop)"
    )
    hours_open: str = Field(description="Opening time in HH:MM format")
    hours_close: str = Field(description="Closing time in HH:MM format")
    description: str = Field(
        description="Description of the amenity and what it offers"
    )
    capacity: Optional[int] = Field(
        default=None, description="Maximum capacity (if applicable)"
    )
    currently_available: bool = Field(
        description="Whether the amenity is currently accessible"
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional amenity-specific information"
    )


class ServiceRequest(BaseModel):
    """Represents a guest service request"""

    request_id: str = Field(description="Unique identifier for the service request")
    guest_id: str = Field(description="ID of the guest making the request")
    request_type: ServiceRequestType = Field(description="Type of service requested")
    details: str = Field(description="Detailed description of the request")
    status: ServiceRequestStatus = Field(description="Current status of the request")
    created_at: str = Field(
        description="Timestamp when request was created (ISO format)"
    )
    completed_at: Optional[str] = Field(
        default=None, description="Timestamp when request was completed (ISO format)"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or staff comments"
    )


class Experience(BaseModel):
    """Represents a local experience available for booking"""

    experience_id: str = Field(description="Unique identifier for the experience")
    name: str = Field(description="Name of the experience")
    category: ExperienceCategory = Field(description="Category of the experience")
    description: str = Field(description="Detailed description")
    location: str = Field(description="Location/address of the experience")
    price: float = Field(description="Price per person in dollars")
    duration: str = Field(description="Duration (e.g., 2 hours, Half day)")
    available_times: List[str] = Field(
        description="Available time slots (e.g., ['09:00', '14:00', '18:00'])"
    )
    max_participants: int = Field(description="Maximum number of participants")
    min_age: Optional[int] = Field(default=None, description="Minimum age requirement")
    cancellation_hours: int = Field(
        description="Hours before start time when cancellation is allowed"
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional information"
    )


class PaymentInfo(BaseModel):
    """Represents payment information for a booking"""

    method: PaymentMethodType = Field(description="Payment method used")
    amount: float = Field(description="Amount paid in dollars")
    transaction_id: str = Field(description="Transaction identifier")
    timestamp: str = Field(description="Payment timestamp (ISO format)")


class ExperienceBooking(BaseModel):
    """Represents a booked experience"""

    booking_id: str = Field(description="Unique booking identifier")
    guest_id: str = Field(description="ID of the guest who booked")
    experience_id: str = Field(description="ID of the experience booked")
    date: str = Field(description="Date of the experience in YYYY-MM-DD format")
    time: str = Field(description="Time of the experience in HH:MM format")
    participants: int = Field(description="Number of participants")
    total_price: float = Field(description="Total price for all participants")
    status: BookingStatus = Field(description="Booking status")
    payment: PaymentInfo = Field(description="Payment information")
    created_at: str = Field(description="Booking creation timestamp (ISO format)")
    cancelled_at: Optional[str] = Field(
        default=None, description="Cancellation timestamp if cancelled (ISO format)"
    )


class StaffRoute(BaseModel):
    """Represents a query routed to hotel staff"""

    route_id: str = Field(description="Unique identifier for the routed message")
    guest_id: str = Field(description="ID of the guest who sent the message")
    message_type: str = Field(
        description="Type of message (e.g., complaint, special_request, question)"
    )
    content: str = Field(description="Message content")
    timestamp: str = Field(description="When the message was routed (ISO format)")
    handled: bool = Field(default=False, description="Whether staff has responded")


class LocalInfo(BaseModel):
    """Represents local information and points of interest"""

    category: str = Field(description="Category (e.g., pharmacy, ATM, restaurant)")
    name: str = Field(description="Name of the place")
    address: str = Field(description="Street address")
    distance: str = Field(
        description="Distance from hotel (e.g., 0.3 miles, 5 minute walk)"
    )
    hours: Optional[str] = Field(default=None, description="Operating hours")
    additional_info: Optional[str] = Field(
        default=None, description="Additional information"
    )


class HotelDB(DB):
    """Main database containing all hotel-related data"""

    guests: Dict[str, Guest] = Field(
        description="Dictionary of guests indexed by guest_id"
    )
    amenities: Dict[str, Amenity] = Field(
        description="Dictionary of amenities indexed by amenity_id"
    )
    service_requests: Dict[str, ServiceRequest] = Field(
        description="Dictionary of service requests indexed by request_id"
    )
    experiences: Dict[str, Experience] = Field(
        description="Dictionary of experiences indexed by experience_id"
    )
    experience_bookings: Dict[str, ExperienceBooking] = Field(
        description="Dictionary of experience bookings indexed by booking_id"
    )
    staff_routes: Dict[str, StaffRoute] = Field(
        description="Dictionary of staff-routed messages indexed by route_id"
    )
    local_info: List[LocalInfo] = Field(
        default_factory=list, description="List of local points of interest"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics"""
        return {
            "num_guests": len(self.guests),
            "num_amenities": len(self.amenities),
            "num_service_requests": len(self.service_requests),
            "num_experiences": len(self.experiences),
            "num_experience_bookings": len(self.experience_bookings),
            "num_staff_routes": len(self.staff_routes),
            "num_local_info": len(self.local_info),
        }


def get_db():
    """Load and return the hotel database"""
    return HotelDB.load(HOTEL_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
