"""Data model for the hospitality domain."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.hospitality.utils import HOSPITALITY_DB_PATH
from tau2.environment.db import DB

ReservationStatus = Literal["confirmed", "cancelled", "checked_in", "checked_out"]


class Hotel(BaseModel):
    """Static information about the hotel."""

    name: str = Field(description="Name of the hotel")
    address: str = Field(description="Street address of the hotel")
    city: str = Field(description="City the hotel is located in")
    phone: str = Field(description="Front desk phone number")
    email: str = Field(description="Front desk email address")
    current_time: str = Field(
        description="Current time at the hotel, format YYYY-MM-DD HH:MM:SS"
    )
    check_in_time: str = Field(description="Earliest check-in time, format HH:MM")
    check_out_time: str = Field(description="Latest check-out time, format HH:MM")


class RoomType(BaseModel):
    """A bookable room category."""

    room_type_id: str = Field(description="Unique identifier for the room type")
    name: str = Field(description="Display name of the room type")
    capacity: int = Field(description="Maximum number of guests the room sleeps")
    count: int = Field(description="Number of rooms of this type in the hotel")
    amenities: list[str] = Field(description="Amenities included with the room")
    rates: dict[str, int] = Field(
        description="Nightly rate in EUR per rate plan, keyed by rate plan ID"
    )


class RatePlan(BaseModel):
    """A rate plan controlling price, cancellation and modification rules."""

    rate_plan_id: str = Field(description="Unique identifier for the rate plan")
    name: str = Field(description="Display name of the rate plan")
    refundable: bool = Field(
        description="Whether cancellations are eligible for a refund"
    )
    date_changes_allowed: bool = Field(
        description="Whether the stay dates of a reservation can be changed"
    )
    breakfast_included: bool = Field(
        description="Whether breakfast is included for all guests"
    )
    description: str = Field(description="Short description of the plan conditions")


class StayPackage(BaseModel):
    """A curated stay package tied to a specific room type."""

    package_id: str = Field(description="Unique identifier for the package")
    name: str = Field(description="Display name of the package")
    room_type_id: str = Field(description="Room type the package is sold with")
    nightly_rate: int = Field(description="Nightly rate in EUR, inclusions included")
    min_nights: int = Field(description="Minimum number of nights for the package")
    inclusions: list[str] = Field(description="What the package includes")
    description: str = Field(description="Short description of the package")


class ExtraService(BaseModel):
    """An extra service that can be added to a reservation."""

    service_id: str = Field(description="Unique identifier for the service")
    name: str = Field(description="Display name of the service")
    unit_price: int = Field(description="Price in EUR per unit")
    pricing_unit: str = Field(
        description="What one unit covers, e.g. 'per night', 'per person per night', "
        "'per trip', 'per stay'"
    )
    description: str = Field(description="Short description of the service")


class Guest(BaseModel):
    """A guest profile."""

    guest_id: str = Field(description="Unique identifier for the guest")
    name: str = Field(description="Full name of the guest")
    email: str = Field(description="Email address of the guest")
    phone: str = Field(description="Phone number of the guest")


class ReservationExtra(BaseModel):
    """An extra service line on a reservation."""

    service_id: str = Field(description="ID of the booked extra service")
    quantity: int = Field(description="Number of units booked")
    amount: int = Field(description="Total price in EUR for this line")


class Reservation(BaseModel):
    """A room reservation."""

    reservation_id: str = Field(
        description="Confirmation number of the reservation, e.g. HV-1001"
    )
    guest_id: str = Field(description="ID of the guest the reservation belongs to")
    room_type_id: str = Field(description="ID of the reserved room type")
    rate_plan_id: Optional[str] = Field(
        default=None, description="ID of the rate plan; None for package bookings"
    )
    package_id: Optional[str] = Field(
        default=None, description="ID of the stay package; None for regular bookings"
    )
    check_in: str = Field(description="Check-in date, format YYYY-MM-DD")
    check_out: str = Field(description="Check-out date, format YYYY-MM-DD")
    num_guests: int = Field(description="Number of guests staying")
    nightly_rate: int = Field(description="Nightly room rate in EUR")
    extras: list[ReservationExtra] = Field(
        default_factory=list, description="Extra services booked on the reservation"
    )
    total_amount: int = Field(
        description="Total price in EUR: nights * nightly_rate + extras"
    )
    status: ReservationStatus = Field(description="Status of the reservation")
    refund_amount: Optional[int] = Field(
        default=None, description="Refund in EUR granted on cancellation"
    )
    online_checkin_completed: bool = Field(
        default=False,
        description="Whether online check-in has been completed for this reservation",
    )


class KnowledgeBaseArticle(BaseModel):
    """An article in the hotel knowledge base."""

    article_id: str = Field(description="Unique identifier for the article")
    title: str = Field(description="Title of the article")
    category: str = Field(description="Category of the article")
    content: str = Field(description="Content of the article")


class HospitalityDB(DB):
    """Database for the hospitality domain. A single hotel."""

    hotel: Hotel = Field(description="Static hotel information")
    room_types: dict[str, RoomType] = Field(
        description="Room types, keyed by room type ID"
    )
    rate_plans: dict[str, RatePlan] = Field(
        description="Rate plans, keyed by rate plan ID"
    )
    stay_packages: dict[str, StayPackage] = Field(
        description="Stay packages, keyed by package ID"
    )
    extra_services: dict[str, ExtraService] = Field(
        description="Extra services, keyed by service ID"
    )
    guests: dict[str, Guest] = Field(description="Guest profiles, keyed by guest ID")
    reservations: dict[str, Reservation] = Field(
        description="Reservations, keyed by confirmation number"
    )
    knowledge_base: dict[str, KnowledgeBaseArticle] = Field(
        description="Hotel knowledge base articles, keyed by article ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the database."""
        return {
            "num_room_types": len(self.room_types),
            "num_rate_plans": len(self.rate_plans),
            "num_stay_packages": len(self.stay_packages),
            "num_extra_services": len(self.extra_services),
            "num_guests": len(self.guests),
            "num_reservations": len(self.reservations),
            "num_knowledge_base_articles": len(self.knowledge_base),
        }


def get_db() -> HospitalityDB:
    """Load the hospitality database from db.json."""
    return HospitalityDB.load(HOSPITALITY_DB_PATH)
