"""Data models for the vacation rental domain."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from tau2.domains.vacation_rental.utils import VACATION_RENTAL_DB_PATH
from tau2.environment.db import DB


class UserName(BaseModel):
    """Represents a user's full name."""

    first_name: str = Field(description="User's first name")
    last_name: str = Field(description="User's last name")


class PaymentMethodBase(BaseModel):
    """Base class for payment methods."""

    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    """Credit card payment method."""

    source: Literal["credit_card"] = Field(
        description="Indicates this is a credit card payment method"
    )
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")
    expiration: str = Field(description="Expiration date in YYYY-MM format")


class BankAccount(PaymentMethodBase):
    """Bank account payment method."""

    source: Literal["bank_account"] = Field(
        description="Indicates this is a bank account payment method"
    )
    last_four: str = Field(description="Last four digits of the bank account")


PaymentMethod = Union[CreditCard, BankAccount]


class User(BaseModel):
    """Represents a user (guest or host) with their profile information."""

    user_id: str = Field(description="Unique identifier for the user")
    name: UserName = Field(description="User's full name")
    email: str = Field(description="User's email address")
    phone: str = Field(description="User's phone number")
    payment_methods: dict[str, PaymentMethod] = Field(
        description="Dictionary of payment methods indexed by payment method ID"
    )
    reservations: list[str] = Field(
        description="List of reservation IDs associated with this user"
    )


class ListingAddress(BaseModel):
    """Represents a listing's physical address."""

    address1: str = Field(description="Primary address line")
    address2: str | None = Field(default=None, description="Secondary address line")
    city: str = Field(description="City name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")
    country: str = Field(description="Country name")


CancellationPolicy = Literal["flexible", "moderate", "firm", "strict"]


class Listing(BaseModel):
    """Represents a vacation rental listing."""

    listing_id: str = Field(description="Unique identifier for the listing")
    host_user_id: str = Field(description="User ID of the host")
    title: str = Field(description="Title of the listing")
    address: ListingAddress = Field(description="Physical address of the listing")
    nightly_rate: float = Field(description="Price per night in USD")
    cancellation_policy: CancellationPolicy = Field(
        description="Cancellation policy type: flexible, moderate, firm, or strict"
    )


ReservationStatus = Literal["pending", "confirmed", "cancelled", "completed"]


class Reservation(BaseModel):
    """Represents a vacation rental reservation."""

    reservation_id: str = Field(description="Unique identifier for the reservation")
    guest_user_id: str = Field(description="User ID of the guest")
    listing_id: str = Field(description="Listing ID for this reservation")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format")
    total_amount: float = Field(description="Total amount for the reservation in USD")
    amount_paid: float = Field(description="Amount already paid in USD")
    status: ReservationStatus = Field(
        description="Status of the reservation: pending, confirmed, cancelled, or completed"
    )
    created_at: str = Field(
        description="Timestamp when the reservation was created in ISO format"
    )
    payment_method_id: str = Field(
        description="Payment method ID used for this reservation"
    )
    refund_amount: float | None = Field(
        default=None, description="Refund amount if cancelled"
    )
    cancelled_by: str | None = Field(
        default=None, description="Who cancelled: 'guest' or 'host'"
    )


class VacationRentalDB(DB):
    """Database containing all vacation rental data including users, listings, and reservations."""

    users: dict[str, User] = Field(
        description="Dictionary of all users indexed by user ID"
    )
    listings: dict[str, Listing] = Field(
        description="Dictionary of all listings indexed by listing ID"
    )
    reservations: dict[str, Reservation] = Field(
        description="Dictionary of all reservations indexed by reservation ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_users = len(self.users)
        num_listings = len(self.listings)
        num_reservations = len(self.reservations)
        num_guests = sum(1 for u in self.users.values() if len(u.reservations) > 0)
        num_hosts = sum(
            1
            for u in self.users.values()
            if any(
                l.host_user_id == u.user_id for l in self.listings.values()
            )
        )
        return {
            "num_users": num_users,
            "num_listings": num_listings,
            "num_reservations": num_reservations,
            "num_guests": num_guests,
            "num_hosts": num_hosts,
        }


def get_db() -> VacationRentalDB:
    """Load the vacation rental database."""
    return VacationRentalDB.load(VACATION_RENTAL_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
