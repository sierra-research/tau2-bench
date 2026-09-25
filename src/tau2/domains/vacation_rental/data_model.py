"""Data models for the vacation rental domain."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from tau2.domains.vacation_rental.utils import VACATION_RENTAL_DB_PATH
from tau2.environment.db import DB

IssueType = Literal[
    "property_condition",
    "cleanliness",
    "amenity_malfunction",
    "not_as_described",
    "rule_violation",
    "safety_concern",
    "cancellation_dispute",
]
IssueSeverity = Literal["minor", "moderate", "major", "critical"]
EvidenceStatus = Literal["pending", "validated", "invalidated", "inconclusive"]
IssueStatus = Literal["open", "investigating", "resolved", "escalated"]
SituationType = Literal[
    "cancellation_exception",
    "partial_refund",
    "issue_compensation",
    "early_checkin_request",
    "late_checkout_request",
    "rule_violation_response",
]
DecisionOutcome = Literal["approve", "approve_partial", "deny", "defer_to_policy"]


class UserName(BaseModel):
    """Represents a user's full name."""

    first_name: str = Field(description="User's first name")
    last_name: str = Field(description="User's last name")


class PaymentMethodBase(BaseModel):
    """Base class for payment methods."""

    source: str = Field(
        description="Payment method type identifier (e.g., 'credit_card', 'bank_account')"
    )
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
        default=None,
        description="Who initiated the cancellation ('guest' or 'host'). Only populated if status is 'cancelled'.",
    )


class HostPhilosophy(BaseModel):
    """Host's business philosophy and approach."""

    primary_focus: Literal[
        "reviews", "revenue", "relationships", "policy_adherence"
    ] = Field(description="What the host prioritizes in their business")
    risk_tolerance: Literal["low", "medium", "high"] = Field(
        description="How willing the host is to make exceptions"
    )


class FlexibilitySettings(BaseModel):
    """Host's flexibility on refunds and accommodations."""

    refund_flexibility: Literal["none", "case_by_case", "generous"] = Field(
        description="How flexible the host is with refunds beyond policy"
    )
    max_goodwill_refund_pct: int = Field(
        ge=0,
        le=100,
        description="Maximum percentage of booking the host will refund as goodwill (0-100)",
    )
    repeat_guest_bonus_pct: int = Field(
        default=0,
        ge=0,
        le=30,
        description="Additional flexibility percentage for repeat guests (0-30)",
    )


class HostProfile(BaseModel):
    """Host preferences that influence agent decision-making."""

    host_user_id: str = Field(description="User ID of the host")
    philosophy: HostPhilosophy = Field(description="Host's business philosophy")
    flexibility_settings: FlexibilitySettings = Field(
        description="Host's flexibility settings"
    )
    hard_limits: list[str] = Field(
        default_factory=list,
        description="Things the host will never budge on (e.g., 'no_parties', 'no_pets')",
    )
    soft_spots: list[str] = Field(
        default_factory=list,
        description="Things that sway the host toward leniency (e.g., 'medical_emergency', 'repeat_guests')",
    )
    deal_breakers: list[str] = Field(
        default_factory=list,
        description="Guest behaviors that trigger strict enforcement (e.g., 'dishonesty', 'rule_violations')",
    )
    dispute_resolution_preference: Literal[
        "platform_decides", "host_involved", "always_escalate"
    ] = Field(description="How the host prefers disputes to be handled")


class Issue(BaseModel):
    """A reported problem with a reservation."""

    issue_id: str = Field(description="Unique identifier for the issue")
    reservation_id: str = Field(description="Associated reservation ID")
    guest_user_id: str = Field(description="User ID of the guest who reported")
    reported_by: Literal["guest", "host"] = Field(description="Who reported the issue")
    issue_type: IssueType = Field(description="Category of the issue")
    description: str = Field(description="Description of the problem")
    severity: IssueSeverity = Field(description="Severity level of the issue")
    evidence_submitted: bool = Field(
        default=False, description="Whether evidence was provided"
    )
    evidence_status: EvidenceStatus | None = Field(
        default=None, description="Status of evidence validation"
    )
    validation_result: str | None = Field(
        default=None, description="Details of what evidence showed"
    )
    status: IssueStatus = Field(
        default="open", description="Current status of the issue"
    )
    resolution: str | None = Field(
        default=None, description="How the issue was resolved"
    )
    resolution_amount: float | None = Field(
        default=None, description="Amount refunded or credited as resolution"
    )
    created_at: str = Field(description="When the issue was reported (ISO format)")


class GuestHistory(BaseModel):
    """Aggregated guest stay history for host consideration."""

    guest_user_id: str = Field(description="User ID of the guest")
    total_stays: int = Field(default=0, description="Total number of completed stays")
    stays_by_host: dict[str, int] = Field(
        default_factory=dict,
        description="Number of stays per host (host_user_id -> count)",
    )
    issues_reported: int = Field(
        default=0, description="Number of issues this guest has reported"
    )
    cancellation_count: int = Field(
        default=0, description="Number of cancellations by this guest"
    )


class HostDecision(BaseModel):
    """Pre-computed host decision for a specific situation."""

    decision_id: str = Field(description="Unique identifier for this decision")
    host_user_id: str = Field(description="Host this decision applies to")
    situation_type: SituationType = Field(description="Type of situation")
    guest_context: str | None = Field(
        default=None,
        description="Guest context that triggers this decision (e.g., 'repeat_guest', 'has_documentation')",
    )
    decision: DecisionOutcome = Field(description="The host's decision")
    approved_amount_pct: int | None = Field(
        default=None,
        description="Percentage of requested amount approved (for partial approvals)",
    )
    reasoning: str = Field(
        description="Explanation of why the host would make this decision"
    )
    conditions: list[str] = Field(
        default_factory=list, description="Conditions attached to the decision"
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
    host_profiles: dict[str, HostProfile] = Field(
        default_factory=dict,
        description="Host profiles indexed by host user ID",
    )
    issues: dict[str, Issue] = Field(
        default_factory=dict,
        description="Issues indexed by issue ID",
    )
    guest_history: dict[str, GuestHistory] = Field(
        default_factory=dict,
        description="Guest history indexed by guest user ID",
    )
    host_decisions: dict[str, HostDecision] = Field(
        default_factory=dict,
        description="Pre-computed host decisions indexed by decision ID",
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_guests = sum(1 for u in self.users.values() if u.reservations)
        num_hosts = sum(
            1
            for u in self.users.values()
            if any(lst.host_user_id == u.user_id for lst in self.listings.values())
        )
        return {
            "num_users": len(self.users),
            "num_listings": len(self.listings),
            "num_reservations": len(self.reservations),
            "num_guests": num_guests,
            "num_hosts": num_hosts,
        }


def get_db() -> VacationRentalDB:
    """Load the vacation rental database."""
    return VacationRentalDB.load(VACATION_RENTAL_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
