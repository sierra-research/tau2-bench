"""User data models for the hospitality domain."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


class CustomerMood(str, Enum):
    """Customer's current mood."""

    HAPPY = "happy"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    DISAPPOINTED = "disappointed"
    DESPERATE = "desperate"
    UPSET = "upset"
    ANXIOUS = "anxious"


class CustomerType(str, Enum):
    """Type of customer persona."""

    REGULAR = "regular"
    VIP = "vip"
    FIRST_TIME = "first_time"
    PRICE_SENSITIVE = "price_sensitive"
    COMPLAINT_PRONE = "complaint_prone"
    FAMILY_WITH_KIDS = "family_with_kids"


class ReceivedSMS(BaseModelNoExtra):
    """An SMS message received by the customer."""

    date: str = Field(description="Date received")
    content: str = Field(description="SMS content")
    promo_code: Optional[str] = Field(default=None)
    missing_terms: Optional[str] = Field(
        default=None, description="Terms that were accidentally omitted"
    )


class CustomerContext(BaseModelNoExtra):
    """Context about the customer's situation."""

    name: str = Field(description="Customer's name")
    phone: str = Field(description="Customer's phone number")
    customer_type: CustomerType = Field(default=CustomerType.REGULAR)
    mood: CustomerMood = Field(default=CustomerMood.NEUTRAL)
    party_size: int = Field(default=2)
    has_kids: bool = Field(default=False)
    num_kids: int = Field(default=0)
    is_birthday: bool = Field(default=False)
    is_anniversary: bool = Field(default=False)
    is_business_meal: bool = Field(default=False)
    allergies: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    previous_visit_count: int = Field(default=0)
    membership_tier: Optional[str] = Field(default=None)
    points_balance: int = Field(default=0)


class CustomerExpectations(BaseModelNoExtra):
    """What the customer expects from this interaction."""

    wants_compensation: bool = Field(default=False)
    minimum_acceptable_discount: Optional[float] = Field(default=None)
    will_accept_alternative: bool = Field(default=True)
    will_threaten_review: bool = Field(default=False)
    willing_to_wait: bool = Field(default=True)
    max_wait_minutes: int = Field(default=30)


class HospitalityUserDB(DB):
    """User database for the hospitality domain."""

    context: CustomerContext = Field(
        default_factory=lambda: CustomerContext(
            name="Guest",
            phone="555-000-0000",
        )
    )
    expectations: CustomerExpectations = Field(default_factory=CustomerExpectations)
    received_sms: List[ReceivedSMS] = Field(default_factory=list)

    # Current order/visit state
    current_table_id: Optional[str] = Field(default=None)
    current_order_id: Optional[str] = Field(default=None)
    current_reservation_id: Optional[str] = Field(default=None)

    # Interaction tracking
    issues_reported: List[str] = Field(default_factory=list)
    compensation_received: List[str] = Field(default_factory=list)
    satisfaction_score: int = Field(default=5, description="1-10 satisfaction")

    def update_context(self, update_data: Dict[str, Any]) -> None:
        """Update the customer context."""
        for key, value in update_data.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)


def get_default_user_db() -> HospitalityUserDB:
    """Get a default user database instance."""
    return HospitalityUserDB()


if __name__ == "__main__":
    db = get_default_user_db()
    print(db.model_dump_json(indent=2))
