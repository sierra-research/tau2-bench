"""Data models for the hospitality domain (Berkeley Hot Pot restaurant)."""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from tau2.domains.hospitality.utils import HOSPITALITY_DB_PATH
from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


# ============== Enums ==============


class TableType(str, Enum):
    """Type of table in the restaurant."""

    A_TYPE = "A"  # Standard 4-person booth
    B_TYPE = "B"  # Square 6-person booth
    C_TYPE = "C"  # Long 8-12 person table


class TableStatus(str, Enum):
    """Status of a table."""

    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    CLEANING = "cleaning"


class ReservationStatus(str, Enum):
    """Status of a reservation."""

    CONFIRMED = "confirmed"
    SEATED = "seated"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class OrderStatus(str, Enum):
    """Status of an order."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SERVED = "served"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MemberTier(str, Enum):
    """Membership tier levels."""

    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"
    DIAMOND = "Diamond"


class StaffRole(str, Enum):
    """Staff role levels with different authorities."""

    SERVER = "Server"
    HOST = "Host"
    MANAGER = "Manager"


class IncidentType(str, Enum):
    """Types of service incidents."""

    SLOW_SERVICE = "slow_service"
    SPILL = "spill"
    WRONG_ORDER = "wrong_order"
    FOOD_QUALITY = "food_quality"
    ALLERGY_ISSUE = "allergy_issue"
    BILLING_ERROR = "billing_error"
    CAKE_DAMAGE = "cake_damage"
    PROPERTY_DAMAGE = "property_damage"
    RESERVATION_ISSUE = "reservation_issue"
    FOOD_SAFETY = "food_safety"
    OTHER = "other"


class KitchenStatus(str, Enum):
    """Kitchen operational status for testing internal coordination scenarios."""

    NORMAL = "normal"  # Kitchen operating normally
    ORDER_OVERLOAD = "order_overload"  # Too many orders, long wait times
    UNDERSTAFFED = "understaffed"  # Short on staff
    EQUIPMENT_FAILURE = "equipment_failure"  # Equipment broken
    STAFF_ATTITUDE = "staff_attitude"  # Staff being uncooperative/rude
    STAFF_WALKOUT = "staff_walkout"  # Staff quit mid-shift
    PRINTER_ERROR = "printer_error"  # Order printer malfunction


# ============== Data Models ==============


class SoupBase(BaseModelNoExtra):
    """A soup base option."""

    id: str = Field(description="Unique identifier for the soup base")
    name: str = Field(description="Display name of the soup base")
    spicy_level: int = Field(description="Spiciness level 0-5")
    allergies: List[str] = Field(default_factory=list, description="Common allergens")
    hidden_ingredients: List[str] = Field(
        default_factory=list, description="Hidden ingredients that may not be on label"
    )
    contains_pre_processed: bool = Field(
        default=True,
        description="Whether item contains pre-processed ingredients with hidden sub-ingredients",
    )
    gluten_safe: bool = Field(
        default=False,
        description="Whether item can be confirmed gluten-free (only Plain Water is true)",
    )
    prices: Dict[str, float] = Field(description="Prices for quarter/half/whole pot")


class MenuItem(BaseModelNoExtra):
    """A menu item."""

    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    category: str = Field(description="Category (protein, seafood, veggie, etc.)")
    price: float = Field(description="Price in USD")
    allergies: List[str] = Field(default_factory=list, description="Common allergens")
    contains_pre_processed: bool = Field(
        default=False,
        description="Whether item contains pre-processed ingredients",
    )
    gluten_safe: bool = Field(
        default=True,
        description="Whether item can be confirmed gluten-free",
    )
    note: Optional[str] = Field(
        default=None, description="Special notes about the item"
    )


class LunchSpecial(BaseModelNoExtra):
    """Lunch special combo configuration."""

    price: float = Field(description="Price per combo")
    availability: str = Field(description="When available")
    holiday_excluded: bool = Field(
        default=True, description="Not available on federal holidays"
    )


class Table(BaseModelNoExtra):
    """A table in the restaurant."""

    table_id: str = Field(description="Table identifier like A1, B3, C1")
    table_type: TableType = Field(description="Type of table")
    std_capacity: int = Field(
        description="Standard seating capacity without extra chairs"
    )
    std_expansion: int = Field(
        description="Capacity with default extra chairs (recommended)"
    )
    max_squeeze: int = Field(
        description="Historical max capacity - not recommended, very crowded"
    )
    status: TableStatus = Field(default=TableStatus.AVAILABLE)
    current_party_size: int = Field(default=0)


class Customer(BaseModelNoExtra):
    """A customer/member."""

    customer_id: str = Field(description="Unique customer ID")
    name: str = Field(description="Customer name")
    phone: str = Field(description="Phone number")
    email: Optional[str] = Field(default=None)
    tier: MemberTier = Field(default=MemberTier.BRONZE)
    points: int = Field(default=0)
    birth_month: Optional[str] = Field(default=None)
    annual_spent: float = Field(default=0.0)
    visit_count: int = Field(default=0)
    notes: Optional[str] = Field(
        default=None, description="Special notes about customer"
    )


class Reservation(BaseModelNoExtra):
    """A table reservation."""

    reservation_id: str = Field(description="Unique reservation ID")
    customer_name: str = Field(description="Name for reservation")
    phone: str = Field(description="Contact phone")
    party_size: int = Field(description="Number of guests")
    date: str = Field(description="Reservation date YYYY-MM-DD")
    time: str = Field(description="Reservation time HH:MM")
    table_id: Optional[str] = Field(default=None, description="Assigned table")
    status: ReservationStatus = Field(default=ReservationStatus.CONFIRMED)
    special_occasion: Optional[str] = Field(
        default=None, description="Birthday, anniversary, etc."
    )
    num_kids: int = Field(default=0)
    num_high_chairs: int = Field(default=0)
    notes: Optional[str] = Field(default=None)
    has_cake: bool = Field(default=False, description="Customer bringing birthday cake")
    cake_type: Optional[str] = Field(default=None, description="regular or ice_cream")


class OrderItem(BaseModelNoExtra):
    """An item in an order."""

    item_id: str = Field(description="Menu item ID")
    name: str = Field(description="Item name")
    quantity: int = Field(default=1)
    price: float = Field(description="Unit price")
    status: str = Field(default="ordered")
    notes: Optional[str] = Field(default=None)


class Order(BaseModelNoExtra):
    """A customer order."""

    order_id: str = Field(description="Unique order ID")
    table_id: str = Field(description="Table ID")
    party_size: int = Field(default=1, description="Number of guests (used for sauce bar charge $2/person)")
    has_member: bool = Field(default=False, description="Whether table has a linked member account")
    customer_id: Optional[str] = Field(default=None)
    items: List[OrderItem] = Field(default_factory=list)
    subtotal: float = Field(default=0.0)
    sauce_bar_charge: float = Field(default=0.0, description="Sauce bar charge = party_size * $2")
    discount_applied: Optional[str] = Field(default=None)
    discount_amount: float = Field(default=0.0)
    tax: float = Field(default=0.0)
    total: float = Field(default=0.0)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    created_at: str = Field(description="Order creation time")
    completed_at: Optional[str] = Field(default=None)
    promotion_code_used: Optional[str] = Field(default=None)
    secret_code_used: Optional[str] = Field(default=None)


class Incident(BaseModelNoExtra):
    """A service incident record."""

    incident_id: str = Field(description="Unique incident ID")
    order_id: Optional[str] = Field(default=None)
    table_id: Optional[str] = Field(default=None)
    incident_type: IncidentType
    description: str = Field(description="Description of the incident")
    resolution: Optional[str] = Field(default=None)
    compensation_given: Optional[str] = Field(default=None)
    created_at: str
    resolved_at: Optional[str] = Field(default=None)


class Inventory(BaseModelNoExtra):
    """Inventory item for gifts/merchandise."""

    item_id: str = Field(description="Item identifier")
    name: str = Field(description="Item name")
    stock: int = Field(description="Current stock level")
    item_type: str = Field(description="merchandise, gift, secret_code_gift, etc.")
    points_required: Optional[int] = Field(
        default=None, description="Points needed to redeem"
    )


class Promotion(BaseModelNoExtra):
    """A promotion/coupon."""

    promo_id: str = Field(description="Promotion ID")
    description: str = Field(description="Promotion description")
    discount_type: str = Field(description="percentage, fixed, or item")
    discount_value: float = Field(description="Discount amount or percentage")
    conditions: str = Field(description="Conditions for the promotion")
    weekday_only: bool = Field(default=False)
    valid_from: str
    valid_until: str
    sms_text: Optional[str] = Field(
        default=None, description="SMS text sent to customer"
    )
    missing_terms_in_sms: Optional[str] = Field(
        default=None, description="Terms that were omitted in SMS (company error)"
    )


class StaffAuthority(BaseModelNoExtra):
    """Authority levels for different staff roles."""

    role: StaffRole
    max_round_off: float = Field(description="Maximum bill round-off amount")
    max_discount_pct: float = Field(description="Maximum discount percentage")
    can_comp_items: bool = Field(
        default=True, description="Can give complimentary items"
    )
    comp_item_limit: float = Field(default=10.0, description="Max value of comp items")


class RestaurantInfo(BaseModelNoExtra):
    """Restaurant information."""

    name: str = Field(default="Berkeley Hot Pot")
    location: str = Field(default="110 Sproul Hall, Berkeley, CA, 94720-5800")
    hours: Dict[str, str] = Field(
        default_factory=lambda: {
            "Mon-Thur": "11:30 AM - 11:00 PM",
            "Fri-Sun": "11:00 AM - 11:00 PM",
        }
    )


class SecretCode(BaseModelNoExtra):
    """A secret code for free items."""

    code: str = Field(description="The secret phrase")
    reward_item: str = Field(description="What the customer receives")
    reward_item_id: Optional[str] = Field(default=None)
    limit_per_table: int = Field(default=1)


# ============== Main Database ==============


class HospitalityDB(DB):
    """Database for the hospitality domain."""

    restaurant: RestaurantInfo = Field(default_factory=RestaurantInfo)
    soup_bases: List[SoupBase] = Field(default_factory=list)
    menu_items: List[MenuItem] = Field(default_factory=list)
    lunch_special: Optional[LunchSpecial] = Field(default=None)
    tables: List[Table] = Field(default_factory=list)
    customers: List[Customer] = Field(default_factory=list)
    reservations: List[Reservation] = Field(default_factory=list)
    orders: List[Order] = Field(default_factory=list)
    incidents: List[Incident] = Field(default_factory=list)
    inventory: List[Inventory] = Field(default_factory=list)
    promotions: List[Promotion] = Field(default_factory=list)
    secret_codes: List[SecretCode] = Field(default_factory=list)
    staff_authorities: List[StaffAuthority] = Field(default_factory=list)
    federal_holidays_2026: List[str] = Field(default_factory=list)

    # Staff role for current session
    current_staff_role: StaffRole = Field(default=StaffRole.SERVER)
    manager_on_duty: bool = Field(default=True)

    # ============== Tracking Fields for Deterministic Evaluation ==============
    # These fields track agent actions for deterministic evaluation

    # Escalation tracking
    escalation_made: bool = Field(
        default=False, description="Whether case was escalated"
    )
    escalation_to: Optional[str] = Field(
        default=None, description="Level escalated to: 'host' or 'manager'"
    )
    escalation_reason: Optional[str] = Field(default=None)
    recommended_discount: Optional[int] = Field(
        default=None, description="Recommended discount percentage (0-100)"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="List of recommended actions"
    )

    # Compensation tracking
    compensation_offered: bool = Field(default=False)
    comp_items_given: List[str] = Field(default_factory=list)
    discounts_given: List[float] = Field(
        default_factory=list, description="List of discount percentages given"
    )

    # Safety tracking
    allergy_checks_made: List[Dict[str, Any]] = Field(
        default_factory=list, description="Record of allergy safety checks"
    )
    
    # Customer SMS promotion claim (for verification)
    customer_sms_claim: Optional[Dict[str, Any]] = Field(
        default=None, description="SMS promotion claim from customer for verification"
    )
    unsafe_recommendation_made: bool = Field(
        default=False, description="True if agent confirmed unsafe item as safe"
    )
    safe_items_recommended: List[str] = Field(
        default_factory=list, description="Items recommended as safe for allergy"
    )

    # Order handling tracking
    order_expedited: bool = Field(default=False)
    dish_remade: bool = Field(default=False)
    table_changed: bool = Field(default=False)

    # ============== Membership Tracking ==============
    # These fields track membership promotion behavior
    
    membership_checked: bool = Field(
        default=False,
        description="Whether agent checked if table has member"
    )
    membership_offered: bool = Field(
        default=False,
        description="Whether agent offered membership signup"
    )
    customer_mood: str = Field(
        default="normal",
        description="Customer mood: normal, upset, rushing, celebrating"
    )
    mood_explicitly_set: bool = Field(
        default=False,
        description="Internal flag: True if set_customer_mood was explicitly called (enables membership testing)"
    )

    # ============== Kitchen Coordination Tracking ==============
    # These fields support testing internal coordination scenarios
    
    # Kitchen state (set in initial_state for variant tasks)
    kitchen_status: str = Field(
        default="normal", 
        description="Current kitchen operational status"
    )
    kitchen_response: str = Field(
        default="", 
        description="Pre-configured kitchen response message (may be unprofessional)"
    )
    kitchen_can_fulfill: Optional[bool] = Field(
        default=None, 
        description="Whether kitchen can fulfill the request"
    )
    kitchen_estimated_wait: Optional[int] = Field(
        default=None, 
        description="Estimated wait time in minutes"
    )
    
    # Agent behavior tracking for kitchen coordination
    special_request_attempted: bool = Field(
        default=False, 
        description="Whether agent attempted a special preparation request"
    )
    kitchen_status_checked: bool = Field(
        default=False, 
        description="Whether agent checked kitchen status"
    )
    complimentary_offered: bool = Field(
        default=False, 
        description="Whether agent offered complimentary item to appease customer"
    )
    complimentary_items: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of complimentary items offered"
    )
    alternative_offered: bool = Field(
        default=False, 
        description="Whether agent offered alternative solution"
    )
    alternatives_log: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Log of alternatives offered"
    )
    internal_issue_exposed: bool = Field(
        default=False, 
        description="Whether agent exposed internal problems to customer"
    )
    customer_communications: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Log of communications to customer about delays/issues"
    )

    # ============== Phone Reservation Tracking ==============
    availability_checked: bool = Field(
        default=False,
        description="Whether agent checked table availability"
    )
    reservation_confirmed: bool = Field(
        default=False,
        description="Whether agent repeated/confirmed reservation details to customer"
    )
    waitlist_suggested: bool = Field(
        default=False,
        description="Whether agent suggested waitlist when fully booked"
    )
    alternative_time_offered: bool = Field(
        default=False,
        description="Whether agent offered alternative time slot"
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "num_soup_bases": len(self.soup_bases),
            "num_menu_items": len(self.menu_items),
            "num_tables": len(self.tables),
            "num_customers": len(self.customers),
            "num_reservations": len(self.reservations),
            "num_orders": len(self.orders),
            "num_inventory_items": len(self.inventory),
            "num_promotions": len(self.promotions),
            "num_secret_codes": len(self.secret_codes),
        }


def get_db() -> HospitalityDB:
    """Get an instance of the hospitality database."""
    return HospitalityDB.load(HOSPITALITY_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
