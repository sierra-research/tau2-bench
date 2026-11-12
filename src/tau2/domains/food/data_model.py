"""Pydantic models for food domain menus, carts, orders, reservations, and state."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB
from tau2.utils import load_file


class MenuTimeRange(BaseModel):
    """A daily time window in 'HH:MM' 24-hour format."""

    start_time: str = Field(description="Start time in 'HH:MM' format (e.g., '14:00')")
    end_time: str = Field(description="End time in 'HH:MM' format (e.g., '16:00')")


class MenuDateRange(BaseModel):
    """An inclusive calendar date range in 'YYYY-MM-DD' format."""

    start_date: str = Field(
        description="Start date in 'YYYY-MM-DD' format (e.g., '2026-07-04')"
    )
    end_date: str = Field(
        description="End date in 'YYYY-MM-DD' format (e.g., '2026-07-04')"
    )


class MenuScheduleRule(BaseModel):
    """A schedule rule restricting availability by day of week, date range, and time range."""

    days_of_week: List[str] = Field(
        default_factory=list,
        description="Allowed days of week (e.g., ['SATURDAY', 'SUNDAY'])",
    )
    dates: List[MenuDateRange] = Field(
        default_factory=list,
        description="Allowed calendar date ranges",
    )
    time_ranges: List[MenuTimeRange] = Field(
        default_factory=list,
        description="Allowed time-of-day windows",
    )


class MenuSchedule(BaseModel):
    """A named menu availability schedule."""

    schedule_id: str = Field(description="Unique identifier of the schedule")
    rules: List[MenuScheduleRule] = Field(
        default_factory=list, description="Rules that make up this schedule"
    )


class ModifierOption(BaseModel):
    """A selectable option within a modifier group."""

    option_id: str = Field(
        description="Identifier of the modifier option, unique within its group"
    )
    name: str = Field(description="Display name of the modifier option")
    price_cents: int = Field(
        default=0, description="Additional price in cents when this option is selected"
    )
    available: bool = Field(
        default=True, description="Whether this option is currently in stock"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative spoken names or slang terms for this option",
    )
    child_group_ids: List[str] = Field(
        default_factory=list,
        description="Modifier group IDs unlocked when this option is selected (e.g., combo side/drink)",
    )
    child_group_defaults: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping from child group_id to default option_ids applied when this parent option is selected",
    )


class ModifierGroup(BaseModel):
    """A group of modifier options with selection cardinality constraints."""

    group_id: str = Field(description="Unique identifier of the modifier group")
    name: str = Field(description="Display name of the modifier group")
    min_select: int = Field(
        default=0,
        description="Minimum total selections required from this group (0 if optional)",
    )
    max_select: int = Field(
        default=1,
        description="Maximum total selections allowed across all options in this group",
    )
    max_per_option: int = Field(
        default=1,
        description="Maximum quantity allowed for a single option within this group",
    )
    default_option_ids: List[str] = Field(
        default_factory=list,
        description="Option IDs automatically applied if the caller makes no selection for this group",
    )
    supports_half_and_half: bool = Field(
        default=False,
        description=(
            "Whether options in this group support 'left' and 'right' pizza placement; "
            "also allows a 'light' or 'extra' volume on options whose names carry no portion"
        ),
    )
    options: Dict[str, ModifierOption] = Field(
        default_factory=dict,
        description="Mapping from option_id to ModifierOption",
    )


class MenuItem(BaseModel):
    """A menu item offered by a restaurant store."""

    item_id: str = Field(description="Unique identifier of the menu item")
    name: str = Field(description="Display name of the menu item")
    category: str = Field(description="Menu category name")
    description: str = Field(description="Description and standard recipe ingredients")
    base_price_cents: int = Field(description="Base price of the item in cents")
    available: bool = Field(
        default=True, description="Whether the item is currently available to order"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative spoken names or aliases for this item",
    )
    modifier_group_ids: List[str] = Field(
        default_factory=list,
        description="Ordered list of top-level modifier group IDs applicable to this item",
    )
    group_defaults: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-item default option_ids keyed by modifier group_id",
    )
    schedule_ids: List[str] = Field(
        default_factory=list,
        description="MenuSchedule IDs restricting when this item is available",
    )
    # "dine_in" is reserved for future dine-in support; active sessions use "pickup" or "delivery".
    allowed_fulfillment_methods: List[Literal["pickup", "delivery", "dine_in"]] = Field(
        default_factory=list,
        description="Allowed fulfillment methods if restricted (empty means all methods allowed)",
    )


class ReservationPolicy(BaseModel):
    """Table reservation rules of a store."""

    min_party_size: int = Field(default=1, description="Smallest party that can book")
    max_party_size: int = Field(default=8, description="Largest party that can book")
    slot_minutes: int = Field(
        default=30, description="Length of a booking slot in minutes"
    )
    first_seating_after_open_minutes: int = Field(
        default=30, description="Minutes after opening of the first bookable slot"
    )
    last_seating_before_close_minutes: int = Field(
        default=60, description="Minutes before closing of the last bookable slot"
    )
    max_days_ahead: int = Field(
        default=30, description="How many days ahead a table can be booked"
    )
    tables_per_slot: Dict[str, int] = Field(
        default_factory=lambda: {"indoor": 4},
        description="Bookable tables per slot for each seating area (one booking uses one table)",
    )
    high_chairs_per_slot: int = Field(
        default=0, description="High chairs available per slot"
    )


class Reservation(BaseModel):
    """A table reservation."""

    reservation_id: str = Field(
        description="Deterministic reservation ID (e.g., 'RES-0001')"
    )
    store_id: str = Field(description="Store the table is booked at")
    date: str = Field(description="Date in 'YYYY-MM-DD' format")
    time: str = Field(description="Slot start time in 'HH:MM' format")
    party_size: int = Field(description="Number of guests")
    customer_name: str = Field(description="Name the booking is under")
    customer_phone: str = Field(
        description="Contact phone number (e.g., '415-555-0199')"
    )
    seating: str = Field(
        default="indoor", description="Seating area (e.g., 'indoor', 'outdoor')"
    )
    high_chairs: int = Field(default=0, description="High chairs requested")
    status: Literal["confirmed", "cancelled"] = Field(
        default="confirmed", description="Status"
    )


class Store(BaseModel):
    """A restaurant store with its complete menu and modifier catalog."""

    store_id: str = Field(description="Unique store identifier")
    name: str = Field(description="Display name of the restaurant store")
    concept: Literal["pizza", "cafe", "burger"] = Field(
        description="Restaurant concept category"
    )
    cuisine: str = Field(description="Short description of the store's cuisine")
    delivery_fee_cents: int = Field(
        default=399, description="Flat delivery fee in cents for delivery orders"
    )
    min_delivery_subtotal_cents: int = Field(
        default=1200,
        description="Minimum subtotal in cents required for delivery orders",
    )
    open_hours: str = Field(default="07:00-22:00", description="Store operating hours")
    pickup_lead_minutes: int = Field(
        default=20,
        description="Minimum minutes between ordering ahead and a pickup time",
    )
    delivery_lead_minutes: int = Field(
        default=45,
        description="Minimum minutes between ordering ahead and a delivery time",
    )
    models_default_ingredients: bool = Field(
        default=False,
        description="Whether recipe ingredients are modeled as defaults, so a 'No X' for an X that is not on the line changes nothing",
    )
    reservation_policy: Optional[ReservationPolicy] = Field(
        default=None,
        description="Table reservation rules, or None if the store takes no reservations",
    )
    categories: List[str] = Field(
        default_factory=list, description="Ordered list of menu categories"
    )
    items: Dict[str, MenuItem] = Field(
        default_factory=dict, description="Mapping from item_id to MenuItem"
    )
    modifier_groups: Dict[str, ModifierGroup] = Field(
        default_factory=dict,
        description="Mapping from group_id to ModifierGroup",
    )
    schedules: Dict[str, MenuSchedule] = Field(
        default_factory=dict,
        description="Mapping from schedule_id to MenuSchedule",
    )


class ModifierSelection(BaseModel):
    """A modifier selection entry passed to add_to_cart or update_cart_item."""

    option_id: str = Field(
        default="",
        description=(
            "Exact modifier name without price suffixes; to remove a default, choose "
            "the group's 'No <name>' option or send quantity 0"
        ),
    )
    group_id: str = Field(
        default="",
        description=(
            "Numbered group name (e.g., 'Options (3)'); required when the option name "
            "appears in more than one of the item's groups, optional otherwise"
        ),
    )
    quantity: int = Field(
        default=1,
        description=(
            "Quantity of this modifier (default 1; for counted options, the full amount; "
            "0 removes a default)"
        ),
    )
    placement: str = Field(
        default="whole",
        description="Placement on the item ('whole', 'left', or 'right'; default 'whole')",
    )
    volume: str = Field(
        default="normal",
        description="Portion volume ('light', 'normal', or 'extra'; default 'normal')",
    )


class SelectedModifier(BaseModel):
    """A validated modifier option applied to an order line item."""

    group_id: str = Field(description="Modifier group ID")
    option_id: str = Field(description="Selected modifier option ID")
    name: str = Field(description="Display name of the selected modifier option")
    quantity: int = Field(default=1, description="Quantity of this modifier option")
    placement: Literal["whole", "left", "right"] = Field(
        default="whole",
        description="Placement on the item ('whole', 'left', or 'right')",
    )
    volume: Literal["light", "normal", "extra"] = Field(
        default="normal",
        description="Portion volume of the modifier ('light', 'normal', or 'extra')",
    )
    unit_price_cents: int = Field(
        default=0, description="Unit price in cents of this modifier option"
    )


class OrderLineItem(BaseModel):
    """A line item in the active working cart or a submitted order."""

    line_item_id: str = Field(description="Line item identifier (e.g., 'line_1')")
    item_id: str = Field(description="Menu item ID")
    name: str = Field(description="Display name of the menu item")
    quantity: int = Field(default=1, description="Quantity of this configured item")
    unit_base_price_cents: int = Field(
        description="Base unit price of the menu item in cents"
    )
    modifiers: List[SelectedModifier] = Field(
        default_factory=list,
        description="Canonicalized list of selected modifiers for this item",
    )
    line_total_cents: int = Field(
        description="Total price in cents for this line item (quantity * (base + modifiers))"
    )


class CartState(BaseModel):
    """Working shopping cart state during an active ordering session."""

    store_id: Optional[str] = Field(
        default=None, description="Store ID associated with the active cart"
    )
    current_time: Optional[str] = Field(
        default=None,
        description=(
            "Simulated store timestamp in 'YYYY-MM-DD HH:MM' format; drives menu schedules, "
            "order lead times, and reservation dates"
        ),
    )
    order_type: Literal["pickup", "delivery", "dine_in"] = Field(
        default="pickup",
        description="Active fulfillment method for the session ('pickup', 'delivery', or 'dine_in')",
    )
    channel: Literal["drive_thru", "mobile"] = Field(
        default="drive_thru",
        description="How the customer orders: at the drive-thru now, or ahead by phone/app",
    )
    scheduled_time: str = Field(
        default="ASAP",
        description="Requested pickup/delivery time: 'ASAP' or 'HH:MM' today",
    )
    items: List[OrderLineItem] = Field(
        default_factory=list, description="Items currently in the cart"
    )
    next_line_id: int = Field(
        default=1, description="Deterministic counter for assigning line_item_id"
    )
    transferred_to_human: bool = Field(
        default=False,
        description="Whether the session was transferred to a human restaurant staff member",
    )


class SubmittedOrder(BaseModel):
    """A finalized, confirmed food order."""

    order_id: str = Field(description="Deterministic order ID (e.g., '#ORD001')")
    store_id: str = Field(description="Store ID where the order was placed")
    order_type: Literal["pickup", "delivery", "dine_in"] = Field(
        description="Fulfillment method ('pickup', 'delivery', or 'dine_in')"
    )
    channel: Literal["drive_thru", "mobile"] = Field(
        default="drive_thru", description="Channel the order was placed through"
    )
    scheduled_time: str = Field(
        default="ASAP", description="Requested pickup/delivery time: 'ASAP' or 'HH:MM'"
    )
    customer_name: str = Field(description="Normalized customer name")
    customer_phone: str = Field(description="Normalized customer phone number")
    delivery_address: Optional[str] = Field(
        default=None,
        description="Normalized delivery address (required when order_type is 'delivery')",
    )
    items: List[OrderLineItem] = Field(
        default_factory=list,
        description="Canonicalized list of ordered line items",
    )
    subtotal_cents: int = Field(description="Order items subtotal in cents")
    delivery_fee_cents: int = Field(
        default=0, description="Delivery fee in cents (0 for pickup or dine_in)"
    )
    total_cents: int = Field(description="Grand total in cents")
    status: Literal["confirmed"] = Field(
        default="confirmed", description="Order status"
    )


class FoodDB(DB):
    """Database containing all restaurant store menus, active cart, and submitted orders."""

    stores: Dict[str, Store] = Field(
        default_factory=dict, description="Mapping from store_id to Store"
    )
    cart: CartState = Field(
        default_factory=CartState, description="Current working cart state"
    )
    orders: Dict[str, SubmittedOrder] = Field(
        default_factory=dict,
        description="Mapping from order_id to SubmittedOrder",
    )
    reservations: Dict[str, Reservation] = Field(
        default_factory=dict,
        description="Mapping from reservation_id to Reservation",
    )

    @classmethod
    def load(
        cls,
        path: str | Path,
        menus_dir: Optional[str | Path] = None,
    ) -> "FoodDB":
        """Load the food database from db.json and per-store menu JSON files in menus/."""
        data = dict(load_file(path))
        if not data.get("stores"):
            resolved_menus_dir = (
                Path(menus_dir)
                if menus_dir is not None
                else Path(path).parent / "menus"
            )
            if resolved_menus_dir.is_dir():
                stores: Dict[str, Any] = {}
                for menu_file in sorted(resolved_menus_dir.glob("*.json")):
                    store_data = load_file(menu_file)
                    store_id = store_data.get("store_id") or menu_file.stem
                    stores[store_id] = store_data
                data["stores"] = stores
        return cls.model_validate(data)

    def get_statistics(self) -> dict[str, Any]:
        """Return summary statistics of the food ordering database."""
        total_items = sum(len(store.items) for store in self.stores.values())
        total_groups = sum(len(store.modifier_groups) for store in self.stores.values())
        return {
            "num_stores": len(self.stores),
            "num_menu_items": total_items,
            "num_modifier_groups": total_groups,
            "num_orders": len(self.orders),
            "num_reservations": len(self.reservations),
        }
