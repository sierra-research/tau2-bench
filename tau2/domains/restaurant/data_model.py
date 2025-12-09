from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.environment.db import DB


class PlateModifiers(BaseModel):
    """Modifier options available/applied to a plate selection"""

    spice_level: Optional[str] = Field(
        description="Spice level for the plate (e.g., mild, medium, hot)", default=None
    )
    portion: Optional[str] = Field(
        description="Portion size for the plate (e.g., small, regular, large)",
        default=None,
    )
    protein: Optional[str] = Field(
        description="Protein choice for the plate (e.g., chicken, tofu, beef)",
        default=None,
    )
    preparation: Optional[str] = Field(
        description="Preparation style for the plate (e.g., grilled, fried, steamed)",
        default=None,
    )


class Plate(BaseModel):
    """Represents a specific plate (selection) of a dish with modifiers, availability and cost"""

    plate_ref: str = Field(description="Unique identifier for the plate selection")
    modifiers: PlateModifiers = Field(
        description="Modifier options for this plate selection"
    )
    served_today: bool = Field(description="Whether this plate is served today")
    cost: float = Field(description="Cost of this plate selection")


class Dish(BaseModel):
    """Represents a dish with its available plate selections"""

    title: str = Field(description="Title of the dish")
    dish_ref: str = Field(description="Unique identifier for the dish")
    selections: Dict[str, Plate] = Field(
        description="Dictionary of plate selections indexed by plate_ref"
    )


class GuestIdentity(BaseModel):
    """Represents a guest's name"""

    given: str = Field(description="Guest's given (first) name")
    family: str = Field(description="Guest's family (last) name")


class PatronLocation(BaseModel):
    """Represents a physical address/location"""

    line_one: str = Field(description="Primary address line")
    line_two: str = Field(description="Secondary address line")
    municipality: str = Field(description="City or municipality")
    nation: str = Field(description="Country name")
    province: str = Field(description="State or province")
    postal_code: str = Field(description="Postal or ZIP code")


class PaymentInstrumentMetadata(BaseModel):
    """Metadata associated with a saved payment instrument"""

    issuer: str = Field(description="Issuer or provider of the instrument")
    tail_digits: str = Field(description="Last digits of the instrument (e.g., card)")


class PaymentInstrument(BaseModel):
    """Represents a saved payment instrument for a patron"""

    origin: str = Field(
        description="Origin/type of the instrument (e.g., credit_card, paypal, gift_card)"
    )
    instrument_ref: str = Field(description="Unique identifier for the instrument")
    metadata: PaymentInstrumentMetadata = Field(
        description="Additional metadata about the instrument"
    )


class Patron(BaseModel):
    """Represents a patron with identity, location, contact, saved instruments and ticket history"""

    guest_ref: str = Field(description="Unique identifier for the patron")
    identity: GuestIdentity = Field(description="Patron's identity")
    location: PatronLocation = Field(description="Patron's primary location/address")
    contact_email: str = Field(description="Patron's contact email")
    saved_instruments: Dict[str, PaymentInstrument] = Field(
        description="Dictionary of saved instruments indexed by instrument_ref"
    )
    ticket_log: List[str] = Field(
        description="List of service ticket references associated with this patron"
    )


ServiceMode = Literal["dine_in", "takeout", "delivery"]


class TableInfo(BaseModel):
    """Table information for dine-in service"""

    zone: str = Field(description="Dining area or zone identifier")
    table_no: str = Field(description="Table number")
    seat_count: int = Field(description="Number of seats at the table")


class LineEntryMods(BaseModel):
    """Modifiers applied to a specific line entry"""

    heat: Optional[str] = Field(
        description="Heat/spice preference for the line entry", default=None
    )
    sauce: Optional[str] = Field(
        description="Sauce preference for the line entry", default=None
    )
    side: Optional[str] = Field(
        description="Side selection for the line entry", default=None
    )


class LineEntry(BaseModel):
    """Represents an item in a service ticket"""

    label: str = Field(description="Display label for the line entry")
    dish_ref: str = Field(description="Reference to the dish")
    plate_ref: str = Field(description="Reference to the selected plate")
    cost: float = Field(description="Cost of this line entry at time of order")
    mods: LineEntryMods = Field(
        description="Modifiers applied to this line entry (heat, sauce, side)"
    )


class PrepBatch(BaseModel):
    """Represents a preparation batch grouping plates and parcel tags"""

    parcel_tags: List[str] = Field(
        description="List of parcel/bag tags associated with this batch"
    )
    plate_refs: List[str] = Field(
        description="List of plate references included in this batch"
    )


class TicketCharge(BaseModel):
    """Represents a charge or refund on a service ticket"""

    kind: str = Field(
        description="Type of charge (e.g., payment, refund, tip, tax, delivery_fee)"
    )
    total: float = Field(description="Total amount for this charge")
    instrument_ref: str = Field(
        description="Reference to the payment instrument used for this charge"
    )


class ServiceTicket(BaseModel):
    """Represents a service ticket with guest, mode, items, state, preparation and charges"""

    ticket_ref: str = Field(description="Unique identifier for the ticket")
    guest_ref: str = Field(description="Reference to the patron (guest)")
    service_mode: ServiceMode = Field(
        description="Mode of service (dine_in, takeout, delivery)"
    )
    dropoff: Optional[PatronLocation] = Field(
        description="Dropoff address (required for delivery orders)", default=None
    )
    table_info: Optional[TableInfo] = Field(
        description="Table information (required for dine-in orders)", default=None
    )
    line_entries: List[LineEntry] = Field(
        description="Line entries (items) included in the ticket"
    )
    state: str = Field(
        description="Current state of the ticket (e.g., placed, in_progress, ready, delivered, cancelled)"
    )
    prep_batches: List[PrepBatch] = Field(
        description="Preparation batches associated with the ticket"
    )
    charges: List[TicketCharge] = Field(
        description="List of charges or refunds applied to the ticket"
    )


class RestaurantDB(DB):
    """Database containing restaurant-related data including menu, patrons and service tickets"""

    menu_board: Dict[str, Dish] = Field(
        description="Dictionary of all dishes indexed by dish_ref"
    )
    patron_registry: Dict[str, Patron] = Field(
        description="Dictionary of all patrons indexed by guest_ref"
    )
    service_tickets: Dict[str, ServiceTicket] = Field(
        description="Dictionary of all service tickets indexed by ticket_ref"
    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get the statistics of the restaurant database."""
        num_dishes = len(self.menu_board)
        num_patrons = len(self.patron_registry)
        num_tickets = len(self.service_tickets)
        total_num_plates = sum(len(dish.selections) for dish in self.menu_board.values())
        return {
            "num_dishes": num_dishes,
            "num_patrons": num_patrons,
            "num_tickets": num_tickets,
            "total_num_plates": total_num_plates,
        }