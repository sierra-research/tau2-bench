from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.environment.db import DB


# Common annotated scalar types
CurrencyCode = Annotated[str, Field(description="Currency code (e.g., USD, EUR)")]
DateYY = Annotated[str, Field(description="Date in YY-MM-DD format")]
TimeRange = Annotated[str, Field(description="Time range in HH:MM-HH:MM format")]


# Reusable value objects
class PersonName(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class PostalAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province name")
    country: str = Field(description="Country name")
    zip: str = Field(description="Postal code")


class ContactInfo(BaseModel):
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")


class PassportInfo(BaseModel):
    number: str = Field(description="Passport number")
    nationality: str = Field(description="Passport nationality")
    expiry: DateYY = Field(description="Passport expiry date in YY-MM-DD format")


class TravelerPreferences(BaseModel):
    dietary: Optional[str] = Field(
        None, description="Dietary preferences or restrictions (optional)"
    )
    accessibility: Optional[str] = Field(
        None, description="Accessibility needs (optional)"
    )
    room_type: Optional[str] = Field(None, description="Preferred room type (optional)")
    bed_preference: Optional[str] = Field(
        None, description="Preferred bed type (optional)"
    )
    other: Optional[str] = Field(None, description="Other preferences (optional)")


class PaymentExtraInfo(BaseModel):
    brand: Optional[str] = Field(
        None, description="Payment brand (e.g., Visa, Mastercard) if applicable"
    )
    last_four: Optional[str] = Field(
        None, description="Last four digits of the payment method if applicable"
    )


class TravelerPaymentMethod(BaseModel):
    source: str = Field(description="Type of payment method (e.g., credit_card)")
    payment_method_id: str = Field(description="Unique identifier for the payment method")
    extra_info: Optional[PaymentExtraInfo] = Field(
        None, description="Additional payment method info"
    )


class Companion(BaseModel):
    first_name: str = Field(description="Companion's first name")
    last_name: str = Field(description="Companion's last name")
    dob: DateYY = Field(description="Companion's date of birth in YY-MM-DD format")
    passport: Optional[PassportInfo] = Field(
        None, description="Companion's passport information"
    )


class BookingTravelerInfo(BaseModel):
    first_name: str = Field(description="Traveler's first name")
    last_name: str = Field(description="Traveler's last name")
    dob: DateYY = Field(description="Traveler's date of birth in YY-MM-DD format")


class RoomingInfo(BaseModel):
    room_type: str = Field(description="Selected room type (e.g., double, suite)")
    occupancy: int = Field(description="Number of occupants in the room")


class BookingAddOn(BaseModel):
    type: str = Field(description="Type of add-on (e.g., excursion, transfer)")
    description: str = Field(description="Description of the add-on")
    price: float = Field(description="Add-on price")


class BookingPayment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment")
    amount: float = Field(description="Payment amount")


class Location(BaseModel):
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class PackageDeparture(BaseModel):
    status: str = Field(description="Departure status (e.g., available, sold_out)")
    base_price: float = Field(description="Base price for this departure")
    currency: CurrencyCode = Field(description="Currency of the base price")
    available_slots: int = Field(description="Available booking slots")
    early_bird_deadline: DateYY = Field(
        description="Early bird deadline in YY-MM-DD format"
    )


class ItineraryItem(BaseModel):
    day: int = Field(description="Day number in the itinerary")
    title: str = Field(description="Title of the day's activities")
    description: str = Field(description="Detailed description of the day's plan")
    location: Location = Field(description="Location for the day's activities")
    included_meals: List[str] = Field(
        description="List of included meals (e.g., breakfast, lunch, dinner)"
    )


class Accommodation(BaseModel):
    name: str = Field(description="Accommodation name")
    type: str = Field(description="Accommodation type (e.g., hotel, lodge)")
    location: Location = Field(description="Accommodation location")
    nights: int = Field(description="Number of nights at this accommodation")


class TransportationInfo(BaseModel):
    international: str = Field(description="International transport details")
    domestic: str = Field(description="Domestic transport details")
    ground: str = Field(description="Ground transport details")


class Activity(BaseModel):
    name: str = Field(description="Activity name")
    description: str = Field(description="Activity description")
    day: int = Field(description="Day number on which the activity occurs")


class PackagePolicies(BaseModel):
    cancellation_policy: str = Field(description="Cancellation policy")
    change_policy: str = Field(description="Change policy")
    refund_policy: str = Field(description="Refund policy")


class Package(BaseModel):
    package_id: str = Field(description="Unique identifier for the package")
    name: str = Field(description="Package name")
    category: str = Field(description="Package category (e.g., adventure, luxury)")
    description: str = Field(description="Package description")
    destinations: List[Location] = Field(description="List of destinations")
    duration_days: int = Field(description="Duration of the package in days")
    departure_points: List[Location] = Field(description="List of departure points")
    departures: Dict[DateYY, PackageDeparture] = Field(
        description="Departures by date (YY-MM-DD)"
    )
    itinerary: List[ItineraryItem] = Field(description="Daily itinerary")
    inclusions: List[str] = Field(description="List of included items/services")
    exclusions: List[str] = Field(description="List of excluded items/services")
    accommodations: List[Accommodation] = Field(
        description="List of accommodations included in the package"
    )
    transportation: TransportationInfo = Field(
        description="Transportation details for the package"
    )
    activities: List[Activity] = Field(description="Activities included in the package")
    policies: PackagePolicies = Field(description="Package policies")
    notes: Optional[str] = Field(None, description="Additional notes about the package")
    managed_by_agents: List[str] = Field(
        description="List of agent IDs who manage this package"
    )


class Booking(BaseModel):
    booking_id: str = Field(description="Unique identifier for the booking")
    package_id: str = Field(description="ID of the booked package")
    agent_id: str = Field(description="ID of the handling agent")
    booking_date: DateYY = Field(description="Booking date in YY-MM-DD format")
    departure_date: DateYY = Field(description="Departure date in YY-MM-DD format")
    status: str = Field(description="Booking status (e.g., confirmed, cancelled)")
    travelers: List[BookingTravelerInfo] = Field(
        description="Travelers included in this booking"
    )
    rooming: RoomingInfo = Field(description="Rooming configuration")
    add_ons: List[BookingAddOn] = Field(description="Selected add-ons")
    insurance: Optional[str] = Field(
        None, description="Travel insurance selection or policy (if any)"
    )
    payment_history: List[BookingPayment] = Field(
        description="Payment history for this booking"
    )
    total_price: float = Field(description="Total price of the booking")
    notes: Optional[str] = Field(None, description="Additional booking notes")


class Traveler(BaseModel):
    traveler_id: str = Field(description="Unique identifier for the traveler")
    name: PersonName = Field(description="Traveler's full name")
    address: PostalAddress = Field(description="Traveler's address")
    contact: ContactInfo = Field(description="Traveler's contact information")
    dob: DateYY = Field(description="Traveler's date of birth in YY-MM-DD format")
    passport: Optional[PassportInfo] = Field(
        None, description="Traveler's passport information"
    )
    preferences: Optional[TravelerPreferences] = Field(
        None, description="Traveler's preferences"
    )
    payment_methods: Dict[str, TravelerPaymentMethod] = Field(
        description="Traveler's saved payment methods indexed by payment_method_id"
    )
    saved_companions: List[Companion] = Field(
        description="Saved travel companions for this traveler"
    )
    memberships: List[str] = Field(description="Membership programs")
    bookings: List[Booking] = Field(description="Traveler's bookings")


class AgencyInfo(BaseModel):
    name: str = Field(description="Agency name")
    address: PostalAddress = Field(description="Agency address")


class AvailabilityDay(BaseModel):
    slots: List[TimeRange] = Field(
        description="List of available time slots in HH:MM-HH:MM format"
    )
    timezone: str = Field(description="Timezone for the provided slots")


class Agent(BaseModel):
    agent_id: str = Field(description="Unique identifier for the agent")
    name: PersonName = Field(description="Agent's full name")
    contact: ContactInfo = Field(description="Agent's contact information")
    agency: AgencyInfo = Field(description="Agent's agency information")
    certifications: List[str] = Field(description="Agent's certifications")
    languages: List[str] = Field(description="Languages spoken by the agent")
    regions_of_expertise: List[str] = Field(
        description="Regions where the agent has expertise"
    )
    commission_rate: float = Field(
        description="Commission rate as a decimal (e.g., 0.10 for 10%)"
    )
    employment_status: str = Field(
        description="Employment status (e.g., full_time, contractor)"
    )
    assigned_travelers: List[str] = Field(
        description="List of traveler IDs assigned to the agent"
    )
    managed_packages: List[str] = Field(
        description="List of package IDs managed by the agent"
    )
    bookings_handled: List[str] = Field(
        description="List of booking IDs handled by the agent"
    )
    availability: Dict[DateYY, AvailabilityDay] = Field(
        description="Agent availability by date (YY-MM-DD)"
    )
    notes: Optional[str] = Field(None, description="Additional notes about the agent")


class TravelAgencyDB(DB):
    """Database containing travel packages, travelers, and agents."""

    packages: Dict[str, Package] = Field(
        description="Dictionary of all packages indexed by package ID"
    )
    travelers: Dict[str, Traveler] = Field(
        description="Dictionary of all travelers indexed by traveler ID"
    )
    agents: Dict[str, Agent] = Field(
        description="Dictionary of all agents indexed by agent ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get high-level statistics of the database."""
        num_packages = len(self.packages)
        num_departure_instances = sum(
            len(pkg.departures) for pkg in self.packages.values()
        )
        num_travelers = len(self.travelers)
        num_agents = len(self.agents)
        num_bookings = sum(len(t.bookings) for t in self.travelers.values())
        return {
            "num_packages": num_packages,
            "num_departure_instances": num_departure_instances,
            "num_travelers": num_travelers,
            "num_agents": num_agents,
            "num_bookings": num_bookings,
        }
