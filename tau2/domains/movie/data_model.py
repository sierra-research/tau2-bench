from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.environment.db import DB

# Enumerations and basic aliases
MovieGenre = Literal[
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Horror",
    "Thriller",
    "Romance",
    "Science Fiction",
    "Fantasy",
    "Mystery",
]

MovieFormat = Literal["2D", "3D", "IMAX", "Dolby"]
ShowStatus = Literal["scheduled", "canceled", "completed"]
BookingStatus = Literal["confirmed", "canceled", "refunded", "pending"]
TicketType = Literal["adult", "child", "senior"]
DeliveryMethod = Literal["e-ticket", "box-office", "kiosk"]
PaymentSource = Literal["card", "wallet", "cash", "gift_card"]
MPAARating = Literal["G", "PG", "PG-13", "R", "NC-17", "NR"]


# Movies
class CastMember(BaseModel):
    name: str = Field(description="Cast member name")
    role: str = Field(description="Role portrayed by the cast member")


class Crew(BaseModel):
    director: str = Field(description="Director name")
    writer: str = Field(description="Writer name")
    producer: str = Field(description="Producer name")
    music: str = Field(description="Music composer name")


class Movie(BaseModel):
    movie_id: str = Field(description="Unique movie identifier")
    title: str = Field(description="Movie title")
    genres: List[MovieGenre] = Field(description="List of genres for the movie")
    runtime_minutes: int = Field(description="Runtime in minutes")
    mpaa_rating: MPAARating = Field(description="MPAA rating")
    languages_audio: List[str] = Field(description="Available audio languages")
    subtitles: List[str] = Field(description="Available subtitle languages")
    formats: List[MovieFormat] = Field(description="Supported presentation formats")
    release_date: str = Field(
        description="Release date in the format YY-MM-DD, e.g. 24-05-01"
    )
    end_of_run_est: str = Field(
        description="Estimated end-of-run date in the format YY-MM-DD, e.g. 24-06-30"
    )
    cast: List[CastMember] = Field(description="List of cast members")
    crew: Crew = Field(description="Key crew members")
    synopsis: str = Field(description="Synopsis of the movie")


# Theaters and showtimes
class TheaterAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province")
    country: str = Field(description="Country name")
    zip: str = Field(description="Postal code")


class TheaterContact(BaseModel):
    phone: str = Field(description="Contact phone number")
    email: str = Field(description="Contact email address")
    website: str = Field(description="Website URL")


class Seat(BaseModel):
    seat_id: str = Field(description="Unique seat identifier")
    type: str = Field(description="Seat type, e.g., standard, recliner")
    wheelchair_accessible: bool = Field(description="Whether seat is wheelchair accessible")


SeatRowMap = Annotated[
    Dict[str, List[Seat]],
    Field(description="Mapping of row_id to list of seats in that row"),
]


class SeatMap(BaseModel):
    rows: SeatRowMap = Field(description="Seat rows with their seats")


class Auditorium(BaseModel):
    auditorium_id: str = Field(description="Unique auditorium identifier")
    name: str = Field(description="Auditorium display name")
    capacity: int = Field(description="Total seating capacity")
    features: List[str] = Field(description="List of auditorium features")
    seat_map: SeatMap = Field(description="Detailed seat map")


class PricingBase(BaseModel):
    adult: float = Field(description="Base price for adult tickets in dollars")
    child: float = Field(description="Base price for child tickets in dollars")
    senior: float = Field(description="Base price for senior tickets in dollars")


FormatSurcharges = Annotated[
    Dict[MovieFormat, float],
    Field(description="Surcharges by presentation format in dollars"),
]


class TimeBasedPricing(BaseModel):
    matinee_discount: float = Field(
        description="Discount applied during matinee times in dollars"
    )
    peak_surcharge: float = Field(
        description="Surcharge applied during peak times in dollars"
    )


class TheaterFees(BaseModel):
    convenience_fee: float = Field(description="Convenience fee in dollars")
    booking_fee: float = Field(description="Booking fee in dollars")


class TheaterPricingRules(BaseModel):
    base: PricingBase = Field(description="Base ticket prices")
    format_surcharges: FormatSurcharges = Field(
        description="Additional surcharges based on format"
    )
    time_based: TimeBasedPricing = Field(description="Time-based discounts/surcharges")
    fees: TheaterFees = Field(description="Fees applied at checkout")
    tax_rate_percent: float = Field(description="Tax rate as percentage")


class PriceFees(BaseModel):
    convenience_fee: float = Field(description="Convenience fee per ticket in dollars")


class PriceSchema(BaseModel):
    adult: float = Field(description="Adult ticket price in dollars")
    child: float = Field(description="Child ticket price in dollars")
    senior: float = Field(description="Senior ticket price in dollars")
    fees: PriceFees = Field(description="Per-ticket fee breakdown")


class Show(BaseModel):
    show_id: str = Field(description="Unique show identifier")
    movie_id: str = Field(description="Associated movie ID")
    auditorium_id: str = Field(description="Auditorium where the show is scheduled")
    start_time_local: str = Field(
        description="Local start time in the format YY-MM-DD-HH-MM, e.g. 24-05-01-19-30"
    )
    end_time_local: str = Field(
        description="Local end time in the format YY-MM-DD-HH-MM, e.g. 24-05-01-21-50"
    )
    format: MovieFormat = Field(description="Presentation format")
    language: str = Field(description="Show audio language")
    subtitles: str = Field(description="Subtitle language, if any")
    status: ShowStatus = Field(description="Show status")
    price_schema: PriceSchema = Field(description="Pricing for the show")


class TheaterDaySchedule(BaseModel):
    shows: List[Show] = Field(description="List of shows for the date")


class Theater(BaseModel):
    theater_id: str = Field(description="Unique theater identifier")
    name: str = Field(description="Theater name")
    address: TheaterAddress = Field(description="Theater address")
    contact: TheaterContact = Field(description="Theater contact information")
    amenities: List[str] = Field(description="Amenities offered at the theater")
    auditoriums: Dict[str, Auditorium] = Field(
        description="Auditoriums keyed by auditorium_id"
    )
    pricing_rules: TheaterPricingRules = Field(description="Pricing rules for the theater")
    dates: Dict[str, TheaterDaySchedule] = Field(
        description="Schedule by date in the format YY-MM-DD"
    )


# Bookings and payments
class CustomerName(BaseModel):
    first_name: str = Field(description="Customer first name")
    last_name: str = Field(description="Customer last name")


class Customer(BaseModel):
    name: CustomerName = Field(description="Customer name")
    email: str = Field(description="Customer email")
    phone: str = Field(description="Customer phone")
    loyalty_id: Optional[str] = Field(
        default=None, description="Customer loyalty identifier (optional)"
    )


class BookedSeat(BaseModel):
    seat_id: str = Field(description="Seat identifier")
    ticket_type: TicketType = Field(description="Type of ticket")
    price: float = Field(description="Ticket price in dollars")
    convenience_fee: float = Field(description="Convenience fee in dollars")
    tax: float = Field(description="Tax amount in dollars")


class ConcessionItem(BaseModel):
    item_id: str = Field(description="Concession item identifier")
    name: str = Field(description="Concession item name")
    size: str = Field(description="Size or variant")
    quantity: int = Field(description="Quantity purchased")
    price_each: float = Field(description="Price per item in dollars")
    tax_each: float = Field(description="Tax per item in dollars")
    total: float = Field(description="Total for this concession line in dollars")


class PromotionApplied(BaseModel):
    code: str = Field(description="Promotion code")
    description: str = Field(description="Promotion description")
    discount_amount: float = Field(description="Discount amount in dollars")


class PaymentExtraInfo(BaseModel):
    brand: Optional[str] = Field(
        default=None, description="Card or wallet brand (if applicable)"
    )
    last_four: Optional[str] = Field(
        default=None, description="Last four digits of card (if applicable)"
    )


class BookingPaymentMethod(BaseModel):
    source: PaymentSource = Field(description="Payment source type")
    payment_method_id: str = Field(description="Payment method identifier")
    extra_info: Optional[PaymentExtraInfo] = Field(
        default=None, description="Additional method details"
    )


class BookingPayment(BaseModel):
    payment_id: str = Field(description="Unique payment identifier")
    amount: float = Field(description="Payment amount in dollars")
    method: BookingPaymentMethod = Field(description="Payment method details")
    created_at: str = Field(description="Payment timestamp in the format YY-MM-DD-HH-MM")


class BookingTotals(BaseModel):
    tickets_subtotal: float = Field(description="Subtotal for tickets in dollars")
    concessions_subtotal: float = Field(description="Subtotal for concessions in dollars")
    fees_total: float = Field(description="Total fees in dollars")
    tax_total: float = Field(description="Total tax in dollars")
    grand_total: float = Field(description="Grand total in dollars")
    amount_paid: float = Field(description="Total amount paid in dollars")
    amount_due: float = Field(description="Remaining amount due in dollars")


class TicketDeliveryItem(BaseModel):
    ticket_id: str = Field(description="Ticket identifier")
    barcode: str = Field(description="Ticket barcode")


class Delivery(BaseModel):
    method: DeliveryMethod = Field(description="Delivery method for tickets")
    tickets: List[TicketDeliveryItem] = Field(description="Delivered tickets")


class Booking(BaseModel):
    booking_id: str = Field(description="Unique identifier for the booking")
    theater_id: str = Field(description="Theater ID")
    movie_id: str = Field(description="Movie ID")
    show_id: str = Field(description="Show ID")
    date: str = Field(description="Show date in the format YY-MM-DD")
    start_time_local: str = Field(
        description="Local start time in the format YY-MM-DD-HH-MM"
    )
    status: BookingStatus = Field(description="Booking status")
    created_at: str = Field(description="Timestamp when booking was created YY-MM-DD-HH-MM")
    canceled_at: Optional[str] = Field(
        default=None, description="Timestamp when booking was canceled YY-MM-DD-HH-MM"
    )
    customer: Customer = Field(description="Customer information")
    seats: List[BookedSeat] = Field(description="List of booked seats")
    concessions: List[ConcessionItem] = Field(
        description="List of concessions purchased"
    )
    promotions_applied: List[PromotionApplied] = Field(
        description="Promotions applied to this booking"
    )
    payment_history: List[BookingPayment] = Field(
        description="History of payments for this booking"
    )
    totals: BookingTotals = Field(description="Aggregated monetary totals")
    delivery: Delivery = Field(description="Ticket delivery details")
    special_requests: Optional[str] = Field(
        default=None, description="Any special requests noted by the customer"
    )


# Database container
class MovieTheaterDB(DB):
    """Database of all movies, theaters, and bookings."""

    movies: Dict[str, Movie] = Field(
        description="Dictionary of all movies indexed by movie ID"
    )
    theaters: Dict[str, Theater] = Field(
        description="Dictionary of all theaters indexed by theater ID"
    )
    bookings: Dict[str, Booking] = Field(
        description="Dictionary of all bookings indexed by booking ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the movie theater database."""
        num_movies = len(self.movies)
        num_theaters = len(self.theaters)
        num_bookings = len(self.bookings)
        num_shows = 0
        num_auditoriums = 0
        total_capacity = 0

        for theater in self.theaters.values():
            num_auditoriums += len(theater.auditoriums)
            total_capacity += sum(a.capacity for a in theater.auditoriums.values())
            for day in theater.dates.values():
                num_shows += len(day.shows)

        return {
            "num_movies": num_movies,
            "num_theaters": num_theaters,
            "num_auditoriums": num_auditoriums,
            "total_capacity": total_capacity,
            "num_shows": num_shows,
            "num_bookings": num_bookings,
        }
