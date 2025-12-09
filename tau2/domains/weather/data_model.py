from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.environment.db import DB

FlightType = Literal["round_trip", "one_way"]
CabinClass = Literal["business", "economy", "basic_economy"]
Insurance = Literal["yes", "no"]


MembershipLevel = Annotated[
    Literal["gold", "silver", "regular"], Field(description="Membership level")
]

# ------------------------------------------------------------------------------
# Pydantic models for the Weather schema
# ------------------------------------------------------------------------------

# Location-related models
class Name(BaseModel):
    city: str
    state: str
    country: str


class Coordinates(BaseModel):
    lat: float
    lon: float
    elevation_m: float


class ClimateNormalsMonth(BaseModel):
    avg_high_c: float
    avg_low_c: float
    precip_mm: float
    snow_cm: float


class SunTimesEntry(BaseModel):
    sunrise_local: str
    sunset_local: str
    day_length_minutes: int


# Forecast-related models
class Units(BaseModel):
    temperature: str  # C
    wind_speed: str   # kph
    precipitation: str  # mm
    pressure: str  # hPa


class Wind(BaseModel):
    speed_kph: float
    gust_kph: float
    direction_deg: int


class PrecipitationDetails(BaseModel):
    probability_pct: int
    type: str  # "rain", "snow", "sleet", "none"
    amount_mm: float


class HourlyForecastEntry(BaseModel):
    time_utc: str
    summary: str
    temperature_c: float
    feels_like_c: float
    dewpoint_c: float
    humidity_pct: int
    pressure_hpa: float
    wind: Wind
    visibility_km: float
    cloud_cover_pct: int
    precipitation: PrecipitationDetails
    uv_index: int


class DailyPrecip(BaseModel):
    probability_pct: int
    total_mm: float
    snow_cm: float


class DailyForecastEntry(BaseModel):
    date: str
    summary: str
    temp_min_c: float
    temp_max_c: float
    precipitation: DailyPrecip
    wind_max_kph: float
    uv_index_max: int
    sunrise_local: str
    sunset_local: str


class VerificationEntry(BaseModel):
    status: str  # "pending", "verified", "revised"
    actual_high_c: float
    actual_low_c: float
    actual_precip_mm: float
    notes: str = ""


# Observation-related models
class ObservationVariables(BaseModel):
    temperature_c: float
    feels_like_c: float
    dewpoint_c: float
    humidity_pct: int
    pressure_hpa: float
    wind_speed_kph: float
    wind_gust_kph: float
    wind_direction_deg: int
    precip_1h_mm: float
    precip_24h_mm: float
    snow_depth_cm: float
    visibility_km: float
    uv_index: int
    cloud_cover_pct: int


class QCCheck(BaseModel):
    check: str  # e.g., "range", "spike"
    result: str
    details: str


class QualityControl(BaseModel):
    qc_flag: str  # "passed", "suspect", "failed"
    checks: List[QCCheck] = Field(default_factory=list)


class Observation(BaseModel):
    observation_id: str
    station_id: str
    location_id: Optional[str] = None
    timestamp_utc: str
    variables: ObservationVariables
    quality_control: QualityControl
    ingested_at_utc: str


# User-related models
class DeliveryWebhook(BaseModel):
    enabled: str  # "yes" or "no"
    url: str


class DeliveryChannels(BaseModel):
    email: str  # "yes" or "no"
    sms: str    # "yes" or "no"
    push: str   # "yes" or "no"
    webhook: DeliveryWebhook


class QuietHoursLocal(BaseModel):
    start: str
    end: str


class AlertPreferences(BaseModel):
    hazards: List[str]
    min_severity: str
    delivery_channels: DeliveryChannels
    quiet_hours_local: QuietHoursLocal


class SavedLocation(BaseModel):
    location_id: str
    label: str  # "Home", "Office"


class PaymentMethod(BaseModel):
    source: str  # "card", "paypal", "wallet", "credit"
    id: str
    extra_info: Dict[str, Any] = Field(default_factory=dict)  # {"brand":..., "last_four":...} or {"amount": int}


class UserName(BaseModel):
    first_name: str
    last_name: str


class Address(BaseModel):
    address1: str
    address2: str
    city: str
    country: str
    state: str
    zip: str


class WeatherUser(BaseModel):
    user_id: str
    name: UserName
    address: Address
    email: str
    dob: str
    payment_methods: Dict[str, PaymentMethod] = Field(default_factory=dict)
    saved_locations: List[SavedLocation] = Field(default_factory=list)
    alert_preferences: AlertPreferences
    membership: str  # "free", "premium", "pro"
    subscriptions: List[str] = Field(default_factory=list)

class Location(BaseModel):
    location_id: str
    name: Name
    coordinates: Coordinates
    timezone: str
    nearby_station_ids: List[str]
    climate_normals: Dict[str, ClimateNormalsMonth]
    sun_times: Dict[str, SunTimesEntry]

class Location1(BaseModel):
    location_id: str
    name: Name

class Forecast(BaseModel):
    forecast_id: str
    location_id: str
    source_model: str  # e.g., "GFS", "ECMWF", "HRRR"
    issued_at_utc: str
    valid_from_utc: str
    valid_to_utc: str
    units: Units
    hourly: List[HourlyForecastEntry] = Field(default_factory=list)
    daily: List[DailyForecastEntry] = Field(default_factory=list)
    verification_by_date: Dict[str, VerificationEntry] = Field(default_factory=dict)
    attached_alert_ids: List[str] = Field(default_factory=list)

# DB container
class WeatherDB(BaseModel):
    locations: Dict[str, Location] = Field(default_factory=dict)
    forecasts: Dict[str, Forecast] = Field(default_factory=dict)
    observations: Dict[str, Observation] = Field(default_factory=dict)
    users: Dict[str, WeatherUser] = Field(default_factory=dict)


# Small helper models for convenience outputs
class LocationCode(BaseModel):
    location_id: str
    label: str  # e.g., "San Francisco, CA, US"

class AirportCode(BaseModel):
    iata: str = Field(description="IATA code")
    city: str = Field(description="City name")


AirportInfo = Annotated[list[AirportCode], Field(description="Airport information")]


class Name(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class Address(BaseModel):
    address1: str = Field(description="Primary address line")
    address2: Optional[str] = Field(
        None, description="Secondary address line (optional)"
    )
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


# Payment Related Models
class Payment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment")
    amount: int = Field(description="Payment amount in dollars")


class PaymentMethodBase(BaseModel):
    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = Field(
        description="Indicates this is a credit card payment method"
    )
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = Field(
        description="Indicates this is a gift card payment method"
    )
    amount: float = Field(description="Gift card value amount")
    id: str = Field(description="Unique identifier for the gift card")


class Certificate(PaymentMethodBase):
    source: Literal["certificate"] = Field(
        description="Indicates this is a certificate payment method"
    )
    amount: float = Field(description="Certificate value amount")


PaymentMethod = Union[CreditCard, GiftCard, Certificate]


class Passenger(BaseModel):
    first_name: str = Field(description="Passenger's first name")
    last_name: str = Field(description="Passenger's last name")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")


SeatPrices = Annotated[
    dict[CabinClass, int], Field(description="Prices for different cabin classes")
]
AvailableSeats = Annotated[
    dict[CabinClass, int],
    Field(description="Available seats for different cabin classes"),
]


class FlightDateStatusAvailable(BaseModel):
    status: Literal["available"] = Field(
        description="Indicates flight is available for booking"
    )
    available_seats: AvailableSeats = Field(description="Available seats by class")
    prices: SeatPrices = Field(description="Current prices by class")


class FlightDataStatusOnTime(BaseModel):
    status: Literal["on time"] = Field(description="Indicates flight is on time")
    estimated_departure_time_est: str = Field(
        description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDataStatusFlying(BaseModel):
    status: Literal["flying"] = Field(description="Indicates flight is in flight")
    actual_departure_time_est: str = Field(
        description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDateStatusLanded(BaseModel):
    status: Literal["landed"] = Field(description="Indicates flight has landed")
    actual_departure_time_est: str = Field(
        description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    actual_arrival_time_est: str = Field(
        description="Actual arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDateStatusCancelled(BaseModel):
    status: Literal["cancelled"] = Field(description="Indicates flight was cancelled")


class FlightDateStatusDelayed(BaseModel):
    status: Literal["delayed"] = Field(description="Indicates flight was delayed")
    estimated_departure_time_est: str = Field(
        description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


FlightDateStatus = Union[
    FlightDateStatusAvailable,
    FlightDateStatusLanded,
    FlightDateStatusCancelled,
    FlightDateStatusDelayed,
    FlightDataStatusFlying,
    FlightDataStatusOnTime,
]


class FlightBase(BaseModel):
    flight_number: str = Field(description="Unique flight identifier")
    origin: str = Field(description="IATA code for origin airport")
    destination: str = Field(description="IATA code for destination airport")


class Flight(FlightBase):
    scheduled_departure_time_est: str = Field(
        description="Scheduled departure time in EST in the format HH:MM:SS, e.g 06:00:00"
    )
    scheduled_arrival_time_est: str = Field(
        description="Scheduled arrival time in EST in the format HH:MM:SS, e.g 07:00:00"
    )
    dates: Dict[str, FlightDateStatus] = Field(
        description="Flight status by date (YYYY-MM-DD)"
    )


class DirectFlight(FlightBase):
    status: Literal["available"] = Field(
        description="Indicates flight is available for booking"
    )
    scheduled_departure_time_est: str = Field(
        description="Scheduled departure time in EST in the format HH:MM:SS, e.g 06:00:00"
    )
    scheduled_arrival_time_est: str = Field(
        description="Scheduled arrival time in EST in the format HH:MM:SS, e.g 07:00:00"
    )
    date: Optional[str] = Field(
        description="Flight date in YYYY-MM-DD format", default=None
    )
    available_seats: AvailableSeats = Field(description="Available seats by class")
    prices: SeatPrices = Field(description="Current prices by class")


class ReservationFlight(FlightBase):
    date: str = Field(description="Flight date in YYYY-MM-DD format")
    price: int = Field(description="Flight price in dollars.")


class FlightInfo(BaseModel):
    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(
        description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'."
    )


class User(BaseModel):
    user_id: str
    name: Name = Field(description="User's full name")
    address: Address = Field(description="User's address information")
    email: str = Field(description="User's email address")
    dob: str = Field(
        description="User's date of birth in the format YYYY-MM-DD, e.g 1990-04-05"
    )
    payment_methods: Dict[str, PaymentMethod] = Field(
        description="User's payment methods"
    )
    saved_locations: List[Location1] = Field(
        description="User's saved locations"
    )
    # alert_preferences: Dict[str, AlertPreferences] = Field(
    #     description="User's alert preferences"
    # )
    membership: MembershipLevel = Field(description="User's membership level")
    subscriptions: List[str] = Field(description="List of user's reservation IDs")


# Reservation Models
class Reservation(BaseModel):
    reservation_id: str = Field(description="Unique identifier for the reservation")
    user_id: str = Field(description="ID of the user who made the reservation")
    origin: str = Field(description="IATA code for trip origin")
    destination: str = Field(description="IATA code for trip destination")
    flight_type: FlightType = Field(description="Type of trip")
    cabin: CabinClass = Field(description="Selected cabin class")
    flights: List[ReservationFlight] = Field(
        description="List of flights in the reservation"
    )
    passengers: List[Passenger] = Field(
        description="List of passengers on the reservation"
    )
    payment_history: List[Payment] = Field(
        description="History of payments for this reservation"
    )
    created_at: str = Field(
        description="Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS"
    )
    total_baggages: int = Field(description="Total number of bags in reservation")
    nonfree_baggages: int = Field(description="Number of paid bags in reservation")
    insurance: Insurance = Field(description="Whether travel insurance was purchased")
    status: Optional[Literal["cancelled"]] = Field(
        description="Status of the reservation", default=None
    )


class WeatherDB(DB):
    """Database of all locations, forecasts, and observations."""

    locations: Dict[str, Location] = Field(
        description="Dictionary of all locations"
    )
    forecasts: Dict[str, Forecast] = Field(
        description="Dictionary of all forecasts"
    )
    observations: Dict[str, Observation] = Field(
        description="Dictionary of all observations"
    )
    # users: Dict[str, User] = Field(
    #     description="Dictionary of all users"
    # )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_locations = len(self.locations)
        num_forecasts = len(self.forecasts)
        num_observations = len(self.observations)
        return {
            "num_locations": num_locations,
            "num_forecasts": num_forecasts,
            "num_observations": num_observations,
        }


def get_db():
    return FlightDB.load(AIRLINE_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
