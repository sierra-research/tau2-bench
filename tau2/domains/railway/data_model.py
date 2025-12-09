from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.environment.db import DB

# Reuse existing enums where applicable
Insurance = Literal["yes", "no"]
MembershipLevel = Annotated[
    Literal["gold", "silver", "regular"], Field(description="Membership level")
]


# Shared/reusable models from airline domain
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


class Passenger(BaseModel):
    first_name: str = Field(description="Passenger's first name")
    last_name: str = Field(description="Passenger's last name")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")


# Railway-specific enums
ServiceType = Literal["high_speed", "express", "regional"]

TripType = Literal["one_way", "round_trip", "multi_city"]

TravelClass = Literal["sleeper", "ac_2_tier", "first_class"]

MealPreference = Literal["veg", "non_veg", "none"]

ReservationStatus = Literal["confirmed", "waitlisted", "cancelled"]


# Railway payment-related models
class RailPayment(BaseModel):
    payment_id: str = Field(description="Unique identifier for the payment")
    amount: float = Field(description="Payment amount in dollars (can include cents)")


class CardExtraInfo(BaseModel):
    brand: str = Field(description="Card brand, e.g., visa, mastercard")
    last_four: str = Field(description="Last four digits of the card")


class WalletExtraInfo(BaseModel):
    amount: int = Field(description="Stored wallet amount or balance")


class PaymentMethodRailBase(BaseModel):
    source: str = Field(description="Type of payment method, e.g., card, wallet")
    id: str = Field(description="Unique identifier for the payment method")


class CardPayment(PaymentMethodRailBase):
    source: Literal["card"] = Field(
        description="Indicates this is a card payment method"
    )
    extra_info: CardExtraInfo = Field(description="Additional card details")


class WalletPayment(PaymentMethodRailBase):
    source: Literal["wallet"] = Field(
        description="Indicates this is a wallet payment method"
    )
    extra_info: WalletExtraInfo = Field(description="Additional wallet details")


class OtherPayment(PaymentMethodRailBase):
    source: Literal["other"] = Field(
        description="Indicates a different or custom payment source"
    )
    extra_info: Dict[str, Any] = Field(
        description="Arbitrary extra info for other payment types"
    )


PaymentMethodRail = Union[CardPayment, WalletPayment, OtherPayment]


# Train status by date
class TrainDateStatusOnTime(BaseModel):
    status: Literal["on time"] = Field(description="Indicates train is on time")
    platform: str = Field(description="Assigned platform for the train")
    actual_departure_time_local: Optional[str] = Field(
        None,
        description="Actual departure time in local timezone in the format YYYY-MM-DDTHH:MM:SS",
    )
    actual_arrival_time_local: Optional[str] = Field(
        None,
        description="Actual arrival time in local timezone in the format YYYY-MM-DDTHH:MM:SS",
    )


class TrainDateStatusDelayed(BaseModel):
    status: Literal["delayed"] = Field(description="Indicates train is delayed")
    platform: str = Field(description="Assigned platform for the train")
    actual_departure_time_local: Optional[str] = Field(
        None,
        description="Actual departure time in local timezone in the format YYYY-MM-DDTHH:MM:SS",
    )
    actual_arrival_time_local: Optional[str] = Field(
        None,
        description="Actual arrival time in local timezone in the format YYYY-MM-DDTHH:MM:SS",
    )


class TrainDateStatusCancelled(BaseModel):
    status: Literal["cancelled"] = Field(description="Indicates train is cancelled")
    platform: Optional[str] = Field(
        None, description="Platform (if applicable or known)"
    )
    actual_departure_time_local: Optional[str] = Field(
        None,
        description="Actual departure time in local timezone in the format YYYY-MM-DDTHH:MM:SS, if any",
    )
    actual_arrival_time_local: Optional[str] = Field(
        None,
        description="Actual arrival time in local timezone in the format YYYY-MM-DDTHH:MM:SS, if any",
    )


TrainDateStatus = Union[
    TrainDateStatusOnTime, TrainDateStatusDelayed, TrainDateStatusCancelled
]


# Train models
class TrainBase(BaseModel):
    origin: str = Field(description="Station code for origin")
    destination: str = Field(description="Station code for destination")
    train_number: str = Field(description="Unique train number identifier")
    train_name: str = Field(description="Official train name")
    service_type: ServiceType = Field(description="Type of train service")


class Train(TrainBase):
    scheduled_departure_time_local: str = Field(
        description="Scheduled departure time in local timezone, format HH:MM:SS, e.g., 06:00:00"
    )
    scheduled_arrival_time_local: str = Field(
        description="Scheduled arrival time in local timezone, format HH:MM:SS, e.g., 07:00:00"
    )
    dates: Dict[str, TrainDateStatus] = Field(
        description="Train status by date (YYYY-MM-DD)"
    )


class TrainInfo(BaseModel):
    train_number: str = Field(description="Train number, e.g., 'HS123'")
    date: str = Field(
        description="The date for the train in the format 'YYYY-MM-DD', e.g., '2024-05-01'."
    )


# User and railcard models
class Railcard(BaseModel):
    type: str = Field(description="Railcard type, e.g., Senior, Student")
    number: str = Field(description="Railcard number")
    expiry: str = Field(
        description="Railcard expiry date in the format YYYY-MM-DD, e.g., 2026-12-31"
    )


class RailUser(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: Name = Field(description="User's full name")
    address: Address = Field(description="User's address information")
    email: str = Field(description="User's email address")
    dob: str = Field(
        description="User's date of birth in the format YYYY-MM-DD, e.g., 1990-04-05"
    )
    payment_methods: Dict[str, PaymentMethodRail] = Field(
        description="User's saved payment methods"
    )
    saved_passengers: List[Passenger] = Field(
        description="User's saved passenger information"
    )
    membership: MembershipLevel = Field(description="User's membership level")
    railcards: List[Railcard] = Field(description="List of user's railcards")
    reservations: List[str] = Field(description="List of user's reservation IDs")


# Reservation models
class ReservationTrainSegment(BaseModel):
    origin: str = Field(description="Station code for origin")
    destination: str = Field(description="Station code for destination")
    train_number: str = Field(description="Train number for this segment")
    date: str = Field(description="Travel date for this segment in YYYY-MM-DD format")
    coach: str = Field(description="Coach identifier, e.g., S3, A1")
    seat_numbers: List[str] = Field(
        description='List of seat/berth numbers, e.g., ["21", "22"]'
    )
    price: int = Field(description="Price for this segment in dollars")


class TrainReservation(BaseModel):
    reservation_id: str = Field(description="Unique identifier for the reservation")
    user_id: str = Field(description="ID of the user who made the reservation")
    origin: str = Field(description="Station code for trip origin")
    destination: str = Field(description="Station code for trip destination")
    trip_type: TripType = Field(description="Type of trip")
    trains: List[ReservationTrainSegment] = Field(
        description="List of train segments in the reservation"
    )
    passengers: List[Passenger] = Field(
        description="List of passengers on the reservation"
    )
    payment_history: List[RailPayment] = Field(
        description="History of payments for this reservation"
    )
    created_at: str = Field(
        description="Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS"
    )
    total_bags: int = Field(description="Total number of bags in reservation")
    bikes: int = Field(
        description="Number of bikes for conveyance if applicable", default=0
    )
    meal_preference: MealPreference = Field(
        description="Meal preference for the reservation"
    )
    insurance: Insurance = Field(description="Whether travel insurance was purchased")
    pnr: str = Field(description="Passenger Name Record associated with the reservation")
    status: ReservationStatus = Field(description="Status of the reservation")


# Database model
class TrainDB(DB):
    """Database of all trains, users, and reservations."""

    trains: Dict[str, Train] = Field(
        description="Dictionary of all trains indexed by unique train ID"
    )
    users: Dict[str, RailUser] = Field(
        description="Dictionary of all users indexed by user ID"
    )
    reservations: Dict[str, TrainReservation] = Field(
        description="Dictionary of all reservations indexed by reservation ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_trains = len(self.trains)
        num_train_instances = sum(len(train.dates) for train in self.trains.values())
        num_users = len(self.users)
        num_reservations = len(self.reservations)
        return {
            "num_trains": num_trains,
            "num_train_instances": num_train_instances,
            "num_users": num_users,
            "num_reservations": num_reservations,
        }