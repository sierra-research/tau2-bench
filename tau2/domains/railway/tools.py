"""
Railway tools: Toolkit for the railway reservation system.

This toolkit mirrors the structure and common operations of the AirlineTools,
adapted to railway data models. It provides frequently used tools for:
- searching trains,
- booking reservations,
- updating reservations (bags, bikes, trains, passengers),
- cancelling reservations,
- querying users/reservations,
- getting train status,
- wallet top-ups,
- and a few generic helpers.

Assumptions:
- Train DB does not model seat availability; we simulate seat assignment and pricing.
- Pricing is derived from service_type and travel_class using fixed base fares.
- Membership discounts apply to fare components only (not to fees/insurance).
- Wallet balances are whole-dollar integers; wallet payments must be in whole dollars.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel

from tau2.domains.railway.data_model import (
    Insurance,
    MealPreference,
    Passenger,
    RailPayment,
    RailUser,
    ReservationStatus,
    ServiceType,
    Train,
    TrainDateStatus,
    TrainDateStatusCancelled,
    TrainDateStatusDelayed,
    TrainDateStatusOnTime,
    TrainDB,
    TrainInfo,
    TrainReservation,
    TravelClass,
    TripType,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


# Simple result model for direct train search
class DirectTrain(BaseModel):
    train_number: str
    train_name: str
    origin: str
    destination: str
    service_type: ServiceType
    date: str
    status: str
    scheduled_departure_time_local: str
    scheduled_arrival_time_local: str
    platform: Optional[str] = None


class RailwayTools(ToolKitBase):
    """All the tools for the railway domain."""

    db: TrainDB

    def __init__(self, db: TrainDB) -> None:
        super().__init__(db)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _get_user(self, user_id: str) -> RailUser:
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> TrainReservation:
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_train(self, train_number: str) -> Train:
        if train_number not in self.db.trains:
            raise ValueError(f"Train {train_number} not found")
        return self.db.trains[train_number]

    def _get_train_instance(self, train_number: str, date: str) -> TrainDateStatus:
        train = self._get_train(train_number)
        if date not in train.dates:
            raise ValueError(f"Train {train_number} not found on date {date}")
        return train.dates[date]

    def _get_new_reservation_id(self) -> str:
        # Assume at most 3 reservations per task
        for reservation_id in ["TRN001", "TRN002", "TRN003"]:
            if reservation_id not in self.db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_pnr(self) -> str:
        # Assume at most 3 PNRs per task
        candidate_pnrs = ["PNR1AA", "PNR2BB", "PNR3CC"]
        used = {res.pnr for res in self.db.reservations.values()}
        for pnr in candidate_pnrs:
            if pnr not in used:
                return pnr
        raise ValueError("Too many PNRs")

    def _get_datetime(self) -> str:
        return "2024-05-15T15:00:00"

    def _coach_for_class(self, travel_class: TravelClass) -> str:
        mapping = {
            "sleeper": "S1",
            "ac_2_tier": "A1",
            "first_class": "FC1",
        }
        return mapping[travel_class]

    def _compute_segment_price(self, train: Train, travel_class: TravelClass) -> int:
        """
        Compute a base price per passenger per segment as an integer (dollars).
        This is a simple model using service_type and travel_class.
        """
        base_by_service = {
            "high_speed": {"sleeper": 120, "ac_2_tier": 90, "first_class": 150},
            "express": {"sleeper": 80, "ac_2_tier": 65, "first_class": 110},
            "regional": {"sleeper": 50, "ac_2_tier": 40, "first_class": 75},
        }
        return base_by_service[train.service_type][travel_class]

    def _apply_membership_discount(self, amount: float, user: RailUser) -> float:
        """
        Apply membership discount only on fare component (not fees/insurance).
        - gold: 10%
        - silver: 5%
        - regular: 0%
        """
        discounts = {"gold": 0.10, "silver": 0.05, "regular": 0.0}
        rate = discounts.get(user.membership, 0.0)
        return round(amount * (1 - rate), 2)

    def _payment_for_update(
        self, user: RailUser, payment_id: str, total_price: float
    ) -> Optional[RailPayment]:
        """
        Process payment for update reservation.
        - Wallet payments must be whole dollars and sufficient in balance.
        - Card/Other payments are assumed to be always processable (no stored balance).
        - For refunds (negative total_price), we add a negative RailPayment record and do not adjust wallet balances.

        Returns a RailPayment if total_price != 0; otherwise None.
        """
        if payment_id not in user.payment_methods:
            raise ValueError(f"Payment method not found: {payment_id}, {user.payment_methods}")
        method = user.payment_methods[payment_id]

        if total_price > 0:
            # Deduct from wallet if used
            if method.source == "wallet":
                # wallet has integer amount (whole dollars)
                if abs(total_price - int(total_price)) > 1e-9:
                    raise ValueError("Wallet payments must be whole dollars")
                if method.extra_info.amount < int(total_price):
                    raise ValueError("Wallet balance is not enough")
                method.extra_info.amount -= int(total_price)

        # Create payment record if any charge/refund applies
        if abs(total_price) > 1e-9:
            return RailPayment(payment_id=payment_id, amount=round(total_price, 2))
        return None

    def _search_direct_train(
        self,
        date: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        leave_after: Optional[str] = None,
    ) -> list[DirectTrain]:
        """
        Search for direct trains on a given date with optional filters.
        """
        results: list[DirectTrain] = []
        for train in self.db.trains.values():
            # Must run on that date
            if date not in train.dates:
                # print(190,origin,destination,date,train.dates)
                continue
            # Filter by route
            if origin is not None and train.origin != origin:
                # print(194,origin,train.origin)
                continue
            if destination is not None and train.destination != destination:
                # pringt(197,destination,train.destination)
                continue
            # Get date status
            date_status = train.dates[date]
            status = date_status.status
            platform = None
            if isinstance(date_status, (TrainDateStatusOnTime, TrainDateStatusDelayed)):
                platform = date_status.platform
            # Optional time filter - use scheduled departure time
            if leave_after is not None and train.scheduled_departure_time_local < leave_after:
                # print(207,leave_after,train.scheduled_departure_time_local,leave_after)
                continue

            results.append(
                DirectTrain(
                    train_number=train.train_number,
                    train_name=train.train_name,
                    origin=train.origin,
                    destination=train.destination,
                    service_type=train.service_type,
                    date=date,
                    status=status,
                    scheduled_departure_time_local=train.scheduled_departure_time_local,
                    scheduled_arrival_time_local=train.scheduled_arrival_time_local,
                    platform=platform,
                )
            )
        return results

    # -------------------------
    # Tools
    # -------------------------
    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> RailUser:
        """
        Get the details of a user, including their reservations.
        """
        return self._get_user(user_id)

    @is_tool(ToolType.READ)
    def get_reservation_details(self, reservation_id: str) -> TrainReservation:
        """
        Get the details of a reservation.
        """
        return self._get_reservation(reservation_id)

    @is_tool(ToolType.READ)
    def search_direct_train(
        self, origin: str, destination: str, date: str
    ) -> list[DirectTrain]:
        """
        Search for direct trains between two stations on a specific date.
        """
        return self._search_direct_train(date=date, origin=origin, destination=destination)

    @is_tool(ToolType.READ)
    def search_onestop_train(
        self, origin: str, destination: str, date: str
    ) -> list[tuple[DirectTrain, DirectTrain]]:
        """
        Search for one-stop train itineraries between two stations on a specific date.
        Returns a list of (first_leg, second_leg).
        Note: This simple version only considers connections on the same day.
        """
        results: list[Tuple[DirectTrain, DirectTrain]] = []
        for first_leg in self._search_direct_train(date=date, origin=origin, destination=None):
            # Only consider first legs that are not cancelled
            if first_leg.status == "cancelled":
                continue
            leave_after = first_leg.scheduled_arrival_time_local
            for second_leg in self._search_direct_train(
                date=date,
                origin=first_leg.destination,
                destination=destination,
                leave_after=leave_after,
            ):
                if second_leg.status == "cancelled":
                    continue
                results.append((first_leg, second_leg))
        return results

    @is_tool(ToolType.WRITE)
    def book_train_reservation(
        self,
        user_id: str,
        origin: str,
        destination: str,
        trip_type: TripType,
        travel_class: TravelClass,
        trains: List[TrainInfo | dict],
        passengers: List[Passenger | dict],
        payment_methods: List[RailPayment | dict],
        total_bags: int,
        bikes: int,
        meal_preference: MealPreference,
        insurance: Insurance,
    ) -> TrainReservation:
        """
        Book a train reservation.

        Args:
            user_id: The user ID, e.g., 'jane_doe_123'.
            origin: Trip origin station code, e.g., 'NYC'.
            destination: Trip destination station code, e.g., 'BOS'.
            trip_type: 'one_way', 'round_trip', or 'multi_city'.
            travel_class: 'sleeper', 'ac_2_tier', or 'first_class'.
            trains: Array of {train_number, date}.
            passengers: Array of passenger objects.
            payment_methods: Array of {payment_id, amount} (amount in dollars; wallet must be whole dollars).
            total_bags: Total number of bags for the reservation.
            bikes: Number of bikes.
            meal_preference: 'veg', 'non_veg', or 'none'.
            insurance: 'yes' or 'no'.
        """
        if all(isinstance(t, dict) for t in trains):
            trains = [TrainInfo(**t) for t in trains]
        if all(isinstance(p, dict) for p in passengers):
            passengers = [Passenger(**p) for p in passengers]
        if all(isinstance(pm, dict) for pm in payment_methods):
            payment_methods = [RailPayment(**pm) for pm in payment_methods]

        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()
        pnr = self._get_new_pnr()

        # Build segments and compute fare
        segments = []
        fare_total = 0  # fare component only (before fees/insurance)
        num_pax = len(passengers)

        for tinfo in trains:
            train = self._get_train(tinfo.train_number)
            date_status = self._get_train_instance(tinfo.train_number, tinfo.date)
            # Cannot book cancelled trains
            if isinstance(date_status, TrainDateStatusCancelled):
                raise ValueError(
                    f"Train {tinfo.train_number} is cancelled on {tinfo.date}"
                )

            price_each = self._compute_segment_price(train, travel_class)
            fare_total += price_each * num_pax

            # Simple seat assignment: consecutive seat numbers as strings
            seat_numbers = [str(21 + i) for i in range(num_pax)]
            segment = {
                "origin": train.origin,
                "destination": train.destination,
                "train_number": tinfo.train_number,
                "date": tinfo.date,
                "coach": self._coach_for_class(travel_class),
                "seat_numbers": seat_numbers,
                "price": price_each,  # per-passenger price for this segment
            }
            segments.append(segment)

        # Membership discount on fares only
        fare_total_after_discount = self._apply_membership_discount(fare_total, user)

        # Insurance fee (e.g., $20 per passenger if opted-in)
        insurance_fee = 20 * num_pax if insurance == "yes" else 0

        # Bags and bikes fees:
        # - 1 bag per passenger included; additional bags: $15 each
        included_bags = num_pax
        extra_bags = max(0, total_bags - included_bags)
        bags_fee = 15 * extra_bags
        # - Bikes: $10 each
        bikes_fee = 10 * bikes

        total_price = round(fare_total_after_discount + insurance_fee + bags_fee + bikes_fee, 2)

        # Validate payment methods and balances
        for pay in payment_methods:
            if pay.payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {pay.payment_id} not found")
            method = user.payment_methods[pay.payment_id]
            if method.source == "wallet":
                # wallet payments must be whole dollars
                if abs(pay.amount - int(pay.amount)) > 1e-9:
                    raise ValueError("Wallet payments must be in whole dollars")
                if method.extra_info.amount < int(pay.amount):
                    raise ValueError(f"Not enough balance in wallet {pay.payment_id}")

        # Check sum of payments
        total_payment = round(sum(p.amount for p in payment_methods), 2)
        if abs(total_payment - total_price) > 1e-9:
            raise ValueError(
                f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}, {payment_methods}"
            )

        # Deduct wallet balances
        for pay in payment_methods:
            method = user.payment_methods[pay.payment_id]
            if method.source == "wallet":
                method.extra_info.amount -= int(pay.amount)

        # Build reservation segments typed
        typed_segments = [
            # price stored per segment as integer per data model
            # represents per-passenger price for this segment
            # (the total paid can include discounts/fees in payment history)
            # seat_numbers length == number of passengers
            # coach derived from travel_class
            TrainReservation.model_fields["trains"].annotation.__args__[0](**seg)  # ReservationTrainSegment
            for seg in segments
        ]

        reservation = TrainReservation(
            reservation_id=reservation_id,
            user_id=user_id,
            origin=origin,
            destination=destination,
            trip_type=trip_type,
            trains=typed_segments,
            passengers=deepcopy(passengers),
            payment_history=deepcopy(payment_methods),
            created_at=self._get_datetime(),
            total_bags=total_bags,
            bikes=bikes,
            meal_preference=meal_preference,
            insurance=insurance,
            pnr=pnr,
            status="confirmed",
        )

        # Update DB
        self.db.reservations[reservation_id] = reservation
        self.db.users[user_id].reservations.append(reservation_id)
        return reservation

    @is_tool(ToolType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> TrainReservation:
        """
        Cancel the whole reservation: mark as cancelled and append refund payments (negative amounts).
        Note: Seat releases are not modeled in this DB.
        """
        reservation = self._get_reservation(reservation_id)
        logger.debug(reservation.model_dump_json(indent=2))

        refunds = []
        for payment in reservation.payment_history:
            refunds.append(
                RailPayment(
                    payment_id=payment.payment_id,
                    amount=-payment.amount,
                )
            )
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        logger.debug(self._get_reservation(reservation_id).model_dump_json(indent=2))
        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_bags(
        self,
        reservation_id: str,
        total_bags: int,
        payment_id: str,
    ) -> TrainReservation:
        """
        Update the total number of bags on a reservation.
        Charges $15 per additional bag beyond 1 per passenger; refunds if fewer bags now.
        """
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)
        num_pax = len(reservation.passengers)
        included_bags = num_pax

        previous_extra = max(0, reservation.total_bags - included_bags)
        new_extra = max(0, total_bags - included_bags)

        delta_bags = new_extra - previous_extra
        delta_price = 15 * delta_bags  # can be positive or negative

        payment = self._payment_for_update(user, payment_id, float(delta_price))
        if payment is not None:
            reservation.payment_history.append(payment)

        reservation.total_bags = total_bags
        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_bikes(
        self,
        reservation_id: str,
        bikes: int,
        payment_id: str,
    ) -> TrainReservation:
        """
        Update the number of bikes on a reservation.
        Charges $10 per added bike; refunds if fewer bikes now.
        """
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        delta_bikes = bikes - reservation.bikes
        delta_price = 10 * delta_bikes  # can be positive or negative

        payment = self._payment_for_update(user, payment_id, float(delta_price))
        if payment is not None:
            reservation.payment_history.append(payment)

        reservation.bikes = bikes
        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_trains(
        self,
        reservation_id: str,
        travel_class: TravelClass,
        trains: List[TrainInfo | dict],
        payment_id: str,
    ) -> TrainReservation:
        """
        Update the train segments of a reservation.
        Rebuilds the itinerary with the provided ENTIRE list of segments.
        Charges the difference (or refunds) compared to the original fare component.
        """
        if all(isinstance(t, dict) for t in trains):
            trains = [TrainInfo(**t) for t in trains]

        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)
        num_pax = len(reservation.passengers)

        # Build new segments and compute new fare (before discounts/fees)
        new_segments = []
        new_fare_total = 0
        for tinfo in trains:
            train = self._get_train(tinfo.train_number)
            date_status = self._get_train_instance(tinfo.train_number, tinfo.date)
            if isinstance(date_status, TrainDateStatusCancelled):
                raise ValueError(
                    f"Train {tinfo.train_number} is cancelled on {tinfo.date}"
                )
            price_each = self._compute_segment_price(train, travel_class)
            new_fare_total += price_each * num_pax
            seat_numbers = [str(21 + i) for i in range(num_pax)]
            new_segments.append(
                TrainReservation.model_fields["trains"].annotation.__args__[0](  # ReservationTrainSegment
                    origin=train.origin,
                    destination=train.destination,
                    train_number=tinfo.train_number,
                    date=tinfo.date,
                    coach=self._coach_for_class(travel_class),
                    seat_numbers=seat_numbers,
                    price=price_each,
                )
            )

        # Old fare total (sum of per-segment per-passenger prices)
        old_fare_total = sum(seg.price for seg in reservation.trains) * num_pax

        # Apply membership discount to the fare difference only
        fare_diff = new_fare_total - old_fare_total
        fare_diff_after_discount = self._apply_membership_discount(fare_diff, user)

        # Create payment or refund
        payment = self._payment_for_update(user, payment_id, float(fare_diff_after_discount))
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation itinerary and travel class
        reservation.trains = new_segments
        # Note: travel_class is represented via coach in segments; no explicit field on reservation.
        return reservation

    @is_tool(ToolType.WRITE)
    def update_reservation_passengers(
        self, reservation_id: str, passengers: List[Passenger | dict]
    ) -> TrainReservation:
        """
        Update the passenger information of a reservation.
        The passenger count must remain the same.
        """
        if all(isinstance(p, dict) for p in passengers):
            passengers = [Passenger(**p) for p in passengers]
        reservation = self._get_reservation(reservation_id)
        if len(passengers) != len(reservation.passengers):
            raise ValueError("Number of passengers does not match")
        reservation.passengers = deepcopy(passengers)
        return reservation

    @is_tool(ToolType.READ)
    def get_train_status(self, train_number: str, date: str) -> str:
        """
        Get the status ('on time', 'delayed', 'cancelled') of a train on a given date.
        """
        return self._get_train_instance(train_number, date).status

    @is_tool(ToolType.READ)
    def list_all_routes(self) -> list[tuple[str, str]]:
        """
        List all unique (origin, destination) routes in the database.
        """
        routes = {(t.origin, t.destination) for t in self.db.trains.values()}
        return sorted(list(routes))

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.
        Allowed chars: digits, + - * / ( ) . and spaces.
        """
        if not all(c in "0123456789+-*/(). " for c in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if:
          - the user explicitly asks for a human agent, or
          - given the policy and available tools, the issue cannot be solved.
        """
        return "Transfer successful"

    @is_tool(ToolType.WRITE)
    def add_wallet_funds(self, user_id: str, amount: int) -> str:
        """
        Add funds to a user's wallet. If the user does not have a wallet, create one.
        Wallet stores whole-dollar amounts only.

        Args:
            user_id: User identifier.
            amount: Whole-dollar amount to add.

        Returns:
            A message indicating the wallet was funded.

        Raises:
            ValueError: If amount is not a positive integer.
        """
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("Amount must be a positive integer (whole dollars)")
        user = self._get_user(user_id)

        # Find existing wallet or create one
        wallet_id = None
        for pid, method in user.payment_methods.items():
            if method.source == "wallet":
                wallet_id = pid
                break

        if wallet_id is None:
            # Create a new wallet id (up to 3 per task)
            for wid in ["wallet_1001", "wallet_1002", "wallet_1003"]:
                if wid not in user.payment_methods:
                    from tau2.domains.railway.data_model import WalletPayment, WalletExtraInfo

                    user.payment_methods[wid] = WalletPayment(
                        source="wallet",
                        id=wid,
                        extra_info=WalletExtraInfo(amount=0),
                    )
                    wallet_id = wid
                    break
        if wallet_id is None:
            raise ValueError("Too many wallets")

        user.payment_methods[wallet_id].extra_info.amount += amount
        return f"Wallet {wallet_id} funded with ${amount}. New balance: ${user.payment_methods[wallet_id].extra_info.amount}"

    @is_tool(ToolType.READ)
    def get_db_statistics(self) -> dict:
        """
        Get basic statistics of the railway database.
        """
        return self.db.get_statistics()
