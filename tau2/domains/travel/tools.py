# Travel agency tools: frequently used APIs built on tau2.domains.travel.data_model
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from tau2.domains.travel.data_model import (
    Activity,
    Accommodation,
    Agent,
    AgencyInfo,
    AvailabilityDay,
    Booking,
    BookingAddOn,
    BookingPayment,
    BookingTravelerInfo,
    Location,
    Package,
    PackageDeparture,
    PackagePolicies,
    PaymentExtraInfo,
    PersonName,
    PostalAddress,
    RoomingInfo,
    TransportationInfo,
    Traveler,
    TravelerPaymentMethod,
    TravelerPreferences,
    TravelAgencyDB,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class TravelAgencyTools(ToolKitBase):
    """Tools for the travel agency domain."""

    db: TravelAgencyDB

    def __init__(self, db: TravelAgencyDB) -> None:
        super().__init__(db)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _get_traveler(self, traveler_id: str) -> Traveler:
        if traveler_id not in self.db.travelers:
            raise ValueError(f"Traveler {traveler_id} not found")
        return self.db.travelers[traveler_id]

    def _get_agent(self, agent_id: str) -> Agent:
        if agent_id not in self.db.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.db.agents[agent_id]

    def _get_package(self, package_id: str) -> Package:
        if package_id not in self.db.packages:
            raise ValueError(f"Package {package_id} not found")
        return self.db.packages[package_id]

    def _get_booking(self, booking_id: str) -> Tuple[Traveler, Booking]:
        """
        Find a booking by ID across all travelers.

        Returns:
            (Traveler owning the booking, Booking)
        """
        for traveler in self.db.travelers.values():
            for booking in traveler.bookings:
                if booking.booking_id == booking_id:
                    return traveler, booking
        raise ValueError(f"Booking {booking_id} not found")

    def _get_new_booking_id(self) -> str:
        """Get a new booking id. Assume each task makes at most 3 bookings."""
        for booking_id in ["TAU001", "TAU002", "TAU003"]:
            # ensure uniqueness across all travelers' bookings
            try:
                self._get_booking(booking_id)
            except ValueError:
                return booking_id
        raise ValueError("Too many bookings created")

    def _get_today_date(self) -> str:
        """Get current date in YY-MM-DD format (simple stub)."""
        return "2024-05-15"

    def _validate_departure(
        self, package: Package, departure_date: str, required_slots: int
    ) -> PackageDeparture:
        if departure_date not in package.departures:
            raise ValueError(
                f"Package {package.package_id} not offered on {departure_date}"
            )
        dep = package.departures[departure_date]
        if dep.status != "available":
            raise ValueError(
                f"Departure on {departure_date} for package {package.package_id} is not available"
            )
        if dep.available_slots < required_slots:
            raise ValueError(
                f"Not enough slots on {departure_date}. Needed {required_slots}, available {dep.available_slots}"
            )
        return dep

    def _calc_total_price(
        self,
        base_price_per_person: float,
        num_travelers: int,
        add_ons: List[BookingAddOn],
        insurance: Optional[str],
    ) -> float:
        total = base_price_per_person * num_travelers
        total += sum(addon.price for addon in add_ons)
        # Simple insurance mapping (example policy)
        if insurance:
            ins_lower = insurance.strip().lower()
            if ins_lower == "standard":
                total += 50.0 * num_travelers
            elif ins_lower == "premium":
                total += 100.0 * num_travelers
            else:
                # No charge for unknown label; kept for compatibility
                total += 0.0
        return round(total, 2)

    def _validate_payments(
        self,
        traveler: Traveler,
        payments: List[BookingPayment],
        expected_total: float,
    ) -> None:
        # Validate payment methods exist
        for p in payments:
            if p.payment_id not in traveler.payment_methods:
                raise ValueError(f"Payment method {p.payment_id} not found for traveler")
        paid = round(sum(p.amount for p in payments), 2)
        if round(expected_total, 2) != paid:
            raise ValueError(
                f"Payment amount does not add up, total price is {expected_total}, but paid {paid}"
            )

    # ---------------------------
    # Read tools
    # ---------------------------
    @is_tool(ToolType.READ)
    def get_traveler_details(self, traveler_id: str) -> Traveler:
        """Get traveler profile with bookings."""
        return self._get_traveler(traveler_id)

    @is_tool(ToolType.READ)
    def get_agent_details(self, agent_id: str) -> Agent:
        """Get agent profile."""
        return self._get_agent(agent_id)

    @is_tool(ToolType.READ)
    def get_package_details(self, package_id: str) -> Package:
        """Get full package details."""
        return self._get_package(package_id)

    @is_tool(ToolType.READ)
    def get_booking_details(self, booking_id: str) -> Booking:
        """Get a booking's full details."""
        _, booking = self._get_booking(booking_id)
        return booking

    @is_tool(ToolType.READ)
    def search_packages(
        self,
        destination_city: Optional[str] = None,
        destination_country: Optional[str] = None,
        category: Optional[str] = None,
        departure_date: Optional[str] = None,
    ) -> List[Package]:
        """
        Search packages by destination/filter criteria.

        Args:
            destination_city: Filter by destination city name
            destination_country: Filter by destination country name
            category: Filter by package category
            departure_date: Filter packages that depart on this date (YY-MM-DD)
        """
        results: List[Package] = []
        for pkg in self.db.packages.values():
            # destination filter
            if destination_city or destination_country:
                dest_match = any(
                    ((destination_city is None or loc.city == destination_city)
                     and (destination_country is None or loc.country == destination_country))
                    for loc in pkg.destinations
                )
                if not dest_match:
                    continue
            # category filter
            if category and pkg.category != category:
                continue
            # departure_date filter
            if departure_date and departure_date not in pkg.departures:
                continue
            # Only include if there exists at least one available departure
            if any(d.status == "available" and d.available_slots > 0 for d in pkg.departures.values()):
                results.append(pkg)
        return results

    @is_tool(ToolType.READ)
    def list_all_destinations(self) -> List[Location]:
        """
        Return a unique list of all destinations across all packages.
        """
        seen: set[Tuple[str, str]] = set()
        dests: List[Location] = []
        for pkg in self.db.packages.values():
            for loc in pkg.destinations:
                key = (loc.city, loc.country)
                if key not in seen:
                    seen.add(key)
                    dests.append(loc)
        return dests

    # ---------------------------
    # Write tools
    # ---------------------------
    @is_tool(ToolType.WRITE)
    def book_package(
        self,
        traveler_id: str,
        package_id: str,
        departure_date: str,
        travelers: List[BookingTravelerInfo | Dict[str, Any]],
        rooming: RoomingInfo | Dict[str, Any],
        add_ons: List[BookingAddOn | Dict[str, Any]],
        insurance: Optional[str],
        payment_methods: List[BookingPayment | Dict[str, Any]],
        agent_id: Optional[str] = None,
    ) -> Booking:
        """
        Create a new booking for a traveler.

        Payment rules:
          - The payment_ids used must exist in traveler's payment_methods.
          - Sum(payment.amount) must equal computed total price.
        """
        # Normalize inputs
        if all(isinstance(t, dict) for t in travelers):
            travelers = [BookingTravelerInfo(**t) for t in travelers]  # type: ignore
        if isinstance(rooming, dict):
            rooming = RoomingInfo(**rooming)  # type: ignore
        if all(isinstance(a, dict) for a in add_ons):
            add_ons = [BookingAddOn(**a) for a in add_ons]  # type: ignore
        if all(isinstance(p, dict) for p in payment_methods):
            payment_methods = [BookingPayment(**p) for p in payment_methods]  # type: ignore

        traveler = self._get_traveler(traveler_id)
        package = self._get_package(package_id)

        # agent selection
        if agent_id is None:
            if not package.managed_by_agents:
                raise ValueError("No managing agent available for this package")
            agent_id = package.managed_by_agents[0]
        agent = self._get_agent(agent_id)

        num_travelers = len(travelers)  # list of BookingTravelerInfo
        departure = self._validate_departure(package, departure_date, required_slots=num_travelers)

        # Compute total price
        total_price = self._calc_total_price(
            base_price_per_person=departure.base_price,
            num_travelers=num_travelers,
            add_ons=add_ons,  # type: ignore
            insurance=insurance,
        )

        # Validate and record payments
        self._validate_payments(traveler, payment_methods, total_price)

        booking_id = self._get_new_booking_id()
        booking = Booking(
            booking_id=booking_id,
            package_id=package_id,
            agent_id=agent_id,
            booking_date=self._get_today_date(),
            departure_date=departure_date,
            status="confirmed",
            travelers=deepcopy(travelers),  # type: ignore
            rooming=deepcopy(rooming),  # type: ignore
            add_ons=deepcopy(add_ons),  # type: ignore
            insurance=insurance,
            payment_history=deepcopy(payment_methods),  # type: ignore
            total_price=total_price,
            notes=None,
        )

        # Update DB state
        traveler.bookings.append(booking)
        # Decrement available slots
        package.departures[departure_date].available_slots -= num_travelers
        # Link traveler to agent (assigned list)
        if traveler_id not in agent.assigned_travelers:
            agent.assigned_travelers.append(traveler_id)
        # Track booking ID in agent records
        if booking_id not in agent.bookings_handled:
            agent.bookings_handled.append(booking_id)

        return booking

    @is_tool(ToolType.WRITE)
    def cancel_booking(self, booking_id: str) -> Booking:
        """
        Cancel a booking and process refunds (negative payments).
        Restores departure available slots.
        """
        traveler, booking = self._get_booking(booking_id)
        package = self._get_package(booking.package_id)

        # Refund all payments as negative entries
        refunds: List[BookingPayment] = []
        for p in booking.payment_history:
            refunds.append(BookingPayment(payment_id=p.payment_id, amount=-p.amount))
        booking.payment_history.extend(refunds)

        # Update status
        booking.status = "cancelled"

        # Restore available slots based on traveler count
        try:
            package.departures[booking.departure_date].available_slots += len(booking.travelers)
        except KeyError:
            logger.warning(
                f"Departure date {booking.departure_date} missing in package {package.package_id} during cancel."
            )
        return booking

    @is_tool(ToolType.WRITE)
    def update_booking_add_ons(
        self,
        booking_id: str,
        add_ons: List[BookingAddOn | Dict[str, Any]],
        payment_method_id: str,
    ) -> Booking:
        """
        Replace add-ons for a booking and charge/refund the difference.

        Args:
            booking_id: The booking ID
            add_ons: Entire new list of add-ons
            payment_method_id: Traveler's saved payment method ID to use for delta
        """
        traveler, booking = self._get_booking(booking_id)
        package = self._get_package(booking.package_id)

        # Validate payment method
        if payment_method_id not in traveler.payment_methods:
            raise ValueError(f"Payment method {payment_method_id} not found for traveler")

        # Normalize add_ons
        if all(isinstance(a, dict) for a in add_ons):
            add_ons = [BookingAddOn(**a) for a in add_ons]  # type: ignore

        # Retrieve base price of current departure
        departure = self._validate_departure(
            package, booking.departure_date, required_slots=0
        )
        old_total = booking.total_price
        new_total = self._calc_total_price(
            base_price_per_person=departure.base_price,
            num_travelers=len(booking.travelers),
            add_ons=add_ons,  # type: ignore
            insurance=booking.insurance,
        )
        delta = round(new_total - old_total, 2)

        # Record payment if delta != 0
        if delta != 0:
            booking.payment_history.append(
                BookingPayment(payment_id=payment_method_id, amount=delta)
            )

        # Update booking
        booking.add_ons = deepcopy(add_ons)  # type: ignore
        booking.total_price = new_total
        return booking

    @is_tool(ToolType.WRITE)
    def update_booking_departure_date(
        self, booking_id: str, new_departure_date: str, payment_method_id: str
    ) -> Booking:
        """
        Change departure date for a booking and charge/refund base price differences.
        Updates departure slots accordingly.
        """
        traveler, booking = self._get_booking(booking_id)
        package = self._get_package(booking.package_id)

        if payment_method_id not in traveler.payment_methods:
            raise ValueError(f"Payment method {payment_method_id} not found for traveler")

        num_travelers = len(booking.travelers)

        # Validate availability on new date
        new_dep = self._validate_departure(package, new_departure_date, required_slots=num_travelers)

        # Old departure (may not exist if data mutated, handle defensively)
        old_price_per_person = 0.0
        if booking.departure_date in package.departures:
            old_price_per_person = package.departures[booking.departure_date].base_price
        else:
            logger.warning("Old departure date not found in package during update.")

        # Calculate price delta (base price difference only)
        old_total_price = booking.total_price
        # Recompute new total: new base + existing add-ons + insurance
        new_total_price = self._calc_total_price(
            base_price_per_person=new_dep.base_price,
            num_travelers=num_travelers,
            add_ons=booking.add_ons,
            insurance=booking.insurance,
        )
        delta = round(new_total_price - old_total_price, 2)

        # Update slots: return to old, take from new
        try:
            package.departures[booking.departure_date].available_slots += num_travelers
        except KeyError:
            logger.warning(
                f"Old departure {booking.departure_date} missing while restoring slots."
            )
        package.departures[new_departure_date].available_slots -= num_travelers

        # Record payment if needed
        if delta != 0:
            booking.payment_history.append(
                BookingPayment(payment_id=payment_method_id, amount=delta)
            )

        # Update booking fields
        booking.departure_date = new_departure_date
        booking.total_price = new_total_price
        return booking

    @is_tool(ToolType.WRITE)
    def update_booking_travelers(
        self, booking_id: str, travelers: List[BookingTravelerInfo | Dict[str, Any]]
    ) -> Booking:
        """
        Update traveler details for a booking. The number of travelers must remain unchanged.
        """
        _, booking = self._get_booking(booking_id)

        if all(isinstance(t, dict) for t in travelers):
            travelers = [BookingTravelerInfo(**t) for t in travelers]  # type: ignore

        if len(travelers) != len(booking.travelers):
            raise ValueError("Number of travelers does not match current booking")
        booking.travelers = deepcopy(travelers)  # type: ignore
        return booking

    @is_tool(ToolType.WRITE)
    def update_booking_rooming(
        self, booking_id: str, rooming: RoomingInfo | Dict[str, Any]
    ) -> Booking:
        """
        Update rooming configuration. Price neutrality is assumed unless add-ons/policies say otherwise.
        """
        _, booking = self._get_booking(booking_id)
        if isinstance(rooming, dict):
            rooming = RoomingInfo(**rooming)  # type: ignore
        booking.rooming = deepcopy(rooming)  # type: ignore
        return booking

    @is_tool(ToolType.WRITE)
    def schedule_agent_meeting(
        self, agent_id: str, date: str, time_range: str, traveler_id: str
    ) -> str:
        """
        Schedule a meeting with an agent by reserving an available time slot.
        Removes the slot from availability if found.
        """
        agent = self._get_agent(agent_id)
        self._get_traveler(traveler_id)  # validate traveler exists

        if date not in agent.availability:
            raise ValueError(f"Agent is not available on {date}")
        avail_day = agent.availability[date]
        if time_range not in avail_day.slots:
            raise ValueError(f"Time slot {time_range} not available on {date}")
        # Reserve the slot by removing it
        avail_day.slots.remove(time_range)
        return f"Meeting scheduled with agent {agent_id} on {date} at {time_range}"

    # ---------------------------
    # Generic tools
    # ---------------------------
    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if the user explicitly asks for a human agent
        or the issue cannot be solved with available tools.
        """
        return "Transfer successful"

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Simple calculator for arithmetic expressions with numbers, +, -, *, /, parentheses.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))


if __name__ == "__main__":
    from tau2.domains.travel.utils import TRAVEL_DB_PATH

    travel_tools = TravelAgencyTools(TravelAgencyDB.load(TRAVEL_DB_PATH))
    print(travel_tools.get_statistics())
