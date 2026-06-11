"""Agent tools for the hospitality domain."""

import re
from datetime import date
from typing import Literal, Optional

from tau2.domains.hospitality.data_model import (
    ExtraService,
    Guest,
    HospitalityDB,
    RatePlan,
    Reservation,
    ReservationExtra,
    RoomType,
    StayPackage,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

ModificationType = Literal["change_dates", "change_room_type", "change_num_guests"]

# Words too common to carry signal in knowledge base lookups.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "can",
    "do",
    "does",
    "for",
    "hotel",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "the",
    "there",
    "to",
    "we",
    "what",
    "when",
    "where",
    "you",
    "your",
}


class HospitalityTools(ToolKitBase):
    """Tools for the hotel reservations agent."""

    db: HospitalityDB

    def __init__(self, db: HospitalityDB) -> None:
        super().__init__(db)

    # -- helpers ------------------------------------------------------------

    def _today(self) -> date:
        return date.fromisoformat(self.db.hotel.current_time.split(" ")[0])

    def _parse_date(self, value: str, field_name: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"Invalid {field_name} '{value}'. Expected format YYYY-MM-DD."
            )

    def _validate_stay_dates(self, check_in: str, check_out: str) -> int:
        """Validate a stay date range and return the number of nights."""
        check_in_date = self._parse_date(check_in, "check_in")
        check_out_date = self._parse_date(check_out, "check_out")
        if check_in_date < self._today():
            raise ValueError(f"Check-in date {check_in} is in the past.")
        if check_out_date <= check_in_date:
            raise ValueError("Check-out date must be after check-in date.")
        return (check_out_date - check_in_date).days

    def _get_guest(self, guest_id: str) -> Guest:
        if guest_id not in self.db.guests:
            raise ValueError(f"Guest {guest_id} not found.")
        return self.db.guests[guest_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found.")
        return self.db.reservations[reservation_id]

    def _get_room_type(self, room_type_id: str) -> RoomType:
        if room_type_id not in self.db.room_types:
            raise ValueError(f"Room type {room_type_id} not found.")
        return self.db.room_types[room_type_id]

    def _get_rate_plan(self, rate_plan_id: str) -> RatePlan:
        if rate_plan_id not in self.db.rate_plans:
            raise ValueError(f"Rate plan {rate_plan_id} not found.")
        return self.db.rate_plans[rate_plan_id]

    def _available_count(
        self,
        room_type_id: str,
        check_in: str,
        check_out: str,
        exclude_reservation_id: Optional[str] = None,
    ) -> int:
        """Number of rooms of the given type free for the whole date range."""
        room_type = self._get_room_type(room_type_id)
        check_in_date = date.fromisoformat(check_in)
        check_out_date = date.fromisoformat(check_out)
        occupied = 0
        for reservation in self.db.reservations.values():
            if reservation.reservation_id == exclude_reservation_id:
                continue
            if reservation.room_type_id != room_type_id:
                continue
            if reservation.status not in ("confirmed", "checked_in"):
                continue
            existing_in = date.fromisoformat(reservation.check_in)
            existing_out = date.fromisoformat(reservation.check_out)
            if existing_in < check_out_date and check_in_date < existing_out:
                occupied += 1
        return room_type.count - occupied

    def _recompute_total(self, reservation: Reservation) -> None:
        nights = (
            date.fromisoformat(reservation.check_out)
            - date.fromisoformat(reservation.check_in)
        ).days
        extras_amount = sum(extra.amount for extra in reservation.extras)
        reservation.total_amount = nights * reservation.nightly_rate + extras_amount

    # -- catalog ------------------------------------------------------------

    @is_tool(ToolType.READ)
    def list_room_types(self) -> list[RoomType]:
        """
        List all room types of the hotel with capacities, amenities and nightly
        rates per rate plan.
        """
        return list(self.db.room_types.values())

    @is_tool(ToolType.READ)
    def list_rate_plans(self) -> list[RatePlan]:
        """
        List all rate plans with their cancellation and modification conditions.
        """
        return list(self.db.rate_plans.values())

    @is_tool(ToolType.READ)
    def list_stay_packages(self) -> list[StayPackage]:
        """
        List all stay packages with their inclusions, rates and minimum stay.
        """
        return list(self.db.stay_packages.values())

    @is_tool(ToolType.READ)
    def list_extra_services(self) -> list[ExtraService]:
        """
        List all extra services that can be added to a reservation, with prices.
        """
        return list(self.db.extra_services.values())

    # -- offers -------------------------------------------------------------

    @is_tool(ToolType.READ)
    def get_room_offers(self, check_in: str, check_out: str, num_guests: int) -> list:
        """
        Get room offers for a stay. Only returns room types that are available
        for the whole date range and large enough for the party. Prices are
        total for the stay, in EUR, per rate plan.

        Args:
            check_in: Check-in date, format YYYY-MM-DD.
            check_out: Check-out date, format YYYY-MM-DD.
            num_guests: Number of guests staying in the room.

        Raises:
            ValueError: If the dates are invalid.
        """
        nights = self._validate_stay_dates(check_in, check_out)
        offers = []
        for room_type in self.db.room_types.values():
            if room_type.capacity < num_guests:
                continue
            if self._available_count(room_type.room_type_id, check_in, check_out) < 1:
                continue
            rates = []
            for rate_plan_id, nightly_rate in room_type.rates.items():
                rate_plan = self.db.rate_plans[rate_plan_id]
                rates.append(
                    {
                        "rate_plan_id": rate_plan_id,
                        "rate_plan_name": rate_plan.name,
                        "nightly_rate": nightly_rate,
                        "total": nights * nightly_rate,
                        "refundable": rate_plan.refundable,
                        "breakfast_included": rate_plan.breakfast_included,
                    }
                )
            offers.append(
                {
                    "room_type_id": room_type.room_type_id,
                    "name": room_type.name,
                    "capacity": room_type.capacity,
                    "amenities": room_type.amenities,
                    "nights": nights,
                    "rates": rates,
                }
            )
        return offers

    @is_tool(ToolType.READ)
    def get_stay_package_offers(
        self, check_in: str, check_out: str, num_guests: int
    ) -> list:
        """
        Get stay package offers for a stay. Only returns packages whose room
        type is available for the whole date range, large enough for the party,
        and whose minimum stay is met. Prices are total for the stay, in EUR.

        Args:
            check_in: Check-in date, format YYYY-MM-DD.
            check_out: Check-out date, format YYYY-MM-DD.
            num_guests: Number of guests staying in the room.

        Raises:
            ValueError: If the dates are invalid.
        """
        nights = self._validate_stay_dates(check_in, check_out)
        offers = []
        for package in self.db.stay_packages.values():
            room_type = self.db.room_types[package.room_type_id]
            if room_type.capacity < num_guests:
                continue
            if nights < package.min_nights:
                continue
            if self._available_count(package.room_type_id, check_in, check_out) < 1:
                continue
            offers.append(
                {
                    "package_id": package.package_id,
                    "name": package.name,
                    "room_type_id": package.room_type_id,
                    "room_type_name": room_type.name,
                    "nightly_rate": package.nightly_rate,
                    "total": nights * package.nightly_rate,
                    "min_nights": package.min_nights,
                    "inclusions": package.inclusions,
                }
            )
        return offers

    # -- guests -------------------------------------------------------------

    @is_tool(ToolType.READ)
    def find_guest(self, email: str) -> Guest:
        """
        Find a guest profile by email address.

        Args:
            email: Email address of the guest.

        Raises:
            ValueError: If no guest with this email exists.
        """
        for guest in self.db.guests.values():
            if guest.email.lower() == email.lower():
                return guest
        raise ValueError(f"No guest found with email {email}.")

    @is_tool(ToolType.WRITE)
    def create_guest_profile(self, name: str, email: str, phone: str) -> Guest:
        """
        Create a new guest profile.

        Args:
            name: Full name of the guest.
            email: Email address of the guest.
            phone: Phone number of the guest.

        Raises:
            ValueError: If a guest with this email already exists.
        """
        for guest in self.db.guests.values():
            if guest.email.lower() == email.lower():
                raise ValueError(
                    f"A guest with email {email} already exists: {guest.guest_id}."
                )
        guest_id = f"G{len(self.db.guests) + 1:03d}"
        guest = Guest(guest_id=guest_id, name=name, email=email, phone=phone)
        self.db.guests[guest_id] = guest
        return guest

    @is_tool(ToolType.WRITE)
    def update_guest_profile(
        self,
        guest_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> Guest:
        """
        Update a guest profile. Only the provided fields are changed.

        Args:
            guest_id: ID of the guest to update.
            name: New full name, if changing.
            email: New email address, if changing.
            phone: New phone number, if changing.

        Raises:
            ValueError: If the guest is not found or no field is provided.
        """
        guest = self._get_guest(guest_id)
        if name is None and email is None and phone is None:
            raise ValueError("Provide at least one field to update.")
        if name is not None:
            guest.name = name
        if email is not None:
            guest.email = email
        if phone is not None:
            guest.phone = phone
        return guest

    # -- reservations -------------------------------------------------------

    @is_tool(ToolType.READ)
    def read_reservation_info(self, reservation_id: str) -> Reservation:
        """
        Read full details of a reservation by confirmation number.

        Args:
            reservation_id: Confirmation number, e.g. HV-1001.

        Raises:
            ValueError: If the reservation is not found.
        """
        return self._get_reservation(reservation_id)

    @is_tool(ToolType.WRITE)
    def create_reservation(
        self,
        guest_id: str,
        check_in: str,
        check_out: str,
        num_guests: int,
        room_type_id: Optional[str] = None,
        rate_plan_id: Optional[str] = None,
        package_id: Optional[str] = None,
    ) -> Reservation:
        """
        Create a reservation for an existing guest. Either provide room_type_id
        together with rate_plan_id (regular booking), or package_id (package
        booking), never both.

        Args:
            guest_id: ID of the guest the reservation is for.
            check_in: Check-in date, format YYYY-MM-DD.
            check_out: Check-out date, format YYYY-MM-DD.
            num_guests: Number of guests staying in the room.
            room_type_id: Room type to book. Requires rate_plan_id.
            rate_plan_id: Rate plan to book the room under.
            package_id: Stay package to book instead of a regular room rate.

        Raises:
            ValueError: If the guest, room type, rate plan or package is not
                found, the dates are invalid, the room does not fit the party,
                the room is not available, or the package minimum stay is not
                met.
        """
        self._get_guest(guest_id)
        nights = self._validate_stay_dates(check_in, check_out)
        if num_guests < 1:
            raise ValueError("num_guests must be at least 1.")

        is_package = package_id is not None
        is_regular = room_type_id is not None or rate_plan_id is not None
        if is_package and is_regular:
            raise ValueError(
                "Provide either room_type_id with rate_plan_id, or package_id, "
                "not both."
            )
        if is_package:
            if package_id not in self.db.stay_packages:
                raise ValueError(f"Package {package_id} not found.")
            package = self.db.stay_packages[package_id]
            if nights < package.min_nights:
                raise ValueError(
                    f"Package {package.name} requires at least "
                    f"{package.min_nights} nights."
                )
            room_type = self._get_room_type(package.room_type_id)
            nightly_rate = package.nightly_rate
            rate_plan_id = None
        else:
            if room_type_id is None or rate_plan_id is None:
                raise ValueError(
                    "Provide both room_type_id and rate_plan_id, or a package_id."
                )
            room_type = self._get_room_type(room_type_id)
            rate_plan = self._get_rate_plan(rate_plan_id)
            if rate_plan.rate_plan_id not in room_type.rates:
                raise ValueError(
                    f"Rate plan {rate_plan_id} is not offered for "
                    f"room type {room_type_id}."
                )
            nightly_rate = room_type.rates[rate_plan_id]

        if room_type.capacity < num_guests:
            raise ValueError(
                f"Room type {room_type.name} sleeps at most {room_type.capacity} "
                f"guests."
            )
        if self._available_count(room_type.room_type_id, check_in, check_out) < 1:
            raise ValueError(
                f"No {room_type.name} available from {check_in} to {check_out}."
            )

        reservation_id = f"HV-{1000 + len(self.db.reservations) + 1}"
        reservation = Reservation(
            reservation_id=reservation_id,
            guest_id=guest_id,
            room_type_id=room_type.room_type_id,
            rate_plan_id=rate_plan_id,
            package_id=package_id,
            check_in=check_in,
            check_out=check_out,
            num_guests=num_guests,
            nightly_rate=nightly_rate,
            total_amount=nights * nightly_rate,
            status="confirmed",
        )
        self.db.reservations[reservation_id] = reservation
        return reservation

    @is_tool(ToolType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> Reservation:
        """
        Cancel a confirmed reservation and compute the refund. Refund rules:
        non-refundable rates get no refund; refundable rates and packages get a
        full refund up to 2 days before check-in, after that the first night is
        charged and the rest is refunded.

        Args:
            reservation_id: Confirmation number of the reservation to cancel.

        Raises:
            ValueError: If the reservation is not found or not in confirmed
                status.
        """
        reservation = self._get_reservation(reservation_id)
        if reservation.status != "confirmed":
            raise ValueError(
                f"Reservation {reservation_id} is {reservation.status}, "
                f"only confirmed reservations can be cancelled."
            )
        refundable = True
        if reservation.rate_plan_id is not None:
            refundable = self.db.rate_plans[reservation.rate_plan_id].refundable
        if not refundable:
            refund = 0
        else:
            days_until_check_in = (
                date.fromisoformat(reservation.check_in) - self._today()
            ).days
            if days_until_check_in >= 2:
                refund = reservation.total_amount
            else:
                refund = reservation.total_amount - reservation.nightly_rate
        reservation.status = "cancelled"
        reservation.refund_amount = refund
        return reservation

    @is_tool(ToolType.WRITE)
    def modify_reservation(
        self,
        reservation_id: str,
        modification_type: ModificationType,
        new_check_in: Optional[str] = None,
        new_check_out: Optional[str] = None,
        new_room_type_id: Optional[str] = None,
        new_num_guests: Optional[int] = None,
    ) -> Reservation:
        """
        Modify a confirmed reservation. Supported modification types:
        'change_dates' (requires new_check_in and new_check_out; not allowed on
        non-refundable rates), 'change_room_type' (requires new_room_type_id;
        not allowed on package bookings; the rate plan is kept and the price is
        recomputed), 'change_num_guests' (requires new_num_guests; must fit the
        room capacity).

        Args:
            reservation_id: Confirmation number of the reservation to modify.
            modification_type: One of 'change_dates', 'change_room_type',
                'change_num_guests'.
            new_check_in: New check-in date, format YYYY-MM-DD.
            new_check_out: New check-out date, format YYYY-MM-DD.
            new_room_type_id: New room type ID.
            new_num_guests: New number of guests.

        Raises:
            ValueError: If the reservation is not found or not confirmed, the
                modification is not allowed for the rate plan or package, the
                required arguments are missing, or the new room/dates are not
                available.
        """
        reservation = self._get_reservation(reservation_id)
        if reservation.status != "confirmed":
            raise ValueError(
                f"Reservation {reservation_id} is {reservation.status}, "
                f"only confirmed reservations can be modified."
            )

        if modification_type == "change_dates":
            if new_check_in is None or new_check_out is None:
                raise ValueError(
                    "change_dates requires new_check_in and new_check_out."
                )
            if reservation.rate_plan_id is not None:
                rate_plan = self.db.rate_plans[reservation.rate_plan_id]
                if not rate_plan.date_changes_allowed:
                    raise ValueError(
                        f"Rate plan {rate_plan.name} does not allow date changes."
                    )
            nights = self._validate_stay_dates(new_check_in, new_check_out)
            if reservation.package_id is not None:
                package = self.db.stay_packages[reservation.package_id]
                if nights < package.min_nights:
                    raise ValueError(
                        f"Package {package.name} requires at least "
                        f"{package.min_nights} nights."
                    )
            available = self._available_count(
                reservation.room_type_id,
                new_check_in,
                new_check_out,
                exclude_reservation_id=reservation_id,
            )
            if available < 1:
                raise ValueError(
                    f"No {self.db.room_types[reservation.room_type_id].name} "
                    f"available from {new_check_in} to {new_check_out}."
                )
            reservation.check_in = new_check_in
            reservation.check_out = new_check_out

        elif modification_type == "change_room_type":
            if new_room_type_id is None:
                raise ValueError("change_room_type requires new_room_type_id.")
            if reservation.package_id is not None:
                raise ValueError(
                    "Package bookings are tied to a specific room type and "
                    "cannot change rooms."
                )
            new_room_type = self._get_room_type(new_room_type_id)
            if new_room_type.capacity < reservation.num_guests:
                raise ValueError(
                    f"Room type {new_room_type.name} sleeps at most "
                    f"{new_room_type.capacity} guests."
                )
            if reservation.rate_plan_id not in new_room_type.rates:
                raise ValueError(
                    f"Rate plan {reservation.rate_plan_id} is not offered for "
                    f"room type {new_room_type_id}."
                )
            available = self._available_count(
                new_room_type_id,
                reservation.check_in,
                reservation.check_out,
                exclude_reservation_id=reservation_id,
            )
            if available < 1:
                raise ValueError(
                    f"No {new_room_type.name} available from "
                    f"{reservation.check_in} to {reservation.check_out}."
                )
            reservation.room_type_id = new_room_type_id
            reservation.nightly_rate = new_room_type.rates[reservation.rate_plan_id]

        elif modification_type == "change_num_guests":
            if new_num_guests is None:
                raise ValueError("change_num_guests requires new_num_guests.")
            if new_num_guests < 1:
                raise ValueError("new_num_guests must be at least 1.")
            room_type = self.db.room_types[reservation.room_type_id]
            if room_type.capacity < new_num_guests:
                raise ValueError(
                    f"Room type {room_type.name} sleeps at most "
                    f"{room_type.capacity} guests."
                )
            reservation.num_guests = new_num_guests

        else:
            raise ValueError(
                f"Unknown modification_type '{modification_type}'. Use one of: "
                f"change_dates, change_room_type, change_num_guests."
            )

        self._recompute_total(reservation)
        return reservation

    @is_tool(ToolType.WRITE)
    def book_extra_services(
        self, reservation_id: str, service_id: str, quantity: int
    ) -> Reservation:
        """
        Add an extra service to a confirmed reservation. The line price is
        unit_price * quantity; check the service pricing_unit to compute the
        right quantity (e.g. breakfast is per person per night).

        Args:
            reservation_id: Confirmation number of the reservation.
            service_id: ID of the extra service to add.
            quantity: Number of units to book.

        Raises:
            ValueError: If the reservation is not found or not confirmed, the
                service is not found, or the quantity is invalid.
        """
        reservation = self._get_reservation(reservation_id)
        if reservation.status != "confirmed":
            raise ValueError(
                f"Reservation {reservation_id} is {reservation.status}, "
                f"extras can only be added to confirmed reservations."
            )
        if service_id not in self.db.extra_services:
            raise ValueError(f"Extra service {service_id} not found.")
        if quantity < 1:
            raise ValueError("quantity must be at least 1.")
        service = self.db.extra_services[service_id]
        amount = service.unit_price * quantity
        reservation.extras.append(
            ReservationExtra(service_id=service_id, quantity=quantity, amount=amount)
        )
        reservation.total_amount += amount
        return reservation

    # -- knowledge base -----------------------------------------------------

    @is_tool(ToolType.READ)
    def knowledge_base(self, query: str) -> str:
        """
        Search the hotel knowledge base for policies and practical information
        (breakfast, parking, pets, spa, transfers, payment, location, ...).
        Returns the most relevant articles.

        Args:
            query: Free-text search query.
        """
        terms = [
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if term not in _STOPWORDS
        ]
        if not terms:
            return "No matching articles found."
        scored = []
        for article in self.db.knowledge_base.values():
            title = article.title.lower()
            content = article.content.lower()
            score = sum(
                (3 if term in title else 0) + (1 if term in content else 0)
                for term in terms
            )
            if score > 0:
                scored.append((score, article))
        if not scored:
            return "No matching articles found."
        scored.sort(key=lambda item: item[0], reverse=True)
        sections = [
            f"## {article.title}\n{article.content}" for _, article in scored[:3]
        ]
        return "\n\n".join(sections)

    # -- escalation ---------------------------------------------------------

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the
            user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"
