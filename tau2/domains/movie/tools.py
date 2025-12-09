from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from tau2.domains.movie.data_model import (
    Auditorium,
    Booking,
    BookingPayment,
    BookingPaymentMethod,
    BookingStatus,
    BookedSeat,
    ConcessionItem,
    Customer,
    Delivery,
    DeliveryMethod,
    Movie,
    MovieTheaterDB,
    PaymentSource,
    PriceSchema,
    Show,
    ShowStatus,
    Theater,
    TicketDeliveryItem,
    TicketType,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MovieTheaterTools(ToolKitBase):
    """All tools for the movie theater domain."""

    db: MovieTheaterDB

    def __init__(self, db: MovieTheaterDB) -> None:
        super().__init__(db)

    # --------------------
    # Internal helpers
    # --------------------
    def _get_movie(self, movie_id: str) -> Movie:
        if movie_id not in self.db.movies:
            raise ValueError(f"Movie {movie_id} not found")
        return self.db.movies[movie_id]

    def _get_theater(self, theater_id: str) -> Theater:
        if theater_id not in self.db.theaters:
            raise ValueError(f"Theater {theater_id} not found")
        return self.db.theaters[theater_id]

    def _get_booking(self, booking_id: str) -> Booking:
        if booking_id not in self.db.bookings:
            raise ValueError(f"Booking {booking_id} not found")
        return self.db.bookings[booking_id]

    def _get_show(
        self, theater_id: str, show_id: str, date: Optional[str] = None
    ) -> tuple[Show, str]:
        """
        Find a show by ID in a given theater, optionally constrained by date.
        Returns (show, date)
        """
        theater = self._get_theater(theater_id)
        if date is not None:
            if date not in theater.dates:
                raise ValueError(f"No schedule for date {date} at theater {theater_id}")
            for show in theater.dates[date].shows:
                if show.show_id == show_id:
                    return show, date
            raise ValueError(f"Show {show_id} not found on date {date} at theater {theater_id}")
        # search all dates
        for d, day in theater.dates.items():
            for show in day.shows:
                if show.show_id == show_id:
                    return show, d
        raise ValueError(f"Show {show_id} not found at theater {theater_id}")

    def _get_auditorium(self, theater: Theater, auditorium_id: str) -> Auditorium:
        if auditorium_id not in theater.auditoriums:
            raise ValueError(f"Auditorium {auditorium_id} not found at theater {theater.theater_id}")
        return theater.auditoriums[auditorium_id]

    def _new_booking_id(self) -> str:
        """
        Get a new booking id. Assume at most 3 bookings will be created per task.
        """
        for booking_id in ["BKG001", "BKG002", "BKG003"]:
            if booking_id not in self.db.bookings:
                return booking_id
        raise ValueError("Too many new bookings in this session")

    def _now_str(self) -> str:
        """
        Current timestamp in YY-MM-DD-HH-MM.
        """
        return "24-05-15-15-00"

    def _extract_date_from_show(self, show: Show) -> str:
        # start_time_local format: YY-MM-DD-HH-MM
        return "-".join(show.start_time_local.split("-")[0:3])

    def _list_all_seat_ids(self, auditorium: Auditorium) -> Set[str]:
        seat_ids: Set[str] = set()
        for row_id, seats in auditorium.seat_map.rows.items():
            for seat in seats:
                seat_ids.add(seat.seat_id)
        return seat_ids

    def _booked_seats_for_show(self, theater_id: str, show_id: str) -> Set[str]:
        """
        Seats already booked for this show (only confirmed bookings count).
        """
        seat_ids: Set[str] = set()
        for booking in self.db.bookings.values():
            if booking.theater_id == theater_id and booking.show_id == show_id:
                if booking.status == "confirmed":
                    for s in booking.seats:
                        seat_ids.add(s.seat_id)
        return seat_ids

    def _validate_seats_exist(self, auditorium: Auditorium, seat_ids: Set[str]) -> None:
        all_seats = self._list_all_seat_ids(auditorium)
        missing = seat_ids - all_seats
        if missing:
            raise ValueError(f"Unknown seat ids: {sorted(list(missing))}")

    def _validate_seats_available(
        self, already_booked: Set[str], requested: Set[str]
    ) -> None:
        conflict = requested & already_booked
        if conflict:
            raise ValueError(f"Seats not available: {sorted(list(conflict))}")

    def _per_ticket_price(
        self, price_schema: PriceSchema, ticket_type: TicketType
    ) -> float:
        if ticket_type == "adult":
            return float(price_schema.adult)
        if ticket_type == "child":
            return float(price_schema.child)
        if ticket_type == "senior":
            return float(price_schema.senior)
        raise ValueError(f"Unsupported ticket type {ticket_type}")

    def _build_booked_seats(
        self,
        show: Show,
        theater: Theater,
        seat_specs: List[dict],  # {"seat_id": str, "ticket_type": TicketType}
    ) -> List[BookedSeat]:
        """
        Build BookedSeat list with computed price, convenience_fee, tax.
        """
        tax_rate = float(theater.pricing_rules.tax_rate_percent) / 100.0
        per_ticket_fee = float(show.price_schema.fees.convenience_fee)

        booked: List[BookedSeat] = []
        for spec in seat_specs:
            seat_id = spec["seat_id"]
            tt: TicketType = spec["ticket_type"]
            price = self._per_ticket_price(show.price_schema, tt)
            fee = per_ticket_fee
            tax = (price + fee) * tax_rate
            booked.append(
                BookedSeat(
                    seat_id=seat_id,
                    ticket_type=tt,
                    price=round(price, 2),
                    convenience_fee=round(fee, 2),
                    tax=round(tax, 2),
                )
            )
        return booked

    def _compute_concession_lines(
        self, theater: Theater, concessions: List[dict]
    ) -> List[ConcessionItem]:
        """
        concessions spec: [{item_id, name, size, quantity, price_each}]
        Add tax_each and total per line.
        """
        tax_rate = float(theater.pricing_rules.tax_rate_percent) / 100.0
        lines: List[ConcessionItem] = []
        for c in concessions:
            qty = int(c["quantity"])
            price_each = float(c["price_each"])
            tax_each = price_each * tax_rate
            total = qty * (price_each + tax_each)
            lines.append(
                ConcessionItem(
                    item_id=c["item_id"],
                    name=c["name"],
                    size=c["size"],
                    quantity=qty,
                    price_each=round(price_each, 2),
                    tax_each=round(tax_each, 2),
                    total=round(total, 2),
                )
            )
        return lines

    def _sum_concessions_subtotal_and_tax(self, items: List[ConcessionItem]) -> tuple[float, float]:
        subtotal = 0.0
        tax_total = 0.0
        for i in items:
            subtotal += i.price_each * i.quantity
            tax_total += i.tax_each * i.quantity
        return round(subtotal, 2), round(tax_total, 2)

    def _make_delivery(
        self, method: DeliveryMethod, booking_id: str, count: int
    ) -> Delivery:
        tickets: List[TicketDeliveryItem] = []
        for i in range(count):
            tid = f"{booking_id}-TKT-{i+1:03d}"
            tickets.append(
                TicketDeliveryItem(
                    ticket_id=tid,
                    barcode=f"BAR-{tid}"
                )
            )
        return Delivery(method=method, tickets=tickets)

    # --------------------
    # Tools
    # --------------------
    @is_tool(ToolType.READ)
    def list_movies(self) -> list[Movie]:
        """List all movies."""
        movies = [{"movie_id": t.movie_id, "title": t.title} for t in list(self.db.movies.values())]
        return movies

    @is_tool(ToolType.READ)
    def list_theaters(self) -> list[Theater]:
        """List all theaters."""
        theaters = [{"theater_id": t.theater_id, "name": t.name} for t in list(self.db.theaters.values())]
        return theaters

    @is_tool(ToolType.READ)
    def get_movie_details(self, movie_id: str) -> Movie:
        """Get details for a movie."""
        return self._get_movie(movie_id)

    @is_tool(ToolType.READ)
    def get_theater_details(self, theater_id: str) -> Theater:
        """Get details for a theater, including schedules and auditoriums."""
        return self._get_theater(theater_id)

    @is_tool(ToolType.READ)
    def list_shows(
        self, theater_id: str, date: str, movie_id: Optional[str] = None
    ) -> list[Show]:
        """
        List shows at a given theater for a date, optionally filtered by movie_id.
        """
        theater = self._get_theater(theater_id)
        if date not in theater.dates:
            return []
        shows = theater.dates[date].shows
        if movie_id is None:
            return shows
        return [s for s in shows if s.movie_id == movie_id]

    @is_tool(ToolType.READ)
    def get_show_details(
        self, theater_id: str, show_id: str, date: Optional[str] = None
    ) -> Show:
        """Get show details."""
        show, _ = self._get_show(theater_id, show_id, date)
        return show

    @is_tool(ToolType.READ)
    def get_show_status(self, theater_id: str, show_id: str) -> ShowStatus:
        """Get current status of a show."""
        show, _ = self._get_show(theater_id, show_id, None)
        return show.status

    @is_tool(ToolType.READ)
    def get_seat_availability(
        self, theater_id: str, show_id: str
    ) -> dict[str, list[str]]:
        """
        Get seat availability for a show.
        Returns dict with keys: available, booked.
        """
        theater = self._get_theater(theater_id)
        show, _ = self._get_show(theater_id, show_id, None)
        auditorium = self._get_auditorium(theater, show.auditorium_id)

        all_seats = self._list_all_seat_ids(auditorium)
        booked = self._booked_seats_for_show(theater_id, show_id)
        available = sorted(list(all_seats - booked))
        booked_list = sorted(list(booked))
        return {"available": available, "booked": booked_list}

    @is_tool(ToolType.GENERIC)
    def price_preview(
        self,
        theater_id: str,
        show_id: str,
        seats: List[dict],  # [{"seat_id": str, "ticket_type": TicketType}, ...]
        concessions: Optional[List[dict]] = None,  # [{"item_id","name","size","quantity","price_each"}]
    ) -> dict:
        """
        Preview price breakdown for a potential booking (no booking created).
        """
        concessions = concessions or []
        theater = self._get_theater(theater_id)
        show, _ = self._get_show(theater_id, show_id, None)
        if show.status != "scheduled":
            raise ValueError(f"Show {show_id} is not available for booking (status={show.status})")

        auditorium = self._get_auditorium(theater, show.auditorium_id)
        requested_seat_ids = {s["seat_id"] for s in seats}
        self._validate_seats_exist(auditorium, requested_seat_ids)
        self._validate_seats_available(self._booked_seats_for_show(theater_id, show_id), requested_seat_ids)

        # Build seats and concessions
        booked_seats = self._build_booked_seats(show, theater, seats)
        cons_lines = self._compute_concession_lines(theater, concessions)

        # Totals
        tickets_subtotal = round(sum(s.price for s in booked_seats), 2)
        ticket_fees_total = round(sum(s.convenience_fee for s in booked_seats), 2)
        ticket_tax_total = round(sum(s.tax for s in booked_seats), 2)
        concessions_subtotal, concessions_tax = self._sum_concessions_subtotal_and_tax(cons_lines)

        # Booking-level fee (if any)
        booking_fee = float(theater.pricing_rules.fees.booking_fee)

        fees_total = round(ticket_fees_total + booking_fee, 2)
        tax_total = round(ticket_tax_total + concessions_tax, 2)
        grand_total = round(tickets_subtotal + concessions_subtotal + fees_total + tax_total, 2)

        return {
            "tickets_subtotal": tickets_subtotal,
            "concessions_subtotal": concessions_subtotal,
            "fees_total": fees_total,
            "tax_total": tax_total,
            "grand_total": grand_total,
            "ticket_count": len(booked_seats),
            "booking_fee": round(booking_fee, 2),
        }

    @is_tool(ToolType.WRITE)
    def create_booking(
        self,
        theater_id: str,
        show_id: str,
        customer: Customer | dict,
        seats: List[dict],  # [{"seat_id": str, "ticket_type": TicketType}]
        delivery_method: DeliveryMethod,
        payments: List[BookingPayment | dict],
        concessions: Optional[List[dict]] = None,
        special_requests: Optional[str] = None,
    ) -> Booking:
        """
        Create a booking for a show with selected seats and optional concessions.

        Args:
            theater_id: Theater ID.
            show_id: Show ID.
            customer: Customer info (object or dict).
            seats: List of seat spec dicts: {"seat_id": str, "ticket_type": TicketType}
            delivery_method: Delivery method for tickets.
            payments: List of BookingPayment (object or dict). Amounts must sum to grand total.
            concessions: Optional list of concession item dicts: {"item_id","name","size","quantity","price_each"}
            special_requests: Optional string.

        Returns:
            Booking object created.

        Raises:
            ValueError: For not found entities, seat conflicts, invalid payment totals, or show not bookable.
        """
        if isinstance(customer, dict):
            customer = Customer(**customer)
        concessions = concessions or []
        if all(isinstance(p, dict) for p in payments):
            payments = [BookingPayment(**p) for p in payments]

        theater = self._get_theater(theater_id)
        show, date = self._get_show(theater_id, show_id, None)
        if show.status != "scheduled":
            raise ValueError(f"Show {show_id} is not available for booking (status={show.status})")

        auditorium = self._get_auditorium(theater, show.auditorium_id)

        requested_seat_ids = {s["seat_id"] for s in seats}
        self._validate_seats_exist(auditorium, requested_seat_ids)
        self._validate_seats_available(self._booked_seats_for_show(theater_id, show_id), requested_seat_ids)

        # Build seats and concessions
        booked_seats = self._build_booked_seats(show, theater, seats)
        cons_lines = self._compute_concession_lines(theater, concessions)

        # Totals
        tickets_subtotal = round(sum(s.price for s in booked_seats), 2)
        ticket_fees_total = round(sum(s.convenience_fee for s in booked_seats), 2)
        ticket_tax_total = round(sum(s.tax for s in booked_seats), 2)
        concessions_subtotal, concessions_tax = self._sum_concessions_subtotal_and_tax(cons_lines)

        booking_fee = float(theater.pricing_rules.fees.booking_fee)
        fees_total = round(ticket_fees_total + booking_fee, 2)
        tax_total = round(ticket_tax_total + concessions_tax, 2)
        grand_total = round(tickets_subtotal + concessions_subtotal + fees_total + tax_total, 2)

        # Validate payments sum
        total_paid = round(sum(float(p.amount) for p in payments), 2)
        if total_paid != grand_total:
            raise ValueError(f"Payment amount does not add up, total price is {grand_total}, but paid {total_paid}")

        booking_id = self._new_booking_id()
        delivery = self._make_delivery(delivery_method, booking_id, len(booked_seats))

        totals = {
            "tickets_subtotal": tickets_subtotal,
            "concessions_subtotal": concessions_subtotal,
            "fees_total": fees_total,
            "tax_total": tax_total,
            "grand_total": grand_total,
            "amount_paid": total_paid,
            "amount_due": round(grand_total - total_paid, 2),
        }

        from tau2.domains.movie.data_model import BookingTotals  # to build model

        booking = Booking(
            booking_id=booking_id,
            theater_id=theater_id,
            movie_id=show.movie_id,
            show_id=show_id,
            date=date,
            start_time_local=show.start_time_local,
            status="confirmed",
            created_at=self._now_str(),
            canceled_at=None,
            customer=deepcopy(customer),
            seats=booked_seats,
            concessions=cons_lines,
            promotions_applied=[],
            payment_history=deepcopy(payments),
            totals=BookingTotals(**totals),
            delivery=delivery,
            special_requests=special_requests,
        )

        # Save
        self.db.bookings[booking_id] = booking
        return booking

    @is_tool(ToolType.WRITE)
    def cancel_booking(self, booking_id: str) -> Booking:
        """
        Cancel a booking and add a refund payment for the full amount paid.
        """
        booking = self._get_booking(booking_id)
        if booking.status in ("canceled", "refunded"):
            return booking

        # Compute refund as negative of amount_paid
        refund_amount = -float(booking.totals.amount_paid)
        if refund_amount != 0.0:
            refund_payment = BookingPayment(
                payment_id=f"{booking_id}-refund",
                amount=round(refund_amount, 2),
                method=BookingPaymentMethod(
                    source="cash",  # Method placeholder; real systems track original source
                    payment_method_id="refund",
                    extra_info=None,
                ),
                created_at=self._now_str(),
            )
            booking.payment_history.append(refund_payment)
            # Update totals
            booking.totals.amount_paid = round(booking.totals.amount_paid + refund_amount, 2)
            booking.totals.amount_due = round(booking.totals.grand_total - booking.totals.amount_paid, 2)

        booking.status = "canceled"
        booking.canceled_at = self._now_str()
        logger.warning("Seat release is implicit via availability calculation (ignores canceled bookings).")
        return booking

    @is_tool(ToolType.WRITE)
    def update_booking_seats(
        self,
        booking_id: str,
        seats: List[dict],  # [{"seat_id": str, "ticket_type": TicketType}]
        payment: Optional[BookingPayment | dict] = None,
    ) -> Booking:
        """
        Update seats for an existing booking. Recalculates totals and processes payment/refund difference.

        Args:
            booking_id: ID of the booking.
            seats: New list of seat specs: {"seat_id": str, "ticket_type": TicketType}
            payment: Optional additional payment to cover increase (if any).

        Returns:
            Updated booking.

        Raises:
            ValueError: If seats invalid, unavailable, or payment mismatch.
        """
        if isinstance(payment, dict):
            payment = BookingPayment(**payment)

        booking = self._get_booking(booking_id)
        theater = self._get_theater(booking.theater_id)
        show, _ = self._get_show(booking.theater_id, booking.show_id, None)
        if booking.status != "confirmed":
            raise ValueError(f"Only confirmed bookings can be updated (status={booking.status})")
        if show.status != "scheduled":
            raise ValueError(f"Show {booking.show_id} is not bookable (status={show.status})")

        auditorium = self._get_auditorium(theater, show.auditorium_id)

        new_seat_ids = {s["seat_id"] for s in seats}
        self._validate_seats_exist(auditorium, new_seat_ids)

        # Seats currently held by this booking should be considered "free" for the update.
        globally_booked = self._booked_seats_for_show(booking.theater_id, booking.show_id)
        current_seat_ids = {s.seat_id for s in booking.seats}
        unavailable = (new_seat_ids - current_seat_ids) & globally_booked
        if unavailable:
            raise ValueError(f"Seats not available: {sorted(list(unavailable))}")

        # Build new seats and recompute totals
        new_booked_seats = self._build_booked_seats(show, theater, seats)
        new_tickets_subtotal = round(sum(s.price for s in new_booked_seats), 2)
        new_ticket_fees_total = round(sum(s.convenience_fee for s in new_booked_seats), 2)
        new_ticket_tax_total = round(sum(s.tax for s in new_booked_seats), 2)

        # Concessions unchanged
        cons_subtotal, cons_tax = self._sum_concessions_subtotal_and_tax(booking.concessions)
        booking_fee = float(theater.pricing_rules.fees.booking_fee)
        new_fees_total = round(new_ticket_fees_total + booking_fee, 2)
        new_tax_total = round(new_ticket_tax_total + cons_tax, 2)
        new_grand_total = round(new_tickets_subtotal + cons_subtotal + new_fees_total + new_tax_total, 2)

        old_paid = float(booking.totals.amount_paid)
        old_grand = float(booking.totals.grand_total)

        diff = round(new_grand_total - old_grand, 2)
        if diff > 0:
            # Need additional payment
            if payment is None or round(float(payment.amount), 2) != diff:
                raise ValueError(f"Additional payment of {diff} is required to complete the update")
            booking.payment_history.append(payment)
            booking.totals.amount_paid = round(booking.totals.amount_paid + diff, 2)
        elif diff < 0:
            # Refund
            refund_payment = BookingPayment(
                payment_id=f"{booking_id}-adj-refund",
                amount=diff,  # negative
                method=BookingPaymentMethod(
                    source="cash",
                    payment_method_id="refund",
                    extra_info=None,
                ),
                created_at=self._now_str(),
            )
            booking.payment_history.append(refund_payment)
            booking.totals.amount_paid = round(booking.totals.amount_paid + diff, 2)

        # Update booking fields
        booking.seats = new_booked_seats
        booking.totals.tickets_subtotal = new_tickets_subtotal
        booking.totals.fees_total = new_fees_total
        booking.totals.tax_total = new_tax_total
        booking.totals.grand_total = new_grand_total
        booking.totals.amount_due = round(booking.totals.grand_total - booking.totals.amount_paid, 2)

        return booking

    @is_tool(ToolType.READ)
    def get_booking_details(self, booking_id: str) -> Booking:
        """Get booking details."""
        return self._get_booking(booking_id)

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.
        Allowed characters: digits, +, -, *, /, parentheses, dot, and spaces.
        """
        if not all(c in "0123456789+-*/(). " for c in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a concise summary.
        Only transfer if requested by user or issue cannot be solved by available tools/policies.
        """
        return "Transfer successful"


if __name__ == "__main__":
    # Example usage (requires actual DB path and data)
    from tau2.domains.movie.utils import MOVIE_DB_PATH

    tools = MovieTheaterTools(MovieTheaterDB.load(MOVIE_DB_PATH))
    print(tools.get_statistics())
