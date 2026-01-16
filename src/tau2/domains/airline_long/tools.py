"""Toolkit for the airline reservation system with noisy output."""

import hashlib
import random
import string
import uuid
from copy import deepcopy
from typing import Any, List, Optional

from loguru import logger

from tau2.domains.airline_long.data_model import (
    AirportCode,
    CabinClass,
    Certificate,
    DirectFlight,
    Flight,
    FlightDateStatus,
    FlightDateStatusAvailable,
    FlightDB,
    FlightInfo,
    FlightType,
    Insurance,
    Passenger,
    Payment,
    Reservation,
    ReservationFlight,
    User,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# TODO: Add an abstract base class for the tools


# ============================================================================
# NOISE GENERATION UTILITIES
# These functions add realistic noise to tool outputs, simulating the kind of
# artifacts found in crawled web pages or raw trace JSON files.
# All functions take a Random instance (rng) for deterministic output.
# ============================================================================

def _generate_trace_id(rng: random.Random) -> str:
    """Generate a random trace ID like those found in distributed systems."""
    return f"trace-{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}-{rng.randint(1000000, 9999999)}"


def _generate_request_metadata(rng: random.Random) -> str:
    """Generate fake request metadata noise."""
    trace_id = _generate_trace_id(rng)
    span_id = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
    timestamp = f"2024-05-15T{rng.randint(10,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}.{rng.randint(100,999)}Z"

    return f"""
<!-- BEGIN_REQUEST_METADATA -->
<!-- x-request-id: {uuid.UUID(int=rng.getrandbits(128))} -->
<!-- x-trace-id: {trace_id} -->
<!-- x-span-id: {span_id} -->
<!-- x-correlation-id: corr_{hashlib.md5(trace_id.encode()).hexdigest()[:12]} -->
<!-- x-timestamp: {timestamp} -->
<!-- x-server-region: us-west-2a -->
<!-- x-cache-status: MISS -->
<!-- x-response-time-ms: {rng.randint(50, 500)} -->
<!-- END_REQUEST_METADATA -->
"""


def _generate_html_noise(rng: random.Random) -> str:
    """Generate HTML-like artifacts commonly found in scraped web content."""
    classes = [''.join(rng.choices(string.ascii_lowercase, k=rng.randint(5,12))) for _ in range(5)]
    attrs = ['data-' + ''.join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(3)]

    noise_parts = [
        f'<div class="{classes[0]} {classes[1]}" {attrs[0]}="{rng.randint(1000,9999)}" {attrs[1]}="true">',
        f'<!-- rendered at {rng.randint(1000000000, 9999999999)} -->',
        f'<span class="sr-only visually-hidden">&nbsp;</span>',
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<!-- gtm.start: {rng.randint(1000000000000, 9999999999999)} -->',
        f'<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-{"".join(rng.choices(string.ascii_uppercase + string.digits, k=7))}" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>',
        f'<!-- build hash: {hashlib.sha256(str(rng.random()).encode()).hexdigest()[:12]} -->',
        f'<link rel="preconnect" href="https://fonts.googleapis.com">',
        f'<link rel="preconnect" href="https://cdn.example-airline.com" crossorigin>',
        f'<!-- cache-key: ck_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]} -->',
        f'<script type="application/ld+json">{{"@context":"https://schema.org","@type":"WebPage","breadcrumb":{{"@type":"BreadcrumbList","itemListElement":[]}}}}</script>',
        '</div>',
    ]
    return '\n'.join(noise_parts)


def _generate_json_trace_noise(rng: random.Random) -> str:
    """Generate JSON-like trace artifacts found in raw API logs."""
    return f'''
{{
  "_meta": {{
    "version": "2.1.{rng.randint(0,99)}",
    "schema_version": "v{rng.randint(1,5)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
    "api_version": "2024-05-01",
    "deprecated_fields": ["legacy_id", "old_status_code", "v1_reference"],
    "warnings": [
      "Field 'internal_ref' will be removed in version 3.0",
      "Consider using 'new_booking_flow' parameter for improved performance"
    ]
  }},
  "_debug": {{
    "query_time_ms": {rng.randint(10, 200)},
    "db_queries": {rng.randint(3, 15)},
    "cache_hits": {rng.randint(0, 10)},
    "cache_misses": {rng.randint(1, 5)},
    "serialization_time_ms": {rng.randint(1, 20)},
    "internal_routing": "svc-airline-api-{rng.choice(['primary', 'secondary', 'fallback'])}-{rng.randint(1,99):02d}",
    "datacenter": "{rng.choice(['us-west-2', 'us-east-1', 'eu-west-1'])}",
    "pod_id": "airline-api-deployment-{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}"
  }},
  "_links": {{
    "self": "/api/v2/resource/{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
    "related": "/api/v2/related/{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
    "documentation": "https://api.example-airline.com/docs/v2/endpoints"
  }},
  "_embedded": {{
    "audit_log_reference": "audit_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "compliance_check_id": "ccheck_{rng.randint(100000, 999999)}"
  }},
'''


def _generate_json_trace_noise_end(rng: random.Random) -> str:
    """Generate closing JSON trace artifacts."""
    return f'''
  "_pagination": {{
    "cursor": "{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:32]}",
    "has_more": false,
    "total_estimated": null,
    "page_info": {{
      "start_cursor": "c_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}",
      "end_cursor": "c_{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}",
      "has_previous_page": false,
      "has_next_page": false
    }}
  }},
  "_rate_limit": {{
    "limit": 1000,
    "remaining": {rng.randint(800, 999)},
    "reset_at": "2024-05-15T{rng.randint(15,23):02d}:00:00Z",
    "retry_after": null
  }},
  "_telemetry": {{
    "trace_id": "{_generate_trace_id(rng)}",
    "span_id": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "parent_span_id": "{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "sampled": true,
    "baggage": {{
      "user_segment": "{rng.choice(['premium', 'standard', 'basic'])}",
      "experiment_bucket": "{rng.choice(['control', 'treatment_a', 'treatment_b'])}"
    }}
  }}
}}
'''


def _generate_legacy_field_noise(rng: random.Random) -> str:
    """Generate deprecated/legacy field noise commonly found in old APIs."""
    return f'''
  "_legacy_fields": {{
    "old_booking_reference": "LEGACY-{uuid.UUID(int=rng.getrandbits(128)).hex[:8].upper()}",
    "deprecated_status": "{rng.choice(['ACTIVE', 'PENDING', 'CONFIRMED'])}",
    "v1_customer_id": "v1_cust_{rng.randint(10000000, 99999999)}",
    "migration_status": "completed",
    "legacy_system_reference": "LS-{rng.randint(1000000, 9999999)}",
    "old_fare_class": "{rng.choice(['Y', 'B', 'M', 'H', 'K', 'L', 'V'])}",
    "internal_notes": "[SYSTEM] Record migrated from legacy platform on 2023-06-15. Original reference: REF-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}. Please use new API fields.",
    "compatibility_mode": true,
    "schema_migration_id": "mig_{rng.randint(1000, 9999)}"
  }},
'''


def _generate_verbose_timestamps(rng: random.Random) -> str:
    """Generate multiple redundant timestamp formats."""
    base_hour = rng.randint(10, 20)
    base_min = rng.randint(0, 59)
    base_sec = rng.randint(0, 59)

    return f'''
  "_timestamps": {{
    "created_at_utc": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}Z",
    "created_at_unix": {1715781600 + base_hour*3600 + base_min*60 + base_sec},
    "created_at_unix_ms": {1715781600000 + base_hour*3600000 + base_min*60000 + base_sec*1000 + rng.randint(0, 999)},
    "created_at_iso8601": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}.{rng.randint(100,999)}+00:00",
    "created_at_rfc2822": "Wed, 15 May 2024 {base_hour:02d}:{base_min:02d}:{base_sec:02d} +0000",
    "created_at_human_readable": "May 15, 2024 at {base_hour:02d}:{base_min:02d} UTC",
    "last_modified_at_utc": "2024-05-15T{base_hour:02d}:{base_min+1 if base_min < 59 else 0:02d}:{base_sec:02d}Z",
    "server_timestamp": "2024-05-15T{base_hour:02d}:{base_min:02d}:{base_sec:02d}.{rng.randint(100,999)}Z",
    "client_timestamp": null,
    "timezone_offset": "+00:00",
    "dst_active": false
  }},
'''


def _generate_internal_ids(rng: random.Random) -> str:
    """Generate multiple internal tracking IDs."""
    return f'''
  "_internal_ids": {{
    "record_id": "{uuid.UUID(int=rng.getrandbits(128))}",
    "partition_key": "pk_{hashlib.md5(str(rng.random()).encode()).hexdigest()[:16]}",
    "sort_key": "sk_{rng.randint(1000000000, 9999999999)}",
    "shard_id": "shard-{rng.randint(0, 15):02d}",
    "sequence_number": "{rng.randint(10000000000000000, 99999999999999999)}",
    "checksum": "{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:16]}",
    "etag": "\\"{hashlib.md5(str(rng.random()).encode()).hexdigest()}\\"",
    "revision": {rng.randint(1, 50)},
    "cluster_id": "cluster-{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}-{rng.randint(1, 10):02d}"
  }},
'''


def _add_noise_to_result(result: Any, operation_name: str, seed_key: str) -> str:
    """Wrap a tool result with realistic noise from web scraping or raw traces.

    Args:
        result: The actual tool result data
        operation_name: Name of the operation for metadata
        seed_key: A string derived from tool call ARGUMENTS (not results) to ensure
                  deterministic noise across simulation and evaluation replay
    """
    import json

    # Convert pydantic models or other objects to dict
    if hasattr(result, 'model_dump'):
        result_data = result.model_dump()
    elif isinstance(result, list):
        result_data = [item.model_dump() if hasattr(item, 'model_dump') else item for item in result]
    else:
        result_data = result

    # Create a local Random instance seeded based on ARGUMENTS (seed_key), not results
    # This ensures deterministic noise for same tool calls (critical for evaluation replay)
    seed_str = operation_name + ":" + seed_key
    seed_value = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_value)

    # Build the noisy output
    noise_output = []

    # Add HTML noise at the beginning
    noise_output.append(_generate_html_noise(rng))

    # Add request metadata
    noise_output.append(_generate_request_metadata(rng))

    # Start JSON trace wrapper
    noise_output.append(_generate_json_trace_noise(rng))

    # Add legacy fields
    noise_output.append(_generate_legacy_field_noise(rng))

    # Add verbose timestamps
    noise_output.append(_generate_verbose_timestamps(rng))

    # Add internal IDs
    noise_output.append(_generate_internal_ids(rng))

    # Add operation-specific metadata
    noise_output.append(f'''
  "_operation": {{
    "name": "{operation_name}",
    "type": "{rng.choice(['query', 'mutation', 'read', 'write'])}",
    "idempotency_key": "idem_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}",
    "retry_count": 0,
    "max_retries": 3,
    "timeout_ms": {rng.randint(5000, 30000)},
    "circuit_breaker_status": "closed"
  }},
''')

    # Add the actual data
    noise_output.append(f'''
  "data": {json.dumps(result_data, indent=4, default=str)},
''')

    # Add status and response metadata
    noise_output.append(f'''
  "_response": {{
    "status": "success",
    "status_code": 200,
    "message": "Operation completed successfully",
    "error": null,
    "error_code": null,
    "error_details": null,
    "warnings": [],
    "info": [
      "Response generated by airline-api-v2.{rng.randint(10,99)}.{rng.randint(0,999)}",
      "Processed by handler: {operation_name}Handler",
      "Backend latency within acceptable range"
    ]
  }},
''')

    # Close JSON trace
    noise_output.append(_generate_json_trace_noise_end(rng))

    # Add more HTML noise at the end
    noise_output.append(f'''
<!-- END_API_RESPONSE -->
<div class="clearfix"></div>
<!-- analytics: page_view_id={uuid.UUID(int=rng.getrandbits(128)).hex[:16]} session_id={uuid.UUID(int=rng.getrandbits(128)).hex[:24]} -->
<!-- performance: dns={rng.randint(1,50)}ms tcp={rng.randint(10,100)}ms ttfb={rng.randint(50,300)}ms -->
<script>window.__INITIAL_STATE__=window.__INITIAL_STATE__||{{}};window.__INITIAL_STATE__.apiResponse=true;</script>
<!-- Served by: web-{rng.randint(1,50):03d}.{rng.choice(['us-west-2', 'us-east-1'])}.example-airline.com -->
''')

    return '\n'.join(noise_output)


class AirlineTools(ToolKitBase):  # Tools
    """All the tools for the airline domain."""

    db: FlightDB

    def __init__(self, db: FlightDB) -> None:
        super().__init__(db)

    def _get_user(self, user_id: str) -> User:
        """Get user from database."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        """Get reservation from database."""
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_flight(self, flight_number: str) -> Flight:
        """Get flight from database."""
        if flight_number not in self.db.flights:
            raise ValueError(f"Flight {flight_number} not found")
        return self.db.flights[flight_number]

    def _get_flight_instance(self, flight_number: str, date: str) -> FlightDateStatus:
        """Get flight instance from database."""
        flight = self._get_flight(flight_number)
        if date not in flight.dates:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight.dates[date]

    def _get_flights_from_flight_infos(
        self, flight_infos: List[FlightInfo]
    ) -> list[FlightDateStatus]:
        """Get the flight from the reservation."""
        flights = []
        for flight_info in flight_infos:
            flights.append(
                self._get_flight_instance(flight_info.flight_number, flight_info.date)
            )
        return flights

    def _get_new_reservation_id(self) -> str:
        """Get a new reservation id.
        Assume each task makes at most 3 reservations

        Returns:
            A new reservation id.

        Raises:
            ValueError: If too many reservations are made.
        """
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
            if reservation_id not in self.db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_payment_id(self) -> str:
        """Get a new payment id.
        Assume each task makes at most 3 payments

        Returns:
            A new payment id.
        """
        return [3221322, 3221323, 3221324]

    def _get_datetime(self) -> str:
        """Get the current datetime."""
        return "2024-05-15T15:00:00"

    def _search_direct_flight(
        self,
        date: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        leave_after: Optional[str] = None,
    ) -> list[DirectFlight]:
        """Search for direct flights

        Args:
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            leave_after: The time to leave after the flight, such as '15:00:00'.
        """
        results = []
        for flight in self.db.flights.values():
            check = (
                (origin is None or flight.origin == origin)
                and (destination is None or flight.destination == destination)
                and (date in flight.dates)
                and (flight.dates[date].status == "available")
                and (
                    leave_after is None
                    or flight.scheduled_departure_time_est >= leave_after
                )
            )
            if check:
                direct_flight = DirectFlight(
                    flight_number=flight.flight_number,
                    origin=flight.origin,
                    destination=flight.destination,
                    status="available",
                    scheduled_departure_time_est=flight.scheduled_departure_time_est,
                    scheduled_arrival_time_est=flight.scheduled_arrival_time_est,
                    available_seats=flight.dates[date].available_seats,
                    prices=flight.dates[date].prices,
                )
                results.append(direct_flight)
        return results

    def _payment_for_update(
        self, user: User, payment_id: str, total_price: int
    ) -> Optional[Payment]:
        """
        Process payment for update reservation

        Args:
            user: The user to process payment for.
            payment_id: The payment id to process.
            total_price: The total price to process.
            reservation: The reservation to process payment for.

        Raises:
            ValueError: If the payment method is not found.
            ValueError: If the certificate is used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        # Check payment
        if payment_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        payment_method = user.payment_methods[payment_id]
        if payment_method.source == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        elif (
            payment_method.source == "gift_card" and payment_method.amount < total_price
        ):
            raise ValueError("Gift card balance is not enough")

        # Deduct payment
        if payment_method.source == "gift_card":
            payment_method.amount -= total_price

        payment = None
        # Create payment if total price is not 0
        if total_price != 0:
            payment = Payment(
                payment_id=payment_id,
                amount=total_price,
            )
        return payment

    @is_tool(ToolType.WRITE)
    def book_reservation(
        self,
        user_id: str,
        origin: str,
        destination: str,
        flight_type: FlightType,
        cabin: CabinClass,
        flights: List[FlightInfo | dict],
        passengers: List[Passenger | dict],
        payment_methods: List[Payment | dict],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: Insurance,
    ) -> str:
        """
        Book a reservation.

        Args:
            user_id: The ID of the user to book the reservation such as 'sara_doe_496'`.
            origin: The IATA code for the origin city such as 'SFO'.
            destination: The IATA code for the destination city such as 'JFK'.
            flight_type: The type of flight such as 'one_way' or 'round_trip'.
            cabin: The cabin class such as 'basic_economy', 'economy', or 'business'.
            flights: An array of objects containing details about each piece of flight.
            passengers: An array of objects containing details about each passenger.
            payment_methods: An array of objects containing details about each payment method.
            total_baggages: The total number of baggage items to book the reservation.
            nonfree_baggages: The number of non-free baggage items to book the reservation.
            insurance: Whether the reservation has insurance.
        """
        if all(isinstance(flight, dict) for flight in flights):
            flights = [FlightInfo(**flight) for flight in flights]
        if all(isinstance(passenger, dict) for passenger in passengers):
            passengers = [Passenger(**passenger) for passenger in passengers]
        if all(isinstance(payment_method, dict) for payment_method in payment_methods):
            payment_methods = [
                Payment(**payment_method) for payment_method in payment_methods
            ]
        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()

        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            cabin=cabin,
            flights=[],
            passengers=deepcopy(passengers),
            payment_history=deepcopy(payment_methods),
            created_at=self._get_datetime(),
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            insurance=insurance,
        )

        # Update flights and calculate price
        total_price = 0
        all_flights_date_data: list[FlightDateStatusAvailable] = []

        for flight_info in flights:
            flight_number = flight_info.flight_number
            flight = self._get_flight(flight_number)
            flight_date_data = self._get_flight_instance(
                flight_number=flight_number, date=flight_info.date
            )
            # Checking flight availability
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(
                    f"Flight {flight_number} not available on date {flight_info.date}"
                )
            # Checking seat availability
            if flight_date_data.available_seats[cabin] < len(passengers):
                raise ValueError(f"Not enough seats on flight {flight_number}")
            # Calculate price
            price = flight_date_data.prices[cabin]
            # Update reservation
            reservation.flights.append(
                ReservationFlight(
                    origin=flight.origin,
                    destination=flight.destination,
                    flight_number=flight_number,
                    date=flight_info.date,
                    price=price,
                )
            )
            all_flights_date_data.append(flight_date_data)
            total_price += price * len(passengers)

        # Add insurance fee
        if insurance == "yes":
            total_price += 30 * len(passengers)

        # Add baggage fee
        total_price += 50 * nonfree_baggages

        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            if payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {payment_id} not found")

            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source in {"gift_card", "certificate"}:
                if user_payment_method.amount < amount:
                    raise ValueError(
                        f"Not enough balance in payment method {payment_id}"
                    )

        total_payment = sum(payment.amount for payment in payment_methods)
        if total_payment != total_price:
            raise ValueError(
                f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}"
            )

        # if checks pass, deduct payment
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source == "gift_card":
                user_payment_method.amount -= amount
            elif user_payment_method.source == "certificate":
                user.payment_methods.pop(payment_id)

        # Update DB
        for flight_date_data in all_flights_date_data:
            flight_date_data.available_seats[cabin] -= len(passengers)
        self.db.reservations[reservation_id] = reservation
        self.db.users[user_id].reservations.append(reservation_id)
        return _add_noise_to_result(reservation, "book_reservation", f"{user_id}:{origin}:{destination}")

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        calc_result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        result = {"expression": expression, "result": calc_result}
        return _add_noise_to_result(result, "calculate", expression)

    @is_tool(ToolType.WRITE)
    def cancel_reservation(self, reservation_id: str) -> str:
        """
        Cancel the whole reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
        """
        reservation = self._get_reservation(reservation_id)
        logger.debug(reservation.model_dump_json(indent=4))
        # reverse the payment
        refunds = []
        for payment in reservation.payment_history:
            refunds.append(
                Payment(
                    payment_id=payment.payment_id,
                    amount=-payment.amount,
                )
            )
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        logger.debug(self._get_reservation(reservation_id).model_dump_json(indent=4))
        # Release seats
        logger.warning("Seats release not implemented for cancellation!!!")
        return _add_noise_to_result(reservation, "cancel_reservation", reservation_id)

    @is_tool(ToolType.READ)
    def get_reservation_details(self, reservation_id: str) -> str:
        """
        Get the details of a reservation.

        Args:
            reservation_id: The reservation ID, such as '8JX2WO'.

        Returns:
            The reservation details.

        Raises:
            ValueError: If the reservation is not found.
        """
        result = self._get_reservation(reservation_id)
        return _add_noise_to_result(result, "get_reservation_details", reservation_id)

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> str:
        """
        Get the details of a user, including their reservations.

        Args:
            user_id: The user ID, such as 'sara_doe_496'.

        Returns:
            The user details.

        Raises:
            ValueError: If the user is not found.
        """
        result = self._get_user(user_id)
        return _add_noise_to_result(result, "get_user_details", user_id)

    @is_tool(ToolType.READ)
    def list_all_airports(self) -> str:
        """Returns a list of all available airports.

        Returns:
            A dictionary mapping IATA codes to AirportInfo objects.
        """
        result = [
            AirportCode(iata="SFO", city="San Francisco"),
            AirportCode(iata="JFK", city="New York"),
            AirportCode(iata="LAX", city="Los Angeles"),
            AirportCode(iata="ORD", city="Chicago"),
            AirportCode(iata="DFW", city="Dallas"),
            AirportCode(iata="DEN", city="Denver"),
            AirportCode(iata="SEA", city="Seattle"),
            AirportCode(iata="ATL", city="Atlanta"),
            AirportCode(iata="MIA", city="Miami"),
            AirportCode(iata="BOS", city="Boston"),
            AirportCode(iata="PHX", city="Phoenix"),
            AirportCode(iata="IAH", city="Houston"),
            AirportCode(iata="LAS", city="Las Vegas"),
            AirportCode(iata="MCO", city="Orlando"),
            AirportCode(iata="EWR", city="Newark"),
            AirportCode(iata="CLT", city="Charlotte"),
            AirportCode(iata="MSP", city="Minneapolis"),
            AirportCode(iata="DTW", city="Detroit"),
            AirportCode(iata="PHL", city="Philadelphia"),
            AirportCode(iata="LGA", city="LaGuardia"),
        ]
        return _add_noise_to_result(result, "list_all_airports", "all")

    @is_tool(ToolType.READ)
    def search_direct_flight(
        self, origin: str, destination: str, date: str
    ) -> str:
        """
        Search direct flights between two cities on a specific date. It provides information about departure and arrival times, flight number, and price per cabin.

        Args:
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.

        Returns:
            The direct flights between the two cities on the specific date.
        """
        result = self._search_direct_flight(
            date=date, origin=origin, destination=destination
        )
        return _add_noise_to_result(result, "search_direct_flight", f"{origin}:{destination}:{date}")

    @is_tool(ToolType.READ)
    def search_onestop_flight(
        self, origin: str, destination: str, date: str
    ) -> str:
        """
        Search one-stop flights between two cities on a specific date. It provides information about departure and arrival times, flight number, and price per cabin.

        Args:
            origin: The origin city airport in three letters, such as 'JFK'.
            destination: The destination city airport in three letters, such as 'LAX'.
            date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.

        Returns:
            A list of pairs of DirectFlight objects.
        """
        results = []
        for result1 in self._search_direct_flight(
            date=date, origin=origin, destination=None
        ):
            result1.date = date
            date2 = (
                f"2024-05-{int(date[-2:]) + 1}"
                if "+1" in result1.scheduled_arrival_time_est
                else date
            )
            # TODO: flight1.scheduled_arrival_time_est could have a +1?
            for result2 in self._search_direct_flight(
                date=date2,
                origin=result1.destination,
                destination=destination,
                leave_after=result1.scheduled_arrival_time_est,
            ):
                result2.date = date2
                results.append([result1, result2])
        return _add_noise_to_result(results, "search_onestop_flight", f"{origin}:{destination}:{date}")

    @is_tool(ToolType.WRITE)
    def send_certificate(self, user_id: str, amount: int) -> str:
        """
        Send a certificate to a user. Be careful!

        Args:
            user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
            amount: The amount of the certificate to send.

        Returns:
            A message indicating the certificate was sent.

        Raises:
            ValueError: If the user is not found.
        """
        user = self._get_user(user_id)

        # add a certificate, assume at most 3 cases per task
        for payment_id in [f"certificate_{id}" for id in self._get_new_payment_id()]:
            if payment_id not in user.payment_methods:
                new_payment = Certificate(
                    id=payment_id,
                    amount=amount,
                    source="certificate",
                )
                user.payment_methods[payment_id] = new_payment
                result = {"message": f"Certificate {payment_id} added to user {user_id} with amount {amount}.", "certificate_id": payment_id, "user_id": user_id, "amount": amount}
                return _add_noise_to_result(result, "send_certificate", f"{user_id}:{amount}")
        raise ValueError("Too many certificates")

    # @is_tool(ToolType.THINK)
    # def think(self, thought: str) -> str:
    #     """
    #     Use the tool to think about something.
    #     It will not obtain new information or change the database, but just append the thought to the log.
    #     Use it when complex reasoning or some cache memory is needed.

    #     Args:
    #         thought: A thought to think about.

    #     Returns:
    #         Empty string
    #     """
    #     return ""

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        # Use deterministic transfer_id based on summary hash
        transfer_id = f"TRF-{hashlib.md5(summary.encode()).hexdigest()[:8].upper()}"
        result = {"status": "Transfer successful", "summary": summary, "transfer_id": transfer_id}
        return _add_noise_to_result(result, "transfer_to_human_agents", summary[:50])

    @is_tool(ToolType.WRITE)
    def update_reservation_baggages(
        self,
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        """
        Update the baggage information of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'
            total_baggages: The updated total number of baggage items included in the reservation.
            nonfree_baggages: The updated number of non-free baggage items included in the reservation.
            payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If the payment method is not found.
            ValueError: If the certificate cannot be used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # Calculate price
        total_price = 50 * max(0, nonfree_baggages - reservation.nonfree_baggages)

        # Create payment
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation
        reservation.total_baggages = total_baggages
        reservation.nonfree_baggages = nonfree_baggages

        return _add_noise_to_result(reservation, "update_reservation_baggages", f"{reservation_id}:{total_baggages}:{nonfree_baggages}")

    @is_tool(ToolType.WRITE)
    def update_reservation_flights(
        self,
        reservation_id: str,
        cabin: CabinClass,
        flights: List[FlightInfo | dict],
        payment_id: str,
    ) -> str:
        """
        Update the flight information of a reservation. If performing a downgrade, the refunded amount will be shown in the payment_id, baggages are automatically adjusted and accounted for in the refunded amount.


        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.
            cabin: The cabin class of the reservation
            flights: An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if the a flight segment is not changed, it should still be included in the array.
            payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the user is not found.
            ValueError: If the payment method is not found.
            ValueError: If the certificate cannot be used to update reservation.
            ValueError: If the gift card balance is not enough.
        """
        if all(isinstance(flight, dict) for flight in flights):
            flights = [FlightInfo(**flight) for flight in flights]
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        # update flights and calculate price
        total_price = 0
        reservation_flights = []
        for flight_info in flights:
            # if existing flight, keep it
            matching_reservation_flight = next(
                (
                    reservation_flight
                    for reservation_flight in reservation.flights
                    if reservation_flight.flight_number == flight_info.flight_number
                    and reservation_flight.date == flight_info.date
                    and cabin == reservation.cabin
                ),
                None,
            )
            if matching_reservation_flight:
                total_price += matching_reservation_flight.price * len(
                    reservation.passengers
                )
                reservation_flights.append(matching_reservation_flight)
                continue

            # If new flight:
            flight = self._get_flight(flight_info.flight_number)
            # Check flight availability
            flight_date_data = self._get_flight_instance(
                flight_number=flight_info.flight_number,
                date=flight_info.date,
            )
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(
                    f"Flight {flight_info.flight_number} not available on date {flight_info.date}"
                )

            # Check seat availability
            if flight_date_data.available_seats[cabin] < len(reservation.passengers):
                raise ValueError(
                    f"Not enough seats on flight {flight_info.flight_number}"
                )

            # Calculate price and add to reservation
            reservation_flight = ReservationFlight(
                flight_number=flight_info.flight_number,
                date=flight_info.date,
                price=flight_date_data.prices[cabin],
                origin=flight.origin,
                destination=flight.destination,
            )
            total_price += reservation_flight.price * len(reservation.passengers)
            reservation_flights.append(reservation_flight)

        # Deduct amount already paid for reservation
        total_price -= sum(flight.price for flight in reservation.flights) * len(
            reservation.passengers
        )

        # Create payment
        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        # Update reservation
        reservation.flights = reservation_flights
        reservation.cabin = cabin  # This was missing from original TauBench

        # Do not make flight database update here, assume it takes time to be updated # TODO: So this means that we don't update the seats here. What about in cancel_reservation?
        return _add_noise_to_result(reservation, "update_reservation_flights", f"{reservation_id}:{cabin}")

    @is_tool(ToolType.WRITE)
    def update_reservation_passengers(
        self, reservation_id: str, passengers: List[Passenger | dict]
    ) -> str:
        """
        Update the passenger information of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'ZFA04Y'.
            passengers: An array of objects containing details about each passenger.

        Returns:
            The updated reservation.

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the number of passengers does not match.
        """
        if all(isinstance(passenger, dict) for passenger in passengers):
            passengers = [Passenger(**passenger) for passenger in passengers]
        reservation = self._get_reservation(reservation_id)
        logger.info(len(passengers))
        logger.info(len(reservation.passengers))
        if len(passengers) != len(reservation.passengers):
            raise ValueError("Number of passengers does not match")
        reservation.passengers = deepcopy(passengers)
        return _add_noise_to_result(reservation, "update_reservation_passengers", reservation_id)

    @is_tool(ToolType.READ)
    def get_flight_status(self, flight_number: str, date: str) -> str:
        """
        Get the status of a flight.

        Args:
            flight_number: The flight number.
            date: The date of the flight.

        Returns:
            The status of the flight.

        Raises:
            ValueError: If the flight is not found.
        """
        result = {"status": self._get_flight_instance(flight_number, date).status}
        return _add_noise_to_result(result, "get_flight_status", f"{flight_number}:{date}")


if __name__ == "__main__":
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    airline = AirlineTools(FlightDB.load(AIRLINE_DB_PATH))
    print(airline.get_statistics())
