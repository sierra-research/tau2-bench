"""Toolkit for the hotel concierge system."""

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.hotel.data_model import (
    Amenity,
    BookingStatus,
    Experience,
    ExperienceBooking,
    Guest,
    HotelDB,
    LocalInfo,
    PaymentInfo,
    PaymentMethodType,
    ServiceRequest,
    ServiceRequestStatus,
    ServiceRequestType,
    StaffRoute,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HotelTools(ToolKitBase):
    """All the tools for the hotel concierge domain."""

    db: HotelDB

    def __init__(self, db: HotelDB) -> None:
        super().__init__(db)

    def _get_guest(self, guest_id: str) -> Guest:
        """Get guest from database."""
        if guest_id not in self.db.guests:
            raise ValueError(f"Guest {guest_id} not found")
        return self.db.guests[guest_id]

    def _get_amenity(self, amenity_id: str) -> Amenity:
        """Get amenity from database."""
        if amenity_id not in self.db.amenities:
            raise ValueError(f"Amenity {amenity_id} not found")
        return self.db.amenities[amenity_id]

    def _get_experience(self, experience_id: str) -> Experience:
        """Get experience from database."""
        if experience_id not in self.db.experiences:
            raise ValueError(f"Experience {experience_id} not found")
        return self.db.experiences[experience_id]

    def _get_experience_booking(self, booking_id: str) -> ExperienceBooking:
        """Get experience booking from database."""
        if booking_id not in self.db.experience_bookings:
            raise ValueError(f"Experience booking {booking_id} not found")
        return self.db.experience_bookings[booking_id]

    def _get_service_request(self, request_id: str) -> ServiceRequest:
        """Get service request from database."""
        if request_id not in self.db.service_requests:
            raise ValueError(f"Service request {request_id} not found")
        return self.db.service_requests[request_id]

    def _get_new_request_id(self) -> str:
        """Generate a new service request ID."""
        base_id = len(self.db.service_requests) + 1
        while f"SR{base_id:04d}" in self.db.service_requests:
            base_id += 1
        return f"SR{base_id:04d}"

    def _get_new_booking_id(self) -> str:
        """Generate a new experience booking ID."""
        base_id = len(self.db.experience_bookings) + 1
        while f"EB{base_id:04d}" in self.db.experience_bookings:
            base_id += 1
        return f"EB{base_id:04d}"

    def _get_new_route_id(self) -> str:
        """Generate a new staff route ID."""
        base_id = len(self.db.staff_routes) + 1
        while f"RT{base_id:04d}" in self.db.staff_routes:
            base_id += 1
        return f"RT{base_id:04d}"

    def _get_new_transaction_id(self) -> str:
        """Generate a new transaction ID."""
        return f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _get_current_datetime(self) -> str:
        """Get current datetime in ISO format."""
        return "2024-05-15T15:00:00"

    # ============ READ TOOLS ============

    @is_tool(ToolType.READ)
    def get_guest_details(self, guest_id: str) -> Guest:
        """
        Retrieve detailed information about a guest including their profile, current booking, preferences, and history.

        Args:
            guest_id: The unique identifier for the guest

        Returns:
            Complete guest information including booking details, preferences, and service history

        Raises:
            ValueError: If the guest_id is not found in the database
        """
        return self._get_guest(guest_id)

    @is_tool(ToolType.READ)
    def get_amenity_info(self, amenity_name: str) -> Optional[Amenity]:
        """
        Get detailed information about a hotel amenity including hours, location, and availability.

        Args:
            amenity_name: Name of the amenity (e.g., "Pool", "Gym", "Spa", "Restaurant")

        Returns:
            Amenity information if found, None otherwise
        """
        for amenity in self.db.amenities.values():
            if amenity.name.lower() == amenity_name.lower():
                return amenity
        return None

    @is_tool(ToolType.READ)
    def search_experiences(
        self,
        category: Optional[str] = None,
        date: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> List[Experience]:
        """
        Search for local experiences available for booking.

        Args:
            category: Filter by category (dining, spa, tours, activities, transportation)
            date: Filter by available date in YYYY-MM-DD format
            max_price: Maximum price per person in dollars

        Returns:
            List of matching experiences
        """
        results = []
        for experience in self.db.experiences.values():
            matches = True
            if category and experience.category != category:
                matches = False
            if max_price and experience.price > max_price:
                matches = False
            # Note: date filtering would require checking availability calendar
            # For now, we assume all experiences are available on requested dates
            if matches:
                results.append(experience)
        return results

    @is_tool(ToolType.READ)
    def get_experience_details(self, experience_id: str) -> Experience:
        """
        Get complete details about a specific experience.

        Args:
            experience_id: Unique identifier for the experience

        Returns:
            Complete experience information including pricing, duration, and availability

        Raises:
            ValueError: If the experience_id is not found
        """
        return self._get_experience(experience_id)

    @is_tool(ToolType.READ)
    def check_room_service_menu(self) -> Dict[str, Any]:
        """
        Get the room service menu with available items and delivery times.

        Returns:
            Dictionary containing menu categories and items with prices
        """
        return {
            "breakfast": [
                {
                    "name": "Continental Breakfast",
                    "price": 18.00,
                    "delivery_time": "30-45 min",
                },
                {
                    "name": "American Breakfast",
                    "price": 24.00,
                    "delivery_time": "30-45 min",
                },
                {"name": "Pancake Stack", "price": 16.00, "delivery_time": "30-45 min"},
            ],
            "lunch_dinner": [
                {"name": "Caesar Salad", "price": 16.00, "delivery_time": "20-30 min"},
                {"name": "Club Sandwich", "price": 18.00, "delivery_time": "20-30 min"},
                {
                    "name": "Grilled Salmon",
                    "price": 32.00,
                    "delivery_time": "35-45 min",
                },
                {
                    "name": "Pasta Primavera",
                    "price": 22.00,
                    "delivery_time": "30-40 min",
                },
            ],
            "beverages": [
                {"name": "Coffee", "price": 5.00, "delivery_time": "15-20 min"},
                {"name": "Fresh Juice", "price": 8.00, "delivery_time": "15-20 min"},
                {"name": "Wine (bottle)", "price": 45.00, "delivery_time": "15-20 min"},
            ],
            "delivery_hours": "24/7",
            "note": "Prices do not include 18% service charge and applicable taxes",
        }

    @is_tool(ToolType.READ)
    def get_local_info(self, query_type: str) -> List[LocalInfo]:
        """
        Get information about local points of interest near the hotel.

        Args:
            query_type: Type of information requested (pharmacy, ATM, restaurant, transit, etc.)

        Returns:
            List of relevant local information
        """
        results = []
        for info in self.db.local_info:
            if query_type.lower() in info.category.lower():
                results.append(info)
        return results

    @is_tool(ToolType.READ)
    def get_service_request_status(self, request_id: str) -> ServiceRequest:
        """
        Check the status of a service request.

        Args:
            request_id: Unique identifier for the service request

        Returns:
            Service request details including current status

        Raises:
            ValueError: If the request_id is not found
        """
        return self._get_service_request(request_id)

    @is_tool(ToolType.READ)
    def get_experience_booking_details(self, booking_id: str) -> ExperienceBooking:
        """
        Get details about an experience booking.

        Args:
            booking_id: Unique identifier for the booking

        Returns:
            Complete booking information including payment and status

        Raises:
            ValueError: If the booking_id is not found
        """
        return self._get_experience_booking(booking_id)

    # ============ WRITE TOOLS ============

    @is_tool(ToolType.WRITE)
    def create_service_request(
        self,
        guest_id: str,
        request_type: ServiceRequestType,
        details: str,
    ) -> ServiceRequest:
        """
        Create a new service request for a guest.

        Args:
            guest_id: ID of the guest making the request
            request_type: Type of service (towels, pillows, maintenance, housekeeping, wake_up_call, room_service, other)
            details: Detailed description of the request

        Returns:
            The created service request

        Raises:
            ValueError: If the guest_id is not found
        """
        guest = self._get_guest(guest_id)
        request_id = self._get_new_request_id()

        service_request = ServiceRequest(
            request_id=request_id,
            guest_id=guest_id,
            request_type=request_type,
            details=details,
            status="pending",
            created_at=self._get_current_datetime(),
        )

        self.db.service_requests[request_id] = service_request
        guest.service_requests.append(request_id)

        logger.info(f"Created service request {request_id} for guest {guest_id}")
        return service_request

    @is_tool(ToolType.WRITE)
    def book_experience(
        self,
        guest_id: str,
        experience_id: str,
        date: str,
        time: str,
        participants: int,
        payment_method: PaymentMethodType,
    ) -> ExperienceBooking:
        """
        Book an experience for a guest.

        Args:
            guest_id: ID of the guest making the booking
            experience_id: ID of the experience to book
            date: Date of the experience in YYYY-MM-DD format
            time: Time of the experience in HH:MM format
            participants: Number of participants
            payment_method: Payment method (credit_card, apple_pay, google_pay, room_charge)

        Returns:
            The created booking

        Raises:
            ValueError: If guest or experience not found, or if time is not available
        """
        guest = self._get_guest(guest_id)
        experience = self._get_experience(experience_id)

        # Validate time is available
        if time not in experience.available_times:
            raise ValueError(
                f"Time {time} is not available for this experience. Available times: {experience.available_times}"
            )

        # Validate participant count
        if participants > experience.max_participants:
            raise ValueError(
                f"Number of participants ({participants}) exceeds maximum ({experience.max_participants})"
            )

        booking_id = self._get_new_booking_id()
        total_price = experience.price * participants

        payment = PaymentInfo(
            method=payment_method,
            amount=total_price,
            transaction_id=self._get_new_transaction_id(),
            timestamp=self._get_current_datetime(),
        )

        booking = ExperienceBooking(
            booking_id=booking_id,
            guest_id=guest_id,
            experience_id=experience_id,
            date=date,
            time=time,
            participants=participants,
            total_price=total_price,
            status="confirmed",
            payment=payment,
            created_at=self._get_current_datetime(),
        )

        self.db.experience_bookings[booking_id] = booking
        guest.experience_bookings.append(booking_id)

        logger.info(f"Created experience booking {booking_id} for guest {guest_id}")
        return booking

    @is_tool(ToolType.WRITE)
    def cancel_experience_booking(self, booking_id: str) -> ExperienceBooking:
        """
        Cancel an experience booking according to the cancellation policy.

        Args:
            booking_id: ID of the booking to cancel

        Returns:
            The updated booking with cancelled status

        Raises:
            ValueError: If booking not found or if cancellation window has passed
        """
        booking = self._get_experience_booking(booking_id)

        if booking.status == "cancelled":
            raise ValueError(f"Booking {booking_id} is already cancelled")

        if booking.status == "completed":
            raise ValueError(f"Cannot cancel completed booking {booking_id}")

        # In a real system, we would check if we're within the cancellation window
        # For now, we'll allow cancellation

        booking.status = "cancelled"
        booking.cancelled_at = self._get_current_datetime()

        logger.info(f"Cancelled experience booking {booking_id}")
        return booking

    @is_tool(ToolType.WRITE)
    def send_to_staff(
        self,
        guest_id: str,
        message_type: str,
        content: str,
    ) -> StaffRoute:
        """
        Route a message to hotel staff for handling.

        Args:
            guest_id: ID of the guest sending the message
            message_type: Type of message (complaint, special_request, question, emergency)
            content: The message content

        Returns:
            The staff route record

        Raises:
            ValueError: If guest not found
        """
        self._get_guest(guest_id)  # Validate guest exists
        route_id = self._get_new_route_id()

        staff_route = StaffRoute(
            route_id=route_id,
            guest_id=guest_id,
            message_type=message_type,
            content=content,
            timestamp=self._get_current_datetime(),
            handled=False,
        )

        self.db.staff_routes[route_id] = staff_route

        logger.info(f"Routed message {route_id} from guest {guest_id} to staff")
        return staff_route

    @is_tool(ToolType.WRITE)
    def update_service_request_status(
        self,
        request_id: str,
        status: ServiceRequestStatus,
        notes: Optional[str] = None,
    ) -> ServiceRequest:
        """
        Update the status of a service request (typically used by staff).

        Args:
            request_id: ID of the service request
            status: New status (pending, in_progress, completed, cancelled)
            notes: Optional notes about the update

        Returns:
            The updated service request

        Raises:
            ValueError: If request not found
        """
        service_request = self._get_service_request(request_id)
        service_request.status = status

        if notes:
            service_request.notes = notes

        if status == "completed":
            service_request.completed_at = self._get_current_datetime()

        logger.info(f"Updated service request {request_id} status to {status}")
        return service_request
