import pytest
from loguru import logger

from tau2.domains.hotel.data_model import (
    Amenity,
    Experience,
    ExperienceBooking,
    Guest,
    GuestName,
    HotelDB,
    RoomBooking,
    ServiceRequest,
)
from tau2.domains.hotel.environment import get_environment
from tau2.domains.hotel.tools import HotelTools
from tau2.environment.environment import Environment


@pytest.fixture
def hotel_db() -> HotelDB:
    """Create a minimal hotel database for testing"""
    return HotelDB(
        guests={
            "G001": Guest(
                guest_id="G001",
                name=GuestName(first_name="Sarah", last_name="Johnson"),
                email="sarah.j@email.com",
                phone="+1-555-0101",
                loyalty_tier="platinum",
                loyalty_points=15000,
                current_booking=RoomBooking(
                    booking_id="RB001",
                    room_number="305",
                    room_type="deluxe",
                    check_in="2024-05-14",
                    check_out="2024-05-18",
                    guests=2,
                    rate_per_night=250.00,
                ),
                service_requests=[],
                experience_bookings=[],
            ),
            "G002": Guest(
                guest_id="G002",
                name=GuestName(first_name="Michael", last_name="Chen"),
                email="m.chen@email.com",
                phone="+1-555-0102",
                loyalty_tier="gold",
                loyalty_points=8500,
                current_booking=RoomBooking(
                    booking_id="RB002",
                    room_number="412",
                    room_type="suite",
                    check_in="2024-05-13",
                    check_out="2024-05-17",
                    guests=1,
                    rate_per_night=350.00,
                ),
                service_requests=[],
                experience_bookings=[],
            ),
        },
        amenities={
            "A001": Amenity(
                amenity_id="A001",
                name="Pool",
                location="Rooftop",
                hours_open="06:00",
                hours_close="22:00",
                description="Heated outdoor pool",
                capacity=50,
                currently_available=True,
            ),
            "A002": Amenity(
                amenity_id="A002",
                name="Gym",
                location="2nd Floor",
                hours_open="05:00",
                hours_close="23:00",
                description="Fully equipped fitness center",
                capacity=30,
                currently_available=True,
            ),
        },
        service_requests={},
        experiences={
            "E001": Experience(
                experience_id="E001",
                name="City Walking Tour",
                category="tours",
                description="Guided walking tour",
                location="Hotel lobby",
                price=45.00,
                duration="2 hours",
                available_times=["09:00", "14:00"],
                max_participants=15,
                min_age=None,
                cancellation_hours=24,
            ),
            "E002": Experience(
                experience_id="E002",
                name="Wine Tasting",
                category="tours",
                description="Wine tasting tour",
                location="Pickup at hotel",
                price=120.00,
                duration="5 hours",
                available_times=["10:00", "14:00"],
                max_participants=12,
                min_age=21,
                cancellation_hours=48,
            ),
        },
        experience_bookings={},
        staff_routes={},
        local_info=[],
    )


@pytest.fixture
def hotel_env(hotel_db: HotelDB) -> Environment:
    """Create hotel environment for testing"""
    return get_environment(db=hotel_db)


@pytest.fixture
def hotel_tools(hotel_db: HotelDB) -> HotelTools:
    """Create hotel tools for testing"""
    return HotelTools(hotel_db)


class TestHotelTools:
    """Test hotel tools functionality"""

    def test_get_guest_details(self, hotel_tools: HotelTools):
        """Test retrieving guest details"""
        guest = hotel_tools.get_guest_details("G001")
        assert guest.guest_id == "G001"
        assert guest.name.first_name == "Sarah"
        assert guest.loyalty_tier == "platinum"
        assert guest.current_booking.room_number == "305"

    def test_get_guest_details_not_found(self, hotel_tools: HotelTools):
        """Test retrieving non-existent guest"""
        with pytest.raises(ValueError, match="Guest .* not found"):
            hotel_tools.get_guest_details("G999")

    def test_get_amenity_info(self, hotel_tools: HotelTools):
        """Test retrieving amenity information"""
        amenity = hotel_tools.get_amenity_info("Pool")
        assert amenity is not None
        assert amenity.amenity_id == "A001"
        assert amenity.hours_open == "06:00"
        assert amenity.hours_close == "22:00"

    def test_get_amenity_info_case_insensitive(self, hotel_tools: HotelTools):
        """Test amenity search is case insensitive"""
        amenity = hotel_tools.get_amenity_info("pool")
        assert amenity is not None
        assert amenity.name == "Pool"

    def test_get_amenity_info_not_found(self, hotel_tools: HotelTools):
        """Test retrieving non-existent amenity"""
        amenity = hotel_tools.get_amenity_info("Spa")
        assert amenity is None

    def test_search_experiences_all(self, hotel_tools: HotelTools):
        """Test searching all experiences"""
        experiences = hotel_tools.search_experiences()
        assert len(experiences) == 2

    def test_search_experiences_by_category(self, hotel_tools: HotelTools):
        """Test searching experiences by category"""
        experiences = hotel_tools.search_experiences(category="tours")
        assert len(experiences) == 2
        assert all(e.category == "tours" for e in experiences)

    def test_search_experiences_by_price(self, hotel_tools: HotelTools):
        """Test searching experiences by max price"""
        experiences = hotel_tools.search_experiences(max_price=50.0)
        assert len(experiences) == 1
        assert experiences[0].experience_id == "E001"

    def test_get_experience_details(self, hotel_tools: HotelTools):
        """Test retrieving experience details"""
        experience = hotel_tools.get_experience_details("E001")
        assert experience.name == "City Walking Tour"
        assert experience.price == 45.00
        assert "09:00" in experience.available_times

    def test_get_experience_details_not_found(self, hotel_tools: HotelTools):
        """Test retrieving non-existent experience"""
        with pytest.raises(ValueError, match="Experience .* not found"):
            hotel_tools.get_experience_details("E999")

    def test_check_room_service_menu(self, hotel_tools: HotelTools):
        """Test retrieving room service menu"""
        menu = hotel_tools.check_room_service_menu()
        assert "breakfast" in menu
        assert "lunch_dinner" in menu
        assert "beverages" in menu
        assert len(menu["breakfast"]) > 0

    def test_create_service_request(self, hotel_tools: HotelTools):
        """Test creating a service request"""
        request = hotel_tools.create_service_request(
            guest_id="G001",
            request_type="towels",
            details="Need 2 extra bath towels",
        )
        assert request.guest_id == "G001"
        assert request.request_type == "towels"
        assert request.status == "pending"
        assert request.request_id in hotel_tools.db.service_requests
        assert request.request_id in hotel_tools.db.guests["G001"].service_requests

    def test_create_service_request_invalid_guest(self, hotel_tools: HotelTools):
        """Test creating service request for non-existent guest"""
        with pytest.raises(ValueError, match="Guest .* not found"):
            hotel_tools.create_service_request(
                guest_id="G999",
                request_type="towels",
                details="Need towels",
            )

    def test_book_experience(self, hotel_tools: HotelTools):
        """Test booking an experience"""
        booking = hotel_tools.book_experience(
            guest_id="G001",
            experience_id="E001",
            date="2024-05-16",
            time="09:00",
            participants=2,
            payment_method="room_charge",
        )
        assert booking.guest_id == "G001"
        assert booking.experience_id == "E001"
        assert booking.participants == 2
        assert booking.status == "confirmed"
        assert booking.total_price == 90.00  # 45 * 2
        assert booking.booking_id in hotel_tools.db.experience_bookings
        assert booking.booking_id in hotel_tools.db.guests["G001"].experience_bookings

    def test_book_experience_invalid_time(self, hotel_tools: HotelTools):
        """Test booking experience with unavailable time"""
        with pytest.raises(ValueError, match="Time .* is not available"):
            hotel_tools.book_experience(
                guest_id="G001",
                experience_id="E001",
                date="2024-05-16",
                time="12:00",  # Not in available_times
                participants=1,
                payment_method="room_charge",
            )

    def test_book_experience_exceeds_capacity(self, hotel_tools: HotelTools):
        """Test booking experience with too many participants"""
        with pytest.raises(ValueError, match="exceeds maximum"):
            hotel_tools.book_experience(
                guest_id="G001",
                experience_id="E001",
                date="2024-05-16",
                time="09:00",
                participants=20,  # Exceeds max_participants of 15
                payment_method="room_charge",
            )

    def test_cancel_experience_booking(self, hotel_tools: HotelTools):
        """Test cancelling an experience booking"""
        # First create a booking
        booking = hotel_tools.book_experience(
            guest_id="G001",
            experience_id="E001",
            date="2024-05-16",
            time="09:00",
            participants=1,
            payment_method="room_charge",
        )
        booking_id = booking.booking_id

        # Then cancel it
        cancelled_booking = hotel_tools.cancel_experience_booking(booking_id)
        assert cancelled_booking.status == "cancelled"
        assert cancelled_booking.cancelled_at is not None

    def test_cancel_already_cancelled_booking(self, hotel_tools: HotelTools):
        """Test cancelling an already cancelled booking"""
        # Create and cancel a booking
        booking = hotel_tools.book_experience(
            guest_id="G001",
            experience_id="E001",
            date="2024-05-16",
            time="09:00",
            participants=1,
            payment_method="room_charge",
        )
        hotel_tools.cancel_experience_booking(booking.booking_id)

        # Try to cancel again
        with pytest.raises(ValueError, match="already cancelled"):
            hotel_tools.cancel_experience_booking(booking.booking_id)

    def test_send_to_staff(self, hotel_tools: HotelTools):
        """Test routing message to staff"""
        staff_route = hotel_tools.send_to_staff(
            guest_id="G001",
            message_type="complaint",
            content="Room is too noisy",
        )
        assert staff_route.guest_id == "G001"
        assert staff_route.message_type == "complaint"
        assert staff_route.handled is False
        assert staff_route.route_id in hotel_tools.db.staff_routes

    def test_update_service_request_status(self, hotel_tools: HotelTools):
        """Test updating service request status"""
        # Create a request first
        request = hotel_tools.create_service_request(
            guest_id="G001",
            request_type="towels",
            details="Need towels",
        )

        # Update its status
        updated_request = hotel_tools.update_service_request_status(
            request_id=request.request_id,
            status="completed",
            notes="Delivered to room",
        )
        assert updated_request.status == "completed"
        assert updated_request.notes == "Delivered to room"
        assert updated_request.completed_at is not None


class TestHotelEnvironment:
    """Test hotel environment setup"""

    def test_environment_creation(self, hotel_env: Environment):
        """Test that environment is created correctly"""
        assert hotel_env.domain_name == "hotel"
        assert hotel_env.policy is not None
        assert hotel_env.tools is not None
        assert isinstance(hotel_env.tools, HotelTools)

    def test_environment_has_policy(self, hotel_env: Environment):
        """Test that policy is loaded"""
        assert len(hotel_env.policy) > 0
        assert (
            "hotel" in hotel_env.policy.lower()
            or "concierge" in hotel_env.policy.lower()
        )

    def test_environment_tools_accessible(self, hotel_env: Environment):
        """Test that tools are accessible through environment"""
        tools = hotel_env.tools
        assert hasattr(tools, "get_guest_details")
        assert hasattr(tools, "get_amenity_info")
        assert hasattr(tools, "book_experience")

    def test_solo_mode_not_supported(self):
        """Test that solo mode raises an error"""
        with pytest.raises(ValueError, match="does not support solo mode"):
            get_environment(solo_mode=True)
