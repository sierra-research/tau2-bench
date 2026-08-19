"""Tests for vacation rental domain tools.

Follows the pattern from test_tools_airline.py:
- Creates minimal fixture databases
- Tests each tool via environment.get_response(ToolCall(...))
- Tests both success and error cases
- Verifies state changes after write operations
"""

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.vacation_rental.data_model import (
    CreditCard,
    FlexibilitySettings,
    GuestHistory,
    HostDecision,
    HostPhilosophy,
    HostProfile,
    Issue,
    Listing,
    ListingAddress,
    Reservation,
    User,
    UserName,
    VacationRentalDB,
)
from tau2.domains.vacation_rental.environment import get_environment
from tau2.domains.vacation_rental.utils import CURRENT_TIME
from tau2.environment.environment import Environment

# === Fixtures ===


@pytest.fixture
def vacation_rental_db() -> VacationRentalDB:
    """Create a minimal test database."""
    return VacationRentalDB(
        users={
            "guest_001": User(
                user_id="guest_001",
                name=UserName(first_name="Test", last_name="Guest"),
                email="test@example.com",
                phone="+1-555-0001",
                payment_methods={
                    "card_001": CreditCard(
                        source="credit_card",
                        id="card_001",
                        brand="visa",
                        last_four="1234",
                        expiration="2027-12",
                    ),
                },
                reservations=[
                    "RES001",
                    "RES_FLEX_FULL",
                    "RES_FLEX_PARTIAL",
                    "RES_MOD_FULL",
                    "RES_MOD_PARTIAL",
                    "RES_FIRM_FULL",
                    "RES_FIRM_50",
                    "RES_FIRM_ZERO",
                    "RES_STRICT_50",
                    "RES_STRICT_ZERO",
                    "RES_CANCELLED",
                    "RES_PAST",
                    "RES_GRACE",
                    "RES_HOST",
                ],
            ),
            "host_001": User(
                user_id="host_001",
                name=UserName(first_name="Test", last_name="Host"),
                email="host@example.com",
                phone="+1-555-0002",
                payment_methods={},
                reservations=[],
            ),
        },
        listings={
            "LST001": Listing(
                listing_id="LST001",
                host_user_id="host_001",
                title="Test Property",
                address=ListingAddress(
                    address1="123 Test St",
                    city="Test City",
                    state="CA",
                    zip="90000",
                    country="USA",
                ),
                nightly_rate=100.0,
                cancellation_policy="flexible",
            ),
            "LST_MOD": Listing(
                listing_id="LST_MOD",
                host_user_id="host_001",
                title="Moderate Property",
                address=ListingAddress(
                    address1="456 Mod Ave",
                    city="Test City",
                    state="CA",
                    zip="90001",
                    country="USA",
                ),
                nightly_rate=100.0,
                cancellation_policy="moderate",
            ),
            "LST_FIRM": Listing(
                listing_id="LST_FIRM",
                host_user_id="host_001",
                title="Firm Property",
                address=ListingAddress(
                    address1="789 Firm Blvd",
                    city="Test City",
                    state="CA",
                    zip="90002",
                    country="USA",
                ),
                nightly_rate=100.0,
                cancellation_policy="firm",
            ),
            "LST_STRICT": Listing(
                listing_id="LST_STRICT",
                host_user_id="host_001",
                title="Strict Property",
                address=ListingAddress(
                    address1="321 Strict Ln",
                    city="Test City",
                    state="CA",
                    zip="90003",
                    country="USA",
                ),
                nightly_rate=100.0,
                cancellation_policy="strict",
            ),
        },
        reservations={
            "RES001": Reservation(
                reservation_id="RES001",
                guest_user_id="guest_001",
                listing_id="LST001",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Flexible: >=1 day before check-in -> full refund ($300)
            "RES_FLEX_FULL": Reservation(
                reservation_id="RES_FLEX_FULL",
                guest_user_id="guest_001",
                listing_id="LST001",
                check_in_date="2025-03-03",
                check_out_date="2025-03-06",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Flexible: <1 day before check-in -> forfeit 1st night ($200)
            "RES_FLEX_PARTIAL": Reservation(
                reservation_id="RES_FLEX_PARTIAL",
                guest_user_id="guest_001",
                listing_id="LST001",
                check_in_date="2025-03-02",
                check_out_date="2025-03-05",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Moderate: >=5 days before check-in -> full refund ($300)
            "RES_MOD_FULL": Reservation(
                reservation_id="RES_MOD_FULL",
                guest_user_id="guest_001",
                listing_id="LST_MOD",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Moderate: <5 days before check-in -> 1st night forfeit + 50% remaining
            # remaining_nights = 3-1 = 2, refund = 2 * 100 * 0.5 = $100
            "RES_MOD_PARTIAL": Reservation(
                reservation_id="RES_MOD_PARTIAL",
                guest_user_id="guest_001",
                listing_id="LST_MOD",
                check_in_date="2025-03-03",
                check_out_date="2025-03-06",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Firm: >=30 days before check-in -> full refund ($300)
            "RES_FIRM_FULL": Reservation(
                reservation_id="RES_FIRM_FULL",
                guest_user_id="guest_001",
                listing_id="LST_FIRM",
                check_in_date="2025-04-05",
                check_out_date="2025-04-08",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Firm: 7-29 days before check-in -> 50% refund ($150)
            "RES_FIRM_50": Reservation(
                reservation_id="RES_FIRM_50",
                guest_user_id="guest_001",
                listing_id="LST_FIRM",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Firm: <7 days before check-in -> no refund ($0)
            "RES_FIRM_ZERO": Reservation(
                reservation_id="RES_FIRM_ZERO",
                guest_user_id="guest_001",
                listing_id="LST_FIRM",
                check_in_date="2025-03-03",
                check_out_date="2025-03-06",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Strict: >=7 days before check-in -> 50% refund ($150)
            "RES_STRICT_50": Reservation(
                reservation_id="RES_STRICT_50",
                guest_user_id="guest_001",
                listing_id="LST_STRICT",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Strict: <7 days before check-in -> no refund ($0)
            "RES_STRICT_ZERO": Reservation(
                reservation_id="RES_STRICT_ZERO",
                guest_user_id="guest_001",
                listing_id="LST_STRICT",
                check_in_date="2025-03-03",
                check_out_date="2025-03-06",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Already cancelled — for process_refund tests
            "RES_CANCELLED": Reservation(
                reservation_id="RES_CANCELLED",
                guest_user_id="guest_001",
                listing_id="LST001",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="cancelled",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
                refund_amount=300.0,
                cancelled_by="guest",
            ),
            # Check-in already passed — for error path test
            "RES_PAST": Reservation(
                reservation_id="RES_PAST",
                guest_user_id="guest_001",
                listing_id="LST001",
                check_in_date="2025-02-20",
                check_out_date="2025-02-23",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
            # Grace period: strict policy, booked 10h ago, 9+ days before check-in
            # -> within 24h of booking AND 7+ days before check-in -> full refund
            "RES_GRACE": Reservation(
                reservation_id="RES_GRACE",
                guest_user_id="guest_001",
                listing_id="LST_STRICT",
                check_in_date="2025-03-10",
                check_out_date="2025-03-13",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-03-01T00:00:00",
                payment_method_id="card_001",
            ),
            # Host cancellation: strict policy, <7 days -> but host cancel = full refund
            "RES_HOST": Reservation(
                reservation_id="RES_HOST",
                guest_user_id="guest_001",
                listing_id="LST_STRICT",
                check_in_date="2025-03-03",
                check_out_date="2025-03-06",
                total_amount=300.0,
                amount_paid=300.0,
                status="confirmed",
                created_at="2025-02-01T10:00:00",
                payment_method_id="card_001",
            ),
        },
        host_profiles={
            "host_001": HostProfile(
                host_user_id="host_001",
                philosophy=HostPhilosophy(
                    primary_focus="reviews", risk_tolerance="high"
                ),
                flexibility_settings=FlexibilitySettings(
                    refund_flexibility="generous",
                    max_goodwill_refund_pct=50,
                    repeat_guest_bonus_pct=20,
                ),
                hard_limits=["no_parties"],
                soft_spots=["medical_emergency", "repeat_guests"],
                deal_breakers=["dishonesty"],
                dispute_resolution_preference="host_involved",
            ),
        },
        guest_history={
            "guest_001": GuestHistory(
                guest_user_id="guest_001",
                total_stays=5,
                stays_by_host={"host_001": 4},
                issues_reported=0,
                cancellation_count=0,
            ),
        },
        issues={
            "ISS_RES001_001": Issue(
                issue_id="ISS_RES001_001",
                reservation_id="RES001",
                guest_user_id="guest_001",
                reported_by="guest",
                issue_type="cleanliness",
                description="Test issue",
                severity="minor",
                evidence_submitted=True,
                evidence_status="validated",
                validation_result="Photos confirm issue",
                status="investigating",
                created_at="2025-03-01T10:00:00",
            ),
            "ISS_NO_EVIDENCE": Issue(
                issue_id="ISS_NO_EVIDENCE",
                reservation_id="RES001",
                guest_user_id="guest_001",
                reported_by="guest",
                issue_type="cleanliness",
                description="No evidence issue",
                severity="minor",
                evidence_submitted=False,
                evidence_status="pending",
                status="open",
                created_at="2025-03-01T10:00:00",
            ),
            "ISS_INVALIDATED": Issue(
                issue_id="ISS_INVALIDATED",
                reservation_id="RES001",
                guest_user_id="guest_001",
                reported_by="guest",
                issue_type="not_as_described",
                description="Invalidated evidence issue",
                severity="moderate",
                evidence_submitted=True,
                evidence_status="invalidated",
                validation_result="Photos do not match claim",
                status="investigating",
                created_at="2025-03-01T10:00:00",
            ),
            "ISS_INCONCLUSIVE": Issue(
                issue_id="ISS_INCONCLUSIVE",
                reservation_id="RES001",
                guest_user_id="guest_001",
                reported_by="guest",
                issue_type="amenity_malfunction",
                description="Inconclusive evidence issue",
                severity="major",
                evidence_submitted=True,
                evidence_status="inconclusive",
                validation_result="Cannot determine from photos",
                status="investigating",
                created_at="2025-03-01T10:00:00",
            ),
        },
        host_decisions={
            "HD_001": HostDecision(
                decision_id="HD_001",
                host_user_id="host_001",
                situation_type="cancellation_exception",
                guest_context="medical_emergency",
                decision="approve",
                approved_amount_pct=100,
                reasoning="Host approves medical emergencies",
                conditions=[],
            ),
        },
    )


@pytest.fixture
def environment(vacation_rental_db: VacationRentalDB) -> Environment:
    """Create environment with test database."""
    return get_environment(vacation_rental_db)


# === Existing Tool Tests ===


def test_get_host_profile(environment: Environment):
    """Test retrieving host profile."""
    call = ToolCall(
        id="1",
        name="get_host_profile",
        arguments={"host_user_id": "host_001"},
    )
    response = environment.get_response(call)
    assert not response.error
    # Content is serialized to string, check for expected values
    assert "reviews" in str(response.content)
    assert "medical_emergency" in str(response.content)


def test_get_host_profile_not_found(environment: Environment):
    """Test error when host profile not found."""
    call = ToolCall(
        id="1",
        name="get_host_profile",
        arguments={"host_user_id": "nonexistent"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_get_guest_history(environment: Environment):
    """Test retrieving guest history."""
    call = ToolCall(
        id="1",
        name="get_guest_history",
        arguments={"guest_user_id": "guest_001"},
    )
    response = environment.get_response(call)
    assert not response.error
    # Content can be either JSON or Python repr format
    content = str(response.content)
    assert "5" in content  # total_stays
    assert "host_001" in content


def test_get_guest_history_not_found(environment: Environment):
    """Test error when guest history not found."""
    call = ToolCall(
        id="1",
        name="get_guest_history",
        arguments={"guest_user_id": "nonexistent"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_submit_issue_report(environment: Environment):
    """Test submitting an issue report."""
    call = ToolCall(
        id="1",
        name="submit_issue_report",
        arguments={
            "reservation_id": "RES001",
            "issue_type": "cleanliness",
            "description": "Kitchen not clean",
            "severity": "minor",
            "evidence_submitted": True,
        },
    )
    response = environment.get_response(call)
    assert not response.error
    # Content can be either JSON or Python repr format
    content = str(response.content)
    assert "open" in content  # status
    assert "pending" in content  # evidence_status


def test_submit_issue_report_invalid_type(environment: Environment):
    """Test error on invalid issue type."""
    call = ToolCall(
        id="1",
        name="submit_issue_report",
        arguments={
            "reservation_id": "RES001",
            "issue_type": "invalid_type",
            "description": "Test",
            "severity": "minor",
        },
    )
    response = environment.get_response(call)
    assert response.error or "invalid" in str(response.content).lower()


def test_get_issue_details(environment: Environment):
    """Test retrieving issue details."""
    call = ToolCall(
        id="1",
        name="get_issue_details",
        arguments={"issue_id": "ISS_RES001_001"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "cleanliness" in str(response.content)


def test_get_issue_details_not_found(environment: Environment):
    """Test error when issue not found."""
    call = ToolCall(
        id="1",
        name="get_issue_details",
        arguments={"issue_id": "ISS_NONEXISTENT"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_validate_issue_evidence(environment: Environment):
    """Test validating issue evidence."""
    call = ToolCall(
        id="1",
        name="validate_issue_evidence",
        arguments={"issue_id": "ISS_RES001_001"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "validated" in str(response.content)
    assert "recommendation" in str(response.content)


def test_validate_issue_evidence_not_found(environment: Environment):
    """Test error when issue not found for validation."""
    call = ToolCall(
        id="1",
        name="validate_issue_evidence",
        arguments={"issue_id": "ISS_NONEXISTENT"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_request_host_decision(environment: Environment):
    """Test requesting host decision."""
    call = ToolCall(
        id="1",
        name="request_host_decision",
        arguments={
            "host_user_id": "host_001",
            "situation_type": "cancellation_exception",
            "guest_context": "medical_emergency",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "approve" in str(response.content)


def test_request_host_decision_default_fallback(environment: Environment):
    """Test fallback to defer_to_policy when no match."""
    call = ToolCall(
        id="1",
        name="request_host_decision",
        arguments={
            "host_user_id": "host_001",
            "situation_type": "late_checkout_request",
            "guest_context": "unknown",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "defer_to_policy" in str(response.content)


def test_process_goodwill_refund(environment: Environment):
    """Test processing goodwill refund."""
    call = ToolCall(
        id="1",
        name="process_goodwill_refund",
        arguments={
            "reservation_id": "RES001",
            "amount": 50.0,
            "justification": "Host-approved exception",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "success" in str(response.content).lower()
    assert "50" in str(response.content)


def test_process_goodwill_refund_exceeds_limit(environment: Environment):
    """Test error when goodwill refund exceeds host limit."""
    # Host max is 50% of 300 = 150, so 200 should fail
    call = ToolCall(
        id="1",
        name="process_goodwill_refund",
        arguments={
            "reservation_id": "RES001",
            "amount": 200.0,
            "justification": "Test",
        },
    )
    response = environment.get_response(call)
    assert response.error or "exceeds" in str(response.content).lower()


def test_apply_service_credit(environment: Environment):
    """Test applying service credit."""
    call = ToolCall(
        id="1",
        name="apply_service_credit",
        arguments={
            "user_id": "guest_001",
            "amount": 25.0,
            "reason": "Apology for cleanliness issue",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "success" in str(response.content).lower()
    assert "25" in str(response.content)


def test_apply_service_credit_user_not_found(environment: Environment):
    """Test error when user not found for credit."""
    call = ToolCall(
        id="1",
        name="apply_service_credit",
        arguments={
            "user_id": "nonexistent",
            "amount": 25.0,
            "reason": "Test",
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_add_reservation_note(environment: Environment):
    """Test adding reservation note."""
    call = ToolCall(
        id="1",
        name="add_reservation_note",
        arguments={
            "reservation_id": "RES001",
            "note": "Early check-in approved by host",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "success" in str(response.content).lower()


def test_add_reservation_note_not_found(environment: Environment):
    """Test error when reservation not found for note."""
    call = ToolCall(
        id="1",
        name="add_reservation_note",
        arguments={
            "reservation_id": "NONEXISTENT",
            "note": "Test",
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_calculate(environment: Environment):
    """Test calculate tool."""
    call = ToolCall(
        id="1",
        name="calculate",
        arguments={"expression": "100 * 0.5"},
    )
    response = environment.get_response(call)
    assert response.content == "50.0"


def test_transfer_to_human_agents(environment: Environment):
    """Test transfer to human agents."""
    call = ToolCall(
        id="1",
        name="transfer_to_human_agents",
        arguments={"summary": "Guest dispute requires human review"},
    )
    response = environment.get_response(call)
    assert not response.error


# === Simple Read Tool Tests ===


def test_get_current_time(environment: Environment):
    """Test get_current_time returns the expected constant."""
    call = ToolCall(
        id="1",
        name="get_current_time",
        arguments={},
    )
    response = environment.get_response(call)
    assert not response.error
    assert response.content == CURRENT_TIME


def test_get_cancellation_policy_rules(environment: Environment):
    """Test get_cancellation_policy_rules returns all policy types."""
    call = ToolCall(
        id="1",
        name="get_cancellation_policy_rules",
        arguments={},
    )
    response = environment.get_response(call)
    assert not response.error
    content = str(response.content)
    for policy in ("flexible", "moderate", "firm", "strict"):
        assert policy in content


def test_get_user_details(environment: Environment):
    """Test retrieving user details."""
    call = ToolCall(
        id="1",
        name="get_user_details",
        arguments={"user_id": "guest_001"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "guest_001" in str(response.content)


def test_get_user_details_not_found(environment: Environment):
    """Test error when user not found."""
    call = ToolCall(
        id="1",
        name="get_user_details",
        arguments={"user_id": "nonexistent"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_get_reservation_details(environment: Environment):
    """Test retrieving reservation details."""
    call = ToolCall(
        id="1",
        name="get_reservation_details",
        arguments={"reservation_id": "RES001"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "RES001" in str(response.content)


def test_get_reservation_details_not_found(environment: Environment):
    """Test error when reservation not found."""
    call = ToolCall(
        id="1",
        name="get_reservation_details",
        arguments={"reservation_id": "nonexistent"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_get_listing_details(environment: Environment):
    """Test retrieving listing details."""
    call = ToolCall(
        id="1",
        name="get_listing_details",
        arguments={"listing_id": "LST001"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "LST001" in str(response.content)


def test_get_listing_details_not_found(environment: Environment):
    """Test error when listing not found."""
    call = ToolCall(
        id="1",
        name="get_listing_details",
        arguments={"listing_id": "nonexistent"},
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


# === cancel_reservation Tests: Policy Branch Tests ===


def test_cancel_flexible_full_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Flexible policy, >=1 day before check-in -> full refund."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_FLEX_FULL",
            "expected_refund_amount": 300.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "success" in str(response.content).lower()
    assert "300" in str(response.content)

    res = vacation_rental_db.reservations["RES_FLEX_FULL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0
    assert res.cancelled_by == "guest"


def test_cancel_flexible_partial_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Flexible policy, <1 day before check-in -> forfeit 1st night ($200 refund)."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_FLEX_PARTIAL",
            "expected_refund_amount": 200.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "200" in str(response.content)

    res = vacation_rental_db.reservations["RES_FLEX_PARTIAL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 200.0
    assert res.cancelled_by == "guest"


def test_cancel_moderate_full_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Moderate policy, >=5 days before check-in -> full refund."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_MOD_FULL",
            "expected_refund_amount": 300.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "300" in str(response.content)

    res = vacation_rental_db.reservations["RES_MOD_FULL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0
    assert res.cancelled_by == "guest"


def test_cancel_moderate_partial_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Moderate policy, <5 days -> 1st night forfeit + 50% remaining = $100."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_MOD_PARTIAL",
            "expected_refund_amount": 100.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "100" in str(response.content)

    res = vacation_rental_db.reservations["RES_MOD_PARTIAL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 100.0
    assert res.cancelled_by == "guest"


def test_cancel_firm_full_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Firm policy, >=30 days before check-in -> full refund."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_FIRM_FULL",
            "expected_refund_amount": 300.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "300" in str(response.content)

    res = vacation_rental_db.reservations["RES_FIRM_FULL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0
    assert res.cancelled_by == "guest"


def test_cancel_firm_50_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Firm policy, 7-29 days before check-in -> 50% refund ($150)."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_FIRM_50",
            "expected_refund_amount": 150.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "150" in str(response.content)

    res = vacation_rental_db.reservations["RES_FIRM_50"]
    assert res.status == "cancelled"
    assert res.refund_amount == 150.0
    assert res.cancelled_by == "guest"


def test_cancel_firm_zero_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Firm policy, <7 days before check-in -> no refund."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_FIRM_ZERO",
            "expected_refund_amount": 0.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error

    res = vacation_rental_db.reservations["RES_FIRM_ZERO"]
    assert res.status == "cancelled"
    assert res.refund_amount == 0.0
    assert res.cancelled_by == "guest"


def test_cancel_strict_50_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Strict policy, >=7 days before check-in -> 50% refund ($150)."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_STRICT_50",
            "expected_refund_amount": 150.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "150" in str(response.content)

    res = vacation_rental_db.reservations["RES_STRICT_50"]
    assert res.status == "cancelled"
    assert res.refund_amount == 150.0
    assert res.cancelled_by == "guest"


def test_cancel_strict_zero_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Strict policy, <7 days before check-in -> no refund."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_STRICT_ZERO",
            "expected_refund_amount": 0.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error

    res = vacation_rental_db.reservations["RES_STRICT_ZERO"]
    assert res.status == "cancelled"
    assert res.refund_amount == 0.0
    assert res.cancelled_by == "guest"


def test_cancel_host_cancellation_full_refund(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Host cancellation always results in full refund regardless of policy."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_HOST",
            "expected_refund_amount": 300.0,
            "cancelled_by": "host",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "300" in str(response.content)

    res = vacation_rental_db.reservations["RES_HOST"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0
    assert res.cancelled_by == "host"


# === cancel_reservation Tests: Grace Period ===


def test_cancel_grace_period_override(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """Within 24h of booking + 7+ days before check-in -> full refund (grace period)."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_GRACE",
            "expected_refund_amount": 300.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "300" in str(response.content)

    res = vacation_rental_db.reservations["RES_GRACE"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0
    assert res.cancelled_by == "guest"


# === cancel_reservation Tests: Error Paths ===


def test_cancel_not_found(environment: Environment):
    """Error when reservation not found."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "NOPE",
            "expected_refund_amount": 0.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_cancel_not_confirmed(environment: Environment):
    """Error when reservation is already cancelled."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_CANCELLED",
            "expected_refund_amount": 300.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert response.error or "cannot be cancelled" in str(response.content).lower()


def test_cancel_check_in_passed(environment: Environment):
    """Error when check-in date has already passed."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_PAST",
            "expected_refund_amount": 0.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert response.error or "has already passed" in str(response.content).lower()


def test_cancel_invalid_cancelled_by(environment: Environment):
    """Error when cancelled_by is not 'guest' or 'host'."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES001",
            "expected_refund_amount": 300.0,
            "cancelled_by": "system",
        },
    )
    response = environment.get_response(call)
    assert response.error or "must be" in str(response.content).lower()


def test_cancel_refund_mismatch(environment: Environment):
    """Error when expected_refund_amount does not match calculated amount."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES001",
            "expected_refund_amount": 999.0,
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert response.error or "mismatch" in str(response.content).lower()


# === cancel_reservation Tests: Float Coercion ===


def test_cancel_float_coercion(
    environment: Environment, vacation_rental_db: VacationRentalDB
):
    """String expected_refund_amount is coerced to float without error."""
    call = ToolCall(
        id="1",
        name="cancel_reservation",
        arguments={
            "reservation_id": "RES_MOD_FULL",
            "expected_refund_amount": "300.0",
            "cancelled_by": "guest",
        },
    )
    response = environment.get_response(call)
    assert not response.error

    res = vacation_rental_db.reservations["RES_MOD_FULL"]
    assert res.status == "cancelled"
    assert res.refund_amount == 300.0


# === process_refund Tests ===


def test_process_refund_success(environment: Environment):
    """Successfully process refund on a cancelled reservation."""
    call = ToolCall(
        id="1",
        name="process_refund",
        arguments={
            "reservation_id": "RES_CANCELLED",
            "payment_method_id": "card_001",
            "amount": 300.0,
        },
    )
    response = environment.get_response(call)
    assert not response.error
    assert "success" in str(response.content).lower()


def test_process_refund_not_found(environment: Environment):
    """Error when reservation not found for refund."""
    call = ToolCall(
        id="1",
        name="process_refund",
        arguments={
            "reservation_id": "NOPE",
            "payment_method_id": "card_001",
            "amount": 100.0,
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()


def test_process_refund_not_cancelled(environment: Environment):
    """Error when reservation is not in cancelled status."""
    call = ToolCall(
        id="1",
        name="process_refund",
        arguments={
            "reservation_id": "RES001",
            "payment_method_id": "card_001",
            "amount": 100.0,
        },
    )
    response = environment.get_response(call)
    assert response.error or "must be cancelled" in str(response.content).lower()


def test_process_refund_exceeds_paid(environment: Environment):
    """Error when refund amount exceeds amount paid."""
    call = ToolCall(
        id="1",
        name="process_refund",
        arguments={
            "reservation_id": "RES_CANCELLED",
            "payment_method_id": "card_001",
            "amount": 999.0,
        },
    )
    response = environment.get_response(call)
    assert response.error or "exceeds" in str(response.content).lower()


def test_process_refund_invalid_payment(environment: Environment):
    """Error when payment method not found for user."""
    call = ToolCall(
        id="1",
        name="process_refund",
        arguments={
            "reservation_id": "RES_CANCELLED",
            "payment_method_id": "bad_card",
            "amount": 100.0,
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found for user" in str(response.content).lower()


# === validate_issue_evidence Gap Tests ===


def test_validate_evidence_no_submission(environment: Environment):
    """Evidence not submitted -> pending status."""
    call = ToolCall(
        id="1",
        name="validate_issue_evidence",
        arguments={"issue_id": "ISS_NO_EVIDENCE"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "pending" in str(response.content)


def test_validate_evidence_invalidated(environment: Environment):
    """Evidence invalidated -> invalidated status."""
    call = ToolCall(
        id="1",
        name="validate_issue_evidence",
        arguments={"issue_id": "ISS_INVALIDATED"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "invalidated" in str(response.content)
    assert "standard policy" in str(response.content).lower()


def test_validate_evidence_inconclusive(environment: Environment):
    """Evidence inconclusive -> inconclusive status."""
    call = ToolCall(
        id="1",
        name="validate_issue_evidence",
        arguments={"issue_id": "ISS_INCONCLUSIVE"},
    )
    response = environment.get_response(call)
    assert not response.error
    assert "inconclusive" in str(response.content)


# === calculate Error Tests ===


def test_calculate_invalid_expression(environment: Environment):
    """Test error on invalid expression with disallowed characters."""
    call = ToolCall(
        id="1",
        name="calculate",
        arguments={"expression": "import os"},
    )
    response = environment.get_response(call)
    assert response.error or "invalid" in str(response.content).lower()


def test_calculate_division_by_zero(environment: Environment):
    """Test error on division by zero."""
    call = ToolCall(
        id="1",
        name="calculate",
        arguments={"expression": "100 / 0"},
    )
    response = environment.get_response(call)
    assert response.error or "error" in str(response.content).lower()


# === submit_issue_report Gap Tests ===


def test_submit_issue_invalid_severity(environment: Environment):
    """Test error on invalid severity value."""
    call = ToolCall(
        id="1",
        name="submit_issue_report",
        arguments={
            "reservation_id": "RES001",
            "issue_type": "cleanliness",
            "description": "Test",
            "severity": "extreme",
        },
    )
    response = environment.get_response(call)
    assert response.error or "invalid" in str(response.content).lower()


def test_submit_issue_reservation_not_found(environment: Environment):
    """Test error when reservation not found for issue submission."""
    call = ToolCall(
        id="1",
        name="submit_issue_report",
        arguments={
            "reservation_id": "NOPE",
            "issue_type": "cleanliness",
            "description": "Test",
            "severity": "minor",
        },
    )
    response = environment.get_response(call)
    assert response.error or "not found" in str(response.content).lower()
