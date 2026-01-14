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
                payment_methods={},
                reservations=["RES001"],
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
        },
        host_profiles={
            "host_001": HostProfile(
                host_user_id="host_001",
                philosophy=HostPhilosophy(primary_focus="reviews", risk_tolerance="high"),
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


# === Tool Tests ===


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
            "reservation_id": "RES001",
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
            "reservation_id": "RES001",
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
