import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.hospitality.data_model import HospitalityDB, get_db
from tau2.domains.hospitality.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.environment.environment import Environment


@pytest.fixture
def env() -> Environment:
    """Fresh environment loaded from the shipped db.json."""
    return get_environment()


def test_db_loads():
    db = get_db()
    assert isinstance(db, HospitalityDB)
    stats = db.get_statistics()
    assert stats["num_room_types"] == 5
    assert stats["num_rate_plans"] == 3
    assert stats["num_stay_packages"] == 3
    assert stats["num_guests"] == 8
    assert stats["num_reservations"] == 10
    assert stats["num_knowledge_base_articles"] == 16


def test_seeded_totals_consistent():
    """Every seeded reservation total must equal nights * nightly + extras."""
    db = get_db()
    from datetime import date

    for reservation in db.reservations.values():
        nights = (
            date.fromisoformat(reservation.check_out)
            - date.fromisoformat(reservation.check_in)
        ).days
        extras = sum(extra.amount for extra in reservation.extras)
        assert reservation.total_amount == nights * reservation.nightly_rate + extras, (
            reservation.reservation_id
        )
        room_type = db.room_types[reservation.room_type_id]
        if reservation.rate_plan_id is not None:
            assert reservation.nightly_rate == room_type.rates[reservation.rate_plan_id]
        assert reservation.num_guests <= room_type.capacity


def test_get_room_offers_capacity_filter(env: Environment):
    # 3 guests only fit the Family Suite.
    offers = env.tools.get_room_offers("2025-06-24", "2025-06-26", 3)
    assert [offer["room_type_id"] for offer in offers] == ["RT_FAMILY"]
    saver = next(
        rate for rate in offers[0]["rates"] if rate["rate_plan_id"] == "RP_SAVER"
    )
    assert saver["total"] == 442


def test_get_room_offers_availability_filter(env: Environment):
    # Both Family Suites are taken June 20-22 (HV-1006, HV-1007).
    offers = env.tools.get_room_offers("2025-06-20", "2025-06-22", 2)
    room_type_ids = [offer["room_type_id"] for offer in offers]
    assert "RT_FAMILY" not in room_type_ids
    assert "RT_STANDARD" in room_type_ids
    standard = next(o for o in offers if o["room_type_id"] == "RT_STANDARD")
    flex = next(r for r in standard["rates"] if r["rate_plan_id"] == "RP_FLEX")
    assert flex["total"] == 280


def test_get_room_offers_invalid_dates(env: Environment):
    with pytest.raises(ValueError):
        env.tools.get_room_offers("2025-06-20", "2025-06-20", 2)
    with pytest.raises(ValueError):
        env.tools.get_room_offers("2025-06-01", "2025-06-03", 2)  # in the past
    with pytest.raises(ValueError):
        env.tools.get_room_offers("June 20", "June 22", 2)


def test_get_stay_package_offers(env: Environment):
    offers = env.tools.get_stay_package_offers("2025-06-21", "2025-06-23", 2)
    by_id = {offer["package_id"]: offer for offer in offers}
    assert by_id["PKG_ROMANCE"]["total"] == 480
    assert by_id["PKG_SPA"]["total"] == 420
    # One night is below every package's minimum stay.
    offers = env.tools.get_stay_package_offers("2025-06-21", "2025-06-22", 2)
    assert offers == []


def test_create_reservation_regular(env: Environment):
    reservation = env.tools.create_reservation(
        guest_id="G001",
        check_in="2025-06-20",
        check_out="2025-06-22",
        num_guests=2,
        room_type_id="RT_STANDARD",
        rate_plan_id="RP_FLEX",
    )
    assert reservation.reservation_id == "HV-1011"
    assert reservation.total_amount == 280
    assert reservation.status == "confirmed"


def test_create_reservation_errors(env: Environment):
    with pytest.raises(ValueError):  # unknown guest
        env.tools.create_reservation(
            guest_id="G999",
            check_in="2025-06-20",
            check_out="2025-06-22",
            num_guests=2,
            room_type_id="RT_STANDARD",
            rate_plan_id="RP_FLEX",
        )
    with pytest.raises(ValueError):  # over capacity
        env.tools.create_reservation(
            guest_id="G001",
            check_in="2025-06-20",
            check_out="2025-06-22",
            num_guests=3,
            room_type_id="RT_STANDARD",
            rate_plan_id="RP_FLEX",
        )
    with pytest.raises(ValueError):  # family suites sold out on these dates
        env.tools.create_reservation(
            guest_id="G001",
            check_in="2025-06-20",
            check_out="2025-06-22",
            num_guests=4,
            room_type_id="RT_FAMILY",
            rate_plan_id="RP_FLEX",
        )
    with pytest.raises(ValueError):  # both package and room type
        env.tools.create_reservation(
            guest_id="G001",
            check_in="2025-06-20",
            check_out="2025-06-22",
            num_guests=2,
            room_type_id="RT_STANDARD",
            rate_plan_id="RP_FLEX",
            package_id="PKG_ROMANCE",
        )
    with pytest.raises(ValueError):  # missing rate plan
        env.tools.create_reservation(
            guest_id="G001",
            check_in="2025-06-20",
            check_out="2025-06-22",
            num_guests=2,
            room_type_id="RT_STANDARD",
        )


def test_create_reservation_package(env: Environment):
    reservation = env.tools.create_reservation(
        guest_id="G008",
        check_in="2025-06-21",
        check_out="2025-06-23",
        num_guests=2,
        package_id="PKG_ROMANCE",
    )
    assert reservation.total_amount == 480
    assert reservation.rate_plan_id is None
    assert reservation.room_type_id == "RT_DELUXE"
    with pytest.raises(ValueError):  # below minimum stay
        env.tools.create_reservation(
            guest_id="G008",
            check_in="2025-06-24",
            check_out="2025-06-25",
            num_guests=2,
            package_id="PKG_ROMANCE",
        )


def test_create_guest_profile(env: Environment):
    guest = env.tools.create_guest_profile(
        name="Lena Hoffmann",
        email="lena.hoffmann@mailbox.org",
        phone="+49 152 555 8090",
    )
    assert guest.guest_id == "G009"
    with pytest.raises(ValueError):  # duplicate email
        env.tools.create_guest_profile(
            name="Lena H.", email="LENA.HOFFMANN@mailbox.org", phone="+49 0"
        )


def test_find_and_update_guest(env: Environment):
    guest = env.tools.find_guest("ANNA.KELLER@webmail.de")
    assert guest.guest_id == "G002"
    with pytest.raises(ValueError):
        env.tools.find_guest("nobody@nowhere.org")
    updated = env.tools.update_guest_profile(
        guest_id="G002",
        email="anna.keller@neumail.de",
        phone="+49 160 555 7788",
    )
    assert updated.email == "anna.keller@neumail.de"
    assert updated.phone == "+49 160 555 7788"
    assert updated.name == "Anna Keller"
    with pytest.raises(ValueError):  # nothing to update
        env.tools.update_guest_profile(guest_id="G002")


def test_cancel_full_refund(env: Environment):
    reservation = env.tools.cancel_reservation("HV-1001")
    assert reservation.status == "cancelled"
    assert reservation.refund_amount == 420


def test_cancel_late_fee(env: Environment):
    # HV-1003 checks in tomorrow: first night charged.
    reservation = env.tools.cancel_reservation("HV-1003")
    assert reservation.refund_amount == 140


def test_cancel_non_refundable(env: Environment):
    reservation = env.tools.cancel_reservation("HV-1002")
    assert reservation.refund_amount == 0


def test_cancel_errors(env: Environment):
    with pytest.raises(ValueError):
        env.tools.cancel_reservation("HV-9999")
    env.tools.cancel_reservation("HV-1001")
    with pytest.raises(ValueError):  # already cancelled
        env.tools.cancel_reservation("HV-1001")


def test_modify_dates_flex(env: Environment):
    reservation = env.tools.modify_reservation(
        reservation_id="HV-1004",
        modification_type="change_dates",
        new_check_in="2025-06-26",
        new_check_out="2025-06-29",
    )
    assert reservation.check_in == "2025-06-26"
    assert reservation.total_amount == 570  # same number of nights


def test_modify_dates_refused_on_saver(env: Environment):
    with pytest.raises(ValueError):
        env.tools.modify_reservation(
            reservation_id="HV-1002",
            modification_type="change_dates",
            new_check_in="2025-06-23",
            new_check_out="2025-06-25",
        )


def test_modify_upgrade_executive(env: Environment):
    reservation = env.tools.modify_reservation(
        reservation_id="HV-1004",
        modification_type="change_room_type",
        new_room_type_id="RT_EXEC",
    )
    assert reservation.nightly_rate == 320
    assert reservation.total_amount == 960


def test_modify_room_unavailable(env: Environment):
    # The single Executive Suite is taken June 18-20 (HV-1008).
    blocker = env.tools.create_reservation(
        guest_id="G001",
        check_in="2025-06-18",
        check_out="2025-06-20",
        num_guests=2,
        room_type_id="RT_STANDARD",
        rate_plan_id="RP_FLEX",
    )
    with pytest.raises(ValueError):
        env.tools.modify_reservation(
            reservation_id=blocker.reservation_id,
            modification_type="change_room_type",
            new_room_type_id="RT_EXEC",
        )


def test_modify_num_guests(env: Environment):
    reservation = env.tools.modify_reservation(
        reservation_id="HV-1001",
        modification_type="change_num_guests",
        new_num_guests=1,
    )
    assert reservation.num_guests == 1
    assert reservation.total_amount == 420  # per-room pricing, total unchanged
    with pytest.raises(ValueError):  # over capacity
        env.tools.modify_reservation(
            reservation_id="HV-1001",
            modification_type="change_num_guests",
            new_num_guests=3,
        )


def test_book_extra_services(env: Environment):
    env.tools.book_extra_services("HV-1005", "SVC_PARKING", 2)
    reservation = env.tools.book_extra_services("HV-1005", "SVC_TRANSFER", 1)
    assert reservation.total_amount == 455
    assert len(reservation.extras) == 2
    with pytest.raises(ValueError):  # unknown service
        env.tools.book_extra_services("HV-1005", "SVC_NOPE", 1)
    with pytest.raises(ValueError):  # cancelled reservation
        env.tools.book_extra_services("HV-1010", "SVC_PARKING", 1)


def test_book_extra_services_order_insensitive():
    """Extras lines merge per service and stay sorted by service ID, so the
    final DB state is identical regardless of booking order."""
    env_a = get_environment()
    env_a.tools.book_extra_services("HV-1006", "SVC_BREAKFAST", 12)
    env_a.tools.book_extra_services("HV-1006", "SVC_SPA", 2)
    env_a.tools.book_extra_services("HV-1006", "SVC_PARKING", 3)
    env_b = get_environment()
    env_b.tools.book_extra_services("HV-1006", "SVC_PARKING", 3)
    env_b.tools.book_extra_services("HV-1006", "SVC_SPA", 2)
    env_b.tools.book_extra_services("HV-1006", "SVC_BREAKFAST", 12)
    res_a = env_a.tools.db.reservations["HV-1006"]
    res_b = env_b.tools.db.reservations["HV-1006"]
    assert res_a == res_b
    assert res_a.total_amount == 1158
    # Booking the same service again merges into the existing line.
    env_a.tools.book_extra_services("HV-1006", "SVC_PARKING", 1)
    assert len(res_a.extras) == 3
    parking = next(e for e in res_a.extras if e.service_id == "SVC_PARKING")
    assert parking.quantity == 4
    assert parking.amount == 96


def test_knowledge_base(env: Environment):
    pets = env.tools.knowledge_base("can I bring my dog")
    assert "32" in pets
    parking = env.tools.knowledge_base("parking garage height")
    assert "1.85" in parking
    breakfast = env.tools.knowledge_base("breakfast time")
    assert "6:30" in breakfast
    assert env.tools.knowledge_base("zzzz qqqq") == "No matching articles found."


def test_user_check_confirmation_email(env: Environment):
    confirmations = env.user_tools.check_booking_confirmation_email(
        "maya.patel@inbox.com"
    )
    assert [c["confirmation_number"] for c in confirmations] == ["HV-1005"]
    with pytest.raises(ValueError):
        env.user_tools.check_booking_confirmation_email("nobody@nowhere.org")


def test_user_online_checkin(env: Environment):
    result = env.user_tools.submit_online_checkin("HV-1005", "Patel", "16:00")
    assert "completed" in result
    assert env.tools.db.reservations["HV-1005"].online_checkin_completed
    with pytest.raises(ValueError):  # already done
        env.user_tools.submit_online_checkin("HV-1005", "Patel", "16:00")
    with pytest.raises(ValueError):  # window not open (check-in June 20)
        env.user_tools.submit_online_checkin("HV-1001", "Dvorak", "15:00")
    # A failed attempt must not leave any trace in the DB.
    assert not env.tools.db.reservations["HV-1001"].online_checkin_completed
    with pytest.raises(ValueError):  # wrong name
        env.user_tools.submit_online_checkin("HV-1004", "Patel", "15:00")
    with pytest.raises(ValueError):  # bad time format
        env.user_tools.submit_online_checkin("HV-1004", "Rossi", "afternoon")


def test_user_tools_share_agent_db(env: Environment):
    """User-side writes must be visible to the agent tools (shared DB)."""
    env.user_tools.submit_online_checkin("HV-1005", "Patel", "16:00")
    reservation = env.tools.read_reservation_info("HV-1005")
    assert reservation.online_checkin_completed


def test_get_response_tool_call(env: Environment):
    response = env.get_response(
        ToolCall(
            id="1",
            name="read_reservation_info",
            arguments={"reservation_id": "HV-1001"},
        )
    )
    assert not response.error
    response = env.get_response(
        ToolCall(
            id="2",
            name="read_reservation_info",
            arguments={"reservation_id": "HV-9999"},
        )
    )
    assert response.error


def test_tasks_load_and_split():
    tasks = get_tasks()
    assert len(tasks) == 25
    splits = get_tasks_split()
    assert set(splits["base"]) == {task.id for task in tasks}


def test_all_golden_actions_replay():
    """Every task's expected actions must execute without error on a fresh
    environment. This is what the DB reward check replays, so a failure here
    means the task is broken by construction."""
    for task in get_tasks():
        env = get_environment()
        if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
            continue
        for action in task.evaluation_criteria.actions:
            env.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
