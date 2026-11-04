"""Regional execution uses one catalog, price scale, policy and scenario clock."""

from datetime import date

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.airline.data_model import AirportCode
from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.airline.region import AirlineRegionalSettings
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_environment
from tau2.domains.telecom.environment import get_environment as telecom_environment
from tau2.domains.telecom.region import TelecomRegionalSettings
from tau2.domains.telecom.user_data_model import PaymentRequest
from tau2.environment.regional import DomainRegionalSettings
from tests.test_domains.test_airline import test_tools_airline as airline_fixtures

airline_db = airline_fixtures.airline_db
reservation_call = airline_fixtures.reservation_call


@pytest.fixture
def region():
    return DomainRegionalSettings(
        country_code="BR",
        currency_code="BRL",
        currency_scale="5",
        time_zone="America/Sao_Paulo",
    )


@pytest.fixture
def airline_region(region):
    return AirlineRegionalSettings(
        **region.model_dump(),
        current_datetime="2024-05-15T15:00:00",
        airports=[
            AirportCode(iata=code, city=city)
            for code, city in [
                ("CGH", "São Paulo"),
                ("GRU", "Guarulhos"),
                ("GIG", "Rio de Janeiro"),
            ]
        ],
    )


def regional_airline_db(db):
    """A fixture whose flight graph and prices are already localized."""
    db = db.model_copy(deep=True)
    airports = {"PHL": "CGH", "LGA": "GRU", "LAX": "GIG"}
    for flight in db.flights.values():
        flight.origin = airports[flight.origin]
        flight.destination = airports[flight.destination]
        for instance in flight.dates.values():
            if hasattr(instance, "prices"):
                instance.prices = {
                    cabin: price * 5 for cabin, price in instance.prices.items()
                }
    for reservation in db.reservations.values():
        reservation.origin = airports[reservation.origin]
        reservation.destination = airports[reservation.destination]
        for flight in reservation.flights:
            flight.origin = airports[flight.origin]
            flight.destination = airports[flight.destination]
            flight.price *= 5
    return db


def test_airline_booking_scales_fixed_fees_once_and_uses_local_catalog(
    airline_db, airline_region, reservation_call
):
    environment = airline_environment(
        db=regional_airline_db(airline_db),
        policy="Reviewed regional policy.",
        regional_settings=airline_region,
    )
    reservation_call.arguments.update(
        origin="CGH", destination="GRU", insurance="yes", nonfree_baggages=1
    )
    # Localized fare 610 + local insurance 150 + local bag fee 250.
    reservation_call.arguments["payment_methods"][0].amount = 1010
    response = environment.get_response(reservation_call)
    assert not response.error
    reservation = environment.tools.db.reservations["HATHAT"]
    assert reservation.payment_history[0].amount == 1010
    assert reservation.flights[0].price == 610
    assert reservation.created_at == "2024-05-15T15:00:00"
    assert environment.get_policy() == "Reviewed regional policy."
    assert [(a.iata, a.city) for a in environment.tools.list_all_airports()] == [
        ("CGH", "São Paulo"),
        ("GRU", "Guarulhos"),
        ("GIG", "Rio de Janeiro"),
    ]
    catalog = environment.tools.list_all_airports()
    catalog[0].city = "Changed caller copy"
    assert environment.tools.list_all_airports()[0].city == "São Paulo"
    assert airline_environment(db=airline_db).tools.list_all_airports()[0].iata == "SFO"


def test_airline_baggage_update_uses_same_fee_rule(airline_db, airline_region):
    environment = airline_environment(
        db=regional_airline_db(airline_db), regional_settings=airline_region
    )
    response = environment.get_response(
        ToolCall(
            id="bags",
            name="update_reservation_baggages",
            arguments={
                "reservation_id": "4WQ150",
                "total_baggages": 7,
                "nonfree_baggages": 2,
                "payment_id": "credit_card_4421486",
            },
        )
    )
    assert not response.error
    assert environment.tools.db.reservations["4WQ150"].payment_history[-1].amount == 500


def test_regional_airports_must_cover_the_loaded_routes(airline_db, airline_region):
    with pytest.raises(ValueError, match="flight graph"):
        airline_environment(db=airline_db, regional_settings=airline_region)
    with pytest.raises(ValueError, match="distinct"):
        AirlineRegionalSettings(
            **{
                **airline_region.model_dump(),
                "airports": [airline_region.airports[0]] * 2,
            }
        )


def test_telecom_billing_and_caller_outputs_share_currency_and_clock(region):
    settings = TelecomRegionalSettings(
        **region.model_dump(), current_datetime="2025-02-26T12:08:00"
    )
    environment = telecom_environment(
        regional_settings=settings, policy="Regional telecom policy."
    )
    customer = environment.tools.get_customer_by_id("C1001")
    line = environment.tools._get_line_by_id("L1001")
    plan = environment.tools._get_plan_by_id(line.plan_id)
    plan.data_refueling_price_per_gb = (
        25  # Already localized; must not multiply by 5 again.
    )
    result = environment.tools.refuel_data(customer.customer_id, line.line_id, 2)
    assert result["charge"] == 50
    assert result["message"].endswith("BRL 50.00")
    item = next(
        item
        for bill in environment.tools.db.bills
        for item in bill.line_items
        if item.description.startswith("Data refueling:")
    )
    assert item.amount == 50
    assert "BRL 25.00/GB" in item.description
    assert item.date == date(2025, 2, 26)
    result = environment.tools.suspend_line("C1001", "L1001", "Customer requested")
    assert (
        result["message"]
        == "Line suspended successfully. BRL 25.00/month holding fee will apply."
    )
    assert result["line"].suspension_start_date == date(2025, 2, 26)
    environment.user_tools.db.surroundings.payment_request = PaymentRequest(
        bill_id="B1001", amount_due=50
    )
    assert "50.0 BRL" in environment.user_tools.check_payment_request()
    assert "50.0 BRL" in environment.user_tools.make_payment()
    assert environment.user_tools.db.surroundings.payment_request.paid
    assert environment.get_policy() == "Regional telecom policy."
    canonical = telecom_environment()
    assert canonical.tools.suspend_line("C1001", "L1001", "Requested")["message"] == (
        "Line suspended successfully. $5/month holding fee will apply."
    )
    canonical.user_tools.db.surroundings.payment_request = PaymentRequest(
        bill_id="B1001", amount_due=50
    )
    assert "50.0 USD" in canonical.user_tools.check_payment_request()


def test_regional_tool_descriptions_do_not_mutate_canonical_schemas(
    region, airline_db, airline_region
):
    database = RetailDB(users={}, orders={}, products={})
    canonical = retail_environment(db=database)
    before = canonical.tools.get_tools()["modify_user_address"].openai_schema
    localized = retail_environment(
        db=database.model_copy(deep=True),
        regional_settings=region,
        policy="Regional retail policy.",
    )
    schema = localized.tools.get_tools()["modify_user_address"].openai_schema
    assert "USA" not in str(schema)
    assert "BRL" in schema["function"]["description"]
    assert "America/Sao_Paulo" in schema["function"]["description"]
    assert (
        schema["function"]["parameters"]["required"]
        == before["function"]["parameters"]["required"]
    )
    assert canonical.tools.get_tools()["modify_user_address"].openai_schema == before
    assert localized.get_policy() == "Regional retail policy."
    airline = airline_environment(
        db=regional_airline_db(airline_db), regional_settings=airline_region
    )
    parameters = airline.tools.get_tools()["book_reservation"].openai_schema[
        "function"
    ]["parameters"]
    assert "list_all_airports" in parameters["properties"]["origin"]["description"]
    assert "SFO" not in parameters["properties"]["origin"]["description"]
    assert (
        parameters["$defs"]["Payment"]["properties"]["amount"]["description"]
        == "Payment amount in BRL"
    )
    canonical_payment = (
        airline_environment(db=airline_db)
        .tools.get_tools()["book_reservation"]
        .openai_schema["function"]["parameters"]
    )
    assert (
        canonical_payment["$defs"]["Payment"]["properties"]["amount"]["description"]
        == "Payment amount in dollars"
    )
    regional_returns = airline.tools.get_tools()[
        "get_reservation_details"
    ].returns.model_json_schema()
    assert "Flight price in BRL." in str(regional_returns)
    assert "Flight price in dollars." not in str(regional_returns)


def test_telecom_generic_lookup_accepts_name_derived_customer_ids(region):
    settings = TelecomRegionalSettings(
        **region.model_dump(), current_datetime="2025-02-25T12:08:00"
    )
    environment = telecom_environment(regional_settings=settings)
    customer = environment.tools.get_customer_by_id("C1001")
    customer.customer_id = "marcos_silva_1001"
    response = environment.get_response(
        ToolCall(
            id="lookup",
            name="get_details_by_id",
            arguments={"id": customer.customer_id},
        )
    )
    assert not response.error
    assert environment.tools.get_details_by_id(customer.customer_id) is customer
    with pytest.raises(ValueError, match="Unknown ID"):
        environment.tools.get_details_by_id("someone_else_1001")


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_same_constructor_kwargs_create_independent_databases(domain, airline_db):
    factory = {
        "airline": airline_environment,
        "retail": retail_environment,
        "telecom": telecom_environment,
    }[domain]
    original = factory(db=airline_db) if domain == "airline" else factory()
    kwargs = {"db": original.tools.db}
    if domain == "telecom":
        kwargs["user_db"] = original.user_tools.db
    before = original.tools.get_db_hash()
    first, second = factory(**kwargs), factory(**kwargs)
    if domain == "airline":
        first.tools.db.users["mia_li_3668"].name.first_name = "Changed"
    elif domain == "retail":
        next(iter(first.tools.db.users.values())).name.first_name = "Changed"
    else:
        first.tools.db.customers[0].full_name = "Changed"
        first.user_tools.db.surroundings.name = "Changed caller DB"
        assert second.user_tools.db.surroundings.name != "Changed caller DB"
        assert original.user_tools.db.surroundings.name != "Changed caller DB"
        assert first.user_tools.db is not first.tools.db
    assert first.tools.get_db_hash() != before
    assert second.tools.get_db_hash() == before
    assert original.tools.get_db_hash() == before


def test_evaluator_missing_write_does_not_pass_through_shared_db(airline_db):
    from tau2.data_model.tasks import Action, EvaluationCriteria, Task, UserScenario
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    before = airline_db.model_dump()
    task = Task(
        id="missing-cancel",
        user_scenario=UserScenario(instructions="Cancel your reservation."),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="cancel",
                    name="cancel_reservation",
                    arguments={"reservation_id": "4WQ150"},
                )
            ],
            reward_basis=["DB"],
        ),
    )
    reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=airline_environment,
        task=task,
        full_trajectory=[],
        env_kwargs={"db": airline_db},
    )
    assert reward.reward == 0
    assert not reward.db_check.db_match
    assert airline_db.model_dump() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"currency_scale": "NaN"},
        {"currency_scale": "0"},
        {"time_zone": "Unknown/Place"},
    ],
)
def test_invalid_regional_units_fail_before_environment_creation(region, changes):
    with pytest.raises(ValueError):
        DomainRegionalSettings(**{**region.model_dump(), **changes})
