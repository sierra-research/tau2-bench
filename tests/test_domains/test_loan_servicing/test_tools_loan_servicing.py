import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.domains.loan_servicing.data_model import LoanServicingDB
from tau2.domains.loan_servicing.environment import get_environment, get_tasks
from tau2.environment.environment import Environment


def make_db() -> LoanServicingDB:
    no_autopay = {"enabled": False, "method_id": None, "day": None}
    return LoanServicingDB(
        borrowers={
            "BF-10001": {
                "borrower_id": "BF-10001",
                "first_name": "Anne",
                "last_name": "Roy",
                "date_of_birth": "1980-01-02",
                "postal_code": "H2X 1Y4",
                "email": "anne.roy@example.com",
                "phone": "514-555-0101",
                "preferred_language": "fr",
                "authorized_third_parties": ["Paul Roy"],
                "bank_accounts": [
                    {"method_id": "BA-40001", "bank_name": "Bank A", "last4": "1111"},
                    {"method_id": "BA-40002", "bank_name": "Bank B", "last4": "2222"},
                ],
                "loan_ids": ["LN-20001", "LN-20003"],
            },
            "BF-10002": {
                "borrower_id": "BF-10002",
                "first_name": "Luc",
                "last_name": "Côté",
                "date_of_birth": "1975-05-06",
                "postal_code": "K1A 0B1",
                "email": "luc.cote@example.com",
                "phone": "613-555-0102",
                "preferred_language": "en",
                "authorized_third_parties": [],
                "bank_accounts": [
                    {"method_id": "BA-40003", "bank_name": "Bank C", "last4": "3333"}
                ],
                "loan_ids": ["LN-20002", "LN-20003"],
            },
        },
        loans={
            "LN-20001": {
                "loan_id": "LN-20001",
                "borrower_ids": ["BF-10001"],
                "product": "personal",
                "origination_date": "2024-01-10",
                "original_principal": 15000.0,
                "annual_rate": 0.12,
                "term_months": 60,
                "monthly_payment": 333.67,
                "principal_balance": 10000.0,
                "accrued_interest": 20.0,
                "due_day": 20,
                "next_due_date": "2026-04-20",
                "past_due_amount": 0.0,
                "status": "current",
                "autopay": no_autopay,
                "fees": [
                    {
                        "fee_id": "FE-50001",
                        "type": "late_fee",
                        "amount": 35.0,
                        "assessed_date": "2026-02-01",
                        "status": "open",
                        "waived_date": None,
                    },
                    {
                        "fee_id": "FE-50002",
                        "type": "late_fee",
                        "amount": 50.0,
                        "assessed_date": "2026-03-01",
                        "status": "open",
                        "waived_date": None,
                    },
                ],
                "due_date_changes": [],
                "hardship_history": [],
            },
            "LN-20002": {
                "loan_id": "LN-20002",
                "borrower_ids": ["BF-10002"],
                "product": "auto",
                "origination_date": "2023-06-05",
                "original_principal": 9000.0,
                "annual_rate": 0.0999,
                "term_months": 48,
                "monthly_payment": 228.22,
                "principal_balance": 5000.0,
                "accrued_interest": 40.0,
                "due_day": 5,
                "next_due_date": "2026-04-05",
                "past_due_amount": 200.0,
                "status": "past_due_30",
                "autopay": no_autopay,
                "fees": [
                    {
                        "fee_id": "FE-50003",
                        "type": "late_fee",
                        "amount": 25.0,
                        "assessed_date": "2026-03-15",
                        "status": "open",
                        "waived_date": None,
                    }
                ],
                "due_date_changes": [],
                "hardship_history": [
                    {
                        "hardship_id": "HP-70001",
                        "plan": "deferral_1",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    }
                ],
            },
            "LN-20003": {
                "loan_id": "LN-20003",
                "borrower_ids": ["BF-10001", "BF-10002"],
                "product": "auto",
                "origination_date": "2025-01-15",
                "original_principal": 1000.0,
                "annual_rate": 0.0599,
                "term_months": 12,
                "monthly_payment": 86.07,
                "principal_balance": 300.0,
                "accrued_interest": 0.5,
                "due_day": 15,
                "next_due_date": "2026-04-15",
                "past_due_amount": 0.0,
                "status": "current",
                "autopay": {"enabled": True, "method_id": "BA-40001", "day": 12},
                "fees": [],
                "due_date_changes": ["2025-06-01"],
                "hardship_history": [],
            },
        },
        payments={
            "PM-30001": {
                "payment_id": "PM-30001",
                "loan_id": "LN-20001",
                "amount": 333.67,
                "date": "2026-02-20",
                "method_id": "BA-40001",
                "status": "posted",
                "allocation": {"fees": 0.0, "interest": 100.0, "principal": 233.67},
            },
            "PM-30002": {
                "payment_id": "PM-30002",
                "loan_id": "LN-20001",
                "amount": 200.0,
                "date": "2026-03-25",
                "method_id": "BA-40002",
                "status": "scheduled",
                "allocation": None,
            },
        },
        document_requests={},
        transfers={},
    )


@pytest.fixture
def environment() -> Environment:
    return get_environment(db=make_db())


def call(environment: Environment, name: str, **arguments):
    """Call a tool through the environment and return (content, error)."""
    response = environment.get_response(
        ToolCall(id="1", name=name, arguments=arguments)
    )
    try:
        content = json.loads(response.content)
    except json.JSONDecodeError:
        content = response.content
    return content, response.error


def loan(environment: Environment, loan_id: str):
    return environment.tools.db.loans[loan_id]


# ---------- lookups ----------


def test_find_borrower_by_email(environment: Environment):
    assert call(environment, "find_borrower_by_email", email=" Anne.Roy@Example.com")[
        0
    ] == ("BF-10001")
    _, error = call(environment, "find_borrower_by_email", email="x@example.com")
    assert error


@pytest.mark.parametrize("phone", ["613-555-0102", "(613) 555 0102", "16135550102"])
def test_find_borrower_by_phone(environment: Environment, phone: str):
    assert call(environment, "find_borrower_by_phone", phone=phone) == (
        "BF-10002",
        False,
    )


def test_find_borrower_by_name_dob_ignores_case_and_accents(environment: Environment):
    content, error = call(
        environment,
        "find_borrower_by_name_dob",
        first_name="luc",
        last_name="Cote",
        date_of_birth="1975-05-06",
    )
    assert (content, error) == ("BF-10002", False)
    _, error = call(
        environment,
        "find_borrower_by_name_dob",
        first_name="Luc",
        last_name="Côté",
        date_of_birth="1975-05-07",
    )
    assert error


def test_get_details(environment: Environment):
    borrower, error = call(environment, "get_borrower_details", borrower_id="BF-10001")
    assert not error and borrower["loan_ids"] == ["LN-20001", "LN-20003"]
    details, error = call(environment, "get_loan_details", loan_id="LN-20002")
    assert not error and details["status"] == "past_due_30"
    assert call(environment, "get_loan_details", loan_id="LN-99999")[1]
    assert call(environment, "get_borrower_details", borrower_id="BF-99999")[1]


def test_list_payments(environment: Environment):
    payments, error = call(environment, "list_payments", loan_id="LN-20001")
    assert not error
    assert [p["payment_id"] for p in payments] == ["PM-30002", "PM-30001"]
    payments, _ = call(environment, "list_payments", loan_id="LN-20001", limit=1)
    assert [p["payment_id"] for p in payments] == ["PM-30002"]
    assert call(environment, "list_payments", loan_id="LN-20001", limit=0)[1]


# ---------- payoff and calculate ----------


def test_calculate_payoff(environment: Environment):
    quote, error = call(
        environment, "calculate_payoff", loan_id="LN-20001", payoff_date="2026-03-26"
    )
    assert not error
    # 20.00 accrued + 10000 * 0.12 / 365 * 10 days = 52.88
    assert quote == {
        "loan_id": "LN-20001",
        "payoff_date": "2026-03-26",
        "principal": 10000.0,
        "interest": 52.88,
        "fees": 85.0,
        "total": 10137.88,
    }
    quote, _ = call(
        environment, "calculate_payoff", loan_id="LN-20001", payoff_date="2026-03-16"
    )
    assert quote["total"] == 10105.0


@pytest.mark.parametrize("payoff_date", ["2026-03-15", "03/20/2026"])
def test_calculate_payoff_bad_date(environment: Environment, payoff_date: str):
    assert call(
        environment, "calculate_payoff", loan_id="LN-20001", payoff_date=payoff_date
    )[1]


def test_calculate(environment: Environment):
    assert call(environment, "calculate", expression="228.22 * 2") == (456.44, False)
    assert call(environment, "calculate", expression="import os")[1]


# ---------- payments ----------


def test_posted_payment_allocates_fees_interest_principal(environment: Environment):
    payment, error = call(
        environment,
        "make_payment",
        loan_id="LN-20001",
        amount=1000,
        method_id="BA-40002",
        payment_date="2026-03-16",
    )
    assert not error
    assert payment["payment_id"] == "PM-30003"
    assert payment["status"] == "posted"
    assert payment["allocation"] == {"fees": 85.0, "interest": 20.0, "principal": 895.0}
    updated = loan(environment, "LN-20001")
    assert [f.status for f in updated.fees] == ["paid", "paid"]
    assert updated.accrued_interest == 0.0
    assert updated.principal_balance == 9105.0


def test_small_payment_pays_only_fees_it_fully_covers(environment: Environment):
    payment, _ = call(
        environment,
        "make_payment",
        loan_id="LN-20001",
        amount=40,
        method_id="BA-40001",
        payment_date="2026-03-16",
    )
    # 35 pays the first fee; the remaining 5 does not cover the 50 fee.
    assert payment["allocation"] == {"fees": 35.0, "interest": 5.0, "principal": 0.0}
    assert [f.status for f in loan(environment, "LN-20001").fees] == ["paid", "open"]


def test_payment_brings_past_due_loan_current(environment: Environment):
    payment, _ = call(
        environment,
        "make_payment",
        loan_id="LN-20002",
        amount=300,
        method_id="BA-40003",
        payment_date="2026-03-16",
    )
    assert payment["allocation"] == {"fees": 25.0, "interest": 40.0, "principal": 235.0}
    updated = loan(environment, "LN-20002")
    assert updated.past_due_amount == 0.0
    assert updated.status == "current"


def test_partial_past_due_payment_stays_past_due(environment: Environment):
    call(
        environment,
        "make_payment",
        loan_id="LN-20002",
        amount=100,
        method_id="BA-40003",
        payment_date="2026-03-16",
    )
    updated = loan(environment, "LN-20002")
    # 25 fee, 40 interest, 35 principal: past due drops by 75.
    assert updated.past_due_amount == 125.0
    assert updated.status == "past_due_30"


def test_scheduled_payment_changes_nothing_yet(environment: Environment):
    payment, _ = call(
        environment,
        "make_payment",
        loan_id="LN-20001",
        amount=250.5,
        method_id="BA-40001",
        payment_date="2026-04-15",
    )
    assert payment["status"] == "scheduled"
    assert payment["allocation"] is None
    assert loan(environment, "LN-20001").principal_balance == 10000.0


def test_payment_by_co_borrower_method(environment: Environment):
    _, error = call(
        environment,
        "make_payment",
        loan_id="LN-20003",
        amount=50,
        method_id="BA-40003",
        payment_date="2026-03-16",
    )
    assert not error


def test_full_payoff_closes_loan(environment: Environment):
    payment, error = call(
        environment,
        "make_payment",
        loan_id="LN-20003",
        amount=300.5,
        method_id="BA-40001",
        payment_date="2026-03-16",
    )
    assert not error
    updated = loan(environment, "LN-20003")
    assert updated.status == "paid_off"
    assert updated.next_due_date is None
    assert not updated.autopay.enabled


@pytest.mark.parametrize(
    "arguments",
    [
        {"amount": 0},
        {"amount": -5},
        {"amount": 10.005},
        {"amount": 10138},  # more than the payoff amount
        {"method_id": "BA-40003"},  # not a borrower on the loan
        {"payment_date": "2026-03-15"},
        {"payment_date": "tomorrow"},
        {"loan_id": "LN-99999"},
    ],
)
def test_make_payment_errors(environment: Environment, arguments: dict):
    base = {
        "loan_id": "LN-20001",
        "amount": 100,
        "method_id": "BA-40001",
        "payment_date": "2026-03-16",
    }
    base.update(arguments)
    _, error = call(environment, "make_payment", **base)
    assert error
    assert len(environment.tools.db.payments) == 2


def test_cancel_scheduled_payment(environment: Environment):
    payment, error = call(
        environment, "cancel_scheduled_payment", payment_id="PM-30002"
    )
    assert not error and payment["status"] == "cancelled"
    assert call(environment, "cancel_scheduled_payment", payment_id="PM-30002")[1]
    assert call(environment, "cancel_scheduled_payment", payment_id="PM-30001")[1]
    assert call(environment, "cancel_scheduled_payment", payment_id="PM-39999")[1]


# ---------- autopay, due date, fees, hardship ----------


def test_enable_and_disable_autopay(environment: Environment):
    content, error = call(
        environment,
        "enable_autopay",
        loan_id="LN-20001",
        method_id="BA-40002",
        day=18,
    )
    assert not error
    assert content == {"enabled": True, "method_id": "BA-40002", "day": 18}
    content, error = call(environment, "disable_autopay", loan_id="LN-20003")
    assert not error
    assert content == {"enabled": False, "method_id": None, "day": None}
    assert call(environment, "disable_autopay", loan_id="LN-20003")[1]


@pytest.mark.parametrize(
    "arguments",
    [{"day": 29}, {"day": 0}, {"method_id": "BA-40003"}, {"loan_id": "LN-99999"}],
)
def test_enable_autopay_errors(environment: Environment, arguments: dict):
    base = {"loan_id": "LN-20001", "method_id": "BA-40001", "day": 15}
    base.update(arguments)
    assert call(environment, "enable_autopay", **base)[1]
    assert not loan(environment, "LN-20001").autopay.enabled


def test_change_due_date(environment: Environment):
    content, error = call(environment, "change_due_date", loan_id="LN-20001", new_day=3)
    assert not error
    assert content["due_day"] == 3
    assert content["next_due_date"] == "2026-04-03"
    assert content["due_date_changes"] == ["2026-03-16"]


@pytest.mark.parametrize("new_day", [20, 29, 0])
def test_change_due_date_errors(environment: Environment, new_day: int):
    assert call(environment, "change_due_date", loan_id="LN-20001", new_day=new_day)[1]


def test_waive_late_fee(environment: Environment):
    fee, error = call(
        environment, "waive_late_fee", loan_id="LN-20001", fee_id="FE-50002"
    )
    assert not error
    assert fee["status"] == "waived" and fee["waived_date"] == "2026-03-16"
    assert call(environment, "waive_late_fee", loan_id="LN-20001", fee_id="FE-50002")[1]
    assert call(environment, "waive_late_fee", loan_id="LN-20001", fee_id="FE-50003")[1]


def test_enroll_deferral(environment: Environment):
    content, error = call(
        environment, "enroll_hardship_plan", loan_id="LN-20002", plan="deferral_1"
    )
    assert not error
    assert content["status"] == "in_hardship"
    assert content["past_due_amount"] == 0.0
    assert content["next_due_date"] == "2026-05-05"
    assert content["hardship_history"][-1] == {
        "hardship_id": "HP-70002",
        "plan": "deferral_1",
        "start_date": "2026-03-16",
        "end_date": "2026-04-15",
    }
    assert call(
        environment, "enroll_hardship_plan", loan_id="LN-20002", plan="deferral_1"
    )[1]


def test_enroll_reduced_payment_keeps_due_date(environment: Environment):
    content, _ = call(
        environment,
        "enroll_hardship_plan",
        loan_id="LN-20001",
        plan="reduced_payment_3",
    )
    assert content["next_due_date"] == "2026-04-20"
    assert content["hardship_history"][-1]["end_date"] == "2026-06-15"


def test_enroll_invalid_plan(environment: Environment):
    assert call(
        environment, "enroll_hardship_plan", loan_id="LN-20001", plan="deferral_3"
    )[1]


# ---------- contact, documents, transfers ----------


def test_update_email_and_phone_normalize(environment: Environment):
    content, error = call(
        environment,
        "update_email",
        borrower_id="BF-10001",
        email=" Anne.New@Example.COM ",
    )
    assert not error and content["email"] == "anne.new@example.com"
    content, error = call(
        environment, "update_phone", borrower_id="BF-10001", phone="(438) 555 0177"
    )
    assert not error and content["phone"] == "438-555-0177"


def test_update_contact_errors(environment: Environment):
    assert call(environment, "update_email", borrower_id="BF-10001", email="nope")[1]
    assert call(environment, "update_phone", borrower_id="BF-10001", phone="555-0177")[
        1
    ]
    assert call(environment, "update_phone", borrower_id="BF-99999", phone="x")[1]
    borrower = environment.tools.db.borrowers["BF-10001"]
    assert borrower.email == "anne.roy@example.com"
    assert borrower.phone == "514-555-0101"


def test_no_tool_has_optional_parameters(environment: Environment):
    # Some providers reject "anyOf [type, null]" parameter schemas.
    for tool in environment.get_tools():
        properties = tool.openai_schema["function"]["parameters"]["properties"]
        for name, schema in properties.items():
            assert "anyOf" not in schema, f"{tool.name}.{name}"


def test_send_document(environment: Environment):
    request, error = call(
        environment,
        "send_document",
        loan_id="LN-20003",
        borrower_id="BF-10002",
        doc_type="statement",
    )
    assert not error
    assert request == {
        "request_id": "DR-60001",
        "loan_id": "LN-20003",
        "borrower_id": "BF-10002",
        "doc_type": "statement",
        "sent_to": "luc.cote@example.com",
    }
    request, _ = call(
        environment,
        "send_document",
        loan_id="LN-20001",
        borrower_id="BF-10001",
        doc_type="tax_summary",
    )
    assert request["request_id"] == "DR-60002"


def test_send_document_errors(environment: Environment):
    assert call(
        environment,
        "send_document",
        loan_id="LN-20001",
        borrower_id="BF-10002",
        doc_type="statement",
    )[1]
    assert call(
        environment,
        "send_document",
        loan_id="LN-20001",
        borrower_id="BF-10001",
        doc_type="receipt",
    )[1]


def test_transfer_to_human_agents(environment: Environment):
    content, error = call(
        environment, "transfer_to_human_agents", reason="dispute", summary="Balance"
    )
    assert (content, error) == ({"transfer_id": "TR-80001", "reason": "dispute"}, False)
    content, _ = call(
        environment, "transfer_to_human_agents", reason="legal", summary="Other text"
    )
    assert content["transfer_id"] == "TR-80002"
    assert call(environment, "transfer_to_human_agents", reason="angry", summary="x")[1]


# ---------- hashing and replay ----------


def run_calls(environment: Environment, calls: list[tuple[str, dict]]) -> list:
    messages = []
    for n, (name, arguments) in enumerate(calls):
        tool_call = ToolCall(id=f"call_{n}", name=name, arguments=arguments)
        messages.append(AssistantMessage(role="assistant", tool_calls=[tool_call]))
        messages.append(environment.get_response(tool_call))
    return messages


CALLS = [
    ("waive_late_fee", {"loan_id": "LN-20001", "fee_id": "FE-50001"}),
    (
        "make_payment",
        {
            "loan_id": "LN-20001",
            "amount": 500,
            "method_id": "BA-40001",
            "payment_date": "2026-03-16",
        },
    ),
    ("get_loan_details", {"loan_id": "LN-20001"}),
    ("transfer_to_human_agents", {"reason": "complaint", "summary": "Anything"}),
]


def test_db_hash_is_stable_and_replayable():
    first = get_environment(db=make_db())
    messages = run_calls(first, CALLS)
    second = get_environment(db=make_db())
    run_calls(second, CALLS)
    assert first.get_db_hash() == second.get_db_hash()
    assert first.get_db_hash() != get_environment(db=make_db()).get_db_hash()

    replayed = get_environment(db=make_db())
    replayed.set_state(
        initialization_data=None, initialization_actions=None, message_history=messages
    )
    assert replayed.get_db_hash() == first.get_db_hash()


def test_transfer_summary_does_not_affect_hash():
    a = get_environment(db=make_db())
    b = get_environment(db=make_db())
    call(a, "transfer_to_human_agents", reason="fraud", summary="short")
    call(b, "transfer_to_human_agents", reason="fraud", summary="a much longer text")
    assert a.get_db_hash() == b.get_db_hash()


def test_order_of_waiver_and_payment_matters():
    waive_first = get_environment(db=make_db())
    run_calls(waive_first, CALLS[:2])
    pay_first = get_environment(db=make_db())
    run_calls(pay_first, [CALLS[1], CALLS[0]])
    assert waive_first.get_db_hash() != pay_first.get_db_hash()


# ---------- real data ----------


def test_real_environment_and_tasks_load():
    environment = get_environment()
    assert environment.get_domain_name() == "loan_servicing"
    assert "Boréal Finance" in environment.get_policy()
    assert len(environment.get_tools()) == 19
    assert get_tasks()
