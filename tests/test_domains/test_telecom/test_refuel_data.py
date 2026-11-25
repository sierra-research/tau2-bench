"""Regression checks for data refueling preconditions and billing side effects."""

from pathlib import Path

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.telecom.data_model import BillStatus, LineStatus, TelecomDB
from tau2.domains.telecom.tools import TelecomTools
from tau2.environment.environment import Environment

DB_PATH = Path(__file__).resolve().parents[3] / "data/tau2/domains/telecom/db.toml"


@pytest.fixture
def tools():
    return TelecomTools(TelecomDB.load(DB_PATH))


@pytest.mark.parametrize(
    "status",
    [LineStatus.SUSPENDED, LineStatus.PENDING_ACTIVATION, LineStatus.CLOSED],
)
def test_refuel_data_rejects_inactive_line_without_mutation(tools, status):
    tools._get_line_by_id("L1001").status = status
    before = tools.db.model_dump()

    with pytest.raises(ValueError, match="Line must be active to refuel data"):
        tools.refuel_data("C1001", "L1001", 2.0)

    assert tools.db.model_dump() == before


def test_refuel_data_active_line_adds_data_and_charge(tools):
    line = tools._get_line_by_id("L1001")
    assert line.status == LineStatus.ACTIVE
    initial_data = line.data_refueling_gb
    initial_total_due = sum(bill.total_due for bill in tools.db.bills)

    result = tools.refuel_data("C1001", "L1001", 2.0)

    assert result["charge"] == 10.0
    assert result["new_data_refueling_gb"] == initial_data + 2.0
    assert line.data_refueling_gb == initial_data + 2.0
    assert sum(bill.total_due for bill in tools.db.bills) == initial_total_due + 10.0
    draft = next(
        bill
        for bill in tools.db.bills
        if bill.customer_id == "C1001" and bill.status == BillStatus.DRAFT
    )
    assert draft.line_items[-1].amount == 10.0


@pytest.mark.parametrize("gb_amount", [0.0, -1.0])
def test_refuel_data_rejects_nonpositive_amount_without_mutation(tools, gb_amount):
    before = tools.db.model_dump()

    with pytest.raises(ValueError, match="Refuel amount must be positive"):
        tools.refuel_data("C1001", "L1001", gb_amount)

    assert tools.db.model_dump() == before


def test_refuel_data_rejects_unowned_line_without_mutation(tools):
    before = tools.db.model_dump()

    with pytest.raises(ValueError):
        tools.refuel_data("C1001", "L1004", 2.0)

    assert tools.db.model_dump() == before


def test_refuel_data_environment_reports_error_without_mutation(tools):
    tools._get_line_by_id("L1001").status = LineStatus.CLOSED
    before = tools.db.model_dump()
    environment = Environment(domain_name="telecom", policy="", tools=tools)

    response = environment.get_response(
        ToolCall(
            id="refuel-inactive",
            name="refuel_data",
            arguments={"customer_id": "C1001", "line_id": "L1001", "gb_amount": 2.0},
        )
    )

    assert response.error
    assert "Line must be active to refuel data" in response.content
    assert tools.db.model_dump() == before
