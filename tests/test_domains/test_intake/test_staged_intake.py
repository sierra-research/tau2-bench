"""The intake_staged domain: staged tools, verification, and policy.

Unit-level coverage of :mod:`tau2.domains.intake.staged` — the chain-band
integration (twin alignment, golden replay through the evaluator) lives in
``test_compose_intake.py``.
"""

import pytest

from tau2.domains.intake.data_model import MISSING
from tau2.domains.intake.staged import (
    STAGED_POLICY_SECTION,
    StagedIntakeDB,
    StagedIntakeTools,
    get_environment,
)
from tau2.domains.intake.utils import INTAKE_MAIN_POLICY_PATH


def _staged_tools() -> StagedIntakeTools:
    """A two-stage record, staged and blanked the way chain tasks do it."""
    tools = StagedIntakeTools(StagedIntakeDB())
    tools.seed_record(
        {
            "record_id": "REC-1",
            "vertical": "clinic",
            "callee_full_name": "Ana Flores",
            "submitted_on": "2025-06-03",
            "fields": [
                {
                    "name": "contact_phone",
                    "description": "Callback phone.",
                    "fold": "phone",
                    "value": "555-014-2233",
                },
                {
                    "name": "appointment_time",
                    "description": "Appointment time, H:MM AM/PM.",
                    "fold": "time",
                    "value": "3:30 PM",
                },
                {
                    "name": "home_address",
                    "description": "Full home address.",
                    "fold": "name",
                    "value": "741 Elm Drive, Lakewood",
                },
            ],
        }
    )
    tools.stage_fields("REC-1", ["appointment_time", "home_address"])
    tools.blank_fields("REC-1", ["appointment_time", "home_address"])
    tools.create_callback_order("REC-1", "Maple Grove Family Clinic")
    return tools


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def test_stage_fields_pins_folded_targets():
    tools = _staged_tools()
    # seed_record stores fold-canonical values, so the pinned targets are
    # already folded (names case-folded and separator-stripped).
    assert tools.db.stage_expected == {
        "REC-1": {
            "appointment_time": "3:30 PM",
            "home_address": "741 elm drive lakewood",
        }
    }


def test_stage_fields_must_run_before_blank():
    tools = StagedIntakeTools(StagedIntakeDB())
    tools.seed_record(
        {
            "record_id": "REC-2",
            "vertical": "clinic",
            "callee_full_name": "Ana Flores",
            "submitted_on": "2025-06-03",
            "fields": [
                {
                    "name": "appointment_time",
                    "description": "Appointment time.",
                    "fold": "time",
                    "value": "3:30 PM",
                }
            ],
        }
    )
    tools.blank_fields("REC-2", ["appointment_time"])
    with pytest.raises(ValueError, match="before blank_fields"):
        tools.stage_fields("REC-2", ["appointment_time"])


def test_create_callback_order_requires_staged_targets():
    tools = StagedIntakeTools(StagedIntakeDB())
    tools.seed_record(
        {
            "record_id": "REC-3",
            "vertical": "clinic",
            "callee_full_name": "Ana Flores",
            "submitted_on": "2025-06-03",
            "fields": [
                {
                    "name": "appointment_time",
                    "description": "Appointment time.",
                    "fold": "time",
                    "value": "3:30 PM",
                }
            ],
        }
    )
    tools.blank_fields("REC-3", ["appointment_time"])
    with pytest.raises(ValueError, match="stage_fields must run"):
        tools.create_callback_order("REC-3", "Maple Grove Family Clinic")


# ---------------------------------------------------------------------------
# The staged protocol
# ---------------------------------------------------------------------------


def test_order_reveals_one_field_at_a_time():
    tools = _staged_tools()
    order = tools.get_callback_order()
    assert [spec.name for spec in order.missing_fields] == ["appointment_time"]
    # The underlying order is not mutated by the redacted view.
    assert len(tools.db.callback_orders[0].missing_fields) == 2


def test_wrong_value_is_refused_and_does_not_advance():
    tools = _staged_tools()
    with pytest.raises(ValueError, match="did not verify"):
        tools.submit_fields(
            record_id="REC-1",
            fields={"appointment_time": "3:30 AM"},
            confirmed_with_user=True,
        )
    record = tools.db.intake_records[0]
    assert record.fields[1].value == MISSING
    order = tools.get_callback_order()
    assert [spec.name for spec in order.missing_fields] == ["appointment_time"]


def test_correct_value_clears_and_names_the_next_field():
    tools = _staged_tools()
    message = tools.submit_fields(
        record_id="REC-1",
        fields={"appointment_time": "3:30 pm"},  # fold-equivalent form
        confirmed_with_user=True,
    )
    assert "Verified and recorded appointment_time" in message
    assert "home_address" in message
    order = tools.get_callback_order()
    assert [spec.name for spec in order.missing_fields] == ["home_address"]


def test_retry_after_refusal_succeeds():
    tools = _staged_tools()
    with pytest.raises(ValueError, match="did not verify"):
        tools.submit_fields(
            record_id="REC-1",
            fields={"appointment_time": "4:30 PM"},
            confirmed_with_user=True,
        )
    message = tools.submit_fields(
        record_id="REC-1",
        fields={"appointment_time": "3:30 PM"},
        confirmed_with_user=True,
    )
    assert "Verified and recorded appointment_time" in message


def test_next_field_cannot_be_submitted_early():
    tools = _staged_tools()
    with pytest.raises(ValueError, match="not the field the order currently asks"):
        tools.submit_fields(
            record_id="REC-1",
            fields={"home_address": "741 Elm Drive, Lakewood"},
            confirmed_with_user=True,
        )


def test_multi_field_submission_is_refused():
    tools = _staged_tools()
    with pytest.raises(ValueError, match="one field per submission"):
        tools.submit_fields(
            record_id="REC-1",
            fields={
                "appointment_time": "3:30 PM",
                "home_address": "741 Elm Drive, Lakewood",
            },
            confirmed_with_user=True,
        )


def test_completing_the_chain_and_submitting_again():
    tools = _staged_tools()
    tools.submit_fields(
        record_id="REC-1",
        fields={"appointment_time": "3:30 PM"},
        confirmed_with_user=True,
    )
    message = tools.submit_fields(
        record_id="REC-1",
        fields={"home_address": "741 elm drive, lakewood"},  # fold-equivalent
        confirmed_with_user=True,
    )
    assert "complete" in message
    record = tools.db.intake_records[0]
    assert not any(field.value == MISSING for field in record.fields)
    with pytest.raises(ValueError, match="already submitted"):
        tools.submit_fields(
            record_id="REC-1",
            fields={"home_address": "741 Elm Drive, Lakewood"},
            confirmed_with_user=True,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_staged_policy_extends_the_canonical_policy_verbatim():
    env = get_environment()
    canonical = INTAKE_MAIN_POLICY_PATH.read_text()
    assert env.policy == canonical + STAGED_POLICY_SECTION
    assert env.domain_name == "intake_staged"


def test_solo_mode_is_rejected():
    with pytest.raises(ValueError, match="Solo mode not supported"):
        get_environment(solo_mode=True)
