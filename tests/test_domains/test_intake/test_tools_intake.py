"""Tests for the v4 intake agent tools (design doc §5.2).

Every tool is exercised through ``environment.get_response(ToolCall(...))``:
the five-tool surface (get_callback_order / log_capture / submit_fields /
get_today / end_call), the
fold-canonical write seam, the code-fold record-id echo, the fail-loud
submission contract (wrong / partial / empty / double submission), and the
seed_record / blank_fields / create_callback_order derivation functions that
tasks replay through ``initialization_actions``.
"""

import json
from copy import deepcopy

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.intake.data_model import MISSING, IntakeDB
from tau2.domains.intake.environment import get_environment
from tau2.domains.intake.tools import IntakeTools
from tau2.environment.environment import Environment

RECORD = {
    "record_id": "REC-1001",
    "vertical": "auto",
    "callee_full_name": "Michael Cook",
    "submitted_on": "2025-06-10",
    "fields": [
        {
            "name": "drop_off_date",
            "description": "The drop-off date, format YYYY-MM-DD.",
            "fold": "date",
            "value": "2025-07-22",
        },
        {
            "name": "contact_phone",
            "description": "The phone number the customer can be reached back on.",
            "fold": "phone",
            "value": "(206) 555-0111",
        },
        {
            "name": "second_owner_full_name",
            "description": "The second registered owner's full name.",
            "fold": "name",
            "value": "Ana de la Vega-Marchetti",
        },
    ],
}

MISSING_NAMES = ["contact_phone", "second_owner_full_name"]
ORG = "Kestrel Point Auto Care"


@pytest.fixture
def environment() -> Environment:
    env = get_environment()
    tools: IntakeTools = env.tools
    tools.seed_record(deepcopy(RECORD))
    tools.blank_fields("REC-1001", MISSING_NAMES)
    tools.create_callback_order("REC-1001", ORG)
    return env


def call(environment: Environment, name: str, requestor: str = "assistant", **kwargs):
    """Raw tool-response content (plain string for str-returning tools)."""
    response = environment.get_response(
        ToolCall(id="1", name=name, requestor=requestor, arguments=kwargs)
    )
    assert not response.error, response.content
    return response.content


def call_json(environment: Environment, name: str, **kwargs):
    """Structured tool-response content (model-returning tools)."""
    return json.loads(call(environment, name, **kwargs))


def call_error(
    environment: Environment, name: str, requestor: str = "assistant", **kwargs
) -> str:
    response = environment.get_response(
        ToolCall(id="1", name=name, requestor=requestor, arguments=kwargs)
    )
    assert response.error, response.content
    return response.content


def _record(environment: Environment) -> dict:
    db: IntakeDB = environment.tools.db
    return db.intake_records[0].model_dump()


# ---------------------------------------------------------------------------
# get_callback_order
# ---------------------------------------------------------------------------


def test_get_callback_order_returns_the_derived_order(environment):
    order = call_json(environment, "get_callback_order")
    assert order["record_id"] == "REC-1001"
    assert order["callee_full_name"] == "Michael Cook"
    assert order["org_name"] == ORG
    # Missing fields in record (form) order, with their type descriptions.
    assert [f["name"] for f in order["missing_fields"]] == MISSING_NAMES
    assert order["missing_fields"][0]["description"] == (
        "The phone number the customer can be reached back on."
    )


def test_get_callback_order_fails_loud_on_empty_queue():
    env = get_environment()
    assert "exactly one callback order" in call_error(env, "get_callback_order")


# ---------------------------------------------------------------------------
# log_capture
# ---------------------------------------------------------------------------


def test_log_capture_logs_missing_fields_and_writes_nothing(environment):
    before = _record(environment)
    assert "Logged contact_phone." in call(
        environment, "log_capture", field_name="contact_phone", value="206 555 0111"
    )
    # Re-logging after a correction is allowed.
    assert "Logged contact_phone." in call(
        environment, "log_capture", field_name="contact_phone", value="(206) 555-0111"
    )
    # Stateless: the DB is untouched — the tool-call trace is the record.
    assert _record(environment) == before


def test_log_capture_rejects_fields_not_on_the_order(environment):
    error = call_error(
        environment, "log_capture", field_name="drop_off_date", value="2025-07-23"
    )
    assert "not a missing field" in error
    assert "contact_phone" in error and "second_owner_full_name" in error


def test_log_capture_rejects_empty_values(environment):
    assert "must not be empty" in call_error(
        environment, "log_capture", field_name="contact_phone", value="  "
    )


def test_log_capture_fails_loud_on_empty_queue():
    env = get_environment()
    assert "exactly one callback order" in call_error(
        env, "log_capture", field_name="contact_phone", value="206 555 0111"
    )


# ---------------------------------------------------------------------------
# submit_fields
# ---------------------------------------------------------------------------


def test_submit_fields_folds_values_and_fills_exactly_the_missing_cells(environment):
    result = call(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={
            "contact_phone": "+1 (206) 555-0111",
            "second_owner_full_name": "ANA DE LA VEGA MARCHETTI",
        },
        confirmed_with_user=True,
    )
    assert "REC-1001" in result
    record = _record(environment)
    values = {f["name"]: f["value"] for f in record["fields"]}
    assert values == {
        "drop_off_date": "2025-07-22",  # untouched pre-filled cell
        "contact_phone": "2065550111",  # PHONE fold: digits, NANP code dropped
        "second_owner_full_name": "ana de la vega marchetti",  # NAME fold
    }
    assert MISSING not in values.values()


def test_submit_fields_record_id_echo_folds_like_a_code(environment):
    call(
        environment,
        "submit_fields",
        record_id="rec 1001",
        fields={
            "contact_phone": "(206) 555-0111",
            "second_owner_full_name": "Ana de la Vega-Marchetti",
        },
        confirmed_with_user=True,
    )
    record = _record(environment)
    # The record keeps its own canonical id; only the echo folded.
    assert record["record_id"] == "REC-1001"


def test_submit_fields_unknown_record_fails(environment):
    assert "not found" in call_error(
        environment,
        "submit_fields",
        record_id="REC-9999",
        fields={"vin": "X"},
        confirmed_with_user=True,
    )


def test_submit_fields_rejects_non_missing_and_unknown_fields(environment):
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={
            "contact_phone": "(206) 555-0111",
            "second_owner_full_name": "Ana de la Vega-Marchetti",
            "drop_off_date": "2025-07-23",  # pre-filled, not missing
        },
        confirmed_with_user=True,
    )
    assert "Not missing" in error and "drop_off_date" in error
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={"vin": "4T7TNV1G1RM882620"},  # not a field of this record
        confirmed_with_user=True,
    )
    assert "Not missing" in error and "vin" in error


def test_submit_fields_requires_all_missing_fields_in_one_call(environment):
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={"contact_phone": "(206) 555-0111"},
        confirmed_with_user=True,
    )
    assert "second_owner_full_name" in error
    assert "all missing fields at once" in error
    # Nothing was written by the rejected call.
    values = {f["name"]: f["value"] for f in _record(environment)["fields"]}
    assert values["contact_phone"] == MISSING


def test_submit_fields_rejects_empty_values(environment):
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={"contact_phone": "  ", "second_owner_full_name": "Ana"},
        confirmed_with_user=True,
    )
    assert "must not be an empty value" in error


def test_submit_fields_submits_exactly_once(environment):
    fields = {
        "contact_phone": "(206) 555-0111",
        "second_owner_full_name": "Ana de la Vega-Marchetti",
    }
    call(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields=fields,
        confirmed_with_user=True,
    )
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields=fields,
        confirmed_with_user=True,
    )
    assert "already submitted" in error


def test_submit_fields_requires_the_confirmation_attestation(environment):
    error = call_error(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={
            "contact_phone": "(206) 555-0111",
            "second_owner_full_name": "Ana de la Vega-Marchetti",
        },
    )
    assert "confirmed_with_user" in error
    # The rejected call wrote nothing.
    values = {f["name"]: f["value"] for f in _record(environment)["fields"]}
    assert values["contact_phone"] == MISSING


def test_submit_fields_accepts_an_honest_false_attestation(environment):
    """confirmed_with_user is a recorded self-report, not a gate: a false
    attestation still writes (the trace carries the flag; reward stays
    final-DB hash)."""
    call(
        environment,
        "submit_fields",
        record_id="REC-1001",
        fields={
            "contact_phone": "(206) 555-0111",
            "second_owner_full_name": "Ana de la Vega-Marchetti",
        },
        confirmed_with_user=False,
    )
    values = {f["name"]: f["value"] for f in _record(environment)["fields"]}
    assert values["contact_phone"] == "2065550111"
    # Not stored on the record: the DB shape is unchanged by the attestation.
    assert "confirmed_with_user" not in _record(environment)


# ---------------------------------------------------------------------------
# get_today
# ---------------------------------------------------------------------------


def test_get_today_is_pinned(environment):
    assert call(environment, "get_today") == "2025-06-12"


def test_end_call_is_a_recognized_call_terminator(environment):
    """end_call responds cleanly and both agent classes treat it as a stop."""
    from tau2.agent.discrete_time_audio_native_agent import (
        DiscreteTimeAudioNativeAgent,
    )
    from tau2.agent.llm_agent import LLMAgent

    assert call(environment, "end_call") == "Call ended."
    assert "end_call" in DiscreteTimeAudioNativeAgent.STOP_TOOL_NAMES
    assert "end_call" in LLMAgent.STOP_TOOL_NAMES


# ---------------------------------------------------------------------------
# Derivation functions (initialization_actions surface)
# ---------------------------------------------------------------------------


def test_seed_record_folds_values_at_write():
    env = get_environment()
    tools: IntakeTools = env.tools
    tools.seed_record(deepcopy(RECORD))
    values = {f.name: f.value for f in tools.db.intake_records[0].fields}
    assert values["contact_phone"] == "2065550111"
    assert values["second_owner_full_name"] == "ana de la vega marchetti"
    assert values["drop_off_date"] == "2025-07-22"


def test_seed_record_rejects_duplicates_and_incomplete_records():
    env = get_environment()
    tools: IntakeTools = env.tools
    tools.seed_record(deepcopy(RECORD))
    with pytest.raises(ValueError, match="already exists"):
        tools.seed_record(deepcopy(RECORD))
    incomplete = deepcopy(RECORD)
    incomplete["record_id"] = "REC-1002"
    incomplete["fields"][1]["value"] = MISSING
    with pytest.raises(ValueError, match="complete record"):
        tools.seed_record(incomplete)


def test_blank_fields_fails_loud():
    env = get_environment()
    tools: IntakeTools = env.tools
    tools.seed_record(deepcopy(RECORD))
    with pytest.raises(ValueError, match="not found"):
        tools.blank_fields("REC-9999", ["contact_phone"])
    with pytest.raises(ValueError, match="no field named"):
        tools.blank_fields("REC-1001", ["vin"])
    with pytest.raises(ValueError, match="at least one field"):
        tools.blank_fields("REC-1001", [])
    tools.blank_fields("REC-1001", ["contact_phone"])
    with pytest.raises(ValueError, match="already missing"):
        tools.blank_fields("REC-1001", ["contact_phone"])


def test_create_callback_order_is_derived_and_unique():
    env = get_environment()
    tools: IntakeTools = env.tools
    tools.seed_record(deepcopy(RECORD))
    with pytest.raises(ValueError, match="no missing fields"):
        tools.create_callback_order("REC-1001", ORG)
    tools.blank_fields("REC-1001", MISSING_NAMES)
    with pytest.raises(ValueError, match="must not be blank"):
        tools.create_callback_order("REC-1001", "  ")
    tools.create_callback_order("REC-1001", ORG)
    with pytest.raises(ValueError, match="already exists"):
        tools.create_callback_order("REC-1001", ORG)
    order = tools.db.callback_orders[0]
    assert [f.name for f in order.missing_fields] == MISSING_NAMES


def test_tool_surface_is_exactly_five_tools():
    env = get_environment()
    tools = env.get_tools()
    assert sorted(t.name for t in tools) == [
        "end_call",
        "get_callback_order",
        "get_today",
        "log_capture",
        "submit_fields",
    ]


class TestChannelPolicy:
    """The agent policy is per run channel: voice keeps the historical
    spoken-channel policy (spell-out pinning, read-backs, capture logging);
    text loads the text policy where the typed value is authoritative. Tools,
    DB, and tasks are channel-invariant."""

    def test_default_channel_is_the_voice_policy(self):
        assert "letter by letter" in get_environment().get_policy()

    def test_text_channel_loads_the_text_policy(self):
        policy = get_environment(channel="text").get_policy()
        assert "text chat" in policy
        # The spoken-channel mechanics are gone: no spell-out pinning, no
        # per-entity read-back protocol.
        assert "letter by letter" not in policy
        assert "read the value back" not in policy

    def test_unknown_channel_fails_loud(self):
        with pytest.raises(ValueError, match="Unknown intake channel"):
            get_environment(channel="carrier-pigeon")

    def test_channel_is_invariant_over_tools(self):
        voice_tools = sorted(t.name for t in get_environment().get_tools())
        text_tools = sorted(t.name for t in get_environment(channel="text").get_tools())
        assert voice_tools == text_tools

    def test_run_config_modality_reaches_env_kwargs(self):
        from tau2.data_model.simulation import (
            AudioNativeConfig,
            TextRunConfig,
            VoiceRunConfig,
        )
        from tau2.runner.build import _build_env_kwargs

        text = TextRunConfig(domain="intake", workers=0)
        assert _build_env_kwargs(text, task=None)["channel"] == "text"
        voice = VoiceRunConfig(domain="intake", audio_native_config=AudioNativeConfig())
        assert _build_env_kwargs(voice, task=None)["channel"] == "voice"
