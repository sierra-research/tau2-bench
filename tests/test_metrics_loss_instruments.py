# Copyright Sierra
"""Deterministic loss instruments: caller-cost + entity-trace on constructed,
real-shaped fixtures (known ticks -> known milestone/counter values; known
tool traces -> known capture/fabrication events)."""

from pathlib import Path
from typing import Optional

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import (
    EnvFunctionCall,
    InitialState,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.metrics.caller_cost import (
    CallerCostCellAggregate,
    CallerCostConfig,
    CallerCostCounters,
    MetricSummary,
    aggregate_cells,
    build_caller_cost,
    extract_caller_cost,
)
from tau2.metrics.entity_trace import (
    EntityKind,
    EntityTraceConfig,
    _kind_for_arg,
    build_entity_trace,
    expected_entities,
    normalize_date,
    normalize_entity_value,
    trace_call_entities,
    value_conveyed_in,
)
from tau2.metrics.run_loading import LoadedCall, RunMeta

DUR = 0.2  # tick clock used by every voice fixture


# ---------------------------------------------------------------------------
# Fixture builders (shapes mirror real stored sims)
# ---------------------------------------------------------------------------


def make_tick(
    i: int,
    *,
    agent_speech: bool = False,
    user_speech: bool = False,
    user_text: Optional[str] = None,
    user_action: Optional[str] = None,
    agent_tool_calls: Optional[list[ToolCall]] = None,
    agent_tool_results: Optional[list[ToolMessage]] = None,
) -> Tick:
    agent_chunk = (
        AssistantMessage.voice(content="a", is_audio=False, contains_speech=True)
        if agent_speech
        else None
    )
    user_chunk = None
    if user_speech or user_action:
        user_chunk = UserMessage.voice(
            content=(
                user_text if user_text is not None else ("u" if user_speech else None)
            ),
            is_audio=False,
            contains_speech=user_speech,
            turn_taking_action=(
                TurnTakingAction(action=user_action) if user_action else None
            ),
        )
    return Tick(
        tick_id=i,
        timestamp=f"2026-01-01T00:00:{i:02d}",
        tick_duration_seconds=DUR,
        agent_chunk=agent_chunk,
        user_chunk=user_chunk,
        agent_tool_calls=agent_tool_calls or [],
        agent_tool_results=agent_tool_results or [],
    )


def make_voice_sim(ticks: list[Tick], *, sim_id: str = "sim_v") -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id="task_1",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1.0),
        ticks=ticks,
        mode="full_duplex",
    )


def telecom_task(task_id: str = "task_1") -> Task:
    """Telecom-shaped task: structured identity init + prose DOB."""
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="telecom",
                task_instructions="Get your data refueled.",
                reason_for_call="Mobile data issue.",
                known_info=(
                    "You are Song Jiayi and your date of birth is March 21, 1992."
                ),
            )
        ),
        initial_state=InitialState(
            initialization_actions=[
                EnvFunctionCall(
                    env_type="user",
                    func_name="set_user_info",
                    arguments={
                        "name": "Song Jiayi",
                        "phone_number": "555-123-2002",
                    },
                ),
            ]
        ),
    )


def retail_task(task_id: str = "task_1") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="retail",
                task_instructions="You are cautious.",
                reason_for_call="Exchange an item.",
                known_info=("You name is Javier Dominguez and your zip code is 56557."),
                unknown_info="You do not remember your email address",
            )
        ),
    )


def airline_task(task_id: str = "task_1") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                task_instructions="Change your flight.",
                reason_for_call="Rebooking.",
                known_info=(
                    "You are Gonzalo Navarro.\nYour user id is gonzalo_navarro_7340."
                ),
            )
        ),
    )


def loaded_call(
    sim: SimulationRun,
    task: Optional[Task],
    *,
    domain: str,
    language: str = "en",
    modality: str = "voice",
) -> LoadedCall:
    return LoadedCall(
        meta=RunMeta(
            results_path="/tmp/run/results.json",
            experiment_label="run",
            domain=domain,
            modality=modality,
            provider="openai" if modality == "voice" else "",
            agent_model="openai:gpt-realtime-2",
            reasoning_effort="xhigh" if modality == "voice" else None,
            run_git_commit="0" * 40,
        ),
        sim=sim,
        task=task,
        language=language,
    )


# ---------------------------------------------------------------------------
# Normalization / conveyance
# ---------------------------------------------------------------------------


def test_normalize_date_parses_iso_numeric_and_prose():
    assert normalize_date("1992-03-21") == "1992-03-21"
    assert normalize_date("March 21, 1992") == "1992-03-21"
    assert normalize_date("3/21/1992") == "1992-03-21"
    # unparseable shapes fall back to the digit string, still comparable
    assert normalize_date("21.03.1992") == normalize_date("21031992")


def test_normalize_entity_value_folds_per_kind():
    assert normalize_entity_value(EntityKind.PHONE, "555-123-2002") == "5551232002"
    assert normalize_entity_value(
        EntityKind.USER_ID, "Laura_Ortega_8020"
    ) == normalize_entity_value(EntityKind.USER_ID, "laura ortega 8020")
    assert (
        normalize_entity_value(EntityKind.NAME, "Álvaro Domínguez")
        == "alvaro dominguez"
    )
    assert normalize_entity_value(EntityKind.DATE, "March 21, 1992") == "1992-03-21"


def test_value_conveyed_in_detects_spelled_spanish_digits():
    zip_norm = normalize_entity_value(EntityKind.ZIP, "07584")
    text = "El código postal: cero, siete, cinco, ocho, cuatro."
    assert value_conveyed_in(EntityKind.ZIP, zip_norm, text, "es")
    assert not value_conveyed_in(EntityKind.ZIP, zip_norm, text, "zh")


def test_value_conveyed_in_matches_cjk_name_as_substring():
    # \b never fires between CJK word characters, so tokens match by
    # containment; a romanized ground truth still misses (documented limit).
    assert value_conveyed_in(EntityKind.NAME, "罗强", "我叫罗强，谢谢。", "zh")
    assert not value_conveyed_in(EntityKind.NAME, "luo qiang", "我叫罗强。", "zh")


def test_value_conveyed_in_names_are_fold_and_order_aware():
    name = normalize_entity_value(EntityKind.NAME, "Javier Domínguez")
    assert value_conveyed_in(EntityKind.NAME, name, "Soy JAVIER DOMINGUEZ.", "es")
    assert value_conveyed_in(EntityKind.NAME, name, "Dominguez, Javier aquí.", "es")
    assert not value_conveyed_in(EntityKind.NAME, name, "Soy Javier Serrano.", "es")


# ---------------------------------------------------------------------------
# Entity pinning
# ---------------------------------------------------------------------------


def test_expected_entities_telecom_reads_init_actions_and_prose_dob():
    entities = {(e.kind, e.value) for e in expected_entities(telecom_task(), "telecom")}
    assert (EntityKind.NAME, "Song Jiayi") in entities
    assert (EntityKind.PHONE, "555-123-2002") in entities
    assert (EntityKind.DATE, "March 21, 1992") in entities


def test_expected_entities_retail_pins_name_and_zip():
    entities = {(e.kind, e.value) for e in expected_entities(retail_task(), "retail")}
    assert (EntityKind.NAME, "Javier Dominguez") in entities
    assert (EntityKind.ZIP, "56557") in entities


def test_expected_entities_airline_pins_handle_and_prose_name():
    by_kind = {e.kind: e for e in expected_entities(airline_task(), "airline")}
    assert by_kind[EntityKind.USER_ID].value == "gonzalo_navarro_7340"
    assert by_kind[EntityKind.NAME].value == "Gonzalo Navarro"


def test_expected_entities_derives_name_from_handle_when_no_prose_name():
    task = Task(
        id="t",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="retail",
                task_instructions="x",
                reason_for_call="x",
                known_info="You are laura_ortega_8020 living in zipcode 82000.",
            )
        ),
    )
    by_kind = {e.kind: e for e in expected_entities(task, "retail")}
    assert by_kind[EntityKind.NAME].value == "laura ortega"
    assert by_kind[EntityKind.NAME].source == "derived_from_user_id"
    assert by_kind[EntityKind.ZIP].value == "82000"


# ---------------------------------------------------------------------------
# Argument kind assignment
# ---------------------------------------------------------------------------


def test_kind_for_arg_name_hints_win():
    assert _kind_for_arg("phone_number", "5551232002") is EntityKind.PHONE
    assert _kind_for_arg("date_of_birth", "1992-03-21") is EntityKind.DATE
    assert _kind_for_arg("zip", "10001") is EntityKind.ZIP
    assert _kind_for_arg("user_id", "mia_li_3668") is EntityKind.USER_ID


def test_kind_for_arg_record_handles_never_shape_fallback():
    # digit-run record handles must not masquerade as phones
    assert _kind_for_arg("item_ids.0", "2913673670") is None
    assert _kind_for_arg("product_id", "4760268021") is None
    assert _kind_for_arg("order_id", "#W4316152") is None


def test_kind_for_arg_shape_fallback_is_conservative():
    assert _kind_for_arg("contact", "a.b@example.com") is EntityKind.EMAIL
    assert _kind_for_arg("value", "555-123-2002") is EntityKind.PHONE
    # a bare digit run without separators is ambiguous -> not traced
    assert _kind_for_arg("value", "5551232002") is None


# ---------------------------------------------------------------------------
# Entity tracing (voice)
# ---------------------------------------------------------------------------


def _result(call_id: str, content: str, *, error: bool = False) -> ToolMessage:
    return ToolMessage(
        id=call_id, role="tool", content=content, requestor="assistant", error=error
    )


@pytest.fixture
def traced_voice_call() -> LoadedCall:
    """A telecom voice call with one mishearing, one record-sourced value,
    one state-changing fabrication, and a final correct capture.

    tick  0-1: agent greeting
    tick  3-4: caller gives the phone number (text on the first tick)
    tick  6  : lookup with a DROPPED-DIGIT mishearing -> not-found error
    tick  8-9: caller re-dictates the number in spelled Spanish digits
    tick 10  : lookup with the correct number -> success (record carries a
               contact DOB of 1985-06-14 and customer id C1001)
    tick 12  : a WRITE tool call carrying a fabricated phone value
    tick 14  : verification call reusing the record's DOB (wrong vs the
               caller's ground-truth DOB, but read off the record)
    """
    ticks = [
        make_tick(0, agent_speech=True),
        make_tick(1, agent_speech=True),
        make_tick(2),
        make_tick(3, user_speech=True, user_text="Mi número es 555-123-2002."),
        make_tick(4, user_speech=True, user_text=None),
        make_tick(5),
        make_tick(
            6,
            agent_tool_calls=[
                ToolCall(
                    id="c1",
                    name="get_customer_by_phone",
                    arguments={"phone_number": "551-232-002"},
                    requestor="assistant",
                )
            ],
            agent_tool_results=[_result("c1", "Error: Customer not found", error=True)],
        ),
        make_tick(7),
        make_tick(
            8,
            user_speech=True,
            user_text="Se lo repito: cinco cinco cinco, uno dos tres, dos cero cero dos.",
        ),
        make_tick(9, user_speech=True, user_text=None),
        make_tick(
            10,
            agent_tool_calls=[
                ToolCall(
                    id="c2",
                    name="get_customer_by_phone",
                    arguments={"phone_number": "555-123-2002"},
                    requestor="assistant",
                )
            ],
            agent_tool_results=[
                _result(
                    "c2",
                    '{"customer_id": "C1001", "name": "Song Jiayi", '
                    '"contact_dob": "1985-06-14"}',
                )
            ],
        ),
        make_tick(11),
        make_tick(
            12,
            agent_tool_calls=[
                ToolCall(
                    id="c3",
                    name="update_contact_phone",
                    arguments={"phone_number": "999-999-9999"},
                    requestor="assistant",
                )
            ],
            agent_tool_results=[_result("c3", "ok")],
        ),
        make_tick(13),
        make_tick(
            14,
            agent_tool_calls=[
                ToolCall(
                    id="c4",
                    name="verify_dob",
                    arguments={"date_of_birth": "1985-06-14"},
                    requestor="assistant",
                )
            ],
            agent_tool_results=[_result("c4", "no", error=True)],
        ),
    ]
    return loaded_call(
        make_voice_sim(ticks), telecom_task(), domain="telecom", language="es"
    )


WRITE_MAP = {
    "get_customer_by_phone": False,
    "update_contact_phone": True,
    "verify_dob": False,
}


def test_trace_marks_dropped_digit_mishearing_as_transient_fabrication(
    traced_voice_call,
):
    records, _ = trace_call_entities(traced_voice_call, write_map=WRITE_MAP)
    phone = next(r for r in records if r.entity_kind is EntityKind.PHONE)
    assert phone.first_capture_correct is False
    assert phone.wrong_arg_event_count == 2  # mishearing + write fabrication
    assert phone.fabricated_event_count == 2
    # the mishearing was later corrected (tick 10); the write one never was
    assert phone.transient_fabrication_count == 1
    misheard = phone.events[0]
    assert misheard.value_raw == "551-232-002"
    # substring-of-truth artifact: never counted as caller-uttered
    assert misheard.value_uttered_by_caller_before is False
    assert misheard.fabricated and misheard.corrected_later


def test_trace_splits_read_only_vs_state_changing(traced_voice_call):
    records, rollup = trace_call_entities(traced_voice_call, write_map=WRITE_MAP)
    phone = next(r for r in records if r.entity_kind is EntityKind.PHONE)
    assert phone.read_only_fabricated_count == 1  # the misheard lookup
    assert phone.state_changing_fabricated_count == 1  # the write tool
    assert rollup.state_changing_fabrication_count == 1
    assert rollup.any_fabrication_reached_state_change


def test_trace_record_sourced_values_are_not_fabrications(traced_voice_call):
    records, _ = trace_call_entities(traced_voice_call, write_map=WRITE_MAP)
    date = next(r for r in records if r.entity_kind is EntityKind.DATE)
    # the verify_dob arg came off the tick-10 record: flagged, not counted
    assert date.wrong_arg_event_count == 0
    assert date.fabricated_event_count == 0
    assert date.first_capture_correct is None  # no eligible realization
    event = date.events[0]
    assert event.value_seen_in_prior_tool_result is True
    assert not event.fabricated


def test_trace_conveyance_counts_spelled_redictation(traced_voice_call):
    records, _ = trace_call_entities(traced_voice_call, write_map=WRITE_MAP)
    phone = next(r for r in records if r.entity_kind is EntityKind.PHONE)
    # tick 3 (verbatim) + tick 8 (spelled Spanish digits)
    assert phone.conveyance_count == 2
    assert phone.first_conveyance_order == 3


def test_trace_text_run_with_correct_args_is_clean():
    messages = [
        UserMessage(role="user", content="Hi, I am gonzalo_navarro_7340."),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="get_user_details",
                    arguments={"user_id": "gonzalo_navarro_7340"},
                    requestor="assistant",
                )
            ],
        ),
        ToolMessage(
            id="c1", role="tool", content='{"name": "Gonzalo"}', requestor="assistant"
        ),
        AssistantMessage(role="assistant", content="Found you."),
    ]
    sim = SimulationRun(
        id="sim_t",
        task_id="task_1",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=1.0),
        messages=messages,
    )
    call = loaded_call(sim, airline_task(), domain="airline", modality="text")
    records, rollup = trace_call_entities(call, write_map={"get_user_details": False})
    user_id = next(r for r in records if r.entity_kind is EntityKind.USER_ID)
    assert user_id.first_capture_correct is True
    assert user_id.first_conveyance_order == 0
    assert rollup.fabricated_event_count == 0
    assert not rollup.any_fabrication_reached_state_change


# ---------------------------------------------------------------------------
# Caller cost
# ---------------------------------------------------------------------------


def test_caller_cost_voice_milestones_on_the_tick_clock(traced_voice_call):
    record = extract_caller_cost(traced_voice_call)
    counters = record.counters
    assert counters.call_duration_s == pytest.approx(15 * DUR)
    assert counters.time_to_first_agent_speech_s == pytest.approx(0.0)
    assert counters.time_to_first_tool_call_s == pytest.approx(6 * DUR)
    assert counters.time_to_auth_s == pytest.approx(10 * DUR)
    assert counters.auth_attempt_count == 2  # mishearing + the success
    assert counters.auth_arg_mismatch_count == 1
    assert counters.agent_tool_call_count == 4
    assert counters.tool_error_count == 2
    assert counters.caller_speaking_time_s == pytest.approx(4 * DUR)
    assert counters.agent_speaking_time_s == pytest.approx(2 * DUR)


def test_caller_cost_counts_redictation_after_mismatch(traced_voice_call):
    counters = extract_caller_cost(traced_voice_call).counters
    assert counters.entity_count == 3  # name, phone, dob
    assert counters.entity_redictation_count == 1  # the spelled repeat
    assert counters.entities_redictated_count == 1
    # the re-dictation happened after the tick-6 not-found
    assert counters.post_mismatch_redictation_count == 1


def test_caller_cost_counts_barge_ins():
    ticks = [
        make_tick(0, agent_speech=True),
        make_tick(1, agent_speech=True, user_speech=True, user_text="wait—"),
        make_tick(2, user_speech=True, user_text=None),
        make_tick(3),
        make_tick(4, agent_speech=True),
        # backchannel during agent speech: never a barge-in
        make_tick(
            5,
            agent_speech=True,
            user_speech=True,
            user_text="mhm",
            user_action="backchannel",
        ),
    ]
    call = loaded_call(make_voice_sim(ticks), telecom_task(), domain="telecom")
    counters = extract_caller_cost(call).counters
    assert counters.caller_barge_in_count == 1
    assert counters.caller_turn_count == 1  # the backchannel is not a turn


def test_caller_cost_text_run_has_counts_but_no_tick_timing():
    messages = [
        UserMessage(role="user", content="Hi, I am gonzalo_navarro_7340."),
        AssistantMessage(role="assistant", content="How can I help?"),
        UserMessage(role="user", content="My id is gonzalo_navarro_7340."),
        AssistantMessage(role="assistant", content="Thanks."),
    ]
    sim = SimulationRun(
        id="sim_t",
        task_id="task_1",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=0.0),
        messages=messages,
    )
    call = loaded_call(sim, airline_task(), domain="airline", modality="text")
    counters = extract_caller_cost(call).counters
    assert counters.call_duration_s == 60.0  # wall clock, not tick clock
    assert counters.time_to_first_agent_speech_s is None
    assert counters.caller_barge_in_count is None
    assert counters.caller_turn_count == 2
    assert counters.agent_turn_count == 2
    assert counters.entity_redictation_count == 1  # the handle, twice


# ---------------------------------------------------------------------------
# Aggregation + artifacts
# ---------------------------------------------------------------------------


def test_aggregate_cells_covers_every_counter_field(traced_voice_call):
    cells = aggregate_cells([extract_caller_cost(traced_voice_call)])
    assert len(cells) == 1
    cell = cells[0]
    assert (cell.language, cell.domain, cell.provider) == ("es", "telecom", "openai")
    assert set(cell.metrics) == set(CallerCostCounters.model_fields)
    assert cell.metrics["caller_turn_count"].n == 1


def test_cell_aggregate_rejects_unknown_metric_keys():
    with pytest.raises(ValueError, match="not CallerCostCounters fields"):
        CallerCostCellAggregate(
            language="en",
            domain="telecom",
            modality="voice",
            provider="openai",
            n_calls=1,
            metrics={"not_a_counter": MetricSummary(mean=1.0, median=1.0, n=1)},
        )


@pytest.fixture
def run_on_disk(tmp_path: Path) -> Path:
    """A real dir-format run written via Results.save (telecom, one sim)."""
    from fixtures_runs import make_hi_results

    ticks = [
        make_tick(0, agent_speech=True),
        make_tick(1, user_speech=True, user_text="I am Song Jiayi, 555-123-2002."),
        make_tick(
            2,
            agent_tool_calls=[
                ToolCall(
                    id="c1",
                    name="get_customer_by_phone",
                    arguments={"phone_number": "555-123-2002"},
                    requestor="assistant",
                )
            ],
            agent_tool_results=[_result("c1", '{"customer_id": "C1001"}')],
        ),
    ]
    sim = make_voice_sim(ticks, sim_id="sim_disk")
    from tau2.config import ReasoningEffort
    from tau2.data_model.simulation import AudioNativeConfig

    return make_hi_results(
        tmp_path,
        [sim],
        tasks=[telecom_task()],
        domain="telecom",
        audio_native_config=AudioNativeConfig(
            provider="openai",
            model="gpt-realtime-2",
            reasoning_effort=ReasoningEffort.XHIGH,
        ),
        agent_llm="openai:gpt-realtime-2",
    )


def test_build_caller_cost_artifact_is_content_deterministic(run_on_disk):
    first = build_caller_cost([run_on_disk], CallerCostConfig())
    second = build_caller_cost([run_on_disk], CallerCostConfig())
    assert first.artifact_id == second.artifact_id
    assert len(first.records) == 1
    assert first.source_runs[0].n_calls == 1
    assert first.source_runs[0].meta.domain == "telecom"
    assert first.records[0].counters.auth_attempt_count == 1


def test_build_entity_trace_artifact_over_disk_run(run_on_disk):
    artifact = build_entity_trace([run_on_disk], EntityTraceConfig())
    assert (
        artifact.artifact_id
        == build_entity_trace([run_on_disk], EntityTraceConfig()).artifact_id
    )
    phone = next(r for r in artifact.records if r.entity_kind is EntityKind.PHONE)
    assert phone.first_capture_correct is True
    assert artifact.rollups[0].fabricated_event_count == 0
    # join keys for the fce30 labels are present
    assert phone.sim_id == "sim_disk"
    assert phone.task_id == "task_1"
    assert phone.entity_value_normalized == "5551232002"


def test_cli_verbs_write_validating_artifacts(run_on_disk, tmp_path):
    import argparse

    from tau2.metrics.caller_cost import CallerCostArtifact
    from tau2.metrics.cli import run_metrics_caller_cost, run_metrics_entity_trace
    from tau2.metrics.entity_trace import EntityTraceArtifact

    cc_out = tmp_path / "cc.json"
    run_metrics_caller_cost(
        argparse.Namespace(
            results=[str(run_on_disk)],
            output=cc_out,
            langs=None,
            domain=None,
            max_sims=None,
        )
    )
    loaded = CallerCostArtifact.model_validate_json(cc_out.read_text())
    assert loaded.records and loaded.cells

    et_out = tmp_path / "et.json"
    run_metrics_entity_trace(
        argparse.Namespace(
            results=[str(run_on_disk)],
            output=et_out,
            langs=None,
            domain=None,
            max_sims=None,
            no_events=False,
        )
    )
    loaded_trace = EntityTraceArtifact.model_validate_json(et_out.read_text())
    assert loaded_trace.records and loaded_trace.rollups
