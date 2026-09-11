# Copyright Sierra
"""Universal call-quality rubric: catalog, deterministic checks, and judge."""

import json
from types import SimpleNamespace

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import (
    JudgeOutcome,
    QualityJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import (
    Action,
    EvaluationCriteria,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.judges.quality.factors import (
    HYBRID_FACTORS,
    QUALITY_FACTOR_BY_ID,
    QUALITY_FACTORS,
    SEMANTIC_FACTORS,
)
from tau2.judges.quality.harness import evaluate_quality
from tau2.judges.quality.judge import (
    _SYSTEM_PROMPT,
    QUALITY_JUDGE_PROMPT_VERSION,
    QualityFactorReply,
    UnnecessaryToolCallReply,
    build_quality_factor_prompt,
    build_quality_judge_input,
)
from tau2.metrics.interaction_quality import (
    InteractionQualityMetrics,
    extract_spoken_turns,
)
from tau2.metrics.voice_interaction_metrics import (
    VoiceInteractionMetricCounts,
    VoiceInteractionMetrics,
)


def _task() -> Task:
    return Task(
        id="t1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="mock",
                reason_for_call="test quality",
                task_instructions="Help the caller.",
            )
        ),
    )


def _sim(*, messages=None, ticks=None) -> SimulationRun:
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=10.0,
        termination_reason="agent_stop",
        messages=messages,
        ticks=ticks,
    )


def test_catalog_is_closed_and_excludes_transfer():
    ids = [factor.id for factor in QUALITY_FACTORS]
    assert len(ids) == len(set(ids))
    assert not any("transfer" in factor_id for factor_id in ids)
    assert {factor.id for factor in SEMANTIC_FACTORS} == {
        "unnecessary_repetition",
        "incorrect_tool_parameters",
        "unnecessary_tool_call",
    }
    assert {factor.id for factor in HYBRID_FACTORS} == {
        "agent_caused_tool_error",
        "auth_arg_mismatch",
    }
    assert len(QUALITY_FACTORS) == 12
    # redundant_successful_tool_call was retired 2026-08-20: on the 480-call
    # es+pt judge_calibration_200 corpus the exact-repeat checker failed 2
    # calls total — a near-dead factor.
    assert {
        factor.id for factor in QUALITY_FACTORS if factor.evaluator == "deterministic"
    } == {
        "responsiveness",
        "yielding",
        "inappropriate_interruption",
        "backchannel_selectivity",
        "vocal_tic_selectivity",
        "non_directed_selectivity",
        "monologue",
    }
    unnecessary = next(
        factor for factor in QUALITY_FACTORS if factor.id == "unnecessary_tool_call"
    )
    assert unnecessary.judge_model == "gpt-5.5"
    assert unnecessary.judge_args["reasoning_effort"] == "xhigh"
    # Evaluation levels are declared at the source (owner ruling 2026-08-20):
    # utterance = the violation lives in clickable agent-utterance content;
    # tool-mechanics and timing/event factors stay call-level — their
    # violations live in the tool trace or timing, not in any one utterance.
    assert {
        factor.id
        for factor in QUALITY_FACTORS
        if factor.evaluation_level == "utterance"
    } == {"unnecessary_repetition", "monologue"}
    assert all(
        factor.evaluation_level == "call"
        for factor in QUALITY_FACTORS
        if factor.id not in {"unnecessary_repetition", "monologue"}
    )


def test_repetition_is_semantic_and_defers_without_the_llm_judge():
    call1 = ToolCall(id="c1", name="lookup", arguments={"id": "123"})
    call2 = ToolCall(id="c2", name="lookup", arguments={"id": "123"})
    sim = _sim(
        messages=[
            AssistantMessage.text("Let me check."),
            AssistantMessage.text("Let me check!"),
            AssistantMessage.text("", tool_calls=[call1]),
            ToolMessage(id="c1", role="tool", content="ok"),
            AssistantMessage.text("", tool_calls=[call2]),
            ToolMessage(id="c2", role="tool", content="ok"),
        ]
    )
    info = evaluate_quality(
        sim,
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=False),
    )
    by_id = {check.id: check for check in info.factor_checks}
    assert by_id["unnecessary_repetition"].outcome == JudgeOutcome.DEFERRED
    assert all(
        by_id[factor.id].outcome == JudgeOutcome.DEFERRED for factor in SEMANTIC_FACTORS
    )
    # No ticks and no LLM: every remaining factor is deferred or without
    # opportunity, so no score exists (the retired exact-repeat checker was
    # this text-only sim's only firing factor).
    assert info.score is None
    assert all(check.judge_model is None for check in info.factor_checks)


def test_voice_timing_checks_use_recorded_tick_clock():
    ticks = []
    for i in range(9):
        agent_chunk = None
        user_chunk = None
        if i == 0:
            user_chunk = UserMessage.voice(content="u", contains_speech=True)
        elif i >= 5:
            agent_chunk = AssistantMessage.voice(content="a", contains_speech=True)
        ticks.append(
            Tick(
                tick_id=i,
                timestamp=str(i),
                tick_duration_seconds=1.0,
                agent_chunk=agent_chunk,
                user_chunk=user_chunk,
            )
        )
    info = evaluate_quality(
        _sim(ticks=ticks),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(
            llm_judge=False,
            response_latency_seconds=3.0,
            monologue_seconds=3.0,
        ),
    )
    by_id = {check.id: check for check in info.factor_checks}
    assert by_id["responsiveness"].outcome == JudgeOutcome.FAIL
    assert by_id["responsiveness"].metrics["response_latency_mean"] == 4.0
    assert by_id["monologue"].outcome == JudgeOutcome.FAIL
    assert by_id["monologue"].metrics["max_agent_floor_hold_seconds"] == 4.0


def test_interruption_is_strict_deterministic_overlap():
    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(content="u", contains_speech=True),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(content="u", contains_speech=True),
            agent_chunk=AssistantMessage.voice(content="a", contains_speech=True),
        ),
    ]
    info = evaluate_quality(
        _sim(ticks=ticks),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=False),
    )
    check = next(
        check
        for check in info.factor_checks
        if check.id == "inappropriate_interruption"
    )
    assert check.evaluator == "deterministic"
    assert check.outcome == JudgeOutcome.FAIL
    assert check.metrics["agent_interrupts_count"] == 1.0


def test_zero_interruptions_pass_when_voice_metrics_are_available():
    import tau2.judges.quality.checkers as checkers

    metrics = InteractionQualityMetrics(
        voice=VoiceInteractionMetrics(
            counts=VoiceInteractionMetricCounts(
                n_simulations=1,
                response_total=0,
                yield_total=0,
                backchannel_total=0,
                vocal_tic_total=0,
                non_directed_total=0,
                agent_interrupts_count=0,
            )
        )
    )

    result = checkers.check_inappropriate_interruption(metrics, QualityJudgeSettings())

    assert result.outcome == JudgeOutcome.PASS
    assert result.metrics == {"agent_interrupts_count": 0.0}


def test_monologue_includes_internal_agent_pauses_until_caller_speaks():
    ticks = []
    for i in range(8):
        agent_chunk = None
        user_chunk = None
        if i in {0, 1, 5, 6}:
            agent_chunk = AssistantMessage.voice(content="a", contains_speech=True)
        if i == 7:
            user_chunk = UserMessage.voice(content="u", contains_speech=True)
        ticks.append(
            Tick(
                tick_id=i,
                timestamp=str(i),
                tick_duration_seconds=1.0,
                agent_chunk=agent_chunk,
                user_chunk=user_chunk,
            )
        )
    info = evaluate_quality(
        _sim(ticks=ticks),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=False, monologue_seconds=5.0),
    )
    check = next(check for check in info.factor_checks if check.id == "monologue")
    assert check.outcome == JudgeOutcome.FAIL
    assert check.metrics["max_agent_floor_hold_seconds"] == 7.0


def test_audio_transcript_preserves_turns_without_utterance_ids():
    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(content="first", contains_speech=True),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(content="reply", contains_speech=True),
        ),
        Tick(
            tick_id=2,
            timestamp="2",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(content="second", contains_speech=True),
        ),
    ]
    assert [
        (turn.speaker, turn.text) for turn in extract_spoken_turns(_sim(ticks=ticks))
    ] == [
        ("agent", "first"),
        ("caller", "reply"),
        ("agent", "second"),
    ]


def test_spoken_turns_flag_interruptions_but_keep_raw_text():
    # Dual detection, matching the nativeness judge: a markup gold whose
    # delivered/undelivered chunk split shows undelivered chunks, or the
    # runner's per-chunk raw_data.was_truncated on pre-markup runs. The turn
    # text itself stays raw — only LLM judge prompts render the marker.
    gold = (
        '<message uuid="abc" active="0">'
        "<chunk id=0>Su transcripció</chunk>"
        "<chunk id=1>n está lista.</chunk>"
        "</message>"
    )
    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="Su transcripció",
                contains_speech=True,
                audio_script_gold=gold,
                utterance_ids=["u1"],
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(content="wait!", contains_speech=True),
        ),
        Tick(
            tick_id=2,
            timestamp="2",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="claro",
                contains_speech=True,
                utterance_ids=["u2"],
                raw_data={"was_truncated": True},
            ),
        ),
        Tick(
            tick_id=3,
            timestamp="3",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="algo más?", contains_speech=True, utterance_ids=["u3"]
            ),
        ),
    ]
    turns = extract_spoken_turns(_sim(ticks=ticks))
    assert [(t.speaker, t.text, t.interrupted) for t in turns] == [
        ("agent", "Su transcripció", True),  # markup-gold undelivered chunk
        ("caller", "wait!", False),
        ("agent", "claro", True),  # pre-markup raw_data.was_truncated
        ("agent", "algo más?", False),
    ]


def test_tick_overlap_barge_in_flags_spoken_turn_without_chunk_signal():
    # Realtime cancellation: every emitted chunk stays "active" (plain
    # content, no undelivered gold chunks, no raw_data.was_truncated) — the
    # tick timeline is the ONLY record of the cut. The spoken turn must carry
    # interrupted=True and the judge transcript the marker.
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.quality.judge import _full_transcript

    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="Voy a leer su número de refer",
                contains_speech=True,
                utterance_ids=["u1"],
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="encia com", contains_speech=True, utterance_ids=["u1"]
            ),
            user_chunk=UserMessage.voice(content="no hace falta", contains_speech=True),
        ),
        Tick(
            tick_id=2,
            timestamp="2",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="claro", contains_speech=True, utterance_ids=["u2"]
            ),
        ),
    ]
    sim = _sim(ticks=ticks)
    turns = extract_spoken_turns(sim)
    # The barged span maps onto the turn by tick-range overlap (never
    # ordinal); the caller turn and the post-barge agent turn stay clean.
    assert [(t.speaker, t.interrupted) for t in turns] == [
        ("agent", True),
        ("caller", False),
        ("agent", False),
    ]
    transcript = _full_transcript(sim)
    assert (
        f"agent: Voy a leer su número de referencia com {INTERRUPTED_TURN_MARKER}"
        in transcript
    )
    assert f"claro {INTERRUPTED_TURN_MARKER}" not in transcript
    assert f"caller: no hace falta {INTERRUPTED_TURN_MARKER}" not in transcript
    # Raw turn text never carries the marker.
    assert all(INTERRUPTED_TURN_MARKER not in turn.text for turn in turns)


def test_quality_transcript_marks_interrupted_agent_turns_for_the_judge_only():
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.quality.judge import _full_transcript

    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="Perguntas apareceram trunca",
                contains_speech=True,
                utterance_ids=["u1"],
                raw_data={"was_truncated": True},
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(
                content="sim",
                contains_speech=True,
                raw_data={"was_truncated": True},
            ),
        ),
    ]
    sim = _sim(ticks=ticks)
    transcript = _full_transcript(sim)
    assert f"agent: Perguntas apareceram trunca {INTERRUPTED_TURN_MARKER}" in transcript
    # Only agent turns are marked: the marker names the caller as interrupter.
    assert f"caller: sim {INTERRUPTED_TURN_MARKER}" not in transcript
    assert "caller: sim" in transcript
    # Deterministic evidence keeps the raw text.
    assert all(
        INTERRUPTED_TURN_MARKER not in turn.text for turn in extract_spoken_turns(sim)
    )


def test_quality_prompt_explains_the_interruption_marker_and_requires_reasoning():
    prompt = build_quality_factor_prompt(
        _sim(messages=[]),
        _task(),
        QUALITY_FACTOR_BY_ID["unnecessary_repetition"],
        domain="mock",
    )
    assert '"[cut off by caller]"' in prompt
    assert "never violations" in prompt
    assert "reasoning must give a brief concrete explanation" in prompt


def test_evidence_free_violations_are_loud_errors_not_silent_fails(monkeypatch):
    # Fix for the calibration-corpus finding: a violated reply with empty
    # reasoning fails validation and surfaces as an ERROR outcome.
    import tau2.judges.base as base

    def fake_generate(*, model, messages, call_name, **kwargs):
        return SimpleNamespace(
            content=json.dumps(
                {"opportunity": True, "violated": True, "reasoning": "", "quote": ""}
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    info = evaluate_quality(
        _sim(
            messages=[
                AssistantMessage.text("Hello."),
                AssistantMessage.text("Hello again."),
            ]
        ),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=True, model="judge-model"),
    )
    check = next(c for c in info.factor_checks if c.id == "unnecessary_repetition")
    assert check.outcome == JudgeOutcome.ERROR
    assert "reasoning" in (check.evidence or "")


def test_silent_voice_run_has_no_timing_opportunity():
    ticks = [
        Tick(tick_id=i, timestamp=str(i), tick_duration_seconds=1.0) for i in range(4)
    ]
    info = evaluate_quality(
        _sim(ticks=ticks),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=False),
    )
    by_id = {check.id: check for check in info.factor_checks}
    assert by_id["responsiveness"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["yielding"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["backchannel_selectivity"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["vocal_tic_selectivity"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["non_directed_selectivity"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["monologue"].outcome == JudgeOutcome.NO_OPPORTUNITY


def test_complete_tau_voice_panel_maps_to_non_overlapping_quality_factors():
    import tau2.judges.quality.checkers as checkers

    metrics = InteractionQualityMetrics(
        voice=VoiceInteractionMetrics(
            response_latency_mean=4.0,
            yield_latency_mean=0.6,
            response_rate=0.5,
            yield_rate=0.5,
            agent_interruption_rate=0.5,
            selectivity_backchannel=0.5,
            selectivity_vocal_tic=0.5,
            selectivity_non_directed=0.5,
            counts=VoiceInteractionMetricCounts(
                n_simulations=1,
                response_total=2,
                yield_total=2,
                backchannel_total=2,
                vocal_tic_total=2,
                non_directed_total=2,
                agent_interrupts_count=1,
            ),
        )
    )
    settings = QualityJudgeSettings(response_latency_seconds=3.0)

    results = {
        factor_id: checkers.CHECKER_REGISTRY[factor_id](metrics, settings)
        for factor_id in (
            "responsiveness",
            "yielding",
            "inappropriate_interruption",
            "backchannel_selectivity",
            "vocal_tic_selectivity",
            "non_directed_selectivity",
        )
    }

    assert all(result.outcome == JudgeOutcome.FAIL for result in results.values())
    assert results["responsiveness"].metrics == {
        "response_rate": 0.5,
        "response_total": 2.0,
        "response_latency_mean": 4.0,
    }
    assert results["yielding"].metrics == {
        "yield_rate": 0.5,
        "yield_total": 2.0,
        "yield_latency_mean": 0.6,
    }


def test_semantic_judges_run_per_factor_with_factor_specific_effort(monkeypatch):
    import tau2.judges.base as base

    calls = []

    def fake_generate(*, model, messages, call_name, **kwargs):
        calls.append((model, call_name, messages, kwargs))
        factor_id = call_name.removeprefix("quality_judge_")
        payload = {
            "opportunity": True,
            "violated": factor_id == "unnecessary_tool_call",
            "reasoning": "extra lookup",
            "quote": "",
        }
        if factor_id == "unnecessary_tool_call":
            payload.update(
                {
                    "unnecessary_call": "c2: diagnose",
                    "correct_path_without_call": "Use the successful identity lookup.",
                }
            )
        return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(base, "generate", fake_generate)
    info = evaluate_quality(
        _sim(
            messages=[
                UserMessage.text("My phone is 5551234"),
                AssistantMessage.text("Let me check."),
                AssistantMessage.text(
                    "I am checking now.",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="get_customer_by_phone",
                            arguments={"phone_number": "5550000"},
                        )
                    ],
                ),
                ToolMessage(id="c1", role="tool", content="No records found"),
                AssistantMessage.text(
                    "One more check.",
                    tool_calls=[
                        ToolCall(id="c2", name="diagnose", arguments={"id": "bad"})
                    ],
                ),
                ToolMessage(id="c2", role="tool", content="bad id", error=True),
            ]
        ),
        _task(),
        domain="telecom",
        settings=QualityJudgeSettings(llm_judge=True, model="judge-model"),
    )
    assert {call[1] for call in calls} == {
        "quality_judge_unnecessary_repetition",
        "quality_judge_agent_caused_tool_error",
        "quality_judge_auth_arg_mismatch",
        "quality_judge_incorrect_tool_parameters",
        "quality_judge_unnecessary_tool_call",
    }
    by_call_name = {call[1]: call for call in calls}
    assert (
        by_call_name["quality_judge_unnecessary_repetition"][3]["reasoning_effort"]
        == "none"
    )
    assert (
        by_call_name["quality_judge_unnecessary_tool_call"][3]["reasoning_effort"]
        == "xhigh"
    )
    assert all(call[0] == "judge-model" for call in calls)
    by_id = {check.id: check for check in info.factor_checks}
    assert by_id["unnecessary_tool_call"].outcome == JudgeOutcome.FAIL
    assert by_id["incorrect_tool_parameters"].outcome == JudgeOutcome.PASS
    assert by_id["unnecessary_tool_call"].judge_model == "judge-model"
    assert by_id["unnecessary_tool_call"].judge_args == {"reasoning_effort": "xhigh"}
    assert (
        by_id["unnecessary_tool_call"].judge_prompt_version
        == QUALITY_JUDGE_PROMPT_VERSION
    )
    assert info.num_fail == 1
    assert (
        info.num_pass
        + info.num_fail
        + info.num_no_opportunity
        + info.num_deferred
        + info.num_errors
        == len(info.factor_checks)
    )
    assert info.score_coverage == pytest.approx(5 / 12)


def test_quality_prompt_is_concise_and_protects_necessary_order_discovery():
    sim = _sim(
        messages=[
            UserMessage.text(
                "I need to return the blue shirt, but I do not know which order."
            ),
            AssistantMessage.text(
                "I will check your recent orders.",
                tool_calls=[
                    ToolCall(
                        id="c1", name="get_order_details", arguments={"order_id": "o1"}
                    )
                ],
            ),
            ToolMessage(id="c1", role="tool", content="No blue shirt"),
        ]
    )
    prompt = build_quality_factor_prompt(
        sim,
        _task(),
        QUALITY_FACTOR_BY_ID["unnecessary_tool_call"],
        domain="retail",
    )

    assert _SYSTEM_PROMPT == (
        "You evaluate process quality in AI customer-service calls. Return JSON only."
    )
    assert "multiple candidate orders" in prompt
    assert "lookups needed to identify that order" in prompt
    assert "not hindsight" in prompt
    assert "shortest correct" not in prompt
    assert "FACTOR-SPECIFIC" not in prompt
    assert "INTERACTION METRICS" not in prompt


def test_quality_reply_schema_uses_process_quality_language():
    violated = QualityFactorReply.model_json_schema()["properties"]["violated"]

    assert "process-quality criterion" in violated["description"]
    assert "non-natively" not in violated["description"]


def test_unnecessary_call_reply_coerces_structured_tool_descriptions():
    # Judges at high reasoning effort sometimes return the offending call as a
    # structured object instead of prose; that is usable evidence, not a
    # validation error.
    reply = UnnecessaryToolCallReply.model_validate(
        {
            "opportunity": True,
            "violated": True,
            "reasoning": "the transfer was avoidable",
            "unnecessary_call": {
                "tool_name": "transfer_to_human_agents",
                "arguments": {"summary": "requires human handling"},
            },
            "correct_path_without_call": "Resolve with the refund tool.",
        }
    )
    assert "transfer_to_human_agents" in reply.unnecessary_call
    assert reply.correct_path_without_call == "Resolve with the refund tool."


def test_quality_task_contract_excludes_evaluator_only_reference_actions():
    task = Task(
        id="retail-1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="retail",
                reason_for_call="Return a blue shirt.",
                known_info="The shirt was in a recent order.",
                unknown_info="The caller does not know the order id.",
                task_instructions="Ask the agent to find and return the shirt.",
            )
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="gold-1",
                    name="gold_reference_action",
                    arguments={"order_id": "secret-order"},
                )
            ]
        ),
    )

    request = build_quality_judge_input(
        _sim(messages=[]),
        task,
        QUALITY_FACTOR_BY_ID["unnecessary_tool_call"],
        domain="retail",
    )
    contract = request.task_contract.model_dump_json(exclude_none=True)

    assert "Return a blue shirt" in contract
    assert "recent order" in contract
    assert "does not know the order id" in contract
    assert "find and return the shirt" in contract
    assert "gold_reference_action" not in contract
    assert "secret-order" not in contract


def test_semantic_parse_failure_is_recorded_not_raised(monkeypatch):
    import tau2.judges.base as base

    monkeypatch.setattr(
        base,
        "generate",
        lambda **_kwargs: SimpleNamespace(content='{"verdicts": []}'),
    )
    info = evaluate_quality(
        _sim(
            messages=[
                AssistantMessage.text("Checking."),
                AssistantMessage.text(
                    "Still checking.",
                    tool_calls=[ToolCall(id="c1", name="lookup", arguments={})],
                ),
                ToolMessage(id="c1", role="tool", content="ok"),
            ]
        ),
        _task(),
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=True),
    )
    semantic = [
        check
        for check in info.factor_checks
        if check.evaluator in {"llm", "hybrid"}
        and check.outcome != JudgeOutcome.NO_OPPORTUNITY
    ]
    assert semantic
    assert all(check.outcome == JudgeOutcome.ERROR for check in semantic)
    assert info.num_errors == len(semantic)
