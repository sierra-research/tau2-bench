# Copyright Sierra
"""Execution-unit and aggregation contracts for the nativeness judge."""

import json
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tau2.judges.base as base
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.judges.nativeness.caller_identity import korean_caller_name
from tau2.judges.nativeness.checkers import (
    BACKCHANNEL_FREQUENCY_SPECS,
    CheckerContext,
    check_backchannel_frequency,
)
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.nativeness.harness import (
    aggregate_utterance_results,
    build_agent_turns,
    evaluate_nativeness,
)
from tau2.judges.nativeness.judge import NativenessJudgeResult


def _sim(messages):
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        messages=messages,
    )


def _task():
    return Task(
        id="t1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="help",
                task_instructions="help",
            )
        ),
    )


def _result(*, opportunity=True, violated=False, severity=0, quote=""):
    return NativenessJudgeResult(
        factor_id="translationese",
        opportunity=opportunity,
        violated=violated,
        severity=severity,
        reasoning="reason",
        quote=quote,
    )


def test_agent_turns_keep_only_the_immediately_preceding_user_turn():
    turns = build_agent_turns(
        _sim(
            [
                UserMessage(role="user", content="hola"),
                AssistantMessage(role="assistant", content="buenas tardes"),
                UserMessage(role="user", content="necesito ayuda"),
                AssistantMessage(role="assistant", content="dame el código"),
            ]
        )
    )

    assert [turn.text for turn in turns] == ["buenas tardes", "dame el código"]
    assert [turn.preceding_user_text for turn in turns] == ["hola", "necesito ayuda"]
    assert [turn.index for turn in turns] == [0, 1]


def test_voice_agent_turn_has_preceding_delivered_user_utterance():
    sim = _sim([])
    sim.messages = None
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            user_chunk=UserMessage.voice(
                content="necesito ayuda", utterance_ids=["u1"]
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            agent_chunk=AssistantMessage.voice(
                content="deme el código", utterance_ids=["a1"]
            ),
        ),
    ]

    turns = build_agent_turns(sim)
    assert [(turn.text, turn.preceding_user_text) for turn in turns] == [
        ("deme el código", "necesito ayuda")
    ]


def test_factor_execution_levels_match_the_reviewed_split():
    es = {factor.id: factor for factor in judge_factors_for("es")}
    assert es["register_formality"].evaluation_level == "deterministic"
    assert es["register_formality"].type == "es_register_formality"
    assert es["register_formality"].params is None
    assert "regional_consistency" not in es
    assert es["gender_agreement"].evaluation_level == "utterance"
    assert es["gender_agreement"].type == "judge"
    assert es["gender_agreement"].precheck_type == "es_gender_agreement"
    assert es["gender_agreement"].params is not None
    assert es["natural_word_choice"].precheck_type is None
    assert es["natural_word_choice"].aggregation == "any"
    assert "translationese" not in es
    assert "verb_morphology" not in es

    pt = {factor.id: factor for factor in judge_factors_for("pt")}
    assert "brazilian_variety" not in pt
    assert pt["regional_consistency"].evaluation_level == "call"
    assert pt["regional_consistency"].precheck_type == "pt_regional_consistency"
    assert pt["gender_agreement"].precheck_type == "pt_gender_agreement"
    assert pt["natural_word_choice"].precheck_type is None

    ko = {factor.id: factor for factor in judge_factors_for("ko")}
    assert ko["honorific_levels"].evaluation_level == "deterministic"
    assert ko["honorific_levels"].type == "ko_honorific_levels"
    assert ko["honorific_levels"].params is None
    assert "subject_honorific" not in ko
    assert ko["honorific_agreement"].evaluation_level == "utterance"
    assert ko["honorific_agreement"].precheck_type == "ko_honorific_agreement"
    assert ko["request_politeness"].evaluation_level == "utterance"
    assert ko["request_politeness"].aggregation == "any"
    assert ko["name_address_conventions"].evaluation_level == "utterance"
    assert ko["name_address_conventions"].precheck_type == "ko_name_address_conventions"
    assert not ko["name_address_conventions"].shadow
    assert ko["natural_word_choice"].precheck_type is None
    assert ko["counting_units"].precheck_type == "ko_counting_units"

    hi = {factor.id: factor for factor in judge_factors_for("hi")}
    assert hi["register_formality"].type == "hi_register_formality"
    assert hi["register_formality"].params is None
    assert hi["gender_agreement"].evaluation_level == "utterance"
    assert hi["gender_agreement"].precheck_type == "hi_gender_agreement"
    assert hi["honorific_agreement"].precheck_type == "hi_honorific_agreement"
    assert not hi["gender_agreement"].shadow
    assert hi["natural_word_choice"].precheck_type is None

    zh = {factor.id: factor for factor in judge_factors_for("zh")}
    assert zh["register_formality"].type == "zh_register_formality"
    assert zh["register_formality"].params is None
    assert zh["regional_consistency"].precheck_type == "zh_regional_consistency"
    assert zh["natural_word_choice"].precheck_type is None
    assert zh["counting_units"].precheck_type == "zh_counting_units"

    pt = {factor.id: factor for factor in judge_factors_for("pt")}
    assert pt["register_formality"].type == "judge"
    assert pt["register_formality"].evaluation_level == "call"
    assert pt["register_formality"].params is not None

    old_ids = {
        "code_switching",
        "hinglish_codeswitch",
        "untranslated_terms",
        "measure_words",
    }
    for language in ("es", "hi", "ko", "pt", "zh"):
        ids = {factor.id for factor in judge_factors_for(language)}
        assert "natural_word_choice" in ids
        assert ids.isdisjoint(old_ids)

    assert "counters" not in {factor.id for factor in judge_factors_for("ko")}


@pytest.mark.parametrize(
    ("language", "acknowledgment"),
    [
        ("en", "Okay, I can help with that."),
        ("hi", "जी, मैं अभी जाँच करता हूँ।"),
        ("ko", "네, 지금 확인하겠습니다."),
        ("zh", "好的，我来查一下。"),
        ("pt", "Certo, vou verificar."),
        ("es", "Claro, voy a revisarlo."),
    ],
)
def test_backchannel_frequency_counts_attached_acknowledgments(
    language, acknowledgment
):
    outcome, evidence = check_backchannel_frequency(
        CheckerContext(
            agent_text=acknowledgment,
            agent_turns=[acknowledgment],
            user_turns=["I need help", "The booking failed"],
            user_speech_seconds=60,
            has_email=False,
            language=language,
        )
    )

    assert outcome == JudgeOutcome.PASS
    assert "1.00/min" in (evidence or "")


def test_backchannel_frequency_enforces_opportunity_and_rate_bounds():
    base_context = dict(
        agent_text="Okay",
        user_turns=["I need help", "The booking failed"],
        has_email=False,
        language="en",
    )
    no_duration, _ = check_backchannel_frequency(
        CheckerContext(
            **base_context,
            agent_turns=["Okay"],
            user_speech_seconds=None,
        )
    )
    too_short, _ = check_backchannel_frequency(
        CheckerContext(
            **base_context,
            agent_turns=["Okay"],
            user_speech_seconds=29.99,
        )
    )
    fail_low, low_evidence = check_backchannel_frequency(
        CheckerContext(
            **base_context,
            agent_turns=[],
            user_speech_seconds=60,
        )
    )
    fail_high, high_evidence = check_backchannel_frequency(
        CheckerContext(
            **base_context,
            agent_turns=["Okay"] * 7,
            user_speech_seconds=60,
        )
    )

    assert no_duration == too_short == JudgeOutcome.NO_OPPORTUNITY
    assert fail_low == fail_high == JudgeOutcome.FAIL
    assert (low_evidence or "").startswith("fail-low")
    assert (high_evidence or "").startswith("fail-high")


def test_backchannel_frequency_requires_two_substantive_user_turns():
    spec = BACKCHANNEL_FREQUENCY_SPECS["en"]
    assert "yes" in spec.acknowledgments
    outcome, _ = check_backchannel_frequency(
        CheckerContext(
            agent_text="Okay",
            agent_turns=["Okay"],
            user_turns=["yes", "I need help"],
            user_speech_seconds=60,
            has_email=False,
            language="en",
        )
    )
    assert outcome == JudgeOutcome.NO_OPPORTUNITY


def test_backchannel_frequency_upper_bound_is_inclusive_and_one_per_turn():
    outcome, evidence = check_backchannel_frequency(
        CheckerContext(
            agent_text="yes okay right",
            agent_turns=["yes okay right"] * 6,
            user_turns=["I need help", "The booking failed"],
            user_speech_seconds=60,
            has_email=False,
            language="en",
        )
    )
    assert outcome == JudgeOutcome.PASS
    assert "6 acknowledgment turn(s)" in (evidence or "")


def test_backchannel_frequency_uses_tick_aligned_user_speech_in_harness():
    ticks = [
        Tick(
            tick_id=index,
            timestamp=str(index),
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(content="I need help. ", utterance_ids=["u1"]),
        )
        for index in range(15)
    ]
    ticks.append(
        Tick(
            tick_id=15,
            timestamp="15",
            agent_chunk=AssistantMessage.voice(content="Okay.", utterance_ids=["a1"]),
        )
    )
    ticks.extend(
        Tick(
            tick_id=index,
            timestamp=str(index),
            tick_duration_seconds=1.0,
            user_chunk=UserMessage.voice(
                content="The booking failed. ", utterance_ids=["u2"]
            ),
        )
        for index in range(16, 31)
    )
    ticks.append(
        Tick(
            tick_id=31,
            timestamp="31",
            agent_chunk=AssistantMessage.voice(content="I see.", utterance_ids=["a2"]),
        )
    )
    sim = _sim([])
    sim.messages = None
    sim.ticks = ticks

    info = evaluate_nativeness(
        sim,
        _task(),
        "en",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=False),
    )

    assert info is not None
    check = next(
        check for check in info.factor_checks if check.id == "backchannel_frequency"
    )
    assert check.outcome == JudgeOutcome.PASS
    assert "4.00/min" in (check.evidence or "")


def test_judge_result_enforces_opportunity_violation_and_severity_contract():
    no_opp = _result(opportunity=False, violated=False, severity=0)
    assert no_opp.severity == 0

    with pytest.raises(ValidationError):
        _result(opportunity=False, violated=True, severity=2)
    with pytest.raises(ValidationError):
        _result(opportunity=True, violated=False, severity=1)
    with pytest.raises(ValidationError):
        _result(opportunity=True, violated=True, severity=0)


def test_any_aggregation_fails_on_one_local_violation():
    aggregated = aggregate_utterance_results(
        [
            _result(opportunity=True, violated=False, severity=0),
            _result(opportunity=True, violated=True, severity=2, quote="bad"),
        ],
        strategy="any",
    )
    assert aggregated.outcome == JudgeOutcome.FAIL
    assert aggregated.severity == 2
    assert aggregated.violation_count == 1
    assert aggregated.quote == "bad"


def test_dose_aggregation_requires_one_severe_or_repeated_mild_evidence():
    one_mild = aggregate_utterance_results(
        [_result(opportunity=True, violated=True, severity=1, quote="mild")],
        strategy="dose",
    )
    assert one_mild.outcome == JudgeOutcome.PASS
    assert one_mild.severity == 0
    assert one_mild.violation_count == 1

    repeated = aggregate_utterance_results(
        [
            _result(opportunity=True, violated=True, severity=1, quote="first"),
            _result(opportunity=True, violated=True, severity=1, quote="second"),
        ],
        strategy="dose",
    )
    assert repeated.outcome == JudgeOutcome.FAIL
    assert repeated.severity == 2
    assert repeated.violation_count == 2

    severe = aggregate_utterance_results(
        [_result(opportunity=True, violated=True, severity=3, quote="severe")],
        strategy="dose",
    )
    assert severe.outcome == JudgeOutcome.FAIL
    assert severe.severity == 3


def test_aggregation_preserves_no_opportunity():
    aggregated = aggregate_utterance_results(
        [_result(opportunity=False, violated=False, severity=0)],
        strategy="any",
    )
    assert aggregated.outcome == JudgeOutcome.NO_OPPORTUNITY
    assert aggregated.severity == 0
    assert aggregated.violation_count == 0


def test_any_aggregation_counts_every_violated_utterance():
    aggregated = aggregate_utterance_results(
        [
            _result(opportunity=True, violated=True, severity=1, quote="one"),
            _result(opportunity=True, violated=False, severity=0),
            _result(opportunity=True, violated=True, severity=2, quote="two"),
            _result(opportunity=True, violated=True, severity=1, quote="three"),
        ],
        strategy="any",
    )
    assert aggregated.outcome == JudgeOutcome.FAIL
    assert aggregated.violation_count == 3


def test_runtime_isolates_combined_naturalness_per_agent_utterance(monkeypatch):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    info = evaluate_nativeness(
        _sim(
            [
                UserMessage(role="user", content="hola"),
                AssistantMessage(role="assistant", content="buenas tardes"),
                UserMessage(role="user", content="necesito ayuda"),
                AssistantMessage(role="assistant", content="deme el código"),
            ]
        ),
        _task(),
        "es",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert info is not None
    assert len(prompts) == 4  # two utterances x (combined naturalness + other factors)
    call_prompts = [prompt for prompt in prompts if "all agent turns" in prompt]
    utterance_prompts = [
        prompt for prompt in prompts if "one agent utterance" in prompt
    ]
    assert call_prompts == []
    assert len(utterance_prompts) == 4
    batches = [
        list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        for prompt in utterance_prompts
    ]
    assert sum(batch == ["natural_word_choice"] for batch in batches) == 2
    assert all(
        batch == ["natural_word_choice"] or "natural_word_choice" not in batch
        for batch in batches
    )
    assert "register_formality" not in "\n".join(prompts)
    assert sum("hola" in prompt for prompt in utterance_prompts) == 2
    assert sum("necesito ayuda" in prompt for prompt in utterance_prompts) == 2
    assert all("CUSTOMER CONTEXT" in prompt for prompt in utterance_prompts)


def test_combined_naturalness_failure_does_not_poison_other_utterance_factors(
    monkeypatch,
):
    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        if factor_ids == ["natural_word_choice"]:
            raise RuntimeError("combined judge unavailable")
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    info = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="मैं आपकी मदद करूँगा।")]),
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert info is not None
    checks = {check.id: check for check in info.factor_checks}
    assert checks["natural_word_choice"].outcome == JudgeOutcome.ERROR
    assert checks["name_address_conventions"].outcome == JudgeOutcome.PASS
    assert checks["name_address_conventions"].unit_results


def test_hybrid_gender_precheck_skips_only_a_definite_failure(monkeypatch):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    # First assistant message = the injected opener (pinned, never scored by
    # the gender precheck); the model-authored second turn carries the
    # definite contradiction.
    info = evaluate_nativeness(
        _sim(
            [
                AssistantMessage(role="assistant", content="Olá! Como posso ajudar?"),
                UserMessage(role="user", content="preciso de ajuda"),
                AssistantMessage(role="assistant", content="Estou pronto para ajudar."),
            ]
        ),
        _task(),
        "pt",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
        agent_gender="female",
    )

    assert info is not None
    agreement = next(c for c in info.factor_checks if c.id == "gender_agreement")
    assert agreement.outcome == JudgeOutcome.FAIL
    assert agreement.evaluation_level == "utterance"
    assert agreement.unit_results == []
    assert "gender-v3" in (agreement.evidence or "")
    assert "gender_agreement" not in "\n".join(prompts)
    assert "natural_word_choice" in "\n".join(prompts)


def test_hybrid_gender_unresolved_case_reaches_llm(monkeypatch):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    info = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="Estou pronta para ajudar.")]),
        _task(),
        "pt",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
        agent_gender="female",
    )

    assert info is not None
    agreement = next(c for c in info.factor_checks if c.id == "gender_agreement")
    assert agreement.outcome == JudgeOutcome.PASS
    assert agreement.unit_results
    assert "gender_agreement" in "\n".join(prompts)


def test_portuguese_regional_precheck_skips_fail_but_sends_unresolved_to_llm(
    monkeypatch,
):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    failed = evaluate_nativeness(
        _sim(
            [
                AssistantMessage(
                    role="assistant", content="Abra o ficheiro, por favor."
                ),
                AssistantMessage(role="assistant", content="Confira no telemóvel."),
            ]
        ),
        _task(),
        "pt",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert failed is not None
    regional = next(c for c in failed.factor_checks if c.id == "regional_consistency")
    assert regional.outcome == JudgeOutcome.FAIL
    assert regional.evaluation_level == "call"
    assert regional.unit_results == []
    assert "regional_consistency" not in "\n".join(prompts)

    prompts.clear()
    unresolved = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="Use o celular.")]),
        _task(),
        "pt",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert unresolved is not None
    regional = next(
        c for c in unresolved.factor_checks if c.id == "regional_consistency"
    )
    assert regional.outcome == JudgeOutcome.PASS
    assert regional.unit_results
    assert "regional_consistency" in "\n".join(prompts)


def test_hindi_combined_natural_word_choice_always_returns_utterance_units(monkeypatch):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    failed = evaluate_nativeness(
        _sim(
            [
                AssistantMessage(
                    role="assistant",
                    content="आपकी request पर वापस आते हुए।",
                )
            ]
        ),
        _task(),
        "hi",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert failed is not None
    choices = [
        check for check in failed.factor_checks if check.id == "natural_word_choice"
    ]
    assert len(choices) == 1
    assert choices[0].outcome == JudgeOutcome.PASS
    assert choices[0].unit_results
    assert "natural_word_choice" in "\n".join(prompts)
    naturalness_batches = [
        list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        for prompt in prompts
    ]
    assert ["natural_word_choice"] in naturalness_batches
    assert all(
        batch == ["natural_word_choice"]
        for batch in naturalness_batches
        if "natural_word_choice" in batch
    )

    prompts.clear()
    unresolved = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="booking तैयार है।")]),
        _task(),
        "hi",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert unresolved is not None
    choices = [
        check for check in unresolved.factor_checks if check.id == "natural_word_choice"
    ]
    assert len(choices) == 1
    assert choices[0].outcome == JudgeOutcome.PASS
    assert choices[0].unit_results
    assert "natural_word_choice" in "\n".join(prompts)

    deferred = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="booking तैयार है।")]),
        _task(),
        "hi",
        "latn",
        settings=NativenessJudgeSettings(llm_judge=False),
    )
    assert deferred is not None
    choices = [
        check for check in deferred.factor_checks if check.id == "natural_word_choice"
    ]
    assert len(choices) == 1
    assert choices[0].outcome == JudgeOutcome.DEFERRED


def test_hybrid_honorific_agreement_skips_fail_but_falls_back_once(monkeypatch):
    import tau2.judges.nativeness.harness as harness

    prompts = []

    # Korean honorific agreement is enabled in production; Hindi retains the
    # same implementation but remains disabled. Enable both locally so this
    # shared hybrid-checker test covers each language.
    configured_factors_for = harness.judge_factors_for

    def factors_with_honorific(language):
        return [
            factor.model_copy(update={"enabled": True})
            if factor.id == "honorific_agreement"
            else factor
            for factor in configured_factors_for(language)
        ]

    monkeypatch.setattr(harness, "judge_factors_for", factors_with_honorific)

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    monkeypatch.setattr(base, "generate", fake_generate)
    failed = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="आप तैयार है।")]),
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert failed is not None
    checks = [
        check for check in failed.factor_checks if check.id == "honorific_agreement"
    ]
    assert len(checks) == 1
    assert checks[0].outcome == JudgeOutcome.FAIL
    assert checks[0].unit_results == []
    assert "honorific_agreement" not in "\n".join(prompts)

    prompts.clear()
    unresolved = evaluate_nativeness(
        _sim(
            [
                AssistantMessage(
                    role="assistant",
                    content="제가 고객님께서 신청하신 예약을 확인하겠습니다.",
                )
            ]
        ),
        _task(),
        "ko",
        "hang",
        settings=NativenessJudgeSettings(llm_judge=True),
    )

    assert unresolved is not None
    checks = [
        check for check in unresolved.factor_checks if check.id == "honorific_agreement"
    ]
    assert len(checks) == 1
    assert checks[0].outcome == JudgeOutcome.PASS
    assert checks[0].unit_results
    assert "honorific_agreement" in "\n".join(prompts)

    deferred = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="제가 확인하겠습니다.")]),
        _task(),
        "ko",
        "hang",
        settings=NativenessJudgeSettings(llm_judge=False),
    )
    assert deferred is not None
    checks = [
        check for check in deferred.factor_checks if check.id == "honorific_agreement"
    ]
    assert len(checks) == 1
    assert checks[0].outcome == JudgeOutcome.DEFERRED


def test_hybrid_ko_name_address_skips_fail_but_sends_unresolved_to_llm(monkeypatch):
    prompts = []

    def fake_generate(model, messages, call_name=None, **kwargs):
        prompt = messages[1].content
        prompts.append(prompt)
        factor_ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
                if not factor_id.startswith("<")
            )
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "fine",
                            "quote": "",
                        }
                        for factor_id in factor_ids
                    ]
                }
            )
        )

    name = korean_caller_name("Minjun", "Kim")
    assert name is not None
    monkeypatch.setattr(base, "generate", fake_generate)
    failed = evaluate_nativeness(
        _sim(
            [
                AssistantMessage(
                    role="assistant",
                    content="민준 김 고객님, 잠시만 기다려 주세요.",
                )
            ]
        ),
        _task(),
        "ko",
        "hang",
        settings=NativenessJudgeSettings(llm_judge=True),
        caller_name=name,
    )

    assert failed is not None
    check = next(
        check
        for check in failed.factor_checks
        if check.id == "name_address_conventions"
    )
    assert check.outcome == JudgeOutcome.FAIL
    assert check.unit_results == []
    assert "name_address_conventions" not in "\n".join(prompts)

    prompts.clear()
    unresolved = evaluate_nativeness(
        _sim([AssistantMessage(role="assistant", content="김민준 님, 확인했습니다.")]),
        _task(),
        "ko",
        "hang",
        settings=NativenessJudgeSettings(llm_judge=True),
        caller_name=name,
    )
    assert unresolved is not None
    check = next(
        check
        for check in unresolved.factor_checks
        if check.id == "name_address_conventions"
    )
    assert check.outcome == JudgeOutcome.PASS
    assert check.unit_results
    assert "name_address_conventions" in "\n".join(prompts)
