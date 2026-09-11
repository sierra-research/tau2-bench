# Copyright Sierra
"""Delivery judge: audio→WAV conversion, response model, harness aggregation."""

import base64
import io
import json
import wave
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tau2.judges.base as judge_base
from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    DeliveryFactorCheck,
    DeliveryFinding,
    DeliveryJudgeSettings,
    DeliveryUtteranceResult,
    JudgeOutcome,
    SimulationRun,
)
from tau2.judges.delivery.audio import audio_data_to_wav_b64, message_audio_to_wav_b64
from tau2.judges.delivery.factors import DeliveryFactorConfig, DeliveryRubric
from tau2.judges.delivery.harness import (
    _delivered_reference,
    evaluate_delivery,
)
from tau2.judges.delivery.judge import (
    DeliveryJudgeFinding,
    DeliveryJudgeResponse,
    render_language_rubric,
    run_delivery_judge,
)
from tau2.metrics.interaction_quality import barge_in_utterance_indices


def _pack_rubric():
    return DeliveryRubric(
        language="pl",
        display_name="Polish",
        source="pack",
        factors=[
            DeliveryFactorConfig(
                id="word_pronunciation",
                category="pronunciation_fidelity",
                severity=3,
                listen_for="hallucinated word forms",
            ),
            DeliveryFactorConfig(
                id="digit_readout",
                category="digit_readout_delivery",
                severity=3,
                listen_for="garbled longer numbers",
                shadow=True,
            ),
            DeliveryFactorConfig(
                id="lexical_stress",
                category="lexical_stress",
                severity=2,
                listen_for="disabled factor, never rendered",
                enabled=False,
            ),
        ],
    )


def _fallback_rubric(language="hi", display_name="Hindi"):
    return DeliveryRubric(
        language=language, display_name=display_name, source="fallback"
    )


# --- audio conversion ------------------------------------------------------


def _ulaw_audio(n_samples=800, rate=8000):
    return AudioData(
        data=b"\x7f" * n_samples,
        format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=rate, channels=1),
    )


def test_ulaw_audio_to_valid_wav_b64():
    b64 = audio_data_to_wav_b64(_ulaw_audio(rate=8000))
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2  # μ-law -> PCM_S16LE
        assert wav.getnframes() == 800


def test_message_audio_to_wav_b64_roundtrip_and_none():
    msg = AssistantMessage.voice(
        audio_content=audio_bytes_to_string(b"\x7f" * 400),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        audio_script_gold="hello",
    )
    assert message_audio_to_wav_b64(msg) is not None
    # No audio -> None
    assert message_audio_to_wav_b64(AssistantMessage.text("no audio here")) is None


# --- response model ----------------------------------------------------------


def test_finding_infers_axis_and_clamps_severity():
    f = DeliveryJudgeFinding.model_validate(
        {"category": "mispronunciation", "severity": 2}
    )
    assert f.axis == "fidelity" and f.severity == 2
    f = DeliveryJudgeFinding.model_validate(
        {"category": "unnatural_pause", "severity": 9}
    )
    assert f.axis == "intonation" and f.severity == 3  # inferred + clamped
    f = DeliveryJudgeFinding.model_validate(
        {"axis": "intonation", "category": "other", "severity": 0}
    )
    assert f.axis == "intonation" and f.severity == 1  # clamped up


def test_garbage_finding_fails_the_whole_response():
    # A finding no coercion can repair fails the RESPONSE (loud ERROR upstream)
    # instead of being silently dropped like the old _coerce_findings did.
    with pytest.raises(ValidationError):
        DeliveryJudgeResponse.model_validate({"findings": ["garbage"]})
    with pytest.raises(ValidationError):
        DeliveryJudgeResponse.model_validate({"findings": "not-a-list"})


# --- run_delivery_judge (mocked generate) ------------------------------------


def _mock_generate(monkeypatch, payload: dict, capture=None):
    def fake(model, messages, call_name=None, **kw):
        if capture is not None:
            capture.update(model=model, messages=messages, call_name=call_name)
        return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(judge_base, "generate", fake)


def test_run_delivery_judge_derives_verdict_from_findings(monkeypatch):
    """outcome/severity/flag come from findings only — an unarticulated
    'has_issue' without findings must not desync overall vs axis scores."""
    # Model claims an issue but articulates no findings -> PASS, severity 0.
    _mock_generate(monkeypatch, {"has_issue": True, "severity": 2, "findings": []})
    result = run_delivery_judge("UklGd0g=", "hello", utterance_idx=0)
    assert result.outcome == JudgeOutcome.PASS
    assert result.severity == 0 and not result.flag_for_review

    # Findings present -> FAIL with severity/flag derived from the worst finding.
    _mock_generate(
        monkeypatch,
        {
            "has_issue": False,  # contradicts findings; findings win
            "severity": 0,
            "findings": [
                {"axis": "intonation", "category": "unnatural_pause", "severity": 2}
            ],
        },
    )
    result = run_delivery_judge("UklGd0g=", "hello", utterance_idx=1)
    assert result.outcome == JudgeOutcome.FAIL
    assert result.severity == 2 and result.flag_for_review
    assert result.findings[0].axis == "intonation"


def test_run_delivery_judge_sends_audio_multipart(monkeypatch):
    capture: dict = {}
    _mock_generate(monkeypatch, {"findings": []}, capture=capture)
    run_delivery_judge("UklGd0g=", "hello", utterance_idx=0, rubric=_fallback_rubric())
    assert capture["call_name"] == "delivery_judge"
    user_msg = capture["messages"][1]
    assert user_msg.audio_content == "UklGd0g="  # audio rides the generate() seam
    assert "hello" in user_msg.content


# --- language-specific prompt rendering (pack / fallback / generic) ----------


def test_prompt_pack_mode_renders_enabled_factors_only():
    block = render_language_rubric(_pack_rubric())
    assert "source: language pack" in block
    assert "Polish (pl)" in block
    assert "factor_id: word_pronunciation (severity 3)" in block
    assert "garbled longer numbers" in block
    # Disabled factors are never shown to the judge.
    assert "lexical_stress" not in block and "disabled factor" not in block


def test_prompt_fallback_mode_is_language_specific_fixed_text():
    """No pack factors -> the fixed native-listener template, parameterized ONLY
    on the language name/code — never a generic prompt."""
    block = render_language_rubric(_fallback_rubric())
    assert "source: fallback" in block
    assert "AS A NATIVE LISTENER of Hindi" in block
    assert "typical text-to-speech failure modes for Hindi" in block
    assert 'Return an empty "factor_checks" list' in block
    # Only the language differs between two fallback renders.
    other = render_language_rubric(
        _fallback_rubric(language="ru", display_name="Russian")
    )
    assert block.replace("Hindi (hi)", "X").replace("Hindi", "X") == other.replace(
        "Russian (ru)", "X"
    ).replace("Russian", "X")


def test_prompt_generic_mode_has_no_rubric():
    assert render_language_rubric(DeliveryRubric()) is None


def test_user_prompt_carries_rubric_mode(monkeypatch):
    capture: dict = {}
    _mock_generate(monkeypatch, {"findings": []}, capture=capture)
    run_delivery_judge("UklGd0g=", "hello", utterance_idx=0, rubric=_pack_rubric())
    assert "source: language pack" in capture["messages"][1].content

    _mock_generate(monkeypatch, {"findings": []}, capture=capture)
    run_delivery_judge("UklGd0g=", "hello", utterance_idx=0, rubric=_fallback_rubric())
    assert "source: fallback" in capture["messages"][1].content

    _mock_generate(monkeypatch, {"findings": []}, capture=capture)
    run_delivery_judge("UklGd0g=", "hello", utterance_idx=0)  # no rubric at all
    content = capture["messages"][1].content
    assert "source:" not in content and "(not provided)" in content


def test_user_prompt_uses_regional_locale_when_available(monkeypatch):
    capture: dict = {}
    rubric = _pack_rubric().model_copy(update={"locale": "ES-MD"})
    _mock_generate(monkeypatch, {"findings": []}, capture=capture)
    run_delivery_judge("UklGd0g=", "hola", utterance_idx=0, rubric=rubric)
    assert "Target locale/language:\nES-MD" in capture["messages"][1].content


# --- factor checks (pack mode) ------------------------------------------------


def test_factor_checks_mapped_onto_requested_factors(monkeypatch):
    """Every requested factor gets exactly one check; verdicts map to
    PASS/FAIL/NO_OPPORTUNITY; a factor the judge skipped is a loud ERROR; a
    hallucinated factor_id is dropped (closed catalog, no un-reviewed axes)."""
    _mock_generate(
        monkeypatch,
        {
            "findings": [],
            "factor_checks": [
                {
                    "factor_id": "word_pronunciation",
                    "opportunity": True,
                    "violated": True,
                    "reasoning": "heard a non-word",
                    "quote": "identyfikator",
                },
                # digit_readout omitted by the judge -> ERROR for that factor
                {
                    "factor_id": "made_up_factor",
                    "opportunity": True,
                    "violated": True,
                    "reasoning": "hallucinated",
                },
            ],
        },
    )
    result = run_delivery_judge(
        "UklGd0g=", "hello", utterance_idx=0, rubric=_pack_rubric()
    )
    by_id = {c.id: c for c in result.factor_checks}
    assert set(by_id) == {"word_pronunciation", "digit_readout"}  # enabled only
    assert by_id["word_pronunciation"].outcome == JudgeOutcome.FAIL
    assert by_id["word_pronunciation"].evidence == "heard a non-word"
    assert by_id["word_pronunciation"].quote == "identyfikator"
    assert by_id["word_pronunciation"].category == "pronunciation_fidelity"
    assert by_id["word_pronunciation"].severity == 3
    assert by_id["digit_readout"].outcome == JudgeOutcome.ERROR
    # Factor checks never alter the axis verdict (findings are empty -> PASS).
    assert result.outcome == JudgeOutcome.PASS and result.severity == 0


def test_factor_checks_no_opportunity_and_stringly_bools(monkeypatch):
    _mock_generate(
        monkeypatch,
        {
            "findings": [],
            "factor_checks": [
                {"factor_id": "word_pronunciation", "opportunity": "false"},
                {
                    "factor_id": "digit_readout",
                    "opportunity": "true",
                    "violated": "no",
                },
            ],
        },
    )
    result = run_delivery_judge(
        "UklGd0g=", "hello", utterance_idx=0, rubric=_pack_rubric()
    )
    by_id = {c.id: c for c in result.factor_checks}
    assert by_id["word_pronunciation"].outcome == JudgeOutcome.NO_OPPORTUNITY
    assert by_id["digit_readout"].outcome == JudgeOutcome.PASS


def test_factor_checks_empty_outside_pack_mode(monkeypatch):
    """Fallback mode never produces factor checks, even if the judge invents
    some (there are no factor ids to score against)."""
    _mock_generate(
        monkeypatch,
        {
            "findings": [],
            "factor_checks": [
                {
                    "factor_id": "word_pronunciation",
                    "opportunity": True,
                    "violated": True,
                    "reasoning": "mispronounced brand name",
                }
            ],
        },
    )
    result = run_delivery_judge(
        "UklGd0g=", "hello", utterance_idx=0, rubric=_fallback_rubric()
    )
    assert result.factor_checks == []


def test_run_delivery_judge_invalid_reply_raises(monkeypatch):
    monkeypatch.setattr(
        judge_base,
        "generate",
        lambda *a, **k: SimpleNamespace(content="not json"),
    )
    with pytest.raises(ValueError):
        run_delivery_judge("UklGd0g=", "hello", utterance_idx=0)


# --- harness ---------------------------------------------------------------


def _voice_sim(n_utterances=3, sim_id="sim-1"):
    ticks = []
    for i in range(n_utterances):
        chunk = AssistantMessage.voice(
            audio_content=audio_bytes_to_string(b"\x7f" * 400),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            audio_script_gold=f"utterance {i}",
            utterance_ids=[f"u{i}"],  # distinct -> each is its own merged message
        )
        ticks.append(Tick(tick_id=i, timestamp=f"t{i}", agent_chunk=chunk))
    return SimulationRun(
        id=sim_id,
        task_id="task-1",
        start_time="s",
        end_time="e",
        duration=1.0,
        termination_reason="user_stop",
        ticks=ticks,
    )


def _fake_judge(severity_by_idx, axis="fidelity"):
    def _judge(wav_b64, expected_text, *, utterance_idx, **kwargs):
        sev = severity_by_idx.get(utterance_idx, 0)
        findings = (
            [DeliveryFinding(axis=axis, category="other", severity=sev)] if sev else []
        )
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            expected_text=expected_text,
            outcome=JudgeOutcome.FAIL if sev else JudgeOutcome.PASS,
            flag_for_review=sev >= 2,
            severity=sev,
            findings=findings,
        )

    return _judge


def test_non_voice_run_returns_none():
    sim = SimulationRun(
        id="x",
        task_id="t",
        start_time="s",
        end_time="e",
        duration=1.0,
        termination_reason="user_stop",
    )
    assert evaluate_delivery(sim, "es") is None


def test_harness_aggregates_scores(monkeypatch):
    # 3 utterances: one clean, one sev-2 fidelity, one sev-3 fidelity.
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge",
        _fake_judge({1: 2, 2: 3}, axis="fidelity"),
    )
    info = evaluate_delivery(
        _voice_sim(3), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert info is not None
    assert info.num_judged == 3
    assert info.num_flagged == 2
    # fidelity cleanliness: 1.0, 1-2/3, 1-3/3 -> mean = (1 + .3333 + 0)/3
    assert info.fidelity_score == pytest.approx((1 + 1 / 3 + 0) / 3)
    # no intonation findings -> perfect intonation
    assert info.intonation_score == pytest.approx(1.0)
    assert info.score == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_harness_records_error_without_aborting(monkeypatch):
    def _judge(wav_b64, expected_text, *, utterance_idx, **kwargs):
        if utterance_idx == 1:
            raise ValueError("boom")  # invalid reply
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            outcome=JudgeOutcome.PASS,
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    info = evaluate_delivery(
        _voice_sim(3), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert len(info.utterance_results) == 3
    errors = [r for r in info.utterance_results if r.outcome == JudgeOutcome.ERROR]
    assert len(errors) == 1 and "boom" in errors[0].summary
    assert info.num_judged == 3  # every call sent to the judge, ERROR included
    assert info.num_errors == 1  # ...but ERRORs are broken out and not scored
    assert info.score == pytest.approx(1.0)


def test_harness_api_failure_recorded_as_error(monkeypatch):
    def _judge(wav_b64, expected_text, *, utterance_idx, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    info = evaluate_delivery(
        _voice_sim(2), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert info.num_errors == 2


def test_silence_only_utterances_are_not_judged(monkeypatch):
    """Discrete-time padding ticks (silence audio, no gold/content) must not be
    sent to the judge — they waste budget and would skew aggregates."""
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _fake_judge({})
    )
    sim = _voice_sim(2)
    silence = AssistantMessage.voice(
        content=None,
        audio_content=audio_bytes_to_string(b"\x7f" * 400),  # μ-law silence
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        audio_script_gold=None,
        contains_speech=False,
    )
    sim.ticks.append(Tick(tick_id=2, timestamp="t2", agent_chunk=silence))
    info = evaluate_delivery(sim, "es", settings=DeliveryJudgeSettings(sample_rate=1.0))
    assert info.num_judged == 2  # the 2 speech utterances only
    assert all(r.expected_text for r in info.utterance_results)


def test_harness_aggregates_factor_checks_and_score(monkeypatch):
    """Per-utterance factor verdicts fold into one sim-level check per factor
    (FAIL beats PASS beats ERROR beats NO_OPPORTUNITY) with an equal-weight binary
    factor_score (excluding shadow factors), and the rubric source is stamped
    as provenance."""
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.build_delivery_rubric",
        lambda language, locale=None: _pack_rubric(),
    )

    def _judge(wav_b64, expected_text, *, utterance_idx, rubric, **kwargs):
        # utt 0: word FAIL + digit NO_OPPORTUNITY; utt 1: word PASS + digit PASS.
        if utterance_idx == 0:
            checks = [
                DeliveryFactorCheck(
                    id="word_pronunciation",
                    category="pronunciation_fidelity",
                    severity=3,
                    outcome=JudgeOutcome.FAIL,
                    evidence="non-word heard",
                    quote="archeśnika",
                ),
                DeliveryFactorCheck(
                    id="digit_readout",
                    category="digit_readout_delivery",
                    severity=3,
                    outcome=JudgeOutcome.NO_OPPORTUNITY,
                ),
            ]
        else:
            checks = [
                DeliveryFactorCheck(
                    id="word_pronunciation",
                    category="pronunciation_fidelity",
                    severity=3,
                    outcome=JudgeOutcome.PASS,
                ),
                DeliveryFactorCheck(
                    id="digit_readout",
                    category="digit_readout_delivery",
                    severity=3,
                    outcome=JudgeOutcome.PASS,
                ),
            ]
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            outcome=JudgeOutcome.PASS,
            factor_checks=checks,
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    info = evaluate_delivery(
        _voice_sim(2), "pl", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert info.rubric_source == "pack"
    by_id = {c.id: c for c in info.factor_checks}
    # FAIL on any utterance fails the factor at sim level (evidence carried).
    assert by_id["word_pronunciation"].outcome == JudgeOutcome.FAIL
    assert by_id["word_pronunciation"].quote == "archeśnika"
    assert by_id["digit_readout"].outcome == JudgeOutcome.PASS
    assert by_id["digit_readout"].shadow
    # The digit factor is recorded in shadow but excluded: scored word FAIL -> 0.
    assert info.factor_score == pytest.approx(0.0)
    # Axis scores are untouched by factor verdicts.
    assert info.score == pytest.approx(1.0)


def test_harness_fallback_rubric_source_and_no_factor_channel(monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.build_delivery_rubric",
        lambda language, locale=None: _fallback_rubric(),
    )
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _fake_judge({})
    )
    info = evaluate_delivery(
        _voice_sim(2), "hi", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert info.rubric_source == "fallback"
    assert info.factor_checks == [] and info.factor_score is None


def test_harness_builds_rubric_once_and_passes_it(monkeypatch):
    """The rubric is built once per sim and rides every judge call."""
    calls = {"built": 0, "seen": []}

    def _build(language, locale=None):
        calls["built"] += 1
        return _pack_rubric()

    def _judge(wav_b64, expected_text, *, utterance_idx, rubric, **kwargs):
        calls["seen"].append(rubric.source)
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx, outcome=JudgeOutcome.PASS
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.build_delivery_rubric", _build)
    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    evaluate_delivery(
        _voice_sim(3), "pl", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert calls["built"] == 1
    assert calls["seen"] == ["pack", "pack", "pack"]


def test_max_segments_cap_spreads_evenly(monkeypatch):
    """The cost cap keeps a deterministic even spread across the WHOLE call
    (first + last always in), not the first N utterances (opening bias)."""
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _fake_judge({})
    )
    info = evaluate_delivery(
        _voice_sim(5),
        "es",
        settings=DeliveryJudgeSettings(sample_rate=1.0, max_segments=2),
    )
    assert info.num_judged == 2
    assert [r.utterance_idx for r in info.utterance_results] == [0, 4]

    info3 = evaluate_delivery(
        _voice_sim(9),
        "es",
        settings=DeliveryJudgeSettings(sample_rate=1.0, max_segments=3),
    )
    assert [r.utterance_idx for r in info3.utterance_results] == [0, 4, 8]
    # Deterministic: the same sim yields the same selection on a re-run.
    info3_again = evaluate_delivery(
        _voice_sim(9),
        "es",
        settings=DeliveryJudgeSettings(sample_rate=1.0, max_segments=3),
    )
    assert [r.utterance_idx for r in info3_again.utterance_results] == [0, 4, 8]


def test_delivery_judge_config_stamped_only_when_judged(monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _fake_judge({})
    )
    from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION

    judged = evaluate_delivery(
        _voice_sim(2), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert judged.judge_model is not None
    assert judged.judge_prompt_version == DELIVERY_JUDGE_PROMPT_VERSION
    # 'es' carries a seeded pack delivery rubric; its source is recorded.
    assert judged.rubric_source == "pack"

    # A sampled voice sim with only silence: nothing sent to the judge -> no
    # judge model/prompt config for a judge that never ran.
    sim = _voice_sim(1)
    sim.ticks[0].agent_chunk.audio_script_gold = None
    sim.ticks[0].agent_chunk.content = None
    sim.ticks[0].agent_chunk.contains_speech = False
    unjudged = evaluate_delivery(
        sim, "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert unjudged is not None and unjudged.num_judged == 0
    assert unjudged.judge_model is None and unjudged.judge_args is None
    assert unjudged.judge_prompt_version is None
    assert unjudged.rubric_source is None  # no call, no rubric recorded


def test_attach_delivery_populates_info(monkeypatch):
    """attach_delivery end-to-end: voice sim gets delivery_info (with language
    resolved for context); a non-voice sim stays None."""
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_delivery

    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge",
        _fake_judge({1: 2}, axis="intonation"),
    )
    sim = _voice_sim(2)
    sim.speech_environment = SpeechEnvironment(language="es")
    attach_delivery(sim, settings=DeliveryJudgeSettings(sample_rate=1.0))
    assert sim.delivery_info is not None
    assert sim.delivery_info.language == "es"
    assert sim.delivery_info.num_judged == 2
    assert sim.delivery_info.intonation_score == pytest.approx((1 + 1 / 3) / 2)

    text_sim = SimulationRun(
        id="text",
        task_id="t",
        start_time="s",
        end_time="e",
        duration=1.0,
        termination_reason="user_stop",
    )
    attach_delivery(text_sim)
    assert text_sim.delivery_info is None


def test_per_conversation_sampling_is_deterministic(monkeypatch):
    """Sampling is per CONVERSATION: a sampled sim judges ALL its utterances;
    an unsampled sim returns None (drops out of aggregates) — never a noisy
    partial estimate."""
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _fake_judge({})
    )
    infos = [
        evaluate_delivery(
            _voice_sim(3, sim_id=f"sim-{i}"),
            "es",
            settings=DeliveryJudgeSettings(sample_rate=0.3),
        )
        for i in range(40)
    ]
    sampled = [info for info in infos if info is not None]
    assert 0 < len(sampled) < 40  # a strict subset of conversations
    assert all(info.num_judged == 3 for info in sampled)  # full coverage within
    # Deterministic: the same sims are sampled on a re-run.
    infos_again = [
        evaluate_delivery(
            _voice_sim(3, sim_id=f"sim-{i}"),
            "es",
            settings=DeliveryJudgeSettings(sample_rate=0.3),
        )
        for i in range(40)
    ]
    assert [i is not None for i in infos] == [i is not None for i in infos_again]


# --- delivered reference (markup gold / interruption) -----------------------


_MARKUP_GOLD = (
    '<message uuid="abc" active="0,1">'
    "<chunk id=0>Hello, your ref</chunk>"
    "<chunk id=1>und is on the way</chunk>"
    "<chunk id=2> and will arrive Friday.</chunk>"
    "</message>"
)


def _markup_msg(gold):
    return AssistantMessage.voice(
        audio_content=audio_bytes_to_string(b"\x7f" * 400),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        audio_script_gold=gold,
        utterance_ids=["u0"],
    )


def test_delivered_reference_plain_text_passthrough():
    ref, interrupted = _delivered_reference(_markup_msg("just plain text"))
    assert ref == "just plain text" and interrupted is False


def test_delivered_reference_markup_active_subset_is_interrupted():
    # Only chunks 0,1 delivered; chunk 2 cut off by the caller.
    ref, interrupted = _delivered_reference(_markup_msg(_MARKUP_GOLD))
    assert ref == "Hello, your refund is on the way"  # chunks joined w/o separator
    assert interrupted is True
    assert "<chunk" not in ref  # never leak markup to the judge


def test_delivered_reference_markup_all_active_not_interrupted():
    gold = _MARKUP_GOLD.replace('active="0,1"', 'active="0,1,2"')
    ref, interrupted = _delivered_reference(_markup_msg(gold))
    assert ref == "Hello, your refund is on the way and will arrive Friday."
    assert interrupted is False


def test_delivered_reference_legacy_raw_truncation_is_interrupted():
    msg = _markup_msg("clipped transcripció")
    msg.raw_data = {"was_truncated": True}

    ref, interrupted = _delivered_reference(msg)

    assert ref == "clipped transcripció"
    assert interrupted is True


def test_harness_passes_plain_reference_and_interruption_to_judge(monkeypatch):
    captured = {}

    def _judge(wav_b64, expected_text, *, utterance_idx, was_interrupted, **kwargs):
        captured[utterance_idx] = (expected_text, was_interrupted)
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx, outcome=JudgeOutcome.PASS
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    sim = _voice_sim(1)
    sim.ticks[0].agent_chunk.audio_script_gold = _MARKUP_GOLD
    evaluate_delivery(sim, "es", settings=DeliveryJudgeSettings(sample_rate=1.0))
    ref, interrupted = captured[0]
    assert ref == "Hello, your refund is on the way"
    assert interrupted is True


# ---------------------------------------------------------------------------
# Tick-level barge-in detection (provider-cancelled tails stay marked "active"
# in the chunk accounting, so the tick timeline is the only interruption
# signal). ALL caller speech overlapping the utterance's speech tail counts —
# provider VAD cancels on collision starts and backchannels too.
# ---------------------------------------------------------------------------


def _agent_tick(tick_id, uid, contains_speech=True):
    chunk = AssistantMessage.voice(
        audio_content=audio_bytes_to_string(b"\x7f" * 400),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        audio_script_gold=f"utterance {uid}",
        utterance_ids=[uid],
        contains_speech=contains_speech,
    )
    # 1s ticks -> a 2-tick grace window, so small fixtures can sit on either
    # side of the tail-grace boundary.
    return Tick(
        tick_id=tick_id,
        timestamp=f"t{tick_id}",
        tick_duration_seconds=1.0,
        agent_chunk=chunk,
    )


def _user_speech(action=None):
    return UserMessage(
        role="user",
        content="wait, actually",
        contains_speech=True,
        turn_taking_action=TurnTakingAction(action=action) if action else None,
    )


def _ticks_sim(ticks, sim_id="sim-bargein"):
    return SimulationRun(
        id=sim_id,
        task_id="task-1",
        start_time="s",
        end_time="e",
        duration=1.0,
        termination_reason="user_stop",
        ticks=ticks,
    )


def test_barge_in_marks_the_airing_utterance():
    # u0 airs over ticks 0-2; the caller starts speaking at tick 2 while the
    # agent chunk still carries speech, and the agent stops right after. u1
    # (tick 4) is untouched.
    ticks = [_agent_tick(0, "u0"), _agent_tick(1, "u0"), _agent_tick(2, "u0")]
    ticks[2].user_chunk = _user_speech(action="interrupt")
    ticks.append(Tick(tick_id=3, timestamp="t3", user_chunk=_user_speech()))
    ticks.append(_agent_tick(4, "u1"))
    assert barge_in_utterance_indices(_ticks_sim(ticks)) == {0}


def test_collision_start_still_marks_the_cut_utterance():
    # The caller began speaking one tick BEFORE the agent's utterance (no
    # caller segment ever *starts* during agent speech), keeps talking over
    # its whole airing, and the agent yields mid-word — observed in stored
    # runs; the segment-start interruption signal misses it.
    ticks = [Tick(tick_id=0, timestamp="t0", user_chunk=_user_speech())]
    for tick_id in (1, 2):
        tick = _agent_tick(tick_id, "u0")
        tick.user_chunk = _user_speech()
        ticks.append(tick)
    assert barge_in_utterance_indices(_ticks_sim(ticks)) == {0}


def test_backchannel_overlap_at_tail_marks_the_cut_utterance():
    # Provider VAD cancels on backchannels too ("mm-hm" over the tail, agent
    # audio stops mid-word) — the turn-taking classification is ignored.
    ticks = [_agent_tick(0, "u0"), _agent_tick(1, "u0"), _agent_tick(2, "u0")]
    ticks[2].user_chunk = _user_speech(action="backchannel")
    assert barge_in_utterance_indices(_ticks_sim(ticks)) == {0}


def test_caller_speech_after_agent_finished_is_not_a_barge_in():
    # Normal turn-taking: the caller starts on a tick with no agent speech.
    ticks = [_agent_tick(0, "u0"), _agent_tick(1, "u0")]
    ticks.append(Tick(tick_id=2, timestamp="t2", user_chunk=_user_speech()))
    assert barge_in_utterance_indices(_ticks_sim(ticks)) == set()


def test_overlap_the_agent_talks_through_is_not_a_barge_in():
    # The caller overlaps at tick 0 but the agent keeps speaking well past the
    # grace window (1s ticks -> 2-tick grace): the
    # provider did not cancel, so the full fidelity check stays on.
    ticks = [_agent_tick(i, "u0") for i in range(6)]
    ticks[0].user_chunk = _user_speech()
    assert barge_in_utterance_indices(_ticks_sim(ticks)) == set()


def test_harness_ors_tick_barge_in_into_was_interrupted(monkeypatch):
    # All chunks "active" (plain script gold -> chunk accounting says NOT
    # interrupted), but the tick timeline shows the barge-in cut.
    captured = {}

    def _judge(wav_b64, expected_text, *, utterance_idx, was_interrupted, **kwargs):
        captured[utterance_idx] = was_interrupted
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            was_interrupted=was_interrupted,
            outcome=JudgeOutcome.PASS,
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    ticks = [_agent_tick(0, "u0"), _agent_tick(2, "u1")]
    ticks[0].user_chunk = _user_speech(action="interrupt")
    # Trailing agent-silent tick: the recording outlives u1, so only the
    # barge-in signal (never the recording-end rule) can set u1's flag.
    ticks.append(Tick(tick_id=3, timestamp="t3", tick_duration_seconds=1.0))
    sim = _ticks_sim(ticks)
    info = evaluate_delivery(sim, "es", settings=DeliveryJudgeSettings(sample_rate=1.0))
    assert info is not None and info.num_judged == 2
    assert captured == {0: True, 1: False}
    stored = {u.utterance_idx: u.was_interrupted for u in info.utterance_results}
    assert stored == {0: True, 1: False}


def test_harness_preserves_legacy_raw_truncation_across_merged_chunks(monkeypatch):
    captured = {}

    def _judge(wav_b64, expected_text, *, utterance_idx, was_interrupted, **kwargs):
        captured[utterance_idx] = was_interrupted
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            was_interrupted=was_interrupted,
            outcome=JudgeOutcome.PASS,
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    ticks = [_agent_tick(0, "u0"), _agent_tick(1, "u0")]
    assert ticks[0].agent_chunk is not None
    assert ticks[1].agent_chunk is not None
    ticks[0].agent_chunk.content = "clipped transcrip"
    ticks[1].agent_chunk.content = "ció"
    ticks[1].agent_chunk.raw_data = {"was_truncated": True}
    # Ensure neither the tick-overlap nor recording-end fallback can supply
    # the flag: raw_data on the final original chunk is the only signal.
    ticks.append(Tick(tick_id=2, timestamp="t2", tick_duration_seconds=1.0))

    info = evaluate_delivery(
        _ticks_sim(ticks), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )

    assert info is not None and info.num_judged == 1
    assert captured == {0: True}
    assert info.utterance_results[0].was_interrupted is True


def test_harness_flags_recording_end_cut_as_interrupted(monkeypatch):
    # A call that stops (step ceiling / hang-up) while the agent is still
    # talking clamps the audio at the tail with no barge-in anywhere. The
    # utterance whose tick span reaches the recording's final tick gets the
    # interrupted tolerance: the missing tail is the call ending, not a
    # synthesis defect (measured as the largest fidelity FP class on the
    # recall_20 calibration gold).
    captured = {}

    def _judge(wav_b64, expected_text, *, utterance_idx, was_interrupted, **kwargs):
        captured[utterance_idx] = was_interrupted
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            was_interrupted=was_interrupted,
            outcome=JudgeOutcome.PASS,
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    # No trailing tick: u1 is still speaking in the recording's final tick.
    ticks = [_agent_tick(0, "u0"), _agent_tick(2, "u1")]
    info = evaluate_delivery(
        _ticks_sim(ticks), "es", settings=DeliveryJudgeSettings(sample_rate=1.0)
    )
    assert info is not None and info.num_judged == 2
    # u0 ends mid-recording; u1's span reaches the final tick.
    assert captured == {0: False, 1: True}


def test_delivered_reference_strips_stop_token():
    # transfer_to_human appends ###STOP### to content; it is never spoken and
    # must not reach the judge as expected speech.
    ref, interrupted = _delivered_reference(
        _markup_msg("Your SIM card is locked. ###STOP###")
    )
    assert ref == "Your SIM card is locked."
    assert interrupted is False
