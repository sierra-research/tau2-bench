# Copyright Sierra
"""LLM nativeness judge: verdict parsing, prompt contents, error policy.

All tests mock ``generate`` at the judge-call seam — no real API calls. The
judge parses replies through ``judges.base.judge_structured`` into the shared
``VerdictReplyBase``, so these tests exercise the real parse/validate path.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import tau2.judges.base as base
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.judges.nativeness.harness import evaluate_nativeness
from tau2.judges.nativeness.judge import (
    NativenessJudgeCriterion,
    NativenessJudgeInput,
    run_nativeness_judge,
)

PASS = JudgeOutcome.PASS
FAIL = JudgeOutcome.FAIL
NONE = JudgeOutcome.NO_OPPORTUNITY
DEFERRED = JudgeOutcome.DEFERRED


def _request(**over):
    base_kwargs = dict(
        agent_text="नमस्ते, मैं आपकी मदद कर सकता हूँ",
        language="hi",
        evaluation_level="utterance",
        criteria=[
            NativenessJudgeCriterion(
                factor_id="register_formality",
                question="Did the agent keep the आप form?",
                opportunity="agent addresses the customer",
                positive_examples="consistent आप",
                negative_examples="slips to तुम",
            )
        ],
        agent_context="The speaker is an AI customer-service voice agent. Agent gender: female.",
    )
    base_kwargs.update(over)
    return NativenessJudgeInput(**base_kwargs)


def _mock_generate(monkeypatch, payload: dict, capture: dict | None = None):
    """Mock generate at the base seam (the one place all judges call through)."""

    def fake(model, messages, call_name=None, **kw):
        if capture is not None:
            capture["model"] = model
            capture["messages"] = messages
            capture["call_name"] = call_name
        return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(base, "generate", fake)


def _mock_batched_results(monkeypatch, *, failing_factor: str | None = None):
    """Reply once per runtime batch with exactly the factor ids in its prompt."""

    def fake(model, messages, call_name=None, **kw):
        ids = list(
            dict.fromkeys(
                factor_id
                for factor_id in re.findall(
                    r'"factor_id":\s*"([^"]+)"', messages[1].content
                )
                if not factor_id.startswith("<")
            )
        )
        results = []
        for factor_id in ids:
            violated = factor_id == failing_factor
            results.append(
                {
                    "factor_id": factor_id,
                    "opportunity": True,
                    "violated": violated,
                    "severity": 2 if violated else 0,
                    "reasoning": "x" if violated else "fine",
                    "quote": "कुछ टेक्स्ट" if violated else "",
                }
            )
        return SimpleNamespace(content=json.dumps({"results": results}))

    monkeypatch.setattr(base, "generate", fake)


# --- settings gate ----------------------------------------------------------


def test_judge_model_settings_are_used(monkeypatch):
    capture: dict = {}
    _mock_generate(
        monkeypatch,
        {
            "results": [
                {
                    "factor_id": "register_formality",
                    "opportunity": True,
                    "violated": False,
                    "severity": 0,
                }
            ]
        },
        capture,
    )
    run_nativeness_judge(
        _request(), model="my-judge-model", model_args={"reasoning_effort": "low"}
    )
    assert capture["model"] == "my-judge-model"
    assert capture["call_name"] == "nativeness_judge_batch"


# --- verdict parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {
                "results": [
                    {
                        "factor_id": "register_formality",
                        "opportunity": True,
                        "violated": True,
                        "severity": 3,
                        "reasoning": "drifted to तुम",
                        "quote": "तुम कहाँ हो",
                    }
                ],
            },
            FAIL,
        ),
        (
            {
                "results": [
                    {
                        "factor_id": "register_formality",
                        "opportunity": True,
                        "violated": False,
                        "severity": 0,
                        "reasoning": "fine",
                    }
                ]
            },
            PASS,
        ),
        (
            {
                "results": [
                    {
                        "factor_id": "register_formality",
                        "opportunity": False,
                        "violated": False,
                        "severity": 0,
                        "reasoning": "n/a",
                    }
                ]
            },
            NONE,
        ),
    ],
)
def test_verdict_parsing(monkeypatch, payload, expected):
    _mock_generate(monkeypatch, payload)
    result = run_nativeness_judge(_request(agent_text="नमस्ते, तुम कहाँ हो"))[0]
    outcome = NONE if not result.opportunity else FAIL if result.violated else PASS
    assert outcome == expected
    if expected == FAIL:
        assert result.reasoning
        assert result.quote == "तुम कहाँ हो"
    else:
        assert not result.quote


def test_quote_empty_on_diffuse_fail(monkeypatch):
    # A FAIL with no localizable span (diffuse factor) returns quote=None, not "".
    _mock_generate(
        monkeypatch,
        {
            "results": [
                {
                    "factor_id": "register_formality",
                    "opportunity": True,
                    "violated": True,
                    "severity": 2,
                    "reasoning": "translationese",
                    "quote": "",
                }
            ],
        },
    )
    result = run_nativeness_judge(_request(evaluation_level="call"))[0]
    assert result.violated
    assert result.quote == ""


def test_prompt_includes_rubric_and_agent_context(monkeypatch):
    capture: dict = {}
    _mock_generate(
        monkeypatch,
        {
            "results": [
                {
                    "factor_id": "register_formality",
                    "opportunity": True,
                    "violated": False,
                    "severity": 0,
                }
            ]
        },
        capture,
    )
    run_nativeness_judge(_request())
    assert capture["messages"][0].content == (
        "You evaluate language use in an AI agent's speech. Return JSON only."
    )
    user_prompt = capture["messages"][1].content
    assert "consistent आप" in user_prompt
    assert "slips to तुम" in user_prompt
    assert "Did the agent keep the आप form?" in user_prompt
    assert "shared_allowed" not in user_prompt
    assert "language_criterion" not in user_prompt
    assert set(_request().criteria[0].model_dump()) == {
        "factor_id",
        "question",
        "opportunity",
        "positive_examples",
        "negative_examples",
    }
    assert "Do not infer them from names" not in user_prompt
    assert "Agent gender: female" in user_prompt
    assert "नमस्ते" in user_prompt  # agent transcript


@pytest.mark.parametrize(
    "payload,expected",
    [
        # Some models emit string booleans; "false" must not read as truthy.
        (
            {
                "factor_id": "register_formality",
                "opportunity": "false",
                "violated": "false",
                "severity": 0,
            },
            NONE,
        ),
        (
            {
                "factor_id": "register_formality",
                "opportunity": "true",
                "violated": "false",
                "severity": 0,
            },
            PASS,
        ),
        (
            {
                "factor_id": "register_formality",
                "opportunity": "true",
                "violated": "true",
                "severity": 2,
                "reasoning": "x",
                "quote": "नमस्ते",
            },
            FAIL,
        ),
    ],
)
def test_string_json_flags_coerced(monkeypatch, payload, expected):
    _mock_generate(monkeypatch, {"results": [payload]})
    result = run_nativeness_judge(_request())[0]
    outcome = NONE if not result.opportunity else FAIL if result.violated else PASS
    assert outcome == expected


def test_fenced_reply_parses(monkeypatch):
    # A markdown-fenced reply must parse (fence-strip lives in judges.base).
    payload = json.dumps(
        {
            "results": [
                {
                    "factor_id": "register_formality",
                    "opportunity": True,
                    "violated": False,
                    "severity": 0,
                }
            ]
        }
    )

    def fake(model, messages, call_name=None, **kw):
        return SimpleNamespace(content=f"```json\n{payload}\n```")

    monkeypatch.setattr(base, "generate", fake)
    result = run_nativeness_judge(_request())[0]
    assert result.opportunity and not result.violated


def test_empty_transcript_no_opportunity(monkeypatch):
    # Should not even call the model.
    called = {"n": 0}

    def fake(*a, **k):
        called["n"] += 1
        return SimpleNamespace(content="{}")

    monkeypatch.setattr(base, "generate", fake)
    result = run_nativeness_judge(_request(agent_text="   "))[0]
    assert not result.opportunity
    assert called["n"] == 0


def test_judge_errors_propagate(monkeypatch):
    # Errors must surface (not be masked as NO_OPPORTUNITY) so issues are caught.
    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(base, "generate", boom)
    with pytest.raises(RuntimeError):
        run_nativeness_judge(_request())

    # A malformed (non-JSON) response raises ValueError.
    monkeypatch.setattr(
        base,
        "generate",
        lambda *a, **k: SimpleNamespace(content="not json"),
    )
    with pytest.raises(ValueError):
        run_nativeness_judge(_request())


# --- harness integration ----------------------------------------------------


def _sim(text):
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        messages=[AssistantMessage(role="assistant", content=text)],
    )


def _task():
    return Task(
        id="t1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline", reason_for_call="x", task_instructions="help"
            )
        ),
    )


def test_judge_factor_deferred_when_gate_off():
    info = evaluate_nativeness(
        _sim("कुछ टेक्स्ट"),
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=False),
    )
    wording = [c for c in info.factor_checks if c.id == "natural_word_choice"]
    assert wording and wording[0].outcome == DEFERRED
    # Judge never ran -> no judge configuration recorded.
    assert info.judge_model is None and info.judge_args is None
    assert info.num_errors == 0


def test_the_gate_is_off_unless_a_run_asks_for_it():
    # The judge is opt-in (2026-08-03): a production run should not pay for it,
    # and every collected pool turned it off explicitly when it was the default.
    # The tests above that exercise judge behaviour therefore have to say
    # llm_judge=True; a bare settings object judges nothing.
    assert NativenessJudgeSettings().llm_judge is False
    info = evaluate_nativeness(
        _sim("कुछ टेक्स्ट"), _task(), "hi", "deva", settings=NativenessJudgeSettings()
    )
    wording = [c for c in info.factor_checks if c.id == "natural_word_choice"]
    assert wording and wording[0].outcome == DEFERRED
    assert info.judge_model is None


def test_judge_error_recorded_not_crashed(monkeypatch):
    # A judge failure is recorded as ERROR (with the message) and does NOT crash
    # the run — so a batch post-hoc pass survives one flaky call.
    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(base, "generate", boom)
    info = evaluate_nativeness(
        _sim("कुछ टेक्स्ट JMO1MG"),
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=True),
    )
    wording = [c for c in info.factor_checks if c.id == "natural_word_choice"]
    assert wording and wording[0].outcome == JudgeOutcome.ERROR
    assert "api down" in (wording[0].evidence or "")
    # ERROR is excluded from the score. With no email-symbol opportunity, no
    # deterministic factor fires either, so the aggregate is intentionally None.
    assert info.score is None
    # ERROR checks are counted for the record.
    assert info.num_errors == len(
        [
            c
            for c in info.factor_checks
            if c.outcome == JudgeOutcome.ERROR and not c.shadow
        ]
    )
    assert info.num_errors >= 1


def test_invalid_reply_recorded_as_error(monkeypatch):
    # A reply that came back but failed validation is a loud ERROR outcome.
    monkeypatch.setattr(
        base,
        "generate",
        lambda *a, **k: SimpleNamespace(content="not json"),
    )
    info = evaluate_nativeness(
        _sim("कुछ टेक्स्ट"),
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=True),
    )
    errors = [
        c
        for c in info.factor_checks
        if c.outcome == JudgeOutcome.ERROR and not c.shadow
    ]
    assert errors and info.num_errors == len(errors)


def test_attach_nativeness_includes_known_agent_gender(monkeypatch):
    # The reviewed provider-voice catalog establishes grammatical-gender context.
    import tau2.judges.attach as attach_mod
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_nativeness

    captured: dict = {}

    def fake_eval(
        sim,
        task,
        language,
        script,
        *,
        settings,
        agent_context,
        agent_gender,
        caller_gender,
        caller_name,
    ):
        captured["agent_context"] = agent_context
        captured["agent_gender"] = agent_gender
        captured["caller_gender"] = caller_gender
        captured["caller_name"] = caller_name
        return None

    monkeypatch.setattr(attach_mod, "evaluate_nativeness", fake_eval)
    sim = _sim("कुछ टेक्स्ट")
    sim.speech_environment = SpeechEnvironment(
        language="hi", persona_tags={"gender": "female"}
    )
    sim.agent_provider = "xai"
    sim.agent_voice = "Rex"  # male
    # The agent context is built for the judge, so this needs the judge on.
    attach_nativeness(
        sim,
        _task(),
        settings=NativenessJudgeSettings(llm_judge=True),
        agent_provider="openai",  # fallback ignored
    )
    assert "AI customer-service voice agent" in captured["agent_context"]
    assert "agent gender: male" in captured["agent_context"].lower()
    assert "caller gender: female" in captured["agent_context"].lower()
    assert "do not infer" in captured["agent_context"].lower()
    assert captured["agent_gender"] == "male"
    assert captured["caller_gender"] == "female"
    assert captured["caller_name"] is None


def test_attach_nativeness_supplies_separately_labeled_caller_gender(monkeypatch):
    import tau2.judges.attach as attach_mod
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_nativeness

    captured: dict = {}

    def fake_eval(
        sim,
        task,
        language,
        script,
        *,
        settings,
        agent_context,
        agent_gender,
        caller_gender,
        caller_name,
    ):
        captured["agent_context"] = agent_context
        captured["agent_gender"] = agent_gender
        captured["caller_gender"] = caller_gender
        captured["caller_name"] = caller_name
        return None

    monkeypatch.setattr(attach_mod, "evaluate_nativeness", fake_eval)
    monkeypatch.setattr(
        attach_mod,
        "caller_gender_for_task",
        lambda task_id, language, domain: "female",
    )
    sim = _sim("कुछ टेक्स्ट")
    sim.speech_environment = SpeechEnvironment(language="hi")
    sim.agent_provider = "xai"
    sim.agent_voice = "Rex"
    attach_nativeness(
        sim,
        _task(),
        settings=NativenessJudgeSettings(llm_judge=True),
        domain="airline",
    )
    assert "Agent gender: male" in captured["agent_context"]
    assert "Caller gender: female" in captured["agent_context"]
    assert captured["agent_gender"] == "male"
    assert captured["caller_gender"] == "female"
    assert captured["caller_name"] is None


def test_attach_nativeness_supplies_typed_genders_when_llm_is_disabled(monkeypatch):
    import tau2.judges.attach as attach_mod
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_nativeness

    captured: dict = {}

    def fake_eval(
        sim,
        task,
        language,
        script,
        *,
        settings,
        agent_context,
        agent_gender,
        caller_gender,
        caller_name,
    ):
        captured.update(
            agent_context=agent_context,
            agent_gender=agent_gender,
            caller_gender=caller_gender,
            caller_name=caller_name,
        )
        return None

    monkeypatch.setattr(attach_mod, "evaluate_nativeness", fake_eval)
    sim = _sim("कुछ टेक्स्ट")
    sim.speech_environment = SpeechEnvironment(
        language="hi", persona_tags={"gender": "female"}
    )
    sim.agent_provider = "xai"
    sim.agent_voice = "Rex"

    attach_nativeness(sim, _task(), settings=NativenessJudgeSettings(llm_judge=False))

    assert captured == {
        "agent_context": None,
        "agent_gender": "male",
        "caller_gender": "female",
        "caller_name": None,
    }


def test_attach_nativeness_supplies_typed_caller_name_only_to_local_checker(
    monkeypatch,
):
    import tau2.judges.attach as attach_mod
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_nativeness
    from tau2.judges.nativeness.caller_identity import korean_caller_name

    name = korean_caller_name("Minjun", "Kim")
    assert name is not None
    captured: dict = {}

    def fake_eval(
        sim,
        task,
        language,
        script,
        *,
        settings,
        agent_context,
        agent_gender,
        caller_gender,
        caller_name,
    ):
        captured.update(agent_context=agent_context, caller_name=caller_name)
        return None

    monkeypatch.setattr(attach_mod, "evaluate_nativeness", fake_eval)
    monkeypatch.setattr(
        attach_mod,
        "resolve_korean_caller_name",
        lambda task, language, domain: name,
    )
    sim = _sim("민준 김 고객님, 잠시만 기다려 주세요.")
    sim.speech_environment = SpeechEnvironment(language="ko")
    attach_nativeness(
        sim,
        _task(),
        settings=NativenessJudgeSettings(llm_judge=False),
        domain="airline",
    )

    assert captured == {"agent_context": None, "caller_name": name}


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_attach_nativeness_resolves_actual_korean_identity_task_shapes(
    monkeypatch, domain
):
    import tau2.judges.attach as attach_mod
    from tau2.data_model.voice import SpeechEnvironment
    from tau2.judges.attach import attach_nativeness

    path = Path(f"data/tau2/multilingual/ko/{domain}_tasks_ko_identity.json")
    task = Task.model_validate(json.loads(path.read_text())[0])
    captured: dict = {}

    def fake_eval(
        sim,
        task,
        language,
        script,
        *,
        settings,
        agent_context,
        agent_gender,
        caller_gender,
        caller_name,
    ):
        captured["caller_name"] = caller_name
        return None

    monkeypatch.setattr(attach_mod, "evaluate_nativeness", fake_eval)
    sim = _sim("고객님, 확인했습니다.")
    sim.speech_environment = SpeechEnvironment(language="ko")
    attach_nativeness(
        sim,
        task,
        settings=NativenessJudgeSettings(llm_judge=False),
        domain=domain,
    )

    name = captured["caller_name"]
    assert name is not None
    assert name.full_spoken == f"{name.family_spoken}{name.given_spoken}"


def test_judge_factor_runs_when_gate_on(monkeypatch):
    _mock_batched_results(monkeypatch, failing_factor="natural_word_choice")
    settings = NativenessJudgeSettings(llm_judge=True, model="my-judge-model")
    info = evaluate_nativeness(
        _sim("कुछ टेक्स्ट"),
        _task(),
        "hi",
        "deva",
        settings=settings,
        agent_context="The speaker is an AI customer-service voice agent.",
    )
    wording = [c for c in info.factor_checks if c.id == "natural_word_choice"]
    assert wording and wording[0].outcome == FAIL
    # FAIL (sev 3) now contributes to the score alongside deterministic factors.
    assert info.score is not None
    # Judge configuration recorded from the settings that produced the verdicts.
    assert info.judge_model == "my-judge-model"
    assert info.judge_args == settings.model_args
    # Prompt version stamped (a retune between reruns must be visible).
    from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

    assert info.judge_prompt_version == NATIVENESS_JUDGE_PROMPT_VERSION


def test_empty_transcript_records_no_judge_config(monkeypatch):
    """Judge ON but nothing to judge: the factors record NO_OPPORTUNITY without
    any LLM call, so no judge model / prompt version is stamped (a judge that
    never ran must not claim one)."""

    def fail_generate(*a, **k):
        pytest.fail("no LLM call may happen on an empty transcript")

    monkeypatch.setattr(base, "generate", fail_generate)
    info = evaluate_nativeness(
        _sim(""),  # empty agent transcript
        _task(),
        "hi",
        "deva",
        settings=NativenessJudgeSettings(llm_judge=True),
    )
    judge_checks = [c for c in info.factor_checks if c.id in _judge_ids("hi")]
    assert judge_checks
    assert all(c.outcome == NONE for c in judge_checks)
    assert info.judge_model is None and info.judge_args is None
    assert info.judge_prompt_version is None


def _judge_ids(language):
    from tau2.judges.export import judge_factor_ids

    return judge_factor_ids(language)
