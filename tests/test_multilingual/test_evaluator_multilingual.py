# Copyright Sierra
"""Tests for language-aware evaluation (PR4).

- communicate_info: LLM judge auto-activates for non-English runs; English
  runs keep exact substring matching unless the judge is forced via
  parameter or the TAU2_FORCE_LLM_COMMUNICATE_JUDGE env var.
- nl_assertions: the judge prompt carries a language/script hint for
  non-English runs.
- English default behavior and result formats are unchanged.
"""

import json

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import (
    CommunicateCheck,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.data_model.voice import SpeechEnvironment
from tau2.evaluator import evaluator as evaluator_module
from tau2.evaluator import evaluator_communicate, evaluator_nl_assertions
from tau2.evaluator.evaluator import (
    EvaluationType,
    evaluate_simulation,
    get_simulation_language_info,
)
from tau2.evaluator.evaluator_communicate import (
    FORCE_LLM_COMMUNICATE_JUDGE_ENV,
    LLM_JUDGE_JUSTIFICATION_PREFIX,
    CommunicateEvaluator,
    should_use_llm_communicate_judge,
)
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

# ---- Helpers ----


def _trajectory():
    return [
        UserMessage(role="user", content="mera refund kab aayega?"),
        AssistantMessage(
            role="assistant",
            content="Aapka refund 5 din mein process ho jayega.",
        ),
    ]


def _english_trajectory():
    return [
        UserMessage(role="user", content="When will my refund arrive?"),
        AssistantMessage(
            role="assistant",
            content="Your refund will be processed in 5 business days.",
        ),
    ]


def _fake_generate(captured: list, met: bool = True, reasoning: str = "ok"):
    def fake(model, messages, call_name=None, **kwargs):
        captured.append(
            {
                "model": model,
                "messages": messages,
                "call_name": call_name,
                "kwargs": kwargs,
            }
        )
        return AssistantMessage(
            role="assistant",
            content=json.dumps({"reasoning": reasoning, "met": met}),
        )

    return fake


def _make_task(communicate_info=None, nl_assertions=None) -> Task:
    return Task(
        id="task-1",
        user_scenario=UserScenario(instructions="test"),
        evaluation_criteria=EvaluationCriteria(
            communicate_info=communicate_info or [],
            nl_assertions=nl_assertions or [],
        ),
    )


def _make_sim(speech_environment=None) -> SimulationRun:
    return SimulationRun(
        id="sim-1",
        task_id="task-1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=_trajectory(),
        trial=0,
        seed=42,
        speech_environment=speech_environment,
    )


# ---- should_use_llm_communicate_judge ----


class TestJudgeActivation:
    def test_default_english_no_judge(self):
        assert should_use_llm_communicate_judge(None) is False
        assert should_use_llm_communicate_judge("en") is False
        assert should_use_llm_communicate_judge("EN") is False

    def test_non_english_activates_judge(self):
        assert should_use_llm_communicate_judge("hi") is True
        assert should_use_llm_communicate_judge("zh") is True

    def test_explicit_param_wins(self):
        assert should_use_llm_communicate_judge(None, True) is True
        assert should_use_llm_communicate_judge("hi", False) is False

    def test_env_var_forces_judge(self, monkeypatch):
        monkeypatch.setenv(FORCE_LLM_COMMUNICATE_JUDGE_ENV, "1")
        assert should_use_llm_communicate_judge(None) is True
        assert should_use_llm_communicate_judge("en") is True

    def test_env_var_does_not_override_explicit_false(self, monkeypatch):
        monkeypatch.setenv(FORCE_LLM_COMMUNICATE_JUDGE_ENV, "true")
        assert should_use_llm_communicate_judge("hi", False) is False

    def test_env_var_off_values(self, monkeypatch):
        monkeypatch.setenv(FORCE_LLM_COMMUNICATE_JUDGE_ENV, "0")
        assert should_use_llm_communicate_judge(None) is False
        monkeypatch.setenv(FORCE_LLM_COMMUNICATE_JUDGE_ENV, "")
        assert should_use_llm_communicate_judge(None) is False


# ---- CommunicateEvaluator ----


class TestCommunicateEvaluator:
    def test_english_default_uses_substring(self, monkeypatch):
        """Default English path must not call any LLM and keep exact behavior."""

        def boom(*args, **kwargs):
            raise AssertionError("LLM judge must not be called for English default")

        monkeypatch.setattr(evaluator_communicate, "generate", boom)
        checks = CommunicateEvaluator.evaluate_communicate_info(
            _english_trajectory(), ["5 business days", "not communicated info"]
        )
        assert [c.met for c in checks] == [True, False]
        assert checks[0].info == "5 business days"
        assert "communicated in the message" in checks[0].justification
        assert checks[1].justification == (
            "Information 'not communicated info' not communicated."
        )
        # Result format unchanged: only the historical fields, no extras.
        assert set(checks[0].model_dump().keys()) == {"info", "met", "justification"}

    def test_hindi_run_uses_llm_judge(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            evaluator_communicate,
            "generate",
            _fake_generate(captured, met=True, reasoning="refund timing conveyed"),
        )
        checks = CommunicateEvaluator.evaluate_communicate_info(
            _trajectory(), ["refund will be processed in 5 days"], language="hi"
        )
        assert len(captured) == 1  # one LLM call per info item
        assert len(checks) == 1
        assert checks[0].met is True
        assert checks[0].justification.startswith(LLM_JUDGE_JUSTIFICATION_PREFIX)
        assert "refund timing conveyed" in checks[0].justification
        # The judge prompt contains the question, the info, and the transcript.
        system_prompt = captured[0]["messages"][0].content
        user_prompt = captured[0]["messages"][1].content
        assert "Did the agent communicate the following information" in user_prompt
        assert "refund will be processed in 5 days" in user_prompt
        assert "Aapka refund 5 din mein process ho jayega." in user_prompt
        assert "'hi'" in system_prompt  # language hint

    def test_judge_one_call_per_info_item(self, monkeypatch):
        captured = []
        monkeypatch.setattr(evaluator_communicate, "generate", _fake_generate(captured))
        checks = CommunicateEvaluator.evaluate_communicate_info(
            _trajectory(), ["info a", "info b", "info c"], language="hi"
        )
        assert len(captured) == 3
        assert [c.info for c in checks] == ["info a", "info b", "info c"]

    def test_force_flag_on_english(self, monkeypatch):
        """The English baseline arm can be scored with the identical judge."""
        captured = []
        monkeypatch.setattr(
            evaluator_communicate, "generate", _fake_generate(captured, met=False)
        )
        checks = CommunicateEvaluator.evaluate_communicate_info(
            _english_trajectory(),
            ["5 business days"],
            language=None,
            llm_communicate_judge=True,
        )
        assert len(captured) == 1
        assert checks[0].met is False
        # No language hint for English.
        assert "ISO 639-1" not in captured[0]["messages"][0].content

    def test_env_var_forces_judge_on_english(self, monkeypatch):
        captured = []
        monkeypatch.setattr(evaluator_communicate, "generate", _fake_generate(captured))
        monkeypatch.setenv(FORCE_LLM_COMMUNICATE_JUDGE_ENV, "1")
        CommunicateEvaluator.evaluate_communicate_info(
            _english_trajectory(), ["5 business days"]
        )
        assert len(captured) == 1

    def test_explicit_false_disables_judge_for_hindi(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("judge disabled explicitly")

        monkeypatch.setattr(evaluator_communicate, "generate", boom)
        checks = CommunicateEvaluator.evaluate_communicate_info(
            _trajectory(),
            ["refund will be processed in 5 days"],
            language="hi",
            llm_communicate_judge=False,
        )
        assert checks[0].met is False  # substring miss, as for English

    def test_judge_uses_configured_model(self, monkeypatch):
        captured = []
        monkeypatch.setattr(evaluator_communicate, "generate", _fake_generate(captured))
        CommunicateEvaluator.evaluate_communicate_info(
            _trajectory(), ["x"], language="hi"
        )
        assert (
            captured[0]["model"] == evaluator_communicate.DEFAULT_LLM_COMMUNICATE_JUDGE
        )
        assert captured[0]["call_name"] == "communicate_info_eval"


# ---- NLAssertionsEvaluator language hint ----


class TestNLAssertionsLanguageHint:
    def _capture_generate(self, monkeypatch, captured):
        def fake(model, messages, call_name=None, **kwargs):
            captured.append(messages)
            return AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "results": [
                            {
                                "expectedOutcome": "assertion",
                                "reasoning": "ok",
                                "metExpectation": True,
                            }
                        ]
                    }
                ),
            )

        monkeypatch.setattr(evaluator_nl_assertions, "generate", fake)

    def test_hint_present_for_hindi(self, monkeypatch):
        captured = []
        self._capture_generate(monkeypatch, captured)
        checks = NLAssertionsEvaluator.evaluate_nl_assertions(
            _trajectory(), ["assertion"], language="hi", script="Deva"
        )
        system_prompt = captured[0][0].content
        assert "LANGUAGE" in system_prompt
        assert "'hi'" in system_prompt
        assert "(script: Deva)" in system_prompt
        assert "responses in that language are valid" in system_prompt
        assert checks[0].met is True

    def test_hint_without_script(self, monkeypatch):
        captured = []
        self._capture_generate(monkeypatch, captured)
        NLAssertionsEvaluator.evaluate_nl_assertions(
            _trajectory(), ["assertion"], language="hi"
        )
        system_prompt = captured[0][0].content
        assert "'hi'" in system_prompt
        assert "script:" not in system_prompt

    def test_no_hint_for_english_default(self, monkeypatch):
        captured = []
        self._capture_generate(monkeypatch, captured)
        NLAssertionsEvaluator.evaluate_nl_assertions(_trajectory(), ["assertion"])
        system_prompt = captured[0][0].content
        assert "LANGUAGE" not in system_prompt
        assert "ISO 639-1" not in system_prompt

    def test_no_hint_for_explicit_en(self, monkeypatch):
        captured = []
        self._capture_generate(monkeypatch, captured)
        NLAssertionsEvaluator.evaluate_nl_assertions(
            _trajectory(), ["assertion"], language="en"
        )
        assert "LANGUAGE" not in captured[0][0].content


# ---- Language resolution from SimulationRun ----


class TestGetSimulationLanguageInfo:
    def test_no_speech_environment(self):
        assert get_simulation_language_info(_make_sim()) == (None, None)

    def test_english_speech_environment(self):
        sim = _make_sim(SpeechEnvironment())
        assert get_simulation_language_info(sim) == (None, None)

    def test_language_without_persona(self):
        sim = _make_sim(SpeechEnvironment(language="hi"))
        assert get_simulation_language_info(sim) == ("hi", None)

    def test_language_and_script_from_persona(self, monkeypatch):
        class FakePersona:
            language = "hi"
            script = "Deva"

        monkeypatch.setattr(
            evaluator_module,
            "get_multilingual_persona",
            lambda persona_id: (object(), FakePersona()),
        )
        sim = _make_sim(SpeechEnvironment(language="hi", persona_id="priya_hindi_v1"))
        assert get_simulation_language_info(sim) == ("hi", "Deva")

    def test_unknown_persona_id(self, monkeypatch):
        monkeypatch.setattr(
            evaluator_module, "get_multilingual_persona", lambda persona_id: None
        )
        sim = _make_sim(SpeechEnvironment(language="hi", persona_id="nope"))
        assert get_simulation_language_info(sim) == ("hi", None)


# ---- evaluate_simulation integration ----


class TestEvaluateSimulationIntegration:
    def test_hindi_run_activates_judge(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            evaluator_communicate, "generate", _fake_generate(captured, met=True)
        )
        sim = _make_sim(SpeechEnvironment(language="hi"))
        task = _make_task(communicate_info=["refund will be processed in 5 days"])
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.COMMUNICATE,
            solo_mode=False,
            domain="mock",
        )
        assert len(captured) == 1
        assert reward_info.reward == 1.0
        assert reward_info.communicate_checks[0].justification.startswith(
            LLM_JUDGE_JUSTIFICATION_PREFIX
        )

    def test_english_run_unchanged(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("LLM judge must not be called for English runs")

        monkeypatch.setattr(evaluator_communicate, "generate", boom)
        sim = _make_sim()
        sim.messages = _english_trajectory()
        task = _make_task(communicate_info=["5 business days"])
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.COMMUNICATE,
            solo_mode=False,
            domain="mock",
        )
        assert reward_info.reward == 1.0
        check = reward_info.communicate_checks[0]
        assert isinstance(check, CommunicateCheck)
        assert set(check.model_dump().keys()) == {"info", "met", "justification"}
        assert "communicated in the message" in check.justification

    def test_force_judge_param_on_english_run(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            evaluator_communicate, "generate", _fake_generate(captured, met=True)
        )
        sim = _make_sim()
        sim.messages = _english_trajectory()
        task = _make_task(communicate_info=["5 business days"])
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.COMMUNICATE,
            solo_mode=False,
            domain="mock",
            llm_communicate_judge=True,
        )
        assert len(captured) == 1
        assert reward_info.reward == 1.0

    def test_hindi_run_nl_assertions_hint(self, monkeypatch):
        captured = []

        def fake(model, messages, call_name=None, **kwargs):
            captured.append(messages)
            return AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "results": [
                            {
                                "expectedOutcome": "assertion",
                                "reasoning": "ok",
                                "metExpectation": True,
                            }
                        ]
                    }
                ),
            )

        monkeypatch.setattr(evaluator_nl_assertions, "generate", fake)

        class FakePersona:
            language = "hi"
            script = "Deva"

        monkeypatch.setattr(
            evaluator_module,
            "get_multilingual_persona",
            lambda persona_id: (object(), FakePersona()),
        )
        sim = _make_sim(SpeechEnvironment(language="hi", persona_id="priya_hindi_v1"))
        task = _make_task(nl_assertions=["assertion"])
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.NL_ASSERTIONS,
            solo_mode=False,
            domain="mock",
        )
        system_prompt = captured[0][0].content
        assert "'hi' (script: Deva)" in system_prompt
        assert reward_info.reward == 1.0


# ---- Bad judge output ----


def test_judge_invalid_json_raises(monkeypatch):
    def fake(model, messages, call_name=None, **kwargs):
        return AssistantMessage(role="assistant", content="not json")

    monkeypatch.setattr(evaluator_communicate, "generate", fake)
    with pytest.raises(json.JSONDecodeError):
        CommunicateEvaluator.evaluate_communicate_info(
            _trajectory(), ["x"], language="hi"
        )
