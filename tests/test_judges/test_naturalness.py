"""Calibrated text adapters: frozen context, batching, strict parsing and hybrid gate."""

import json
from types import SimpleNamespace

import pytest

from tau2.data_model.judges import Measure
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.judges.catalog import load_catalog
from tau2.judges.criteria import CriteriaJudge
from tau2.judges.run_evidence import RunEvidence


def sim(text="Necesito ayudarle."):
    return SimulationRun(
        id="s",
        task_id="t",
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            UserMessage(role="user", content="Ayuda"),
            AssistantMessage(role="assistant", content=text),
        ],
    )


def contract(identifier):
    return next(c for c in load_catalog().contracts if c.contract_id == identifier)


def test_naturalness_exact_context_and_false_string(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": "false",
                            "severity": 0,
                            "reasoning": "Natural",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    result = CriteriaJudge(
        contract("naturalness-v17-rubric-v21-language-specific-draft2")
    ).evaluate(
        RunEvidence(
            simulation=sim(),
            task=Task(id="t", user_scenario={"instructions": "Help."}),
            language="es",
            agent_gender="female",
            caller_gender="male",
            policy="",
        ),
        Measure.NATURALNESS,
    )
    assert not result.units[0].violation
    assert calls[0]["model"] == "gpt-5.5" and calls[0]["reasoning_effort"] == "xhigh"
    assert calls[0]["call_name"] == "naturalness_judge"
    prompt = calls[0]["messages"][1].content
    assert "Caller gender: male." in prompt and "Agent gender: female." in prompt
    assert "Ayuda" in prompt and "one agent utterance" in prompt


def test_naturalness_rejects_retired_factor_in_model_reply(monkeypatch):
    def generate(**kwargs):
        prompt = kwargs["messages"][1].content
        assert '"factor_id": "naturalness"' in prompt
        assert "natural_word_choice" not in prompt
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "natural_word_choice",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Natural",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    result = CriteriaJudge(
        contract("naturalness-v17-rubric-v21-language-specific-draft2")
    ).evaluate(
        RunEvidence(
            simulation=sim(),
            task=Task(id="t", user_scenario={"instructions": "Help."}),
            language="es",
            agent_gender="female",
            caller_gender="male",
        ),
        Measure.NATURALNESS,
    )
    assert not result.units
    assert result.unit_errors and all(
        "factor batch mismatch" in error.error.lower() for error in result.unit_errors
    )


def test_hindi_hybrid_short_circuit_and_participant_gate(monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: pytest.fail("deterministic contradiction should short circuit"),
    )
    c = contract("hi-gender-v16-rubric-v26")
    task = Task(id="t", user_scenario={"instructions": "Help."})
    result = CriteriaJudge(c).evaluate(
        RunEvidence(
            simulation=sim("मैं मदद कर सकती हूँ"),
            task=task,
            language="hi",
            agent_gender="male",
            caller_gender="female",
            policy="",
        ),
        Measure.GENDER_AGREEMENT,
    )
    assert result.units[0].violation
    with pytest.raises(ValueError, match="male-agent"):
        CriteriaJudge(c).evaluate(
            RunEvidence(
                simulation=sim(),
                task=task,
                language="hi",
                agent_gender="female",
                caller_gender="male",
                policy="",
            ),
            Measure.GENDER_AGREEMENT,
        )


def test_batch_mismatch_is_error(monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: SimpleNamespace(content='{"results":[]}'),
    )
    result = CriteriaJudge(contract("nativeness-v15-rubric-v19")).evaluate(
        RunEvidence(
            simulation=sim(),
            task=Task(id="t", user_scenario={"instructions": "Help."}),
            language="pt",
            agent_gender="female",
            caller_gender="male",
            policy="",
        ),
        Measure.COUNTING_UNITS,
    )
    assert not result.units
    assert result.unit_errors and all(
        "factor" in error.error.lower() for error in result.unit_errors
    )


def test_portuguese_gender_preserves_validated_participant_frame(monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: pytest.fail(
            "Unsupported participant frame must fail before judging"
        ),
    )
    with pytest.raises(ValueError, match="female-agent"):
        CriteriaJudge(contract("nativeness-v15-rubric-v19")).evaluate(
            RunEvidence(
                simulation=sim(),
                task=Task(id="t", user_scenario={"instructions": "Help."}),
                language="pt",
                agent_gender="male",
                caller_gender="female",
                policy="",
            ),
            Measure.GENDER_AGREEMENT,
        )


@pytest.mark.parametrize(
    "language,measure,contract_id",
    [
        ("es", Measure.COUNTING_UNITS, "nativeness-v15-rubric-v19"),
        ("pt", Measure.REGISTER_FORMALITY, "nativeness-v15-rubric-v19"),
        ("pt", Measure.GENDER_AGREEMENT, "nativeness-v15-rubric-v19"),
        ("hi", Measure.NAME_ADDRESS, "nativeness-v15-rubric-v19"),
        ("ko", Measure.HONORIFIC_AGREEMENT, "nativeness-v15-rubric-v19"),
        ("zh", Measure.MODAL_PARTICLES, "nativeness-v15-rubric-v19"),
    ],
)
def test_frozen_peer_batch_and_unit_level(language, measure, contract_id, monkeypatch):
    from tau2.judges.criteria import CALIBRATED_CALL_FACTORS

    c = contract(contract_id)
    call = measure.value in CALIBRATED_CALL_FACTORS
    expected = [
        criterion
        for criterion in c.nativeness_criteria[language]
        if (criterion.factor_id in CALIBRATED_CALL_FACTORS) == call
    ]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": criterion.factor_id,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Valid",
                            "quote": "",
                        }
                        for criterion in expected
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    verdict = CriteriaJudge(c).evaluate(
        RunEvidence(
            simulation=sim("Reviewed agent speech."),
            task=Task(id="t", user_scenario={"instructions": "Help."}),
            language=language,
            agent_gender="female",
            caller_gender="male",
            policy="",
        ),
        measure,
    )
    assert len(verdict.units) == 1 and not verdict.units[0].violation
    prompt = calls[0]["messages"][1].content
    assert ("all agent turns in this call" if call else "one agent utterance") in prompt
    for criterion in expected:
        assert criterion.question in prompt


def test_multiple_selected_measures_reuse_frozen_peer_batch(monkeypatch):
    calls = []
    c = contract("nativeness-v15-rubric-v19")

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": factor,
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Valid",
                            "quote": "",
                        }
                        for factor in ["gender_agreement", "counting_units"]
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    judge = CriteriaJudge(c)
    task = Task(id="t", user_scenario={"instructions": "Help."})
    evidence = RunEvidence(
        simulation=sim(),
        task=task,
        language="pt",
        agent_gender="female",
        caller_gender="male",
    )
    for measure in [Measure.COUNTING_UNITS, Measure.GENDER_AGREEMENT]:
        judge.evaluate(evidence, measure)
    assert len(calls) == 1


def test_criteria_cache_does_not_reuse_a_different_call(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Valid",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    judge = CriteriaJudge(
        contract("naturalness-v17-rubric-v21-language-specific-draft2")
    )
    task = Task(id="t", user_scenario={"instructions": "Help."})
    first = RunEvidence(simulation=sim("First call."), task=task, language="es")
    second = RunEvidence(simulation=sim("Different call."), task=task, language="es")
    judge.evaluate(first, Measure.NATURALNESS)
    judge.evaluate(first, Measure.NATURALNESS)
    judge.evaluate(second, Measure.NATURALNESS)
    assert len(calls) == 2
    assert "First call." in calls[0]["messages"][1].content
    assert "Different call." in calls[1]["messages"][1].content


def test_mutating_evidence_invalidates_in_memory_batch(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Valid",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    judge = CriteriaJudge(
        contract("naturalness-v17-rubric-v21-language-specific-draft2")
    )
    evidence = RunEvidence(
        simulation=sim("First."),
        task=Task(id="t", user_scenario={"instructions": "Help."}),
        language="es",
    )
    judge.evaluate(evidence, Measure.NATURALNESS)
    evidence.simulation.messages[-1].content = "Second."
    judge.evaluate(evidence, Measure.NATURALNESS)
    assert len(calls) == 2
    assert "Second." in calls[-1]["messages"][1].content


def test_bad_quote_does_not_discard_other_units_and_retry_reuses_them(monkeypatch):
    call = sim("Primera.")
    call.messages.extend(
        [
            UserMessage(role="user", content="Más"),
            AssistantMessage(role="assistant", content="Segunda."),
        ]
    )
    replies = iter(["Invented quote", ""])

    def generate(**kwargs):
        quote = next(replies)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": bool(quote),
                            "severity": 1 if quote else 0,
                            "reasoning": "Recorded",
                            "quote": quote,
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    evidence = RunEvidence(
        simulation=call,
        task=Task(id="t", user_scenario={"instructions": "Fixture"}),
        language="es",
        agent_gender="female",
        caller_gender="male",
    )
    judge = CriteriaJudge(
        contract("naturalness-v17-rubric-v21-language-specific-draft2")
    )
    result = judge.evaluate(evidence, "naturalness")
    assert len(result.units) == len(result.unit_errors) == 1
    assert "Quote is not present" in result.unit_errors[0].error
    assert "Invented quote" in result.unit_errors[0].response_text
    replies = iter([""])
    resumed = judge.resume(evidence, "naturalness", result)
    assert len(resumed.units) == 2 and not resumed.unit_errors
    assert result.units[0] in resumed.units
