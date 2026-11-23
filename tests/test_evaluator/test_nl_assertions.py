import json
from types import SimpleNamespace

import pytest

from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator_nl_assertions import (
    FullDuplexNLAssertionsEvaluator,
    NLAssertionsEvaluator,
)


@pytest.mark.parametrize(
    "evaluator",
    [NLAssertionsEvaluator, FullDuplexNLAssertionsEvaluator],
    ids=["text", "voice"],
)
@pytest.mark.parametrize(
    "wrapped,fenced",
    [(True, False), (True, True), (False, False), (False, True)],
    ids=["object", "fenced-object", "array", "fenced-array"],
)
def test_nl_assertions_parse_judge_response(monkeypatch, evaluator, wrapped, fenced):
    assertions = ["The agent greets the customer.", "The agent confirms the refund."]
    verdicts = [
        {
            "expectedOutcome": assertion,
            "metExpectation": met,
            "reasoning": "Observed in the transcript.",
        }
        for assertion, met in zip(assertions, [True, False], strict=True)
    ]
    content = json.dumps({"results": verdicts} if wrapped else verdicts)
    if fenced:
        content = f"```json\n{content}\n```"
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: SimpleNamespace(content=content),
    )

    task = Task(
        id="judge-response",
        user_scenario=UserScenario(instructions="fixture"),
        evaluation_criteria=EvaluationCriteria(nl_assertions=assertions),
    )
    reward = evaluator.calculate_reward(task, [])

    assert [(check.nl_assertion, check.met) for check in reward.nl_assertions] == list(
        zip(assertions, [True, False], strict=True)
    )
    assert reward.reward == 0.0


@pytest.mark.parametrize("content", ["{}", "[]", '{"results": "invalid"}', "null"])
def test_nl_assertions_reject_invalid_response_shape(monkeypatch, content):
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: SimpleNamespace(content=content),
    )

    with pytest.raises(ValueError, match="list of results"):
        NLAssertionsEvaluator.evaluate_nl_assertions(
            [], ["The agent greets the customer."]
        )
