import json

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.tasks import EvaluationCriteria, RewardType
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator_nl_assertions import (
    FullDuplexNLAssertionsEvaluator,
    NLAssertionsEvaluator,
)


@pytest.fixture(params=[NLAssertionsEvaluator, FullDuplexNLAssertionsEvaluator])
def evaluator(request):
    return request.param


@pytest.mark.parametrize(
    "wrapper",
    ["{}", "```json\n{}\n```", "```\n{}\n```", " \n```json \r\n{}\r\n```\t\n"],
)
@pytest.mark.parametrize("met", [True, False])
@pytest.mark.parametrize(
    "reasoning",
    ["Checked the conversation.", "The user wrote ```not JSON``` and ```json."],
)
def test_judge_json_formats(evaluator, base_task, monkeypatch, wrapper, met, reasoning):
    """JSON formatting must not change either evaluator's assertion verdict."""
    assertion = "The agent confirms the request."
    base_task.evaluation_criteria = EvaluationCriteria(
        nl_assertions=[assertion], reward_basis=[RewardType.NL_ASSERTION]
    )
    payload = json.dumps(
        {
            "results": [
                {
                    "expectedOutcome": assertion,
                    "metExpectation": met,
                    "reasoning": reasoning,
                }
            ]
        }
    )
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=wrapper.format(payload)
        ),
    )

    reward = evaluator.calculate_reward(base_task, [])

    assert reward.reward == float(met)
    assert reward.reward_breakdown == {RewardType.NL_ASSERTION: float(met)}
    assert len(reward.nl_assertions) == 1
    check = reward.nl_assertions[0]
    assert check.nl_assertion == assertion
    assert check.met is met
    assert check.justification == reasoning


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '```json\n{"results":\n```',
        'Here is the result:\n```json\n{"results": []}\n```',
        '```json\n{"results": []}\n```\nTrailing prose',
        '```json\n{"results": []}\n```\n```json\n{"results": []}\n```',
        '```json\n{"results": []}',
        '```python\n{"results": []}\n```',
    ],
)
def test_malformed_judge_json_still_raises(evaluator, monkeypatch, response):
    """A parse failure must not be converted to an empty set of passing checks."""
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: AssistantMessage(role="assistant", content=response),
    )

    with pytest.raises(json.JSONDecodeError):
        evaluator.evaluate_nl_assertions(
            [UserMessage(role="user", content="Please confirm my request.")],
            ["The agent confirms the request."],
        )


def test_missing_judge_content_still_raises(evaluator, monkeypatch):
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "generate",
        lambda **kwargs: AssistantMessage(role="assistant", content=None),
    )

    with pytest.raises(TypeError):
        evaluator.evaluate_nl_assertions([], ["The agent confirms the request."])
