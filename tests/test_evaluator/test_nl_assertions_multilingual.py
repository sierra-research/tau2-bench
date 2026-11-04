"""NL-assertion judging over non-English transcripts with English assertions."""

import json
from types import SimpleNamespace

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.multilingual.runtime import load_pack

ASSERTIONS = [
    "The agent greets the customer.",
    "The agent confirms the refund amount.",
]


@pytest.mark.parametrize("language", ["es", "hi", "ko", "zh"])
def test_nl_assertions_pass_native_transcript_and_parse_reply(monkeypatch, language):
    import tau2.evaluator.evaluator_nl_assertions as module

    # Reviewed pack strings only: no target-language text is authored here.
    greeting = load_pack(language).agent_greeting
    trajectory = [
        AssistantMessage(role="assistant", content=greeting),
        UserMessage(role="user", content="###STOP###"),
    ]
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": ASSERTIONS[0],
                            "reasoning": "Greeting is present.",
                            "metExpectation": True,
                        },
                        {
                            "expectedOutcome": ASSERTIONS[1],
                            "reasoning": "No refund amount was confirmed.",
                            "metExpectation": False,
                        },
                    ]
                }
            )
        )

    monkeypatch.setattr(module, "generate", fake_generate)
    task = Task(
        id="nl-multilingual",
        user_scenario=UserScenario(instructions="fixture"),
        evaluation_criteria=EvaluationCriteria(nl_assertions=ASSERTIONS),
    )
    reward = NLAssertionsEvaluator.calculate_reward(task, trajectory)

    assert len(calls) == 1
    assert calls[0]["call_name"] == "nl_assertions_eval"
    user_prompt = calls[0]["messages"][-1].content
    assert f"assistant: {greeting}" in user_prompt
    for assertion in ASSERTIONS:
        assert assertion in user_prompt
    assert [check.met for check in reward.nl_assertions] == [True, False]
    assert [check.nl_assertion for check in reward.nl_assertions] == ASSERTIONS
    assert reward.reward == 0.0
