import json
from unittest import mock

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.evaluator import evaluator_nl_assertions as nl_module
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

ASSERTIONS = [
    "Agent should cancel order #W1",
    "Agent should confirm the refund amount",
]
TRAJECTORY = [UserMessage(role="user", content="cancel my order")]


def _verdict(assertion: str, met: bool = True) -> dict:
    return {"expectedOutcome": assertion, "metExpectation": met, "reasoning": "r"}


def _judge(payload: dict):
    reply = AssistantMessage(role="assistant", content=json.dumps(payload))
    return mock.patch.object(nl_module, "generate", return_value=reply)


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        {"verdicts": "unexpected shape"},
        {"results": [_verdict(ASSERTIONS[0])]},
        {"results": [_verdict(a) for a in ASSERTIONS] * 2},
    ],
    ids=["empty", "missing-key", "too-few", "too-many"],
)
def test_verdict_count_mismatch_is_not_scored(payload):
    with _judge(payload):
        with pytest.raises(ValueError, match="verdict"):
            NLAssertionsEvaluator.evaluate_nl_assertions(TRAJECTORY, list(ASSERTIONS))


def test_matching_verdicts_are_scored():
    payload = {"results": [_verdict(ASSERTIONS[0]), _verdict(ASSERTIONS[1], False)]}
    with _judge(payload):
        checks = NLAssertionsEvaluator.evaluate_nl_assertions(
            TRAJECTORY, list(ASSERTIONS)
        )
    assert [c.met for c in checks] == [True, False]
