"""Parsing of NL-assertion judge replies (issue #735)."""

import json
from types import SimpleNamespace

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.evaluator.evaluator_nl_assertions import (
    FullDuplexNLAssertionsEvaluator,
    NLAssertionsEvaluator,
    parse_judge_results,
)

RESULTS = [
    {
        "expectedOutcome": "The agent greets the customer.",
        "reasoning": "It says ``` hello ``` first.",
        "metExpectation": True,
    },
    {
        "expectedOutcome": "The agent confirms the refund amount.",
        "reasoning": "No amount was confirmed.",
        "metExpectation": False,
    },
]
OBJECT = json.dumps({"results": RESULTS})
ARRAY = json.dumps(RESULTS)

ACCEPTED_REPLIES = {
    "object": OBJECT,
    "object in json fence": f"```json\n{OBJECT}\n```",
    "bare array": ARRAY,
    "bare array in plain fence": f"```\n{json.dumps(RESULTS, indent=2)}\n```",
    "bare array in fence with CRLF": f"```json\r\n{ARRAY}\r\n```\r\n",
    "object with surrounding whitespace": f"\n  {OBJECT}  \n",
}

REJECTED_REPLIES = {
    "object without results key": (json.dumps({"verdicts": RESULTS}), ValueError),
    "results is not a list": (json.dumps({"results": RESULTS[0]}), ValueError),
    "array of strings": (json.dumps(["a", "b"]), ValueError),
    "scalar": ("true", ValueError),
    "prose": ("The agent did fine.", json.JSONDecodeError),
    "fenced block inside prose": (
        f"Here you go:\n```json\n{OBJECT}\n```",
        json.JSONDecodeError,
    ),
    "two fenced blocks": (
        f"```json\n{ARRAY}\n```\n```json\n{ARRAY}\n```",
        json.JSONDecodeError,
    ),
}


@pytest.mark.parametrize(
    "content", ACCEPTED_REPLIES.values(), ids=ACCEPTED_REPLIES.keys()
)
def test_parse_judge_results_accepts_object_and_bare_array(content):
    assert parse_judge_results(content) == RESULTS


@pytest.mark.parametrize(
    "content,error",
    REJECTED_REPLIES.values(),
    ids=REJECTED_REPLIES.keys(),
)
def test_parse_judge_results_rejects_reply_without_results(content, error):
    with pytest.raises(error):
        parse_judge_results(content)


def test_parse_judge_results_rejects_empty_reply():
    with pytest.raises(ValueError):
        parse_judge_results(None)


@pytest.mark.parametrize(
    "evaluator", [NLAssertionsEvaluator, FullDuplexNLAssertionsEvaluator]
)
@pytest.mark.parametrize(
    "content",
    [OBJECT, ARRAY, f"```json\n{ARRAY}\n```"],
    ids=["object", "bare array", "fenced bare array"],
)
def test_evaluate_nl_assertions_reads_every_reply_shape(
    monkeypatch, evaluator, content
):
    import tau2.evaluator.evaluator_nl_assertions as module

    monkeypatch.setattr(
        module, "generate", lambda **kwargs: SimpleNamespace(content=content)
    )
    trajectory = [
        AssistantMessage(role="assistant", content="Hello."),
        UserMessage(role="user", content="###STOP###"),
    ]
    checks = evaluator.evaluate_nl_assertions(
        trajectory, [r["expectedOutcome"] for r in RESULTS]
    )

    assert [c.nl_assertion for c in checks] == [r["expectedOutcome"] for r in RESULTS]
    assert [c.met for c in checks] == [True, False]
    assert [c.justification for c in checks] == [r["reasoning"] for r in RESULTS]


@pytest.mark.parametrize(
    "reply",
    [json.dumps({"verdicts": RESULTS}), json.dumps({"results": []}), "[]"],
    ids=["no results key", "empty results", "empty array"],
)
def test_evaluate_nl_assertions_raises_when_reply_has_no_results(monkeypatch, reply):
    import tau2.evaluator.evaluator_nl_assertions as module

    monkeypatch.setattr(
        module, "generate", lambda **kwargs: SimpleNamespace(content=reply)
    )

    with pytest.raises(ValueError):
        NLAssertionsEvaluator.evaluate_nl_assertions(
            [UserMessage(role="user", content="hi")], ["a"]
        )
