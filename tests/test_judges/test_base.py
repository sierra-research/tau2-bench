# Copyright Sierra
"""Shared judge plumbing: reply parsing, structured calls, ordered concurrency."""

import json
import threading
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import BaseModel

import tau2.judges.base as base
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.judges.base import judge_structured, ordered_map

# --- judge_structured --------------------------------------------------------


class _Verdict(BaseModel):
    ok: bool
    note: Optional[str] = None


def _messages():
    return [
        SystemMessage(role="system", content="sys"),
        UserMessage(role="user", content="user"),
    ]


def _mock_generate(monkeypatch, content: str, capture=None):
    def fake(model, messages, call_name=None, **kw):
        if capture is not None:
            capture.update(model=model, messages=messages, call_name=call_name, kw=kw)
        return SimpleNamespace(content=content)

    monkeypatch.setattr(base, "generate", fake)


def test_judge_structured_happy_path(monkeypatch):
    capture: dict = {}
    _mock_generate(
        monkeypatch,
        json.dumps({"ok": True, "note": "fine"}),
        capture=capture,
    )
    response = judge_structured(
        model="judge-model",
        messages=_messages(),
        response_model=_Verdict,
        call_name="test_judge",
        model_args={"reasoning_effort": "low"},
    )
    assert response.ok is True and response.note == "fine"
    # Args flow through the generate() seam with a stable call_name.
    assert capture["model"] == "judge-model"
    assert capture["call_name"] == "test_judge"
    assert capture["kw"]["reasoning_effort"] == "low"


def test_judge_structured_invalid_shape_raises(monkeypatch):
    # Valid JSON, invalid shape -> loud ValueError, never a silent drop.
    _mock_generate(monkeypatch, json.dumps({"note": "missing ok"}))
    with pytest.raises(ValueError, match="test_judge"):
        judge_structured(
            model="m",
            messages=_messages(),
            response_model=_Verdict,
            call_name="test_judge",
        )


def test_judge_structured_api_failure_propagates(monkeypatch):
    # The raw exception propagates untouched.
    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(base, "generate", boom)
    with pytest.raises(RuntimeError, match="api down"):
        judge_structured(
            model="m",
            messages=_messages(),
            response_model=_Verdict,
            call_name="test_judge",
        )


@pytest.mark.parametrize(
    "reply",
    [
        '{"ok": true}',
        '```json\n{"ok": true}\n```',
        '```\n{"ok": true}\n```',
        'Here is the analysis: {"ok": true} — hope that helps!',
    ],
)
def test_judge_structured_tolerates_fences_and_prose(monkeypatch, reply):
    _mock_generate(monkeypatch, reply)
    response = judge_structured(
        model="m", messages=_messages(), response_model=_Verdict, call_name="t"
    )
    assert response.ok is True


@pytest.mark.parametrize("reply", ["", "no json here at all", "[1, 2, 3]"])
def test_judge_structured_rejects_non_object_replies(monkeypatch, reply):
    # Garbage and JSON-but-not-an-object are loud failures, never silent drops.
    _mock_generate(monkeypatch, reply)
    with pytest.raises(ValueError):
        judge_structured(
            model="m", messages=_messages(), response_model=_Verdict, call_name="t"
        )


# --- shared verdict shape / scoring helpers ----------------------------------


def test_verdict_reply_lenient_bools_and_null_text():
    from tau2.judges.base import VerdictReplyBase

    v = VerdictReplyBase.model_validate(
        {"opportunity": "false", "violated": "no", "reasoning": None, "quote": None}
    )
    assert v.opportunity is False and v.violated is False
    assert v.reasoning == "" and v.quote == ""
    v = VerdictReplyBase.model_validate(
        {"opportunity": "true", "violated": "1", "reasoning": "why"}
    )
    assert v.opportunity is True and v.violated is True


def test_verdict_outcome_mapping():
    from tau2.data_model.simulation import JudgeOutcome
    from tau2.judges.base import VerdictReplyBase, verdict_outcome

    no_opp = VerdictReplyBase(opportunity=False, violated=True, reasoning="r")
    assert verdict_outcome(no_opp) == (JudgeOutcome.NO_OPPORTUNITY, None, None)

    passed = VerdictReplyBase(opportunity=True, violated=False, reasoning="fine")
    assert verdict_outcome(passed) == (JudgeOutcome.PASS, None, None)

    failed = VerdictReplyBase(
        opportunity=True, violated=True, reasoning=" bad ", quote=" span "
    )
    assert verdict_outcome(failed) == (JudgeOutcome.FAIL, "bad", "span")
    # A diffuse violation may omit the quote (exports as None, never "").
    diffuse = VerdictReplyBase(opportunity=True, violated=True, reasoning="why")
    assert verdict_outcome(diffuse) == (JudgeOutcome.FAIL, "why", None)


def test_violation_verdicts_require_reasoning():
    # A FAIL without an explanation is unactionable evidence: on the es+pt
    # judge-calibration corpus LLM quality-factor FAILs were ~100%
    # evidence-free before this constraint. Enforced on the shared base so
    # every verdict-shaped judge reply carries it.
    from pydantic import ValidationError

    from tau2.judges.base import VerdictReplyBase

    with pytest.raises(ValidationError):
        VerdictReplyBase(opportunity=True, violated=True)
    with pytest.raises(ValidationError):
        VerdictReplyBase(opportunity=True, violated=True, reasoning="   ")
    # Non-violations and no-opportunity replies never require reasoning.
    assert VerdictReplyBase(opportunity=True, violated=False).reasoning == ""
    assert VerdictReplyBase(opportunity=False, violated=False).reasoning == ""


def test_severity_weighted_score():
    from tau2.data_model.simulation import JudgeOutcome, NativenessFactorCheck
    from tau2.judges.base import severity_weighted_score

    def check(outcome, sev):
        return NativenessFactorCheck(
            id="f", category="c", severity=sev, outcome=outcome
        )

    assert severity_weighted_score([]) is None
    assert (
        severity_weighted_score(
            [check(JudgeOutcome.NO_OPPORTUNITY, 3), check(JudgeOutcome.DEFERRED, 3)]
        )
        is None
    )
    score = severity_weighted_score(
        [
            check(JudgeOutcome.PASS, 3),
            check(JudgeOutcome.FAIL, 1),
            check(JudgeOutcome.ERROR, 3),  # excluded
        ]
    )
    assert score == pytest.approx(3 / 4)


# --- ordered_map -------------------------------------------------------------


def test_ordered_map_preserves_order():
    items = list(range(20))
    assert ordered_map(lambda x: x * 2, items, concurrency=5) == [x * 2 for x in items]


def test_ordered_map_serial_when_concurrency_one():
    thread_ids = set()

    def fn(x):
        thread_ids.add(threading.get_ident())
        return x

    assert ordered_map(fn, [1, 2, 3], concurrency=1) == [1, 2, 3]
    assert thread_ids == {threading.get_ident()}  # ran on the caller's thread


def test_ordered_map_empty():
    assert ordered_map(lambda x: x, [], concurrency=4) == []
