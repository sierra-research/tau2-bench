"""Frozen tool-use context and combined measure, with model calls mocked."""

import json
from types import SimpleNamespace

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EvaluationCriteria, Task
from tau2.judges.catalog import load_catalog
from tau2.judges.run_evidence import RunEvidence
from tau2.judges.tool_use import (
    QualityReply,
    ToolUseJudge,
    build_context,
    render_prompt,
)
from tau2.utils.transcript.audio_text import extract_delivered_text
from tau2.utils.transcript.views import extract_spoken_turns


def simulation(messages=None, ticks=None):
    return SimulationRun(
        id="sim",
        task_id="task",
        start_time="start",
        end_time="end",
        duration=1,
        trial=0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
        ticks=ticks,
        policy="Recorded policy.",
    )


def contract():
    return next(
        c for c in load_catalog().contracts if c.contract_id == "quality-v7-rubric-v4"
    )


def task():
    return Task(
        id="task", user_scenario={"instructions": "The customer wants an order refund."}
    )


def test_exact_prompt_preserves_recorded_policy_and_excludes_evaluator_data():
    call = ToolCall(
        id="lookup", name="find_user_id_by_email", arguments={"email": "a@example.com"}
    )
    sim = simulation(
        [
            UserMessage(role="user", content="My email is a@example.com."),
            AssistantMessage(role="assistant", content="Checking.", tool_calls=[call]),
            ToolMessage(
                role="tool",
                id="lookup",
                requestor="assistant",
                content="null",
                error=False,
            ),
        ]
    )
    recorded_task = task()
    recorded_task.evaluation_criteria = EvaluationCriteria(
        communicate_info=["PRIVATE_EVALUATOR_GOLDEN"]
    )
    context = build_context(sim, recorded_task, "retail")
    assert context.auth_calls == context.auth_no_match == context.tool_calls == 1
    assert (
        'tool_call id=lookup name=find_user_id_by_email arguments={"email": "a@example.com"}'
        in context.tool_trace
    )
    assert "tool_result id=lookup error=False null" in context.tool_trace
    criterion = contract().quality_criteria[0]
    prompt = render_prompt(context, criterion, contract())
    assert prompt == contract().prompt.user_prompt_template.format(
        criterion_json=json.dumps(criterion.model_dump(), ensure_ascii=False, indent=2),
        domain="retail",
        task_contract_json=context.task_contract.model_dump_json(exclude_none=True),
        domain_policy="Recorded policy.",
        transcript=context.transcript,
        tool_trace=context.tool_trace,
    )
    assert "evaluation_criteria" not in prompt
    assert "PRIVATE_EVALUATOR_GOLDEN" not in prompt


def test_preclassification_and_combination_do_not_report_subtypes_as_separate_scores(
    monkeypatch,
):
    import tau2.judges.tool_use as module

    call = ToolCall(
        id="lookup",
        name="find_user_id_by_email",
        arguments={"email": "wrong@example.com"},
    )
    sim = simulation(
        [
            AssistantMessage(role="assistant", content="Checking.", tool_calls=[call]),
            ToolMessage(
                role="tool",
                id="lookup",
                requestor="assistant",
                content="null",
                error=False,
            ),
        ]
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                dict(
                    criterion_id=kwargs["call_name"].removeprefix("quality_judge_"),
                    opportunity="true",
                    violated="false",
                    reasoning="No proven agent error.",
                    quote="",
                )
            )
        )

    monkeypatch.setattr(module, "generate", generate)
    result = ToolUseJudge(contract()).evaluate(
        RunEvidence(simulation=sim, task=task(), domain="retail", language="en"),
        "tool_use",
    )
    assert len(result.units) == 1 and result.units[0].unit_id == "call"
    assert not result.units[0].violation
    assert [c["call_name"] for c in calls] == [
        "quality_judge_auth_arg_mismatch",
        "quality_judge_incorrect_tool_parameters",
    ]
    assert all(
        c["model"] == "gpt-5.5" and c["reasoning_effort"] == "high" for c in calls
    )
    monkeypatch.setattr(
        module,
        "generate",
        lambda **kwargs: SimpleNamespace(
            content='{"opportunity": true, "violated": true, "reasoning": "Wrong supplied email.", "quote": "Checking."}'
        ),
    )
    assert (
        ToolUseJudge(contract())
        .evaluate(
            RunEvidence(simulation=sim, task=task(), domain="retail", language="en"),
            "tool_use",
        )
        .units[0]
        .violation
    )


def test_no_tools_is_no_opportunity_without_model_calls(monkeypatch):
    import tau2.judges.tool_use as module

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: pytest.fail("No opportunity must not call model"),
    )
    assert (
        not ToolUseJudge(contract())
        .evaluate(
            RunEvidence(
                simulation=simulation([]), task=task(), domain="retail", language="en"
            ),
            "tool_use",
        )
        .units[0]
        .opportunity
    )


@pytest.mark.parametrize(
    "criterion_id",
    ["agent_caused_tool_error", "auth_arg_mismatch", "incorrect_tool_parameters"],
)
@pytest.mark.parametrize("violated", [False, True])
def test_quality_reply_ignores_echoed_criterion_id(criterion_id, violated):
    verdict = {
        "opportunity": True,
        "violated": violated,
        "reasoning": "The argument differs from the caller's supplied value."
        if violated
        else "The argument matches the caller's supplied value.",
        "quote": "",
    }
    reply = QualityReply.model_validate({"criterion_id": criterion_id, **verdict})
    assert reply.model_dump() == verdict


def test_malformed_reply_never_becomes_a_pass():
    assert (
        QualityReply.model_validate(
            {"opportunity": "true", "violated": "false"}
        ).violated
        is False
    )
    for row in (
        {"opportunity": True},
        {"opportunity": True, "violated": "nonsense"},
        {"opportunity": True, "violated": True, "reasoning": ""},
        {
            "opportunity": False,
            "violated": True,
            "reasoning": "Contradictory opportunity.",
        },
    ):
        with pytest.raises(ValueError):
            QualityReply.model_validate(
                {"criterion_id": "incorrect_tool_parameters", **row}
            )


def test_interruption_uses_tick_overlap_and_delivered_gold_chunks():
    sim = simulation(
        ticks=[
            Tick(
                tick_id=0,
                timestamp="0",
                tick_duration_seconds=0.2,
                agent_chunk=AssistantMessage(
                    role="assistant", content="The total is ", utterance_ids=["a"]
                ),
            ),
            Tick(
                tick_id=1,
                timestamp="1",
                tick_duration_seconds=0.2,
                agent_chunk=AssistantMessage(
                    role="assistant", content="twenty", utterance_ids=["a"]
                ),
                user_chunk=UserMessage(
                    role="user", content="Wait.", utterance_ids=["u"]
                ),
            ),
        ]
    )
    turns = extract_spoken_turns(sim)
    assert turns[0].text == "The total is twenty" and turns[0].interrupted
    assert (
        "agent: The total is twenty [cut off by caller]"
        in build_context(sim, task(), "retail").transcript
    )
    gold = '<message uuid="a" active="0"><chunk id=0>Hello</chunk><chunk id=1> there</chunk></message>'
    assert extract_delivered_text(gold) == ("Hello", True)


def test_tool_use_contract_identity_is_data_but_criteria_must_be_complete():
    definition = contract().model_copy(deep=True)
    definition.contract_id = "reviewed-tool-use-revision"
    evidence = RunEvidence(
        simulation=simulation([]), task=task(), domain="retail", language="en"
    )
    assert (
        not ToolUseJudge(definition).evaluate(evidence, "tool_use").units[0].opportunity
    )
    definition.quality_criteria.pop()
    with pytest.raises(ValueError, match="three calibrated criteria"):
        ToolUseJudge(definition).evaluate(evidence, "tool_use")
