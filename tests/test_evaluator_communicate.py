import pytest

from tau2.data_model.message import AssistantMessage, Tick, ToolCall, UserMessage
from tau2.evaluator.evaluator_communicate import (
    CommunicateEvaluator,
    FullDuplexCommunicateEvaluator,
)


@pytest.mark.parametrize(
    ("amount", "text"),
    [
        ("5244", "Refund is five thousand two hundred forty four dollars."),
        ("23553", "twenty three thousand five hundred and fifty three dollars"),
        ("1286", "Your card was charged twelve eighty-six dollars."),
        ("1000", "You have one thousand dollars."),
        ("1628", "The total is 1,628 dollars."),
    ],
)
def test_spoken_amounts(amount, text):
    checks = CommunicateEvaluator.evaluate_communicate_info(
        [AssistantMessage(role="assistant", content=text)], [amount]
    )
    assert checks[0].met
    assert text in checks[0].justification


@pytest.mark.parametrize(
    "text",
    [
        "5244 dollars",
        "4.5 dollars",
        "forty dollars",
        "one hundred four dollars",
        "four thousand dollars",
        "four point five",
        "ID4",
    ],
)
def test_wrong_amount_does_not_match(text):
    assert not CommunicateEvaluator.evaluate_communicate_info(
        [AssistantMessage(role="assistant", content=text)], ["4"]
    )[0].met


def ticks(parts, ids=None):
    return [
        Tick(
            tick_id=i,
            timestamp="2026-09-09T00:00:00",
            tick_duration_seconds=0.2,
            agent_chunk=AssistantMessage(
                role="assistant", content=p, utterance_ids=ids
            ),
        )
        for i, p in enumerate(parts)
    ]


@pytest.mark.parametrize(
    "parts,amount",
    [
        (["1,", "628 dollars"], "1628"),
        (["10", "00 dollars"], "1000"),
        (["five thousand ", "two hundred ", "forty four dollars"], "5244"),
    ],
)
def test_missing_ids_reconstruct_amount(parts, amount):
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(ticks(parts))
    assert FullDuplexCommunicateEvaluator.evaluate_communicate_info(messages, [amount])[
        0
    ].met


def test_explicit_utterance_boundaries():
    trajectory = ticks(["10", "00"])
    trajectory[0].agent_chunk.utterance_ids = ["first"]
    trajectory[1].agent_chunk.utterance_ids = ["second"]
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
    assert not FullDuplexCommunicateEvaluator.evaluate_communicate_info(
        messages, ["1000"]
    )[0].met


@pytest.mark.parametrize("boundary", ["silence", "tool"])
def test_missing_ids_do_not_join_unrelated_responses(boundary):
    trajectory = ticks(["10"] + [None] * 6 + ["00"])
    if boundary == "tool":
        trajectory = ticks(["10", None, "00"])
        trajectory[1].agent_tool_calls = [
            ToolCall(id="call", name="lookup", arguments={})
        ]
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
    assert not FullDuplexCommunicateEvaluator.evaluate_communicate_info(
        messages, ["1000"]
    )[0].met


def test_user_and_tools_cannot_supply_required_information():
    messages = [
        UserMessage(role="user", content="5244"),
        AssistantMessage(role="assistant", content="Done"),
    ]
    assert not CommunicateEvaluator.evaluate_communicate_info(messages, ["5244"])[0].met


def test_non_numeric_criteria_keep_existing_matching():
    assert CommunicateEvaluator.evaluate_communicate_info(
        [AssistantMessage(role="assistant", content="Your CODE is ABC")],
        ["code is abc"],
    )[0].met


def test_whitespace_only_chunks_preserve_word_boundaries():
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(
        ticks(["total", " ", "327", " ", "dollars"])
    )
    assert messages[0].content == "total 327 dollars"
    assert FullDuplexCommunicateEvaluator.evaluate_communicate_info(messages, ["327"])[
        0
    ].met


def test_numeric_sentence_punctuation():
    assert CommunicateEvaluator.evaluate_communicate_info(
        [AssistantMessage(role="assistant", content="Your allowance is 4.")], ["4"]
    )[0].met


def test_shared_utterance_ids_still_merge():
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(
        ticks(["10", "00 dollars"], ids=["same"])
    )
    assert len(messages) == 1
    assert messages[0].content == "1000 dollars"


def test_all_required_information_controls_reward(base_task):
    from tau2.data_model.tasks import EvaluationCriteria

    task = base_task.model_copy(
        update={
            "evaluation_criteria": EvaluationCriteria(communicate_info=["5244", "1000"])
        }
    )
    messages = [
        AssistantMessage(
            role="assistant",
            content="Refund is five thousand two hundred forty four dollars.",
        )
    ]
    assert CommunicateEvaluator.calculate_reward(task, messages).reward == 0
    assert (
        FullDuplexCommunicateEvaluator.calculate_reward(
            task, ticks([messages[0].content])
        ).reward
        == 0
    )
    task.evaluation_criteria.communicate_info = ["5244"]
    assert CommunicateEvaluator.calculate_reward(task, messages).reward == 1
    assert (
        FullDuplexCommunicateEvaluator.calculate_reward(
            task, ticks([messages[0].content])
        ).reward
        == 1
    )


@pytest.mark.parametrize(
    "text",
    [
        "minus four dollars",
        "negative four dollars",
        "-four dollars",
        "minus 4 dollars",
        "negative 4 dollars",
        "-$4",
        "-4",
    ],
)
def test_negative_amount_cannot_match_positive(text):
    assert not CommunicateEvaluator.evaluate_communicate_info(
        [AssistantMessage(role="assistant", content=text)], ["4"]
    )[0].met


@pytest.mark.parametrize("duration,count", [(0.1, 10), (0.2, 5), (0.05, 20), (0.25, 4)])
def test_exact_one_second_pause_separates_responses(duration, count):
    trajectory = ticks(["10"] + [None] * count + ["00"])
    for tick in trajectory:
        tick.tick_duration_seconds = duration
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
    assert [message.content for message in messages] == ["10", "00"]


@pytest.mark.parametrize("event", ["call", "result", "inline_call"])
def test_simultaneous_tool_event_preserves_speech(event):
    from tau2.data_model.message import ToolMessage

    trajectory = ticks(["before", "Refund is 5244 dollars.", "after"])
    call = ToolCall(id="call", name="lookup", arguments={})
    if event == "call":
        trajectory[1].agent_tool_calls = [call]
    elif event == "result":
        trajectory[1].agent_tool_results = [
            ToolMessage(role="tool", id="call", content="1000")
        ]
    else:
        trajectory[1].agent_chunk.tool_calls = [call]
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
    assert [m.content for m in messages] == [
        "before",
        "Refund is 5244 dollars.",
        "after",
    ]
    checks = FullDuplexCommunicateEvaluator.evaluate_communicate_info(
        messages, ["5244", "1000"]
    )
    assert [c.met for c in checks] == [True, False]


@pytest.mark.parametrize(
    "ids",
    [
        [["first"], None],
        [None, ["second"]],
        [["first"], ["first", "second"], ["second"]],
    ],
)
def test_mixed_utterance_ids(ids):
    trajectory = ticks(["10", "00"] if len(ids) == 2 else ["1", "0", "00"])
    for tick, utterance_ids in zip(trajectory, ids):
        tick.agent_chunk.utterance_ids = utterance_ids
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
    assert [m.content for m in messages] == (
        ["10", "00"] if len(ids) == 2 else ["1000"]
    )


def test_short_pause_preserves_number_fragments():
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(
        ticks(["10", None, None, "00"])
    )
    assert [m.content for m in messages] == ["1000"]


def test_public_airline_trajectory_excerpts():
    """Keep small real-tick regressions for both observed false-negative causes."""
    import json
    from pathlib import Path

    cases = json.loads(
        (
            Path(__file__).parent / "fixtures" / "communication_airline_excerpts.json"
        ).read_text()
    )
    for case in cases:
        trajectory = [Tick.model_validate(t) for t in case["ticks"]]
        messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(trajectory)
        assert FullDuplexCommunicateEvaluator.evaluate_communicate_info(
            messages, [case["amount"]]
        )[0].met, case["task_id"]
