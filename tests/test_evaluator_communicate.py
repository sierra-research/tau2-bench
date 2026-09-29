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
