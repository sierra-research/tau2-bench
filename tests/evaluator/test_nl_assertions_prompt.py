from unittest.mock import patch

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator


def test_nl_assertion_prompt_omits_contentless_messages() -> None:
    trajectory = [
        UserMessage(role="user", content="Please check order #W1"),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="get_order_details",
                    arguments={"order_id": "#W1"},
                )
            ],
        ),
        ToolMessage(role="tool", content="Order #W1 is pending", id="call-1"),
        AssistantMessage(role="assistant", content="The order is pending."),
    ]

    judge_response = AssistantMessage(
        role="assistant",
        content=(
            '{"results": [{"expectedOutcome": "The agent reports the order status", '
            '"metExpectation": true, "reasoning": "The status was reported."}]}'
        ),
    )

    with patch(
        "tau2.evaluator.evaluator_nl_assertions.generate",
        return_value=judge_response,
    ) as generate_mock:
        checks = NLAssertionsEvaluator.evaluate_nl_assertions(
            trajectory,
            ["The agent reports the order status"],
        )

    prompt = generate_mock.call_args.kwargs["messages"][1].content
    assert "assistant: None" not in prompt
    assert "user: Please check order #W1" in prompt
    assert "tool: Order #W1 is pending" in prompt
    assert "assistant: The order is pending." in prompt
    assert len(checks) == 1
    assert checks[0].met is True
