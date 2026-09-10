"""Anthropic must start a new reply when the voice simulator resumes after silence."""

import pytest
from litellm import ModelResponse

from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.utils import llm_utils


@pytest.mark.parametrize("after_reply", [False, True])
def test_customer_start_and_silence_are_not_assistant_prefills(
    monkeypatch, after_reply
):
    messages = [SystemMessage(role="system", content="You are the customer.")]
    if after_reply:
        messages += [
            UserMessage(role="user", content="One moment."),
            AssistantMessage(role="assistant", content="Okay."),
            SystemMessage(role="system", content="Both parties silent for 5 seconds."),
        ]
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Hello?"}}]
        )

    monkeypatch.setattr(llm_utils, "completion", complete)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda response: {})
    result = llm_utils.generate("anthropic/claude-haiku-4-5-20251001", messages)
    assert result.content == "Hello?"
    assert captured["messages"][-1] == {"role": "user", "content": "Please continue."}
    assert messages[-1].role == "system"
