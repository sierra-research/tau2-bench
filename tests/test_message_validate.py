"""Tests for ParticipantMessageBase.validate() placeholder injection."""

import logging

from tau2.data_model.message import AssistantMessage


def test_validate_injects_placeholder_when_no_content_or_tool_calls(caplog):
    """validate() should inject a placeholder instead of raising ValueError."""
    msg = AssistantMessage(role="assistant", content=None, tool_calls=None)

    with caplog.at_level(logging.WARNING, logger="tau2.data_model.message"):
        msg.validate()

    assert msg.content == "[No response generated]"
    assert any("injecting placeholder" in record.message for record in caplog.records)


def test_validate_does_not_modify_message_with_content():
    """validate() should leave messages with text content unchanged."""
    msg = AssistantMessage(role="assistant", content="Hello!")
    msg.validate()
    assert msg.content == "Hello!"


def test_validate_does_not_modify_message_with_tool_calls():
    """validate() should leave messages with tool calls unchanged."""
    from tau2.data_model.message import ToolCall

    tc = ToolCall(name="my_tool", arguments={})
    msg = AssistantMessage(role="assistant", content=None, tool_calls=[tc])
    msg.validate()
    assert msg.tool_calls == [tc]
    assert msg.content is None
