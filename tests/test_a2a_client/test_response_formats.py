"""Tests for extract_response() — SDK typed extraction."""

import uuid

import pytest
from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithUri,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

import json

from tau2.a2a.translation import a2a_to_tau2_assistant_message, extract_response


def _make_text_part(text: str) -> Part:
    """Helper to create a Part containing a TextPart."""
    return Part(root=TextPart(text=text))


def _make_message(text: str, context_id: str | None = None) -> Message:
    """Helper to create a simple agent text message."""
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.agent,
        parts=[_make_text_part(text)],
        context_id=context_id,
    )


def _make_task(
    *,
    artifacts: list[Artifact] | None = None,
    status_message: Message | None = None,
    history: list[Message] | None = None,
    context_id: str = "ctx-123",
) -> Task:
    """Helper to create a Task with specified fields."""
    status = TaskStatus(state=TaskState.completed)
    if status_message is not None:
        status = TaskStatus(state=TaskState.completed, message=status_message)
    return Task(
        id=str(uuid.uuid4()),
        context_id=context_id,
        status=status,
        artifacts=artifacts,
        history=history,
    )


class TestMessageExtraction:
    """Tests for extract_response() on Message objects."""

    def test_simple_text(self):
        msg = _make_message("Hello world", context_id="ctx-1")
        text, ctx = extract_response(msg)
        assert text == "Hello world"
        assert ctx == "ctx-1"

    def test_multi_part_message(self):
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[_make_text_part("Part one"), _make_text_part("Part two")],
            context_id="ctx-2",
        )
        text, ctx = extract_response(msg)
        assert "Part one" in text
        assert "Part two" in text
        assert ctx == "ctx-2"

    def test_none_context_id(self):
        msg = _make_message("No context")
        text, ctx = extract_response(msg)
        assert text == "No context"
        assert ctx is None

    def test_empty_parts(self):
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[],
        )
        text, ctx = extract_response(msg)
        assert text == ""


class TestTaskExtraction:
    """Tests for extract_response() on Task objects."""

    def test_task_with_artifacts(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("Artifact text")],
        )
        task = _make_task(artifacts=[artifact])
        text, ctx = extract_response(task)
        assert text == "Artifact text"
        assert ctx == "ctx-123"

    def test_task_with_multiple_artifacts(self):
        a1 = Artifact(artifact_id="a1", parts=[_make_text_part("First")])
        a2 = Artifact(artifact_id="a2", parts=[_make_text_part("Second")])
        task = _make_task(artifacts=[a1, a2])
        text, ctx = extract_response(task)
        assert "First" in text
        assert "Second" in text

    def test_task_with_status_message(self):
        status_msg = _make_message("Status update")
        task = _make_task(status_message=status_msg)
        text, ctx = extract_response(task)
        assert text == "Status update"

    def test_task_with_history(self):
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("User said")],
        )
        agent_msg = _make_message("Agent replied")
        task = _make_task(history=[user_msg, agent_msg])
        text, ctx = extract_response(task)
        assert text == "Agent replied"

    def test_task_history_picks_last_agent_message(self):
        agent1 = _make_message("First reply")
        user = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("Follow up")],
        )
        agent2 = _make_message("Second reply")
        task = _make_task(history=[agent1, user, agent2])
        text, _ = extract_response(task)
        assert text == "Second reply"

    def test_task_history_skips_trailing_user_messages(self):
        agent_msg = _make_message("Agent reply")
        user_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[_make_text_part("User follow up")],
        )
        task = _make_task(history=[agent_msg, user_msg])
        text, _ = extract_response(task)
        assert text == "Agent reply"

    def test_empty_task(self):
        task = _make_task()
        text, ctx = extract_response(task)
        assert text == ""
        assert ctx == "ctx-123"


class TestTaskFieldPriority:
    """Tests for extract_response() field priority: artifacts > status > history."""

    def test_artifacts_win_over_status(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("From artifact")],
        )
        status_msg = _make_message("From status")
        task = _make_task(artifacts=[artifact], status_message=status_msg)
        text, _ = extract_response(task)
        assert text == "From artifact"

    def test_status_wins_over_history(self):
        status_msg = _make_message("From status")
        agent_msg = _make_message("From history")
        task = _make_task(status_message=status_msg, history=[agent_msg])
        text, _ = extract_response(task)
        assert text == "From status"

    def test_artifacts_win_over_all(self):
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("From artifact")],
        )
        status_msg = _make_message("From status")
        agent_msg = _make_message("From history")
        task = _make_task(
            artifacts=[artifact],
            status_message=status_msg,
            history=[agent_msg],
        )
        text, _ = extract_response(task)
        assert text == "From artifact"

    def test_non_text_artifact_falls_through_to_status(self):
        """File artifact (no text parts) should not block status.message fallback."""
        file_artifact = Artifact(
            artifact_id="a1",
            parts=[Part(root=FilePart(file=FileWithUri(uri="s3://bucket/report.pdf")))],
        )
        status_msg = _make_message("Your report is ready.")
        task = _make_task(artifacts=[file_artifact], status_message=status_msg)
        text, _ = extract_response(task)
        assert text == "Your report is ready."

    def test_non_text_artifact_falls_through_to_history(self):
        """File artifact with no status should fall through to history."""
        file_artifact = Artifact(
            artifact_id="a1",
            parts=[Part(root=FilePart(file=FileWithUri(uri="s3://bucket/report.pdf")))],
        )
        agent_msg = _make_message("Here is the file.")
        task = _make_task(artifacts=[file_artifact], history=[agent_msg])
        text, _ = extract_response(task)
        assert text == "Here is the file."

    def test_data_artifact_falls_through_to_status(self):
        """Data artifact (no text parts) should not block status.message fallback."""
        data_artifact = Artifact(
            artifact_id="a1",
            parts=[Part(root=DataPart(data={"key": "value"}))],
        )
        status_msg = _make_message("Data processed.")
        task = _make_task(artifacts=[data_artifact], status_message=status_msg)
        text, _ = extract_response(task)
        assert text == "Data processed."

    def test_mixed_artifacts_only_text_ones_used(self):
        """Mix of file and text artifacts — only text artifacts contribute."""
        file_artifact = Artifact(
            artifact_id="a1",
            parts=[Part(root=FilePart(file=FileWithUri(uri="s3://bucket/f.pdf")))],
        )
        text_artifact = Artifact(
            artifact_id="a2",
            parts=[_make_text_part("Actual content")],
        )
        status_msg = _make_message("Should not appear")
        task = _make_task(
            artifacts=[file_artifact, text_artifact], status_message=status_msg
        )
        text, _ = extract_response(task)
        assert text == "Actual content"

    def test_all_non_text_artifacts_empty_status_falls_to_history(self):
        """Non-text artifacts + empty-parts status → falls through to history."""
        file_artifact = Artifact(
            artifact_id="a1",
            parts=[Part(root=DataPart(data={"x": 1}))],
        )
        empty_status = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[],
        )
        agent_msg = _make_message("From history")
        task = _make_task(
            artifacts=[file_artifact],
            status_message=empty_status,
            history=[agent_msg],
        )
        text, _ = extract_response(task)
        assert text == "From history"


class TestClientMessageExtraction:
    """Full pipeline: SDK response → extract_response → a2a_to_tau2_assistant_message."""

    def test_tool_call_from_message(self):
        """Agent returns tool call JSON in a Message → parsed as tool call."""
        tool_json = json.dumps(
            {"tool_call": {"name": "search_flights", "arguments": {"origin": "SFO"}}}
        )
        msg = _make_message(tool_json)
        result = a2a_to_tau2_assistant_message(msg)
        assert result.is_tool_call()
        assert result.tool_calls[0].name == "search_flights"
        assert result.tool_calls[0].arguments == {"origin": "SFO"}

    def test_tool_call_in_markdown_block(self):
        """Agent returns tool call wrapped in markdown code block."""
        content = '```json\n{"tool_call": {"name": "book_flight", "arguments": {"id": "1"}}}\n```'
        msg = _make_message(content)
        result = a2a_to_tau2_assistant_message(msg)
        assert result.is_tool_call()
        assert result.tool_calls[0].name == "book_flight"

    def test_multiple_tool_calls(self):
        """Agent returns multiple tool calls."""
        content = json.dumps(
            {
                "tool_calls": [
                    {"tool_call": {"name": "search_flights", "arguments": {"origin": "SFO"}}},
                    {"tool_call": {"name": "search_hotels", "arguments": {"city": "NYC"}}},
                ]
            }
        )
        msg = _make_message(content)
        result = a2a_to_tau2_assistant_message(msg)
        assert result.is_tool_call()
        assert len(result.tool_calls) == 2

    def test_plain_text_from_message(self):
        """Agent returns plain text in a Message → text content, no tool calls."""
        msg = _make_message("Your flight is confirmed.")
        result = a2a_to_tau2_assistant_message(msg)
        assert result.content == "Your flight is confirmed."
        assert result.tool_calls is None

    def test_plain_text_from_task_artifact(self):
        """Agent returns text via Task artifact."""
        artifact = Artifact(
            artifact_id="a1",
            parts=[_make_text_part("Booking confirmed for AA123.")],
        )
        task = _make_task(artifacts=[artifact])
        result = a2a_to_tau2_assistant_message(task)
        assert result.content == "Booking confirmed for AA123."
        assert result.tool_calls is None

    def test_plain_text_from_task_status(self):
        """Agent returns text via Task status message."""
        status_msg = _make_message("Processing complete.")
        task = _make_task(status_message=status_msg)
        result = a2a_to_tau2_assistant_message(task)
        assert result.content == "Processing complete."

    def test_plain_text_from_task_history(self):
        """Agent returns text via Task history."""
        agent_msg = _make_message("Here are your options.")
        task = _make_task(history=[agent_msg])
        result = a2a_to_tau2_assistant_message(task)
        assert result.content == "Here are your options."

    def test_empty_message_fallback(self):
        """Empty message → fallback text."""
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[],
        )
        result = a2a_to_tau2_assistant_message(msg)
        assert "unable to generate" in result.content.lower()

    def test_json_that_is_not_tool_call(self):
        """JSON that doesn't match tool call format → treated as text."""
        msg = _make_message('{"status": "ok", "flights": 5}')
        result = a2a_to_tau2_assistant_message(msg)
        assert result.content == '{"status": "ok", "flights": 5}'
        assert result.tool_calls is None
