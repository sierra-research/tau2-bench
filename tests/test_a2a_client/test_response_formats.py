"""Tests for extract_response() — SDK typed extraction."""

import uuid

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from tau2.a2a.translation import extract_response


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
