# Copyright Sierra
"""Agent-context clause for the nativeness judge."""

from tau2.judges.nativeness.agent_voice import agent_context


def test_agent_context_includes_gender_when_known():
    ctx = agent_context("nova", "tiffany")
    assert ctx is not None
    assert "voice agent" in ctx
    assert "Agent gender: female" in ctx


def test_agent_context_labels_agent_and_caller_gender_separately():
    ctx = agent_context("openai", "cedar", caller_gender="female")
    assert ctx is not None
    assert "Agent gender: male" in ctx
    assert "Caller gender: female" in ctx
    assert "directly addresses or refers to the caller" in ctx


def test_agent_context_no_gender_clause_when_neutral():
    ctx = agent_context("openai", "alloy")
    assert ctx is not None
    assert "voice agent" in ctx
    assert "known agent gender" not in ctx.lower()


def test_agent_context_includes_recorded_customer_gender():
    ctx = agent_context("openai", "alloy", "male")
    assert ctx is not None
    assert "Caller gender: male" in ctx


def test_agent_context_none_when_nothing_known():
    assert agent_context(None) is None
