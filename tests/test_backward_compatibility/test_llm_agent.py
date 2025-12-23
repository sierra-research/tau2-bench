"""
Test backward compatibility for LLM agent interface.

Verifies that existing LLM agent interface methods work unchanged after A2A integration.
These tests complement tests/test_agent.py with interface-specific checks.
"""

from unittest.mock import Mock, patch

import pytest

from tau2.agent.base import LocalAgent
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.environment.tool import Tool


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    tool1 = Mock(spec=Tool)
    tool1.name = "search_flights"
    tool1.description = "Search for available flights"
    tool1.parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"}
        },
        "required": ["origin", "destination"]
    }
    return [tool1]


@pytest.fixture
def llm_agent(mock_tools):
    """Create an LLMAgent instance."""
    return LLMAgent(
        tools=mock_tools,
        domain_policy="Help users with flight bookings.",
        llm="gpt-4o",
        llm_args={"temperature": 0.7}
    )


class TestLLMAgentInterface:
    """Test that LLMAgent interface methods are unchanged."""

    def test_stop_method_exists(self, llm_agent):
        """Test stop method exists and can be called."""
        state = llm_agent.get_init_state()
        user_msg = UserMessage(role="user", content="Goodbye")

        # Should not raise exception
        llm_agent.stop(message=user_msg, state=state)

    def test_is_stop_method_exists(self, llm_agent):
        """Test is_stop class method exists."""
        msg = AssistantMessage(role="assistant", content="Goodbye!", tool_calls=[])
        result = llm_agent.is_stop(msg)
        assert result is False  # Default behavior

    def test_set_seed_method_exists(self, llm_agent):
        """Test set_seed method exists and can be called."""
        # Should not raise exception
        llm_agent.set_seed(42)


class TestLLMAgentNoBreakingChanges:
    """Test that no breaking changes were introduced."""

    def test_llm_agent_imports_unchanged(self):
        """Test that LLMAgent imports work as before."""
        from tau2.agent.base import BaseAgent
        from tau2.agent.llm_agent import LLMGTAgent, LLMSoloAgent

        assert LLMAgent is not None
        assert LLMAgentState is not None
        assert LLMGTAgent is not None
        assert LLMSoloAgent is not None
        assert BaseAgent is not None
        assert LocalAgent is not None

    def test_llm_agent_inherits_from_local_agent(self):
        """Test LLMAgent still inherits from LocalAgent."""
        assert issubclass(LLMAgent, LocalAgent)

    def test_llm_agent_constructor_signature_unchanged(self, mock_tools):
        """Test LLMAgent constructor accepts same parameters."""
        # Full parameters
        agent1 = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy",
            llm="gpt-4o",
            llm_args={"temperature": 0.7}
        )
        assert agent1 is not None

        # Optional parameters as None
        agent2 = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy",
            llm=None,
            llm_args=None
        )
        assert agent2 is not None

        # Minimal parameters
        agent3 = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy"
        )
        assert agent3 is not None
