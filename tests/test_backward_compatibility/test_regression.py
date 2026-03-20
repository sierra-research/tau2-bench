"""
Regression test suite for backward compatibility.

Tests critical imports and interfaces to catch breaking changes.
"""

import pytest
from unittest.mock import Mock, patch

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from tau2.environment.tool import Tool


@pytest.fixture
def mock_tools():
    """Create mock tools."""
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool.parameters = {"type": "object", "properties": {}}
    return [tool]


@pytest.mark.unit
class TestLLMAgentRegressionBehavior:
    """Test that LLM agent behavior is unchanged."""

    @patch("tau2.agent.llm_agent.generate")
    def test_simple_conversation_unchanged(self, mock_generate, mock_tools):
        """Test simple back-and-forth conversation produces same results."""
        responses = [
            AssistantMessage(role="assistant", content="Hello! How can I help?", tool_calls=[]),
            AssistantMessage(role="assistant", content="I can help you with that.", tool_calls=[]),
        ]
        mock_generate.side_effect = responses

        agent = LLMAgent(
            tools=mock_tools,
            domain_policy="Test policy",
            llm="gpt-4o",
            llm_args={}
        )

        state = agent.get_init_state()
        msg1 = UserMessage(role="user", content="Hello")
        response1, state = agent.generate_next_message(msg1, state)

        msg2 = UserMessage(role="user", content="I need help")
        response2, state = agent.generate_next_message(msg2, state)

        assert response1.content == "Hello! How can I help?"
        assert response2.content == "I can help you with that."
        assert len(state.messages) == 4  # 2 user + 2 assistant

    @patch("tau2.agent.llm_agent.generate")
    def test_tool_calling_sequence_unchanged(self, mock_generate, mock_tools):
        """Test tool calling sequence produces same results."""
        tool_call = ToolCall(
            id="call_1",
            name="test_tool",
            arguments={"param": "value"}
        )
        mock_generate.return_value = AssistantMessage(
            role="assistant",
            content="",
            tool_calls=[tool_call]
        )

        agent = LLMAgent(
            tools=mock_tools,
            domain_policy="Test policy",
            llm="gpt-4o",
            llm_args={}
        )

        state = agent.get_init_state()
        msg = UserMessage(role="user", content="Use the tool")
        response, state = agent.generate_next_message(msg, state)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert response.tool_calls[0].arguments == {"param": "value"}


@pytest.mark.unit
class TestAgentStateRegressionBehavior:
    """Test agent state management is unchanged."""

    def test_init_state_structure_unchanged(self, mock_tools):
        """Test get_init_state returns expected structure."""
        agent = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy",
            llm="gpt-4o",
            llm_args={}
        )

        state = agent.get_init_state()

        assert hasattr(state, "system_messages")
        assert hasattr(state, "messages")
        assert len(state.system_messages) == 1
        assert len(state.messages) == 0

    def test_state_serialization_unchanged(self, mock_tools):
        """Test state can be serialized/deserialized."""
        agent = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy",
            llm="gpt-4o",
            llm_args={}
        )

        state = agent.get_init_state()
        state_dict = state.model_dump()

        from tau2.agent.llm_agent import LLMAgentState
        restored_state = LLMAgentState(**state_dict)

        assert restored_state.system_messages == state.system_messages
        assert restored_state.messages == state.messages


@pytest.mark.unit
class TestAgentInterfaceRegressionBehavior:
    """Test BaseAgent interface unchanged."""

    def test_base_agent_interface_methods_exist(self):
        """Test BaseAgent interface has required methods."""
        from tau2.agent.base import BaseAgent

        assert hasattr(BaseAgent, "generate_next_message")
        assert hasattr(BaseAgent, "stop")
        assert hasattr(BaseAgent, "get_init_state")
        assert hasattr(BaseAgent, "is_stop")
        assert hasattr(BaseAgent, "set_seed")

    def test_llm_agent_implements_interface(self, mock_tools):
        """Test LLMAgent implements BaseAgent interface."""
        from tau2.agent.base import BaseAgent

        agent = LLMAgent(
            tools=mock_tools,
            domain_policy="Policy",
            llm="gpt-4o",
            llm_args={}
        )

        assert isinstance(agent, BaseAgent)
        assert callable(agent.generate_next_message)
        assert callable(agent.stop)
        assert callable(agent.get_init_state)
        assert callable(agent.is_stop)
        assert callable(agent.set_seed)


@pytest.mark.unit
class TestImportRegressionBehavior:
    """Test imports unchanged - critical for catching breaking changes."""

    def test_core_imports_unchanged(self):
        """Test core imports still work."""
        from tau2.agent.llm_agent import LLMAgent, LLMAgentState
        from tau2.agent.base import BaseAgent, LocalAgent, ValidAgentInputMessage
        from tau2.data_model.message import (
            UserMessage,
            AssistantMessage,
            ToolMessage,
            MultiToolMessage,
        )
        from tau2.registry import registry
        from tau2.run import run_domain, get_options

        assert LLMAgent is not None
        assert BaseAgent is not None
        assert registry is not None
        assert run_domain is not None

    def test_a2a_agent_registered(self):
        """Test A2A agent is properly registered."""
        from tau2.registry import registry

        agents = registry.get_agents()
        assert "a2a_agent" in agents

    def test_legacy_agents_still_registered(self):
        """Test pre-A2A agents still registered."""
        from tau2.registry import registry

        agents = registry.get_agents()
        assert "llm_agent" in agents
        assert "llm_agent_gt" in agents
        assert "llm_agent_solo" in agents
