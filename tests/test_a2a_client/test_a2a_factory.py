"""
Tests for A2A agent factory and build_agent integration.

Verifies that create_a2a_agent and build_agent correctly wire CLI arguments
through to A2AAgent instances with proper A2AConfig.
"""

from unittest.mock import Mock

import pytest

from tau2.agent.a2a_agent import A2AAgent, create_a2a_agent
from tau2.environment.tool import Tool


@pytest.fixture
def mock_tools():
    """Create mock tools for factory tests."""
    tool = Mock(spec=Tool)
    tool.name = "lookup"
    tool.description = "Look up information"
    tool.parameters = {"type": "object", "properties": {}}
    return [tool]


class TestCreateA2AAgent:
    """Test the create_a2a_agent factory function."""

    def test_create_with_required_args(self, mock_tools):
        """Factory with just endpoint produces a valid A2AAgent."""
        agent = create_a2a_agent(
            tools=mock_tools,
            domain_policy="Be helpful.",
            a2a_agent_args={"endpoint": "http://localhost:8080"},
        )

        assert isinstance(agent, A2AAgent)
        assert agent.config.endpoint == "http://localhost:8080"
        assert agent.config.auth_token is None
        assert agent.config.timeout == 300

    def test_create_with_all_args(self, mock_tools):
        """Factory with all args passes them through to A2AConfig."""
        agent = create_a2a_agent(
            tools=mock_tools,
            domain_policy="Be helpful.",
            a2a_agent_args={
                "endpoint": "https://agent.example.com",
                "auth_token": "bearer-xyz",
                "timeout": 600,
            },
        )

        assert agent.config.endpoint == "https://agent.example.com"
        assert agent.config.auth_token == "bearer-xyz"
        assert agent.config.timeout == 600

    def test_missing_endpoint_raises(self, mock_tools):
        """Factory raises KeyError when endpoint is missing from dict."""
        with pytest.raises(KeyError):
            create_a2a_agent(
                tools=mock_tools,
                domain_policy="Policy",
                a2a_agent_args={},
            )

    def test_no_a2a_args_raises(self, mock_tools):
        """Factory raises when a2a_agent_args is not provided."""
        with pytest.raises(KeyError):
            create_a2a_agent(
                tools=mock_tools,
                domain_policy="Policy",
            )

    def test_auth_token_in_config(self, mock_tools):
        """auth_token flows from dict to A2AConfig."""
        agent = create_a2a_agent(
            tools=mock_tools,
            domain_policy="Policy",
            a2a_agent_args={
                "endpoint": "http://localhost:5000",
                "auth_token": "secret-token",
            },
        )

        assert agent.config.auth_token == "secret-token"


class TestBuildAgentA2AIntegration:
    """Test build_agent creates A2AAgent with correct config."""

    def test_build_agent_creates_a2a(self, mock_tools):
        """build_agent('a2a_agent', ...) returns A2AAgent with correct config."""
        from tau2.runner.build import build_agent

        mock_env = Mock()
        mock_env.get_policy.return_value = "Test policy"
        mock_env.get_tools.return_value = mock_tools

        a2a_args = {
            "endpoint": "http://localhost:9090",
            "auth_token": "tok-abc",
            "timeout": 120,
        }

        agent = build_agent(
            "a2a_agent",
            mock_env,
            a2a_agent_args=a2a_args,
        )

        assert isinstance(agent, A2AAgent)
        assert agent.config.endpoint == "http://localhost:9090"
        assert agent.config.auth_token == "tok-abc"
        assert agent.config.timeout == 120
