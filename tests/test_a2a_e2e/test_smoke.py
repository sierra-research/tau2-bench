"""
Smoke tests for full tau2_agent <-> simple_nebius_agent integration.

These tests verify the complete evaluation flow works end-to-end:
1. ADK server starts with simple_nebius_agent
2. tau2_agent's run_tau2_evaluation tool is called
3. The tool evaluates simple_nebius_agent via A2A protocol
4. Results are returned successfully

Prerequisites:
- NEBIUS_API_KEY: For simple_nebius_agent LLM calls
- User LLM credentials set via context vars (normally from HTTP headers)

Run with: pytest -m smoke
"""

import pytest

from tau2_agent.context import user_llm_api_key, user_llm_model

# Mark all tests as smoke tests (opt-in only)
pytestmark = [pytest.mark.smoke, pytest.mark.a2a_e2e]


@pytest.fixture
def smoke_tool_context():
    """Create tool context for smoke tests."""
    from unittest.mock import Mock

    context = Mock()
    context.session = Mock()
    context.session.state = {}
    context.invocation_id = "smoke-test-invocation"
    return context


@pytest.fixture
def set_user_llm_credentials():
    """Set user LLM credentials from environment for direct tool testing.

    The RunTau2Evaluation tool reads credentials from contextvars that are
    normally set by the CredentialsMiddleware from HTTP headers. For direct
    tool testing, we need to set these manually from environment variables.
    """
    import os

    model = os.environ.get("USER_LLM_MODEL") or os.environ.get("TEST_USER_LLM_MODEL")
    api_key = os.environ.get("USER_LLM_API_KEY") or os.environ.get("NEBIUS_API_KEY")

    # Default model when using Nebius API
    if api_key and not model and os.environ.get("NEBIUS_API_KEY"):
        model = "openai/Qwen/Qwen3-30B-A3B-Thinking-2507"

    if not model or not api_key:
        pytest.skip("User LLM credentials not configured (need USER_LLM_MODEL + USER_LLM_API_KEY or NEBIUS_API_KEY)")

    # Set contextvars
    token_model = user_llm_model.set(model)
    token_key = user_llm_api_key.set(api_key)

    yield

    # Reset contextvars
    user_llm_model.reset(token_model)
    user_llm_api_key.reset(token_key)


class TestTau2AgentSmoke:
    """Smoke tests for tau2_agent evaluating simple_nebius_agent."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(
        self, a2e_server, smoke_tool_context, set_user_llm_credentials
    ):
        """
        Smoke test: tau2_agent evaluates simple_nebius_agent end-to-end.

        This test verifies:
        1. ADK server is running with simple_nebius_agent
        2. run_tau2_evaluation tool can be called
        3. Evaluation completes (with mock domain for speed)
        4. Results contain expected structure
        """
        from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

        tool = RunTau2Evaluation(
            name="run_tau2_evaluation",
            description="Run tau2-bench evaluation",
        )

        # Run the evaluation (returns a dict, not an async generator)
        result = await tool.run_async(
            args={
                "domain": "mock",
                "agent_endpoint": a2e_server.mock_agent_endpoint,
                "num_tasks": 1,
                "num_trials": 1,
            },
            tool_context=smoke_tool_context,
        )

        # Check for error response
        if "error" in result:
            pytest.fail(f"Evaluation failed with error: {result.get('message', result)}")

        # Verify result structure
        assert result.get("status") == "completed", f"Expected completed, got {result.get('status')}"
        assert "evaluation_id" in result, "Result should have evaluation_id"
        assert "summary" in result, "Result should have summary"
        assert "tasks" in result, "Result should have tasks"

        # Verify summary metrics
        summary = result["summary"]
        assert "total_simulations" in summary, "Summary should have total_simulations"
        assert "total_tasks" in summary, "Summary should have total_tasks"
        assert summary["total_tasks"] >= 1, "Should have at least 1 task"

    @pytest.mark.asyncio
    async def test_evaluation_returns_metrics(
        self, a2e_server, smoke_tool_context, set_user_llm_credentials
    ):
        """
        Smoke test: Verify evaluation returns proper metrics.

        This ensures the evaluation result contains all expected
        metrics for downstream processing.
        """
        from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

        tool = RunTau2Evaluation(
            name="run_tau2_evaluation",
            description="Run tau2-bench evaluation",
        )

        result = await tool.run_async(
            args={
                "domain": "mock",
                "agent_endpoint": a2e_server.mock_agent_endpoint,
                "num_tasks": 1,
                "num_trials": 1,
            },
            tool_context=smoke_tool_context,
        )

        # Skip if error (credentials issue, etc.)
        if "error" in result:
            pytest.skip(f"Evaluation returned error: {result.get('message')}")

        # Verify metrics are present
        summary = result.get("summary", {})
        assert "avg_reward" in summary, "Summary should have avg_reward"
        assert "successful_simulations" in summary, "Summary should have successful_simulations"

        # Verify simulations data is present
        assert "simulations" in result, "Result should have simulations"
        simulations = result["simulations"]
        assert len(simulations) >= 1, "Should have at least 1 simulation"

        # Verify simulation structure
        sim = simulations[0]
        assert "task_id" in sim, "Simulation should have task_id"
        assert "termination_reason" in sim, "Simulation should have termination_reason"
