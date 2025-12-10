"""
Integration test for ADK tools (T024).

Tests that RunTau2Evaluation and ListDomains tools work correctly.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tau2_agent.tools.get_evaluation_results import GetEvaluationResults
from tau2_agent.tools.list_domains import ListDomains
from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


@pytest.fixture
def mock_tool_context():
    """Create mock tool context for ADK tools"""
    context = Mock()
    context.session = Mock()
    context.session.state = {}
    return context


@pytest.mark.asyncio
async def test_list_domains_tool(mock_tool_context):
    """Test ListDomains tool returns available domains"""
    tool = ListDomains(
        name="list_domains",
        description="List all available tau2-bench evaluation domains",
    )

    result = await tool.run_async(args={}, tool_context=mock_tool_context)

    assert "domains" in result, "ListDomains should return domains list"
    domains = result["domains"]
    assert len(domains) > 0, "Should have at least one domain"

    # Check required domains are present
    domain_names = [d["name"] for d in domains]
    assert "airline" in domain_names, "Should include airline domain"
    assert "retail" in domain_names, "Should include retail domain"
    assert "telecom" in domain_names, "Should include telecom domain"

    # Check domain structure
    for domain in domains:
        assert "name" in domain, "Domain should have name"
        assert "description" in domain, "Domain should have description"
        assert "num_tasks" in domain, "Domain should have num_tasks"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_tool_success(mock_tool_context):
    """Test RunTau2Evaluation tool with successful evaluation"""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    # Create mock task with proper structure
    mock_task = Mock()
    mock_task.id = "task-1"
    mock_task.description = Mock()
    mock_task.description.purpose = "Test purpose"

    # Create mock reward_info for simulations
    mock_reward_info = Mock()
    mock_reward_info.reward = 1.0  # Successful reward

    # Create mock simulations with proper structure
    mock_simulations = []
    for _ in range(10):
        sim = Mock()
        sim.task_id = "task-1"
        sim.task = mock_task
        sim.success = True
        sim.reward_info = mock_reward_info
        mock_simulations.append(sim)

    mock_results = Mock()
    mock_results.timestamp = "2025-11-24T10:00:00Z"
    mock_results.simulations = mock_simulations
    mock_results.tasks = [mock_task]

    # Mock both run_domain and compute_metrics to avoid complex pandas operations
    mock_metrics = Mock()
    mock_metrics.avg_reward = 1.0
    mock_metrics.pass_hat_ks = {1: 1.0}
    mock_metrics.avg_agent_cost = 0.001

    # Patch at source since imports are inside _execute function
    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "user_llm": "gpt-4o",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

    assert result["status"] == "completed", "Evaluation should complete successfully"
    assert result["timestamp"] == "2025-11-24T10:00:00Z", "Should include timestamp"
    assert "summary" in result, "Should include summary"
    assert result["summary"]["successful_simulations"] == 10, (
        "All simulations should succeed"
    )
    assert result["summary"]["total_simulations"] == 10, "Should have 10 simulations"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_tool_invalid_domain(mock_tool_context):
    """Test RunTau2Evaluation tool with invalid domain"""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    with pytest.raises(ValueError, match="Invalid domain"):
        await tool.run_async(
            args={
                "domain": "invalid_domain",
                "agent_endpoint": "https://agent.example.com",
                "user_llm": "gpt-4o",
            },
            tool_context=mock_tool_context,
        )


@pytest.mark.asyncio
async def test_get_evaluation_results_tool(mock_tool_context):
    """Test GetEvaluationResults tool (placeholder implementation)"""
    tool = GetEvaluationResults(
        name="get_evaluation_results", description="Get evaluation results"
    )

    result = await tool.run_async(
        args={"evaluation_id": "eval-123"},
        tool_context=mock_tool_context,
    )

    # For now, this tool returns an error message
    assert "error" in result or "message" in result, (
        "GetEvaluationResults should return error/message"
    )
