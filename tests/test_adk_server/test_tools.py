"""
Integration test for ADK tools (T024).

Tests that RunTau2Evaluation and ListDomains tools work correctly.
"""

from unittest.mock import Mock, patch

import pytest

from tau2_agent.context import user_llm_api_key, user_llm_model
from tau2_agent.tools.get_evaluation_results import GetEvaluationResults
from tau2_agent.tools.list_domains import ListDomains
from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


@pytest.fixture
def mock_user_credentials():
    """Set up mock user LLM credentials in context variables."""
    token_model = user_llm_model.set("gpt-4o")
    token_key = user_llm_api_key.set("test-api-key-12345")
    yield
    user_llm_model.reset(token_model)
    user_llm_api_key.reset(token_key)


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


def _create_mock_registry():
    """Create a mock registry that returns valid domains."""
    mock_registry = Mock()
    mock_registry.get_domains.return_value = ["airline", "retail", "telecom", "mock"]
    return mock_registry


@pytest.mark.asyncio
async def test_run_tau2_evaluation_tool_success(mock_tool_context, mock_user_credentials):
    """Test RunTau2Evaluation tool with successful evaluation returns results dict"""
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
    # model_dump is called to serialize reward_info
    mock_reward_info.model_dump = Mock(return_value={"reward": 1.0})

    # Create mock simulations with proper structure
    mock_simulations = []
    for _ in range(10):
        sim = Mock()
        sim.task_id = "task-1"
        sim.task = mock_task
        sim.success = True
        sim.reward_info = mock_reward_info
        # Required fields for simulation data extraction (007-datadog Phase 4.5)
        sim.duration = 10.5
        sim.termination_reason = Mock()
        sim.termination_reason.value = "user_stop"
        sim.messages = []  # Empty list, not a Mock object
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
        patch("tau2.run.load_tasks", return_value=[mock_task]),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        # run_async returns a dict directly
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

    # Should return completed status
    assert result["status"] == "completed", "Result should have status='completed'"

    # Check summary contains expected fields
    assert "summary" in result, "Result should contain summary"
    summary = result["summary"]
    assert summary["total_simulations"] == 10, "Should have 10 simulations"
    assert summary["successful_simulations"] == 10, "All simulations should succeed"

    # Check evaluation_id is present
    assert "evaluation_id" in result, "Result should contain evaluation_id"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_tool_invalid_domain(mock_tool_context, mock_user_credentials):
    """Test RunTau2Evaluation tool with invalid domain returns error dict"""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    # Invalid domain should return error dict, not raise exception
    result = await tool.run_async(
        args={
            "domain": "invalid_domain",
            "agent_endpoint": "https://agent.example.com",
        },
        tool_context=mock_tool_context,
    )

    # Should return an error response
    assert "error" in result, "Invalid domain should return error"
    assert "message" in result, "Error response should have message"
    assert "invalid_domain" in result["message"].lower() or "invalid" in result["message"].lower(), (
        "Error message should mention invalid domain"
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


# =============================================================================
# User Story 2: RunTau2Evaluation SSE Streaming Tests (T015, T016)
# =============================================================================


def _create_mock_evaluation_results(num_tasks: int = 3):
    """Helper to create mock tau2 evaluation results."""
    mock_tasks = []
    mock_simulations = []

    for i in range(num_tasks):
        mock_task = Mock()
        mock_task.id = f"task-{i + 1}"
        mock_task.description = Mock()
        mock_task.description.purpose = f"Test purpose {i + 1}"
        mock_tasks.append(mock_task)

        # Create mock simulation for each task
        mock_reward_info = Mock()
        mock_reward_info.reward = 1.0
        # model_dump is called to serialize reward_info (007-datadog Phase 4.5)
        mock_reward_info.model_dump = Mock(return_value={"reward": 1.0})
        sim = Mock()
        sim.task_id = mock_task.id
        sim.task = mock_task
        sim.success = True
        sim.reward_info = mock_reward_info
        # Required fields for simulation data extraction (007-datadog Phase 4.5)
        sim.duration = 10.5
        sim.termination_reason = Mock()
        sim.termination_reason.value = "user_stop"
        sim.messages = []  # Empty list, not a Mock object
        mock_simulations.append(sim)

    mock_results = Mock()
    mock_results.timestamp = "2025-12-23T10:00:00Z"
    mock_results.simulations = mock_simulations
    mock_results.tasks = mock_tasks

    mock_metrics = Mock()
    mock_metrics.avg_reward = 1.0
    mock_metrics.pass_hat_ks = {1: 1.0}
    mock_metrics.avg_agent_cost = 0.001

    return mock_results, mock_metrics, mock_tasks


@pytest.mark.asyncio
async def test_run_tau2_evaluation_returns_dict(mock_tool_context, mock_user_credentials):
    """Test RunTau2Evaluation returns a dict with expected structure."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        # run_async returns a dict directly
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "status" in result, "Result should have status"
        assert "summary" in result, "Result should have summary"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_completed_status(mock_tool_context, mock_user_credentials):
    """Test successful evaluation returns status='completed'."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        # Result should have completed status
        assert result.get("status") == "completed", (
            "Result should have status='completed'"
        )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_summary_metrics(mock_tool_context, mock_user_credentials):
    """Test result includes summary metrics for all tasks."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=3
    )

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        # Should have summary with metrics
        assert "summary" in result, "Result should have summary"
        summary = result["summary"]

        # Check required summary fields
        assert "total_simulations" in summary, "Summary should have total_simulations"
        assert "total_tasks" in summary, "Summary should have total_tasks"
        assert "successful_simulations" in summary, "Summary should have successful_simulations"
        assert "avg_reward" in summary, "Summary should have avg_reward"

        # Verify counts match mock data
        assert summary["total_tasks"] == 3, "Should have 3 tasks"
        assert summary["total_simulations"] == 3, "Should have 3 simulations"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_result_contains_tasks(mock_tool_context, mock_user_credentials):
    """Test result contains task details."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        # Result should have completed status
        assert result.get("status") == "completed", (
            "Result should have status='completed'"
        )

        # Result should contain tasks list
        assert "tasks" in result, "Result should contain tasks"
        tasks = result["tasks"]
        assert len(tasks) == 2, "Should have 2 tasks"

        # Each task should have required fields
        for task in tasks:
            assert "task_id" in task, "Task should have task_id"
            assert "purpose" in task, "Task should have purpose"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_error_handling(mock_tool_context, mock_user_credentials):
    """Test error handling returns error dict."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    _mock_results, _mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", side_effect=RuntimeError("Evaluation failed")),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        # Should return an error response
        assert "error" in result, "Error should return error dict"
        assert "message" in result, "Error response should have message"
        assert "Evaluation failed" in result["message"], (
            "Error message should contain error details"
        )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_includes_evaluation_id(mock_tool_context, mock_user_credentials):
    """Test result contains evaluation_id for tracing."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", return_value=mock_results),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        result = await tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        )

        # Result should include evaluation_id for tracing
        assert "evaluation_id" in result, (
            "Result should include evaluation_id for tracing"
        )
        assert result["evaluation_id"] is not None, (
            "evaluation_id should not be None"
        )

        # Result should include simulations data
        assert "simulations" in result, "Result should include simulations"
        assert len(result["simulations"]) == 2, "Should have 2 simulations"

        # Each simulation should have task_id for tracing
        for sim in result["simulations"]:
            assert "task_id" in sim, "Simulation should have task_id"
