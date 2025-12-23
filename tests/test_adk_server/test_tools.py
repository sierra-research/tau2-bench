"""
Integration test for ADK tools (T024).

Tests that RunTau2Evaluation and ListDomains tools work correctly.
"""

from unittest.mock import Mock, patch

import pytest
from google.adk.events.event import Event

from tau2_agent.tools.get_evaluation_results import GetEvaluationResults
from tau2_agent.tools.list_domains import ListDomains
from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

# Mark all tests in this module as mock-based (no real endpoints)
pytestmark = pytest.mark.a2a_mock


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
async def test_run_tau2_evaluation_tool_success(mock_tool_context):
    """Test RunTau2Evaluation tool with successful evaluation returns streaming events"""
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
        patch("tau2.run.load_tasks", return_value=[mock_task]),
        patch("tau2.registry.registry", _create_mock_registry()),
        patch("tau2.metrics.agent_metrics.compute_metrics", return_value=mock_metrics),
        patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
    ):
        # run_async now returns an async iterator, collect events
        events = []
        async for event in tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "user_llm": "gpt-4o",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        ):
            events.append(event)

    # Should have multiple events
    assert len(events) >= 2, "Should have at least submitted and result events"

    # Final event should be completed with results
    last_event = events[-1]
    assert last_event.custom_metadata.get("tau2.state") == "completed", (
        "Final event should have state='completed'"
    )

    # Check results in final event content
    content_text = last_event.content.parts[0].text if last_event.content else ""
    assert "10/10" in content_text or "Results:" in content_text, (
        "Result event should contain results info"
    )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_tool_invalid_domain(mock_tool_context):
    """Test RunTau2Evaluation tool with invalid domain emits error event"""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    # Invalid domain should emit error event, not raise exception
    events = []
    async for event in tool.run_async(
        args={
            "domain": "invalid_domain",
            "agent_endpoint": "https://agent.example.com",
            "user_llm": "gpt-4o",
        },
        tool_context=mock_tool_context,
    ):
        events.append(event)

    # Should emit an error event
    assert len(events) >= 1, "Should emit at least one event"
    error_event = events[-1]
    assert error_event.custom_metadata.get("tau2.state") == "failed", (
        "Invalid domain should result in failed state"
    )
    assert "tau2.error" in error_event.custom_metadata, (
        "Error event should have tau2.error"
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
        sim = Mock()
        sim.task_id = mock_task.id
        sim.task = mock_task
        sim.success = True
        sim.reward_info = mock_reward_info
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
async def test_run_tau2_evaluation_yields_events(mock_tool_context):
    """Test RunTau2Evaluation yields Event objects (not just dict)."""
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
        # run_async returns an async generator directly, iterate over it
        events = []
        async for event in tool.run_async(
            args={
                "domain": "airline",
                "agent_endpoint": "https://agent.example.com",
                "num_trials": 1,
            },
            tool_context=mock_tool_context,
        ):
            assert isinstance(event, Event), f"Expected Event, got {type(event)}"
            events.append(event)

        assert len(events) >= 2, "Should yield at least submitted and result events"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_submitted_event(mock_tool_context):
    """Test first event has state='submitted'."""
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
        events = [
            event
            async for event in tool.run_async(
                args={
                    "domain": "airline",
                    "agent_endpoint": "https://agent.example.com",
                    "num_trials": 1,
                },
                tool_context=mock_tool_context,
            )
        ]

        # First event should be submitted state
        first_event = events[0]
        assert first_event.custom_metadata.get("tau2.state") == "submitted", (
            "First event should have state='submitted'"
        )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_progress_events(mock_tool_context):
    """Test working events emitted during evaluation."""
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
        events = [
            event
            async for event in tool.run_async(
                args={
                    "domain": "airline",
                    "agent_endpoint": "https://agent.example.com",
                    "num_trials": 1,
                },
                tool_context=mock_tool_context,
            )
        ]

        # Should have progress events with state='working'
        working_events = [
            e for e in events if e.custom_metadata.get("tau2.state") == "working"
        ]

        # At minimum, working events should be emitted per task completion
        assert len(working_events) >= 1, "Should emit at least one working event"

        # Working events should have progress metadata
        for event in working_events:
            assert "tau2.progress" in event.custom_metadata, (
                "Working events should include tau2.progress"
            )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_result_event(mock_tool_context):
    """Test final event contains results."""
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
        events = [
            event
            async for event in tool.run_async(
                args={
                    "domain": "airline",
                    "agent_endpoint": "https://agent.example.com",
                    "num_trials": 1,
                },
                tool_context=mock_tool_context,
            )
        ]

        # Last event should be completed state with results
        last_event = events[-1]
        assert last_event.custom_metadata.get("tau2.state") == "completed", (
            "Last event should have state='completed'"
        )
        assert last_event.custom_metadata.get("tau2.progress") == 100, (
            "Last event should have progress=100"
        )

        # Content should include results
        content_text = last_event.content.parts[0].text if last_event.content else ""
        assert "Results:" in content_text, "Result event should contain results"


@pytest.mark.asyncio
async def test_run_tau2_evaluation_error_event(mock_tool_context):
    """Test failed state on error."""
    tool = RunTau2Evaluation(
        name="run_tau2_evaluation", description="Run tau2-bench evaluation"
    )

    mock_results, mock_metrics, mock_tasks = _create_mock_evaluation_results(
        num_tasks=2
    )

    with (
        patch("tau2.run.run_domain", side_effect=RuntimeError("Evaluation failed")),
        patch("tau2.run.load_tasks", return_value=mock_tasks),
        patch("tau2.registry.registry", _create_mock_registry()),
    ):
        events = [
            event
            async for event in tool.run_async(
                args={
                    "domain": "airline",
                    "agent_endpoint": "https://agent.example.com",
                    "num_trials": 1,
                },
                tool_context=mock_tool_context,
            )
        ]

        # Should emit an error event
        error_events = [
            e for e in events if e.custom_metadata.get("tau2.state") == "failed"
        ]

        assert len(error_events) >= 1, "Should emit error event on failure"

        error_event = error_events[0]
        assert "tau2.error" in error_event.custom_metadata, (
            "Error event should have tau2.error"
        )


@pytest.mark.asyncio
async def test_run_tau2_evaluation_trace_context(mock_tool_context):
    """Test events contain evaluation_id, task_id, domain for OTel instrumentation."""
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
        events = [
            event
            async for event in tool.run_async(
                args={
                    "domain": "airline",
                    "agent_endpoint": "https://agent.example.com",
                    "num_trials": 1,
                },
                tool_context=mock_tool_context,
            )
        ]

        # All events should have trace context metadata
        for event in events:
            metadata = event.custom_metadata

            # Required trace context fields for 007-datadog instrumentation
            assert "tau2.evaluation_id" in metadata, (
                "Events should include tau2.evaluation_id for tracing"
            )
            assert "tau2.domain" in metadata, (
                "Events should include tau2.domain for tracing"
            )
            assert "tau2.agent_endpoint" in metadata, (
                "Events should include tau2.agent_endpoint for tracing"
            )

        # Working events should also have current_task_id
        working_events = [
            e for e in events if e.custom_metadata.get("tau2.state") == "working"
        ]
        for event in working_events:
            assert "tau2.current_task_id" in event.custom_metadata, (
                "Working events should include tau2.current_task_id"
            )
