"""
Fixtures for ADK server tests.

Provides mock tau2 backends and helper fixtures for testing
ADK tools and streaming events.
"""

from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from google.adk.events.event import Event


@pytest.fixture
def mock_tool_context():
    """Create mock tool context for ADK tools."""
    context = Mock()
    context.session = Mock()
    context.session.state = {}
    context.invocation_id = "test-invocation-123"
    return context


@pytest.fixture
def mock_tau2_backend():
    """Fast mock that returns controllable evaluation results.

    This fixture patches tau2.run.run_domain to return quickly without
    making actual LLM calls, enabling fast integration tests.
    """
    mock_tasks = []
    mock_simulations = []

    for i in range(3):
        mock_task = Mock()
        mock_task.id = f"task-{i + 1}"
        mock_task.description = Mock()
        mock_task.description.purpose = f"Test purpose {i + 1}"
        mock_tasks.append(mock_task)

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

    mock_registry = Mock()
    mock_registry.get_domains.return_value = ["airline", "retail", "telecom", "mock"]

    return {
        "results": mock_results,
        "metrics": mock_metrics,
        "tasks": mock_tasks,
        "registry": mock_registry,
    }


@pytest.fixture
def apply_tau2_mocks(mock_tau2_backend):
    """Apply tau2 mocks for integration testing.

    Returns a context manager that patches all tau2 components needed
    for running evaluations without actual LLM calls.
    """

    def create_patches():
        return (
            patch("tau2.run.run_domain", return_value=mock_tau2_backend["results"]),
            patch("tau2.run.load_tasks", return_value=mock_tau2_backend["tasks"]),
            patch("tau2.registry.registry", mock_tau2_backend["registry"]),
            patch(
                "tau2.metrics.agent_metrics.compute_metrics",
                return_value=mock_tau2_backend["metrics"],
            ),
            patch("tau2.metrics.agent_metrics.is_successful", return_value=True),
        )

    return create_patches


@pytest_asyncio.fixture
async def streaming_tool():
    """Provide RunTau2Evaluation tool configured for streaming tests."""
    from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

    tool = RunTau2Evaluation(
        name="run_tau2_evaluation",
        description="Run tau2-bench evaluation",
    )

    return tool


@pytest_asyncio.fixture
async def collect_streaming_events(mock_tool_context, streaming_tool, apply_tau2_mocks):
    """Helper fixture to collect all streaming events from a tool invocation.

    Returns an async function that runs the tool and collects all yielded events.
    """

    async def _collect_events(
        domain: str = "airline",
        agent_endpoint: str = "https://test-agent.example.com",
        num_tasks: int | None = None,
        num_trials: int = 1,
    ) -> list[Event]:
        """Run tool and collect all emitted events."""
        events = []
        patches = apply_tau2_mocks()

        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            async for event in streaming_tool.run_async(
                args={
                    "domain": domain,
                    "agent_endpoint": agent_endpoint,
                    "num_trials": num_trials,
                    **({"num_tasks": num_tasks} if num_tasks else {}),
                },
                tool_context=mock_tool_context,
            ):
                events.append(event)

        return events

    return _collect_events


@pytest.fixture
def extract_event_states():
    """Helper to extract states from a list of events."""

    def _extract(events: list[Event]) -> list[str]:
        return [
            e.custom_metadata.get("tau2.state") for e in events if e.custom_metadata
        ]

    return _extract


@pytest.fixture
def filter_events_by_state():
    """Helper to filter events by state."""

    def _filter(events: list[Event], state: str) -> list[Event]:
        return [
            e
            for e in events
            if e.custom_metadata and e.custom_metadata.get("tau2.state") == state
        ]

    return _filter
