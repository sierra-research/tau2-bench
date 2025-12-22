"""
Shared fixtures for evaluation store integration tests.

Provides fixtures for:
- A2A agent connectivity checking
- Isolated temporary data directories
- Evaluation store and event logger instances
- Mock domain tasks for testing
"""

import os
from pathlib import Path

import httpx
import pytest

from tau2.store import EvaluationStore, EventLogger, create_event_logger, create_store

# Default A2A agent endpoint (can be overridden via environment)
DEFAULT_A2A_AGENT_ENDPOINT = "http://tau2-agent:8001/a2a/tau2_agent"
AGENT_HEALTH_TIMEOUT = 5  # seconds


@pytest.fixture(scope="session")
def a2a_agent_endpoint() -> str:
    """Return the A2A agent endpoint URL from environment or default."""
    return os.environ.get("TAU2_A2A_AGENT_ENDPOINT", DEFAULT_A2A_AGENT_ENDPOINT)


@pytest.fixture(scope="session")
def agent_available(a2a_agent_endpoint: str) -> bool:
    """Check if the A2A agent is reachable.

    Tries multiple methods to verify agent availability:
    1. Agent card at base URL /.well-known/agent.json
    2. Agent card at agent path /.well-known/agent.json
    3. Direct endpoint check (405 Method Not Allowed is valid)

    Returns:
        True if agent responds, False otherwise.
    """
    from urllib.parse import urlparse

    parsed = urlparse(a2a_agent_endpoint)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    try:
        # Method 1: Try agent card at base URL
        response = httpx.get(
            f"{base_url}/.well-known/agent.json",
            timeout=AGENT_HEALTH_TIMEOUT,
        )
        if response.status_code == 200:
            return True

        # Method 2: Try agent card at agent path
        response = httpx.get(
            f"{a2a_agent_endpoint}/.well-known/agent.json",
            timeout=AGENT_HEALTH_TIMEOUT,
        )
        if response.status_code == 200:
            return True

        # Method 3: Check if endpoint exists (405 Method Not Allowed is valid)
        response = httpx.get(
            a2a_agent_endpoint,
            timeout=AGENT_HEALTH_TIMEOUT,
        )
        # 405 = Method Not Allowed means the endpoint exists
        return response.status_code in (200, 405)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


@pytest.fixture
def skip_if_agent_unavailable(agent_available: bool, a2a_agent_endpoint: str):
    """Skip the test if the A2A agent is not available."""
    if not agent_available:
        pytest.skip(
            f"A2A agent not available at {a2a_agent_endpoint}. "
            "Set TAU2_A2A_AGENT_ENDPOINT or start the agent."
        )


@pytest.fixture
def integration_data_dir(tmp_path: Path) -> Path:
    """Create an isolated temporary data directory for integration tests.

    Sets TAU2_DATA_DIR environment variable for the test.
    Automatically cleans up after test completion.

    Yields:
        Path to the temporary data directory.
    """
    data_dir = tmp_path / "integration_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Store old environment value
    old_env = os.environ.get("TAU2_DATA_DIR")
    os.environ["TAU2_DATA_DIR"] = str(data_dir)

    try:
        yield data_dir
    finally:
        # Restore environment
        if old_env is not None:
            os.environ["TAU2_DATA_DIR"] = old_env
        else:
            os.environ.pop("TAU2_DATA_DIR", None)


@pytest.fixture
def integration_store(integration_data_dir: Path) -> EvaluationStore:
    """Create an evaluation store instance for integration testing."""
    return create_store(integration_data_dir)


@pytest.fixture
def integration_logger(integration_data_dir: Path) -> EventLogger:
    """Create an event logger for integration testing (stdout disabled)."""
    return create_event_logger(data_dir=integration_data_dir, stdout=False)


@pytest.fixture
def mock_task_config() -> dict:
    """Configuration for mock domain evaluation."""
    return {
        "domain": "mock",
        "num_tasks": 3,
        "num_trials": 1,
        "max_steps": 15,
        "max_errors": 5,
    }
