"""
Fixtures for A2A end-to-end tests.

These fixtures provide real server instances and clients for E2E testing.
The test suite manages its own isolated server to avoid conflicts with
any user-running servers.
"""

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from tau2.a2a.client import A2AClient
from tau2.a2a.models import A2AConfig

# Test configuration - use unique port to avoid conflicts
ADK_SERVER_HOST = "localhost"
ADK_SERVER_PORT = int(os.environ.get("E2E_TEST_PORT", "8765"))  # Unique test port
ADK_SERVER_BASE_URL = f"http://{ADK_SERVER_HOST}:{ADK_SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 30  # seconds
SERVER_HEALTH_CHECK_INTERVAL = 0.5  # seconds

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_available_agent() -> str | None:
    """
    Find the first project directory that appears to be a runnable ADK agent.

    Checks a preferred candidate ("simple_nebius_agent") first, then scans PROJECT_ROOT for any directory that contains both `agent.py` and `__init__.py`.

    Returns:
        The name of the first directory that looks like a valid agent, or `None` if no such directory is found.
    """
    # Priority list of agent directories to check
    # Note: tau2_agent is the evaluator, not a target agent for evaluation
    agent_candidates = [
        "simple_nebius_agent",
    ]

    for agent_name in agent_candidates:
        agent_dir = PROJECT_ROOT / agent_name
        agent_py = agent_dir / "agent.py"
        init_py = agent_dir / "__init__.py"

        if agent_dir.exists() and agent_py.exists() and init_py.exists():
            return agent_name

    # Fallback: scan for any valid agent directory
    for item in PROJECT_ROOT.iterdir():
        if (
            item.is_dir()
            and not item.name.startswith((".", "_", "test", "src"))
            and (item / "agent.py").exists()
            and (item / "__init__.py").exists()
        ):
            return item.name

    return None


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


@pytest.fixture(scope="session")
def adk_server():
    """
    Start ADK server as subprocess for the test session.

    This fixture starts its own isolated ADK dev server on a unique port
    and waits for it to be ready. The server is automatically stopped
    when tests complete.

    Yields:
        str: The agent endpoint URL (including /a2a/{agent_name} path)
    """
    # Find an available agent
    agent_name = find_available_agent()
    if agent_name is None:
        pytest.skip(
            "No valid ADK agent found. Create an agent directory with "
            "agent.py and __init__.py"
        )

    agent_endpoint = f"{ADK_SERVER_BASE_URL}/a2a/{agent_name}"
    agent_card_url = f"{agent_endpoint}/.well-known/agent-card.json"

    # Check if our test port is already in use
    if is_port_in_use(ADK_SERVER_PORT):
        # Try to see if it's our expected agent
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                # Port has our agent, reuse it
                yield agent_endpoint
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        pytest.fail(
            f"Port {ADK_SERVER_PORT} is already in use by another process. "
            f"Set E2E_TEST_PORT env var to use a different port, or stop "
            f"the process using port {ADK_SERVER_PORT}."
        )

    # Ensure agent directory exists
    agent_dir = PROJECT_ROOT / agent_name
    if not agent_dir.exists():
        pytest.fail(f"Agent directory not found: {agent_dir.absolute()}")

    # Start ADK server using adk api_server command
    cmd = [
        "adk",
        "api_server",
        "--a2a",
        str(PROJECT_ROOT),
        "--port",
        str(ADK_SERVER_PORT),
        "--host",
        ADK_SERVER_HOST,
    ]

    process = None
    try:
        # Start server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Use process group for clean shutdown
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        last_error = None

        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                response = httpx.get(agent_card_url, timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    break
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e

            # Check if process crashed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(
                    f"ADK server process terminated unexpectedly.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )

            time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

        if not server_ready:
            # Terminate and get output
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"ADK server did not become ready within {SERVER_STARTUP_TIMEOUT}s.\n"
                f"Agent: {agent_name}\n"
                f"URL checked: {agent_card_url}\n"
                f"Last error: {last_error}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )

        yield agent_endpoint

    finally:
        # Cleanup: stop server and entire process group
        if process is not None and process.poll() is None:
            try:
                if os.name != "nt":
                    # Kill entire process group on Unix
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
                except (ProcessLookupError, OSError):
                    pass


@pytest_asyncio.fixture
async def a2a_client_to_local(adk_server):
    """
    Create A2AClient connected to local ADK server.

    This fixture provides a real A2AClient that communicates with
    the local ADK server over HTTP.

    Args:
        adk_server: Agent endpoint URL from adk_server fixture

    Returns:
        A2AClient: Client connected to local server
    """
    config = A2AConfig(
        endpoint=adk_server,
        timeout=120,  # Longer timeout for LLM responses
    )

    # Create httpx client with base_url and redirect following
    http_client = httpx.AsyncClient(
        base_url=adk_server,
        timeout=120.0,  # Longer timeout for LLM responses
        follow_redirects=True,
    )
    client = A2AClient(config, http_client=http_client)

    # Verify connection by discovering agent
    try:
        await client.discover_agent()
    except Exception as e:
        await http_client.aclose()
        pytest.fail(f"Failed to connect to ADK server at {adk_server}: {e}")

    yield client

    # Cleanup
    await http_client.aclose()


@pytest.fixture
def sample_test_tools():
    """
    Create sample tools for testing E2E flows.

    These tools match the structure expected by tau2-bench domains
    but are simpler for testing purposes.

    Returns:
        list[Tool]: Sample tool instances
    """
    from tau2.environment.tool import Tool

    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Search for available flights."""
        return {
            "flights": [
                {
                    "id": "TEST123",
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "price": 350.0,
                    "departure": "10:00",
                    "arrival": "14:00",
                }
            ]
        }

    def book_flight(flight_id: str, passenger_name: str, passenger_email: str) -> dict:
        """Book a specific flight."""
        return {
            "booking_id": f"BK-{flight_id}-001",
            "confirmation": f"Booked flight {flight_id} for {passenger_name}",
            "status": "confirmed",
        }

    def cancel_booking(booking_id: str) -> dict:
        """Cancel a flight booking."""
        return {"status": "cancelled", "booking_id": booking_id, "refund_amount": 350.0}

    return [
        Tool(search_flights),
        Tool(book_flight),
        Tool(cancel_booking),
    ]


@pytest_asyncio.fixture
async def verify_server_health(adk_server):
    """
    Verify ADK server health before each test.

    This fixture ensures the server is responding correctly
    before running each E2E test.

    Args:
        adk_server: Agent endpoint URL from adk_server fixture
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Check agent card endpoint
            response = await client.get(f"{adk_server}/.well-known/agent-card.json")
            assert response.status_code == 200, (
                f"Server health check failed: {response.status_code}"
            )

            # Verify agent card structure
            card = response.json()
            assert "name" in card, "Agent card missing required 'name' field"

        except Exception as e:
            pytest.fail(f"Server health check failed: {e}")


@pytest.fixture
def mock_evaluation_agent_endpoint(adk_server):
    """
    Provide agent endpoint for evaluation testing.

    Returns:
        str: Agent endpoint URL (includes /a2a/{agent_name} path)
    """
    return adk_server


@pytest_asyncio.fixture
async def wait_for_async():
    """
    Provide a helper that awaits a coroutine with a timeout and fails the test on timeout.

    Intended for use in async tests; returns a callable that awaits the given coroutine up to `timeout` seconds.

    Parameters:
        coro (Awaitable): The coroutine or awaitable to run.
        timeout (float): Maximum seconds to wait before failing the test (default 30).

    Returns:
        The value produced by awaiting `coro`. If the timeout is reached, the test is failed via `pytest.fail`.
    """

    async def wait_with_timeout(coro, timeout=30):
        """
        Await a coroutine and fail the test if it does not complete within the given timeout.

        Parameters:
            coro (Awaitable): The coroutine or awaitable to run.
            timeout (float): Maximum seconds to wait for completion (default 30).

        Returns:
            The result produced by the awaited coroutine.

        Raises:
            pytest.fail: Fails the current test if the operation times out.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {timeout}s")

    return wait_with_timeout
