"""
Fixtures for A2A end-to-end tests.

These fixtures provide real server instances and clients for E2E testing.
The test suite manages its own isolated servers to avoid conflicts with
any user-running servers.

Key design decisions:
- tau2_agent and mock_agent run on SEPARATE ports to avoid async deadlock
- Uses A2EServer dataclass to encapsulate server state
- SSE streaming helpers for evaluation request handling
- EvaluationStore verification fixtures
"""

import json
import os
import signal
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from tau2.a2a.client import A2AClient
from tau2.a2a.models import A2AConfig

# Test configuration - use unique ports to avoid conflicts
# CRITICAL: tau2_agent and mock_agent MUST run on SEPARATE ports to avoid
# async deadlock. When both run on the same port, the evaluation request from
# tau2_agent to the mock agent blocks the event loop that needs to handle
# the mock agent's request.
#
# Port assignments:
# - 8765: Legacy single-server (deprecated)
# - 8766/8767: test_datadog_e2e (tau2_agent/mock_agent)
# - 8768/8769: test_a2a_e2e (tau2_agent/mock_agent)
ADK_SERVER_HOST = "localhost"
TAU2_AGENT_PORT = int(os.environ.get("A2A_E2E_TAU2_PORT", "8768"))
MOCK_AGENT_PORT = int(os.environ.get("A2A_E2E_MOCK_PORT", "8769"))
TAU2_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{TAU2_AGENT_PORT}"
MOCK_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{MOCK_AGENT_PORT}"
SERVER_STARTUP_TIMEOUT = 60  # seconds
SERVER_HEALTH_CHECK_INTERVAL = 0.5  # seconds

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class A2EServer:
    """Represents running ADK servers for A2A E2E tests.

    Encapsulates the state of both tau2_agent and mock_agent servers,
    providing computed properties for common paths and endpoints.
    """

    tau2_process: subprocess.Popen
    mock_process: subprocess.Popen
    data_dir: Path
    tau2_agent_endpoint: str
    mock_agent_endpoint: str

    @property
    def evaluations_dir(self) -> Path:
        """Path to the evaluations directory."""
        return self.data_dir / "evaluations"

    @property
    def sessions_dir(self) -> Path:
        """Path to the sessions directory."""
        return self.data_dir / "sessions"


@dataclass
class MockAgentServer:
    """Represents a running mock agent server for evaluation targets."""

    process: subprocess.Popen
    endpoint: str
    agent_endpoint: str


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_agent() -> str | None:
    """
    Find the first project directory that appears to be a runnable ADK agent.

    Checks a preferred candidate ("simple_nebius_agent") first, then scans
    PROJECT_ROOT for any directory that contains both `agent.py` and `__init__.py`.

    Returns:
        The name of the first directory that looks like a valid agent, or `None`
        if no such directory is found.
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


def parse_sse_event(event_text: str) -> dict | None:
    """Parse a single SSE event into a dictionary.

    Args:
        event_text: Raw SSE event text (e.g., "event: message\\ndata: {...}")

    Returns:
        dict or None: Parsed event data, or None if not parseable.
        If JSON parsing fails, returns {"_raw": data_str, "_event_type": event_type}
    """
    lines = event_text.strip().split("\n")
    event_type = None
    data_lines = []

    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif line.startswith(":"):
            # Comment line, skip
            continue

    if not data_lines:
        return None

    data_str = "".join(data_lines)
    try:
        data = json.loads(data_str)
        if event_type:
            data["_event_type"] = event_type
        return data
    except json.JSONDecodeError:
        return {"_raw": data_str, "_event_type": event_type}


def build_a2a_evaluation_request(
    domain: str,
    agent_endpoint: str,
    num_tasks: int = 2,
    num_trials: int = 1,
    message_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    """Build a JSON-RPC 2.0 A2A message requesting tau2 evaluation.

    Args:
        domain: The tau2 domain to evaluate (e.g., "mock", "airline")
        agent_endpoint: URL of the agent to evaluate
        num_tasks: Number of tasks to run (default: 2)
        num_trials: Number of trials per task (default: 1)
        message_id: Optional message ID (auto-generated if not provided)
        request_id: Optional JSON-RPC request ID (auto-generated if not provided)

    Returns:
        dict: JSON-RPC 2.0 formatted A2A request
    """
    return {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id or str(uuid.uuid4()),
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Run an evaluation on the {domain} domain for agent at "
                            f"{agent_endpoint}. Use {num_tasks} tasks and {num_trials} trial(s)."
                        )
                    }
                ],
            }
        },
        "id": request_id or str(uuid.uuid4()),
    }


async def send_a2a_evaluation_request(
    endpoint: str,
    domain: str = "mock",
    agent_endpoint: str = "http://mock-agent:8000",
    num_tasks: int = 2,
    num_trials: int = 1,
    stream: bool = True,
    timeout: float = 180.0,
) -> AsyncIterator[dict]:
    """Send an A2A evaluation request and stream SSE events.

    Args:
        endpoint: The A2A endpoint URL (e.g., "http://localhost:8768/a2a/tau2_agent")
        domain: The tau2 domain to evaluate
        agent_endpoint: URL of the agent to evaluate
        num_tasks: Number of tasks to run
        num_trials: Number of trials per task
        stream: Whether to use SSE streaming (default: True)
        timeout: Request timeout in seconds (default: 180)

    Yields:
        dict: Parsed SSE event data containing evaluation progress/results
    """
    request = build_a2a_evaluation_request(
        domain=domain,
        agent_endpoint=agent_endpoint,
        num_tasks=num_tasks,
        num_trials=num_trials,
    )

    if stream:
        # Use message/stream for SSE streaming
        request["method"] = "message/stream"

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "POST",
                endpoint,
                json=request,
                headers={"Accept": "text/event-stream"},
            ) as response,
        ):
            response.raise_for_status()

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk

                # Process complete SSE events (separated by \n\n or single lines)
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    event_data = parse_sse_event(event_text)
                    if event_data:
                        yield event_data

                # Process single-line events
                lines = buffer.split("\n")
                complete_lines = lines[:-1]
                buffer = lines[-1] if lines else ""

                for line in complete_lines:
                    if line.strip():
                        event_data = parse_sse_event(line)
                        if event_data:
                            yield event_data

            # Handle remaining buffer
            if buffer.strip():
                event_data = parse_sse_event(buffer)
                if event_data:
                    yield event_data
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=request)
            response.raise_for_status()
            yield response.json()


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    """Create an isolated temporary data directory for the test session.

    Creates a temp directory with:
    - sessions/ for EvaluationStore session data
    - evaluations/ for EvaluationStore completed evaluations
    - tau2/ symlinked to project's data/tau2/ for domain task files
    """
    data_dir = tmp_path_factory.mktemp("a2a_e2e_data")
    # Create subdirectories for EvaluationStore
    (data_dir / "sessions").mkdir(exist_ok=True)
    (data_dir / "evaluations").mkdir(exist_ok=True)

    # Symlink tau2 domains from project data directory
    # This allows tau2-bench to find domain task files when TAU2_DATA_DIR is set
    source_tau2_dir = PROJECT_ROOT / "data" / "tau2"
    if source_tau2_dir.exists():
        target_tau2_dir = data_dir / "tau2"
        target_tau2_dir.symlink_to(source_tau2_dir)

    return data_dir


@pytest.fixture(scope="session")
def mock_agent_server(tmp_path_factory):
    """
    Start a separate ADK server for simple_nebius_agent on a different port.

    This fixture starts simple_nebius_agent on a SEPARATE port to avoid the async
    deadlock that occurs when tau2_agent and simple_nebius_agent share the same
    ADK server. The deadlock happens because:
    - tau2_agent blocks its event loop waiting for run_in_executor
    - A2AAgent in run_domain makes HTTP request back to same server
    - Server can't process new request while blocked on the first one

    Yields:
        MockAgentServer: Server info including process and endpoint
    """
    agent_name = find_available_agent()
    if agent_name is None:
        pytest.skip(
            "No valid ADK agent found. Create an agent directory with "
            "agent.py and __init__.py"
        )

    mock_agent_endpoint = f"{MOCK_AGENT_BASE_URL}/a2a/{agent_name}"
    agent_card_url = f"{mock_agent_endpoint}/.well-known/agent-card.json"

    # Check if port is already in use
    if is_port_in_use(MOCK_AGENT_PORT):
        pytest.fail(
            f"Port {MOCK_AGENT_PORT} is already in use. "
            f"Set A2A_E2E_MOCK_PORT to use a different port."
        )

    # Create a temp directory with agent symlinked
    mock_agents_dir = tmp_path_factory.mktemp("mock_agents")
    mock_agent_link = mock_agents_dir / agent_name
    mock_agent_link.symlink_to(PROJECT_ROOT / agent_name)

    # Build environment
    env = os.environ.copy()

    # Start ADK server for mock agent only
    cmd = [
        "adk",
        "api_server",
        "--a2a",
        str(mock_agents_dir),
        "--port",
        str(MOCK_AGENT_PORT),
        "--host",
        ADK_SERVER_HOST,
    ]

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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
                    f"Mock agent server terminated unexpectedly.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )

            time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

        if not server_ready:
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"Mock agent server did not become ready within {SERVER_STARTUP_TIMEOUT}s.\n"
                f"URL checked: {agent_card_url}\n"
                f"Last error: {last_error}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )

        yield MockAgentServer(
            process=process,
            endpoint=MOCK_AGENT_BASE_URL,
            agent_endpoint=mock_agent_endpoint,
        )

    finally:
        # Cleanup: stop server and entire process group
        if process is not None and process.poll() is None:
            try:
                if os.name != "nt":
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


@pytest.fixture(scope="session")
def a2e_server(temp_data_dir, mock_agent_server) -> A2EServer:
    """
    Start ADK server for tau2_agent with isolated data directory.

    This fixture starts tau2_agent on a separate port from mock_agent_server
    to avoid async deadlock issues during evaluation.

    Yields:
        A2EServer: Server info including processes, data_dir, and endpoints
    """
    tau2_agent_endpoint = f"{TAU2_AGENT_BASE_URL}/a2a/tau2_agent"
    agent_card_url = f"{tau2_agent_endpoint}/.well-known/agent-card.json"

    # Check if port is already in use
    if is_port_in_use(TAU2_AGENT_PORT):
        pytest.fail(
            f"Port {TAU2_AGENT_PORT} is already in use. "
            f"Set A2A_E2E_TAU2_PORT to use a different port."
        )

    # Build environment
    env = os.environ.copy()

    # Use isolated data directory for EvaluationStore
    env["TAU2_DATA_DIR"] = str(temp_data_dir)

    # Create a temp directory with only tau2_agent symlinked
    tau2_agents_dir = temp_data_dir / "agents"
    tau2_agents_dir.mkdir(exist_ok=True)
    tau2_agent_link = tau2_agents_dir / "tau2_agent"
    if not tau2_agent_link.exists():
        tau2_agent_link.symlink_to(PROJECT_ROOT / "tau2_agent")

    # Start ADK server for tau2_agent only (mock agent runs separately)
    cmd = [
        "adk",
        "api_server",
        "--a2a",
        str(tau2_agents_dir),
        "--port",
        str(TAU2_AGENT_PORT),
        "--host",
        ADK_SERVER_HOST,
    ]

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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
                    f"tau2_agent server terminated unexpectedly.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )

            time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

        if not server_ready:
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"tau2_agent server did not become ready within {SERVER_STARTUP_TIMEOUT}s.\n"
                f"URL checked: {agent_card_url}\n"
                f"Last error: {last_error}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )

        yield A2EServer(
            tau2_process=process,
            mock_process=mock_agent_server.process,
            data_dir=temp_data_dir,
            tau2_agent_endpoint=tau2_agent_endpoint,
            mock_agent_endpoint=mock_agent_server.agent_endpoint,
        )

    finally:
        # Cleanup: stop server and entire process group
        if process is not None and process.poll() is None:
            try:
                if os.name != "nt":
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


# Legacy fixture for backwards compatibility with existing tests
@pytest.fixture(scope="session")
def adk_server(a2e_server) -> str:
    """
    Legacy fixture that returns just the tau2_agent endpoint URL.

    For new tests, prefer using a2e_server directly for access to
    both endpoints and the data directory.

    Yields:
        str: The tau2_agent endpoint URL
    """
    return a2e_server.tau2_agent_endpoint


@pytest_asyncio.fixture
async def a2a_client_to_local(a2e_server):
    """
    Create A2AClient connected to local ADK server.

    This fixture provides a real A2AClient that communicates with
    the local tau2_agent server over HTTP.

    Args:
        a2e_server: A2EServer from a2e_server fixture

    Returns:
        A2AClient: Client connected to local server
    """
    config = A2AConfig(
        endpoint=a2e_server.tau2_agent_endpoint,
        timeout=120,  # Longer timeout for LLM responses
    )

    # Create httpx client with base_url and redirect following
    http_client = httpx.AsyncClient(
        base_url=a2e_server.tau2_agent_endpoint,
        timeout=120.0,  # Longer timeout for LLM responses
        follow_redirects=True,
    )
    client = A2AClient(config, http_client=http_client)

    # Verify connection by discovering agent
    try:
        await client.discover_agent()
    except Exception as e:
        await http_client.aclose()
        pytest.fail(
            f"Failed to connect to ADK server at {a2e_server.tau2_agent_endpoint}: {e}"
        )

    yield client

    # Cleanup
    await http_client.aclose()


@pytest.fixture
def evaluation_store(a2e_server):
    """Provide access to EvaluationStore for verification."""
    # Set the data dir environment for store operations
    os.environ["TAU2_DATA_DIR"] = str(a2e_server.data_dir)

    from tau2.store import create_store

    return create_store()


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
async def verify_server_health(a2e_server):
    """
    Verify ADK server health before each test.

    This fixture ensures the server is responding correctly
    before running each E2E test.

    Args:
        a2e_server: A2EServer from a2e_server fixture
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Check tau2_agent endpoint
            response = await client.get(
                f"{a2e_server.tau2_agent_endpoint}/.well-known/agent-card.json"
            )
            assert response.status_code == 200, (
                f"tau2_agent health check failed: {response.status_code}"
            )

            # Check mock_agent endpoint
            response = await client.get(
                f"{a2e_server.mock_agent_endpoint}/.well-known/agent-card.json"
            )
            assert response.status_code == 200, (
                f"mock_agent health check failed: {response.status_code}"
            )

        except Exception as e:
            pytest.fail(f"Server health check failed: {e}")


@pytest.fixture
def mock_evaluation_agent_endpoint(a2e_server):
    """
    Provide agent endpoint for evaluation testing.

    Returns:
        str: Mock agent endpoint URL for evaluations
    """
    return a2e_server.mock_agent_endpoint


# =============================================================================
# Module-scoped evaluation fixtures for test consolidation
# =============================================================================
# These fixtures run ONE evaluation per module and cache results,
# avoiding the cost of 20+ evaluations when tests could share one.


@dataclass
class EvaluationResult:
    """Cached result from a single evaluation run."""

    events: list[dict]
    eval_file: Path | None
    eval_data: dict | None


@pytest.fixture(scope="module")
def _cached_evaluation(a2e_server: A2EServer, request) -> EvaluationResult:
    """Run one evaluation and cache results for module.

    This fixture is module-scoped so all tests in a file share
    the same evaluation result, reducing 6+ evaluations to 1.
    """
    import asyncio

    events = []

    async def run_eval():
        async for event in send_a2a_evaluation_request(
            endpoint=a2e_server.tau2_agent_endpoint,
            domain="mock",
            agent_endpoint=a2e_server.mock_agent_endpoint,
            num_tasks=1,
            num_trials=1,
        ):
            events.append(event)

    # Run the async evaluation with a fresh event loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run_eval())
    finally:
        loop.close()

    # Find and load the evaluation file
    eval_files = list(a2e_server.evaluations_dir.glob("*.json"))
    eval_file = None
    eval_data = None

    if eval_files:
        eval_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        eval_file = eval_files[0]
        with open(eval_file) as f:
            eval_data = json.load(f)

    return EvaluationResult(events=events, eval_file=eval_file, eval_data=eval_data)


@pytest.fixture(scope="module")
def evaluation_events(_cached_evaluation: EvaluationResult) -> list[dict]:
    """SSE events from cached evaluation."""
    return _cached_evaluation.events


@pytest.fixture(scope="module")
def evaluation_data(_cached_evaluation: EvaluationResult) -> dict:
    """Evaluation JSON data from cached evaluation."""
    assert _cached_evaluation.eval_data is not None, "No evaluation file found"
    return _cached_evaluation.eval_data


@pytest.fixture(scope="module")
def evaluation_file(_cached_evaluation: EvaluationResult) -> Path:
    """Path to evaluation file from cached evaluation."""
    assert _cached_evaluation.eval_file is not None, "No evaluation file found"
    return _cached_evaluation.eval_file
