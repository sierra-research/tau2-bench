"""
Fixtures for A2A end-to-end tests.

These fixtures provide real server instances and clients for E2E testing.
The test suite manages its own isolated servers to avoid conflicts with
any user-running servers.

Key design decisions:
- MODULE-SCOPED SERVERS: Each test module gets its own fresh server instances,
  providing complete isolation between test modules. This prevents state leakage
  from error/timeout tests affecting subsequent modules.
- tau2_agent and mock_agent run on SEPARATE ports to avoid async deadlock
- Uses A2EServer dataclass to encapsulate server state
- CONNECTION DRAIN: Module boundary fixtures ensure connections are properly
  drained between test modules with health verification
- SAFE CLOSE: All async client cleanup uses timeout guards to prevent hanging
- SSE streaming helpers for evaluation request handling
- EvaluationStore verification fixtures
- PARALLEL EXECUTION: pytest-xdist worker isolation via disjoint port ranges
"""

import asyncio
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
from dotenv import load_dotenv

from tau2.a2a.client import A2AClient
from tau2.a2a.models import A2AConfig
from tau2_agent.utils import SSEEvent, SSEParser

# Load .env for NEBIUS_API_KEY and other credentials
load_dotenv()


def pytest_configure(config):
    """Configure pytest-xdist worker isolation with disjoint port ranges.

    Each xdist worker gets a unique 10-port range to avoid collisions:
    - gw0: 8700-8709
    - gw1: 8710-8719
    - gw2: 8720-8729
    - etc.

    This enables parallel execution of module-scoped server fixtures
    without port conflicts between workers.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id:
        # Extract worker number from id like "gw0", "gw1", etc.
        worker_num = int(worker_id.replace("gw", ""))
        port_base = 8700 + (worker_num * 10)
        os.environ["A2A_E2E_TAU2_PORT"] = str(port_base)
        os.environ["A2A_E2E_MOCK_PORT"] = str(port_base + 1)


# Test configuration - use unique ports to avoid conflicts
# CRITICAL: tau2_agent and mock_agent MUST run on SEPARATE ports to avoid
# async deadlock. When both run on the same port, the evaluation request from
# tau2_agent to the mock agent blocks the event loop that needs to handle
# the mock agent's request.
#
# Port assignments (dynamic for xdist workers):
# - 8765: Legacy single-server (deprecated)
# - 8766/8767: test_datadog_e2e (tau2_agent/mock_agent)
# - 8768/8769: test_a2a_e2e default (tau2_agent/mock_agent)
# - 8700+N*10: xdist worker N base port
ADK_SERVER_HOST = "localhost"


def get_worker_ports() -> tuple[int, int]:
    """Get port assignments for this worker.

    For pytest-xdist workers, returns unique port ranges to avoid conflicts.
    Falls back to default ports for non-parallel execution.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id:
        worker_num = int(worker_id.replace("gw", ""))
        port_base = 8700 + (worker_num * 10)
        return port_base, port_base + 1
    # Default ports for serial execution
    tau2_port = int(os.environ.get("A2A_E2E_TAU2_PORT", "8768"))
    mock_port = int(os.environ.get("A2A_E2E_MOCK_PORT", "8769"))
    return tau2_port, mock_port


# Legacy constants for backwards compatibility (will be overridden by fixtures)
TAU2_AGENT_PORT, MOCK_AGENT_PORT = get_worker_ports()
TAU2_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{TAU2_AGENT_PORT}"
MOCK_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{MOCK_AGENT_PORT}"
SERVER_STARTUP_TIMEOUT = 60  # seconds
SERVER_HEALTH_CHECK_INTERVAL = 0.5  # seconds

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Connection management constants
CONNECTION_DRAIN_DELAY = 0.5  # seconds to wait for connections to drain
SAFE_CLOSE_TIMEOUT = 5.0  # seconds to wait for client close
HEALTH_CHECK_RETRIES = 10  # number of health check retries
HEALTH_CHECK_RETRY_DELAY = 0.5  # seconds between health check retries


async def _safe_close(client: httpx.AsyncClient, timeout: float = SAFE_CLOSE_TIMEOUT):
    """Close httpx client with timeout guard.

    Ensures client cleanup completes within a timeout, preventing
    hanging connections from blocking test teardown.

    Args:
        client: The httpx.AsyncClient to close
        timeout: Maximum seconds to wait for close (default: 5.0)
    """
    try:
        async with asyncio.timeout(timeout):
            await client.aclose()
    except (TimeoutError, Exception):
        # Connection will be cleaned up by garbage collection
        pass


def _create_temp_data_dir(base_dir: Path) -> Path:
    """Create and configure a temporary data directory.

    Creates a temp directory with:
    - sessions/ for EvaluationStore session data
    - evaluations/ for EvaluationStore completed evaluations
    - tau2/ symlinked to project's data/tau2/ for domain task files

    Args:
        base_dir: Base directory to create subdirectories in

    Returns:
        Path: The configured data directory
    """
    # Create subdirectories for EvaluationStore
    (base_dir / "sessions").mkdir(exist_ok=True)
    (base_dir / "evaluations").mkdir(exist_ok=True)

    # Symlink tau2 domains from project data directory
    # This allows tau2-bench to find domain task files when TAU2_DATA_DIR is set
    source_tau2_dir = PROJECT_ROOT / "data" / "tau2"
    target_tau2_dir = base_dir / "tau2"
    if source_tau2_dir.exists() and not target_tau2_dir.exists():
        target_tau2_dir.symlink_to(source_tau2_dir)

    return base_dir


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
    port: int


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_port(
    start_port: int, host: str = "localhost", max_attempts: int = 100
) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: The port to start searching from.
        host: The host to check port availability on.
        max_attempts: Maximum number of ports to try.

    Returns:
        int: An available port number.

    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port, host):
            return port
    msg = f"No available port found in range {start_port}-{start_port + max_attempts}"
    raise RuntimeError(msg)


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


def sse_event_to_dict(event: SSEEvent) -> dict | None:
    """Convert SSEEvent to dict format expected by tests.

    Args:
        event: Parsed SSEEvent from SSEParser.

    Returns:
        dict or None: Parsed event data with _event_type field if present.
        If JSON parsing fails, returns {"_raw": data, "_event_type": event_type}
    """
    parsed = event.json()
    if parsed is not None:
        if event.event:
            parsed["_event_type"] = event.event
        return parsed
    if event.data:
        return {"_raw": event.data, "_event_type": event.event}
    return None


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


def get_user_llm_headers() -> dict[str, str]:
    """Get user LLM headers from environment variables."""
    headers = {}
    model = os.environ.get("USER_LLM_MODEL") or os.environ.get("TEST_USER_LLM_MODEL")
    api_key = os.environ.get("USER_LLM_API_KEY") or os.environ.get("NEBIUS_API_KEY")

    # Default model when using Nebius API
    if api_key and not model and os.environ.get("NEBIUS_API_KEY"):
        model = "nebius/Qwen/Qwen3-30B-A3B-Thinking-2507"

    if model:
        headers["X-User-LLM-Model"] = model
    if api_key:
        headers["X-User-LLM-API-Key"] = api_key
    return headers


async def send_a2a_evaluation_request(
    endpoint: str,
    domain: str = "mock",
    agent_endpoint: str = "http://mock-agent:8000",
    num_tasks: int = 2,
    num_trials: int = 1,
    stream: bool = True,
    timeout: float = 180.0,
    user_llm_model: str | None = None,
    user_llm_api_key: str | None = None,
) -> AsyncIterator[dict]:
    """Send an A2A evaluation request and stream SSE events.

    Args:
        endpoint: The A2A endpoint URL
        domain: The tau2 domain to evaluate
        agent_endpoint: URL of the agent to evaluate
        num_tasks: Number of tasks to run
        num_trials: Number of trials per task
        stream: Whether to use SSE streaming (default: True)
        timeout: Request timeout in seconds (default: 180)
        user_llm_model: LLM model for user simulator (falls back to env)
        user_llm_api_key: API key for user simulator (falls back to env)

    Yields:
        dict: Parsed SSE event data containing evaluation progress/results
    """
    request = build_a2a_evaluation_request(
        domain=domain,
        agent_endpoint=agent_endpoint,
        num_tasks=num_tasks,
        num_trials=num_trials,
    )

    # Build headers with user LLM credentials
    headers = {"Accept": "text/event-stream"}
    if user_llm_model:
        headers["X-User-LLM-Model"] = user_llm_model
    if user_llm_api_key:
        headers["X-User-LLM-API-Key"] = user_llm_api_key

    # Fall back to environment if not explicitly provided
    if "X-User-LLM-Model" not in headers or "X-User-LLM-API-Key" not in headers:
        env_headers = get_user_llm_headers()
        for k, v in env_headers.items():
            if k not in headers:
                headers[k] = v

    if stream:
        request["method"] = "message/stream"

        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", endpoint, json=request, headers=headers) as response,
        ):
            response.raise_for_status()

            parser = SSEParser()
            async for chunk in response.aiter_text():
                for event in parser.feed(chunk):
                    event_data = sse_event_to_dict(event)
                    if event_data:
                        yield event_data

            # Flush any remaining buffered event
            for event in parser.flush():
                event_data = sse_event_to_dict(event)
                if event_data:
                    yield event_data
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=request, headers=headers)
            response.raise_for_status()
            yield response.json()


@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    """Create an isolated temporary data directory for the test module.

    Each test module gets its own fresh data directory, providing
    complete isolation between test modules.

    Creates a temp directory with:
    - sessions/ for EvaluationStore session data
    - evaluations/ for EvaluationStore completed evaluations
    - tau2/ symlinked to project's data/tau2/ for domain task files
    """
    data_dir = tmp_path_factory.mktemp("a2a_e2e_data")
    return _create_temp_data_dir(data_dir)


@pytest.fixture(scope="module")
def mock_agent_server(tmp_path_factory):
    """
    Start a separate ADK server for simple_nebius_agent on a different port.

    This fixture is module-scoped, meaning each test module gets its own
    fresh server instance. This provides complete isolation between test
    modules, preventing state leakage from error/timeout tests affecting
    subsequent test modules.

    The server starts simple_nebius_agent on a SEPARATE port to avoid the async
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

    # Find an available port dynamically
    mock_agent_port = find_available_port(MOCK_AGENT_PORT, ADK_SERVER_HOST)
    mock_agent_base_url = f"http://{ADK_SERVER_HOST}:{mock_agent_port}"
    mock_agent_endpoint = f"{mock_agent_base_url}/a2a/{agent_name}"
    agent_card_url = f"{mock_agent_endpoint}/.well-known/agent-card.json"

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
        str(mock_agent_port),
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
            endpoint=mock_agent_base_url,
            agent_endpoint=mock_agent_endpoint,
            port=mock_agent_port,
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

        # Verify port is released
        for _ in range(10):
            if not is_port_in_use(mock_agent_port, ADK_SERVER_HOST):
                break
            time.sleep(0.1)


@pytest.fixture(scope="module")
def a2e_server(temp_data_dir, mock_agent_server) -> A2EServer:
    """
    Start ADK server for tau2_agent with isolated data directory.

    This fixture is module-scoped, meaning each test module gets its own
    fresh server instance. Combined with module-scoped mock_agent_server
    and temp_data_dir, this provides complete isolation between test
    modules.

    The fixture starts tau2_agent on a separate port from mock_agent_server
    to avoid async deadlock issues during evaluation.

    Yields:
        A2EServer: Server info including processes, data_dir, and endpoints
    """
    # Find an available port dynamically
    tau2_agent_port = find_available_port(TAU2_AGENT_PORT, ADK_SERVER_HOST)
    tau2_agent_base_url = f"http://{ADK_SERVER_HOST}:{tau2_agent_port}"
    tau2_agent_endpoint = f"{tau2_agent_base_url}/a2a/tau2_agent"
    agent_card_url = f"{tau2_agent_endpoint}/.well-known/agent-card.json"

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

    # Start custom server for tau2_agent with credentials middleware
    # Uses tau2_agent.server which wraps ADK with our middleware
    env["AGENTS_DIR"] = str(tau2_agents_dir)
    env["PORT"] = str(tau2_agent_port)
    env["HOST"] = ADK_SERVER_HOST
    cmd = [
        "python",
        "-m",
        "tau2_agent.server",
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

        # Verify port is released
        for _ in range(10):
            if not is_port_in_use(tau2_agent_port, ADK_SERVER_HOST):
                break
            time.sleep(0.1)


# Legacy fixture for backwards compatibility with existing tests
@pytest.fixture(scope="module")
def adk_server(a2e_server) -> str:
    """
    Legacy fixture that returns just the tau2_agent endpoint URL.

    For new tests, prefer using a2e_server directly for access to
    both endpoints and the data directory.

    Yields:
        str: The tau2_agent endpoint URL
    """
    return a2e_server.tau2_agent_endpoint


@pytest_asyncio.fixture(scope="module", autouse=True, loop_scope="module")
async def _module_server_health_gate(a2e_server: A2EServer):
    """
    Verify server health at module boundaries.

    This autouse fixture runs automatically for each test module:
    1. At module start: verifies both servers are healthy
    2. At module end: drains connections and verifies recovery

    This provides defense-in-depth alongside module-scoped servers,
    ensuring clean state even if individual test cleanup fails.
    """
    # === Module startup: verify servers are ready ===
    async with httpx.AsyncClient(timeout=5.0) as client:
        for endpoint_name, endpoint_url in [
            ("tau2_agent", a2e_server.tau2_agent_endpoint),
            ("mock_agent", a2e_server.mock_agent_endpoint),
        ]:
            for _attempt in range(HEALTH_CHECK_RETRIES):
                try:
                    response = await client.get(
                        f"{endpoint_url}/.well-known/agent-card.json"
                    )
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(HEALTH_CHECK_RETRY_DELAY)
            else:
                pytest.fail(
                    f"{endpoint_name} health check failed after {HEALTH_CHECK_RETRIES} attempts"
                )

    yield  # Tests in this module run here

    # === Module teardown: drain connections ===
    # Allow TIME_WAIT sockets and in-flight requests to complete
    await asyncio.sleep(CONNECTION_DRAIN_DELAY)

    # Verify servers recovered (especially after error/timeout tests)
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _endpoint_name, endpoint_url in [
            ("tau2_agent", a2e_server.tau2_agent_endpoint),
            ("mock_agent", a2e_server.mock_agent_endpoint),
        ]:
            try:
                response = await client.get(
                    f"{endpoint_url}/.well-known/agent-card.json"
                )
                if response.status_code != 200:
                    # Log warning but don't fail - server will be torn down anyway
                    pass
            except httpx.HTTPError:
                # Server may already be shutting down, which is fine
                pass


@pytest_asyncio.fixture
async def a2a_client_to_local(a2e_server):
    """
    Create A2AClient connected to local ADK server.

    This fixture provides a real A2AClient that communicates with
    the local tau2_agent server over HTTP. Uses timeout-guarded cleanup
    to prevent hanging connections from blocking test teardown.

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
        await _safe_close(http_client)
        pytest.fail(
            f"Failed to connect to ADK server at {a2e_server.tau2_agent_endpoint}: {e}"
        )

    yield client

    # Cleanup with timeout guard to prevent hanging on abandoned connections
    await _safe_close(http_client)


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


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _cached_evaluation(a2e_server: A2EServer) -> EvaluationResult:
    """Run one evaluation and cache results for module.

    This fixture is module-scoped so all tests in a file share
    the same evaluation result, reducing 6+ evaluations to 1.

    Uses pytest-asyncio's event loop management to avoid conflicts
    with the function-scoped loops used by async tests.
    """
    events = []

    async for event in send_a2a_evaluation_request(
        endpoint=a2e_server.tau2_agent_endpoint,
        domain="mock",
        agent_endpoint=a2e_server.mock_agent_endpoint,
        num_tasks=1,
        num_trials=1,
    ):
        events.append(event)

    # Find and load the evaluation file
    eval_files = list(a2e_server.evaluations_dir.glob("*.json"))
    eval_file = None
    eval_data = None

    if eval_files:
        eval_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        eval_file = eval_files[0]
        with open(eval_file, encoding="utf-8") as f:
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
