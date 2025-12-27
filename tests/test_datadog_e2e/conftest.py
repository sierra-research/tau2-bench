"""
Fixtures for Datadog E2E observability tests.

These fixtures provide a tau2_agent ADK server with ddtrace enabled
and an isolated data directory for EvaluationStore verification.
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
from dotenv import load_dotenv

from tau2_agent.utils import SSEEvent, SSEParser

# Load environment variables from .env file (for NEBIUS_API_KEY, etc.)
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Test configuration - use unique ports to avoid conflicts
# CRITICAL: tau2_agent and the test agent MUST run on SEPARATE ports to avoid
# async deadlock. When both run on the same port, the evaluation request from
# tau2_agent to the test agent blocks the event loop that needs to handle
# the test agent's request.
ADK_SERVER_HOST = "localhost"
TAU2_AGENT_PORT = int(os.environ.get("DATADOG_E2E_TAU2_PORT", "8766"))
MOCK_AGENT_PORT = int(os.environ.get("DATADOG_E2E_MOCK_PORT", "8767"))
TAU2_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{TAU2_AGENT_PORT}"
MOCK_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{MOCK_AGENT_PORT}"
SERVER_STARTUP_TIMEOUT = 60  # Longer timeout for traced startup
SERVER_HEALTH_CHECK_INTERVAL = 0.5

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class TracedServer:
    """Represents a running ADK server with tracing enabled."""

    process: subprocess.Popen
    data_dir: Path
    endpoint: str
    tau2_agent_endpoint: str
    mock_agent_endpoint: str  # On separate port to avoid deadlock

    @property
    def evaluations_dir(self) -> Path:
        """Path to the evaluations directory."""
        return self.data_dir / "evaluations"


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


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    """Create an isolated temporary data directory for the test session.

    Creates a temp directory with:
    - sessions/ for EvaluationStore session data
    - evaluations/ for EvaluationStore completed evaluations
    - tau2/ symlinked to project's data/tau2/ for domain task files
    """
    data_dir = tmp_path_factory.mktemp("tau2_data")
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

    Uses simple_nebius_agent (real LLM via Nebius API) which properly integrates
    with tau2's A2A protocol for tool calls.

    Yields:
        MockAgentServer: Server info including process and endpoint
    """
    # Use simple_nebius_agent which properly integrates with tau2's A2A protocol
    agent_name = "simple_nebius_agent"

    # Find an available port dynamically
    mock_agent_port = find_available_port(MOCK_AGENT_PORT, ADK_SERVER_HOST)
    mock_agent_base_url = f"http://{ADK_SERVER_HOST}:{mock_agent_port}"
    mock_agent_endpoint = f"{mock_agent_base_url}/a2a/{agent_name}"
    agent_card_url = f"{mock_agent_endpoint}/.well-known/agent-card.json"

    # Create a temp directory with simple_nebius_agent symlinked
    # ADK expects AGENTS_DIR to contain subdirectories, each being an agent
    mock_agents_dir = tmp_path_factory.mktemp("mock_agents")
    mock_agent_link = mock_agents_dir / agent_name
    mock_agent_link.symlink_to(PROJECT_ROOT / agent_name)

    # Build environment with ddtrace for mock agent LLM tracing
    env = os.environ.copy()
    env["DD_TRACE_ENABLED"] = "1"
    env["DD_SERVICE"] = "simple-nebius-agent"
    env["DD_ENV"] = "test"
    if os.getenv("DD_API_KEY"):
        env["DD_LLMOBS_ENABLED"] = "1"
        env["DD_LLMOBS_AGENTLESS_ENABLED"] = "1"
        env["DD_LLMOBS_ML_APP"] = os.getenv("DD_LLMOBS_ML_APP", "tau2-bench-agent")

    # Start ADK server for mock agent with ddtrace-run for LLM tracing
    cmd = [
        "ddtrace-run",
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
                f"Mock agent server did not become ready "
                f"within {SERVER_STARTUP_TIMEOUT}s.\n"
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


@pytest.fixture(scope="session")
def traced_adk_server(temp_data_dir, mock_agent_server):
    """
    Start ADK server with ddtrace enabled via environment variables.

    This fixture starts an isolated ADK server with:
    - DD_TRACE_ENABLED=true for Datadog tracing
    - TAU2_DATA_DIR pointing to isolated temp directory
    - tau2_agent registered for evaluation requests
    - simple_nebius_agent runs on a SEPARATE server (via mock_agent_server fixture)

    Yields:
        TracedServer: Server info including process, data_dir, and endpoints
    """
    # Find an available port dynamically
    tau2_agent_port = find_available_port(TAU2_AGENT_PORT, ADK_SERVER_HOST)
    tau2_agent_base_url = f"http://{ADK_SERVER_HOST}:{tau2_agent_port}"
    tau2_agent_endpoint = f"{tau2_agent_base_url}/a2a/tau2_agent"
    agent_card_url = f"{tau2_agent_endpoint}/.well-known/agent-card.json"

    # Build environment with ddtrace configuration
    env = os.environ.copy()

    # Enable ddtrace via environment (auto-patches on import)
    env["DD_TRACE_ENABLED"] = "1"
    env["DD_SERVICE"] = "tau2-bench-agent"
    env["DD_ENV"] = "test"

    # Enable LLM Observability if DD_API_KEY is set
    if os.getenv("DD_API_KEY"):
        env["DD_LLMOBS_ENABLED"] = "1"
        env["DD_LLMOBS_AGENTLESS_ENABLED"] = "1"
        env["DD_LLMOBS_ML_APP"] = os.getenv("DD_LLMOBS_ML_APP", "tau2-bench-agent")

    # Use isolated data directory
    env["TAU2_DATA_DIR"] = str(temp_data_dir)

    # Create a temp directory with only tau2_agent symlinked
    # ADK expects AGENTS_DIR to contain subdirectories, each being an agent
    tau2_agents_dir = temp_data_dir / "agents"
    tau2_agents_dir.mkdir(exist_ok=True)
    tau2_agent_link = tau2_agents_dir / "tau2_agent"
    if not tau2_agent_link.exists():
        tau2_agent_link.symlink_to(PROJECT_ROOT / "tau2_agent")

    # Start custom server for tau2_agent with credentials middleware
    # Use ddtrace-run to enable Datadog tracing and LLM Observability
    env["AGENTS_DIR"] = str(tau2_agents_dir)
    env["PORT"] = str(tau2_agent_port)
    env["HOST"] = ADK_SERVER_HOST
    cmd = [
        "ddtrace-run",
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
                    f"Traced ADK server terminated unexpectedly.\n"
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
                f"Traced ADK server did not become ready "
                f"within {SERVER_STARTUP_TIMEOUT}s.\n"
                f"URL checked: {agent_card_url}\n"
                f"Last error: {last_error}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )

        yield TracedServer(
            process=process,
            data_dir=temp_data_dir,
            endpoint=tau2_agent_base_url,
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


@pytest.fixture
def evaluation_store(traced_adk_server, monkeypatch):
    """Provide access to EvaluationStore for verification."""
    # Set the data dir environment for store operations
    monkeypatch.setenv("TAU2_DATA_DIR", str(traced_adk_server.data_dir))

    from tau2.store import create_store

    return create_store()


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
                            f"Run an evaluation on the {domain} domain "
                            f"for agent at {agent_endpoint}. "
                            f"Use {num_tasks} tasks and {num_trials} trial(s)."
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
        model = "openai/Qwen/Qwen3-30B-A3B-Thinking-2507"

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
        timeout: Request timeout in seconds (default: 180.0)
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

        async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
            "POST", endpoint, json=request, headers=headers
        ) as response:
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
