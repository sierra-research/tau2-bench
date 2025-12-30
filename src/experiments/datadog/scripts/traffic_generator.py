#!/usr/bin/env python3
"""Traffic generator for Datadog observability demo.

Generates evaluation telemetry by running concurrent A2A requests against tau2_agent.
Emits custom metrics via emit_metrics.py and enables LLM Observability in local mode.

Modes:
    GCP (default): Uses deployed Cloud Run agents.
    Local (--local): Starts local servers with ddtrace-run for LLM Observability.

Environment Variables:
    DD_API_KEY: Required for Datadog metric/trace submission.
    DD_SITE: Datadog site (default: us3.datadoghq.com).
    GEMINI_API_KEY: Required for --local mode with simple_gemini_agent.
    NEBIUS_API_KEY: Required for --local mode with simple_nebius_agent.

Usage:
    # GCP mode (default)
    uv run python -m experiments.datadog.scripts.traffic_generator --count 5

    # Local mode with LLM Observability
    uv run python -m experiments.datadog.scripts.traffic_generator --local --count 5

    # Failure mode to generate low-reward evaluations
    uv run python -m experiments.datadog.scripts.traffic_generator --mode failure --count 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).parent.parent.parent.parent.parent
load_dotenv(_project_root / ".env")

from tau2_agent.utils import SSEEvent, SSEParser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

# ============================================================================
# Configuration Constants
# ============================================================================

# GCP deployed endpoints - configurable via environment variables
_DEFAULT_TAU2_URL = "https://tau2-agent-676371821546.us-west2.run.app"
_DEFAULT_MOCK_URLS = [
    "https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app",
    "https://kimi-litellm-agent-4twyiz3sqq-wl.a.run.app",
]
GCP_TAU2_AGENT_URL = os.environ.get("TAU2_AGENT_URL", _DEFAULT_TAU2_URL)
GCP_MOCK_AGENT_URL = os.environ.get("MOCK_AGENT_URL", _DEFAULT_MOCK_URLS[0])
# Support multiple mock agents via comma-separated URLs
_mock_urls_env = os.environ.get("MOCK_AGENT_URLS", "")
GCP_MOCK_AGENT_URLS = [
    url.strip() for url in _mock_urls_env.split(",") if url.strip()
] if _mock_urls_env else _DEFAULT_MOCK_URLS

# A2A endpoint paths
TAU2_A2A_PATH = "/a2a/tau2_agent"

# Mock agent URL to A2A path mapping
# Maps service name (from URL) to agent module name
MOCK_AGENT_PATH_MAP = {
    "simple-gemini-agent": "/a2a/simple_gemini_agent",
    "kimi-litellm-agent": "/a2a/kimi_litellm_agent",
}


def get_mock_a2a_path(base_url: str) -> str:
    """Get the A2A path for a mock agent URL.

    Infers the agent name from the Cloud Run service URL pattern.
    Falls back to simple_gemini_agent if not recognized.

    Args:
        base_url: The base URL of the mock agent (e.g., https://simple-gemini-agent-xxx.run.app)

    Returns:
        str: The A2A path (e.g., /a2a/simple_gemini_agent)
    """
    for service_name, a2a_path in MOCK_AGENT_PATH_MAP.items():
        if service_name in base_url:
            return a2a_path
    # Fallback for unknown agents
    logger.warning(f"Unknown mock agent URL pattern: {base_url}, using simple_gemini_agent path")
    return "/a2a/simple_gemini_agent"

# Use unique ports to avoid conflicts with other services (local mode only)
# CRITICAL: tau2_agent and mock agent MUST run on SEPARATE ports to avoid
# async deadlock. See issue_tracker/concurrency-fix.md for details.
ADK_SERVER_HOST = "localhost"
TAU2_AGENT_PORT = int(os.environ.get("TRAFFIC_GEN_TAU2_PORT", "8766"))
MOCK_AGENT_PORT = int(os.environ.get("TRAFFIC_GEN_MOCK_PORT", "8767"))
LOCAL_TAU2_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{TAU2_AGENT_PORT}"
LOCAL_MOCK_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{MOCK_AGENT_PORT}"

# Legacy aliases (for backward compatibility in local functions)
TAU2_AGENT_BASE_URL = LOCAL_TAU2_AGENT_BASE_URL
MOCK_AGENT_BASE_URL = LOCAL_MOCK_AGENT_BASE_URL

# Server startup configuration
SERVER_STARTUP_TIMEOUT = 60  # Longer timeout for traced startup
SERVER_HEALTH_CHECK_INTERVAL = 0.5

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Default evaluation configuration
DEFAULT_NUM_TASKS = 2
DEFAULT_NUM_TRIALS = 1

# Domains available for evaluation
AVAILABLE_DOMAINS = ["mock", "airline", "retail", "telecom"]

# Available mock agents for evaluation
AVAILABLE_AGENTS = ["simple_gemini_agent", "simple_nebius_agent", "kimi_litellm_agent"]

# Task IDs for failure mode (tasks that may produce varied/low rewards)
# These are adversarial/complex tasks designed to test edge cases:
# - mock: Task requires unavailable tool (must transfer to human)
# - airline: Social engineering resistance tests (users make false claims)
# - retail: Complex multi-item exchanges with specific product requirements
# - telecom: [PERSONA:Hard] tasks with anxious user persona that gets flustered
FAILURE_MODE_TASK_IDS = {
    "mock": ["impossible_task_1"],  # Requires unavailable delete_task tool
    "airline": ["0", "1", "2"],  # Users claim false approvals/policies
    "retail": ["0", "1", "2"],  # Multi-item exchanges with product lookups
    "telecom": [
        "[mobile_data_issue]user_abroad_roaming_disabled_off[PERSONA:Hard]",
        "[service_issue]lock_sim_card_pin[PERSONA:Hard]",
        "[mms_issue]break_apn_mms_setting[PERSONA:Hard]",
    ],
}


# ============================================================================
# Server Management (copied from conftest.py)
# ============================================================================


@dataclass
class TracedServer:
    """Represents a running ADK server with tracing enabled."""

    process: subprocess.Popen | None
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

    process: subprocess.Popen | None
    endpoint: str
    agent_endpoint: str
    temp_dir: Path | None = None  # Temp directory to clean up on shutdown


@dataclass
class ServerManager:
    """Manages tau2_agent and mock agent server lifecycles."""

    tau2_server: TracedServer | None = None
    mock_server: MockAgentServer | None = None
    _cleanup_functions: list[Callable[[], None]] = field(default_factory=list)

    def add_cleanup(self, func):
        """Add a cleanup function to be called on shutdown."""
        self._cleanup_functions.append(func)

    def cleanup(self):
        """Stop all servers and run cleanup functions."""
        for func in self._cleanup_functions:
            try:
                func()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

        if self.mock_server:
            if self.mock_server.process:
                self._stop_process(self.mock_server.process)
            # Clean up temp directory if it exists
            if self.mock_server.temp_dir and self.mock_server.temp_dir.exists():
                import shutil
                try:
                    shutil.rmtree(self.mock_server.temp_dir)
                    logger.debug(f"Cleaned up temp dir: {self.mock_server.temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp dir: {e}")
            self.mock_server = None

        if self.tau2_server and self.tau2_server.process:
            self._stop_process(self.tau2_server.process)
            self.tau2_server = None

    def _stop_process(self, process: subprocess.Popen):
        """Stop a subprocess and its process group."""
        if process.poll() is None:
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


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def detect_mock_agent() -> str:
    """Detect which mock agent to use based on available API keys.

    Prefers Gemini (GEMINI_API_KEY) over Nebius (NEBIUS_API_KEY) since
    Gemini is typically faster and more widely available.

    Returns:
        str: Name of the agent to use ("simple_gemini_agent" or "simple_nebius_agent")

    Raises:
        ValueError: If no API keys are available
    """
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "simple_gemini_agent"
    if os.getenv("NEBIUS_API_KEY"):
        return "simple_nebius_agent"

    raise ValueError(
        "No mock agent API keys available. Set one of:\n"
        "  - GEMINI_API_KEY or GOOGLE_API_KEY (for simple_gemini_agent)\n"
        "  - NEBIUS_API_KEY (for simple_nebius_agent)"
    )


def start_mock_agent_server(
    agents_dir: Path | None = None,
    agent_name: str | None = None,
) -> MockAgentServer:
    """Start a separate ADK server for a mock agent on a different port.

    Supports simple_gemini_agent (Gemini Flash 2.0) or simple_nebius_agent
    (Nebius Qwen3). Auto-detects which agent to use based on available API keys.

    Args:
        agents_dir: Optional directory containing agents. If None, uses PROJECT_ROOT.
        agent_name: Optional agent name. If None, auto-detects based on API keys.

    Returns:
        MockAgentServer: Server info including process and endpoint
    """
    if agent_name is None:
        agent_name = detect_mock_agent()
        logger.info(f"Auto-detected mock agent: {agent_name}")
    mock_agent_endpoint = f"{MOCK_AGENT_BASE_URL}/a2a/{agent_name}"
    agent_card_url = f"{mock_agent_endpoint}/.well-known/agent-card.json"

    # Check if port is already in use - if so, assume server is running
    if is_port_in_use(MOCK_AGENT_PORT):
        logger.info(f"Port {MOCK_AGENT_PORT} already in use, checking if mock agent is running...")
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Mock agent already running at {mock_agent_endpoint}")
                return MockAgentServer(
                    process=None,
                    endpoint=MOCK_AGENT_BASE_URL,
                    agent_endpoint=mock_agent_endpoint,
                )
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        msg = (
            f"Port {MOCK_AGENT_PORT} is in use but mock agent not responding. "
            f"Set TRAFFIC_GEN_MOCK_PORT to use a different port."
        )
        raise RuntimeError(msg)

    # Copy agent to temp directory (ADK rejects symlinks outside base dir)
    temp_dir_path: Path | None = None
    if agents_dir is None:
        import shutil
        import tempfile
        temp_dir_path = Path(tempfile.mkdtemp(prefix="traffic_gen_mock_"))
        mock_agent_dest = temp_dir_path / agent_name
        # Exclude docker_setup and __pycache__ as they interfere with ADK agent discovery
        shutil.copytree(
            PROJECT_ROOT / agent_name,
            mock_agent_dest,
            ignore=shutil.ignore_patterns("docker_setup", "__pycache__"),
        )
        agents_dir = temp_dir_path

    env = os.environ.copy()

    # ADK's Gemini class expects GOOGLE_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        env["GOOGLE_API_KEY"] = gemini_key

    env["AGENTS_DIR"] = str(agents_dir)
    env["PORT"] = str(MOCK_AGENT_PORT)
    env["HOST"] = ADK_SERVER_HOST

    # Add agents directory to PYTHONPATH for module import
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{agents_dir}:{pythonpath}" if pythonpath else str(agents_dir)

    cmd = ["uv", "run", "python", "-m", f"{agent_name}.server"]

    logger.info(f"Starting mock agent server: {' '.join(cmd)}")
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
            msg = (
                f"Mock agent server terminated unexpectedly.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )
            raise RuntimeError(msg)

        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

    if not server_ready:
        if process.poll() is None:
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        msg = (
            f"Mock agent server did not become ready within {SERVER_STARTUP_TIMEOUT}s.\n"
            f"URL checked: {agent_card_url}\n"
            f"Last error: {last_error}\n"
            f"STDOUT: {stdout}\n"
            f"STDERR: {stderr}"
        )
        raise RuntimeError(msg)

    logger.info(f"Mock agent server started at {mock_agent_endpoint}")
    return MockAgentServer(
        process=process,
        endpoint=MOCK_AGENT_BASE_URL,
        agent_endpoint=mock_agent_endpoint,
        temp_dir=temp_dir_path,
    )


def start_tau2_server(data_dir: Path, mock_agent_endpoint: str) -> TracedServer:
    """Start ADK server with ddtrace enabled via environment variables.

    This function starts an isolated ADK server with:
    - DD_TRACE_ENABLED=true for Datadog tracing
    - TAU2_DATA_DIR pointing to the specified data directory
    - tau2_agent registered for evaluation requests

    Args:
        data_dir: Path to the data directory for EvaluationStore.
        mock_agent_endpoint: URL of the mock agent for evaluations.

    Returns:
        TracedServer: Server info including process, data_dir, and endpoints
    """
    tau2_agent_endpoint = f"{TAU2_AGENT_BASE_URL}/a2a/tau2_agent"
    agent_card_url = f"{tau2_agent_endpoint}/.well-known/agent-card.json"

    # Check if port is already in use - if so, assume server is running
    if is_port_in_use(TAU2_AGENT_PORT):
        logger.info(f"Port {TAU2_AGENT_PORT} already in use, checking if tau2_agent is running...")
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                logger.info(f"tau2_agent already running at {tau2_agent_endpoint}")
                return TracedServer(
                    process=None,
                    data_dir=data_dir,
                    endpoint=TAU2_AGENT_BASE_URL,
                    tau2_agent_endpoint=tau2_agent_endpoint,
                    mock_agent_endpoint=mock_agent_endpoint,
                )
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        msg = (
            f"Port {TAU2_AGENT_PORT} is in use but tau2_agent not responding. "
            f"Set TRAFFIC_GEN_TAU2_PORT to use a different port."
        )
        raise RuntimeError(msg)

    env = os.environ.copy()

    # Configure ddtrace for LLM Observability
    env["DD_TRACE_ENABLED"] = "true"
    env["DD_SERVICE"] = "tau2-bench-agent"
    env["DD_ENV"] = os.getenv("DD_ENV", "dev")

    # LLM Observability supports agentless mode (sends directly to Datadog intake).
    # APM tracing requires a local Datadog Agent and is not enabled here.
    dd_api_key = os.getenv("DD_API_KEY")
    dd_site = os.getenv("DD_SITE", "us3.datadoghq.com")
    if dd_api_key:
        env["DD_API_KEY"] = dd_api_key
        env["DD_SITE"] = dd_site
        env["DD_LLMOBS_ENABLED"] = "true"
        env["DD_LLMOBS_AGENTLESS_ENABLED"] = "true"
        env["DD_LLMOBS_ML_APP"] = os.getenv("DD_LLMOBS_ML_APP", "tau2-bench-agent")
        logger.info(f"LLM Observability enabled for {dd_site}")

    env["TAU2_DATA_DIR"] = str(data_dir)

    # ADK's Gemini class expects GOOGLE_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        env["GOOGLE_API_KEY"] = gemini_key

    # Copy tau2_agent to agents directory (ADK rejects symlinks outside base dir)
    import shutil
    tau2_agents_dir = data_dir / "agents"
    tau2_agents_dir.mkdir(exist_ok=True)
    tau2_agent_dest = tau2_agents_dir / "tau2_agent"
    if not tau2_agent_dest.exists():
        # Exclude docker_setup and __pycache__ as they interfere with ADK agent discovery
        shutil.copytree(
            PROJECT_ROOT / "tau2_agent",
            tau2_agent_dest,
            ignore=shutil.ignore_patterns("docker_setup", "__pycache__"),
        )

    env["AGENTS_DIR"] = str(tau2_agents_dir)
    env["PORT"] = str(TAU2_AGENT_PORT)
    env["HOST"] = ADK_SERVER_HOST

    # Use ddtrace-run for LLM Observability when DD_API_KEY is available
    base_cmd = ["uv", "run"]
    if dd_api_key:
        base_cmd.extend(["ddtrace-run"])
    cmd = base_cmd + ["python", "-m", "tau2_agent.server"]

    logger.info(f"Starting tau2_agent server: {' '.join(cmd)}")
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
            msg = (
                f"tau2_agent server terminated unexpectedly.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )
            raise RuntimeError(msg)

        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

    if not server_ready:
        if process.poll() is None:
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        msg = (
            f"tau2_agent server did not become ready within {SERVER_STARTUP_TIMEOUT}s.\n"
            f"URL checked: {agent_card_url}\n"
            f"Last error: {last_error}\n"
            f"STDOUT: {stdout}\n"
            f"STDERR: {stderr}"
        )
        raise RuntimeError(msg)

    logger.info(f"tau2_agent server started at {tau2_agent_endpoint}")
    return TracedServer(
        process=process,
        data_dir=data_dir,
        endpoint=TAU2_AGENT_BASE_URL,
        tau2_agent_endpoint=tau2_agent_endpoint,
        mock_agent_endpoint=mock_agent_endpoint,
    )


# ============================================================================
# A2A Request Helpers (copied from conftest.py)
# ============================================================================


def build_a2a_evaluation_request(
    domain: str,
    agent_endpoint: str,
    num_tasks: int = DEFAULT_NUM_TASKS,
    num_trials: int = DEFAULT_NUM_TRIALS,
    task_ids: list[str] | None = None,
    message_id: str | None = None,
    request_id: str | None = None,
) -> dict:
    """Build a JSON-RPC 2.0 A2A message requesting tau2 evaluation.

    Args:
        domain: The tau2 domain to evaluate (e.g., "mock", "airline")
        agent_endpoint: URL of the agent to evaluate
        num_tasks: Number of tasks to run (default: 2)
        num_trials: Number of trials per task (default: 1)
        task_ids: Optional list of specific task IDs to run
        message_id: Optional message ID (auto-generated if not provided)
        request_id: Optional JSON-RPC request ID (auto-generated if not provided)

    Returns:
        dict: JSON-RPC 2.0 formatted A2A request
    """
    # Build the natural language request
    task_spec = ""
    if task_ids:
        task_spec = f" Run tasks: {', '.join(task_ids)}."
    else:
        task_spec = f" Use {num_tasks} tasks and {num_trials} trial(s)."

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
                            f"{agent_endpoint}.{task_spec}"
                        )
                    }
                ],
            }
        },
        "id": request_id or str(uuid.uuid4()),
    }


def get_user_llm_credentials() -> dict[str, str]:
    """Get user LLM credentials for tau2_agent requests.

    Returns headers required by tau2_agent's CredentialsMiddleware.

    Returns:
        dict: Headers with X-User-LLM-Model and X-User-LLM-API-Key

    Raises:
        ValueError: If no LLM API key is configured
    """
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LLM API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env"
        )

    # Get model from environment or use default
    model = os.getenv("USER_LLM_MODEL", "gemini/gemini-3-flash-preview")

    return {
        "X-User-LLM-Model": model,
        "X-User-LLM-API-Key": api_key,
    }


async def send_a2a_evaluation_request(
    endpoint: str,
    domain: str,
    agent_endpoint: str,
    num_tasks: int = DEFAULT_NUM_TASKS,
    num_trials: int = DEFAULT_NUM_TRIALS,
    task_ids: list[str] | None = None,
    stream: bool = True,
    timeout: float = 300.0,  # 5 minutes timeout for concurrent evaluations
) -> AsyncIterator[dict]:
    """Send an A2A evaluation request and stream SSE events.

    Args:
        endpoint: The A2A endpoint URL (e.g., "http://localhost:8766/a2a/tau2_agent")
        domain: The tau2 domain to evaluate
        agent_endpoint: URL of the agent to evaluate
        num_tasks: Number of tasks to run
        num_trials: Number of trials per task
        task_ids: Optional list of specific task IDs to run
        stream: Whether to use SSE streaming (default: True)
        timeout: Request timeout in seconds (default: 300)

    Yields:
        dict: Parsed SSE event data containing evaluation progress/results
    """
    request = build_a2a_evaluation_request(
        domain=domain,
        agent_endpoint=agent_endpoint,
        num_tasks=num_tasks,
        num_trials=num_trials,
        task_ids=task_ids,
    )

    # Get user LLM credentials for tau2_agent
    llm_headers = get_user_llm_credentials()

    if stream:
        # Use message/stream for SSE streaming
        request["method"] = "message/stream"

        headers = {"Accept": "text/event-stream", **llm_headers}
        async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
            "POST",
            endpoint,
            json=request,
            headers=headers,
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
            response = await client.post(endpoint, json=request, headers=llm_headers)
            response.raise_for_status()
            yield response.json()


def sse_event_to_dict(event: SSEEvent) -> dict | None:
    """Convert SSEEvent to dict format expected by callers.

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


# ============================================================================
# Evaluation Runner
# ============================================================================


@dataclass
class EvaluationResult:
    """Result of a single evaluation request."""

    domain: str
    task_ids: list[str] | None
    success: bool
    events: list[dict]
    error: str | None = None
    final_state: str | None = None


async def run_single_evaluation(
    tau2_endpoint: str,
    mock_agent_endpoint: str,
    domain: str,
    num_tasks: int = DEFAULT_NUM_TASKS,
    num_trials: int = DEFAULT_NUM_TRIALS,
    task_ids: list[str] | None = None,
) -> EvaluationResult:
    """Run a single evaluation and collect all events.

    Args:
        tau2_endpoint: The tau2_agent A2A endpoint
        mock_agent_endpoint: The mock agent endpoint for evaluation
        domain: The domain to evaluate
        num_tasks: Number of tasks
        num_trials: Number of trials
        task_ids: Optional specific task IDs

    Returns:
        EvaluationResult: The result of the evaluation
    """
    events = []
    error = None
    final_state = None
    success = True

    try:
        async for event in send_a2a_evaluation_request(
            endpoint=tau2_endpoint,
            domain=domain,
            agent_endpoint=mock_agent_endpoint,
            num_tasks=num_tasks,
            num_trials=num_trials,
            task_ids=task_ids,
        ):
            events.append(event)

            # Check for error in function_response (tool returned error dict)
            # The error is nested in: result.status.message.parts[].data.response.error
            try:
                result = event.get("result", {})
                status = result.get("status", {})
                message = status.get("message", {})
                parts = message.get("parts", [])
                for part in parts:
                    data = part.get("data", {})
                    metadata = part.get("metadata", {})
                    # Check if this is a function_response with an error
                    if metadata.get("adk_type") == "function_response":
                        response = data.get("response", {})
                        if "error" in response:
                            success = False
                            error = f"{response.get('error')}: {response.get('message', 'Unknown error')}"
                            logger.error(f"Evaluation error from tau2_agent: {error}")
            except Exception:
                pass  # Ignore parsing errors, continue with other checks

            # Track final state from last message
            if "result" in event and "status" in event["result"]:
                final_state = event["result"]["status"].get("state")

            # Check for error state
            if final_state == "failed":
                success = False
                error_info = event.get("result", {}).get("status", {}).get("error")
                if error_info:
                    error = str(error_info)

    except Exception as e:
        success = False
        error = str(e)
        logger.error(f"Evaluation failed: {e}")

    return EvaluationResult(
        domain=domain,
        task_ids=task_ids,
        success=success,
        events=events,
        error=error,
        final_state=final_state,
    )


async def run_concurrent_evaluations(
    tau2_endpoint: str,
    mock_agent_endpoints: list[str],
    count: int,
    mode: str,
    domain: str | None = None,
    num_tasks: int = DEFAULT_NUM_TASKS,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> list[EvaluationResult]:
    """Run N concurrent evaluation requests.

    Args:
        tau2_endpoint: The tau2_agent A2A endpoint
        mock_agent_endpoints: List of mock agent endpoints for evaluation (cycles through)
        count: Number of evaluations to run
        mode: "normal" or "failure"
        domain: Optional domain (random if not specified)
        num_tasks: Number of tasks per evaluation
        num_trials: Number of trials per task

    Returns:
        list[EvaluationResult]: Results of all evaluations
    """
    tasks = []

    for i in range(count):
        # Select domain - random if not specified
        eval_domain = domain or random.choice(AVAILABLE_DOMAINS)

        # Cycle through mock agents for load distribution
        mock_agent_endpoint = mock_agent_endpoints[i % len(mock_agent_endpoints)]

        # In failure mode, use specific task IDs known to produce low rewards
        task_ids = None
        if mode == "failure":
            task_ids = FAILURE_MODE_TASK_IDS.get(eval_domain, ["1", "2"])

        logger.info(
            f"Queuing evaluation {i + 1}/{count}: domain={eval_domain}, "
            f"agent={mock_agent_endpoint}, mode={mode}, task_ids={task_ids}"
        )

        tasks.append(
            run_single_evaluation(
                tau2_endpoint=tau2_endpoint,
                mock_agent_endpoint=mock_agent_endpoint,
                domain=eval_domain,
                num_tasks=num_tasks,
                num_trials=num_trials,
                task_ids=task_ids,
            )
        )

    # Run all evaluations concurrently
    logger.info(f"Starting {count} concurrent evaluations...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to EvaluationResult
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(
                EvaluationResult(
                    domain=domain or "unknown",
                    task_ids=None,
                    success=False,
                    events=[],
                    error=str(result),
                )
            )
        else:
            processed_results.append(result)

    return processed_results


# ============================================================================
# Metrics Emission
# ============================================================================


def emit_metrics(dry_run: bool = False) -> int:
    """Run emit_metrics.py to send metrics to Datadog.

    Args:
        dry_run: If True, use --dry-run flag

    Returns:
        int: Exit code from emit_metrics.py
    """
    cmd = [
        sys.executable,
        "-m",
        "experiments.datadog.scripts.emit_metrics",
        "--all",
    ]

    if dry_run:
        cmd.append("--dry-run")

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"emit_metrics.py failed: {result.stderr}")
    else:
        logger.info(f"emit_metrics.py output: {result.stdout}")

    return result.returncode


# ============================================================================
# Main Entry Point
# ============================================================================


async def async_main(args: argparse.Namespace) -> int:
    """Async main function."""
    manager = ServerManager()

    try:
        # Set up data directory
        data_dir = Path(os.getenv("TAU2_DATA_DIR", "./data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "sessions").mkdir(exist_ok=True)
        (data_dir / "evaluations").mkdir(exist_ok=True)

        # Symlink tau2 domains if needed
        tau2_dir = data_dir / "tau2"
        source_tau2_dir = PROJECT_ROOT / "data" / "tau2"
        if source_tau2_dir.exists() and not tau2_dir.exists():
            tau2_dir.symlink_to(source_tau2_dir)

        logger.info(f"Using data directory: {data_dir}")

        # Determine mode: GCP (default) or local (--local flag)
        if args.local:
            # Local mode: Start local servers
            logger.info("Local mode: Starting local servers...")

            # Start mock agent first
            logger.info("Starting mock agent server...")
            manager.mock_server = start_mock_agent_server(agent_name=args.agent)
            mock_agent_endpoints = [manager.mock_server.agent_endpoint]

            # Start tau2_agent with tracing
            logger.info("Starting tau2_agent server with ddtrace...")
            manager.tau2_server = start_tau2_server(data_dir, mock_agent_endpoints[0])
            tau2_agent_endpoint = manager.tau2_server.tau2_agent_endpoint
        else:
            # GCP mode (default): Use deployed GCP agents
            logger.info("GCP mode: Using deployed agents")
            logger.info(f"  tau2_agent:  {args.tau2_url}")

            # Build list of mock agent endpoints
            # Priority: --mock-urls > --mock-url (if changed) > MOCK_AGENT_URLS env > defaults
            if args.mock_urls:
                mock_base_urls = args.mock_urls
            elif args.mock_url != _DEFAULT_MOCK_URLS[0]:
                # User explicitly set --mock-url to something different
                mock_base_urls = [args.mock_url]
            elif _mock_urls_env:
                # User set MOCK_AGENT_URLS env var
                mock_base_urls = GCP_MOCK_AGENT_URLS
            else:
                # Use all default mock agents
                mock_base_urls = _DEFAULT_MOCK_URLS

            # Build full A2A endpoints, inferring path from URL pattern
            mock_agent_endpoints = [
                f"{url.strip()}{get_mock_a2a_path(url)}" for url in mock_base_urls
            ]
            logger.info(f"  mock agents ({len(mock_agent_endpoints)}):")
            for endpoint in mock_agent_endpoints:
                logger.info(f"    - {endpoint}")

            # Construct A2A endpoints from base URLs
            tau2_agent_endpoint = f"{args.tau2_url}{TAU2_A2A_PATH}"

            # Verify GCP agents are reachable
            tau2_agent_card = f"{tau2_agent_endpoint}/.well-known/agent-card.json"

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    # Check tau2_agent
                    logger.info(f"Checking tau2_agent at {tau2_agent_card}...")
                    resp = await client.get(tau2_agent_card)
                    if resp.status_code != 200:
                        logger.warning(f"tau2_agent returned {resp.status_code}, may be cold starting")

                    # Check all mock agents
                    for endpoint in mock_agent_endpoints:
                        mock_agent_card = f"{endpoint}/.well-known/agent-card.json"
                        logger.info(f"Checking mock agent at {mock_agent_card}...")
                        resp = await client.get(mock_agent_card)
                        if resp.status_code != 200:
                            logger.warning(f"mock agent returned {resp.status_code}, may be cold starting")

                logger.info("GCP agents are reachable!")
            except httpx.ConnectError as e:
                logger.error(f"Failed to connect to GCP agents: {e}")
                logger.error("Try --local flag to run with local servers instead")
                return 1
            except httpx.TimeoutException:
                logger.warning("GCP agents timed out (may be cold starting), proceeding anyway...")

        # Run evaluations
        logger.info(
            f"Running {args.count} evaluations in {args.mode} mode "
            f"(domain={args.domain or 'random'})"
        )

        results = await run_concurrent_evaluations(
            tau2_endpoint=tau2_agent_endpoint,
            mock_agent_endpoints=mock_agent_endpoints,
            count=args.count,
            mode=args.mode,
            domain=args.domain,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
        )

        # Report results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(f"Evaluations complete: {successful} succeeded, {failed} failed")

        for i, result in enumerate(results):
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(
                f"  [{i + 1}] {status}: domain={result.domain}, "
                f"events={len(result.events)}, error={result.error}"
            )

        # Emit metrics - only in local mode
        # In GCP mode, metrics are emitted from the container immediately after each evaluation
        if args.local:
            logger.info("Emitting metrics to Datadog (local mode)...")
            emit_result = emit_metrics(dry_run=args.dry_run)

            if emit_result != 0:
                logger.warning(f"Metrics emission returned exit code {emit_result}")
        else:
            logger.info("Skipping local metric emission (GCP mode - metrics emitted from container)")

        return 0 if failed == 0 else 1

    finally:
        # Cleanup servers
        manager.cleanup()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Traffic generator for Datadog LLM observability demo"
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of evaluations to run (default: 5)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "failure"],
        default="normal",
        help="Traffic mode: normal (varied) or failure (trigger DR-002)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=AVAILABLE_DOMAINS,
        default=None,
        help="Domain to evaluate (default: random)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=DEFAULT_NUM_TASKS,
        help=f"Number of tasks per evaluation (default: {DEFAULT_NUM_TASKS})",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Number of trials per task (default: {DEFAULT_NUM_TRIALS})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run with local servers instead of GCP (requires GEMINI_API_KEY or equivalent)",
    )
    parser.add_argument(
        "--tau2-url",
        type=str,
        default=GCP_TAU2_AGENT_URL,
        help=f"tau2_agent base URL (env: TAU2_AGENT_URL, default: {_DEFAULT_TAU2_URL})",
    )
    parser.add_argument(
        "--mock-url",
        type=str,
        default=GCP_MOCK_AGENT_URL,
        help=f"Mock agent base URL (env: MOCK_AGENT_URL, default: {_DEFAULT_MOCK_URLS[0]})",
    )
    parser.add_argument(
        "--mock-urls",
        type=str,
        nargs="+",
        default=None,
        help="Multiple mock agent base URLs (space-separated). Overrides --mock-url.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=AVAILABLE_AGENTS,
        default=None,
        help="Mock agent to use for local mode (default: auto-detect from API keys)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: run evaluations but emit_metrics uses --dry-run",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Check for required environment variables
    if not args.dry_run and not os.getenv("DD_API_KEY"):
        logger.warning(
            "DD_API_KEY not set. Metrics will not be sent to Datadog. "
            "Use --dry-run for local testing."
        )

    # Local mode requires mock agent API keys
    if args.local:
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY") and not os.getenv("NEBIUS_API_KEY"):
            logger.error(
                "Local mode requires a mock agent API key. Set one of:\n"
                "  - GEMINI_API_KEY or GOOGLE_API_KEY (for simple_gemini_agent)\n"
                "  - NEBIUS_API_KEY (for simple_nebius_agent)"
            )
            return 1
    else:
        logger.info("GCP mode: Using deployed agents")

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
