#!/usr/bin/env python3
"""Traffic generator for Datadog LLM observability demo.

This script generates varied telemetry by running concurrent A2A evaluation requests
against tau2_agent. It starts the necessary servers (tau2_agent and mock agent),
fires N concurrent evaluations, and emits metrics to Datadog.

Architecture: Concurrent-by-default A2A requests to tau2_agent
- Fires N concurrent evaluation requests using asyncio.gather()
- tau2_agent handles up to 10 concurrent evaluations (via _EVALUATION_EXECUTOR)
- Generates more data points in less time for richer Datadog visualizations

Pattern Source: Inline copy from tests/test_datadog_e2e/conftest.py:
- TracedServer / MockAgentServer dataclasses
- build_a2a_evaluation_request(), send_a2a_evaluation_request(), parse_sse_event()
- Server startup with DD environment variables

Environment Variables:
    DD_TRACE_ENABLED: Enable ddtrace instrumentation (default: true)
    DD_SERVICE: Datadog service name (default: tau2-bench-agent)
    DD_API_KEY: Required for Datadog metric submission (optional for --dry-run)
    TAU2_DATA_DIR: Base data directory (default: ./data)
    NEBIUS_API_KEY: Required for mock agent LLM calls

Usage:
    # Generate 5 normal evaluations (dry-run, no Datadog needed)
    uv run python -m experiments.datadog.scripts.traffic_generator --count 5 --dry-run

    # Generate evaluations with metrics to Datadog
    uv run python -m experiments.datadog.scripts.traffic_generator --count 5

    # Generate failure-mode evaluations to trigger DR-002
    uv run python -m experiments.datadog.scripts.traffic_generator --count 3 --mode failure

    # Skip server startup (use existing servers)
    uv run python -m experiments.datadog.scripts.traffic_generator --count 5 --skip-server-start
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
from loguru import logger

from tau2_agent.utils import SSEEvent, SSEParser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

# ============================================================================
# Configuration Constants
# ============================================================================

# Use unique ports to avoid conflicts with other services
# CRITICAL: tau2_agent and mock agent MUST run on SEPARATE ports to avoid
# async deadlock. See issue_tracker/concurrency-fix.md for details.
ADK_SERVER_HOST = "localhost"
TAU2_AGENT_PORT = int(os.environ.get("TRAFFIC_GEN_TAU2_PORT", "8766"))
MOCK_AGENT_PORT = int(os.environ.get("TRAFFIC_GEN_MOCK_PORT", "8767"))
TAU2_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{TAU2_AGENT_PORT}"
MOCK_AGENT_BASE_URL = f"http://{ADK_SERVER_HOST}:{MOCK_AGENT_PORT}"

# Server startup configuration
SERVER_STARTUP_TIMEOUT = 60  # Longer timeout for traced startup
SERVER_HEALTH_CHECK_INTERVAL = 0.5

# Project root for finding agents
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Default evaluation configuration
DEFAULT_NUM_TASKS = 2
DEFAULT_NUM_TRIALS = 1

# Domains available for evaluation
AVAILABLE_DOMAINS = ["mock", "airline"]

# Task IDs known to produce low rewards (for failure mode)
# These are tasks that are intentionally difficult or have edge cases
FAILURE_MODE_TASK_IDS = {
    "mock": ["1", "2", "3"],  # Mock domain tasks
    "airline": ["1", "2", "3"],  # Airline domain tasks
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


def start_mock_agent_server(agents_dir: Path | None = None) -> MockAgentServer:
    """Start a separate ADK server for simple_nebius_agent on a different port.

    Uses simple_nebius_agent (real LLM via Nebius API) which properly integrates
    with tau2's A2A protocol for tool calls.

    Args:
        agents_dir: Optional directory containing agents. If None, uses PROJECT_ROOT.

    Returns:
        MockAgentServer: Server info including process and endpoint
    """
    agent_name = "simple_nebius_agent"
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

    # Create a temp directory with simple_nebius_agent symlinked
    # ADK expects AGENTS_DIR to contain subdirectories, each being an agent
    # Note: temp_dir is stored in MockAgentServer.temp_dir for cleanup by ServerManager
    temp_dir_path: Path | None = None
    if agents_dir is None:
        import tempfile
        temp_dir_path = Path(tempfile.mkdtemp(prefix="traffic_gen_mock_"))
        mock_agent_link = temp_dir_path / agent_name
        mock_agent_link.symlink_to(PROJECT_ROOT / agent_name)
        agents_dir = temp_dir_path

    # Build environment - no ddtrace needed for mock agent
    env = os.environ.copy()

    # Start ADK server for mock agent only
    cmd = [
        "adk",
        "api_server",
        "--a2a",
        str(agents_dir),
        "--port",
        str(MOCK_AGENT_PORT),
        "--host",
        ADK_SERVER_HOST,
    ]

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

    # Build environment with ddtrace configuration
    env = os.environ.copy()

    # Enable ddtrace via environment (auto-patches on import)
    env["DD_TRACE_ENABLED"] = "true"
    env["DD_SERVICE"] = "tau2-bench-agent"
    env["DD_ENV"] = os.getenv("DD_ENV", "demo")

    # Optional: Enable LLM Observability if DD_API_KEY is set
    if os.getenv("DD_API_KEY"):
        env["DD_LLMOBS_ENABLED"] = "true"
        env["DD_LLMOBS_AGENTLESS_ENABLED"] = "true"

    # Use the specified data directory
    env["TAU2_DATA_DIR"] = str(data_dir)

    # Create a temp directory with only tau2_agent symlinked
    # ADK expects AGENTS_DIR to contain subdirectories, each being an agent
    tau2_agents_dir = data_dir / "agents"
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

    if stream:
        # Use message/stream for SSE streaming
        request["method"] = "message/stream"

        async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
            "POST",
            endpoint,
            json=request,
            headers={"Accept": "text/event-stream"},
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
            response = await client.post(endpoint, json=request)
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
    mock_agent_endpoint: str,
    count: int,
    mode: str,
    domain: str | None = None,
    num_tasks: int = DEFAULT_NUM_TASKS,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> list[EvaluationResult]:
    """Run N concurrent evaluation requests.

    Args:
        tau2_endpoint: The tau2_agent A2A endpoint
        mock_agent_endpoint: The mock agent endpoint for evaluation
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

        # In failure mode, use specific task IDs known to produce low rewards
        task_ids = None
        if mode == "failure":
            task_ids = FAILURE_MODE_TASK_IDS.get(eval_domain, ["1", "2"])

        logger.info(
            f"Queuing evaluation {i + 1}/{count}: domain={eval_domain}, "
            f"mode={mode}, task_ids={task_ids}"
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

        # Start servers if needed
        if args.skip_server_start:
            logger.info("Skipping server startup (--skip-server-start)")
            mock_agent_endpoint = f"{MOCK_AGENT_BASE_URL}/a2a/simple_nebius_agent"
            tau2_agent_endpoint = f"{TAU2_AGENT_BASE_URL}/a2a/tau2_agent"
        else:
            # Start mock agent first
            logger.info("Starting mock agent server...")
            manager.mock_server = start_mock_agent_server()
            mock_agent_endpoint = manager.mock_server.agent_endpoint

            # Start tau2_agent with tracing
            logger.info("Starting tau2_agent server with ddtrace...")
            manager.tau2_server = start_tau2_server(data_dir, mock_agent_endpoint)
            tau2_agent_endpoint = manager.tau2_server.tau2_agent_endpoint

        # Run evaluations
        logger.info(
            f"Running {args.count} evaluations in {args.mode} mode "
            f"(domain={args.domain or 'random'})"
        )

        results = await run_concurrent_evaluations(
            tau2_endpoint=tau2_agent_endpoint,
            mock_agent_endpoint=mock_agent_endpoint,
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

        # Emit metrics
        logger.info("Emitting metrics to Datadog...")
        emit_result = emit_metrics(dry_run=args.dry_run)

        if emit_result != 0:
            logger.warning(f"Metrics emission returned exit code {emit_result}")

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
        "--skip-server-start",
        action="store_true",
        help="Skip starting servers (use existing servers)",
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

    if not os.getenv("NEBIUS_API_KEY"):
        logger.warning(
            "NEBIUS_API_KEY not set. Mock agent LLM calls may fail."
        )

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
