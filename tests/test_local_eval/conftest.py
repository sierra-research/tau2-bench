"""
Pytest fixtures for local A2A agent testing.

These fixtures manage the lifecycle of the Nebius agent server
for automated testing.
"""

import os
import subprocess
import time
from typing import Generator

import httpx
import pytest


@pytest.fixture
def nebius_api_configured():
    """Skip test if NEBIUS_API_KEY is not configured."""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        pytest.skip("NEBIUS_API_KEY not set - skipping test requiring Nebius API")
    return api_key


@pytest.fixture
def anthropic_user_llm_config():
    """
    Provide Anthropic LLM configuration for user simulation.

    Skip test if ANTHROPIC_API_KEY is not configured.

    Returns:
        dict with 'llm' (model string) and 'llm_args' (additional args)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set - skipping test requiring Anthropic API")

    return {
        "llm": "anthropic/claude-3-haiku-20240307",
        "llm_args": {},
    }


@pytest.fixture
def simple_agent_port() -> int:
    """Port for simple agent server."""
    return 8001


@pytest.fixture
def simple_agent_endpoint(simple_agent_port: int) -> str:
    """Endpoint URL for simple agent (A2A agent path)."""
    return f"http://localhost:{simple_agent_port}/a2a/simple_nebius_agent"


@pytest.fixture
def check_port_available(simple_agent_port: int):
    """Check if the port is available, fail if not."""
    result = subprocess.run(
        ["lsof", "-Pi", f":{simple_agent_port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        pytest.fail(
            f"Port {simple_agent_port} is already in use. "
            f"Kill the process: kill {result.stdout.strip()}"
        )


@pytest.fixture
def wait_for_agent(simple_agent_endpoint: str):
    """Helper function to wait for agent to be ready."""
    def _wait(timeout: int = 30) -> bool:
        """
        Wait for agent to respond to health check.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if agent is ready, False otherwise
        """
        start_time = time.time()
        agent_card_url = f"{simple_agent_endpoint}/.well-known/agent-card.json"

        while time.time() - start_time < timeout:
            try:
                response = httpx.get(agent_card_url, timeout=2.0)
                if response.status_code == 200:
                    return True
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            time.sleep(1)

        return False

    return _wait


@pytest.fixture
def simple_agent_server(
    simple_agent_port: int,
    simple_agent_endpoint: str,
    nebius_api_configured: str,
    check_port_available,
    wait_for_agent,
) -> Generator[str, None, None]:
    """
    Start and manage the simple Nebius agent server for testing.

    Yields:
        Agent endpoint URL

    Raises:
        RuntimeError: If agent fails to start
    """
    # Start the agent server
    # Note: Pass current directory (parent of agent dirs), not agent dir itself
    process = subprocess.Popen(
        [
            "adk",
            "api_server",
            "--a2a",
            ".",
            "--port",
            str(simple_agent_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    try:
        # Wait for agent to be ready
        if not wait_for_agent(timeout=30):
            # Get any error output
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Agent failed to start within 30 seconds.\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}"
            )

        # Yield control to test
        yield simple_agent_endpoint

    finally:
        # Cleanup: Stop the agent server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def agent_card_url(simple_agent_endpoint: str) -> str:
    """Agent card discovery URL."""
    return f"{simple_agent_endpoint}/.well-known/agent-card.json"


@pytest.fixture
def jsonrpc_endpoint(simple_agent_endpoint: str) -> str:
    """A2A JSON-RPC endpoint URL (root of agent endpoint)."""
    return simple_agent_endpoint
