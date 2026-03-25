"""Fixtures for A2A end-to-end tests.

Provides:
- --a2a-endpoint CLI option for targeting external servers
- a2a_e2e_endpoint: session-scoped fixture that starts a local A2A server
  or yields an external endpoint URL
- a2a_e2e_agent: function-scoped fixture that creates a fresh A2AAgent
"""

import os
import socket
import threading
import time

import pytest
import uvicorn

from tau2.a2a.models import A2AConfig
from tau2.agent.a2a_agent import A2AAgent
from tau2.domains.mock.environment import get_environment
from tests.test_a2a_e2e.harness import build_test_server


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--a2a-endpoint",
        action="store",
        default=None,
        help="URL of an external A2A endpoint to test against (skips local server startup)",
    )


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def a2a_e2e_endpoint(request: pytest.FixtureRequest):
    """Yield the base URL of a running A2A server.

    If --a2a-endpoint was passed, yields that URL directly.
    Otherwise, starts a local uvicorn server on a random port.
    Skips if OPENAI_API_KEY is not set (local server needs it).
    """
    external_url = request.config.getoption("--a2a-endpoint")
    if external_url is not None:
        yield external_url
        return

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (required for local A2A E2E server)")

    port = _find_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    app = build_test_server(url=base_url)

    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("A2A E2E server failed to start within 10 seconds")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def a2a_e2e_agent(a2a_e2e_endpoint: str):
    """Create a fresh A2AAgent per test, configured against the E2E endpoint.

    Uses mock domain tools and policy. Fresh agent state per test.
    """
    env = get_environment()
    tools = env.get_tools()
    policy = env.policy

    config = A2AConfig(
        endpoint=a2a_e2e_endpoint,
        timeout=120,
        connect_timeout=10,
        verify_ssl=False,
    )

    agent = A2AAgent(config=config, tools=tools, domain_policy=policy)
    yield agent
    agent.stop()
