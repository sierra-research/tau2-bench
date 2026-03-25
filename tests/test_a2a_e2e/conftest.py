"""Fixtures for A2A end-to-end tests.

Two modes of operation:

  Local (default):
    OPENAI_API_KEY=sk-... pytest -m full_a2a_integration

    Spins up a local A2A server backed by gpt-4o on a random port.
    Requires OPENAI_API_KEY; skips otherwise.

  External endpoint:
    pytest -m full_a2a_integration --a2a-endpoint https://my-agent.example.com

    Runs the same test suite against an external A2A-compliant agent.
    The endpoint must serve /.well-known/agent-card.json and accept
    message/send JSON-RPC requests.
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


def _make_a2a_config(endpoint: str) -> A2AConfig:
    """Build A2AConfig for E2E tests."""
    return A2AConfig(
        endpoint=endpoint,
        timeout=120,
        connect_timeout=10,
        verify_ssl=False,
    )


@pytest.fixture(scope="session")
def a2a_e2e_endpoint(request: pytest.FixtureRequest):
    """Start a local A2A server or yield an external endpoint URL."""
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

    # Poll until server accepts connections
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


@pytest.fixture
def a2a_e2e_agent(a2a_e2e_endpoint: str):
    """Fixture providing an A2AAgent with mock domain tools."""
    env = get_environment()
    config = _make_a2a_config(a2a_e2e_endpoint)
    agent = A2AAgent(config=config, tools=env.get_tools(), domain_policy=env.policy)
    yield agent
    agent.stop()
