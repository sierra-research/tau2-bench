"""
Shared utilities for agent card URL management.

Provides functions for deriving and updating agent card URLs in containerized
deployments where the external URL may differ from localhost.
"""

import json
import os
import socket
from pathlib import Path

from loguru import logger


def derive_card_url(agent_name: str, port: int) -> str:
    """Derive CARD_URL from container hostname and port.

    For ADK agents, the URL pattern is:
        http://{hostname}:{port}/a2a/{agent_name}

    This allows automatic URL discovery in Docker/Kubernetes environments
    where the container hostname matches the service name.

    Args:
        agent_name: Name of the agent.
        port: The server port number.

    Returns:
        Derived CARD_URL string.
    """
    hostname = socket.gethostname()
    return f"http://{hostname}:{port}/a2a/{agent_name}"


def update_agent_card_url(
    agent_name: str, card_url: str | None, port: int
) -> str | None:
    """Update agent.json URL, auto-deriving if not explicitly provided.

    This enables containerized deployments where the agent's external URL
    differs from localhost. When CARD_URL is not set, we auto-derive it from
    the container hostname and port.

    Priority:
    1. Explicit CARD_URL (escape hatch for custom configurations)
    2. Auto-derived from hostname + port + agent_name

    Args:
        agent_name: Name of the agent.
        card_url: Explicit URL override, or None to auto-derive.
        port: Server port for auto-derivation.

    Returns:
        The effective CARD_URL (explicit or derived), or None if agent.json not found.
    """
    effective_url = card_url or derive_card_url(agent_name, port)

    agents_dir = Path(os.getenv("AGENTS_DIR", "/app/agents"))
    agent_json = agents_dir / agent_name / "agent.json"

    if not agent_json.exists():
        logger.warning(f"agent.json not found at {agent_json}")
        return effective_url

    data = json.loads(agent_json.read_text())
    data["url"] = effective_url
    agent_json.write_text(json.dumps(data, indent=2))
    logger.info(f"Updated agent.json URL to: {effective_url}")

    return effective_url
