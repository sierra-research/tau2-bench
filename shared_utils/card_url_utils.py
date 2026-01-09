"""Shared utilities for agent card URL management."""

import json
import os
from pathlib import Path

from loguru import logger


def update_agent_card_url(
    agent_name: str, card_url: str | None, port: int
) -> str:
    """Update agent.json URL field for A2A discovery.

    The URL should point to the root of the agent server (e.g., http://example.com),
    not to the ADK-specific path (/a2a/{agent_name}/). Root-based discovery is
    handled by RootA2AMiddleware and the root agent card endpoint.

    In production, set CARD_URL via environment variable or --card-url CLI arg.
    For local development, defaults to localhost root (http://localhost:{port}).

    Args:
        agent_name: Name of the agent.
        card_url: Explicit URL from CARD_URL env var or --card-url arg.
            If None, defaults to http://localhost:{port}.
        port: Server port (used for localhost default).

    Returns:
        The URL written to agent.json.
    """
    if not card_url:
        card_url = f"http://localhost:{port}"

    agents_dir = Path(os.getenv("AGENTS_DIR", "/app/agents"))
    agent_json = agents_dir / agent_name / "agent.json"

    if not agent_json.exists():
        logger.warning(f"agent.json not found at {agent_json}")
        return card_url

    try:
        data = json.loads(agent_json.read_text())
        data["url"] = card_url
        agent_json.write_text(json.dumps(data, indent=2))
        logger.info(f"Updated agent.json URL to: {card_url}")
    except OSError as e:
        logger.error(f"Failed to read/write agent.json at {agent_json}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse agent.json at {agent_json}: {e}")

    return card_url
