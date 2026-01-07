"""
Simple server entrypoint for kimi_litellm_agent.

Usage:
    python -m kimi_litellm_agent.server
"""

import json
import os
from pathlib import Path

from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger

from kimi_litellm_agent.logging_config import configure_logging

AGENT_NAME = "kimi_litellm_agent"


def _update_agent_card_url(card_url: str | None) -> None:
    """Update agent.json URL if CARD_URL is provided.

    This enables containerized deployments where the agent's external URL
    differs from localhost (e.g., Docker networking, Kubernetes services).

    Args:
        card_url: The external URL for agent card discovery. If None, no update.
    """
    if not card_url:
        return

    agents_dir = Path(os.getenv("AGENTS_DIR", "/app/agents"))
    agent_json = agents_dir / AGENT_NAME / "agent.json"

    if not agent_json.exists():
        logger.warning(f"agent.json not found at {agent_json}")
        return

    data = json.loads(agent_json.read_text())
    data["url"] = card_url
    agent_json.write_text(json.dumps(data, indent=2))
    logger.info(f"Updated agent.json URL to: {card_url}")


def create_app():
    """Create and configure the FastAPI application.

    Creates the ADK FastAPI app with A2A endpoints enabled for the
    kimi_litellm_agent.

    Returns:
        FastAPI: Configured FastAPI application with A2A endpoints.
    """
    # Use agents/ directory which contains symlink to kimi_litellm_agent
    project_root = Path(__file__).resolve().parent.parent
    agents_dir = os.getenv("AGENTS_DIR", str(project_root / "agents"))

    # Create ADK FastAPI app with A2A enabled
    app = get_fast_api_app(agents_dir=agents_dir, web=False, a2a=True)
    return app


def main():
    """Run the kimi_litellm_agent server.

    Supports both environment variables and CLI arguments for configuration.
    CLI arguments take precedence over environment variables.

    CLI Args (AgentBeats compatible):
        --host: Host to bind the server (default: 0.0.0.0)
        --port: Port to bind the server (default: 8002)
        --card-url: External URL for agent card discovery
    """
    import argparse

    import uvicorn

    # Parse CLI arguments (AgentBeats compatibility)
    parser = argparse.ArgumentParser(description="Run the kimi_litellm_agent A2A server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8002")),
        help="Port to bind the server (default: 8002)",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=os.getenv("CARD_URL"),
        help="External URL for agent card discovery",
    )
    args = parser.parse_args()

    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure structured logging (JSON for GCP, human-readable locally)
    configure_logging(level=log_level)

    # Update agent.json URL if provided (for containerized deployments)
    _update_agent_card_url(args.card_url)

    logger.info(
        "Starting kimi_litellm_agent server",
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        log_level=log_level,
    )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
