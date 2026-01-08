"""
Simple server entrypoint for kimi_litellm_agent.

Usage:
    python -m kimi_litellm_agent.server
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger

from kimi_litellm_agent.card_url_utils import update_agent_card_url
from kimi_litellm_agent.logging_config import configure_logging

AGENT_NAME = "kimi_litellm_agent"


def _add_root_agent_card(app: FastAPI, agents_dir: str) -> None:
    """Add /.well-known/agent-card.json route for AgentBeats compatibility.

    AgentBeats health checks probe the root path, but ADK serves agent cards at
    /a2a/{agent_name}/.well-known/agent-card.json. This adds a root-level route
    that serves the same agent card content, enabling AgentBeats health checks
    to pass while keeping the standard ADK routes intact.

    Args:
        app: FastAPI application to add the route to.
        agents_dir: Directory containing agent subdirectories with agent.json files.
    """
    agent_json_path = Path(agents_dir) / AGENT_NAME / "agent.json"

    @app.get("/.well-known/agent-card.json")
    async def root_agent_card():
        """Serve agent card at root for AgentBeats health checks."""
        if agent_json_path.exists():
            data = json.loads(agent_json_path.read_text())
            return JSONResponse(content=data)
        logger.warning(f"Agent card not found at {agent_json_path}")
        return JSONResponse(
            content={"error": "agent.json not found"},
            status_code=404,
        )

    logger.info("Added root agent card route for AgentBeats compatibility")


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

    # Add root agent card for AgentBeats health check compatibility
    # AgentBeats probes /.well-known/agent-card.json at root, but ADK serves
    # at /a2a/{agent_name}/.well-known/agent-card.json
    if os.getenv("AGENTBEATS_MODE", "").lower() == "true":
        _add_root_agent_card(app, agents_dir)

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
    parser = argparse.ArgumentParser(
        description="Run the kimi_litellm_agent A2A server"
    )
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

    # Update agent.json URL (auto-derive if not provided, for containerized deployments)
    effective_card_url = update_agent_card_url(AGENT_NAME, args.card_url, args.port)

    logger.info(
        "Starting kimi_litellm_agent server",
        host=args.host,
        port=args.port,
        card_url=effective_card_url,
        card_url_auto_derived=args.card_url is None,
        log_level=log_level,
    )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
