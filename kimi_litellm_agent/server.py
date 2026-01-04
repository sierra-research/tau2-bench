"""
Simple server entrypoint for kimi_litellm_agent.

Usage:
    python -m kimi_litellm_agent.server
"""

import os
from pathlib import Path

from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger

from kimi_litellm_agent.logging_config import configure_logging


def create_app():
    """Create and configure the FastAPI application."""
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
