"""
Custom server entrypoint for tau2_agent with credentials middleware.

This module provides a custom FastAPI server that wraps the ADK-generated
app with credentials middleware for extracting user LLM credentials from
HTTP headers.

Usage:
    # Run directly
    python -m tau2_agent.server

    # Or import the app for testing
    from tau2_agent.server import create_app
    app = create_app()
"""

import os
from pathlib import Path

from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger

from tau2_agent.logging_config import configure_logging


def create_app():
    """Create and configure the FastAPI application.

    Creates the ADK FastAPI app with A2A endpoints enabled and adds
    credentials middleware for extracting user LLM credentials from HTTP headers.

    Returns:
        FastAPI: Configured FastAPI application with A2A and credentials middleware.
    """
    # Default: use agents/ directory which contains only valid agents (via symlinks)
    # This ensures /list-apps only returns actual agents, not data/src/etc
    project_root = Path(__file__).resolve().parent.parent
    default_agents_dir = str(project_root / "agents")

    # Allow override via AGENTS_DIR env var (for test isolation and deployment flexibility)
    agents_dir = os.getenv("AGENTS_DIR", default_agents_dir)

    # Create ADK FastAPI app with A2A enabled
    # - web=False: disable ADK web UI, API only
    # - a2a=True: enable A2A JSON-RPC endpoints for direct tool invocation
    app = get_fast_api_app(agents_dir=agents_dir, web=False, a2a=True)

    # Import and add credentials middleware
    # Delayed import to avoid circular dependencies
    try:
        from tau2_agent.middleware import CredentialsMiddleware

        app.add_middleware(CredentialsMiddleware)
        logger.info("Credentials middleware registered")
    except ImportError:
        logger.warning("CredentialsMiddleware not found, running without credential extraction")

    return app


def main():
    """Run the tau2_agent server.

    Supports both environment variables and CLI arguments for configuration.
    CLI arguments take precedence over environment variables.

    CLI Args (AgentBeats compatible):
        --host: Host to bind the server (default: 0.0.0.0)
        --port: Port to bind the server (default: 8001)
        --card-url: External URL for agent card discovery
    """
    import argparse

    import uvicorn

    # Parse CLI arguments (AgentBeats compatibility)
    parser = argparse.ArgumentParser(description="Run the tau2_agent A2A server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8001")),
        help="Port to bind the server (default: 8001)",
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

    # Configure Datadog tracing (opt-in via DD_TRACE_ENABLED=true)
    try:
        from tau2.tracing import configure_ddtrace

        configure_ddtrace()
    except ImportError:
        logger.debug("tau2.tracing not available, skipping ddtrace configuration")

    logger.info(
        "Starting tau2_agent server",
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        log_level=log_level,
    )

    # Create app
    app = create_app()

    # Run with uvicorn (use lowercase for uvicorn)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
