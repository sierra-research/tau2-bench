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

from shared_utils.card_url_utils import update_agent_card_url
from shared_utils.server_utils import add_root_agent_card
from tau2_agent.logging_config import configure_logging

AGENT_NAME = "tau2_agent"


def create_app():
    """Create and configure the FastAPI application.

    Creates the ADK FastAPI app with A2A endpoints enabled, adds root-based
    discovery routes, and wraps with credentials and path-rewriting middleware.

    Returns:
        ASGIApp: Configured application with A2A, credentials, and root discovery.
    """
    project_root = Path(__file__).resolve().parent.parent
    agents_dir = os.getenv("AGENTS_DIR", str(project_root / "agents"))

    app = get_fast_api_app(agents_dir=agents_dir, web=False, a2a=True)

    # Add root agent card route for A2A discovery
    add_root_agent_card(app, AGENT_NAME, agents_dir)

    try:
        from tau2_agent.middleware import CredentialsMiddleware, RootA2AMiddleware

        app.add_middleware(CredentialsMiddleware)
        logger.info("Credentials middleware registered")

        # Wrap with RootA2AMiddleware as outermost layer
        # This rewrites POST / to POST /a2a/{agent_name}/ for root-based A2A
        return RootA2AMiddleware(app, agent_name=AGENT_NAME)
    except ImportError:
        logger.warning(
            "Middleware not found, running without credential extraction or root A2A"
        )
        return app


def main():
    """Run the tau2_agent server.

    Supports both environment variables and CLI arguments for configuration.
    CLI arguments take precedence over environment variables.

    CLI Args:
        --host: Host to bind the server (default: 0.0.0.0)
        --port: Port to bind the server (default: 8001)
        --card-url: External URL for agent card discovery
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the tau2_agent A2A server")
    parser.add_argument(
        "--host",
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
        default=os.getenv("CARD_URL"),
        help="External URL for agent card discovery",
    )
    args = parser.parse_args()

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    configure_logging(level=log_level)

    try:
        from tau2.tracing import configure_ddtrace

        configure_ddtrace()
    except ImportError:
        logger.debug("tau2.tracing not available, skipping ddtrace configuration")

    card_url = update_agent_card_url(AGENT_NAME, args.card_url, args.port)

    logger.info(
        "Starting tau2_agent server",
        host=args.host,
        port=args.port,
        card_url=card_url,
        log_level=log_level,
    )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
