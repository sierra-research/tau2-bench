"""
Custom server entrypoint for tau2_agent with credentials middleware.

This module provides a custom FastAPI server that wraps the ADK-generated
app with credentials middleware for extracting user LLM credentials from
HTTP headers.

The server exposes two A2A endpoints:
- POST /a2a/tau2_agent: LlmAgent for natural language requests (local testing)
- POST / or /a2a/tau2_green: GreenExecutor for AgentBeats (returns DataPart)

Usage:
    # Run directly
    python -m tau2_agent.server

    # Or import the app for testing
    from tau2_agent.server import create_app
    app = create_app()
"""

import os
from pathlib import Path

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from google.adk.cli.fast_api import get_fast_api_app
from loguru import logger

from shared_utils.card_url_utils import update_agent_card_url
from tau2_agent.green_executor import Tau2GreenExecutor, create_green_agent_card
from tau2_agent.logging_config import configure_logging

AGENT_NAME = "tau2_agent"


def create_app():
    """Create and configure the FastAPI application.

    Creates the ADK FastAPI app with A2A endpoints enabled, mounts the
    GreenExecutor for AgentBeats compatibility, and adds credentials middleware.

    Routes:
        - POST /a2a/tau2_agent: LlmAgent (natural language, local testing)
        - POST /a2a/tau2_green: GreenExecutor (structured DataPart results)
        - POST /: GreenExecutor (root-based A2A for AgentBeats)
        - GET /.well-known/agent-card.json: GreenExecutor agent card

    Returns:
        FastAPI: Configured application with dual A2A routes.
    """
    project_root = Path(__file__).resolve().parent.parent
    agents_dir = os.getenv("AGENTS_DIR", str(project_root / "agents"))
    card_url = os.getenv("CARD_URL", "http://localhost:8001")

    # Create ADK app with LlmAgent at /a2a/tau2_agent
    app = get_fast_api_app(agents_dir=agents_dir, web=False, a2a=True)

    # Create GreenExecutor for AgentBeats (returns DataPart artifacts)
    green_executor = Tau2GreenExecutor()
    green_handler = DefaultRequestHandler(
        agent_executor=green_executor,
        task_store=InMemoryTaskStore(),
    )
    green_app = A2AStarletteApplication(
        agent_card=create_green_agent_card(card_url),
        http_handler=green_handler,
    )
    green_starlette = green_app.build()

    # Mount green executor at explicit path
    app.mount("/a2a/tau2_green", green_starlette)
    logger.info("GreenExecutor mounted at /a2a/tau2_green")

    # Root-based agent card for AgentBeats discovery
    @app.get("/.well-known/agent-card.json")
    async def root_agent_card():
        """Return green executor agent card for root-based A2A discovery."""
        return create_green_agent_card(card_url).model_dump()

    # Mount green executor at root (AFTER explicit routes to avoid conflicts)
    app.mount("/", green_starlette)
    logger.info("GreenExecutor mounted at root for AgentBeats")

    # Add credentials middleware (needed for /a2a/tau2_agent LlmAgent path)
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
