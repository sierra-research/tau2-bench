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
    """Run the kimi_litellm_agent server."""
    import uvicorn

    port = int(os.getenv("PORT", "8002"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure structured logging (JSON for GCP, human-readable locally)
    configure_logging(level=log_level)

    logger.info(
        "Starting kimi_litellm_agent server",
        host=host,
        port=port,
        log_level=log_level,
    )

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
