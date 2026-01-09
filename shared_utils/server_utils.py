"""Shared server utilities for A2A agent servers."""

import json
from pathlib import Path

import aiofiles
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger


def add_root_agent_card(app: FastAPI, agent_name: str, agents_dir: str) -> None:
    """Add /.well-known/agent-card.json route for root-based A2A discovery.

    Standard A2A clients and AgentBeats expect agent cards at the root path,
    but ADK serves agent cards at /a2a/{agent_name}/.well-known/agent-card.json.
    This adds a root-level route that serves the same agent card content.

    Args:
        app: FastAPI application to add the route to.
        agent_name: Name of the agent (used to find agent.json).
        agents_dir: Directory containing agent subdirectories with agent.json files.
    """
    agent_json_path = Path(agents_dir) / agent_name / "agent.json"

    @app.get("/.well-known/agent-card.json")
    async def root_agent_card():
        """Serve agent card at root for A2A discovery."""
        if not agent_json_path.exists():
            logger.warning(f"Agent card not found at {agent_json_path}")
            return JSONResponse(
                content={"error": "agent.json not found"},
                status_code=404,
            )

        try:
            async with aiofiles.open(agent_json_path) as f:
                content = await f.read()
            data = json.loads(content)
            return JSONResponse(content=data)
        except OSError as e:
            logger.error(f"Failed to read agent card at {agent_json_path}: {e}")
            return JSONResponse(
                content={"error": "Failed to read agent.json"},
                status_code=500,
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent card JSON at {agent_json_path}: {e}")
            return JSONResponse(
                content={"error": "Invalid agent.json format"},
                status_code=500,
            )

    logger.info("Added root agent card route at /.well-known/agent-card.json")
