"""Shared middleware for A2A agent servers."""

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send


class RootA2AMiddleware:
    """ASGI middleware that rewrites POST / to POST /a2a/{agent_name}/.

    This enables root-based A2A discovery, allowing agents to be accessed at
    the root path for compatibility with AgentBeats and standard A2A clients
    that expect endpoints at "/".

    Security:
        - Only rewrites exact "/" path (not "//" or "/foo")
        - Only rewrites POST method (GET / passes through for health checks)
        - agent_name validated at init (alphanumeric + underscore only)
        - Target path is hardcoded (no user-controllable components)

    This middleware must be the outermost middleware to ensure path rewriting
    happens before other middleware processes the request.
    """

    def __init__(self, app: ASGIApp, agent_name: str) -> None:
        """Initialize RootA2AMiddleware.

        Args:
            app: The ASGI application to wrap.
            agent_name: The agent name used to construct the target path.
                Must be alphanumeric with underscores only.

        Raises:
            ValueError: If agent_name contains invalid characters.
        """
        if not agent_name or not agent_name.replace("_", "").isalnum():
            msg = f"agent_name must be alphanumeric with underscores, got: {agent_name!r}"
            raise ValueError(msg)
        self.app = app
        self.agent_name = agent_name
        # ADK routes without trailing slash; trailing slash causes 307 redirect
        self._target_path = f"/a2a/{agent_name}"
        logger.info(
            "RootA2AMiddleware configured",
            source_path="/",
            target_path=self._target_path,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI request, rewriting path if conditions are met.

        Args:
            scope: ASGI connection scope dictionary.
            receive: ASGI receive callable for request body.
            send: ASGI send callable for response.
        """
        if (
            scope["type"] == "http"
            and scope.get("path") == "/"
            and scope.get("method") == "POST"
        ):
            scope = dict(scope)
            scope["path"] = self._target_path

        await self.app(scope, receive, send)
