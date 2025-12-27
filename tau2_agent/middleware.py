"""
User LLM credentials middleware for tau2_agent.

This module provides HTTP middleware that extracts user LLM credentials
from request headers and stores them in request-scoped context variables.
Credentials are optional at the middleware level; individual tools validate
if they require credentials (e.g., run_tau2_evaluation validates credentials
because it needs them to call the user's LLM).

Security Note:
    API keys are NEVER logged. The middleware uses loguru for structured
    logging but explicitly excludes API key values from all log output.

Usage:
    from tau2_agent.middleware import CredentialsMiddleware

    app = FastAPI()
    app.add_middleware(CredentialsMiddleware)

    # With optional service authentication:
    app.add_middleware(CredentialsMiddleware, service_api_keys=["key1", "key2"])
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from loguru import logger

from tau2_agent.context import user_llm_model, user_llm_api_key, request_id
from tau2_agent.errors import ErrorCode, EvaluationError


# Paths that bypass credential and service auth validation (health checks, root, etc.)
BYPASS_PATHS = frozenset({"/", "/health", "/healthz", "/ready", "/readiness"})

# Path suffixes that bypass validation (e.g., agent discovery endpoints)
BYPASS_SUFFIXES = (".well-known/agent-card.json",)

# Header names (lowercase for case-insensitive matching)
HEADER_MODEL = "x-user-llm-model"
HEADER_API_KEY = "x-user-llm-api-key"
HEADER_REQUEST_ID = "x-request-id"
HEADER_AUTHORIZATION = "authorization"


class CredentialsMiddleware(BaseHTTPMiddleware):
    """Middleware for extracting user LLM credentials from HTTP headers.

    This middleware:
    1. Validates optional service authentication (Authorization: Bearer <token>)
    2. Extracts X-User-LLM-Model and X-User-LLM-API-Key headers (if present)
    3. Stores credentials in request-scoped context variables
    4. Cleans up context after request completes (via finally block)

    Credentials are OPTIONAL at the middleware level. Individual tools that
    require credentials (e.g., run_tau2_evaluation) validate them at the tool
    level and return appropriate errors if missing.

    Attributes:
        service_api_keys: Optional list of valid service API keys for authentication.
            When provided and non-empty, requests must include a valid
            Authorization: Bearer <token> header.

    Example:
        >>> from starlette.applications import Starlette
        >>> from tau2_agent.middleware import CredentialsMiddleware
        >>>
        >>> app = Starlette()
        >>> app.add_middleware(CredentialsMiddleware)
        >>>
        >>> # With optional service authentication:
        >>> app.add_middleware(CredentialsMiddleware, service_api_keys=["key1", "key2"])
    """

    def __init__(
        self,
        app,
        service_api_keys: list[str] | None = None,
    ):
        """Initialize CredentialsMiddleware.

        Args:
            app: The ASGI application.
            service_api_keys: Optional list of valid service API keys.
                When provided and non-empty, requests must include a valid
                Authorization: Bearer <token> header. When None or empty,
                service authentication is bypassed.
        """
        super().__init__(app)
        # Convert to frozenset for O(1) lookup; treat None as empty
        self._service_api_keys: frozenset[str] = frozenset(service_api_keys or [])
        self._auth_enabled = bool(self._service_api_keys)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request, extracting credential headers and managing context.

        Args:
            request: The incoming HTTP request.
            call_next: Callable to invoke the next middleware/handler.

        Returns:
            Response: Either error response (400/401) or the handler response.
        """
        # Bypass validation for health/status endpoints
        if self._should_bypass(request):
            return await call_next(request)

        # Validate service authentication first (if enabled)
        auth_error = self._validate_service_auth(request)
        if auth_error:
            return auth_error

        # Extract headers (case-insensitive via Starlette)
        model = request.headers.get(HEADER_MODEL, "").strip() or None
        api_key = request.headers.get(HEADER_API_KEY, "").strip() or None
        req_id = request.headers.get(HEADER_REQUEST_ID, "").strip() or None

        # Set context variables
        token_model = user_llm_model.set(model)
        token_key = user_llm_api_key.set(api_key)
        token_req_id = request_id.set(req_id)

        try:
            logger.info(
                "Request received",
                model=model,
                request_id=req_id,
                path=request.url.path,
                method=request.method,
            )

            # Process request
            response = await call_next(request)
            return response

        finally:
            # Always reset context to prevent leakage between requests
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)
            request_id.reset(token_req_id)

    def _should_bypass(self, request: Request) -> bool:
        """Check if request should bypass credential validation.

        Bypass conditions:
        - Path is in BYPASS_PATHS (health checks, root)
        - Path ends with a BYPASS_SUFFIXES entry (agent discovery endpoints)

        Args:
            request: The incoming HTTP request.

        Returns:
            True if request should bypass validation, False otherwise.
        """
        path = request.url.path

        # Always bypass health check paths
        if path in BYPASS_PATHS:
            return True

        # Bypass agent discovery endpoints (used by Cloud Run health probes)
        if path.endswith(BYPASS_SUFFIXES):
            return True

        return False

    def _validate_service_auth(self, request: Request) -> JSONResponse | None:
        """Validate service authentication via Authorization header.

        When service_api_keys is configured (non-empty), this method validates
        that the request includes a valid Authorization: Bearer <token> header
        where the token matches one of the configured service API keys.

        Args:
            request: The incoming HTTP request.

        Returns:
            JSONResponse with 401 error if auth is enabled and validation fails,
            None if auth is disabled or validation succeeds.
        """
        # Skip if service auth is not enabled
        if not self._auth_enabled:
            return None

        # Extract Authorization header
        auth_header = request.headers.get(HEADER_AUTHORIZATION, "").strip()

        # Check if header is present
        if not auth_header:
            error = EvaluationError(
                code=ErrorCode.INVALID_AUTH,
                message="Missing Authorization header",
            )
            logger.warning("Missing Authorization header for service auth")
            return JSONResponse(
                status_code=401,
                content=error.to_dict(),
            )

        # Parse Bearer token (case-insensitive 'Bearer' per RFC 7235)
        token = self._extract_bearer_token(auth_header)
        if token is None:
            error = EvaluationError(
                code=ErrorCode.INVALID_AUTH,
                message="Invalid authorization scheme, expected Bearer",
            )
            logger.warning("Invalid authorization scheme")
            return JSONResponse(
                status_code=401,
                content=error.to_dict(),
            )

        # Validate token against configured keys (never log the token)
        if token not in self._service_api_keys:
            error = EvaluationError(
                code=ErrorCode.INVALID_AUTH,
                message="Invalid authorization",
            )
            logger.warning("Invalid service API key attempted")
            return JSONResponse(
                status_code=401,
                content=error.to_dict(),
            )

        return None

    def _extract_bearer_token(self, auth_header: str) -> str | None:
        """Extract Bearer token from Authorization header.

        The 'Bearer' keyword is matched case-insensitively per RFC 7235,
        but the token itself is treated as case-sensitive.

        Args:
            auth_header: The full Authorization header value.

        Returns:
            The token string if valid Bearer format, None otherwise.
        """
        # Split into parts (e.g., "Bearer token123" -> ["Bearer", "token123"])
        parts = auth_header.split(None, 1)
        if len(parts) != 2:
            return None

        scheme, token = parts

        # Check scheme (case-insensitive per RFC 7235)
        if scheme.lower() != "bearer":
            return None

        # Return stripped token (must be non-empty)
        token = token.strip()
        return token if token else None