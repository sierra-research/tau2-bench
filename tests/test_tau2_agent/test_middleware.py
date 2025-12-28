"""Unit tests for tau2_agent credentials middleware."""

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from tau2_agent.context import request_id, user_llm_api_key, user_llm_model
from tau2_agent.errors import ErrorCode
from tau2_agent.middleware import CredentialsMiddleware


# Test app that echoes back context values
async def echo_context(request):
    """Handler that returns current context values."""
    return JSONResponse({
        "model": user_llm_model.get(),
        "has_api_key": user_llm_api_key.get() is not None,
        "request_id": request_id.get(),
    })


def create_test_app():
    """Create a test Starlette app with credentials middleware."""
    app = Starlette(
        routes=[Route("/test", echo_context, methods=["POST", "GET"])],
    )
    app.add_middleware(CredentialsMiddleware)
    return app


@pytest.fixture
def client():
    """Create a test client with credentials middleware."""
    app = create_test_app()
    return TestClient(app)


class TestCredentialsMiddlewareHeaderExtraction:
    """Tests for credential header extraction."""

    def test_extracts_headers_openai(self, client):
        """Middleware should extract credential headers for OpenAI models."""
        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test-key-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-4o"
        assert data["has_api_key"] is True

    def test_extracts_headers_anthropic(self, client):
        """Middleware should extract credential headers for Anthropic models."""
        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "claude-3-5-sonnet-20241022",
                "X-User-LLM-API-Key": "sk-ant-test-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert data["has_api_key"] is True

    def test_extracts_headers_gemini(self, client):
        """Middleware should extract credential headers for Gemini models."""
        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gemini/gemini-2.0-flash",
                "X-User-LLM-API-Key": "AIza-test-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gemini/gemini-2.0-flash"
        assert data["has_api_key"] is True

    def test_headers_are_case_insensitive(self, client):
        """Header names should be case-insensitive per HTTP spec."""
        response = client.post(
            "/test",
            headers={
                "x-user-llm-model": "claude-3-5-sonnet-20241022",
                "x-user-llm-api-key": "sk-ant-test-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert data["has_api_key"] is True

    def test_mixed_case_headers(self, client):
        """Mixed case headers should work."""
        response = client.post(
            "/test",
            headers={
                "X-USER-LLM-MODEL": "gemini/gemini-2.0-flash",
                "X-USER-LLM-API-KEY": "AIza-test-key",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gemini/gemini-2.0-flash"
        assert data["has_api_key"] is True

    def test_context_reset_after_request(self, client):
        """Context should be reset after request completes."""
        # Make a request to set context
        client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
            },
        )

        # After request, context should be reset to None
        assert user_llm_model.get() is None
        assert user_llm_api_key.get() is None

    def test_extracts_request_id_header(self, client):
        """Middleware should extract X-Request-ID header if present."""
        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "X-Request-ID": "req-abc-123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-abc-123"

    def test_request_id_optional(self, client):
        """X-Request-ID should be optional."""
        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o-mini",
                "X-User-LLM-API-Key": "sk-test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] is None


class TestCredentialsMiddlewareOptionalHeaders:
    """Middleware is permissive; tools validate credentials at execution time."""

    def test_missing_model_header_succeeds(self, client):
        """Missing X-User-LLM-Model still allows request through."""
        response = client.post("/test", headers={"X-User-LLM-API-Key": "sk-test-key"})

        assert response.status_code == 200
        data = response.json()
        assert data["model"] is None
        assert data["has_api_key"] is True

    def test_missing_api_key_header_succeeds(self, client):
        """Missing X-User-LLM-API-Key still allows request through."""
        response = client.post("/test", headers={"X-User-LLM-Model": "gpt-4o"})

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-4o"
        assert data["has_api_key"] is False

    def test_missing_both_headers_succeeds(self, client):
        """Missing both headers still allows request through."""
        response = client.post("/test", headers={})

        assert response.status_code == 200
        data = response.json()
        assert data["model"] is None
        assert data["has_api_key"] is False

    def test_empty_model_header_treated_as_none(self, client):
        """Empty X-User-LLM-Model treated as None."""
        response = client.post(
            "/test",
            headers={"X-User-LLM-Model": "", "X-User-LLM-API-Key": "sk-test"},
        )

        assert response.status_code == 200
        assert response.json()["model"] is None

    def test_empty_api_key_header_treated_as_none(self, client):
        """Empty X-User-LLM-API-Key treated as None."""
        response = client.post(
            "/test",
            headers={"X-User-LLM-Model": "gpt-4o", "X-User-LLM-API-Key": ""},
        )

        assert response.status_code == 200
        assert response.json()["has_api_key"] is False

    def test_whitespace_only_headers_treated_as_none(self, client):
        """Whitespace-only headers treated as None."""
        response = client.post(
            "/test",
            headers={"X-User-LLM-Model": "   ", "X-User-LLM-API-Key": "sk-test"},
        )

        assert response.status_code == 200
        assert response.json()["model"] is None


class TestCredentialsMiddlewareHealthEndpoints:
    """Tests for health/status endpoints that should bypass credential validation."""

    def create_app_with_health(self):
        """Create app with health endpoint."""
        async def health(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(
            routes=[
                Route("/health", health, methods=["GET"]),
                Route("/test", echo_context, methods=["POST"]),
            ],
        )
        app.add_middleware(CredentialsMiddleware)
        return app

    def test_health_endpoint_bypasses_validation(self):
        """GET /health should not require credential headers."""
        app = self.create_app_with_health()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint_bypasses_validation(self):
        """GET / should not require credential headers for root path."""
        async def root(request):
            return JSONResponse({"service": "tau2_agent"})

        app = Starlette(
            routes=[
                Route("/", root, methods=["GET"]),
                Route("/test", echo_context, methods=["POST"]),
            ],
        )
        app.add_middleware(CredentialsMiddleware)
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200

    def test_agent_card_endpoint_bypasses_validation(self):
        """Agent card endpoint should bypass credentials for Cloud Run health probes."""
        async def agent_card(request):
            return JSONResponse({"name": "tau2_agent", "version": "1.0"})

        app = Starlette(
            routes=[
                Route(
                    "/a2a/tau2_agent/.well-known/agent-card.json",
                    agent_card,
                    methods=["GET"],
                ),
                Route("/test", echo_context, methods=["POST"]),
            ],
        )
        app.add_middleware(CredentialsMiddleware)
        client = TestClient(app)

        # No credential headers - should still succeed (bypass for health probes)
        response = client.get("/a2a/tau2_agent/.well-known/agent-card.json")

        assert response.status_code == 200
        assert response.json()["name"] == "tau2_agent"


class TestCredentialsMiddlewareContextIsolation:
    """Tests for context isolation between requests."""

    def test_concurrent_requests_have_isolated_context(self, client):
        """Different requests should have isolated context."""
        # First request with OpenAI model
        response1 = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-key-a",
            },
        )

        # Second request with Anthropic model
        response2 = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "claude-3-5-sonnet-20241022",
                "X-User-LLM-API-Key": "sk-ant-key-b",
            },
        )

        # Each should see their own values
        assert response1.json()["model"] == "gpt-4o"
        assert response2.json()["model"] == "claude-3-5-sonnet-20241022"

    def test_failed_request_still_cleans_context(self, client):
        """Context should be cleaned up even when request processing fails."""
        # Make a request that will be rejected
        client.post("/test", headers={})

        # Context should still be clean
        assert user_llm_model.get() is None
        assert user_llm_api_key.get() is None


class TestLimitEnforcement:
    """Tests for evaluation limit enforcement in run_tau2_evaluation.py.

    These tests verify that the EvaluationLimits constants (MAX_TASKS=30,
    MAX_TRIALS=3) are properly enforced and return appropriate errors.
    """

    def test_max_tasks_limit_constant(self):
        """EvaluationLimits.MAX_TASKS should be 30."""
        from tau2_agent.config import EvaluationLimits

        assert EvaluationLimits.MAX_TASKS == 30

    def test_max_trials_limit_constant(self):
        """EvaluationLimits.MAX_TRIALS should be 3."""
        from tau2_agent.config import EvaluationLimits

        assert EvaluationLimits.MAX_TRIALS == 3

    def test_limit_exceeded_error_code_exists(self):
        """ErrorCode.LIMIT_EXCEEDED should be defined."""
        from tau2_agent.errors import ErrorCode

        assert hasattr(ErrorCode, "LIMIT_EXCEEDED")
        assert ErrorCode.LIMIT_EXCEEDED.value == "LIMIT_EXCEEDED"

    def test_evaluation_error_with_limit_exceeded(self):
        """EvaluationError should format LIMIT_EXCEEDED error correctly."""
        from tau2_agent.errors import ErrorCode, EvaluationError

        error = EvaluationError(
            code=ErrorCode.LIMIT_EXCEEDED,
            message="num_tasks must be between 1 and 30",
            details={"num_tasks": 50, "max_tasks": 30},
        )

        result = error.to_dict()

        assert result["code"] == "LIMIT_EXCEEDED"
        assert result["error"] == "num_tasks must be between 1 and 30"
        assert result["details"]["num_tasks"] == 50
        assert result["details"]["max_tasks"] == 30

    def test_evaluation_error_with_trials_exceeded(self):
        """EvaluationError should format LIMIT_EXCEEDED for trials correctly."""
        from tau2_agent.errors import ErrorCode, EvaluationError

        error = EvaluationError(
            code=ErrorCode.LIMIT_EXCEEDED,
            message="num_trials must be between 1 and 3",
            details={"num_trials": 5, "max_trials": 3},
        )

        result = error.to_dict()

        assert result["code"] == "LIMIT_EXCEEDED"
        assert result["error"] == "num_trials must be between 1 and 3"
        assert result["details"]["num_trials"] == 5

    def test_user_llm_auth_failed_error_code_exists(self):
        """ErrorCode.USER_LLM_AUTH_FAILED should be defined."""
        from tau2_agent.errors import ErrorCode

        assert hasattr(ErrorCode, "USER_LLM_AUTH_FAILED")
        assert ErrorCode.USER_LLM_AUTH_FAILED.value == "USER_LLM_AUTH_FAILED"

    def test_evaluation_error_with_user_llm_auth_failed(self):
        """EvaluationError should format USER_LLM_AUTH_FAILED error correctly."""
        from tau2_agent.errors import ErrorCode, EvaluationError

        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="User LLM authentication failed",
            details={"model": "gpt-4o"},
        )

        result = error.to_dict()

        assert result["code"] == "USER_LLM_AUTH_FAILED"
        assert result["error"] == "User LLM authentication failed"
        # API key should never appear in details
        assert "api_key" not in result.get("details", {})

    def test_context_from_context_method(self):
        """CredentialsContext.from_context() should return context when set."""
        from tau2_agent.context import (
            CredentialsContext,
            user_llm_api_key,
            user_llm_model,
        )

        # Set context
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-test-key")

        try:
            ctx = CredentialsContext.from_context()
            assert ctx is not None
            assert ctx.user_llm_model == "gpt-4o"
            assert ctx.user_llm_api_key == "sk-test-key"
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    def test_context_from_context_returns_none_when_not_set(self):
        """CredentialsContext.from_context() should return None when context not set."""
        from tau2_agent.context import CredentialsContext

        ctx = CredentialsContext.from_context()
        assert ctx is None


class TestOptionalServiceAuth:
    """Tests for optional service authentication via Authorization header.

    When SERVICE_API_KEYS is configured, requests must include a valid
    Authorization: Bearer <token> header matching one of the configured keys.
    When not configured, authorization is bypassed.
    """

    def create_app_with_service_auth(self, service_api_keys: list[str] | None = None):
        """Create a test Starlette app with optional service auth.

        Args:
            service_api_keys: List of valid service API keys. None disables auth.
        """
        app = Starlette(
            routes=[Route("/test", echo_context, methods=["POST", "GET"])],
        )
        app.add_middleware(CredentialsMiddleware, service_api_keys=service_api_keys)
        return app

    def test_no_auth_when_service_keys_not_configured(self):
        """When SERVICE_API_KEYS is not configured, auth is bypassed."""
        app = self.create_app_with_service_auth(service_api_keys=None)
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
            },
        )

        assert response.status_code == 200

    def test_no_auth_when_service_keys_empty(self):
        """When SERVICE_API_KEYS is empty list, auth is bypassed."""
        app = self.create_app_with_service_auth(service_api_keys=[])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
            },
        )

        assert response.status_code == 200

    def test_valid_bearer_token_accepted(self):
        """Valid Bearer token matching SERVICE_API_KEYS should be accepted."""
        app = self.create_app_with_service_auth(
            service_api_keys=["valid-key-123", "valid-key-456"]
        )
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer valid-key-123",
            },
        )

        assert response.status_code == 200

    def test_valid_bearer_token_second_key(self):
        """Any valid key in SERVICE_API_KEYS should be accepted."""
        app = self.create_app_with_service_auth(
            service_api_keys=["key-a", "key-b", "key-c"]
        )
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer key-c",
            },
        )

        assert response.status_code == 200

    def test_missing_auth_header_returns_401(self):
        """Missing Authorization header returns 401 when auth is enabled."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value
        assert "error" in data

    def test_invalid_token_returns_401(self):
        """Invalid Bearer token returns 401."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer wrong-key",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value

    def test_non_bearer_auth_returns_401(self):
        """Non-Bearer authorization scheme returns 401."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Basic dXNlcjpwYXNz",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value

    def test_empty_bearer_token_returns_401(self):
        """Empty Bearer token returns 401."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer ",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value

    def test_bearer_token_case_sensitive(self):
        """Bearer token matching is case-sensitive."""
        app = self.create_app_with_service_auth(service_api_keys=["Valid-Key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer valid-key",  # lowercase
            },
        )

        assert response.status_code == 401

    def test_bearer_keyword_case_insensitive(self):
        """'Bearer' keyword should be case-insensitive per RFC 7235."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "bearer valid-key",  # lowercase bearer
            },
        )

        assert response.status_code == 200

    def test_auth_checked_before_credentials(self):
        """Service auth should be validated before credential headers."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        # Missing credential headers but also missing auth
        response = client.post(
            "/test",
            headers={},
        )

        # Should get 401 (auth error), not 400 (missing credential headers)
        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value

    def test_health_endpoints_bypass_service_auth(self):
        """Health endpoints should bypass service auth."""
        async def health(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(
            routes=[
                Route("/health", health, methods=["GET"]),
                Route("/test", echo_context, methods=["POST"]),
            ],
        )
        app.add_middleware(CredentialsMiddleware, service_api_keys=["valid-key"])
        client = TestClient(app)

        # Health endpoint without auth header
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_error_response_does_not_echo_token(self):
        """Error responses must never include the attempted token value."""
        app = self.create_app_with_service_auth(service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-test",
                "Authorization": "Bearer secret-attempt-123",
            },
        )

        assert response.status_code == 401
        response_text = response.text
        # Token value should never appear in response
        assert "secret-attempt-123" not in response_text
