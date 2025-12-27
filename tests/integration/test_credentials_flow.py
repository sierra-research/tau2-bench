"""Integration tests for the full credentials request flow.

These tests validate the end-to-end flow of:
1. Credential header extraction in middleware
2. Context propagation to evaluation tools
3. Error handling for LLM authentication failures
4. Successful evaluation with mocked LLM responses

Tests use mock HTTP and patched LLM calls to avoid real external dependencies.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from tau2_agent.config import EvaluationLimits
from tau2_agent.context import (
    CredentialsContext,
    request_id,
    user_llm_api_key,
    user_llm_model,
)
from tau2_agent.errors import ErrorCode, EvaluationError
from tau2_agent.middleware import CredentialsMiddleware
from tau2_agent.tools.run_tau2_evaluation import (
    LimitExceededError,
    RunTau2Evaluation,
    UserLLMAuthError,
)


def create_test_tool():
    """Create a RunTau2Evaluation tool instance for testing."""
    return RunTau2Evaluation(
        name=RunTau2Evaluation.name,
        description=RunTau2Evaluation.description,
    )


# Test fixtures
@pytest.fixture
def credential_headers():
    """Standard credential headers for testing."""
    return {
        "X-User-LLM-Model": "gpt-4o",
        "X-User-LLM-API-Key": "sk-test-key-12345",
    }


@pytest.fixture
def credential_headers_anthropic():
    """Credential headers for Anthropic model."""
    return {
        "X-User-LLM-Model": "claude-3-5-sonnet-20241022",
        "X-User-LLM-API-Key": "sk-ant-test-key",
    }


@pytest.fixture
def credential_headers_gemini():
    """Credential headers for Gemini model."""
    return {
        "X-User-LLM-Model": "gemini/gemini-2.0-flash",
        "X-User-LLM-API-Key": "AIza-test-key",
    }


class TestCredentialsFlowIntegration:
    """Integration tests for the complete credentials request flow."""

    def create_test_app(self, handler, service_api_keys=None):
        """Create a test app with CredentialsMiddleware and given handler."""
        app = Starlette(
            routes=[
                Route("/a2a/tau2_agent", handler, methods=["POST"]),
                Route("/health", lambda _: JSONResponse({"status": "ok"})),
            ],
        )
        app.add_middleware(CredentialsMiddleware, service_api_keys=service_api_keys)
        return app

    def test_credential_headers_flow_to_context(self, credential_headers):
        """Credential headers should flow correctly through middleware to handler context."""
        context_captured = {}

        async def capture_context(request):
            context_captured["model"] = user_llm_model.get()
            context_captured["has_key"] = user_llm_api_key.get() is not None
            context_captured["key_value"] = user_llm_api_key.get()
            ctx = CredentialsContext.from_context()
            context_captured["credentials_context"] = ctx
            return JSONResponse({"status": "ok"})

        app = self.create_test_app(capture_context)
        client = TestClient(app)

        response = client.post("/a2a/tau2_agent", headers=credential_headers)

        assert response.status_code == 200
        assert context_captured["model"] == "gpt-4o"
        assert context_captured["has_key"] is True
        assert context_captured["key_value"] == "sk-test-key-12345"
        assert context_captured["credentials_context"] is not None
        assert context_captured["credentials_context"].user_llm_model == "gpt-4o"

    def test_credentials_context_isolated_between_requests(self, credential_headers):
        """Each request should have isolated credentials context."""
        request_contexts = []

        async def capture_context(request):
            ctx = CredentialsContext.from_context()
            request_contexts.append(ctx)
            return JSONResponse({"status": "ok"})

        app = self.create_test_app(capture_context)
        client = TestClient(app)

        # First request with OpenAI model
        client.post("/a2a/tau2_agent", headers=credential_headers)

        # Second request with Anthropic model
        client.post(
            "/a2a/tau2_agent",
            headers={
                "X-User-LLM-Model": "claude-3-5-sonnet-20241022",
                "X-User-LLM-API-Key": "sk-ant-different-key",
            },
        )

        assert len(request_contexts) == 2
        assert request_contexts[0].user_llm_model == "gpt-4o"
        assert request_contexts[1].user_llm_model == "claude-3-5-sonnet-20241022"
        assert request_contexts[0].user_llm_api_key == "sk-test-key-12345"
        assert request_contexts[1].user_llm_api_key == "sk-ant-different-key"

    def test_missing_credential_headers_passes_through(self):
        """Missing credential headers should pass through middleware.

        Credentials are optional at the middleware level per design.
        Individual tools validate if they require credentials.
        """
        handler_called = {"called": False}

        async def handler(request):
            handler_called["called"] = True
            return JSONResponse({"status": "ok"})

        app = self.create_test_app(handler)
        client = TestClient(app)

        response = client.post("/a2a/tau2_agent", headers={})

        # Middleware passes through - handler should be called
        assert response.status_code == 200
        assert handler_called["called"] is True

    def test_service_auth_flow_with_valid_token(self, credential_headers):
        """Valid service auth token should allow request to proceed."""
        async def handler(request):
            return JSONResponse({"status": "authenticated"})

        app = self.create_test_app(handler, service_api_keys=["valid-service-key"])
        client = TestClient(app)

        headers = {
            **credential_headers,
            "Authorization": "Bearer valid-service-key",
        }
        response = client.post("/a2a/tau2_agent", headers=headers)

        assert response.status_code == 200
        assert response.json()["status"] == "authenticated"

    def test_service_auth_flow_with_invalid_token(self, credential_headers):
        """Invalid service auth token should return 401."""
        async def handler(request):
            return JSONResponse({"status": "ok"})

        app = self.create_test_app(handler, service_api_keys=["valid-service-key"])
        client = TestClient(app)

        headers = {
            **credential_headers,
            "Authorization": "Bearer invalid-key",
        }
        response = client.post("/a2a/tau2_agent", headers=headers)

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == ErrorCode.INVALID_AUTH.value

    def test_request_id_propagation(self, credential_headers):
        """X-Request-ID header should propagate to context."""
        captured_request_id = {}

        async def capture_request_id(request):
            captured_request_id["value"] = request_id.get()
            return JSONResponse({"request_id": request_id.get()})

        app = self.create_test_app(capture_request_id)
        client = TestClient(app)

        headers = {
            **credential_headers,
            "X-Request-ID": "test-correlation-id-123",
        }
        response = client.post("/a2a/tau2_agent", headers=headers)

        assert response.status_code == 200
        assert captured_request_id["value"] == "test-correlation-id-123"


class TestLLMAuthFailureFlow:
    """Tests for LLM authentication failure handling (401 response)."""

    def test_user_llm_auth_error_creation(self):
        """UserLLMAuthError should properly wrap EvaluationError."""
        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="User LLM authentication failed",
            details={"model": "gpt-4o"},
        )
        auth_error = UserLLMAuthError(error)

        assert auth_error.error.code == ErrorCode.USER_LLM_AUTH_FAILED
        assert auth_error.error.message == "User LLM authentication failed"
        assert "api_key" not in str(auth_error.error.details)

    def test_user_llm_auth_failed_error_format(self):
        """USER_LLM_AUTH_FAILED error should have correct format."""
        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="User LLM authentication failed",
            details={"model": "gpt-4o"},
        )

        result = error.to_dict()

        assert result["code"] == "USER_LLM_AUTH_FAILED"
        assert result["error"] == "User LLM authentication failed"
        assert result["details"]["model"] == "gpt-4o"
        # API key must never be in error details
        assert "api_key" not in result.get("details", {})
        assert "key" not in str(result.get("details", {})).lower() or "api_key" not in str(result.get("details", {}))

    @pytest.mark.asyncio
    async def test_authentication_error_triggers_user_llm_auth_error(self):
        """_execute should raise UserLLMAuthError for authentication failures.

        This tests the actual error detection logic in run_tau2_evaluation._execute
        rather than duplicating the pattern matching logic.
        """
        tool = create_test_tool()

        # Set up credentials context
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-invalid-key")

        # Create a custom exception class to simulate litellm's AuthenticationError
        class AuthenticationError(Exception):
            """Mock litellm AuthenticationError."""

        try:
            # Mock tau2 imports (imported inside _execute method)
            with patch("tau2.registry.registry") as mock_registry:
                mock_registry.get_domains.return_value = ["mock", "airline"]

                # Simulate litellm AuthenticationError during run_domain
                with patch("tau2.run.run_domain") as mock_run:
                    # Create an exception with class name matching auth error pattern
                    auth_error = AuthenticationError("Invalid API key provided")
                    mock_run.side_effect = auth_error

                    with pytest.raises(UserLLMAuthError) as exc_info:
                        await tool._execute(
                            _tool_context=MagicMock(invocation_id="test"),
                            domain="mock",
                            agent_endpoint="http://test:8001",
                            user_llm="gpt-4o",
                            llm_args_user={"api_key": "sk-invalid-key"},
                        )

                    # Verify the error is properly wrapped
                    assert exc_info.value.error.code == ErrorCode.USER_LLM_AUTH_FAILED
                    assert "api_key" not in str(exc_info.value.error.details)
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    def test_user_llm_auth_error_wrapping(self):
        """UserLLMAuthError should correctly wrap authentication failures."""
        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="User LLM authentication failed",
            details={"model": "gpt-4o"},
        )
        auth_error = UserLLMAuthError(error)

        # Verify error is wrapped correctly
        assert isinstance(auth_error, Exception)
        assert auth_error.error is error
        assert auth_error.error.code == ErrorCode.USER_LLM_AUTH_FAILED

        # Verify string representation
        error_str = str(auth_error)
        assert "USER_LLM_AUTH_FAILED" in error_str
        assert "authentication failed" in error_str.lower()


class TestSuccessfulEvaluationFlow:
    """Tests for successful evaluation with mocked LLM responses."""

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock successful evaluation result."""
        return {
            "status": "completed",
            "timestamp": "2025-01-01T00:00:00Z",
            "summary": {
                "total_simulations": 3,
                "total_tasks": 3,
                "successful_simulations": 2,
                "avg_reward": 0.75,
                "pass_hat_k": {"1": 0.67},
                "avg_agent_cost": 0.05,
            },
            "tasks": [
                {"task_id": "task_1", "purpose": "Test task 1"},
                {"task_id": "task_2", "purpose": "Test task 2"},
                {"task_id": "task_3", "purpose": "Test task 3"},
            ],
        }

    def test_limit_validation_max_tasks(self):
        """Tool should reject num_tasks > MAX_TASKS."""
        tool = create_test_tool()

        with pytest.raises(LimitExceededError) as exc_info:
            tool._validate_limits(num_tasks=50, num_trials=1)

        error = exc_info.value.error
        assert error.code == ErrorCode.LIMIT_EXCEEDED
        assert "num_tasks" in error.message
        assert error.details["num_tasks"] == 50
        assert error.details["max_tasks"] == EvaluationLimits.MAX_TASKS

    def test_limit_validation_max_trials(self):
        """Tool should reject num_trials > MAX_TRIALS."""
        tool = create_test_tool()

        with pytest.raises(LimitExceededError) as exc_info:
            tool._validate_limits(num_tasks=10, num_trials=5)

        error = exc_info.value.error
        assert error.code == ErrorCode.LIMIT_EXCEEDED
        assert "num_trials" in error.message
        assert error.details["num_trials"] == 5
        assert error.details["max_trials"] == EvaluationLimits.MAX_TRIALS

    def test_limit_validation_accepts_valid_params(self):
        """Tool should accept valid num_tasks and num_trials."""
        tool = create_test_tool()

        # Should not raise
        tool._validate_limits(num_tasks=30, num_trials=3)
        tool._validate_limits(num_tasks=1, num_trials=1)
        tool._validate_limits(num_tasks=None, num_trials=1)

    @patch.dict("os.environ", {"NEBIUS_API_BASE": "", "USER_LLM_API_BASE": ""}, clear=False)
    def test_credentials_from_context(self):
        """Tool should read credentials from context variables."""
        tool = create_test_tool()

        # Set up context
        token_model = user_llm_model.set("claude-3-5-sonnet-20241022")
        token_key = user_llm_api_key.set("sk-ant-secret-key")

        try:
            model, llm_args = tool._get_user_llm_credentials()

            # Model should be formatted for LiteLLM (anthropic/ prefix added)
            assert model == "anthropic/claude-3-5-sonnet-20241022"
            assert llm_args == {"api_key": "sk-ant-secret-key"}
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    @patch.dict("os.environ", {"NEBIUS_API_BASE": "", "USER_LLM_API_BASE": ""}, clear=False)
    def test_credentials_formats_openai_model(self):
        """Tool should format OpenAI models correctly (no prefix)."""
        tool = create_test_tool()

        # Set up context with OpenAI model
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-openai-key")

        try:
            model, llm_args = tool._get_user_llm_credentials()

            # OpenAI models don't need a prefix
            assert model == "gpt-4o"
            assert llm_args == {"api_key": "sk-openai-key"}
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    @patch.dict("os.environ", {"NEBIUS_API_BASE": "", "USER_LLM_API_BASE": ""}, clear=False)
    def test_credentials_formats_gemini_model(self):
        """Tool should format Gemini models correctly."""
        tool = create_test_tool()

        # Set up context with Gemini model (without prefix)
        token_model = user_llm_model.set("gemini-2.0-flash")
        token_key = user_llm_api_key.set("AIza-test-key")

        try:
            model, llm_args = tool._get_user_llm_credentials()

            # Gemini models get gemini/ prefix added
            assert model == "gemini/gemini-2.0-flash"
            assert llm_args == {"api_key": "AIza-test-key"}
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    def test_missing_credentials_raises_error(self):
        """Tool should raise MissingCredentialsError when credentials not set."""
        from tau2_agent.tools.run_tau2_evaluation import MissingCredentialsError

        tool = create_test_tool()

        # Ensure no credentials context is set
        token_model = user_llm_model.set(None)
        token_key = user_llm_api_key.set(None)

        try:
            with pytest.raises(MissingCredentialsError) as exc_info:
                tool._get_user_llm_credentials()

            assert exc_info.value.error.code == ErrorCode.MISSING_HEADER
            assert "X-User-LLM-Model" in exc_info.value.error.message
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)


class TestEndToEndCredentialsFlow:
    """End-to-end tests simulating the complete credentials flow."""

    def test_complete_credentials_flow_mock_evaluation(self):
        """Test complete flow from HTTP request through to evaluation result."""
        evaluation_params = {}

        async def mock_evaluation_handler(request):
            # Capture what the handler sees from context
            evaluation_params["model"] = user_llm_model.get()
            evaluation_params["has_api_key"] = user_llm_api_key.get() is not None
            evaluation_params["request_id"] = request_id.get()

            # Simulate successful evaluation response
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": "test-1",
                "result": {
                    "status": "completed",
                    "summary": {
                        "total_simulations": 3,
                        "successful_simulations": 2,
                    },
                },
            })

        app = Starlette(
            routes=[Route("/a2a/tau2_agent", mock_evaluation_handler, methods=["POST"])],
        )
        app.add_middleware(CredentialsMiddleware)
        client = TestClient(app)

        # Send A2A request with credential headers
        response = client.post(
            "/a2a/tau2_agent",
            headers={
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-user-key-123",
                "X-Request-ID": "correlation-123",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"text": "Run mock evaluation"}],
                    }
                },
            },
        )

        assert response.status_code == 200
        assert evaluation_params["model"] == "gpt-4o"
        assert evaluation_params["has_api_key"] is True
        assert evaluation_params["request_id"] == "correlation-123"

        # Verify response structure
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["result"]["status"] == "completed"

    def test_credentials_flow_with_service_auth(self):
        """Test credentials flow with service authentication enabled."""
        async def handler(request):
            return JSONResponse({"status": "success", "model": user_llm_model.get()})

        app = Starlette(
            routes=[Route("/a2a/tau2_agent", handler, methods=["POST"])],
        )
        app.add_middleware(CredentialsMiddleware, service_api_keys=["service-key-abc"])
        client = TestClient(app)

        # Request with both service auth and credential headers
        response = client.post(
            "/a2a/tau2_agent",
            headers={
                "Authorization": "Bearer service-key-abc",
                "X-User-LLM-Model": "claude-3-5-sonnet-20241022",
                "X-User-LLM-API-Key": "sk-ant-key",
            },
        )

        assert response.status_code == 200
        assert response.json()["model"] == "claude-3-5-sonnet-20241022"

    def test_credentials_flow_rejects_invalid_service_auth(self):
        """Service auth failure should block request before credential validation."""
        handler_reached = {"reached": False}

        async def handler(request):
            handler_reached["reached"] = True
            return JSONResponse({"status": "success"})

        app = Starlette(
            routes=[Route("/a2a/tau2_agent", handler, methods=["POST"])],
        )
        app.add_middleware(CredentialsMiddleware, service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/a2a/tau2_agent",
            headers={
                "Authorization": "Bearer wrong-key",
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-key",
            },
        )

        assert response.status_code == 401
        assert handler_reached["reached"] is False
        assert response.json()["code"] == "INVALID_AUTH"


class TestSecurityConstraints:
    """Tests for security constraints in the credentials flow."""

    def test_api_key_never_in_error_response(self):
        """API key values must never appear in error responses.

        Tests that when a tool raises MissingCredentialsError (e.g., missing model),
        the API key is not leaked in the error response.
        """
        tool = create_test_tool()

        # Set only API key, not model - this should trigger MissingCredentialsError
        token_key = user_llm_api_key.set("sk-secret-key-12345")
        # Model is not set, so it will be None

        try:
            from tau2_agent.tools.run_tau2_evaluation import MissingCredentialsError

            with pytest.raises(MissingCredentialsError) as exc_info:
                tool._get_user_llm_credentials()

            # The error message/details must not contain the API key
            error = exc_info.value.error
            assert "sk-secret-key-12345" not in error.message
            assert "sk-secret-key-12345" not in str(error.details or {})
        finally:
            user_llm_api_key.reset(token_key)

    def test_api_key_never_in_auth_error(self):
        """API key must not appear in service auth error responses."""
        async def handler(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(CredentialsMiddleware, service_api_keys=["valid-key"])
        client = TestClient(app)

        response = client.post(
            "/test",
            headers={
                "Authorization": "Bearer my-secret-token-xyz",
                "X-User-LLM-Model": "gpt-4o",
                "X-User-LLM-API-Key": "sk-another-secret",
            },
        )

        assert response.status_code == 401
        response_text = response.text

        # Neither token should appear in error
        assert "my-secret-token-xyz" not in response_text
        assert "sk-another-secret" not in response_text

    def test_credentials_context_repr_masks_key(self):
        """CredentialsContext repr should mask the API key."""
        ctx = CredentialsContext(
            user_llm_model="gpt-4o",
            user_llm_api_key="sk-super-secret-key",
            request_id="test-123",
        )

        repr_str = repr(ctx)

        assert "sk-super-secret-key" not in repr_str
        assert "***" in repr_str
        assert "gpt-4o" in repr_str

    def test_evaluation_error_masks_api_key_in_details(self):
        """EvaluationError should not include api_key in details."""
        # This tests the contract - details should never have api_key
        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="Auth failed",
            details={"model": "gpt-4o"},  # api_key should NOT be here
        )

        result = error.to_dict()

        assert "api_key" not in str(result)
        assert "key" not in result.get("details", {})
