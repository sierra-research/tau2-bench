"""Unit tests for tau2_agent context variables."""

import asyncio

import pytest

from tau2_agent.context import (
    CredentialsContext,
    request_id,
    user_llm_api_key,
    user_llm_model,
)


class TestContextVariables:
    """Tests for context variable set/get/reset lifecycle."""

    def test_default_values_are_none(self):
        """Context variables should default to None."""
        # Reset to ensure clean state
        user_llm_model.set(None)
        user_llm_api_key.set(None)
        request_id.set(None)

        assert user_llm_model.get() is None
        assert user_llm_api_key.get() is None
        assert request_id.get() is None

    def test_set_and_get(self):
        """Setting context variable should make it retrievable."""
        token = user_llm_model.set("gpt-4o")
        try:
            assert user_llm_model.get() == "gpt-4o"
        finally:
            user_llm_model.reset(token)

    def test_reset_restores_previous_value(self):
        """Resetting context variable should restore previous value."""
        # Set initial value
        token1 = user_llm_model.set("initial-model")
        try:
            # Set new value
            token2 = user_llm_model.set("new-model")
            assert user_llm_model.get() == "new-model"

            # Reset should restore previous value
            user_llm_model.reset(token2)
            assert user_llm_model.get() == "initial-model"
        finally:
            user_llm_model.reset(token1)

    def test_reset_to_default(self):
        """Resetting all the way should return to default None."""
        token = user_llm_model.set("test-model")
        assert user_llm_model.get() == "test-model"

        user_llm_model.reset(token)
        assert user_llm_model.get() is None

    def test_multiple_context_vars_independent(self):
        """Different context variables should be independent."""
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-test-key")
        token_req = request_id.set("req-123")

        try:
            assert user_llm_model.get() == "gpt-4o"
            assert user_llm_api_key.get() == "sk-test-key"
            assert request_id.get() == "req-123"

            # Reset one, others should remain
            user_llm_model.reset(token_model)
            assert user_llm_model.get() is None
            assert user_llm_api_key.get() == "sk-test-key"
            assert request_id.get() == "req-123"
        finally:
            user_llm_api_key.reset(token_key)
            request_id.reset(token_req)


class TestContextVariablesAsync:
    """Tests for async-safe context variable behavior."""

    @pytest.mark.asyncio
    async def test_context_isolated_between_tasks(self):
        """Context variables should be isolated between async tasks."""
        results = {}

        async def task_a():
            token = user_llm_model.set("model-a")
            try:
                await asyncio.sleep(0.01)  # Yield to other task
                results["task_a"] = user_llm_model.get()
            finally:
                user_llm_model.reset(token)

        async def task_b():
            token = user_llm_model.set("model-b")
            try:
                await asyncio.sleep(0.01)  # Yield to other task
                results["task_b"] = user_llm_model.get()
            finally:
                user_llm_model.reset(token)

        await asyncio.gather(task_a(), task_b())

        # Each task should see its own value
        assert results["task_a"] == "model-a"
        assert results["task_b"] == "model-b"

    @pytest.mark.asyncio
    async def test_context_preserved_across_await(self):
        """Context should be preserved across await points."""
        token = user_llm_model.set("persistent-model")
        try:
            assert user_llm_model.get() == "persistent-model"
            await asyncio.sleep(0.01)
            assert user_llm_model.get() == "persistent-model"
            await asyncio.sleep(0.01)
            assert user_llm_model.get() == "persistent-model"
        finally:
            user_llm_model.reset(token)

    @pytest.mark.asyncio
    async def test_middleware_pattern(self):
        """Test the middleware set/reset pattern."""

        async def simulated_middleware(model: str, api_key: str):
            """Simulate middleware setting context."""
            token_model = user_llm_model.set(model)
            token_key = user_llm_api_key.set(api_key)
            try:
                # Simulate handler execution
                return await simulated_handler()
            finally:
                user_llm_model.reset(token_model)
                user_llm_api_key.reset(token_key)

        async def simulated_handler():
            """Simulate handler reading context."""
            await asyncio.sleep(0.01)  # Simulate async work
            return {
                "model": user_llm_model.get(),
                "has_key": user_llm_api_key.get() is not None,
            }

        result = await simulated_middleware("claude-3-5-sonnet", "sk-test")

        assert result["model"] == "claude-3-5-sonnet"
        assert result["has_key"] is True

        # After middleware completes, context should be reset
        assert user_llm_model.get() is None
        assert user_llm_api_key.get() is None


class TestCredentialsContext:
    """Tests for CredentialsContext dataclass."""

    def test_from_context_returns_none_when_empty(self):
        """from_context should return None when context vars are not set."""
        # Ensure clean state
        user_llm_model.set(None)
        user_llm_api_key.set(None)

        assert CredentialsContext.from_context() is None

    def test_from_context_returns_none_when_partial(self):
        """from_context should return None when only some vars are set."""
        token = user_llm_model.set("gpt-4o")
        try:
            user_llm_api_key.set(None)
            assert CredentialsContext.from_context() is None
        finally:
            user_llm_model.reset(token)

    def test_from_context_creates_instance(self):
        """from_context should create CredentialsContext when all required vars set."""
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-test-key")
        token_req = request_id.set("req-123")

        try:
            ctx = CredentialsContext.from_context()
            assert ctx is not None
            assert ctx.user_llm_model == "gpt-4o"
            assert ctx.user_llm_api_key == "sk-test-key"
            assert ctx.request_id == "req-123"
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)
            request_id.reset(token_req)

    def test_from_context_without_request_id(self):
        """from_context should work without request_id."""
        token_model = user_llm_model.set("gpt-4o")
        token_key = user_llm_api_key.set("sk-test-key")
        request_id.set(None)

        try:
            ctx = CredentialsContext.from_context()
            assert ctx is not None
            assert ctx.user_llm_model == "gpt-4o"
            assert ctx.user_llm_api_key == "sk-test-key"
            assert ctx.request_id is None
        finally:
            user_llm_model.reset(token_model)
            user_llm_api_key.reset(token_key)

    def test_repr_hides_api_key(self):
        """repr should not expose the API key."""
        ctx = CredentialsContext(
            user_llm_model="gpt-4o",
            user_llm_api_key="sk-secret-key-12345",
            request_id="req-123",
        )
        repr_str = repr(ctx)

        assert "gpt-4o" in repr_str
        assert "req-123" in repr_str
        assert "sk-secret-key-12345" not in repr_str
        assert "***" in repr_str
