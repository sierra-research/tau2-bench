"""
Request context management for tau2_agent user LLM credentials.

This module provides async-safe, request-scoped context variables for storing
user LLM credentials extracted from HTTP headers. Uses Python's contextvars
module which is safe for async/await patterns used by FastAPI/Starlette.

Usage:
    # In middleware - set context
    from tau2_agent.context import user_llm_model, user_llm_api_key

    token_model = user_llm_model.set("gpt-4o")
    token_key = user_llm_api_key.set("sk-...")
    try:
        response = await call_next(request)
    finally:
        user_llm_model.reset(token_model)
        user_llm_api_key.reset(token_key)

    # In tool code - read context
    model = user_llm_model.get()
    api_key = user_llm_api_key.get()
"""

from contextvars import ContextVar
from dataclasses import dataclass

# Context variables for user LLM credentials
# These are set by middleware and read by tools deep in the call stack

user_llm_model: ContextVar[str | None] = ContextVar("user_llm_model", default=None)
"""LiteLLM model identifier provided by client (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')"""

user_llm_api_key: ContextVar[str | None] = ContextVar("user_llm_api_key", default=None)
"""API key for the user's LLM provider. Never logged."""

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
"""Optional correlation ID for request tracing."""


@dataclass
class CredentialsContext:
    """User LLM credentials extracted from request headers.

    This dataclass provides a structured representation of the credentials context
    for cases where passing individual values is inconvenient.

    Attributes:
        user_llm_model: LiteLLM model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
        user_llm_api_key: API key for the user's LLM provider (never logged)
        request_id: Optional correlation ID for tracing
    """

    user_llm_model: str
    user_llm_api_key: str
    request_id: str | None = None

    @classmethod
    def from_context(cls) -> "CredentialsContext | None":
        """Create CredentialsContext from current context variables.

        Returns:
            CredentialsContext if both model and api_key are set, None otherwise.
        """
        model = user_llm_model.get()
        api_key = user_llm_api_key.get()

        if model is None or api_key is None:
            return None

        return cls(
            user_llm_model=model,
            user_llm_api_key=api_key,
            request_id=request_id.get(),
        )

    def __repr__(self) -> str:
        """Return string representation without exposing API key."""
        return (
            f"CredentialsContext(user_llm_model={self.user_llm_model!r}, "
            f"user_llm_api_key='***', "
            f"request_id={self.request_id!r})"
        )
