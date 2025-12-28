"""
Error codes and error response types for tau2_agent.

This module defines standardized error codes and a structured error response
dataclass for consistent error handling across the tau2_agent service.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for tau2_agent service.

    Each error code maps to a specific HTTP status code and error category:

    - MISSING_HEADER (400): Required credential header not provided
        Returned when X-User-LLM-Model or X-User-LLM-API-Key headers are missing.

    - INVALID_AUTH (401): Service authentication failed
        Returned when SERVICE_API_KEYS is configured but the provided
        Authorization Bearer token is invalid or missing.

    - USER_LLM_AUTH_FAILED (401): User's LLM API key is invalid
        Returned when the user-provided API key fails authentication
        with the LLM provider (e.g., OpenAI, Anthropic).

    - LIMIT_EXCEEDED (400): Evaluation parameter limits exceeded
        Returned when num_tasks > 30 or num_trials > 3, which would
        exceed Cloud Run's 60-minute timeout.

    - EVALUATION_FAILED (500): Evaluation execution failed
        Returned when the tau2-bench evaluation fails due to an
        unexpected error during execution.
    """

    MISSING_HEADER = "MISSING_HEADER"
    INVALID_AUTH = "INVALID_AUTH"
    USER_LLM_AUTH_FAILED = "USER_LLM_AUTH_FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    EVALUATION_FAILED = "EVALUATION_FAILED"


@dataclass
class EvaluationError:
    """Structured error response for tau2_agent.

    Provides a consistent format for error responses that can be
    serialized to JSON for HTTP responses.

    Attributes:
        code: The error code identifying the error type.
        message: Human-readable error message.
        details: Optional additional context about the error.

    Example:
        >>> error = EvaluationError(
        ...     code=ErrorCode.LIMIT_EXCEEDED,
        ...     message="num_tasks must be between 1 and 30",
        ...     details={"num_tasks": 50, "max_tasks": 30}
        ... )
        >>> error.to_dict()
        {'error': 'num_tasks must be between 1 and 30', 'code': 'LIMIT_EXCEEDED', 'details': {'num_tasks': 50, 'max_tasks': 30}}
    """

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON serialization.

        Returns:
            Dictionary with 'error', 'code', and optionally 'details' keys.
            The 'error' key contains the human-readable message.
            The 'code' key contains the ErrorCode value.
        """
        result: dict[str, Any] = {
            "error": self.message,
            "code": self.code.value,
        }
        if self.details is not None:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"[{self.code.value}] {self.message} - {self.details}"
        return f"[{self.code.value}] {self.message}"
