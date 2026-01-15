"""Exception types for A2A protocol integration."""

from typing import Any


class A2AError(Exception):
    """Base exception for A2A protocol errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize A2A error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"A2AError: {self.message}"]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)


class A2ATimeoutError(A2AError):
    """Exception raised when A2A agent response times out."""

    def __init__(
        self, message: str = "A2A agent response timeout", timeout: int | None = None
    ):
        """
        Initialize timeout error.

        Args:
            message: Error message
            timeout: Timeout value in seconds
        """
        details = {"timeout": timeout} if timeout else {}
        super().__init__(message, status_code=408, details=details)


class A2AAuthError(A2AError):
    """Exception raised when A2A authentication fails."""

    def __init__(self, message: str = "A2A authentication failed"):
        """
        Initialize authentication error.

        Args:
            message: Error message
        """
        super().__init__(message, status_code=401)


class A2ADiscoveryError(A2AError):
    """Exception raised when agent discovery fails."""

    def __init__(
        self, message: str = "Agent discovery failed", endpoint: str | None = None
    ):
        """
        Initialize discovery error.

        Args:
            message: Error message
            endpoint: Agent endpoint that failed discovery
        """
        details = {"endpoint": endpoint} if endpoint else {}
        super().__init__(message, status_code=404, details=details)


class A2AMessageError(A2AError):
    """Exception raised when message parsing or validation fails."""

    def __init__(
        self, message: str = "A2A message error", message_id: str | None = None
    ):
        """
        Initialize message error.

        Args:
            message: Error message
            message_id: ID of the message that caused the error
        """
        details = {"message_id": message_id} if message_id else {}
        super().__init__(message, status_code=400, details=details)
