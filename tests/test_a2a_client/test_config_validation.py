"""Tests for A2AConfig validation.

A2AConfig validates endpoint URL scheme, timeout bounds,
and normalizes trailing slashes. These tests verify each validation rule.
"""

import pytest

from tau2.a2a.models import A2AConfig


class TestA2AConfigValidation:
    """Verify A2AConfig validation rules."""

    # --- Valid configs ---

    def test_valid_http_config(self):
        """Basic http:// config succeeds with correct defaults."""
        config = A2AConfig(endpoint="http://localhost:8080")

        assert config.endpoint == "http://localhost:8080"
        assert config.timeout == 300
        assert config.connect_timeout == 5
        assert config.auth_token is None
        assert config.verify_ssl is True

    def test_valid_https_config(self):
        """https:// endpoint is accepted."""
        config = A2AConfig(endpoint="https://agent.example.com")

        assert config.endpoint == "https://agent.example.com"

    # --- Trailing slash normalization ---

    def test_trailing_slash_normalized(self):
        """Single trailing slash is stripped."""
        config = A2AConfig(endpoint="http://localhost:8080/")

        assert config.endpoint == "http://localhost:8080"

    def test_multiple_trailing_slashes_normalized(self):
        """Multiple trailing slashes are all stripped."""
        config = A2AConfig(endpoint="http://localhost:8080///")

        assert config.endpoint == "http://localhost:8080"

    # --- Timeout validation ---

    def test_timeout_zero_raises(self):
        """timeout=0 is not positive and must raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            A2AConfig(endpoint="http://localhost:8080", timeout=0)

    def test_timeout_negative_raises(self):
        """Negative timeout must raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            A2AConfig(endpoint="http://localhost:8080", timeout=-1)

    def test_connect_timeout_zero_raises(self):
        """connect_timeout=0 must raise ValueError."""
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            A2AConfig(endpoint="http://localhost:8080", connect_timeout=0)

    def test_connect_timeout_negative_raises(self):
        """Negative connect_timeout must raise ValueError."""
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            A2AConfig(endpoint="http://localhost:8080", connect_timeout=-5)

    # --- URL scheme validation ---

    def test_invalid_scheme_ftp_raises(self):
        """ftp:// scheme is rejected."""
        with pytest.raises(ValueError, match="must start with http:// or https://"):
            A2AConfig(endpoint="ftp://agent.example.com")

    def test_invalid_scheme_no_scheme_raises(self):
        """Bare hostname without scheme is rejected."""
        with pytest.raises(ValueError, match="must start with http:// or https://"):
            A2AConfig(endpoint="agent.example.com")

    def test_invalid_scheme_websocket_raises(self):
        """ws:// scheme is rejected."""
        with pytest.raises(ValueError, match="must start with http:// or https://"):
            A2AConfig(endpoint="ws://agent.example.com")
