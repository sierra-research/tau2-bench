"""Connection settings for the Ava Zero Pipecat provider.

The Ava Zero server owns its own model, speech, and turn-taking configuration.
tau2 only needs to know where the server is, which agent to start, and how
long to wait for each step of the session handshake.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AVA_ZERO_BASE_URL,
    DEFAULT_AVA_ZERO_BOOT_ATTEMPTS,
    DEFAULT_AVA_ZERO_BOOT_TIMEOUT_SECONDS,
    DEFAULT_AVA_ZERO_COST_REPORT_TIMEOUT_SECONDS,
    DEFAULT_AVA_ZERO_MODEL,
    DEFAULT_AVA_ZERO_TOOL_REGISTRATION_TIMEOUT_SECONDS,
)


class AvaZeroPipecatConfig(BaseModel):
    """Connection configuration for an already-running Ava Zero server."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_AVA_ZERO_BASE_URL
    agent: str = Field(
        default=DEFAULT_AVA_ZERO_MODEL,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    websocket_token_env: Optional[str] = None

    connect_timeout_seconds: float = Field(
        default=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
        gt=0,
    )
    boot_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_BOOT_TIMEOUT_SECONDS,
        gt=0,
    )
    tool_registration_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_TOOL_REGISTRATION_TIMEOUT_SECONDS,
        gt=0,
    )
    cost_report_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_COST_REPORT_TIMEOUT_SECONDS,
        gt=0,
    )
    boot_attempts: int = Field(default=DEFAULT_AVA_ZERO_BOOT_ATTEMPTS, ge=1)
