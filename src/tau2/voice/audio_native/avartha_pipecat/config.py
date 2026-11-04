"""Connection settings for the Avartha Pipecat provider.

The hosted agent owns its own model, speech, and turn-taking configuration.
tau2 only needs the agent's URL and how long to wait for each step of the call
handshake.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AVARTHA_BOOT_ATTEMPTS,
    DEFAULT_AVARTHA_BOOT_TIMEOUT_SECONDS,
    DEFAULT_AVARTHA_TOOL_REGISTRATION_TIMEOUT_SECONDS,
)


class AvarthaPipecatConfig(BaseModel):
    """Connection configuration for a hosted Avartha agent."""

    model_config = ConfigDict(extra="forbid")

    # The hosted agent's URL,
    # https://<host>/inference/serverless/pipecat/v1/public/<agent_id>.
    base_url: str

    connect_timeout_seconds: float = Field(
        default=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
        gt=0,
    )
    boot_timeout_seconds: float = Field(
        default=DEFAULT_AVARTHA_BOOT_TIMEOUT_SECONDS,
        gt=0,
    )
    tool_registration_timeout_seconds: float = Field(
        default=DEFAULT_AVARTHA_TOOL_REGISTRATION_TIMEOUT_SECONDS,
        gt=0,
    )
    boot_attempts: int = Field(default=DEFAULT_AVARTHA_BOOT_ATTEMPTS, ge=1)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, base_url: str) -> str:
        """Require the hosted agent's http(s) URL."""

        if not base_url.startswith(("https://", "http://")):
            raise ValueError(
                "base_url must be the hosted agent's http(s) URL, "
                "https://<host>/inference/serverless/pipecat/v1/public/<agent_id>"
            )
        return base_url.rstrip("/")
