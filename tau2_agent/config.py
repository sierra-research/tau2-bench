"""
Configuration constants for tau2_agent GCP deployment.

This module defines evaluation limits and server configuration for Cloud Run deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class EvaluationLimits:
    """Limits enforced for Cloud Run deployment.

    These limits ensure evaluations complete within Cloud Run's 60-minute
    request timeout. With ~2 minutes per task, 30 tasks provides a safe margin.
    """

    MAX_TASKS: ClassVar[int] = 30
    """Maximum tasks per evaluation (Cloud Run 60-min timeout)"""

    MAX_TRIALS: ClassVar[int] = 3
    """Maximum trials per task"""

    TIMEOUT_SECONDS: ClassVar[int] = 3600
    """Cloud Run request timeout (60 minutes)"""


@dataclass
class ServerConfig:
    """Server configuration loaded from environment.

    This dataclass encapsulates all server-side configuration for the tau2_agent
    Cloud Run deployment. Values are loaded from environment variables with sensible
    defaults for local development.

    Attributes:
        tau2_agent_model: Model for tau2_agent orchestrator LLM.
        google_api_key: Gemini API key (from Secret Manager in production).
        port: Server port (Cloud Run sets PORT env var).
        log_level: Logging verbosity.
        service_api_keys: Optional list of keys for service access control.
    """

    tau2_agent_model: str = "gemini-2.0-flash"
    """Model for tau2_agent orchestrator LLM"""

    google_api_key: str | None = None
    """Gemini API key (from Secret Manager in production)"""

    port: int = 8001
    """Server port (Cloud Run sets PORT env var)"""

    log_level: str = "INFO"
    """Logging verbosity"""

    service_api_keys: list[str] = field(default_factory=list)
    """Optional: list of keys for service access control"""

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Load configuration from environment variables.

        Returns:
            ServerConfig: Configuration instance with values from environment.
        """
        service_keys_str = os.getenv("SERVICE_API_KEYS", "")
        service_keys = [k.strip() for k in service_keys_str.split(",") if k.strip()]

        return cls(
            tau2_agent_model=os.getenv("TAU2_AGENT_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            port=int(os.getenv("PORT", "8001")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            service_api_keys=service_keys,
        )
