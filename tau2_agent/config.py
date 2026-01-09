"""
Configuration constants for tau2_agent GCP deployment.

This module defines evaluation limits and server configuration for Cloud Run deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar

# Orchestrator model defaults (internal, not user-provided)
DEFAULT_ORCHESTRATOR_MODEL = "litellm/nebius/Qwen/Qwen3-Coder-480B-A35B-Instruct"
"""Default model for tau2_agent orchestrator (costs borne by tau2-bench-agent)"""

DEFAULT_ORCHESTRATOR_API_BASE = "https://api.studio.nebius.com/v1/"
"""Default API base for Nebius orchestrator model"""


@dataclass
class EvaluationLimits:
    """Limits enforced for Cloud Run deployment.

    These limits ensure evaluations complete within Cloud Run's 60-minute
    request timeout. With ~2 minutes per task, 30 tasks provides a safe margin.
    """

    MAX_TASKS: ClassVar[int] = 30
    MAX_TRIALS: ClassVar[int] = 3
    TIMEOUT_SECONDS: ClassVar[int] = 3600


@dataclass
class ServerConfig:
    """Server configuration loaded from environment.

    Encapsulates all server-side configuration for tau2_agent Cloud Run deployment.
    Values are loaded from environment variables with defaults for local development.

    Attributes:
        tau2_orchestrator_model: Model for orchestrator (internal, not user-provided).
        tau2_orchestrator_api_key: API key for orchestrator model (internal secret).
        tau2_orchestrator_api_base: API base URL for orchestrator model.
        google_api_key: Gemini API key (from Secret Manager in production).
        port: Server port (Cloud Run sets PORT env var).
        log_level: Logging verbosity.
        service_api_keys: Optional list of keys for service access control.
    """

    tau2_orchestrator_model: str = field(
        default_factory=lambda: DEFAULT_ORCHESTRATOR_MODEL
    )
    tau2_orchestrator_api_key: str | None = None
    tau2_orchestrator_api_base: str = field(
        default_factory=lambda: DEFAULT_ORCHESTRATOR_API_BASE
    )
    google_api_key: str | None = None
    port: int = 8001
    log_level: str = "INFO"
    service_api_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Load configuration from environment variables.

        Returns:
            ServerConfig: Configuration instance with values from environment.
        """
        service_keys_str = os.getenv("SERVICE_API_KEYS", "")
        service_keys = [k.strip() for k in service_keys_str.split(",") if k.strip()]

        # Orchestrator model: prefer TAU2_ORCHESTRATOR_MODEL, fall back to TAU2_AGENT_MODEL for backward compat
        orchestrator_model = (
            os.getenv("TAU2_ORCHESTRATOR_MODEL")
            or os.getenv("TAU2_AGENT_MODEL")
            or DEFAULT_ORCHESTRATOR_MODEL
        )

        return cls(
            tau2_orchestrator_model=orchestrator_model,
            tau2_orchestrator_api_key=os.getenv("TAU2_ORCHESTRATOR_API_KEY"),
            tau2_orchestrator_api_base=os.getenv(
                "TAU2_ORCHESTRATOR_API_BASE", DEFAULT_ORCHESTRATOR_API_BASE
            ),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            port=int(os.getenv("PORT", "8001")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            service_api_keys=service_keys,
        )
