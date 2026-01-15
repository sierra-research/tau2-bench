"""A2A Protocol Data Models for tau2-bench integration."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class A2AConfig:
    """Configuration bundle for A2A agent connection and behavior."""

    endpoint: str
    auth_token: str | None = None
    timeout: int = 300
    connect_timeout: int = 5
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        # Normalize endpoint (remove trailing slash)
        self.endpoint = self.endpoint.rstrip("/")

        # Validate timeouts
        if self.timeout <= 0:
            msg = f"timeout must be positive, got {self.timeout}"
            raise ValueError(msg)

        if self.connect_timeout <= 0:
            msg = f"connect_timeout must be positive, got {self.connect_timeout}"
            raise ValueError(msg)

        # Validate URL scheme
        if not self.endpoint.startswith(("http://", "https://")):
            msg = f"endpoint must start with http:// or https://, got {self.endpoint}"
            raise ValueError(msg)


class AgentCapabilities(BaseModel):
    """Agent capabilities from agent card."""

    streaming: bool = False
    push_notifications: bool = False


class AgentSkill(BaseModel):
    """Agent skill metadata (informational only)."""

    id: str
    name: str
    description: str | None = None
    tags: list[str] | None = None


class AgentCard(BaseModel):
    """Agent capability metadata from /.well-known/agent-card.json."""

    name: str
    url: str
    description: str | None = None
    version: str | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    security_schemes: dict[str, Any] | None = None
    security: list[str] | None = None
    skills: list[AgentSkill] | None = None

    model_config = ConfigDict(use_enum_values=True)


@dataclass
class A2AAgentState:
    """Agent execution state for single task evaluation."""

    context_id: str | None = None
    conversation_history: list[Any] = field(default_factory=list)
    agent_card: AgentCard | None = None
    request_count: int = 0
