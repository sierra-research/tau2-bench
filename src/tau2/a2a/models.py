"""A2A Protocol Data Models for tau2-bench integration."""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class A2AConfig(BaseModel):
    """Configuration bundle for A2A agent connection and behavior."""

    endpoint: str
    auth_token: Optional[str] = None
    timeout: int = 300
    connect_timeout: int = 5
    verify_ssl: bool = True

    def model_post_init(self, __context: Any) -> None:
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

    model_config = ConfigDict(validate_assignment=True)


class AgentCapabilities(BaseModel):
    """Agent capabilities from agent card."""

    streaming: bool = False
    push_notifications: bool = False


class AgentSkill(BaseModel):
    """Agent skill metadata (informational only)."""

    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class AgentCard(BaseModel):
    """Agent capability metadata from /.well-known/agent-card.json."""

    name: str
    url: str
    description: Optional[str] = None
    version: Optional[str] = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    security_schemes: Optional[dict[str, Any]] = None
    security: Optional[List[str]] = None
    skills: Optional[List[AgentSkill]] = None

    model_config = ConfigDict(use_enum_values=True)


class A2AAgentState(BaseModel):
    """Agent execution state for single task evaluation."""

    context_id: Optional[str] = None
    conversation_history: List[Any] = Field(default_factory=list)
    agent_card: Optional[AgentCard] = None
    request_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)
