"""The one generation-settings shape shared by every factory authoring call."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

FactoryReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class FactoryModelConfig(BaseModel):
    """Explicit authoring settings, separate from the benchmark customer model.

    Reasoning effort is always pinned; there is no unpinned or "none" setting.
    """

    model_config = ConfigDict(extra="forbid")

    model: Annotated[str, Field(min_length=1, description="Authoring generation model")]
    reasoning_effort: Annotated[
        FactoryReasoningEffort,
        Field(description="Explicit reasoning setting for this authoring request"),
    ]

    def provenance_args(self) -> dict[str, str]:
        """Model arguments exactly as recorded in generation provenance."""
        return {"reasoning_effort": self.reasoning_effort}
