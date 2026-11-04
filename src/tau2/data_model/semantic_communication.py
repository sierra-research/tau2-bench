"""Explicit settings and validated replies for communicate-info scoring."""

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tau2.config import (
    DEFAULT_SEMANTIC_COMMUNICATE_MODEL,
    DEFAULT_SEMANTIC_COMMUNICATE_MODEL_ARGS,
)


class SemanticCommunicateSettings(BaseModel):
    """Task communication judge settings, resolved eagerly and independently of the customer."""

    model_config = ConfigDict(extra="forbid")
    model: Annotated[
        str,
        Field(
            default=DEFAULT_SEMANTIC_COMMUNICATE_MODEL,
            min_length=1,
            description="Resolved communicate-info judge model, independent of customer model.",
        ),
    ]
    model_args: Annotated[
        dict[str, JsonValue],
        Field(
            default_factory=lambda: dict(DEFAULT_SEMANTIC_COMMUNICATE_MODEL_ARGS),
            description="Resolved judge arguments; explicit overrides are preserved.",
        ),
    ]
    prompt_version: Annotated[
        Literal["communicate-semantic-v1"],
        Field(
            default="communicate-semantic-v1",
            description="Version of the preserved semantic communication prompt.",
        ),
    ]


class CommunicateScorer(str, Enum):
    """How required communicate_info facts are matched in the agent's messages."""

    SEMANTIC = "semantic"  # pinned LLM judge; language- and paraphrase-tolerant
    SUBSTRING = "substring"  # original case-insensitive substring match


class CommunicateScoring(BaseModel):
    """One explicit communicate-info scorer; semantic judging is the default."""

    model_config = ConfigDict(extra="forbid")
    scorer: Annotated[
        CommunicateScorer,
        Field(
            default=CommunicateScorer.SEMANTIC,
            description="Semantic LLM judge (default) or original substring matching.",
        ),
    ]
    judge: Annotated[
        Optional[SemanticCommunicateSettings],
        Field(
            default=None,
            description="Pinned judge settings; present exactly for the semantic scorer.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def default_judge(cls, value):
        """The semantic scorer resolves its judge eagerly so results record it."""
        if isinstance(value, dict):
            scorer = value.get("scorer", CommunicateScorer.SEMANTIC)
            if (
                CommunicateScorer(scorer) == CommunicateScorer.SEMANTIC
                and value.get("judge") is None
            ):
                value = {**value, "judge": SemanticCommunicateSettings()}
        return value

    @model_validator(mode="after")
    def judge_matches_scorer(self) -> "CommunicateScoring":
        """Substring scoring never carries (or silently ignores) judge settings."""
        if (self.scorer == CommunicateScorer.SEMANTIC) != (self.judge is not None):
            raise ValueError("judge settings apply exactly to the semantic scorer")
        return self


class SemanticCommunicateReply(BaseModel):
    """Pydantic handles known boolean strings; malformed answers raise errors."""

    model_config = ConfigDict(extra="forbid")
    reasoning: Annotated[
        str,
        Field(
            min_length=1,
            description="Judge explanation of whether meaning was conveyed.",
        ),
    ]
    met: Annotated[
        bool,
        Field(
            description="Whether the agent conveyed the expected meaning; string false parses as false."
        ),
    ]
