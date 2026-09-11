# Copyright Sierra
"""Shared contract for the closed nativeness and delivery factor catalogs."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FactorModality(str, Enum):
    """Which run modality a factor's opportunity profile exists in.

    ``both`` is the conservative default: grammar, lexicon, and politeness
    factors apply to typed chat exactly as to speech. ``voice`` marks factors
    whose rubric polarity is speech-framed (spoken discourse markers,
    spoken-variety diglossia) — judging them on text produces false FAILs,
    not signal. ``text`` marks script/orthography factors that only typed
    text supports (voice transcripts normalize script). Per-factor reasoning:
    docs/designs/text-mode-multilingual.md, Item 3.
    """

    VOICE = "voice"
    TEXT = "text"
    BOTH = "both"

    def applies(self, *, is_voice: bool) -> bool:
        """Whether a factor of this modality fires in a run of this mode."""
        if self is FactorModality.BOTH:
            return True
        return self is (FactorModality.VOICE if is_voice else FactorModality.TEXT)


class FactorDefinition(BaseModel):
    """Language-independent metadata for one judge or delivery factor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable catalog id referenced by language packs.")
    category: str = Field(description="Universal factor category.")
    description: str = Field(
        description="Language-independent explanation of the factor."
    )
    annotator_label: str = Field(
        description="Plain-language violation phrasing shown to annotators."
    )
    default_severity: int = Field(
        ge=1,
        le=3,
        description="Default severity from 1 (subtle) to 3 (severe).",
    )
    opportunity: str = Field(
        description="Documentary note describing when the factor can fire."
    )
    modality: FactorModality = Field(
        default=FactorModality.BOTH,
        description="Run modality the factor applies to (voice | text | "
        "both). The nativeness harness filters factors by the run's mode.",
    )


def factor_definition(
    id: str,
    category: str,
    opportunity: str,
    description: str,
    annotator_label: str,
    *,
    default_severity: int,
    modality: FactorModality = FactorModality.BOTH,
) -> FactorDefinition:
    """Build a catalog entry while keeping the tables compact and readable."""
    return FactorDefinition(
        id=id,
        category=category,
        opportunity=opportunity,
        description=description,
        annotator_label=annotator_label,
        default_severity=default_severity,
        modality=modality,
    )


def catalog_factor(
    catalog: dict[str, FactorDefinition],
    factor_id: str,
    *,
    axis: str,
    module: str,
) -> FactorDefinition:
    """Look up a closed-catalog factor with one consistent error contract."""
    try:
        return catalog[factor_id]
    except KeyError:
        raise KeyError(
            f"Unknown {axis} factor '{factor_id}'. It must be one of the "
            f"catalog ids in {module}: {sorted(catalog)}"
        ) from None
