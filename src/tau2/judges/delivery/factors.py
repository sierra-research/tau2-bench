# Copyright Sierra
"""Delivery factor configs: the bridge from pack data to the audio judge.

Per-language delivery rubrics live in each ``pack.yaml`` (``delivery:`` block)
and are assembled against the closed catalog
(``tau2.multilingual.delivery_catalog``) by ``build_delivery_rubric``. Unlike
nativeness (where a pack without factors is simply not scored), the delivery
judge is language-specific for EVERY language: a pack with no ``delivery:``
block yields a ``fallback`` rubric — the fixed native-listener instruction in
``tau2.judges.delivery.judge``, parameterized only on the language name/code.

The rubric's ``source`` ("pack" / "fallback" / None for language-less runs) is
recorded on ``DeliveryInfo`` so results always say where the judge's criteria
came from.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from tau2.multilingual.delivery_catalog import get_delivery_catalog_factor

DeliveryRubricSource = Literal["pack", "fallback"]


class DeliveryFactorConfig(BaseModel):
    """One delivery judge factor: pack rubric + universal catalog metadata."""

    id: Annotated[str, Field(description="Catalog factor id, used as the metric key.")]
    category: Annotated[
        str,
        Field(description="Universal audio-defect bucket, e.g. 'tone_meaning'."),
    ]
    severity: Annotated[
        int,
        Field(
            description="1 (subtle) .. 3 (breaks the illusion). Weights the score.",
            ge=1,
            le=3,
        ),
    ]
    listen_for: Annotated[
        str,
        Field(
            description="Language-specific listening instruction for the judge "
            "(from the pack)."
        ),
    ]
    enabled: bool = True
    shadow: bool = Field(
        default=False,
        description="Calibration mode: recorded but excluded from factor scoring.",
    )


class DeliveryRubric(BaseModel):
    """The per-language rubric handed to the delivery judge for one run.

    ``source`` distinguishes the two prompt modes (and is stamped onto the
    result as provenance):
    - "pack"     — the pack defines delivery factors; they are rendered as the
                   rubric and scored per-factor.
    - "fallback" — no pack factors; the fixed native-listener instruction is
                   rendered for the language (no per-factor scoring).
    - None       — no language on the run; the judge stays language-generic.
    """

    language: Annotated[
        Optional[str], Field(description="ISO 639-1 language code, lowercased.")
    ] = None
    display_name: Annotated[
        Optional[str],
        Field(description="Human-readable language name (from the pack, if any)."),
    ] = None
    locale: Annotated[
        Optional[str],
        Field(description="Configured regional locale, e.g. ES-MD, when known."),
    ] = None
    source: Annotated[
        Optional[DeliveryRubricSource],
        Field(description="'pack' / 'fallback', or None for language-less runs."),
    ] = None
    factors: Annotated[
        list[DeliveryFactorConfig],
        Field(
            default_factory=list,
            description="Factors to score (pack mode only; empty in fallback).",
        ),
    ]

    @property
    def enabled_factors(self) -> list[DeliveryFactorConfig]:
        """The factors the judge actually renders and scores."""
        return [f for f in self.factors if f.enabled]


def delivery_factors_for(language: Optional[str]) -> list[DeliveryFactorConfig]:
    """Delivery factors for a language, assembled from its pack + the catalog.

    Reads the pack's ``delivery.judge_factors`` rubrics (per-language text) and
    fills in the universal metadata (category / severity) from the
    closed catalog. Returns an empty list for an unregistered language or a pack
    that declares no delivery factors — never an error. Disabled rubrics are
    still returned (with ``enabled=False``); the judge renders and scores only
    enabled ones.
    """
    # Lazy import: tau2.multilingual.registry pulls in the pack schema, which
    # lazily references this package — importing at call time avoids any cycle.
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack((language or "").lower())
    if pack is None or pack.delivery is None:
        return []
    factors: list[DeliveryFactorConfig] = []
    for rubric in pack.delivery.judge_factors:
        catalog = get_delivery_catalog_factor(rubric.factor_id)
        factors.append(
            DeliveryFactorConfig(
                id=rubric.factor_id,
                category=catalog.category,
                severity=rubric.severity or catalog.default_severity,
                listen_for=rubric.listen_for,
                enabled=rubric.enabled,
                shadow=rubric.shadow,
            )
        )
    return factors


def build_delivery_rubric(
    language: Optional[str], locale: Optional[str] = None
) -> DeliveryRubric:
    """The delivery rubric for a language: pack factors, or the fallback.

    - Registered pack with enabled ``delivery`` factors → ``source="pack"``.
    - Any other language (no pack, no ``delivery:`` block, all factors
      disabled) → ``source="fallback"`` — the judge still gets a
      language-specific (native-listener) instruction, never a generic one.
    - No language at all → ``source=None`` (language-generic judging).
    """
    lang = (language or "").strip().lower()
    if not lang:
        return DeliveryRubric()

    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(lang)
    display_name = pack.display_name if pack is not None else None
    factors = delivery_factors_for(lang)
    if any(f.enabled for f in factors):
        return DeliveryRubric(
            language=lang,
            display_name=display_name,
            locale=locale,
            source="pack",
            factors=factors,
        )
    return DeliveryRubric(
        language=lang,
        display_name=display_name,
        locale=locale,
        source="fallback",
    )
