# Copyright Sierra
"""Nativeness factor configs: universal defaults + the bridge from pack data.

Language-independent implementations (the deterministic ``DEFAULT_FACTORS``
and ``ENGLISH_SYMBOL_LEAKS``) live here in code; each ``pack.yaml`` selects the
deterministic checks it runs and owns its language-specific judge rubrics and
native email-symbol verbalizations. ``judge_factors_for`` assembles those
rubrics against the closed catalog.

Factor params are TYPED: ``NativenessRubricText`` composes the catalog-owned
shared base with one pack's language-specific rules and examples. Deterministic
checkers take no params — their inputs ride ``CheckerContext``. Hybrid judge
factors retain their rubric and name a deterministic FAIL-only precheck.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from tau2.multilingual.factor_catalog import FactorModality
from tau2.multilingual.nativeness_catalog import (
    get_catalog_factor,
    get_factor_prompt_base,
)

NativenessEvaluationLevel = Literal["call", "utterance", "deterministic"]
NativenessAggregation = Literal["direct", "any", "dose"]


# These factors require comparison across agent turns or a call-level judgment
# about absence/dosage. Every other LLM factor is judged one agent utterance at
# a time and then aggregated to the call.
CALL_LEVEL_FACTOR_IDS = frozenset(
    {
        "register_formality",
        "regional_consistency",
        "honorific_levels",
        "routines",
        "modal_particles",
        "discourse_markers",
    }
)


class NativenessRubricText(BaseModel):
    """One shared factor base plus one language's rules and examples."""

    question: Annotated[
        str, Field(description="Catalog-owned binary question shown to the judge.")
    ]
    opportunity: Annotated[
        str,
        Field(description="Catalog-owned description of when the factor can fire."),
    ]
    shared_allowed: Annotated[
        str, Field(description="Language-independent passing behavior.")
    ]
    shared_violation: Annotated[
        str, Field(description="Language-independent failing behavior.")
    ]
    language_criterion: Annotated[
        str, Field(description="Short language-specific name from the pack.")
    ]
    language_question: Annotated[
        Optional[str],
        Field(description="Optional language-specific clarification of the criterion."),
    ] = None
    language_allowed: Annotated[
        str,
        Field(description="Language-specific rules, exceptions, and good examples."),
    ]
    language_violation: Annotated[
        str,
        Field(description="Language-specific failing rules and concrete bad examples."),
    ]


class NativenessFactorConfig(BaseModel):
    """One nativeness factor: which checker to run, and how to weight it."""

    id: Annotated[str, Field(description="Stable identifier, used as the metric key.")]
    category: Annotated[
        str, Field(description="Universal axis, e.g. 'numbers', 'symbols'.")
    ]
    type: Annotated[
        str,
        Field(
            description="Checker key in CHECKER_REGISTRY, or 'judge' for an "
            "LLM-judge factor."
        ),
    ]
    severity: Annotated[
        int,
        Field(
            description="1 (subtle) .. 3 (breaks the illusion). Weights the score.",
            ge=1,
            le=3,
        ),
    ]
    precheck_type: Annotated[
        Optional[str],
        Field(
            description="Optional CHECKER_REGISTRY key for a deterministic, "
            "FAIL-only precheck before this factor's LLM fallback."
        ),
    ] = None
    params: Annotated[
        Optional[NativenessRubricText],
        Field(
            description="Per-language rubric text for judge factors; None for "
            "deterministic checkers (their inputs ride CheckerContext)."
        ),
    ] = None
    enabled: bool = True
    shadow: Annotated[
        bool,
        Field(
            description="Calibration mode: judged and recorded on every "
            "simulation but excluded from the aggregate score and coverage."
        ),
    ] = False
    evaluation_level: NativenessEvaluationLevel = Field(
        default="deterministic",
        description="Text unit supplied to this factor's evaluator.",
    )
    aggregation: NativenessAggregation = Field(
        default="direct",
        description="How utterance verdicts become one call-level verdict.",
    )
    modality: FactorModality = Field(
        default=FactorModality.BOTH,
        description="Run modality the factor applies to (voice | text | "
        "both). The harness filters by the run's mode: a voice-only factor "
        "never scores a text transcript and vice versa. Catalog factors "
        "inherit their catalog modality; deterministic factors set it here.",
    )


# The deterministic factors available to applicable runs. Language packs
# select which of these are active; English retains both without a nativeness
# pack. Both are
# VOICE-only phenomena: backchannels are a floor-holding behavior half-duplex
# chat has no channel for, and native symbol verbalization ('arroba',
# 'punto') only exists when values are SPOKEN — typed emails are literal.
DEFAULT_FACTORS: list[NativenessFactorConfig] = [
    NativenessFactorConfig(
        id="backchannel_frequency",
        category="discourse",
        type="backchannel_frequency",
        severity=2,
        modality=FactorModality.VOICE,
    ),
    NativenessFactorConfig(
        id="email_symbol_verbalization",
        category="symbols",
        type="email_symbol_verbalization",
        severity=2,
        modality=FactorModality.VOICE,
    ),
]

# Deterministic factors that apply only to text-mode runs of one language
# (modality="text"). Voice transcripts may normalize script and therefore
# cannot support a script verdict, even when the spoken wording is native.
TEXT_ONLY_FACTORS_BY_LANGUAGE: dict[str, list[NativenessFactorConfig]] = {
    "zh": [
        NativenessFactorConfig(
            id="script_consistency",
            category="script",
            type="zh_script_consistency",
            severity=2,
            modality=FactorModality.TEXT,
        )
    ]
}

# Pack-authored factors whose reviewed rubric is precise enough to execute
# deterministically. The pack remains the source of factor order, severity,
# enabled/shadow state, and stable metric id; this table selects only the
# runtime checker. Portuguese deliberately remains on the LLM judge because
# natural Brazilian address systems cannot be captured by this lexical split.
PACK_DETERMINISTIC_FACTOR_TYPES: dict[str, dict[str, str]] = {
    "es": {"register_formality": "es_register_formality"},
    "hi": {"register_formality": "hi_register_formality"},
    "ko": {"honorific_levels": "ko_honorific_levels"},
    "zh": {"register_formality": "zh_register_formality"},
}

# Precision-first hybrid factors. These remain ordinary LLM factors for every
# unresolved case; a checker may short-circuit only an unambiguous FAIL.
PACK_HYBRID_FACTOR_TYPES: dict[str, dict[str, str]] = {
    "es": {
        "gender_agreement": "es_gender_agreement",
    },
    "hi": {
        "gender_agreement": "hi_gender_agreement",
        "honorific_agreement": "hi_honorific_agreement",
    },
    "ko": {
        "counting_units": "ko_counting_units",
        "honorific_agreement": "ko_honorific_agreement",
        "name_address_conventions": "ko_name_address_conventions",
    },
    "pt": {
        "gender_agreement": "pt_gender_agreement",
        "regional_consistency": "pt_regional_consistency",
    },
    "zh": {
        "counting_units": "zh_counting_units",
        "regional_consistency": "zh_regional_consistency",
    },
}


def judge_factors_for(language: Optional[str]) -> list[NativenessFactorConfig]:
    """Pack factors for a language, assembled from its pack + the catalog.

    Reads the pack's ``nativeness.judge_factors`` rubrics (per-language text) and
    fills in the universal metadata (category / severity) from the
    closed catalog. Returns an empty list for English, an unregistered language,
    or a pack that declares no nativeness factors — never an error. Disabled
    rubrics are still returned (with ``enabled=False``), but the harness does not
    schedule or record them.
    """
    # Lazy import: tau2.multilingual.registry pulls in the pack schema, which
    # lazily references this package — importing at call time avoids any cycle.
    from tau2.multilingual.registry import get_language_pack

    language_code = (language or "").lower()
    pack = get_language_pack(language_code)
    if pack is None or pack.nativeness is None:
        return []
    factors: list[NativenessFactorConfig] = []
    for rubric in pack.nativeness.judge_factors:
        catalog = get_catalog_factor(rubric.factor_id)
        checker_type = PACK_DETERMINISTIC_FACTOR_TYPES.get(language_code, {}).get(
            rubric.factor_id
        )
        precheck_type = PACK_HYBRID_FACTOR_TYPES.get(language_code, {}).get(
            rubric.factor_id
        )
        base = None if checker_type else get_factor_prompt_base(rubric.factor_id)
        factors.append(
            NativenessFactorConfig(
                id=rubric.factor_id,
                category=catalog.category,
                type=checker_type or "judge",
                precheck_type=precheck_type,
                severity=rubric.severity or catalog.default_severity,
                params=(
                    None
                    if base is None
                    else NativenessRubricText(
                        question=base.question,
                        opportunity=base.opportunity,
                        shared_allowed=base.allowed,
                        shared_violation=base.violation,
                        language_criterion=rubric.nuance,
                        language_question=rubric.question,
                        language_allowed=rubric.native_does,
                        language_violation=rubric.ai_likely_does,
                    )
                ),
                enabled=rubric.enabled,
                shadow=rubric.shadow,
                evaluation_level=(
                    "deterministic"
                    if checker_type
                    else "call"
                    if rubric.factor_id in CALL_LEVEL_FACTOR_IDS
                    else "utterance"
                ),
                aggregation=(
                    "direct"
                    if checker_type or rubric.factor_id in CALL_LEVEL_FACTOR_IDS
                    else "dose"
                    if rubric.dose
                    else "any"
                ),
                modality=catalog.modality,
            )
        )
    return factors


def email_symbols_for(language: Optional[str]) -> dict[str, list[str]]:
    """Native email-symbol verbalizations for a language (empty if none/unknown)."""
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack((language or "").lower())
    if pack is None or pack.nativeness is None:
        return {}
    return pack.nativeness.email_symbols


def enabled_deterministic_factor_ids_for(language: Optional[str]) -> set[str]:
    """Pack-selected deterministic factor ids for a language.

    English has no nativeness pack and retains the universal deterministic
    diagnostics. Registered non-English packs opt in explicitly so a retired
    checker can remain implemented without continuing to run.
    """
    from tau2.multilingual.registry import get_language_pack

    language_code = (language or "").lower()
    pack = get_language_pack(language_code)
    if pack is None or pack.nativeness is None:
        return (
            {factor.id for factor in DEFAULT_FACTORS}
            if language_code == "en"
            else set()
        )
    return set(pack.nativeness.enabled_deterministic_factors)


# English symbol-words that signal an English readout. Language-agnostic.
# A language whose accept-list (pack email_symbols) contains one of these (e.g.
# German "at") will NOT be flagged for it — it is native there.
ENGLISH_SYMBOL_LEAKS: dict[str, list[str]] = {
    "@": ["at", "at the rate"],
    ".": ["dot"],
}
