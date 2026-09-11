# Copyright Sierra
"""Controlled vocabulary for persona tags.

Tags are structured metadata on language-pack personas (see
``MultilingualPersonaConfig.tags``) so results can be sliced by persona
properties — code-switching density, formality, age band — without parsing
the freeform pragmatics prose. The vocabulary is deliberately closed: every
tag key must be a dimension below and every value must come from that
dimension's list, so analyses never fragment on spelling variants. There is no
``environment`` tag: acoustic environment is derived from the persona's
``acoustic_preset_id``, not tagged.

Validation happens in two layers:
- schema level (``tau2.multilingual.schema``): any tag PROVIDED must be valid;
- factory level (``tau2.multilingual.factory.guardrails``): every persona must
  carry ALL of :data:`REQUIRED_TAG_DIMENSIONS`. Completeness is enforced at the
  factory level, not the schema level.
"""

# Dimension -> allowed values. Extend deliberately; never rename values in
# place (persisted runs join on them).
TAG_VOCABULARY: dict[str, list[str]] = {
    # How much English is mixed into the persona's matrix language.
    "code_switch": ["none", "low", "medium", "high"],
    # Politeness/formality register toward the agent.
    "formality": ["low", "medium", "high"],
    # How well the persona copes with an English-only agent.
    "english_tolerance": ["low", "medium", "high"],
    # Rough age band of the speaker.
    "age_band": ["20s", "30s", "40s", "50s", "60s_plus"],
    # Social/occupational register. OPTIONAL (not in REQUIRED_TAG_DIMENSIONS)
    # analysis-slicing metadata, so the drafter omits it when nothing fits.
    # APPEND new values only; never rename/remove (persisted runs join on them).
    "register": [
        "urban_professional",
        "homemaker",
        "small_business",
        "student",
        "retiree",
        "rural",
        "gig_worker",
        "service_worker",
        "manual_trade",
    ],
    # Whether the persona's goodbyes are brisk or ritually extended.
    "closing_style": ["brisk", "extended"],
    # Speaker gender (drives TTS voice + the one-male/one-female pack invariant
    # enforced by tau2.multilingual.factory.voice_generation). REQUIRED: the
    # first-person gender-agreement clause in schema.to_guidelines_text fires
    # only when tags.gender is male/female.
    "gender": ["male", "female"],
}

# Dimensions the factory guardrails require on EVERY persona (completeness is
# a factory-level requirement, not a schema-level one).
REQUIRED_TAG_DIMENSIONS: list[str] = [
    "code_switch",
    "formality",
    "english_tolerance",
    "age_band",
    "gender",
]


def validate_tags(tags: dict[str, str]) -> list[str]:
    """Problems with the PROVIDED tags (empty == valid).

    Checks only that what is present is in-vocabulary; completeness against
    :data:`REQUIRED_TAG_DIMENSIONS` is the factory guardrails' job.
    """
    problems: list[str] = []
    for key, value in tags.items():
        if key not in TAG_VOCABULARY:
            problems.append(
                f"unknown tag dimension '{key}' (known: {sorted(TAG_VOCABULARY)})"
            )
        elif value not in TAG_VOCABULARY[key]:
            problems.append(
                f"tag '{key}' has invalid value '{value}' "
                f"(allowed: {TAG_VOCABULARY[key]})"
            )
    return problems
