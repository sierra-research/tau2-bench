# Copyright Sierra
"""Closed catalog of language-independent delivery factor metadata.

Language packs select these ids and supply language-specific listening rubrics.
The pack loader rejects unknown ids, so a pack cannot invent an unreviewed axis.
"""

from functools import partial

from tau2.multilingual.factor_catalog import (
    FactorDefinition,
    catalog_factor,
    factor_definition,
)

DEFAULT_DELIVERY_SEVERITY = 2


_f = partial(factor_definition, default_severity=DEFAULT_DELIVERY_SEVERITY)


# The closed set. Keyed by id. Every pack delivery factor MUST reference one of
# these. Grounded in the Round-1/2/3 human-annotation findings
# (docs/multilingual/annotation/nuance_map.json): pronunciation and audio-level
# delivery was the biggest catalog gap of the transcript-based judge.
#
# Deliberately small: the generic audio-quality taxonomy (fidelity /
# mispronunciation / readout / intonation questions) already covers the
# language-independent delivery surfaces, so the catalog holds only axes that
# taxonomy cannot express — phenomena that exist in some languages and not
# others.
DELIVERY_FACTOR_CATALOG: dict[str, FactorDefinition] = {
    f.id: f
    for f in [
        # --- lexical tone ----------------------------------------------------------
        _f(
            "tone_meaning_flip",
            "tone_meaning",
            "agent speaks tone-bearing words whose tone selects the meaning",
            "In a tonal language, lexical tones are realized correctly on "
            "meaning-bearing words; a wrong or collapsed tone that turns the "
            "word into a different word is a delivery failure.",
            "Gets a word's tone wrong in a way that turns it into a different word",
            default_severity=3,
        ),
        _f(
            "regional_accent_consistency",
            "accent_consistency",
            "the target locale specifies a regional variety and the agent speaks aloud",
            "Pronunciation remains compatible with the target regional variety; "
            "ordinary within-variety variation is acceptable.",
            "Audibly switches between incompatible regional pronunciation systems "
            "or repeatedly uses an accent inconsistent with the target locale",
        ),
    ]
}


def get_delivery_catalog_factor(factor_id: str) -> FactorDefinition:
    """Look up a catalog factor by id, or raise if it is not in the closed set."""
    return catalog_factor(
        DELIVERY_FACTOR_CATALOG,
        factor_id,
        axis="delivery judge",
        module="tau2.multilingual.delivery_catalog",
    )
