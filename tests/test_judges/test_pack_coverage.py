# Copyright Sierra
"""Coverage guard: every registered non-English pack is scorable for nativeness,
and every pack resolves a language-specific DELIVERY rubric (pack or fallback).

These tests are what prevent the "works for some languages but not others"
failure mode: a new pack that forgets its ``nativeness:`` block, or references a
factor outside a closed catalog, fails here (and at load) rather than silently
scoring nothing. Delivery differs deliberately: a pack without a ``delivery:``
block still resolves — to the native-listener FALLBACK rubric, never a generic
one — so the guard asserts the mode, not mere presence.
"""

import hashlib
import json
import re

import pytest

from tau2.judges.delivery.factors import build_delivery_rubric, delivery_factors_for
from tau2.judges.nativeness.checkers import (
    BACKCHANNEL_FREQUENCY_SPECS,
    CHECKER_REGISTRY,
    COUNTING_UNIT_SPECS,
    HONORIFIC_AGREEMENT_MARKERS,
    REGIONAL_CONSISTENCY_SPECS,
)
from tau2.judges.nativeness.factors import (
    DEFAULT_FACTORS,
    PACK_DETERMINISTIC_FACTOR_TYPES,
    PACK_HYBRID_FACTOR_TYPES,
    TEXT_ONLY_FACTORS_BY_LANGUAGE,
    enabled_deterministic_factor_ids_for,
    judge_factors_for,
)
from tau2.multilingual.delivery_catalog import DELIVERY_FACTOR_CATALOG
from tau2.multilingual.nativeness_catalog import (
    FACTOR_QUESTIONS,
    JUDGE_FACTOR_CATALOG,
)
from tau2.multilingual.registry import get_language_pack, list_language_packs
from tau2.multilingual.schema import (
    DeliveryPackConfig,
    DeliveryPackFactorRubric,
    NativenessPackConfig,
    NativenessPackFactorRubric,
)

NON_ENGLISH = [lang for lang in list_language_packs() if lang != "en"]

# The languages whose packs must carry a real ``delivery:`` block; other
# languages legitimately ride the fallback. Only languages with delivery
# phenomena the generic audio-quality taxonomy cannot express carry pack
# delivery factors (tone, pinned regional accent): es/zh were seeded from
# the confirmed human-annotation nuance map
# (docs/multilingual/annotation/nuance_map.json); the 2026-08-20 rubric-parity
# wave pinned the remaining paper languages' regional accents (pt Brazilian,
# hi standard urban Hindustani, ko standard Seoul, zh Mainland Putonghua).
DELIVERY_SEEDED = ["es", "hi", "ko", "pt", "zh"]

REVIEWED_NATIVENESS_FACTOR_IDS = {
    "es": [
        "register_formality",
        "gender_agreement",
        "natural_word_choice",
        "request_politeness",
        "counting_units",
        "name_address_conventions",
    ],
    "pt": [
        "register_formality",
        "regional_consistency",
        "gender_agreement",
        "natural_word_choice",
        "request_politeness",
        "counting_units",
        "name_address_conventions",
    ],
    "hi": [
        "register_formality",
        "regional_consistency",
        "honorific_agreement",
        "gender_agreement",
        "natural_word_choice",
        "request_politeness",
        "counting_units",
        "name_address_conventions",
    ],
    "zh": [
        "register_formality",
        "regional_consistency",
        "counting_units",
        "modal_particles",
        "natural_word_choice",
        "request_politeness",
        "name_address_conventions",
    ],
    "ko": [
        "honorific_levels",
        "honorific_agreement",
        "request_politeness",
        "natural_word_choice",
        "counting_units",
        "name_address_conventions",
    ],
}

ENABLED_NATIVENESS_FACTOR_IDS = {
    "es": {"natural_word_choice", "counting_units"},
    "pt": {
        "natural_word_choice",
        "register_formality",
        "regional_consistency",
        "gender_agreement",
        "counting_units",
    },
    "hi": {"natural_word_choice", "gender_agreement", "name_address_conventions"},
    "ko": {
        "natural_word_choice",
        "honorific_agreement",
        "name_address_conventions",
    },
    "zh": {
        "natural_word_choice",
        "counting_units",
        "modal_particles",
        "name_address_conventions",
    },
}

ENABLED_DETERMINISTIC_FACTOR_IDS = {
    "es": {"email_symbol_verbalization"},
    "pt": {"email_symbol_verbalization"},
    "hi": set(),
    "ko": {"email_symbol_verbalization"},
    "zh": {"email_symbol_verbalization"},
}


def test_there_are_registered_languages():
    assert NON_ENGLISH, "no non-English packs registered — discovery is broken"


def test_every_catalog_factor_has_one_canonical_question():
    assert set(FACTOR_QUESTIONS) == set(JUDGE_FACTOR_CATALOG)
    assert all(question.endswith("?") for question in FACTOR_QUESTIONS.values())


def test_shared_factors_resolve_the_same_base_across_languages():
    for factor_id, languages in {
        "gender_agreement": ("hi", "pt", "es"),
        "honorific_agreement": ("hi", "ko"),
        "natural_word_choice": ("hi", "ko", "zh", "pt", "es"),
    }.items():
        resolved = []
        for language in languages:
            factor = next(f for f in judge_factors_for(language) if f.id == factor_id)
            assert factor.params is not None
            resolved.append(
                (
                    factor.params.question,
                    factor.params.opportunity,
                    factor.params.shared_allowed,
                    factor.params.shared_violation,
                )
            )
        assert len(set(resolved)) == 1


@pytest.mark.parametrize("language", NON_ENGLISH)
def test_pack_has_catalog_judge_factors(language):
    """Each non-English pack resolves >=1 runtime factor from the catalog."""
    factors = judge_factors_for(language)
    enabled = [f for f in factors if f.enabled]
    assert enabled, f"pack '{language}' has no enabled nativeness judge factors"
    for f in factors:
        assert f.type == "judge" or f.type in CHECKER_REGISTRY
        assert (f.params is not None) == (f.type == "judge")
        assert f.id in JUDGE_FACTOR_CATALOG, f"{language}: '{f.id}' not in catalog"
        # category/severity must agree with the catalog (severity may be overridden
        # but defaults to the catalog value).
        assert f.category == JUDGE_FACTOR_CATALOG[f.id].category


def test_english_has_no_judge_factors():
    assert judge_factors_for("en") == []


def test_backchannel_frequency_covers_the_six_reviewed_languages():
    assert set(BACKCHANNEL_FREQUENCY_SPECS) == {"en", "hi", "ko", "zh", "pt", "es"}


def test_reviewed_register_factors_have_exact_deterministic_wiring():
    assert PACK_DETERMINISTIC_FACTOR_TYPES == {
        "es": {"register_formality": "es_register_formality"},
        "hi": {"register_formality": "hi_register_formality"},
        "ko": {"honorific_levels": "ko_honorific_levels"},
        "zh": {"register_formality": "zh_register_formality"},
    }
    for language, by_factor in PACK_DETERMINISTIC_FACTOR_TYPES.items():
        resolved = {factor.id: factor for factor in judge_factors_for(language)}
        for factor_id, checker_type in by_factor.items():
            factor = resolved[factor_id]
            assert checker_type in CHECKER_REGISTRY
            assert factor.type == checker_type
            assert factor.evaluation_level == "deterministic"
            assert factor.params is None

    portuguese = {factor.id: factor for factor in judge_factors_for("pt")}
    assert portuguese["register_formality"].type == "judge"
    assert portuguese["register_formality"].params is not None


def test_reviewed_hybrid_factors_have_exact_wiring():
    assert PACK_HYBRID_FACTOR_TYPES == {
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
    for language, by_factor in PACK_HYBRID_FACTOR_TYPES.items():
        resolved = {factor.id: factor for factor in judge_factors_for(language)}
        for factor_id, checker_type in by_factor.items():
            factor = resolved[factor_id]
            assert checker_type in CHECKER_REGISTRY
            assert factor.type == "judge"
            assert factor.precheck_type == checker_type
            expected_level = (
                "call" if factor_id == "regional_consistency" else "utterance"
            )
            assert factor.evaluation_level == expected_level
            assert factor.params is not None


@pytest.mark.parametrize(
    ("language", "digest", "absorbed"),
    [
        (
            "es",
            "76d1ba0f60c53498d03a2d47f049ee60b29f92ab3df417d270d4dc386e9fa532",
            {"regional_consistency", "translationese", "verb_morphology"},
        ),
        (
            "hi",
            "70a15b9e154db0b4c4448d33e0606df38d00adf99252ddf0ad1014940c7d5a84",
            {"translationese"},
        ),
        (
            "ko",
            "afee1093f88fa350479f19381fcb2aa89bf48aa956e768cb569e0b3ab5ecb179",
            {"translationese"},
        ),
        (
            "pt",
            "6a78ce90053b068f912bf873dce6170447b1e0195a22c97a847b5d44bb32de18",
            {"translationese", "verb_morphology"},
        ),
        (
            "zh",
            "bc4453c7c531dd78e035b863930a111dc1dc11669755f96466c68dc5d0c0dd6a",
            {"translationese"},
        ),
    ],
)
def test_combined_natural_word_choice_is_the_frozen_utterance_judge(
    language, digest, absorbed
):
    """Every paper language ships the exact validated combined criterion."""
    by_id = {factor.id: factor for factor in judge_factors_for(language)}
    combined = by_id["natural_word_choice"]

    assert combined.enabled
    assert combined.type == "judge"
    assert combined.precheck_type is None
    assert combined.evaluation_level == "utterance"
    assert combined.aggregation == "any"
    assert combined.params is not None
    assert combined.params.opportunity == "every non-empty agent utterance"
    assert absorbed.isdisjoint(by_id)

    prompt_fields = {
        "question": combined.params.language_question,
        "native_does": combined.params.language_allowed,
        "ai_likely_does": combined.params.language_violation,
    }
    encoded = json.dumps(
        prompt_fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest


def test_regional_marker_catalog_is_closed_and_provenance_bearing():
    assert set(REGIONAL_CONSISTENCY_SPECS) == {"pt", "zh"}
    for language, spec in REGIONAL_CONSISTENCY_SPECS.items():
        assert spec.language == language
        assert spec.min_occurrences >= 2
        assert spec.min_agent_turns >= 2
        assert len({marker.id for marker in spec.markers}) == len(spec.markers)
        for marker in spec.markers:
            assert marker.provenance
            re.compile(marker.pattern)


def test_counting_unit_catalog_is_closed_and_provenance_bearing():
    assert set(COUNTING_UNIT_SPECS) == {"ko", "zh"}
    for language, spec in COUNTING_UNIT_SPECS.items():
        assert spec.language == language
        assert len({marker.id for marker in spec.markers}) == len(spec.markers)
        for marker in spec.markers:
            assert marker.provenance
            re.compile(marker.pattern)


def test_honorific_agreement_catalog_is_closed_and_provenance_bearing():
    assert set(HONORIFIC_AGREEMENT_MARKERS) == {"hi", "ko"}
    for markers in HONORIFIC_AGREEMENT_MARKERS.values():
        assert len({marker.id for marker in markers}) == len(markers)
        for marker in markers:
            assert marker.provenance
            re.compile(marker.pattern)


def test_every_nativeness_checker_is_referenced_by_factor_configuration():
    configured_types = {factor.type for factor in DEFAULT_FACTORS}
    configured_types.update(
        factor.type
        for factors in TEXT_ONLY_FACTORS_BY_LANGUAGE.values()
        for factor in factors
    )
    configured_types.update(
        checker_type
        for by_factor in PACK_HYBRID_FACTOR_TYPES.values()
        for checker_type in by_factor.values()
    )
    configured_types.update(
        checker_type
        for by_factor in PACK_DETERMINISTIC_FACTOR_TYPES.values()
        for checker_type in by_factor.values()
    )
    assert set(CHECKER_REGISTRY) == configured_types


def test_unknown_factor_id_rejected_at_construction():
    """A pack can never introduce a factor outside the closed catalog."""
    with pytest.raises(ValueError):
        NativenessPackConfig(
            judge_factors=[
                NativenessPackFactorRubric(
                    factor_id="totally_made_up_factor",
                    nuance="x",
                    native_does="x",
                    ai_likely_does="x",
                )
            ]
        )


def test_pack_email_symbols_are_lists():
    """email_symbols, where present, map a symbol to a list of native tokens."""
    for language in NON_ENGLISH:
        pack = get_language_pack(language)
        if pack.nativeness is None:
            continue
        for symbol, tokens in pack.nativeness.email_symbols.items():
            assert isinstance(tokens, list) and tokens, f"{language} {symbol} empty"


# --- delivery rubric coverage -------------------------------------------------


@pytest.mark.parametrize("language", NON_ENGLISH)
def test_every_language_resolves_a_language_specific_delivery_rubric(language):
    """No language is ever judged generically: a pack with delivery factors
    resolves 'pack', every other registered language resolves 'fallback'."""
    rubric = build_delivery_rubric(language)
    assert rubric.source in ("pack", "fallback"), f"{language}: generic rubric"
    assert rubric.language == language
    if rubric.source == "pack":
        assert rubric.enabled_factors, f"{language}: pack mode with no factors"
    else:
        assert rubric.factors == []


@pytest.mark.parametrize("language", DELIVERY_SEEDED)
def test_annotated_language_has_pack_delivery_factors(language):
    """Languages with taxonomy-uncoverable phenomena carry real delivery
    factors, all from the closed catalog with catalog-consistent categories."""
    rubric = build_delivery_rubric(language)
    assert rubric.source == "pack", f"{language} lost its seeded delivery block"
    for f in delivery_factors_for(language):
        assert f.id in DELIVERY_FACTOR_CATALOG, f"{language}: '{f.id}' not in catalog"
        assert f.category == DELIVERY_FACTOR_CATALOG[f.id].category
        assert 1 <= f.severity <= 3
        assert f.listen_for.strip(), f"{language}: '{f.id}' has empty listen_for"


def test_unknown_delivery_factor_id_rejected_at_construction():
    """A pack can never introduce a delivery factor outside the closed catalog."""
    with pytest.raises(ValueError):
        DeliveryPackConfig(
            judge_factors=[
                DeliveryPackFactorRubric(
                    factor_id="totally_made_up_factor",
                    listen_for="x",
                )
            ]
        )


def test_spanish_delivery_rubric_covers_selected_audio_factors():
    """Spanish keeps only the factor the generic audio taxonomy cannot ask:
    consistency with its pinned regional variety."""
    assert {factor.id for factor in delivery_factors_for("es")} == {
        "regional_accent_consistency",
    }


@pytest.mark.parametrize("language", ["es", "hi", "ko", "pt", "zh"])
def test_nativeness_judge_factors_have_explicit_binary_questions(language):
    factors = [
        factor for factor in judge_factors_for(language) if factor.type == "judge"
    ]
    assert factors
    assert all(
        factor.params is not None
        and factor.params.question
        and factor.params.question.rstrip().endswith("?")
        for factor in factors
    )


def test_spanish_delivery_rubric_preserves_regional_locale():
    rubric = build_delivery_rubric("es", "ES-MD")
    assert rubric.locale == "ES-MD"


def test_reviewed_language_pack_factors_are_never_shadowed():
    for language in ("en", "hi", "ko", "zh", "pt", "es"):
        assert all(not factor.shadow for factor in judge_factors_for(language))
        assert all(not factor.shadow for factor in delivery_factors_for(language))


@pytest.mark.parametrize(
    ("language", "expected_ids"), REVIEWED_NATIVENESS_FACTOR_IDS.items()
)
def test_reviewed_packs_emit_the_finalized_nativeness_metrics(language, expected_ids):
    """The shipped pack is the runtime selection seam; guard the finalized set."""
    factors = judge_factors_for(language)
    assert [factor.id for factor in factors] == expected_ids
    assert {factor.id for factor in factors if factor.enabled} == (
        ENABLED_NATIVENESS_FACTOR_IDS[language]
    )
    assert all(not factor.shadow for factor in factors)
    assert all(
        (factor.params is not None) == (factor.type == "judge") for factor in factors
    )


@pytest.mark.parametrize(
    ("language", "expected_ids"), ENABLED_DETERMINISTIC_FACTOR_IDS.items()
)
def test_reviewed_packs_select_exact_deterministic_metrics(language, expected_ids):
    assert enabled_deterministic_factor_ids_for(language) == expected_ids


def test_duplicate_delivery_factor_id_rejected():
    with pytest.raises(ValueError):
        DeliveryPackConfig(
            judge_factors=[
                DeliveryPackFactorRubric(factor_id="tone_meaning_flip", listen_for="a"),
                DeliveryPackFactorRubric(factor_id="tone_meaning_flip", listen_for="b"),
            ]
        )


def test_delivery_rubric_no_language_is_generic():
    """Language-less runs (no pack context at all) stay language-generic."""
    rubric = build_delivery_rubric(None)
    assert rubric.source is None and rubric.factors == []


def test_delivery_rubric_unregistered_language_falls_back():
    """An unregistered language still gets the native-listener fallback (the
    fixed template parameterized on the code), never a silent generic prompt."""
    rubric = build_delivery_rubric("xx")
    assert rubric.source == "fallback"
    assert rubric.language == "xx" and rubric.display_name is None
