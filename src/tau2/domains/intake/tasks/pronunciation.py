# Copyright Sierra
"""Closed pronunciation metadata catalogs for the checked-in entity banks.

The released task generator consumes reviewed pronunciations directly from
the banks. It does not synthesize or invent pronunciation variants.
"""

from enum import Enum


class DistortionOperator(str, Enum):
    """Closed catalog of mispronunciation operators."""

    METATHESIS = "metathesis"  # transpose adjacent phonemes in a syllable
    VOWEL_SWAP = "vowel_swap"  # substitute the stressed vowel
    SYLLABLE_DROP = "syllable_drop"  # drop an unstressed medial syllable
    SPELLING_PRONUNCIATION = "spelling_pronunciation"


_NAME_LIKE_OPERATORS = frozenset(
    {
        DistortionOperator.METATHESIS,
        DistortionOperator.VOWEL_SWAP,
        DistortionOperator.SPELLING_PRONUNCIATION,
    }
)


OPERATOR_APPLICABILITY: dict[str, frozenset[DistortionOperator]] = {
    "person_names": _NAME_LIKE_OPERATORS,
    "medications": frozenset(
        {
            DistortionOperator.METATHESIS,
            DistortionOperator.VOWEL_SWAP,
            DistortionOperator.SYLLABLE_DROP,
            DistortionOperator.SPELLING_PRONUNCIATION,
        }
    ),
    "properties": _NAME_LIKE_OPERATORS,
    "vehicles": _NAME_LIKE_OPERATORS,
}


NO_VARIANT_TOKENS: dict[str, frozenset[str]] = {
    "person_names": frozenset(),
    "medications": frozenset({"Valsartan", "Losartan"}),
    # Gasthof (GAHST-hohf) and Harom (HAH-rohm) are vowel-swap-only tokens
    # whose swapped natural rendering IS the token's own spelling
    # ("gasthof", "harom") — deliberately columned without a variant.
    "properties": frozenset({"Gasthof", "Harom"}),
    "vehicles": frozenset(),
}
