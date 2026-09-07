# Copyright Sierra
"""The intake run's pre-synthesis pronunciation substitution map (design doc
docs/designs/intake-mispronunciation.md sec. 3 + 8).

:func:`build_pronunciation_map` assembles the ``token -> spoken form`` map the
voice USER simulator applies to its synthesis text (via
``VoiceSettings.pronunciation_map`` and
:func:`tau2.voice.utils.pronunciation_swap.apply_pronunciations`).

**Default reading always** (owner decision after the 2026-08-26 audition
listen: the provider's default reading beat the curated respellings almost
everywhere, and hyphenated caps respellings misrender — short caps syllables
read as initialisms): an uncomplicated run gets an EMPTY map, so every value
is synthesized from its gold spelling. The banks' correct ``respelling``
columns and the oddball token table stay as reviewed reference data (they
still define the canonical readings and drive the mispronunciation
operators) but are never swapped in.

**`mispronounced_term` active**: the map contains exactly the drawn term,
mapped to its curated bad variant. The variant is stored TTS-ready in the
banks — natural orthography ("iburprofen", lowercase plain letters, via
:func:`tau2.domains.intake.tasks.pronunciation.render_natural`), because
the hyphenated caps-stress respelling notation misrenders (initialisms,
letter-by-letter spell-outs). The swap repeats on every utterance, so the
mispronunciation is consistent for free.

Built once per simulation at orchestrator-build time (the runner's
domain-keyed dispatch in :mod:`tau2.runner.complications` calls this for
intake voice runs only — non-intake domains get no map and synthesis is
identity). Everything here is the banks' curated data verbatim; nothing is
invented at run time.
"""

from functools import lru_cache

from tau2.data_model.simulation import SampledComplication
from tau2.domains.intake.complications import (
    MISPRONOUNCED_TERM_BANKS,
    ComplicationKind,
)
from tau2.domains.intake.tasks.banks import Banks, load_banks


@lru_cache(maxsize=1)
def _mispronounceable_tokens() -> frozenset[str]:
    """Every bank token that carries a curated mispronounced variant.

    Cached: pure function of the checked-in banks; used to fail loud when a
    draw references a token the banks no longer carry (catalog/bank drift).
    """
    banks: Banks = load_banks()
    tokens: set[str] = set()
    for bank_name in sorted(MISPRONOUNCED_TERM_BANKS):
        for entry in getattr(banks, bank_name):
            for pronunciation in entry.pronunciations or []:
                if pronunciation.mispronounced is not None:
                    tokens.add(pronunciation.token)
    return frozenset(tokens)


def build_pronunciation_map(
    complication: SampledComplication | None,
) -> dict[str, str]:
    """The substitution map for one intake voice simulation.

    Empty (identity synthesis, default TTS readings) unless the simulation
    drew ``mispronounced_term`` — then exactly the drawn term maps to its
    curated bad respelling, TTS-rendered. Returns a fresh dict — callers may
    hand it to per-simulation settings without aliasing anything cached.
    """
    if (
        complication is None
        or complication.kind != ComplicationKind.MISPRONOUNCED_TERM.value
    ):
        return {}
    term = complication.params["term"]
    if term not in _mispronounceable_tokens():
        raise ValueError(
            f"mispronounced_term drew {term!r}, which carries no mispronounced "
            "variant in the checked-in banks — bank/catalog drift"
        )
    if not complication.mispronunciation:
        raise ValueError(
            f"mispronounced_term draw for {term!r} carries no mispronunciation"
        )
    return {term: complication.mispronunciation}
