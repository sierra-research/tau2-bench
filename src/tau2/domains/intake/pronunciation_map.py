# Copyright Sierra
"""The intake caller's pre-synthesis pronunciation substitution map.

:func:`build_pronunciation_map` assembles the ``token -> spoken form`` map the
voice USER simulator applies to its synthesis text (via
``VoiceSettings.pronunciation_map`` and
:func:`tau2.voice.utils.pronunciation_swap.apply_pronunciations`).

An uncomplicated run gets an empty map, so every value is synthesized from
its gold spelling using the provider's default reading. The banks' reviewed
``respelling`` columns remain reference metadata and are not substituted.

When ``mispronounced_term`` is active, the map contains exactly the drawn
term and its curated bad variant. The bank stores that variant as natural
orthography ready for TTS (for example, "iburprofen"). This avoids the
initialisms and letter-by-letter readings that caps-stress notation can
produce. The same substitution applies to every utterance in the call.

Built once per simulation by the runner for elicitation voice users only.
Other domains receive no map and retain the original synthesis text. Every
substitution comes from the banks' reviewed data; nothing is invented at runtime.
"""

from functools import lru_cache

from tau2.data_model.elicitation import SampledComplication
from tau2.domains.intake.complications import (
    MISPRONOUNCED_TERM_BANKS,
    ComplicationKind,
)
from tau2.domains.intake.localized import SelectedStimulus
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
    selected: SelectedStimulus | None = None,
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
    term = complication.params.term
    if selected is not None:
        matches = [
            p
            for p in selected.pronunciations
            if p.token == term and p.distorted == complication.mispronunciation
        ]
        if len(matches) != 1:
            raise ValueError(
                "Pronunciation stimulus differs from the reviewed selected entity"
            )
        return {term: complication.mispronunciation}
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
