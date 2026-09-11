# Copyright Sierra
"""Tolerant matching for identifiers and names the benchmark says out loud.

The voice benchmark round-trips every lookup key through speech: the caller
pronounces a name or an id, the agent transcribes it, and the agent's
transcription is what reaches the domain tool. Transcription is faithful to
the *sound*, not to the DB's byte string — a Spanish caller saying "Álvaro
Fernández" is transcribed with its diacritics whether or not the record
carries them, and a spelled-out user id comes back mis-cased. An exact string
comparison rejects both as "not found", manufacturing an agent failure out of
an orthography difference.

:func:`fold_for_match` is the single normalization those lookups compare
through: case-folded and diacritic-folded, nothing else. It does NOT
transliterate, collapse internal whitespace, or drop punctuation — folding
beyond accents would start merging genuinely distinct records.
"""

import unicodedata


def fold_for_match(text: str) -> str:
    """Case- and diacritic-folded form of ``text`` for tolerant lookups.

    Surrounding whitespace is stripped, the string is decomposed with Unicode
    NFD and its combining marks dropped (``Álvaro Fernández`` -> ``Alvaro
    Fernandez``), then case-folded. Marks are stripped *before* case folding so
    that case folding cannot reintroduce one (``İ`` case-folds to ``i`` plus a
    combining dot).

    Only use this where a fold cannot merge two distinct records — the caller
    is responsible for knowing its key space is unique modulo case and accents.
    """
    decomposed = unicodedata.normalize("NFD", text.strip())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
