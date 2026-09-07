# Copyright Sierra
"""Pre-synthesis pronunciation substitution (intake pronunciation program,
design doc docs/designs/intake-mispronunciation.md sec. 3 + 8).

:func:`apply_pronunciations` rewrites pronunciation-bearing tokens of a
text into their pinned respellings right before TTS. It is a pure function
over ``(text, mapping)`` — deterministic, no I/O, no domain knowledge; the
mapping is built per run from the active domain's banks (see
``tau2.domains.intake.pronunciation_map``) and handed to the voice user
simulator via ``VoiceSettings.pronunciation_map``. The stored transcript is
never touched: callers apply this to the synthesis text only.

Matching semantics (fixed, unit-tested):

- **Whole token**: matches are anchored at word boundaries, so spell-outs
  never match (letter-by-letter sequences like ``N-a-r-b-u-t`` contain no
  whole token) while tokens embedded in emails do (``wendee.narbut@...`` —
  ``.`` is a word boundary).
- **Case**: ALL-CAPS tokens (letters only, length >= 2, e.g. ``AS``, ``IT``,
  ``PPO``) match case-SENSITIVELY — several are common English words in
  lowercase and must never swap in running prose. Every other token matches
  case-insensitively (a surname lowercased into an email local part still
  swaps), via a scoped ``(?i:...)`` group per token.
- **Literal tokens**: tokens are regex-escaped (``B&B`` is a literal match;
  boundary assertions degrade to non-word lookarounds when a token edge is
  itself a non-word character).
- **One pass, longest first**: all tokens compile into a SINGLE alternation
  ordered longest-first, so replacements never cascade into each other
  (a respelling is never re-scanned) and overlapping tokens resolve
  deterministically.
"""

import re
from functools import lru_cache
from typing import Annotated, Optional

from pydantic import Field

from tau2.utils.pydantic_utils import BaseModelNoExtra

_ALL_CAPS = re.compile(r"[A-Z]{2,}")
_WORD_CHAR = re.compile(r"\w")


class SwapEvent(BaseModelNoExtra):
    """One token swapped during a pre-synthesis pronunciation pass."""

    token: Annotated[
        str, Field(description="The matched text as it appeared in the input.")
    ]
    replacement: Annotated[
        str, Field(description="The respelling the occurrences were replaced with.")
    ]
    count: Annotated[
        int, Field(description="How many occurrences were replaced in this text.")
    ]


def matches_case_sensitively(token: str) -> bool:
    """Whether a mapping token matches case-sensitively (the fixed rule:
    ALL-CAPS letters-only tokens do — ``AS``/``IT``/``ID`` are common English
    words in lowercase and must never swap in running prose)."""
    return _ALL_CAPS.fullmatch(token) is not None


def _token_pattern(token: str) -> str:
    """The anchored, escaped, case-scoped pattern for one mapping token.

    ``\\b`` is only a boundary next to a word character, so a token edge that
    is itself a non-word character (defensive; no current token has one) gets
    an explicit non-word lookaround instead.
    """
    prefix = r"\b" if _WORD_CHAR.match(token[0]) else r"(?<!\w)"
    suffix = r"\b" if _WORD_CHAR.match(token[-1]) else r"(?!\w)"
    body = re.escape(token)
    if not matches_case_sensitively(token):
        body = f"(?i:{body})"
    return f"{prefix}{body}{suffix}"


@lru_cache(maxsize=64)
def _compile(
    mapping_items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern, dict[str, str], dict[str, str]]:
    """Compile a mapping into its single-pass alternation.

    Returns ``(pattern, sensitive_lookup, insensitive_lookup)``; the
    insensitive lookup is keyed by casefolded token. Cached per mapping (one
    mapping per run in practice). Fails loud on tokens whose casefolded forms
    collide within the insensitive set — the match text could not be resolved
    to a single replacement.
    """
    sensitive: dict[str, str] = {}
    insensitive: dict[str, str] = {}
    for token, replacement in mapping_items:
        if not token:
            raise ValueError("Pronunciation mapping has an empty token")
        if matches_case_sensitively(token):
            sensitive[token] = replacement
        else:
            folded = token.casefold()
            if folded in insensitive and insensitive[folded] != replacement:
                raise ValueError(
                    f"Pronunciation mapping tokens collide case-insensitively "
                    f"on {folded!r} with different respellings"
                )
            insensitive[folded] = replacement
    longest_first = sorted(
        (token for token, _ in mapping_items), key=lambda t: (-len(t), t)
    )
    pattern = re.compile(
        "|".join(_token_pattern(token) for token in longest_first) or r"(?!x)x"
    )
    return pattern, sensitive, insensitive


def apply_pronunciations(
    text: str, mapping: Optional[dict[str, str]]
) -> tuple[str, list[SwapEvent]]:
    """Replace every whole-token occurrence of a mapping key with its
    respelling. Pure and deterministic; an empty/None mapping is identity.

    Returns the rewritten text plus one :class:`SwapEvent` per distinct
    (matched text, replacement) pair, in first-occurrence order, so the
    caller can log exactly what the TTS will say differently from the
    transcript.
    """
    if not mapping or not text:
        return text, []
    pattern, sensitive, insensitive = _compile(tuple(sorted(mapping.items())))
    counts: dict[tuple[str, str], int] = {}

    def _sub(match: re.Match) -> str:
        matched = match.group(0)
        replacement = sensitive.get(matched)
        if replacement is None:
            replacement = insensitive.get(matched.casefold())
        if replacement is None:
            # An ALL-CAPS token matched in a different case: only possible if
            # the pattern and lookups disagree — a bug, never data.
            raise AssertionError(
                f"Pronunciation swap matched {matched!r} with no replacement"
            )
        key = (matched, replacement)
        counts[key] = counts.get(key, 0) + 1
        return replacement

    swapped = pattern.sub(_sub, text)
    events = [
        SwapEvent(token=matched, replacement=replacement, count=count)
        for (matched, replacement), count in counts.items()
    ]
    return swapped, events
