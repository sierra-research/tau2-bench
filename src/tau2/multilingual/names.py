"""Personal-name ORDER — how a locale composes a display full name.

The English source data is given-first ("Wei Hu"), and until this module
existed every localized identity inherited that order verbatim: the caller
generator composed ``f"{first} {last}"`` with no locale awareness. For
Mandarin, Korean, and Vietnamese that is the wrong order. The caller then
introduces itself the way its language actually works (family name first —
"Hu Wei") while the agent's customer record holds the English-ordered string,
and an exact-match lookup misses. That is not an agent failure and not a
designed challenge; it is English word order leaking through the localizer, so
it is fixed at the source: the order is a declared property of the language
pack and every full-name composition in the pipeline routes through here.

Two rules keep the pieces consistent:

- ``first_name``/``given`` and ``last_name``/``family`` are always ROLES, never
  positions. A record's ``first_name`` is the given name in every language.
  Only the DISPLAY string reorders.
- The ENGLISH source is always :attr:`NameOrder.GIVEN_FIRST`. Only the locale
  side of a rename takes the pack's order — pass it explicitly rather than
  assuming both sides match.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class NameOrder(str, Enum):
    """How a locale writes and says a full name."""

    GIVEN_FIRST = "given_first"
    """Given name then family name — "Wei Hu" (English, most European locales)."""

    FAMILY_FIRST = "family_first"
    """Family name then given name — "Hu Wei" (Mandarin, Korean, Vietnamese)."""


# A full name written in Han characters joins WITHOUT a space (郑志强, never
# 郑 志强) — the script's own orthography, applied only when BOTH roles are
# entirely Han so romanized-pinyin names ("Zheng Zhiqiang") keep their space.
_HAN_NAME_RE = re.compile(r"^[一-鿿]+$")


def _name_separator(given: str, family: str) -> str:
    """The joiner between the two name roles — '' for an all-Han name."""
    if _HAN_NAME_RE.match(given) and _HAN_NAME_RE.match(family):
        return ""
    return " "


def compose_full_name(
    given: str, family: str, order: NameOrder = NameOrder.GIVEN_FIRST
) -> str:
    """The display full name for ``order`` — the ONLY place order is applied.

    ``given``/``family`` are roles; the return value is what a caller says and
    what the agent-side record stores.

    An all-Han name joins without a space (native-script DB variant); such a
    display name does NOT round-trip through :func:`split_full_name`, which
    only structured (telecom-shape) renames use — Han native identities are
    currently retail-only, where records carry the roles separately.
    """
    given, family = given.strip(), family.strip()
    separator = _name_separator(given, family)
    if order is NameOrder.FAMILY_FIRST:
        return f"{family}{separator}{given}".strip()
    return f"{given}{separator}{family}".strip()


def split_full_name(
    full_name: str, order: NameOrder = NameOrder.GIVEN_FIRST
) -> tuple[str, str]:
    """``(given, family)`` from a display full name written in ``order``.

    The inverse of :func:`compose_full_name`: recomposing the result in the
    same order reconstructs the input, so multi-token names round-trip. The
    GIVEN name is the single token and the family name absorbs the remainder,
    which is the side that actually carries extra tokens — particles under
    given-first ("Ana de Souza" -> given "Ana", family "de Souza") and a
    middle name under family-first ("Nguyen Van Minh" -> given "Minh", family
    "Nguyen Van").
    """
    full_name = full_name.strip()
    if order is NameOrder.FAMILY_FIRST:
        family, _, given = full_name.rpartition(" ")
        return (given, family) if family else (full_name, "")
    given, _, family = full_name.partition(" ")
    return given, family


def name_order_for_language(language: Optional[str]) -> NameOrder:
    """The declared name order of ``language``'s pack.

    ``GIVEN_FIRST`` for English, an unregistered language, or None — the
    English source order, so every un-flipped language behaves exactly as it
    did before this module existed.
    """
    if not language:
        return NameOrder.GIVEN_FIRST
    # Local import: the registry imports the pack schema, which imports this
    # module for NameOrder.
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(language)
    return pack.name_order if pack else NameOrder.GIVEN_FIRST
