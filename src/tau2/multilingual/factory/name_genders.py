# Copyright Sierra
"""Closed given-name → gender catalog for the multilingual factory.

Two consumers, one source of truth:

1. **Entity localization** (``tau2 factory localize-entities``): the gender of
   every locale identity is DERIVED from the English source caller's given
   name via :func:`source_caller_gender`, so a task keeps one caller gender
   across its plain-localized and ``_identity`` variants (and across
   languages), and the voice sampler can pin the persona to it. Coverage is
   mandatory: an unresolvable source caller name fails loudly at identity-map
   build time — extend the catalog (or, for a genuinely unisex name,
   :data:`UNISEX_SOURCE_CALLER_GENDERS`) rather than letting a caller fall
   through to un-pinned voice sampling.
2. **Factory guardrails**: ``display_name`` ↔ ``tags.gender`` consistency for
   pack personas via :func:`known_given_name_gender` — cheap insurance against
   Olivia-tagged-male / Noah-tagged-female pack authoring. This path is
   advisory: a name the catalog does not know simply passes.

The catalog must stay UNAMBIGUOUS: only names whose predominant gender is
clear-cut belong in :data:`GIVEN_NAME_GENDERS`. A genuinely unisex name (e.g.
"Chen") must NOT be added here — the guardrail would then flag correct packs.
Unisex SOURCE-DOMAIN caller names instead get a reviewed per-name assignment
in :data:`UNISEX_SOURCE_CALLER_GENDERS`, which only the entity-localization
path consults.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Unambiguous given names → gender. Covers (a) every gendered first name used
# by the English source-domain callers (airline db.json, telecom db.toml) and
# (b) the persona display names of the shipped language packs. Keys are
# normalized: lowercase ASCII (see _normalize_given_name).
GIVEN_NAME_GENDERS: dict[str, str] = {
    # --- English source-domain caller names (telecom) ---
    # "john" is the pool's original single caller; the rest are the reviewed
    # caller_pool.yaml identities the seed diversification assigns per task
    # ("olivia" is shared with the airline section below).
    "john": "male",
    "allison": "female",
    "andre": "male",
    "bridget": "female",
    "caleb": "male",
    "colleen": "female",
    "darnell": "male",
    "diane": "female",
    "felicia": "female",
    "franklin": "male",
    "gerald": "male",
    "hector": "male",
    "latoya": "female",
    "marcus": "male",
    "monique": "female",
    "nicole": "female",
    "rachel": "female",
    "raymond": "male",
    "roland": "male",
    "spencer": "male",
    "tamara": "female",
    "trevor": "male",
    "vanessa": "female",
    "wesley": "male",
    "whitney": "female",
    # --- English source-domain caller names (airline) ---
    "aarav": "male",
    "amelia": "female",
    "anya": "female",
    "daiki": "male",
    "emma": "female",
    "ethan": "male",
    "ivan": "male",
    "james": "male",
    "liam": "male",
    "lucas": "male",
    "mei": "female",
    "mia": "female",
    "mohamed": "male",
    "noah": "male",
    "olivia": "female",
    "omar": "male",
    "raj": "male",
    "sofia": "female",
    "sophia": "female",
    "yara": "female",
    # --- English source-domain caller names (retail) ---
    # Only the names retail adds; it shares most of its given-name pool with
    # airline above ("Lei" is unisex and lives in the unisex table instead).
    "ava": "female",
    "fatima": "female",
    "isabella": "female",
    "yusuf": "male",
    # --- shipped language-pack persona display names ---
    "adriana": "female",
    "alejandro": "male",
    "alessandro": "male",
    "aoi": "female",
    "artyom": "male",
    "bartek": "male",
    "bogdan": "male",
    "camila": "female",
    "camille": "female",
    "etienne": "male",
    "friedrich": "male",
    "giulia": "female",
    "hiroshi": "male",
    "imran": "male",
    "jianguo": "male",
    "jihun": "male",
    "juliana": "female",
    "katarzyna": "female",
    "khaled": "male",
    "lena": "female",
    "lina": "female",
    "lisa": "female",
    "matt": "male",
    "mehmet": "male",
    "minh": "male",
    "ricardo": "male",
    "rishika": "female",
    "sem": "male",
    "soyeon": "female",
    "trang": "female",
    "willemijn": "female",
    "xiaomeng": "female",
    "yelena": "female",
    "zeynep": "female",
    # --- stock English voice personas (tau2.data_model.voice_personas) ---
    # matt/lisa (control) live in the language-pack section above; these are
    # the regular-complexity pool, consulted by the English caller-gender
    # voice pinning ("wei" is unisex and lives in the reviewed table below).
    "arjun": "male",
    "mamadou": "male",
    "mildred": "female",
    "priya": "female",
}

# Reviewed gender assignments for UNISEX English source-domain caller and
# stock-persona names. Consulted ONLY by :func:`source_caller_gender` (never
# by the guardrail): a unisex name carries no signal about a persona author's
# intent, but a source caller still needs one fixed gender so their locale
# identities and voice pinning are deterministic. "Chen" is assigned female
# so the 50-task airline set splits exactly 25 male / 25 female at task level
# (24F/25M without it). "Pat"/"Sam" are the toyair test-fixture callers,
# assigned to keep the fixture mixed-gender. "Wei" is the stock English
# persona Wei Lin (a woman) — the romanized name itself is unisex.
UNISEX_SOURCE_CALLER_GENDERS: dict[str, str] = {
    "chen": "female",
    # Retail caller. Unisex in Mandarin (雷 reads male, 蕾 female) with nothing
    # in the task or the DB record to disambiguate, so this is a reviewed
    # assignment, not a lookup — chosen male to keep retail's caller-gender
    # split even (the other 22 given names sit at 9F/9M plus 4 female).
    "lei": "male",
    "pat": "male",
    "sam": "female",
    "wei": "female",
}


def _normalize_given_name(name: str) -> str:
    """First name token, lowercase ASCII (diacritics stripped: É → e)."""
    first_token = name.strip().split()[0] if name.strip() else ""
    lowered = first_token.lower()
    stripped = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z]", "", stripped)


def known_given_name_gender(display_name: str) -> Optional[str]:
    """The catalog gender of a (persona) given name, or None if unknown.

    Advisory lookup for the display_name ↔ tags.gender guardrail: only
    unambiguous catalog names resolve; anything else returns None and the
    guardrail stays silent. Unisex assignments are deliberately excluded.
    """
    return GIVEN_NAME_GENDERS.get(_normalize_given_name(display_name))


def source_caller_gender(first_name: str) -> str:
    """The gender of an English source-domain caller's given name (mandatory).

    Resolution order: reviewed unisex assignment, then the unambiguous
    catalog. Raises when neither knows the name — entity localization must
    never silently produce an ungendered identity.

    Raises:
        ValueError: If the name is not covered; the fix is to extend
            ``GIVEN_NAME_GENDERS`` (or ``UNISEX_SOURCE_CALLER_GENDERS`` for a
            unisex name) in this module.
    """
    key = _normalize_given_name(first_name)
    gender = UNISEX_SOURCE_CALLER_GENDERS.get(key) or GIVEN_NAME_GENDERS.get(key)
    if gender is None:
        raise ValueError(
            f"source caller given name '{first_name}' has no gender in the "
            "catalog — add it to GIVEN_NAME_GENDERS (or, if genuinely unisex, "
            "to UNISEX_SOURCE_CALLER_GENDERS) in "
            "tau2.multilingual.factory.name_genders"
        )
    return gender
