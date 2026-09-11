# Copyright Sierra
"""Shared factory path + JSON-state helpers.

The translation/parity/names stages each need the same things: the current
``DATA_DIR`` (resolved dynamically, not snapshotted at import, so isolated test
environments work), the source/localized task sets, and pretty UTF-8 JSON
persistence with a trailing newline.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from tau2.multilingual.factory.state import factory_root_dir
from tau2.multilingual.localize_lib import multilingual_data_dir

# The per-pack subdirectory that holds the localized user-simulator guidelines
# markdown. No run reads these files — the runtime guidelines are always
# English — but the draft chain authors them because Call B's body grounds the
# LIVE localization drafting (Call D), and Call B/C reuse a shipped pack's
# guidelines as their exemplar.
GUIDELINES_PACK_SUBDIR = "guidelines"

# Written by the retired parity probe; still read by annotation-side agreement
# reporting for rounds that recorded one.
PARITY_PROBE_FILENAME = "parity_probe_verdicts.json"

# Continuous-bed filename basenames for the two CANONICAL locale beds, plus the
# shared top-level English office bed. ``pack_assembly`` scaffolds acoustic
# presets against these names. Locale bed PRODUCTION is retired (every language
# runs on shared stock-English acoustics); playback of any already-committed
# bed still resolves through here.
OUTDOOR_BED_BASENAME = "busy_street_iphone_mic.wav"
TV_KITCHEN_BED_BASENAME = "medium_size_room_tv_news_iphone_mic.wav"
SHARED_OFFICE_BED_BASENAME = "people_talking.wav"


def _country_from_locale(loc: Optional[str]) -> Optional[str]:
    """The ISO 3166-1 alpha-2 country from an ISO 3166-2 persona locale.

    ``"ES-MD"`` (Madrid) -> ``"ES"``, ``"EG-C"`` (Cairo) -> ``"EG"``. Returns
    ``None`` for anything that is not a 2-letter country head.
    """
    if not loc:
        return None
    head = str(loc).split("-", 1)[0].strip().upper()
    return head if len(head) == 2 and head.isalpha() else None


def derive_locale_dir(language: str, locales) -> str:
    """The ``language_COUNTRY`` locale dir for a pack's voice/acoustic assets.

    Derived from the personas' ISO 3166-2 ``locale`` codes (the locale info the
    factory already carries): the most common country wins, ties broken by first
    occurrence (so the primary/first persona anchors it). Reproduces the merged
    convention without a hardcoded table — ``hi`` personas (IN-*) -> ``hi_IN``,
    ``ro`` (RO-*) -> ``ro_RO``, ``es`` (ES-MD, AR-C) -> ``es_ES``. Falls back to
    the bare language code when no persona carries a usable locale.
    """
    countries = [c for c in (_country_from_locale(loc) for loc in locales) if c]
    if not countries:
        return language
    country = Counter(countries).most_common(1)[0][0]
    return f"{language}_{country}"


def default_voice_guidelines_filename(lang: str) -> str:
    """The conventional final VOICE guidelines filename for a language.

    The fallback used whenever a draft did not declare an explicit filename
    (``drafting`` declares it, ``FactoryProject`` records it, ``finalize``
    restores it) — defined once so the convention cannot fork. Pack-relative
    and namespaced under ``guidelines/``: no run reads the localized
    guidelines (the runtime is always English), but the draft still authors
    them because Call B's body grounds the LIVE localization drafting (Call D).
    """
    return f"{GUIDELINES_PACK_SUBDIR}/simulation_guidelines_voice_{lang}.md"


def default_text_guidelines_filename(lang: str) -> str:
    """The conventional final TEXT (typed chat) guidelines filename.

    Same ``guidelines/`` namespacing as
    :func:`default_voice_guidelines_filename`.
    """
    return f"{GUIDELINES_PACK_SUBDIR}/simulation_guidelines_text_{lang}.md"


def calibration_dir(lang: str) -> Path:
    """The language's calibration workspace under ``_factory/<lang>/``."""
    return factory_root_dir() / lang / "calibration"


def load_localized_tasks(lang: str, domain: str) -> list[dict]:
    """The language's localized task set (registered by filename convention)."""
    path = multilingual_data_dir() / lang / f"{domain}_tasks_{lang}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No localized task set for '{lang}/{domain}' at {path}. "
            "Task translation is retired — the live flow is English-prose "
            "seed tasks (`tau2 factory seed-tasks`), and only packs from the "
            "translation era carry localized sets."
        )
    return json.loads(path.read_text())


def save_json(path: Path, obj) -> None:
    """Persist ``obj`` as pretty UTF-8 JSON with a trailing newline.

    The factory's state-file convention (``ensure_ascii=False`` so native-script
    content stays readable in the file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())
