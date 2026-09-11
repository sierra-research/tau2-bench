# Copyright Sierra
"""Language Pack registry for multilingual user simulation.

A :class:`LanguagePack` bundles all language-specific *content* for one
language: persona definitions (each carrying its own backchannel/out-of-turn
speech phrase lists), the localized guidelines file, acoustic-preset
registrations, the language-level backchannel decision prompt, and default
behavior parameters. A language owner authors exactly one
``data/tau2/multilingual/<lang>/pack.yaml`` and edits no shared code; with no
pack registered (or no multilingual persona selected), behavior is the English
default.

See ``docs/multilingual/ADDING_A_LANGUAGE.md`` for the language-owner playbook.
"""

from tau2.multilingual.registry import (
    get_agent_language_clause,
    get_language_pack,
    get_multilingual_persona,
    list_language_packs,
    register_language_pack,
    resolve_run_language,
)
from tau2.multilingual.schema import (
    AcousticPreset,
    LanguagePack,
    MultilingualPersonaConfig,
)

__all__ = [
    "AcousticPreset",
    "LanguagePack",
    "MultilingualPersonaConfig",
    "get_agent_language_clause",
    "get_language_pack",
    "get_multilingual_persona",
    "list_language_packs",
    "register_language_pack",
    "resolve_run_language",
]
