# Copyright Sierra
"""Language Pack registry for multilingual user simulation.

A :class:`LanguagePack` bundles all language-specific *content* for one
language: persona definitions (each carrying its own backchannel/out-of-turn speech
phrase lists), the localized guidelines file, acoustic-preset registrations,
the language-level backchannel decision prompt, and default behavior
parameters.

Shared infra (this package) defines the pack interface, the registry, the
loader, and the integration touchpoints. A language owner implements exactly
one pack under ``tau2/multilingual/languages/<lang>/`` and edits zero shared
code. English remains the default everywhere: with no pack registered (or no
multilingual persona selected), behavior is byte-identical to before this
package existed.

See ``docs/multilingual/ADDING_A_LANGUAGE.md`` (authored in the final PR of
the v1 milestone) for the language-owner playbook.
"""

from tau2.multilingual.registry import (
    get_language_pack,
    get_multilingual_persona,
    list_language_packs,
    register_language_pack,
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
    "get_language_pack",
    "get_multilingual_persona",
    "list_language_packs",
    "register_language_pack",
]
