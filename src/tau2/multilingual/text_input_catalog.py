# Copyright Sierra
"""Closed catalog of TEXT input styles (how a chat user types the language).

Real multilingual chat users do not type textbook orthography: Hindi chat is
dominantly romanized ("mera naam Rahul hai") and code-mixed with English;
Spanish/Portuguese users routinely type without diacritics ("Jose", "nao",
"voce"). A run is pinned to ONE style (``TextRunConfig.text_input_style``),
recorded in provenance, so arms stay comparable.

Closed-catalog contract (same shape as the nativeness/delivery factor
catalogs): the styles are a fixed enum here; a language pack SELECTS the
styles it supports (``text_input.styles``) and supplies only localized
example pairs — the instructional prose is a fixed, versioned in-code
template below, persona-neutral per the localization-content rule (script /
orthography mechanics only; register, affect, and HOW MUCH English mixes in
stay persona-owned).

``native_script`` is the default: every pack supports it implicitly and it
renders NO extra prompt block, so the baseline arm is byte-identical to a run
without the knob.
"""

from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tau2.multilingual.schema import TextInputExample

TEXT_INPUT_STYLE_PROMPT_VERSION = "v1"


class TextInputStyle(str, Enum):
    """The closed set of text input styles a run can be pinned to."""

    NATIVE_SCRIPT = "native_script"
    """Textbook native orthography — the default; renders no directive."""

    ROMANIZED = "romanized"
    """The matrix language typed in the Latin alphabet the way chat users
    transliterate ad hoc (no scholarly scheme, no diacritics)."""

    DIACRITIC_FREE = "diacritic_free"
    """The language's own alphabet typed without accents or diacritics
    (the mobile-keyboard register)."""

    CODE_MIXED = "code_mixed"
    """Romanized matrix language with everyday English words kept in English
    spelling (the Hinglish chat register). The style owns only the
    script/orthography mechanics; mixing PROPORTION stays persona-owned."""


# Fixed, versioned directive templates — calibrated prompt material: never
# edit in place without bumping TEXT_INPUT_STYLE_PROMPT_VERSION. Each
# non-default template explicitly supersedes the native-orthography bullet of
# the target-language directive (tau2.multilingual.english_prompts), which
# would otherwise contradict it inside one prompt. ``{language_name}`` is the
# pack's display name; the worked example pairs come from the pack.
_STYLE_DIRECTIVE_TEMPLATES: dict[TextInputStyle, str] = {
    TextInputStyle.ROMANIZED: """
## HOW YOU TYPE (ROMANIZED {language_name})

You type {language_name} in the Latin alphabet, the way {language_name} speakers write casual chat messages — NOT in the native script. This overrides the native-orthography instruction above: your words are still {language_name}, but every message is typed in Latin letters.
- Transliterate the way an everyday chat user does: informal and readable, no scholarly transliteration marks, no diacritics.
- Concrete values (names, IDs, emails, codes, numbers) stay exactly as written in your scenario.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) stay verbatim.
""".strip(),
    TextInputStyle.DIACRITIC_FREE: """
## HOW YOU TYPE (NO ACCENTS)

You type {language_name} without accents or diacritics, the way casual chat users type on a phone keyboard. This overrides the native-orthography instruction above: keep the language's own alphabet and spelling, but drop every accent mark and language-specific punctuation that needs extra keystrokes.
- Never add accents or special punctuation back, even for words that formally require them.
- Concrete values (names, IDs, emails, codes, numbers) stay exactly as written in your scenario.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) stay verbatim.
""".strip(),
    TextInputStyle.CODE_MIXED: """
## HOW YOU TYPE (ROMANIZED, CODE-MIXED {language_name})

You type {language_name} in the Latin alphabet, the way {language_name} speakers write casual chat messages — NOT in the native script — and everyday English words that belong in your speech stay in their English spelling inside the sentence. This overrides the native-orthography instruction above.
- Transliterate the {language_name} words the way an everyday chat user does: informal and readable, no diacritics.
- English words keep normal English spelling; never transliterate them phonetically.
- How much English you mix in comes from your persona — this section only fixes HOW words are typed, not how many are English.
- Concrete values (names, IDs, emails, codes, numbers) stay exactly as written in your scenario.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) stay verbatim.
""".strip(),
}

_EXAMPLES_LABEL = "How this looks (standard orthography → how you type it):"


def render_input_style_directive(
    style: TextInputStyle,
    language_name: str,
    examples: list["TextInputExample"],
) -> Optional[str]:
    """The typing-style directive block for one run, or None for the default.

    Fixed in-code template (never model-improvised); only the pack-supplied
    example pairs vary. ``native_script`` returns None — the baseline arm
    renders nothing.
    """
    if TextInputStyle(style) is TextInputStyle.NATIVE_SCRIPT:
        return None
    template = _STYLE_DIRECTIVE_TEMPLATES[TextInputStyle(style)]
    lines = [template.format(language_name=language_name)]
    if examples:
        lines.append(_EXAMPLES_LABEL)
        for example in examples:
            lines.append(f"- {example.standard} → {example.typed}")
    return "\n".join(lines)
