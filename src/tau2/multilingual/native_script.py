# Copyright Sierra
"""Native-script DB identity variant — suffix contract + name spell-outs.

The romanization x localization ablation adds, per language, a THIRD retail
task-set variant: ``<domain>_<lang>_identity_native``, where the caller's DB
name keeps its native script (Devanagari / Hanzi) instead of the ASCII fold
the ordinary ``_identity`` sets apply (PR #534). This module owns everything
that is script-specific about that variant:

- the ``identity_native`` suffix contract (one constant, three predicates),
  shared by the factory build, the run-time english-prompt swap, the agent
  clause plumbing, and the pool specs;
- the deterministic native NAME SPELL-OUTS the user simulator dictates when
  asked to spell or confirm the caller's name:

  * **Devanagari** — algorithmic akshara segmentation with matra
    clarifications for the ambiguous pairs (ि छोटी इ vs ी बड़ी ई, ु छोटा उ vs
    ू बड़ा ऊ), conjunct decomposition (आधा <C>), and nukta call-outs;
  * **Han** — per-character clarification glosses from the locale corpus's
    curated closed catalog (标准 phone conventions: 弓长张, 木子李, 口天吴, …;
    common-word references otherwise). Never derived: a character without a
    reviewed gloss fails loudly.

- the fixed, versioned prompt block (``NATIVE_NAME_SPELLOUT_VERSION``) the
  runner appends to the user-sim task instructions for native-variant tasks.

Everything here is pure and deterministic; nothing calls an LLM.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Suffix contract
# ---------------------------------------------------------------------------

# The task-set variant token: file <domain>_tasks_<lang>_identity_native.json
# -> task set <domain>_<lang>_identity_native -> ids <id>_<lang>_identity_native.
NATIVE_IDENTITY_VARIANT = "identity_native"

# Task-id suffixes that mark an identity-variant task (caller renamed via
# initialization_data). Longest first, so suffix-stripping helpers can use it.
IDENTITY_TASK_ID_SUFFIXES = (f"_{NATIVE_IDENTITY_VARIANT}", "_identity")


def native_identity_task_set_name(domain: str, lang: str) -> str:
    return f"{domain}_{lang}_{NATIVE_IDENTITY_VARIANT}"


def is_native_identity_task_id(task_id: str) -> bool:
    """Whether a task id belongs to a native-script identity task set."""
    return task_id.endswith(f"_{NATIVE_IDENTITY_VARIANT}")


def is_identity_task_id(task_id: str) -> bool:
    """Whether a task id belongs to ANY identity-variant set (folded or native)."""
    return task_id.endswith(IDENTITY_TASK_ID_SUFFIXES)


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

_HAN_RE = re.compile(r"^[一-鿿]+$")
_DEVANAGARI_RE = re.compile(r"^[ऀ-ॿ]+$")


def is_han(text: str) -> bool:
    return bool(text) and bool(_HAN_RE.match(text))


def is_devanagari(text: str) -> bool:
    return bool(text) and bool(_DEVANAGARI_RE.match(text))


# ---------------------------------------------------------------------------
# Devanagari akshara spell-out (algorithmic)
# ---------------------------------------------------------------------------

# Matra names as a Hindi speaker clarifies them over the phone. The ambiguous
# pairs get the explicit छोटी/बड़ी disambiguation — the point of the exercise:
# ि and ी (and ु / ू) are the standard confusions when dictating a name.
_MATRA_NAMES: dict[str, str] = {
    "ा": "आ की मात्रा",  # ा
    "ि": "छोटी इ की मात्रा",  # ि
    "ी": "बड़ी ई की मात्रा",  # ी
    "ु": "छोटा उ की मात्रा",  # ु
    "ू": "बड़ा ऊ की मात्रा",  # ू
    "ृ": "ऋ की मात्रा",  # ृ
    "ॅ": "चंद्र ए की मात्रा",  # ॅ
    "े": "ए की मात्रा",  # े
    "ै": "ऐ की मात्रा",  # ै
    "ॉ": "चंद्र ओ की मात्रा",  # ॉ
    "ो": "ओ की मात्रा",  # ो
    "ौ": "औ की मात्रा",  # ौ
}

_SIGN_NAMES: dict[str, str] = {
    "ं": "अनुस्वार बिंदु",  # ं
    "ँ": "चंद्रबिंदु",  # ँ
    "ः": "विसर्ग",  # ः
}

_VIRAMA = "्"  # ्
_NUKTA = "़"  # ़

_DEVANAGARI_CONSONANT_RE = re.compile(r"[क-हक़-य़ॹ-ॿ]")
_DEVANAGARI_INDEPENDENT_VOWEL_RE = re.compile(r"[ऄ-औॲ-ॷ]")


def _segment_aksharas(word: str) -> list[str]:
    """Split a Devanagari word into aksharas (orthographic syllables).

    An akshara is ``(C [nukta] virama)* C [nukta] [matra] [sign]`` or an
    independent vowel ``V [sign]``. Input is NFC-normalized first; NFC leaves
    the nukta consonants decomposed (they are composition exclusions), so the
    nukta is always a combining char here.
    """
    word = unicodedata.normalize("NFC", word)
    aksharas: list[str] = []
    current = ""
    pending_conjunct = False
    for ch in word:
        if _DEVANAGARI_CONSONANT_RE.match(ch):
            if current and not pending_conjunct:
                aksharas.append(current)
                current = ""
            current += ch
            pending_conjunct = False
        elif ch == _VIRAMA:
            current += ch
            pending_conjunct = True
        elif ch == _NUKTA:
            current += ch
        elif _DEVANAGARI_INDEPENDENT_VOWEL_RE.match(ch):
            if current:
                aksharas.append(current)
            current = ch
            pending_conjunct = False
        elif ch in _MATRA_NAMES or ch in _SIGN_NAMES:
            current += ch
        else:
            raise ValueError(
                f"devanagari_spellout: unsupported character {ch!r} (U+{ord(ch):04X}) "
                f"in {word!r} — extend the akshara segmenter before shipping a "
                "name that uses it."
            )
    if current:
        aksharas.append(current)
    return aksharas


def _akshara_rendition(akshara: str) -> str:
    """One akshara as the caller dictates it: the akshara + clarifications.

    - conjuncts name each halved consonant: क्से -> 'क्से (आधा क, फिर से, ए की मात्रा)'
    - the ambiguous matras are named छोटी/बड़ी: मि -> 'मि (म में छोटी इ की मात्रा)'
    - nukta consonants are called out: ड़ -> 'ड में नुक़्ता'
    - a bare consonant or independent vowel needs no clarification: त -> 'त'
    """
    clarifications: list[str] = []
    # Split the cluster at viramas: every part but the last is a halved
    # consonant; the last carries the matra/signs.
    parts = akshara.split(_VIRAMA)
    for half in parts[:-1]:
        base = half.replace(_NUKTA, "")
        clarifications.append(f"आधा {base}")
        if _NUKTA in half:
            clarifications.append(f"{base} में नुक़्ता")
    final = parts[-1]
    if len(parts) > 1 and final:
        final_base = final[0]
        clarifications.append(f"फिर {final_base}")
    for ch in final:
        if ch == _NUKTA:
            clarifications.append(f"{final[0]} में नुक़्ता")
        elif ch in _MATRA_NAMES:
            prefix = "" if len(parts) > 1 else f"{final[0]} में "
            clarifications.append(f"{prefix}{_MATRA_NAMES[ch]}")
        elif ch in _SIGN_NAMES:
            clarifications.append(_SIGN_NAMES[ch])
    if not clarifications:
        return akshara
    return f"{akshara} ({', '.join(clarifications)})"


def devanagari_spellout(word: str) -> list[str]:
    """The akshara-by-akshara spell-out of one Devanagari word.

    Returns one rendition string per akshara, in order — what a Hindi speaker
    dictates over the phone, matra clarifications included. Deterministic and
    total over the Devanagari block; an unsupported character raises.
    """
    if not is_devanagari(unicodedata.normalize("NFC", word)):
        raise ValueError(f"devanagari_spellout: {word!r} is not entirely Devanagari.")
    return [_akshara_rendition(a) for a in _segment_aksharas(word)]


# ---------------------------------------------------------------------------
# Han character clarification glosses (curated closed catalog)
# ---------------------------------------------------------------------------


def han_spellout(word: str, character_clarifications: dict[str, str]) -> list[str]:
    """Per-character clarification glosses for one Han word.

    ``character_clarifications`` is the locale corpus's curated catalog
    (character -> gloss, e.g. 张 -> 弓长张, 伟 -> 伟大的伟). Curated content is
    the whole point: a character without a reviewed gloss raises rather than
    improvising one.
    """
    if not is_han(word):
        raise ValueError(f"han_spellout: {word!r} is not entirely Han characters.")
    missing = [ch for ch in word if ch not in character_clarifications]
    if missing:
        raise ValueError(
            f"han_spellout: no reviewed clarification gloss for {missing!r} — "
            "add it to the locale corpus's character_clarifications catalog."
        )
    return [f"{ch} — {character_clarifications[ch]}" for ch in word]


# ---------------------------------------------------------------------------
# Per-identity spell-out payload (shared by the factory build and the runner)
# ---------------------------------------------------------------------------


def native_name_spellout(
    first_name: str,
    last_name: str,
    *,
    character_clarifications: dict[str, str] | None = None,
) -> list[str]:
    """The caller's full-name spell-out payload, one line per name part.

    Dispatches on the SCRIPT of the name itself (the two never mix):

    - Han: one clarification gloss per character, family name first (the
      display order), from the curated catalog — fail-loud on a missing gloss.
    - Devanagari: given name first (the display order), each word spelled
      akshara by akshara with matra clarifications.

    This single function is what the factory embeds into the native identity
    map's ``character_clarifications`` payload AND what the coverage tests
    recompute — the committed payload can never drift from the algorithm/catalog.
    """
    if is_han(first_name) and is_han(last_name):
        catalog = character_clarifications or {}
        # Family-first display order — the order the caller says and spells it.
        return han_spellout(last_name + first_name, catalog)
    if is_devanagari(first_name) and is_devanagari(last_name):
        lines = []
        for word in (first_name, last_name):
            lines.append(f"{word}: " + ", ".join(devanagari_spellout(word)))
        return lines
    raise ValueError(
        "native_name_spellout: name is neither all-Han nor all-Devanagari "
        f"({first_name!r} {last_name!r}) — no native spell-out convention is "
        "defined for its script."
    )


# ---------------------------------------------------------------------------
# User-simulator directive block (fixed, versioned prompt text)
# ---------------------------------------------------------------------------

NATIVE_NAME_SPELLOUT_VERSION = "v1"

# Appended to the user-sim task instructions for every native-script identity
# task (see tau2.runner.build.user_prompt_task). Calibrated prompt material:
# never edit in place without bumping NATIVE_NAME_SPELLOUT_VERSION. The block
# SUPERSEDES the pack's generic spelling_alphabet guidance (which assumes the
# romanized-DB regime) for the caller's own name.
NATIVE_NAME_SPELLOUT_TEMPLATE_HAN_V1 = """
## SPELLING YOUR NAME (NATIVE SCRIPT)

Your name is written {full_name}, in Chinese characters, and the agent's records store it exactly that way. When asked to spell, clarify, or confirm your name, disambiguate each character the way a Mandarin speaker does on the phone — by its standard decomposition or a well-known word that contains it — using exactly these clarifications:
{clarification_lines}
Never spell your own name in Latin letters or pinyin, and never use an A-for-Alpha alphabet for it: the record is in characters, so letter-spelling cannot identify you. (Latin codes, emails, and IDs are still spelled letter by letter as usual.)
""".strip()

NATIVE_NAME_SPELLOUT_TEMPLATE_DEVANAGARI_V1 = """
## SPELLING YOUR NAME (NATIVE SCRIPT)

Your name is written {full_name}, in Devanagari, and the agent's records store it exactly that way. When asked to spell, clarify, or confirm your name, dictate it akshara by akshara the way a Hindi speaker does on the phone, naming the matras (छोटी इ vs बड़ी ई, छोटा उ vs बड़ा ऊ) and half letters, using exactly this spell-out:
{clarification_lines}
Never spell your own name in Latin letters or a romanized form: the record is in Devanagari, so English letter-spelling cannot identify you. (Latin codes, emails, and IDs are still spelled letter by letter as usual.)
""".strip()


def native_spellout_block_for_task(
    task: dict,
    language: str,
    domain: str,
) -> str:
    """The rendered spell-out block for one native-variant task (fail-loud).

    Reads the caller's native name from the task's own ``initialization_data``
    users patch and the per-identity ``character_clarifications`` payload from
    the committed native identity map — the REVIEWED artifact, so what the sim
    dictates is exactly what a human signed off on (a coverage test asserts the
    committed payload equals the recomputation, so it cannot silently drift).

    Raises on a task without a single-user patch, a missing map, or a caller
    absent from it: a native-variant run whose caller cannot spell their name
    is a corrupted arm and must die at build time, not mid-call.
    """
    import json

    from tau2.multilingual.factory.entity_localization import (
        native_identity_map_path,
    )
    from tau2.multilingual.names import compose_full_name, name_order_for_language

    users = (
        ((task.get("initial_state") or {}).get("initialization_data") or {}).get(
            "agent_data"
        )
        or {}
    ).get("users") or {}
    if len(users) != 1:
        raise ValueError(
            f"native spell-out: task {task.get('id')!r} patches {len(users)} "
            "user records; exactly one caller record is expected."
        )
    record = next(iter(users.values()))
    name = record.get("name") or {}
    first, last = name.get("first_name"), name.get("last_name")
    if not first or not last:
        raise ValueError(
            f"native spell-out: task {task.get('id')!r} caller record has no name."
        )
    map_path = native_identity_map_path(language, domain)
    if not map_path.is_file():
        raise FileNotFoundError(
            f"native spell-out: no native identity map at {map_path}; run "
            f"`tau2 factory localize-entities --lang {language} --domain "
            f"{domain} --native-script` first."
        )
    identity_map: dict[str, dict] = json.loads(map_path.read_text())
    by_user_id = {identity["user_id"]: identity for identity in identity_map.values()}
    identity = by_user_id.get(record.get("user_id"))
    if identity is None or not identity.get("character_clarifications"):
        raise ValueError(
            f"native spell-out: caller {record.get('user_id')!r} has no "
            f"character_clarifications payload in {map_path.name}."
        )
    full_name = compose_full_name(first, last, name_order_for_language(language))
    return render_native_name_spellout_block(
        first, last, full_name, list(identity["character_clarifications"])
    )


def render_native_name_spellout_block(
    first_name: str,
    last_name: str,
    full_name: str,
    payload_lines: list[str],
) -> str:
    """The fixed directive block carrying one caller's spell-out payload.

    ``payload_lines`` is the per-identity ``character_clarifications`` payload
    from the native identity map (the reviewed artifact); the template is
    selected by the name's script.
    """
    lines = "\n".join(f"- {line}" for line in payload_lines)
    if is_han(first_name) and is_han(last_name):
        template = NATIVE_NAME_SPELLOUT_TEMPLATE_HAN_V1
    elif is_devanagari(first_name) and is_devanagari(last_name):
        template = NATIVE_NAME_SPELLOUT_TEMPLATE_DEVANAGARI_V1
    else:
        raise ValueError(
            f"native spell-out block: unsupported name script for {full_name!r}"
        )
    return template.format(full_name=full_name, clarification_lines=lines)
