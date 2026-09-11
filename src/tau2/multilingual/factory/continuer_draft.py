# Copyright Sierra
"""Give a shipped pack's personas a pure backchannel continuer
(``tau2 factory draft-continuers``).

The backchannel DECISION is LLM-gated, but the phrase is a uniform random
draw with zero context, so every entry in ``backchannel_phrases`` must be a
pure "I'm listening, keep going" continuer — the language's "mm-hmm". Packs
drafted before that rule mixed in full acknowledgments; most were fixed by
SELECTING the continuer already present in the pack's own authored material,
but a pack can have none anywhere (Italian: its lists hold only
acknowledgments and speaker-side discourse markers, and Italian's "mh mh" is
absent).

Inventing one inline is exactly the agent-improvised content the factory
forbids, so this verb does it the sanctioned way: one fixed, versioned prompt
that is shown ALL of the pack's authored continuer-candidate material and is
required to SELECT from it when anything qualifies, authoring the language's
canonical continuer only when nothing does — and to say which it did. The
reply records per-persona provenance, the write marks the content as pending
native review, and ``REVIEWED_CONTINUERS`` in the pack-invariant suite pins
the result so it cannot drift afterwards.

The write is SURGICAL: only each persona's ``backchannel_phrases`` block
(plus one provenance marker comment above ``personas:``) changes; the merged
pack is re-validated through ``LanguagePack`` before anything is written.
"""

from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.backchannel import MAX_BACKCHANNEL_PHRASES
from tau2.config import DEFAULT_FACTORY_MODEL
from tau2.multilingual.brevity import (
    BACKCHANNEL_INVENTORY,
    phrase_budget_label,
    phrase_counting_rule,
    phrase_inventory_problems,
)
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.backfill_lib import (
    draft_validated,
    load_shipped_pack,
    validate_merged_pack,
)
from tau2.multilingual.factory.draft_prompts import (
    BACKCHANNEL_CONTINUER_PURITY_RULE,
)
from tau2.multilingual.factory.llm import FactoryLLM, get_default_llm
from tau2.multilingual.factory.pack_text_edit import (
    insert_marker_before_personas,
    replace_persona_list_blocks,
    strip_marker_block,
)
from tau2.multilingual.invariants import SCRIPT_RANGES, script_regex
from tau2.utils.utils import prompt_sha256

# LLM call name (the observable contract for tests and llm logs).
DRAFT_CONTINUERS_CALL_NAME = "factory_draft_continuers"

CONTINUER_DRAFT_PROMPT_VERSION = "v1"

DRAFT_CONTINUERS_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory backchannel editor. You give each \
persona in one language pack a PURE listener continuer — the language's \
"mm-hmm" — preferring one already present in the pack's authored material \
and authoring the language's canonical continuer only when the pack contains \
none. You never port a continuer from another language, never transliterate \
one, and never pass off an acknowledgment as a continuer. You respond with \
exactly one JSON object and nothing else."""

# The selection-first contract. Held as a module constant so the provenance
# hash covers the actual instructions.
DRAFT_CONTINUERS_PROMPT_TEMPLATE = """\
Give every persona in the Tau-Voice {display_name} ('{language}') language \
pack a pure backchannel continuer.

## The binding backchannel rule

{purity_rule}

## Where the continuer must come from

- FIRST, look for a pure continuer in the pack's own authored material below
  — any persona's current list, or the language-level palettes. If one is
  there, SELECT it verbatim and mark that persona "selected". This is the
  normal case and it is strongly preferred: the material was written by
  someone briefed on this language.
- ONLY IF nothing in the material below is a pure continuer may you AUTHOR
  one. Then give the language's ordinary, canonical listener hum — the sound
  a native speaker actually makes on the phone while the other person is
  still talking — and mark that persona "authored". Do not coin anything
  novel, stylised or regional-specific, and do not transliterate another
  language's continuer into this script.
- When you AUTHOR, do NOT reach for a second form to tell the personas
  apart. Give every persona the SAME canonical hum unless the language
  genuinely has a second form of equal everyday currency — a rarer variant
  chosen for variety is a worse continuer, and personas sharing one is the
  expected outcome. Prefer the language's own conventional spelling of the
  sound over one borrowed from English orthography.
- Judge the material as a LISTENER continuer, not by how short it is. A
  hesitation filler the speaker uses inside their OWN turn (Italian "eh",
  "mah", "boh"; English "um") is not a continuer. An acknowledgment is not a
  continuer however brief ("ok", "sì", "va bene").

## Shape

- Each persona gets at most {max_phrases} entries; one is fine.
- Each entry is at most {budget_label} — a continuer is a hum, not an
  utterance. The mechanical checker that will reject your reply counts it
  this way: {counting_rule}.
- Write it in the pack's script ('{script}'), spelled the way that language
  conventionally writes the sound.

## The pack's authored material

Current backchannel lists (these are what you are replacing — they are known
to be contaminated with acknowledgments):
{current_lists}

Language-level palettes from the localization block:
{palettes}

## The personas

{persona_lines}

Respond with ONE JSON object and nothing else. For every persona id above:
{{"personas": {{"<persona_id>": {{"phrases": [<1-{max_phrases} strings>], \
"provenance": "selected" | "authored", "note": "<one short sentence: which \
material you selected it from, or why nothing qualified and what the \
authored form is>"}}}}}}"""


def _render_material(entries: list[tuple[str, object]]) -> str:
    """Bullet lines for a labelled block of the pack's authored material."""
    if not entries:
        return "  (none)"
    return "\n".join(f"  - {label}: {value}" for label, value in entries)


def build_draft_continuers_prompt(
    *,
    language: str,
    display_name: str,
    script: str,
    personas: dict[str, dict],
    palettes: dict[str, list[str]],
) -> str:
    """The fixed continuer prompt with its typed slots filled.

    The model sees EVERY continuer-candidate surface at once — both personas'
    current lists and the language-level palettes — because the rule is
    select-first and it cannot honour that without seeing what there is to
    select from.
    """
    current_lists = _render_material(
        [
            (persona_id, persona.get("backchannel_phrases") or [])
            for persona_id, persona in personas.items()
        ]
    )
    palette_lines = _render_material(sorted(palettes.items()))
    persona_lines = "\n".join(
        f"- {persona_id} ({persona.get('display_name', persona_id)}; "
        f"{persona.get('short_description', '')})"
        for persona_id, persona in personas.items()
    )
    return DRAFT_CONTINUERS_PROMPT_TEMPLATE.format(
        display_name=display_name,
        language=language,
        purity_rule=BACKCHANNEL_CONTINUER_PURITY_RULE,
        max_phrases=MAX_BACKCHANNEL_PHRASES,
        budget_label=phrase_budget_label(language, BACKCHANNEL_INVENTORY),
        counting_rule=phrase_counting_rule(language, BACKCHANNEL_INVENTORY),
        script=script,
        current_lists=current_lists,
        palettes=palette_lines,
        persona_lines=persona_lines,
    )


def continuer_draft_prompt_sha256() -> str:
    """sha256 of the fixed prompt material — recordable in provenance."""
    return prompt_sha256(
        DRAFT_CONTINUERS_SYSTEM_PROMPT,
        BACKCHANNEL_CONTINUER_PURITY_RULE,
        DRAFT_CONTINUERS_PROMPT_TEMPLATE,
        CONTINUER_DRAFT_PROMPT_VERSION,
    )


class PersonaContinuers(BaseModel):
    """One persona's continuer inventory and where it came from."""

    phrases: list[str] = Field(description="One or two pure continuers")
    provenance: Literal["selected", "authored"] = Field(
        description="'selected' from the pack's own material, or 'authored' "
        "because the pack contained none"
    )
    note: str = Field(default="", description="One sentence of justification")


def _reply_problems(
    reply: object,
    *,
    persona_ids: list[str],
    language: str,
    script: Optional[str],
) -> list[str]:
    """Contract violations in a continuer reply.

    Deterministic checks only — coverage, the inventory bounds, and the
    script. Whether a phrase is genuinely a pure CONTINUER is a judgement no
    checker can make; that is what the pinned reviewed-continuer catalog and
    a native reviewer are for.
    """
    if not isinstance(reply, dict):
        return ["reply must be a JSON object keyed by persona id"]
    missing = [pid for pid in persona_ids if pid not in reply]
    if missing:
        return [f"reply is missing persona(s) {missing}"]
    extra = [pid for pid in reply if pid not in persona_ids]
    if extra:
        return [f"reply carries unknown persona(s) {extra}"]

    problems: list[str] = []
    script_re = script_regex(script) if script in SCRIPT_RANGES else None
    for persona_id in persona_ids:
        entry = reply[persona_id]
        if not isinstance(entry, PersonaContinuers):
            problems.append(f"persona '{persona_id}' entry is malformed")
            continue
        phrases = entry.phrases
        if not phrases:
            problems.append(f"persona '{persona_id}' returned no continuer")
            continue
        problems.extend(
            phrase_inventory_problems(
                persona_id,
                phrases,
                inventory=BACKCHANNEL_INVENTORY,
                language=language,
                script=script,
            )
        )
        for phrase in phrases:
            if not phrase.strip():
                problems.append(f"persona '{persona_id}' returned an empty continuer")
            elif script_re and not script_re.search(phrase):
                problems.append(
                    f"persona '{persona_id}' continuer {phrase!r} is not in the "
                    f"pack's script ('{script}')"
                )
    return problems


def _parse_reply(data: object) -> dict[str, PersonaContinuers]:
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    personas = data.get("personas")
    if not isinstance(personas, dict):
        raise ValueError("reply has no 'personas' object")
    return {
        persona_id: PersonaContinuers.model_validate(entry)
        for persona_id, entry in personas.items()
    }


def draft_pack_continuers(
    *,
    language: str,
    display_name: str,
    personas: dict[str, dict],
    palettes: dict[str, list[str]],
    llm: Optional[FactoryLLM] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> dict[str, PersonaContinuers]:
    """One fixed-prompt LLM call -> a continuer inventory for every persona.

    ONE call for the whole pack, not one per persona: the select-first rule
    is pack-wide (a continuer in persona A's list is material persona B may
    select from), so the model has to see every surface at once to apply it.
    """
    llm = llm if llm is not None else get_default_llm()
    model = model or DEFAULT_FACTORY_MODEL
    persona_ids = list(personas)
    scripts = {p.get("script") for p in personas.values() if p.get("script")}
    if len(scripts) != 1:
        raise FactoryDraftError(
            f"pack '{language}' personas declare scripts {sorted(scripts)}; "
            "the continuer prompt needs exactly one"
        )
    script = scripts.pop()

    prompt = build_draft_continuers_prompt(
        language=language,
        display_name=display_name,
        script=str(script),
        personas=personas,
        palettes=palettes,
    )
    return draft_validated(
        llm,
        model,
        system=DRAFT_CONTINUERS_SYSTEM_PROMPT,
        user=prompt,
        call_name=DRAFT_CONTINUERS_CALL_NAME,
        parse=_parse_reply,
        problems=lambda candidate: _reply_problems(
            candidate, persona_ids=persona_ids, language=language, script=script
        ),
        subject=language,
        failure_noun="continuer inventory",
        reasoning_effort=reasoning_effort,
    )


# =============================================================================
# The `tau2 factory draft-continuers` entry point
# =============================================================================

_MARKER_PREFIX = "# --- persona backchannel continuers"


def _render_marker(*, model: str, provenance: dict[str, str]) -> str:
    summary = ", ".join(f"{pid}: {kind}" for pid, kind in sorted(provenance.items()))
    return (
        f"{_MARKER_PREFIX} (pure continuers) ---\n"
        "# backchannel_phrases set by `tau2 factory draft-continuers` "
        f"(prompt {CONTINUER_DRAFT_PROMPT_VERSION} "
        f"sha256:{continuer_draft_prompt_sha256()[:12]}, model:{model}).\n"
        f"# Provenance — {summary}.\n"
        "# The phrase is a uniform random draw with no context, so every entry\n"
        "# must be a PURE continuer; acknowledgments live in the localization\n"
        "# block's conversational_confirmations palette instead.\n"
        "# Native content pending per-language owner review.\n"
    )


class ContinuerDraftOutcome(BaseModel):
    """What ``run_draft_continuers`` did, for the CLI and tests."""

    language: str
    pack_path: Path
    written: bool = Field(
        description="False when skipped (already drafted; --force to redo)"
    )
    personas: dict[str, PersonaContinuers] = Field(default_factory=dict)
    prompt_version: str = CONTINUER_DRAFT_PROMPT_VERSION
    prompt_sha256: str = Field(default_factory=continuer_draft_prompt_sha256)
    model: str = ""


def _pack_palettes(pack_data: dict) -> dict[str, list[str]]:
    """The language-level continuer-candidate palettes, if the pack has them."""
    localization = pack_data.get("localization")
    if not isinstance(localization, dict):
        return {}
    examples = localization.get("guideline_examples")
    if not isinstance(examples, list):
        return {}
    wanted = {"conversational_confirmations", "disfluency_fillers"}
    return {
        entry["kind"]: entry.get("utterances") or []
        for entry in examples
        if isinstance(entry, dict) and entry.get("kind") in wanted
    }


def run_draft_continuers(
    lang: str,
    *,
    force: bool = False,
    llm: Optional[FactoryLLM] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> ContinuerDraftOutcome:
    """Replace a SHIPPED pack's backchannel inventory with pure continuers.

    Reuse-by-default: a pack whose ``personas:`` block already carries the
    continuer marker is skipped unless ``force`` is set. Only the
    ``backchannel_phrases`` blocks and the provenance marker change; the
    merged pack is re-validated through ``LanguagePack`` before the write.
    """
    pack = load_shipped_pack(lang)
    pack_path, original = pack.pack_path, pack.original
    personas = pack.personas
    if not personas:
        raise FactoryDraftError(f"pack '{lang}' has no personas")

    resolved_model = model or DEFAULT_FACTORY_MODEL
    if _MARKER_PREFIX in original and not force:
        logger.info(
            f"pack '{lang}' continuers already drafted; skipping (--force to redo)"
        )
        return ContinuerDraftOutcome(
            language=lang, pack_path=pack_path, written=False, model=resolved_model
        )

    drafted = draft_pack_continuers(
        language=lang,
        display_name=pack.display_name,
        personas=personas,
        palettes=_pack_palettes(pack.data),
        llm=llm,
        model=resolved_model,
        reasoning_effort=reasoning_effort,
    )

    base_text = strip_marker_block(original, _MARKER_PREFIX)
    new_text = replace_persona_list_blocks(
        base_text,
        key="backchannel_phrases",
        new_by_persona={pid: entry.phrases for pid, entry in drafted.items()},
    )
    new_text = insert_marker_before_personas(
        new_text,
        _render_marker(
            model=resolved_model,
            provenance={pid: entry.provenance for pid, entry in drafted.items()},
        ),
    )

    # Validate the MERGED pack before writing anything (loud, not best-effort).
    validate_merged_pack(new_text, pack_path.parent)

    pack_path.write_text(new_text)
    logger.info(
        f"drafted continuers for '{lang}': "
        + ", ".join(f"{pid}={entry.phrases}" for pid, entry in drafted.items())
    )
    return ContinuerDraftOutcome(
        language=lang,
        pack_path=pack_path,
        written=True,
        personas=drafted,
        model=resolved_model,
    )
