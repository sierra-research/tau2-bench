# Copyright Sierra
"""LLM drafting of a pack's ``localization:`` block (spoken-localization data).

One fixed, versioned prompt authors the per-language content for
``LocalizationPackConfig`` — native symbol readouts, worked examples for the
single-language date/numeral rule, the domain-term glossary, and the speech
conventions (phone-number readout, amount readout, spelling alphabet) —
validated against the closed catalogs in
``tau2.multilingual.localization_catalog``.

Two entry points share the prompt:

- ``draft_localization`` — called by the drafting pipeline (``tau2 factory
  draft``) so NEW languages get the block at draft time; and
- ``run_backfill_localization`` — the ``tau2 factory draft-localization`` verb
  that backfills a SHIPPED pack. A pack with NO block gets a fresh one
  appended (only the new top-level ``localization:`` key is written, so
  concurrent edits to other pack keys rebase trivially). A pack whose block
  predates the speech-convention fields or the guideline example palettes is
  UPGRADED in place: the missing fields are drafted (grounded on the pack's
  localized voice guidelines + the personas' backchannel phrases, so the
  drafted content agrees with the reviewed native content) and merged in,
  while every already-reviewed field keeps its exact existing values.

An in-place edit of an EXISTING block is spliced field by field
(``pack_text_edit.set_nested_block``), not re-dumped: a shipped block carries
authored review notes as YAML comments — measured-defect rationale, PENDING
NATIVE REVIEW markers — and a whole-block re-dump would delete them.
"""

from pathlib import Path
from typing import Optional, Sequence

import yaml
from loguru import logger
from pydantic import Field

from tau2.config import DEFAULT_FACTORY_MODEL, DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.domain_profiles import get_domain_profile
from tau2.multilingual.factory.author_form import (
    FactoryDraftError,
    render_field_docs,
)
from tau2.multilingual.factory.backfill_lib import (
    BackfillOutcome,
    ShippedPack,
    draft_validated,
    load_shipped_pack,
    parse_existing_block,
    render_backfill_block,
    render_backfill_provenance,
    write_backfill_block,
    write_pack_text,
)
from tau2.multilingual.factory.llm import FactoryLLM, get_default_llm
from tau2.multilingual.factory.pack_text_edit import (
    insert_marker_before_key,
    set_nested_block,
    strip_marker_block,
)
from tau2.multilingual.localization_catalog import (
    GUIDELINE_EXAMPLE_CATALOG,
    SYMBOL_CATALOG,
    get_domain_term_catalog,
)
from tau2.multilingual.schema import (
    AmountReadout,
    DomainTermGloss,
    GuidelineExample,
    LocalizationPackConfig,
    PhoneNumberReadout,
    SpellingAlphabet,
    SpokenFormExample,
    SpokenValueExample,
    SpokenValueKind,
    SymbolReadout,
)
from tau2.utils.utils import prompt_sha256

# LLM call name (the observable contract for tests and llm logs).
DRAFT_LOCALIZATION_CALL_NAME = "factory_draft_localization"

# v2 added the speech conventions (phone_number_readout, amount_readout,
# spelling_alphabet), the worked spoken-value examples, the localized-
# guidelines grounding slot, and the persona-neutrality contract.
# v3 adds the guideline example palettes (guideline_examples, grounded on the
# localized guidelines + the personas' backchannel phrases) and the email/URL
# comma convention (values flow as words; comma+space only for
# character-by-character spelling).
# v4 makes the drafted glossary per-domain: the prompt carries ONE domain's
# term catalog + call-subject grounding (from the domain profile) and the
# drafted terms are filed under `domain_glossaries.<domain>`; the previously
# airline-specific prompt wording is domain-parameterized.
# v5 adds the gendered guideline-example variant (`male_utterances`) for the
# languages that mark speaker gender in these mechanics — grounded in the ja
# review's [Gender] finding that one shared palette reads as a woman's speech.
# v6 splits the two acknowledgment surfaces that v3 had collapsed:
# conversational_confirmations is the BROAD, context-chosen turn-level palette
# and must contain (not shrink to) the personas' one-or-two PURE continuers,
# which are a separate uniform-random-draw channel (see
# tau2.backchannel.MAX_BACKCHANNEL_PHRASES).
# v7 pins the glossary `note` LANGUAGE to English (native forms quoted inside).
# The notes render into the unconditionally-English prompt scaffold, and 7 of
# the 12 shipped two-domain packs had drifted to target-language notes in ONE
# of their two domains — a split with no rule to appeal to. Also restates
# persona-neutrality for notes, which several packs violated by naming the
# casual persona ("casual (Lena): oft ...").
LOCALIZATION_PROMPT_VERSION = "v7"

DRAFT_LOCALIZATION_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory localization author. For one language \
you produce the SPOKEN-LOCALIZATION data of its language pack: how a native \
caller voices technical symbols, dates, numbers, phone numbers, money \
amounts, spelled-out names/codes, and the service-domain vocabulary on a \
phone call. You select ONLY from the closed catalogs you are given, you \
author every native form idiomatically in the target language, and you \
respond with exactly one JSON object and nothing else."""


def _render_symbol_catalog() -> str:
    lines = ["Closed symbol catalog (symbol — English name — where it appears):"]
    for symbol, s in SYMBOL_CATALOG.items():
        lines.append(f"- '{symbol}' — {s.english_name} — {s.description}")
    return "\n".join(lines)


def _render_domain_term_catalog(domain: str) -> str:
    lines = [
        f"Closed domain-term catalog for the '{domain}' domain "
        "(term_id — English term — meaning):"
    ]
    for term_id, t in get_domain_term_catalog(domain).items():
        lines.append(f"- {term_id} — '{t.english}' — {t.description}")
    return "\n".join(lines)


def _render_guideline_example_catalog() -> str:
    lines = [
        "Closed guideline-example catalog (kind — item count — what the "
        "native utterances must demonstrate — the English default they "
        "replace):"
    ]
    for kind_id, e in GUIDELINE_EXAMPLE_CATALOG.items():
        count = (
            str(e.min_items)
            if e.min_items == e.max_items
            else f"{e.min_items}-{e.max_items}"
        )
        english = (
            "; ".join(f'"{item}"' for item in e.english_items)
            or "(no English example today — the rule is stated bare)"
        )
        lines.append(f"- {kind_id} — {count} utterance(s) — {e.description}")
        lines.append(f"  English default: {english}")
    return "\n".join(lines)


def _render_localization_field_docs() -> str:
    """Schema docs generated from the validating models (cannot drift)."""
    return "\n\n".join(
        render_field_docs(model)
        for model in (
            LocalizationPackConfig,
            SymbolReadout,
            SpokenFormExample,
            DomainTermGloss,
            PhoneNumberReadout,
            AmountReadout,
            SpellingAlphabet,
            SpokenValueExample,
            GuidelineExample,
        )
    )


def build_localization_prompt(
    language: str,
    display_name: str,
    script: str,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    translation_guidance: Optional[str] = None,
    persona_summaries: Optional[list[str]] = None,
    speech_guidelines: Optional[str] = None,
    backchannel_phrases: Optional[list[str]] = None,
) -> str:
    """The fixed localization-drafting prompt with its typed slots filled.

    ``domain`` selects the closed term catalog the glossary drafts against
    and the call-subject grounding sentence (both from the domain profile).
    ``translation_guidance`` (the pack's author-written script/number/loanword
    norms) and the persona register summaries ground the loanword-vs-native
    decisions; ``speech_guidelines`` (the pack's localized voice-guidelines
    markdown, when it exists) grounds the speech conventions AND the guideline
    example palettes so the drafted content AGREES with the reviewed native
    conventions; ``backchannel_phrases`` (the pack personas' one-or-two PURE
    continuers) are shown so the conversational-confirmations palette
    CONTAINS them without collapsing to them — confirmations is the broad
    turn-level acknowledgment palette, the continuers are the narrow
    random-draw channel. All render a graceful placeholder when absent.
    """
    profile = get_domain_profile(domain)
    guidance = (translation_guidance or "").strip() or "(none authored)"
    personas = "\n".join(f"- {p}" for p in (persona_summaries or [])) or "(none)"
    guidelines = (speech_guidelines or "").strip() or "(none authored yet)"
    backchannels = (
        ", ".join(f'"{p}"' for p in (backchannel_phrases or [])) or "(none authored)"
    )
    return f"""\
Author the `localization` block for the Tau-Voice {display_name} \
('{language}', script '{script}') language pack: how a native {display_name} \
caller VOICES technical symbols, dates, numbers, phone numbers, money \
amounts, spelled-out names/codes, and the '{domain}' domain vocabulary on a \
phone call — plus the native guideline example palettes (disfluencies, \
confirmations, check-ins, don't-know responses, …) the behavioral \
guidelines demonstrate.

Domain grounding: {profile.vocabulary_context}

Respond with ONE JSON object and nothing else, with exactly these keys:
"symbol_readouts", "date_examples", "number_examples", "domain_glossary",
"phone_number_readout", "amount_readout", "spelling_alphabet",
"spoken_value_examples", "guideline_examples" — the JSON serialization of
the schema documented below.

## Schema (from the validating models)

{_render_localization_field_docs()}

## Closed catalogs — select ONLY these; never invent a symbol, term_id, or kind

{_render_symbol_catalog()}

{_render_domain_term_catalog(domain)}

{_render_guideline_example_catalog()}

## Requirements

- COVER THE FULL CATALOGS: one symbol_readouts entry for EVERY catalog symbol,
  one domain_glossary entry for EVERY catalog term_id, and one
  guideline_examples entry for EVERY catalog example kind (a validator rejects
  unknown ids; a coverage guard rejects missing ones).
- Every `spoken` / `native` / example-`spoken` / `utterances` value is
  authored NATIVELY in {display_name} (script '{script}'), exactly as a
  native says it on the phone. Never leave an English name as the readout
  unless that English form IS the established native usage — then say so in
  a `note`.
- Symbol readouts are the words a native SAYS for the symbol (e.g. German '_'
  -> "Unterstrich", French '_' -> "tiret du bas"), not descriptions. Give the
  most natural form first; add genuinely-used alternatives after it.
- date_examples: at least 2 entries; number_examples: at least 2 entries. Each
  pairs a written value from a task (a date like 'May 17', an amount
  like '253 dollars', a count like '21') with its FULLY-NATIVE spoken form —
  entirely in {display_name}, no English words mixed in, no run-together
  halves. These are the worked examples for the single-language consistency
  rule (the grounded failures are half-and-half readouts like 'May 십칠' and
  run-together forms like 'May21').
- domain_glossary `native` values are what a native caller actually SAYS: use
  the natural everyday term, not a stilted official coinage. When the natural
  usage is an established English loanword (possibly transliterated into the
  native script), use that loanword form and mark it in `note`.
- domain_glossary `note` values are written IN ENGLISH, with any {display_name}
  form they cite quoted in its own script ("established loanword; also
  '<native form>' in formal register"). The note is rendered into an
  English prompt line, so English prose keeps that line one language. A note
  may name a REGISTER ('casual register') but NEVER a persona or speaker.
- phone_number_readout: the `rule` states IN ENGLISH how a native groups and
  reads a local phone number aloud (grouping blocks, digit-by-digit vs.
  grouped numbers, when to fall back to digit-by-digit); the examples pair a
  written local-format number with its fully-native spoken form.
- amount_readout: the `rule` states IN ENGLISH how a native reads money
  amounts aloud (currency-word placement, the decimal-separator word,
  large-amount units); the examples pair written amounts (include one with a
  decimal part) with their fully-native spoken forms.
- spelling_alphabet: the `rule` states IN ENGLISH what a native does to
  disambiguate a misheard name or code (anchor words on native
  names/cities/words, or character decomposition where the language spells
  that way); `avoid` states IN ENGLISH the non-native convention to reject
  (e.g. the NATO/anglo "B for boy" alphabet, unless that IS the native
  convention); anchor_examples pair a written character with its native
  spoken anchor phrase.
- spoken_value_examples: EXACTLY one example per kind — "email", "user_id",
  "phone", "spelling_name", "account_number", "website". `written` is a
  locale-plausible value; `spoken` is the complete utterance a native says
  for it — symbols voiced per your symbol_readouts, the phone example
  following your phone_number_readout rule, the spelling_name example using
  your spelling_alphabet anchors. A plain neutral carrier phrase (the native
  "it's …" / "my account is …") is fine.
- COMMA CONVENTION for spoken_value_examples: emails and website addresses
  are read as a CONTINUOUS FLOW of words — never insert commas between the
  spoken tokens ('juan guion bajo perez arroba gmail punto com', NOT 'juan,
  guion bajo, perez, arroba, gmail, punto, com'). Comma-and-space separation
  applies ONLY where a value is spelled character by character (letters,
  digits, codes — 'A, B, one, two'). If the reviewed native guidelines below
  genuinely read emails/URLs character by character (e.g. Japanese katakana
  letter readouts), follow the guidelines — they win.
- guideline_examples: EXACTLY one entry per catalog kind, utterance counts
  within each kind's bounds. These are the target-language replacements for
  the English behavioral examples quoted in the catalog: author what a
  native ACTUALLY says on a service call, not literal translations of the
  English defaults. The pack's localized voice guidelines below already
  demonstrate native versions of ALL of these (disfluency palette, restarts,
  confirmations, silence check-ins, frustrated endings, don't-know
  responses, vague openers, …) — mine them and stay consistent with them.
- SPEAKER GENDER in guideline_examples: if the language marks the speaker's
  own gender in these mechanics (first-person pronouns, sentence-final
  particles, softeners, apology-word choice — Japanese ごめんなさい vs すみません),
  add `male_utterances` to the kinds where it genuinely shows. It is the
  male-speaker realization of the SAME utterance: parallel item-for-item,
  same situation, same length and tone, differing ONLY where the language
  differs by gender — not a second style. Omit it on every kind (and in
  every language) where a native would say the identical thing regardless of
  gender; a copy of `utterances` is rejected. This is grammar/usage, NOT
  affect: it must not smuggle in a personality.
- conversational_confirmations is the LANGUAGE-LEVEL TURN-of-speech palette:
  the full range of acknowledgments a native says when it IS their turn —
  "I see", "understood", "right", "okay", plus the pure continuers. It is
  chosen IN CONTEXT by the simulator, so breadth here is correct and wanted.
  It must CONTAIN the personas' backchannel continuers listed below (they are
  the same list family), but must NOT collapse to them: the personas'
  `backchannel_phrases` are a deliberately narrow one-or-two-entry channel
  emitted by UNIFORM RANDOM DRAW while the agent is still speaking, which is
  why acknowledgments are banned there and live here instead. Do not invent a
  divergent second inventory.
- PERSONA-ATTITUDE NEUTRALITY, in EVERY field: language mechanics and
  speech-pattern palettes — yes; personality, affect prescriptions, or
  politeness dressing — no. No personality adjectives ("friendly", "casual",
  "polite"), no fixed-mood phrasing: attitude belongs to the TASK persona
  elsewhere in the prompt stack. The guideline_examples palettes naturally
  CONTAIN the language's hesitation fillers and discourse particles — that
  is their purpose — but the spoken_value_examples readouts stay
  filler-free ("um"/"uh" or {display_name} equivalents never appear there).
- Keep glossary terms consistent with the pack's translation guidance and the
  personas' registers below.
- The pack's localized voice guidelines below are REVIEWED native content:
  where they already state a symbol readout, phone grouping, amount reading,
  spelling-anchor convention, or an example palette, your drafted content
  MUST agree with it (same grouping, same anchor style, same currency
  reading, same filler family) — never contradict it.

## The pack's translation guidance (authored norms — stay consistent with it)

{guidance}

## The pack's personas (register grounding)

{personas}

## The pack personas' pure continuers (must APPEAR IN, not replace, confirmations)

{backchannels}

## The pack's localized voice guidelines (reviewed conventions — must agree)

{guidelines}

Now produce the JSON object."""


def localization_prompt_sha256() -> str:
    """sha256 of the fixed prompt template — recordable in provenance."""
    return prompt_sha256(DRAFT_LOCALIZATION_SYSTEM_PROMPT, LOCALIZATION_PROMPT_VERSION)


# The speech-convention fields added by prompt v2. The backfill's upgrade path
# drafts exactly these for a pack whose reviewed block predates them (an
# unset field is None for the singular models, [] for the examples list).
SPEECH_CONVENTION_FIELDS = (
    "phone_number_readout",
    "amount_readout",
    "spelling_alphabet",
    "spoken_value_examples",
)

# Everything the backfill's upgrade path can draft into an existing block:
# the v2 speech conventions plus the v3 guideline example palettes.
UPGRADE_FIELDS = SPEECH_CONVENTION_FIELDS + ("guideline_examples",)


def _parse_drafted(domain: str):
    """Parser for the drafted JSON: the LLM emits a FLAT ``domain_glossary``
    list (one domain per drafting call); it is filed under
    ``domain_glossaries.<domain>`` before schema validation."""

    def parse(raw: dict) -> LocalizationPackConfig:
        data = dict(raw)
        glossary = data.pop("domain_glossary", None)
        if glossary is not None:
            data["domain_glossaries"] = {domain: glossary}
        return LocalizationPackConfig.model_validate(data)

    return parse


def _full_coverage_problems(config: LocalizationPackConfig, domain: str) -> list[str]:
    """Catalog-coverage gaps (the schema validates ids; this checks fullness)."""
    problems: list[str] = []
    have_symbols = {r.symbol for r in config.symbol_readouts}
    missing_symbols = sorted(set(SYMBOL_CATALOG) - have_symbols)
    if missing_symbols:
        problems.append(f"missing symbol_readouts for symbols: {missing_symbols}")
    have_terms = {g.term_id for g in config.domain_glossaries.get(domain, [])}
    missing_terms = sorted(set(get_domain_term_catalog(domain)) - have_terms)
    if missing_terms:
        problems.append(f"missing domain_glossary entries for: {missing_terms}")
    if len(config.date_examples) < 2:
        problems.append("need at least 2 date_examples")
    if len(config.number_examples) < 2:
        problems.append("need at least 2 number_examples")
    for field in SPEECH_CONVENTION_FIELDS:
        if not getattr(config, field):
            problems.append(f"missing {field}")
    have_kinds = {example.kind for example in config.spoken_value_examples}
    missing_kinds = sorted(k.value for k in set(SpokenValueKind) - have_kinds)
    if config.spoken_value_examples and missing_kinds:
        problems.append(f"missing spoken_value_examples kinds: {missing_kinds}")
    have_example_kinds = {example.kind for example in config.guideline_examples}
    missing_example_kinds = sorted(set(GUIDELINE_EXAMPLE_CATALOG) - have_example_kinds)
    if missing_example_kinds:
        problems.append(f"missing guideline_examples kinds: {missing_example_kinds}")
    return problems


def draft_localization(
    language: str,
    display_name: str,
    script: str,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    translation_guidance: Optional[str] = None,
    persona_summaries: Optional[list[str]] = None,
    speech_guidelines: Optional[str] = None,
    backchannel_phrases: Optional[list[str]] = None,
    llm: Optional[FactoryLLM] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> LocalizationPackConfig:
    """One fixed-prompt LLM call -> a validated ``LocalizationPackConfig``.

    The drafted glossary covers ``domain``'s closed term catalog and is filed
    under ``domain_glossaries.<domain>``. The reply is parsed and
    ``model_validate``d (unknown symbols/term_ids are rejected by the
    schema's catalog validators) and checked for FULL catalog coverage +
    speech-convention presence through ``backfill_lib.draft_validated``.
    Invalid output raises a loud ``FactoryDraftError`` without a hidden
    re-prompt.
    """
    llm = llm if llm is not None else get_default_llm()
    model = model or DEFAULT_FACTORY_MODEL
    prompt = build_localization_prompt(
        language,
        display_name,
        script,
        domain=domain,
        translation_guidance=translation_guidance,
        persona_summaries=persona_summaries,
        speech_guidelines=speech_guidelines,
        backchannel_phrases=backchannel_phrases,
    )
    return draft_validated(
        llm,
        model,
        system=DRAFT_LOCALIZATION_SYSTEM_PROMPT,
        user=prompt,
        call_name=DRAFT_LOCALIZATION_CALL_NAME,
        parse=_parse_drafted(domain),
        problems=lambda config: _full_coverage_problems(config, domain),
        subject=language,
        failure_noun="valid localization block",
        reasoning_effort=reasoning_effort,
    )


# =============================================================================
# Backfill of shipped packs (`tau2 factory draft-localization`)
# =============================================================================


class LocalizationBackfillOutcome(BackfillOutcome):
    """What ``run_backfill_localization`` did, for the CLI and tests."""

    domain: str = DEFAULT_MULTILINGUAL_DOMAIN
    symbols: int = 0
    glossary_terms: int = 0
    prompt_version: str = LOCALIZATION_PROMPT_VERSION
    prompt_sha256: str = Field(default_factory=localization_prompt_sha256)


# The appended block's marker comment (the strip/splice matches on it) and
# the fixed header/comment lines around the rendered block.
_LOCALIZATION_MARKER = "# --- spoken localization"
_LOCALIZATION_HEADER = (
    f"{_LOCALIZATION_MARKER} (symbols / date-numeral rule / speech "
    "conventions / domain glossary)"
)
_LOCALIZATION_COMMENT = (
    "# Symbols and term_ids select from the closed catalogs in\n"
    "# tau2.multilingual.localization_catalog (the loader rejects unknown "
    "ids).\n"
    "# Native content pending per-language owner review.\n"
)


def _write_spliced_localization(
    pack: ShippedPack,
    config: LocalizationPackConfig,
    *,
    fields: Sequence[str],
    glossary_domain: Optional[str],
    descriptors: list[str],
    upgrade_action: str,
    model: str,
) -> None:
    """Splice ONLY the merge's changed fields into the pack's existing block.

    A shipped ``localization:`` block accumulates authored linguistic notes as
    YAML comments (a defect rationale above
    ``honorific_self_reference``, a ``PENDING NATIVE REVIEW`` marker, the
    native-language reasoning behind ``letter_name_tts``). Re-dumping the
    whole block would delete all of them, so each changed field is replaced
    on its own — ``fields`` by name, and the requested domain's glossary at
    ``domain_glossaries.<domain>`` so OTHER domains' glossaries keep their
    exact lines too. The provenance comment above the key is refreshed.

    The spliced text must parse back to exactly ``config``: a text edit that
    lost or moved content is a loud error, never a silent write.
    """
    payload = config.model_dump(mode="json", exclude_none=True)
    text = pack.original
    for field in fields:
        text = set_nested_block(
            text, path=("localization", field), new_value=payload[field]
        )
    if glossary_domain is not None:
        text = set_nested_block(
            text,
            path=("localization", "domain_glossaries", glossary_domain),
            new_value=payload["domain_glossaries"][glossary_domain],
        )

    text = strip_marker_block(text, _LOCALIZATION_MARKER)
    text = insert_marker_before_key(
        text,
        marker=f"{_LOCALIZATION_HEADER} ---\n"
        + render_backfill_provenance(
            verb="draft-localization",
            prompt_version=LOCALIZATION_PROMPT_VERSION,
            sha256=localization_prompt_sha256(),
            model=model,
            upgraded_fields=descriptors,
            upgrade_action=upgrade_action,
        )
        + _LOCALIZATION_COMMENT,
        key="localization",
    )

    spliced = LocalizationPackConfig.model_validate(
        yaml.safe_load(text)["localization"]
    )
    if spliced != config:
        raise FactoryDraftError(
            f"the surgical localization splice for '{pack.language}' did not "
            "reproduce the merged config; refusing to write a pack.yaml whose "
            "content does not match what was drafted"
        )
    write_pack_text(pack, text)


def _pack_speech_guidelines(data: dict) -> Optional[str]:
    """The pack's localized voice-guidelines markdown, or None.

    ``data`` is the loader-normalized pack view, so ``guidelines_voice_path``
    is already absolute; the text grounds the drafted speech conventions on
    the reviewed native content. A pack without the file (or the key) grounds
    on nothing.
    """
    rel = data.get("guidelines_voice_path")
    if not rel:
        return None
    path = Path(rel)
    if not path.exists():
        return None
    return path.read_text()


def run_backfill_localization(
    lang: str,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    force: bool = False,
    llm: Optional[FactoryLLM] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> LocalizationBackfillOutcome:
    """Draft the ``localization:`` block into a SHIPPED pack.yaml.

    Reuse-by-default, with three paths:

    - No ``localization`` key: draft a fresh block and APPEND only the new
      top-level key — every other line of pack.yaml stays byte-identical, so
      concurrent edits to other keys rebase trivially.
    - Existing block MISSING fields (the v2 speech conventions, the v3
      guideline example palettes, or the requested ``domain``'s glossary
      under ``domain_glossaries``): UPGRADE in place — the missing fields
      are drafted (grounded on the pack's localized voice guidelines + the
      personas' backchannel phrases) and merged in, while every
      already-reviewed field (including OTHER domains' glossaries) keeps its
      exact existing values; the rest of pack.yaml stays byte-identical.
    - Existing complete block: skipped unless ``force``, which RE-ROLLS the
      requested ``domain``'s glossary — and only that. The verb is
      domain-scoped, so ``force`` is too: the drafted symbol readouts, date/
      number examples, speech conventions and guideline palettes are
      discarded and every one of those reviewed fields keeps its exact
      existing value, exactly as on the upgrade path. (To re-roll a whole
      language's block, delete the ``localization:`` key and re-run — that is
      the fresh-draft path.)

    Both in-place paths SPLICE the changed fields into the existing block
    (``pack_text_edit.set_nested_block``) rather than re-dumping it, so the
    block's authored review comments — and every line of the fields that did
    not change — stay byte-identical. The spliced text must parse back to the
    merged config, and the merged pack is re-validated through
    ``LanguagePack``, before anything is written.
    """
    pack = load_shipped_pack(lang)
    pack_path, data = pack.pack_path, pack.data

    resolved_model = model or DEFAULT_FACTORY_MODEL
    existing = parse_existing_block(
        pack, key="localization", model_cls=LocalizationPackConfig
    )
    missing_fields = [
        field
        for field in UPGRADE_FIELDS
        if existing is not None and not getattr(existing, field)
    ]
    # The requested domain's glossary is per-domain state: missing means the
    # block predates the domain (other domains' glossaries don't count).
    glossary_missing = existing is not None and not existing.domain_glossaries.get(
        domain
    )
    if glossary_missing:
        missing_fields.append(f"domain_glossaries[{domain}]")
    if existing is not None and not missing_fields and not force:
        logger.info(
            f"pack '{lang}' already has a complete localization block "
            f"(incl. the '{domain}' glossary); skipping (--force to re-draft)"
        )
        return LocalizationBackfillOutcome(
            language=lang,
            domain=domain,
            pack_path=pack_path,
            written=False,
            model=resolved_model,
        )

    personas = pack.personas
    persona_summaries = [
        str(p.get("short_description", pid)) for pid, p in personas.items()
    ]
    backchannel_phrases = sorted(
        {
            str(phrase)
            for p in personas.values()
            for phrase in (p.get("backchannel_phrases") or [])
        }
    )

    drafted = draft_localization(
        lang,
        pack.display_name,
        pack.script,
        domain=domain,
        translation_guidance=data.get("translation_guidance"),
        speech_guidelines=_pack_speech_guidelines(data),
        persona_summaries=persona_summaries,
        backchannel_phrases=backchannel_phrases,
        llm=llm,
        model=resolved_model,
        reasoning_effort=reasoning_effort,
    )

    glossary_key = f"domain_glossaries[{domain}]"
    merging = existing is not None
    glossary_changed = merging and (glossary_missing or force)
    if merging:
        # Preserve every reviewed field verbatim; merge in ONLY the drafted
        # fields the block was missing — plus, under --force, a re-rolled
        # glossary for the REQUESTED domain. Other domains' glossaries and
        # every non-glossary field come from `existing` either way, so a
        # domain-scoped verb can never quietly re-roll a language's reviewed
        # symbol readouts, speech conventions or guideline palettes.
        update: dict = {
            field: getattr(drafted, field)
            for field in missing_fields
            if field != glossary_key
        }
        if glossary_changed:
            update["domain_glossaries"] = {
                **existing.domain_glossaries,
                domain: drafted.domain_glossaries[domain],
            }
        config = existing.model_copy(update=update)
    else:
        config = drafted

    descriptors = list(missing_fields)
    if force and glossary_key not in descriptors:
        descriptors.append(glossary_key)
    if merging:
        # In-place edit of a block that may carry authored review comments:
        # splice the changed fields, never re-dump the block.
        _write_spliced_localization(
            pack,
            config,
            fields=[f for f in missing_fields if f != glossary_key],
            glossary_domain=domain if glossary_changed else None,
            descriptors=descriptors,
            upgrade_action="re-drafted" if force else "drafted",
            model=resolved_model,
        )
    else:
        # First write: append the whole new top-level key.
        write_backfill_block(
            pack,
            key="localization",
            marker=_LOCALIZATION_MARKER,
            replacing=False,
            block=render_backfill_block(
                "localization",
                config,
                header=_LOCALIZATION_HEADER,
                verb="draft-localization",
                prompt_version=LOCALIZATION_PROMPT_VERSION,
                sha256=localization_prompt_sha256(),
                model=resolved_model,
                comment=_LOCALIZATION_COMMENT,
            ),
        )
    action = "re-drafted" if force and merging else "upgraded" if merging else "wrote"
    logger.info(
        f"{action} localization block for '{lang}' "
        f"({len(config.symbol_readouts)} symbols, "
        f"{len(config.domain_glossaries.get(domain, []))} '{domain}' "
        "glossary terms"
        + (f"; {action} {', '.join(descriptors)}" if merging else "")
        + ")"
    )
    return LocalizationBackfillOutcome(
        language=lang,
        domain=domain,
        pack_path=pack_path,
        written=True,
        replaced=force and merging,
        upgraded=merging,
        upgraded_fields=descriptors if merging else [],
        symbols=len(config.symbol_readouts),
        glossary_terms=len(config.domain_glossaries.get(domain, [])),
        model=resolved_model,
    )
