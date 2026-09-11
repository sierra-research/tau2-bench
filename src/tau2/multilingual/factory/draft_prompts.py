# Copyright Sierra
"""Prompt assembly for the drafting stage: exemplar rendering, schema/tag
docs, and the Call A/B/C system prompts + prompt builders."""

from pathlib import Path

import yaml

from tau2.multilingual.factory.author_form import (
    AuthorForm,
    FactoryDraftError,
)
from tau2.multilingual.schema import (
    LanguagePack,
    MultilingualPersonaConfig,
)
from tau2.multilingual.tags import REQUIRED_TAG_DIMENSIONS, TAG_VOCABULARY
from tau2.utils.utils import prompt_sha256

# Shipped packs embedded verbatim as few-shot exemplars. Deliberately
# anchored to the SOURCE TREE next to this module (unlike the dynamic
# validate/finalize target dir, which resolves through DATA_DIR): exemplars
# are prompt material from the shipped repo data, not validation targets,
# and must keep resolving even when tests point DATA_DIR at a temporary
# workspace — including when this module is first imported while such a
# patch is active.
EXEMPLAR_LANGUAGES = ("hi",)
# The guidelines (Call B) exemplar is a FULLY-TRANSLATED pack, separate from the
# persona (Call A) exemplar above: the guidelines must be authored entirely in
# the target language, so the exemplar has to demonstrate that (the `hi` pack
# keeps its instruction prose in English, which would model the wrong thing).
# `es` is fully localized end-to-end and audits clean.
GUIDELINES_EXEMPLAR_LANGUAGES = ("es",)
EXEMPLAR_PACKS_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "tau2" / "multilingual"
)


def render_tag_vocabulary() -> str:
    """The closed tag vocabulary plus the completeness requirement.

    Tags are analysis-slicing metadata only (NOT behavioral — real register
    lives in ``pragmatics_clauses``). The required dimensions must always be
    present, but optional dimensions are best omitted than force-fit: telling
    the model to OMIT an optional dimension when no listed value genuinely fits
    keeps the closed vocabulary clean instead of accreting near-misses (e.g. a
    gig/service driver getting force-labelled ``register: small_business``).
    """
    optional_dimensions = [
        dimension
        for dimension in TAG_VOCABULARY
        if dimension not in REQUIRED_TAG_DIMENSIONS
    ]
    lines = ["Closed persona-tag vocabulary (dimension: allowed values):"]
    for dimension, values in TAG_VOCABULARY.items():
        lines.append(f"- {dimension}: {values}")
    lines.append(
        "Every persona MUST carry ALL of these required tag dimensions: "
        f"{REQUIRED_TAG_DIMENSIONS}. The other dimensions "
        f"({optional_dimensions}) are OPTIONAL: include one only when a listed "
        "value genuinely fits the persona; if none fits, OMIT that dimension "
        "entirely — do NOT force-fit the nearest value. Never invent new "
        "dimensions or values: every key and value you emit must come from the "
        "lists above."
    )
    return "\n".join(lines)


def _exemplar_pack_text(language: str) -> str:
    pack_path = EXEMPLAR_PACKS_DIR / language / "pack.yaml"
    if not pack_path.exists():
        raise FactoryDraftError(
            f"exemplar pack for '{language}' not found at {pack_path}"
        )
    return pack_path.read_text()


def _exemplar_guidelines_body(language: str, key: str) -> str:
    """The exemplar pack's guidelines markdown for one declared path key.

    ``key`` is ``guidelines_voice_path`` (Call B's exemplar) or
    ``guidelines_text_path`` (Call C's). The declared value is PACK-RELATIVE
    and may name a subdirectory — shipped packs point both keys into their
    ``guidelines/`` subdir — so it is joined whole, never truncated to its
    basename.

    A missing or unreadable exemplar is a loud ``FactoryDraftError``, matching
    :func:`_exemplar_pack_text`: these bodies are grounding for fixed,
    versioned drafting prompts, and Call B's in particular grounds Call D (the
    LIVE localization block). Substituting a placeholder string would ship a
    silently degraded prompt.
    """
    data = yaml.safe_load(_exemplar_pack_text(language))
    rel = (data or {}).get(key) if isinstance(data, dict) else None
    if not rel:
        raise FactoryDraftError(
            f"exemplar pack '{language}' declares no {key} — the drafting "
            "prompt has no guidelines exemplar to ground on."
        )
    path = EXEMPLAR_PACKS_DIR / language / rel
    if not path.exists():
        raise FactoryDraftError(
            f"exemplar pack '{language}' declares {key}: '{rel}', but "
            f"{path} does not exist."
        )
    return path.read_text()


def _exemplar_guidelines_text(language: str) -> str:
    """The exemplar pack's VOICE guidelines markdown (Call B's exemplar)."""
    return _exemplar_guidelines_body(language, "guidelines_voice_path")


def _exemplar_text_guidelines_text(language: str) -> str:
    """The exemplar pack's TEXT guidelines markdown (Call C's exemplar)."""
    return _exemplar_guidelines_body(language, "guidelines_text_path")


# =============================================================================
# Call A — persona / voice content (+ translation_guidance), authored as JSON.
# These fields are linguistically coupled, so they MUST be authored together.
# The deterministic fields (experiment/clause/rates/acoustics) are NOT asked
# for here — pack_assembly generates them.
# =============================================================================

# Version of the persona-drafting prompt material (system prompt + the
# neutrality rule + the builder requirements). v1 was the pre-neutrality
# prompt whose personas prescribed fixed affect ("You open casually and
# friendly"); v2 adds PERSONA_ATTITUDE_NEUTRALITY_RULE: pack personas own the
# language profile, tasks own attitude, and every attitude-flavored clause is
# a conditional realization.
# v3: acoustic_preset_id wiring became conditional on the form's locale_beds
# opt-in — the default pack has no acoustic presets (shared English
# environments at run time) and the model must not emit the field at all.
# v4: backchannel_phrases became a PURE-CONTINUER channel (1-2 entries) —
# BACKCHANNEL_CONTINUER_PURITY_RULE. Earlier drafts authored 6-8 mixed
# phrases per persona, and since the phrase is drawn uniformly at random with
# no context, full acknowledgments ("I understand", "okay") landed mid-
# explanation and read to the agent as resolution.
DRAFT_PERSONAS_PROMPT_VERSION = "v4"

# The binding backchannel-inventory rule, shared verbatim by Call A
# (drafting), the Call A repair prompt, and the retired variety re-pin, so the
# contract cannot drift between drafting a new language and rewriting a
# shipped one.
BACKCHANNEL_CONTINUER_PURITY_RULE = """\
Backchannel inventory — PURE CONTINUERS ONLY (BINDING):
- "backchannel_phrases" holds ONE or TWO phrases, no more. They are the
  language's equivalent of English "mm-hmm" / "uh-huh": the brief sound a
  listener makes to say "I'm listening, keep going" and NOTHING else.
- A pure continuer is NOT an answer, NOT agreement, NOT "I understand" / "I
  see" / "got it", NOT "okay / fine / right", and NOT a closing signal.
  Forbidden shapes: 好的, 明白了, vale, de acuerdo, entiendo, tamam, anladım,
  alles klar, oczywiście, đúng rồi, 알겠습니다, ठीक है — every one of those is a
  turn-level response.
- WHY: only the TIMING of a backchannel is decided in context; the phrase
  itself is drawn UNIFORMLY AT RANDOM from this list, so any entry can land at
  any eligible moment with zero context. An acknowledgment drawn mid-
  explanation reads to the agent as "the user is done / the issue is
  resolved".
- The rich acknowledgment palette is NOT lost: it belongs in the
  language-level `conversational_confirmations` guideline examples, where the
  simulator picks a phrase IN CONTEXT at turn level.
- Where the language has two distinct continuer forms (e.g. a single vs. a
  doubled hum), give the two personas different ones; never at the cost of
  purity — two personas sharing the same continuer is correct and expected."""

# The binding persona content rule, shared verbatim by Call A (drafting), the
# Call A repair prompt, and the retired pragmatics redraft so the contract
# cannot drift between drafting a new language and redrafting a shipped one.
PERSONA_ATTITUDE_NEUTRALITY_RULE = """\
Persona content model — who owns what (BINDING):
- Every task assigns the caller's ATTITUDE/AFFECT for that scenario (impatient,
  friendly, frustrated, chatty, hurried, patient). The pack persona is layered
  into the SAME prompt, so a persona that prescribes a fixed attitude
  contradicts the task.
- The pack persona owns the LANGUAGE PROFILE only: dialect/variety realization
  (pronunciation traits, voseo, distinción/seseo), the address system (T-V
  forms such as usted/vos/tú and when they shift), the register/formality AXIS
  (e.g. 'high formality, usted at all times, low code-switch' is a language
  profile and STAYS), code-switching proportion and which domains it covers,
  letter names and spelling anchors, English-tolerance behavior, orthography,
  and characteristic discourse forms.
- pragmatics_clauses must NEVER assert an attitude, mood, pace, or temperament
  as the persona's fixed nature. Forbidden shapes: 'You open casually and
  friendly', 'You never rush the agent', 'Your courtesy is unwavering',
  'Lively interjections', 'Your closings are short and brisk — you do not
  linger', 'You stay composed'. No affect adjectives (friendly, warm, warmly,
  lively, cheerful, patient, impatient, composed, brisk, energetic, chatty) as
  prescriptions.
- Cultural and linguistic realization knowledge is VALUABLE — keep all of it,
  but write it as a CONDITIONAL REALIZATION of an attitude the task may
  assign: 'When you open casually, natural forms are «Hola, ¿qué tal?»,
  «Buenas, mirá, te llamo por una cosa»'; 'When frustrated, a high-register
  Peninsular speaker becomes MORE formal and insistent, not rude — longer
  polite preambles, repeated «por favor»'; 'If you are unhurried and
  courteous, closings extend across several turns: …'.
- Do not restate the tts_voice_prompt's voice-design affect (warm/composed
  timbre, measured pacing) in pragmatics_clauses: the voice prompt is TTS
  voice-design material and never reaches the behavioral prompt."""

DRAFT_PERSONAS_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory persona author. From an author form you \
produce the LINGUISTICALLY-COUPLED persona content for a new language: the \
personas (language profile, code-switching, conditional attitude realizations, \
phrases, voice-identity prompts, tags) plus the translation guidance — all \
authored together so they cohere. You write persona content natively and \
idiomatically in the target language, with inline native-script examples \
inside English instructions, exactly like the exemplar packs. Pack personas \
own language mechanics; the TASK owns the caller's attitude — you never \
prescribe a fixed affect. You respond with exactly one JSON object and \
nothing else."""


def personas_prompt_sha256() -> str:
    """sha256 of the fixed Call A prompt material — recordable in provenance."""
    return prompt_sha256(
        DRAFT_PERSONAS_SYSTEM_PROMPT,
        PERSONA_ATTITUDE_NEUTRALITY_RULE,
        BACKCHANNEL_CONTINUER_PURITY_RULE,
        DRAFT_PERSONAS_PROMPT_VERSION,
    )


# Persona/identity fields the model is asked to author for Call A. Subset of
# MultilingualPersonaConfig that is CREATIVE — the deterministic
# acoustic_preset_id wiring is constrained to the scaffolded ids; voice_id is a
# human bottleneck (left unset, stripped if emitted).
_CREATIVE_PERSONA_FIELDS = (
    "persona_id",
    "display_name",
    "short_description",
    "language",
    "locale",
    "script",
    "verbosity",
    "interrupt_tendency",
    "pragmatics_clauses",
    "backchannel_phrases",
    "non_directed_phrases",
    "tts_voice_prompt",
    "acoustic_preset_id",
    "tags",
)


def _render_creative_persona_field_docs(include_acoustic: bool) -> str:
    """Field docs for just the CREATIVE persona fields (Call A's portion).

    ``acoustic_preset_id`` only exists on packs that opted into locale beds;
    the default pack has no presets to wire, so the field is omitted from the
    contract entirely.
    """
    lines = [f"{MultilingualPersonaConfig.__name__} (author these fields):"]
    model_fields = MultilingualPersonaConfig.model_fields
    for name in _CREATIVE_PERSONA_FIELDS:
        if name == "acoustic_preset_id" and not include_acoustic:
            continue
        field = model_fields.get(name)
        if field is None:
            continue
        requirement = "required" if field.is_required() else "optional"
        description = field.description or "(no description)"
        lines.append(f"- {name} ({requirement}): {description}")
    return "\n".join(lines)


def build_personas_prompt(form: AuthorForm, preset_ids: list[str]) -> str:
    """Call A's prompt: author the persona content + translation_guidance.

    Exemplar-driven (the hi pack verbatim) but scoped to the persona/
    translation portion. The acoustic preset ids are the deterministic
    scaffolding ids — the model only WIRES each persona to one of them, it does
    not author preset structure or file paths.
    """
    exemplars = "\n\n".join(
        f"### Exemplar pack.yaml ({language}) — VERBATIM (study its `personas:` "
        f"and `translation_guidance:`)\n```yaml\n{_exemplar_pack_text(language)}```"
        for language in EXEMPLAR_LANGUAGES
    )
    form_yaml = yaml.safe_dump(
        form.model_dump(mode="json"), sort_keys=False, allow_unicode=True
    )
    translation_doc = next(
        field.description
        for name, field in LanguagePack.model_fields.items()
        if name == "translation_guidance"
    )
    if preset_ids:
        acoustic_rule = f"""\
- `acoustic_preset_id`: set it to ONE of these scaffolded preset ids (do NOT
  invent ids or file paths): {preset_ids}. This is the ONLY acoustic choice —
  the persona's calling environment is IMPLIED by the bed you pick, so choose
  the bed that fits where the persona is most likely calling from: a mobile,
  out-and-about persona -> the *_outdoor_traffic id (street traffic); an
  at-home/indoor persona -> the *_household id (a TV news broadcast over kitchen
  ambience); a desk/office persona -> the *_office id. Do NOT emit a separate
  environment tag — there is none."""
    else:
        acoustic_rule = """\
- `acoustic_preset_id`: DO NOT emit it — this pack uses the shared benchmark
  environments (selected at run time), so personas carry no acoustic wiring
  and no environment tag."""
    return f"""\
Author the persona content and translation guidance for a Tau-Voice \
{form.display_name} ('{form.language}', script '{form.script}') language pack.

Respond with ONE JSON object and nothing else, with exactly these keys:
- "personas": a JSON LIST of persona objects (one per persona sketch in the
  form), each authored from the fields documented below;
- "translation_guidance": a single JSON string (prose, newlines allowed) —
  {translation_doc}
- "agent_greeting": a single JSON string — the localized agent opening greeting
  in {form.display_name} (native script).

## Persona fields to author (from the validating schema)

{_render_creative_persona_field_docs(include_acoustic=bool(preset_ids))}

## Persona content model (BINDING)

{PERSONA_ATTITUDE_NEUTRALITY_RULE}

{BACKCHANNEL_CONTINUER_PURITY_RULE}

Requirements:
- Author every persona NATIVELY and idiomatically in {form.display_name}, with
  inline native-script examples inside the English instructions, exactly like
  the exemplars.
- pragmatics_clauses follow the persona content model above: language profile
  as fixed facts, attitude realizations ONLY in conditional form ('When you
  open casually…', 'When frustrated…', 'If you are unhurried…').
- `persona_id` is `<name>_{form.language}_v1` (lowercase name from the form).
- `language` MUST be '{form.language}' and `script` MUST be '{form.script}' on
  every persona.
- `voice_id`: DO NOT emit it — voice selection is a human bottleneck (designed
  and auditioned through the telephony codec before pinning).
{acoustic_rule}

## Persona tags

{render_tag_vocabulary()}

## Exemplar packs

{exemplars}

## The author form (the content to draft from)

```yaml
{form_yaml}```

Now produce the JSON object."""


def build_personas_repair_prompt(
    form: AuthorForm, personas_json: str, problems: list[str]
) -> str:
    """Call A repair: the current persona JSON + the guardrail findings."""
    problem_lines = "\n".join(f"- {problem}" for problem in problems)
    acoustic_rule = (
        "- `acoustic_preset_id`: one of the scaffolded ids; do not invent ids/paths."
        if form.locale_beds
        else "- `acoustic_preset_id`: do not emit it (shared benchmark environments)."
    )
    return f"""\
Your authored {form.display_name} ('{form.language}') persona content failed the
deterministic guardrails. Fix EVERY problem listed and respond with the SAME
JSON object as before (keys "personas", "translation_guidance",
"agent_greeting"), complete and self-contained — not a diff.

## Guardrail problems

{problem_lines}

## Contract reminders

{render_tag_vocabulary()}

{PERSONA_ATTITUDE_NEUTRALITY_RULE}

{BACKCHANNEL_CONTINUER_PURITY_RULE}

- `language`='{form.language}' and `script`='{form.script}' on every persona.
{acoustic_rule}
- `voice_id`: do not emit it (human bottleneck).

## The current persona JSON

```json
{personas_json}```

Now produce the corrected JSON object."""


# =============================================================================
# Call B — derived guidelines markdown, authored GIVEN Call A's personas so the
# conventions stay consistent with the persona phrases. (Backchannel density is
# now a single `backchannel_level` knob on the form, not a drafted prompt.)
# =============================================================================

DRAFT_ARTIFACTS_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory artifact author. GIVEN the already- \
authored personas for a language, you derive the re-authored voice \
user-simulator guidelines (letter-anchoring/email/number conventions matching \
the personas). You respond with exactly one fenced code block and nothing \
else."""


def _personas_context_for_artifacts(
    personas: list[MultilingualPersonaConfig],
) -> str:
    """A compact view of Call A's personas to ground Call B's artifacts."""
    lines: list[str] = []
    for persona in personas:
        backchannels = persona.backchannel_phrases or []
        lines.append(f"- {persona.persona_id}: {persona.short_description}")
        if backchannels:
            lines.append(f"    backchannel phrases: {backchannels}")
    return "\n".join(lines) if lines else "(no personas authored)"


def build_artifacts_prompt(
    form: AuthorForm, personas: list[MultilingualPersonaConfig]
) -> str:
    """Call B's prompt: derive the guidelines markdown GIVEN personas.

    Sees Call A's personas so the guidelines conventions stay consistent with
    them — the cross-artifact coherence the single call gave for free.
    (Backchannel density is a `backchannel_level` knob on the form, not a
    drafted prompt, so Call B no longer authors a decision prompt.)
    """
    exemplar_artifacts = "\n\n".join(
        f"### Exemplar guidelines ({language}) — fully localized; match this\n"
        f"```markdown\n{_exemplar_guidelines_text(language)}```"
        for language in GUIDELINES_EXEMPLAR_LANGUAGES
    )
    return f"""\
Derive the voice user-simulator guidelines for a Tau-Voice {form.display_name} \
('{form.language}', script '{form.script}') language pack, GIVEN its personas.

Respond with EXACTLY ONE fenced code block and nothing else:
- a ```markdown block: the voice user-simulator guidelines written ENTIRELY in
  {form.display_name} (script '{form.script}').

## Language contract — FULLY localize, do not leave instructions in English

Write the WHOLE document in {form.display_name}: every section heading and
every instruction sentence, not only the examples. Translate the meaning and
force of each instruction faithfully — a strong imperative in the source stays
a strong imperative in {form.display_name}. Also localize the conventions
themselves: native spelling/letter-anchoring style, phone-number chunking, and
how emails and amounts are read aloud.

The ONLY things that stay verbatim (never translated, transliterated, or
re-scripted) are:
- the literal `<PERSONA_GUIDELINES>` slot, on its own line; and
- the ASCII control tokens (`###STOP###`, `###TRANSFER###`,
  `###OUT-OF-SCOPE###`).

The exemplar below is itself fully localized — match how completely it renders
the instructions in its own language. Do NOT copy any English instruction prose
into your output.

## The personas these guidelines must stay consistent with

{_personas_context_for_artifacts(personas)}

## Exemplar guidelines

{exemplar_artifacts}

Now produce the fenced block."""


def build_artifacts_repair_prompt(
    form: AuthorForm,
    personas: list[MultilingualPersonaConfig],
    guidelines_markdown: str,
    problems: list[str],
) -> str:
    """Call B repair: current guidelines + guardrail findings (personas fixed)."""
    problem_lines = "\n".join(f"- {problem}" for problem in problems)
    return f"""\
Your derived {form.display_name} ('{form.language}') guidelines failed the
deterministic guardrails. Fix EVERY problem listed and respond with the SAME
single ```markdown guidelines block as before, complete and self-contained —
not a diff.

## Guardrail problems

{problem_lines}

## The personas these guidelines must stay consistent with

{_personas_context_for_artifacts(personas)}

## Contract reminders

- Write the WHOLE document in {form.display_name} (script '{form.script}') —
  every heading and instruction, not just the examples. Do not leave any
  instruction prose in English.
- Keep ONLY the `<PERSONA_GUIDELINES>` slot and the ASCII control tokens
  (e.g. `###STOP###`) verbatim.

## The current guidelines markdown

```markdown
{guidelines_markdown}```

Now produce the corrected fenced block."""


# =============================================================================
# Call C — derived TEXT (half-duplex chat) guidelines markdown, authored GIVEN
# Call A's personas, in parallel to Call B's voice guidelines. Text guidelines
# drop the spoken-form conventions (number/letter anchoring, backchannels,
# silence handling) and cover typed register/script/punctuation instead. The
# harness is strictly turn-based (one user message per turn), so the guidelines
# must NOT instruct multi-message splitting.
# =============================================================================

DRAFT_TEXT_ARTIFACTS_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory artifact author. GIVEN the already- \
authored personas for a language, you derive the re-authored TEXT (typed chat) \
user-simulator guidelines: native written register, code-switching, and how \
numbers / dates / emails / booking codes are TYPED (not spoken). You respond \
with exactly one fenced code block and nothing else."""


def build_text_artifacts_prompt(
    form: AuthorForm, personas: list[MultilingualPersonaConfig]
) -> str:
    """Call C's prompt: derive the TEXT guidelines markdown GIVEN personas.

    Parallel to ``build_artifacts_prompt`` (voice) but for turn-based typed
    chat: no spoken-form conventions, no multi-message splitting. Anchored on
    the English text guidelines contract (the ``<PERSONA_GUIDELINES>`` slot +
    ASCII control tokens), fully localized like the exemplar.
    """
    exemplar_artifacts = "\n\n".join(
        f"### Exemplar TEXT guidelines ({language}) — fully localized; match this\n"
        f"```markdown\n{_exemplar_text_guidelines_text(language)}```"
        for language in GUIDELINES_EXEMPLAR_LANGUAGES
    )
    return f"""\
Derive the TEXT (typed chat) user-simulator guidelines for a Tau-Voice \
{form.display_name} ('{form.language}', script '{form.script}') language pack, \
GIVEN its personas. This is a TEXT chat channel, NOT a voice call.

Respond with EXACTLY ONE fenced code block and nothing else:
- a ```markdown block: the text user-simulator guidelines written ENTIRELY in
  {form.display_name} (script '{form.script}').

## What TEXT guidelines must and must NOT contain

- DO cover, natively for {form.display_name}: the written register a real
  customer types (formality, politeness, code-switching / loanword habits),
  and how numbers, dates, emails, and booking codes are TYPED (native script vs
  Latin digits, spacing, @ / . written literally — NOT spelled out as words).
- DO reflect persona-appropriate typing habits (punctuation, capitalization,
  common typos/abbreviations, emoji only if it fits the persona).
- DO NOT include any VOICE/spoken conventions: no "spell it out", no
  number/letter anchoring ("J, O, H, N"), no backchannels ("uh-huh"), no
  disfluencies/pauses, no handling of agent silence.
- DO NOT instruct the user to split a message across multiple turns: the
  channel is strictly turn-based (one user message per turn).

## Language contract — FULLY localize, do not leave instructions in English

Write the WHOLE document in {form.display_name}: every section heading and
every instruction sentence, not only the examples. Translate the meaning and
force of each instruction faithfully.

The ONLY things that stay verbatim (never translated, transliterated, or
re-scripted) are:
- the literal `<PERSONA_GUIDELINES>` slot, on its own line; and
- the ASCII control tokens (`###STOP###`, `###TRANSFER###`,
  `###OUT-OF-SCOPE###`).

The exemplar below is itself fully localized — match how completely it renders
the instructions in its own language. Do NOT copy any English instruction prose
into your output, and do NOT copy the VOICE guidelines (this is a distinct
text-chat document).

## The personas these guidelines must stay consistent with

{_personas_context_for_artifacts(personas)}

## Exemplar TEXT guidelines

{exemplar_artifacts}

Now produce the fenced block."""


def build_text_artifacts_repair_prompt(
    form: AuthorForm,
    personas: list[MultilingualPersonaConfig],
    guidelines_markdown: str,
    problems: list[str],
) -> str:
    """Call C repair: current text guidelines + guardrail findings."""
    problem_lines = "\n".join(f"- {problem}" for problem in problems)
    return f"""\
Your derived {form.display_name} ('{form.language}') TEXT guidelines failed the
deterministic guardrails. Fix EVERY problem listed and respond with the SAME
single ```markdown guidelines block as before, complete and self-contained —
not a diff.

## Guardrail problems

{problem_lines}

## The personas these guidelines must stay consistent with

{_personas_context_for_artifacts(personas)}

## Contract reminders

- Write the WHOLE document in {form.display_name} (script '{form.script}') —
  every heading and instruction, not just the examples.
- Keep ONLY the `<PERSONA_GUIDELINES>` slot and the ASCII control tokens
  (e.g. `###STOP###`) verbatim.
- TEXT channel: no spoken/voice conventions, no multi-message splitting.
- Do NOT duplicate the voice guidelines — this is a distinct text-chat document.

## The current text guidelines markdown

```markdown
{guidelines_markdown}```

Now produce the corrected fenced block."""
