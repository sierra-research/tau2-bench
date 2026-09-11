# Copyright Sierra
"""Schema for Language Packs and multilingual personas.

These models are the frozen interface between the shared infrastructure and
per-language content owners: a language owner authors one ``LanguagePack`` per
language and edits no shared code. Typed fields exist only where code branches
on them (registry keys, voice/preset references, injectable phrase lists);
behavioral language content (register, code-switching, politeness, rituals)
lives in the author-written ``pragmatics_clauses``, with language-wide
behavior (decision prompts, firing rates) on the pack. The
``tts_voice_prompt`` is voice-DESIGN material only (TTS voice design and
pinning); it never reaches the LLM behavioral prompt.

Persona content model — who owns what:

- The PACK persona owns the LANGUAGE PROFILE: dialect/variety realization,
  the address system (T-V forms and when), register/formality axis,
  code-switching proportion and domains, letter names / spelling anchors,
  English-tolerance behavior, orthography, and characteristic discourse forms.
- The TASK owns ATTITUDE/AFFECT: impatient, friendly, frustrated, chatty,
  hurried. Task instructions assign the caller's disposition per scenario.
- Pragmatics clauses therefore describe HOW a given attitude sounds in this
  language/variety (conditional realizations: "When frustrated, …"), never
  WHICH attitude the persona has — a fixed-affect clause would contradict the
  task persona inside one prompt.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.backchannel import BackchannelLevel
from tau2.config import (
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_MATRIX_SPEECH_COMPLEXITY,
)
from tau2.data_model.persona import PersonaConfig, PromptMode
from tau2.multilingual.delivery_catalog import get_delivery_catalog_factor
from tau2.multilingual.localization_catalog import (
    get_domain_term,
    get_domain_term_catalog,
    get_guideline_example_kind,
    get_honorific_concept,
    get_symbol_def,
)
from tau2.multilingual.names import NameOrder
from tau2.multilingual.nativeness_catalog import get_catalog_factor
from tau2.multilingual.spelled_entity_normalizer import SpelledEntityTables
from tau2.multilingual.tags import validate_tags
from tau2.multilingual.text_input_catalog import TextInputStyle


class AcousticPreset(BaseModel):
    """A locale-specific acoustic environment (background + burst noise files).

    File names are relative to the verified background-noise data directories,
    and may include a subdirectory (e.g. ``hi_IN/busy_street_iphone_mic.wav``).
    Audio files themselves are language-owner deliverables.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique preset id within the pack")
    display_name: str = Field(description="Human-readable name for visualizations")
    background_noise_files: list[str] = Field(
        default_factory=list,
        description="Continuous background noise files (relative to the continuous "
        "noise dir)",
    )
    burst_noise_files: list[str] = Field(
        default_factory=list,
        description="Burst noise files (relative to the burst noise dir)",
    )


class MultilingualPersonaConfig(PersonaConfig):
    """A language-pack persona: identity, language metadata, and behavior knobs.

    Extends the runtime :class:`PersonaConfig` (verbosity, interrupt tendency)
    with language-aware fields. The persona's full prompt contribution flows
    through the existing ``<PERSONA_GUIDELINES>`` slot via
    :meth:`to_guidelines_text` — no user-simulator changes are required for a
    persona to take effect.

    A registered persona also becomes a ``VoicePersona`` (for TTS voice-id
    resolution) automatically; see ``tau2.multilingual.registry``.
    """

    model_config = ConfigDict(extra="forbid")

    persona_id: str = Field(
        description="Globally unique persona id, e.g. 'priya_hindi_v1'. Must not "
        "collide with existing voice persona names."
    )
    display_name: str = Field(description="Human-readable name, e.g. 'Priya'")
    short_description: str = Field(
        description="One-line description for logs and visualizations"
    )
    language: str = Field(description="ISO 639-1 language code, e.g. 'hi'")
    locale: Optional[str] = Field(
        default=None,
        description="ISO 3166-2 subdivision code, e.g. 'IN-MH' (Maharashtra), "
        "'IN-TG' (Telangana). Recorded on the trajectory and rendered into the "
        "agent prompt as the caller's original regional background; it does not "
        "assert the caller's current physical location.",
    )
    script: Optional[str] = Field(
        default=None,
        description="ISO 15924 script code for the matrix language, e.g. 'deva'",
    )
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Freeform prompt-injected clauses, authored natively by the "
        "language owner. This is where ALL behavioral language content lives: "
        "the persona's LANGUAGE PROFILE (dialect/variety realization, address "
        "system, register/formality axis, code-switching proportion and "
        "domains, letter names / spelling anchors, orthography, English-"
        "tolerance behavior) plus CONDITIONAL REALIZATIONS of attitudes — how "
        "a casual opening, frustration, an indirect refusal, or an unhurried "
        "closing SOUNDS in this language/variety ('When you open casually, "
        "natural forms are …'; 'When frustrated, a high-register speaker "
        "becomes MORE formal and insistent, not rude: …'). Clauses must NEVER "
        "prescribe which attitude/affect the persona has (friendly, warm, "
        "patient, impatient, brisk, composed): attitude is owned by the TASK "
        "persona, and a fixed-affect clause would contradict it inside one "
        "prompt.",
    )
    backchannel_phrases: Optional[list[str]] = Field(
        default=None,
        description="PURE CONTINUERS ONLY — one or two, in the persona's "
        "language/script: the equivalent of English 'mm-hmm'/'uh-huh' (see "
        "tau2.voice_config.BACKCHANNEL_PHRASES). A pure continuer means 'I am "
        "listening, keep going' and NOTHING else — it is not an answer, not "
        "agreement, not 'I understand', not 'okay/fine', not a closing "
        "signal. WHY THE LIST MUST STAY THIS NARROW: only the TIMING of a "
        "backchannel is LLM-gated; the phrase itself is a uniform random draw "
        "from this list with zero context (see "
        "tau2.user.user_simulator_streaming._generate_backchannel_message), "
        "so anything in it can land at any eligible moment. Acknowledgments "
        "(zh 好的/明白了, es vale, tr tamam, …) belong in the language-level "
        "`conversational_confirmations` guideline examples, where the "
        "simulator picks them IN CONTEXT at turn level. At most "
        "tau2.backchannel.MAX_BACKCHANNEL_PHRASES entries; the factory "
        "guardrails enforce the bound. None falls back to the English "
        "defaults.",
    )
    non_directed_phrases: Optional[list[str]] = Field(
        default=None,
        description="Out-of-turn speech phrases for this persona — speech "
        "directed at family, a colleague, a driver, etc., not at the agent. "
        "CLIPPED, at most tau2.voice_config.MAX_NON_DIRECTED_PHRASE_WORDS "
        "words each (space-less scripts and Vietnamese convert that budget "
        "in tau2.multilingual.brevity): the phrase is spliced into the call "
        "as one uninterrupted burst, so anything sentence-length stops "
        "reading as a glance away from the mic and becomes a second "
        "conversation the agent has to sit through. Real away-from-the-phone "
        "speech is elliptical — 'one sec', 'not now', 'coming' — so keep the "
        "addressee and the register but drop everything the listener can "
        "infer: no please/thank-you tails, no explanations, no two-clause "
        "sentences. At most tau2.voice_config.MAX_NON_DIRECTED_PHRASES "
        "entries, because the phrase is drawn uniformly at random with no "
        "context; the factory guardrails enforce both bounds. Vary the "
        "ADDRESSEE across the list (someone in the room, someone on the "
        "street, the caller's own household) rather than padding one phrase. "
        "None falls back to the English defaults.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="ElevenLabs voice id for this persona. Overridable at runtime "
        "via TAU2_VOICE_ID_<PERSONA_ID_UPPER>.",
    )
    tts_voice_prompt: str = Field(
        default="",
        description="Speaker-identity prompt (accent, age, tone, pacing) for "
        "TTS voice DESIGN and pinning only (the VoicePersona prompt consumed "
        "by tau2.multilingual.registry / factory voice generation). It is "
        "NEVER injected into the user-simulator LLM guidelines: voice-design "
        "affect (warm/composed/brisk timbre and pacing) must not steer the "
        "sim's behavior — attitude is owned by the task persona.",
    )
    acoustic_preset_id: Optional[str] = Field(
        default=None,
        description="Id of an AcousticPreset in the same pack",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Structured persona tags from the controlled vocabulary in "
        "tau2.multilingual.tags (e.g. code_switch, formality, age_band). Any "
        "tag provided must be in-vocabulary; completeness is enforced only by "
        "the factory guardrails, so existing packs without tags keep loading.",
    )

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, tags: dict[str, str]) -> dict[str, str]:
        problems = validate_tags(tags)
        if problems:
            raise ValueError("; ".join(problems))
        return tags

    def to_guidelines_text(
        self,
        mode: PromptMode = PromptMode.VOICE,
    ) -> Optional[str]:
        """Persona guidelines for the ``<PERSONA_GUIDELINES>`` system-prompt slot.

        Extends the base PersonaConfig guidelines (verbosity etc.) with the
        author's verbatim persona content: the pragmatics clauses. No prose is
        generated from metadata fields — the language owner controls all
        language/register phrasing natively. The ``tts_voice_prompt`` is
        deliberately NOT included in any mode: it is voice-DESIGN material
        (timbre, pacing, warmth for TTS voice pinning), and splicing it into
        the behavioral prompt would prescribe an affect that can contradict
        the task persona (which owns attitude).

        Args:
            mode: Voice or text rendering. The gender-agreement clause and the
                pragmatics clauses (language profile + conditional attitude
                realizations) apply in both modes.
        """
        sections: list[str] = []
        base = super().to_guidelines_text(mode=mode)
        if base:
            sections.append(base)

        lines: list[str] = []
        gender = self.tags.get("gender")
        if gender in ("male", "female"):
            lines.append(
                english_prompt_string(f"gender_preamble_{gender}", self.language)
            )
        for clause in self.pragmatics_clauses:
            lines.append(clause.strip())
        if lines:
            header = english_prompt_string("persona_language_header", self.language)
            sections.append("\n\n".join([header] + lines))

        return "\n\n".join(sections) if sections else None


class NativenessPackFactorRubric(BaseModel):
    """One language's rubric for a catalog nativeness judge factor.

    The pack SELECTS a factor from the closed catalog
    (``tau2.multilingual.nativeness_catalog.JUDGE_FACTOR_CATALOG``) by ``factor_id`` and
    supplies the language-specific text the judge reads. It may NOT invent a new
    factor: ``factor_id`` is validated against the catalog and an unknown id
    raises, so authoring a language can never introduce an un-reviewed axis.

    The catalog fixes the universal metadata (category, default severity,
    documentary opportunity); only ``severity`` may be overridden per language,
    and rarely should be.

    Authoring style is part of the contract. Write short, direct sentences in
    simple language. State exactly what the agent may do and what counts as a
    violation, then give concrete natural and violating examples in the target
    language. Judge only agent speech. Do not paste research history, calibration
    notes, abstract linguistic prose, or evaluator jargon into these fields.
    """

    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(description="Catalog factor id (must exist in the catalog).")
    nuance: str = Field(description="Short language-specific name shown to the judge.")
    question: Optional[str] = Field(
        default=None,
        description="Optional short, direct language-specific clarification shown "
        "beneath the catalog-owned canonical question.",
    )
    native_does: str = Field(
        description="Simple language-specific passing rule with concrete natural "
        "examples from agent speech."
    )
    ai_likely_does: str = Field(
        description="Simple language-specific violation rule with concrete bad "
        "examples from agent speech."
    )
    severity: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Optional per-language override of the catalog default severity.",
    )
    enabled: bool = Field(
        default=True, description="Set False to keep the rubric but skip scoring it."
    )
    shadow: bool = Field(
        default=False,
        description="Calibration mode: the factor is judged and its check is "
        "recorded on every simulation, but it is excluded from the aggregate "
        "score and coverage. Current language-pack factors are configured live.",
    )
    dose: bool = Field(
        default=False,
        description="Opt-in dose aggregation for utterance-level factors: the "
        "call fails only on one severity>=3 violation or two violated "
        "utterances. Default (False) fails the call on any violation at any "
        "severity.",
    )

    @field_validator("factor_id")
    @classmethod
    def _factor_in_catalog(cls, v: str) -> str:
        try:
            get_catalog_factor(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v


class NativenessPackConfig(BaseModel):
    """Per-language nativeness data, sourced from the pack instead of code.

    Deterministic implementations and the language-agnostic English-leak list
    stay in code (``tau2.judges.nativeness.factors``); the pack selects which
    deterministic checks run and owns all per-language content.
    """

    model_config = ConfigDict(extra="forbid")

    judge_factors: list[NativenessPackFactorRubric] = Field(
        default_factory=list,
        description="Catalog factors available for this language, with their rubrics. "
        "A retained rubric may be retired from runtime with enabled=False.",
    )
    enabled_deterministic_factors: list[
        Literal["backchannel_frequency", "email_symbol_verbalization"]
    ] = Field(
        default_factory=list,
        description="Code-owned deterministic voice factors selected for this "
        "language. Omitted factors remain implemented but are not scheduled.",
    )
    email_symbols: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Native verbalizations of email symbols, keyed by symbol "
        "('@', '.'), e.g. {'@': ['arroba'], '.': ['punto']}. Used by the "
        "deterministic email checker; absent → that checker is a no-op for this "
        "language (never a false FAIL).",
    )

    @model_validator(mode="after")
    def _no_duplicate_factor_ids(self) -> "NativenessPackConfig":
        seen = set()
        for rubric in self.judge_factors:
            if rubric.factor_id in seen:
                raise ValueError(
                    f"Duplicate nativeness judge factor '{rubric.factor_id}' in pack"
                )
            seen.add(rubric.factor_id)
        if len(self.enabled_deterministic_factors) != len(
            set(self.enabled_deterministic_factors)
        ):
            raise ValueError("Duplicate enabled deterministic nativeness factor")
        return self


class SymbolReadout(BaseModel):
    """One language's native verbalization(s) of a catalog technical symbol.

    The pack SELECTS a symbol from the closed catalog
    (``tau2.multilingual.localization_catalog.SYMBOL_CATALOG``) by its literal
    character and supplies the native readout(s). It may NOT invent a new
    symbol key: ``symbol`` is validated against the catalog and an unknown
    character raises, so authoring a language can never add an un-reviewed
    symbol axis.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        description="The literal symbol character (must exist in the closed "
        "symbol catalog in tau2.multilingual.localization_catalog)."
    )
    spoken: list[str] = Field(
        min_length=1,
        description="Native verbalizations of the symbol, most natural first "
        "(e.g. de '_' -> ['Unterstrich']). Alternatives a native also says may "
        "follow (e.g. de '@' -> ['at', 'Klammeraffe']).",
    )

    @field_validator("symbol")
    @classmethod
    def _symbol_in_catalog(cls, v: str) -> str:
        try:
            get_symbol_def(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v

    @field_validator("spoken")
    @classmethod
    def _spoken_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [token.strip() for token in v]
        if any(not token for token in cleaned):
            raise ValueError("symbol readout tokens must be non-empty strings")
        return cleaned


class SpokenFormExample(BaseModel):
    """A written value paired with its fully-native spoken form.

    Worked examples for the single-language date/numeral consistency rule:
    the ``spoken`` side must be voiced ENTIRELY in the pack's language (the
    grounded failures are half-and-half readouts like ko 'May 십칠' /
    '이십two' and run-together forms like vi 'May21').
    """

    model_config = ConfigDict(extra="forbid")

    written: str = Field(
        description="The value as written in a task/DB (e.g. 'May 17', '21')."
    )
    spoken: str = Field(
        description="How a native voices it, entirely in the pack's language "
        "(native script; no English words mixed in)."
    )


class PhoneNumberReadout(BaseModel):
    """How a native caller groups and reads a phone number aloud.

    The ``rule`` is stated in English (it is injected into the English-scaffold
    localization block); the worked examples are fully native. Grounded in the
    per-language guideline conventions (e.g. es '612 34 56 78' digit-by-digit
    or pairs; ja 3-4-4 chunks joined by 'の') that previously reached only the
    retired native-prompt arm.
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(
        description="The native grouping/reading convention, stated in English: "
        "grouping blocks (e.g. pairs, 3-4-4), digit-by-digit vs. grouped-number "
        "reading, and when a native falls back to digit-by-digit."
    )
    examples: list[SpokenFormExample] = Field(
        min_length=1,
        description="Worked phone-number readouts (written number -> "
        "fully-native spoken form; at least 1).",
    )


class AmountReadout(BaseModel):
    """How a native caller reads money amounts and decimals aloud.

    The ``rule`` is stated in English; the worked examples are fully native.
    Covers currency-word placement, the decimal-separator word, and
    large-amount units (e.g. es 'doce mil euros', 'treinta y cinco coma
    cinco'; ja 万; ko 만/억).
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(
        description="The native amount-reading convention, stated in English: "
        "currency-word placement, the decimal-separator word (e.g. 'coma', "
        "'virgule', 'Komma'), and large-amount units."
    )
    examples: list[SpokenFormExample] = Field(
        min_length=2,
        description="Worked amount readouts (written amount -> fully-native "
        "spoken form; at least 2).",
    )


# The prompt supplies the prohibition once, as its own 'Do NOT use:' label, so
# an ``avoid`` string that repeats it renders 'Do NOT use: Do NOT use the
# NATO…'. Eight packs (de es hi it ko nl pl vi) shipped that way until the
# 2026-07-27 data correction. Gated on the DATA at pack load rather than by
# teaching the renderer to sniff for a leading negation and suppress its label
# — that heuristic would rot the moment an avoid string is worded differently.
AVOID_REDUNDANT_LEAD_RE = re.compile(r"^(do\s+not|don't)\s+use\b", re.IGNORECASE)


class SpellingAlphabet(BaseModel):
    """The native convention for disambiguating a misheard name or code.

    The ``rule`` and ``avoid`` are stated in English; the anchors are fully
    native. Most languages anchor letters on native names/cities (es 'B de
    Barcelona', de 'A wie Anton'); some disambiguate differently (zh character
    decomposition '弓长张', ko jamo decomposition) — the rule states whatever a
    native actually does. ``avoid`` names the non-native convention to reject
    (typically the NATO/English 'B for boy' alphabet).
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(
        description="The native spelling/disambiguation convention, stated in "
        "English: the anchor-word pattern (e.g. 'B de Barcelona' on native "
        "names/cities), or character/jamo decomposition where the language "
        "spells that way."
    )
    avoid: str = Field(
        description="The convention a native does NOT use, stated in English "
        'as a NOUN PHRASE naming the thing to reject (e.g. "The NATO/anglo '
        "'B for boy' / 'Bravo' alphabet\"). It renders after the prompt's own "
        "'Do NOT use:' label, so it must NOT begin with 'Do not use' — that "
        "duplicates the label ('Do NOT use: Do NOT use the NATO…'), which is "
        "how eight packs shipped before the 2026-07-27 correction."
    )
    anchor_examples: list[SpokenFormExample] = Field(
        min_length=3,
        description="Worked anchors (written character -> native spoken anchor "
        "phrase, e.g. 'B' -> 'B de Barcelona'; at least 3).",
    )

    @field_validator("avoid")
    @classmethod
    def _avoid_does_not_repeat_the_label(cls, v: str) -> str:
        cleaned = v.strip()
        if AVOID_REDUNDANT_LEAD_RE.match(cleaned):
            raise ValueError(
                "spelling_alphabet.avoid must not begin with 'Do not use' — "
                "the prompt already prefixes its own 'Do NOT use:' label, so "
                "this renders as 'Do NOT use: Do NOT use …'. State the "
                "convention as a noun phrase (\"The NATO/anglo 'B for boy' "
                'alphabet")'
            )
        return cleaned


class SpokenValueKind(str, Enum):
    """The closed set of worked spoken-value example kinds.

    Mirrors the labeled example list of the English voice guidelines'
    'Speaking Special Characters and Numbers' section; every pack supplies
    exactly one example per kind so the rendered per-language section always
    covers the same value types.
    """

    EMAIL = "email"
    USER_ID = "user_id"
    PHONE = "phone"
    SPELLING_NAME = "spelling_name"
    ACCOUNT_NUMBER = "account_number"
    WEBSITE = "website"

    @property
    def label(self) -> str:
        """The rendered list label (matches the English section's labels)."""
        return {
            SpokenValueKind.EMAIL: "Email",
            SpokenValueKind.USER_ID: "User ID",
            SpokenValueKind.PHONE: "Phone",
            SpokenValueKind.SPELLING_NAME: "Spelling name",
            SpokenValueKind.ACCOUNT_NUMBER: "Account number",
            SpokenValueKind.WEBSITE: "Website",
        }[self]


class SpokenValueExample(BaseModel):
    """One worked, fully-voiced value readout for the guidelines spellout section.

    These are the per-language counterparts of the English guidelines' worked
    examples ("it's john underscore doe at gmail dot com", …): a
    locale-plausible written value plus the complete utterance a native says
    on the phone — symbols voiced natively, digits/letters separated.
    PERSONA-NEUTRAL by contract: pure readout mechanics, no hesitation
    fillers, no register/affect dressing — tone belongs to the task persona.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SpokenValueKind = Field(
        description="Which value type this example demonstrates (closed set)."
    )
    written: str = Field(
        description="The value as written (locale-plausible, e.g. "
        "'juan_perez@gmail.com', a local-format phone number)."
    )
    spoken: str = Field(
        description="The complete utterance a native says for it on the "
        "phone — fully in the pack's language, symbols voiced natively. "
        "COMMA CONVENTION: emails and website addresses read as a continuous "
        "flow of words with NO commas between the spoken tokens ('juan guion "
        "bajo perez arroba gmail punto com'); comma-and-space separation "
        "applies ONLY where a value is spelled character by character "
        "(letters, digits, codes — 'A, B, one, two'). A plain neutral "
        'carrier phrase (the native "it\'s …" / "my account is …") is '
        "fine; NO hesitation fillers ('um', 'uh' or native equivalents) and "
        "NO personality/register dressing — persona-neutral mechanics only."
    )


class GuidelineExample(BaseModel):
    """One language's native utterances for a catalog guidelines example kind.

    The pack SELECTS a kind from the closed catalog
    (``tau2.multilingual.localization_catalog.GUIDELINE_EXAMPLE_CATALOG``) by
    ``kind`` and supplies the native utterances the English voice guidelines
    demonstrate inline (disfluency palette, confirmations, silence check-ins,
    …). It may NOT invent a new kind: ``kind`` is validated against the
    catalog and an unknown id raises, so authoring a language can never add an
    un-reviewed example axis.

    Content contract: utterances are PERSONA-NEUTRAL language mechanics. A
    palette naturally contains the language's own hesitation fillers and
    discourse particles — that is its purpose — but no personality adjectives
    and no politeness/affect dressing prescribing an attitude: attitude
    belongs to the task persona.

    Speaker gender is language MECHANICS, not persona affect, in the languages
    that mark it (Japanese first-person pronouns, sentence-final particles and
    softeners; the ja reviewer's finding that a single shared palette reads as
    a woman's speech). ``male_utterances`` is the optional male-speaker
    realization of the SAME utterances: parallel item-for-item, same kind and
    same situation, differing only where the language genuinely differs by
    gender. Supply it only for kinds that really differ — a kind without it
    renders ``utterances`` for every speaker, which is the right answer for
    the many languages (and the many kinds) that do not mark speaker gender
    here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description="Catalog example kind id (must exist in the closed "
        "guideline-example catalog in tau2.multilingual.localization_catalog)."
    )
    utterances: list[str] = Field(
        min_length=1,
        description="The native utterances/phrases for this kind, most natural "
        "first, item count within the kind's catalog bounds. Fully in the "
        "pack's language and native orthography (code-switched English words, "
        "where natural, keep English spelling). The default palette: it "
        "renders for every speaker unless a gendered variant applies.",
    )
    male_utterances: Optional[list[str]] = Field(
        default=None,
        description="Male-speaker realization of the same utterances, for "
        "languages that mark speaker gender in these mechanics. Parallel "
        "item-for-item to 'utterances' (same length, same situations) and "
        "must differ from it — a no-op copy is rejected. None means the "
        "language does not mark gender for this kind and 'utterances' serves "
        "every speaker.",
    )

    @field_validator("kind")
    @classmethod
    def _kind_in_catalog(cls, v: str) -> str:
        try:
            get_guideline_example_kind(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v

    @field_validator("utterances", "male_utterances")
    @classmethod
    def _utterances_non_empty(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        cleaned = [utterance.strip() for utterance in v]
        if any(not utterance for utterance in cleaned):
            raise ValueError("guideline example utterances must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _count_within_kind_bounds(self) -> "GuidelineExample":
        kind_def = get_guideline_example_kind(self.kind)
        if not (kind_def.min_items <= len(self.utterances) <= kind_def.max_items):
            raise ValueError(
                f"guideline example kind '{self.kind}' requires between "
                f"{kind_def.min_items} and {kind_def.max_items} utterances, "
                f"got {len(self.utterances)}"
            )
        if self.male_utterances is not None:
            if len(self.male_utterances) != len(self.utterances):
                raise ValueError(
                    f"guideline example kind '{self.kind}': male_utterances "
                    "must be parallel item-for-item to utterances "
                    f"({len(self.utterances)} items), got "
                    f"{len(self.male_utterances)}"
                )
            if self.male_utterances == self.utterances:
                raise ValueError(
                    f"guideline example kind '{self.kind}': male_utterances "
                    "duplicates utterances — drop it instead (absent means "
                    "the shared palette serves every speaker)"
                )
        return self

    def items_for_gender(self, gender: Optional[str]) -> list[str]:
        """The utterances this kind renders for a speaker of ``gender``.

        Male speakers get ``male_utterances`` where the pack supplies them;
        everyone else (female, unset, a pack without the variant) gets the
        shared ``utterances``.
        """
        if gender == "male" and self.male_utterances is not None:
            return self.male_utterances
        return self.utterances


# The fixed English template strings of the prompt scaffold, keyed by field
# name. Single source of truth: the localization block and the persona
# preamble render THESE (see :func:`english_prompt_string`), and the retired
# native arm's the retired native-strings draft prompt shows THESE as the
# translation source. ``{display_name}`` / ``{gender}`` are formatted at render
# time. The end-of-prompt reminder's English source lives in
# ``tau2.multilingual.english_prompts`` (versioned prompt material), not here.
ENGLISH_GENDER_PREAMBLE_TEMPLATE = (
    "You are {gender}. When you refer to yourself in the first person, use "
    "grammatically correct first-person gender agreement for {gender} "
    "speakers in the language you are speaking (verb forms, adjectives, "
    "participles, and any other gendered elements). This matters only in "
    "languages that mark grammatical gender; in languages that do not, it "
    "has no effect."
)

ENGLISH_PROMPT_SCAFFOLD: dict[str, str] = {
    "localization_header": "## LANGUAGE LOCALIZATION ({display_name})",
    "dates_numbers_heading": "### Dates and numbers — one language only",
    "dates_numbers_rule_voice": (
        "Say every date and number ENTIRELY in {display_name}. Never mix "
        "English and {display_name} words inside one date or number, and "
        "never run the parts together — the whole value is voiced in "
        "{display_name}, not half in each language."
    ),
    "dates_numbers_rule_text": (
        "Write every date and number consistently in {display_name} "
        "conventions. Never mix English and {display_name} words inside "
        "one date or number."
    ),
    "native_readouts_label": "Native readouts:",
    "phone_heading": "### Phone numbers",
    "amounts_heading": "### Amounts and prices",
    "spelling_heading": "### Spelling out a misheard name or code",
    "do_not_use_label": "Do NOT use:",
    "native_anchors_label": "Native anchors:",
    "honorific_self_heading": "### Talking about yourself, not the other person",
    "honorific_self_rule": (
        "These words have honorific forms that apply only to the other "
        "person. Use the plain form about yourself, never the honorific one."
    ),
    "glossary_heading": "### Domain terms — say them in {display_name}",
    "glossary_rule": (
        "Use these natural {display_name} terms for the service "
        "vocabulary below; do not switch into the English term "
        "mid-sentence. Where a note marks the English form as the "
        "established native usage, that form IS the natural term. If "
        "your persona's register genuinely code-switches a term, the "
        "persona instructions take precedence."
    ),
    "persona_language_header": "## PERSONA AND LANGUAGE",
    "gender_preamble_male": ENGLISH_GENDER_PREAMBLE_TEMPLATE.format(gender="male"),
    "gender_preamble_female": ENGLISH_GENDER_PREAMBLE_TEMPLATE.format(gender="female"),
}


def english_prompt_string(field: str, display_name: str) -> str:
    """The fixed English scaffold string for a field, language-parameterized.

    ``field`` must be an ``ENGLISH_PROMPT_SCAFFOLD`` key — a loud ValueError
    otherwise, so a renderer can never silently reference a string that does
    not exist. The retired native prompt-language arm overlaid a pack's
    translated rendering here; that lookup now lives in
    was deleted along with the arm on 2026-08-03.
    """
    if field not in ENGLISH_PROMPT_SCAFFOLD:
        raise ValueError(
            f"'{field}' is not a prompt-scaffold field; legal fields: "
            f"{sorted(ENGLISH_PROMPT_SCAFFOLD)}"
        )
    return ENGLISH_PROMPT_SCAFFOLD[field].format(display_name=display_name)


class DomainTermGloss(BaseModel):
    """One language's natural spoken term for a catalog domain-vocabulary entry.

    The pack SELECTS a term from its domain's closed catalog
    (``tau2.multilingual.localization_catalog.DOMAIN_TERM_CATALOGS``) by
    ``term_id`` and supplies the term a native caller actually says. It may NOT
    invent a new term id — the ``domain_glossaries`` validator checks each
    entry against the catalog of the domain it is filed under. Where the
    natural native usage IS the English loanword (e.g. an established
    borrowing in the local script), ``native`` holds that loanword form and
    ``note`` says so — the glossary encodes natural usage, not forced purism.
    """

    model_config = ConfigDict(extra="forbid")

    term_id: str = Field(
        description="Catalog term id (must exist in the term catalog of the "
        "domain this gloss is filed under)."
    )
    native: str = Field(
        description="The natural native spoken term (native script; may be an "
        "established loanword when that is what natives say)."
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional register nuance, written IN ENGLISH with any "
        "cited native form quoted (e.g. 'established loanword — the English "
        "form is the natural usage', or a formal/casual split). English "
        "because the note is rendered into the unconditionally-English prompt "
        "scaffold (see to_localization_section); a note may name a REGISTER "
        "but never a persona.",
    )


class HonorificSelfReference(BaseModel):
    """One language's honorific/plain word pair for a catalog concept.

    Some languages have a word whose honorific form may be used ONLY about the
    person you are talking to; said of yourself it is ungrammatical (ko 성함 vs
    이름 for 'name', ja ご住所 vs 住所 for 'address', zh 贵姓 vs 姓 for a surname).
    A caller simulator that reaches for the polite-sounding form self-applies
    it, so the pack supplies the corresponding plain self-reference form.

    The pack SELECTS a concept from the closed catalog
    (``tau2.multilingual.localization_catalog.HONORIFIC_CONCEPT_CATALOG``) by
    ``concept_id``; it may NOT invent one. HAND-CURATED, native-reviewed
    content: it is NOT drafted by ``tau2 factory draft-localization`` and must
    only be supplied for languages with observed evidence of the trap — a
    speculative pair teaches the simulator a distinction the language may not
    make.
    """

    model_config = ConfigDict(extra="forbid")

    concept_id: str = Field(
        description="Catalog honorific-concept id (must exist in the closed "
        "catalog in tau2.multilingual.localization_catalog)."
    )
    honorific: str = Field(
        description="The honorific word, which applies ONLY to the other "
        "person (native script) — the form the simulator must not self-apply."
    )
    plain: str = Field(
        description="The plain word to use about oneself (native script)."
    )

    @field_validator("concept_id")
    @classmethod
    def _concept_in_catalog(cls, v: str) -> str:
        try:
            get_honorific_concept(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v

    @field_validator("honorific", "plain")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("honorific self-reference words must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _forms_differ(self) -> "HonorificSelfReference":
        if self.honorific == self.plain:
            raise ValueError(
                f"honorific concept '{self.concept_id}': the honorific and "
                "plain forms are identical — the language does not make this "
                "distinction, so drop the entry"
            )
        return self


class LocalizationPackConfig(BaseModel):
    """Per-language SPOKEN-LOCALIZATION data: symbol readouts, the
    single-language date/numeral rule's worked examples, and the domain-term
    glossary.

    Governs how the user simulator SPEAKS entities and domain terms in the
    target language — orthogonal to which language its instructions are in.
    Rendered into the user-simulator system prompt via :meth:`to_prompt_text`
    (both text and voice modes); a pack without this block renders nothing and
    keeps the previous behavior.
    """

    model_config = ConfigDict(extra="forbid")

    symbol_readouts: list[SymbolReadout] = Field(
        default_factory=list,
        description="Native verbalizations of catalog technical symbols "
        "(underscore, hyphen, zero, at sign, …) for reading emails/IDs/codes "
        "aloud.",
    )
    date_examples: list[SpokenFormExample] = Field(
        default_factory=list,
        description="Worked examples of fully-native date readouts (written "
        "value -> spoken form, entirely in this language).",
    )
    number_examples: list[SpokenFormExample] = Field(
        default_factory=list,
        description="Worked examples of fully-native number/amount readouts "
        "(written value -> spoken form, entirely in this language).",
    )
    domain_glossaries: dict[str, list[DomainTermGloss]] = Field(
        default_factory=dict,
        description="Natural native terms for each benchmark domain's catalog "
        "vocabulary, keyed by domain (airline: user ID, refund, reservation, "
        "…; telecom: plan, roaming, SIM card, …) so callers do not drop "
        "English terms mid-sentence. A pack carries one glossary per domain "
        "it has an experiment for; only the run's domain renders.",
    )
    phone_number_readout: Optional[PhoneNumberReadout] = Field(
        default=None,
        description="How a native groups and reads phone numbers aloud "
        "(rule stated in English + fully-native worked examples). None "
        "renders nothing (pre-speech-conventions packs still load).",
    )
    amount_readout: Optional[AmountReadout] = Field(
        default=None,
        description="How a native reads money amounts and decimals aloud "
        "(currency placement, decimal-separator word, large-amount units). "
        "None renders nothing.",
    )
    spelling_alphabet: Optional[SpellingAlphabet] = Field(
        default=None,
        description="The native anchor-word spelling convention for misheard "
        "names/codes, with the non-native convention to avoid. None renders "
        "nothing.",
    )
    honorific_self_reference: list[HonorificSelfReference] = Field(
        default_factory=list,
        description="Honorific/plain word pairs for catalog concepts whose "
        "honorific form is other-directed only (ko 성함 vs 이름 for 'name'), so "
        "the caller does not apply the honorific word to ITSELF. Renders in "
        "both text and voice modes; empty renders nothing, which is what "
        "every language without observed evidence of the trap must stay. NOT "
        "DRAFTED — do not author this field: it is hand-curated, "
        "native-reviewed data, supplied only for ko/ja/zh today.",
    )
    spoken_value_examples: list[SpokenValueExample] = Field(
        default_factory=list,
        description="Worked, fully-voiced value readouts (one per "
        "SpokenValueKind: email, user ID, phone, spelling a name, account "
        "number, website) rendered into the voice guidelines' spellout "
        "section in place of the English worked examples. Empty renders "
        "nothing (the English section applies unchanged).",
    )
    guideline_examples: list[GuidelineExample] = Field(
        default_factory=list,
        description="Native utterances for the catalog guideline example "
        "kinds (disfluency palette, conversational confirmations, silence "
        "check-ins, don't-know responses, …). Rendered into the English "
        "voice guidelines' inline <EXAMPLE:kind> slots so english-prompt-mode "
        "runs demonstrate target-language speech patterns; a missing kind "
        "renders the fixed English default.",
    )
    spelled_entity_normalization: bool = Field(
        default=False,
        description="Opt-in pre-TTS spelled-entity normalization: when "
        "True, entity-shaped tokens and spelling runs in the TTS-BOUND text "
        "are expanded into the native letter/digit/symbol words below, "
        "immediately before the ElevenLabs call (see "
        "tau2.multilingual.spelled_entity_normalizer). The stored transcript "
        "keeps the original text. Default False: languages whose TTS already "
        "renders codes correctly are completely untouched. Requires complete "
        "letter_names + digit_names tables and an '_' symbol readout.",
    )
    letter_names: dict[str, str] = Field(
        default_factory=dict,
        description="Native spoken letter names, keyed by UPPERCASE letter. "
        "When present the table must be complete (all 26 of A-Z); a partial "
        "table is rejected at pack load. May ship with "
        "spelled_entity_normalization=False (staged pending native review).",
    )
    digit_names: dict[str, str] = Field(
        default_factory=dict,
        description="Native spoken digit words, keyed by digit '0'-'9'. When "
        "present the table must be complete (all 10 digits); a partial table "
        "is rejected at pack load. May ship with "
        "spelled_entity_normalization=False (staged pending native review).",
    )
    letter_name_tts: dict[str, str] = Field(
        default_factory=dict,
        description="Pronunciation-hint respelling of a letter name for the "
        "TTS-BOUND text only, keyed by UPPERCASE letter — for the letters "
        "whose orthographic name the TTS voices wrong (pt 'S': 'ésse' is "
        "read with a European final -e, so the pack hints 'éssi'). Partial "
        "by design; the transcript always keeps the orthographic letter_names "
        "form. Requires letter_names, and a hint may not collide with another "
        "letter's name (that would make normalization non-idempotent).",
    )

    @model_validator(mode="after")
    def _validate_spelled_entity_tables(self) -> "LocalizationPackConfig":
        if self.letter_names:
            expected = {chr(c) for c in range(ord("A"), ord("Z") + 1)}
            if set(self.letter_names) != expected:
                missing = sorted(expected - set(self.letter_names))
                extra = sorted(set(self.letter_names) - expected)
                raise ValueError(
                    "letter_names must cover exactly the uppercase letters "
                    f"A-Z (missing: {missing}, unexpected: {extra})"
                )
            if any(not name.strip() for name in self.letter_names.values()):
                raise ValueError("letter_names values must be non-empty")
        if self.digit_names:
            expected = {str(d) for d in range(10)}
            if set(self.digit_names) != expected:
                missing = sorted(expected - set(self.digit_names))
                extra = sorted(set(self.digit_names) - expected)
                raise ValueError(
                    "digit_names must cover exactly the digits 0-9 "
                    f"(missing: {missing}, unexpected: {extra})"
                )
            if any(not name.strip() for name in self.digit_names.values()):
                raise ValueError("digit_names values must be non-empty")
        if self.letter_name_tts:
            if not self.letter_names:
                raise ValueError("letter_name_tts requires a letter_names table")
            unknown = sorted(set(self.letter_name_tts) - set(self.letter_names))
            if unknown:
                raise ValueError(
                    f"letter_name_tts keys must be letters A-Z (unexpected: {unknown})"
                )
            if any(hint.strip() == "" for hint in self.letter_name_tts.values()):
                raise ValueError("letter_name_tts values must be non-empty")
            spelled = {name.casefold() for name in self.letter_names.values()}
            colliding = sorted(
                letter
                for letter, hint in self.letter_name_tts.items()
                if hint.casefold() in spelled
            )
            if colliding:
                raise ValueError(
                    "letter_name_tts hints must not repeat a letter_names "
                    f"spelling (colliding letters: {colliding})"
                )
        if self.spelled_entity_normalization:
            if not self.letter_names or not self.digit_names:
                raise ValueError(
                    "spelled_entity_normalization requires complete "
                    "letter_names AND digit_names tables"
                )
            if "_" not in {readout.symbol for readout in self.symbol_readouts}:
                raise ValueError(
                    "spelled_entity_normalization requires an '_' symbol "
                    "readout (user-id expansion voices the underscore)"
                )
        return self

    def spelled_entity_tables(self) -> Optional[SpelledEntityTables]:
        """The native word tables for pre-TTS normalization, or None.

        None unless the pack opts in (``spelled_entity_normalization: true``),
        so holding a ``SpelledEntityTables`` implies normalization applies.
        Symbol readouts contribute their most-natural (first) spoken form.
        """
        if not self.spelled_entity_normalization:
            return None
        return SpelledEntityTables(
            letter_names=dict(self.letter_names),
            digit_names=dict(self.digit_names),
            symbol_names={
                readout.symbol: readout.spoken[0] for readout in self.symbol_readouts
            },
            letter_name_tts=dict(self.letter_name_tts),
        )

    @model_validator(mode="after")
    def _no_duplicates(self) -> "LocalizationPackConfig":
        seen_symbols: set[str] = set()
        for readout in self.symbol_readouts:
            if readout.symbol in seen_symbols:
                raise ValueError(
                    f"Duplicate localization symbol '{readout.symbol}' in pack"
                )
            seen_symbols.add(readout.symbol)
        for glossary_domain, glosses in self.domain_glossaries.items():
            try:
                catalog = get_domain_term_catalog(glossary_domain)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            seen_terms: set[str] = set()
            for gloss in glosses:
                if gloss.term_id not in catalog:
                    raise ValueError(
                        f"Unknown localization domain term '{gloss.term_id}' "
                        f"for domain '{glossary_domain}'; catalog ids: "
                        f"{sorted(catalog)}"
                    )
                if gloss.term_id in seen_terms:
                    raise ValueError(
                        f"Duplicate localization domain term '{gloss.term_id}' "
                        f"in pack ('{glossary_domain}' glossary)"
                    )
                seen_terms.add(gloss.term_id)
        seen_kinds: set[SpokenValueKind] = set()
        for example in self.spoken_value_examples:
            if example.kind in seen_kinds:
                raise ValueError(
                    f"Duplicate spoken-value example kind '{example.kind.value}' "
                    "in pack"
                )
            seen_kinds.add(example.kind)
        seen_concepts: set[str] = set()
        for pair in self.honorific_self_reference:
            if pair.concept_id in seen_concepts:
                raise ValueError(
                    f"Duplicate honorific concept '{pair.concept_id}' in pack"
                )
            seen_concepts.add(pair.concept_id)
        seen_example_kinds: set[str] = set()
        for guideline_example in self.guideline_examples:
            if guideline_example.kind in seen_example_kinds:
                raise ValueError(
                    f"Duplicate guideline example kind "
                    f"'{guideline_example.kind}' in pack"
                )
            seen_example_kinds.add(guideline_example.kind)
        return self

    def guideline_example_items(
        self, kind_id: str, gender: Optional[str] = None
    ) -> Optional[list[str]]:
        """The pack's native utterances for a catalog example kind, or None.

        None means the pack does not carry the kind — the guidelines slot then
        renders the kind's fixed English default. ``gender`` is the speaking
        persona's gender tag: a kind carrying a gendered variant renders the
        male realization for male speakers (see
        :class:`GuidelineExample`), the shared palette otherwise.
        """
        for example in self.guideline_examples:
            if example.kind == kind_id:
                return example.items_for_gender(gender)
        return None

    def to_prompt_text(
        self,
        display_name: str,
        mode: PromptMode = PromptMode.VOICE,
        domain: Optional[str] = None,
    ) -> Optional[str]:
        """Render the localization data as a user-simulator prompt section.

        Fixed in-code template (never model-improvised); only the per-language
        data varies. In ``voice`` mode the date/numeral rule, phone-number and
        amount conventions, the spelling alphabet, and the glossary render; in
        ``text`` mode the spoken-only sub-sections are omitted (values are
        TYPED literally in chat) while the single-language consistency rule
        and the domain glossary still apply. The self-honorification block is
        deliberately NOT voice-specific — applying an other-directed honorific
        to yourself is ungrammatical typed or spoken — so it renders in both
        modes alongside the glossary. Symbol readouts, the worked
        spoken-value examples, and the guideline examples do NOT render here —
        the first two render inside the voice guidelines' 'Speaking Special
        Characters and Numbers' section (:meth:`to_spellout_section`) and the
        examples inside the guidelines' ``<EXAMPLE:kind>`` slots; each has a
        single owner. Returns None when nothing would render.

        Args:
            display_name: The pack's human-readable language name.
            mode: Voice or text rendering.
            domain: The run's benchmark domain; selects which of the pack's
                ``domain_glossaries`` renders. None (or a domain the pack has
                no glossary for) renders no glossary section — the
                language-mechanics sections are domain-independent.
        """
        glossary = self.domain_glossaries.get(domain) if domain else None
        if not (
            self.date_examples
            or self.number_examples
            or glossary
            or self.phone_number_readout
            or self.amount_readout
            or self.spelling_alphabet
            or self.honorific_self_reference
        ):
            return None

        def scaffold(field: str) -> str:
            return english_prompt_string(field, display_name)

        voice = PromptMode(mode) is PromptMode.VOICE
        sections: list[str] = []

        rule_lines = [
            scaffold("dates_numbers_heading"),
            scaffold("dates_numbers_rule_voice")
            if voice
            else scaffold("dates_numbers_rule_text"),
        ]
        examples = (self.date_examples + self.number_examples) if voice else []
        if examples:
            rule_lines.append(scaffold("native_readouts_label"))
            for example in examples:
                rule_lines.append(f"- '{example.written}' → “{example.spoken}”")
        sections.append("\n".join(rule_lines))

        if voice and self.phone_number_readout:
            lines = [
                scaffold("phone_heading"),
                self.phone_number_readout.rule,
                scaffold("native_readouts_label"),
            ]
            for example in self.phone_number_readout.examples:
                lines.append(f"- '{example.written}' → “{example.spoken}”")
            sections.append("\n".join(lines))

        if voice and self.amount_readout:
            lines = [
                scaffold("amounts_heading"),
                self.amount_readout.rule,
                scaffold("native_readouts_label"),
            ]
            for example in self.amount_readout.examples:
                lines.append(f"- '{example.written}' → “{example.spoken}”")
            sections.append("\n".join(lines))

        if voice and self.spelling_alphabet:
            lines = [
                scaffold("spelling_heading"),
                self.spelling_alphabet.rule,
                f"{scaffold('do_not_use_label')} {self.spelling_alphabet.avoid}",
                scaffold("native_anchors_label"),
            ]
            for example in self.spelling_alphabet.anchor_examples:
                lines.append(f"- '{example.written}' → “{example.spoken}”")
            sections.append("\n".join(lines))

        if self.honorific_self_reference:
            lines = [
                scaffold("honorific_self_heading"),
                scaffold("honorific_self_rule"),
                f"{scaffold('do_not_use_label')} "
                + ", ".join(pair.honorific for pair in self.honorific_self_reference),
                scaffold("native_anchors_label"),
            ]
            for pair in self.honorific_self_reference:
                english = get_honorific_concept(pair.concept_id).english
                lines.append(f"- {english} → “{pair.plain}” (not “{pair.honorific}”)")
            sections.append("\n".join(lines))

        if glossary:
            lines = [
                scaffold("glossary_heading"),
                scaffold("glossary_rule"),
            ]
            for gloss in glossary:
                english = get_domain_term(domain, gloss.term_id).english
                line = f"- {english} → “{gloss.native}”"
                if gloss.note:
                    line += f" ({gloss.note})"
                lines.append(line)
            sections.append("\n".join(lines))

        header = scaffold("localization_header")
        return "\n\n".join([header] + sections)

    def to_spellout_section(self, display_name: str) -> Optional[str]:
        """Render the voice guidelines' spellout section for this language.

        The single owner of the symbol table and the worked value readouts:
        it replaces the ``<SPOKEN_VALUES_SPELLOUT>`` slot in the ENGLISH
        voice guidelines — which is what every run reads — so the
        instructional prose stays English while every symbol name and worked
        example is native. (The packs' localized guidelines files are
        retired-arm material: they carry their own translated section and
        have no slot.)

        Fixed in-code template; only the per-language data varies. Worked
        examples render in ``SpokenValueKind`` declaration order. Returns
        None unless BOTH the symbol readouts and the worked examples are
        present — a partial section (native table, English examples) would
        be worse than the English fallback.

        Args:
            display_name: The pack's human-readable language name.
        """
        if not (self.symbol_readouts and self.spoken_value_examples):
            return None
        lines = [
            "## Speaking Special Characters and Numbers",
            f"When providing emails, user IDs, or any text with special "
            f"characters, SPELL THEM OUT as you would on a phone, voicing "
            f"each symbol the native {display_name} way — never with its "
            f"English name:",
        ]
        for readout in self.symbol_readouts:
            english = get_symbol_def(readout.symbol).english_name
            spoken = " or ".join(f'"{s}"' for s in readout.spoken)
            lines.append(f"- {readout.symbol} ({english}) = {spoken}")
        lines.append("")
        lines.append(
            "Comma convention: emails and website addresses are read as a "
            "continuous flow of words — never insert commas or pauses "
            "between the spoken parts. When a value IS spelled out character "
            "by character (letters, digits, codes), ALWAYS separate the "
            "characters with comma and space and never run them together. "
            "The examples below show the native convention for each value "
            "type — follow them exactly."
        )
        lines.append("")
        lines.append(
            f"Examples (every value is read out the native {display_name} "
            "way — these show pure readout mechanics; your persona controls "
            "tone and register):"
        )
        by_kind = {example.kind: example for example in self.spoken_value_examples}
        for kind in SpokenValueKind:
            example = by_kind.get(kind)
            if example is not None:
                lines.append(f'- {kind.label}: "{example.spoken}"')
        return "\n".join(lines)


class TextInputExample(BaseModel):
    """One worked typing example for a declared text input style.

    Persona-neutral typing MECHANICS by contract: the pair shows how standard
    orthography is typed under the style (script/diacritics/spelling only) —
    no register, affect, or hesitation dressing (tone belongs to the task
    persona).
    """

    model_config = ConfigDict(extra="forbid")

    standard: str = Field(
        description="A short utterance in the language's standard/native orthography."
    )
    typed: str = Field(
        description="The same utterance as a chat user types it under this "
        "input style (e.g. romanized, or with diacritics dropped)."
    )

    @field_validator("standard", "typed")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("text input example sides must be non-empty")
        return cleaned


class TextInputStyleSpec(BaseModel):
    """One text input style this language supports, with worked examples.

    The pack SELECTS a style from the closed catalog
    (``tau2.multilingual.text_input_catalog.TextInputStyle``) and supplies the
    localized example pairs the fixed directive template renders. It may NOT
    declare ``native_script``: every pack supports it implicitly (it is the
    default arm and renders no directive), so declaring it would be redundant
    data with nothing to carry.
    """

    model_config = ConfigDict(extra="forbid")

    style: TextInputStyle = Field(
        description="Catalog input style (closed set; never 'native_script')."
    )
    examples: list[TextInputExample] = Field(
        min_length=2,
        description="Worked typing examples (standard orthography → typed "
        "form; at least 2). Persona-neutral mechanics only.",
    )

    @field_validator("style")
    @classmethod
    def _not_native_script(cls, v: TextInputStyle) -> TextInputStyle:
        if v is TextInputStyle.NATIVE_SCRIPT:
            raise ValueError(
                "'native_script' is the implicit default for every pack — "
                "declare only the additional styles the language supports"
            )
        return v


class TextInputPackConfig(BaseModel):
    """Per-language TEXT input-style applicability + examples.

    Declares which non-default input styles this language's chat register
    actually has (hi: romanized + code_mixed; es/pt: diacritic_free), each
    with localized worked examples. A pack WITHOUT this block supports only
    ``native_script`` — pinning any other style on it fails loudly at build
    time, never silently falling back.
    """

    model_config = ConfigDict(extra="forbid")

    styles: list[TextInputStyleSpec] = Field(
        min_length=1,
        description="The non-default input styles this language supports.",
    )

    @model_validator(mode="after")
    def _no_duplicate_styles(self) -> "TextInputPackConfig":
        seen: set[TextInputStyle] = set()
        for spec in self.styles:
            if spec.style in seen:
                raise ValueError(
                    f"Duplicate text input style '{spec.style.value}' in pack"
                )
            seen.add(spec.style)
        return self

    def style_spec(self, style: TextInputStyle) -> Optional[TextInputStyleSpec]:
        """The declared spec for a style, or None if not supported."""
        for spec in self.styles:
            if spec.style is TextInputStyle(style):
                return spec
        return None


class DeliveryPackFactorRubric(BaseModel):
    """One language's rubric for a catalog DELIVERY judge factor.

    The pack SELECTS a factor from the closed catalog
    (``tau2.multilingual.delivery_catalog.DELIVERY_FACTOR_CATALOG``) by
    ``factor_id`` and supplies the language-specific listening instruction the
    multimodal audio judge reads. It may NOT invent a new factor: ``factor_id``
    is validated against the catalog and an unknown id raises, so authoring a
    language can never introduce an un-reviewed axis.

    The catalog fixes the universal metadata (category, default severity,
    documentary opportunity); only ``severity`` may be overridden per language,
    and rarely should be.
    """

    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(description="Catalog factor id (must exist in the catalog).")
    listen_for: str = Field(
        description="What the judge should listen for in the delivered audio — a "
        "concise, language-specific description grounded in confirmed human-"
        "annotation findings."
    )
    severity: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Optional per-language override of the catalog default severity.",
    )
    enabled: bool = Field(
        default=True, description="Set False to keep the rubric but skip scoring it."
    )
    shadow: bool = Field(
        default=False,
        description="Calibration mode: judge and record the factor, but exclude "
        "it from the aggregate delivery factor score.",
    )

    @field_validator("factor_id")
    @classmethod
    def _factor_in_catalog(cls, v: str) -> str:
        try:
            get_delivery_catalog_factor(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v


class DeliveryPackConfig(BaseModel):
    """Per-language delivery-judge rubrics, sourced from the pack.

    Sibling of ``NativenessPackConfig`` for the delivery (audio-layer) axis:
    the universal judge instructions stay in code
    (``tau2.judges.delivery.judge``); only per-language content lives here. A
    pack WITHOUT this block is still judged language-specifically via the fixed
    native-listener fallback prompt (see ``tau2.judges.delivery.factors``).
    """

    model_config = ConfigDict(extra="forbid")

    judge_factors: list[DeliveryPackFactorRubric] = Field(
        default_factory=list,
        description="Catalog factors enabled for this language, with what to "
        "listen for in the delivered audio.",
    )

    @model_validator(mode="after")
    def _no_duplicate_factor_ids(self) -> "DeliveryPackConfig":
        seen = set()
        for rubric in self.judge_factors:
            if rubric.factor_id in seen:
                raise ValueError(
                    f"Duplicate delivery judge factor '{rubric.factor_id}' in pack"
                )
            seen.add(rubric.factor_id)
        return self


class TurnTakingPackConfig(BaseModel):
    """Per-language overrides for the EVA turn-taking latency/silence curve.

    Every field is optional: an unset field keeps the EVA default
    (``tau2.metrics.turn_taking.TurnTakingParams``). Values are
    PROVISIONAL calibration surface — they exist so the per-language fits
    from the caller-tolerance annotations (E7) have a home in the pack; no
    shipped pack sets them until a human fit lands. All times are
    milliseconds on the tick clock.
    """

    model_config = ConfigDict(extra="forbid")

    hard_early_ms: Optional[float] = Field(
        default=None,
        description="Response onset at or before this (negative) latency "
        "scores 0 (premature interruption).",
    )
    sweet_low_ms: Optional[float] = Field(
        default=None,
        description="Lower bound of the optimal response window (score reaches 1).",
    )
    sweet_high_ms: Optional[float] = Field(
        default=None,
        description="Upper bound of the optimal response window for turns "
        "without tool calls.",
    )
    hard_late_ms: Optional[float] = Field(
        default=None,
        description="Latency at which a non-tool response scores 0.",
    )
    tool_sweet_high_ms: Optional[float] = Field(
        default=None,
        description="Upper bound of the optimal window for tool-call turns.",
    )
    tool_hard_late_ms: Optional[float] = Field(
        default=None,
        description="Latency at which a tool-call response scores 0.",
    )
    agent_overlap_hard_ms: Optional[float] = Field(
        default=None,
        description="Total agent-over-caller overlap at which the overlap "
        "sub-score reaches 0.",
    )
    user_yield_hard_ms: Optional[float] = Field(
        default=None,
        description="Yield latency at which the caller-interruption score reaches 0.",
    )


class LanguagePack(BaseModel):
    """All language-specific content for one language, bundled.

    Shared code consumes packs only through the registry
    (``tau2.multilingual.registry``); language owners author exactly one
    ``data/tau2/multilingual/<lang>/pack.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO 639-1 language code, e.g. 'hi'")
    display_name: str = Field(description="Human-readable language name")
    name_order: NameOrder = Field(
        default=NameOrder.GIVEN_FIRST,
        description="How this locale composes a DISPLAY full name: "
        "'given_first' (Wei Hu — English and most European locales) or "
        "'family_first' (Hu Wei — Mandarin, Korean, Vietnamese). The identity "
        "generator composes every localized caller name through this, so the "
        "name the caller says and the name in the agent's customer record are "
        "both in the order the language actually uses. Structured "
        "first_name/last_name fields are ROLES and never reorder; only the "
        "display string does.",
    )
    guidelines_voice_path: Optional[Path] = Field(
        default=None,
        description="Path to the localized VOICE user-simulator guidelines "
        "markdown (re-authored, not translated). RETIRED-ARM material: the "
        "runtime always reads the English guidelines, and this file is read "
        "by no run at all. Shipped packs point it into their guidelines/ "
        "subdirectory. It stays a pack field because "
        "the draft chain still authors it — Call B's body grounds the LIVE "
        "localization drafting (Call D).",
    )
    guidelines_text_path: Optional[Path] = Field(
        default=None,
        description="Path to the localized TEXT (half-duplex chat) "
        "user-simulator guidelines markdown. Same retired-arm status as "
        "guidelines_voice_path; distinct from it in that text guidelines omit "
        "spoken-form conventions (number/letter anchoring, backchannels, "
        "silence handling) and cover typed register/script/punctuation "
        "instead.",
    )
    translation_guidance: Optional[str] = Field(
        default=None,
        description="Freeform, author-written guidance for the translation factory "
        "(translator/fixer/verifier), injected into all three factory prompts. This "
        "is the per-language surface for translation-specific norms that must NOT "
        "live in the shared templates: calque traps and their natural rewrites, "
        "loanword / code-switching policy, the language's gendered-verb behavior "
        "and unmarked default, script/number conventions, and short register "
        "examples. Free text (not typed sub-fields) per the schema's "
        "'free text generalizes across languages' rule. None renders a graceful "
        "fallback string so packs without it still build.",
    )
    personas: dict[str, MultilingualPersonaConfig] = Field(
        default_factory=dict,
        description="Personas keyed by persona_id",
    )
    acoustic_presets: dict[str, AcousticPreset] = Field(default_factory=dict)
    backchannel_level: Optional[BackchannelLevel] = Field(
        default=None,
        description="Backchannel density knob for this language: low / medium / "
        "high. This is the SINGLE source of truth for how often the user "
        "backchannels — it expands (via tau2.backchannel) into a shared decision "
        "prompt whose correctness gates are identical across languages; only the "
        "density axes vary. None falls back to the English default.",
    )
    agent_language_clause: Optional[str] = Field(
        default=None,
        description="Agent-side system-prompt clause describing how to respond in "
        "this language (script conventions, mirroring the user's register, keeping "
        "tool calls in English). At runtime the resolved persona's locale is "
        "prepended as caller-background context.",
    )
    agent_native_script_db_clause: Optional[str] = Field(
        default=None,
        description="Agent-side clause for the NATIVE-SCRIPT DB ablation task "
        "sets (<domain>_<lang>_identity_native): tells the agent the customer "
        "database stores names in the language's native script and that name "
        "arguments to tools must be written in that script. Appended AFTER "
        "agent_language_clause only on native-variant runs — it must supersede "
        "that clause's keep-tool-arguments-in-English rule for name fields, and "
        "it must never ship on the folded '_identity' sets, whose DB it would "
        "lie about. Required (fail-loud) when a native-variant task set runs.",
    )
    agent_greeting: Optional[str] = Field(
        default=None,
        description="Localized agent opening greeting, used as the agent's first "
        "turn instead of the hardcoded English default ('Hi! How can I help you "
        "today?'). None keeps the English default, so English/unlocalized runs are "
        "unaffected.",
    )
    default_out_of_turn_events_per_minute: Optional[float] = Field(
        default=None,
        description="Language-level default out-of-turn speech rate. None keeps the global "
        "default.",
    )
    nativeness: Optional[NativenessPackConfig] = Field(
        default=None,
        description="Per-language nativeness judge rubrics + email symbols (see "
        "NativenessPackConfig). None means this language is not scored for nativeness "
        "yet — judge factors resolve to empty (never an error). Replaces the old "
        "hardcoded per-language dicts in tau2.judges.nativeness.factors.",
    )
    delivery: Optional[DeliveryPackConfig] = Field(
        default=None,
        description="Per-language DELIVERY judge rubrics (audio-layer defects: "
        "pronunciation, tone, stress, contour, letter/digit readout; see "
        "DeliveryPackConfig). None does NOT disable language-specific judging: "
        "the delivery judge falls back to a fixed native-listener prompt "
        "parameterized on the language (tau2.judges.delivery.factors).",
    )
    turn_taking: Optional[TurnTakingPackConfig] = Field(
        default=None,
        description="Per-language overrides for the shadow turn-taking "
        "scorer's latency/silence curve breakpoints (see "
        "TurnTakingPackConfig). None keeps the EVA defaults; overrides are "
        "provisional until fit from per-language caller-tolerance "
        "annotations.",
    )
    localization: Optional[LocalizationPackConfig] = Field(
        default=None,
        description="Per-language SPOKEN-LOCALIZATION data (see "
        "LocalizationPackConfig): native symbol readouts, worked examples for "
        "the single-language date/numeral rule, and the domain-term glossary. "
        "Rendered into the user-simulator prompt so the sim SPEAKS entities "
        "and domain terms natively — orthogonal to instruction language. None "
        "renders nothing (previous behavior).",
    )
    text_input: Optional[TextInputPackConfig] = Field(
        default=None,
        description="Per-language TEXT input-style applicability + worked "
        "examples (see TextInputPackConfig): which non-default typing "
        "registers (romanized, diacritic_free, code_mixed) this language's "
        "chat actually has. None means the pack supports only native_script "
        "typing — the default arm and today's behavior.",
    )

    @model_validator(mode="after")
    def _validate_pack(self) -> "LanguagePack":
        for persona_id, persona in self.personas.items():
            if persona.persona_id != persona_id:
                raise ValueError(
                    f"Persona key '{persona_id}' does not match its persona_id "
                    f"'{persona.persona_id}'"
                )
            if persona.language != self.language:
                raise ValueError(
                    f"Persona '{persona_id}' has language '{persona.language}' but "
                    f"the pack language is '{self.language}'"
                )
            if (
                persona.acoustic_preset_id is not None
                and persona.acoustic_preset_id not in self.acoustic_presets
            ):
                raise ValueError(
                    f"Persona '{persona_id}' references unknown acoustic preset "
                    f"'{persona.acoustic_preset_id}'"
                )
        if (
            self.guidelines_voice_path is not None
            and not Path(self.guidelines_voice_path).exists()
        ):
            raise ValueError(
                f"Guidelines file does not exist: {self.guidelines_voice_path}"
            )
        if (
            self.guidelines_text_path is not None
            and not Path(self.guidelines_text_path).exists()
        ):
            raise ValueError(
                f"Text guidelines file does not exist: {self.guidelines_text_path}"
            )
        return self

    def get_persona(self, persona_id: str) -> Optional[MultilingualPersonaConfig]:
        return self.personas.get(persona_id)

    def get_acoustic_preset(
        self, persona: MultilingualPersonaConfig
    ) -> Optional[AcousticPreset]:
        if persona.acoustic_preset_id is None:
            return None
        return self.acoustic_presets.get(persona.acoustic_preset_id)


# Matched-condition defaults shared by every generated run preset (overridable
# per experiment block): the preset-scoped audio provider plus the pinned
# DEFAULT_MATRIX_* conditions from ``tau2.config``.
PRESET_DEFAULT_AUDIO_NATIVE_PROVIDER = "openai"


class ExtraArm(BaseModel):
    """One optional extra experiment arm (e.g. an identity-variant task set)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Arm name, used in save paths and reports")
    task_set_suffix: str = Field(
        description="Localized task-set suffix the arm runs "
        "(task set '<domain>_<suffix>')"
    )


class ExperimentSpec(BaseModel):
    """One entry of the typed ``experiments`` list of a pack.yaml.

    A pack carries one spec per benchmark domain it runs (airline, telecom,
    …), each generating its own :class:`RunPreset`. Validated at pack LOAD
    time (``tau2.multilingual.loader``), so a broken entry fails discovery
    loudly instead of surfacing as a skipped run preset. ``extra='forbid'``
    rejects unknown keys — an experiment knob can only be added by extending
    this model. ``pack_assembly.build_experiment_block`` (and the
    ``tau2 factory add-experiment`` verb) emit entries that validate against
    it.
    """

    model_config = ConfigDict(extra="forbid")

    preset_name: str = Field(
        description="Registry key of the generated preset "
        "(convention: multilingual_v1_<main_arm_name>)."
    )
    description: str = Field(description="Human-readable experiment description.")
    domain: str = Field(description="Domain to run on (environment).")
    task_set_suffix: str = Field(
        description="Localized task-set suffix of the MAIN arm (must equal "
        "the pack language)."
    )
    smoke_task_stem: str = Field(
        description="Id stem of the smoke-stage task "
        "(smoke task id = '<stem>_<suffix>')."
    )
    main_arm_name: str = Field(description="Short name of the main arm.")
    include_identity_arm: bool = Field(
        default=True,
        description="Whether the identity-variant ('_identity') task set, when "
        "generated, REPLACES the plain localized set as the main arm. An "
        "explicit null in the YAML means the default (True), never False.",
    )
    extra_arms: list[ExtraArm] = Field(
        default_factory=list,
        description="Optional extra arms on other localized task sets.",
    )
    audio_native_provider: str = Field(
        default=PRESET_DEFAULT_AUDIO_NATIVE_PROVIDER,
        description="Audio-native provider for every arm (matched conditions).",
    )
    speech_complexity: str = Field(
        default=DEFAULT_MATRIX_SPEECH_COMPLEXITY,
        description="Speech complexity level for every arm (matched conditions).",
    )
    max_steps_seconds: Optional[int] = Field(
        default=DEFAULT_MATRIX_MAX_STEPS_SECONDS,
        description="Conversation duration cap in simulated seconds. An "
        "explicit null means 'use the global default' (no preset-level cap "
        "argument).",
    )
    seed: int = Field(
        default=DEFAULT_MATRIX_SEED, description="Seed shared by every arm."
    )

    @field_validator("smoke_task_stem", mode="before")
    @classmethod
    def _smoke_stem_to_str(cls, v: object) -> object:
        # Authoring convenience: YAML integers (airline's numeric task ids)
        # are accepted and normalized.
        return str(v) if isinstance(v, int) else v

    @field_validator("include_identity_arm", "extra_arms", mode="before")
    @classmethod
    def _null_means_default(cls, v: object, info) -> object:
        # An explicit `include_identity_arm: null` / `extra_arms: null` in the
        # YAML means "the default", never False/[] by accident.
        if v is None:
            return cls.model_fields[info.field_name].get_default(
                call_default_factory=True
            )
        return v
