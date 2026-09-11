# Copyright Sierra
"""The persisted translation-review row artifact and its judging provenance.

The task-translation *loop* (translate -> hard-check -> verify -> fix) is
retired and deleted: the paper language set is frozen at 11 languages plus
English, every shipped pack already carries verified localized task sets, and
the live flow is English-prose seed tasks (``tau2 factory seed-tasks``). What
survives here is the loop's **on-disk output format**, because
``tau2.annotation`` still reads it:

- :class:`TranslationState` / :class:`RowState` — the per-domain
  ``<factory_root>/<lang>/translation/<domain>_rows.json`` file that
  ``tau2 annotate translation-review`` exports for native annotators and
  ``tau2 annotate ingest`` reads filled sheets back against.
- :func:`judging_provenance` — the translator/verifier configuration an
  exported artifact was produced under, so a round traces back to it.
- :func:`build_prompt_context` — renders the translator/verifier system
  prompts for display in prompt-bed review packets.

The three system-prompt templates below are **frozen**: their sha256s are
stamped into the provenance of already-exported artifacts, so editing one
silently invalidates the traceability of every round that recorded the old
hash. A real edit must bump ``TRANSLATION_PROMPT_VERSION``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.config import (
    DEFAULT_FACTORY_TRANSLATOR_MODEL,
    DEFAULT_FACTORY_TRANSLATOR_REASONING,
    DEFAULT_FACTORY_VERIFIER_MODEL,
    DEFAULT_FACTORY_VERIFIER_REASONING,
    DEFAULT_MULTILINGUAL_DOMAIN,
)
from tau2.multilingual.domain_profiles import DOMAIN_PROFILES
from tau2.multilingual.factory.paths import (
    load_json,
    save_json,
)
from tau2.multilingual.factory.state import (
    FactoryProject,
    factory_root_dir,
)
from tau2.multilingual.registry import get_language_pack
from tau2.utils.utils import prompt_sha256

# ---------------------------------------------------------------------------
# Models and call names
# ---------------------------------------------------------------------------

# Translator/fixer prompt version. v1 stamped the first versioned hash of the
# two .format() templates (TRANSLATOR_SYSTEM_PROMPT_TEMPLATE +
# FIXER_SYSTEM_PROMPT_TEMPLATE) into provenance — any edit to either template
# must bump this (a calibration boundary, like the verifier prompt).
# v2: the translator's verbatim-value examples come from the domain profile
# ({example_entities} — airline reservation codes vs telecom phone numbers)
# instead of hardcoded airline nouns.
TRANSLATION_PROMPT_VERSION = "v2"

CALL_TRANSLATE = "factory_translate"
CALL_VERIFY = "factory_verify"
CALL_FIX = "factory_fix"


class RowStatus(str, Enum):
    """Translation-row lifecycle outcome."""

    PENDING = "pending"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    UNTRANSLATED = "untranslated"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRANSLATOR_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert {display_name} localizer for voice-agent benchmark tasks.
You translate ENGLISH user-simulator task prose into {display_name} (ISO \
639-1 '{language}'), written in the '{script_code}' script (ISO 15924).

Hard rules — violations are mechanically rejected:
- Every concrete value listed for a row ({example_entities}, and every
  numeric run) must appear VERBATIM in Latin script with Latin digits. Never
  translate, transliterate, reformat, or localize a digit.
- The prose itself must actually be localized: it must contain
  '{script_code}'-script characters, not romanized {display_name}.
- Translations are PERSONA-NEUTRAL. Do not bake in any register, dialect,
  politeness level, or persona voice: speaker register is injected at runtime
  via persona guidelines, never via task prose. Use a neutral, natural,
  standard register.
- Value/unit PRINCIPLE: a concrete VALUE stays verbatim in Latin script with
  Latin digits (see the hard rule above), but a unit or word attached to it
  localizes — keep the digit and translate the attached unit/word into the
  target language (e.g. "3h" -> the digit 3 plus the local word for "hours").
- Use a single, consistent speaker voice throughout: do not oscillate between
  imperative and descriptive phrasing within the same text.
- Preserve the meaning exactly: conditions ("if the agent refuses..."),
  directions (from X to Y), dates, amounts, and the ordering of steps must
  survive precisely.
- Preserve line breaks and paragraph structure where natural.

Per-language translation guidance from the language pack (calque traps,
loanword/code-switching norms, gender behavior, script/number conventions):
{translation_guidance}

Respond with ONLY the translation text. No preamble, no quotes, no notes.
"""

FIXER_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert {display_name} localizer fixing a rejected translation of
voice-agent benchmark task prose (English -> {display_name}, '{script_code}'
script).

You will be given the English source, the current (rejected) translation, and
the list of issues found. Produce a corrected translation that fixes EVERY
listed issue while obeying the hard rules:
- every listed concrete value verbatim in Latin script with Latin digits;
- prose genuinely in the '{script_code}' script;
- persona-neutral, natural standard register;
- value/unit PRINCIPLE: a concrete VALUE stays verbatim in Latin script with
  Latin digits, but a unit or word attached to it localizes — keep the digit
  and translate the unit/word into the target language (e.g. "3h" -> the digit
  3 plus the local word for "hours").
- a single, consistent speaker voice: do not oscillate between imperative and
  descriptive phrasing within the text.
- meaning preserved exactly (conditions, directions, dates, amounts,
  sequencing).

Per-language translation guidance from the language pack (calque traps,
loanword/code-switching norms, gender behavior, script/number conventions):
{translation_guidance}

Respond with ONLY the corrected translation text.
"""

# NOTE: this is a .format() template (the constant name is kept as
# VERIFIER_SYSTEM_PROMPT so calibration's verifier_prompt_sha256() keeps
# hashing it for provenance). The instantiated verifier prompt =
# this template ⊕ the pack's per-language translation_guidance @ its
# pack_revision. Literal JSON braces in the response block are escaped
# ({{ }}) so .format() leaves them intact. Any edit here invalidates
# (intentionally) calibration-round comparisons.
VERIFIER_SYSTEM_PROMPT = """\
You are a bilingual quality judge for localized voice-agent benchmark tasks.
You will be given an English source text and its candidate translation, and
you judge the PAIR DIRECTLY — do not assume either text is correct.

Judge two things:

1. SEMANTIC EQUIVALENCE — the translation must preserve the source meaning
   exactly. Scrutinize especially:
   - conditional structure: "if the agent refuses, then..." must keep the
     same trigger and the same consequence, never become unconditional;
   - directions and routes: "from Philadelphia to LaGuardia" must not flip;
   - dates, times, and amounts: same values, same roles (a refund OF $250 is
     not a fee of $250);
   - sequencing and ordering of steps ("first insist once, then accept");
   - negations and modality (must / may / refuse / don't want).
   Grammatical gender agreement that the target language FORCES (e.g. a verb
   inflecting for the speaker's gender) is NOT a meaning difference: "you want
   X" and its gendered target rendering are semantically equivalent. Do not
   set equivalent=false for forced gender agreement.

2. NATURALNESS (advisory) AND REGISTER (gating). Judge naturalness against how
   a real speaker of this language's everyday spoken register actually talks
   (see the per-language translation guidance below), NOT against
   formal/literary standard prose. Set "register_ok" to false ONLY for a WRONG
   FORMALITY LEVEL (e.g. stiff literary/formal prose where a casual spoken
   register is expected, or vice versa) — NOT for loanword density. When you
   set register_ok=false, name
   the specific formality problem in "register_issues" (that list is what the
   fixer acts on); keep advisory observations (calques, awkwardness) in
   "naturalness_issues". In particular:
   - HARD rule: a loanword for a travel/tech/number/everyday term that matches
     the target spoken register is NEVER, by itself, a naturalness or register
     defect. Code-switching and loanwords that match this register (e.g.
     English work/tech/number terms woven into the matrix language, where the
     pack's guidance states that is the norm) are the INTENDED style — do NOT
     flag them as unnatural, as "too much English," or as a register problem.
     Only flag borrowing that a real speaker of this
     register would genuinely find odd, and even then as a naturalness note,
     not as register_ok=false.
   - Still flag genuine machine-translationese and awkward calques as
     naturalness issues.

Per-language translation guidance from the language pack (the target everyday
register, calque traps, loanword/code-switching norms, the language's gender
behavior, script/number conventions). Apply it when judging this pair — e.g. a
calque it warns about is a naturalness issue, and a loanword policy it states
overrides any instinct to flag code-switching:
{translation_guidance}

Also produce a faithful literal back-translation of the candidate into
English. The back-translation is SUPPORTING EVIDENCE for human reviewers —
base your verdict on the bilingual pair itself, not on the back-translation
alone.

Respond with ONLY a JSON object:
{{
  "equivalent": <bool>,
  "equivalence_issues": ["<each concrete meaning problem, or empty>"],
  "natural": <bool>,
  "register_ok": <bool>,
  "register_issues": ["<each WRONG-FORMALITY / register problem when register_ok is false (this is what the fixer acts on), or empty>"],
  "naturalness_issues": ["<each advisory naturalness problem: calques, machine-translationese, awkwardness — NOT loanword density, or empty>"],
  "back_translation": "<literal English back-translation>"
}}
"""


# ---------------------------------------------------------------------------
# Row state
# ---------------------------------------------------------------------------


class VerifierVerdict(BaseModel):
    """The verifier's parsed JSON verdict (or a synthesized ERROR verdict).

    The typed LLM boundary for ``CALL_VERIFY``: quirk-absorbing validators
    accept the model's occasional null/non-list issue fields, and every gate
    reads typed attributes instead of ``dict.get`` chains.

    Gating is ``equivalent AND register_ok`` only. ``natural`` /
    ``naturalness_issues`` are ADVISORY: recorded for the calibrator, never
    fed to the fixer, never blocking (naturalness is scored by the judge
    pillar, not the translation loop).
    """

    equivalent: bool = Field(
        default=False, description="Semantic equivalence gate (blocks when false)"
    )
    equivalence_issues: list[str] = Field(
        default_factory=list,
        description="Concrete meaning problems (fed to the fixer)",
    )
    natural: bool = Field(
        default=False, description="ADVISORY naturalness flag (never gates)"
    )
    register_ok: bool = Field(
        default=False, description="Formality/register gate (blocks when false)"
    )
    register_issues: list[str] = Field(
        default_factory=list,
        description="Wrong-formality problems (fed to the fixer)",
    )
    naturalness_issues: list[str] = Field(
        default_factory=list,
        description="ADVISORY calque/translationese notes (never fed to the fixer)",
    )
    back_translation: str = Field(
        default="", description="The verifier's literal English back-translation"
    )
    error: Optional[str] = Field(
        default=None,
        description="Synthesized when the verifier returned no usable JSON "
        "(the row then fails this attempt loudly)",
    )

    @field_validator(
        "equivalence_issues", "register_issues", "naturalness_issues", mode="before"
    )
    @classmethod
    def _absorb_issue_list(cls, v: object) -> list[str]:
        # LLM quirks: null instead of [], a bare string instead of a list.
        if v is None:
            return []
        if not isinstance(v, list):
            return [str(v)]
        return [str(item) for item in v]

    @field_validator("equivalent", "natural", "register_ok", mode="before")
    @classmethod
    def _null_bool_is_false(cls, v: object) -> object:
        return False if v is None else v

    @field_validator("back_translation", mode="before")
    @classmethod
    def _absorb_null_text(cls, v: object) -> str:
        return "" if v is None else str(v)

    @property
    def passes(self) -> bool:
        """Whether the verdict accepts the translation (equivalent AND
        register_ok; ``natural`` is advisory and never gates)."""
        return self.equivalent and self.register_ok

    def gating_issues(self) -> list[str]:
        """The GATING issues, for the fixer prompt.

        Only equivalence and register problems drive the fix loop —
        ``naturalness_issues`` are deliberately excluded (advisory only). A
        rejection that names nothing still yields one explicit issue so the
        fixer prompt is never empty.
        """
        issues: list[str] = []
        if not self.equivalent:
            issues.extend(self.equivalence_issues)
        if not self.register_ok:
            issues.extend(self.register_issues)
        if self.error:
            issues.append(self.error)
        if not issues and not self.passes:
            issues.append("verifier rejected the translation without naming issues")
        return issues


class ParityEvidence(BaseModel):
    """Parity-probe evidence attached to a flagged row.

    Was produced by the retired parity probe and rendered into the retired
    fixer prompt. Retained as a schema field only so archived
    ``<domain>_rows.json`` files that recorded it still validate on load —
    nothing writes it any more.
    """

    delta: Optional[float] = Field(
        default=None, description="Task reward delta (localized - english)"
    )
    aggregate_gap: Optional[float] = Field(
        default=None, description="Language-level mean delta at flag time"
    )
    excess: Optional[float] = Field(
        default=None, description="delta - aggregate_gap (how far below the gap)"
    )
    flagged_criterion: Optional[str] = Field(
        default=None, description="The most-diverging evaluation criterion"
    )
    failure_excerpt: Optional[str] = Field(
        default=None, description="Failure excerpt from the worst localized trial"
    )


class TranslationAttempt(BaseModel):
    """One translator/fixer output and what the gates said about it."""

    translation: str = Field(description="The candidate translation text")
    hard_problems: list[str] = Field(
        default_factory=list,
        description="Hard-check problems (empty == passed; verifier ran)",
    )
    verdict: Optional[VerifierVerdict] = Field(
        default=None,
        description="The verifier's typed verdict (None if the hard check "
        "failed, so the verifier was never called)",
    )

    @property
    def passed_hard(self) -> bool:
        return not self.hard_problems

    @property
    def passed(self) -> bool:
        return self.passed_hard and self.verdict is not None and self.verdict.passes


class RowState(BaseModel):
    """One extract row's full translation history and outcome."""

    row_id: int = Field(description="1-based extract row id (stable ordering)")
    task_id: str = Field(description="SOURCE (English) task id")
    field: str = Field(
        description="Field label as emitted by localize_lib.iter_rows, e.g. "
        "'task_instructions' or 'purpose (metadata)'"
    )
    english: str = Field(description="The English source text for this row")
    status: RowStatus = Field(
        default=RowStatus.PENDING,
        description="pending | verified | flagged | untranslated",
    )
    attempts: list[TranslationAttempt] = Field(default_factory=list)
    translator_model: Optional[str] = Field(default=None)
    verifier_model: Optional[str] = Field(default=None)
    # Historical: the retired parity probe attached this when it flagged a
    # task as a behavioral regression. Read-only now — kept so archived rows
    # that carry it still validate.
    parity_evidence: Optional[ParityEvidence] = Field(
        default=None,
        description="Parity-probe evidence recorded by the retired parity "
        "probe: delta stats, failing criterion, failure excerpt from the "
        "worst localized trial. Never written any more.",
    )

    @property
    def final_translation(self) -> Optional[str]:
        """The row's best translation: the verified one, or for flagged rows
        the last attempt that passed the hard check. None if nothing ever
        passed the hard check."""
        for attempt in reversed(self.attempts):
            if attempt.passed_hard:
                return attempt.translation
        return None

    @property
    def last_issues(self) -> list[str]:
        """The most recent rejection reasons (hard problems or verdict issues)."""
        if not self.attempts:
            return []
        last = self.attempts[-1]
        if last.hard_problems:
            return list(last.hard_problems)
        if last.verdict is None:
            return ["verifier returned no verdict"]
        return last.verdict.gating_issues()


class TranslationState(BaseModel):
    """The persisted per-domain row file (``<domain>_rows.json``)."""

    language: str
    domain: str
    script_code: str
    rows: list[RowState] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        save_json(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> "TranslationState":
        return cls.model_validate(load_json(path))


def translation_dir(lang: str) -> Path:
    return factory_root_dir() / lang / "translation"


def rows_path(lang: str, domain: str) -> Path:
    return translation_dir(lang) / f"{domain}_rows.json"


def load_translation_state(lang: str, domain: str) -> Optional[TranslationState]:
    """The persisted translation rows for one (language, domain), if any.

    The typed loader downstream consumers (annotation exports) use instead of
    hand-opening ``rows_path`` JSON.
    """
    path = rows_path(lang, domain)
    if not path.exists():
        return None
    return TranslationState.load(path)


def verifier_prompt_sha256() -> str:
    """sha256 of the verifier prompt template — recorded in export provenance."""
    return prompt_sha256(VERIFIER_SYSTEM_PROMPT)


def translator_prompt_sha256() -> str:
    """sha256 of the fixed translator+fixer templates — recorded in provenance."""
    return prompt_sha256(
        TRANSLATOR_SYSTEM_PROMPT_TEMPLATE,
        FIXER_SYSTEM_PROMPT_TEMPLATE,
        TRANSLATION_PROMPT_VERSION,
    )


def pack_revision(lang: str) -> int:
    """The factory project's pack revision (0 if no project exists).

    Read through the typed ``FactoryProject`` loader — a corrupt project.json
    fails LOUD instead of silently reporting revision 0 in provenance.
    """
    project = FactoryProject.load(lang)
    return 0 if project is None else project.pack_revision


class FactoryProvenance(BaseModel):
    """The judging configuration a calibration/annotation round measured.

    Everything needed to trace an exported artifact back to the exact
    translator/verifier setup that produced the judged rows.
    """

    translator_model: Annotated[
        str, Field(description="Model id used for translation drafts.")
    ]
    verifier_model: Annotated[
        str, Field(description="Model id used for row verification.")
    ]
    verifier_reasoning: Annotated[
        str, Field(description="Reasoning effort the verifier ran with.")
    ]
    verifier_prompt_sha256: Annotated[
        str, Field(description="sha256 of the verifier prompt template.")
    ]
    translator_prompt_sha256: Annotated[
        str,
        Field(
            description="sha256 of the translator+fixer prompt templates "
            f"(version {TRANSLATION_PROMPT_VERSION})."
        ),
    ]
    pack_revision: Annotated[
        int, Field(description="Factory project's pack revision at export time.")
    ]


def judging_provenance(lang: str) -> FactoryProvenance:
    """The current judging configuration for one language's factory workspace."""
    return FactoryProvenance(
        translator_model=DEFAULT_FACTORY_TRANSLATOR_MODEL,
        verifier_model=DEFAULT_FACTORY_VERIFIER_MODEL,
        verifier_reasoning=DEFAULT_FACTORY_VERIFIER_REASONING,
        verifier_prompt_sha256=verifier_prompt_sha256(),
        translator_prompt_sha256=translator_prompt_sha256(),
        pack_revision=pack_revision(lang),
    )


# ---------------------------------------------------------------------------
# Prompt context (rendered into prompt-bed review packets)
# ---------------------------------------------------------------------------


class PromptContext(BaseModel):
    """The row-independent half of the retired loop's prompt configuration.

    Still built for display: prompt-bed review packets render these system
    prompts so a reviewer sees the exact text the translator/verifier ran
    under.
    """

    model_config = ConfigDict(frozen=True)

    language: str
    display_name: str
    script_code: str
    translator_model: str
    verifier_model: str
    translator_reasoning: str
    verifier_reasoning: str
    translator_system: str
    fixer_system: str
    verifier_system: str


def build_prompt_context(
    lang: str,
    script_code: str,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    translator_model: Optional[str] = None,
    verifier_model: Optional[str] = None,
    translator_reasoning: Optional[str] = None,
    verifier_reasoning: Optional[str] = None,
) -> PromptContext:
    """Build the run's prompt context from the language pack.

    Model/reasoning overrides come from the CLI flags; None means the
    ``DEFAULT_FACTORY_*`` constants in ``tau2.config``. The translator's
    illustrative verbatim-value entities come from the domain profile; an
    unprofiled (test) domain gets a generic description.
    """
    pack = get_language_pack(lang)
    display_name = pack.display_name if pack else lang
    translation_guidance = (
        pack.translation_guidance
        if pack and pack.translation_guidance
        else "(no per-language translation guidance provided — apply the general "
        "rules above for this language)"
    )
    profile = DOMAIN_PROFILES.get(domain)
    example_entities = (
        profile.translation_example_entities
        if profile
        else "emails, ids, codes, amounts, dates"
    )
    fields = {
        "display_name": display_name,
        "language": lang,
        "script_code": script_code,
        "example_entities": example_entities,
    }
    return PromptContext(
        language=lang,
        display_name=display_name,
        script_code=script_code,
        translator_model=translator_model or DEFAULT_FACTORY_TRANSLATOR_MODEL,
        verifier_model=verifier_model or DEFAULT_FACTORY_VERIFIER_MODEL,
        translator_reasoning=translator_reasoning
        or DEFAULT_FACTORY_TRANSLATOR_REASONING,
        verifier_reasoning=verifier_reasoning or DEFAULT_FACTORY_VERIFIER_REASONING,
        translator_system=TRANSLATOR_SYSTEM_PROMPT_TEMPLATE.format(
            translation_guidance=translation_guidance,
            **fields,
        ),
        fixer_system=FIXER_SYSTEM_PROMPT_TEMPLATE.format(
            translation_guidance=translation_guidance,
            **fields,
        ),
        verifier_system=VERIFIER_SYSTEM_PROMPT.format(
            translation_guidance=translation_guidance,
        ),
    )
