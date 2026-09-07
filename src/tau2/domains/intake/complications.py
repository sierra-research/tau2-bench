# Copyright Sierra
"""Scripted caller complications for the intake domain.

A complication is ONE scripted caller behavior sampled for a triggered intake
call. Most kinds render as an additive line appended to the user-simulator
prompt; prompt-silent kinds (``mispronounced_term``) leave the prompt
byte-identical and take effect elsewhere. Which calls trigger is governed by
the run's :class:`tau2.data_model.simulation.ComplicationProfile` (catalog
v2.0.0, replacing the flat 25%-then-uniform draw): ONE categorical draw per
task over ``{none} ∪ kinds``, where every FEASIBLE kind carries its
profile rate (literature-anchored constants below for the ``default``
profile; 1.0 everywhere for ``hard``) and infeasible kinds' probability mass
goes to "none" — never renormalized among the feasible kinds, so each kind's
conditional-on-feasible rate is a task-independent constant and a constrained
task simply triggers less often. Everything is deterministic from
``(run seed, task id, catalog version)`` plus the profile, so the same seed
renders byte-identical lines in the pre-run review packet
(``tau2 intake-tasks complications-packet``) and in the run itself. The
sampled complication is recorded on each ``SimulationRun`` (see
:class:`tau2.data_model.simulation.SampledComplication`).

Phase-1 kinds (:class:`ComplicationKind`):

- ``self_correction`` — the caller first misspeaks the value as a decoy drawn
  from the SAME bank and tier's value pool (gated fold-distinct from the gold
  value), then immediately corrects themselves in the same turn and is
  accurate afterwards.
- ``spelling_style`` — a letter/number rendering variation in spell-outs,
  from the closed style sub-catalog (:class:`SpellingStyle`): grouped
  numbers, "oh" for zero, "zed" for Z, doubled characters. VOICE-ONLY as of
  catalog v2.3.0: every style scripts how a value is voiced and has no
  typed analogue.
- ``lazy_omission`` — the caller FIRST gives the value with a component
  missing (AM/PM for times, the year for dates, the last name for person
  names) and only supplies it when asked.
- ``wrong_slot`` — on the first ask the caller misunderstands and answers
  with a different personal fact (a value from one of the task's pre-filled
  context fields), then gives the right value when the agent re-asks.
- ``spell_correction`` (catalog v2.4.0) — when the caller actually spells the
  value character by character (or reads it out digit by digit), they falter
  partway — a SEEDED single edit (substitute or drop one character) applied
  to a prefix of the gold sequence — break off with a correction marker from
  a small closed set, and restart, giving the complete correct sequence.
  Distinct from ``self_correction`` (a wrong VALUE, then the right value):
  this is a wrong RENDERING mid-spell-out, then a restart. VOICE-ONLY, and
  conditional in effect on a spell-out actually happening — in the
  free-strategy arm (``intake_free``) that opportunity is usually
  AGENT-elicited, so the effective per-call rate further conditions on agent
  behavior (like ``spelling_style``).

Catalog v1.2.0 adds ``mispronounced_term`` (spec:
docs/designs/intake-mispronunciation.md sec. 5) — the caller consistently
mispronounces a pronunciation-bearing token of the gold value (a drug name or
a hard-tier person-name token). It is VOICE-ONLY (``sample_complication``
takes an explicit ``channel``; text runs never draw it) and PROMPT-SILENT
(``user_prompt_task`` skips it — the sim's prompt is byte-identical to an
uncomplicated run; ``SampledComplication.line`` is packet-display text only).
The bad respelling comes from the bank's curated ``pronunciations`` column,
never invented, and is recorded on ``SampledComplication.mispronunciation``;
it takes effect at the voice-synthesis seam (the drawn term's entry in the
run's pre-synthesis substitution map is overridden with the bad respelling —
see :mod:`tau2.domains.intake.pronunciation_map`).

Every line template is fixed in-code (machine-not-scripts mandate), ASCII
only, and SELF-CONTAINED: it carries its own carve-out from the simulator's
"never invent values" rule, so a complication never licenses free-form
invention beyond the scripted behavior.
"""

import hashlib
import json
import random
import re
from enum import Enum
from functools import lru_cache
from typing import Annotated, Literal, Optional

from pydantic import Field

from tau2.data_model.simulation import ComplicationProfile, SampledComplication
from tau2.data_model.tasks import Task
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.tasks.banks import (
    BANK_NAMES,
    Banks,
    Difficulty,
    TokenPronunciation,
    instantiate_email,
    load_banks,
)
from tau2.domains.intake.utils import spoken_form
from tau2.utils.pydantic_utils import BaseModelNoExtra

# v2.0.0: complication profiles (owner directive 2026-08-26) — the flat
# rate-then-uniform draw is replaced by one categorical draw over
# {none} ∪ kinds with per-kind rates (PROFILE_KIND_RATES), so every draw
# re-keys against v1.2.0. The version keys the per-task RNG — re-render the
# review packet after any bump.
# v2.1.0: mispronounced_term rate split per bank (owner directive
# 2026-08-26): medications 0.50, person_names 0.15
# (MISPRONOUNCED_TERM_BANK_RATES) — trigger outcomes change on
# mispronounceable tasks.
# v2.2.0: properties + vehicles gain foreign-token pronunciation columns
# (design doc docs/designs/intake-mispronunciation.md sec. 8b), so
# mispronounced_term now applies to both at the person_names rate —
# trigger outcomes change on property/vehicle tasks. Re-render the seeded
# complication review packet before the next armed run.
# v2.3.0: spelling_style is voice-gated (owner directive 2026-08-27, text
# arm): every style scripts HOW a value is VOICED ("oh" for zero, "zed",
# grouped digits, "double five") and has no typed analogue, so a text run
# never draws it — per the mass-to-none rule, text triggers correspondingly
# less often. Voice draws re-key on the version bump; re-render the review
# packet for both channels before the next armed run.
# v2.4.0: adds spell_correction (owner directive 2026-09-01, free-strategy
# arm): a mid-spell-out falter-and-restart — seeded single edit on a prefix
# of the gold sequence, closed correction-marker set, then the complete
# correct sequence. Voice-only, all banks, feasibility-gated on the gold
# value carrying >= 4 spellable characters. Every draw re-keys on the bump;
# re-render the review packet before the next armed run.
COMPLICATION_CATALOG_VERSION = "2.4.0"

Channel = Literal["text", "voice"]

_CHANNELS: tuple[str, ...] = ("text", "voice")


class ComplicationKind(str, Enum):
    """Closed set of scripted caller complications."""

    SELF_CORRECTION = "self_correction"
    SPELLING_STYLE = "spelling_style"
    LAZY_OMISSION = "lazy_omission"
    WRONG_SLOT = "wrong_slot"
    MISPRONOUNCED_TERM = "mispronounced_term"
    SPELL_CORRECTION = "spell_correction"


class SpellingStyle(str, Enum):
    """Closed sub-catalog of spell-out rendering styles."""

    GROUPED_NUMBERS = "grouped_numbers"  # "forty-four", not "four, four"
    OH_FOR_ZERO = "oh_for_zero"  # says "oh" for the digit 0
    UK_LETTERS = "uk_letters"  # says "zed" for the letter Z
    DOUBLED_LETTERS = "doubled_letters"  # "double five" for 55


# ---------------------------------------------------------------------------
# Per-kind rates (ComplicationProfile.DEFAULT), literature-anchored.
#
# Each constant is the kind's PER-OPPORTUNITY CONDITIONAL rate: the
# probability that a task on which the kind is feasible draws it. The
# constructs in the literature rarely match ours exactly — every constant
# below says how far its anchor is from what we script, and rates marked
# JUDGMENT have no direct corpus number at all. Where the literature supports
# a range, the constant sits at the conservative (low) end. Full citations:
# docs/designs/intake-complication-profiles.md ("Rate grounding") and the
# references bundled with the tau-Elicitation paper.
# ---------------------------------------------------------------------------

# Spontaneous-speech self-repair. Anchors: Levelt 1983 (Cognition 14) —
# taxonomy of overt self-repairs; Shriberg 1994 (PhD diss., UC Berkeley) —
# disfluency rates in three speech corpora; Fox Tree 1995 (JML 34) — false
# starts/repetitions in spontaneous speech; Bortfeld et al. 2001
# (Language & Speech 44) — ~5.97 disfluencies per 100 words, of which
# restarts/repairs ~1.7-2.2 per 100 words. CONSTRUCT GAP: those counts
# include abandoned starts and appropriateness repairs of ANY word; ours is
# one substantive misspeak-then-correct of the single information-bearing
# value in the call. Substantive repairs of content words run single-digit
# percent per utterance; conservative end.
SELF_CORRECTION_RATE = 0.05

# Marked spell-out rendering conventions ("oh" for zero, grouped digits,
# "double five", "zed"). This is DIALECT PREVALENCE, not an error rate:
# "oh" dominates casual US phone-number speech and digit-grouping is
# routine; "double"/"treble" collapsing is British/Commonwealth (Murphy,
# "Separated by a Common Language", 2007). JUDGMENT: no corpus prevalence
# number found for any single convention; prevalence of using SOME marked
# convention plausibly exceeds 0.5, but our construct scripts one style
# applied consistently for the whole call, so the rate sits far below raw
# prevalence.
SPELLING_STYLE_RATE = 0.15

# Under-specification of inferable value components (year, AM/PM, last
# name). JUDGMENT: no direct corpus number found (searched
# appointment-scheduling dialog corpora and the TimeML/TempEval literature
# 2026-08-26). Qualitative anchors: Grice 1975 (quantity maxim — omission of
# inferable components is unmarked, cooperative speech); the TimeML/TempEval
# normalization literature (Pustejovsky et al. 2003; UzZaman et al. 2013)
# treats contextually underspecified temporal expressions as pervasive
# enough to be the main normalization error source.
LAZY_OMISSION_RATE = 0.10

# Mid-spell-out restart: falter partway through a character-by-character
# spell-out (or long digit read-out), break off, and start the sequence
# over. Anchors: Levelt 1983 (Cognition 14) — restarts are a core class in
# the self-repair taxonomy; Shriberg 1994 (PhD diss., UC Berkeley) —
# disfluency (incl. restart) rates rise with utterance length, which
# unit-by-unit sequences maximize; Bortfeld et al. 2001 (Language & Speech
# 44) — restarts/repairs ~1.7-2.2 per 100 words. CONSTRUCT GAP: no corpus
# number found for restarts WITHIN letter-by-letter spell-outs specifically;
# JUDGMENT at the conservative end of the 0.10-0.15 range read off the
# adjacent restart-disfluency rates. PER-OPPORTUNITY CONDITIONAL twice over:
# the drawn line only fires when a spell-out/read-out actually happens, and
# in the free-strategy arm (intake_free) that opportunity is usually
# AGENT-elicited — the effective per-call rate further conditions on agent
# behavior, like spelling_style (docs/designs/intake-free-strategy.md).
SPELL_CORRECTION_RATE = 0.10

# Answering a different question than asked (misunderstanding the slot).
# Anchor: Dingemanse et al. 2015 (PLOS ONE 10:e0136100) — other-initiated
# repair runs ~once per 1.4 minutes of conversation across 12 languages,
# i.e. a few percent of turns need the recipient to signal trouble; Purver
# et al. 2018 (Topics in Cognitive Science 10) surveys computational models
# of the same phenomena. CONSTRUCT GAP: OIR covers all trouble sources
# (mishearing, non-understanding); ours is the specific
# answered-the-wrong-question subset, so the rate sits below the OIR rate.
WRONG_SLOT_RATE = 0.03

# Consistently mispronouncing a pronunciation-bearing token of the gold
# value (voice-only; conditional on the gold value carrying a curated
# ``mispronounced`` column). Split PER BANK (owner directive 2026-08-26):
# the grounding literature is medication-specific, and a caller struggles
# with a drug name far more than with a name they chose to give.
#
# Medications — the adjacent literature is medication NAMING failure, not
# spoken mispronunciation: Persell et al. 2007 (JGIM 22; Northwestern
# Feinberg) — 40.5% vs 68.3% of hypertensive patients with inadequate vs
# adequate health literacy could name any of their antihypertensives;
# naming-recall failure spans roughly 30-78% across studies. Drug-name
# pronunciation difficulty itself is documented qualitatively
# ("Unpronounceable drug names", PMC6299177; the DrugSpeak
# pharmacy-education program, Pharmacy Education 2023) but without a
# population rate. JUDGMENT anchored to the ADJACENT naming-recall
# construct, at the middle of its 30-78% range.
MISPRONOUNCED_TERM_MEDICATIONS_RATE = 0.50

# Person names — no quantitative literature found for callers
# mispronouncing personal names they are reporting; rarer than drug names
# (a caller often knows the bearer) but real for hard-tier foreign or
# unusual spellings. JUDGMENT.
MISPRONOUNCED_TERM_PERSON_NAMES_RATE = 0.15

# The DEFAULT profile's mispronounced_term rate per bank. Closed: keys must
# equal MISPRONOUNCED_TERM_BANKS exactly (validate_profile_rates), so a
# bank gaining pronunciation columns without a rate decision fails loud.
MISPRONOUNCED_TERM_BANK_RATES: dict[str, float] = {
    "medications": MISPRONOUNCED_TERM_MEDICATIONS_RATE,
    "person_names": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
    # Properties and vehicles carry NAME-LIKE foreign tokens (hotel and car
    # model names, design doc sec. 8b) — same construct as person names (a
    # caller reading an unfamiliar proper name off a document), so they
    # share the person_names rate. JUDGMENT, no independent literature.
    "properties": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
    "vehicles": MISPRONOUNCED_TERM_PERSON_NAMES_RATE,
}

# Per-profile per-kind rates for the BANK-INVARIANT kinds. DEFAULT carries
# the literature-anchored constants above and deliberately OMITS
# mispronounced_term — that kind's DEFAULT rate is per bank
# (MISPRONOUNCED_TERM_BANK_RATES), resolved by profile_kind_rates(). HARD
# puts every kind (mispronounced_term included) at 1.0 — each call draws a
# complication, uniform among the feasible kinds (equal rates make the
# proportional draw uniform), for measurement power. HARD additionally
# restricts the run to hard-tier tasks (tau2.runner.complications
# profile_task_filter); that restriction lives at task-selection level, not
# here. Closed catalog: validate_profile_rates() gates coverage.
PROFILE_KIND_RATES: dict[ComplicationProfile, dict[ComplicationKind, float]] = {
    ComplicationProfile.DEFAULT: {
        ComplicationKind.SELF_CORRECTION: SELF_CORRECTION_RATE,
        ComplicationKind.SPELLING_STYLE: SPELLING_STYLE_RATE,
        ComplicationKind.LAZY_OMISSION: LAZY_OMISSION_RATE,
        ComplicationKind.WRONG_SLOT: WRONG_SLOT_RATE,
        ComplicationKind.SPELL_CORRECTION: SPELL_CORRECTION_RATE,
    },
    ComplicationProfile.HARD: {kind: 1.0 for kind in ComplicationKind},
}


def profile_kind_rates(
    profile: ComplicationProfile, bank: str
) -> dict[ComplicationKind, float]:
    """The complete per-kind rate vector for one task's bank.

    HARD is 1.0 everywhere. DEFAULT takes the bank-invariant constants plus
    the bank's mispronounced_term rate — 0.0 for banks with no curated
    variants (the kind is structurally infeasible there, so the zero is the
    honest mass). Unknown banks fail loud.
    """
    if bank not in BANK_COMPLICATIONS:
        raise ValueError(f"Unknown bank {bank!r}: no complication mapping")
    if profile is ComplicationProfile.HARD:
        return dict(PROFILE_KIND_RATES[ComplicationProfile.HARD])
    rates = dict(PROFILE_KIND_RATES[ComplicationProfile.DEFAULT])
    rates[ComplicationKind.MISPRONOUNCED_TERM] = MISPRONOUNCED_TERM_BANK_RATES.get(
        bank, 0.0
    )
    return rates


def effective_kind_rates(
    profile: ComplicationProfile, rate_override: Optional[float], bank: str
) -> dict[ComplicationKind, float]:
    """The per-kind rates a task in ``bank`` draws with at this profile +
    override.

    No override returns the bank's profile vector
    (:func:`profile_kind_rates`) verbatim. An explicit
    ``--complication-rate`` R is a UNIFORM TRIGGER SCALING: every per-kind
    rate is multiplied by ``R / sum(the bank vector)``, so a task on which
    every kind applicable to its bank is feasible triggers with probability
    exactly R while the profile keeps contributing the kind MIX (and
    constrained tasks still trigger proportionally less — the mass-to-none
    rule is untouched). R=0 runs clean. R outside [0, 1] fails loud.
    """
    rates = profile_kind_rates(profile, bank)
    if rate_override is None:
        return rates
    if not 0.0 <= rate_override <= 1.0:
        raise ValueError(
            f"--complication-rate must be within [0, 1], got {rate_override}"
        )
    scale = rate_override / sum(rates.values())
    return {kind: rate * scale for kind, rate in rates.items()}


def validate_profile_rates() -> None:
    """Gate the profile-rate catalog. Fails loud on any hole.

    Called at import time (below) and directly by the test suite: every
    profile covers every kind, every rate is a probability, DEFAULT's rates
    sum below 1 (the remainder is the "none" mass on a fully-feasible task),
    and HARD is 1.0 everywhere (every call draws; uniform among feasible).
    """
    if set(PROFILE_KIND_RATES) != set(ComplicationProfile):
        raise ValueError(
            "PROFILE_KIND_RATES out of sync with ComplicationProfile: "
            f"missing={sorted(p.value for p in set(ComplicationProfile) - set(PROFILE_KIND_RATES))}"
        )
    bank_invariant = set(ComplicationKind) - {ComplicationKind.MISPRONOUNCED_TERM}
    if set(PROFILE_KIND_RATES[ComplicationProfile.DEFAULT]) != bank_invariant:
        raise ValueError(
            "DEFAULT profile carries exactly the bank-invariant kinds — "
            "mispronounced_term's DEFAULT rate lives per bank in "
            "MISPRONOUNCED_TERM_BANK_RATES"
        )
    if set(MISPRONOUNCED_TERM_BANK_RATES) != set(MISPRONOUNCED_TERM_BANKS):
        raise ValueError(
            "MISPRONOUNCED_TERM_BANK_RATES out of sync with "
            "MISPRONOUNCED_TERM_BANKS: every mispronounceable bank carries "
            f"exactly one rate; got {sorted(MISPRONOUNCED_TERM_BANK_RATES)} "
            f"vs {sorted(MISPRONOUNCED_TERM_BANKS)}"
        )
    if set(PROFILE_KIND_RATES[ComplicationProfile.HARD]) != set(ComplicationKind):
        raise ValueError("HARD profile rates cover every kind")
    for profile, rates in PROFILE_KIND_RATES.items():
        for kind, rate in rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"Profile {profile.value!r} rate for {kind.value} is not "
                    f"a probability: {rate}"
                )
    for bank, rate in MISPRONOUNCED_TERM_BANK_RATES.items():
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"mispronounced_term rate for bank {bank!r} is not a "
                f"probability: {rate}"
            )
        bank_total = sum(profile_kind_rates(ComplicationProfile.DEFAULT, bank).values())
        if not 0.0 < bank_total < 1.0:
            raise ValueError(
                f"DEFAULT profile rates for bank {bank!r} must sum inside "
                "(0, 1) — the remainder is the 'none' mass on a "
                f"fully-feasible task; got {bank_total}"
            )
    if any(
        rate != 1.0 for rate in PROFILE_KIND_RATES[ComplicationProfile.HARD].values()
    ):
        raise ValueError("HARD profile puts every kind at 1.0")


# Fixed in-code line templates, one per kind (per style for spelling_style;
# per omitted component for lazy_omission). Each line is self-contained: it
# includes its own carve-out from the sim's "never invent values" rule. ASCII
# only (the repo gates non-ASCII in task content).
COMPLICATION_LINES: dict[str, str] = {
    "self_correction": (
        "Scripted complication (self-correction): the first time you say the "
        'value the caller asked you to provide, misspeak it: say "{decoy}" '
        "first, then immediately correct yourself in the same turn with words "
        'like "no wait, sorry" and give the true value from your get_entity '
        'look-up. The misspoken value "{decoy}" is a scripted slip handed to '
        "you here, not something you invented: say it exactly once, never "
        "spell it, never confirm it, and never repeat it after the "
        "correction. From the correction onward you are accurate. The true "
        "value still comes only from your get_entity look-up; this scripted "
        "slip is the ONLY exception to your rule against speaking values you "
        "have not fetched."
    ),
    "spelling_style.grouped_numbers": (
        "Scripted complication (spelling style): whenever you read out or "
        "spell a value that contains digits, group adjacent digits into "
        'natural numbers instead of naming each digit alone: say "forty-four" '
        'for 44, not "four, four". This changes only HOW you voice the '
        "digits. The value itself is unchanged and still comes only from your "
        "get_entity look-up; you never invent, add, or drop a digit."
    ),
    "spelling_style.oh_for_zero": (
        "Scripted complication (spelling style): whenever you read out or "
        'spell a value that contains the digit 0, say the word "oh" for it '
        'instead of "zero". This changes only HOW you voice that digit. The '
        "value itself is unchanged and still comes only from your get_entity "
        "look-up; you never invent, add, or drop a digit."
    ),
    "spelling_style.uk_letters": (
        "Scripted complication (spelling style): whenever you spell out a "
        'value that contains the letter Z, say "zed" for it, the way a '
        "British speaker would. This changes only HOW you voice that letter. "
        "The value itself is unchanged and still comes only from your "
        "get_entity look-up; you never invent, add, or drop a letter."
    ),
    "spelling_style.doubled_letters": (
        "Scripted complication (spelling style): whenever you read out or "
        "spell a value that contains the same character twice in a row, voice "
        'the pair with the word "double": say "double five" for 55 and '
        '"double L" for LL. This changes only HOW you voice those '
        "characters. The value itself is unchanged and still comes only from "
        "your get_entity look-up; you never invent, add, or drop a character."
    ),
    "lazy_omission.am_pm": (
        "Scripted complication (lazy omission): the first time you give the "
        "requested time, leave off the AM or PM part and say only the clock "
        "time. Supply the AM/PM part only when the caller asks for it or "
        "asks you to confirm which one it is. You are omitting a part of the "
        "true value, never changing it: the full value, including its AM/PM "
        "part, still comes only from your get_entity look-up."
    ),
    "lazy_omission.year": (
        "Scripted complication (lazy omission): the first time you give the "
        "requested date, leave off the year and say only the month and day. "
        "Supply the year only when the caller asks for it or asks you to "
        "confirm it. You are omitting a part of the true value, never "
        "changing it: the full value, including its year, still comes only "
        "from your get_entity look-up."
    ),
    "lazy_omission.last_name": (
        "Scripted complication (lazy omission): the first time you give the "
        "requested name, say only the first name. Supply the last name only "
        "when the caller asks for the last name or the full name "
        "specifically. You are omitting a part of the true value, never "
        "changing it: the full name, including the last name, still comes "
        "only from your get_entity look-up."
    ),
    "wrong_slot": (
        "Scripted complication (wrong slot): the first time the caller asks "
        "you for the missing detail, misunderstand which detail they want "
        "and answer with a different fact from your own records instead: say "
        '"{off_slot_value}" (your {off_slot_phrase}). When the caller asks '
        "again, clarifies, or points out that this is not what they asked "
        "for, give the correct value from your get_entity look-up. The "
        'mistaken answer "{off_slot_value}" is a real value from your own '
        "record, handed to you here as a script, not something you invented: "
        "say it once and do not bring it up again. Every other value you "
        "speak still comes only from your get_entity look-up; this scripted "
        "misunderstanding is the ONLY exception to answering exactly what "
        "you are asked."
    ),
    "spell_correction": (
        "Scripted complication (spell correction): the FIRST time you spell "
        "the requested value out character by character, or read it out "
        'digit by digit, falter partway through: say "{faltered_partial}", '
        'break off, say "{marker}", and then start over, giving the complete '
        "correct sequence character by character from the beginning, from "
        'your get_entity look-up. The faltered partial "{faltered_partial}" '
        "is a scripted slip handed to you here, not something you invented: "
        "say it exactly once, in that first faltered attempt only, and never "
        "repeat it after the restart. This applies only when you actually "
        "spell or read the value out character by character; if that never "
        "happens on this call, this script never fires. After the restart "
        "you are accurate and complete. The true value still comes only from "
        "your get_entity look-up; this scripted slip is the ONLY exception "
        "to your rule against speaking values you have not fetched."
    ),
    # Packet-display text ONLY (SampledComplication.injected=False): this kind
    # is never appended to the user-sim prompt — the sim is unaware and speaks
    # normally. The bad pronunciation is applied at the voice-synthesis seam
    # (tau2.domains.intake.pronunciation_map + VoiceSettings.pronunciation_map).
    "mispronounced_term": (
        "Scripted complication (mispronounced term): the caller consistently "
        'mispronounces "{term}" as "{mispronounced}" (correct respelling '
        '"{respelling}"; distortion operator: {operator}). NOT injected; '
        "applied at the voice-synthesis seam. The user-simulator prompt is "
        "byte-identical to an uncomplicated run and the transcript keeps the "
        "gold spelling; only the synthesized audio carries the bad "
        "pronunciation."
    ),
}


# Which spell-out rendering styles a bank's values can carry. Digit-bearing
# banks take the number styles; letter-spelled banks take the letter styles;
# mixed grammars (codes, emails, addresses) take both. Feasibility is
# additionally gated per gold value at sample time (_style_applies) so a
# rendered line is never vacuous (no "say oh for zero" on a value without a
# zero).
BANK_SPELLING_STYLES: dict[str, tuple[SpellingStyle, ...]] = {
    "person_names": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "codes": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.UK_LETTERS,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "phones": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "dates": (SpellingStyle.GROUPED_NUMBERS, SpellingStyle.OH_FOR_ZERO),
    "times": (SpellingStyle.GROUPED_NUMBERS, SpellingStyle.OH_FOR_ZERO),
    "addresses": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.UK_LETTERS,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "properties": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "amounts": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "emails": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.UK_LETTERS,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "medications": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "insurance_plans": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "vehicles": (
        SpellingStyle.GROUPED_NUMBERS,
        SpellingStyle.OH_FOR_ZERO,
        SpellingStyle.UK_LETTERS,
        SpellingStyle.DOUBLED_LETTERS,
    ),
    "rate_plans": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "coined": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
    "shops": (SpellingStyle.UK_LETTERS, SpellingStyle.DOUBLED_LETTERS),
}

# The omitted component per bank for lazy_omission (design: banks whose
# canonical form has a droppable component the caller can lazily leave off).
LAZY_OMISSION_COMPONENTS: dict[str, str] = {
    "times": "am_pm",
    "dates": "year",
    "person_names": "last_name",
}

# The banks whose entries can carry a curated `mispronounced` respelling:
# medications (every entry, design doc sec. 5), person_names (hard tier
# only, sec. 5), and properties/vehicles (foreign tokens only, sec. 8b).
# mispronounced_term is additionally gated per gold value at sample time
# (_mispronounceable_tokens) and per run channel (voice-only).
MISPRONOUNCED_TERM_BANKS: frozenset[str] = frozenset(
    {"medications", "person_names", "properties", "vehicles"}
)

# Which complication kinds apply per bank. self_correction applies to every
# bank (so coverage holds), spell_correction to every bank (any value can be
# asked to spell; per-value feasibility additionally requires >= 4 spellable
# characters and a voice run), spelling_style to digit/letter-bearing banks
# (all 15 carry at least one style), lazy_omission only to
# times/dates/person_names, wrong_slot to all, mispronounced_term only to the
# pronunciation-column banks (MISPRONOUNCED_TERM_BANKS). Written out
# explicitly so a reviewer reads the catalog, not a derivation.
BANK_COMPLICATIONS: dict[str, list[ComplicationKind]] = {
    "person_names": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.LAZY_OMISSION,
        ComplicationKind.WRONG_SLOT,
        ComplicationKind.MISPRONOUNCED_TERM,
    ],
    "codes": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "phones": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "dates": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.LAZY_OMISSION,
        ComplicationKind.WRONG_SLOT,
    ],
    "times": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.LAZY_OMISSION,
        ComplicationKind.WRONG_SLOT,
    ],
    "addresses": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "properties": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
        ComplicationKind.MISPRONOUNCED_TERM,
    ],
    "amounts": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "emails": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "medications": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
        ComplicationKind.MISPRONOUNCED_TERM,
    ],
    "insurance_plans": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "vehicles": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
        ComplicationKind.MISPRONOUNCED_TERM,
    ],
    "rate_plans": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "coined": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
    "shops": [
        ComplicationKind.SELF_CORRECTION,
        ComplicationKind.SPELL_CORRECTION,
        ComplicationKind.SPELLING_STYLE,
        ComplicationKind.WRONG_SLOT,
    ],
}


def validate_complication_catalog() -> None:
    """Gate the catalog's internal consistency. Fails loud on any hole.

    Called at import time (below) and directly by the test suite, so a bank
    added to the banks package without a complication mapping, a kind without
    a line, or a non-ASCII template can never reach a run.
    """
    missing = set(BANK_NAMES) - set(BANK_COMPLICATIONS)
    extra = set(BANK_COMPLICATIONS) - set(BANK_NAMES)
    if missing or extra:
        raise ValueError(
            f"BANK_COMPLICATIONS out of sync with BANK_NAMES: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for bank, kinds in BANK_COMPLICATIONS.items():
        if not kinds:
            raise ValueError(f"Bank {bank!r} has no applicable complication kinds")
        if ComplicationKind.SELF_CORRECTION not in kinds:
            raise ValueError(
                f"Bank {bank!r} must carry self_correction (coverage guarantee)"
            )
        if ComplicationKind.SPELL_CORRECTION not in kinds:
            raise ValueError(
                f"Bank {bank!r} must carry spell_correction (any value can be "
                "asked to spell; per-value feasibility gates at sample time)"
            )
        if ComplicationKind.SPELLING_STYLE in kinds and not BANK_SPELLING_STYLES.get(
            bank
        ):
            raise ValueError(f"Bank {bank!r} has spelling_style but no styles")
        if (
            ComplicationKind.LAZY_OMISSION in kinds
            and bank not in LAZY_OMISSION_COMPONENTS
        ):
            raise ValueError(f"Bank {bank!r} has lazy_omission but no component")
    if set(LAZY_OMISSION_COMPONENTS) != {"times", "dates", "person_names"}:
        raise ValueError(
            "lazy_omission applies to exactly the times, dates, and person_names banks"
        )
    mispronounced_banks = {
        bank
        for bank, kinds in BANK_COMPLICATIONS.items()
        if ComplicationKind.MISPRONOUNCED_TERM in kinds
    }
    if mispronounced_banks != set(MISPRONOUNCED_TERM_BANKS):
        raise ValueError(
            "mispronounced_term applies to exactly the banks with curated "
            f"pronunciation columns {sorted(MISPRONOUNCED_TERM_BANKS)}, "
            f"got {sorted(mispronounced_banks)}"
        )
    expected_lines = {
        "self_correction",
        "wrong_slot",
        "mispronounced_term",
        "spell_correction",
    }
    expected_lines |= {f"spelling_style.{style.value}" for style in SpellingStyle}
    expected_lines |= {
        f"lazy_omission.{component}" for component in LAZY_OMISSION_COMPONENTS.values()
    }
    if set(COMPLICATION_LINES) != expected_lines:
        raise ValueError(
            f"COMPLICATION_LINES keys out of sync: "
            f"missing={sorted(expected_lines - set(COMPLICATION_LINES))}, "
            f"extra={sorted(set(COMPLICATION_LINES) - expected_lines)}"
        )
    for key, line in COMPLICATION_LINES.items():
        if not line.isascii():
            raise ValueError(f"Complication line {key!r} is not ASCII")


validate_complication_catalog()
validate_profile_rates()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

_TASK_ID_PATTERN = re.compile(
    rf"^intake_({'|'.join(sorted(BANK_NAMES, key=len, reverse=True))})"
    rf"_(easy|hard)_\d+$"
)

# Value-level feasibility per style: a style is only drawable when the gold
# value actually exercises it, so a rendered line is never a no-op.
_STYLE_FEASIBLE = {
    SpellingStyle.GROUPED_NUMBERS: re.compile(r"\d\d"),
    SpellingStyle.OH_FOR_ZERO: re.compile(r"0"),
    SpellingStyle.UK_LETTERS: re.compile(r"[zZ]"),
    SpellingStyle.DOUBLED_LETTERS: re.compile(r"([0-9A-Za-z])\1"),
}

# Value-level feasibility per omitted component: the component must be
# present in the gold value the sim will speak (a spoken relative date like
# "tomorrow" has no year to omit).
_OMISSION_FEASIBLE = {
    "am_pm": re.compile(r"\b[APap]\.?[Mm]\.?\b"),
    "year": re.compile(r"\b\d{4}\b"),
    "last_name": re.compile(r"\S\s+\S"),
}

# spell_correction machinery. The spelled sequence is the WRITTEN form's
# spellable characters (a caller asked to spell spells the written value —
# for dual-form payloads that is the annotated real value, not the spoken
# lead). Feasible when the sequence carries at least
# _SPELL_CORRECTION_MIN_UNITS characters — a 2-3 character sequence is not
# restart-worthy. The seeded single edit lands within the first
# _SPELL_CORRECTION_MAX_ERROR_INDEX units (people falter early in long
# sequences, and an early restart keeps the faltered partial short), and the
# correction marker comes from the closed SPELL_CORRECTION_MARKERS set —
# fixed in-code, never improvised (machine-not-scripts).
_SPELL_CORRECTION_MIN_UNITS = 4
_SPELL_CORRECTION_MAX_ERROR_INDEX = 6
SPELL_CORRECTION_MARKERS: tuple[str, ...] = (
    "sorry --",
    "wait, no --",
    "let me start again",
)
_SPELL_UNIT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SPELL_UNIT_DIGITS = "0123456789"


def _spell_units(gold_value_raw: str) -> list[str]:
    """The character-by-character spell sequence of one entity payload.

    Dual-form payloads spell their annotated WRITTEN value (what a caller
    asked for the exact written form spells); plain payloads spell the value
    itself. Units are the spellable (alphanumeric) characters in order,
    letters uppercased the way a spell-out voices them.
    """
    from tau2.domains.intake.spell_events import entity_match_texts

    written = entity_match_texts(gold_value_raw)[-1]
    return [char.upper() for char in written if char.isalnum()]


def _rng(run_seed: int, task_id: str) -> random.Random:
    """Deterministic per-(seed, task, catalog) RNG: a stable hash, never
    Python's salted ``hash()``."""
    digest = hashlib.sha256(
        f"{COMPLICATION_CATALOG_VERSION}:{run_seed}:{task_id}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@lru_cache(maxsize=1)
def _banks() -> Banks:
    return load_banks()


def _parse_task_id(task_id: str) -> tuple[str, Difficulty]:
    match = _TASK_ID_PATTERN.match(task_id)
    if match is None:
        raise ValueError(
            f"Task id {task_id!r} does not parse as intake_<bank>_<tier>_NN; "
            "complications only run against the atomic intake task sets"
        )
    return match.group(1), Difficulty(match.group(2))


def is_hard_tier(task_id: str) -> bool:
    """Whether an intake task id names a hard-tier task (the HARD profile's
    task selection — tau2.runner.complications.profile_task_filter)."""
    _bank, tier = _parse_task_id(task_id)
    return tier is Difficulty.HARD


def _init_action_args(task: Task, func_name: str) -> dict:
    actions = (task.initial_state and task.initial_state.initialization_actions) or []
    for action in actions:
        if action.func_name == func_name:
            return action.arguments
    raise ValueError(f"Task {task.id} has no {func_name!r} initialization action")


def _pool_values(bank: str, tier: Difficulty, callee_full_name: str) -> list[str]:
    """The bank x tier value pool as the CALLER would speak it: emails are
    instantiated with the record's callee, relative dates use their spoken
    form (that is what ``get_entity`` returns)."""
    entries = [entry for entry in getattr(_banks(), bank) if entry.difficulty is tier]
    values = []
    for entry in entries:
        if bank == "emails":
            values.append(instantiate_email(entry, callee_full_name).value)
        elif bank == "dates":
            values.append(entry.spoken or entry.value)
        else:
            values.append(entry.value)
    return values


def sample_complication(
    run_seed: int,
    task: Task,
    *,
    profile: ComplicationProfile,
    rate_override: Optional[float] = None,
    channel: Channel,
) -> Optional[SampledComplication]:
    """Deterministically sample this task's complication, or None.

    Derivation is pure: seeded from a stable hash of ``(run_seed, task.id,
    COMPLICATION_CATALOG_VERSION)``. ONE categorical draw over
    ``{none} ∪ kinds``: each of the bank's FEASIBLE kinds carries its
    profile rate (:data:`PROFILE_KIND_RATES`, uniformly scaled by
    ``rate_override`` when given — see :func:`effective_kind_rates`) and the
    infeasible kinds' mass goes to "none" — never renormalized among the
    feasible kinds, so a kind's conditional-on-feasible rate is constant
    across tasks and a constrained task triggers less often. When the
    feasible rates sum to >= 1 (the HARD profile), every call triggers and
    the draw is proportional among feasible kinds — uniform, since HARD's
    rates are all equal. Then the kind's parameter draws. The same seed
    therefore renders byte-identical lines in the review packet and the run.

    ``channel`` is the run's modality and is REQUIRED (no default: an
    indeterminable channel must fail loud upstream, never silently sample as
    text). It gates channel-bound kinds: ``mispronounced_term`` and
    ``spelling_style`` are voice-only, so a text run never draws them — and,
    per the mass-to-none rule, a text run triggers correspondingly less often.
    """
    if channel not in _CHANNELS:
        raise ValueError(
            f"Unknown run channel {channel!r}: complications sample per "
            f"channel and only know {_CHANNELS}"
        )
    bank, tier = _parse_task_id(task.id)
    rates = effective_kind_rates(profile, rate_override, bank)
    if sum(rates.values()) <= 0.0:
        return None
    rng = _rng(run_seed, task.id)
    roll = rng.random()

    record = _init_action_args(task, "seed_record")["record"]
    blanked = set(_init_action_args(task, "blank_fields")["field_names"])
    entities = _init_action_args(task, "set_entities")["entities"]
    if len(entities) != 1:
        raise ValueError(
            f"Task {task.id} pins {len(entities)} entities; phase-1 "
            "complications support single-entity tasks only"
        )
    ((gold_field, gold_value_raw),) = entities.items()
    # Dual-form payloads (relative dates) annotate the real written value;
    # every complication draws against the SPOKEN part — what the sim says —
    # so the annotation can never flip a feasibility gate (e.g. its 4-digit
    # year making year-omission look feasible on a spoken relative date).
    # spell_correction is the one exception: a caller asked to SPELL spells
    # the written form, so its sequence derives from the raw payload
    # (_spell_units).
    gold_value = spoken_form(gold_value_raw)
    fold_kinds = {field["name"]: FoldKind(field["fold"]) for field in record["fields"]}
    if gold_field not in fold_kinds:
        raise ValueError(
            f"Task {task.id}: entity field {gold_field!r} is not on the record"
        )

    kinds = [
        kind
        for kind in BANK_COMPLICATIONS[bank]
        if _kind_feasible(kind, bank, tier, gold_value, channel, gold_value_raw)
    ]
    if not kinds:
        raise ValueError(f"Task {task.id} has no feasible complication kinds")
    feasible_total = sum(rates[kind] for kind in kinds)
    if feasible_total < 1.0:
        # Mass-to-none: the roll lands in a kind's interval or in the
        # remaining "none" mass.
        if roll >= feasible_total:
            return None
        point = roll
    else:
        # Saturated (HARD): every call triggers; the roll is stretched over
        # the feasible kinds' rates — proportional, i.e. uniform when the
        # rates are all equal.
        point = roll * feasible_total
    kind = kinds[-1]  # guard against float edge; the loop always breaks
    cumulative = 0.0
    for candidate in kinds:
        cumulative += rates[candidate]
        if point < cumulative:
            kind = candidate
            break

    if kind is ComplicationKind.SELF_CORRECTION:
        return _sample_self_correction(
            rng, bank, tier, record, gold_value, fold_kinds[gold_field]
        )
    if kind is ComplicationKind.SPELLING_STYLE:
        return _sample_spelling_style(rng, bank, gold_value)
    if kind is ComplicationKind.LAZY_OMISSION:
        return _sample_lazy_omission(bank)
    if kind is ComplicationKind.WRONG_SLOT:
        return _sample_wrong_slot(rng, task, record, blanked, gold_field)
    if kind is ComplicationKind.MISPRONOUNCED_TERM:
        return _sample_mispronounced_term(rng, bank, tier, gold_value)
    if kind is ComplicationKind.SPELL_CORRECTION:
        return _sample_spell_correction(rng, gold_value_raw)
    raise ValueError(f"Unhandled complication kind: {kind}")  # pragma: no cover


def _kind_feasible(
    kind: ComplicationKind,
    bank: str,
    tier: Difficulty,
    gold_value: str,
    channel: Channel,
    gold_value_raw: str,
) -> bool:
    """Whether a kind can render a non-vacuous draw for this gold value.

    self_correction and wrong_slot are always feasible (every bank pool holds
    fold-distinct neighbors, and every record carries a filled context field
    next to the missing one); spelling_style needs a VOICE run (every style
    scripts how a value is voiced, with no typed analogue) plus at least one
    style the gold value exercises; lazy_omission needs the omittable
    component present in the gold value; mispronounced_term needs a VOICE run
    (the effect lives in synthesized audio — a text run has no audio to
    distort) and at least one gold-value token with a curated
    ``mispronounced`` respelling in the bank; spell_correction needs a VOICE
    run (the construct is a voiced restart mid-spell-out) and a written form
    with at least _SPELL_CORRECTION_MIN_UNITS spellable characters — a
    shorter sequence is not restart-worthy.
    """
    if kind is ComplicationKind.SPELLING_STYLE:
        if channel != "voice":
            return False
        return bool(_feasible_styles(bank, gold_value))
    if kind is ComplicationKind.LAZY_OMISSION:
        component = LAZY_OMISSION_COMPONENTS[bank]
        return bool(_OMISSION_FEASIBLE[component].search(gold_value))
    if kind is ComplicationKind.MISPRONOUNCED_TERM:
        if channel != "voice":
            return False
        return bool(_mispronounceable_tokens(bank, tier, gold_value))
    if kind is ComplicationKind.SPELL_CORRECTION:
        if channel != "voice":
            return False
        return len(_spell_units(gold_value_raw)) >= _SPELL_CORRECTION_MIN_UNITS
    return True


def _feasible_styles(bank: str, gold_value: str) -> list[SpellingStyle]:
    return [
        style
        for style in BANK_SPELLING_STYLES.get(bank, ())
        if _STYLE_FEASIBLE[style].search(gold_value)
    ]


def _sample_self_correction(
    rng: random.Random,
    bank: str,
    tier: Difficulty,
    record: dict,
    gold_value: str,
    fold_kind: FoldKind,
) -> SampledComplication:
    pool = _pool_values(bank, tier, record["callee_full_name"])
    rng.shuffle(pool)
    gold_folded = fold_value(fold_kind, gold_value)
    for candidate in pool:
        if fold_value(fold_kind, candidate) != gold_folded:
            decoy = candidate
            break
    else:
        raise ValueError(
            f"Bank {bank} ({tier.value}) pool cannot produce a fold-distinct "
            f"decoy for {gold_value!r}"
        )
    return SampledComplication(
        kind=ComplicationKind.SELF_CORRECTION.value,
        line=COMPLICATION_LINES["self_correction"].format(decoy=decoy),
        params={"decoy_value": decoy},
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )


def _sample_spelling_style(
    rng: random.Random, bank: str, gold_value: str
) -> SampledComplication:
    style = rng.choice(_feasible_styles(bank, gold_value))
    return SampledComplication(
        kind=ComplicationKind.SPELLING_STYLE.value,
        line=COMPLICATION_LINES[f"spelling_style.{style.value}"],
        params={"style": style.value},
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )


def _sample_lazy_omission(bank: str) -> SampledComplication:
    component = LAZY_OMISSION_COMPONENTS[bank]
    return SampledComplication(
        kind=ComplicationKind.LAZY_OMISSION.value,
        line=COMPLICATION_LINES[f"lazy_omission.{component}"],
        params={"omitted_component": component},
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )


def _sample_wrong_slot(
    rng: random.Random,
    task: Task,
    record: dict,
    blanked: set[str],
    gold_field: str,
) -> SampledComplication:
    candidates = [
        field
        for field in record["fields"]
        if field["name"] != gold_field
        and field["name"] not in blanked
        and field.get("value")
    ]
    if not candidates:
        raise ValueError(f"Task {task.id} has no filled context field for wrong_slot")
    chosen = rng.choice(sorted(candidates, key=lambda field: field["name"]))
    off_slot_phrase = chosen["name"].replace("_", " ")
    return SampledComplication(
        kind=ComplicationKind.WRONG_SLOT.value,
        line=COMPLICATION_LINES["wrong_slot"].format(
            off_slot_value=chosen["value"], off_slot_phrase=off_slot_phrase
        ),
        params={
            "off_slot_field": chosen["name"],
            "off_slot_value": chosen["value"],
        },
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )


def _mispronounceable_tokens(
    bank: str, tier: Difficulty, gold_value: str
) -> list[TokenPronunciation]:
    """The gold value's bank tokens that carry a curated ``mispronounced``
    respelling, in bank order (empty for banks without pronunciation columns).

    The bank entry lookup fails loud: for the pronunciation-column banks the
    gold entity is by construction a bank value, so a miss is a task/bank
    drift bug, never a reason to silently skip the kind.
    """
    if bank not in MISPRONOUNCED_TERM_BANKS:
        return []
    for entry in getattr(_banks(), bank):
        if entry.difficulty is tier and entry.value == gold_value:
            return [p for p in entry.pronunciations if p.mispronounced is not None]
    raise ValueError(
        f"Gold value {gold_value!r} not found in bank {bank!r} ({tier.value}); "
        "mispronounced_term reads pronunciations straight off the bank entry"
    )


def _sample_mispronounced_term(
    rng: random.Random, bank: str, tier: Difficulty, gold_value: str
) -> SampledComplication:
    """Draw the mispronounced token (seeded choice among the gold value's
    mispronounceable tokens) straight off the bank's pronunciation columns.

    Every value in the record (term, respelling, mispronounced, operator) is
    the bank's curated data verbatim — never invented here. ``injected=False``:
    the line is packet-display text; the sim's prompt stays byte-identical to
    an uncomplicated run, and the effect lives at the voice-synthesis seam
    (the drawn term's map entry is overridden with the bad respelling — see
    :mod:`tau2.domains.intake.pronunciation_map`).
    """
    tokens = _mispronounceable_tokens(bank, tier, gold_value)
    chosen = rng.choice(tokens)
    return SampledComplication(
        kind=ComplicationKind.MISPRONOUNCED_TERM.value,
        line=COMPLICATION_LINES["mispronounced_term"].format(
            term=chosen.token,
            mispronounced=chosen.mispronounced,
            respelling=chosen.respelling,
            operator=chosen.operator.value,
        ),
        params={
            "term": chosen.token,
            "respelling": chosen.respelling,
            "operator": chosen.operator.value,
        },
        catalog_version=COMPLICATION_CATALOG_VERSION,
        injected=False,
        mispronunciation=chosen.mispronounced,
    )


def _sample_spell_correction(
    rng: random.Random, gold_value_raw: str
) -> SampledComplication:
    """Draw the seeded faltered partial for a mid-spell-out restart.

    Deterministic single edit on a prefix of the gold spell sequence
    (:func:`_spell_units`): a seeded error position within the first
    _SPELL_CORRECTION_MAX_ERROR_INDEX units, a seeded operator —
    ``substitute`` (a different same-class character) or ``drop`` (the unit
    is skipped) — and 0-2 seeded extra CORRECT units spoken past the error
    before breaking off (a drop always speaks at least one following unit so
    the slip is audible). The faltered partial is handed to the sim verbatim
    in the fixed line template together with a marker from the closed
    SPELL_CORRECTION_MARKERS set; the complete correct sequence after the
    restart comes only from the sim's get_entity look-up — never from this
    line.
    """
    units = _spell_units(gold_value_raw)
    if len(units) < _SPELL_CORRECTION_MIN_UNITS:  # pragma: no cover - gated
        raise ValueError(
            f"spell_correction drawn on a {len(units)}-unit sequence; "
            "feasibility gates at _SPELL_CORRECTION_MIN_UNITS"
        )
    operator = rng.choice(("substitute", "drop"))
    error_index = rng.randrange(
        0, min(len(units) - 1, _SPELL_CORRECTION_MAX_ERROR_INDEX)
    )
    extra_units = rng.randrange(0, 3)
    if operator == "substitute":
        original = units[error_index]
        alphabet = _SPELL_UNIT_DIGITS if original.isdigit() else _SPELL_UNIT_LETTERS
        wrong = rng.choice([char for char in alphabet if char != original])
        partial = (
            units[:error_index]
            + [wrong]
            + units[error_index + 1 : error_index + 1 + extra_units]
        )
    else:
        partial = (
            units[:error_index] + units[error_index + 1 : error_index + 2 + extra_units]
        )
    faltered_partial = ", ".join(partial)
    marker = rng.choice(SPELL_CORRECTION_MARKERS)
    return SampledComplication(
        kind=ComplicationKind.SPELL_CORRECTION.value,
        line=COMPLICATION_LINES["spell_correction"].format(
            faltered_partial=faltered_partial, marker=marker
        ),
        params={
            "operator": operator,
            "error_index": error_index,
            "extra_units": extra_units,
            "faltered_partial": faltered_partial,
            "marker": marker,
            "sequence_length": len(units),
        },
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )


# ---------------------------------------------------------------------------
# Pre-verification review packet (tau2 intake-tasks complications-packet)
# ---------------------------------------------------------------------------


class ComplicationsPacket(BaseModelNoExtra):
    """One rendered complications review packet plus its summary counts."""

    seed: Annotated[int, Field(description="Run seed the draw was made with.")]
    profile: Annotated[
        ComplicationProfile,
        Field(
            description="Complication profile of the draw (per-kind rates; "
            "HARD additionally restricts the packet to hard-tier tasks, "
            "matching the run's task selection)."
        ),
    ]
    rate_override: Annotated[
        Optional[float],
        Field(
            description="Explicit --complication-rate override the draw was "
            "made with (uniform trigger scaling of the profile's rates), or "
            "None for the profile's rates verbatim."
        ),
    ]
    per_kind_rates: Annotated[
        dict[str, float],
        Field(
            description="The profile's BASE per-kind conditional rates "
            "(mispronounced_term expanded per bank on DEFAULT, e.g. "
            "'mispronounced_term (medications)'). An override rescales each "
            "task's bank vector uniformly at draw time and is recorded in "
            "rate_override, not folded in here."
        ),
    ]
    channel: Annotated[
        Channel,
        Field(
            description="Run channel the draw was made for (voice-only kinds "
            "like mispronounced_term only appear in a voice draw)."
        ),
    ]
    total_tasks: Annotated[
        int, Field(description="Tasks the profile selects from the frozen set.")
    ]
    tier_mix: Annotated[
        dict[str, int],
        Field(description="Selected-task count per tier (easy/hard)."),
    ]
    triggered: Annotated[int, Field(description="Tasks that drew a complication.")]
    per_kind: Annotated[
        dict[str, int],
        Field(description="Triggered count per complication kind."),
    ]
    markdown: Annotated[
        str,
        Field(
            description="The full packet body (deterministic per "
            "seed+profile+override+channel)."
        ),
    ]


def _load_frozen_tasks() -> tuple[list[Task], str]:
    """The frozen canonical task set plus its file sha (provenance)."""
    from tau2.domains.intake.tasks.generator import TASKS_FILENAME
    from tau2.domains.intake.utils import INTAKE_DATA_DIR

    path = INTAKE_DATA_DIR / TASKS_FILENAME
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    tasks = [Task.model_validate(raw) for raw in payload["tasks"]]
    return tasks, hashlib.sha256(raw_bytes).hexdigest()


def _packet_rate_rows(profile: ComplicationProfile) -> dict[str, float]:
    """The profile's base rates as labeled packet rows, catalog-ordered —
    mispronounced_term expanded per bank on DEFAULT ('mispronounced_term
    (medications)'), a single 1.0 row on HARD."""
    rows: dict[str, float] = {}
    for kind in ComplicationKind:
        if (
            kind is ComplicationKind.MISPRONOUNCED_TERM
            and profile is ComplicationProfile.DEFAULT
        ):
            for bank_name, bank_rate in sorted(MISPRONOUNCED_TERM_BANK_RATES.items()):
                rows[f"{kind.value} ({bank_name})"] = bank_rate
        else:
            rows[kind.value] = PROFILE_KIND_RATES[profile][kind]
    return rows


def build_complications_packet(
    seed: int,
    profile: ComplicationProfile,
    channel: Channel,
    rate_override: Optional[float] = None,
) -> ComplicationsPacket:
    """Render the seeded complication draw over the profile's task selection.

    Runs the SAME :func:`sample_complication` a run at this seed, profile,
    override, and channel runs, over the SAME tasks the profile selects
    (DEFAULT: the full frozen set — 100 easy / 100 hard by construction;
    HARD: the hard tier only), so the packet is a pre-verification of the
    exact draws: summary counts first, then one section per triggered task
    with the full line — as it will be appended for injected kinds, as
    packet-display text for prompt-silent ones. Deterministic: same inputs
    render a byte-identical body.
    """
    all_tasks, tasks_sha = _load_frozen_tasks()
    if profile is ComplicationProfile.HARD:
        tasks = [task for task in all_tasks if is_hard_tier(task.id)]
    else:
        tasks = all_tasks
    tier_mix = {tier.value: 0 for tier in Difficulty}
    for task in tasks:
        _bank, tier = _parse_task_id(task.id)
        tier_mix[tier.value] += 1
    if rate_override is not None and not 0.0 <= rate_override <= 1.0:
        raise ValueError(
            f"--complication-rate must be within [0, 1], got {rate_override}"
        )
    draws: list[tuple[Task, SampledComplication]] = []
    per_kind = {kind.value: 0 for kind in ComplicationKind}
    for task in tasks:
        sampled = sample_complication(
            seed, task, profile=profile, rate_override=rate_override, channel=channel
        )
        if sampled is None:
            continue
        draws.append((task, sampled))
        per_kind[sampled.kind] += 1

    lines: list[str] = []
    lines.append("# Intake scripted-complications review packet")
    lines.append("")
    lines.append(
        f"Provenance: catalog {COMPLICATION_CATALOG_VERSION}, seed {seed}, "
        f"profile {profile.value}, rate override "
        f"{'none' if rate_override is None else rate_override}, "
        f"channel {channel}, tasks {len(tasks)} "
        f"(easy {tier_mix['easy']} / hard {tier_mix['hard']}) from the frozen "
        f"task set tasks.json (sha256 {tasks_sha}). "
        "Regenerable byte-identically via `tau2 intake-tasks "
        "complications-packet`; do not hand-edit."
    )
    lines.append("")
    lines.append(
        "Per-kind base rates (per feasible opportunity; infeasible kinds' "
        "mass goes to none; a rate override rescales each task's bank "
        "vector uniformly — mispronounced_term's DEFAULT rate is per bank):"
    )
    lines.append("")
    for label, rate in _packet_rate_rows(profile).items():
        lines.append(f"- {label}: {rate:.4f}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- triggered: {len(draws)}/{len(tasks)} tasks")
    for kind, count in per_kind.items():
        lines.append(f"- {kind}: {count}")
    for task, sampled in draws:
        bank, tier = _parse_task_id(task.id)
        lines.append("")
        lines.append(f"## {task.id}")
        lines.append("")
        lines.append(f"- bank/tier: {bank} / {tier.value}")
        lines.append(f"- kind: {sampled.kind}")
        lines.append(f"- params: {json.dumps(sampled.params, sort_keys=True)}")
        lines.append("")
        if sampled.injected:
            lines.append("Line as appended to the user-sim prompt:")
        else:
            lines.append(
                "Packet-display line (NOT injected; applied at the "
                "voice-synthesis seam):"
            )
        lines.append("")
        lines.append(f"> {sampled.line}")
    lines.append("")
    return ComplicationsPacket(
        seed=seed,
        profile=profile,
        rate_override=rate_override,
        per_kind_rates=_packet_rate_rows(profile),
        channel=channel,
        total_tasks=len(tasks),
        tier_mix=tier_mix,
        triggered=len(draws),
        per_kind=per_kind,
        markdown="\n".join(lines),
    )
