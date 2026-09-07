# Intake complication profiles: literature-anchored default + hard

Status: IMPLEMENTED (owner directive 2026-08-26). Catalog v2.0.0.
Companion to [intake-mispronunciation.md](intake-mispronunciation.md) (which
specs the `mispronounced_term` kind and the pronunciation columns it reads)
and [intake-domain.md](intake-domain.md) (the v4 task frame). Code:
`src/tau2/domains/intake/complications.py` (catalog + sampler + packet),
`src/tau2/runner/complications.py` (dispatcher + profile task selection).

## 1. What a profile is

A `ComplicationProfile` (`tau2.data_model.simulation`, closed str-enum:
`default` | `hard`) names two things:

1. **Per-kind trigger rates** (`PROFILE_KIND_RATES` in the intake catalog),
   replacing catalog v1.x's flat 25%-then-uniform draw.
2. **Task selection** the profile imposes, applied at run-config /
   task-selection level (`resolve_tasks` via `profile_task_filter`) — the
   frozen `tasks.json` is never edited and there is no re-freeze.

The profile rides on `RunConfig.complication_profile`
(`--complication-profile`, default `default`), is stamped into
`Info.complication_profile` next to `complication_rate`, and is restored by
`tau2 run refill` / auto-resume exactly like the rate (PR #928 pattern).

## 2. The draw (default profile)

ONE categorical draw per task over `{none} ∪ kinds`, seeded from
`(run seed, task id, catalog version)` as before:

- every kind that is FEASIBLE for the task (bank applicability × gold-value
  gates × channel) occupies an interval of width equal to its rate;
- the mass of INFEASIBLE kinds goes to **none** — never renormalized among
  the feasible kinds. A kind's conditional-on-feasible rate is therefore a
  task-independent constant, and a constrained task (e.g. a text run, where
  `mispronounced_term` is off-channel) simply triggers less often.
  Renormalizing instead would silently inflate every rate exactly on the
  constrained tasks.

Consequence worth knowing: the v1.x invariant "the trigger gate is
channel-independent" is intentionally gone. At the same seed, the set of
text-triggered tasks is a subset of the voice-triggered ones.

On the frozen set (voice, seed-dependent): most banks carry
`self_correction + spelling_style + wrong_slot` ≈ 0.23 total trigger;
times/dates/person_names add `lazy_omission` where the component is present;
medications and hard-tier person names add `mispronounced_term` mass (up to
≈ 0.63 on a fully-feasible task). Empirically ≈ 20% of the 300 tasks trigger
per voice run.

## 3. The hard profile

Every kind at rate 1.0: the feasible rates sum to ≥ 1, so **every call draws
a complication** and the draw is proportional among feasible kinds — with
equal rates, exactly the old uniform-among-feasible choice (measurement
power: per-kind cells stay comparable).

Additionally, `hard` restricts the run to **hard-tier tasks only**
(`intake_<bank>_hard_NN`), enforced in `resolve_tasks` through the
domain-registered predicate (`DOMAIN_HARD_TIER_PREDICATES`). A `hard`
profile on a domain without a tier predicate fails loud.

## 4. Entity-tier mix

The frozen `tasks.json` (300 tasks, manifest-pinned) is **already the
designed 50/50 mix**: 150 easy / 150 hard, 10 × 15 banks per tier (checked
2026-08-26; guarded by `test_frozen_set_is_a_50_50_tier_mix`). So:

- `default` = the full frozen set; the profile only asserts/records the mix
  (no subsampling, no seeded frame needed);
- `hard` = the 150-task hard-tier subset, selected at task-selection level.

The review packet header records the selected tier mix per profile.

## 5. `--complication-rate` under profiles

Kept, as an explicit override with one documented meaning: a **uniform
trigger scaling**. The override R multiplies every per-kind rate by
`R / sum(all kinds' profile rates)`, so a task on which every kind is
feasible triggers with probability exactly R while the profile keeps
contributing the kind MIX (and the mass-to-none rule is untouched — a
constrained task still triggers proportionally less). `R = 0` runs clean;
R outside [0, 1] fails loud at the CLI and in `effective_kind_rates`.
Silently combining a flat rate with per-kind rates in any other way was the
rejected alternative.

Provenance: `Info.complication_profile` is always recorded; its presence
marks a post-profiles result, under which `Info.complication_rate` is the
override verbatim (None = the profile's rates ran as-is, 0.0 = explicitly
clean). Pre-profiles results keep their old reading (flat resolved rate);
a refill of one carries that flat rate over as the override and reports the
profile unrecorded — pre-bump draws are not reproducible anyway (sec. 6).

## 6. Catalog v2.0.0

`sample_complication` changed shape (per-kind categorical instead of
rate-then-uniform), so every draw re-keys: `COMPLICATION_CATALOG_VERSION` is
bumped to 2.0.0 (the version keys the per-task RNG). No seed packet is
committed in-repo (packets are rendered per run seed and reviewed on Drive);
re-render before the next armed run:

```
tau2 intake-tasks complications-packet --seed <run seed> --profile default -o <out.md>
tau2 intake-tasks complications-packet --seed <run seed> --profile hard -o <out.md>
```

The packet verb gained `--profile`; its provenance header records profile,
rate override, per-kind effective rates, channel, tier mix, and the frozen
tasks.json sha. A `hard` packet renders hard-tier tasks only — exactly the
tasks a hard run runs.

## 7. Rate grounding (default profile)

The per-kind rates are named in-code constants in
`src/tau2/domains/intake/complications.py`, each carrying its citation
comment. **The constructs in the literature rarely match ours exactly**;
every rate below says how far its anchor is, and rates with no quantitative
anchor at all are labeled JUDGMENT. Where the literature supports a range,
the constant sits at the conservative (low) end. BibTeX for the citations
lives in `papers/tau-intake/v1/references.bib`.

### self_correction = 0.05

Spontaneous-speech self-repair. Anchors:

- Levelt, W. J. M. (1983). Monitoring and self-repair in speech.
  *Cognition*, 14(1), 41–104. — The repair taxonomy (error repairs vs
  appropriateness repairs) our kind's misspeak-then-correct pattern
  ("no wait, sorry") is modeled on.
- Shriberg, E. (1994). *Preliminaries to a Theory of Speech Disfluencies.*
  PhD dissertation, UC Berkeley. — Disfluency rates across three corpora;
  disfluency probability grows with utterance length; substitutions/repairs
  are a minority of total disfluency mass.
- Fox Tree, J. E. (1995). The effects of false starts and repetitions on the
  processing of subsequent words in spontaneous speech. *Journal of Memory
  and Language*, 34(6), 709–738.
- Bortfeld, H., Leon, S. D., Bloom, J. E., Schober, M. F., & Brennan, S. E.
  (2001). Disfluency rates in conversation: Effects of age, relationship,
  topic, role, and gender. *Language and Speech*, 44(2), 123–147. — ~5.97
  disfluencies per 100 words in task-oriented conversation, of which
  restarts/repairs ~1.7–2.2 per 100 words.

CONSTRUCT GAP: the corpus rates count all repairs of any word, including
abandoned starts and appropriateness repairs; our kind scripts one
substantive misspeak-then-correct of the single information-bearing value
in the call. Substantive content repairs run single-digit percent per
utterance; 0.05 is the conservative end.

### spelling_style = 0.15 (JUDGMENT — prevalence, not error)

Marked spell-out rendering conventions. This is DIALECT PREVALENCE, not an
error rate: "oh" for zero dominates casual US phone-number speech;
"double"/"treble" collapsing is British/Commonwealth usage; digit grouping
is routine. Anchors are qualitative only (no corpus prevalence number found
for any single convention, searched 2026-08-26):

- Murphy, L. (2007). "double". *Separated by a Common Language* (blog),
  documenting the British "double"/"treble" convention and its absence in
  American speech.
- Usage guides on English number-reading conventions (e.g. Woodward
  English, "Telephone numbers in English").

JUDGMENT: raw prevalence of using SOME marked convention plausibly exceeds
0.5, but our construct scripts one specific style applied consistently for
the whole call, so the rate sits far below raw prevalence, at 0.15.

### lazy_omission = 0.10 (JUDGMENT)

Under-specification of inferable value components (year, AM/PM, last name).
NO direct corpus number exists that we could find (searched
appointment-scheduling dialog corpora and the TimeML/TempEval literature,
2026-08-26). Qualitative anchors:

- Grice, H. P. (1975). Logic and conversation. In *Syntax and Semantics 3:
  Speech Acts*, 41–58. — The quantity maxim: omission of inferable
  components is unmarked, cooperative speech, which is why callers do it.
- Pustejovsky, J., Castaño, J., Ingria, R., Saurí, R., Gaizauskas, R.,
  Setzer, A., & Katz, G. (2003). TimeML: Robust specification of event and
  temporal expressions in text. *New Directions in Question Answering*. —
  TimeML is explicitly designed around contextually underspecified temporal
  expressions.
- UzZaman, N., Llorens, H., Derczynski, L., Allen, J., Verhagen, M., &
  Pustejovsky, J. (2013). SemEval-2013 Task 1: TempEval-3. — Error analyses
  (e.g. HeidelTime's) identify underspecified expressions as the main
  normalization error source, i.e. the phenomenon is pervasive in real text.

JUDGMENT rate 0.10 with the qualitative citations above.

### wrong_slot = 0.03

Misunderstanding which detail was asked for. Anchors:

- Dingemanse, M., Roberts, S. G., Baranova, J., Blythe, J., Drew, P.,
  Floyd, S., et al. (2015). Universal principles in the repair of
  communication problems. *PLOS ONE*, 10(9), e0136100. — Other-initiated
  repair occurs about once per 1.4 minutes of conversation across 12
  languages: a few percent of turns carry a misunderstanding the recipient
  must flag.
- Purver, M., Hough, J., & Howes, C. (2018). Computational models of
  miscommunication phenomena. *Topics in Cognitive Science*, 10(2),
  425–451.

CONSTRUCT GAP: OIR covers all trouble sources (mishearing, non-hearing,
non-understanding); our kind is the specific answered-the-wrong-question
subset, so 0.03 sits below the overall OIR rate.

### mispronounced_term: per-bank — medications 0.50, person_names 0.15 (JUDGMENT, adjacent construct)

Conditional on the gold value carrying a curated `mispronounced` column
(medications; hard-tier person names) and on a voice run. Split PER BANK
(owner directive 2026-08-26, catalog v2.1.0,
`MISPRONOUNCED_TERM_BANK_RATES`): the grounding literature below is
medication-specific, and a caller struggles with a drug name far more than
with a personal name they are reporting. The adjacent literature is
medication NAMING failure, not spoken mispronunciation — stated plainly:
**no study measures how often patients mispronounce a drug name they do
produce**, so these rates are judgment anchored to the nearest measured
construct:

- Persell, S. D., Osborn, C. Y., Richard, R., Skripkauskas, S., & Wolf,
  M. S. (2007). Limited health literacy is a barrier to medication
  reconciliation in ambulatory care. *Journal of General Internal
  Medicine*, 22(11), 1523–1526. (Northwestern Feinberg.) — 40.5% vs 68.3%
  of hypertensive patients with inadequate vs adequate health literacy
  could name ANY of their antihypertensives (~60% of the low-health-
  literacy group could not); overall ~40% could not accurately recall
  their regimen.
- Naming-recall failure spans roughly **30–78% across studies** of
  patients' ability to name their own medications.
- Drug-name pronunciation difficulty itself is documented qualitatively:
  Patterson, C. (2018). Unpronounceable drug names. *Australian
  Prescriber*, 41(6), 176–177 (PMC6299177); Cheesman, M., Do, D., Alcorn,
  S., Grant, G., & Cardell, E. (2022). Impact of the "DrugSpeak" programme
  on drug name pronunciation skills and perceptions in a pharmacy student
  cohort. *Pharmacy Education*, 22(1), 348–359 — even pharmacy students
  need explicit phonetics training to pronounce drug names.

Chosen rates: **medications 0.50** = the middle of the 30–78%
naming-recall range; **person_names 0.15** = pure judgment (no quantitative
literature on callers mispronouncing personal names they report; rarer than
drug names since a caller often knows the bearer, but real for hard-tier
foreign or unusual spellings). Both labeled JUDGMENT — for medications
because naming-recall (cannot produce the name) is an ADJACENT construct to
ours (produces the name, mispronounced).

## 8. Tests

`tests/test_domains/test_intake/test_complications.py`: profile enum closed;
default rates equal the documented constants (sum < 1); hard = 1.0
everywhere + always triggers + hard-tier-only task selection (and fails loud
on domains without tiers); the mass-to-none property (empirical: fewer
feasible kinds → lower total trigger probability, per-kind conditional rates
unchanged; deterministic: text triggers ⊆ voice triggers per seed); override
scaling and validation; catalog version bump asserted; packet provenance
(profile, rates, tier mix, hard-only rendering); Info/refill round-trips
including the no-override-is-a-real-record case.
`tests/test_runner/test_refill.py`: legacy results report
`complication_profile` unrecorded.
