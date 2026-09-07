# Intake pronunciation program: grounded names, correct TTS, `mispronounced_term`

Status: SPEC v2 — approved direction (owner, 2026-08-25); details under review.
Catalog target: v1.2.0. Supersedes v1 of this doc (pre-synthesis-swap section
carried forward unchanged).

> **Catalog v2.0.0 (2026-08-26):** WHICH calls trigger a complication — and
> at what per-kind rates — is now owned by the named complication profiles;
> see [intake-complication-profiles.md](intake-complication-profiles.md)
> (literature-anchored `default` rates incl. `mispronounced_term` at 0.30,
> the all-1.0 `hard` profile + hard-tier task selection, and the
> `--complication-rate` override semantics). This doc still owns the
> `mispronounced_term` kind itself and the pronunciation columns it reads.

## 0. Owner decisions this spec implements

1. Re-source the person_names **hard tier backwards from pronunciation
   lexicons**: every hard name must have a dataset-attested pronunciation.
2. Store **correct pronunciations for all names and medications**, and apply
   them at synthesis so TTS says them right by default.
3. Every medication (and every hard name) also carries a **mispronounced
   variant** (e.g. "azi-thro-MY-cin" -> "azi-THOR-my-cin") for the
   `mispronounced_term` complication.
4. **Pairing stays the pure seeded shuffle** (owner, 2026-08-25, reversing an
   earlier ask): no dataset of real first+last pairings exists that is not a
   registry of real identifiable people, and independently sampled pairs
   guarantee the benchmark contains no real person's actual name. No
   coherence gate.

## 1. Dataset findings (2026-08-25)

CMUdict master (135k entries) + WikiPron `eng_latn_us_broad` (107k):

| pool | coverage |
|---|---|
| current hard names | 0/40 full names, 25/100 tokens (surnames absent) |
| medications easy/hard | 38/40 and 32/40 drug tokens (WikiPron-carried) |
| census tail band (count 100-129), covered + non-word + len>=5 | **857 surnames** (610 CMUdict-only, 146 WikiPron-only, 101 both) |
| SSA rare given names (total<1000), covered + non-word + len>=5 | **2,408 names** |

The hard tier needs ~40 first names + ~60 surname tokens: the covered pool is
10-20x oversubscribed **within the existing rarity band** — no band change.
CMUdict (built for ASR) carries the surnames; WikiPron carries the drugs.

## 2. Pipeline changes (`tau2 intake-names`, extractor v2 / builder v3)

### 2.1 New pinned sources

Add both lexicons to the extract stage as raw sources with URL + retrieval
date + sha256 in `provenance.json`, exactly like the SSA/Census zips:

- CMUdict `cmudict.dict` (github.com/cmusphinx/cmudict, pin a commit).
- WikiPron `eng_latn_us_broad.tsv` (github.com/CUNY-CL/wikipron, pin a commit).

### 2.2 Hard-tier coverage filter

Hard given-name and tail-surname pools are restricted to lexicon-covered
names, with junk filters (fixed, in-code): not a common English word
(`/usr/share/dict/words` is not pinnable — vendor a pinned wordlist source),
singular-stripped form also not a word (kills "Diets"/"Fowls"), length >= 5,
no place-name/brand leakage (small fixed denylist, reviewed). Easy tiers
unchanged (top-band names are lexicon-covered ~100% anyway).

### 2.3 Pronunciation columns

Every person_names entry and every medications entry gains:

```yaml
pronunciation:
  source: cmudict | wikipron | manual   # manual = USAN/MedlinePlus, cited
  phonemes: "Z OW1 G B IY"              # ARPABET (cmudict) or IPA (wikipron)
  respelling: "ZOHG-bee"                # derived, ASCII, reference notation
  mispronounced: "zogbye"               # bad variant, natural orthography —
                                        # TTS-ready verbatim (sec. 4)
```

- `respelling` is derived from `phonemes` by a **fixed in-code converter**
  (ARPABET->respelling and IPA->respelling phoneme tables; deterministic,
  unit-tested). Hand-curated only for `source: manual` (the ~10 uncovered
  combination drugs, from USAN/MedlinePlus, per-entry citation).
- person_names: pronunciation per TOKEN (first name and each surname token).
- Bank pydantic models gain the sub-model; validation: ASCII, respelling
  differs from `mispronounced` case-insensitively, `mispronounced` never
  collides with any decoy token.

### 2.4 Regeneration cascade

person_names bank regen -> identity maps / user_db -> task re-freeze. This
invalidates comparability with all prior intake runs (including the k=3
robustness cells) and re-keys complication draws — accepted by owner as the
cost of grounding. Snapshot per-sim rewards before the re-freeze
(snapshot-before-redo rule).

## 3. Synthesis: default readings always

Same seam as the swap (v1 sec. 2): `synthesize_voice` in
`src/tau2/agent/base/voice.py` derives `text_to_synthesize` from
`message.content` and already mutates it (speech inserts) without touching
the transcript. Substitution pass, USER side only:

- **Default (no complication)**: EMPTY map — every value synthesizes from
  its gold spelling and TTS gives its default reading. (Owner decision
  after the 2026-08-26 audition listen, RETIRING the original
  correct-by-default design: eleven_v3's default readings beat the curated
  respellings almost everywhere, hyphenated respellings slow and
  over-articulate the reading, and short ALL-CAPS syllables misrender as
  initialisms — "IL" -> "eye-ell". The `respelling` columns and the oddball
  token table stay as reviewed reference data — they define the canonical
  readings and drive the sec. 4 operators — but are never swapped in.)
- **`mispronounced_term` active**: the drawn term — and only it — maps to
  its `mispronounced` variant, which is stored TTS-READY in NATURAL
  ORTHOGRAPHY (a plausible normal English word, "iburprofen" — second
  owner listen, 2026-08-26: even dehyphenated-lowercase respellings
  sometimes get spelled out letter by letter, so the spoken form is
  rendered by `render_natural`, a closed respelling-unit -> ordinary-
  grapheme table; the hyphenated caps-stress respelling stays reference
  notation and never reaches TTS).
- Whole-token, case-insensitive, `\b`-bounded; transcript keeps gold;
  spell-outs pass through untouched (letter tokens never match); the swap
  repeats on every utterance, so the mispronunciation is CONSISTENT for
  free. Swap events (turn idx, token, form used) logged.

## 4. Mispronounced variants: seeded distortion operators

Machine, not scripts: bad variants are not free-hand. A closed in-code
catalog of distortion operators produces the bad variant, chosen per token
by seeded draw at BUILD time (recorded in the bank, so the run reads fixed
data):

- `metathesis` — transpose adjacent phonemes within a syllable
  ("azi-THRO-mycin" -> "azi-THOR-mycin" — the owner's example).
- `vowel_swap` — substitute one stressed vowel ("LIS-in-o-pril" ->
  "LEES-in-o-pril").
- `syllable_drop` — drop one unstressed medial syllable
  ("o-MEP-ra-zole" -> "o-MEP-zole"). **Medications only**.
- `spelling_pronunciation` (2026-08-26) — read the token AS SPELLED, the way
  a naive American reader would ("Chaim" -> "CHAYM", chain with an m,
  instead of the correct "HY-ihm"). Unlike the four syllable operators it
  runs on the token SPELLING, through a fixed in-code letter-to-sound rule
  set (vowel teams, r-controlled vowels, consonant digraphs, hard/soft c/g,
  silent/magic e, Mc-prefix; `ei` reads "ay" as in "eight" — the one
  documented choice for the ambiguous team), emitting the same
  hyphenated-syllable caps-stress alphabet as the converters. Feasibility:
  the naive reading must differ AUDIBLY from the correct respelling
  (hyphens stripped, case folded — transparent spellings like "Metformin"
  ARE their naive reading, so nothing is mispronounced) and must not
  collide with any decoy token of the entry.

A fifth operator, `stress_shift` (move primary stress one syllable), was
RETIRED outright (owner decision, 2026-08-26 audition listen): its variant
differed from the correct respelling ONLY by caps position, and the caps
stress marker never reaches TTS — the operator produced no audible
mispronunciation. Its 30 person_names draws were redistributed by the
balance-greedy draw over the remaining operators.

**Natural-orthography variants** (owner decision, 2026-08-26 second
listen): operators run in respelling space (syllables), but the STORED and
SPOKEN variant is rendered by `render_natural` — a closed
respelling-unit -> ordinary-grapheme table producing a plausible normal
English word ("iburprofen", "chaym"), lowercase, no hyphens, no stress —
because the respelling notation misrenders in eleven_v3 (initialisms,
letter-by-letter spell-outs, hyphen over-articulation). Consequences,
all fail-loud:

- Feasibility and vacuousness are judged in natural space: a variant that
  spells the correct reading's spoken form, or the token's own spelling,
  is vacuous (sending the token's spelling gives TTS's own — correct —
  lexicon reading). Every `VOWEL_SWAPS` pair maps to distinct graphemes,
  so a swap is never erased by the rendering itself.
- 14 `spelling_pronunciation` hard pins were dropped: their naive readings
  spelled naturally reconstruct the token ("Olayan" -> "olayan"); those
  tokens fall back to the balance draw.
- Tokens where NO operator yields a non-vacuous variant are in the closed
  `NO_VARIANT_TOKENS` allowlist (currently medications `Valsartan`,
  `Losartan` — transparent "-sartan" spellings), carry no `mispronounced`
  column (enforced both directions at bank load), and are never drawn by
  `mispronounced_term`.

**Per-bank applicability** (owner directive 2026-08-26), consulted before
the per-value feasibility gates and validated on bank load: `syllable_drop`
applies to medications only (a person name missing a syllable reads as a
DIFFERENT name, not a mispronounced one).

| operator | person_names | medications |
|---|---|---|
| metathesis | yes | yes |
| vowel_swap | yes | yes |
| syllable_drop | no | yes |
| spelling_pronunciation | yes | yes |

Operator feasibility is value-gated (a 2-syllable name can't syllable_drop;
`vowel_swap` is always feasible, so the feasible set is never empty). The
**draw is balance-greedy per bank** (2026-08-26, replacing
uniform-among-feasible, whose histogram was an artifact of feasibility
rates — stress_shift 70 / vowel_swap 70 / syllable_drop 30 / metathesis 14):
tokens are visited in the existing seeded build order and each takes the
feasible operator with the lowest running count in its bank, ties broken by
the token's own seeded rng stream — still a pure function of the build
seed. Rationale: per-operator analysis of mispronounced_term outcomes needs
non-trivial cell sizes. Counts run per bank because the applicability sets
differ. The rendered (correct, mispronounced, operator) triple for every
entry goes in the review packet; owner review gates the bank write.

**Hand-tuned hard pins** (owner directive 2026-08-26, second pass): the
balance draw is blind to how HARD each operator's variant is for a given
token, so it wastes the hard-hitting cases — Chaim drew a mild vowel_swap
("HAY-ihm") when its naive spelling reading ("CHAYM") is barely
recognizable, and the drawn variants overall were too easy. Where one
operator's variant diverges far more audibly than the balance pick, the
owner pins that operator per token in the closed in-code table
`HARD_PIN_OPERATORS` (`pronunciation.py`): 25 person-name tokens + 24
medication tokens, curated from a divergence ranking of every feasible
operator's variant and then hand-reviewed (a pinned variant must sound
WRONG, never like a different valid entity — Janey was rejected because its
naive "JEH-nee" is exactly the name Jenny). Pinned tokens take their pinned
operator; everything else stays balance-greedy. With the pins, roughly half
of each bank's variants are hard (audibly far from correct) and half stay
mild — both regimes are measured. A pin whose operator stops being feasible
and a pin no build consumes both fail loud. Same techniques, no new
machinery: the pins select among the section-4 operators, per token.

## 5. Catalog v1.2.0: `mispronounced_term`

Unchanged from spec v1 except feasibility now reads the bank column:

- New `ComplicationKind.MISPRONOUNCED_TERM`; banks: medications,
  person_names (hard-tier entries — the only ones with `mispronounced`).
- Voice-only: `sample_complication` gains `channel: Literal["text","voice"]`;
  text runs never draw it. Packet records channel in provenance.
- **No prompt injection** — `user_prompt_task` skips this kind; the sim is
  unaware. `SampledComplication.line` carries fixed packet-display text
  marked "NOT injected; applied at synthesis".
- `params = {"term", "respelling", "operator"}`;
  `SampledComplication.mispronunciation` = the bad respelling.
- Version keys the RNG: all draws re-key; re-render the seed packet.

## 6. Verification gates (before first run)

1. **Bank packet**: per-entry
   (gold, phonemes, respelling, mispronounced, operator, source) table.
2. **Audition verb**: `tau2 intake-tasks audition-pronunciations -o DIR` —
   for every mispronounced-bearing entry synthesize a PAIR of wavs: gold
   (the default TTS reading of the gold spelling — what every
   uncomplicated run says, sec. 3) and mispronounced (TTS-rendered).
   Owner spot-listens: mispronounced must be audibly wrong. (The original
   three-cut design with a "correct respelling" cut was retired with
   correct-by-default, sec. 3.)
3. **Unit tests**: phoneme->respelling converters (golden table), distortion
   operators (deterministic per seed, feasibility gates), swap function
   (whole-token, spell-out untouched, correct-vs-mispronounced selection),
   catalog validation, prompt NON-injection,
   packet determinism.

## 7. PR plan (base=intake)

- **PR-A — pipeline + banks**: lexicon sources in extract v2, coverage
  filter, pronunciation columns + converters, distortion
  operators, bank regen, task re-freeze, bank packet. (Biggest; everything
  else reads its data.)
- **PR-B — catalog v1.2.0**: the kind, channel parameter, packet rendering,
  provenance fields.
- **PR-C — synthesis**: correct-by-default substitution + complication
  selection at the voice.py seam, swap logging, audition verb.
- Gates: owner reviews PR-A's bank packet and PR-C's audition wavs before
  the first complicated voice run.

## 8. Oddball-token pronunciation table (PR-C scope; owner, 2026-08-25)

A mechanical scan of the invented-content banks (coined, insurance_plans,
rate_plans, shops, addresses, vehicles, properties; values AND decoys) for
TTS-risk token classes found 318 unique tokens:

- 38 mixed-case-inner (GmbH, XyloNine, gCaorach, MzK) — genuinely
  unpredictable readings;
- 31 all-caps initialisms (PPO, HDHP, MICE, SMERF, BAR, FIT) — several are
  real words TTS may speak instead of spelling;
- 2 ampersand tokens (B&B);
- 247 letter-digit mixes (3B, 228i, 4MATIC) — read predictably; **ignored
  by owner decision**.

(Re-measured at PR-C build time over the post-#907 banks with the pinned
extractor: 276 unique tokens — 40 mixed-case-inner, 60 all-caps, 1
ampersand, 175 letter-digit ignored. The classes are unchanged; the counts
above were the pre-#907 scan.)

(Amended in the 2026-08-26 packet review: the owner graduated 21
letter-digit tokens out of the ignored class — leet-spelled coinages whose
intended word is not the naive reading (Dr4gline, Qw1kfit, Xen0dyne, ...)
plus convention-bound vehicle/rate codes (4MATIC, sDrive28i, 228i, P300e,
15Pax). They carry the new `letter_digit` class in the table, pinned via the
closed in-code allowlist `GRADUATED_LETTER_DIGIT`; the remaining 154 stay
ignored. Table now 122 entries.)

The non-ignored tokens get a shared token-level respelling table,
`data/tau2/domains/intake/banks/token_pronunciations.yaml`
(token -> {respelling, class, note}). REFERENCE DATA ONLY since the sec. 3
default-readings-always decision (2026-08-26): the table defines the
canonical reading of each invented or convention-bound token for reviewers
and judges, but is never swapped into synthesis — the audition listen
showed eleven_v3's default readings beat the respellings and the respelling
notation misrenders. Real initialisms pin the documented industry
convention (MICE and SMERF spoken as words; PPO/HDHP/GmbH as letter names —
letter names written out in the respelling, e.g. "gee-em-bee-aitch");
coined tokens are authored and owner-reviewed in the packet. The 247
ignored tokens are listed in the packet as ignored.

## 8b. Phase 3: properties + vehicles foreign tokens (BUILT 2026-08-26; see "As built" below)

Foreign-language tokens in the properties and vehicles banks are the same
TTS-risk shape as hard names, and (unlike coinages) an external truth exists
for most of them. Owner directives (2026-08-26): every covered token carries
BOTH a correct respelling AND an incorrect (mispronounced) variant, exactly
as names/medications do, so the complicator can draw the wrong one; the
per-entry language pin is delegated (curated in the phase-3 PR, reviewed via
the packet). (Post default-readings-always, sec. 3: the correct respelling
is reference data — only the mispronounced variant, TTS-rendered, is ever
swapped in.)

### Measured scope (2026-08-26)

114 unique foreign tokens (alphabetic, absent from CMUdict + the English
wordlist): 90 in properties, 24 in vehicles. First-pass attestation against
12 WikiPron lexicons (deu, fra, ita, spa, por, pol, tur, swe, nob, ron, gle,
ell): properties 43/90, vehicles 5/24. The 47 property misses decompose:

- missing lexicons (Dutch, Welsh, Scottish Gaelic, Czech, Hungarian) —
  recovered by adding those TSVs;
- inflected forms WikiPron lemmatizes away (Bialym/Bocianem Polish
  instrumentals, Lupului Romanian genitive, Weisses/Krummen German
  declensions, gCaorach Irish eclipsis) — recovered by per-token Wiktionary
  entry lookup (the inflected forms have their own entries with
  pronunciations);
- Germanic compounds (Alpenblick, Fjellstua, Gjestgiveri, Kvarnviken,
  Vertshuset) — Wiktionary lookup or compound decomposition;
- romanized Greek (Asimeniou, Feggariou) — transliteration step or authored;
- transparent English compounds misflagged as foreign (Amberfield, Bayfront,
  Harborview, Silverpine, ...) — reclassified tier 3 (default), no column.

Vehicles' 19 misses are mostly manufacturer coinages (Evoque, Kodiaq,
Captur, Aircross) — tier 2 authored by design — plus trim codes (GLB, TFSI,
xDrive) already owned by the section-8 oddball table. Genuinely unattested
residual after all recovery steps: ~10-15 tokens (archaisms like Brycgstow,
the Greek transliterations), authored and owner-reviewed.

### Language pin

Each properties/vehicles entry gets a one-time `language` tag (~80 entries,
curated from the entry's linguistic flavor; several tokens are attested in
multiple languages, so first-hit lookup is NOT acceptable — the pin decides
which lexicon and which adaptation table apply). The pin is data, reviewed
in the packet like everything else.

### Sourcing ladder (per token, in order; source recorded per token)

1. Pinned-language WikiPron bulk TSV (pinned commit + sha, like PR-A's
   lexicon sources). Lexicon keys are native-script/diacritic forms; bank
   values are ASCII-folded (#534), so lookups fold lexicon keys to ASCII.
2. Per-token Wiktionary entry lookup for inflected forms and compounds —
   attested data with a per-token source URL in the column; no LLM anywhere.
3. Authored respelling (tier 2) for coinages, archaisms, and transliteration
   leftovers — owner-reviewed in the packet.

### Loanword adaptation

Target pronunciation is the FLUENT-AMERICAN anglicized reading, not native:
a fixed in-code per-language phoneme->English-respelling adaptation table
(German /ç/ -> "kh"/"sh", Portuguese nasals -> "-ng", etc.) maps
source-language IPA into the same respelling alphabet the PR-A converters
emit. Unit-tested with golden vectors per language.

### Mispronounced variants + complication extension

Every token with a correct respelling also gets a mispronounced variant via
the SAME seeded distortion-operator catalog (section 4). mispronounced_term
feasibility extends to the properties and vehicles banks (entry has a
mispronounced column). This is a CATALOG bump (draws re-key; packet
re-render) — the columns themselves change no values, so there is NO task
re-freeze; the catalog bump is the only comparability event in phase 3.

### Gates

Same as phases 1-2: packet renders (token, language pin, source, phonemes,
respelling, mispronounced, operator) for every covered token; the audition
verb synthesizes gold/correct/mispronounced triples; owner review gates the
bank write and the first post-bump complicated run.

### As built (2026-08-26, phase-3 PR)

Pipeline: `tau2 intake-names extract-foreign` (candidate scan + all-language
fold index against the 16 pinned WikiPron TSVs, checked-in extracts with
per-lexicon URL/date/sha provenance) -> `build-foreign` (in-place bank
rewrite, seeded balance-greedy variant draw, manifest) -> `foreign-packet`
(review packet). Curated tables + adaptation live in
`tasks/foreign_banks.py` and `tasks/loanwords.py`. Deviations and decisions
vs the plan above, all reviewable in the packet:

- **Counts**: 110 candidates (91 properties / 19 vehicles — the 24 vehicles
  figure above counted 5 trim codes the oddball table already owns). 92
  column-bearing tokens: 46 pinned-WikiPron, 2 exact-form Wiktionary
  (Herberge, Syv), 44 authored; 18 transparent English compounds tier 3.
  37 property + 8 vehicle entries carry language pins across 16 languages.
- **Greek (ell) dropped from the lexicon set**: its keys are Greek-script
  and can never fold-match an ASCII bank token; the two romanized-Greek
  tokens (Asimeniou, Feggariou) are authored.
- **gCaorach stays oddball-owned**: mixed-case (Irish eclipsis) tokens are
  section-8 territory; the per-token Wiktionary rung therefore carried only
  exact-form entries whose printed IPA covers the whole token.
- **Curated tables take precedence over the bulk lookup** (a deliberate
  re-ordering of ladder rungs 1-2): ASCII folding creates false matches —
  deu kruemmen (verb, wrong vowel) shadows Krummen; pol impreza (a party)
  is a false friend of the Subaru coinage; ita Nitti (surname) is the wrong
  language for Norwegian nitti. Every shadowed bulk match is recorded in
  the manifest and rendered in the packet.
- **Accent-marked stress goes authored**: the ASCII fold loses written
  accents, so the Romance stress rule misplaces zaguan (native final),
  animas (native antepenult), hospederia (native -RI-); all three are
  authored with the native reading.
- **NO_VARIANT_TOKENS**: Gasthof and Harom drew only vacuous variants
  (their vowel-swapped natural rendering IS their own spelling) and carry
  columns without a mispronounced variant.
- **Operator applicability** (PROPOSED, owner review via the packet):
  properties/vehicles use the person_names set — metathesis, vowel_swap,
  spelling_pronunciation, NO syllable_drop (a place/model name missing a
  syllable reads as a different name).
- **Rates** (JUDGMENT): properties/vehicles share
  MISPRONOUNCED_TERM_PERSON_NAMES_RATE (0.15) — foreign tokens in
  hotel/car names are the same caller construct as unfamiliar person names.
- **Catalog bump v2.1.0 -> v2.2.0**; `tau2 intake-tasks freeze` verified
  byte-identical for tasks.json/split_tasks.json (tasks.manifest.json
  churns — it embeds bank shas). Packet review + audition gate the first
  post-bump armed run, not the merge.

## 9. Measurement (unchanged from v1)

Reward + `log_capture` give first-capture vs corrected; cross-tab swap
counts vs reward; the spell-out policy is the designed recovery path.
