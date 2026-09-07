# Intake entity provenance

The living per-bank provenance reference for the 15 intake entity banks in
`banks/` (40 easy + 40 hard values each). Ten of them form the canonical
benchmark draw (`CANONICAL_BANKS`, owner decision 2026-08-26); the five rows
marked RETIRED below keep their bank data and machinery but no longer appear
in the frozen task set. Says what each bank's values are
grounded in TODAY, by which mechanism, and what remains on the roadmap.
Design rationale lives in `docs/designs/intake-grounding.md` (grounding-mode
taxonomy) and `docs/designs/intake-mispronunciation.md` (pronunciation
program). Update this table in the same PR as any change to a bank's
grounding, screening, or pronunciation status.

Grounding modes: **positive** (values ARE real, from a pinned dataset or
validated against a registry) · **algorithmic** (values satisfy a checkable
rule) · **negative** (values are provably NOT real; re-screened at build
time because reality moves) · **self-grounded** (validity is internal:
calendar, clock, arithmetic).

| Bank | Mode | Grounded in | Enforcement / status |
|---|---|---|---|
| `person_names` | positive | SSA given names + Census 2010 surnames + pronunciation lexicons (CMUdict `cmusphinx/cmudict@0f8072f`, WikiPron `eng_latn_us_broad` `CUNY-CL/wikipron@d282e84`, FreeBSD web2 wordlist `freebsd-src@72f0bc8`), pinned zips + shas (`tau2 intake-names`, `name_sources/provenance.json`) | pipeline-built (PR-A); hard given / tail-surname pools filtered to lexicon-covered, non-dictionary-word, length ≥ 5; per-token pronunciation columns on hard tier incl. seeded `mispronounced` variant + operator (balance-greedy per-bank draw over the closed 4-operator catalog incl. `spelling_pronunciation`, with 25 owner hand-pinned hard tokens via `HARD_PIN_OPERATORS`; `syllable_drop` never on names; `stress_shift` retired 2026-08-26 — design doc §4); review packet via `tau2 intake-names packet` |
| `medications` | positive | real generic drug names, hand-curated; phonemes from WikiPron, 9 lexicon-gap entries hand-filled from MedlinePlus with citations | pronunciation columns on every drug-name token incl. `mispronounced` variant (PR-A; balance-greedy per-bank operator draw with 24 owner hand-pinned hard tokens via `HARD_PIN_OPERATORS` — design doc §4); values AND decoys screened vs NLM RxNorm (`tau2 intake-grounding rxnorm`, `grounding_rxnorm.json`): 131/134 pass, 3 findings — value `Prilocaine 2% cream` (US market only has the 2.5% lidocaine-prilocaine combination cream), decoys `Dapoxetine 30 mg tablet` (ingredient not in RxNorm; not US-marketed) and `Sulfasalazine 1000 mg tablet` (only 500 mg marketed) |
| `vehicles` | positive | real makes/models, informally curated | values AND decoys screened vs NHTSA vPIC (`tau2 intake-grounding vpic`, `grounding_vpic.json`): 124/160 pass, 36 findings — 18 hard-tier EU-market values + their decoys (makes vPIC lacks: Skoda, Seat, Cupra, Vauxhall, DS; EU-only models under US-known makes: Citroen C4 Picasso / C5 Aircross, Peugeot 3008 / 508, Renault Megane / Captur, Alfa Romeo Giulia Ti, Range Rover Evoque; Mercedes trims — vPIC lists class names, not `CLA 250`-style trims). vPIC covers US-market vehicles, so EU-only findings are expected facts, not defects; 14 foreign tokens carry pronunciation columns + entry language pins (phase 3, design doc §8b — 3 pinned-WikiPron, 11 authored coinages/place names with cited manufacturer readings; 5 transparent English compounds tier 3; `tau2 intake-names build-foreign`); **RETIRED from the canonical draw 2026-08-26** (bank kept, dropped-languages precedent) |
| `codes` | algorithmic | VINs: ISO 3779 check digit + real WMI prefixes; member IDs: mod-97 (`check_digits.py`) | enforced at bank load and at submit |
| `phones` | algorithmic (regulator ranges) | officially reserved fictional ranges only: NANP 555-0100..0199 (NANPA), UK Ofcom drama +44 20 7946 0xxx, AU ACMA +61 2 5550 xxxx, FR ARCEP +33 6 39 98 xx xx — values AND decoys | enforced at bank load (#907, `grounding.py`) |
| `emails` | negative | coined domains, must not resolve (A/AAAA/MX) | screened by `tau2 intake-grounding screen` (#907); first screen found 5 resolving domains — replaced (basename pairs swapped whole) and re-screened clean |
| `coined` | negative | invented companies, must not be registered entities | screened vs SEC EDGAR + GLEIF (#907): clean |
| `insurance_plans` | negative | invented carriers + real plan-type vocabulary (PPO/HMO/HDHP/EPO) | carriers screened vs EDGAR + GLEIF (#907): clean; **RETIRED from the canonical draw 2026-08-26** (bank kept, dropped-languages precedent) |
| `properties` | negative | invented hotels with real foreign-language flavor | screened vs EDGAR + GLEIF (#907): clean; 78 foreign tokens carry pronunciation columns + entry language pins across 16 languages (phase 3, design doc §8b — 43 pinned-WikiPron, 2 exact-form Wiktionary, 33 authored with cited rationale; 13 transparent English compounds tier 3; `tau2 intake-names build-foreign`, packet-reviewed) |
| `shops` | negative | invented repair shops | screened vs EDGAR + GLEIF (#907): clean; **RETIRED from the canonical draw 2026-08-26** (bank kept, dropped-languages precedent) |
| `rate_plans` | negative | invented plan names over real industry vocabulary | no registry exists; unscreened by design; **RETIRED from the canonical draw 2026-08-26** (bank kept, dropped-languages precedent) |
| `addresses` | grammar-generated | synthetic street grammar, format-plausible postcodes | OS Open Names / Census TIGER component screens possible; low priority |
| `dates` | self-grounded | calendar validity against the pinned clock | nothing needed |
| `times` | self-grounded | clock times (hard tier = deliberate 12:00 AM/PM ambiguity) | nothing needed |
| `amounts` | self-grounded | plain currency amounts | nothing needed; **RETIRED from the canonical draw 2026-08-26** (bank kept, dropped-languages precedent) |

## Pronunciation tiers (cross-cutting; program in `docs/designs/intake-mispronunciation.md`)

1. **Attested** — a real pronunciation exists: hard-tier names and all
   medications carry lexicon-sourced phonemes + derived respelling (PR-A);
   properties/vehicles foreign tokens (phase 3, design doc §8b) resolve
   through the sourcing ladder — 46 from the 16 pinned per-language WikiPron
   TSVs (`CUNY-CL/wikipron@d282e84`, entry-level language pins, anglicized
   via the fixed in-code adaptation tables in `tasks/loanwords.py`), 2 from
   exact-form en.wiktionary IPA (per-token source URL), 44 authored with
   cited rationale (coinages, archaisms, romanized Greek, inflected forms,
   accent-marked stress the ASCII fold loses); 18 transparent English
   compounds reclassified tier 3. Pipeline: `tau2 intake-names
   extract-foreign / build-foreign / foreign-packet`
   (`name_sources/foreign_provenance.json`,
   `foreign_banks.manifest.json`).
2. **Authored** — no external truth exists: 122 oddball tokens (mixed-case
   coinages, initialisms, `B&B`, plus 21 owner-graduated letter-digit
   tokens — leet coinages and trim codes flagged in the 2026-08-26 packet
   review) get owner-reviewed respellings that DEFINE the canonical reading
   (`banks/token_pronunciations.yaml`).
3. **Default** — reliable TTS reading: easy-tier names, the remaining 154
   letter-digit tokens, numerals, dates. No column, by decision.

Since the default-readings-always decision (owner, 2026-08-26 audition
listen — design doc §3), ALL tiers synthesize from the gold spelling:
tier 1/2 respellings are reference data (they define the canonical reading
and drive the distortion operators) and are never swapped in. The only
synthesis swap left is the `mispronounced_term` complication's drawn term,
whose `mispronounced` column is stored TTS-ready in natural orthography
(a plausible normal word, e.g. "chaym" — design doc §4). Two medication
tokens (`Valsartan`, `Losartan`) and two property tokens (`Gasthof`,
`Harom`) have no non-vacuous natural variant and carry no variant (closed
`NO_VARIANT_TOKENS` allowlist).

Screen reports (dated, provenance-bearing): `grounding_screen.json` (re-run
`tau2 intake-grounding screen` after any change to a negative-grounded bank),
`grounding_vpic.json` (re-run `tau2 intake-grounding vpic` after any vehicles
change), and `grounding_rxnorm.json` (re-run `tau2 intake-grounding rxnorm`
after any medications change).
