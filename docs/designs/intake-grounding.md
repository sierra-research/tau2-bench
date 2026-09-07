# Intake bank grounding

How the intake entity banks (`data/tau2/domains/intake/banks/*.yaml`, 15 banks,
40 easy + 40 hard values each — see `src/tau2/domains/intake/tasks/banks.py`)
stay anchored to reality in exactly the ways the benchmark needs, and provably
detached from it everywhere else.

## Grounding-mode taxonomy

Every bank sits in one of three grounding modes:

1. **Positive grounding** — values must BE real (or derived from real
   distributions). A caller spelling a real medication or a real vehicle
   model exercises the same recognition priors production traffic does;
   an invented one would test something easier. Positive grounding is
   enforced by building the bank from an authoritative dataset (SSA/Census
   for names) or by validating each value against a reference registry
   (`tau2 intake-grounding vpic` for vehicles vs NHTSA vPIC,
   `tau2 intake-grounding rxnorm` for medications vs RxNorm — below).
2. **Algorithmic grounding** — values must satisfy a checkable internal
   structure rather than exist in the world. VINs carry the ISO 3779
   check digit and real WMI prefixes; member IDs carry mod-97 check digits
   (`src/tau2/domains/intake/check_digits.py`, enforced at bank load and at
   submit time). Phone numbers are the special case where the *algorithm is
   the regulator's*: every value must sit inside an officially reserved
   fictional range (`src/tau2/domains/intake/grounding.py`, enforced at bank
   load — this PR).
3. **Negative grounding** — values must NOT be real. Fictional email
   domains, coined companies, insurance carriers, hotels, and repair shops
   must stay fictional: a bank value that names a real registered entity
   leaks reality into a fictional scenario (and puts real parties into
   generated transcripts). Negative grounding cannot be proven once and
   frozen — reality moves — so it is a *screen*, re-run at build time:
   `tau2 intake-grounding screen` (`grounding_screen.py`, this PR) checks
   DNS non-resolution for email domains and exact/case-folded name absence
   from SEC EDGAR and GLEIF LEI registries, and writes a provenance-bearing
   report to `data/tau2/domains/intake/grounding_screen.json`. The verb
   never edits a bank; findings are for the owner to act on.

Load-time validation is pure and offline (fail-loud `ValueError`s in the
bank loader); network access happens only in the screen verb, at build time.

## Per-bank grounding table

The living table moved next to the banks it describes:
`data/tau2/domains/intake/ENTITY_PROVENANCE.md`. Update it in the same PR as
any change to a bank's grounding, screening, or pronunciation status; this
design doc keeps only the rationale.

## The phone range table (deliverable 1)

`grounding.py` carries the closed, in-code range table (machine-not-scripts:
fixed data, cited per regulator) and `load_banks` rejects any phones-bank
value or decoy outside every range. The checked-in bank is 40 easy NANP
values plus a hard tier of 10 UK + 10 AU + 10 FR + 10 NANP-with-extension —
all inside the reserved ranges by construction, now provably so on every
load.

## The negative screen (deliverable 2)

`tau2 intake-grounding screen`:

- **emails** — every `domain`/`decoy_domain` looked up (A, AAAA, MX) through
  the Google Public DNS DoH JSON API. Any answer record ⇒ FINDING (someone
  owns the domain); NXDOMAIN everywhere ⇒ PASS; transport failure ⇒
  UNCHECKED (never PASS).
- **registry names** — coined values+decoys, property values+decoys, shop
  values+decoys, and extracted insurance carriers, checked exact and
  case-folded against (a) SEC EDGAR `company_tickers.json` (one pinned
  download, sha256 recorded in the report) and (b) GLEIF LEI fulltext search
  (one polite, rate-limited request per unique name; lookup failure ⇒
  UNCHECKED).
- Output: `GroundingScreenReport` (screen date, tau2 version, screen schema
  version, screened-bank shas, sources with URLs, per-check verdicts) at
  `data/tau2/domains/intake/grounding_screen.json`. Exit 0 with findings
  listed; nonzero only when the screen itself cannot run.

## The positive screens (deliverable 3)

Same shape as the negative screen — build-time verbs, report-only, network
only at the screen, transport failure ⇒ UNCHECKED (never PASS), findings are
facts for the owner rather than auto-failures:

- **`tau2 intake-grounding vpic`** (`grounding_vpic.py`) — every
  vehicles-bank value AND decoy. The make must be the longest word-boundary
  prefix of the value in the vPIC `getallmakes` catalog (one pinned
  download, sha256 recorded), and a `GetModelsForMake` model (one cached
  request per unique make, URLs recorded) must match the remainder exactly
  or as a leading word-boundary prefix (vPIC lists base models without
  trim). Names are diacritic-stripped, casefolded, whitespace-collapsed.
  vPIC covers US-market vehicles, so EU-only makes/models surface as
  expected FINDINGs. Report: `grounding_vpic.json`.
- **`tau2 intake-grounding rxnorm`** (`grounding_rxnorm.py`) — every
  medications-bank value AND decoy, parsed as ``<Drug> <strength> <form>``.
  Every ingredient must be known to RxNorm (`rxcui.json?search=2`), and a
  marketed product concept (`drugs.json` by first ingredient) must match
  the strength (rendered into RxNorm vocabulary: mcg also as MG, percent as
  MG/ML / MG/MG, ratios per-mL, combination strengths set-matched) and the
  form (closed map into RxNorm dose-form phrases). Report:
  `grounding_rxnorm.json`.

## Roadmap

1. **PR-A (merged)** — pronunciation lexicons for `person_names`
   (same-file work in `name_banks.py` / the person_names bank).
2. **Deliverables 1+2 (merged, #907)** — phone fictional-range load
   validation + the negative screen verb and its first committed report.
3. **Deliverable 3 (this PR)** — positive grounding for `vehicles` (NHTSA
   vPIC) and `medications` (RxNorm), reports committed; findings recorded
   in `ENTITY_PROVENANCE.md`.
4. **Foreign-token pronunciation for `properties`/`vehicles`** — phase 3 of
   the pronunciation program (see `docs/designs/intake-mispronunciation.md`).
