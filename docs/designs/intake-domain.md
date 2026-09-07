# The `intake` domain — an outbound entity-capture benchmark

**Status:** design v4, 2026-08-24 — full respecification (rip-and-redo).
**Owner:** soham. **Scope:** the English domain; the localization seam
(§10) is kept but not built here.

**What changed in v4 (owner decisions, 2026-08-23/24):** the domain is
rebuilt around what the v3 study proved and discarded what it disproved.
The 240-call voice study (artifacts `c49269fbad41`, `687fe924d2e0`;
comparison in `data/analysis/intake_study_spellout_comparison_2026-08-23.md`)
established that: task failure is driven almost entirely by per-entity
**capture**; verify-role slots are free; complications cost ≈ nothing;
multi-entity failure is multiplicative (per-slot hazards do not degrade
with count); and the capture hazard decomposes into procedural,
orthographic, and acoustic channels. Therefore v4:

- **Flips the call direction: outbound.** The agent calls the person to
  collect field values missing from a record they already submitted.
  This makes single-entity calls natural (a "missing field callback" is
  a common real CS call), removes intent recognition and lookup from the
  shell entirely, and reduces the reward diff to exactly the missing
  cells.
- **Rips out all complication machinery** — the operator catalog, repair
  contracts, falsification checks, belief staleness, documents,
  `double_check_*`/`read_*` tools, unsolvable recipes, the transfer
  action. Deleted, not parked (v3 history and code remain in git).
- **Rips out the inbound flows** — the six intake types, the arrival
  flow, directory lookups, identity cross-check ceremony,
  `configure_intake_requirements` (the mechanism behind the placeholder
  defect found in the study), `log_verification`.
- **Two knobs only**: entity tier (easy | hard) and entity count
  (1 | 2 | 3 | 5). Everything else is fixed. Entity families are
  balanced across the freeze, not ranked.
- **Keeps the crown jewels**: the value banks (census names, ASCII-only
  rule, real-near-neighbor decoys), the folds and check-digit machinery,
  the letter-by-letter pinning policy validated by the study, the
  enumerate → verify → freeze provenance discipline, and the
  `intake-study analyze` attribution seam (simplified).

## 1. Goal

Measure **interactive entity capture** in voice: given one short
cooperative call per task, does the correct *written* value land in the
database? The domain isolates the capture exchange from everything else
a CS call contains, so that:

- per-**bank** × tier hazards are directly readable from pass rates
  (every canonical task is n=1, so task pass **is** the entity hazard —
  no attribution machinery in the headline numbers);
- localization later swaps banks per language while the shell, policy,
  and protocol stay fixed (§10).

Multiplicativity (task pass ≈ ∏ per-entity success at n > 1) was
established once on the v3 study (e2c ≡ e3, artifacts `c49269fbad41` /
`687fe924d2e0`) and is **deferred here**: the generator keeps the
`n_entities` knob, but the canonical set is atomic-only; a multi-entity
band is a later additive draw if ever needed.

## 2. Cover story

The same multi-client **service bureau intake desk**, now working its
callback queue. A person recently submitted a form (clinic
registration, auto service request, hotel booking — the vertical only
picks which banks a task draws from). The record went in with one or
more required fields missing. The desk calls the person back to
complete it. Outbound data-completion callbacks are a standard
real-world CS workload, which is what makes a one-entity call natural
rather than contrived.

## 3. The call, turn by turn

The shape is fixed and identical across every task:

```
agent  "Hello, this is {org} calling."            ← deterministic opener:
                                                     task-supplied text, injected
                                                     unsynthesized (silence audio)
                                                     via create_initial_message(content=…)
user   "Hello?"                                   ← near-deterministic: the fixed
                                                     user-sim template instructs the
                                                     callee to open their first turn
                                                     with "Hello?"
agent  states the purpose, naming the person: "I'm calling for
       {callee_name} about the form submitted {when} — the {field(s)}
       are missing." The callee's natural acknowledgment ("Yes, that's
       me") covers identity; there is no mandated confirmation step.
       — then per missing field —
agent  asks for the field
user   calls get_entity(field), answers with the value
agent  pins the written form per policy (letter-by-letter proper nouns,
       char-by-char codes, symbol-by-symbol emails, …)
agent  "To confirm, your {field} is {value}?"     ← mandatory per-entity read-back
user   confirms (or corrects; the agent re-pins and re-confirms)
       — after all fields —
agent  submit_fields(...), then "Thanks, that's all I need." + agent_stop
                                                  ← standard unsynthesized stop
                                                     marker, as in all domains
```

Fixed properties: the callee is always cooperative, always reachable,
always the right person, and always able to produce every value via
`get_entity`. There is **no transfer** — the endings are `agent_stop`
(normal) or `user_stop`, exactly like the other domains. There are no
complications of any kind.

## 4. Knobs

The task spec is `vertical × tier (easy|hard) × n_entities (1|2|3|5)`.

- **Tier** applies to every slot in the task at once (per-task mode, as
  in v3). Easy/hard semantics per family are unchanged from v3 §6
  (frequency-tiered census names, code length/alternation, email symbol
  density, …) and live in the banks.
- **n_entities** counts the missing fields the agent must collect. The
  callee's identity confirmation is conversational (name spoken, no DB
  write) and does **not** count — every counted entity is a captured,
  DB-written value. This supersedes the v3 counting rule, which existed
  to price lookup keys in inbound flows that no longer exist.
- **The bank is the unit, not a family** (owner call, 2026-08-24):
  every hazard cell is one bank × one tier. Banks are homogeneous (one
  value grammar each), so a cell measures one clean thing; semantic
  families (place, jargon, coined, …) exist only as optional analysis
  roll-ups over bank rows, never as design objects. The 15 banks after
  the providers/parts retirement: person_names, codes, phones, dates,
  times, addresses, properties, amounts, emails, medications,
  insurance_plans, vehicles, rate_plans, coined, shops.
  **Canonical-10 cut (owner decision 2026-08-26, off the full300 run's
  entity analysis)**: the benchmark set draws from `CANONICAL_BANKS` =
  person_names, codes, phones, dates, times, addresses, properties,
  emails, medications, coined. Retired from the canonical draw (bank
  data and machinery stay, dropped-languages precedent): amounts and
  rate_plans (at ceiling, thin failure profiles), insurance_plans and
  shops (redundant with the invented-name cluster — coined stays as its
  hard representative), vehicles (its channels remain covered:
  mispronounced_term keeps 3 carrier banks, foreign tokens mostly
  duplicated properties' signal). Phones kept by fiat (ubiquity)
  despite its 17/20 near-ceiling score.
- **Bank contract**: every bank ships exactly **40 easy + 40 hard**
  values, validated at build time (flat, no per-bank exceptions). The
  canonical set draws 10 per cell without replacement, so the pool
  covers two full additive doublings (10 → 20 → 40) with no value
  reuse. Banks are the materialized, reviewed value universe the seeded
  freeze draws from — dataset-derived where difficulty is a fact about
  the world (census names, real medications), materialized generator
  output for the grammar families — so validation is once-per-pool and
  values stay byte-identical across generator rewrites.

## 5. Environment

### 5.1 DB (`data_model.py`, generated `db.toml` + per-task derivation)

- `intake_records`: pre-existing records with the task's missing fields
  set to an explicit MISSING sentinel. The expected final DB is the base
  DB with exactly those cells filled with the pinned true values
  (fold-compared, as today).
- `callback_orders`: one row per task — record id, callee full name,
  org display name (feeds the opener), and the ordered list of missing
  field names.
- No directories, no arrivals, no verification log. The pinned clock
  stays (dates are still an entity family).

Initial state remains a **derivation, not a blob**: seed data plus the
recorded env calls that blank the missing fields. The diff from the
complete record is readable in the task JSON.

### 5.2 Agent tools (`tools.py`)

Four tools, total:

- `get_callback_order()` (READ) — returns the work order: record id,
  callee name, org name, missing field names with their type
  descriptions. No arguments; one order per task.
- `submit_fields(record_id, fields)` (WRITE) — one typed pydantic call
  writing all collected values at once; ends the task's write phase.
  Submit exactly once, after all per-entity confirmations.
- `get_today()` (READ) — the pinned clock.
- `end_call()` (GENERIC) — hangs up; the agent's `agent_stop`. Both agent
  classes treat it as a call-terminating tool (marked with the stop token,
  never synthesized) — the outbound counterpart of the inbound
  transfer-ends-the-call convention, since intake has no transfer.

### 5.3 User side (`user_tools.py`)

One tool: **`get_entity(field)`** (READ) — returns the callee's true
value for that field, always. It models the person checking their own
records; it never fails, never returns stale values, never points to a
document. The sim never holds a hard value in prose — values enter the
conversation only through this tool, preserving the
environment-heavy/prompt-light property.

### 5.4 Policy (`main_policy.md` — rewritten, one page)

Sections, in order:

1. **Role and call frame**: you are calling on behalf of {org} to
   complete a record; state the purpose naming the person the record
   belongs to (no separate identity-confirmation step); collect only
   the fields named in the callback order — and every value you record
   must come from this call.
2. **Collecting each field**: ask, then pin the written form — the
   **"Pinning the written form" section carries over verbatim from v3**
   (letter-by-letter proper nouns with Bravo-anchoring, char-by-char
   codes with 0/O 1/I, symbol-by-symbol emails, decimal digits,
   unambiguous dates; the do-not-confirm list for case/accents/
   hyphens). It is the study-validated asset.
3. **Per-entity confirmation**: after pinning, read the value back and
   get an explicit yes before moving on. (The always-read-back rule is
   now policy, not habit.)
4. **Closing**: submit once via `submit_fields`, thank the callee, and
   end the call. No transfer exists; if the callee ends the call early,
   submit nothing.

### 5.5 The user-sim scenario — one common template

One fixed template for every task (no per-task prose). It specifies:
the callee persona (name from the record), the "Hello?" opening rule
for the first turn, cooperative disposition, answer-only-when-asked,
spell-when-asked (never volunteer spelling unprompted — keeping the
procedural channel measurable), and the `get_entity` tool as the only
source of values.

## 6. Task space: enumerate, verify, freeze

The v3 discipline survives with a much smaller frame:

- **Enumeration**: vertical × tier × n_entities × the valid bank
  assignments for that count. The frame size is a reportable fact,
  recomputed by the generator.
- **Verification gates** (generation time, per task): the record's
  missing-field set matches the order; every missing field has a pinned
  true value in the banks; the expected final DB is reachable by
  exactly one `submit_fields` call; fold-distinctness of values within
  a task; ASCII-only everywhere.
- **Freeze**: stratified draws with the same provenance machinery as
  v3 (named rng streams `intake-freeze|<draw>|{seed}`, recorded
  constraint sets, additive extensions keep earlier task bodies
  byte-identical). `GENERATOR_VERSION` 6.0.0.
- **The canonical set** (first freeze, owner-approved 2026-08-24):
  **15 banks × 2 tiers × 10 calls = 300 atomic (n=1) tasks.** Each
  call captures a distinct value (draw without replacement within a
  cell). No multi-entity cells and no multiplicativity band — deferred
  per §1; any bank whose hazard needs tighter resolution gets its cell
  extended additively (10 → 20) in a follow-up draw without touching
  the rest. **Re-frozen 2026-08-26 on the canonical-10 cut (§5):
  10 banks × 2 tiers × 10 calls = 200 tasks**, `GENERATOR_VERSION`
  7.0.0. Per-bank rng streams keep every kept task's values, callees,
  and complication draws byte-identical to the 15-bank freeze (verified
  against the full300 run: 200/200 gold values unchanged); only the
  sequential record ids shift.

## 7. Scoring

Reward = expected-DB hash match, as today: base DB + the missing cells
filled with the pinned values under the fold rules. No LLM judge in the
reward path. With one record and a known missing-field set, per-slot
attribution is a direct cell diff; `tau2 intake-study analyze` is
simplified accordingly (slot outcomes correct / wrong_value / missing;
the v3 cause ladder collapses since flows and complications are gone).

## 8. What is deleted (rip-and-redo, not parked)

From `src/tau2/domains/intake/`: the complication catalog and all
operator machinery in `tasks/` (scenarios, instantiation branches,
repair contracts, falsification checks), the six intake types and
`tasks/intake_types.py`, arrival flow, `find_*` directory tools,
`find_intake_subject`, `log_verification`, `confirm_arrival`,
`configure_intake_requirements`, `transfer_to_human_agents`, the
belief-record staleness machinery and document tools in
`user_tools.py`, and the v3 frozen task set (`tasks.json`,
`tasks.manifest.json`, `split_tasks.json`, `tasks_voice.json` — all
regenerated). The v3 study data and analysis artifacts stay in
`data/analysis/` and on Drive as the record that motivated v4.

Kept: `banks/` (all 15, unchanged), `name_sources/` + `tau2
intake-names`, `folds.py`, `check_digits.py`, the environment seam, the
freeze/manifest provenance pattern, the study package (simplified).

## 9. Build phases

- **Phase A (this doc)**: v4 respecification. PR: doc only.
- **Phase B — rip + core**: delete per §8; new `data_model.py`,
  `tools.py`, `user_tools.py`, `environment.py`, `main_policy.md`;
  orchestrator wiring for the task-supplied deterministic opener
  (`create_initial_message(content=…)` already accepts it); unit tests
  for tools, folds integration, and the opener seam.
- **Phase C — generator + freeze**: template, enumeration, gates,
  balanced draw, frozen set + manifest; text sanity smoke (a handful of
  tasks at reward 1.0).
- **Phase D — analyze + voice smoke**: simplified analyze verb over the
  new manifest; small voice smoke with the standard pins; then the
  first full battery is a run decision, not a build phase.

Each phase is a PR into `intake` (bg agent in a fresh worktree,
critique pass, bugbot, squash merge), per the standing pipeline.

## 10. Localization seam (kept, not built)

The v3 locale identity swap survives unchanged in concept: per-language
banks replace value pools; the shell, policy structure, spell-out
protocol (with per-language letter-naming conventions), and reward path
stay fixed, so cross-language hazard differences are attributable to
the language. The v3 §12 decision table remains the reference; it
re-enters scope only when localization work resumes.

## 11. Resolved questions

- **Why outbound?** Makes n=1 natural, deletes intent/lookup shell
  variance, shrinks the reward diff to the missing cells, and matches a
  real deployment category.
- **Why no verify-mode arm?** Owner call: capture is the measured
  quantity; verify slots were shown to be free. The directories left
  with the inbound flows.
- **What happens to the v3 directory-resolved fields?** Owner call
  (2026-08-24): they become plain spoken-value captures. Insurance
  plan, medication, part, rate plan, shop, and property names are
  recorded as text (jargon / coined / place families, per bank);
  referring providers join the person-name family. No ids, no
  resolution — the banks keep earning their keep as capture pools, and
  six formerly-free verify slots become genuine capture material.
- **Why one submit call?** Keeps the DB diff and reward semantics
  identical to the validated v3 machinery.
- **Why a money-amount family?** Owner call (2026-08-24): monetary
  amounts are a standard task-oriented-dialogue capture (SGD Banks_1
  `amount`/`balance`, fares and prices across its travel services) that
  the v3 banks lacked. Added as `banks/amounts.yaml` (USD, digit
  strings; easy = round whole dollars, hard = non-round with cents;
  decoys are digit-transposition near-misses), assigned to the **auto
  vertical** as the repair callback's `approved_budget` field. The
  "Decimal quantities" pinning rule already covers its written form.
- **Why a time-of-day family?** Owner call (2026-08-24): times are the
  one capture-worthy staple present in both MultiWOZ (`booktime`,
  `leaveat`) and SGD (`appointment_time`) that v3 lacked, with genuine
  written-form ambiguity (meridiem, non-round minutes). Added as
  `banks/times.yaml` (canonical written form "h:mm AM/PM" - the
  meridiem is identity-bearing; easy = hour/half-hour daytime, hard =
  non-round minutes plus AM/PM- and noon/midnight-confusable classes;
  decoys are meridiem flips and minute transpositions), assigned to the
  **clinic vertical** as the callback's `appointment_time` field. Phase
  C adds the matching pinning bullet (record the meridiem explicitly,
  never a bare "2:30") alongside the generator wiring.
- **Why retire providers.yaml and parts.yaml?** Owner call
  (2026-08-24). Providers duplicate person_names post-census-refactor
  (same SSA/Census pools, a "Dr." prefix); referring-provider fields
  draw from person_names. Parts are dictionary compounds ("rear brake
  pad set") — the exact class the pinning policy exempts from spelling,
  so near-zero expected hazard. Both yamls are deleted in Phase C with
  the generator rewrite (v3 code still references them).
- **Why banks instead of families, and 10 calls per cell?** Owner call
  (2026-08-24). Families average heterogeneous banks (a drug name and a
  trim level need not fail alike); bank × tier cells are homogeneous,
  and families survive as analysis roll-ups. 10 calls/cell separates
  the large effects and keeps the canonical set at 300 calls; additive
  extension is the resolution mechanism, not a bigger first draw.
- **Why is there no identity-confirmation step?** Owner call
  (2026-08-24, maximum minimalism): the callee is always the right
  person, so a mandated check never fails and measures nothing. The
  purpose statement names the person and the callee's natural
  acknowledgment covers it. Counted entities are exactly the DB-written
  captures, so pass rates compose multiplicatively over them.
- **Spell-out stance**: mandated always (owner call, 2026-08-24); no
  free-policy arm.
- **TIME and AMOUNT fold rules** (Phase C, 2026-08-24): the closed fold
  catalog gains `time` (canonical "h:mm AM/PM"; 24-hour readings,
  leading zeros, meridiem spellings, and noon/midnight normalize; a
  bare 1–12-hour reading like "2:30" deliberately stays as given — the
  policy pins the meridiem, and an agent that failed to pin it fails
  the match) and `amount` (minimal decimal digit string; "$", thousands
  commas, "USD"/"dollars", and trailing cent zeros fold away). The
  policy's "get AM or PM explicit" pinning bullet covers the time
  family.
- **Bank flattening + deterministic trims** (Phase C, 2026-08-24): all
  banks share the flat `value/difficulty/decoys` entry shape (closed
  tags kept where the design splits a bank: code `family`, date `kind`,
  coined `role`; person names keep `gender`/`alt_name`; emails stay
  owner-linked patterns). Over-full banks were trimmed to the exact
  40+40 contract deterministically: person names by the builder count,
  phones by file order, codes and dates by fixed per-family/per-kind
  recomposition; short banks were extended in each bank's existing
  grammar. Medications formerly in the capture-only pool carry no
  decoys (decoys have no runtime role in v4; they anchor the fold
  collision sweep).
- **Unit catalog: bank → record field** (Phase C, 2026-08-24, fixed
  in-code in `generator.py`): person_names→hotel
  `second_guest_full_name`; codes by family (vin→auto `vin`, plate→auto
  `license_plate`, member_id→clinic `member_id`, loyalty→hotel
  `loyalty_number`); phones→auto `contact_phone`; dates by kind
  (birth→clinic `date_of_birth`, future and relative→auto
  `drop_off_date`); times→clinic `appointment_time`; addresses→clinic
  `home_address`; properties→hotel `destination_property`; amounts→auto
  `approved_budget`; emails→hotel `contact_email` (instantiated against
  the callee); medications→clinic `current_medication`;
  insurance_plans→clinic `insurance_plan_name`; vehicles→auto
  `vehicle_model`; rate_plans→hotel `rate_plan_name`; coined by role
  (employer→clinic, company→hotel, brand→auto); shops→auto
  `previous_service_shop`.
- **Filler scheme** (Phase C, 2026-08-24): fixed in-code org catalogs
  per vertical (six coined names each, asserted fold-distinct from
  every bank value); callee drawn from the easy person-name pool
  (re-drawn if it folds onto a measured name); one filled context cell
  per record from an ordered per-vertical candidate list (contact
  phone, then a vertical-specific date/address) drawn from easy pools;
  `submitted_on` in 2025-06-02..11 under the pinned clock. All filler
  comes from per-task `intake-freeze|filler|{task_id}|{seed}` streams,
  cell draws from `intake-freeze|cell|{bank}|{tier}|{seed}`, so filler
  changes can never perturb a measured draw. Canonical seed: 20260824.
  Record ids REC-1001… sequential; task ids
  `intake_{bank}_{tier}_{01..10}`.
- **Relative dates through the callee** (Phase C, 2026-08-24; dual-form
  2026-08-26, frame 4.5.0): for relative-date draws the callee's
  `get_entity` returns the DUAL-FORM payload — the spoken form plus the
  real written date it refers to ("the fourth Monday of next month (the
  real date this refers to, from your records: 2025-07-28)",
  `render_dual_form` in `utils.py`) — while the golden submission and
  seeded record hold the ISO resolution. The sim leads with the spoken
  form (one fixed frame sentence tells it how to use the pair) and gives
  the written date when the caller asks for the exact date; the agent can
  still resolve against `get_today` instead of asking. Spoken-only
  payloads made the 2026-08-26 smoke sim spell the PHRASE letter by
  letter ("T, H, E, space, F, O, U, R, T, H, ..."), which no real caller
  does. Complications draw against the spoken part only (`spoken_form`
  strips the annotation), so the written year can never flip a
  feasibility gate. Every other family's entity equals the written
  value.
- **Multi-entity bands** (Phase C, 2026-08-24): `n_entities` in {2,3,5}
  bundles n same-vertical, distinct-field bank units per call, 10 calls
  per vertical × tier cell, banks picked per call by remaining pool
  capacity (deterministic). Bands are written only to an explicit
  output directory — the canonical `tasks.json` stays atomic-only. A
  bundle that measures every context candidate simply carries no filler
  cell.
- **Call frame 4.0.1** (Phase C, 2026-08-24): the ASCII-only directive
  extends to the whole serialized task (a generation gate), so the
  three em-dashes in the 4.0.0 user-scenario templates were replaced
  with ASCII punctuation and the call-frame version bumped.
- **Known-info disclosure split** (Phase D, 2026-08-24, owner call;
  call frame 4.1.0, generator 6.1.0): the all-fields-through-get_entity
  frame proved intrinsically leaky — across three grounding hardenings
  (4.0.1→4.0.3) and three user-sim reasoning efforts the sim kept
  inventing plausible values for fields a real person would know by
  heart, because fetching your own guest's name from a tool is
  unnatural. The generator now carries a fixed disclosure catalog:
  KNOWN banks (person_names, phones, emails, addresses, vehicles,
  shops, coined, properties) render their pinned value VERBATIM into
  the scenario's known-information block and the sim answers character
  for character from that text; FETCHED banks (times, amounts, dates,
  codes, medications, insurance_plans, rate_plans) never appear in the
  scenario and reach the sim only through `get_entity`. The unknown-info
  clause now scopes the will-be-wrong warning to unlisted fields.
  Reward semantics are unchanged — the split moves where the sim's
  truth lives, not what the agent must capture.
- **Tool discipline + named look-up fields; split retired** (Phase D,
  2026-08-24; call frame 4.3.0, generator 6.3.0): wave-v5 attribution
  found one residual sim defect — a `get_entity` call with an
  improvised field-name string, followed by the sim asking the AGENT
  aloud about internal field names. Telecom's proven user-sim design
  shows why it never fumbles: its per-question tool NAMES
  (`get_status_bar`, `can_send_mms`) carry the grounding map, so the
  sim never invents an argument. That reframed the diagnosis: the
  v2–v4 grounding failures were field-name ambiguity, not the
  unnaturalness of fetching known facts — so the known-info disclosure
  split (4.1.0, one wave old) is RETIRED (owner call, 2026-08-24) and
  ALL fields return to tool-only grounding, now with the missing
  mapping: the scenario's unknown-info block lists the task's FIELD
  NAMES exactly as `get_entity` expects them (names only, never
  values), and the task instructions declare the tool private — never
  mention the tool, the lookup, or field names to the caller; on an
  error listing the record's fields, silently retry with the listed
  name. Wave v6 is the hypothesis test.
