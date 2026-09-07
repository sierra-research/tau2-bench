# Entity composition experiments — compounding, interference, hierarchy

**Status:** design v1, 2026-08-28 — awaiting owner review.
**Owner:** soham. **Scope:** two small formative experiments on the v4
intake (τ-Elicitation) domain. English, voice-first, one provider. Additive
draws only — the canonical 200-task freeze is untouched.

## 1. Why

The τ-Elicitation paper (PR #888) evaluates atomic tasks only and names the
gap explicitly: *"The main benchmark is atomic and does not test
cross-entity interference."* The generator already carries the
`n_entities` knob (1|2|3) as a deferred seam (design doc
`intake-domain.md` §1, §11).

Prior art: the **v3** study (generator 5.2.0, inbound flow, artifacts
`c49269fbad41` / `687fe924d2e0`) found multi-entity failure roughly
multiplicative — per-slot hazards flat in count for hard tiers, mildly
rising for easy — but that evidence is (a) from the discarded inbound
shell, (b) unpaired (each band drew fresh values), and (c) confounded
between e1 and multi-entity bands by flow type (v3 design §8.2). The v4
design deliberately deferred re-establishing it. These experiments
re-test it on the v4 outbound shell with a **paired** design, and add a
genuinely new axis: **hierarchical gating**, where entity k+1 is only
reachable after entity k verifies — the authentication structure the
paper's introduction motivates but the benchmark does not yet contain.

Questions:

- **Q1 (compounding):** at n>1, is task success the product of the
  constituent entities' atomic success rates?
- **Q2 (interference):** does bundling change *per-entity* capture —
  same value, same condition, worse (or better) capture when other
  entities share the call? Includes cross-field contamination (value B
  written into field A) and slot-position effects.
- **Q3 (hierarchy):** when entity 2 is locked behind a verified entity
  1, how do task outcomes change against the flat bundle of the same
  values — how much failure converts into *blocking*, do agents recover
  via retry, and is downstream capture among gate-survivors still at the
  atomic rate?

## 2. Experiment E-COMP — paired compositional bundles

### 2.1 Draw

A new **paired compose draw** (additive, never touches the freeze):
bundles are built from the *exact values of the canonical 200 tasks*,
so every bundled entity has a known atomic outcome under the same
condition. This is the difference from the existing `draw --n-entities`
band, which draws fresh values and only supports rate-level comparison.

- **n=2 band:** 3 verticals × 2 tiers × 10 calls = **60 calls**
  (120 slots, 120 distinct canonical values). Tier-pure (easy+easy,
  hard+hard) so cells align with the canonical bank×tier cells.
- **n=3 band:** 3 verticals × 2 tiers × 5 calls = **30 calls**
  (90 slots). Values may be reused from the n=2 band — reuse is a
  feature: the same value observed at n=1 (canonical), n=2, and n=3
  gives a within-value count gradient.
- **Position balance:** within each vertical×tier cell, each bank
  appears in slot 1 and slot 2 (and 3) equally, by deterministic
  rotation. Slot order = callback-order position = record field order.
- **Constraints inherited:** same-vertical bundling, fold-distinctness
  within task, ASCII, replay gates. A candidate pair whose values fold
  together is rejected and redrawn deterministically.
- **Provenance:** the compose manifest records, per slot, the parent
  canonical task id, bank, tier, value, and slot index. All analysis
  keys off this manifest — no transcript archaeology.

### 2.2 Run

- Voice: `gpt-realtime-2`, the standard intake caller sim, **regular
  condition, seed 42** — one condition; the robustness cross is out of
  scope for a formative wave.
- **Complications off (clean).** The atomic reference outcomes come
  from the seed-42 regular-condition canonical run; pairing is only
  clean-vs-clean. Values whose atomic reference call was
  realism-triggered are still drawable but are excluded from the paired
  test set (they still count for the product-null and contamination
  analyses). The compose draw prefers clean-reference values when
  filling cells.
- Text control: same 90 tasks in text mode (cheap; expected near
  ceiling — separates protocol/logic load from speech).
- Duration cap: keep `DEFAULT_MAX_STEPS_SECONDS` (1200 s simulated);
  record per-call duration so cap pressure at n=3 is visible before
  anyone interprets pass rates.

Voice total: **90 calls.**

**Execution note (2026-09-07).** The frozen voice roots repeat the 60 n=2 and
30 n=3 tasks. The reported analysis pools trials 0--2 in each arm (180 and 90
executed calls, respectively). The n=3 root also contains a concurrently
launched fourth trial; it is excluded only to keep the arms balanced. Analysis
uses the compose manifest at the runs' recorded commit (`ce74dff8`), before a
later complication-catalog redraw changed one slot.

### 2.3 Analysis (one verb, facts first)

`tau2 metrics composition` reads the run + compose manifest and emits a
facts JSON plus a rendered md (the `interaction-facts` /
`robustness` precedent). Reported:

1. **Product null (Q1):** per cell and pooled, observed task pass vs
   ∏ of clean atomic bank×tier rates, task-level bootstrap CI on the
   difference. Secondary: predicted-by-realized-outcomes (task passes
   iff all constituents passed atomically) as a paired binary
   comparison.
2. **Per-slot parity (Q2):** each bundled slot vs the same value's
   atomic outcome — McNemar on discordant pairs, pooled and by bank.
   ~120 pairs at n=2 detects a per-slot drop of roughly 10 points; this
   wave is formative, not confirmatory, and the doc says so.
3. **Per-slot outcome table:** unconditional per-entity capture rate
   (the flat x% that E-HIER's y% is compared against) and the joint
   2×2 per call — both pass / first-pass-second-fail /
   first-fail-second-pass / both fail — plus within-call outcome
   correlation (failures clustering in bad *calls* vs bad *entities*).
   Calls that end with no submission count as all slots failed,
   reported as their own row.
4. **Slot-position effect:** slot-1 vs slot-2(3) per-entity pass,
   within-call paired.
5. **Cross-field contamination:** deterministic — final DB field A
   equals `fold(value B)` for a co-bundled B. The signature interference
   failure; reported as a count with task ids.
6. **Count gradient:** per-slot failure at n=1 → 2 → 3, overall and for
   the within-value reuse subset.

## 3. Experiment E-HIER — hierarchical gating

E-HIER is the **gated twin of E-COMP**: the identical task set — same
bundles, same values, same slot order (60 n=2 + 30 n=3 calls) — with
one change: the slots form a chain, and slot k+1 only becomes
reachable once slot k **clears**. At n=3 the chain has two gates.
Every flat-vs-gated comparison is therefore paired on identical tasks.

### 3.1 Mechanism (new, environment-side only)

A **staged callback order**: the order carries its missing fields as a
chain; `get_callback_order` reveals only the lowest uncleared field.
Every slot is submitted alone via `submit_fields` and must clear:

- The environment fold-compares the submitted value against the seeded
  truth. On match, the field is written and the next slot is revealed;
  on mismatch, the tool returns a generic verification failure ("the
  value did not verify against the record") — the field stays missing
  and the agent may re-elicit and retry until the duration cap.
- The rule is uniform: every slot, including the last, is a verified
  write. Per-slot clearance is the measured outcome; final-DB exact
  match remains the task score.

The verify oracle is deliberate and scoped: it is the definition of
the gated workflow (each field validates before the process moves on,
as authentication does), and it leaks only a pass/fail bit per
attempt. The retry channel is part of what is being measured, not a
confound — flat E-COMP measures one-shot capture, E-HIER measures
verified-workflow capture. The user simulator is unchanged — all
entities sit in `set_entities`; gating is entirely environment-side.
The staged protocol gets one new fixed paragraph in the policy
(versioned; the flat "one submit writes everything" rule is explicitly
scoped to non-staged orders).

### 3.2 Cells

Identical to E-COMP: 3 verticals x 2 tiers, tier-pure; n=2 (10
calls/cell, 60 total) + n=3 (5 calls/cell, 30 total). Voice total:
**90 calls** (+ 90-task text control).

### 3.3 Analysis (same verb, gated section)

**The headline is a per-entity rate drop.** E-COMP measures the
unconditional per-entity capture rate in flat bundles (x%); E-HIER
measures the same rate for the same values in the chain (y%). The
claim: an entity that lands x% of the time in a flat call lands only
y% inside a gated chain — same values, same call shell, so x - y is
the cost of workflow structure alone. Reported pooled, per slot
position (the drop should deepen down the chain), and per bank.

Supporting decomposition:

1. **Why y fell:** per-slot failures split into *blocked* (an earlier
   slot never cleared, this one never reachable) vs *reached but never
   cleared* vs *cap terminal*. If reached slots clear at ~x%, gating
   is pure upstream blocking; if reached slots also degrade, the gate
   ceremony hurts capture itself.
2. **Slot-1 clearance vs flat slot-1:** identical position and values,
   so this isolates the verify+retry channel alone (retry can push
   clearance *above* the flat one-shot rate — report
   attempts-to-clear and time burned per slot).
3. **Chain survival:** P(slot k reachable) and
   P(clear | reachable) down the chain; n=3 gives the two-gate read.
4. **Task level:** gated vs flat task pass, paired on identical
   bundles.

## 4. What is out of scope

Condition crossing / Pass^k_robust for bundles, provider comparison,
mixed-vertical bundles, complication × composition interactions, n=5,
chains deeper than two gates, and any change to the
canonical freeze or the paper's prespecified atomic analyses. Each is a
follow-on draw if the formative read warrants it.

## 5. Build phases (phase-PRs into `intake`)

1. **C1 — compose draw.** `tau2 intake-tasks compose --n-entities {2,3}
   --from <canonical task dir+manifest> --out <dir>`: paired bundling,
   position rotation, fold gates, parent-id
   provenance. Freeze regression untouched; new unit tests replay the
   draw byte-identically.
2. **C2 — staged orders.** Data model (`CallbackOrder` stages + unlock
   state), `submit_fields` staged behavior + generic verify-fail
   message, policy paragraph, call-frame note, generator `--hierarchical`
   band (the compose draw re-emitted as a staged chain), replay gates walk the golden staged path.
3. **C3 — analysis verb + runs.** `tau2 metrics composition` (facts
   JSON + md), run presets for the four runs (E-COMP voice/text,
   E-HIER voice/text), analysis artifacts into `data/analysis/`.

Known dependency: text controls want the #969 end_call bundled-turn fix
merged first (bundled end_call turns drop calls and score 0 in text
mode).

## 6. Open owner decisions

1. Gate verify-failure wording — generic pass/fail proposed; any
   objection to the bounded oracle bit?
2. Complications off for both experiments (proposed) vs default profile.
3. Cap: keep 1200 s for gated calls (retries burn time — measured, not
   pre-tuned) vs a per-band override.
