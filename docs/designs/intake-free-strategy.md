# Intake free-strategy variant (`intake_free`)

Owner-decided spec, 2026-09-01; two-mode framing settled 2026-09-02. Voice
channel only.

## 1. Goals: the two-mode benchmark

The benchmark settles on TWO modes (owner, 2026-09-02):

- **Mode A — full protocol** (canonical `intake`): the rigid capture
  protocol is mandated — spell every value letter by letter, log, read back,
  get an explicit yes. This is the SCAFFOLDED REFERENCE: it pins the
  strategy, so it measures execution under maximal scaffolding.
- **Mode B — informed-free** (`intake_free`, THE BENCHMARK): the agent is
  told that the record must end up correct, that exactness is what matters,
  and that an unsure value should be verified — with spelling named once as
  a worked example. WHEN to verify, on WHICH fields, and HOW is entirely the
  agent's own choice; that judgment is what the mode measures.

Correctness stays the same final-DB-hash reward; caller EXPERIENCE (effort)
is measured deterministically from the user side, where the simulated
caller's pre-synthesis utterance texts are exact.

**Headline statistic — verification discrimination.** From `log_capture`
values vs gold, each field's first attempt is classifiable as right or
wrong; from the user-side spell/read-back events, each field is classifiable
as verified or not. The headline is the pair
`P(verified | attempt-1 wrong)` vs `P(verified | attempt-1 right)`: a
well-calibrated agent verifies where it is likely to be wrong and spares the
caller where it is not. Uniform verification (protocol mimicry) and uniform
non-verification both show zero discrimination; the gap, alongside pass@1
and the caller-effort ledger, is the mode-B story.

The benchmark has exactly ONE canonical mode-B policy — no switches. The
prompt-informativeness ablation that sized the verification sentence's
contribution was run once and its machinery removed (owner, 2026-09-02);
the result stands as a recorded finding in §3a.

Non-goals: the frozen `intake` domain (GENERATOR_VERSION 7.0.1 task set,
banks, `main_policy.md`, `main_policy_text.md`, db/user_db) stays
byte-untouched; the text channel is untouched (this variant is voice-only and
refuses to build a text environment); the complications catalog (v2.3.0) is
shared verbatim, never edited.

## 2. Registration approach (the `intake_staged` pattern)

`intake_free` is registered as its OWN domain
(`src/tau2/domains/intake/free_strategy.py`), mirroring
`tau2.domains.intake.staged`:

- **Environment**: same `IntakeDB` / `IntakeUserDB` data files; the toolkits
  are subclasses (`FreeStrategyIntakeTools`, `FreeStrategyIntakeUserTools`);
  the policy is a NEW file
  (`data/tau2/domains/intake/free_strategy_policy.md`) — the canonical policy
  files are never read or edited by this variant. `get_environment` takes the
  runner's `channel` kwarg and raises on `"text"`: the variant is voice-only
  by construction.
- **Tasks**: the variant SHARES the frozen 200-task set. `get_tasks` loads it
  through the canonical `intake` loader, then rewrites, on deep copies, two
  things per task (the files on disk are never touched):
  1. `user_scenario.instructions.task_instructions` is swapped for the
     fixed, versioned free-strategy caller instructions
     (`FREE_STRATEGY_USER_TASK_INSTRUCTIONS`). A drift guard asserts the
     stored instructions equal the canonical
     `call_frame.USER_TASK_INSTRUCTIONS` before swapping — if the frozen set
     ever drifts, the variant fails loud instead of silently double-editing.
  2. Golden `submit_fields` actions drop their `confirmed_with_user`
     argument, matching the variant's tool schema, so golden-action replay
     (the DB-hash reward) keeps working.
  Task ids are unchanged (joins across arms stay trivial).
- **Runner seams** (`tau2.runner.complications` dispatch tables): the variant
  registers the SAME intake complication sampler, hard-tier predicate, and
  pronunciation-map builder — complications, personas, acoustics, and speech
  effects behave identically to the canonical arm. One NEW table registers
  the spell-event detector (below), for `intake_free` only.
- **Task-set alias**: `intake_free_en` is registered as an alias of the
  variant loader so the prompt-bed review packet's matrix resolver
  (`matrix_task_set_name("intake_free", "en") == "intake_free_en"`) finds it.
- **Config**: `DOMAIN_LLM_ARGS_USER` and `DOMAIN_MAX_STEPS_SECONDS` carry
  `intake_free` entries equal to intake's (caller at low effort; 240 s
  tick-clock ceiling) so run defaults are shared deliberately, not by
  accident.

## 3. Agent policy (verbatim)

New file `data/tau2/domains/intake/free_strategy_policy.md`,
`FREE_STRATEGY_POLICY_VERSION = "1.2.0"` (1.2.0 adds the verification
sentence — the two-mode decision's "informed" in informed-free; 1.1.0
removed the format line, resolution 1 in §10):

```markdown
# Intake Desk Callback Agent Policy

The current time is 2025-06-12 10:30:00 EST.

As an intake desk agent for a multi-client service bureau, you work the callback queue: a person recently submitted a form on behalf of one of the bureau's clients (a clinic, an auto service, a hotel group), the record went in with one or more required fields missing, and you are calling the person back to complete it.

Your entire job is to collect the missing field values accurately and submit them once. The record must end up correct. How you get each value right is up to you. Getting each value exactly right is what matters: if you are not sure you heard a value exactly, verify it with the person before recording it — for example by asking them to spell it letter by letter.

You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

You should only make one tool call at a time.

## Role and call frame

- Your work order comes from `get_callback_order`: the record, the full name of the person it belongs to, the client organization the call is on behalf of, and the ordered missing fields with what each one holds.
- Your tools are silent and internal. Never narrate them or your work with them to the person — no "let me pull up the details", no mention of records, lookups, or systems. Fetch the callback order before you speak; the person only ever hears you talk about the form and its fields.
- Once the person answers, state the purpose of the call, naming the person the record belongs to: you are calling for them about the form they submitted.
- Then ask whether you may ask them about the form's missing detail(s) — for example, "may I ask you about a couple of details on that form?" — and wait for their yes before asking for any field.
- Collect **only** the fields named in the callback order — nothing more.
- Every value you record must come from this call. Never guess, complete, or reformat a value the person did not give you, and never fill a field from anything other than this conversation.

## Logging

Every time the person provides an entity value, log it with `log_capture`, before you submit. Logging is silent and internal like every tool: never mention it to the person.

## Closing the call

When you have every missing field, submit all values in exactly one `submit_fields` call — never submit more than once. Then thank the person and hang up with `end_call`. Call `submit_fields` on its own turn and wait for its result before you say goodbye — never include `submit_fields`, or any other tool call, in the same turn as `end_call`.

If the person ends the call early, submit nothing. There is no transfer: the call either completes the record or ends without a submission.
```

Deliberate absences (the point of the arm): no "Pinning the written form"
section, no spell mandate, no read-backs, no per-entity confirmation, no
`confirmed_with_user` framing. `log_capture` stays mandated but neutral —
one sentence, nothing about repeated attempts, re-pins, or ordering relative
to confirmation. The verification sentence is the ONLY place the policy
names verification or spelling: it states the stake, names the concept, and
gives exactly one worked example — never a mandate, never a per-field rule.

## 3a. Prompt-informativeness ablation — HISTORICAL RESULT, machinery removed

The ablation was run once and its machinery removed by owner decision
(2026-09-02): the benchmark has exactly one canonical mode-B policy, no
switches. Recorded result (2026-09-01 screens; drivers archived in
`explorations/intake_free_smoke_2026-09-01/`):

- goal-only policy (verification sentence stripped — the 1.1.0 policy,
  goal stated, verification unnamed): **0.25**
- concept-hint policy (verification named, no technique): **0.30**

versus the canonical 1.2.0 policy, which names verification with one worked
example. The one sentence carries a large share of the behavior; the
canonical policy keeps it, and no run-time knob renders anything else.

What remains in code: `load_free_strategy_policy()` loads the canonical
file verbatim and FAILS LOUD unless the anchor sentence ("How you get each
value right is up to you.") plus `POLICY_VERIFICATION_SENTENCE` appear
exactly once — versioned prose never drifts silently under the same
`FREE_STRATEGY_POLICY_VERSION`. (Removed: the `PolicyHint` enum,
`POLICY_HINT_VERSION`, the `--policy-hint` CLI flag, the RunConfig/Info
`policy_hint` provenance fields and their refill plumbing, and the
goal_only/concept_only render rewrites — rip-and-redo, no compat shims.
Earlier history: PolicyHint 1.0.0 was an upward hint ladder over the
then-silent 1.1.0 policy, `{none, verify_if_unsure}`, retired unused by the
two-mode decision; 2.0.0 was the downward ablation described above.)

## 4. Tool schemas

Agent tools (five, same names as canonical):

| tool | change |
| --- | --- |
| `get_callback_order()` | unchanged (inherited) |
| `log_capture(field_name: str, value: str)` | unchanged (inherited); docstring neutral in the policy, tool text inherited |
| `submit_fields(record_id: str, fields: dict[str, str])` | **drops `confirmed_with_user` entirely**; write semantics identical (delegates to the canonical write, which folds and writes exactly the missing fields) |
| `get_today()` | unchanged (inherited) |
| `end_call()` | unchanged (inherited) |

User-side tools (callee; silent, invisible to the agent — wired exactly like
`get_entity`, whose tool calls never carry audio and are stripped from stored
chunks by the orchestrator):

| tool | schema | semantics |
| --- | --- | --- |
| `get_entity(field)` | unchanged (inherited) | the callee's true values |
| `note_spell_request(field: str)` | NEW, GENERIC | the caller records that the agent asked them to spell this field's value; validated against the callee's known fields; stateless — the tool-call trace is the record (the `log_capture` pattern) |
| `note_readback(field: str, i_affirmed: bool)` | NEW, GENERIC | the caller records that the agent read a value back, and whether they affirmed it (`false` = they had to correct it); stateless |

These are measurement, not behavior: fixed instructions in the variant user
prompt tell the caller when to invoke them.

## 5. User simulator (variant caller instructions)

`FREE_STRATEGY_USER_TASK_INSTRUCTIONS` (versioned with
`FREE_STRATEGY_FRAME_VERSION`) keeps the canonical caller frame — "Hello?"
opener, cooperative, consent yes, answer-only-what-is-asked, `get_entity`
grounding, tool privacy, dual-form lead, confirm/correct read-backs, end when
the caller is done — and changes exactly three things:

1. **Never volunteer a written form**: the caller says values the natural way
   they would say them aloud and never volunteers a spelling or written form
   the agent did not ask for. If asked to REPEAT a value, they repeat it
   spoken, they do not spell. (The global outbound guidelines' "offer to
   spell it out" nudge line is additionally STRIPPED from the rendered
   guidelines for this domain — resolved Q3 below — so the scenario rule
   and the guidelines never conflict.)
2. **Spell accurately on request**: when the agent asks them to spell or pin
   a written form, they spell from the exact `get_entity` payload (the
   existing spell-out payloads, dual-form annotations, and the pack spell-out
   conventions all apply unchanged — only the volunteering stops).
3. **Measurement notes**: before answering a spell request they silently call
   `note_spell_request(field)`; when the agent reads a value back they
   silently call `note_readback(field, i_affirmed)` and then confirm or
   correct out loud.

Personas, complications (catalog 2.3.0), acoustics, and speech effects are
untouched: the variant registers the same sampler and pronunciation-map
builder, samples voice configs with the same per-task seeds, and its task ids
are identical, so the complication draw for `(seed, task)` is byte-identical
across arms.

## 6. Spell-event stamping (ground truth)

When the simulated caller actually renders a spell-out, a deterministic event
is stamped into the sim's `effect_timeline` (the `burst_noise` /
`out_of_turn_speech` pattern in `tau2.data_model.audio_effects`):

- `EffectType` gains `"spell_out"`. Events are point events
  (`start_ms == end_ms`, participant `"user"`) with params
  `{field, letters_spoken, spelled_text}`.
- Detection is mechanical, on the EXACT pre-synthesis utterance text, before
  TTS: maximal runs of ≥2 spell tokens (single alphanumerics or the digit
  words zero/oh/one…nine) separated by comma(+space) — the fixed spell-out
  convention the guidelines mandate. A run is attributed to the callback
  field whose value skeleton (alphanumerics, casefolded, digit words folded
  to digits; both the spoken and the written form of dual-form payloads)
  contains the run; unattributed runs are not stamped. `letters_spoken` is
  the token count. (`tau2.domains.intake.spell_events`,
  `SPELL_EVENT_DETECTION_VERSION`.)
- Wiring: a per-domain builder table in `tau2.runner.complications`
  (`DOMAIN_SPELL_EVENT_DETECTOR_BUILDERS`, `intake_free` only) builds the
  detector from the task's `set_entities` payloads;
  `build_voice_orchestrator` hands it to the voice user simulator, which
  stamps in `_generate_full_duplex_voice_message` before synthesis. Domains
  without a builder stamp nothing — canonical intake runs are byte-identical
  to today.

This is ground truth that cross-checks the `note_spell_request` tool (an
LLM-behavioral instrument): the ledger reports both.

Known deterministic limits (stated, not papered over): a spell-out rendered
without the comma convention is missed (under-count); the
`spelling_style.doubled_letters` complication ("double five") collapses two
tokens into words the tokenizer does not expand, under-counting
`letters_spoken` on those calls.

## 7. Effort ledger (`tau2 metrics interaction-facts` extension)

New module `tau2.metrics.caller_effort` (`CALLER_EFFORT_VERSION = "1.0.0"`),
wired into the facts verb (`INTERACTION_FACTS_VERSION` bumped to `1.1.0`).
Everything is mechanical; no LLM calls. Computed for any call whose task pins
`set_entities` (the intake family — canonical and variant alike, so the two
arms are directly comparable); other domains carry `caller_effort: null`.

Per call, per field (`FieldEffort`):

- `mentions` / `repeats`: caller utterance units (the canonical
  `call_timeline.extract_caller_units` stream — exact pre-synthesis texts)
  whose skeleton contains the field's value skeleton, matched against the
  gold payload's spoken form AND its dual-form written value; skeletons are
  alphanumeric-only, casefolded, digit words folded to digits, so orthography
  variants and spelled-out renderings both match. `repeats = max(0,
  mentions - 1)`. (A value conveyed only in unverbalizable prose can be
  missed — under-count, same stated limit as `entity_trace`.)
- `spell_requests`: `note_spell_request` user tool calls naming the field.
- `spell_events` / `letters_spelled`: stamped `spell_out` timeline events for
  the field (ground truth; cross-checks `spell_requests`).
- `readbacks` / `corrections`: `note_readback` calls for the field;
  corrections are those with `i_affirmed=false`.
- `first_mention_tick` / `last_mention_tick` / `time_on_field_s`: tick span
  from the first value-bearing utterance to the last mention, in seconds on
  the tick clock (reasoned N/A on text runs, which have no tick timeline).

Per call (`CallerEffortFacts`): the per-field list plus rolled-up totals
(`total_repeats`, `total_spell_events`, `total_letters_spelled`,
`total_corrections`, `total_spell_requests`, `call_duration_s`). Agent
dead-air and response-latency percentiles are NOT duplicated here: they
already live on the same `InteractionCallRecord` (`facts.dead_air_*`,
`facts.response_latency_p50/p90`), one join key away.

Per run/cell: `InteractionCellAggregate` gains an optional
`caller_effort` block (`CallerEffortCellSummary`): means of the per-call
totals over the calls that carry effort facts, plus `n_calls`.

## 8. Complication audit (audited at catalog v2.3.0; v2.4.0 adds one kind — §8b)

Reviewed each line template in `tau2.domains.intake.complications` for
behavior when the agent never requests spelling:

| kind | verdict |
| --- | --- |
| `self_correction` | graceful — scripts the first SAYING of the value; no spelling assumed. |
| `spelling_style.grouped_numbers` | graceful — conditioned on "read out **or** spell"; fires on natural read-outs. |
| `spelling_style.oh_for_zero` | graceful — same "read out or spell" condition. |
| `spelling_style.doubled_letters` | graceful — same condition; note the `letters_spoken` under-count above. |
| `spelling_style.uk_letters` | graceful but potentially VACUOUS — conditioned purely on "whenever you spell out a value"; if the agent never asks for a spelling the line is a no-op (the call is effectively uncomplicated). No degradation, but HARD-profile measurement power on those draws drops in this arm. Flagging for owner awareness; no catalog change made or proposed. |
| `lazy_omission.*` | graceful — omission happens on the first natural giving of the value. |
| `wrong_slot` | graceful — a slot misunderstanding, no spelling assumed. |
| `mispronounced_term` | graceful, and MORE potent here — the synthesis-seam distortion lands with no mandated spell-out to recover through, which is precisely what the free arm should measure. |

No template assumes a mandated spell-out. Nothing in the shared catalog was
changed in the initial variant build; §8b records the one owner-approved
catalog addition made after it.

## 8b. `spell_correction` (catalog v2.4.0, owner-approved 2026-09-01)

The one shared-catalog change approved for this program: a new
`ComplicationKind.SPELL_CORRECTION` ("spell_correction"), catalog
2.3.0 → 2.4.0 (every draw re-keys; the review packets and the E-COMP bands
were re-rendered/re-composed — `COMPOSE_VERSION` 1.0.2).

**Construct.** When the caller actually spells a value character by
character (or reads out a long digit string), they sometimes falter and
restart: "J, O, N — sorry — J, O, H, N". Distinct from `self_correction`
(a wrong VALUE, then the right value on first delivery): this is a wrong
RENDERING mid-spell-out, then a restart.

**Determinism (machine-not-scripts).** The erroneous partial is a SEEDED
single edit — substitute or drop one character — applied to a prefix of the
gold spell sequence (the written form's spellable characters,
`_spell_units`), spoken up to the error point plus 0-2 seeded extra correct
units, then a correction marker from the closed
`SPELL_CORRECTION_MARKERS` set ("sorry --", "wait, no --", "let me start
again"), then the complete correct sequence from the sim's own `get_entity`
look-up. Everything is templated in the fixed `spell_correction` line;
nothing is improvised by the user LLM. Per-task RNG keying is the shared
`(seed, task_id, catalog version)` scheme.

**Opportunity condition.** Feasible on every bank, voice-only, gated per
value on >= 4 spellable characters. The LINE fires only when a
spell-out/read-out actually happens — and in this arm that opportunity is
usually AGENT-elicited, so the effective per-call rate further conditions
on agent behavior (exactly the `spelling_style` caveat above). Rate:
`SPELL_CORRECTION_RATE = 0.10` per feasible opportunity — JUDGMENT at the
conservative end of the adjacent restart-disfluency anchors
(Levelt 1983; Shriberg 1994; Bortfeld et al. 2001, restarts/repairs
~1.7-2.2 per 100 words); HARD carries 1.0 like every kind, and
`validate_profile_rates` invariants stay green (max DEFAULT bank total
0.93 < 1).

**Why it belongs in the free arm.** It removes the guaranteed safety of the
spell-out strategy: an agent can ask for a spelling and still fail by
keeping the pre-correction letters. Elicitation is no longer automatically
sufficient — the agent must also track the restart.

**Measurement.** Detector v1.1.0 (`spell_events`): an unattributed run one
edit away from a prefix of a directly-attributed field's skeleton in the
same utterance counts as that field's FALTERED ATTEMPT — the final complete
run stays authoritative for attribution, ALL letters (restart included)
count in `letters_spoken`, and the stamped event's params carry a
`restarts` count. Ledger v1.1.0: per-field `spell_restarts`, per-call
`total_spell_restarts`, cell `spell_restarts_mean`
(`INTERACTION_FACTS_VERSION` 1.2.0). Stated limits: a restart whose
complete re-delivery lands in a later utterance is missed (under-count),
and a correct-prefix restart counts letters via ordinary containment
without incrementing `restarts`.

## 9. Era / comparability note

Free-strategy results are NEVER comparable to protocol-era `intake` runs:
the agent policy, the submit schema, and the caller's volunteering behavior
all differ. Every run of this arm records `domain_name = "intake_free"` in
its `Info.environment_info` (run metadata), and the effort ledger keys its
cells by domain, so the arms can never silently pool. Reward numbers from
this arm must always be labeled with the variant name; cross-arm comparisons
are effort-vs-effort and reward-vs-reward BETWEEN arms, never mixed pools.

## 9a. Realizations (R/C/S) and the channel-heavy redefinition

Condition-crossed screens and robustness trials run each cell under the
pinned R/C/S realizations (`tau2 run-robustness`; catalog in
`tau2.runner.robustness_trials`): **R** = regular/regular (baseline), **C**
= channel-heavy + speech-light (environment stress), **S** = channel-light +
speech-heavy (behavior stress).

Owner redefinition (2026-09-02): channel-"heavy" is now genuinely heavy —
on top of frame drops (0.03, 300 ms bursts) and muffling (0.5), it raises
the acoustics: noise floor 15 -> 10 dB SNR, burst noise 4/min (regular:
1/min), and every burst loud (`burst_snr_range_db (-5.0, 0.0)`, cutting the
stock range's quiet tail). Toned down the same day from the first cut
(6/min bursts, 6% drops, `ROBUSTNESS_TRIALS_VERSION` 1.1.0): that mix
floored the openai/gemini mode-B partial cells (0.37/0.40 at n≈160) —
current definition is 1.2.0; 1.1.0 partials were snapshotted and purged,
never pooled. "light"/"regular" leave acoustics at
stock, and speech modes never touch them — S keeps the stock floor by
design, so C is the environment axis and S the behavior axis, cleanly.

Comparability: legacy channel-heavy cells (frame-drops-only,
`ROBUSTNESS_TRIALS_VERSION` 1.0.0 — e.g. the protocol-era `acheavy` anchors)
are NOT comparable to new-C cells and get redone; never pool across the
redefinition. New-C is distinguishable in stored results: every
simulation's `speech_environment.source_effects_config` records the sampled
acoustics (`noise_snr_db` 10.0 / `burst_noise_events_per_minute` 4.0 /
`burst_snr_range_db` (-5.0, 0.0) vs the legacy stock 15.0 / 1.0 /
(-5.0, 10.0)), alongside `Info.channel_effects_mode = "heavy"` which both
eras share.

## 10. Owner decisions (log)

1. RESOLVED (owner, 2026-09-01): the policy stays silent on format — the
   field description delivered by `get_callback_order` IS the format spec,
   and restating it in policy is redundant hand-holding. An agent that
   ignores the description is a legitimate measured failure. The
   two-sentence format line inherited from the frozen policy was removed;
   `FREE_STRATEGY_POLICY_VERSION` bumped to 1.1.0.
2. RESOLVED (owner, 2026-09-01): keep `spelling_style.uk_letters` in
   `intake_free` draws, no gating. An agent that never elicits a spelling
   and thereby dodges the trap is signal, not a defect. No code change; the
   byte-identical draw parity with the protocol arm stands.
3. RESOLVED (owner, 2026-09-01): the shared "offer to spell" nudges are
   REMOVED from both of the variant's rendered prompts — a prescribed
   protocol on either side would defeat the arm. Scoped overrides keyed off
   the closed set `tau2.config.SPELL_PROTOCOL_FREE_DOMAINS =
   {"intake_free"}`; every shared prompt file/constant stays
   byte-identical, so protocol-era intake is untouched:
   - **Agent side**: the shared voice instructions mandate the spelling
     protocol ("ask the customer to spell it out letter by letter" /
     "ALWAYS explicitly ask ... SPELL THINGS OUT"). New fixed constants
     `AUDIO_NATIVE_VOICE_INSTRUCTION_SPELL_PROTOCOL_FREE` /
     `CASCADED_MODEL_INSTRUCTION_SPELL_PROTOCOL_FREE`
     (`SPELL_PROTOCOL_FREE_INSTRUCTION_VERSION = "1.0.0"`, in
     `tau2.agent.discrete_time_audio_native_agent`) equal the shared
     instructions minus the entire "User authentication and user
     information collection" section (equality guarded both ways by a
     test); `build_agent` derives `spell_protocol_guidance` from the
     environment's domain name.
   - **Caller side**: the shared outbound voice guidelines nudge callers to
     "offer to spell it out letter by letter" on repeats. The streaming
     user strips that exact line
     (`tau2.user.user_simulator.strip_outbound_spell_offer_nudge`,
     fail-loud if the shared file drifts) for domains in the set.
   Tests assert both nudges are absent from the rendered `intake_free`
   agent and caller prompts and still present in protocol-era intake.
4. RESOLVED (owner, 2026-09-01): keep the shared 240 s ceiling
   (`DOMAIN_MAX_STEPS_SECONDS["intake_free"] = 240`) for the smoke; revisit
   only if the free arm's pass-median call duration shifts.
5. DECIDED (owner, 2026-09-01): the variant DEFAULTS to complication rate
   1.0 — `tau2.config.DOMAIN_COMPLICATION_RATE = {"intake_free": 1.0}`,
   resolved by the CLI into the stored run config when no explicit
   `--complication-rate` is passed (an explicit flag, including 0, always
   wins), and mirrored by the prompt-bed packet builder so the review
   packet renders full-density prompts. Semantics are exactly
   `--complication-rate 1.0`: one categorical draw per call with the kind
   mix proportional to the profile vector — NOT the HARD profile (no
   uniform mix, no hard-tier task filter) — and the mass-to-none rule
   still discounts constrained tasks, so density over the frozen set is
   high but below 1 (144/200 at seed 42, catalog 2.4.0). Rationale: every
   real call has some quirk, and full density maximizes measurement per
   call. The shared intake default is untouched; an A/B against
   protocol-era intake baselines therefore picks up a complication-density
   delta on top of the policy delta — pass an explicit rate to run a
   density-matched comparison.
6. DECIDED (owner, 2026-09-02): the benchmark settles on TWO modes — full
   protocol (canonical `intake`, the scaffolded reference) and informed-free
   (`intake_free`, the benchmark) — no prompt ladder. The default
   `intake_free` policy carries the verification sentence
   (`FREE_STRATEGY_POLICY_VERSION` 1.2.0, §3); the `PolicyHint` seam was
   briefly repurposed as the downward prompt-informativeness ablation
   `{default, goal_only, concept_only}`; headline statistic is verification
   discrimination `P(verified | attempt-1 wrong)` vs
   `P(verified | attempt-1 right)` from `log_capture` vs gold (§1).
7. DECIDED (owner, 2026-09-02, later the same day): remove the `PolicyHint`
   ablation knob entirely — the benchmark has exactly one canonical mode-B
   policy, no switches. The ablation's one measured result is recorded in
   §3a (goal-only 0.25 / concept-hint 0.30, drivers archived in
   `explorations/intake_free_smoke_2026-09-01/`); the machinery — enum,
   version constant, `--policy-hint` flag, RunConfig/Info provenance
   fields, refill plumbing, render rewrites — was ripped out (rip-and-redo,
   no compat shims). The drift guard on the canonical policy's anchored
   verification sentence stays (`load_free_strategy_policy`), as do the
   protocol-era no-nudge guards.
