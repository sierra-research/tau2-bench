# Intake paper result sources

This file records the local artifacts behind every quantitative statement in
`main.tex` (agent-directed benchmark rewrite, 2026-09-02). Canonical run data:
`data/simulations/paper_runs/tau-elicit/` in the main checkout: `main_runs/`
holds the canonical benchmark dirs plus `INTAKE_FINAL_MANIFEST.md` (the role of
every dir), with `text_channel/` and `ablations/{scaffolded,entity_composition,gated_stages}/` alongside. Drive mirror:
`My Drive/Multilingual Tau/intake_final_2026-09-02/`. Analysis narrative:
`data/analysis/modeb_r_grid_2026-09-02.md`; rollup artifacts and drivers:
`data/analysis/modeb_campaign_2026-09-02/`.

## Agent-directed benchmark — Figure 2, abstract, conclusion

- Cells: `intake_final/modeb_{openai_xhigh,gemini_high,xai_10}_{regular,chanheavy,speechheavy}_2026-09-02`
  (+ `modeb_openai_minimal_regular_2026-09-02` for the reasoning ablation).
- Domain `intake_free`, policy 1.2.0, complication catalog 2.4.0, rate 1.0,
  user `gpt-5.5` xhigh temp 0, 200 tasks, one trial, conc 10 / 8 workers.
- Realization seeds shared across systems: regular=9401, noise-heavy=9402,
  speech-heavy=9403. Noise-heavy is ROBUSTNESS_TRIALS_VERSION 1.2.0
  (10 dB floor, 4 bursts/min at −5..0 dB, 3% frame drops, muffling 0.5) —
  merged in PR #1086 (trunk 759d70853); PR #1076 landed the variant.
- Integrity: 600/600 + 200 sims first-attempt, zero stubs; every sim's
  recorded complication kind matches the offline deterministic draw
  (`modeb_campaign_2026-09-02/expected_draws_{9401,9402,9403}.json`).
- Per-cell pass@1 (passes/200): openai_minimal 83/73/86, openai_xhigh
  89/78/104, gemini_high 103/74/89, and xai_10 133/125/132.
- Figure 2 reports regular-realization Pass@1: .415 / .445 / .515 / .665. The pooled
  three-realization means (.403 / .452 / .443 / .650) are diagnostic only and are not
  labeled Pass@1 in the paper.
- Pass^3 (crossed): .150 / .200 / .135 / .405 for GPT minimal, GPT xhigh,
  Gemini high, and Grok. Independence checks from the three realization rates
  are .065 / .090 / .085 / .274.
- Paired counts (regular, same seed): xai>oai 54 vs 10; xai>gem 53 vs 23;
  gem>oai 41 vs 27. The significance table uses two-sided paired sign permutations with
  100,000 draws, seed 42, a +1 Monte Carlo correction, and separate Holm
  correction across the six system pairs for Pass@1 and Pass^3. Full output:
  `analysis/intake_main_significance_2026-09-05.json`.
- Rollup: `modeb_campaign_2026-09-02/modeb_rollup.json` (+ modeb_analysis.py),
  crossed per-task ledger `modeb_crossed.json`.

## Scaffolded reference — Figure 2

- R and S cells: og matrix `intake_final/intake_m_{lane}_{regular,chanlight_speechheavy}`
  (lanes: openai_minimal, openai_xhigh, gemini_high, xai_10; og seeds
  R=42, S=16319; run commit b220df0a5; pre-2.4.0 complication catalog).
- Noise-heavy cells REDONE 2026-09-02 under trials 1.2.0 and catalog 2.4.0:
  `intake_final/modea_{lane}_chanheavy_2026-09-02`, seed 401 (matches the og
  C seed). og legacy-C cells (frame-drops-only) are retired to
  `archive_intake_explorations/` and never pooled.
- Per-cell success: R .785/.770/.795/.815; tuned-C .635/.595/.640/.720;
  S .725/.735/.700/.780 (min/xh/gem/xai). Figure 2 uses R as Pass@1; pooled
  three-realization means .715/.700/.712/.772 are diagnostic only.
- Pass^3 (og R, new C, og S): .455/.385/.365/.540. Catalog-drift caveat is
  recorded in Limitations.

## Same vs crossed Pass^3 — Table 4

- Same-environment trials combine the canonical scaffolded OpenAI xhigh regular
  cell (`intake_final/intake_m_openai_xhigh_regular`, .770) with two additional
  regular trials in `intake_final/intake_passk_xhigh_regular` (.745 and .755).
  The three-run Pass^3 is .515 (103/200); 59 tasks pass twice, 27 pass once, and
  11 never pass, so 86/200 tasks change outcome at least once.
- Crossed = scaffolded OpenAI xhigh Pass^3 above = .385 (77/200), 13 points and
  26 all-three task successes below the repeated-regular baseline.

## Behavioral measures (agent-directed regular cells, seed 9401)

Computed by `modeb_campaign_2026-09-02/modeb_analysis.py` from agent
`log_capture` calls, gold `submit_fields` action checks, and the caller
simulator's silent `note_spell_request` / `note_readback` tools:

- Attempt-1 wrong (of gold fields with a logged attempt): openai 137/200,
  gemini 129/199, xai 81/193.
- Verification discrimination P(verified | a1 wrong) vs P(verified | a1
  right): .657/.500 (openai), .643/.500 (gemini), .938/.959 (xai).
- Repair of verified wrong captures: .267 / .373 / .237; unverified wrong
  captures recover .106 / .109 / .000.
- Adaptivity delta (noise-heavy − regular, per call): readbacks −0.02 /
  −0.12 / +0.40; gemini call duration 126s → 110s.
- Effort (regular, per call): readbacks .57/.815/1.665, letters spelled
  7.4/7.3/12.0 for OpenAI xhigh/Gemini high/xAI.
- Agent-directed regular call-level verification associations, in OpenAI
  xhigh/Gemini high/xAI order:
  - Calls with at least one read-back: 102/200 (51.0%), 87/200 (43.5%), and
    174/200 (87.0%). Pass@1 with a read-back is 45/102 (.441), 60/87 (.690),
    and 126/174 (.724); without one it is 44/98 (.449), 43/113 (.381), and
    7/26 (.269).
  - Calls with at least one spelling request: 57/200 (28.5%), 52/200 (26.0%),
    and 75/200 (37.5%). Pass@1 with a request is 22/57 (.386), 23/52 (.442),
    and 47/75 (.627); without one it is 67/143 (.469), 80/148 (.541), and
    86/125 (.688).
  - Among 347 initially wrong captures pooled across systems, final recovery is
    10/98 (.102) with neither step, 32/123 (.260) with read-back only, 7/29
    (.241) with spelling only, and 34/97 (.351) with both. These are
    observational associations because verification is agent-selected.
- Figure `fig:main-results` combines the agent-directed and scaffolded Pass@1
  and Pass^3 values previously shown in the main results table. GPT minimal
  Pass^3 is 30/200 (15.0%), using its regular 2026-09-02 run and the channel-
  and speech-heavy 2026-09-05 runs frozen under `main_runs/`.
- The effort comparison uses simulated conversation duration, not execution
  wall time. Agent-directed durations are 59.439/66.370/87.086 s and matched
  scaffolded durations are 87.421/86.942/109.800 s. These values come from
  artifact `4a6629554b1a`, stored at
  `modeb_campaign_2026-09-02/caller_effort_agent_directed_vs_scaffolded.json`.
  The figure uses the regular `intake_free` and `intake` cells, respectively,
  and splits each 200-call cell evenly into 100 easy and 100 hard tasks.
  Agent-directed/scaffolded Pass@1 is .60/.90, .55/.83, and .77/.89 on easy
  tasks and .29/.64, .48/.76, and .56/.74 on hard tasks for OpenAI xhigh,
  Gemini high, and xAI. Added mean simulated duration is 22/34, 19/22, and
  21/24 seconds for easy/hard tasks, respectively. Averaged across systems,
  scaffolding raises Pass@1 by 23.3 points for easy entities and 27.0 points for
  hard entities. The older scaffolded runs do not contain the silent
  event logging needed for a direct read-back or spelling-count comparison.
- Terbinafine vignette: task intake_medications_hard_04, cell
  modeb_openai_xhigh_regular_2026-09-02.
- Reasoning ablation: minimal .415 vs xhigh .445 (26 vs 32 paired wins,
  exact binomial p=.512); readback rate .38 vs .51; discrimination gap
  +.31 vs +.16. Scaffolded null: .785 vs .770 (27 vs 24 paired wins).

## Combined value-bank and entity-results table (Table 3)

- Pass@1 per bank pools the 9 agent-directed cells (180 calls/bank); bootstrap
  percentile CIs over the bank's 20 tasks (10k resamples, seed 42;
  replace with BCa when the significance verb reruns). Pass^3 pools the 60
  system-task pairs per bank.
- Values: medications .222/.017, coined .267/.033, addresses .378/.100,
  emails .389/.083, properties .472/.183, person names .533/.300,
  codes .578/.250, times .711/.433, phones .772/.467, dates .828/.600.

## Voice spread (agent-directed vs scaffolded)

- Scaffolded regular per-voice spreads are 13 points for OpenAI xhigh, 44 for
  Gemini high, and 16 for xAI; arjun_roy is Gemini's worst at .54 and xAI's
  second-best at .88.
- Scaffolded regular Pass@1 rankings (voice, calls, score):
  - OpenAI xhigh: mildred_kaplan (38, .842), wei_lin (41, .805),
    arjun_roy (39, .769), priya_patil (40, .725), mamadou_diallo (42, .714).
  - Gemini high: mildred_kaplan (39, .974), wei_lin (39, .897),
    priya_patil (39, .821), mamadou_diallo (42, .762), arjun_roy (41, .537).
  - xAI: mildred_kaplan (40, .900), arjun_roy (41, .878), mamadou_diallo
    (42, .786), priya_patil (39, .769), wei_lin (38, .737).
- Agent-directed R spread: openai 27 pts (.33--.60), gemini 28 (.41--.69), xai 15
  (.60--.75). Source: modeb_rollup.json per-voice blocks.
- Agent-directed regular Pass@1 rankings (voice, calls, score):
  - OpenAI xhigh: mildred_kaplan (47, .596), wei_lin (39, .487),
    priya_patil (47, .404), mamadou_diallo (31, .355), arjun_roy (36, .333).
  - Gemini high: wei_lin (35, .686), mildred_kaplan (48, .646),
    mamadou_diallo (34, .412), arjun_roy (39, .410), priya_patil (44, .409).
  - xAI: mildred_kaplan (51, .745), arjun_roy (34, .676), priya_patil
    (47, .660), mamadou_diallo (33, .606), wei_lin (35, .600).
- Omnibus 5-voice x binary-outcome Fisher--Freeman--Halton Monte Carlo tests use
  100,000 samples with seed 42 and Holm correction across the three systems.
  Raw/adjusted p-values are .1046/.2091 (OpenAI xhigh), .01337/.04011 (Gemini
  high), and .6044/.6044 (xAI). Ten pairwise two-sided Fisher exact tests within
  Gemini, Holm-corrected, leave no significant pair (smallest adjusted p=.212).
- In the provider-stratified agent-directed comparison, Mildred exceeds Priya,
  Mamadou, and Arjun after Holm correction across ten voice pairs (adjusted
  $p=.027$, $.014$, and $.019$), but not Wei ($p=.938$).

## Realisms (agent-directed benchmark, pooled 9 cells)

- Reproducer: `src/experiments/intake/realism_effects.py`; versioned output:
  `analysis/intake_realism_effects_2026-09-07.json`. The artifact records all
  12 input paths and SHA-256 hashes and verifies every stored assignment against
  the deterministic catalog draw.
- The analysis unit is a task x environment assignment, with exact success
  averaged across the four agent-directed systems. Each contrast retains units
  with positive probability of either the target realism or a clean draw, uses
  normalized inverse-probability (Hajek) means within each environment, and
  averages the three environment-specific effects.
- Point estimates / assigned n and ESS / clean n and ESS: any realism -3.59 /
  326, 291.4 / 166, 135.5; wrong-field answer -11.09 / 29, 26.0 / 166, 135.5;
  spelling variation -8.71 / 94, 89.2 / 65, 61.9; self-correction -4.32 / 50,
  45.6 / 166, 135.5; mispronunciation -0.50 / 49, 43.0 / 18, 15.2; and falter
  and restart +2.04 / 99, 93.4 / 166, 135.5.
- Two-sided randomization tests redraw the catalog's categorical assignment
  100,000 times. Holm correction across the overall contrast and five estimable
  subtypes gives a minimum adjusted p-value of .6664 (reported as .67).
- Assignment is the intention-to-treat exposure and does not guarantee that a
  conditional realism is expressed. Across the four systems, spelling
  variation was assigned in 524 calls and 301 (57.4%) contained a spelling
  event. Falter/restart was assigned in 488 calls; 221 (45.3%) contained a
  spelling event and 101 contained an observed restart.

## Speech fidelity (`tab:fidelity-provider`)

- Source cells are the four 200-call agent-directed regular runs:
  `modeb_{openai_minimal,openai_xhigh,gemini_high,xai_10}_regular_2026-09-02`.
  No noise-heavy or speech-heavy realization is included.
- Every call was judged from stored stereo audio with
  `gemini/gemini-3.1-pro-preview`, delivery prompt v6, sample rate 1.0, and
  finding-filter v1. The filter excludes fidelity findings whose approximate
  span overlaps the last 1.0 second of the judged utterance and retains them in
  `excluded_findings` for audit.
- Metrics are utterance-weighted. For each valid utterance, severity is the
  maximum retained fidelity-finding severity, or zero when there is no retained
  fidelity finding. `Mean` averages this value; `Severity >= 2` is the share of
  utterances whose value is at least two.
- Valid utterances / mean severity / severity-2-or-higher count and rate are:
  GPT minimal 945 / .06984 / 24 (2.54%); GPT xhigh 1,142 / .05867 / 24
  (2.10%); Gemini high 1,046 / .09273 / 38 (3.63%); Grok 1,813 / .10425 / 74
  (4.08%). Table values are rounded to three decimals and one percentage point
  decimal.
- The judge returned valid JSON for 4,946 of 4,948 utterances. One GPT-minimal
  and one Grok utterance repeatedly hit the output limit; both remain explicit
  `ERROR` records and are excluded from the denominator by the scoring
  contract. All 800 calls had usable stereo audio.
- Reproducible rollup and full run provenance:
  `analysis/intake_speech_fidelity_2026-09-04.json`.

## Composition and submission workflow (Tables 3--4)

- Composition roots:
  `ablations/entity_composition/intake_ecomp_n2` and
  `ablations/entity_composition/intake_ecomp_n3`; the single-field reference is
  `main_runs/intake_m_openai_xhigh_regular`.
- The single-field row is contextual rather than a paired count-effect
  estimate: it is 154/200 fields from the canonical scaffolded regular run.
  Its caller reasoning setting and duration cap differ from the later
  composition roots, so the paper's composition claim rests on the within-arm
  gap between field and whole-task accuracy, not a causal n=1-to-n=3 contrast.
- The paper pools trials 0--2 in both multi-field arms. This gives 125/180
  two-field task passes and 290/360 correct fields, and 48/90 three-field task
  passes and 202/270 correct fields. Thus task/field Pass@1 is .694/.806 for
  two fields and .533/.748 for three fields; the table reports two decimals
  with Wald 95% margins (.07/.04 and .10/.05).
- The three-field root contains a fourth trial (21/30 task passes, 78/90 correct
  fields), which is excluded solely to balance the trial count. Pooling all
  four would give task Pass@1 = 69/120 = .575 and field Pass@1 = 280/360 =
  .778; neither changes the qualitative conclusion.
- Run audit: every included trial contains its complete task set, with no null
  rewards or infrastructure-error terminations. The low three-field trial 2
  (11/30) is a broad hard-task decline, not a failed worker. The roots share
  commit `ce74dff8f099fb8560b814592d76e3847ca37ee6`, seed 42, regular speech,
  complications off, GPT Realtime xhigh, and a `gpt-5.5` low-reasoning caller.
  Four discarded caller-hallucination attempts in the two-field root were
  replaced before scoring. The included gated run likewise has three trials.
- The compose manifest at recorded commit `ce74dff8` is authoritative. A later
  catalog-2.4 redraw changed one current-manifest n=2 slot from `vin` to
  `license_plate`; it does not describe these frozen runs. Full audit and
  per-trial counts: `analysis/intake_composition_first3_audit.md`.
- Protocol ablation: scaffolded-era protocol study cells (flat / first-strike
  / oracle), unchanged numbers. Table 3 presents the decision-relevant flat
  and oracle conditions; the first-strike arm remains in the source analysis
  but is omitted from the compact paper table.

## Text control

- `intake_final/intake_text200_xhigh_textpolicy_2026-08-27`: 200/200 = 1.00.

## Pending analyses not shown in the compact manuscript

- Failure attribution on the current agent-directed cells. The completed validation
  below is a separate, earlier provider-default cohort.

## Human validation (separate earlier cohort)

- Frozen annotation directory:
  `data/simulations/paper_runs/tau-elicit/judge_validation/intake_review_100_2026-08-27/`
- Source run:
  `data/simulations/intake_bm200_2026-08-26_chanlight_speechheavy/` (no longer
  kept locally; Drive: `Multilingual Tau/intake_paper_runs_2026-08-29/`)
- Sample: 100 OpenAI `gpt-realtime-2` provider-default,
  channel-light/speech-heavy calls: 70 reward-1 and 30 reward-0 calls.
- Raters: Niko and Ian Belcher. Each reviewed every call first from audio alone
  (`pre_reveal`) and then with transcript and reward (`post_reveal`). Calls
  1--40 were used for rater calibration; blind error-type summaries use calls
  41--100. Both raters also adjudicated all 41 fidelity-judge findings.
- Raw returns:
  `niko_review.csv`, `ian_belcher_review.csv`, `niko_decisions.csv`, and
  `ian_belcher_decisions.csv` in the frozen annotation directory.
- Frozen summary:
  `data/simulations/paper_runs/tau-elicit/judge_validation/intake_review_100_2026-08-27/results.md`.

Post-reveal source attribution on the 30 failing calls is:

| Attribution | n | Share |
| --- | ---: | ---: |
| Agent, both raters | 29 | 96.7% |
| User simulator, both raters | 0 | 0.0% |
| Raters split (agent vs. user) | 1 | 3.3% |

In the blind held-out segment, when both raters identify an error on a failing
call, they agree on its type 88% of the time. The 14 consensus errors are 12
transcription errors and two logical errors; neither rater pair agrees on a
hallucination. Post-reveal, both raters agree on 20 transcription and five
logical errors across the 30 failing calls. Transcription failures concentrate
in emails (6), coined names (4), and properties (3).

Fidelity-judge validation over all 100 calls and 41 findings is:

| Criterion | Precision | Clip-level recall | F1 |
| --- | ---: | ---: | ---: |
| Strict (both raters) | 0.73 | 1.00 | 0.85 |
| Lenient (either rater) | 0.85 | 0.80 | 0.82 |

All 21 calls with a consensus human fidelity defect receive at least one
confirmed judge finding. Finding-decision agreement is 0.88 with Cohen's
kappa 0.63. Per-factor counts range from 1 to 12 and are directional rather
than stable factor-level estimates.
