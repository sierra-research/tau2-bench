# Intake paper result sources

This file records the artifacts behind every quantitative statement in
`main.tex` (agent-directed benchmark rewrite, 2026-09-02). The reviewer-facing
copies live in `reproduction/`: `results/manifest.csv` identifies each frozen
result root and its SHA-256, `transcripts/` contains compact scored records,
`run_configs/` and `prompts/` preserve the run contract, and
`analysis_inputs/` contains the checked rollups. The detached source corpus has
the same `main_runs/`, `text_channel/`, and
`ablations/{scaffolded,entity_composition,gated_stages}/` layout and can be
verified against the manifest with `tau2 paper elicitation-verify`.

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

- Same-environment trials: `intake_final/intake_passk_xhigh_regular`
  (scaffolded openai xhigh, 3 same-config trials) → Pass^3 same = .515.
- Crossed = scaffolded openai xhigh Pass^3 above = .385.

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

- Clean 253/498 = .508 vs realism-triggered 674/1302 = .518.
- Per-kind (regular cells): spell_correction .56/.75/.75 vs clean .51/.49/.63;
  mispronounced_term .18/.41/.41.

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

- Exact task snapshots, transcripts, and recomputed counts for the two- and
  three-field rows, joint submission, and verify/retry are in
  `reproduction/analysis_inputs/composition_protocol_recomputed.json`.
- The single-field reference is reconstructed in
  `reproduction/analysis_inputs/composition_one_field_matched.json`. For each
  of the 210 slot occurrences in the frozen two- and three-field compose
  manifest, it selects the exact atomic parent task from three regular,
  scaffolded GPT-xhigh realizations: trial 0 of
  `intake_m_openai_xhigh_regular` and trials 0--1 of
  `intake_passk_xhigh_regular`. Repeated parents retain their slot-frequency
  weight. The resulting 486/630=.771 exactly matches the paper.
- The two-field result uses all three trials in `intake_ecomp_n2`: task
  125/180=.694 and field 290/360=.806. The three-field result uses the frozen
  three-trial design (trials 0--2) from `intake_ecomp_n3`: task 48/90=.533 and
  field 202/270=.748; its retained fourth exploratory trial is not pooled.
- Combining those rows gives joint submission task 173/270=.641 and field
  492/630=.781. `intake_ehier` gives verify/retry task 222/270=.822 and field
  544/630=.863.
- The separate no-retry source run is not present in the frozen local evidence
  and is listed in `reproduction/SOURCE_GAPS.md`. A first-attempt
  counterfactual over `intake_ehier` reproduces the reported no-retry field
  numerator (440/630=.698), but not its task numerator.

## Text control

- `intake_final/intake_text200_xhigh_textpolicy_2026-08-27`: 200/200 = 1.00.

## Analyses outside the current manuscript

- Failure attribution on the current agent-directed cells. The completed validation
  below is a separate, earlier provider-default cohort.

## Human validation (separate earlier cohort)

- Reviewer archive:
  `reproduction/judge_validation/validation.csv` and `metrics.json`.
- Source run:
  `data/simulations/archive_intake_explorations/intake_bm200_2026-08-26_chanlight_speechheavy/`.
- Sample: 100 OpenAI `gpt-realtime-2` provider-default,
  channel-light/speech-heavy calls: 70 reward-1 and 30 reward-0 calls.
- The reviewer archive maps every final call label and finding decision directly
  to its simulation, task, utterance text, and judge output.

Final source attribution on the 30 failing calls is:

| Attribution | n | Share |
| --- | ---: | ---: |
| Agent, both raters | 29 | 96.7% |
| User simulator, both raters | 0 | 0.0% |
| Raters split (agent vs. user) | 1 | 3.3% |

When both raters identify an error on a failing call, they agree on its type
88% of the time. The 14 consensus errors are 12
transcription errors and two logical errors; neither rater pair agrees on a
hallucination. Across the full set, both raters agree on 20 transcription and five
logical errors across the 30 failing calls. Transcription failures concentrate
in emails (6), coined names (4), and properties (3).

Fidelity-judge validation over all 100 calls and 41 findings is:

| Criterion | Precision | Clip-level recall | F1 |
| --- | ---: | ---: | ---: |
| Strict (both raters) | 0.73 | 1.00 | 0.85 |
| Lenient (either rater) | 0.85 | 0.73 | 0.79 |

All 21 calls with a consensus human fidelity defect receive at least one
confirmed judge finding. Finding-decision agreement is 0.88 with Cohen's
kappa 0.63. Per-factor counts range from 1 to 12 and are directional rather
than stable factor-level estimates.
