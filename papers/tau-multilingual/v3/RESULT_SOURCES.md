# Result source ledger

This ledger maps the manuscript's quantitative claims to portable,
machine-readable artifacts. Paths are relative to the repository root. The
frozen result bundle is distributed separately because the complete audio
corpus is too large for Git. `papers/tau-multilingual/reproduction/VERIFICATION.md`
records the completed release checks and the remaining non-fatal warnings.

## Cohort of record

- Repeated OpenAI/Gemini voice cohort: 72 result files, two trials, 7,200
  scored calls. Fingerprint:
  `c8e470205629b69730f2f6d279b9a23e3f290c372db1a3f17c34b2ece2a994e8`.
- xAI voice cohort: 18 result files, one trial, 900 scored calls. Fingerprint:
  `8ad13688a83f8d8a76e991b2525486e7bfa99693e09663a72230ad5ed871886e`.
- Text controls: 36 result files, one trial, 1,800 scored calls.
- Retail localization ablations: eight result files over the fixed 30-task
  frame, 240 new calls plus the matched baseline slice.
- Manuscript voice slice: trial 0 of the 90 voice cells, 4,500 calls.

The voice fingerprints use
`sha256(sorted "relative_path\\tfile_sha256\\trow_count\\n")`. The complete
file inventory and individual hashes are stored in
`papers/tau-multilingual/reproduction/audit.json`. The Experience analysis
independently binds its 90 trial-0 inputs to cohort fingerprint
`79da0f0731ce1fbe12379581e5eb46d82abd6458042da2baf9b3ddd6d5dc29ff`.

## Reproduction commands

```bash
# Validate the frozen result bundle and rebuild task-success tables.
tau2 paper multilingual-audit \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction \
  --strict

# Verify the checked-in prompt object store, then bind it to source results.
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts \
  --evidence-root /path/to/tau-multi \
  --repo-root /path/to/tau2-bench

# Verify final human labels, stored predictions, metrics, and file hashes.
tau2 judges tau-multi-validation \
  data/simulations/paper_runs/tau-multi/human_annotations

# Rebuild Interaction, latency, utterance Experience, and significance.
tau2 paper multilingual-experience \
  --repo-root /path/to/tau2-bench \
  --naturalness-sidecar /path/to/utterance_naturalness_v16_trial0 \
  --out papers/tau-multilingual/reproduction/experience.json

# Preflight the exact trial-0 naturalness replay against a detached result bundle.
tau2 judges tau-multi-naturalness prepare \
  --repo-root /path/to/tau2-bench \
  --evidence-root /path/to/tau-multi

# Rebuild and verify the compact 360-call localization-ablation transcripts.
tau2 paper multilingual-ablation-transcripts \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction/retail_ablation_transcripts
tau2 paper multilingual-verify-ablation-transcripts \
  --root papers/tau-multilingual/reproduction/retail_ablation_transcripts

# Rebuild the manuscript and figures.
make -C papers/tau-multilingual/v3 clean all
```

All audit and validation commands are read-only with respect to frozen results.
Benchmark and hosted-judge reruns write to new output directories.

## Claim-to-artifact map

| Manuscript claim | Reproduction artifact |
|---|---|
| Thirty trial-0 voice task-completion cells | `papers/tau-multilingual/reproduction/task_success.csv` |
| Twelve matched text-control cells | `papers/tau-multilingual/reproduction/text_success.csv` |
| Six text-minus-voice gaps | `papers/tau-multilingual/reproduction/text_voice_gap.csv` |
| Repeated-trial stability | `papers/tau-multilingual/reproduction/trial_stability.csv` |
| Four retail localization-ablation rows and 360 compact call transcripts | `papers/tau-multilingual/reproduction/retail_ablations.csv`; `papers/tau-multilingual/reproduction/retail_ablation_transcripts/` |
| Interaction components, latency, call duration, utterance Experience, and provider-pair significance | `papers/tau-multilingual/reproduction/experience.json` |
| English-relative Interaction significance and standalone Speech-fidelity totals | `data/analysis/tau_multilingual_experience_without_fluency_2026-09-03.json` |
| Task-success language significance | Frozen two-trial GPT/Gemini results inventoried in `papers/tau-multilingual/reproduction/audit.json`; method and reported contrasts below |
| Human-label precision, recall, F1, kappa, labels, verdicts, and prompt contracts | `data/simulations/paper_runs/tau-multi/human_annotations/validations/` |
| Call-level factor/language packages | `data/simulations/paper_runs/tau-multi/human_annotations/validations/call_level/` |
| Combined-naturalness packages | `data/simulations/paper_runs/tau-multi/human_annotations/validations/utterance_level/naturalness/` |
| Speech-fidelity packages | `data/simulations/paper_runs/tau-multi/human_annotations/validations/utterance_level/speech_fidelity/` |
| Hindi gender-agreement package | `data/simulations/paper_runs/tau-multi/human_annotations/validations/utterance_level/gender_agreement/hi/` |
| English and localized-language first-critical-error and caller-quality review | `data/simulations/paper_runs/tau-multi/human_annotations/user_sim_review/fce/` |
| Exact rendered prompts and prompt-affecting run inputs | `papers/tau-multilingual/reproduction/prompts/manifest.json` |
| Historical/current language-pack differences | `papers/tau-multilingual/reproduction/PACK_SNAPSHOT_DIFFERENCES.md` |
| Three listening examples per language | `papers/tau-multilingual/reproduction/listening_manifest.json`; audio exported from the separate evidence bundle |

The authoritative index is `human_annotations/validations/index.csv`. Each row
resolves to one canonical package at
`<call|utterance>_level/<measure>/<language>/`; there is no sibling validation
tree or normalized duplicate.

## Task and experience table

`papers/tau-multilingual/reproduction/experience.json` is a typed v2.2.0
artifact reconstructed from all 4,500 trial-0 calls. Interaction is the
descriptive equal-weight mean of five call-level failure percentages:
non-response, interruption, selectivity, monologue, and validated tool use.
Experience is the percentage of aligned agent utterances that pass both the
combined-naturalness and material speech-fidelity checks. Each utterance is
counted once even when either judge returns multiple findings or both
components fail.

English contributes to Interaction but not Experience because no validated
combined-naturalness judge is claimed for English. All 150 calls in every
language-system cell remain in the descriptive cohort.

The exact pre-rounding system values are:

| Measure | GPT min. | GPT xhigh | Gem min. | Gem high | Grok | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Latency (s) | 1.1236 | 1.4198 | 1.3554 | 1.8196 | 1.1972 | 1.3831 |
| Interaction failure (%) | 54.6526 | 58.7754 | 57.5854 | 59.3360 | 56.8371 | 57.4373 |
| Non-response (%) | 8.5556 | 27.9586 | 47.2222 | 76.5556 | 9.7778 | 34.0139 |
| Interrupts (%) | 80.3333 | 90.4444 | 73.8889 | 79.3333 | 90.0000 | 82.8000 |
| Selectivity errors (%) | 96.7967 | 95.7220 | 83.7340 | 77.1211 | 97.0900 | 90.0928 |
| Monologue (%) | 17.7778 | 17.9987 | 6.5767 | 10.6332 | 36.5943 | 17.9161 |
| Tool use (%) | 69.7994 | 61.7532 | 76.5050 | 53.0365 | 50.7233 | 62.3635 |
| Experience pass (%) | 65.3690 | 67.4167 | 59.6107 | 57.8826 | 51.8419 | 60.4242 |
| Fluency failure (%) | 31.9028 | 29.7844 | 36.7420 | 38.2954 | 43.9352 | 36.1320 |
| Speech-fidelity failure (%) | 7.4725 | 6.6710 | 7.8715 | 9.2776 | 21.0606 | 10.4706 |

Latency is the six-language macro mean of the equal-weight mean of
event-weighted response and yield latencies. It is reported separately and is
not part of Interaction or Experience.

The utterance join contains 59,514 eligible utterances: 21,583 Fluency
failures, 6,825 Speech-fidelity failures, 4,768 failures shared by both, and
23,640 unique Experience failures. Before joining, 3,750 structurally injected
greetings are excluded. There are 76 unmatched naturalness utterances and zero
unmatched delivery utterances. The stored final-window rule excludes 6,078
fidelity findings; the exact rows and manifest are in:

- `data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.csv`
- `data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.json`

Portuguese and Hindi gender diagnostics use the operative-male verdict files
below. They are keyed by simulation id and do not edit frozen result files:

- `data/analysis/pt_gender_v3_male_full_2026-09-03.csv`
- `data/analysis/hi_gender_v3_male_full_2026-09-03.csv`

The Experience artifact records the paths, row counts, hashes, factor ids, and
judge versions for these inputs.

The same sidecars provide the operative-male v3 failure rates quoted in the
manuscript: 27.2% for Hindi and 29.6% for Portuguese. The Portuguese result
includes 135/750 calls flagged only for speaker-marked `obrigada`; excluding
that convention-sensitive class gives 11.6%.

## Current manuscript claim provenance

### Task outcome, inference, and controls

`papers/tau-multilingual/reproduction/task_success.csv` contains the 30
trial-0 language--system cells plotted in Figure 2(a), and `audit.json` binds
each cell to its frozen `results.json` inputs. The confirmatory language test
uses both trials for the four GPT/Gemini systems: it averages trials within
task, compares languages with English over 150 matched domain--task clusters,
runs 100,000 two-sided sign permutations with seed 42, applies Holm correction
over five language contrasts, and uses 10,000 paired BCa bootstrap resamples.
The resulting fixed-panel losses are 20.7 points for Korean and 13.3 points for
Mandarin (both Holm-adjusted $p<.00005$); the other three contrasts are not
significant. Grok has one trial and remains descriptive.

The matched controls are recorded in
`papers/tau-multilingual/reproduction/text_success.csv` and
`papers/tau-multilingual/reproduction/text_voice_gap.csv`; the latter gives the
manuscript's 16.8--41.5-point text-over-voice range.
`papers/tau-multilingual/reproduction/trial_stability.csv` records the
two-trial aggregate rates and largest cell gap. The four retail identity/script
contrasts and all eight 30-task cells are in
`papers/tau-multilingual/reproduction/retail_ablations.csv`; their compact
frozen transcripts are under
`papers/tau-multilingual/reproduction/retail_ablation_transcripts/`.

### Interaction, Experience, and model tradeoffs

`papers/tau-multilingual/reproduction/experience.json` is the portable,
hash-bound artifact used to render the figures. It records all 30
language--system cells, Interaction components, Experience denominators,
latency, call duration, gender overrides, utterance alignment, and all ten
provider-pair Experience tests. It supports the reported model tradeoffs,
including Grok's 25.6-minute mean call duration, the 11.3--13.8-minute range
for the other systems, and the weakest provider contrast
($p=.01844$).

The checked-in source-analysis artifact
`data/analysis/tau_multilingual_experience_without_fluency_2026-09-03.json`
adds the `interaction_language_significance` block absent from the portable
render artifact. It records English-relative Interaction deltas (Holm-adjusted
$p$) of -4.611 (.000120) Spanish, -3.743 (.000200) Portuguese, -3.136
(.000550) Hindi, -12.266 (.000050) Korean, and -12.189 (.000050) Mandarin.
It also records the all-language standalone Speech-fidelity denominator and
cleanliness values used in Section 4.3. The portable reproduction artifact
remains the primary rendering source. The language-resolved diagnostic table
reads only its standalone Speech-fidelity fields from this supplemental source
artifact; inferential claims likewise use it only where the portable copy lacks
the corresponding field.

The language-resolved latency row uses the response and yield fields already
stored on the frozen trial-0 primary trajectories across airline, retail, and
telecom. Within each language--system cell, each latency is event-weighted over
its scoreable opportunities and the response and yield means then receive equal
weight. This covers all five displayed systems; the diagnostic cells are
unshaded, and printed values remain seconds.

### Figure semantics and model stability

The primary system--language heatmaps report Task completion, the descriptive
Interaction pass composite, and utterance Experience. All panels are
higher-is-better and include an Overall row that equal-weights the five
displayed model configurations within each language; language-marginal averages
remain omitted. Fluency and Speech fidelity remain prose decompositions rather
than additional primary heatmaps.

The model-level diagnostic table reports Task failure and every other row in
the lower-is-better direction. Numeric labels retain the underlying percentages
(and seconds for latency); boldface marks the best model within each row. Its
separated Avg. column reports the unweighted five-model mean and is not eligible
for bolding.

Section 4.5 measures cross-language stability by the displayed best-to-worst
range for each model and outcome. From the same trial-0 task-completion table
and typed analysis artifact, the narrowest ranges are 9.4 points for Grok Task
success, 11.5 for Gemini high Interaction, and 12.3 for GPT xhigh Experience
across the five localized languages. Grok has the widest Interaction and
Experience ranges (18.3 and 23.4 points), respectively.

The language-resolved best-model table is generated from the same trial-0
task-completion table and the two artifacts described above. All rows are
lower-is-better, with Task failure defined as 100 minus Task completion. Within
every language and metric, the table reports the minimum among the five
displayed model configurations; numbered model keys preserve ties using
unrounded values. The unshaded cells avoid implying comparisons across
diagnostics or languages. Interaction components retain their own opportunity
denominators.
Naturalness is blank for English, while Speech-fidelity failure is 100 minus
standalone fidelity cleanliness so that English remains covered.

### Failure-mechanism and qualitative audits

The interaction-component and localization percentages in Sections 4.2--4.4 are
aggregations of the trial-0 cells recorded in the Experience artifacts and
their hash-bound source results. The exact-lookup audit retains English,
Korean, and Mandarin because those source trajectories were unchanged by the
Spanish/Portuguese reruns. It reports exact lookup / task success conditional
on exact lookup ranges of 81--85% / 40--66% for English, 17--55% / 28--51%
for Korean, and 37--58% / 32--73% for Mandarin. The telecom authentication
split uses the fixed 25 name+DOB and 25 phone tasks, pooled over five trial-0
systems (125 calls per language--method stratum), yielding 32.0% versus 35.2%
for Korean and 40.8% versus 33.6% for Mandarin.

The checked-in localized identity-task prompts supply the correct family-first
names (for example, `Liu Feng` and `Hu Wei` in
`telecom_tasks_zh_identity.json`), while paper-run agents often map the spoken
parts into first-name then last-name fields. Mandarin additionally uses
romanized pinyin database names, creating a cross-script mismatch when a caller
uses native-script name decomposition. The Mandarin pack's
`agent_language_clause` in `data/tau2/multilingual/zh/pack.yaml` explicitly
requires all tool calls, arguments, and structured outputs to remain in
English. These observations motivate the manuscript's diagnosis; the retail
ablation remains exploratory.

The language examples are drawn from final human labels under
`data/simulations/paper_runs/tau-multi/human_annotations/validations/` and
`data/simulations/paper_runs/tau-multi/human_annotations/misc/`, together with
the compact retail-ablation transcripts. Standalone Speech-fidelity totals are
in the source-analysis artifact above; the quoted failure-subtype counts come
from its frozen delivery-finding inputs after the recorded final-window
exclusions.

## Human judge validation

The canonical archive stores final human labels and row-aligned judge verdicts
for English and the five localized languages. Each row preserves the language,
public measure, sampling strata, evaluation unit, simulation/task/turn
identity, human label, stored judge verdict and reason, agreement, model, and
prompt/rubric versions. Every measure/language pair has one leaf package, and
its manifest binds the final rows and derived metrics by SHA-256. Final cohort
sizes are read from `validations/index.csv` rather than duplicated in this
ledger.

The runtime naturalness factor is one combined verdict covering natural word
choice, translationese, and the language-specific grammar clauses documented
in the archived prompt contract.

The manuscript reports only passing validation endpoints. Naturalness runs
from Hindi (F1 0.814, kappa 0.634) to Spanish (F1 0.979, kappa 0.965), and
speech fidelity from Korean (F1 0.824, kappa 0.754) to Spanish (F1 0.966,
kappa 0.955). Three additional diagnostic cohorts remain in the archive with
an `insufficient_positive_support` gate status and are not reported.

The final enabled factor sets are language-specific. English uses tool use and
speech fidelity. Each of the five localized languages uses combined
naturalness, tool use, and speech fidelity. The retained localized additions
are:

- ES: counting units and email-symbol verbalization;
- PT: register, gender, counting units, regional consistency, and email;
- HI: gender and name/address conventions;
- KO: honorific agreement, name/address conventions, and email;
  and
- ZH: counting units, modal particles, name/address conventions, email, and
  tone-induced meaning flips.

Deterministic email checks are released as diagnostics and verified in code;
they are not presented as human-validated learned judges.

Hindi `gender_agreement` is stored at
`data/simulations/paper_runs/tau-multi/human_annotations/validations/utterance_level/gender_agreement/hi/`.
Evaluation fixes the agent as male and freezes caller gender,
source-interruption state, and allowed literals rather than inferring them at
scoring time. This is a factor-accuracy regression test, not the source for prevalence; the
Hindi gender prevalence rate remains derived from the operative-male sidecar
documented above. Korean `honorific_agreement` likewise remains a separate
diagnostic. Neither diagnostic changes the paper's combined-naturalness
Fluency result or utterance Experience.

The matched disclosed-voice comparisons use 30 calls per language. Their
gender-agreement failure rates change from 30.0% to 40.0% in Hindi and from
30.7% to 20.0% in Portuguese, supporting the manuscript's conclusion that
disclosure alone does not consistently remove the broader agreement errors.

Speech-fidelity leaf packages validate the underlying detector using retained
fidelity findings after the final-window filter. The paper's downstream
standalone Speech-fidelity rates and Experience aggregation separately apply a
severity-$\geq2$ threshold, plus Mandarin tone-induced meaning flips where
applicable. Each leaf verifier checks final labels, stored verdicts, prompt
contracts, transcript alignment, metrics, and file hashes offline.

## User-simulator review

`human_annotations/user_sim_review/fce/` covers English and the five localized
languages. Final cohort sizes and aggregate first-critical-error and
caller-quality statistics are read from the landed archive rather than copied
into this ledger. Supporting evidence under `human_annotations/misc/` remains
scoped to the five localized languages.

## Configuration and task-frame checks

The audit verifies every expected voice and text cell, row count, trial set,
provider, reasoning-effort setting, and fixed 50-task frame. Four early English
telecom files predate explicit `task_subset` metadata; their exact canonical
task ids still equal `telecom_50`.

The xAI result metadata and retained WebSocket URLs identify
`grok-voice-think-fast-1.0`; no other xAI model URL occurs in the frozen logs.
Exact run-time language-pack values are stored with the prompt objects rather
than reconstructed from the current packs.

## Listening examples

`papers/tau-multilingual/reproduction/listening_manifest.json` deterministically
selects one fixed arm per domain and three calls per language. Selection does not
use task outcome or judge score. The manifest records each simulation, task, and
result-relative audio path; the `multilingual-listening-sample` command exports
the corresponding WAV files from the separate evidence bundle.

## Known limits

- Text versus voice is a system-bundle comparison, not a pure modality
  intervention.
- Provider reasoning-effort labels are not a shared compute scale.
- The five localized languages have one final human label per validation row;
  the paper therefore reports agreement with that reference rather than
  inter-annotator agreement.
- Hosted-model reruns test executable reproducibility but cannot guarantee
  bit-identical outputs after provider-side changes.
- Frozen prompts cannot expose provider instructions that were never returned
  to the client.
