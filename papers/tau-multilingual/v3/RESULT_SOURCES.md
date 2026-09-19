# Result source ledger

This ledger maps the manuscript's quantitative claims to portable,
machine-readable artifacts. Paths are relative to the repository root. The
frozen result bundle is distributed separately because the complete audio
corpus is too large for Git. `papers/tau-multilingual/reproduction/VERIFICATION.md`
records the completed release checks and the remaining non-fatal warnings.

## Cohort of record

- Repeated OpenAI/Gemini voice cohort: 72 result files, two trials, 7,200
  scored calls. Fingerprint:
  `a4bcbc6f0de88870d790de8137080dccc1f6ae038f587c13d74eb75f0361210e`.
- xAI voice cohort: 18 result files, one trial, 900 scored calls. Fingerprint:
  `bcb2451c9e2943a3d8e2e3149336da30b405f37b286aabfc13cb530750123e74`.
- Text controls: 36 result files, one trial, 1,800 scored calls. Fingerprint:
  `d6394eb1ac455a50e511c0b025556315017ca2100a7a11964341902152a4c129`.
- Final task-success source map: 126 result files (90 voice and 36 text),
  including the 14 corrected Korean/Mandarin retail cells. Its fingerprint is
  `7bd909036ebce1782de2b64f0ed8580ff54d9eb1bcd36ab5ba8d162c811c1653`.
  The frozen typed artifact is
  `data/analysis/tau_multilingual_task_success_significance_2026-09-18.json`
  (SHA-256 `c8eacf97c6486d58e7a7d5fffe703cf598044f5a5f76f8470d06367421900189`).
- Retail localization ablations: eight result files over the fixed 30-task
  frame, 240 new calls plus the matched baseline slice.
- Manuscript voice slice: trial 0 of the 90 voice cells, 4,500 calls.

The voice fingerprints use
`sha256(sorted "relative_path\\tfile_sha256\\trow_count\\n")`. The complete
file inventory and individual hashes are stored in
`papers/tau-multilingual/reproduction/audit.json` (artifact id
`ee59caca58b614a4`). The release replaces 14 Korean/Mandarin primary
voice/text cells and four Mandarin ablation cells; the other 112 primary/text
cells and four Hindi ablation cells are unchanged. Old and new cell hashes,
judge provenance, and exclusions are recorded in
`papers/tau-multilingual/reproduction/retail_name_role_replacement.json` and
`papers/tau-multilingual/reproduction/retail_ablation_name_role_replacement.json`.
The primary replacement file is an immutable input receipt: it feeds the
canonical naturalness rebind receipt, whose output manifest and hybrid summary
then feed the active Experience analysis. This direction keeps the provenance
graph acyclic. The input receipt has SHA-256
`12dce708b68e97fbeb8e38cbab96340984a2ee60f35987f52f906f6c6fd89deb`.
The Experience analysis independently binds its 90 trial-0 inputs to cohort
fingerprint
`43ba4a604719c22d127e52cd1faf0459bfbfe4103422a50706f8fc8bf685ab6b`.

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

# Rebuild Interaction, call duration, utterance Experience, and significance.
tau2 paper multilingual-experience \
  --repo-root /path/to/tau2-bench \
  --evidence-root /path/to/tau-multi \
  --naturalness-sidecar /path/to/tau-multi/judge_outputs/utterance_naturalness_v16_trial0 \
  --out data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json

# Rebuild task-success significance and trial stability from the active cohort.
tau2 paper multilingual-task-success \
  --evidence-root /path/to/tau-multi \
  --out data/analysis/tau_multilingual_task_success_significance_2026-09-18.json

# Rebuild active trial-0 latency and compare it with the archived cohort.
# The evidence root must include validation_runs/{ko,zh}/ with the archived
# pre-correction retail result directories.
tau2 paper multilingual-latency \
  --evidence-root /path/to/tau-multi \
  --audit papers/tau-multilingual/reproduction/audit.json \
  --previous-experience papers/tau-multilingual/reproduction/validation_runs/pre-retail-name-role-v1/experience.json \
  --out papers/tau-multilingual/reproduction/latency.json

# Preflight the exact trial-0 naturalness replay against a detached result bundle.
tau2 judges tau-multi-naturalness prepare \
  --repo-root /path/to/tau2-bench \
  --evidence-root /path/to/tau-multi \
  --experience data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json

# Rebuild and verify the compact 360-call localization-ablation transcripts.
tau2 paper multilingual-ablation-transcripts \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction/retail_ablation_transcripts
tau2 paper multilingual-verify-ablation-transcripts \
  --root papers/tau-multilingual/reproduction/retail_ablation_transcripts

# Rebuild the figures unconditionally, then the manuscript.
make -C papers/tau-multilingual/v3 repro
```

All audit and validation commands are read-only with respect to frozen results.
Benchmark and hosted-judge reruns write to new output directories.
The frozen human-validation projection remains tied to the explicitly archived
pre-correction cohort; the naturalness preflight above uses the corrected active
execution cohort.

## Claim-to-artifact map

| Manuscript claim | Reproduction artifact |
|---|---|
| Thirty trial-0 voice task-completion cells | `papers/tau-multilingual/reproduction/task_success.csv` |
| Twelve matched text-control cells | `papers/tau-multilingual/reproduction/text_success.csv` |
| Six text-minus-voice gaps | `papers/tau-multilingual/reproduction/text_voice_gap.csv` |
| Repeated-trial stability | `papers/tau-multilingual/reproduction/trial_stability.csv` |
| Four retail localization-ablation rows and 360 compact call transcripts | `papers/tau-multilingual/reproduction/retail_ablations.csv`; `papers/tau-multilingual/reproduction/retail_ablation_transcripts/` |
| Korean/Mandarin retail replacement provenance | Input receipt `papers/tau-multilingual/reproduction/retail_name_role_replacement.json`; canonical rebind and output metadata under `papers/tau-multilingual/reproduction/judges/utterance_naturalness_v16_trial0/`; active analysis `data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json` |
| Mandarin retail-ablation replacement provenance | `papers/tau-multilingual/reproduction/retail_ablation_name_role_replacement.json` |
| Interaction components, call duration, utterance Experience, provider-pair significance, English-relative Interaction significance, and standalone Speech-fidelity totals | `data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json` |
| Turn-taking latency | `papers/tau-multilingual/reproduction/latency.json` |
| Task-success language significance | `data/analysis/tau_multilingual_task_success_significance_2026-09-18.json`; source results are inventoried in `papers/tau-multilingual/reproduction/audit.json` |
| Human-label precision, recall, F1, kappa, labels, verdicts, and prompt contracts | `data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/` |
| Call-level factor/language packages | `data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/call_level/` |
| Combined-naturalness packages | `data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/utterance_level/naturalness/` |
| Speech-fidelity packages | `data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/utterance_level/speech_fidelity/` |
| Hindi gender-agreement package | `data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/utterance_level/gender_agreement/hi/` |
| English and localized-language first-critical-error and caller-quality review | `data/simulations/paper_runs/tau-multi/human_annotations/user_sim_review/fce/` |
| Exact rendered prompts and prompt-affecting run inputs | `papers/tau-multilingual/reproduction/prompts/manifest.json` |
| Historical/current language-pack differences | `papers/tau-multilingual/reproduction/PACK_SNAPSHOT_DIFFERENCES.md` |
| Three listening examples per language | `papers/tau-multilingual/reproduction/listening_manifest.json`; audio exported from the separate evidence bundle |

The prompt archive has artifact id
`352e2091ed18c3b560b3e0fd4b77412d24c029c0132a9703b207029323dc205b`
and manifest SHA-256
`15546db167c05ef498f0590a46476ab46f6f4a58c8ddf1142a553295a51e40ff`.
It contains 134 cells, 10,140 simulations, and 2,813 content-addressed objects.

The authoritative validation index is
`validation_runs/pre-retail-name-role-v1/human_annotations/validations/index.csv`.
Each row resolves to one canonical package at
`<call|utterance>_level/<measure>/<language>/`; there is no sibling validation
tree or normalized duplicate.

## Task and experience table

`data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json`
is a typed v2.2.0 artifact reconstructed from all 4,500 trial-0 calls, with
SHA-256
`2b1af33f3dfbb75886581c50e2bf8cad82d16a2c222fd162b794545c2bceab05`.
Interaction is the
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
| Latency (s) | 1.1227 | 1.4165 | 1.3429 | 1.8248 | 1.1988 | 1.3811 |
| Interaction failure (%) | 55.0742 | 58.9711 | 57.4682 | 59.1458 | 56.4789 | 57.4276 |
| Non-response (%) | 8.3333 | 27.9586 | 47.6667 | 76.7778 | 10.3333 | 34.2139 |
| Interrupts (%) | 81.4444 | 90.0000 | 74.7778 | 79.5556 | 89.3333 | 83.0222 |
| Selectivity errors (%) | 97.4519 | 95.9364 | 83.0439 | 76.5508 | 96.9765 | 89.9919 |
| Monologue (%) | 18.7778 | 19.4529 | 6.1209 | 10.1394 | 34.4444 | 17.7871 |
| Tool use (%) | 69.3635 | 61.5075 | 75.7318 | 52.7056 | 51.3071 | 62.1231 |
| Experience pass (%) | 65.1864 | 67.5330 | 59.5296 | 57.8700 | 52.2878 | 60.4814 |
| Fluency failure (%) | 31.9912 | 29.6913 | 36.7680 | 38.2248 | 43.4113 | 36.0173 |
| Speech-fidelity failure (%) | 7.6990 | 6.7922 | 7.8419 | 9.3433 | 20.7200 | 10.4793 |

Latency is the six-language macro mean of the equal-weight mean of
event-weighted response and yield latencies. It is reported separately and is
not part of Interaction or Experience. Its frozen artifact is
`papers/tau-multilingual/reproduction/latency.json`, SHA-256
`eb56617f3b216566a9147e49699500d30b9ed9b311af3586bda6b85d19c3e01f`.
It binds all 90 active trial-0 source cells and preserves the pre-replacement
sufficient statistics for a direct 80-unchanged/10-changed comparison.

The promoted naturalness replay contains 60,417 utterances in 3,750 localized
calls. The final join contains 60,265 eligible utterances: 21,722 Fluency
failures, 6,874 Speech-fidelity failures, 4,746 failures shared by both, and
23,850 unique Experience failures. Before joining, 3,750 structurally injected
greetings are excluded. There are 76 unmatched naturalness utterances and zero
unmatched delivery utterances. The final-window rule excludes 6,104 findings
from the localized Experience join and 7,462 from the six-language standalone
Speech-fidelity cohort; the exact rows and manifest are in:

- `data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.csv`
- `data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.json`

The tracked naturalness metadata bundle is
`papers/tau-multilingual/reproduction/judges/utterance_naturalness_v16_trial0/`.
Its manifest, identity, work fingerprint, rebind receipt, and hybrid-summary
hashes are respectively `8f3a94f72f1a740df264562fb363845cf35d9c443726b22b1acd36067e1cb054`,
`b3cde4227d98175c7c5ddcd3be53292f24dcc7f858f0289c1cb5efd64d0db849`,
`36f5d814b9e9372ebeec77f5eb35fa7e7ee21c0c03e1f720853038f4fa1ebc51`,
`08360a565673afe75255e39ddf063b2e5147be669ed791f00c0d74cf6efcc493`,
and `d311b10e59fcb0ad8cbad4f3cc25061886f37fd1975a7c5adfc93e02625ecb6d`.
The versioned validation projection it binds has manifest
`4f6bd97c3d9a1d8ae4179ae5a1b3990b13a055f3922e86c3957fd7da4b4c86a5`.
The final-window CSV and manifest hashes are
`2813e0dc25a80fd3a60775dd1533326523fe0b569220f1010bdf78c91dd52841`
and `d4f4abf5bfc9f56895678d194903d62193f1871aa82efa9a567dcba6f488d48e`.

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
The resulting fixed-panel losses are 18.8 points for Korean and 9.5 points for
Mandarin (both Holm-adjusted $p<.00005$); the other three contrasts are not
significant. Grok has one trial and remains descriptive.

The matched controls are recorded in
`papers/tau-multilingual/reproduction/text_success.csv` and
`papers/tau-multilingual/reproduction/text_voice_gap.csv`; the latter gives the
manuscript's 16.8--41.1-point text-over-voice range.
`papers/tau-multilingual/reproduction/trial_stability.csv` records the
two-trial aggregate rates and largest cell gap. Korean records 33.2% and 31.7%
across the two trials (32.4% mean; 5.3-point maximum system gap), while
Mandarin records 41.5% and 41.8% (41.7% mean; 2.7-point maximum system gap).
Across all repeated systems and languages, trial 1 differs from trial 0 by
-0.6389 points ($p=.47252$). The four retail identity/script contrasts and all
eight 30-task cells are in
`papers/tau-multilingual/reproduction/retail_ablations.csv`; their compact
frozen transcripts are under
`papers/tau-multilingual/reproduction/retail_ablation_transcripts/`.
The corrected Mandarin localized baselines are 36.7% for GPT xhigh and 53.3%
for Gemini high. The corrected Mandarin rows are GPT xhigh: 36.7% localized
and romanized, 20.0% with source-English entities, and 53.3% with native-script
database values; Gemini high is 53.3% in all four columns. The companion
replacement manifest binds these four corrected ablation cells.

### Interaction, Experience, and model tradeoffs

`data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json`
is the hash-bound quality and significance artifact used to render the
figures. It records all 30 language--system cells, Interaction components,
Experience denominators, call duration, gender overrides, utterance alignment,
standalone Speech fidelity, five English-relative Interaction tests, and all
ten provider-pair Experience tests. It supports the reported model tradeoffs,
including Grok's 24.7-minute mean call duration, the 11.7--14.2-minute range
for the other systems, and the weakest provider contrast, Gemini minimal
versus high (1.900 points; Holm-adjusted $p=.02378$).

The artifact records English-relative Interaction deltas (Holm-adjusted $p$)
of -4.611 (.000120) Spanish, -3.743 (.000200) Portuguese, -3.136 (.000550)
Hindi, -12.599 (.000050) Korean, and -11.767 (.000050) Mandarin. Standalone
Speech-fidelity cleanliness is 92.8% English, 91.5% Spanish, 92.0% Portuguese,
88.5% Hindi, 88.1% Korean, and 87.5% Mandarin over 72,164 scored utterances;
7,795 utterances are flagged.

The language-resolved latency row is recomputed from the response and yield
fields stored on all active trial-0 primary trajectories across airline,
retail, and telecom. Within each language--system cell, each latency is
event-weighted over its scoreable opportunities and the response and yield
means then receive equal weight. The typed latency artifact retains exact
previous and active sufficient statistics, result hashes, and simulation
hashes; 80 cells are byte-identical and the ten corrected Korean/Mandarin
retail cells change. This covers all five displayed systems; the diagnostic
cells are unshaded, and printed values remain seconds.

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
and typed analysis artifact, the narrowest localized-language ranges are 8.7
points for Grok Task success and 10.1 for Gemini high Interaction. Experience
ranges are retained as diagnostics but are not interpreted across languages
because Naturalness is not calibrated cross-linguistically; GPT xhigh has the
narrowest such range (12.1 points), and Grok the widest (23.1). Grok's
localized-language Interaction range is 13.4 points; it is 16.4 when English
is included.

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
aggregations of the trial-0 cells recorded in the Experience artifact and its
hash-bound source results. English remains the unchanged reference, while the
Korean and Mandarin retail lookup results are recomputed from the corrected
cells bound by the replacement manifest. Across the four repeated systems,
exact-lookup success / task success conditional on exact lookup is 85.0/47.2,
83.5/66.0, 81.1/39.8, and 83.5/50.9 for English; 26.0/46.2, 34.0/70.6,
42.0/42.9, and 56.0/78.6 for Korean; and 60.0/63.3, 62.0/58.1, 74.0/29.7,
and 68.0/70.6 for Mandarin. Thus exact-lookup ranges are 81--85% for English,
26--56% for Korean, and 60--74% for Mandarin. Across all five corrected
systems, Korean has 115/250 exact successes and Mandarin has 174/250.

The telecom authentication split uses the fixed 25 name+DOB and 25 phone
tasks, pooled over five trial-0 systems (125 calls per language--method
stratum), yielding 32.0% versus 35.2% for Korean and 40.8% versus 33.6% for
Mandarin.

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
`data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/` and
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
sizes are read from the versioned `human_annotations/validations/index.csv`
rather than duplicated in this ledger.

The Korean and Mandarin human-validation rows remain tied to the frozen
pre-correction retail cohort, whose caller name roles were underspecified. The
corresponding raw cells are preserved under `validation_runs/{ko,zh}/`; active
task, interaction, generation, and latency results use the corrected cells.

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
`data/simulations/paper_runs/tau-multi/validation_runs/pre-retail-name-role-v1/human_annotations/validations/utterance_level/gender_agreement/hi/`.
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
