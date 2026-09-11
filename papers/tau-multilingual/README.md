# τ-Multilingual — ICASSP 2027

This directory contains the ICASSP manuscript and its quantitative-source
ledger. The submission includes the author list, following the conference's
review format.

The reviewer release is organized around four reproducible surfaces:

- the six language packs and their fixed airline, retail, and telecom tasks;
- typed run pools for the five voice configurations and two text controls;
- a frozen-evidence audit that recreates the paper tables and checks run
  configuration, task frames, and source-file hashes;
- a unified six-language human-annotation archive, plus a deterministic command
  selecting three listening calls per language from the separate evidence
  bundle.

The frozen call corpus is intentionally separate from Git because it includes
large audio artifacts. Download the
[reviewer evidence bundle](https://drive.google.com/drive/folders/10L02DTRVMRIeKL9ytSMiL2R_Mrl-nC3C?usp=sharing).
Given a local evidence directory, run:

```bash
tau2 paper multilingual-audit \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction \
  --strict

tau2 paper multilingual-listening-sample \
  --evidence-root /path/to/tau-multi \
  --out /path/to/listening-sample

tau2 paper multilingual-experience \
  --repo-root /path/to/reviewer-repository \
  --naturalness-sidecar /path/to/utterance_naturalness_v16_trial0 \
  --out papers/tau-multilingual/reproduction/experience.json

tau2 paper multilingual-ablation-transcripts \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction/retail_ablation_transcripts
```

The ablation export contains the delivered caller/agent transcript and task
outcome for all 360 unique calls supporting the Hindi and Mandarin retail
localization ablation. It omits raw ticks and audio while preserving source
result and simulation hashes. Verify the checked-in export offline with:

```bash
tau2 paper multilingual-verify-ablation-transcripts \
  --root papers/tau-multilingual/reproduction/retail_ablation_transcripts
```

The checked-in prompt archive maps every scored simulation to its exact rendered
agent and user system prompts, recorded inputs, and historical language-pack
values. Verify its content hashes with:

```bash
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts
```

See [`reproduction/prompts/README.md`](reproduction/prompts/README.md) for the
stronger source-to-archive verification using the frozen result bundle.

## Human annotation archive

The archive stores each retained validation measure once, under
`human_annotations/validations/<call|utterance>_level/<measure>/<language>/`.
Its index records each retained measure's evidence-gate result. The
first-critical-error review covers English and the five localized languages;
the supporting miscellaneous language review remains localized-language only.
Verify schemas, hashes, metrics, factor coverage, and the FCE summary offline
with:

```bash
tau2 judges tau-multi-validation \
  data/simulations/paper_runs/tau-multi/human_annotations
```

## Combined naturalness replay

The canonical human-annotation command above verifies the combined-naturalness
labels, predictions, metrics, and prompt provenance. Before a paper replay,
preflight the exact non-English trial-0 cohort selected
by `reproduction/experience.json`:

```bash
tau2 judges tau-multi-naturalness prepare \
  --repo-root /path/to/reviewer-repository \
  --evidence-root /path/to/tau-multi
```

Then run or resume the frozen combined judge into a separate sidecar directory:

```bash
tau2 judges tau-multi-naturalness run \
  --repo-root /path/to/reviewer-repository \
  --evidence-root /path/to/tau-multi \
  --out /path/to/utterance-naturalness-v16-trial0 \
  --max-concurrency 100
```

The audit is offline and never modifies frozen results. Benchmark and judge
reruns are separate, explicit operations.

For a bounded executable smoke check, select one task from a frozen subset and
spell the complete arm in the invocation. For example, this runs one Spanish
airline call in a new result directory without touching the frozen corpus:

```bash
tau2 run \
  --domain airline \
  --task-set-name airline_es_identity \
  --task-ids 3_es_identity \
  --audio-native \
  --audio-native-provider openai \
  --audio-native-model gpt-realtime-2 \
  --reasoning-effort xhigh \
  --speech-complexity regular \
  --user-persona-id es \
  --seed 42 \
  --num-trials 1 \
  --max-concurrency 1 \
  --save-to reviewer_smoke/es_airline_openai_xhigh
```

See `v3/README.md` for manuscript build instructions and
`v3/RESULT_SOURCES.md` for the evidence ledger. The completed release checks and
their documented warnings are summarized in `reproduction/VERIFICATION.md`.
