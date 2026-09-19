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
  --naturalness-sidecar data/simulations/paper_runs/tau-multi/judge_outputs/utterance_naturalness_v16_trial0 \
  --out data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json

tau2 paper multilingual-task-success \
  --evidence-root /path/to/tau-multi \
  --out data/analysis/tau_multilingual_task_success_significance_2026-09-18.json

tau2 paper multilingual-latency \
  --evidence-root /path/to/tau-multi \
  --audit papers/tau-multilingual/reproduction/audit.json \
  --previous-experience papers/tau-multilingual/reproduction/validation_runs/pre-retail-name-role-v1/experience.json \
  --out papers/tau-multilingual/reproduction/latency.json

tau2 paper multilingual-ablation-transcripts \
  --evidence-root /path/to/tau-multi \
  --out papers/tau-multilingual/reproduction/retail_ablation_transcripts
```

Task-success provenance uses the canonical logical
`data/simulations/paper_runs/tau-multi/` namespace, so the JSON and its source
fingerprint are identical regardless of where the detached bundle is mounted.
The latency replay additionally requires the bundle's archived pre-correction
results under `validation_runs/ko/` and `validation_runs/zh/`; these are inputs
to the historical comparison, not active benchmark cells.

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

The complete evidence archive stores each retained validation measure once,
under `human_annotations/validations/<call|utterance>_level/<measure>/<language>/`.
Its index records each retained measure's evidence-gate result. The public,
versioned validation projection is preserved separately under
`validation_runs/pre-retail-name-role-v1/human_annotations/validations/`; it
omits private provenance and free-text fields without changing public rows. The
first-critical-error review covers English and the five localized languages;
the supporting miscellaneous language review remains localized-language only.
Verify schemas, hashes, metrics, factor coverage, and the FCE summary offline
with:

```bash
tau2 judges tau-multi-validation \
  data/simulations/paper_runs/tau-multi/human_annotations
```

## Combined naturalness replay

The complete human-annotation command above verifies the combined-naturalness
labels, predictions, metrics, and prompt provenance. Before a paper replay,
preflight the corrected active non-English trial-0 cohort recorded by the
current Experience analysis:

```bash
tau2 judges tau-multi-naturalness prepare \
  --repo-root /path/to/reviewer-repository \
  --evidence-root /path/to/tau-multi \
  --experience data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json
```

Then run or resume the frozen combined judge into a separate sidecar directory:

```bash
tau2 judges tau-multi-naturalness run \
  --repo-root /path/to/reviewer-repository \
  --evidence-root /path/to/tau-multi \
  --experience data/analysis/tau_multilingual_experience_without_fluency_2026-09-18.json \
  --out /path/to/utterance-naturalness-v16-trial0 \
  --max-concurrency 100
```

The active canonical replay is rooted at
`data/simulations/paper_runs/tau-multi/judge_outputs/utterance_naturalness_v16_trial0/`;
its portable manifest, rebind receipt, and hybrid summary are mirrored under
`reproduction/judges/utterance_naturalness_v16_trial0/`.

The human-validation cohort intentionally remains the frozen original
pre-correction cohort under
`validation_runs/pre-retail-name-role-v1/human_annotations/validations/`; it
validates the judge and is not substituted for the corrected execution cohort.

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

The checked-in `uv.lock` is generated through Sierra's package mirror. Reviewers
without mirror access can install the public dependencies from PyPI and override
the manuscript Makefile's Python launcher:

```bash
python -m pip install -e '.[experiments]'
make -C papers/tau-multilingual/v3 repro PYTHON=python
```
