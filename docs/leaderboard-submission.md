# Submission guide

This guide covers submitting results to the τ-bench leaderboard at
**[taubench.com](https://taubench.com)**. There are four paths, one per
benchmark generation:

| You ran | Package | Leaderboard tab | Section |
|---|---|---|---|
| `text-v1` (2.0 text, 18 cells) | `benchmark-submission.json` | 2.0 | [2.0 benchmarks](#20-benchmarks-text-v1-voice-v2-banking-and-entity) |
| `voice-v2` (2.0 voice, τ-i18n, 18 cells) | `benchmark-submission.json` + `benchmark-diagnostics.json` + `benchmark-quality.json` | 2.0 | [2.0 benchmarks](#20-benchmarks-text-v1-voice-v2-banking-and-entity) |
| `voice-v2-banking` / `voice-v2-entity` (6 languages) | `benchmark-submission.json` | 2.0 (Banking, Intake) | [Banking and τ-Entity](#banking-and-τ-entity-voice) |
| `voice-v1` (English voice, 3 domains) | `submission.json` | English voice | [voice-v1](#voice-v1-english-voice) |
| 1.x text or voice (v1.0.1) | `submission.json` | 1.x | [1.x submissions](#1x-submissions) |

1.x and 2.0 results are not comparable (see
[Comparing with v1.0.1](../CHANGELOG.md#upgrading-from-1x)), so they are shown on
separate leaderboard tabs. 1.x submissions are still accepted.

Every path ends the same way: validate locally, upload the trajectories to
external storage, and open a pull request. Terms such as *cell*, *frame* and
*judge* are defined in the [glossary](glossary.md).

## Rules for every submission

- **Same configuration throughout.** Use one agent model and one user simulator,
  with identical arguments, for every domain and cell.
- **Complete runs.** Do not filter tasks with `--num-tasks` or `--task-ids`. The
  2.0 commands refuse incomplete runs; for 1.x, each domain should appear once
  with all of its tasks.
- **Trials.** The 2.0 benchmarks and voice-v1 are scored at one trial (pass@1).
  For 1.x text, at least 4 trials per domain are strongly preferred.
- **Voice speech complexity.** Voice submissions use `--speech-complexity regular`
  (the default).
- **Use your own caller voices** for voice runs. See
  [Voice personas](voice-personas.md).

### Standard and custom submissions

A **standard** submission uses the default benchmark-side setup for its track:
the standard tasks, policies, tools, user simulator and evaluator, with no
benchmark-side changes to prompts, tools, orchestration, task selection or
grading, no access to hidden task data, and a model that was not trained on
τ-bench tasks, rewards or evaluation data. The system under test can be built
however you like behind the agent interface:

- **Text.** A general-purpose LLM through the default τ-bench agent scaffold.
  Adding a planner, router, extra agent, custom tool or modified control flow on
  the benchmark side makes the submission custom.
- **Voice.** One voice agent behind the standard τ-voice interface. Proprietary
  ASR and TTS, multiple models, internal prompts and tools, and routing are fine
  as long as they live inside the submitted system. A transport adapter that
  only connects the system to the standard interface is allowed.

A **custom** submission changes the benchmark-side setup, for example:

- multi-model routers, ensembles or additional agents in the benchmark-side code;
- tools beyond the standard tool set;
- modified orchestration, control flow, prompts or system instructions;
- a non-default user simulator, task selection, speech configuration or
  evaluation protocol;
- a model trained or fine-tuned on τ-bench domains, tasks, rewards or
  evaluation data.

Custom submissions must set `submission_type` to `"custom"`, explain the changes
in `methodology.notes`, link the implementation in `references`, and set
`methodology.verification.modified_prompts` to `true` if any benchmark prompt
changed.

---

## 2.0 benchmarks (text-v1, voice-v2, banking and entity)

A 2.0 submission is built from one saved matrix root. `tau2 submit prepare`
detects the benchmark from the root's `benchmark-frame.json`, so `--benchmark`
is optional; pass it only to assert which benchmark you expect. See
[Running the benchmark](running-the-benchmark.md) for run details.

### Voice-v2

```bash
# 1. Run the released voice benchmark. Repeat the same command (it resumes) to
#    fill missing calls; it also retries infrastructure errors.
tau2 run --audio-native --audio-native-model <model> --save-to voice-run

# 2. Judge every cell. Resumable: it judges only missing, stale and errored verdicts.
tau2 judges suite data/simulations/voice-run

# 3. Optional: check judge coverage without model calls.
tau2 judges summarize data/simulations/voice-run

# 4. Build the package. Refuses, with per-cell counts and fill commands, if incomplete.
tau2 submit prepare data/simulations/voice-run --output ./submissions \
  --metadata-file ./submitter.json

# 5. Recompute rewards, judge diagnostics and agent identity from the saved matrix.
tau2 submit validate ./submissions/<name> --results data/simulations/voice-run
```

`submitter.json` names the submission (see [Package metadata](#package-metadata)).
`prepare` writes the package to `./submissions/<name>/`, where `<name>` is
`{model_name}_{model_organization}_{submission_date}` in lowercase letters,
numbers and hyphens. voice-v2 and text-v1 names have no suffix. Banking and
entity packages add `_banking` or `_entity`, so that one agent's three packages
can share their metadata (see [Banking and τ-Entity](#banking-and-τ-entity-voice)).

`prepare` refuses unless every cell's rewards are complete and every applicable
judge measure in every cell has a usable saved verdict. It reads saved results
and verdicts only; it never calls a model. An incomplete run exits with code 1
and a message like this:

```
Error: Submission is incomplete for voice-v2 (data/simulations/voice-run).
Rewards: 60/900 calls lack a scored reward (60 missing, 0 unscored).
Rewards: 6 infrastructure-errored calls are excluded from the score, not counted as gaps.
  pt-retail: 10 missing, 2 errored (excluded)
  ko-telecom: 50 missing
Judges: 1/18 cells have applicable measures without a usable saved verdict.
  zh-airline: naturalness 48/50 judged (2 not judged)
To fill the gaps:
  - Rerun the original `tau2 run ... --save-to <root>` command with --auto-resume: ...
  - Then run `tau2 judges suite data/simulations/voice-run`: it judges only missing, stale and errored verdicts.
```

Calls that ended in an infrastructure error (`termination_reason:
infrastructure_error`) do not block a package. They are harness faults, not
model failures, so they are excluded from the score's denominator: a cell is
scored over its scored calls only, and a task whose calls all errored drops out
of its cell's mean. The package reports them as `error_trials` per cell, per
language, per domain and overall, and `prepare` prints the total. A cell is
complete once every call is scored or errored, provided at least one is scored;
a cell whose calls all errored still blocks. Missing and unscored calls still
block. Rerunning with `--auto-resume` retries infrastructure errors if you want
them scored.

Judge verdicts blocked only because recorded Live speech has no transcript
are recording gaps, not judge failures, so they do not block either. A verdict
whose only unit errors are missing-Live-input (`kind: "input"`) errors counts
as judged; its blocked units stay out of the pass fraction and are reported as
`input_error_units`. A call a measure cannot judge at all for that reason (for
call-level measures, any missing transcript) is excluded and reported as
`input_excluded_calls`. Any other judge error (a model or API failure, an
unparseable reply) still blocks. `tau2 judges suite` already asks again, in
the same run, for units whose reply was incomplete or invalid (`--unit-retries`,
default 2, keeping every successful unit); rerun it to retry what is left. A
measure whose applicable calls are all excluded still blocks.

The speech-fidelity judge runs at temperature 0, so a clip whose reply reaches
the judge's output limit (usually a long, looping agent turn) does so on every
retry. Such clips are reported as `output_limit_units` and do not block: the
utterance counts as a Generation-quality issue when naturalness flagged it and
is otherwise left out. At most 0.1% of a package's speech-fidelity clips may
be left out this way; above that, `tau2 submit prepare` fails and lists the
calls, and a maintainer reviews them.

Only the default production judge catalog is accepted. Verdicts made with
`--registry` bundles have a different catalog revision and will not validate.

### Text-v1

```bash
# 1. Run the released text benchmark (repeat the same command to resume).
tau2 run --benchmark text --agent-llm <model> --agent-llm-args '<json>' --save-to text-run

# 2. Build the package (rewards only; no judges needed).
tau2 submit prepare data/simulations/text-run --output ./submissions \
  --metadata-file ./submitter.json

# 3. Recompute rewards from the saved matrix.
tau2 submit validate ./submissions/<name> --results data/simulations/text-run
```

The leaderboard has no 2.0 text tab yet. Text-v1 packages validate the same
way, but `manifest.json` has no entry for them; open an issue before submitting
one.

### Banking and τ-Entity (voice)

`voice-v2-banking` (`--benchmark banking`, 582 calls) and `voice-v2-entity`
(`--benchmark entity`, 1200 calls) cover all six languages. Each is its own
matrix root and its own package, scored on rewards only.

```bash
# 1. Run the released benchmark (repeat the same command to resume).
tau2 run --audio-native --benchmark banking --audio-native-model <model> --save-to banking-run
tau2 run --audio-native --benchmark entity --audio-native-model <model> --save-to entity-run

# 2. Build each package (rewards only; no judges needed).
tau2 submit prepare data/simulations/banking-run --output ./submissions \
  --metadata-file ./submitter.json

# 3. Recompute rewards from the saved matrix.
tau2 submit validate ./submissions/<name>_banking --results data/simulations/banking-run
```

`prepare` names these packages `<name>_banking` and `<name>_entity`, where
`<name>` is the voice-v2 package name. Use the same `submitter.json` for all
three packages, so they share the model name, organization and submission date.
The suffix keeps their directories apart.

Banking always runs with the `alltools` retrieval config, and entity with the
agent-directed strategy. Other retrieval configs and the scaffolded strategy
are custom runs (`--benchmark none`) and have no 2.0 package.

Submit each package as for voice-v2 (see [Submit a 2.0 package](#submit-a-20-package)),
but list it in the `voice_v2_domain_submissions` array of `manifest.json`. It
fills the Banking or Intake row of your voice-v2 row when the model name,
organization, agent provider, agent model and submission type all match; it
never changes Overall. A banking or entity package without a voice-v2 package
is shown only when its domain is selected.

### What the 2.0 package contains

| File | Benchmarks | Contents |
|---|---|---|
| `benchmark-submission.json` | text-v1, voice-v2 | The frozen frame, every task/trial observation, the per-cell, per-language and per-domain reward summary, and a SHA-256 of each cell's `results.json`. |
| `benchmark-diagnostics.json` | voice-v2 only | The judge catalog and transport revisions, each judge contract used, and per cell: the same per-measure summary `tau2 judges summarize` prints, plus a fingerprint for every judged call × measure (`verdict_sha256` is the SHA-256 of the stored verdict). |
| `benchmark-quality.json` | voice-v2 only | Call-level [Interaction](glossary.md#interaction) and [Generation quality](glossary.md#generation-quality) (`experience` in the file) for every scored call: the five Interaction flags, both outcomes, and per-cell pass/fail/unscored counts. `prepare` computes it from the saved calls and verdicts with no model calls; `validate` recomputes it and rejects any difference. Schema: [`schema_benchmark_quality.json`](../web/leaderboard/public/submissions/schema_benchmark_quality.json). |
| `benchmark-metadata.json` | text-v1, voice-v2 | Your `submitter.json` fields, plus the agent provider and model recorded in the run conditions. Maintainers add `trajectories` (the S3 prefix) after upload. Schema: [`schema_benchmark_metadata.json`](../web/leaderboard/public/submissions/schema_benchmark_metadata.json). |

The leaderboard reads these files directly. A package must be complete:
every cell scored, and, for voice-v2, diagnostics and call quality for every
cell. Call quality never blocks a package: a call whose verdicts or ticks
cannot decide a measure is counted as unscored, not failed, and left out of
that measure's rate.

The package does not copy trajectories or audio. Maintainers validate it against
your saved matrix root, which you upload separately (next step). `validate`
rejects changed membership, incompatible conditions, duplicate trials, missing
evidence, and summaries that disagree with the saved results or verdicts.

### Package metadata

```json
{
  "model_name": "my-voice-model",
  "model_organization": "Example Labs",
  "submitting_organization": "Example Labs",
  "submission_date": "2026-10-20",
  "contact_info": {"email": "you@example.org", "name": "Your Name"},
  "submission_type": "standard",
  "references": [{"title": "Model card", "url": "https://example.org/model"}],
  "notes": "How the agent was configured and run."
}
```

`model_name`, `model_organization`, `submitting_organization`,
`submission_date` and `contact_info.email` are required. `submission_type` is
`"standard"` (default) or `"custom"` (see
[Standard and custom submissions](#standard-and-custom-submissions)); explain a
custom setup in `notes` and link it in `references`. Unknown fields are rejected.

### Submit a 2.0 package

1. **Upload the matrix root** (for example `data/simulations/voice-run/`) to
   external storage you can share a link to: Google Drive, Hugging Face,
   institutional storage. Upload the whole root: `benchmark-frame.json`,
   `benchmark-plan.json`, `execution.json` and every cell directory with its
   `results.json`, `simulations/`, `artifacts/` (audio) and `diagnostics/`.
2. **Add the package to the repository** as
   `web/leaderboard/public/submissions/<name>/` (the files `prepare` wrote), and add `<name>` to the `voice_v2_submissions` array in
   `web/leaderboard/public/submissions/manifest.json`. Commit only the package
   files, never the trajectories.
3. **Open a pull request** (see [Pull request](#pull-request)) with the link to
   the uploaded matrix root. Maintainers run `tau2 submit validate` against it,
   then copy the matrix to
   `s3://sierra-tau-bench-public/submissions/<name>/trajectories/<language>/<domain>/`.

voice-v2 packages are the rows of the τ³-Voice **τ-i18n (Multilingual)**
leaderboard (`/leaderboard?benchmark=voice`). Each row shows Task completion
(pass@1, overall or for the selected domain) and, from the package's
`benchmark-quality.json`, Interaction and Generation quality: the share of
scored calls that pass, pooled over the scope shown. A package without that
file shows "—" for both and stays out of the 3D explorer. A package's breakdown
shows each score per language × domain, the five Interaction failures, and each
judge measure per language with its coverage. Judge measures are diagnostics
and never change the rank. 1.x and voice-v1 submissions keep their own tabs and
are still accepted.

The frozen 2026-10-02 voice-v2 release calls predate the matrix layout. A
maintainer-only adapter, `tau2 submit prepare --historical-voice-v2`, reads
those saved exports and writes the same package; see
[Voice V2 release packages](voice-v2-publication.md).

---

## Voice-v1 (English voice)

Voice-v1 is the English frame behind the current English voice leaderboard:
50 airline, 114 retail and 114 telecom tasks, one trial.

```bash
# 1. Run the frame (all three domains, or repeat with --domain for one at a time).
tau2 run --audio-native --benchmark voice-v1 \
  --audio-native-provider <provider> --audio-native-model <model> \
  --save-to voice-v1-run

# 2. Prepare a leaderboard submission from the cell directories.
tau2 submit prepare \
  data/simulations/voice-v1-run/en-airline \
  data/simulations/voice-v1-run/en-retail \
  data/simulations/voice-v1-run/en-telecom \
  --output ./voice-v1-submission

# 3. Validate it.
tau2 submit validate ./voice-v1-submission/<name>
```

Point `prepare` at the cell directories, not the matrix root. Given the root, it
builds a 2.0-style package instead of the leaderboard `submission.json`. Single-domain submissions are allowed; each domain must be
complete.

For voice, `prepare`:

1. keeps only `regular` speech-complexity results;
2. converts results to the directory layout (`results.json` + `simulations/`);
3. copies only the audio of each task's canonical simulation;
4. records `voice_config` (provider, model, tick duration, user TTS) in
   `submission.json`;
5. computes `interaction_metrics` (latency, responsiveness, interruptions,
   selectivity; see [Interaction metrics](interaction-metrics.md)); maintainers
   recompute these from your trajectories;
6. sets `modality: "voice"` and asks for the voice user-simulator version.

Then follow the [leaderboard steps](#leaderboard-steps-voice-v1-and-1x), using
the `voice_submissions` array.

If the provider you want to evaluate has no adapter in
`src/tau2/voice/audio_native/`, implement one (see the existing providers), open
a PR with it, and contact us.

---

## 1.x submissions

1.x results are still accepted and appear on the 1.x tab. To produce scores
comparable to the existing 1.x leaderboard, run from the
[`v1.0.1`](https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.1)
release, not from 2.0:

```bash
git checkout v1.0.1
uv sync
```

Run every domain with the same settings:

```bash
tau2 run --domain retail  --agent-llm <model> --user-llm <user-model> --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm <model> --user-llm <user-model> --num-trials 4 --save-to my_model_airline
tau2 run --domain telecom --agent-llm <model> --user-llm <user-model> --num-trials 4 --save-to my_model_telecom
tau2 run --domain banking_knowledge --retrieval-config alltools \
  --agent-llm <model> --user-llm <user-model> --num-trials 4 --save-to my_model_banking
```

The user-simulator model is reported on the leaderboard.

**`banking_knowledge`** needs a retrieval configuration, recorded as
`retrieval_config` in its results and shown as a badge:

| Config | Description |
|---|---|
| `alltools` | BM25 + OpenAI dense retrieval + shell access |
| `alltools-qwen` | BM25 + Qwen dense retrieval + shell access |
| `terminal_use` | Shell navigation of the knowledge base (grep, cat, find) |
| `openai_embeddings` | Dense retrieval with OpenAI `text-embedding-3-large` |
| `qwen_embeddings` | Dense retrieval with Qwen3-Embedding |
| `bm25` | Sparse BM25 retrieval |

For another method, use a short descriptive string (for example
`"custom_reranker"`) and describe it in `methodology.notes`.

**Cost (optional).** Report the average cost per trajectory in USD in each
domain's `cost` field, and describe how you computed it in `methodology.notes`.

Prepare and validate:

```bash
tau2 submit prepare \
  data/simulations/my_model_retail \
  data/simulations/my_model_airline \
  data/simulations/my_model_telecom \
  data/simulations/my_model_banking \
  --output ./my_submission

tau2 submit validate ./my_submission/<name>
```

`prepare` verifies the trajectory files, checks the requirements above, computes
pass^k, asks for the model name, organization and contact email, and writes
`submission.json` plus a `trajectories/` directory. `--no-verify` skips
trajectory verification while you iterate. To check trajectory files on their
own, use `tau2 submit verify-trajs <paths...>`.

1.x voice submissions follow the [voice-v1](#voice-v1-english-voice) steps
from the 1.x release.

---

## Leaderboard steps (voice-v1 and 1.x)

### 1. Create the submission directory

Copy `submission.json` (only) to
`web/leaderboard/public/submissions/<name>/`, where `<name>` is
`{model_name}_{model_organization}_{submission_date}` in lowercase letters,
numbers, hyphens and underscores, for example `gpt-4.1_openai_2025-01-15`.

```
web/leaderboard/public/submissions/my-model_myorg_2025-01-15/
└── submission.json
```

### 2. Update the manifest

Add `<name>` to the matching array in
`web/leaderboard/public/submissions/manifest.json`:

| Array | Use for |
|---|---|
| `submissions` | 1.x text submissions |
| `voice_submissions` | voice submissions (voice-v1, 1.x voice) |
| `legacy_submissions` | original τ-bench v1 task set; maintainers only |

```json
{
  "submissions": ["existing-submission_org_2024-12-01", "my-model_myorg_2025-01-15"],
  "voice_submissions": ["my-voice-model_myorg_2026-03-01"],
  "legacy_submissions": ["older-model_org_2024-06-20"]
}
```

### 3. Upload the trajectories

Trajectories are not committed to the repository; they are hosted on S3. Upload
the `trajectories/` directory that `prepare` wrote to external storage (Google
Drive, Hugging Face, institutional storage) and keep its structure:

```
trajectories/                         # text: one file per domain
├── my-model_airline_default_gpt-4o_4trials.json
└── ...

trajectories/                         # voice: one directory per domain
└── <experiment_name>/
    ├── results.json
    ├── simulations/
    └── artifacts/                    # canonical audio only
```

A maintainer copies them to S3 after review.

---

## Pull request

1. Fork [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench).
2. Add your submission directory and, for voice-v1 and 1.x, the
   `manifest.json` change.
3. Open a pull request that includes:
   - a short description of the model and results;
   - the link to your uploaded trajectories (or, for 2.0, the matrix root);
   - any framework changes or omitted tasks;
   - a link to the model or paper, if available.

After merge, submission metadata is synced to S3 by a GitHub Actions workflow,
and a maintainer uploads the trajectories from your link. The site at
[taubench.com](https://taubench.com) reads everything from S3.

---

## `submission.json` reference (voice-v1 and 1.x)

This section applies to the leaderboard `submission.json` written for voice-v1
and 1.x submissions. Your `submission.json` must follow the schema defined in [`web/leaderboard/public/submissions/schema.json`](../web/leaderboard/public/submissions/schema.json).

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Name of the model being evaluated |
| `model_organization` | string | Organization that developed the model |
| `submitting_organization` | string | Organization that ran the evaluation |
| `submission_date` | string | Date of submission (YYYY-MM-DD) |
| `contact_info` | object | Contact information (`email`, `name`, `github`) |
| `results` | object | Performance results per domain |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `submission_type` | string | `"standard"` | `"standard"` or `"custom"` |
| `modality` | string | `"text"` | `"text"` or `"voice"` |
| `is_new` | boolean | `false` | Highlight as new on the leaderboard |
| `trajectories_available` | boolean | `false` | Whether trajectory files are available on S3 |
| `trajectory_files` | object | — | Mapping of domain name to trajectory filename (on S3) |
| `references` | array | — | Links to papers, documentation, repos |
| `methodology` | object | — | Evaluation methodology details |
| `voice_config` | object | — | Voice-specific configuration (required for voice) |
| `interaction_metrics` | object | — | Voice interaction metrics computed from trajectories; see [Interaction Metrics](#interaction-metrics) |
| `model_release` | object | — | Model release metadata (release date + announcement link); see [Model Release](#model-release) |

### Domain Results

Each domain in `results` accepts:
- `pass_1` through `pass_4`: Success rate percentage (0-100) or `null`
- `cost`: Average cost in USD per trajectory or `null`
- `retrieval_config`: Required for `banking_knowledge` only

### References

The optional `references` array links to papers, blog posts, documentation, or other resources. Supported `type` values: `paper`, `blog_post`, `documentation`, `model_card`, `github`, `huggingface`, `other`.

### Model Release

The optional `model_release` object captures metadata about the **model itself** (independent of the evaluation), so the leaderboard can track model progress over time:

| Field | Type | Description |
|-------|------|-------------|
| `release_date` | string (YYYY-MM-DD) | Date the model was first made publicly available |
| `announcement_url` | string | URL to the release announcement (blog post, paper, model card, release notes, etc.) |
| `announcement_title` | string | Title of the announcement (used as link text in the UI) |

`release_date` is distinct from the top-level `submission_date` (which is when the evaluation was submitted) and from `methodology.evaluation_date` (which is when the evaluation was run).

If `announcement_url` is provided, `release_date` is required.

Example:

```json
"model_release": {
  "release_date": "2025-08-05",
  "announcement_url": "https://www.anthropic.com/news/claude-opus-4-1",
  "announcement_title": "Claude Opus 4.1"
}
```

### Voice Config Fields

Required when `modality` is `"voice"`:

| Field | Required | Description |
|-------|----------|-------------|
| `provider` | Yes | Audio-native provider (e.g. `"openai"`, `"gemini"`, `"xai"`) |
| `model` | Yes | Model identifier (e.g. `"gpt-realtime-1.5"`) |
| `tick_duration_seconds` | No | Duration of each simulation tick in seconds |
| `max_steps_seconds` | No | Call limit recorded in the run, in simulated seconds (2400 for a 2.0 voice-v1 run; the example below is from an older run) |
| `user_tts_provider` | No | User simulator TTS provider/model (e.g. `"elevenlabs/eleven_v3"`) |

### Interaction Metrics

Voice submissions carry an `interaction_metrics` block with the τ-voice
interaction-quality panel (response/yield latency, response/yield rate, agent
interruption rate, and selectivity for backchannels, vocal tics, and
non-directed speech), computed offline from the tick-level trajectories. It is
generated automatically by `tau2 submit prepare` for voice submissions and
**recomputed by maintainers from the submitted trajectories during review** —
submitter-provided values are always replaced.

The block contains a `version` stamp, the detection-window `config`,
per-domain panels under `domains`, and the cross-domain average under
`overall`; every rate is accompanied by its event counts. See
[Interaction Metrics documentation](interaction-metrics.md) for full metric
definitions, and compute it yourself with:

```bash
tau2 submit interaction-metrics <experiment-dirs> --output interaction_metrics.json
```

## Verification System

Submissions are classified as **Verified** or **Unverified** on the leaderboard:

**Verified submissions** require:
- Trajectory data available for review (`trajectories_available: true`)
- No modifications to standard prompts (`modified_prompts: false`)
- Complete evaluation with no omitted tasks (`omitted_questions: false`)

**Unverified submissions** are marked with a caution icon and may have:
- Missing trajectory data
- Modified prompts or evaluation setup
- Omitted questions or domains

### Verification Fields

Include a `verification` section in the `methodology` object:

```json
"methodology": {
  "verification": {
    "modified_prompts": false,
    "omitted_questions": false,
    "details": "Full evaluation with standard configuration"
  }
}
```

- `modified_prompts` (boolean or null): `true` if prompts were modified, `false` if standard, `null` if unknown
- `omitted_questions` (boolean or null): `true` if tasks were omitted, `false` if all tasks evaluated, `null` if unknown
- `details` (string, optional): Additional context about the evaluation methodology

---

## Examples

### Standard Text Submission

```json
{
  "model_name": "My-Model-v1.0",
  "model_organization": "My Company",
  "submitting_organization": "My Company",
  "submission_date": "2025-01-15",
  "model_release": {
    "release_date": "2025-01-10",
    "announcement_url": "https://mycompany.example.com/blog/my-model-v1",
    "announcement_title": "Introducing My-Model-v1.0"
  },
  "contact_info": {
    "email": "contact@mycompany.com",
    "name": "Research Team"
  },
  "is_new": true,
  "trajectories_available": true,
  "trajectory_files": {
    "retail": "my-model_retail_default_gpt-4o_4trials.json",
    "airline": "my-model_airline_default_gpt-4o_4trials.json",
    "telecom": "my-model_telecom_default_gpt-4o_4trials.json",
    "banking_knowledge": "my-model_banking_knowledge_gpt-4o_4trials.json"
  },
  "results": {
    "retail": { "pass_1": 75.2, "pass_2": 68.1, "pass_3": 62.3, "pass_4": 57.8, "cost": 0.15 },
    "airline": { "pass_1": 65.4, "pass_2": 60.1, "pass_3": 56.2, "pass_4": 53.0, "cost": 0.12 },
    "telecom": { "pass_1": 58.9, "pass_2": 52.1, "pass_3": 47.6, "pass_4": 43.2, "cost": 0.18 },
    "banking_knowledge": { "pass_1": 22.5, "pass_2": 17.3, "pass_3": 13.1, "pass_4": 10.2, "cost": 1.05, "retrieval_config": "alltools" }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "v1.0",
    "user_simulator": "gpt-4.1-2025-04-14",
    "notes": "Evaluated using default settings with 4 trials per task.",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false,
      "details": "Full evaluation with standard τ-bench configuration."
    }
  }
}
```

### Custom Submission (Modified Scaffold)

```json
{
  "model_name": "Custom-Multi-Agent-v1",
  "model_organization": "Research Lab",
  "submitting_organization": "Research Lab",
  "submission_date": "2025-01-15",
  "submission_type": "custom",
  "contact_info": { "email": "research@example.com" },
  "is_new": true,
  "trajectories_available": true,
  "references": [
    { "title": "Custom Agent Implementation", "url": "https://github.com/example/custom-tau-agent", "type": "github" }
  ],
  "results": {
    "retail": { "pass_1": 82.5, "pass_2": 78.1, "pass_3": 74.2, "pass_4": 71.0 },
    "airline": { "pass_1": 68.3, "pass_2": 63.5, "pass_3": 59.8, "pass_4": 56.2 },
    "telecom": { "pass_1": 75.1, "pass_2": 70.4, "pass_3": 66.8, "pass_4": 63.5 }
  },
  "methodology": {
    "evaluation_date": "2025-01-10",
    "tau2_bench_version": "0.2.0",
    "user_simulator": "gpt-4o",
    "notes": "Multi-model router with GPT-4 as planner and Claude-3.5-Sonnet as executor. Custom reflection step after each tool call.",
    "verification": {
      "modified_prompts": true,
      "omitted_questions": false,
      "details": "Custom system prompts for planning, execution, and reflection phases."
    }
  }
}
```

See `web/leaderboard/public/submissions/A_EXAMPLE_new-model_example-org_2025-01-15/` for a complete example.

### Voice Submission

Voice submissions set `modality` to `"voice"` and include a `voice_config` object. Set `methodology.user_simulator` to the voice user simulator version (e.g., `"v1.0"` — see git tag `voice-user-sim-v1.0`).

```json
{
  "model_name": "gpt-realtime-1.5",
  "model_organization": "OpenAI",
  "submitting_organization": "Sierra",
  "submission_date": "2026-03-11",
  "modality": "voice",
  "contact_info": { "email": "research@sierra.ai", "name": "Research Team" },
  "is_new": true,
  "trajectories_available": false,
  "results": {
    "retail": { "pass_1": 43.9 },
    "airline": { "pass_1": 40.0 },
    "telecom": { "pass_1": 21.1 }
  },
  "voice_config": {
    "provider": "openai",
    "model": "gpt-realtime-1.5",
    "tick_duration_seconds": 0.2,
    "max_steps_seconds": 600,
    "user_tts_provider": "elevenlabs/eleven_v3"
  },
  "interaction_metrics": {
    "version": "1.0",
    "config": { "tick_duration_sec": 0.2, "no_yield_window_sec": 2.0, "...": 0.0 },
    "domains": {
      "retail": {
        "response_latency_mean": 1.44,
        "yield_latency_mean": 0.62,
        "response_rate": 0.998,
        "yield_rate": 1.0,
        "agent_interruption_rate": 0.2,
        "selectivity_backchannel": 0.04,
        "selectivity_vocal_tic": 0.06,
        "selectivity_non_directed": 0.15,
        "counts": { "n_simulations": 114, "response_total": 891, "yield_total": 566, "backchannel_total": 77, "vocal_tic_total": 83, "non_directed_total": 134, "agent_interrupts_count": 179 }
      }
    },
    "overall": { "...": "per-metric mean across domains, counts summed" }
  },
  "methodology": {
    "evaluation_date": "2026-03-01",
    "tau2_bench_version": "v2.0",
    "user_simulator": "v1.0",
    "notes": "Full-duplex audio-native evaluation using regular speech complexity.",
    "verification": {
      "modified_prompts": false,
      "omitted_questions": false
    }
  }
}
```

See `web/leaderboard/public/submissions/A_EXAMPLE_voice-model_example-org_2026-03-11/` for a complete example.

---

## For Maintainers: Review Checklist

### 2.0 packages (text-v1, voice-v2)

- [ ] PR links the complete uploaded matrix root (`benchmark-frame.json` and every cell)
- [ ] Package directory name equals `{model_name}_{model_organization}_{submission_date}`, plus `_banking` / `_entity` for those frames; `manifest.json` lists it in `voice_v2_submissions` (banking and entity: `voice_v2_domain_submissions`)
- [ ] `benchmark-metadata.json` has a contact email; `submission_type` is right (if custom: `notes` and `references` explain the change)
- [ ] `review_submission.py <package> <downloaded-root>` passes (it runs `tau2 submit validate`); with `--upload` it copies the matrix to `trajectories/<language>/<domain>/` and sets `trajectories`
- [ ] voice-v2: `benchmark-diagnostics.json` present and made with the production judge catalog
- [ ] voice-v2: `benchmark-quality.json` present (validation recomputes it from the uploaded calls and verdicts)
- [ ] Spot-check: re-judge a sample of uploaded calls and compare unit outcomes with the matching fingerprints
- [ ] No duplicate submissions

### Text Submissions (1.x)

- [ ] `submission.json` follows the [schema](../web/leaderboard/public/submissions/schema.json)
- [ ] Directory name follows convention (`{model_name}_{model_organization}_{date}`)
- [ ] `manifest.json` updated (`submissions` array)
- [ ] Contact info is provided
- [ ] `submission_type` is set correctly (`"standard"` or `"custom"`)
- [ ] **If custom:** detailed `methodology.notes` and `references` with implementation links
- [ ] **If custom:** `methodology.verification.modified_prompts` is set appropriately
- [ ] `trajectory_files` mapping matches actual filenames
- [ ] 4 trials per domain (check trajectory files)
- [ ] **If banking domain:** `retrieval_config` field present in `banking_knowledge` results
- [ ] Results verified against trajectory data
- [ ] Cost information is positive numbers or `null` (if provided)
- [ ] No duplicate submissions
- [ ] PR includes link to externally hosted trajectory files
- [ ] **After merge:** download trajectories and upload to S3: `aws s3 cp <local-trajectories>/ s3://sierra-tau-bench-public/submissions/<submission-dir>/trajectories/ --recursive --profile tau-bench-ci`

### Voice Submissions (voice-v1, 1.x)

- [ ] `submission.json` follows the [schema](../web/leaderboard/public/submissions/schema.json)
- [ ] `modality` is `"voice"`, `trajectories_available` is `false`
- [ ] `voice_config` includes `provider` and `model`
- [ ] `manifest.json` updated (`voice_submissions` array)
- [ ] `methodology.user_simulator` set to voice user sim version (e.g., `"v1.0"`)
- [ ] PR description includes link to externally hosted trajectory data
- [ ] **New provider:** PR includes audio-native adapter implementation and documentation
- [ ] Results use "regular" speech complexity only
- [ ] `interaction_metrics` recomputed from the submitted trajectories (done automatically by `review_submission.py`; submitter-provided values must not be trusted)
- [ ] No duplicate submissions

## Questions?

If you have questions about submitting results, please:
1. Check the [τ-bench documentation](https://github.com/sierra-research/tau2-bench)
2. Open an issue in the repository
3. Contact us at the email provided in the main README
