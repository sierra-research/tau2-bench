# Running the benchmark

This page covers running the released τ-bench 2.0 benchmarks, text and voice:
the commands, the defaults, where results go, how to resume, and how runs are
scored and judged. To submit results afterwards, see the
[submission guide](leaderboard-submission.md). Terms such as *cell*, *frame* and
*pin* are defined in the [glossary](glossary.md).

## The benchmarks

A released benchmark is a [benchmark manifest](glossary.md#benchmark-manifest)
under `data/tau2/benchmarks/`. Five ship with 2.0:

| Benchmark | Manifest | Mode | Languages | Domains and tasks | Calls at 1 trial |
|---|---|---|---|---|---|
| `text-v1` | `text.json` | text | en, es, pt, hi, zh, ko | airline, retail, telecom; 50 frozen tasks each | 900 |
| `voice-v2` (τ-i18n) | `voice.json` | voice | en, es, pt, hi, zh, ko | airline, retail, telecom; 50 frozen tasks each | 900 |
| `voice-v1` | `voice-v1.json` | voice | en | airline 50, retail 114, telecom 114 | 278 |
| `voice-v2-banking` | `banking.json` | voice | en, es, pt, hi, zh, ko | banking_knowledge; all 97 tasks | 582 |
| `voice-v2-entity` (τ-Entity) | `entity.json` | voice | en, es, pt, hi, zh, ko | intake; all 200 tasks | 1200 |

- `text-v1` and `voice-v2` cover the same 18 [cells](glossary.md#cell)
  (6 languages × 3 domains) and the same frozen task subsets (`airline_50`,
  `retail_50`, `telecom_50`). Their [binding revisions](glossary.md#binding-revision)
  differ, because text and voice calls use different persona and prompt content
  from the same pack.
- `voice-v1` is the English-only voice frame used by the current English voice
  leaderboard. It uses the original registry tasks and refuses partial runs.
- `voice-v2-banking` and `voice-v2-entity` score domains that sit outside the
  voice-v2 overall. Non-English cells bind the published pack extensions
  (`banking-dates-v1`, `intake`); the extension revision is part of each
  binding pin. Banking always uses the `alltools` retrieval config. Entity uses
  the default agent-directed strategy and keeps its own 240-second call limit.
- English is a language pack like any other. A run with no `--language` uses
  the English pack.

How `tau2 run` picks a benchmark:

| Command | Benchmark |
|---|---|
| `tau2 run --audio-native ...` | `voice` (voice-v2) by default |
| `tau2 run --audio-native --benchmark voice-v1 ...` | voice-v1 |
| `tau2 run --audio-native --benchmark banking ...` | voice-v2-banking |
| `tau2 run --audio-native --benchmark entity ...` | voice-v2-entity |
| `tau2 run --benchmark text ...` | text-v1 |
| `tau2 run ...` (text, no `--benchmark`) | none: a custom single-domain run |
| `tau2 run ... --benchmark none` | none: a custom single-domain run, text or voice |

A manifest only runs in its own mode: `--benchmark text --audio-native` and
`--benchmark voice` without `--audio-native` are rejected.

## Before you start

1. Install with the extras you need:

   ```bash
   uv sync                    # text
   uv sync --extra voice      # voice (also: brew install portaudio ffmpeg on macOS)
   ```

2. Copy `.env.example` to `.env` and fill in the keys:

   | Key | Needed for |
   |---|---|
   | `OPENAI_API_KEY` | Every benchmark run: the user simulator (`gpt-5.5`), the communication judge and the NL-assertion judge. Also the OpenAI Realtime agent. |
   | `ELEVENLABS_API_KEY` | Voice runs: the caller's speech synthesis. |
   | `DEEPGRAM_API_KEY` | Voice runs: transcription. |
   | `GEMINI_API_KEY` | `tau2 judges suite` (the speech-fidelity judge runs on Gemini), and Gemini Live agents. |
   | `XAI_API_KEY` | xAI agents only. |

3. Voice only: create your own caller voices. The voice IDs in the packs are
   Sierra-internal and will not work with your ElevenLabs account. This creates
   one voice per persona for every voice-v2 language and prints the
   `TAU2_VOICE_ID_<LANG>_<PERSONA>=...` lines to paste into `.env`:

   ```bash
   python -m tau2.voice.scripts.setup_voices --language all
   ```

   Use `--language en` (the default) for voice-v1 only. Your voices will not
   sound identical to Sierra's, and runs with voice overrides are recorded as
   such. See [Voice personas](voice-personas.md).

## Run the text benchmark

```bash
# Preview all 900 conversations. No API calls, nothing written.
tau2 run --benchmark text \
  --agent-llm openai/gpt-5.5 --agent-llm-args '{"reasoning_effort":"low"}' \
  --save-to text-run --dry-run

# Remove --dry-run to execute.
tau2 run --benchmark text \
  --agent-llm openai/gpt-5.5 --agent-llm-args '{"reasoning_effort":"low"}' \
  --save-to text-run
```

Always pass `--agent-llm` (and `--agent-llm-args` if needed). The user simulator
defaults to `openai/gpt-5.5` with `reasoning_effort: low`.

## Run the voice benchmark (voice-v2)

```bash
# Preview all 900 calls.
tau2 run --audio-native --audio-native-model gpt-realtime-2 \
  --reasoning-effort minimal --save-to voice-run --dry-run

# Remove --dry-run to execute.
tau2 run --audio-native --audio-native-model gpt-realtime-2 \
  --reasoning-effort minimal --save-to voice-run
```

Pick the agent with `--audio-native-provider` (`openai`, `gemini`, `xai`, `nova`,
`qwen`, `livekit`, ...) and `--audio-native-model`. Without
`--audio-native-model`, each provider uses its default from `config.py`.

## Run voice-v1 (English)

```bash
tau2 run --audio-native --benchmark voice-v1 \
  --audio-native-provider openai --audio-native-model gpt-realtime-2 \
  --save-to voice-v1-run
```

`--domain` selects whole domains. `--num-tasks` and `--task-ids` are rejected;
for an English sample use `--benchmark none --language en`.

## Run banking and τ-Entity

```bash
# All 97 banking tasks in all six languages: 582 calls.
tau2 run --audio-native --benchmark banking --save-to banking-run

# All 200 entity tasks in all six languages: 1200 calls.
tau2 run --audio-native --benchmark entity --save-to entity-run
```

`--retrieval-config` other than `alltools` and `--elicitation-strategy
scaffolded` are custom runs; use `--benchmark none` for them. Slices and
samples work as for voice-v2.

## Run a slice or a sample

Slices and samples are useful for smoke tests. They have no official score (see
[Scoring](#scoring)).

```bash
# One domain, all six languages: 300 calls.
tau2 run --audio-native --domain retail --save-to retail-run

# One language, all three domains: 150 calls.
tau2 run --audio-native --language en --save-to english-run

# The first 2 frozen retail tasks in every language: 12 calls.
tau2 run --audio-native --domain retail --num-tasks 2 --save-to sample-run
```

- `--language` and `--domain` can be repeated.
- `--num-tasks N` takes the first N tasks of the frozen subset, in subset order,
  in every selected domain. Every language gets the same canonical tasks.
- `--task-ids` selects canonical task IDs from the frozen subset and needs
  exactly one `--domain`. Use either `--num-tasks` or `--task-ids`, not both.
- The same flags work with `--benchmark text`.

## Custom runs

Use `--benchmark none` for anything outside the frozen matrix: other domains,
other task counts, task splits, or a language pack that is not in a manifest.
Custom runs use the ordinary single-domain defaults (see [Defaults](#defaults)).

```bash
# A text run on one domain (text runs are custom unless --benchmark text).
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5

# A shipped language on a task count outside the frozen subset.
tau2 run --audio-native --benchmark none --domain retail --language es --num-tasks 2

# banking_knowledge in voice.
tau2 run --audio-native --benchmark none --domain banking_knowledge \
  --retrieval-config bm25 --num-tasks 2
```

The released τ-Entity frame is `--benchmark entity`. Its default
agent-directed strategy is voice-only. The scaffolded ablation is a custom run
and also runs in text:

```bash
tau2 run --audio-native --benchmark none --domain intake --num-tasks 2
tau2 run --domain intake --elicitation-strategy scaffolded --num-tasks 2
```

See [τ-Entity](elicitation.md) for its task set and defaults.

## Defaults

These are the values in `src/tau2/config.py` and `src/tau2/runner/presets.py`.
An explicit flag always wins.

| Setting | Released benchmark (text-v1, voice-v2, voice-v1, banking, entity) | Custom run (`--benchmark none`) |
|---|---|---|
| User simulator | `openai/gpt-5.5`, `reasoning_effort: low` | `gpt-4.1-2025-04-14`, temperature 0 (intake: `gpt-5.5`, low) |
| Text agent | `gpt-4.1-2025-04-14` unless `--agent-llm` is given | same |
| Voice call limit (`--max-steps-seconds`) | 2400 simulated seconds (entity: 240) | 1200 (intake: 240) |
| OpenAI Realtime agent voice (`--voice`) | `marin` | provider default |
| Trials (`--num-trials`) | 1 | 1 |
| Workers (`--workers`) | 1 worker process (`0` is raised to 1) | in-process |
| Concurrency (`--max-concurrency`) | 3 per worker | 3 |
| Resume | always on | `--auto-resume` |
| Communication check | semantic LLM judge, `gpt-5.5`, `reasoning_effort: low` | same |
| NL-assertion judge | `gpt-5.4-mini-2026-03-17`, `reasoning_effort: low` | same |
| Seed (`--seed`) | 300 | 300 |

Notes:

- **Communication check.** Every run, text or voice, English included, checks
  `communicate_info` with the semantic judge. The judge model, arguments and
  prompt version are saved with the run and reused by `tau2 evaluate-trajs`.
  Override the judge with
  `--communicate-judge '{"model":"...","model_args":{"reasoning_effort":"low"}}'`.
  `--communicate-scorer substring` restores the 1.x substring match and is meant
  for custom runs.
- **Concurrency.** With `--workers 3 --max-concurrency 10`, up to 30 calls run at
  once. `--global-limit` caps the whole fleet and `--provider-limit openai=20`
  caps one provider.
- **Dry run.** `--dry-run` prints the languages, domains, call count, trials and
  workers as a table. Add `--json` for the full resolved plan.

## Where results are saved

`--save-to NAME` writes under `data/simulations/NAME`; an absolute path writes
there instead. `--output` is the same option.

A released benchmark run writes a matrix root with one directory per cell:

```
data/simulations/voice-run/
├── benchmark-frame.json      # the full frame this root is scored against
├── benchmark-plan.json       # the cells and tasks requested so far
├── execution.json            # workers and concurrency
├── en-airline/
│   ├── results.json          # run metadata and task definitions
│   ├── simulations/          # one JSON file per call
│   ├── artifacts/            # audio and per-call logs
│   └── diagnostics/          # judge verdicts (after tau2 judges suite)
├── en-retail/
└── ...
```

Voice cells use this directory layout because each call carries a lot of
tick-level data. Text cells hold a single monolithic `results.json` with every
conversation in it:

```
data/simulations/text-run/
├── benchmark-frame.json
├── benchmark-plan.json
├── execution.json
├── en-airline/results.json
└── ...
```

A custom run writes one directory: a monolithic `results.json` for text, or the
`results.json` + `simulations/` + `artifacts/` layout for voice.
`tau2 convert-results <dir> --to dir|json` converts between the two.

Browse any of these with:

```bash
tau2 view --file data/simulations/voice-run
```

## Resume

Run the same command again. Released benchmark runs always resume: completed
calls are kept, missing calls run, and calls that failed with infrastructure
errors are retried. Custom runs resume with `--auto-resume`.

You can extend a saved benchmark run by raising `--num-tasks`, adding task IDs,
or adding cells; the recorded selection becomes the union. Start a new
`--save-to` root when you change the agent, models or evaluation settings, the
trial count, or the release data. Results saved by 1.x cannot be resumed.

## Scoring

Each call gets a task reward of 0 or 1 from the task's `reward_basis`: database
state, required actions, required information communicated, and NL assertions.
See [Task schema and evaluation](evaluation.md).

- Task success averages trials within a task, tasks within a cell, then cells
  with equal weight. At one trial this is pass@1. Language and domain breakdowns
  use the same equal-cell rule.
- A scored zero is a failed task. Infrastructure errors, unscored calls and
  missing calls are counted separately; they are not zeros.
- Infrastructure errors are excluded from the denominator and reported as
  `error_trials`: a cell scores over its scored calls, and a task whose calls
  all errored drops out of its cell. Retrying them (by resuming) is optional. A
  cell with no scored call has no score.
- Only a complete frame has an official score. A frame is complete when every
  call is scored or infrastructure-errored; missing and unscored calls block it. A slice or sample gets a
  *requested-scope* score, and an incomplete run reports an explicitly partial
  `observed_score`.

At the end of a benchmark run, `tau2 run` prints the score summary as a per-cell
table (add `--json` for the full summary). `tau2 view` shows the same numbers.

## Judges

Judges are model-based diagnostics run after the simulation: naturalness, speech
fidelity, tool use, and language-specific checks such as counting units or
honorifics. They never change the task reward. Which measures apply depends on
the language; for example, there is no English naturalness measure.

```bash
# Show what would be judged and an upper bound on model requests. No model calls.
tau2 judges suite data/simulations/voice-run --dry-run

# Judge every cell. Resumable: it only judges missing, stale and errored verdicts.
tau2 judges suite data/simulations/voice-run

# Coverage and pass fractions per cell and measure. No model calls.
tau2 judges summarize data/simulations/voice-run
```

- Verdicts are saved in each cell's `diagnostics/verdicts.json`.
- `suite` and `summarize` print compact tables. `--json` prints the full typed
  export, and `--output FILE` also saves it. `tau2 judges export` always prints
  JSON.
- `--only naturalness,tool_use` restricts the measures; `--trials 0` restricts
  the trials.
- A voice-v2 submission needs a usable verdict for every applicable measure in
  every cell, so run the judges before `tau2 submit prepare`. Text-v1 and
  voice-v1 submissions carry rewards only.
- The production judges are listed in
  [`production_judges.json`](../src/tau2/judges/production_judges.json). See
  [Judge validation](judge-validation.md) for how they are selected and how to
  author new ones.

## Cost and time

> **TODO (owner):** add rough API cost and wall-clock time for a full text-v1,
> voice-v2 and voice-v1 run, including `tau2 judges suite`. The repository has no
> recorded full runs to derive these from.

What the code fixes: voice-v2 and text-v1 are 900 calls at one trial, voice-v1 is
278. Each voice call is capped at 2400 simulated seconds. A benchmark run uses one
worker with 3 concurrent calls unless you raise `--workers` and
`--max-concurrency`. `tau2 judges suite --dry-run` prints an upper bound on the
number of judge requests before you spend anything.

## Other run modes

### Pass^3

`tau2 run-pass3` runs three ordinary trials per task and reports `pass_hat_3`. It
takes the same domain and task selection as `tau2 run`:

```bash
tau2 run-pass3 --domain airline --num-tasks 5 --save-to airline_pass3
```

For voice, `--pass3-mode crossed` runs one trial per task in each of three
pinned conditions: R (regular channel and speech), C (heavy channel, light
speech) and S (light channel, heavy speech). The result is the share of tasks
passed in all three. This is a different metric from `pass_hat_3`; do not pool
the two.

```bash
tau2 run-pass3 --pass3-mode crossed --audio-native --benchmark none \
  --domain intake --task-ids intake_addresses_easy_01 --save-to intake_rcs \
  --audio-native-model gpt-realtime-2 --reasoning-effort low --dry-run
```

Each condition has its own resumable directory (`<save-to>_regular`,
`<save-to>_channel_heavy`, `<save-to>_speech_heavy`). The summary
`data/simulations/<save-to>_pass3.json` is written once all three finish.

### Canonical-scenario runs

Language runs use the pack's localized database, tasks and policy. To keep the
spoken language but use the original English scenarios, run a custom run with
`--no-localize-db`:

```bash
tau2 run --audio-native --benchmark none --domain retail --language ko --no-localize-db
```

The released benchmarks always use the localized scenarios.

### Review prompts offline

Render the exact prompts a benchmark run would send, without any calls:

```bash
tau2 languages render-prompts --benchmark voice --language en --domain retail \
  --output ./prompt-review
```

This accepts voice manifests only. Omit `--benchmark` to render with the
standalone voice defaults.

## For maintainers: regenerate manifest pins

Subset digests and binding revisions are derived from the checkout, never
written by hand. After any pack, map, task or prompt change:

```bash
tau2 benchmarks list                  # installed manifests, modes and sizes
tau2 benchmarks pin voice             # check voice.json; fails if a pin changed
tau2 benchmarks pin voice --revision regional-release-data-v17   # repin under a new revision
tau2 benchmarks pin text --revision regional-release-data-v17
```

`pin` is idempotent. A changed pin means results saved under the old revision no
longer validate, so `pin` refuses to write changed pins without a new
`--revision`. The current revision of `text.json` and `voice.json` is
`regional-release-data-v16`; `voice-v1.json` is `english-base-v1`.

## Python callers

Python callers use the same resolver as the CLI:

```python
from pathlib import Path
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner.presets import (
    BenchmarkRequest, ExecutionSettings, execute_plan, resolve_benchmark,
)

output = Path("voice-run")
plan = resolve_benchmark(BenchmarkRequest(
    output=output,
    template=VoiceRunConfig(audio_native_config=AudioNativeConfig(reasoning_effort="low")),
    domains=["retail"],
))
execute_plan(plan, output, ExecutionSettings(workers=1), dry_run=True)
```

For text, pass a `TextRunConfig` template with `preset="text"`. `VoiceRunConfig`
and `run_domain()` on their own describe one language/domain cell; only a
`BenchmarkRequest` expands into the matrix. Explicit LiveKit pipelines go through
`CascadedConfigRequest(config=...)`. See [Running simulations](running_simulations.md)
for the lower-level runner API.
