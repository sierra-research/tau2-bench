# CLI Reference

The `tau2` command provides a unified interface for all τ-bench functionality. Use `tau2 <command> --help` to see full details for any command.

Run `tau2 intro` (or just `tau2`) to see an overview of available domains, commands, and a quick-start guide directly in the terminal.

## `tau2 run` — Run Evaluations

Run agent evaluations across different communication modes.

### Basic Usage

```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --num-tasks <task_count>
```

### Common Options

| Option | Description |
|--------|-------------|
| `--domain`, `-d` | Domain to evaluate: `airline`, `retail`, `telecom`, `mock`, `banking_knowledge` |
| `--agent-llm` | LLM model for the agent |
| `--user-llm` | LLM model for the user simulator |
| `--agent-llm-args` | JSON dict of extra args for agent LLM (e.g. `'{"temperature": 0.5}'`) |
| `--user-llm-args` | JSON dict of extra args for user LLM |
| `--agent` | Agent implementation to use (default: `llm_agent`) |
| `--user` | User simulator implementation to use (default: `user_simulator`) |
| `--num-trials` | Number of evaluation trials (default: `1`) |
| `--num-tasks` | Number of tasks to evaluate (omit for all tasks) |
| `--task-ids` | Specific task IDs to evaluate |
| `--task-split-name` | Task split to use (default: `base`) |
| `--task-set-name` | Task set to use (default: domain default) |
| `--max-steps` | Maximum simulation steps (default: `200`) |
| `--max-errors` | Maximum consecutive tool errors allowed (default: `10`) |
| `--max-concurrency` | Maximum concurrent simulations (default: `3`) |
| `--workers` | Worker processes; each holds up to `--max-concurrency` simulations (`0` keeps in-process execution) |
| `--provider-limit` | Per-provider cap in controller mode, e.g. `openai=40,gemini=20` |
| `--seed` | Random seed for reproducibility (default: `300`) |
| `--save-to` | Custom output directory name (saved under `data/simulations/`) |
| `--log-level` | Log level (default: `ERROR`) |
| `--verbose-logs` | Save detailed logs (LLM calls, audio, ticks) |
| `--audio-debug` | Save per-tick audio files and timing analysis (requires `--audio-native`) |
| `--llm-log-mode` | LLM log mode when `--verbose-logs` is on: `all` or `latest` (default: `latest`) |
| `--max-retries` | Max retries for failed tasks (default: `3`) |
| `--retry-delay` | Delay in seconds between retries (default: `1.0`) |
| `--enforce-communication-protocol` | Enforce protocol rules (e.g. no mixed text + tool call messages) |
| `--llm-communicate-judge`, `--no-llm-communicate-judge` | Pin `communicate_info` scoring to the semantic LLM judge or exact matching; default is semantic for non-English and exact for English |
| `--user-persona` | User persona config as JSON dict |
| `--xml-prompt` | Force XML tags in system prompt |
| `--no-xml-prompt` | Force plain text system prompt (no XML tags) |
| `--auto-resume` | Automatically resume from existing save file without prompting |
| `--auto-review` | Automatically run LLM conversation review after each simulation |
| `--review-mode` | Review mode when `--auto-review` is on: `full` or `user` (default: `full`) |
| `--review-model` | LLM model to use for review calls (default: `claude-opus-4-5`) |
| `--hallucination-retries` | Max retries when user simulator hallucination is detected (full-duplex only, default: `3`). Set to `0` to disable |
| `--timeout` | Maximum wallclock time in seconds per simulation (default: `2400`; pass `0` to disable) |
| `--audio-native` | Enable audio native mode (voice full-duplex) |
| `--audio-taps` | Save WAV files at each pipeline stage for debugging (requires `--audio-native`) |
| `--retrieval-config` | Retrieval configuration for `banking_knowledge` domain (e.g., `alltools`, `bm25`, `terminal_use`) |
| `--retrieval-config-kwargs` | JSON arguments for the retrieval config constructor (e.g., `'{"top_k": 10}'`) |
| `--text-input-style` | Text-mode caller typing style arm: `native_script`, `romanized`, `diacritic_free`, `code_mixed`. Non-default styles require a language pack that declares them (text only, incompatible with `--audio-native`) |
| `--text-noise` | Arm deterministic noisy-text entity probes: the caller's first conveyance of each pinned entity (user id / name / phone / email / date) is corrupted; clean truth is recorded in provenance (text only) |
| `--text-noise-seed` | Seed for the noisy-text corruption draw (default: `42`) |

### Audio Native Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio-native` | `false` | Enable audio native mode |
| `--audio-native-provider` | `openai` | Provider: `openai`, `gemini`, `xai` |
| `--audio-native-model` | *(per-provider)* | Model to use (defaults to provider-specific model if not set) |
| `--reasoning-effort` | *(per-provider)* | `minimal`, `low`, `medium`, `high`, `xhigh`, or `provider_default` (send no reasoning setting). Not every level is valid for every provider — `xhigh` is OpenAI-only, xai/nova/qwen/livekit take none (`SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS`), and an unsupported pair is rejected at config time. Defaults to the provider's entry in `DEFAULT_AUDIO_NATIVE_REASONING_EFFORT`; the effective value is recorded in `results.info`. Pin it — see AGENTS.md |
| `--tick-duration` | `0.2` | Tick duration in seconds (simulation timestep) |
| `--max-steps-seconds` | `600` | Maximum conversation duration in seconds |
| `--speech-complexity` | `regular` | Speech complexity: `control`, `regular`, or ablation variants (`control_audio`, `control_accents`, `control_behavior`, `control_audio_accents`, `control_audio_behavior`, `control_accents_behavior`) |
| `--pcm-sample-rate` | `16000` | User simulator PCM synthesis rate |
| `--telephony-rate` | `8000` | API/agent telephony rate |

**Turn-taking thresholds:**

| Option | Default | Description |
|--------|---------|-------------|
| `--wait-to-respond-other` | `1.0` | Min seconds since agent spoke before user responds |
| `--wait-to-respond-self` | `5.0` | Min seconds since user spoke before responding again |
| `--yield-when-interrupted` | `1.0` | How long user keeps speaking when agent interrupts |
| `--yield-when-interrupting` | `5.0` | How long user keeps speaking when interrupting agent |
| `--interruption-check-interval` | `2.0` | Interval for checking interruptions |
| `--integration-duration` | `0.5` | Integration duration for linearization |
| `--silence-annotation-threshold` | `4.0` | Silence threshold for annotations |

### Examples

```bash
# Standard text evaluation
tau2 run --domain airline --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 1 --num-tasks 5

# Audio native (voice full-duplex)
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs

# Audio native with custom provider and settings
tau2 run --domain retail --audio-native --audio-native-provider gemini \
  --tick-duration 0.2 --max-steps-seconds 240 --speech-complexity control \
  --verbose-logs --save-to my_audio_native_run

# Audio native with hallucination retries disabled
tau2 run --domain retail --audio-native --hallucination-retries 0 --num-tasks 1

# Knowledge retrieval with BM25
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-tasks 5

# Knowledge retrieval with embeddings and reranker
tau2 run --domain banking_knowledge --retrieval-config openai_embeddings_reranker \
  --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-tasks 5
```

### `tau2 run refill` — Re-run individual simulations of an existing run

Replace named simulations in a run directory (a hung call, a corrupt one, an
error) **on the configuration that directory records**, not on flags retyped
from memory. The run config is read back from `results.json`'s `info` block —
domain, task set, caller persona, provider, reasoning effort, seed, timeout,
scoring axes — the named simulations and their index entries are deleted, and
the run resumes into the same directory, which fills exactly those cells.

```bash
# Everything the refill needs beyond the run itself: which cells, how fast
tau2 run refill --save-to multilingual/preference_runs_v1_ko/ko_telecom_openai_high \
  --task-ids '[mobile_data_issue]...' --trials 0 --dry-run
```

| Option | Description |
|--------|-------------|
| `--save-to` | The run: its directory, its `results.json`, or the `--save-to` name it was launched with |
| `--task-ids` | Task id(s) to delete and re-run |
| `--trials` | Trial indices, 0-based (default: every trial of those tasks) |
| `--max-concurrency` | Concurrent simulations — the only knob no result records |
| `--user-persona-id`, `--task-set-name` | Pre-recording escapes (see below) |
| `--dry-run` | Print the rebuilt config and the affected cells, delete nothing |

`--dry-run` first: it prints the rebuilt config, the simulations that would be
deleted (with their rewards), any **pre-existing gaps elsewhere in the
directory** that the resume will also fill, and any knob the checkpoint is too
old to record. Directories written before the run config was recorded in full
can supply those knobs with `--user-persona-id`, `--task-set-name`,
`--verbose-logs`, `--timeout`, `--scores` and `--nativeness-llm-judge`; each
is refused when the checkpoint does record the field, so a refilled call cannot
differ from the calls beside it.

```bash
# A pre-#530 pool directory: it records neither its caller nor its task set,
# and the refill refuses to guess either
tau2 run refill --save-to preference_runs_v1_korean_telecom/ko_telecom_gemini_high \
  --task-ids '[mobile_data_issue]...' --trials 0 \
  --user-persona-id ko --task-set-name telecom_ko_identity --dry-run
```

Refusals happen before anything is deleted — an unknown task id, a trial
outside the grid, a run whose caller persona no longer resolves (the guard that
makes a localized task set without `--user-persona-id` fatal), a caller that
contradicts the personas the directory's own calls were spoken by, or a task
set that does not hold the directory's tasks. The last one is what an
unrecorded task set looks like: the refill would otherwise fall back to the
domain's own English set, and the error names the set that does hold them.

---

## `tau2 pool` — Run a Fixed Experiment Grid

A registered pool is a fixed language × arm collection of ordinary runs. The
driver builds one typed `RunConfig` per selected cell and submits the complete
list to `run_domains()` once, so all simulations share one worker fleet and one
set of provider limits.

```bash
tau2 pool list
tau2 pool status preference_v1
tau2 pool run preference_v1 --workers 8 \
  --provider-limit openai=40,gemini=40
```

Each task/trial unit receives the controller's whole-unit retry. If both
attempts fail, the cell records an `infrastructure_error` and the pool command
still finishes: 100 terminal results may mean 99 usable plus one recorded
failure. Re-running the same pool later uses normal checkpoint resume, which
drops that failure and runs only the missing unit. There are no census rounds,
cell subprocesses, host process scans, or automatic dedup repairs.

`--langs` and `--only` select cells; `--dry-run` prints their typed configs.
`tau2 pool status` remains a read-only usable/coverage census and flags recorded
infrastructure errors or legacy duplicates.

---

## `tau2 play` — Interactive Play Mode

Experience τ-bench interactively from either perspective.

```bash
tau2 play
```

Play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools like telecom)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

See the [Gym Documentation](../src/tau2/gym/README.md) for using the gymnasium interface programmatically.

---

## `tau2 view` — View Results

Browse and analyze simulation results.

```bash
tau2 view
```

| Option | Description |
|--------|-------------|
| `--dir` | Directory containing simulation files (defaults to `data/simulations/`) |
| `--file` | Path to a specific results file to view |
| `--only-show-failed` | Only show failed tasks |
| `--only-show-all-failed` | Only show tasks that failed in all trials |
| `--expanded-ticks` | Show expanded tick view (for full-duplex simulations) |

---

## `tau2 domain` — View Domain Documentation

View domain policy and API documentation.

```bash
tau2 domain <domain>
```

Then visit http://127.0.0.1:8004/redoc to see the domain policy and available tools.

---

## `tau2 check-data` — Check Data Configuration

Verify that your data directory is properly configured.

```bash
tau2 check-data
```

---

## `tau2 start` — Start All Servers

Start all domain servers.

```bash
tau2 start
```

---

## `tau2 evaluate-trajs` — Evaluate Trajectories

Re-score stored results: reward via the evaluator; nativeness and delivery
through the judges machinery (complete stored verdicts are reused, gaps and
ERROR verdicts are re-judged; delivery is judged from the stored disk audio).
Rescored results are saved back in their original storage format.

```bash
tau2 evaluate-trajs <paths...>
```

| Option | Description |
|--------|-------------|
| `<paths>` | Paths to trajectory files, directories, or glob patterns |
| `-o`, `--output-dir` | Directory to save rescored results as `updated_<name>`. Default: update each input in place |
| `--scores` | Comma-separated axes to recompute: `reward,nativeness,delivery` (or `all`) |
| `--nativeness-llm-judge` | Run the LLM judge for judge-type nativeness factors. Off by default: deterministic checkers only, LLM factors recorded DEFERRED |

---

## `tau2 review` — LLM Conversation Review

Run LLM-based review on simulation results to detect agent and/or user errors.

```bash
tau2 review <path>
```

| Option | Description |
|--------|-------------|
| `<path>` | Path to a `results.json` file or directory containing them |
| `-m`, `--mode` | Review mode: `full` (agent + user, default) or `user` (user simulator only) |
| `-o`, `--output` | Output path for reviewed results (single file only) |
| `--interruption-enabled` | Flag indicating interruption was enabled in these simulations |
| `--show-details` | Show detailed review for each simulation |
| `-c`, `--max-concurrency` | Max concurrent reviews (default: `32`) |
| `--limit` | Limit review to first N simulations |
| `--task-ids` | Only review simulations for these task IDs |
| `--log-llm` | Log LLM request/response for each review call |
| `--review-model` | LLM model to use for review calls (default: `claude-opus-4-5`) |

---

## `tau2 convert-results` — Convert Results Format

Convert simulation results between monolithic JSON and directory-based formats.

```bash
tau2 convert-results <path> [--to {json,dir}] [--no-backup]
```

| Option | Description |
|--------|-------------|
| `<path>` | Path to a `results.json` file or directory containing one |
| `--to` | Target format: `json` (monolithic) or `dir` (directory with individual sim files). If omitted, converts to the opposite of the current format |
| `--no-backup` | Skip creating a backup before conversion |

Text runs default to monolithic JSON; voice runs default to directory-based format. Use this command to convert between them when needed.

---

## `tau2 leaderboard` — View Leaderboard

Show the τ-bench leaderboard in the terminal.

```bash
tau2 leaderboard
```

| Option | Description |
|--------|-------------|
| `--domain`, `-d` | Show leaderboard for a specific domain: `retail`, `airline`, `telecom`, or `banking_knowledge` |
| `--metric`, `-m` | Metric to rank by: `pass_1`, `pass_2`, `pass_3`, `pass_4`, `cost` (default: `pass_1`) |
| `--limit`, `-n` | Limit the number of entries shown |

---

## `tau2 submit` — Leaderboard Submission

See the full [Leaderboard Submission Guide](leaderboard-submission.md).

```bash
# Prepare a submission
tau2 submit prepare <paths...> --output ./my_submission

# Prepare a voice submission (auto-detected, or force with --voice)
tau2 submit prepare <paths...> --output ./my_submission --voice

# Skip trajectory verification during preparation
tau2 submit prepare <paths...> --output ./my_submission --no-verify

# Validate a submission
tau2 submit validate <submission_dir> [--mode public|private]

# Verify trajectory files
tau2 submit verify-trajs <paths...> [--mode public|private]
```

---

## `tau2 factory` — Language Factory

Build and maintain multilingual language packs end to end (see the full
[Adding a Language guide](multilingual/ADDING_A_LANGUAGE.md)). Owned by
`src/tau2/multilingual/`.

```bash
tau2 factory autoform --lang ro --out form_ro.md   # LLM-draft the author form
tau2 factory draft --lang ro --form form_ro.md     # draft pack + guidelines
tau2 factory draft-nativeness --lang ro            # write review packet; no pack edit
# Review _factory/ro/nativeness_review.yaml, then set review_status: approved
tau2 factory apply-nativeness --lang ro            # add the approved live rubrics
tau2 factory finalize --lang ro                    # promote draft to data/tau2/multilingual/ro/
tau2 factory generate-assets --lang ro             # design + pin persona voices (voice-only)
tau2 factory validate --lang ro                    # deterministic guardrails
```

The production audio-quality command has no language- or rater-specific prompt
switches. Model overrides remain available for controlled experiments, and any
override changes the recorded judge contract and cache key.

| Subcommand | Description |
|------------|-------------|
| `autoform` | LLM-generate an author form (the seed for `draft`) from a language identity |
| `draft` | LLM draft of `pack_draft.yaml` + guidelines from an author form |
| `draft-nativeness` | Select applicable ids from the closed nativeness catalog and author only language-specific agent-speech rules/examples into `_factory/<lang>/nativeness_review.yaml`; never edits the pack or overwrites an existing packet |
| `apply-nativeness` | Validate and apply an `approved` nativeness review packet to `pack_draft.yaml`; refuses pending/stale packets and existing different rubric content |
| `localize-entities` | Swap callers to locale identities; emits the `_identity` task-set variant |
| `finalize` | Promote a guardrail-clean draft into `data/tau2/multilingual/<lang>/` |
| `generate-assets` | Design + pin persona voices (ElevenLabs) — voice-only |
| `validate` | Deterministic guardrails over a pack (draft or final) |
| `smoke-assets` | Cheap ElevenLabs API smoke before `generate-assets` |
| `backchannel-eval` | Offline backchannel-density eval against the curated fixture corpus (the retune instrument; per-pack levels are settled) |
| `text-noise-bank` | Render the full noisy-text stimulus bank (clean → corrupted entity pairs per task) for one language/domain/seed for pre-run human review; writes a provenance-stamped JSON artifact |

### Retired factory machinery

The `tau2 factory legacy` verb group was dissolved on 2026-08-03 and its
machinery deleted (recover from git history if ever needed):

| Retired verb | What it did | Why it went |
|--------------|-------------|-------------|
| `legacy translate` | LLM translation loop (translate → verify → fix) | The τ-Multilingual paper set is frozen at five localized languages plus English with verified task sets already shipped; the live flow is English-prose `seed-tasks` |
| `legacy tasks-csv` | Translator CSV round-trip (`extract` / `inject`) | Same retirement; `localize_lib` keeps the row iteration and inject gate |
| `legacy parity` | Behavioral translation verification via paired probes | Retires with the translation loop it verified |
| `legacy redraft-pragmatics` | Conditional-realization rewrite of `pragmatics_clauses` | Completed one-shot migration; all shipped packs carry the marker |
| `legacy draft-native-strings` | Translate the prompt scaffold into a pack sidecar | The native prompt-language arm is gone; sidecars deleted |
| `legacy repin-variety` | Re-pin a persona to another language variety | Both re-pins (es, tr) are complete |
| `legacy generate-beds` | Locale background beds for an opted-in pack | Every language runs on shared stock-English acoustics; no pack opted in |
| `legacy probe-coverage` | Probe a provider's native speech-output coverage | The language set is frozen; only relevant if a new language is scouted |

Two verbs survived the dissolution by moving:

- `legacy backchannel-eval` → **`tau2 factory backchannel-eval`**
- `legacy translation-review` → **`tau2 annotate translation-review`** (reads
  archived translation rows; filled sheets still ingest via `tau2 annotate ingest`)

---

## `tau2 judges` — Post-hoc Judging

Run or export nativeness/delivery judges over stored results. Owned by
`src/tau2/judges/`. `rejudge` and `export` reuse complete stored verdicts by
default and only fill gaps; `--rejudge` forces a full re-judge.

```bash
tau2 judges suite <results...>                   # one-pass calibrated judging:
                                                 # nativeness + quality + delivery
tau2 judges rejudge <results...>                 # fill verdict gaps in place
tau2 judges export <results...> --out verdicts/  # typed JSONL verdict records
tau2 judges lexical-stats <results...>           # deterministic MATTR report
tau2 judges tau-multi-naturalness prepare \
    --repo-root /path/to/reviewer-repository      # offline paper-cohort preflight
tau2 judges tau-multi-naturalness run \
    --repo-root /path/to/reviewer-repository \
    --out /path/to/judge-sidecars                 # resumable frozen trial-0 replay
tau2 judges tau-multi-validation \
    data/simulations/paper_runs/tau-multi/human_annotations
tau2 judges legacy conversation <results...> \
    --out conversation.json                      # EVA progression + conciseness
                                                 # (composites in shadow_scores)
tau2 judges legacy quality <results...>          # universal binary quality rubric
```

| Subcommand | Description |
|------------|-------------|
| `suite` | One-pass annotation-calibrated judging over stored results (paths only): a text phase (nativeness + quality, `--text-processes` × `--concurrency` in flight, default 8×10) then an audio phase (per-utterance delivery from disk audio, `--audio-processes` × `--concurrency`, default 5×10, every audio-bearing call — no sampling). Gap-filling with version-aware verdict reuse (`--rejudge` forces a full pass); judges trial 0 only by default (`--trials 0,1` / `--trials all` opt in); `--only nativeness,quality,delivery` selects families; en cells silently skip nativeness. Verdicts persist back in the results' original format and each root gets a typed `judge_suite_provenance.json` (per family: judge model, prompt/rubric versions, invocation flags/trials/timestamp, touched sims), merged on write — a later `--only` pass updates its families' entries and keeps the rest |
| `rejudge` | Run the nativeness judge over stored results and save in their original format; `--delivery` also runs the per-utterance delivery judge from disk audio |
| `adopt-verdicts` | Copy current-prompt-version judge verdicts from already-judged sims into results whose verdicts are stale or missing, matched by sim id (never invokes a judge) |
| `export` | Export typed JSONL verdict records (quality + nativeness + delivery + factor rubrics); newly judged verdicts are persisted back to the results |
| `lexical-stats` | Deterministic run-level lexical-diversity report (MATTR over pooled agent speech; within-language comparisons only) |
| `tau-multi-naturalness prepare` | Offline preflight of the canonical τ-Multilingual non-English trial-0 cohort: verify the Experience-manifest roots and hashes, then report the exact simulations and utterance work without making LLM calls |
| `tau-multi-naturalness run` | Run or resume the frozen combined utterance-naturalness judge (`gpt-5.5`, `xhigh`) over that cohort at up to `--max-concurrency` calls (default 100); writes deterministic per-utterance and call-level sidecars without modifying frozen simulations |
| `tau-multi-validation` | Offline verification of the canonical human-annotation archive: nested language-and-measure validation leaves, stored prompt contracts, file hashes, derived precision/recall/F1/κ, first-critical-error reviews, and supporting native-speaker reviews |

### Legacy judge verbs

The `tau2 judges legacy` group holds demoted verbs — fully usable, explicitly
invoked, out of the standard judging flow (the same demotion pattern as the
former `tau2 factory legacy` group):

| Legacy verb | Description |
|-------------|-------------|
| `legacy conversation` | LLM conversation judges: EVA-ported progression per dimension + per-turn conciseness failure modes into a typed resumable sidecar; the EVA-X conjunction and the τ quality composite are computed into `shadow_scores` only; see [Conversation judges](conversation-judge.md) |
| `legacy quality` | Compute the universal binary quality rubric over stored results (deterministic-only by default; `--llm-judge` adds the task-aware semantic factors) |

The calibrated audio-quality reference judge (`audio-quality-run` /
`audio-quality-calibrate`) was removed outright (owner ruling 2026-08-27): the
per-utterance delivery judge, run via `rejudge --delivery`, is the canonical
audio judge everywhere.

---

## `tau2 annotate` — Annotation Artifacts

Build human-facing annotation artifacts (sheets, workbooks, HTML packets) and
ingest filled ones back into calibration metrics. Owned by `src/tau2/annotation/`.

```bash
tau2 annotate nativeness <results...> --lang es        # judge-calibration sheets
tau2 annotate packets --results <dirs...> --out pkt/   # standalone HTML packet
tau2 annotate ingest <filled.csv>                      # ingest any filled sheet
tau2 annotate agreement --lang es                      # persisted agreement summaries
```

| Subcommand | Description |
|------------|-------------|
| `nativeness` | Export nativeness-judge calibration sheets from results |
| `workbook` | Build formatted `.xlsx` workbooks from exported nativeness sheet CSVs |
| `localization-sheet` | Side-by-side task review workbook with approve/deny dropdowns per language |
| `audit` | Nativeness Audit sheet bodies (`--generate` for LLM nuance candidates) |
| `packets` | Build a standalone HTML annotation packet (pages + audio + index + manifest) |
| `prompt-bed-packet` | One language's runtime user-sim prompts + background-bed review packet |
| `ingest` | Ingest any filled annotation sheet (dispatches on the artifact manifest) |
| `feature-table` | Extract the deterministic per-call feature table from run results |
| `translation-review` | Export the translation-review sheet family from archived rows |
| `agreement` | A language's persisted calibration agreement summaries |
| `source-audit` | Check every packet manifest still points at the run it was built from; `--apply` repairs runs that moved |

A packet's provenance records each source run as a path **plus** a
content-derived run identity, so `source-audit` resolves by identity: a moved
run is `relocated` (repairable), a run re-executed to the same `--save-to` is
`replaced`, and a run that is gone is `missing`. Only `relocated` paths are
ever rewritten.

---

## `tau2 metrics` — Deterministic Instruments

Deterministic instruments over stored runs — no LLM calls anywhere. The fact
columns are the headline; every composite score is a permanent shadow column
(labeled `uncalibrated-shadow`, never ranked or headlined). Every verb writes
one provenance-stamped JSON artifact (content-derived `artifact_id`, git sha,
config).

```bash
tau2 metrics interaction-facts <run-dirs...> -o facts.json [--langs es pt] [--no-turns]
tau2 metrics caller-cost <run-dirs...> -o caller_cost.json [--langs es pt] [--domain telecom]
tau2 metrics entity-trace <run-dirs...> -o entity_trace.json [--no-events]
tau2 metrics event-inference-audit <run-dirs...> -o event_audit.json [--no-events]
tau2 metrics paired-power <run-dirs...> -o paired_power.json --arm-a gemini --arm-b openai
```

| Verb | What it computes |
|------|------------------|
| `interaction-facts` | The facts-first interaction suite: per-call deterministic facts (latencies, missed responses, interruption behavior, route mix, τ-voice selectivity, dead air) with typed N/A-with-reason columns, per language×domain×provider×reasoning-effort cell aggregates, and the EVA turn-taking composite demoted to `shadow_scores`; see [Voice Interaction Metrics](interaction-metrics.md) |
| `caller-cost` | Per-call "interaction tax" counters (re-dictations, repeats, barge-ins, milestone timings) + per language×domain×provider cell aggregates |
| `entity-trace` | Per-entity capture / tool-argument tracing with the fabrication gate |
| `event-inference-audit` | E8(a): EVA-style post-hoc event inference (audio-only overlap/duration heuristics) scored against the orchestrator's typed turn-taking ground truth — per-language confusion matrices, precision/recall, and the induced EVA-X score error per call |
| `paired-power` | E8(b): bootstrap CI on a provider reward gap computed stem-paired vs unpaired at the same n; reports the CI-width ratio per language×domain cell and pooled |

`paired-power` selects arms by run provenance: `--arm-a provider` or
`--arm-a provider:effort` (e.g. `openai:xhigh`). The bootstrap is seeded
(`--seed`, default 1234) and per-cell streams derive from the seed, so
artifacts are reproducible and order-independent.

---

## `tau2 run-preset` — Benchmark Presets & Matrix Runs

Run a language's benchmark preset, or the full language × provider matrix.

```bash
tau2 run-preset --list                              # available presets
tau2 run-preset multilingual_v1_spanish --stage full
tau2 run-preset --all-languages --providers openai,gemini --dry-run
```

| Option | Description |
|--------|-------------|
| `--list` | List available presets (generated from the registered language packs) |
| `--stage` | `smoke` (1 task per arm) or `full` (preset mode) |
| `--arm` | Run only one arm of the preset |
| `--all-languages` | Matrix mode: every registered language pack across `--providers` |
| `--providers` / `--lanes` | Matrix providers and concurrent-lane cap |
| `--dry-run` | Print the `tau2 run` commands without executing |

Preset runs land under `data/simulations/multilingual/<preset>/<stage>_<arm>/`
— one subtree for the whole multilingual experiment. Matrix mode
(`--all-languages`) writes to its own dated base instead.

---

## `tau2 paper` — Paper Reproduction

Offline reproduction and evidence-export commands for the τ-Multilingual
reviewer release.

```bash
tau2 paper multilingual-audit --evidence-root /path/to/tau-multi --out /tmp/audit
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts
tau2 paper multilingual-listening-sample \
  --evidence-root /path/to/tau-multi --out /tmp/listening
tau2 paper multilingual-verify-ablation-transcripts \
  --root papers/tau-multilingual/reproduction/retail_ablation_transcripts
```

| Verb | Purpose |
|------|---------|
| `multilingual-audit` | Validate the frozen result cohort and reproduce the paper's core tables |
| `multilingual-prompt-snapshots` | Build a content-addressed archive of exact recorded prompt inputs and historical runtime language-pack values |
| `multilingual-verify-prompts` | Verify the checked-in prompt archive; with `--evidence-root`, compare every input against its frozen source |
| `multilingual-listening-sample` | Export the deterministic three-calls-per-language listening sample |
| `multilingual-experience` | Recompute the Interaction, latency, and utterance-level Experience analysis artifact |
| `multilingual-annotations` | Export the final precision and recall annotation results |
| `multilingual-ablation-transcripts` | Export compact transcripts for the 360-call Hindi and Mandarin retail localization ablation |
| `multilingual-verify-ablation-transcripts` | Verify the checked-in ablation transcript export offline |

---

## Environment CLI (beta)

An interactive CLI for directly querying and testing domain environments.

```bash
make env-cli
```

**Commands:**
- `:q` — quit
- `:d` — change domain
- `:n` — start new session (clears history)

**Example:**
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
```

Useful for testing domain tools, debugging environment responses, and exploring domain functionality without starting the full server stack.

---

## Running Tests

```bash
make test              # Core tests (requires: uv sync --extra dev)
make test-voice        # Voice + streaming tests (requires: uv sync --extra dev --extra voice)
make test-knowledge    # Banking knowledge tests (requires: uv sync --extra dev --extra knowledge)
make test-gym          # Gymnasium tests (requires: uv sync --extra dev --extra gym)
make test-all          # All tests (requires: uv sync --all-extras)
```

---

## Advanced: Ablation Studies

The `telecom` domain supports ablation studies for research purposes.

### No-user mode

The LLM is given all tools and information upfront (no user interaction):

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-5.4-mini \
  --user dummy_user
```

### Oracle-plan mode

The LLM is given an oracle plan, removing the need for action planning:

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-5.4-mini \
  --user-llm gpt-5.4-mini
```

### Workflow policy format

Test the impact of policy format using the workflow policy for telecom:

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-5.4-mini \
  --user-llm gpt-5.4-mini
```
