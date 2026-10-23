# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- voice-v2 packages carry `benchmark-quality.json` (schema
  `schema_benchmark_quality.json`): call-level Interaction
  (`voice-v2-interaction-call-v1`) and Experience, shown as Generation quality
  (`voice-v2-experience-call-v1`), for every scored call. Each record has the
  five Interaction flags (nonresponse, interruption over 3 events, selectivity
  over 3 errors, monologue, tool misuse) and the Experience utterance and issue
  counts. Each cell sums its pass/fail/unscored counts. `tau2 submit prepare`
  and the historical adapter write it from the saved ticks and judge verdicts
  without model calls. `tau2 submit validate` recomputes it and rejects any
  difference. Calls that cannot be decided are unscored, never failed. Banking
  and entity packages carry none. The deterministic turn-taking measurements
  are in `tau2.metrics.interaction_quality` and the rules in
  `tau2.metrics.call_quality`. See [docs/glossary.md](docs/glossary.md).
- The maintainer-only historical adapter (`tau2 submit prepare/validate
  --historical-voice-v2`) also packages the frozen 2026-10-02 release export's
  banking and entity (intake) calls: `--benchmark banking` builds a
  `voice-v2-banking` package and `--benchmark entity` a `voice-v2-entity` one
  (default: `voice`). Those calls predate their manifests, so the adapter
  checks each recorded binding against the released cell and accepts only
  three verified `runtime_rules_sha256` digests. Neither frame is judged, so
  neither takes `--judges-root`. Validation reads the frame from the package.
  See [docs/voice-v2-publication.md](docs/voice-v2-publication.md).
- 2.0 package names: `voice-v2-banking` and `voice-v2-entity` packages are
  named `{model_name}_{model_organization}_{submission_date}_banking` /
  `_entity`. One agent's voice-v2, banking and entity packages share their
  metadata so the leaderboard can group them, and without the suffix they would
  collide. voice-v2 and text-v1 names are unchanged.

### Changed
- 2.0 scoring excludes infrastructure-errored calls from the denominator
  instead of blocking the cell. A cell is complete once every owed call is
  scored or errored and at least one is scored; its score averages the scored
  calls only, and a task whose calls all errored drops out of the cell mean.
  Errored calls stay reported as `error_trials` and are no longer listed as
  gaps by `tau2 submit prepare`. Missing and unscored calls, and cells whose
  calls all errored, still block a submission.
- Judge diagnostics treat missing Live transcripts as recording gaps, not
  judge failures. A verdict whose only unit errors are missing-Live-input
  errors counts as judged (blocked units stay out of pass fractions and are
  reported as `input_error_units`); a call a measure cannot judge for that
  reason is excluded and reported as `input_excluded_calls`. Other judge
  errors still block, as does a measure whose applicable calls are all
  excluded. `VerdictFingerprint` gains `input_error_units`, so voice-v2
  packages prepared before this change must be re-prepared.

## [2.0.0] - 2026-10-19

> **⚠️ Scoring change — scores are not comparable with 1.x.** Communication checks,
> NL-assertion judging, tool-argument lookup and the English prompts all changed.
> Re-run rather than compare 1.x results against 2.0 numbers.

### Upgrading from 1.x
- 2.0 English runs are not input-identical to 1.0.1. The differences:
  - English retail and telecom each use 50 reviewed task replacements (same IDs).
  - User-simulator prompts add a language clause and an entity-order rule
    (`language-guidelines-v4`).
  - The text user simulator gets a `## Persona Guidelines` block.
  - Communication checks are semantic (LLM-judged) by default.
  - The NL-assertion judge is `gpt-5.4-mini-2026-03-17` (was `gpt-4.1`).
  - Name, ID and phone lookups fold case, accents and formatting.
  - Voice benchmark defaults: user simulator `openai/gpt-5.5` at
    `reasoning_effort: low` (was `gpt-4.1`), `max_steps_seconds` 2400 (was
    1200) and OpenAI agent voice `marin` (was `alloy`).
- For scores comparable to the 1.x leaderboard, run the `v1.0.1` tag. 1.x and
  2.0 results are shown on separate leaderboard tabs, and 1.x submissions are
  still accepted.
- No flag combination reproduces 1.0.1 inputs exactly.
- Closest text run: `--communicate-scorer substring`. Still different: the
  English retail/telecom task replacements, the persona block and language
  clause in the user prompt, the NL-assertion judge model, and folded ID lookups.
- Closest voice run: `--audio-native --benchmark voice-v1 --user-llm gpt-4.1
  --user-llm-args '{"temperature":0.0}' --max-steps-seconds 1200 --voice alloy`.
  Still different: the v2 agent prompt, the spelling, persona and greeting
  changes, semantic communication checks, the new judges and folded ID lookups.
- For a custom run without a released manifest, pass `--benchmark none`. To
  keep the 1.x Gemini model, pass
  `--audio-native-model gemini-3.1-flash-live-preview`.
- English LiveKit STT/TTS resolution cannot be changed from the CLI (an explicit
  `CascadedConfig` in Python can still select Deepgram Aura).
- Results saved by 1.x cannot be resumed; re-run into a new output path.

### Added
- Released benchmark manifests `voice-v2-banking` (`--benchmark banking`:
  `banking_knowledge`, all 97 tasks) and `voice-v2-entity` (`--benchmark entity`:
  agent-directed `intake`, all 200 tasks), voice-only in all six languages
  through the `banking-dates-v1` and `intake` pack extensions. Banking runs use
  the `alltools` retrieval config. Both score outside the voice-v2 Overall.
- `tau2 judges suite --processes N --concurrency N --preserve-errors` judges
  saved runs in parallel (defaults stay 1/1).
- Crossed Pass³ `--regular-results` reuses a validated seed-42 regular run
  read-only; the report records source commits and lists infrastructure
  errors separately.
- 2.0 leaderboard packages: `tau2 submit prepare <matrix-root> --metadata-file
  submitter.json --output DIR` writes `DIR/<model>_<org>_<date>/` with
  `benchmark-submission.json`, `benchmark-diagnostics.json` (voice-v2) and
  `benchmark-metadata.json` (schema `schema_benchmark_metadata.json`). Packages are
  listed under the leaderboard manifest key `voice_v2_submissions`
  (`voice-v2-banking` and `voice-v2-entity` packages under
  `voice_v2_domain_submissions`). Evidence is uploaded to
  `s3://sierra-tau-bench-public/submissions/<name>/trajectories/<language>/<domain>/`.
- `intake` domain (τ-Entity): exact entity capture on outbound calls, 200
  tasks, agent-directed (voice) by default with a `scaffolded` ablation that
  also runs in text. See `docs/elicitation.md`.
- Multilingual text and voice runs: `--language` selects a language pack
  (`en`, `es`, `pt`, `hi`, `zh`, `ko`) with a localized database and tasks for
  airline, retail and telecom. See `docs/language-packs.md`.
- Released benchmark manifests `voice` and `text` (18 cells each, frozen 50-task
  subsets) and the `tau2 benchmarks list` / `tau2 benchmarks pin` verbs.
- Released benchmark manifest `voice-v1`: the English-only voice frame (airline,
  retail, telecom) on the original registry tasks, for the current voice
  leaderboard. It refuses partial runs (`--num-tasks`, `--task-ids`); use
  `--benchmark none --language en` for samples.
- `tau2 factory` language-authoring verbs (`init-request`, `approve`,
  `refresh-regional-bundle`, …); see `docs/language-authoring.md`.
- `tau2 judges`: post-hoc judge suite (not part of the reward) and the public
  judge-authoring path. Greeting and transfer turns are excluded from judging.
- Regional localization maps for `banking_knowledge` and `intake` in every
  non-English pack, authored through `tau2 factory`.
- Reviewed language-specific naturalness criteria candidates for the judge catalog
  (drafts; not yet active judges).
- `tau2.task_subsets`: frozen canonical task frames and immutable subset draws
  (e.g. `retail_50`) used by the benchmark manifests.
- `TAU2_VOICE_ID_<LANG>_<PERSONA>` environment overrides for persona voices.
- Key-free pytest CI workflow and a `slow` marker for full-bundle tests.

### Changed
- Gemini Live starts a new audio item at each `turn_complete` and completes an
  interaction once, on `IDLE`, so consecutive agent turns are kept separate.
- Crossed Pass³ uses catalog `voice-v2-rcs-v3` with whole-run seeds 42, 401
  and 16319 (was `voice-v2-rcs-v2` with 9401/9402/9403). Earlier results keep
  the `voice-v2-rcs-v2` label.
- The frozen 2026-10-02 voice-v2 release adapter (`tau2 submit prepare
  --historical-voice-v2 --judges-root ...`) now writes the same 2.0 package as
  new runs instead of its own `VoiceV2Submission` format; judge diagnostics
  record partial and missing verdicts as coverage gaps. New runs still must be
  fully judged before `prepare` writes a package.
- Docs for 2.0: a 2.0 section in the README, one page for running the text and
  voice benchmarks (`docs/running-the-benchmark.md`, replacing
  `docs/benchmark-runs.md`), a submission guide covering text-v1, voice-v2,
  voice-v1 and 1.x, a glossary, and `docs/cli-reference.md` generated from
  `--help` by `scripts/generate_cli_reference.py`.
- `tau2 judges suite`/`summarize` and released-benchmark `tau2 run` (dry-run plan
  and final score summary) print compact tables; `--json` restores the full JSON.
  `tau2 judges ... --output FILE` saves the full export.
- CLI startup no longer logs the registry contents, data directory or noise-bank
  counts, so `--help` prints only help.
- **Every run resolves a language pack.** No `--language` is identical to
  `--language en`; English prompts (personas, agent language clause, persona
  gender line) now come from the English pack.
- **English retail and telecom runs use 50 reviewed task replacements each.**
  The English pack replaces 50 of the 114 base tasks per domain with same-ID
  versions, so every English retail and telecom run (text or voice) differs
  from 1.x on those tasks. Retail callers get an explicit first/last-name
  ordering line; telecom callers are renamed from the shared "John Smith" to
  distinct names, with matching customer records in the agent database (their
  emails still read `john.smith@example.com`). Airline is unchanged.
  `tau2 submit` now verifies tasks against the recorded language binding, so
  these runs pass leaderboard verification; 1.x results are still checked
  against the registry tasks.
- **Communication checks are LLM-judged for every run** (`CommunicateScoring`
  is semantic by default; substring matching only when requested). The judge
  model (`gpt-5.5`, `reasoning_effort: low`) and prompt version are saved with
  the run and reused by `tau2 evaluate-trajs`.
- User-simulator guidelines (`language-guidelines-v4`, text and voice) tell the
  caller to keep multi-part entity values such as addresses in the task's order.
- NL-assertion judge is `gpt-5.4-mini-2026-03-17` with `reasoning_effort: low`
  (was `gpt-4.1` at temperature 0).
- Retail user lookups fold case and diacritics and raise on ambiguous matches.
- Under evaluation type `ALL`, communication is scored only when `COMMUNICATE`
  is in the task's `reward_basis`; otherwise `communicate_checks` is `None`.
- Voice user-simulator decision model is `gpt-5.4-mini-2026-03-17` with
  `reasoning_effort: low` (was `gpt-4.1`).
- The Gymnasium environments build simulations through the same text-run path:
  they resolve a language pack (English by default) and accept `language` and
  `communicate_scoring`.
- Saved results record the simulator guideline scaffold version, and benchmark
  pins include it; `tau2 benchmarks pin` refuses changed pins without a new
  `--revision`.
- Voice agent instructions use prompt `voice-instructions-v2`.
- `--audio-native` defaults to `--benchmark voice`; text runs are custom unless
  `--benchmark text` is given.
- `gpt-5*` models are always routed through the OpenAI Responses API.
- Language pack schema v3: mode-neutral packs with a `voice:` section. Clauses in
  `voice.pragmatics_clauses` render only in voice runs; the persona's gender line
  and mode-neutral pragmatics clauses render in text runs too (see below).
- Judge catalog `validated-measures-v4`; judge `call_name`s renamed.
- Persona voice seeds use stable SHA-256 hashing.
- Default OpenAI live model is `gpt-live-1`.
- A language pack may bind a registry domain with `task_metadata_file` alone; the
  English airline binding now uses the registry tasks directly.
- Regional caller emails use the language's reviewed mailbox domain.
- Default Gemini model is `gemini-3.8-live-extended-thinking` (was
  `gemini-3.1-flash-live-preview`). With it, Gemini tools are `NON_BLOCKING` and
  proactive audio stays off. Requires `google-genai>=2.30.1,<3`.
- xAI voice runs pin `grok-voice-think-fast-2.0` (passed as `?model=`) with
  reasoning effort `high`; 1.x used the endpoint's default model.
- Qwen defaults are `qwen3.5-omni-plus-realtime` with voice `Tina` (was
  `qwen3-omni-flash-realtime` with `Cherry`).
- Gemini input audio is sent at 8 kHz (was 16 kHz); transcription language codes
  default to `["en-US"]`.
- Released benchmark runs (`voice`, `text`, `voice-v1`) default the user
  simulator to `openai/gpt-5.5` at `reasoning_effort: low` (1.x: `gpt-4.1` at
  temperature 0) and always auto-resume. Voice benchmark runs default to
  `max_steps_seconds` 2400 (was 1200) and OpenAI agent voice `marin` (was
  `alloy`), and record the OpenAI transcription language (`en` for English, the
  value 1.x hard-coded).
- Voice agent instructions end with "Your gender is <gender>." when the agent
  voice has a known gender (e.g. `marin` → female).
- `VOICE_USER_SIMULATOR_VERSION` is `v1.1` (was `v1.0`).
- English LiveKit (cascaded) runs resolve TTS to ElevenLabs `eleven_flash_v2_5`
  (was Deepgram `aura-asteria-en`) and STT to Deepgram `nova-3` with language
  `en`. They now need `ELEVENLABS_API_KEY` and cost more per character. Explicit
  configs outside the supported STT/TTS models are rejected.
- English voice runs speak with the English pack's personas (`mamadou_diallo`,
  `priya_patil`) instead of the legacy persona pool. Verbosity, interruptions
  and speech inserts now come from the persona, so `--speech-complexity control`
  and `regular` callers both interrupt. Voice-ID overrides are now
  `TAU2_VOICE_ID_<LANG>_<PERSONA>` (e.g. `TAU2_VOICE_ID_EN_PRIYA_PATIL`); the
  legacy `TAU2_VOICE_ID_<PERSONA>` variables no longer affect `tau2 run`.
  `setup_voices` creates the English pack voices and drops `--complexity`.
- English voice greeting is "Hi, thanks for calling. How can I help you today?"
  (was "Hi! How can I help you today?"). The text greeting is unchanged.
- Voice user-simulator prompt: callers spell names, IDs and emails only when
  the agent asks, and are no longer told to say "at"/"dot"/"underscore" or to
  use `[pause]` markers. The 1.x `## MINIMAL VERBOSITY` block (on every voice
  task) is replaced by the pack persona's pragmatics clauses, including a
  voice-only clause about cutting in and finishing the agent's sentences.
- Text user-simulator prompt adds the address-order rule and a
  `## Persona Guidelines` block. Each task is assigned one of the two English
  pack personas (in every domain, including `banking_knowledge`), and the block
  carries its gender line and 5–6 mode-neutral pragmatics clauses. On telecom
  this stacks on top of the task's own `[PERSONA:Easy/Hard]` persona text.
- Airline user, reservation and flight lookups ignore case, surrounding
  whitespace and accents. Telecom phone-number lookups ignore formatting, and
  telecom name lookups ignore case and accents. Ambiguous matches raise.
- `banking_knowledge` task 102 is rewritten: the user volunteers Ember's correct
  age, and the NL assertion no longer names the startups.
- `tau2 run --audio-native` with a domain outside the voice benchmark (e.g.
  `banking_knowledge`) needs `--benchmark none`.
- `--domain` on `tau2 run` is repeatable (`action="append"`), so `args.domain`
  is a list.
- Resuming results saved by 1.x (no recorded language) is refused.
- `evaluate_simulation(...)` requires the keyword-only `communicate_scoring`
  argument.
- Gymnasium environments seed simulations with `300` by default (1.x was
  unseeded).
- The `es`/`hi`/`ko`/`pt`/`zh` retail and airline bundles are rebuilt from the
  reconciled address and entity maps (benchmark manifests pinned as
  `regional-release-data-v16`); native review of the replacement Korean and
  Mandarin names is complete.

### Removed
- `tau2.scripts.leaderboard.voice_v2_submission`, `schema_voice_v2.json` and the
  `--provider` / `--interaction-facts` flags of `tau2 submit prepare`.
- `tau2.agent.base.streaming_utils` and its re-exports from `tau2.agent`; the
  transcript helpers (`extract_message_uuid`, `merge_audio_script_gold`,
  `extract_gold_text`, `format_transcript_comparison`, …) live in
  `tau2.utils.transcript.audio_text`.
- `tau2.voice_config.BACKCHANNEL_PHRASES`; backchannel phrases come from each
  language pack's persona `voice:` section.
- Cascaded and live config models moved to `tau2.data_model.cascaded` and
  `tau2.data_model.live_config`.
- `data/tau2/user_simulator/simulation_guidelines*.md`, replaced by
  `tau2.user.guidelines`.
- `data/tau2/multilingual/en/airline_tasks_en.json` (identical to the registry
  airline tasks).
- `get_global_user_sim_guidelines`, `get_global_user_sim_guidelines_voice` and
  the `GLOBAL_USER_SIM_GUIDELINES_*` path constants; use `tau2.user.guidelines`.
- `DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE` (the NL judge uses `reasoning_effort`).
- Linearization helpers from `tau2.agent.base.streaming` and the
  `tau2.agent.base` exports (`consolidate_messages`, `merge_homogeneous_chunks`,
  `linearize_ticks`, `can_merge_messages`, `append_or_merge_chunk`, …) moved to
  `tau2.utils.transcript.linearization`; `ParticipantTick` and
  `LinearizationStrategy` moved to `tau2.data_model.transcript`.

### Fixed
- GPT-5 calls routed through the LiteLLM Responses bridge silently dropped
  `tool_choice` (including `"required"` for solo agents), and printed LiteLLM's
  "Provider List" banner twice per call.
- A failed or malformed hallucination review no longer turns a scored voice
  call into an infrastructure failure. The review status is recorded
  separately, malformed replies get up to three repair attempts, and a
  reviewed call is checkpointed if saving its result fails.
- `tau2 factory` draft prompts state the scenario language and phrase
  budgets, with bounded repair.
- The CLI no longer warns when `.env` is absent.

### Security
- Patched vulnerable Python dependencies.

- `metrics.break_down_metrics.analyze_reward()` aborted the whole reward analysis
  when a simulation's `reward_basis` named a `COMMUNICATE` or `ENV_ASSERTION`
  component that its `reward_breakdown` did not carry: the values were read by
  direct subscript, guarded only by a `reward_breakdown is not None` check, and
  the surrounding `except Exception` re-raises. An `EnvironmentEvaluator` result
  on an airline task (basis `[DB, COMMUNICATE]`) and an
  `evaluate_simulation(..., EvaluationType.ALL_IGNORE_BASIS)` result on a
  retail/airline/telecom task (that branch fixes `reward_basis` to four components
  while filling the breakdown per task) both produce that shape today. A named
  component that is absent now reports `0` -- the value the `reward_breakdown
  is None` branch already gave -- so no new state enters the report and no column
  is renamed or redefined.

## [1.0.1] - 2026-07-15

> **⚠️ Grading change — banking_knowledge scores are not comparable across this release.**
> The fixes below change how `banking_knowledge` rewards are computed. For the grading-scheme
> fixes, re-grading existing trajectories moves scores **only upward** (no previously-passing
> simulation fails under the new scheme), by up to ~9 points pass^1 depending on the model
> (e.g. GPT-5.5 xhigh: 37.37 → 46.39; GPT-5.4 xhigh: 30.67 → 39.43). One task-data fix
> (task_074, #374) corrects a gold refund value, so trajectories that reproduced the old,
> incorrect refund fail that task under 1.0.1. Scores produced with tau2-bench < 1.0.1 on
> `banking_knowledge` must not be
> compared against scores produced with >= 1.0.1. Old results files can be re-scored with
> `tau2 evaluate-trajs --fresh-tasks <results.json>`. Other domains are unaffected.
>
> Note that most of these fixes were merged to `main` on 2026-07-14/15, ahead of this release
> and without a version bump, so an install from `main` after that window already includes
> them. The last commit with the pre-1.0.1 `banking_knowledge` behavior is
> `b51a6d6`, tagged `pre-v1.0.1` — to reproduce pre-1.0.1 grading, pin it explicitly:
> `pip install git+https://github.com/sierra-research/tau2-bench@pre-v1.0.1`.
> Grading is identical at every earlier 2026 commit of `main` (verified back to `2be6916`,
> April 2026); changes in that window are simulation-side only, so any pin in that range
> re-grades recorded trajectories identically.

### Added
- `tau2 evaluate-trajs --fresh-tasks` flag: re-grade trajectories against the current task
  definitions from the data directory instead of the ones embedded in each results file, so
  shipped task/criteria fixes are picked up when re-scoring old runs

### Changed
- Leaderboard website now fetches submission and trajectory data from S3 (`sierra-tau-bench-public`) instead of serving from GitHub Pages directly
- `banking_knowledge` tasks 077–086 (lost/stolen card scenarios): gold trajectories made
  agent-realizable, e.g. account-listing reads that any agent must perform are now part of
  the golden trajectory instead of counting against the DB-hash comparison (#402)

### Fixed
- **`banking_knowledge` grading: extra read calls no longer zero the reward** (#329). The
  `call_discoverable_agent_tool` wrapper logged *every* call to the `agent_discoverable_tools`
  table, which is part of the DB-hash reward comparison. A single prudent verification read
  not present in the golden trajectory (e.g. `get_all_user_accounts_by_user_id_3847` after
  opening an account, or `get_pending_replacement_orders_5765` after ordering a replacement
  card) therefore zeroed the reward. Write calls are still always logged; read calls are now
  logged only when required by the task's golden trajectory (per-task allowlist), so
  required-read assertions keep discriminating while extra validation reads are ignored.
  This accounts for nearly all of the score movement in this release.
- **`banking_knowledge` grading: DB hashes no longer depend on int-vs-float argument
  spelling** (#397). Numeric tool arguments are normalized (`25` and `25.0` are the same JSON
  number), so deterministic record IDs and DB-state hashes no longer differ based on how the
  caller spelled a number. In the re-graded leaderboard set this fix changed no final scores
  on its own, but it removes a latent source of spurious failures.
- **`banking_knowledge`: bank account transactions are returned in reverse chronological
  order** (#403). The knowledge base (doc_018) documents `get_bank_account_transactions_9173`
  as returning most-recent-first, but the tool returned raw DB insertion order (oldest
  first), making the "dispute the earliest duplicate" tie-breaker in doc_031 unresolvable
  from tool output for the duplicate-charge tasks (issue #371, part a). Output is now sorted
  by date descending with a stable sort; task_085's two identical CityFit rows were swapped
  in the fixture so its gold duplicate id follows the same tie-break convention as
  task_083/084 (no gold action changes). Re-grading confirmed this fix changes no existing
  leaderboard scores.
- Contradictory cash-back rate in the `banking_knowledge` Platinum Rewards knowledge-base
  document (#388)
- **`banking_knowledge` task_074: Light Blue ATM-fee refund under-counted the documented
  free allowance** (#374). The Light Blue Account docs grant two free out-of-network and two
  free foreign ATM withdrawals per month (docs `_004`/`_006`), but the gold credit honored
  only one of each, under-refunding the account ($8.00 instead of the policy-faithful
  $14.50). The gold credit for `chk_ar72c5d8e3_2` is now $14.50 and the task notes enumerate
  all seven Light Blue fee errors. The Light Blue fee descriptions also carried
  answer-leaking annotations — "(SHOULD BE FREE - 1ST OF 2)", "(AFTER 2 FREE)" — that told
  the agent which fees were erroneous instead of making it derive that from the policy docs;
  all twelve are now plain "NON-RHO ATM FEE"/"FOREIGN ATM FEE", matching the statement-style
  descriptions on the other three accounts. The same class of leak was scrubbed from the
  sibling ATM-fee tasks: four "NON-RHO ATM FEE - AFTER FREE ALLOWANCE" rows on task_072's
  and task_073's Light Green accounts (on task_072 the annotation appeared only on the
  legitimate fee, so its presence separated correct fees from erroneous ones). Description
  changes don't affect grading — gold and predicted environments replay from the same
  db.json, and no task text or assertion references these strings. Unlike the grading
  fixes above, this changes a gold value: trajectories that refunded the old $8.00 figure
  fail task_074 under 1.0.1, while policy-faithful $14.50 refunds now pass.
- `tau2 evaluate-trajs` now reproduces live grading when re-scoring trajectories: it applies
  the per-task `read_log_allowlist` for `banking_knowledge` (previously required-read
  assertions silently stopped discriminating on re-grade), and replays historical
  trajectories leniently (`Environment.set_state(strict=False)`) so recorded tool outputs
  that cosmetically predate current tool code — e.g. `deposit_check_3847` echoing
  `"check_amount": 25` where current code renders `25.0` — no longer abort the replay with
  `ValueError`. Live evaluation remains strict.
- Hallucinated tool calls in agent trajectories are now treated as no-ops during `Environment.set_state` replay (matching the live env, which returns a `ToolMessage(error=True)` and applies no state change), instead of raising `ValueError`. Previously, the exception propagated through `run_with_retry`, causing the entire task to be re-run up to `--max-retries` times before being binned as `INFRASTRUCTURE_ERROR` (which is excluded from `pass^k` and `avg_reward` metrics). Trajectories that hallucinate and then successfully recover now score correctly; trajectories that hallucinate without recovering still fail naturally via DB-state mismatch. Repeated hallucination remains bounded by the orchestrator's `max_errors` guard (`TerminationReason.TOO_MANY_ERRORS`).

## [1.0.0] - 2026-MM-DD

### Added

#### Voice Evaluation
- Full-duplex voice evaluation with tick-based orchestration for simultaneous user/agent interaction
- `DiscreteTimeAudioNativeAgent` with provider-agnostic adapter architecture
- Audio native providers: OpenAI Realtime, xAI Grok Voice, Gemini Live (fully supported); Nova Sonic, Qwen, Deepgram, LiveKit (experimental)
- Voice user simulator with yield/interrupt/wait turn-taking behavior in full-duplex conversations
- ElevenLabs TTS integration for user simulator speech synthesis
- Audio effects pipeline: background noise, burst noise, telephony compression, frame-drop simulation
- AudioTap system for recording audio at each pipeline stage
- Voice-specific task metadata (`tasks_voice.json`, `audio_difficulty.json`) for airline, retail, telecom domains
- Hallucination reviewer for detecting user simulator deviations in voice evaluations
- Automatic hallucination retry system for improved evaluation reliability

#### Knowledge Retrieval + Banking Domain
- New `banking_knowledge` domain with 97 tasks and 698 policy/procedure documents
- Modular knowledge retrieval framework (`src/tau2/knowledge/`) with pluggable components:
  - Embedders: OpenAI, OpenRouter (Qwen)
  - Retrievers: BM25, cosine similarity, grep
  - Postprocessors: BGE reranker, pointwise LLM reranker, Qwen reranker
  - Document and input preprocessors
- 12 retrieval configurations: offline (`no_knowledge`, `full_kb`, `golden_retrieval`, `bm25`, `bm25_grep`, `grep_only`), embedding-backed (`openai_embeddings*`, `qwen_embeddings*`), and agentic (`terminal_use`, `terminal_use_write`)
- Disk-based embedding cache with automatic invalidation on document changes
- Sandboxed shell retrieval via `sandbox-runtime` integration
- Combined transactional tools + knowledge retrieval evaluation paradigm

#### Developer Experience
- `tau2 intro` command with guided introduction to framework and domains
- `tau2 view` enhanced simulation viewer
- Layered runner API (`tau2.runner`) with three levels: simulation, build, and batch execution
- LLM-based conversation review system with automated quality checks
- Per-task summary analysis and diagnostics
- Timeout control via `--timeout` flag
- Multiple results paths comparison support
- 53 new test files covering voice providers, banking tools, and retrieval pipelines
- Comprehensive documentation: Getting Started guide, CLI reference, knowledge retrieval guide, audio native mode guide
- Per-module READMEs and developer guides throughout codebase

### Changed

#### Installation and Dependencies
- Migrated from pip/pdm-backend to `uv` workflow
- Optional dependency groups: `voice`, `knowledge`, `gym`, `dev`, `experiments`
- Python requirement updated to `>=3.12, <3.14`
- Modernized build backend and dependency grouping

#### Architecture
- Runner refactored from monolithic `run.py` to layered `src/tau2/runner/` package
- RunConfig model split: `BaseRunConfig` → `TextRunConfig`, `VoiceRunConfig`, `AudioNativeConfig`
- User simulator refactored: extracted base class, added streaming simulator
- Evaluator enhanced with 5 new modules for review, hallucination detection, and auth classification
- Data model additions: audio, voice, persona, and audio effects models

#### CLI and Tooling
- Enhanced `tau2 view` with richer output and Sierra theme
- Improved action check display showing requestor (user/agent)
- Simulation status tracking with `sim_status.json` for retry provenance

### Fixed

#### Banking Knowledge Domain (20+ tasks)
- Required documents corrections across tasks 027, 046, 056, 061, 062, 075, 081, 083, 084, 088, 100
- Task logic fixes: task 019 (7 incorrect transactions), 048 (tier/retention), 058 (invalid parameter), 061 (deposit transfer), 062 (action ID), 084 (liability amount), 100 (user simulator prompt), 102 (NL assertion)
- Document escaping cleanup: removed double `$$` escaping across 155+ documents, double `%%` escaping
- Policy document corrections: fee mischarges rules, cooldown constraints, dispute eligibility criteria
- Tool validation parity: ported 8+ missing validations, added `card_type` validation, fixed `current_holdings` field
- Rewards calculation corrections (Fatima Al-Hassan transactions)

#### Airline task fixes (27 tasks)

- **Incorrect expected actions:**
  - Tasks 2, 27, 38: Removed incorrect delayed flight compensation — policy does not allow compensation for regular economy passengers without insurance
  - Task 7: Corrected cancellation policy compliance — reservation did not meet cancellation criteria
  - Task 9: Removed incorrect NQNU5R cancellation and fixed date typo in `nl_assertion` (May 12 → May 22)
  - Task 37: Removed incorrect NQNU5R cancellation — reservation did not meet cancellation criteria
  - Task 44: Corrected cancellation logic and clarified user instructions
- **Ambiguous user instructions:**
  - Task 5: Clarified no-compensation reasoning in task purpose
  - Task 12: Clarified baggage request independence from cabin upgrade
  - Tasks 15, 16: Disambiguated economy from basic economy — user said "economy" but expected actions assumed basic economy
  - Task 18: Added payment method lookup and refund timing; moved payment method uncertainty to `unknown_info`
  - Task 19: Added return flight date to `reason_for_call`
  - Task 21: Clarified date and fallback in `reason_for_call`
  - Task 25: Clarified payment decision criteria and passenger scope
  - Task 30: Added payment preference and flight selection guidance
  - Task 36: Noted flight departure in purpose description
  - Task 39: Corrected purpose description about API enforcement
- **Impossible or contradictory constraints:**
  - Task 14: Replaced impossible Mastercard constraint (not in user profile) and corrected payment amounts
  - Task 34: Enforced complete-package budget constraint
  - Task 42: Resolved Boston location contradiction between user address and scenario
- **Policy loophole prevention:**
  - Task 13: Prevented cancel-and-rebook workaround for basic economy flight modification
  - Task 29: Enforced that destination change requires cancel+rebook, not modification
  - Task 32: Enforced two-step cabin upgrade then flight change sequence
  - Task 45: Prevented cabin upgrade loophole for basic economy flight modification
- **Missing fallback behaviors:**
  - Task 43: Added health-issue guardrail to prevent unsafe recommendations
- **Data fixes:**
  - Task 23: Corrected passenger names and purpose description
  - Task 33: Corrected baggage charges for Gold member
  - Task 35: Corrected date of birth

#### Retail task fixes (26 tasks)

- **Incorrect expected actions:**
  - Tasks 12, 13: Removed invalid PayPal refund expected actions — PayPal is not a supported refund method
  - Tasks 33, 34: Removed incorrect `nl_assertions` and `communicate_info` asserting $1,093.34 refund — gold actions follow fallback path (no cancellation/refund occurs); updated `reward_basis` from `["DB", "NL_ASSERTION"]` to `["DB"]`
  - Task 59: Removed trivial `calculate` action from expected actions
- **Ambiguous user instructions:**
  - Tasks 0, 1: Changed "similar one" to "the same one" for keyboard exchange to match expected behavior
  - Tasks 2, 3, 4: Added "exactly" to t-shirt count question to require precise answer
  - Task 8: Added "Make sure to return BOTH orders" to clarify multi-order return
  - Task 15: Added PayPal preference for payment method selection
  - Task 22: Clarified regret timing to prevent premature user abort
  - Task 28: Clarified multi-order return instructions
  - Task 55: Added sequencing and user behavior guidance
  - Task 76: Made cancellation reason explicit to prevent agent divergence
  - Tasks 98, 99: Clarified bicycle frame size priority
  - Task 100: Specified payment method for pending order modification
- **Impossible or contradictory constraints:**
  - Task 18: Fixed invalid same-item exchange for office chair — system does not support same-SKU exchange
  - Task 29: Added item validation for garden hose exchange
  - Task 91: Corrected same-item exchange for e-reader — system does not support same-SKU exchange
  - Task 107: Changed hiking boots exchange to different item (was invalid same-item exchange)
- **Missing fallback behaviors:**
  - Task 20: Relaxed specs constraint and added fallback for non-modifiable orders
  - Task 54: Added fallback behavior when desired boots exchange item is unavailable
  - Task 62: Added fallback behavior when speaker search returns no matching results
- **Tool additions:**
  - Task 21: Added `get_item_details` tool to retail domain for product information retrieval

### Security

## [0.2.1] - 2025-11-07
### Added
- Gymnasium-compatible interface for RL training with `AgentGymEnv` and `UserGymEnv`
- Train/test task splits for all domains
- Interactive play mode (`tau2 play`) supporting both agent and user roles
- Possibility to strictly enforce communication protocol rules (e.g., no mixed messages with text and tool calls)

## [0.2.0] - 2025-10-06

### Added
- Web-based leaderboard system with interactive submission management
- GitHub Pages deployment for leaderboard with automated CI/CD
- Comprehensive submission validation and verification system
- Model comparison interface with performance metrics visualization
- Trajectory visualization in web interface
- Mobile-responsive leaderboard design
- Logo assets and branding for multiple LLM providers
- Live leaderboard deployment at tau-bench.com

### Changed
- Enhanced submission manifest structure
- Improved image handling and asset management
- Updated deployment workflow for better reliability

### Fixed
- Mobile view responsiveness issues
- Missing submissions from manifest
- Image path resolution for GitHub Pages deployment
- Base URL handling for subdirectory deployment

## [0.1.3] - 2025-08-26

### Fixed
- LLM arguments parsing and handling
- Removed default natural language assertion checks that were causing issues

## [0.1.2] - 2025-07-17

### Added
- `tau2 check-data` CLI command for verifying data directory setup
- Support for `TAU2_DATA_DIR` environment variable for non-editable installs
- Fallback to local source when data directory is not set
- `--num-tasks` CLI flag for limiting task count

### Changed
- Made `pip install -e .` the default installation method
- Improved task name display in CLI
- Enhanced data directory configuration flexibility

### Fixed
- Installation issues with data directory discovery
- Task filtering and display problems

## [0.1.1] - 2025-06-12

### Fixed
- Domain viewer CLI functionality
- `tau2 domain` command execution issues

## [0.1.0] - 2025-06-12

### Added
- Initial release of τ-bench framework
- Support for multiple domains: mock, airline, retail, telecom
- Command-line interface with `tau2` command
- Agent evaluation system with LLM integration via LiteLLM
- User simulator for realistic conversation scenarios
- Environment system with domain-specific tools and policies
- Orchestration system for managing agent-user-environment interactions
- Comprehensive test suite
- Domain-specific documentation and API endpoints
- Experimental features: no-user mode, oracle-plan mode, workflow policies
- Support for ablation studies
- Interactive environment CLI for testing and debugging
- Caching system for LLM calls (Redis-based)
- Multi-trial evaluation with concurrent execution support

### Technical Details
- Python 3.10+ support
- FastAPI-based web services
- Pydantic data models
- Rich CLI with tabulated output
- Comprehensive logging with Loguru
- Performance metrics and visualization
- Configurable LLM backends
- Semantic versioning adoption

## Links
- [Repository](https://github.com/sierra-research/tau2-bench)
- [Leaderboard](https://tau-bench.com)
- [Paper](https://arxiv.org/abs/2506.07982)
- [Blog Post](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)