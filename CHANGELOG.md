# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

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