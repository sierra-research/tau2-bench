# Reproduction log

## 2026-08-22 — reference reconstruction

- Identified the public leaderboard source as PR 452 and the S3 submission
  `qwen3-8-max_sierra_2026-08-04`.
- Confirmed the result records Git commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`, release v1.0.1, domain
  `banking_knowledge`, `alltools`, seed 300, 4 trials, 200 steps, and 10 errors.
- Confirmed the four generated trial seeds directly from all 388 trajectory
  records: 626729, 373753, 361454, and 1567.
- Verified public metadata SHA-256
  `afd5525e447a221f6997b6af8a6a72536acda81b0e7775fd7124836a13a9ad76`
  (1,340 bytes) and trajectory SHA-256
  `8c8191c43dfb2d21c1322cc154740e5e6044151837ab817c2cbbcd13ffeb626e`
  (246,971,719 bytes).
- Recomputed 214 reward points across 388 simulations, exact pass@1
  `55.154639175257735%`, and 388 `user_stop` terminations. No infrastructure
  terminations were present.
- Extracted all 97 four-trial reward vectors into `reference.json`. Selected an
  exact-score gate with tasks `001,003,004,007,014,032,034,035,046,102`:
  22/40 (`55.0%`) with per-trial totals `[6,6,4,6]`. Task 102 replaces the
  cheaper task 010 so the gate covers the benchmark's only NL assertion.
- Recomputed historical provider-serialized costs: $0.14192925 for the one-task
  smoke, $8.84201230 for the 40-conversation gate ($8.33811800 agent plus
  $0.50389430 user), and $246.13332615 for all 388 chats. These figures exclude
  embeddings, NL judges, and Modal.
- Recorded immutable data/source Git objects in `reference.json`. The working
  checkout HEAD matched the official commit.

## 2026-08-22 — harness implementation

- Added an atomic reference fetcher with size and SHA-256 enforcement.
- Added a bounded JSON comparator keyed by task and trial. Score parity and
  tool behavior parity are reported separately; a missing or extra simulation
  cannot pass the score gate.
- Added dry-run-by-default smoke, 10-task subset, and full launch modes. Paid
  execution requires an explicit ceiling acknowledgement. Full execution also
  requires `ALLOW_FULL_RUN=1` and a current exact subset gate receipt.
- Added non-logging OpenRouter credential loading and explicit OpenRouter dense
  embedding transport. Pinned the GPT-5.2 user and GPT-4.1 NL judge to the
  OpenAI provider with fallback disabled. Added Modal backend/app/timeout
  environment settings.
- Preserved the official Qwen request arguments and added a free preflight that
  permits paid execution only while the exact dated Alibaba snapshot is Qwen's
  sole active OpenRouter endpoint. Endpoint manifests now retain active and
  matching counts. The committed subset shell-order fixture path is set
  explicitly, so an ambient `TAU2_MODAL_ORDER_MANIFEST` cannot redirect a run.
- No paid model, embedding, or Modal evaluation was run while building the
  harness.

## 2026-08-22 — offline harness validation

- JSON validation and `py_compile` under the locked uv runtime passed for the
  config and all seven harness scripts. Ruff lint and format checks passed.
- Submission fetch and second-pass `--verify-only` both produced the pinned
  1,340-byte SHA-256 exactly.
- Recomputed the reference config independently: 97 task files, 388 binary
  rewards, reward sum 214, exact `55.154639175257735%`, gate reward 22/40 with
  `[6,6,4,6]` trial totals, and all four Python-generated trial seeds matched.
- Strict comparison of the official 40-record gate slice against the pinned
  trajectory returned score, grading-component, and tool behavior parity.
  Gate creation now rejects `--score-only` and requires a guarded execution
  manifest, exact provider/judge routes, and strict paired ToolMessage output.
- Added an explicit known-dense-drift gate mode after proving that OpenRouter
  embedding scores cannot be byte-identical to the unrecoverable direct-OpenAI
  cache. Strict behavior diagnostics remain unchanged; the waiver classifies
  only paired assistant `KB_search_dense` ToolMessage content differences and
  requires zero missing, reordered, argument-changed, unpaired, or other tool
  mismatches. Schema-v3 full gates revalidate the classification and counts.
- Deliberately changed one reward; the comparator exited 1 and reported the
  exact task/trial reward mismatch. Deliberately removed tool calls; it retained
  score parity but exited 1 with tool count and sequence mismatches.
- Full strict comparison of the public trajectory against itself returned
  388/388 records, reward sum 214, exact pass rate, grading components, and
  tool behavior with zero mismatches. Reproduction configuration/provenance
  validation correctly rejected that historical artifact as a new guarded run.
- Smoke and guarded-full launch plans were generated offline. The full plan had
  97 explicit task IDs, four trials, concurrency 10, and the pinned arguments.
  Full mode correctly refused without `ALLOW_FULL_RUN=1`; paid mode correctly
  refused without `--confirm-paid-api-calls`.
- The configured OpenRouter key shape, local Modal credential presence, Modal
  backend wiring, OpenRouter embedding wiring, and NL-judge environment wiring
  were checked without logging secret contents or making remote calls.

## 2026-08-22 — live parity evidence recorded

- Added a reproducible shell oracle over the exact gate task set. The smoke
  slice matched 6/6 unique commands exactly (report SHA-256
  `55218fe1359c61c0dc818ff3267c7aa4ca10526a2abe2a63849b71770d50e589`).
  The archive-based gate replay matched 215/263 unique commands exactly across
  275 calls (81.7490%; report SHA-256
  `ff92d2b5d24937836ca7c36cbad0d693e1c24ec54433f312b1398085dce7c597`).
  All 48 differences were classified: 47 recursive-grep traversal-order
  differences and one root `ls -la` filesystem-metadata difference.
- Derived one disclosed, uniform filesystem order from public gate trace paths
  and corpus matches. A network-blocked Modal tmpfs validation preserved all
  699 insertion positions and reproduced 59/59 recursive gate commands,
  including all 47 prior ordering mismatches. The deterministic manifest has
  451 constrained filenames, 248 lexical tie-breaks, and SHA-256
  `898b4038585ab4bd10be0ed57f396c4ae5a2b46d8e339e39cc5c8d8219e1d32f`.
  It does not inspect commands or replay output at runtime, but is explicitly a
  trace-derived reproduction fixture and is insufficient for a blind/full run.
- Checked in the deterministic 699-entry order manifest and its offline
  generator. Modal now extracts that order directly onto tmpfs and exposes it
  through the unchanged `/knowledge_base` path. The strict recursive oracle
  reproduced 59/59 commands, and the complete gate shell oracle reproduced all
  263/263 unique commands across 275 recorded calls. The latter also uses a
  narrow normalization for only the total and `.`/`..` rows of a leading bare
  `ls -la`; an explicit-path listing is never normalized.
- Corrected Modal's visible file modes to match the official Linux shell
  (`drwxrwxr-x` directories and `-rw-rw-r--` files) while retaining read-only
  enforcement through root ownership and an irreversible uid/gid 65534 drop.
  A live 698-document `ls -la` check matched file modes, names, and byte sizes;
  a write attempt returned permission denied. Filesystem allocation totals and
  the `.`/`..` metadata remain backend-specific.
- Rebuilt all 698 document embeddings with one OpenRouter route pinned to the
  OpenAI provider. The resulting cache SHA-256 is
  `2600dc04615b3a0bf01ff03dd3868bc6ba78298ca0f9aab158b93ca5f176cdc5`.
- Found that document loading order was host-filesystem dependent. The macOS
  order reproduced only 1,774/1,910 official BM25 calls; lexical loading
  reproduced all 1,910/1,910 exactly. The loader is now deterministic and the
  cache reorders existing embedding rows by document ID without a paid rebuild.
  The effective sorted cache is `(698, 3072)`, finite `float64`, with semantic
  SHA-256 `7b1668a5b9afd48edba1ef195c10b534accafb91f1da8229c91ea0c0fabb562b`.
- Compared all 1,806 official dense calls (1,802 unique queries) against that
  fresh cache: 1,188/1,806 (65.7807%) had the exact top-k order, 1,787/1,806
  (98.9480%) had the same top result, mean top-k overlap was 99.5238%, minimum
  overlap was 75%, and displayed scores were exact for 0/1,806 calls.
- The official embedding cache was not recoverable. Dense retrieval remains an
  unresolved exact-parity blocker, so no paid agent/user simulation was
  launched during this phase.
- Explored a uniform full-corpus shell order without model calls. An ephemeral
  Modal replay matched 4,581/4,614 unique commands (99.2848%) and retained
  263/263 gate-command parity. The command was
  `uv run --frozen --extra knowledge python /tmp/tau3_full_modal_oracle.py`;
  it selected the temporary manifest through `TAU2_MODAL_ORDER_MANIFEST`.
  The candidate file SHA-256 was
  `2dd6900198202d40ad466031be4b86d0fb7d699352680821f66f27e594356de2`,
  its order SHA-256 was
  `f925abd0b262c3317cedffbd1dd5d4c2a5a4495eee77e65834119ebeef7f0bf6`,
  and the report SHA-256 was
  `cc28f4bfe1a50815a91acb140f309677c34b848a738c26e5da58809fdeed8d90`.
  This is explicitly an ad-hoc exploratory measurement: neither temporary
  script nor candidate is checked in, so it is not a standalone receipt and is
  not used by the paid runner.
- Classified its 33 unique mismatches: 10 traversal-order cases, 14 historical
  `sandbox-runtime` exclamation-mark escaping cases (15 calls), four random
  historical `pwd` paths, two EROFS-versus-EACCES diagnostics, one Bash
  argv-zero diagnostic, one missing-SciPy command, and one explicit-path `ls`
  metadata difference. Changed Modal agent commands to invoke
  `/usr/bin/bash`, which fixes the authentic argv-zero difference. Deliberately
  did not rewrite commands, paths, stderr, or tool answers, and did not add
  SciPy or the exploratory order to the scored runtime.

## 2026-08-22 — official-trajectory regrade

- Regraded all 388 published trajectories against the pinned v1.0.1 tasks with
  Modal selected. Regrading created no remote Modal sandboxes; retrieval staging
  stayed local and all temporary environments were released.
- Using the moving `openrouter/openai/gpt-4.1` alias changed task 102 trial 0
  from 0 to 1 and produced 215/388. This isolated the only grading drift to the
  benchmark's sole `NL_ASSERTION` task.
- Switched the judge transport to the exact benchmark model name,
  `openrouter/openai/gpt-4.1-2025-04-14`. A four-trial task 102 probe restored
  final rewards `[0,0,0,0]`, though one non-consequential NL classification
  varied where the DB component was already zero.
- Repeated the complete 388-trajectory regrade with the dated model. It matched
  exactly: reward 214/388, pass rate `55.154639175257735%`, task 102 DB
  components `[1,0,0,0]`, NL components `[0,0,1,1]`, and final components
  `[0,0,0,0]`. This confirms the grading path while retaining the warning that
  a model judge is not guaranteed deterministic even at temperature zero. The
  retained gitignored regrade artifact SHA-256 is
  `6248bed0ad1fb5b8fcd9edeedd447ba3197d5b044ad78e5e51188bedfddd8c8e`.

## Open parity risks before any full run

1. The official GPT-5.2 user, GPT-4.1 judge, and `text-embedding-3-large` calls
   used direct OpenAI; reproduction must use OpenRouter. Live dense comparison
   is not exact and the official cache is unavailable.
2. Modal replaces the historical local Anthropic `sandbox-runtime`. The current
   disclosed gate fixture is exact for all 263/263 unique subset shell commands
   (275 recorded calls), including the strict 59/59 recursive slice. It is
   trace-derived. The ad-hoc full-corpus probe reached 4,581/4,614 unique
   commands, but is not a checked-in fixture and is not used by the runner.
3. GPT aliases may have moved since 2026-08-03. Qwen paid execution is blocked
   unless its sole active endpoint remains the exact Alibaba snapshot; still
   confirm raw response provider/model fields in smoke output before the subset.
4. The official result does not serialize concurrency. Ten was reconstructed
   from launch overlap; it is not a signed metadata field.
