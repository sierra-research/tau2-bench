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
  OpenAI provider with fallback disabled. Both OpenRouter API-base environment
  names are manifest-bound and overwritten in the paid child process, so
  ambient endpoint variables cannot redirect requests. Added Modal
  backend/app/timeout environment settings.
- Preserved the official Qwen request arguments and added a free preflight that
  permits paid execution only while the exact dated Alibaba snapshot is Qwen's
  sole active OpenRouter endpoint. Endpoint manifests now retain active and
  matching counts. The committed full shell-order fixture path is set
  explicitly, so an ambient `TAU2_MODAL_ORDER_MANIFEST` cannot redirect a run.
  The wrapper's offline graders and paid children pin `TAU2_DATA_DIR`, disable
  `.env` loading, and resolve `tau2` from this checkout; paid children also
  disable user-site loading and scrub ambient Python/uv project/runtime overrides.
- No paid model, embedding, or Modal evaluation was run while building the
  harness.

## 2026-08-22 — offline harness validation

- JSON validation and `py_compile` under the locked uv runtime passed for the
  config and all eight harness scripts. Ruff lint and format checks passed.
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
  mismatches. This introduced schema-v3 full gates; the later aggregate
  sampling guard supersedes them with schema v4.
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
- After the aggregate-sampling, restart-safe resume, raw-cost, and full-manifest
  guards were integrated, the final offline slices passed: 67 focused harness
  and Modal tests, 13 provider-cost tests, and 457 broader banking/environment
  tests with one skip. Both shell-order generators were current, the BM25 oracle
  remained 1,910/1,910, and targeted Ruff, compilation, JSON, and diff checks
  passed. The two legacy `test_llm_utils` generation tests were deliberately not
  run because they make live OpenAI calls; their 13 cost-only siblings passed.

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

## 2026-08-22 — live seed-replay diagnosis and aggregate gate

- Published the initial standalone implementation to the user fork branch
  `signalrush/tau2-bench:agent/tau3-banking-modal-parity` as commit
  `f1b1658abb0b1382703d8b686a5febf17b3ef4a3`. A fresh network clone at
  `/tmp/tau3-banking-fresh.18HRmA/repo` passed the locked offline install,
  selected-cache validation, clean-worktree check, and dry smoke without making
  remote calls.
- Launched the first paid smoke with
  `uv run --frozen --extra knowledge python reproduction/tau3_banking/run.py smoke --output-dir reproduction/tau3_banking/runs/gate_trial0 --execute --confirm-paid-api-calls --cost-ceiling-usd 0.25`.
  It produced candidate artifact SHA-256
  `7e594d1f98a9c98cb3d4566549034d6dcbf186904c9526e64696d04de410f764`
  and guarded execution-manifest SHA-256
  `48c41e14193aebb8a9309818ad8eceb1e9935de22a90ecd7f83261c825d8c516`.
- The first paid smoke completed task 001 trial 0 with reward 1, but diverged
  from the published trajectory despite the same seed 626729 and identical
  committed prompts, tools, model arguments, and provider pins.
- Repeated only the first hosted-model requests to isolate sampling. GPT-5.2
  first-user content changed from SHA-256
  `ee51d33b867b92ab172bf8d20e9e8491b767aebf0948b04ad9cd96250a4607be`
  (`gen-1787393465-q38LkKjXJYijmvoTfpYQ`) to
  `1296078c1a996c6291887cf3bfefc046de6612604b657486ead2036a73ef55df`
  (`gen-1787393683-avNYfrT0HhPS0BpykrQ5`), both OpenAI/default. The official
  first-user content SHA-256 is
  `c8bcae567670de2c5c2e4daf0446515c42e0f001f40654622f601c6bc84abf40`.
- Qwen's canonical initial tool call changed from SHA-256
  `80458bfe976f892288e8de50e374fe2e6930cee38419cbff00a4cc0d9d5f4a58`
  (`gen-1787393466-MJM6SNmXOPdQInO9Mvd8`) to
  `489768bcc9eeb8de59fa7582f7a15ac3cb92162f76a26bbb9eff65dac7fd8a0d`
  (`gen-1787393717-LrzM8u8Ls8RMJKyUoNQK`), both Alibaba with null service
  tier. This proves that hosted seed replay is not trajectory deterministic.
- Added an explicit aggregate-only model-sampling gate. Strict per-record
  reward/component, text, call, argument, and output diagnostics are retained.
  When explicitly requested, their sampling-attributable differences may be
  waived only while all 40 task/trial keys, seeds, `user_stop` terminations,
  internally recombinable binary grading, configuration, routes,
  manifest/runtime state, and the exact 22/40 aggregate match. Identical
  non-dense calls with different output remain unwaivable backend drift; the
  separate dense waiver still applies only to identical dense calls.
- Hardened that gate before expanding the paid subset: tool results are aligned
  by exact role/name/arguments across call insertion and reordering, with
  ambiguous duplicate results rejected; reward breakdowns are recomputed from
  canonical DB/action/environment/NL/communication evaluator records; and null
  raw user models are rejected. Each differing deterministic component is now
  recomputed with the authoritative task and official local evaluator in a
  retrieval-free environment, so failed/no-op writes and generic/discoverable
  state mutations receive their exact semantics rather than name-based credit.
  NL outcome drift always requires task 102's validated dated judge route, and
  the full gate requires an exact zero-issue attribution receipt. The paid
  endpoint preflight now requires the same exact GPT-5.2 alias inventory proof
  (4 active total, 3 OpenAI-eligible, all 3 matching the dated snapshot) as the
  comparator.
- Added the ten-task trial-0 intermediate (6/10, historical chat cost
  $2.59890095) before the 40-simulation gate. It can resume the smoke, and the
  full subset can resume it. A partially completed four-trial expansion can
  also resume despite upstream retaining stale `num_trials=1` metadata; exact
  task/trial/seed and bound checkpoint-manifest provenance checks remain.
- The smoke's raw OpenRouter usage costs were $0.154672 agent and $0.0082551
  user, $0.1629271 total. The comparator now reports these non-gating raw
  totals because LiteLLM's mapped top-level agent cost may be zero.
- OpenRouter serialized the user model as `openai/gpt-5.2`. This moving alias
  is accepted only when the bound valid manifest contains the exact catalog
  proof for provider OpenAI and resolved endpoint
  `openai/gpt-5.2-20251211` with four active and three matching endpoints.
  Full-gate verification revalidates the route counters and catalog proof.

## 2026-08-22 — deterministic full-corpus Modal fixture

- Added `generate_full_shell_order.py` and a checked-in 699-entry uniform
  filesystem order derived from the pinned 5,135 recorded shell calls (4,614
  unique commands). The generator treats all 767 subset edges as hard
  constraints, admits only acyclic full-trace edges, and handles compound
  commands conservatively: 54 had a unique segmentation, 31 retained only
  edges safe under every valid segmentation, and one was unsegmentable.
- The resulting fixture has 7,548 precedence edges, constrains 698/699 files,
  and uses the subset order to break the final tie. Its order SHA-256 is
  `ddb11f1a583e408079c136805c786f6e53903afb3dad46047c69a06b3b01b6f3`;
  the manifest file SHA-256 is
  `5f8005d162f81d9eadf6836b296d4a334090daec60c0228770e8b8de890d37f8`.
- Verified the generated fixture offline with
  `uv run --frozen --extra knowledge python reproduction/tau3_banking/generate_full_shell_order.py --check`.
  Then replayed the fixed subset against one live network-blocked Modal sandbox
  with
  `TAU2_MODAL_ORDER_MANIFEST=reproduction/tau3_banking/full_shell_order_manifest.json uv run --frozen --extra knowledge python reproduction/tau3_banking/compare_shell_oracle.py --mode subset --scope all --execute --output reproduction/tau3_banking/artifacts/shell_oracle_subset_full_manifest.json`.
  The result was 263/263 unique commands exact across 275 recorded calls, zero
  mismatches, applied order SHA-256 `ddb11f1a...b01b6f3`, and report SHA-256
  `626fa9d613ec8d789a258bc81d686358b23b2a07131da98b21df841009c7a1bf`.
- Switched the Modal manager default and guarded runner environment to this
  single full fixture for smoke, trial-0, subset, and full runs. Runtime order
  selection remains command-independent and no recorded output is replayed.
- Made abrupt-termination resume provenance crash-safe. A `running` manifest may
  authorize the structurally validated checkpoint produced by that exact
  guarded launch; resumed launches additionally bind the already validated
  pre-run checkpoint SHA-256. Finalized manifests retain exact post-run state
  and checkpoint digest requirements.
  Paid `uv` children inherit the output-directory OS lock, preventing a second
  resume from writing or spending concurrently even if the Python wrapper is
  killed first. Fresh-run absence and resume checkpoint/state provenance are
  rechecked after acquiring the lease and before prewarm or paid launch.
  Successful runs with a transient finalization fingerprint error retain the
  original zero runner exit code and can be recovered only by fresh exact
  checkpoint and clean-state validation.
- Made the full live Modal shell-oracle receipt a hard full-run prerequisite.
  The guard binds its path, SHA-256, official-reference and order-fixture
  digests, complete counts/details, and committed score-impact review. Any
  remaining difference is treated as potentially behavior/score affecting;
  reviewed and explicitly accepted nonzero drift additionally requires
  `--allow-known-full-shell-drift`, which is persisted in the run manifest.
- Added official-schema and half-duplex chronology validation for every
  trajectory. Tool results must follow their pending call with matching ID and
  requestor, participant turns must be valid, call IDs must be unique, and
  repeated stateful call outcomes retain occurrence order. These failures are
  structural/non-waivable. Judge response IDs are also globally disjoint from
  participant generation IDs.

## 2026-08-22 — active full Modal shell receipt

- Ran the final no-model, 4,614-unique-command oracle against the active fixture
  and hydrated Modal image with the exact command:

  ```bash
  env -u TAU2_MODAL_EXPECTED_IMAGE_ID \
  TAU2_MODAL_ORDER_MANIFEST=reproduction/tau3_banking/full_shell_order_manifest.json \
  uv run --frozen --extra knowledge python \
    reproduction/tau3_banking/compare_shell_oracle.py \
    --mode full --scope all --execute --max-details 100 \
    --output reproduction/tau3_banking/artifacts/shell_oracle_full_active_manifest.json
  ```

  The tracked receipt SHA-256 is
  `17728b8c8ae721e23506a06bd7dfe9a276009222ab17f9af0ada20ace9fd06eb`.
  It binds image-recipe SHA-256
  `fa738d7f079e0b3cfccf0c7e30f140064409afd04732eb3f5182f737f7126795`
  and hydrated Modal image object ID `im-57yaJoNct9YREpBb74YQ0k`.
- The final result is 4,603/4,614 unique commands exact
  (`99.76159514521024%`) and 5,124/5,135 recorded occurrences exact
  (`99.78578383641675%`). All 11 mismatch details are retained, spanning 11
  task/trial records across 10 tasks and zero gate-subset simulations.
- The 11 residuals are three traversal-order differences (tasks 080/0, 023/1,
  and 070/2), four unrecoverable randomized historical working-directory paths
  (067/0, 063/1, 076/2, and 044/3), two EROFS-versus-EACCES diagnostics (019/0
  and 022/3), one explicit-path `ls` ownership/timestamp difference (044/2),
  and one remaining `srt` conditional-shell difference (064/2).
- The compatibility transform mirrors `sandbox-runtime` 0.0.23 by replacing
  each command-argv `!` with `\!`. A targeted 20-command probe matched 17/20
  historical outputs after this transform; the active receipt retains the one
  remaining full-corpus conditional-shell mismatch rather than spoofing it.
- The Modal image pins the exact install call
  `modal.Image.debian_slim().pip_install("scipy==1.16.3")`. This restores the
  official task 095 trial 2 command whose executable core is:

  ```bash
  python3 - <<'EOF'
  from scipy.optimize import brentq
  bal = 95550.0
  days = 31
  def i(apy): return bal * ((1 + apy) ** (days / 365) - 1)
  ap = brentq(lambda a: i(a) - 450, 0.01, 0.2)
  print("APY that yields $450:", round(ap, 5), f"{ap * 100:.3f}%")
  EOF
  ```

  This is the same numerical operation as the recorded command; formatting-only
  list output omitted here does not alter the executed fixture or receipt.
- Cleanup completed with zero live Modal tasks. The receipt and guard describe
  the residuals honestly: all 11 can change model-visible output, downstream
  behavior, and score. They are explicitly accepted only after an exact 22/40
  live subset and only with `--allow-known-full-shell-drift`; this authorizes a
  distribution-level Modal reproduction, not exact trajectory parity.

## Open parity risks before any full run

1. The official GPT-5.2 user, GPT-4.1 judge, and `text-embedding-3-large` calls
   used direct OpenAI; reproduction must use OpenRouter. Live dense comparison
   is not exact and the official cache is unavailable.
2. Modal replaces the historical local Anthropic `sandbox-runtime`. The active
   disclosed full-trace fixture preserves all 263/263 unique subset shell
   commands (275 recorded calls), including the strict 59/59 recursive slice.
   It is trace-derived and constrains 698/699 files. The earlier ad-hoc
   full-corpus probe reached 4,581/4,614 unique commands with a different order.
   The pinned active-fixture receipt reaches 4,603/4,614; its 11 model-visible
   differences remain potentially behavior/score affecting and require the
   explicit full-run acknowledgement described above.
3. GPT aliases may have moved since 2026-08-03. Qwen paid execution is blocked
   unless its sole active endpoint remains the exact Alibaba snapshot; still
   confirm raw response provider/model fields in smoke output before the subset.
4. The official result does not serialize concurrency. Ten was reconstructed
   from launch overlap; it is not a signed metadata field.
