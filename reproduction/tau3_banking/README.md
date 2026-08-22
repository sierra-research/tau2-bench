# tau3 banking reproduction harness

This folder reproduces only the `banking_knowledge` submission
`qwen3-8-max_sierra_2026-08-04`. It pins the public result, the v1.0.1 data,
the exact 97-by-4 reward matrix, and the run arguments. Every run command is a
dry run unless `--execute` and the paid-call acknowledgement are both present.

## Reference target

- Upstream: `sierra-research/tau2-bench` at
  `fc0055dc4e0a316c3f83133267fbd6faaa770992` (`v1.0.1`).
- Agent: `openrouter/qwen/qwen3.8-max`, reasoning effort `xhigh`.
- Official user simulator: `gpt-5.2`, reasoning effort `low` (the trace resolved
  this through direct OpenAI to `gpt-5.2-2025-12-11`).
- Retrieval: `alltools` (BM25, `text-embedding-3-large`, shell).
- Seed 300; derived trial seeds 626729, 373753, 361454, 1567; 200 steps;
  10 consecutive tool errors; 4 trials; 97 tasks.
- Exact score: 214/388 = `55.154639175257735%` pass@1. Pass@2/3/4 are
  `45.18900343642611%`, `39.69072164948454%`, and
  `35.051546391752574%`.

[`reference.json`](./reference.json) is the machine-readable source of truth.
It also records immutable Git objects, artifact SHA-256 values, costs, mode
scopes, and every task reward vector.

## Setup and reference fetch

From the repository root:

```bash
git rev-parse HEAD
uv sync --frozen --extra knowledge
uv run --frozen --extra knowledge modal setup
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/fetch_reference.py
```

The fetch is atomic and publishes a file only after its byte size and SHA-256
match `reference.json`. The trajectory is about 236 MiB and is gitignored.
Verify an existing download without network access with:

```bash
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/fetch_reference.py --verify-only
```

The reproduction branch deliberately tracks the one selected 698-document
OpenRouter/OpenAI embedding cache even though the upstream cache directory is
normally ignored. The harness validates its document IDs, effective row order,
shape `(698, 3072)`, `float64` dtype, finite values, and semantic SHA-256 before
the first chat call. A fresh clone therefore does not need to pay to rebuild it.

## Cost-minimized progression

First inspect the exact argv and environment plan. This makes no API or Modal
calls and does not read the API key:

```bash
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py smoke
```

Then, if the plan is correct, run one official task into the fixed gate run
directory and compare its score and trace. The ceiling is a preflight check
against historical serialized chat cost, not a provider-side hard spending
cap.

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py smoke \
  --output-dir reproduction/tau3_banking/runs/gate_trial0 \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 0.25
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_results.py \
  reproduction/tau3_banking/runs/gate_trial0/results.json --mode smoke
```

After smoke inspection, first expand the same checkpoint to all ten gate tasks
for trial 0 only. This intermediate costs much less than all four trials, uses
the official concurrency of 10, and targets 6/10:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py subset_trial0 \
  --output-dir reproduction/tau3_banking/runs/gate_trial0 --resume \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 3.5
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_results.py \
  reproduction/tau3_banking/runs/gate_trial0/results.json --mode subset_trial0
```

The comparator deliberately exits nonzero for strict trajectory drift, so read
its aggregate, component, behavior, route, and manifest fields rather than
treating this intermediate diagnostic as a gate. If its aggregate is 6/10 and
the differences are understood, resume into the fixed 10-task, four-trial
gate. The task IDs are `001,003,004,007,014,032,034,035,046,102`; their
official result is 22/40 (`55.0%`) with per-trial reward sums `[6,6,4,6]`.
Task 102 deliberately exercises the benchmark's only natural-language judge.
The already completed trial-0 conversations are skipped:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py subset \
  --output-dir reproduction/tau3_banking/runs/gate_trial0 --resume \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 12
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_results.py \
  reproduction/tau3_banking/runs/gate_trial0/results.json \
  --mode subset \
  --output reproduction/tau3_banking/runs/gate_trial0/strict_compare.json \
  --write-gate reproduction/tau3_banking/.state/subset_score_parity.json \
  --allow-known-dense-drift \
  --allow-model-sampling-drift \
  --reference-results reproduction/tau3_banking/artifacts/banking_knowledge_results.json
```

The strict comparison joins on `(task_id, trial)` and reports reward, seed,
termination, score components, tool-call sequence and arguments, paired tool
outputs, and raw provider routes. Strict `behavior_parity` remains false when
any tool output differs, and `strict_trace_parity` also includes participant
text. Gate creation rejects `--score-only`. Without a sampling waiver it
requires the exact per-task reward vector, `[6,6,4,6]` trial totals, grading
components, and behavior.

`--allow-known-dense-drift` is an explicit, narrow gate waiver for the
demonstrated direct-OpenAI versus OpenRouter embedding difference. It permits
only content differences in paired assistant `KB_search_dense` ToolMessages.
Missing outputs, different calls, order, arguments, roles, tool names, unpaired
results, scoring components, provider routes, and every other behavior mismatch
still reject the dense waiver.

`--allow-model-sampling-drift` is a separate, explicit aggregate-score waiver.
Two live calls with identical prompts, tools, provider pins, model arguments,
and seed produced different first GPT-5.2 user messages; two equivalent Qwen
calls produced different initial tool calls. With this flag, per-record reward,
per-trial totals, grading components, participant text, and model-selected tool
call count/order/arguments may differ, but all 40 exact task/trial keys, trial
seeds, `user_stop` terminations, internally recombinable binary grading,
configuration, provider routes, execution manifest/state, and the exact 22/40
aggregate remain mandatory. Downstream output differences are covered only
when the generated call changed. Tool results are aligned by exact
role/name/arguments across insertions and reordering, so an identical non-dense
call with different stdout is backend drift and remains fatal. Ambiguous
duplicate-call output alignment also fails conservatively; same-call dense
content is covered only by `--allow-known-dense-drift`. Candidate reward
breakdowns are independently recomputed from DB, action, environment, NL, and
communication evaluator records before aggregate parity can pass.
Every trajectory is also parsed with the official message schema and checked
for half-duplex turn order, call-before-result ordering, exact pending call IDs,
requestors, and unique call IDs. Reordered participant/tool messages and
swapped outcomes from repeated stateful calls are non-waivable.
Every differing deterministic component is also regraded from the candidate
trajectory by the official DB/environment, action, or communication evaluator.
The replay uses the authoritative task and the local `no_knowledge` environment,
so retrieval and shell calls are no-ops while successful, failed, generic, and
discoverable-tool mutations retain their real semantics. A serialized outcome
that the exact evaluator cannot reproduce is fatal even if compensating flips
preserve 22/40. Any NL outcome change requires task 102's validated dated judge
route; text drift is recorded but cannot replace judge provenance. The receipt
retains the exact attribution issue count/details and full mode requires zero
issues.

The receipt records strict failures, mismatch types/counts, each requested and
applied waiver, and the aggregate-only decision; full mode recomputes those
invariants. Omit both flags to require strict trace parity. The comparator also
validates the serialized commit ancestry, models, provider-pinned user/judge
arguments, domain, retrieval config, seed, limits, dated NL-judge route,
checkpoint digest, and exact guarded command/environment manifest. A resumed
run may retain
`num_trials=1` in upstream checkpoint metadata; the gate accepts that known
serialization artifact only after all 40 task/trial pairs are present. Raw
OpenRouter `usage.cost` is required for every attributed generated response,
summed by role and in total, and rebound from the candidate by full mode. Cost
coverage/provenance is gating; the observed total remains diagnostic rather
than a hard live spend cap because LiteLLM may serialize an unmapped Qwen
`agent_cost` as zero.
If the four-trial subset is interrupted after the one-trial checkpoint was
expanded, resume the same output directory again. The guard accepts upstream's
stale serialized `num_trials=1` only for this subset transition and still
revalidates every task, trial, derived seed, checkpoint digest, command,
manifest, and committed runtime/cache state.

An abrupt host or process-group termination can leave the execution manifest in
`running` state after the checkpoint has advanced. That state is resumable only
when the manifest proves the exact guarded command/environment/runtime that was
launched and, for a prior resume, binds the validated pre-run checkpoint digest.
The checkpoint is still structurally revalidated in full before another paid
process can start; other unfinished manifest states fail closed.
Every paid wrapper and its `uv` child also inherit one OS-level exclusive lease
on the output directory. A second resume/writer is rejected while either
process is alive, including when the wrapper is abruptly killed but its child
survives. After acquiring that lease, the wrapper rechecks fresh-run absence or
the exact resume checkpoint/state digest before any cache prewarm or paid child
can start, closing the preflight-to-launch race.
A successful runner whose final checkpoint/state fingerprint hit a transient
I/O error records the original zero exit code. A later resume may recover only
after recomputing the exact checkpoint digest and clean runtime/cache state;
unproven finalization failures remain rejected.

Before a paid full run, replay all 4,614 unique official shell commands against
the active fixture. This makes Modal calls but no model calls and writes a
tracked canonical receipt; a nonzero exit means at least one historical
shell result differs and must be classified before proceeding:

```bash
env -u TAU2_MODAL_EXPECTED_IMAGE_ID \
TAU2_MODAL_ORDER_MANIFEST=reproduction/tau3_banking/full_shell_order_manifest.json \
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_shell_oracle.py \
  --mode full --scope all --execute --max-details 100 \
  --output reproduction/tau3_banking/artifacts/shell_oracle_full_active_manifest.json
```

The pinned 2026-08-22 receipt is 4,603/4,614 unique commands exact
(`99.76159514521024%`) across 5,135 recorded calls, with 11 complete mismatch
details and SHA-256 `17728b8c8ae721e23506a06bd7dfe9a276009222ab17f9af0ada20ace9fd06eb`.
The receipt also binds Modal image-recipe SHA-256
`fa738d7f079e0b3cfccf0c7e30f140064409afd04732eb3f5182f737f7126795`,
hydrated image object ID `im-57yaJoNct9YREpBb74YQ0k`, the pinned
`scipy==1.16.3` dependency, and the unprivileged runtime. Every scored sandbox
must hydrate to that same image object ID.
The reviewed residuals are three traversal-order differences, four randomized
working-directory paths, two permission-code differences, one explicit-path
`ls` metadata difference, and one `srt` conditional-shell difference.

Full mode is deliberately hard to trigger. It requires the current gate,
`ALLOW_FULL_RUN=1`, the two paid-run flags, and a cost ceiling above the
historical $246.13 chat cost:

```bash
ALLOW_FULL_RUN=1 uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py full \
  --output-dir reproduction/tau3_banking/runs/full_reproduction \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 300 \
  --allow-known-full-shell-drift
```

The full guard hashes and parses the pinned live 4,614-command shell-oracle
receipt, requires every nonzero mismatch to have a committed score-impact
assessment and explicit review acceptance, and records the CLI acknowledgement
in the execution manifest. These differences can affect model behavior and the
score; acceptance authorizes a distribution-level reproduction, not exact
trajectory parity. A missing, stale, truncated, or unreviewed receipt blocks
full mode.
Do not run this until subset mismatches have been understood. The historical
cost excludes embeddings, NL judges, and Modal; provider prices can change, and
retries can increase spend.

Afterward, write the bounded full comparison report:

```bash
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_results.py \
  reproduction/tau3_banking/runs/full_reproduction/results.json \
  --mode full \
  --output reproduction/tau3_banking/runs/full_reproduction/full_compare.json
```

The comparator remains strict and can exit 1 for per-task, tool, or participant
text drift even when the exact aggregate is reproduced. For the declared full
result, require `candidate_simulation_count=388`, `candidate_reward_sum=214`,
`aggregate_score_parity=true`, `candidate_grading_integrity=true`, and exact
configuration, raw-route, judge-route, and execution-manifest parity. Strict
mismatch fields remain part of the report and must not be relabeled as exact
trajectory parity.

## Credential and sandbox handling

`run.py` takes no key argument. For paid execution,
`~/.rllm/config.json:api_keys.openrouter` is the authoritative credential. If
an ambient `OPENROUTER_API_KEY` is present, it must match that file value or the
runner refuses to start; the ambient value is never preferred. The runner never
prints the value or writes it to the manifest. The local evaluator receives the
same value as
`OPENROUTER_API_KEY` and `OPENAI_API_KEY`, plus
`OPENROUTER_API_BASE=https://openrouter.ai/api/v1` and
`OPENAI_BASE_URL=https://openrouter.ai/api/v1`. Both transports are bound in
the manifest and overwritten in the paid child environment, so an ambient API
base cannot redirect model or embedding requests. The child also pins the
committed `data/` directory, disables ignored `.env` and user-site loading, and
removes ambient Python/uv project overrides that could silently select code,
data, or an environment absent from the manifest. No application secrets are
passed into Modal sandboxes.

The GPT-5.2 user and dated `gpt-4.1-2025-04-14` NL judge both include an
OpenRouter provider order of `OpenAI` with fallback disabled. This prevents a
provider or judge-alias switch from being silently treated as parity.
The official Qwen agent arguments remain unchanged. Before reading the API key,
the free endpoint preflight requires that Qwen's sole active OpenRouter endpoint
is exactly `Alibaba | qwen/qwen3.8-max-20260803`; it records both active and
matching endpoint counts and refuses paid execution if another active route
appears. OpenRouter currently serializes the pinned user response model as the
moving alias `openai/gpt-5.2`. The comparator accepts that alias only when the
bound, otherwise-valid execution manifest records the exact observed catalog
proof: provider `OpenAI`, resolved endpoint
`openai/gpt-5.2-20251211`, four active endpoints in total, and exactly three
OpenAI-eligible endpoints all matching that dated snapshot. Paid preflight
requires those same exact current counts, and a null
raw response model is rejected. A moving alias without that proof is not
treated as a dated route.

The runner sets:

```text
TAU2_SANDBOX_BACKEND=modal
TAU2_MODAL_APP=tau3-banking-sandboxes
TAU2_MODAL_SANDBOX_TIMEOUT=3600
TAU2_MODAL_ORDER_MANIFEST=reproduction/tau3_banking/full_shell_order_manifest.json
TAU2_NL_ASSERTIONS_MODEL=openrouter/openai/gpt-4.1-2025-04-14
TAU2_NL_ASSERTIONS_ARGS={"temperature":0.0,"extra_body":{"provider":{"order":["OpenAI"],"allow_fallbacks":false}}}
```

Each shell-using simulation gets a network-blocked, read-only Modal sandbox.
The one-hour lifetime exceeds the longest official trajectory; individual shell
commands retain the upstream 30-second timeout. The checked-in full-corpus
compatibility fixture is used unchanged by smoke, trial 0, subset, and full
mode. It is verified and materialized in one uniform order on container tmpfs,
exposed at the unchanged `/knowledge_base` path. Regenerate or verify it from
the pinned public trajectory without any model calls with:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/generate_full_shell_order.py --check
```

The checked-in manifest is the runtime input in a fresh clone; regenerating it
is not required to run the benchmark. The derivation check intentionally uses
the host's Bash, `grep`, and `find` semantics for corpus-only membership
queries, so it is environment-sensitive. Its pinned trace, corpus, cardinality,
and order hashes make any semantic drift fail closed instead of silently
publishing or using a different fixture.

## Parity caveats

The public official run used direct OpenAI for GPT-5.2, the GPT-4.1 NL judge,
and dense embeddings. The available reproduction credential requires
OpenRouter. This is a real transport difference. The official embedding cache
could not be recovered. A fresh, consistently OpenAI-pinned OpenRouter cache
was compared against every official dense call: 1,188/1,806 had the exact
top-10 order, 1,787/1,806 had the same top result, mean document overlap was
99.5238%, and 0/1,806 had all displayed scores byte-identical. Dense retrieval
is therefore a demonstrated parity blocker, not merely a theoretical risk.

The official shell used the v1.0.1 local `sandbox-runtime`; Modal is the required
reproduction backend and is not the historical backend. The initial plain
archive extraction matched 6/6 unique smoke commands and 215/263 unique gate
commands. Its 48 differences were fully classified: 47 recursive-search
filesystem-order differences and one root `ls -la` allocation-metadata
difference. The disclosed trace-derived uniform tmpfs fixture plus narrowly
scoped bare-root `ls -la` metadata normalization now matches all 263/263 unique
gate commands. The strict recursive slice is also 59/59:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_shell_oracle.py \
  --mode subset --scope recursive-filename-lines --execute
```

The runtime never inspects a command to choose ordering and never replays a
tool answer. The active uniform order is derived deterministically from the
pinned full public trace, with the proven subset graph retained as hard
constraints. It constrains 698/699 filenames using 7,548 precedence edges; the
remaining filename uses the subset order as its tie-break. A live Modal replay
after switching to this full fixture still matched 263/263 unique subset
commands. This remains a disclosed trace-derived compatibility fixture rather
than an independent blind evaluation, and it does not rewrite irreproducible
historical paths or diagnostics.

An earlier exploratory, ad-hoc full-corpus ordering probe reproduced 4,581/4,614 unique
official shell commands (99.2848%) while preserving the gate's 263/263. The 33
remaining unique differences were 10 mutually incompatible traversal-order
cases, 14 historical `sandbox-runtime` exclamation-mark escaping cases, four
random historical `pwd` paths, two EROFS-versus-EACCES diagnostics, and one
case each for Bash argv-zero text, missing SciPy, and explicit-path `ls`
metadata. The probe used a different temporary order and remains only
historical evidence, not a receipt for the active fixture. The authentic Bash
argv-zero difference was fixed by invoking `/usr/bin/bash`; the other outputs
are not rewritten or replayed.

Document loading is now lexical and deterministic across filesystems. This
change reproduces all 1,910/1,910 official BM25 calls exactly; the prior host
order reproduced 1,774. Verify that zero-model-call oracle with:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_bm25_oracle.py
```

`max_concurrency=10` was reconstructed from overlapping trajectory timestamps
because concurrency is not serialized in the result metadata. Model sampling
is demonstrably non-replayable even with the recorded seed. For task 001 trial
0, identical GPT-5.2 inputs produced content SHA-256 values `ee51d33b...` and
`1296078c...`; identical Qwen inputs produced initial tool-call SHA-256 values
`80458bfe...` and `489768bc...`. The immutable full hashes and response IDs are
in `reference.json`. The smoke's raw OpenRouter cost was $0.154672 agent plus
$0.0082551 user, $0.1629271 total. The aggregate score is the reproduction
target; the strict comparator remains essential for detecting backend drift or
unexplained compensating task changes.
