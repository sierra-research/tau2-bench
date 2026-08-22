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

After smoke inspection, resume that same checkpoint as the fixed 10-task,
four-trial gate. The task IDs are `001,003,004,007,014,032,034,035,046,102`;
their official result is 22/40 (`55.0%`) with per-trial reward sums
`[6,6,4,6]`. Task 102 deliberately exercises the benchmark's only
natural-language judge. The completed smoke conversation is skipped, leaving
39 paid agent/user conversations rather than 40:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py subset \
  --output-dir reproduction/tau3_banking/runs/gate_trial0 --resume \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 10
uv run --offline --frozen --extra knowledge python \
  reproduction/tau3_banking/compare_results.py \
  reproduction/tau3_banking/runs/gate_trial0/results.json \
  --mode subset \
  --output reproduction/tau3_banking/runs/gate_trial0/strict_compare.json \
  --write-gate reproduction/tau3_banking/.state/subset_score_parity.json \
  --allow-known-dense-drift \
  --reference-results reproduction/tau3_banking/artifacts/banking_knowledge_results.json
```

The strict comparison joins on `(task_id, trial)` and reports reward, seed,
termination, score components, tool-call sequence and arguments, paired tool
outputs, and raw provider routes. Strict `behavior_parity` remains false when
any tool output differs. Gate creation rejects `--score-only`: all 40 expected
task/trial pairs, the exact 22 rewards and `[6,6,4,6]` trial totals, official
trial seeds, `user_stop` terminations, action/DB/NL grading components, tool
sequence and arguments, and every non-dense tool result must match.

`--allow-known-dense-drift` is an explicit, narrow gate waiver for the
demonstrated direct-OpenAI versus OpenRouter embedding difference. It permits
only content differences in paired assistant `KB_search_dense` ToolMessages.
Missing outputs, different calls, order, arguments, roles, tool names, unpaired
results, scoring components, provider routes, and every other behavior mismatch
still reject the gate. The receipt records both strict behavior failure and the
exact waived/remaining counts; full mode verifies those invariants again. Omit
the flag to require byte-exact tool behavior. The comparator also validates the
serialized commit ancestry, models, provider-pinned user/judge arguments,
domain, retrieval config, seed, limits, dated NL-judge route, checkpoint digest,
and exact guarded command/environment manifest. A resumed run may retain
`num_trials=1` in upstream checkpoint metadata; the gate accepts that known
serialization artifact only after all 40 task/trial pairs are present and exact.

Full mode is deliberately hard to trigger. It requires the current gate,
`ALLOW_FULL_RUN=1`, the two paid-run flags, and a cost ceiling above the
historical $246.13 chat cost:

```bash
ALLOW_FULL_RUN=1 uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/run.py full \
  --execute --confirm-paid-api-calls --cost-ceiling-usd 300
```

Do not run this until subset mismatches have been understood. The historical
cost excludes embeddings, NL judges, and Modal; provider prices can change, and
retries can increase spend.

## Credential and sandbox handling

`run.py` takes no key argument. For paid execution,
`~/.rllm/config.json:api_keys.openrouter` is the authoritative credential. If
an ambient `OPENROUTER_API_KEY` is present, it must match that file value or the
runner refuses to start; the ambient value is never preferred. The runner never
prints the value or writes it to the manifest. The local evaluator receives the
same value as
`OPENROUTER_API_KEY` and `OPENAI_API_KEY`, plus
`OPENAI_BASE_URL=https://openrouter.ai/api/v1` so the OpenAI embedding SDK uses
OpenRouter. No application secrets are passed into Modal sandboxes.

The GPT-5.2 user and dated `gpt-4.1-2025-04-14` NL judge both include an
OpenRouter provider order of `OpenAI` with fallback disabled. This prevents a
provider or judge-alias switch from being silently treated as parity.
The official Qwen agent arguments remain unchanged. Before reading the API key,
the free endpoint preflight requires that Qwen's sole active OpenRouter endpoint
is exactly `Alibaba | qwen/qwen3.8-max-20260803`; it records both active and
matching endpoint counts and refuses paid execution if another active route
appears.

The runner sets:

```text
TAU2_SANDBOX_BACKEND=modal
TAU2_MODAL_APP=tau3-banking-sandboxes
TAU2_MODAL_SANDBOX_TIMEOUT=3600
TAU2_MODAL_ORDER_MANIFEST=reproduction/tau3_banking/subset_shell_order_manifest.json
TAU2_NL_ASSERTIONS_MODEL=openrouter/openai/gpt-4.1-2025-04-14
TAU2_NL_ASSERTIONS_ARGS={"temperature":0.0,"extra_body":{"provider":{"order":["OpenAI"],"allow_fallbacks":false}}}
```

Each shell-using simulation gets a network-blocked, read-only Modal sandbox.
The one-hour lifetime exceeds the longest official trajectory; individual shell
commands retain the upstream 30-second timeout. The checked-in gate fixture is
verified and materialized in one uniform order on container tmpfs, exposed at
the unchanged `/knowledge_base` path. Regenerate or verify it without any
model calls with:

```bash
uv run --frozen --extra knowledge python \
  reproduction/tau3_banking/generate_subset_shell_order.py --check
```

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
tool answer. This remains a disclosed reproduction fixture rather than an
independent blind evaluation: 451 filenames are constrained by the public gate
trace and 248 use a lexical tie-break. It is not sufficient evidence for the
full 97-task trace.

An exploratory, ad-hoc full-corpus ordering probe reproduced 4,581/4,614 unique
official shell commands (99.2848%) while preserving the gate's 263/263. The 33
remaining unique differences were 10 mutually incompatible traversal-order
cases, 14 historical `sandbox-runtime` exclamation-mark escaping cases, four
random historical `pwd` paths, two EROFS-versus-EACCES diagnostics, and one
case each for Bash argv-zero text, missing SciPy, and explicit-path `ls`
metadata. The probe used temporary scripts and is deliberately not wired into
the scored runner or presented as a reproducible receipt. The authentic Bash
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
because concurrency is not serialized in the result metadata. Finally, model
sampling and provider aliases are not guaranteed immutable even with the
recorded seeds. An equal aggregate score is necessary, but the detailed
comparator should be used to diagnose compensating per-task changes.
