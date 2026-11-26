# v2 CRM evaluation harness

This directory contains the fixed evaluation design for the thesis. It replaces the
current CRM evaluation, which used an LLM judge in the reward and a shared Twenty
CRM instance across trials.

## Changes

| Old | v2 |
|---|---|
| `reward_basis = ["NL_ASSERTION","ENV_ASSERTION"]` | `["ENV_ASSERTION"]` only |
| Existence checks only | Person + company link + opportunity name + stage + task title + record counts |
| Shared Twenty instance | Reset before every trial, clean-state check |
| Empty DB, no fixtures | Fixtures seeded through MCP and mirrored into the local CRMDB with the same UUIDs |
| `train = test = all 10` | A published disjoint dev/test split |
| `pass^4` as "four consecutive" | pass^k is all k of n trials |
| Trust written by the agent | Removed from the reward; dialogue context only |

## Files

- `tasks_v2.json` - the 10 ENV-only tasks.
- `fixtures_v2.json` - companies, people and opportunities seeded per task.
- `task_map.md` - old Table 5 labels to v2 task IDs.
- `generate_tasks_v2.py` - regenerates the two JSON files from the v1 tasks.
- `apply_patch.py` - adds the strict verifiers to `tools.py` and removes the
  silent local fallbacks.
- `twenty_mcp.py` - small MCP client.
- `reset_twenty.py` - per-trial reset and clean-state check.
- `seed_twenty.py` - seeds fixtures and builds matching local state.
- `run_trials.py` - reset, seed, run one trial, save JSON.
- `summarize.py` - pass^k and task-cluster bootstrap from the logs.
- `paper_edits.md` - text changes for the thesis after the rerun.

## 1. Patch the tau2 checkout

```bash
python harness/apply_patch.py --repo /path/to/23-tau2-bench-copy
```

The patch:

- inserts `verify_person_named_company`, `verify_opportunity_named`,
  `verify_task_named`, `verify_people_total`, `verify_opportunities_total`,
  `verify_tasks_total`, `verify_person_count`, `verify_opportunity_count` and
  `verify_clean_state`;
- makes `create_person`, `create_opportunity` and `create_task` raise on MCP
  failure instead of writing a local-only record;
- makes stage/task updates raise on MCP failure.

A backup is written to `tools.py.bak-v2`. The patch is idempotent.

## 2. Configure credentials

The scripts read `TWENTY_MCP_URL` and `TWENTY_MCP_TOKEN` from the environment, or
from `.env` in the tau2 repo root. Example:

```env
TWENTY_MCP_URL=http://host_name_here:port_here/mcp?mode=direct
TWENTY_MCP_TOKEN=...
```

## 3. Check the reset

The default reset soft-deletes records through MCP and then verifies that
`find_many_*` returns zero records:

```bash
python harness/reset_twenty.py --repo /path/to/23-tau2-bench-copy
```

If records remain (soft deletes are visible to `find_many`), use a hard SQL reset.
Run this against the Twenty Postgres database, after confirming the table names:

```sql
-- check the table names first
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- Twenty core tables; adjust if your schema differs
TRUNCATE TABLE person, company, opportunity, task RESTART IDENTITY CASCADE;
```

If the Twenty server caches records, restart it after the truncate.

Do not run the benchmark until `reset_twenty.py` prints `OK: clean state`.
A single contaminated trial invalidates pass^k.

## 4. Smoke test

Run one task and one trial first:

```bash
python harness/run_trials.py \
  --repo /path/to/23-tau2-bench-copy \
  --models Qwen3.5-9B \
  --user-model Qwen3.5-9B \
  --agent-args '{"api_base":"http://127.0.0.1:8080/v1","api_key":"local"}' \
  --user-args  '{"api_base":"http://127.0.0.1:8080/v1","api_key":"local"}' \
  --task-ids 0 --trials 1
```

Then inspect:

- `harness/runs/<model>/task_0/trial_01/simulation.json`
- `harness/runs/<model>/task_0/trial_01/meta.json`
- the reward breakdown and every env assertion result.

If the trial does not start from the expected clean/fixture state, stop and fix the
reset before running more trials.

## 5. Full rerun

The minimum defensible rerun is the two leading models:

```bash
python harness/run_trials.py \
  --repo /path/to/23-tau2-bench-copy \
  --models Qwen3.5-9B,GPT-OSS-20B \
  --user-model <simulator-model> \
  --agent-args '<json>' \
  --user-args '<json>' \
  --trials 8
```

This is 2 x 10 x 8 = 160 trials. Run trials sequentially; the reset is global.

The paper's current 4-model table also contains Qwen 3 4B and Phi-4-mini 3.8B. Their
old counts are contaminated by the shared Twenty instance. Either rerun all four
models or remove the sub-4B rows from the main table and label them as a pilot.

## 6. Summarize

```bash
python harness/summarize.py --runs harness/runs --out harness/summary
```

Outputs:

- `summary/trials.csv` - one row per trial.
- `summary/summary.json` - pass^1, pass^4, pass^8 and task-cluster 95% CIs.

Do not reuse the old `sessions/stats.json` counts. Recompute from the new logs.

## 7. User simulator

The paper says Qwen 3.5 9B played the customer. That is the same model as the best
agent, so the simulator and agent share a family. For the rerun, prefer a model that
is not under test. If only Qwen is available, run the leading comparison twice with
two different simulators and report the sensitivity.

Pin the simulator version, prompt and seed. Record the simulator turns in the logs.

## 8. Split

The current `split_tasks.json` is `base = train = test = all 10`. Create a disjoint
split before the final run, for example:

```json
{
  "base": ["0","1","2","3","4","5","6","7","8","9"],
  "dev":  ["0","1","2","3","4"],
  "test": ["5","6","7","8","9"]
}
```

Use `dev` for prompt/task iteration and `test` only for the final table. If n=5 is
too small for the final table, add tasks; do not evaluate on the tasks you tuned on.

## 9. What to report

For every trial:

- task ID, trial, model, simulator model, seeds;
- initial and final DB state hashes;
- each env assertion and its result;
- tool calls and transcript;
- MCP/infrastructure errors;
- reward.

Report infrastructure failures separately from task failures. Do not silently turn
an MCP error into a task failure or a pass.
