# Smoke Test Run

**Command:**
```
tau2 run --domain banking_knowledge \
  --agent-llm claude-sonnet-5 --agent-llm-args '{}' \
  --user-llm claude-sonnet-5 --user-llm-args '{}' \
  --retrieval-config bm25 \
  --task-ids task_014 task_024 task_057 task_072 task_081 task_087 \
  --num-trials 1 --save-to smoke_test
```

Notes:
- `--task-ids` is a built-in `tau2 run` CLI flag — no code changes were needed to run this specific subset.
- `--retrieval-config bm25` was used instead of the default (OpenAI embeddings) since only an Anthropic key was available.
- `--agent-llm-args '{}'` / `--user-llm-args '{}'` were required to override the default `{"temperature": ...}` arg, which `claude-sonnet-5` rejects (`temperature is deprecated for this model`).
- Full results (transcripts, per-task detail) saved to `data/simulations/smoke_test/results.json`.

## Summary

| Metric | Value |
|---|---|
| Total Simulations | 6 |
| Total Tasks | 6 |
| Average Reward | 0.1667 |
| Pass^1 | 0.167 |
| Avg Cost/Conversation | $3.1958 |
| Write Actions | 38/49 (77.6%) |
| DB Match | 2 / 4 (33.3%) |
| Normal Stop | 6 (User: 6 / Agent: 0) |

## Per-task results

| Task | Reward | DB Check | Partial Action Reward | Duration | Agent Cost | User Cost |
|---|---|---|---|---|---|---|
| task_014 | ❌ 0.0000 | ✅ 1.0 | 0/1 (0.0%) | 45.33s | $0.4209 | $0.0259 |
| task_024 | ❌ 0.0000 | ❌ 0.0 | 0/1 (0.0%) | 120.01s | $2.7708 | $0.0429 |
| task_057 | ✅ 1.0000 | ✅ 1.0 | 6/7 (85.7%) | 197.93s | $3.3176 | $0.2562 |
| task_072 | ❌ 0.0000 | ❌ 0.0 | 8/9 (88.9%) | 153.05s | $1.3366 | $0.0236 |
| task_081 | ❌ 0.0000 | ❌ 0.0 | 28/37 (75.7%) | 231.15s | $5.1588 | $0.2262 |
| task_087 | ❌ 0.0000 | ❌ 0.0 | 18/20 (90.0%) | 262.19s | $6.1699 | $0.1331 |

All 6 tasks terminated normally (`USER_STOP`), no infra errors, no agent/user errors flagged by LLM judge.

## Full console output

```
2026-09-08 13:39:17.694 | INFO     | tau2.utils.utils:<module>:28 - Using data directory from source: /Users/ntb/projects/tau2-bench/data
2026-09-08 13:39:18.655 | INFO     | tau2.utils.llm_utils:<module>:102 - LiteLLM: Cache is disabled
2026-09-08 13:39:18.719 | DEBUG    | tau2.registry:<module>:281 - Registering default components...
2026-09-08 13:39:18.720 | DEBUG    | tau2.registry:<module>:292 - Voice dependencies not installed, skipping voice user registration
2026-09-08 13:39:18.720 | DEBUG    | tau2.registry:<module>:350 - Default components registered successfully. Registry info: {
  "domains": ["mock", "airline", "retail", "telecom", "telecom-workflow", "banking_knowledge"],
  "agents": ["llm_agent", "llm_agent_gt", "llm_agent_solo", "discrete_time_audio_native_agent"],
  "users": ["user_simulator", "dummy_user"],
  "task_sets": ["mock", "airline", "retail", "telecom_full", "telecom_small", "telecom", "telecom-workflow", "banking_knowledge"]
}
╭────────────────────────── Simulation Configuration ──────────────────────────╮
│ Domain: banking_knowledge  Task Set: Default  Tasks: task_014, task_024,     │
│ task_057, task_072, task_081, task_087                                       │
│ Trials: 1  Max Steps: 200  Max Errors: 10                                    │
│                                                                              │
│ Agent: llm_agent → claude-sonnet-5                                           │
│ User:  user_simulator → claude-sonnet-5                                      │
│                                                                              │
│ Save: smoke_test  Concurrency: 3  Verbose: False                             │
╰──────────────────────────────────────────────────────────────────────────────╯
🔄 Loading documents...
✅ Loaded 698 documents

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_014                                                            │
│ Reward: ❌ 0.0000 (ACTION: 0.0)  DB Check: ✅ 1.0                            │
│ Action Checks: - 0: agent transfer_to_human_agents [generic] ❌ 0.0          │
│ Partial Action Reward: 0/1 (0.0%)                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_024                                                            │
│ Reward: ❌ 0.0000 (DB: 0.0)  DB Check: ❌ 0.0                                │
│ Action Checks: - 0: user apply_for_credit_card [write] ❌ 0.0                │
│ Partial Action Reward: 0/1 (0.0%)                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_057                                                            │
│ Reward: ✅ 1.0000 (DB: 1.0)  DB Check: ✅ 1.0                                │
│ Action Checks:                                                               │
│ - 0: agent log_verification [write] ✅ 1.0                                   │
│ - 1: agent unlock_discoverable_agent_tool [generic] ✅ 1.0                   │
│ - 2: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 3: agent unlock_discoverable_agent_tool [generic] ✅ 1.0                   │
│ - 4: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 5: agent give_discoverable_user_tool [generic] ❌ 0.0                      │
│ - 6: user call_discoverable_user_tool [write] ✅ 1.0                         │
│ Partial Action Reward: 6/7 (85.7%)  Write: 4/4 (100.0%)                      │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_072                                                            │
│ Reward: ❌ 0.0000 (DB: 0.0)  DB Check: ❌ 0.0                                │
│ Action Checks:                                                               │
│ - 0: agent log_verification [write] ✅ 1.0                                   │
│ - 1: agent unlock_discoverable_agent_tool [generic] ✅ 1.0                   │
│ - 2: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 3: agent unlock_discoverable_agent_tool [generic] ✅ 1.0                   │
│ - 4: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 5: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 6: agent unlock_discoverable_agent_tool [generic] ✅ 1.0                   │
│ - 7: agent call_discoverable_agent_tool [write] ✅ 1.0                       │
│ - 8: agent call_discoverable_agent_tool [write] ❌ 0.0                       │
│ Partial Action Reward: 8/9 (88.9%)  Write: 5/6 (83.3%)                       │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_081                                                            │
│ Reward: ❌ 0.0000 (DB: 0.0)  DB Check: ❌ 0.0                                │
│ Action Checks: 37 checks, 28 passed (see results.json for full list)         │
│ Partial Action Reward: 28/37 (75.7%)  Write: 19/26 (73.1%)                   │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Simulation Overview ─────────────────────────────╮
│ Task ID: task_087                                                            │
│ Reward: ❌ 0.0000 (DB: 0.0)  DB Check: ❌ 0.0                                │
│ Action Checks: 20 checks, 18 passed (see results.json for full list)         │
│ Partial Action Reward: 18/20 (90.0%)  Write: 10/12 (83.3%)                   │
╰──────────────────────────────────────────────────────────────────────────────╯

Successfully completed all simulations!
To review the simulations, run: tau2 view
╭───────────────────────── Agent Performance Metrics ──────────────────────────╮
│   ═══ Overview ═══                                                           │
│   Total Simulations         6                                                │
│   Total Tasks               6                                                │
│                                                                              │
│   ═══ Reward Metrics ═══                                                     │
│   🏆 Average Reward         0.1667                                           │
│      Pass^1                 0.167                                            │
│   💰 Avg Cost/Conversation  $3.1958                                          │
│                                                                              │
│   ═══ Action Metrics ═══                                                     │
│   ✏️  Write Actions          38/49 (77.6%)                                    │
│                                                                              │
│   ═══ DB Match ═══                                                           │
│   🗄️  DB Match               ✓ 2 / ✗ 4 (33.3%)                                │
│                                                                              │
│   ═══ Termination ═══                                                        │
│   🛑 Normal Stop            6 (👤 6 / 🤖 0)                                  │
│                                                                              │
│   ═══ LLM Judge Review ═══                                                   │
│   🤖 Agent Errors           0 errors                                         │
│   👤 User Errors            0 errors                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Raw results: `data/simulations/smoke_test/results.json`
