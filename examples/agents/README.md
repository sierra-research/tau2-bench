# Agent Examples

Runnable examples showing how to create and evaluate custom tau2 agents.

## Examples

### `minimal_text_agent.py` -- Start here

A single-file example that creates a minimal agent, registers it, and runs it against the mock domain. Shows:

- Implementing `HalfDuplexAgent` (the two required methods)
- Writing a factory function
- Registering with the registry
- Running via `run_single_task` or `run_domain`

```bash
python examples/agents/minimal_text_agent.py
```

### `react_agent.py` -- ReAct pattern

A ReAct (Reasoning + Acting) agent that explicitly thinks before acting. Each turn follows:

1. **THINK** -- reason about the situation (LLM call without tools)
2. **ACT** -- choose a tool call or text response based on the reasoning (LLM call with tools)

Shows how to customize the agent's decision-making process to improve tool-use accuracy.

```bash
python examples/agents/react_agent.py
```

### `return_exchange_agent_tau2.py` — External Return-and-Exchange agent

Evaluates the [Return-and-Exchange agent](https://github.com/annagibaeva/Return-and-Exchange-agent) against the τ-bench **retail** domain (returns/exchanges). Uses your agent's skills and τ-bench retail supervisor with real retail tools.

```powershell
# From tau2-bench root (PowerShell)
uv sync --extra return_exchange
uv run python examples/agents/return_exchange_agent_tau2.py `
  --return-agent-path "C:\dev\Return-and-Exchange-agent-main" `
  --task-ids 0 1 2 `
  --user-llm openai/gpt-4.1-mini
```

Requires `ANTHROPIC_API_KEY` (agent + supervisor) and a user-simulator key in `.env` (e.g. `OPENAI_API_KEY`).

#### Pass^k reliability and cost-per-resolution

τ-bench scores **pass^k** (probability a task passes all *k* trials). Under outcome-based pricing, **margin per resolution = price per case − cost per resolved case** — so track both reliability and cost together.

Each run logs **tokens per conversation** on every simulation (`sim.info.token_usage` in `results.json`) and prints a cost summary when the eval finishes:

| Component | What it bills |
|-----------|----------------|
| `agent` | Claude turns (tool + text) |
| `supervisor` | Retail policy audit (LLM path only; deterministic fast-path = $0) |
| `user_simulator` | τ-bench user simulator (replaces the golden-set **judge** tax) |

**Cost model (claude-sonnet-4-6):** $3/M input, $15/M output tokens.

| Run | Tasks | Trials | Pass^1 | Cost / conversation | Cost / resolved case |
|-----|-------|--------|--------|---------------------|----------------------|
| Pilot (old supervisor, tasks 0–2) | 3 | 1 | 1/3 | — | — |
| Supervisor off (tasks 0 & 2) | 2† | 1 | 1/2 | n/a‡ | n/a‡ |
| Retail supervisor retest (tasks 0 & 2) | 2 | 1 | **2/2** | — | — |
| Supervisor on + cost logging (tasks 0–2) | 3 | 1 | 2/3 | $0.28 | $0.32 (2/3) |
| Full retail base + supervisor | 114 | 1 | TBD | TBD | TBD |

† A/B run (`return-exchange-ab-no-supervisor-0-2`) executed tasks 0 and 2 only; task 1 not run.  
‡ Agent + supervisor costs not logged in this run (user-sim only ≈ $0.002/conversation). Full cost data from `cost-pilot-with-supervisor` (supervisor on, tasks 0–2).

**Pilot sources:** Pass/cost for supervisor-on row from `cost-pilot-with-supervisor` (tasks 0→1.0, 1→0.0, 2→1.0). Retest `return-exchange-retail-supervisor-retest-0-2` agrees on tasks 0 & 2 (both 1.0). Earlier `return-exchange-agent-retail-pilot` scored 1/3 (0, 1, 0) before retest fixes. Supervisor-off from `return-exchange-ab-no-supervisor-0-2` (task 0→0.0, task 2→1.0).

Re-run with `--num-trials 4` for pass^4 reliability. A full pass^k sweep (agent + supervisor + user sim, multi-turn) is roughly **~170 billed LLM calls** for the golden-set harness at k=5; τ-bench retail scales with task count × trials.

Example cost block after a run:

```text
==================================================
COST PER RESOLUTION
==================================================
  Total: $0.0842  (42,100 in / 3,200 out, 18 billed LLM calls across 3 conversations)
  Per conversation (avg over k=1): $0.0281
  Per resolved case (reward=1): $0.0315 (2/3 resolved)
  agent          $0.0610  (12 calls, ...)
  supervisor     $0.0040  (2 calls, ...)
  user_simulator $0.0192  (6 calls, ...)
```

Pass^k without supervisor (A/B):

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python examples/agents/return_exchange_agent_tau2.py `
  --return-agent-path "C:\dev\Return-and-Exchange-agent-main" `
  --task-ids 0 2 `
  --user-llm openai/gpt-4.1-mini `
  --no-supervisor `
  --save-to return-exchange-ab-no-supervisor-0-2
```

Retail supervisor retest (fixed `retail_supervisor.py`):

```powershell
uv run python examples/agents/return_exchange_agent_tau2.py `
  --return-agent-path "C:\dev\Return-and-Exchange-agent-main" `
  --task-ids 0 2 `
  --user-llm openai/gpt-4.1-mini `
  --save-to return-exchange-retail-supervisor-retest-0-2
```

#### Supervisor A/B walkthrough (tasks 0 & 2)

The pilot failed tasks 0 and 2 with the original Singapore Apparel supervisor ported to retail. This A/B isolates whether the supervisor or the agent caused each failure.

**What `--no-supervisor` does.** In `return_exchange_agent_tau2.py`, the supervisor sits between Claude's draft and what the user sees. With `--no-supervisor`, the draft goes straight to the user — no retail audit layer.

| Task | Scenario | Key success criterion |
|------|----------|----------------------|
| 0 | Exchange keyboard (fallback: clicky/no backlight) + thermostat | `exchange_delivered_order_items` with both items |
| 2 | Count t-shirt options + return cleaner, headphones, smart watch | Tell user 10 t-shirt options + `return_delivered_order_items` |

**Results (1 trial each)**

| Task | Pilot (old supervisor) | `--no-supervisor` | Retail supervisor (retest) | Verdict |
|------|------------------------|-------------------|----------------------------|---------|
| 0 | ❌ 0.0 — escalated | ❌ 0.0 — no exchange write | ✅ 1.0 — full exchange with fallback keyboard | Mixed → fixed by retail supervisor |
| 2 | ❌ 0.0 — blocked mid-flow | ✅ 1.0 — "10 available" + return | ✅ 1.0 — same | Supervisor was the blocker |

Average reward: **0.50** (1/2) without supervisor vs **0.33** (1/3) on full pilot vs **1.0** (2/2) on retail-supervisor retest.

**Task 2 — supervisor was the problem.** With the old supervisor, a REVISE stub ("let me bring in a colleague…") stopped the flow before the agent could state variant counts or call the return tool. Without supervisor, Claude completed normally (reward 1.0). The retail supervisor fast-paths read-only lookup and variant-count replies, so the retest also passes.

**Task 0 — not just the supervisor.** Without supervisor, the agent still never called `exchange_delivered_order_items` — it reached the OOS fallback explanation and user confirmation, then the user sim ended (`USER_STOP`) before the write. With the retail supervisor, the agent completed the exchange (reward 1.0); the OOS fast-path lets the flow continue instead of escalating.

| | Task 0 | Task 2 |
|---|---|---|
| Supervisor fault | Yes (blocked escalation) | Yes (blocked mid-flow) |
| Agent fault | Yes (never executed exchange without supervisor) | No (completed without supervisor) |

**Inspect trajectories**

```powershell
# Interactive browser
uv run tau2 view

# Or open raw JSON
code data/simulations/return-exchange-retail-supervisor-retest-0-2/results.json
code data/simulations/return-exchange-ab-no-supervisor-0-2/results.json
```

Look for: Task 0 — is `exchange_delivered_order_items` in the message history? Task 2 — does the assistant mention "10" before the return tool call?

**Simulation folders**

| Folder | Description |
|--------|-------------|
| `return-exchange-agent-retail-pilot` | First pilot (tasks 0–2, old supervisor): 1/3 |
| `return-exchange-ab-no-supervisor-0-2` | A/B without supervisor (tasks 0 & 2): 1/2 |
| `return-exchange-retail-supervisor-retest-0-2` | Fixed `retail_supervisor.py` (tasks 0 & 2): 2/2 |
| `cost-pilot-with-supervisor` | Cost instrumentation (tasks 0–2, supervisor on) |

### `custom_agent_eval.py` — Manual orchestrator wiring

Builds all components manually without the registry. Shows:

- Building environment, agent, user, and orchestrator by hand
- Running `run_simulation()` directly
- Inspecting results (messages, rewards, evaluation details)
- Adding custom behavior (logging, call counting)

```bash
python examples/agents/custom_agent_eval.py
```

## The Agent Interface

Every text agent must subclass `HalfDuplexAgent` and implement two methods:

```python
class MyAgent(HalfDuplexAgent[MyState]):

    def get_init_state(self, message_history=None) -> MyState:
        """Return the initial state (e.g., system prompt + history)."""
        ...

    def generate_next_message(self, message, state) -> tuple[AssistantMessage, MyState]:
        """Given a user/tool message and current state, return (response, new_state)."""
        ...
```

The agent receives `tools: list[Tool]` and `domain_policy: str` in `__init__`.

See `src/tau2/agent/README.md` for the full developer guide.
