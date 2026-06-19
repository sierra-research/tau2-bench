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
| Supervisor on (tasks 0–2) | 3 | 1 | 100% | see run output | see run output |
| Supervisor off (tasks 0–2) | 3 | 1 | 50% | see run output | see run output |
| Full retail base + supervisor | 114 | 1 | TBD | TBD | TBD |

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
uv run python examples/agents/return_exchange_agent_tau2.py `
  --return-agent-path "C:\dev\Return-and-Exchange-agent-main" `
  --task-ids 0 1 2 `
  --no-supervisor `
  --save-to return-exchange-ab-no-supervisor
```

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
