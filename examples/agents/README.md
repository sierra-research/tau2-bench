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

### `custom_agent_eval.py` -- Power user path

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
