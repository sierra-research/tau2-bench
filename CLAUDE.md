# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tau2-bench is a simulation framework for evaluating conversational customer service agents. It simulates interactions between an Agent, User (simulator), and Environment across multiple domains (airline, retail, telecom, mock).

## Common Commands

```bash
# Install (editable mode required for data directory detection)
pip install -e .

# Run benchmark evaluation
tau2 run --domain <domain> --agent-llm <llm> --user-llm <llm> --num-trials <n>

# View domain API documentation (opens at http://127.0.0.1:8004/redoc)
tau2 domain <domain>

# Interactive play mode (play as agent or user)
tau2 play

# View simulation results
tau2 view

# Check data directory setup
tau2 check-data

# Run all tests
make test
pytest tests/

# Run single test file
pytest tests/test_agent.py

# Run domain-specific tests
pytest tests/test_domains/test_airline/

# Linting and formatting (uses ruff)
make lint          # Check for issues
make format        # Format code
make lint-fix      # Auto-fix issues
make check-all     # Both lint and format
```

## Architecture

### Core Flow
The `Orchestrator` (src/tau2/orchestrator/orchestrator.py) manages turn-based message passing:
- **Agent** generates responses and tool calls
- **User** (simulator) responds to agent
- **Environment** executes tool calls and returns results

Messages flow: AGENT <-> USER, AGENT <-> ENV, USER <-> ENV

### Key Components

**Registry** (src/tau2/registry.py): Central registration for agents, users, and domains. All components must be registered here to be usable via CLI.

**Agents** (src/tau2/agent/):
- `BaseAgent` - Abstract base class
- `LLMAgent` - Standard conversational agent
- `LLMSoloAgent` - Operates without user interaction (tool calls only)
- `LLMGTAgent` - Ground truth agent with oracle action guidance
- `LangChainAgent` - Uses LangGraph's create_react_agent

**Domains** (src/tau2/domains/): Each domain contains:
- `data_model.py` - Database models
- `tools.py` - ToolKitBase implementation for agent tools
- `user_tools.py` - Optional user-side tools
- `environment.py` - get_environment() and get_tasks() functions

**Domain Data** (data/tau2/domains/<domain>/):
- `tasks.json` - Task definitions
- `split_tasks.json` - Task splits (must include "base" split)
- `policy.md` - Domain policy document
- `db.json` or `db.toml` - Domain database

### Message Protocol
- Messages must contain EITHER text OR tool calls, never both
- Tool calls must be followed by corresponding ToolMessage responses
- Simulation ends on: agent/user stop signal, max steps, max errors

## Configuration

Edit `src/tau2/config.py` for defaults (max steps, LLM settings, caching).

LLM calls use LiteLLM - configure API keys in `.env` file (copy from `.env.example`).

## Adding New Agents

1. Inherit from `LocalAgent` in `src/tau2/agent/base.py`
2. Implement `generate_next_message()` and `get_init_state()`
3. Register in `src/tau2/registry.py`:
   ```python
   registry.register_agent(MyAgent, "my_agent")
   ```
4. Use via CLI: `--agent my_agent`

## Adding New Domains

1. Create domain folder in `src/tau2/domains/<name>/`
2. Implement required files: data_model.py, tools.py, environment.py
3. Create data folder in `data/tau2/domains/<name>/`
4. Register in `src/tau2/registry.py`:
   ```python
   registry.register_domain(get_environment, "domain_name")
   registry.register_tasks(get_tasks, "domain_name")
   ```
5. Add tests in `tests/test_domains/test_<name>/`

## Testing Notes

- Use `--task-split base` for leaderboard-comparable evaluations
- Results saved to `data/tau2/simulations/`
- Telecom domain supports ablation studies with `llm_agent_solo` and `llm_agent_gt`
