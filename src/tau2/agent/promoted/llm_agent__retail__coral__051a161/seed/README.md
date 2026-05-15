# Tau2 Retail Seed

This repo is the mutable seed workspace consumed by CORAL.

Public contract:

- Export `create_agent(tools, domain_policy, task=None, **kwargs)` from `agent.py`.
- The seed receives model settings through runtime kwargs such as `llm` and
  `llm_args`. It must not read the task package `.env` directly.

Local public checks:

- `uv run pytest`
- `uv run python smoke.py --domain mock --num-tasks 1`

The retail/test evaluation itself is hidden behind the CORAL grader.
