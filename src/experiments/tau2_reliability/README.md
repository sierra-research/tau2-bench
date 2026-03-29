# tau2-reliability

Comprehensive agent reliability analysis for tau2-bench. Measures how consistently, safely, and robustly an agent handles customer service tasks across multiple trials.

## Quick Start

```bash
# 1. Run simulations
uv run tau2 run --domain retail --agent-llm azure/gpt-5.2 --num-trials 5 --save-to my_eval

# 2. Analyze reliability (FREE — no LLM calls)
PYTHONPATH=src/experiments/tau2_reliability \
uv run tau2 reliability analyze \
  --results data/simulations/my_eval/results.json \
  --output web/reliability-dashboard/public/data/

# 3. Safety analysis (LLM judge, ~$0.10-0.50)
PYTHONPATH=src/experiments/tau2_reliability \
uv run tau2 reliability enrich \
  --results data/simulations/my_eval/results.json \
  --judge-model azure/gpt-5.2 \
  --output web/reliability-dashboard/public/data/

# 4. Fault robustness (re-runs with injected errors)
PYTHONPATH=src/experiments/tau2_reliability \
uv run tau2 reliability fault \
  --results data/simulations/my_eval/results.json \
  --fault-rate 0.2 --num-trials 3 \
  --output web/reliability-dashboard/public/data/

# 5. Launch dashboard
cd web/reliability-dashboard && npm install && npm run dev
```

## What It Measures

### Consistency (from multi-trial runs, no extra cost)

| Metric | What it answers |
|--------|-----------------|
| Outcome Consistency | Same pass/fail each run? |
| Tool Usage Consistency | Same tools used across runs? |
| Step Order Consistency | Same action ordering? |
| Cost & Speed Stability | Predictable cost and time? |

### Workflow & Efficiency (from conversation traces, no extra cost)

| Metric | What it answers |
|--------|-----------------|
| Workflow Compliance | Follows authenticate → gather info → execute? |
| Redundant Calls | Wasted consecutive identical tool calls? |
| Read Before Write | Verifies state before making changes? |
| Abstention Detection | Knows when to defer vs attempt? |

### Fault Robustness (requires additional runs)

| Metric | What it answers |
|--------|-----------------|
| Error Recovery | Maintains accuracy when tools fail? |

### Safety (requires LLM judge calls)

| Metric | What it answers |
|--------|-----------------|
| Safety Compliance | Avoids PII exposure, unauthorized actions, financial errors? |

## Dashboard

Three-tier visualization:

1. **Overview** — Dimension scores, task classification, findings, recommendations
2. **Dimension drill-down** — Per-task breakdown, violation details, relevant conversations
3. **Conversation audit** — Full message trace with tool annotations, reward breakdown, policy violations

## Running Tests

```bash
PYTHONPATH=src/experiments/tau2_reliability \
uv run python -m pytest src/experiments/tau2_reliability/tests/ -v
```

## Architecture

```
tau2_reliability/
├── conversation_analyzer.py   Per-conversation: outcome, actions, policy, efficiency, abstention
├── task_analyzer.py           Per-task across K trials: consistency, classification, divergence
├── domain_analyzer.py         Aggregate scores + JSON export for dashboard
├── fault_runner.py            Re-runs tasks with injected tool failures
├── safety_runner.py           LLM-as-judge for 6 safety constraints
├── metrics/                   Consistency, predictability, robustness, safety formulas
├── analysis/                  Policy adherence, divergence, mutation-aware, task taxonomy, abstention
├── visualization/             Matplotlib plots (static)
└── runners/                   Analysis pipeline, confidence elicitation, prompt variation
```
