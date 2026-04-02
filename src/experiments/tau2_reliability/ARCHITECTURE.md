# tau2-reliability Architecture

## Overview

A reliability evaluation framework for tau2-bench agents. Measures how consistently, safely, and robustly an agent handles customer service tasks.

```
tau2 run → traces → tau2 reliability analyze → JSON → dashboard
                  → tau2 reliability fault   → fault_results.json
                  → tau2 reliability enrich  → safety_results.json
```

## Components

### Python Analysis Pipeline (`src/experiments/tau2_reliability/tau2_reliability/`)

```
conversation_analyzer.py    Per-conversation: outcome, actions, policy, efficiency, abstention
task_analyzer.py            Per-task across K trials: consistency, classification, divergence
domain_analyzer.py          Aggregate: dimension scores, task classes, export JSON
fault_runner.py             Re-runs tasks with injected tool failures
safety_runner.py            LLM-as-judge for 6 safety constraints
```

### Metric Modules

```
metrics/consistency.py      C_out (outcome), C_traj_d (JSD), C_traj_s (Levenshtein), C_res (CV)
metrics/predictability.py   P_cal (ECE), P_auroc (Mann-Whitney), P_brier
metrics/robustness.py       R_fault = min(Acc_faulted / Acc_baseline, 1.0)
metrics/safety.py           S_comp, S_harm (from violation data)
```

### Analysis Modules

```
analysis/policy_adherence.py   DAG-based workflow checking (3 domain DAGs)
analysis/divergence.py         Cross-trial branching detection
analysis/mutation_aware.py     SABER write-action failure attribution
analysis/task_taxonomy.py      bimodal/stable_pass/stable_fail/fragile classification
analysis/abstention.py         Regex-based deferral/refusal detection
```

### Web Dashboard (`web/reliability-dashboard/`)

```
Overview.jsx         Hero + dimension cards + task taxonomy + findings + recommendations
DimensionView.jsx    Per-dimension: sub-metrics, heatmap/breakdown, conversation list
ConversationList.jsx Filterable table of all conversations
ConversationAudit.jsx Full message trace with tool annotations + metrics panel
```

## Data Flow

```
SimulationRun (tau2 native)
  → conversation_analyzer.py: per-conversation dict
    {id, task_id, trial, outcome, reward_breakdown, actions, messages,
     policy_adherence, efficiency, abstention}
  → task_analyzer.py: per-task dict
    {task_id, outcomes, consistency{outcome, actions, sequence, resources},
     class, divergence, decisive_action}
  → domain_analyzer.py: domain summary
    {model, domain, accuracy, dimensions{5 scores}, task_classes, efficiency, abstention}
  → reliability_data.json (slim, no messages)
  → reliability_full.json (with messages for audit view)
```

## Metric Definitions

### Consistency (requires K≥2 trials per task)

| Metric | Formula | Note |
|--------|---------|------|
| Outcome | `1 - var(outcomes) / (p(1-p) + ε)` | Uses population variance in both |
| Action | `1 - mean(pairwise JSD)` | Conditioned on successful runs |
| Sequence | `1 - mean(normalized Levenshtein)` | Conditioned on successful runs |
| Resource | `exp(-mean(CV_cost, CV_duration, CV_actions))` | |

### Safety (via LLM judge)

| Metric | Formula |
|--------|---------|
| S_comp | `1 - (conversations with violations / total)` |
| S_harm | `1 - mean(max_severity_per_conversation)` |
| S_safety | `1 - (1 - S_comp)(1 - S_harm)` |

### Fault Robustness (requires fault injection run)

| Metric | Formula |
|--------|---------|
| R_fault | `min(Acc_faulted / Acc_baseline, 1.0)` |

### Policy Adherence (from action sequences)

| Metric | Formula |
|--------|---------|
| Adherence | `matched_transitions / expected_transitions` |

### Abstention (regex detection on agent messages)

| Metric | Formula |
|--------|---------|
| Rate | `abstained_count / total` |
| Precision | `P(fail \| abstain)` |
| Recall | `P(abstain \| fail)` |

## Known Issues and Limitations

1. **Consistency C_out is binary** for small K — with Bessel correction, non-unanimous outcomes score near 0
2. **Fault injection is syntactic only** — timeout/500/empty, not semantic data corruption (AVER-style)
3. **No predictability metrics yet** — requires confidence elicitation (LLM calls)
4. **Abstention detection has false positives** — pattern matching doesn't consider task outcome
5. **Policy DAGs are hardcoded** — should be configuration files
6. **No bootstrap standard errors** on conversation-level or domain-level aggregates

## CLI Commands

```bash
# Analyze (FREE — reads existing traces)
tau2 reliability analyze --results <path> --output <dir>

# Fault robustness (re-runs with injected failures)
tau2 reliability fault --results <path> --fault-rate 0.2 --num-trials 3 --output <dir>

# Safety (LLM-as-judge, ~$0.10-0.50)
tau2 reliability enrich --results <path> --judge-model azure/gpt-5.2 --output <dir>

# Quick summary
tau2 reliability summary --results <path>
```

## Test Coverage

165 tests across 12 test files covering:
- Consistency metrics (30 tests)
- Predictability metrics (16 tests)
- Robustness metrics (10 tests)
- Policy adherence (13 tests)
- Divergence analysis (15 tests)
- Mutation-aware analysis (12 tests)
- Task taxonomy (8 tests)
- Abstention detection (9 tests)
- Confidence elicitation (12 tests)
- Data extraction (9 tests)
- Models serialization (10 tests)
- Integration (6 tests)
