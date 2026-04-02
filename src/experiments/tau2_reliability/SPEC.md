# tau2-reliability: Definitive Specification

## Goal

One command to evaluate agent reliability. Three-tier visualization to understand and improve.

```bash
tau2 run --domain retail --agent-llm azure/gpt-5.4 --num-trials 5 --save-to my_eval
tau2 reliability analyze --results data/simulations/my_eval/results.json --output analysis/
# Open dashboard → Domain → Metric → Conversation
```

## What We Measure (from baseline traces, no extra LLM calls)

### Tier 1: Consistency (requires K≥2 trials)

| Metric | Formula | Range | What it answers |
|--------|---------|-------|-----------------|
| Outcome Consistency | `1 - var(outcomes)/(p(1-p)+ε)` per task, mean across tasks | [0,1] | Same pass/fail each run? |
| Action Consistency | `1 - mean(JSD(action_distributions))` pairwise across successful runs | [0,1] | Same action types across runs? |
| Sequence Consistency | `1 - mean(normalized_levenshtein)` pairwise across successful runs | [0,1] | Same action ordering? |
| Resource Consistency | `exp(-mean(CV_cost, CV_duration, CV_actions))` | [0,1] | Stable cost/time? |
| **Overall Consistency** | `1/3*(outcome + mean(action,sequence) + resource)` | [0,1] | |

### Tier 2: Policy Adherence (from action sequences + domain DAG)

| Metric | Formula | Range | What it answers |
|--------|---------|-------|-----------------|
| Policy Compliance | `matched_transitions / expected_transitions` per conversation | [0,1] | Follows the right workflow? |
| Violations | Count of SKIPPED_NODE, WRONG_ORDER | int | What went wrong? |

### Tier 3: Efficiency (from conversation traces)

| Metric | Source | What it answers |
|--------|--------|-----------------|
| Redundant Calls | Consecutive identical tool calls | Wasted API calls? |
| Loops | Repeated 3-action patterns | Stuck in loops? |
| Tool Errors | Tool responses containing "error" | How many errors hit? |
| Read-Before-Write | % of WRITEs preceded by READ | Verifies before mutating? |

### Tier 4: Task Classification (derived from consistency)

| Class | Rule | Meaning |
|-------|------|---------|
| Stable Pass | c_out > 0.8 AND pass_rate > 0.8 | Reliably succeeds |
| Stable Fail | c_out > 0.8 AND pass_rate < 0.2 | Reliably fails (capability gap) |
| Bimodal | c_out < 0.3 | Flips between pass/fail (reliability frontier) |
| Fragile | c_out > 0.8 AND c_traj_s < 0.5 | Succeeds via different paths |

### Tier 5: Fault Robustness (requires additional runs)

| Metric | Formula | What it answers |
|--------|---------|-----------------|
| R_fault | `min(Acc_faulted / Acc_baseline, 1.0)` | Handles API failures? |

Faults injected: timeout (30%), HTTP 500 (35%), empty response (15%), malformed (20%).

### Future (requires LLM calls — not in initial release)

- Safety Compliance (LLM-as-judge)
- Confidence Calibration (agent self-assessment)
- Prompt Robustness (paraphrase re-runs)

## Retail Domain Policy DAG (Corrected)

```
Phase: AUTHENTICATE
  Tools: find_user_id_by_name_zip, find_user_id_by_email
  Required: YES (must be first)

Phase: GATHER_INFO
  Tools: get_user_details, get_order_details, get_product_details,
         get_item_details, list_all_product_types
  Required: YES (at least get_user_details + get_order_details)
  Prerequisite: AUTHENTICATE

Phase: EXECUTE
  Tools: cancel_pending_order, modify_pending_order_address,
         modify_pending_order_items, modify_pending_order_payment,
         return_delivered_order_items, exchange_delivered_order_items,
         modify_user_address
  Required: Depends on task
  Prerequisite: GATHER_INFO

Phase: ESCALATE
  Tools: transfer_to_human_agents
  Required: NO (only when out of scope)
```

Valid transitions: AUTHENTICATE → GATHER_INFO → EXECUTE → ESCALATE

## Airline Domain Policy DAG (Corrected)

```
Phase: AUTHENTICATE
  Tools: get_user_details
  Required: YES

Phase: GATHER_INFO
  Tools: get_reservation_details, get_flight_status, list_all_airports,
         search_direct_flight, search_onestop_flight
  Required: YES

Phase: EXECUTE
  Tools: book_reservation, cancel_reservation, update_reservation_flights,
         update_reservation_passengers, update_reservation_baggages,
         send_certificate
  Required: Depends on task

Phase: ESCALATE
  Tools: transfer_to_human_agents, calculate
  Required: NO
```

## Dashboard: Three-Tier Visualization

### Level 1: Domain Overview
- Model name, domain, accuracy, trial count
- 5 dimension gauge cards (Outcome Consistency, Action Consistency, Sequence Consistency, Cost Stability, Policy Compliance)
- Task classification distribution bar
- Key stats

### Level 2: Metric Drill-Down
Click any dimension → see:
- All conversations ranked by that metric
- Conversations that FAIL this metric highlighted
- For consistency: tasks grouped by bimodal/stable/fragile
- For policy: tasks grouped by violation type
- For efficiency: tasks sorted by redundant calls / errors

### Level 3: Conversation Audit
Click any conversation → see:
- Full message trace with role-colored cards
- Tool calls annotated: READ (blue), WRITE (red), correct (✓), wrong (✗)
- Reward breakdown: DB ✓/✗, Actions ✓/✗, Communication ✓/✗
- Policy adherence: workflow phases followed/skipped
- Efficiency: redundant calls, loops, errors
- For bimodal tasks: trial switcher to compare pass vs fail

## Output JSON Structure

```json
{
  "runs": [
    {
      "id": "filename_stem",
      "domain_summary": {
        "model": "azure/gpt-5.4",
        "domain": "retail",
        "accuracy": 0.72,
        "num_tasks": 50,
        "num_trials": 5,
        "total_conversations": 250,
        "dimensions": {
          "outcome_consistency": {"score": 0.68, "label": "Outcome Consistency", "question": "Same pass/fail each run?"},
          ...
        },
        "task_classes": {"stable_pass": 20, "stable_fail": 10, "bimodal": 16, "fragile": 4},
        "efficiency": {...}
      },
      "tasks": {
        "task_0": {
          "num_trials": 5,
          "pass_rate": 0.6,
          "outcomes": ["pass","fail","pass","pass","fail"],
          "consistency": {"outcome": 0.0, "actions": 0.65, "sequence": 0.58, "resources": 0.82},
          "class": "bimodal",
          "divergence": {"turn": 2, "common_prefix": [...], "success_path": [...], "failure_path": [...]},
          "decisive_action": "cancel_pending_order"
        }
      },
      "conversations": [
        {
          "id": "sim_uuid",
          "task_id": "0",
          "trial": 0,
          "outcome": "fail",
          "reward": 0.0,
          "reward_breakdown": {...},
          "cost_usd": 0.12,
          "duration_sec": 45.2,
          "actions": [{"name": "find_user_id_by_name_zip", "type": "READ", "correct": true, "turn": 2}, ...],
          "messages": [...],
          "policy_adherence": {"score": 0.67, "violations": [...]},
          "efficiency": {"redundant_calls": 1, "loops": 0, "tool_errors": 0, "read_before_write_rate": 1.0}
        }
      ]
    }
  ]
}
```

## Implementation Checklist

### Python (src/experiments/tau2_reliability/)
- [x] models.py — Pydantic data models
- [x] extract.py — TaskTrialData extraction + build_tool_type_map()
- [x] metrics/consistency.py — 4 consistency metrics
- [x] metrics/predictability.py — ECE, AUROC, Brier
- [x] metrics/robustness.py — Robustness ratios
- [x] analysis/divergence.py — Cross-trial divergence
- [x] analysis/mutation_aware.py — SABER analysis
- [x] analysis/task_taxonomy.py — Task classification
- [x] conversation_analyzer.py — Per-conversation analysis
- [x] task_analyzer.py — Cross-trial task analysis
- [x] domain_analyzer.py — Aggregate + JSON export
- [x] fault_runner.py — Fault injection evaluation
- [ ] analysis/policy_adherence.py — FIX retail DAG with correct tool names
- [x] tau2 reliability CLI subcommand

### Web Dashboard (web/reliability-dashboard/)
- [x] Basic scaffold (React + Vite)
- [x] DomainOverview.jsx — dimension cards + task classes
- [x] ConversationList.jsx — filterable table
- [x] ConversationAudit.jsx — message trace + metrics panel
- [ ] Dark theme matching HAL aesthetic
- [ ] Metric drill-down view (Level 2)
- [ ] Multi-agent comparison charts
- [ ] Auto-generated findings + recommendations

### Tests
- [x] 156 tests passing (consistency, predictability, robustness, divergence, mutation, taxonomy, policy, confidence, models, extract, integration)
